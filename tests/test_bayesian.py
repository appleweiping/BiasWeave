from __future__ import annotations

import json
import math
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from biasweave import BayesianOptimizer as PublicBayesianOptimizer
from biasweave.cli import main
from biasweave.dominance import assess, failed_trial
from biasweave.encoding import make_point
from biasweave.errors import ConfigurationError
from biasweave.model import RunConfig, Scalar, Trial, TrialStatus
from biasweave.optimizers.bayesian import (
    BayesianOptimizer,
    _normalize_rows,
)
from biasweave.problem import parse_problem
from biasweave.strategy import optimize_strategy
from biasweave.surrogate import GaussianPrediction


def _problem(*, choice_count: int = 3, constrained: bool = False):
    constraints = (
        [{"metric": "limit", "relation": "le", "limit": 0.0, "scale": 1.0}] if constrained else []
    )
    return parse_problem(
        {
            "schema_version": 1,
            "variables": {
                "x": {"kind": "real", "low": 0.0, "high": 1.0, "default": 0.5},
                "n": {"kind": "integer", "low": 0, "high": 10, "default": 5},
                "mode": {
                    "kind": "choice",
                    "values": [f"m{index}" for index in range(choice_count)],
                    "default": "m0",
                },
            },
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": constraints,
        }
    )


def _metrics(point: Mapping[str, Scalar], *, limit: float = 0.0) -> dict[str, float]:
    x = float(point["x"])
    n = float(point["n"])
    return {"a": (x - 0.2) ** 2 + n / 100.0, "b": 1.0 - x, "limit": limit}


def test_features_use_canonical_numeric_encoding_and_full_choice_one_hot() -> None:
    problem = _problem()
    optimizer = BayesianOptimizer(problem, 3)
    left = make_point(problem, (0.25, 0.54, 0.34))
    right = make_point(problem, (0.25, 0.50, 0.65))

    assert optimizer.feature_count == 5
    assert optimizer._point_features(left) == (0.25, 0.5, 0.0, 1.0, 0.0)
    assert optimizer._point_features(right) == (0.25, 0.5, 0.0, 1.0, 0.0)
    assert left.values == right.values

    categorical = [
        optimizer._point_features(make_point(problem, (0.25, 0.5, coordinate)))[-3:]
        for coordinate in (0.1, 0.5, 0.9)
    ]
    squared_distances = {
        math.fsum((a - b) ** 2 for a, b in zip(categorical[left_index], right, strict=True))
        for left_index, right in ((0, categorical[1]), (0, categorical[2]), (1, categorical[2]))
    }
    assert squared_distances == {2.0}


def test_feature_expansion_rejects_more_than_64_dimensions_before_fitting() -> None:
    with pytest.raises(ConfigurationError, match="at most 64"):
        BayesianOptimizer(_problem(choice_count=63), 0)


@pytest.mark.parametrize(
    "options",
    [
        {"initial_design_size": True},
        {"initial_design_size": 1.0},
        {"initial_design_size": 0},
        {"initial_design_size": 129},
        {"training_window_size": 1},
        {"training_window_size": 129},
        {"candidate_pool_size": 3},
        {"candidate_pool_size": 65},
        {"length_scale": 0.0},
        {"length_scale": 11.0},
        {"length_scale": "1"},
        {"noise_variance": 0.0},
        {"noise_variance": 1.01},
        {"local_fraction": -0.01},
        {"local_fraction": 1.01},
        {"local_radius": 0.0},
        {"local_radius": 1.01},
        {"exploration": -0.01},
        {"exploration": 1.01},
        {"exploration": float("inf")},
    ],
)
def test_all_bayesian_public_hyperparameters_fail_before_proposal(options):
    with pytest.raises(ConfigurationError):
        BayesianOptimizer(_problem(), 0, **options)


@pytest.mark.parametrize("seed", [True, 1.0, None, "1", -(2**63) - 1, 2**63])
def test_seed_type_and_signed64_range_are_explicit(seed):
    with pytest.raises(ConfigurationError):
        BayesianOptimizer(_problem(), seed)


def test_exact_hyperparameter_endpoints_and_empty_or_invalid_target_rows():
    for options in (
        {
            "length_scale": 1e-4,
            "noise_variance": 1e-10,
            "local_fraction": 0.0,
            "local_radius": 1.0,
            "exploration": 0.0,
        },
        {
            "length_scale": 10.0,
            "noise_variance": 1.0,
            "local_fraction": 1.0,
            "local_radius": 1e-6,
            "exploration": 1.0,
        },
    ):
        assert len(BayesianOptimizer(_problem(), -(2**63), **options).ask(1)) == 1
    assert _normalize_rows(()) == ()
    for rows in (((),), ((1.0,), (2.0, 3.0)), ((math.nan,),), ((math.inf,),)):
        with pytest.raises(ConfigurationError):
            _normalize_rows(rows)


def test_objective_and_diagnostic_history_have_explicit_resource_bounds() -> None:
    data = {
        "schema_version": 1,
        "variables": {"x": {"kind": "real", "low": 0.0, "high": 1.0}},
        "objectives": [
            {"metric": f"metric_{index}", "goal": "min", "scale": 1.0} for index in range(129)
        ],
        "constraints": [],
    }
    with pytest.raises(ConfigurationError, match="at most 128 objectives"):
        BayesianOptimizer(parse_problem(data), 0)

    optimizer = BayesianOptimizer(_problem(), 0)
    for _index in range(130):
        optimizer._objective_weights()
    assert optimizer.scalarization_step == 130
    assert len(optimizer.scalarization_history) == 128
    assert [record.step for record in optimizer.scalarization_history] == list(range(2, 130))

    with pytest.raises(ConfigurationError, match="signed 64-bit"):
        BayesianOptimizer(_problem(), 10**5000)


def test_bayesian_ask_is_strictly_sequential_and_fits_once(monkeypatch) -> None:
    problem = _problem()
    optimizer = BayesianOptimizer(
        problem,
        11,
        initial_design_size=1,
        candidate_pool_size=8,
    )
    first = optimizer.ask(9)
    assert len(first) == 1
    optimizer.tell((assess(problem, 0, first[0], _metrics(first[0].values)),))

    calls: list[tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]] = []
    predictions: list[tuple[float, ...]] = []

    class FakeProcess:
        def __init__(self, points, targets, **_options):
            calls.append((points, targets))

        def predict(self, coordinates):
            predictions.append(coordinates)
            return GaussianPrediction(mean=-coordinates[0], variance=0.25)

    monkeypatch.setattr("biasweave.optimizers.bayesian.GaussianProcess", FakeProcess)
    second = optimizer.ask(9)
    assert len(second) == 1
    assert len(calls) == 1
    assert 1 <= len(predictions) <= optimizer.candidate_pool_size
    assert optimizer._point_features(second[0])[0] == max(item[0] for item in predictions)
    with pytest.raises(ConfigurationError, match="tell must report"):
        optimizer.ask(1)


def _trial(
    optimizer: BayesianOptimizer,
    trial_id: int,
    coordinate: float,
    *,
    feasible: bool,
) -> Trial:
    point = make_point(optimizer.problem, (coordinate, coordinate, coordinate))
    return assess(
        optimizer.problem,
        trial_id,
        point,
        _metrics(point.values, limit=0.0 if feasible else 1.0 + coordinate),
    )


def test_training_window_is_mode_specific_and_keeps_its_constraint_first_incumbent() -> None:
    optimizer = BayesianOptimizer(_problem(constrained=True), 4)
    infeasible = tuple(
        _trial(optimizer, index, (index + 1) / 200.0, feasible=False) for index in range(130)
    )
    optimizer._tell(infeasible)
    recovery = optimizer._training_trials()
    assert len(recovery) == 128
    assert recovery[0].trial_id == 0
    assert {trial.trial_id for trial in recovery[1:]} == set(range(3, 130))

    feasible = _trial(optimizer, 130, 0.9, feasible=True)
    optimizer._tell((feasible,))
    assert optimizer._training_trials() == (feasible,)


def test_failed_trials_are_never_modeled_or_assigned_synthetic_targets() -> None:
    optimizer = BayesianOptimizer(_problem(constrained=True), 5)
    point = make_point(optimizer.problem, (0.1, 0.1, 0.1))
    optimizer._tell((failed_trial(0, point, "simulator failed"),))

    assert optimizer.successful_observations == 0
    assert optimizer.failed_observations == 1
    assert optimizer._training_trials() == ()
    assert optimizer.scalarization_history == ()


def test_all_failed_catalog_run_spends_budget_without_fitting(monkeypatch) -> None:
    class ForbiddenProcess:
        def __init__(self, *_args, **_options):
            raise AssertionError("failed evaluations must not be fitted")

    monkeypatch.setattr("biasweave.optimizers.bayesian.GaussianProcess", ForbiddenProcess)
    result = optimize_strategy(
        _problem(),
        lambda _point: (_ for _ in ()).throw(RuntimeError("simulator failed")),
        evaluator_id="tests:all-failed-bayes",
        config=RunConfig(12, seed=2, batch_size=6),
        strategy="bayes",
    )
    assert result.stop_reason == "budget"
    assert len(result.trials) == result.failed_trials == 12
    assert all(trial.status is TrialStatus.FAILED for trial in result.trials)


def test_extreme_finite_rows_normalize_without_intermediate_overflow() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    normalized = _normalize_rows(((-maximum, maximum), (0.0, 0.0), (maximum, -maximum)))
    assert normalized == ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0))


@pytest.mark.parametrize("domain", ["wide-integer", "wide-real", "narrow-log"])
def test_adaptive_bayesian_path_preserves_extreme_encoding_domains(domain: str) -> None:
    if domain == "wide-integer":
        variable = {"kind": "integer", "low": 0, "high": 2**60}
    elif domain == "wide-real":
        variable = {"kind": "real", "low": -1e308, "high": 1e308}
    else:
        low = 1e308
        high = low
        for _index in range(8):
            high = math.nextafter(high, math.inf)
        variable = {"kind": "real", "low": low, "high": high, "scale": "log"}
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": variable},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )

    def finite_metrics(point: Mapping[str, Scalar]) -> dict[str, float]:
        value = point["x"]
        if isinstance(value, int):
            normalized = float(value // 2**40)
        elif domain == "wide-real":
            normalized = value / 1e308
        else:
            assert isinstance(variable["low"], float)
            assert isinstance(variable["high"], float)
            normalized = (value - variable["low"]) / (variable["high"] - variable["low"])
        return {"a": normalized, "b": -normalized}

    result = optimize_strategy(
        problem,
        finite_metrics,
        evaluator_id=f"tests:bayes-{domain}",
        config=RunConfig(9, seed=13, batch_size=9),
        strategy="bayes",
    )
    assert result.stop_reason == "budget"
    assert len(result.trials) == len({trial.point.key for trial in result.trials}) == 9


def test_discrete_hard_constraint_case_exercises_recovery_without_ranking_claim() -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {
                "n": {"kind": "integer", "low": 0, "high": 11, "default": 3},
                "mode": {
                    "kind": "choice",
                    "values": ["plain", "boost"],
                    "default": "plain",
                },
            },
            "objectives": [
                {"metric": "left", "goal": "min", "scale": 1.0},
                {"metric": "right", "goal": "min", "scale": 1.0},
            ],
            "constraints": [{"metric": "gate", "relation": "ge", "limit": 8.0, "scale": 1.0}],
        }
    )

    def discrete_metrics(point: Mapping[str, Scalar]) -> dict[str, float]:
        count = int(point["n"])
        boost = 2 if point["mode"] == "boost" else 0
        return {
            "left": abs(count - 2.0),
            "right": abs(count - 10.0),
            "gate": float(count + boost),
        }

    result = optimize_strategy(
        problem,
        discrete_metrics,
        evaluator_id="tests:bayes-discrete-constraint",
        config=RunConfig(14, seed=0, batch_size=8),
        strategy="bayes",
    )
    assert len(result.trials) == len({trial.point.key for trial in result.trials}) == 14
    assert any(trial.feasible for trial in result.trials)
    assert any(not trial.feasible for trial in result.trials)
    assert result.frontier and all(trial.feasible for trial in result.frontier)
    assert all(isinstance(trial.point.values["n"], int) for trial in result.trials)
    assert {trial.point.values["mode"] for trial in result.trials} == {"plain", "boost"}


def test_seeded_scalarization_is_positive_normalized_and_replayable() -> None:
    problem = _problem()
    left = BayesianOptimizer(problem, -123, initial_design_size=1)
    right = BayesianOptimizer(problem, -123, initial_design_size=1)
    for optimizer in (left, right):
        point = optimizer.ask(1)[0]
        optimizer.tell((assess(problem, 0, point, _metrics(point.values)),))
        optimizer.ask(1)

    assert left.scalarization_history == right.scalarization_history
    assert left.scalarization_history[0].step == 0
    weights = left.scalarization_history[0].weights
    assert len(weights) == len(problem.objectives)
    assert all(weight > 0.0 for weight in weights)
    assert math.fsum(weights) == pytest.approx(1.0)


def test_simplex_schedule_handles_the_zero_endpoint_and_reaches_extreme_weights(
    monkeypatch,
) -> None:
    optimizer = BayesianOptimizer(_problem(), 8)

    class EndpointRandom:
        def random(self) -> float:
            return 0.0

    monkeypatch.setattr(
        "biasweave.optimizers.bayesian.random.Random", lambda _seed: EndpointRandom()
    )
    endpoint = optimizer._objective_weights()
    assert endpoint == (0.5, 0.5)

    monkeypatch.undo()
    replay = BayesianOptimizer(_problem(), 8)
    realized = tuple(replay._objective_weights() for _index in range(128))
    assert any(max(weights) > 0.9 for weights in realized)


def test_catalog_bayesian_run_is_budget_exact_unique_and_records_effective_parameters(
    tmp_path,
) -> None:
    problem = _problem()
    output = tmp_path / "bayes"
    result = optimize_strategy(
        problem,
        lambda point: _metrics(point),
        evaluator_id="tests:bayes",
        config=RunConfig(12, seed=19, workers=4, batch_size=8),
        strategy="bayes",
        output_directory=output,
    )
    repeated = optimize_strategy(
        problem,
        lambda point: _metrics(point),
        evaluator_id="tests:bayes",
        config=RunConfig(12, seed=19, workers=1, batch_size=1),
        strategy="bayes",
    )

    assert result.stop_reason == "budget"
    assert len(result.trials) == 12
    assert len({trial.point.key for trial in result.trials}) == 12
    assert [trial.as_dict() for trial in result.trials] == [
        trial.as_dict() for trial in repeated.trials
    ]
    metadata = json.loads((output / "catalog-run.json").read_text(encoding="utf-8"))
    assert metadata["strategy"] == "bayes"
    assert metadata["population_size"] is None
    assert metadata["strategy_parameters"] == {
        "candidate_pool_size": 64,
        "exploration": 0.0,
        "feature_encoding": "canonical-unit-choice-one-hot-v1",
        "feature_limit": 64,
        "initial_design_size": 8,
        "length_scale": 0.25,
        "local_fraction": 0.5,
        "local_radius": 0.15,
        "noise_variance": 1e-06,
        "scalarization_schedule": "seeded-exponential-simplex-v1",
        "scalarization_seed": 19,
        "training_window_size": 128,
    }
    schema = json.loads(
        files("biasweave")
        .joinpath("schemas", "catalog-run-v1.schema.json")
        .read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(metadata)


def test_bayesian_recovers_constraints_before_modeling_feasible_objectives(monkeypatch) -> None:
    problem = _problem(constrained=True)
    optimizer = BayesianOptimizer(problem, 7, initial_design_size=1, candidate_pool_size=4)
    first = optimizer.ask(1)[0]
    optimizer.tell((assess(problem, 0, first, _metrics(first.values, limit=3.0)),))
    captured: list[tuple[float, ...]] = []

    class RecoveryProcess:
        def __init__(self, _points, targets, **_options):
            captured.append(targets)

        def predict(self, _coordinates):
            return GaussianPrediction(mean=0.0, variance=0.5)

    monkeypatch.setattr("biasweave.optimizers.bayesian.GaussianProcess", RecoveryProcess)
    optimizer.ask(1)
    assert captured == [(-1.0,)]
    assert optimizer.scalarization_history == ()


def test_population_size_does_not_apply_to_bayesian() -> None:
    with pytest.raises(ConfigurationError, match="does not apply to bayes"):
        optimize_strategy(
            _problem(),
            lambda point: _metrics(point),
            evaluator_id="tests:bayes-population",
            config=RunConfig(1),
            strategy="bayes",
            population_size=4,
        )


def test_bayesian_is_exported_and_selectable_from_the_cli(tmp_path, capsys) -> None:
    assert PublicBayesianOptimizer is BayesianOptimizer
    output = tmp_path / "cli-bayes"
    assert (
        main(
            [
                "run",
                "--problem",
                "examples/two_stage_ota/problem.toml",
                "--evaluator",
                "python:biasweave.demo:evaluate",
                "--strategy",
                "bayes",
                "--budget",
                "9",
                "--batch-size",
                "4",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert "bayes: budget: 9 evaluations" in capsys.readouterr().out
    metadata = json.loads((output / "catalog-run.json").read_text(encoding="utf-8"))
    assert metadata["strategy"] == "bayes"
    assert not output.joinpath("run.json").exists()


def test_release_wheel_smoke_crosses_the_bayesian_initial_design_boundary() -> None:
    workflow = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "config=RunConfig(9, seed=7, batch_size=4)" in workflow
    assert "len(result.trials) != 9" in workflow
