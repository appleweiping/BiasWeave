from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from biasweave.benchmark import compare_optimizer_catalog, load_analog_benchmark
from biasweave.cli import main
from biasweave.demo import evaluate as demo_evaluator
from biasweave.dominance import assess
from biasweave.encoding import make_point
from biasweave.engine import optimize
from biasweave.errors import CheckpointError, ConfigurationError, ProblemError
from biasweave.model import RunConfig, Scalar, VariableKind
from biasweave.optimizers import StrategyName, create_optimizer
from biasweave.problem import load_problem, parse_problem
from biasweave.search_space import finite_axes
from biasweave.strategy import optimize_strategy
from tests.helpers import evaluator, failed_evaluator, make_problem

POPULATION_STRATEGIES = {
    StrategyName.PSO,
    StrategyName.DE,
    StrategyName.NSGA2,
    StrategyName.MOEAD,
}
CONTRACT = Path("benchmarks/manifest.json")
EXAMPLE = Path("examples/two_stage_ota/problem.toml")


def _population(strategy: StrategyName) -> int | None:
    return 6 if strategy in POPULATION_STRATEGIES else None


@pytest.mark.parametrize("force", [False, True])
def test_catalog_cannot_enter_an_active_weave_output_claim(tmp_path: Path, force: bool) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_evaluator(point: Mapping[str, Scalar]) -> dict[str, float]:
        entered.set()
        assert release.wait(10)
        return evaluator(point)

    output = tmp_path / "shared-output"
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(
            optimize,
            make_problem(),
            blocked_evaluator,
            evaluator_id="tests:active-weave",
            config=RunConfig(1, batch_size=1),
            output_directory=output,
        )
        assert entered.wait(10)
        try:
            with pytest.raises(CheckpointError, match="another writer holds the output claim"):
                optimize_strategy(
                    make_problem(),
                    evaluator,
                    evaluator_id="tests:contending-catalog",
                    config=RunConfig(1, batch_size=1),
                    strategy="random",
                    output_directory=output,
                    force=force,
                )
        finally:
            release.set()
        assert len(active.result(timeout=10).trials) == 1


@pytest.mark.parametrize("force", [False, True])
def test_weave_cannot_enter_an_active_catalog_output_claim(tmp_path: Path, force: bool) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_evaluator(point: Mapping[str, Scalar]) -> dict[str, float]:
        entered.set()
        assert release.wait(10)
        return evaluator(point)

    output = tmp_path / "shared-output"
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(
            optimize_strategy,
            make_problem(),
            blocked_evaluator,
            evaluator_id="tests:active-catalog",
            config=RunConfig(1, batch_size=1),
            strategy="random",
            output_directory=output,
        )
        assert entered.wait(10)
        try:
            with pytest.raises(CheckpointError, match="another writer holds the output claim"):
                optimize(
                    make_problem(),
                    evaluator,
                    evaluator_id="tests:contending-weave",
                    config=RunConfig(1, batch_size=1),
                    output_directory=output,
                    force=force,
                )
        finally:
            release.set()
        assert len(active.result(timeout=10).trials) == 1


@pytest.mark.parametrize("strategy", tuple(StrategyName))
def test_every_strategy_is_budget_exact_unique_and_mixed_variable_safe(strategy) -> None:
    result = optimize_strategy(
        make_problem(),
        evaluator,
        evaluator_id="tests:mixed",
        config=RunConfig(25, seed=41, workers=3, batch_size=5),
        strategy=strategy,
        population_size=_population(strategy),
    )
    assert result.stop_reason == "budget"
    assert len(result.trials) == 25
    assert [trial.trial_id for trial in result.trials] == list(range(25))
    assert len({trial.point.key for trial in result.trials}) == 25
    for trial in result.trials:
        assert 0.01 <= float(trial.point.values["x"]) <= 1.0
        assert isinstance(trial.point.values["n"], int)
        assert trial.point.values["mode"] in {"fast", "quiet"}
        assert trial.point.values["twice_x"] == pytest.approx(2.0 * float(trial.point.values["x"]))
        assert all(0.0 <= coordinate <= 1.0 for coordinate in trial.point.coordinates)
    assert result.frontier
    assert all(trial.feasible for trial in result.frontier)


@pytest.mark.parametrize("strategy", tuple(StrategyName))
def test_every_strategy_accepts_a_wide_finite_real_interval(strategy: StrategyName) -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": {"kind": "real", "low": -1e308, "high": 1e308}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )

    result = optimize_strategy(
        problem,
        lambda point: {"a": float(point["x"]) / 1e308, "b": -float(point["x"]) / 1e308},
        evaluator_id="tests:wide-real",
        config=RunConfig(3, seed=9, batch_size=2),
        strategy=strategy,
        population_size=4 if strategy in POPULATION_STRATEGIES else None,
    )
    assert len(result.trials) == 3
    assert all(-1e308 <= float(trial.point.values["x"]) <= 1e308 for trial in result.trials)


@pytest.mark.parametrize("strategy", tuple(StrategyName))
def test_every_strategy_accepts_an_ulp_narrow_log_interval(strategy: StrategyName) -> None:
    low = 1e308
    high = low
    for _ in range(8):
        high = math.nextafter(high, math.inf)
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": {"kind": "real", "low": low, "high": high, "scale": "log"}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )
    result = optimize_strategy(
        problem,
        lambda point: {
            "a": (float(point["x"]) - low) / (high - low),
            "b": -(float(point["x"]) - low) / (high - low),
        },
        evaluator_id="tests:narrow-log",
        config=RunConfig(4, seed=13, batch_size=2),
        strategy=strategy,
        population_size=4 if strategy in POPULATION_STRATEGIES else None,
    )
    assert len(result.trials) == 4
    assert len({trial.point.key for trial in result.trials}) == 4
    assert all(low <= float(trial.point.values["x"]) <= high for trial in result.trials)


def test_ulp_narrow_quantized_log_interval_has_exact_finite_representatives() -> None:
    low = 1e308
    values = [low]
    for _ in range(8):
        values.append(math.nextafter(values[-1], math.inf))
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {
                "x": {
                    "kind": "real",
                    "low": low,
                    "high": values[-1],
                    "scale": "log",
                    "quantum": values[2] - low,
                }
            },
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )
    axes = finite_axes(problem)
    assert axes is not None
    decoded = {make_point(problem, (coordinate,)).values["x"] for coordinate in axes[0]}
    assert decoded == {values[index] for index in (0, 2, 4, 6, 8)}


def test_default_and_catalog_weave_share_the_same_seeded_trajectory() -> None:
    config = RunConfig(24, seed=73, batch_size=4)
    default = optimize(make_problem(), evaluator, evaluator_id="tests:default", config=config)
    catalog = optimize_strategy(
        make_problem(),
        evaluator,
        evaluator_id="tests:catalog",
        config=config,
        strategy="weave",
    )
    assert default.stop_reason == catalog.stop_reason == "budget"
    assert [trial.point.key for trial in default.trials] == [
        trial.point.key for trial in catalog.trials
    ]


@pytest.mark.parametrize("strategy", tuple(StrategyName))
def test_seeded_sequence_is_independent_of_worker_count(strategy) -> None:
    common = {
        "problem": make_problem(),
        "evaluator": evaluator,
        "evaluator_id": "tests:determinism",
        "strategy": strategy,
        "population_size": _population(strategy),
    }
    serial = optimize_strategy(
        **common,
        config=RunConfig(18, seed=77, workers=1, batch_size=4),
    )
    parallel = optimize_strategy(
        **common,
        config=RunConfig(18, seed=77, workers=4, batch_size=4),
    )
    assert [trial.as_dict() for trial in serial.trials] == [
        trial.as_dict() for trial in parallel.trials
    ]
    assert [trial.trial_id for trial in serial.frontier] == [
        trial.trial_id for trial in parallel.frontier
    ]


def test_seeded_catalog_trajectories_match_the_versioned_golden_fixture() -> None:
    golden = json.loads(Path("tests/data/optimizer-golden-v1.json").read_text(encoding="utf-8"))
    problem = load_problem(golden["problem"])
    for strategy in StrategyName:
        result = optimize_strategy(
            problem,
            demo_evaluator,
            evaluator_id="golden:demo-v1",
            config=RunConfig(
                golden["budget"], seed=golden["seed"], batch_size=golden["batch_size"]
            ),
            strategy=strategy,
            population_size=(
                golden["population_size"] if strategy in POPULATION_STRATEGIES else None
            ),
        )
        assert [trial.point.key for trial in result.trials] == golden["trajectories"][
            strategy.value
        ]


def test_packaged_run_schemas_are_valid_draft_2020_12(tmp_path: Path) -> None:
    schema_root = files("biasweave").joinpath("schemas")
    catalog_schema = json.loads(
        schema_root.joinpath("catalog-run-v1.schema.json").read_text(encoding="utf-8")
    )
    checkpoint_schema = json.loads(
        schema_root.joinpath("weave-checkpoint-v2.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(catalog_schema)
    Draft202012Validator.check_schema(checkpoint_schema)

    problem = load_problem(EXAMPLE)
    catalog_out = tmp_path / "catalog-schema"
    optimize_strategy(
        problem,
        demo_evaluator,
        evaluator_id="tests:schema",
        config=RunConfig(4, seed=2, batch_size=2),
        strategy="de",
        population_size=4,
        output_directory=catalog_out,
    )
    Draft202012Validator(catalog_schema).validate(
        json.loads((catalog_out / "catalog-run.json").read_text(encoding="utf-8"))
    )
    weave_out = tmp_path / "weave-schema"
    optimize(
        problem,
        demo_evaluator,
        evaluator_id="tests:schema",
        config=RunConfig(4, seed=2, batch_size=2),
        output_directory=weave_out,
    )
    Draft202012Validator(checkpoint_schema).validate(
        json.loads((weave_out / "run.json").read_text(encoding="utf-8"))
    )


def test_strategies_do_not_alias_one_proposal_path() -> None:
    sequences = {}
    for strategy in StrategyName:
        result = optimize_strategy(
            make_problem(),
            evaluator,
            evaluator_id="tests:independence",
            config=RunConfig(18, seed=12, batch_size=3),
            strategy=strategy,
            population_size=4 if strategy in POPULATION_STRATEGIES else None,
        )
        sequences[strategy] = tuple(trial.point.key for trial in result.trials)
    assert len(set(sequences.values())) == len(StrategyName)


def _finite_problem():
    return parse_problem(
        {
            "schema_version": 1,
            "variables": {"switch": {"kind": "choice", "values": ["off", "on", "auto", "test"]}},
            "objectives": [
                {"metric": "cost", "goal": "min", "scale": 1.0},
                {"metric": "benefit", "goal": "max", "scale": 1.0},
            ],
            "constraints": [],
        }
    )


def _finite_evaluator(point: Mapping[str, Scalar]) -> dict[str, float]:
    level = {"off": 0.0, "on": 1.0, "auto": 2.0, "test": 3.0}[point["switch"]]
    return {"cost": level, "benefit": level}


@pytest.mark.parametrize("strategy", tuple(StrategyName))
def test_finite_decoded_space_stops_without_duplicate_evaluations(strategy) -> None:
    result = optimize_strategy(
        _finite_problem(),
        _finite_evaluator,
        evaluator_id="tests:finite",
        config=RunConfig(20, seed=5, batch_size=4),
        strategy=strategy,
        population_size=4 if strategy in POPULATION_STRATEGIES else None,
    )
    assert result.stop_reason == "search_space_exhausted"
    assert len(result.trials) == 4
    assert {trial.point.values["switch"] for trial in result.trials} == {
        "off",
        "on",
        "auto",
        "test",
    }


@pytest.mark.parametrize("strategy", tuple(StrategyName))
def test_direct_contract_remembers_told_keys_across_asks(strategy) -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"only": {"kind": "integer", "low": 0, "high": 0}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )
    optimizer = create_optimizer(
        problem,
        strategy,
        seed=0,
        population_size=4 if strategy in POPULATION_STRATEGIES else None,
    )
    point = optimizer.ask(1)[0]
    optimizer.tell((assess(problem, 0, point, {"a": 0.0, "b": 0.0}),))
    assert optimizer.ask(1) == ()
    assert optimizer.empty_ask_reason == "search_space_exhausted"


@pytest.mark.parametrize("strategy", tuple(StrategyName))
def test_integer_domain_fallback_spends_budget_before_proven_exhaustion(strategy) -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": {"kind": "integer", "low": 0, "high": 99}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )

    def evaluate(point):
        value = float(point["x"])
        return {"a": value, "b": value}

    result = optimize_strategy(
        problem,
        evaluate,
        evaluator_id="tests:finite-fallback",
        config=RunConfig(100, seed=0, batch_size=1),
        strategy=strategy,
        population_size=4 if strategy in POPULATION_STRATEGIES else None,
    )
    assert result.stop_reason == "budget"
    assert len(result.trials) == 100
    assert {trial.point.values["x"] for trial in result.trials} == set(range(100))


def test_stochastic_stall_is_not_reported_as_proven_exhaustion() -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": {"kind": "real", "low": 0.0, "high": 1.0}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )
    optimizer = create_optimizer(problem, "random", seed=0)
    optimizer.random_vector = lambda: (0.5,)  # type: ignore[method-assign]
    point = optimizer.ask(1)[0]
    optimizer.tell((assess(problem, 0, point, {"a": 0.0, "b": 0.0}),))
    assert optimizer.ask(1) == ()
    assert optimizer.empty_ask_reason == "proposal_stalled"


@pytest.mark.parametrize("scale,low", [("linear", 0.0), ("log", 0.1)])
def test_quantized_choice_domain_is_enumerated_exactly(scale, low) -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {
                "x": {
                    "kind": "real",
                    "low": low,
                    "high": 1.0,
                    "scale": scale,
                    "quantum": 0.3,
                },
                "mode": {"kind": "choice", "values": ["a", "b"]},
            },
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )
    result = optimize_strategy(
        problem,
        lambda point: {"a": float(point["x"]), "b": float(point["x"])},
        evaluator_id="tests:quantized",
        config=RunConfig(20, seed=0, batch_size=3),
        strategy="de",
        population_size=4,
    )
    assert result.stop_reason == "search_space_exhausted"
    expected = 10 if scale == "linear" else 8
    assert len(result.trials) == expected
    assert (
        len({(trial.point.values["x"], trial.point.values["mode"]) for trial in result.trials})
        == expected
    )


def test_finite_enumerator_has_a_hard_resource_limit() -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": {"kind": "integer", "low": 0, "high": 100_000}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )
    optimizer = create_optimizer(problem, "random", seed=0)
    assert optimizer._finite_axes is None
    points = optimizer.ask(4)
    assert len(points) == 4
    optimizer.tell(
        tuple(
            assess(problem, trial_id, point, {"a": float(trial_id), "b": 0.0})
            for trial_id, point in enumerate(points)
        )
    )

    # Oversized finite domains deliberately use the bounded stochastic path.
    # A stalled operator must neither loop forever nor reuse an evaluated point,
    # and must not misreport the resource-bound case as proven exhaustion.
    optimizer.random_vector = lambda: points[0].coordinates  # type: ignore[method-assign]
    assert optimizer.ask(1) == ()
    assert optimizer.empty_ask_reason == "proposal_stalled"


def test_mutating_evaluator_receives_a_defensive_point_copy() -> None:
    problem = make_problem()

    def mutating(point):
        point["x"] = 999.0
        return {"loss": 1.0, "score": 1.0, "quality": 1.0, "window": 1.0}

    result = optimize_strategy(
        problem,
        mutating,
        evaluator_id="tests:mutating",
        config=RunConfig(3, seed=0, batch_size=1),
        strategy="random",
    )
    assert result.successful_trials == 3
    for trial in result.trials:
        assert trial.point == make_point(problem, trial.point.coordinates)
        assert float(trial.point.values["x"]) <= 1.0


def test_failed_trials_count_toward_budget_and_stagnation() -> None:
    result = optimize_strategy(
        make_problem(),
        failed_evaluator,
        evaluator_id="tests:failure",
        config=RunConfig(20, seed=2, batch_size=2, max_stagnation=2),
        strategy=StrategyName.RANDOM,
    )
    assert result.stop_reason == "stagnation"
    assert len(result.trials) == 2
    assert result.failed_trials == 2
    assert not result.frontier


def test_wall_time_can_stop_before_first_evaluation() -> None:
    times = iter((10.0, 11.0))
    result = optimize_strategy(
        make_problem(),
        evaluator,
        evaluator_id="tests:clock",
        config=RunConfig(10, wall_time_seconds=0.5),
        strategy="random",
        clock=lambda: next(times),
    )
    assert result.stop_reason == "wall_time"
    assert result.trials == ()


def test_catalog_run_persists_ledger_metadata_and_results(tmp_path) -> None:
    output = tmp_path / "catalog"
    result = optimize_strategy(
        make_problem(),
        evaluator,
        evaluator_id="tests:output",
        config=RunConfig(9, seed=6, batch_size=3),
        strategy="de",
        population_size=4,
        output_directory=output,
    )
    assert len((output / "trials.jsonl").read_text(encoding="utf-8").splitlines()) == 9
    metadata = json.loads((output / "catalog-run.json").read_text(encoding="utf-8"))
    assert metadata == {
        "batch_size": 3,
        "budget": 9,
        "command_timeout_seconds": 300.0,
        "completed_trials": 9,
        "evaluator_id": "tests:output",
        "max_stagnation": 0,
        "package_version": "0.4.0",
        "population_size": 4,
        "problem_sha256": "f" * 64,
        "schema_version": 1,
        "seed": 6,
        "stop_reason": "budget",
        "strategy": "de",
        "strategy_parameters": {
            "crossover_rate": 0.9,
            "differential_weight": 0.8,
            "population_size": 4,
        },
        "strategy_schema_version": 1,
        "wall_time_seconds": None,
        "workers": 1,
    }
    assert json.loads((output / "frontier.json").read_text())["trial_count"] == 9
    assert (output / "summary.md").is_file()
    with pytest.raises(CheckpointError, match="refusing to overwrite"):
        optimize_strategy(
            make_problem(),
            evaluator,
            evaluator_id="tests:output",
            config=RunConfig(1),
            strategy="random",
            output_directory=output,
        )
    assert len(result.trials) == 9


def test_catalog_runner_validates_identity_and_configuration() -> None:
    with pytest.raises(ConfigurationError, match="evaluator_id"):
        optimize_strategy(
            make_problem(), evaluator, evaluator_id=" ", config=RunConfig(1), strategy="random"
        )
    with pytest.raises(ConfigurationError, match="budget"):
        optimize_strategy(
            make_problem(), evaluator, evaluator_id="x", config=RunConfig(0), strategy="random"
        )


def test_cli_selects_catalog_strategy_and_rejects_irrelevant_population(tmp_path, capsys) -> None:
    output = tmp_path / "pso"
    assert (
        main(
            [
                "run",
                "--problem",
                str(EXAMPLE),
                "--evaluator",
                "python:biasweave.demo:evaluate",
                "--strategy",
                "pso",
                "--population-size",
                "4",
                "--budget",
                "8",
                "--batch-size",
                "4",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert "pso: budget: 8 evaluations" in capsys.readouterr().out
    assert json.loads((output / "catalog-run.json").read_text())["strategy"] == "pso"
    bad = main(
        [
            "run",
            "--problem",
            str(EXAMPLE),
            "--evaluator",
            "python:biasweave.demo:evaluate",
            "--strategy",
            "weave",
            "--population-size",
            "4",
            "--budget",
            "4",
            "--out",
            str(tmp_path / "bad"),
        ]
    )
    assert bad == 2
    assert "does not apply" in capsys.readouterr().err


def test_catalog_benchmark_is_budget_symmetric_and_content_bound(tmp_path, capsys) -> None:
    benchmark = load_analog_benchmark(CONTRACT)
    comparison = compare_optimizer_catalog(benchmark, budget=8, seed=3, population_size=4)
    assert set(comparison["algorithms"]) == {strategy.value for strategy in StrategyName}
    assert all(summary["evaluations"] == 8 for summary in comparison["algorithms"].values())
    body = dict(comparison)
    digest = body.pop("comparison_sha256")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert digest == hashlib.sha256(canonical.encode("ascii")).hexdigest()

    output = tmp_path / "catalog.json"
    assert (
        main(
            [
                "catalog-benchmark",
                "--contract",
                str(CONTRACT),
                "--budget",
                "4",
                "--population-size",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["budget_per_strategy"] == 4
    assert capsys.readouterr().out == ""


def test_catalog_benchmark_rejects_misleading_budget_settings() -> None:
    benchmark = load_analog_benchmark(CONTRACT)
    with pytest.raises(ProblemError, match="at least 4"):
        compare_optimizer_catalog(benchmark, budget=3, seed=0, population_size=4)
    with pytest.raises(ProblemError, match="population_size"):
        compare_optimizer_catalog(benchmark, budget=4, seed=0, population_size=5)


def test_all_free_variable_kinds_are_exercised_by_fixture() -> None:
    kinds = {variable.kind for variable in make_problem().variables}
    assert kinds == {
        VariableKind.REAL,
        VariableKind.INTEGER,
        VariableKind.CHOICE,
        VariableKind.LINKED,
    }


@pytest.mark.parametrize("strategy", tuple(StrategyName))
def test_every_strategy_preserves_large_integer_values_exactly(strategy: StrategyName) -> None:
    positive = 2**53
    negative = -(2**60)
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {
                "p": {"kind": "integer", "low": positive, "high": positive + 2},
                "n": {"kind": "integer", "low": negative, "high": negative + 2},
            },
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )

    def exact_metrics(point: Mapping[str, Scalar]) -> dict[str, float]:
        assert isinstance(point["p"], int)
        assert isinstance(point["n"], int)
        return {
            "a": float(point["p"] - positive),
            "b": float(point["n"] - negative),
        }

    result = optimize_strategy(
        problem,
        exact_metrics,
        evaluator_id="tests:large-integers",
        config=RunConfig(9, seed=17, batch_size=3),
        strategy=strategy,
        population_size=4 if strategy in POPULATION_STRATEGIES else None,
    )
    assert result.stop_reason == "budget"
    assert len({trial.point.key for trial in result.trials}) == 9
    assert {(trial.point.values["p"], trial.point.values["n"]) for trial in result.trials} == {
        (p, n) for p in range(positive, positive + 3) for n in range(negative, negative + 3)
    }


@pytest.mark.parametrize("strategy", tuple(StrategyName))
def test_every_strategy_accepts_a_wide_integer_interval(strategy: StrategyName) -> None:
    high = 2**60
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": {"kind": "integer", "low": 0, "high": high}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )
    result = optimize_strategy(
        problem,
        lambda point: {
            "a": float(int(point["x"]) // 2**40),
            "b": -float(int(point["x"]) // 2**40),
        },
        evaluator_id="tests:wide-integer",
        config=RunConfig(4, seed=5, batch_size=2),
        strategy=strategy,
        population_size=4 if strategy in POPULATION_STRATEGIES else None,
    )
    assert len({trial.point.key for trial in result.trials}) == 4
    assert all(
        isinstance(trial.point.values["x"], int) and 0 <= int(trial.point.values["x"]) <= high
        for trial in result.trials
    )
