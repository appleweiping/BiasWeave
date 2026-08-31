from __future__ import annotations

import json
import threading
import time

import pytest

from biasweave.engine import optimize, problem_fingerprint, validate_run_config
from biasweave.errors import CheckpointError, ConfigurationError
from biasweave.ledger import TrialLedger, read_metadata, write_metadata
from biasweave.model import RunConfig, TrialStatus
from biasweave.problem import parse_problem
from tests.helpers import evaluator, failed_evaluator, make_problem, problem_data


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (RunConfig(0), "budget"),
        (RunConfig(1, seed=True), "seed"),
        (RunConfig(1, workers=0), "workers"),
        (RunConfig(1, batch_size=0), "batch_size"),
        (RunConfig(1, max_stagnation=-1), "max_stagnation"),
        (RunConfig(1, wall_time_seconds=0), "wall_time_seconds"),
        (RunConfig(1, command_timeout_seconds=float("nan")), "command_timeout_seconds"),
    ],
)
def test_validate_run_config_rejects_invalid_values(config, message):
    with pytest.raises(ConfigurationError, match=message):
        validate_run_config(config)


def test_problem_fingerprint_uses_source_hash_or_stable_structure():
    problem = make_problem()
    assert problem_fingerprint(problem) == "fixture-hash"
    first = parse_problem(problem_data())
    second = parse_problem(problem_data())
    assert len(problem_fingerprint(first)) == 64
    assert problem_fingerprint(first) == problem_fingerprint(second)


def test_optimize_evaluates_budget_and_returns_feasible_front():
    result = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(24, seed=3, workers=1, batch_size=4),
    )
    assert len(result.trials) == 24
    assert result.stop_reason == "budget"
    assert result.successful_trials == 24
    assert result.failed_trials == 0
    assert result.frontier
    assert all(trial.feasible for trial in result.frontier)


def test_parallel_results_are_committed_in_trial_id_order_and_use_workers():
    lock = threading.Lock()
    thread_names = set()

    def concurrent(point):
        with lock:
            thread_names.add(threading.current_thread().name)
        time.sleep(0.002 if float(point["x"]) < 0.5 else 0.001)
        return evaluator(point)

    result = optimize(
        make_problem(),
        concurrent,
        evaluator_id="tests:concurrent",
        config=RunConfig(12, seed=4, workers=3, batch_size=6),
    )
    assert [trial.trial_id for trial in result.trials] == list(range(12))
    assert any(name.startswith("biasweave") for name in thread_names)


def test_same_seed_is_deterministic_across_worker_counts():
    serial = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(20, seed=77, workers=1, batch_size=5),
    )
    parallel = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(20, seed=77, workers=4, batch_size=5),
    )
    assert [trial.as_dict() for trial in serial.trials] == [
        trial.as_dict() for trial in parallel.trials
    ]
    assert [trial.trial_id for trial in serial.frontier] == [
        trial.trial_id for trial in parallel.frontier
    ]


def test_evaluator_exception_and_bad_metrics_become_failed_trials():
    failed = optimize(
        make_problem(),
        failed_evaluator,
        evaluator_id="tests:failed",
        config=RunConfig(3),
    )
    assert failed.failed_trials == 3
    assert all(trial.status is TrialStatus.FAILED for trial in failed.trials)
    assert "RuntimeError: synthetic evaluator failure" in failed.trials[0].error

    missing = optimize(
        make_problem(),
        lambda _point: {},
        evaluator_id="tests:missing",
        config=RunConfig(1),
    )
    assert missing.trials[0].status is TrialStatus.FAILED
    assert "omitted required metrics" in missing.trials[0].error


def test_fresh_checkpoint_writes_all_artifacts_and_refuses_overwrite(tmp_path):
    output = tmp_path / "run"
    result = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(7, seed=2, batch_size=3),
        output_directory=output,
    )
    assert len(TrialLedger(output / "trials.jsonl").read()) == 7
    assert read_metadata(output / "run.json")["completed_trials"] == 7
    assert json.loads((output / "frontier.json").read_text())["trial_count"] == 7
    assert (output / "summary.md").is_file()
    assert len(result.trials) == 7
    with pytest.raises(CheckpointError, match="already contains a checkpoint"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(8, seed=2, batch_size=3),
            output_directory=output,
        )


def test_resume_matches_uninterrupted_sequence(tmp_path):
    output = tmp_path / "resumed"
    initial = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(8, seed=91, batch_size=4),
        output_directory=output,
    )
    assert len(initial.trials) == 8
    resumed = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(20, seed=91, workers=3, batch_size=4),
        output_directory=output,
        resume=True,
    )
    uninterrupted = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(20, seed=91, batch_size=4),
    )
    assert [trial.as_dict() for trial in resumed.trials] == [
        trial.as_dict() for trial in uninterrupted.trials
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("problem_sha256", "other", "problem_sha256"),
        ("evaluator_id", "other", "evaluator_id"),
        ("seed", 42, "seed"),
        ("batch_size", 99, "batch_size"),
        ("schema_version", 99, "schema"),
    ],
)
def test_resume_rejects_incompatible_metadata(tmp_path, field, value, message):
    output = tmp_path / "run"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(4, seed=1, batch_size=2),
        output_directory=output,
    )
    metadata = read_metadata(output / "run.json")
    metadata[field] = value
    write_metadata(output / "run.json", metadata)
    with pytest.raises(CheckpointError, match=message):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(6, seed=1, batch_size=2),
            output_directory=output,
            resume=True,
        )


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_resume_rejects_non_integer_metadata_schema(tmp_path, schema_version):
    output = tmp_path / "schema"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(2, batch_size=2),
        output_directory=output,
    )
    metadata = read_metadata(output / "run.json")
    metadata["schema_version"] = schema_version
    write_metadata(output / "run.json", metadata)
    with pytest.raises(CheckpointError, match="metadata schema"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(3, batch_size=2),
            output_directory=output,
            resume=True,
        )


def test_resume_rejects_unknown_metadata_fields(tmp_path):
    output = tmp_path / "unknown-field"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(2, batch_size=2),
        output_directory=output,
    )
    metadata = read_metadata(output / "run.json")
    metadata["injected"] = True
    write_metadata(output / "run.json", metadata)
    with pytest.raises(CheckpointError, match="metadata fields"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(3, batch_size=2),
            output_directory=output,
            resume=True,
        )


def test_resume_rejects_missing_metadata_count_disagreement_and_smaller_budget(tmp_path):
    with pytest.raises(CheckpointError, match="does not exist"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(2),
            output_directory=tmp_path / "missing",
            resume=True,
        )
    output = tmp_path / "run"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(4, batch_size=2),
        output_directory=output,
    )
    metadata = read_metadata(output / "run.json")
    metadata["completed_trials"] = 3
    write_metadata(output / "run.json", metadata)
    with pytest.raises(CheckpointError, match="disagree"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(5, batch_size=2),
            output_directory=output,
            resume=True,
        )
    metadata["completed_trials"] = 4
    write_metadata(output / "run.json", metadata)
    with pytest.raises(ConfigurationError, match="less than completed"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(3, batch_size=2),
            output_directory=output,
            resume=True,
        )


def test_resume_requires_output_and_evaluator_id_is_required():
    with pytest.raises(ConfigurationError, match="output directory"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(1),
            resume=True,
        )
    with pytest.raises(ConfigurationError, match="evaluator_id"):
        optimize(make_problem(), evaluator, evaluator_id="", config=RunConfig(1))


def test_resume_rejects_tampered_point_and_generator_state(tmp_path):
    point_output = tmp_path / "point"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(2, batch_size=2),
        output_directory=point_output,
    )
    ledger_path = point_output / "trials.jsonl"
    records = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    records[0]["point"]["values"]["x"] = 0.9
    ledger_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    with pytest.raises(CheckpointError, match="point identity"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(3, batch_size=2),
            output_directory=point_output,
            resume=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("coverage_index", True), ("proposal_index", -1), ("radius", float("nan"))],
)
def test_resume_rejects_malformed_generator_fields(tmp_path, field, value):
    output = tmp_path / field
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(2, batch_size=2),
        output_directory=output,
    )
    metadata = read_metadata(output / "run.json")
    metadata["generator"][field] = value
    (output / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(CheckpointError, match="proposal-generator state"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(3, batch_size=2),
            output_directory=output,
            resume=True,
        )

    state_output = tmp_path / "state"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(2, batch_size=2),
        output_directory=state_output,
    )
    metadata = read_metadata(state_output / "run.json")
    metadata["generator"] = {"coverage_index": 1}
    write_metadata(state_output / "run.json", metadata)
    with pytest.raises(CheckpointError, match="proposal-generator state"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(3, batch_size=2),
            output_directory=state_output,
            resume=True,
        )


def test_wall_time_can_stop_before_first_batch():
    moments = iter([0.0, 1.0])
    result = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(10, wall_time_seconds=0.5),
        clock=lambda: next(moments),
    )
    assert result.stop_reason == "wall_time"
    assert result.trials == ()


def test_resume_recovers_durable_ledger_ahead_of_metadata(tmp_path):
    output = tmp_path / "recover"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(2, batch_size=2),
        output_directory=output,
    )
    old_metadata = read_metadata(output / "run.json")
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(4, batch_size=2),
        output_directory=output,
        resume=True,
    )
    write_metadata(output / "run.json", old_metadata)
    recovered = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(6, batch_size=2),
        output_directory=output,
        resume=True,
    )
    assert len(recovered.trials) == 6
    assert read_metadata(output / "run.json")["completed_trials"] == 6


def test_resume_completes_a_partially_written_batch_without_sequence_drift(tmp_path):
    uninterrupted = optimize(
        make_problem(), evaluator, evaluator_id="tests:evaluator", config=RunConfig(8, batch_size=3)
    )
    output = tmp_path / "partial"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(3, batch_size=3),
        output_directory=output,
    )
    old_metadata = read_metadata(output / "run.json")
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(6, batch_size=3),
        output_directory=output,
        resume=True,
    )
    ledger = output / "trials.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    ledger.write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")
    write_metadata(output / "run.json", old_metadata)
    resumed = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(8, batch_size=3),
        output_directory=output,
        resume=True,
    )
    assert [trial.point.key for trial in resumed.trials] == [
        trial.point.key for trial in uninterrupted.trials
    ]


def test_resume_replays_pending_batch_with_no_durable_trials(tmp_path):
    config = RunConfig(8, seed=7, batch_size=3)
    uninterrupted = optimize(
        make_problem(), evaluator, evaluator_id="tests:evaluator", config=config
    )
    output = tmp_path / "pending-without-ledger-tail"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(3, seed=7, batch_size=3),
        output_directory=output,
    )

    def interrupt_before_commit(point):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        optimize(
            make_problem(),
            interrupt_before_commit,
            evaluator_id="tests:evaluator",
            config=config,
            output_directory=output,
            resume=True,
        )

    metadata = read_metadata(output / "run.json")
    assert len(metadata["pending_keys"]) == 3
    assert len(TrialLedger(output / "trials.jsonl").read()) == 3
    resumed = optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=config,
        output_directory=output,
        resume=True,
    )
    assert [trial.point.key for trial in resumed.trials] == [
        trial.point.key for trial in uninterrupted.trials
    ]


def test_resume_rejects_budget_that_cannot_finish_committed_pending_batch(tmp_path):
    output = tmp_path / "pending-budget"
    optimize(
        make_problem(),
        evaluator,
        evaluator_id="tests:evaluator",
        config=RunConfig(3, batch_size=3),
        output_directory=output,
    )

    def interrupt_before_commit(point):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        optimize(
            make_problem(),
            interrupt_before_commit,
            evaluator_id="tests:evaluator",
            config=RunConfig(8, batch_size=3),
            output_directory=output,
            resume=True,
        )
    with pytest.raises(ConfigurationError, match="committed pending batch"):
        optimize(
            make_problem(),
            evaluator,
            evaluator_id="tests:evaluator",
            config=RunConfig(4, batch_size=3),
            output_directory=output,
            resume=True,
        )


def test_derived_objective_overflow_becomes_failed_trial():
    data = problem_data()
    data["objectives"][0]["reference"] = -1e308
    problem = parse_problem(data)

    def overflowing(point):
        result = evaluator(point)
        result["loss"] = 1e308
        return result

    result = optimize(
        problem,
        overflowing,
        evaluator_id="tests:overflow",
        config=RunConfig(1),
    )
    assert result.trials[0].status.value == "failed"
    assert "non-finite" in (result.trials[0].error or "")


def test_finite_discrete_space_stops_when_exhausted():
    data = problem_data()
    data["variables"] = {"choice": {"kind": "choice", "values": [1, 2]}}
    problem = parse_problem(data)

    def discrete(point):
        value = float(point["choice"])
        return {"loss": value, "score": value, "quality": 1.0, "window": 1.0}

    result = optimize(
        problem,
        discrete,
        evaluator_id="tests:discrete",
        config=RunConfig(10, batch_size=4),
    )
    assert result.stop_reason == "search_space_exhausted"
    assert len(result.trials) == 2
