from __future__ import annotations

import json

import pytest

from biasweave.dominance import assess, failed_trial
from biasweave.encoding import make_point
from biasweave.errors import CheckpointError
from biasweave.ledger import TrialLedger, read_metadata, trial_from_dict, write_metadata
from biasweave.model import OptimizationResult
from biasweave.results import frontier_table, result_data, summary_markdown, write_result
from tests.helpers import evaluator, make_problem


def successful_trial(trial_id=0):
    problem = make_problem()
    point = make_point(problem, [0.5, 0.5, 0.5])
    return assess(problem, trial_id, point, evaluator(point.values))


def test_ledger_round_trip_success_and_failed_trial(tmp_path):
    success = successful_trial(0)
    failure = failed_trial(1, success.point, "simulator stopped")
    ledger = TrialLedger(tmp_path / "trials.jsonl")
    ledger.append([success])
    ledger.append([failure])
    restored = ledger.read()
    assert restored[0] == success
    assert restored[1].error == "simulator stopped"
    assert restored[1].as_dict()["violation"] is None


def test_empty_append_does_not_create_file(tmp_path):
    path = tmp_path / "trials.jsonl"
    TrialLedger(path).append([])
    assert not path.exists()


def test_missing_ledger_reads_as_empty(tmp_path):
    assert TrialLedger(tmp_path / "absent.jsonl").read() == []


def test_ledger_ignores_only_truncated_final_line(tmp_path):
    first = successful_trial(0)
    path = tmp_path / "trials.jsonl"
    path.write_text(json.dumps(first.as_dict()) + '\n{"trial_id":', encoding="utf-8")
    assert TrialLedger(path).read() == [first]
    path.write_text("{bad}\n", encoding="utf-8")
    with pytest.raises(CheckpointError, match="invalid trial ledger line 1"):
        TrialLedger(path).read()


def test_ledger_requires_contiguous_trial_ids(tmp_path):
    record = successful_trial(0).as_dict()
    record["trial_id"] = 2
    path = tmp_path / "trials.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(CheckpointError, match="contiguous"):
        TrialLedger(path).read()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trial_id", -1, "trial_id"),
        ("status", "unknown", "status"),
        ("feasible", 1, "feasible"),
        ("metrics", [], "metrics"),
        ("objective_vector", {}, "objective_vector"),
        ("error", 5, "error"),
    ],
)
def test_trial_from_dict_rejects_malformed_fields(field, value, message):
    record = successful_trial().as_dict()
    record[field] = value
    with pytest.raises(CheckpointError, match=message):
        trial_from_dict(record)


def test_trial_from_dict_rejects_unknown_and_invalid_failed_penalties():
    record = successful_trial().as_dict()
    record["extra"] = 1
    with pytest.raises(CheckpointError, match="unknown fields"):
        trial_from_dict(record)
    failed = failed_trial(0, successful_trial().point, "failed").as_dict()
    failed["violation"] = float("nan")
    with pytest.raises(CheckpointError, match="must be null"):
        trial_from_dict(failed)


def test_metadata_round_trip_replaces_previous_document(tmp_path):
    path = tmp_path / "run.json"
    write_metadata(path, {"version": 1, "count": 2})
    write_metadata(path, {"version": 1, "count": 3})
    assert read_metadata(path) == {"version": 1, "count": 3}
    assert not path.with_suffix(".json.tmp").exists()


def test_read_metadata_rejects_invalid_and_non_object_json(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(CheckpointError, match="must be an object"):
        read_metadata(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(CheckpointError, match="cannot read"):
        read_metadata(path)


def test_result_views_include_counts_frontier_and_variables(tmp_path):
    problem = make_problem()
    success = successful_trial(0)
    failure = failed_trial(1, success.point, "failed")
    result = OptimizationResult(problem, (success, failure), (success,), "budget", "demo", 5)
    data = result_data(result)
    assert data["trial_count"] == 2
    assert data["successful_trials"] == 1
    assert data["failed_trials"] == 1
    summary = summary_markdown(result)
    assert "Feasible Pareto points: 1" in summary
    assert "| Trial | x | n | mode | twice_x | loss | score |" in summary
    table = frontier_table((success,))
    assert "trial  objectives  variables" in table
    assert '"mode":"quiet"' in table
    frontier_path, summary_path = write_result(result, tmp_path / "result")
    assert json.loads(frontier_path.read_text())["stop_reason"] == "budget"
    assert summary_path.read_text().startswith("# BiasWeave run summary")


def test_empty_result_views_explain_absence_of_frontier():
    problem = make_problem()
    result = OptimizationResult(problem, (), (), "wall_time", "demo", 0)
    assert "No feasible point" in summary_markdown(result)
    assert frontier_table(()) == "No feasible Pareto points."
