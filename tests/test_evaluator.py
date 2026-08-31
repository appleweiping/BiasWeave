from __future__ import annotations

import subprocess
import sys

import pytest

from biasweave.errors import EvaluationError
from biasweave.evaluator import CommandEvaluator, load_python_evaluator, validate_metrics
from tests.helpers import make_problem


def complete_metrics():
    return {"loss": 1, "score": 2, "quality": 3, "window": 1, "extra": 9}


def test_validate_metrics_accepts_extra_finite_metrics_and_converts_numbers():
    metrics = validate_metrics(make_problem(), complete_metrics())
    assert metrics == {"loss": 1.0, "score": 2.0, "quality": 3.0, "window": 1.0, "extra": 9.0}


@pytest.mark.parametrize("raw", [None, [], "metrics"])
def test_validate_metrics_requires_mapping(raw):
    with pytest.raises(EvaluationError, match="metric mapping"):
        validate_metrics(make_problem(), raw)


@pytest.mark.parametrize("value", [True, "one", None, float("nan"), float("inf")])
def test_validate_metrics_rejects_invalid_values(value):
    metrics = complete_metrics()
    metrics["loss"] = value
    with pytest.raises(EvaluationError, match="metric 'loss'"):
        validate_metrics(make_problem(), metrics)


def test_validate_metrics_reports_all_missing_required_metrics():
    with pytest.raises(EvaluationError, match="loss, quality, score, window"):
        validate_metrics(make_problem(), {})


def test_validate_metrics_rejects_empty_metric_name():
    metrics = complete_metrics()
    metrics[""] = 3
    with pytest.raises(EvaluationError, match="non-empty strings"):
        validate_metrics(make_problem(), metrics)


def test_load_python_evaluator_loads_callable():
    evaluator = load_python_evaluator("python:biasweave.demo:evaluate")
    assert callable(evaluator)


@pytest.mark.parametrize(
    "specification",
    ["biasweave.demo:evaluate", "python:no_colon", "python::evaluate"],
)
def test_load_python_evaluator_rejects_bad_form(specification):
    with pytest.raises(EvaluationError, match="python:module:function"):
        load_python_evaluator(specification)


def test_load_python_evaluator_reports_missing_target():
    with pytest.raises(EvaluationError, match="cannot load"):
        load_python_evaluator("python:biasweave.demo:absent")


def test_command_evaluator_round_trip_uses_json_without_shell():
    program = (
        "import json,sys; p=json.load(sys.stdin); "
        "json.dump({'loss':p['x'],'score':2,'quality':3,'window':1},sys.stdout)"
    )
    evaluator = CommandEvaluator([sys.executable, "-c", program], timeout_seconds=5)
    result = evaluator({"x": 0.25})
    assert result["loss"] == pytest.approx(0.25)
    assert evaluator.argv[0] == sys.executable


@pytest.mark.parametrize("argv", [[], [""], ["python", ""]])
def test_command_evaluator_rejects_invalid_argv(argv):
    with pytest.raises(EvaluationError, match="non-empty strings"):
        CommandEvaluator(argv)


@pytest.mark.parametrize("timeout", [0, -1, float("nan")])
def test_command_evaluator_rejects_invalid_timeout(timeout):
    with pytest.raises(EvaluationError, match="positive and finite"):
        CommandEvaluator([sys.executable], timeout)


def test_command_evaluator_reports_nonzero_exit():
    evaluator = CommandEvaluator(
        [sys.executable, "-c", "import sys;sys.stderr.write('broken');sys.exit(7)"]
    )
    with pytest.raises(EvaluationError, match="exited with 7: broken"):
        evaluator({})


def test_command_evaluator_reports_invalid_json_and_non_object():
    invalid = CommandEvaluator([sys.executable, "-c", "print('not-json')"])
    with pytest.raises(EvaluationError, match="invalid JSON"):
        invalid({})
    array = CommandEvaluator([sys.executable, "-c", "print('[]')"])
    with pytest.raises(EvaluationError, match="must be an object"):
        array({})


def test_command_evaluator_wraps_timeout(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("tool", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(EvaluationError, match="failed to run"):
        CommandEvaluator(["tool"], 1)({})
