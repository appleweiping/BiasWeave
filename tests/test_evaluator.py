from __future__ import annotations

import base64
import sys
import time

import pytest

import biasweave.evaluator as evaluator_module
from biasweave._strict_json import JSONLimits
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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"loss":1,"loss":2}', "duplicate key"),
        ('{"loss":NaN}', "non-finite"),
        ('{"loss":' + "9" * 129 + "}", "number longer"),
        ("[" * 17 + "0" + "]" * 17, "complexity"),
        (" " * 1_048_577, "byte limit"),
    ],
    ids=["duplicate", "nonfinite", "huge-number", "deep", "oversized"],
)
def test_command_evaluator_rejects_strict_json_violations(payload, message):
    if len(payload) < 30_000:
        encoded = base64.b64encode(payload.encode()).decode("ascii")
        program = f"import base64,sys;sys.stdout.buffer.write(base64.b64decode('{encoded}'))"
    else:
        program = f"import sys;sys.stdout.write(' '*{len(payload)})"
    with pytest.raises(EvaluationError, match=message):
        CommandEvaluator([sys.executable, "-c", program])({})


def test_command_evaluator_kills_a_timeout_and_bounds_both_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(EvaluationError, match="timeout"):
        CommandEvaluator(
            [sys.executable, "-c", "import time;time.sleep(10)"], timeout_seconds=0.05
        )({})

    monkeypatch.setattr(
        evaluator_module,
        "_EVALUATOR_JSON_LIMITS",
        JSONLimits(max_bytes=1_024, max_depth=16, max_nodes=10_000, max_number_characters=128),
    )
    monkeypatch.setattr(evaluator_module, "_MAX_EVALUATOR_STDERR_BYTES", 1_024)
    program = (
        "import sys,threading;"
        "a=lambda s:(s.buffer.write(b'x'*4096),s.flush());"
        "t=threading.Thread(target=a,args=(sys.stdout,));t.start();a(sys.stderr);t.join()"
    )
    with pytest.raises(EvaluationError, match=r"(stdout|stderr) exceeds 1024 byte limit"):
        CommandEvaluator([sys.executable, "-c", program], timeout_seconds=5)({})


def test_command_evaluator_timeout_applies_while_child_does_not_read_stdin() -> None:
    evaluator = CommandEvaluator(
        [sys.executable, "-c", "import time;time.sleep(10)"], timeout_seconds=0.05
    )
    started = time.monotonic()
    with pytest.raises(EvaluationError, match="timeout"):
        evaluator({"payload": "x" * 200_000})
    # Process teardown is slower under Windows coverage and loaded CI hosts;
    # this still proves cancellation happened well before the 10-second child.
    assert time.monotonic() - started < 4.0


def test_command_evaluator_kills_as_soon_as_a_pipe_exceeds_its_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator_module,
        "_EVALUATOR_JSON_LIMITS",
        JSONLimits(max_bytes=1_024, max_depth=16, max_nodes=10_000, max_number_characters=128),
    )
    program = "import sys,time;sys.stdout.write('x'*4096);sys.stdout.flush();time.sleep(10)"
    started = time.monotonic()
    with pytest.raises(EvaluationError, match="stdout exceeds 1024 byte limit"):
        CommandEvaluator([sys.executable, "-c", program], timeout_seconds=5)({})
    # Leave deterministic scheduler headroom while staying below the adapter's
    # five-second timeout (the overflow path must terminate the child early).
    assert time.monotonic() - started < 4.0
