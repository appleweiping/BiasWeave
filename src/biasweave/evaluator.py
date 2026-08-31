"""Trusted Python and no-shell JSON subprocess evaluator adapters."""

from __future__ import annotations

import importlib
import json
import math
import subprocess  # nosec B404
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from biasweave._strict_json import JSONLimits, StrictJSONError, loads_strict_json
from biasweave.errors import EvaluationError
from biasweave.model import Problem, Scalar

_EVALUATOR_JSON_LIMITS = JSONLimits(
    max_bytes=1_048_576,
    max_depth=16,
    max_nodes=10_000,
    max_number_characters=128,
)


class Evaluator(Protocol):
    def __call__(self, point: Mapping[str, Scalar]) -> Mapping[str, float]: ...


def validate_metrics(problem: Problem, raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise EvaluationError("evaluator result must be a metric mapping")
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise EvaluationError("metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise EvaluationError(f"metric {name!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise EvaluationError(f"metric {name!r} must be finite")
        metrics[name] = numeric
    required = {objective.metric for objective in problem.objectives}
    required.update(constraint.metric for constraint in problem.constraints)
    missing = sorted(required - set(metrics))
    if missing:
        raise EvaluationError(f"evaluator omitted required metrics: {', '.join(missing)}")
    return metrics


def load_python_evaluator(
    specification: str,
) -> Callable[[Mapping[str, Scalar]], Mapping[str, float]]:
    """Load a trusted `python:module.path:function` evaluator."""
    prefix, separator, target = specification.partition(":")
    if prefix != "python" or not separator:
        raise EvaluationError("evaluator must have python:module:function form")
    module_name, separator, attribute_name = target.rpartition(":")
    if not separator or not module_name or not attribute_name:
        raise EvaluationError("evaluator must have python:module:function form")
    try:
        module = importlib.import_module(module_name)
        evaluator = getattr(module, attribute_name)
    except (ImportError, AttributeError) as error:
        raise EvaluationError(f"cannot load evaluator {specification}: {error}") from error
    if not callable(evaluator):
        raise EvaluationError(f"evaluator target is not callable: {specification}")
    return cast(Callable[[Mapping[str, Scalar]], Mapping[str, float]], evaluator)


@dataclass(frozen=True, slots=True)
class CommandEvaluator:
    """Run one trusted executable per point with JSON stdin/stdout."""

    argv: tuple[str, ...]
    timeout_seconds: float = 300.0

    def __init__(self, argv: Sequence[str], timeout_seconds: float = 300.0):
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise EvaluationError("command argv must contain non-empty strings")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise EvaluationError("command timeout must be positive and finite")
        object.__setattr__(self, "argv", tuple(argv))
        object.__setattr__(self, "timeout_seconds", float(timeout_seconds))

    def __call__(self, point: Mapping[str, Scalar]) -> Mapping[str, float]:
        try:
            completed = subprocess.run(  # nosec B603
                self.argv,
                input=json.dumps(dict(point), sort_keys=True),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise EvaluationError(f"evaluator command failed to run: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-500:]
            raise EvaluationError(
                f"evaluator command exited with {completed.returncode}: {detail or 'no stderr'}"
            )
        try:
            result = loads_strict_json(
                completed.stdout,
                limits=_EVALUATOR_JSON_LIMITS,
                context="evaluator command JSON",
            )
        except StrictJSONError as error:
            raise EvaluationError(f"evaluator command returned invalid JSON: {error}") from error
        if not isinstance(result, Mapping):
            raise EvaluationError("evaluator command JSON must be an object")
        return result
