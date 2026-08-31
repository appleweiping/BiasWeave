"""Constraint assessment and exact Pareto relations."""

from __future__ import annotations

import math
from collections.abc import Mapping

from biasweave.errors import ConfigurationError
from biasweave.model import (
    Goal,
    Point,
    Problem,
    Relation,
    Trial,
    TrialStatus,
)


def constraint_violation(
    value: float,
    relation: Relation,
    *,
    limit: float | None,
    lower: float | None,
    upper: float | None,
    scale: float,
    tolerance: float,
) -> float:
    """Return a non-negative normalized constraint violation."""
    if relation is Relation.GE:
        if limit is None:
            raise ConfigurationError("greater-than constraint requires a limit")
        raw = limit - value - tolerance
    elif relation is Relation.LE:
        if limit is None:
            raise ConfigurationError("less-than constraint requires a limit")
        raw = value - limit - tolerance
    else:
        if lower is None or upper is None:
            raise ConfigurationError("range constraint requires lower and upper bounds")
        raw = max(lower - value - tolerance, value - upper - tolerance)
    return max(0.0, raw / scale)


def assess(problem: Problem, trial_id: int, point: Point, metrics: Mapping[str, float]) -> Trial:
    violations = [
        constraint_violation(
            metrics[constraint.metric],
            constraint.relation,
            limit=constraint.limit,
            lower=constraint.lower,
            upper=constraint.upper,
            scale=constraint.scale,
            tolerance=constraint.tolerance,
        )
        for constraint in problem.constraints
    ]
    try:
        total = math.fsum(value * value for value in violations)
    except ArithmeticError as error:
        raise ArithmeticError("constraint assessment overflowed") from error
    maximum = max(violations, default=0.0)
    vector = tuple(
        ((metrics[objective.metric] - objective.reference) / objective.scale)
        * (1.0 if objective.goal is Goal.MIN else -1.0)
        for objective in problem.objectives
    )
    if (
        not math.isfinite(total)
        or not math.isfinite(maximum)
        or not all(math.isfinite(value) for value in vector)
    ):
        raise ArithmeticError("objective or constraint assessment is non-finite")
    return Trial(
        trial_id,
        point,
        TrialStatus.SUCCESS,
        dict(metrics),
        None,
        total == 0.0,
        total,
        maximum,
        vector,
    )


def failed_trial(trial_id: int, point: Point, error: str) -> Trial:
    return Trial(
        trial_id,
        point,
        TrialStatus.FAILED,
        {},
        error,
        False,
        math.inf,
        math.inf,
        (),
    )


def dominates(left: Trial, right: Trial, *, epsilon: float = 0.0) -> bool:
    """Apply feasibility-first dominance with strict objective improvement."""
    if left.status is TrialStatus.FAILED:
        return False
    if right.status is TrialStatus.FAILED:
        return True
    if left.feasible != right.feasible:
        return left.feasible
    if not left.feasible:
        if left.violation < right.violation - epsilon:
            return True
        return (
            abs(left.violation - right.violation) <= epsilon
            and left.max_violation < right.max_violation - epsilon
        )
    no_worse = all(
        a <= b + epsilon for a, b in zip(left.objective_vector, right.objective_vector, strict=True)
    )
    strictly_better = any(
        a < b - epsilon for a, b in zip(left.objective_vector, right.objective_vector, strict=True)
    )
    return no_worse and strictly_better


def pareto_front(trials: list[Trial] | tuple[Trial, ...]) -> tuple[Trial, ...]:
    successful = [
        trial for trial in trials if trial.status is TrialStatus.SUCCESS and trial.feasible
    ]
    frontier = [
        trial
        for trial in successful
        if not any(other is not trial and dominates(other, trial) for other in successful)
    ]
    return tuple(sorted(frontier, key=lambda trial: trial.trial_id))
