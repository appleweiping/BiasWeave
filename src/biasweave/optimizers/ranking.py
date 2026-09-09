"""Deterministic constraint-first ranking and Pareto population utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from biasweave.dominance import dominates
from biasweave.model import Trial, TrialStatus


def preference_key(trial: Trial) -> tuple[object, ...]:
    """Return a total order that never trades feasibility for objectives."""
    identity = (trial.trial_id, trial.point.key)
    if trial.status is TrialStatus.FAILED:
        return (2, math.inf, math.inf, math.inf, (), *identity)
    if not trial.feasible:
        return (
            1,
            trial.violation,
            trial.max_violation,
            math.fsum(math.tanh(value) for value in trial.objective_vector),
            trial.objective_vector,
            *identity,
        )
    return (
        0,
        0.0,
        0.0,
        math.fsum(math.tanh(value) for value in trial.objective_vector),
        trial.objective_vector,
        *identity,
    )


def preferred(left: Trial, right: Trial) -> bool:
    """Choose one trial deterministically with Pareto dominance first."""
    if dominates(left, right):
        return True
    if dominates(right, left):
        return False
    return preference_key(left) < preference_key(right)


def annealing_energy(trial: Trial) -> float:
    """Map constraint-first quality to separated, finite SA energy bands."""
    if trial.status is TrialStatus.FAILED:
        return 100.0
    if not trial.feasible:
        return (
            10.0
            + 0.5 * math.tanh(math.log1p(trial.violation))
            + 0.5 * math.tanh(trial.max_violation)
        )
    if not trial.objective_vector:
        return 0.0
    return math.fsum(math.tanh(value) for value in trial.objective_vector) / len(
        trial.objective_vector
    )


def non_dominated_sort(trials: Sequence[Trial]) -> tuple[tuple[Trial, ...], ...]:
    """Return exact deterministic fronts using BiasWeave constraint dominance."""
    ordered = tuple(sorted(trials, key=lambda trial: (trial.trial_id, trial.point.key)))
    dominates_indices: list[list[int]] = [[] for _ in ordered]
    domination_counts = [0] * len(ordered)
    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            if dominates(left, right):
                dominates_indices[left_index].append(right_index)
                domination_counts[right_index] += 1
            elif dominates(right, left):
                dominates_indices[right_index].append(left_index)
                domination_counts[left_index] += 1

    pending = [index for index, count in enumerate(domination_counts) if count == 0]
    fronts: list[tuple[Trial, ...]] = []
    emitted = 0
    while pending:
        current = sorted(pending)
        fronts.append(tuple(ordered[index] for index in current))
        emitted += len(current)
        following: list[int] = []
        for index in current:
            for dominated_index in dominates_indices[index]:
                domination_counts[dominated_index] -= 1
                if domination_counts[dominated_index] == 0:
                    following.append(dominated_index)
        pending = following
    if emitted != len(ordered):
        raise RuntimeError("dominance graph unexpectedly contains a cycle")
    return tuple(fronts)


def crowding_distance(front: Sequence[Trial]) -> dict[str, float]:
    """Compute standard normalized NSGA-II crowding distance by point identity."""
    items = tuple(front)
    distances = {trial.point.key: 0.0 for trial in items}
    if not items:
        return distances
    dimensions = len(items[0].objective_vector)
    if dimensions == 0 or any(len(trial.objective_vector) != dimensions for trial in items):
        return distances
    if len(items) <= 2:
        if all(len(trial.objective_vector) == dimensions for trial in items) and any(
            len({trial.objective_vector[index] for trial in items}) > 1
            for index in range(dimensions)
        ):
            return {trial.point.key: math.inf for trial in items}
        return distances
    for dimension in range(dimensions):
        ordered = sorted(
            items,
            key=lambda trial: (
                trial.objective_vector[dimension],
                trial.trial_id,
                trial.point.key,
            ),
        )
        low = ordered[0].objective_vector[dimension]
        high = ordered[-1].objective_vector[dimension]
        if high == low:
            continue
        distances[ordered[0].point.key] = math.inf
        distances[ordered[-1].point.key] = math.inf
        denominator = Fraction.from_float(high) - Fraction.from_float(low)
        for index in range(1, len(ordered) - 1):
            key = ordered[index].point.key
            if math.isfinite(distances[key]):
                previous = ordered[index - 1].objective_vector[dimension]
                following = ordered[index + 1].objective_vector[dimension]
                numerator = Fraction.from_float(following) - Fraction.from_float(previous)
                contribution = float(numerator / denominator)
                if not math.isfinite(contribution) or contribution < 0.0:
                    raise RuntimeError("crowding contribution is not finite and non-negative")
                distances[key] += contribution
    return distances


def select_population(trials: Sequence[Trial], size: int) -> tuple[Trial, ...]:
    """Truncate fronts by crowding, then stable trial identity."""
    if size <= 0:
        return ()
    selected: list[Trial] = []
    for front in non_dominated_sort(trials):
        remaining = size - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
            continue
        distances = crowding_distance(front)
        selected.extend(
            sorted(
                front,
                key=lambda trial: (
                    -distances[trial.point.key],
                    preference_key(trial),
                ),
            )[:remaining]
        )
    return tuple(selected)
