from __future__ import annotations

import math
import random

import pytest

from biasweave.dominance import dominates
from biasweave.model import Point, Trial, TrialStatus
from biasweave.optimizers.ranking import (
    annealing_energy,
    crowding_distance,
    non_dominated_sort,
    preference_key,
    preferred,
    select_population,
)


def _trial(
    trial_id: int,
    objectives: tuple[float, ...],
    *,
    feasible: bool = True,
    violation: float = 0.0,
    maximum: float = 0.0,
    failed: bool = False,
) -> Trial:
    return Trial(
        trial_id,
        Point((trial_id / 100.0,), {"x": trial_id}, f"point-{trial_id}"),
        TrialStatus.FAILED if failed else TrialStatus.SUCCESS,
        {} if failed else {"a": 1.0},
        "failed" if failed else None,
        False if failed else feasible,
        math.inf if failed else violation,
        math.inf if failed else maximum,
        () if failed else objectives,
    )


def _oracle_fronts(trials: list[Trial]) -> list[set[int]]:
    remaining = list(trials)
    fronts: list[set[int]] = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                other is not candidate and dominates(other, candidate) for other in remaining
            )
        ]
        fronts.append({trial.trial_id for trial in front})
        remaining = [trial for trial in remaining if trial not in front]
    return fronts


def test_non_dominated_sort_matches_independent_peeling_oracle() -> None:
    generator = random.Random(921)  # nosec B311 - deterministic property sample
    trials = [
        _trial(index, (generator.random(), generator.random(), generator.random()))
        for index in range(30)
    ]
    trials += [
        _trial(30, (0.0, 0.0, 0.0), feasible=False, violation=0.2, maximum=0.2),
        _trial(31, (0.0, 0.0, 0.0), feasible=False, violation=0.4, maximum=0.3),
        _trial(32, (), failed=True),
    ]
    actual = [{trial.trial_id for trial in front} for front in non_dominated_sort(trials)]
    assert actual == _oracle_fronts(trials)
    assert actual == [
        {trial.trial_id for trial in front} for front in non_dominated_sort(tuple(reversed(trials)))
    ]


def test_crowding_distance_has_exact_normalized_interior_values() -> None:
    front = tuple(_trial(index, (float(index), float(3 - index))) for index in range(4))
    distances = crowding_distance(front)
    assert math.isinf(distances["point-0"])
    assert math.isinf(distances["point-3"])
    assert distances["point-1"] == pytest.approx(4.0 / 3.0)
    assert distances["point-2"] == pytest.approx(4.0 / 3.0)


def test_crowding_degenerate_and_small_fronts_are_defined() -> None:
    assert crowding_distance(()) == {}
    singleton = _trial(0, (1.0, 1.0))
    assert crowding_distance((singleton,))[singleton.point.key] == 0.0
    no_objectives = _trial(1, ())
    assert crowding_distance((no_objectives,))[no_objectives.point.key] == 0.0
    mismatched = _trial(2, (1.0,))
    assert crowding_distance((singleton, mismatched)) == {
        singleton.point.key: 0.0,
        mismatched.point.key: 0.0,
    }
    flat = tuple(_trial(index, (1.0, float(index))) for index in range(3))
    distances = crowding_distance(flat)
    assert distances["point-1"] == pytest.approx(1.0)


def test_crowding_skips_constant_axes_and_is_scale_safe_and_permutation_invariant() -> None:
    constant = tuple(_trial(index, (7.0, 7.0)) for index in range(4))
    assert crowding_distance(constant) == {trial.point.key: 0.0 for trial in constant}

    extreme = (
        _trial(10, (-1e308, 1e308)),
        _trial(11, (0.0, 0.0)),
        _trial(12, (1e308, -1e308)),
    )
    forward = crowding_distance(extreme)
    assert forward == crowding_distance(tuple(reversed(extreme)))
    assert math.isinf(forward["point-10"])
    assert math.isinf(forward["point-12"])
    assert math.isfinite(forward["point-11"])
    assert forward["point-11"] >= 0.0


def test_preference_is_constraint_first_and_stably_total() -> None:
    feasible_bad_objectives = _trial(4, (100.0, 100.0))
    infeasible_good_objectives = _trial(
        3, (-100.0, -100.0), feasible=False, violation=0.01, maximum=0.01
    )
    failed = _trial(2, (), failed=True)
    assert preferred(feasible_bad_objectives, infeasible_good_objectives)
    assert preferred(infeasible_good_objectives, failed)
    assert preference_key(feasible_bad_objectives) < preference_key(infeasible_good_objectives)
    assert annealing_energy(feasible_bad_objectives) < annealing_energy(infeasible_good_objectives)
    assert annealing_energy(infeasible_good_objectives) < annealing_energy(failed)
    tied_late = _trial(9, (1.0, 1.0))
    tied_early = _trial(8, (1.0, 1.0))
    assert preferred(tied_early, tied_late)
    assert not preferred(tied_late, tied_early)


def test_population_selection_uses_rank_then_crowding() -> None:
    front = tuple(_trial(index, (float(index), float(4 - index))) for index in range(5))
    selected = select_population(front, 3)
    assert {trial.trial_id for trial in selected} == {0, 1, 4}
    assert select_population(front, 0) == ()
    assert select_population(front, 9) == front
