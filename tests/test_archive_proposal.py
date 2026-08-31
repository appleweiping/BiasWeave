from __future__ import annotations

import pytest

from biasweave.archive import Archive
from biasweave.dominance import assess, failed_trial
from biasweave.encoding import make_point
from biasweave.proposal import ProposalGenerator
from tests.helpers import evaluator, make_problem


def make_trial(trial_id, coordinates, metrics=None):
    problem = make_problem()
    point = make_point(problem, coordinates)
    return assess(problem, trial_id, point, metrics or evaluator(point.values))


def test_archive_tracks_successful_trials_and_frontier_change():
    problem = make_problem()
    archive = Archive(problem)
    first = make_trial(0, [0.5, 0.5, 0.5])
    assert archive.add(first)
    assert archive.frontier == (first,)
    failure = failed_trial(1, first.point, "failed")
    assert not archive.add(failure)
    assert archive.trials == (first,)


def test_archive_selects_lowest_violation_infeasible():
    problem = make_problem()
    bad = make_trial(
        0,
        [0.1, 0.5, 0.5],
        {"loss": 1, "score": 1, "quality": -1, "window": 1},
    )
    closer = make_trial(
        1,
        [0.2, 0.5, 0.5],
        {"loss": 1, "score": 1, "quality": 0.3, "window": 1},
    )
    archive = Archive(problem, [bad, closer])
    assert archive.best_infeasible() is closer
    assert archive.frontier == ()


def test_sparse_anchor_handles_empty_single_and_tradeoff_fronts():
    problem = make_problem()
    archive = Archive(problem)
    assert archive.sparse_anchor() is None
    first = make_trial(0, [0.5, 0.5, 0.5])
    archive.add(first)
    assert archive.sparse_anchor() is first
    second = make_trial(
        1,
        [0.7, 0.5, 0.5],
        {"loss": 0.01, "score": 0.1, "quality": 1, "window": 1},
    )
    archive.add(second)
    assert archive.sparse_anchor() in (first, second)


def test_proposal_first_point_is_problem_default():
    problem = make_problem()
    generator = ProposalGenerator(problem, seed=7)
    points = generator.propose(1, Archive(problem), (), set())
    assert len(points) == 1
    assert points[0].values["x"] == pytest.approx(0.5)
    assert points[0].values["n"] == 2
    assert points[0].values["mode"] == "fast"


def test_proposals_are_unique_and_deterministic_for_seed():
    problem = make_problem()
    first = ProposalGenerator(problem, seed=11).propose(12, Archive(problem), (), set())
    second = ProposalGenerator(problem, seed=11).propose(12, Archive(problem), (), set())
    assert [point.key for point in first] == [point.key for point in second]
    assert len({point.key for point in first}) == len(first)


def test_seen_keys_are_not_proposed_again():
    problem = make_problem()
    generator = ProposalGenerator(problem, seed=0)
    default = generator.propose(1, Archive(problem), (), set())[0]
    later = generator.propose(3, Archive(problem), (), {default.key})
    assert default.key not in {point.key for point in later}


def test_snapshot_restore_reproduces_future_random_proposals():
    problem = make_problem()
    archive = Archive(problem)
    generator = ProposalGenerator(problem, seed=19)
    initial = generator.propose(5, archive, (), set())
    for trial_id, point in enumerate(initial):
        archive.add(assess(problem, trial_id, point, evaluator(point.values)))
    snapshot = generator.snapshot()
    seen = {point.key for point in initial}
    expected = generator.propose(6, archive, archive.trials, seen)
    restored = ProposalGenerator(problem, seed=19)
    restored.restore_snapshot(snapshot)
    actual = restored.propose(6, archive, archive.trials, seen)
    assert [point.key for point in actual] == [point.key for point in expected]


def test_observe_adapts_radius_inside_bounds():
    generator = ProposalGenerator(make_problem(), seed=0)
    start = generator.radius
    generator.observe(True)
    assert generator.radius > start
    for _ in range(100):
        generator.observe(True)
    assert generator.radius == pytest.approx(0.35)
    for _ in range(100):
        generator.observe(False)
    assert generator.radius == pytest.approx(0.015)


def test_repair_strand_uses_infeasible_history_without_leaving_cube():
    problem = make_problem()
    archive = Archive(problem)
    trials = []
    for trial_id, coordinate in enumerate((0.05, 0.15, 0.3)):
        candidate = make_trial(
            trial_id,
            [coordinate, 0.5, 0.5],
            {"loss": 1, "score": 1, "quality": coordinate, "window": 1},
        )
        archive.add(candidate)
        trials.append(candidate)
    generator = ProposalGenerator(problem, seed=2)
    generator.proposal_index = 4
    point = generator.propose(1, archive, tuple(trials), {trial.point.key for trial in trials})[0]
    assert all(0.0 <= coordinate <= 1.0 for coordinate in point.coordinates)
