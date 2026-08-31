from __future__ import annotations

import math

import pytest

from biasweave.dominance import assess, constraint_violation, dominates, failed_trial, pareto_front
from biasweave.encoding import make_point
from biasweave.model import Relation, TrialStatus
from tests.helpers import make_problem


@pytest.mark.parametrize(
    ("value", "relation", "limit", "lower", "upper", "expected"),
    [
        (7.0, Relation.GE, 5.0, None, None, 0.0),
        (3.0, Relation.GE, 5.0, None, None, 1.0),
        (3.0, Relation.LE, 5.0, None, None, 0.0),
        (7.0, Relation.LE, 5.0, None, None, 1.0),
        (4.0, Relation.BETWEEN, None, 3.0, 6.0, 0.0),
        (1.0, Relation.BETWEEN, None, 3.0, 6.0, 1.0),
        (8.0, Relation.BETWEEN, None, 3.0, 6.0, 1.0),
    ],
)
def test_constraint_violation_relations(value, relation, limit, lower, upper, expected):
    actual = constraint_violation(
        value,
        relation,
        limit=limit,
        lower=lower,
        upper=upper,
        scale=2.0,
        tolerance=0.0,
    )
    assert actual == pytest.approx(expected)


def test_constraint_tolerance_widens_boundary():
    violation = constraint_violation(
        4.8,
        Relation.GE,
        limit=5.0,
        lower=None,
        upper=None,
        scale=1.0,
        tolerance=0.25,
    )
    assert violation == 0.0


def trial(trial_id, loss, score, quality=1.0, window=1.0):
    problem = make_problem()
    point = make_point(problem, [0.1 + trial_id * 0.1, 0.5, 0.5])
    return assess(
        problem,
        trial_id,
        point,
        {"loss": loss, "score": score, "quality": quality, "window": window},
    )


def test_assess_builds_minimization_vector_and_feasibility():
    candidate = trial(0, loss=0.2, score=1.6)
    assert candidate.status is TrialStatus.SUCCESS
    assert candidate.feasible
    assert candidate.objective_vector == pytest.approx((0.2, -0.8))
    assert candidate.violation == 0.0


def test_assess_sums_squared_normalized_violations():
    candidate = trial(0, loss=1.0, score=1.0, quality=0.1, window=3.5)
    assert not candidate.feasible
    assert candidate.max_violation == pytest.approx(1.0)
    assert candidate.violation == pytest.approx(1.25)


def test_failed_trial_has_explicit_nonfinite_penalty():
    problem = make_problem()
    point = make_point(problem, [0.5, 0.5, 0.5])
    candidate = failed_trial(4, point, "no convergence")
    assert candidate.status is TrialStatus.FAILED
    assert not candidate.feasible
    assert math.isinf(candidate.violation)
    assert candidate.error == "no convergence"
    assert candidate.as_dict()["violation"] is None


def test_feasible_dominates_infeasible_and_success_dominates_failure():
    feasible = trial(0, 2.0, 0.1)
    infeasible = trial(1, 0.0, 100.0, quality=0.0)
    failure = failed_trial(2, infeasible.point, "failed")
    assert dominates(feasible, infeasible)
    assert not dominates(infeasible, feasible)
    assert dominates(infeasible, failure)
    assert not dominates(failure, infeasible)


def test_infeasible_dominance_uses_total_then_maximum_violation():
    smaller = trial(0, 1, 1, quality=0.1)
    larger = trial(1, 1, 1, quality=-0.1)
    assert dominates(smaller, larger)
    assert not dominates(larger, smaller)


def test_pareto_dominance_requires_no_worse_and_one_strictly_better():
    best = trial(0, 0.1, 2.0)
    dominated = trial(1, 0.2, 1.0)
    tradeoff = trial(2, 0.05, 0.5)
    equal = trial(3, 0.1, 2.0)
    assert dominates(best, dominated)
    assert not dominates(best, tradeoff)
    assert not dominates(best, equal)


def test_pareto_front_excludes_failures_infeasible_and_dominated():
    best = trial(0, 0.1, 2.0)
    dominated = trial(1, 0.2, 1.0)
    tradeoff = trial(2, 0.05, 0.5)
    infeasible = trial(3, 0.0, 4.0, quality=0.0)
    failure = failed_trial(4, infeasible.point, "failed")
    front = pareto_front([failure, dominated, tradeoff, infeasible, best])
    assert [candidate.trial_id for candidate in front] == [0, 2]


def test_reported_front_uses_exact_not_epsilon_dominance():
    better = trial(0, 0.1, 2.0)
    almost = trial(1, 0.1 + 5e-13, 2.0)
    assert dominates(better, almost)
    assert [candidate.trial_id for candidate in pareto_front([almost, better])] == [0]
