"""Front quality indicators, checked against methods that share no structure."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from biasweave.errors import ConfigurationError
from biasweave.model import Point, Trial, TrialStatus
from biasweave.quality import (
    BASE_RECURSIVE_FRONT,
    DEFAULT_REFERENCE_MARGIN,
    Attainment,
    _hypervolume_2d,
    _nondominated,
    attainment_curve,
    compare_fronts,
    coverage,
    derive_reference_point,
    epsilon_indicator,
    front_quality,
    hypervolume,
    recursive_front_limit,
    spacing,
)
from tests.helpers import make_problem


def trial(
    trial_id: int,
    vector: tuple[float, ...],
    *,
    feasible: bool = True,
    status: TrialStatus = TrialStatus.SUCCESS,
) -> Trial:
    point = Point((float(trial_id),), {"x": float(trial_id)}, f"p{trial_id}")
    return Trial(
        trial_id,
        point,
        status,
        {},
        None,
        feasible,
        0.0 if feasible else 1.0,
        0.0 if feasible else 1.0,
        vector,
    )


def trials(*vectors: tuple[float, ...]) -> list[Trial]:
    return [trial(index, vector) for index, vector in enumerate(vectors)]


def random_points(rng: random.Random, count: int, dimension: int) -> list[tuple[float, ...]]:
    return [tuple(rng.uniform(0.0, 1.0) for _ in range(dimension)) for _ in range(count)]


def inclusion_exclusion(points: list[tuple[float, ...]], reference: tuple[float, ...]) -> float:
    """The textbook definition: a signed sum over every subset of the boxes."""

    total = 0.0
    for size in range(1, len(points) + 1):
        for subset in itertools.combinations(points, size):
            volume = 1.0
            for axis, bound in enumerate(reference):
                volume *= max(0.0, bound - max(point[axis] for point in subset))
            total += volume if size % 2 else -volume
    return total


# ---------------------------------------------------------------------------
# Hypervolume against closed forms.
# ---------------------------------------------------------------------------


def test_one_point_covers_its_own_box() -> None:
    assert hypervolume([(0.25, 0.5)], (1.0, 1.0)) == pytest.approx(0.75 * 0.5)


def test_two_points_add_their_disjoint_parts() -> None:
    # (0.2, 0.8) covers 0.8 * 0.2; (0.6, 0.4) adds 0.4 * 0.4 beyond it.
    assert hypervolume([(0.2, 0.8), (0.6, 0.4)], (1.0, 1.0)) == pytest.approx(0.16 + 0.16)


def test_a_dominated_point_adds_nothing() -> None:
    front = [(0.2, 0.2)]
    assert hypervolume([*front, (0.5, 0.5)], (1.0, 1.0)) == hypervolume(front, (1.0, 1.0))


def test_a_repeated_point_adds_nothing() -> None:
    assert hypervolume([(0.3, 0.3), (0.3, 0.3)], (1.0, 1.0)) == pytest.approx(0.49)


def test_a_single_objective_reduces_to_the_best_value() -> None:
    assert hypervolume([(0.4,), (0.9,)], (1.0,)) == pytest.approx(0.6)


def test_an_empty_front_covers_nothing() -> None:
    assert hypervolume([], (1.0, 1.0)) == 0.0


# ---------------------------------------------------------------------------
# Hypervolume against independent algorithms.
# ---------------------------------------------------------------------------


def test_the_recursion_matches_the_two_objective_sweep() -> None:
    """The sweep and the recursion are derived differently and must agree.

    A shared third coordinate scales every box by the same factor, so padding a
    two-objective front routes it through the general recursion while leaving
    the answer proportional to the sweep.
    """

    rng = random.Random(11)
    for _ in range(400):
        points = _nondominated(random_points(rng, rng.randint(1, 10), 2))
        padded = [point + (0.5,) for point in points]
        assert _hypervolume_2d(points, (1.0, 1.0)) == pytest.approx(
            hypervolume(padded, (1.0, 1.0, 1.0)) / 0.5, abs=1e-12
        )


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_the_recursion_matches_inclusion_and_exclusion(dimension: int) -> None:
    rng = random.Random(23 + dimension)
    reference = tuple([1.0] * dimension)
    for _ in range(60):
        points = _nondominated(random_points(rng, rng.randint(1, 8), dimension))
        assert hypervolume(points, reference) == pytest.approx(
            inclusion_exclusion(points, reference), abs=1e-12
        )


def test_improving_a_point_never_lowers_the_hypervolume() -> None:
    """The property that makes the indicator worth reporting at all.

    An indicator that could fall when a front improves would rank a better run
    below a worse one, so this is checked rather than assumed.
    """

    rng = random.Random(31)
    for _ in range(500):
        dimension = rng.choice([2, 3, 4])
        reference = tuple([1.0] * dimension)
        points = _nondominated(random_points(rng, rng.randint(2, 7), dimension))
        before = hypervolume(points, reference)
        index = rng.randrange(len(points))
        moved = list(points[index])
        axis = rng.randrange(dimension)
        moved[axis] = max(0.0, moved[axis] - rng.uniform(0.0, 0.3))
        after = hypervolume([*points[:index], tuple(moved), *points[index + 1 :]], reference)
        assert after >= before - 1e-12


# ---------------------------------------------------------------------------
# The reference point.
# ---------------------------------------------------------------------------


def test_a_point_outside_the_reference_box_contributes_nothing() -> None:
    assert hypervolume([(2.0, 0.1)], (1.0, 1.0)) == 0.0


def test_a_point_on_the_reference_contributes_nothing() -> None:
    assert hypervolume([(1.0, 0.5)], (1.0, 1.0)) == 0.0


def test_an_outside_point_cannot_subtract_volume() -> None:
    inside = [(0.3, 0.3)]
    assert hypervolume([*inside, (5.0, -5.0)], (1.0, 1.0)) == hypervolume(inside, (1.0, 1.0))


def test_the_derived_reference_leaves_room_beyond_the_worst_point() -> None:
    reference = derive_reference_point([(0.0, 0.0), (1.0, 2.0)])
    assert reference == pytest.approx((1.0 + DEFAULT_REFERENCE_MARGIN, 2.0 + 2 * 0.1))


def test_the_derived_reference_gives_an_extreme_point_credit() -> None:
    # With the reference on the worst value, the widest point earns nothing.
    front = [(0.0, 1.0), (1.0, 0.0)]
    assert hypervolume(front, (1.0, 1.0)) == 0.0
    assert hypervolume(front, derive_reference_point(front)) > 0.0


def test_a_flat_objective_still_yields_a_usable_reference() -> None:
    reference = derive_reference_point([(0.5, 1.0), (0.5, 3.0)])
    assert reference[0] > 0.5
    assert reference[1] == pytest.approx(3.0 + 0.2)


@pytest.mark.parametrize("margin", [-0.1, float("inf"), float("nan")])
def test_a_bad_margin_is_refused(margin: float) -> None:
    with pytest.raises(ConfigurationError, match="margin"):
        derive_reference_point([(0.0, 0.0)], margin=margin)


def test_a_reference_cannot_be_derived_from_nothing() -> None:
    with pytest.raises(ConfigurationError, match="no vectors"):
        derive_reference_point([])


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_ragged_vectors_are_refused() -> None:
    with pytest.raises(ConfigurationError, match="components"):
        hypervolume([(0.1, 0.2), (0.3,)], (1.0, 1.0))


def test_a_non_finite_vector_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="finite"):
        hypervolume([(0.1, float("nan"))], (1.0, 1.0))


def test_a_non_finite_reference_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="finite"):
        hypervolume([(0.1, 0.2)], (1.0, float("inf")))


def test_a_problem_without_objectives_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="at least one objective"):
        hypervolume([()], ())


# ---------------------------------------------------------------------------
# The cost limit.
# ---------------------------------------------------------------------------


def test_the_limit_shrinks_as_objectives_are_added() -> None:
    limits = [recursive_front_limit(dimension) for dimension in range(3, 10)]
    assert limits[0] == BASE_RECURSIVE_FRONT
    assert limits == sorted(limits, reverse=True)
    assert limits[-1] >= 16


def test_an_oversized_front_is_refused_rather_than_estimated() -> None:
    dimension = 7
    limit = recursive_front_limit(dimension)
    rng = random.Random(5)
    points = []
    while len(points) <= limit:
        raw = [rng.uniform(0.05, 1.0) for _ in range(dimension)]
        total = sum(raw)
        points = _nondominated([*points, tuple(value / total for value in raw)])
    with pytest.raises(ConfigurationError, match="limited to"):
        hypervolume(points, tuple([1.0] * dimension))


def test_two_objectives_are_never_capped() -> None:
    # The sweep is O(n log n), so a large front is answered, not refused.
    front = [(index / 5000.0, 1.0 - index / 5000.0) for index in range(5000)]
    assert hypervolume(front, (2.0, 2.0)) > 0.0


# ---------------------------------------------------------------------------
# Spacing.
# ---------------------------------------------------------------------------


def test_an_evenly_spread_front_has_no_spacing_variation() -> None:
    assert spacing([(0.0, 3.0), (1.0, 2.0), (2.0, 1.0), (3.0, 0.0)]) == pytest.approx(0.0)


def test_a_clustered_front_has_positive_spacing() -> None:
    assert spacing([(0.0, 3.0), (0.01, 2.99), (3.0, 0.0)]) > 0.0


def test_spacing_needs_two_points() -> None:
    assert spacing([(0.5, 0.5)]) == 0.0
    assert spacing([]) == 0.0


# ---------------------------------------------------------------------------
# Comparing two fronts.
# ---------------------------------------------------------------------------


def test_a_front_covers_itself_entirely() -> None:
    front = [(0.1, 0.9), (0.9, 0.1)]
    assert coverage(front, front) == 1.0


def test_a_dominating_front_covers_the_other() -> None:
    assert coverage([(0.1, 0.1)], [(0.5, 0.5), (0.6, 0.4)]) == 1.0
    assert coverage([(0.5, 0.5)], [(0.1, 0.1)]) == 0.0


def test_coverage_is_reported_per_point() -> None:
    assert coverage([(0.1, 0.9)], [(0.2, 0.95), (0.05, 0.99)]) == pytest.approx(0.5)


def test_neither_front_covers_the_other_when_they_cross() -> None:
    left = [(0.1, 0.9)]
    right = [(0.9, 0.1)]
    assert coverage(left, right) == 0.0
    assert coverage(right, left) == 0.0


def test_coverage_of_nothing_is_zero() -> None:
    assert coverage([(0.1, 0.1)], []) == 0.0
    assert coverage([], [(0.1, 0.1)]) == 0.0


def test_a_dominating_front_needs_no_epsilon_shift() -> None:
    assert epsilon_indicator([(0.1, 0.1)], [(0.5, 0.5)]) == pytest.approx(-0.4)


def test_the_epsilon_shift_is_the_worst_single_gap() -> None:
    assert epsilon_indicator([(0.5, 0.5)], [(0.2, 0.4)]) == pytest.approx(0.3)


def test_the_epsilon_indicator_needs_both_fronts() -> None:
    with pytest.raises(ConfigurationError, match="point in each front"):
        epsilon_indicator([], [(0.1, 0.1)])


# ---------------------------------------------------------------------------
# Trial-level measurement.
# ---------------------------------------------------------------------------


def test_only_feasible_successful_trials_are_measured() -> None:
    quality = front_quality(
        [
            trial(0, (0.2, 0.2)),
            trial(1, (0.05, 0.05), feasible=False),
            trial(2, (0.01, 0.01), status=TrialStatus.FAILED),
        ],
        reference_point=(1.0, 1.0),
    )
    assert quality.front_size == 1
    assert quality.hypervolume == pytest.approx(0.64)


def test_a_supplied_reference_is_reported_as_supplied() -> None:
    quality = front_quality(trials((0.2, 0.2)), reference_point=(1.0, 1.0))
    assert not quality.reference_derived
    assert quality.reference_point == (1.0, 1.0)


def test_an_absent_reference_is_derived_and_labelled() -> None:
    quality = front_quality(trials((0.2, 0.8), (0.8, 0.2)))
    assert quality.reference_derived
    assert quality.hypervolume > 0.0


def test_points_outside_the_reference_are_counted_not_hidden() -> None:
    quality = front_quality(trials((0.2, 0.2), (5.0, 0.1)), reference_point=(1.0, 1.0))
    assert quality.front_size == 2
    assert quality.contributing == 1
    assert quality.ignored == 1


def test_a_wholly_excluded_front_is_distinguishable_from_a_bad_run() -> None:
    # Zero hypervolume with every point ignored means the reference was placed
    # wrongly, which a bare zero could not say.
    quality = front_quality(trials((5.0, 5.0)), reference_point=(1.0, 1.0))
    assert quality.hypervolume == 0.0
    assert quality.ignored == quality.front_size == 1


def test_duplicate_objective_vectors_are_both_counted_as_present() -> None:
    # Two distinct designs can share an objective vector and both sit on the
    # front, so the ignored count must not collapse them.
    quality = front_quality(trials((0.3, 0.3), (0.3, 0.3)), reference_point=(1.0, 1.0))
    assert quality.contributing == 2
    assert quality.ignored == 0


def test_the_extent_reports_the_range_covered_per_objective() -> None:
    quality = front_quality(trials((0.1, 0.9), (0.7, 0.2)), reference_point=(1.0, 1.0))
    assert quality.extent == pytest.approx((0.6, 0.7))


def test_an_empty_front_needs_an_explicit_reference() -> None:
    with pytest.raises(ConfigurationError, match="empty front"):
        front_quality([])
    assert front_quality([], reference_point=(1.0, 1.0)).hypervolume == 0.0


def test_the_quality_serializes_without_losing_the_ignored_count() -> None:
    payload = front_quality(trials((0.2, 0.2), (5.0, 0.1)), reference_point=(1.0, 1.0)).as_dict()
    assert payload["ignored"] == 1
    assert payload["reference_point"] == [1.0, 1.0]


# ---------------------------------------------------------------------------
# Comparison shares one reference point.
# ---------------------------------------------------------------------------


def test_a_comparison_bounds_both_fronts_by_the_same_box() -> None:
    """The reason `compare_fronts` exists rather than two `front_quality` calls.

    Measured separately, each front would be scored against a reference derived
    from itself, and the two numbers would describe their reference points as
    much as their fronts.
    """

    left = trials((0.1, 0.1))
    right = trials((0.4, 0.4))
    comparison = compare_fronts(left, right)
    assert comparison.left.reference_point == comparison.reference_point
    assert comparison.right.reference_point == comparison.reference_point
    separate = front_quality(right)
    assert comparison.right.reference_point != separate.reference_point


def test_a_separate_measurement_would_have_ranked_the_worse_front_higher() -> None:
    """A front that is dominated outright can still score higher measured alone.

    The compact front is beaten on every point, but a reference derived from
    its own narrow range sits close in, so its boxes fill most of a small box.
    The wide front derives a large box and fills a corner of it. Sharing the
    reference is what makes the two numbers describe the fronts.
    """

    better = trials((0.0, 0.0))
    worse = trials((0.5, 5.0), (5.0, 0.5))
    assert front_quality(better).hypervolume < front_quality(worse).hypervolume

    comparison = compare_fronts(better, worse)
    assert comparison.hypervolume_difference > 0.0
    assert comparison.left_covers_right == 1.0


def test_hypervolume_is_not_neutral_between_spread_and_proximity() -> None:
    """Why coverage and the epsilon shift are reported beside the volume.

    Neither of these fronts dominates any point of the other, yet hypervolume
    still ranks them, because volume rewards points near the knee over points
    at the extremes. A single number cannot express that they are incomparable,
    so the indicators that can are reported too.
    """

    wide = trials((0.1, 0.9), (0.9, 0.1))
    compact = trials((0.3, 0.5), (0.5, 0.3))
    comparison = compare_fronts(wide, compact)
    assert comparison.left_covers_right == 0.0
    assert comparison.right_covers_left == 0.0
    assert comparison.left_epsilon > 0.0
    assert comparison.right_epsilon > 0.0
    assert comparison.hypervolume_difference < 0.0


def test_the_better_front_dominates_and_needs_no_shift() -> None:
    comparison = compare_fronts(trials((0.1, 0.1)), trials((0.6, 0.6)))
    assert comparison.hypervolume_difference > 0.0
    assert comparison.left_covers_right == 1.0
    assert comparison.right_covers_left == 0.0
    assert comparison.left_epsilon < 0.0
    assert comparison.right_epsilon > 0.0


def test_a_comparison_reverses_cleanly() -> None:
    left = trials((0.1, 0.8), (0.7, 0.2))
    right = trials((0.3, 0.6), (0.9, 0.1))
    forward = compare_fronts(left, right)
    backward = compare_fronts(right, left)
    assert forward.reference_point == backward.reference_point
    assert forward.hypervolume_difference == pytest.approx(-backward.hypervolume_difference)
    assert forward.left_covers_right == backward.right_covers_left
    assert forward.left_epsilon == pytest.approx(backward.right_epsilon)


def test_a_comparison_against_an_empty_front_reports_an_infinite_shift() -> None:
    comparison = compare_fronts(trials((0.2, 0.2)), [])
    assert comparison.right.front_size == 0
    assert comparison.right.hypervolume == 0.0
    assert math.isinf(comparison.left_epsilon)


def test_two_empty_fronts_cannot_be_compared() -> None:
    with pytest.raises(ConfigurationError, match="feasible point"):
        compare_fronts([], [])


def test_a_comparison_serializes_both_sides() -> None:
    payload = compare_fronts(trials((0.1, 0.1)), trials((0.6, 0.6))).as_dict()
    assert payload["left_covers_right"] == 1.0
    assert isinstance(payload["left"], dict)


# ---------------------------------------------------------------------------
# The attainment curve.
# ---------------------------------------------------------------------------


def improving_run() -> list[Trial]:
    return [
        trial(0, (0.9, 0.9)),
        trial(1, (0.8, 0.7)),
        trial(2, (0.4, 0.6)),
        trial(3, (0.3, 0.3)),
        trial(4, (0.35, 0.35)),
        trial(5, (0.36, 0.36)),
        trial(6, (0.37, 0.37)),
        trial(7, (0.38, 0.38)),
    ]


def test_the_curve_never_falls() -> None:
    curve = attainment_curve(make_problem(), improving_run(), steps=8)
    volumes = [point.hypervolume for point in curve.points]
    assert volumes == sorted(volumes)


def test_the_curve_holds_one_reference_across_every_prefix() -> None:
    """A reference derived per prefix would make the curve fall.

    Later trials widen the front, which pushes a derived reference outward and
    would rescale earlier prefixes that had already been reported.
    """

    curve = attainment_curve(make_problem(), improving_run(), steps=8)
    assert curve.reference_derived
    reference = curve.reference_point
    for cut, point in ((point.evaluations, point) for point in curve.points):
        assert point.hypervolume == pytest.approx(
            hypervolume(
                [
                    item.objective_vector
                    for item in improving_run()[:cut]
                    if item.feasible and item.status is TrialStatus.SUCCESS
                ],
                reference,
            ),
            abs=1e-9,
        )


def test_the_curve_reports_where_improvement_stopped() -> None:
    # Trials 5 through 8 are all dominated by trial 3, so nothing after the
    # fourth evaluation moved the front.
    curve = attainment_curve(make_problem(), improving_run(), steps=8)
    assert curve.last_improvement == 4
    assert curve.final_hypervolume > 0.0


def test_a_run_that_improves_to_the_end_says_so() -> None:
    run = [trial(index, (1.0 - index / 10.0, 1.0 - index / 10.0)) for index in range(8)]
    curve = attainment_curve(make_problem(), run, steps=8)
    assert curve.last_improvement == len(run)


def test_the_curve_thins_to_the_requested_number_of_steps() -> None:
    run = [trial(index, (1.0 - index / 100.0, index / 100.0)) for index in range(40)]
    curve = attainment_curve(make_problem(), run, steps=5)
    assert len(curve.points) == 5
    assert curve.points[-1].evaluations == 40


def test_the_curve_never_reports_more_steps_than_trials() -> None:
    curve = attainment_curve(make_problem(), improving_run()[:3], steps=50)
    assert len(curve.points) == 3


def test_a_supplied_reference_must_match_the_objective_count() -> None:
    with pytest.raises(ConfigurationError, match="components for"):
        attainment_curve(make_problem(), improving_run(), reference_point=(1.0, 1.0, 1.0))


def test_the_curve_needs_at_least_one_step() -> None:
    with pytest.raises(ConfigurationError, match="at least one step"):
        attainment_curve(make_problem(), improving_run(), steps=0)


def test_a_run_with_no_feasible_trial_needs_an_explicit_reference() -> None:
    run = [trial(0, (0.5, 0.5), feasible=False)]
    with pytest.raises(ConfigurationError, match="no feasible trial"):
        attainment_curve(make_problem(), run)
    curve = attainment_curve(make_problem(), run, reference_point=(1.0, 1.0))
    assert curve.final_hypervolume == 0.0
    assert curve.last_improvement == 0


def test_an_empty_run_has_an_empty_curve() -> None:
    curve = attainment_curve(make_problem(), [], reference_point=(1.0, 1.0))
    assert curve.points == ()
    assert curve.final_hypervolume == 0.0


def test_the_curve_serializes_its_reference_and_its_points() -> None:
    payload = attainment_curve(make_problem(), improving_run(), steps=4).as_dict()
    assert payload["last_improvement"] == 4
    assert isinstance(payload["points"], list)
    assert len(payload["points"]) == 4


def test_the_attainment_type_survives_an_empty_construction() -> None:
    empty = Attainment(reference_point=(1.0,), reference_derived=False, points=())
    assert empty.final_hypervolume == 0.0
    assert empty.last_improvement == 0
