"""Quality indicators for a feasible Pareto front.

A run reports the front it found. It does not report whether that front was
worth its budget, whether a second run found a better one, or whether the last
two hundred evaluations changed anything. Those are separate questions, and
each has an established indicator:

- `hypervolume` measures how much of objective space a front dominates. It is
  the only widely used unary indicator that is strictly monotone with Pareto
  dominance: a front cannot score higher without actually being better.
- `attainment_curve` reports that volume as a function of evaluations spent,
  which turns "the run stopped at its budget" into "the front last improved at
  evaluation 340 of 512".
- `coverage` and `epsilon_indicator` compare two fronts directly, because a
  pair of hypervolumes answers "which is larger" but not "does either contain
  the other".

Indicators are computed on `Trial.objective_vector`, which the assessment step
has already normalized to `(metric - reference) / scale` and sign-flipped to
minimization. A volume in that space is unit-free and comparable across the
objectives of one problem. A volume in raw metric units would not be: the
product of a power in watts and a bandwidth in hertz is not a quantity.

None of this measures distance to the true Pareto front, which is unknown. A
larger hypervolume means one run dominated more of the space than another under
one reference point; it does not mean either is near optimal.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from biasweave.dominance import pareto_front
from biasweave.errors import ConfigurationError
from biasweave.model import Problem, Trial, TrialStatus

Vector = tuple[float, ...]

#: Fraction of a front's own per-objective range added beyond its worst point
#: when a reference point is derived rather than supplied. A reference placed
#: exactly on the worst point gives that point zero volume, so an extreme
#: solution would earn no credit for being extreme.
DEFAULT_REFERENCE_MARGIN = 0.1

#: Per-objective span used when a derived reference point has nothing to scale
#: against, which happens when every front point shares one objective value.
DEGENERATE_SPAN = 1.0

#: Largest front the general-dimension recursion accepts at five objectives or
#: fewer. Two-objective fronts use an exact sweep instead and are never capped.
BASE_RECURSIVE_FRONT = 200

#: Objective count above which the budget is halved per additional objective.
#: Measured worst-case cost of a fully nondominated front of 200 points is
#: 0.15 s at three objectives, 3.8 s at five, 29 s at six and 268 s at seven,
#: so a single limit is either uselessly small or indistinguishable from a
#: hang. Halving tracks that growth: 100 points at six objectives cost 3.6 s
#: and 50 at seven cost 1.6 s.
RECURSIVE_FRONT_KNEE = 5

#: Smallest budget the halving may reach, so a many-objective problem still
#: measures a front rather than only ever refusing.
MIN_RECURSIVE_FRONT = 16

#: Share of a run's final hypervolume treated as "near enough" when reporting
#: how much of the budget bought how much of the front. A run whose last
#: improvement is a millionth of a percent is still improving by the letter of
#: the word and finished long before by any practical reading, so both numbers
#: are reported rather than one being chosen for the reader.
DEFAULT_ATTAINMENT_FRACTION = 0.99


def recursive_front_limit(dimension: int) -> int:
    """Return the largest front the exact recursion will attempt.

    The recursion is exponential in the objective count for a fixed front size,
    so the budget halves once past the knee. Above the limit BiasWeave refuses
    rather than estimating: a sampled hypervolume carries around half a percent
    of relative error, and two runs worth comparing routinely differ by less
    than that, so the estimate could not answer the question it exists for.
    """

    if dimension <= RECURSIVE_FRONT_KNEE:
        return BASE_RECURSIVE_FRONT
    return max(MIN_RECURSIVE_FRONT, BASE_RECURSIVE_FRONT >> (dimension - RECURSIVE_FRONT_KNEE))


@dataclass(frozen=True, slots=True)
class FrontQuality:
    """What one front covers, and how much of it the indicator could see."""

    reference_point: Vector
    reference_derived: bool
    hypervolume: float
    front_size: int
    contributing: int
    spacing: float
    extent: Vector

    @property
    def ignored(self) -> int:
        """Front points that are not better than the reference point.

        A point outside the reference box contributes no volume. When this is
        not zero the hypervolume describes a subset of the front, and a
        hypervolume of zero with every point ignored means the reference point
        was placed wrongly rather than that the run found nothing.
        """

        return self.front_size - self.contributing

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_point": list(self.reference_point),
            "reference_derived": self.reference_derived,
            "hypervolume": self.hypervolume,
            "front_size": self.front_size,
            "contributing": self.contributing,
            "ignored": self.ignored,
            "spacing": self.spacing,
            "extent": list(self.extent),
        }


@dataclass(frozen=True, slots=True)
class AttainmentPoint:
    """The front that existed after a prefix of the trial sequence."""

    evaluations: int
    hypervolume: float
    front_size: int


@dataclass(frozen=True, slots=True)
class Attainment:
    """A hypervolume curve over evaluations, under one fixed reference point."""

    reference_point: Vector
    reference_derived: bool
    points: tuple[AttainmentPoint, ...]

    @property
    def final_hypervolume(self) -> float:
        return self.points[-1].hypervolume if self.points else 0.0

    @property
    def last_improvement(self) -> int:
        """Evaluations spent when the hypervolume last increased.

        This is a description of a finished run, not a stopping rule. A run
        whose last improvement came far before its budget spent the remainder
        confirming a front rather than extending one -- on this problem, this
        seed and this evaluator, which is all a single run can say.
        """

        best = 0.0
        improved = 0
        for point in self.points:
            if point.hypervolume > best:
                best = point.hypervolume
                improved = point.evaluations
        return improved

    def evaluations_for(self, fraction: float = DEFAULT_ATTAINMENT_FRACTION) -> int:
        """Evaluations spent before the curve first reached `fraction` of its end.

        `last_improvement` answers a strict question and can name the final
        evaluation over a gain of one part in a million. This answers the
        practical one, and the gap between the two is what says whether a
        budget was spent extending a front or confirming it.

        Zero means the curve never reached that share, which happens only when
        the run found nothing feasible and the final hypervolume is zero.
        """

        if not 0.0 < fraction <= 1.0:
            raise ConfigurationError("attainment fraction must lie in (0, 1]")
        target = self.final_hypervolume * fraction
        if self.final_hypervolume <= 0.0:
            return 0
        for point in self.points:
            if point.hypervolume >= target:
                return point.evaluations
        # The final volume is the last point's, so that point always clears a
        # target of at most the whole of it. Reaching here would mean the curve
        # disagreed with its own endpoint.
        return 0  # pragma: no cover - unreachable while the curve is monotone

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_point": list(self.reference_point),
            "reference_derived": self.reference_derived,
            "final_hypervolume": self.final_hypervolume,
            "last_improvement": self.last_improvement,
            "evaluations_for_99_percent": self.evaluations_for(),
            "points": [
                {
                    "evaluations": point.evaluations,
                    "hypervolume": point.hypervolume,
                    "front_size": point.front_size,
                }
                for point in self.points
            ],
        }


@dataclass(frozen=True, slots=True)
class FrontComparison:
    """Two fronts measured against one shared reference point.

    Hypervolumes taken against reference points derived separately from each
    front are not comparable, so this type is the only way to obtain a pair of
    them: the reference is derived once, from the union.
    """

    reference_point: Vector
    reference_derived: bool
    left: FrontQuality
    right: FrontQuality
    left_covers_right: float
    right_covers_left: float
    left_epsilon: float
    right_epsilon: float

    @property
    def hypervolume_difference(self) -> float:
        return self.left.hypervolume - self.right.hypervolume

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_point": list(self.reference_point),
            "reference_derived": self.reference_derived,
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
            "hypervolume_difference": self.hypervolume_difference,
            "left_covers_right": self.left_covers_right,
            "right_covers_left": self.right_covers_left,
            "left_epsilon": self.left_epsilon,
            "right_epsilon": self.right_epsilon,
        }


def _validated(vectors: Iterable[Sequence[float]], *, dimension: int | None = None) -> list[Vector]:
    checked: list[Vector] = []
    for vector in vectors:
        values = tuple(float(value) for value in vector)
        if dimension is None:
            dimension = len(values)
        if len(values) != dimension:
            raise ConfigurationError(
                f"objective vectors must all have {dimension} components, found {len(values)}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ConfigurationError("objective vectors must be finite")
        checked.append(values)
    if dimension == 0:
        raise ConfigurationError("quality indicators need at least one objective")
    return checked


def _nondominated(vectors: list[Vector]) -> list[Vector]:
    """Drop weakly dominated duplicates from a minimization set."""

    unique = sorted(set(vectors))
    kept: list[Vector] = []
    for candidate in unique:
        if any(
            all(a <= b for a, b in zip(other, candidate, strict=True)) for other in kept
        ):  # `unique` is sorted, so a dominator is always already kept.
            continue
        kept.append(candidate)
    return kept


def _box(vector: Vector, reference: Vector) -> float:
    volume = 1.0
    for value, bound in zip(vector, reference, strict=True):
        volume *= bound - value
    return volume


def _hypervolume_2d(vectors: list[Vector], reference: Vector) -> float:
    """Sweep an exact two-objective hypervolume in O(n log n).

    A nondominated two-objective set sorted by its first component ascending is
    sorted by its second descending, so each point adds one rectangle: its full
    width against the reference, and the height it gains over its predecessor.
    """

    total = 0.0
    previous = reference[1]
    for first, second in sorted(vectors):
        total += (reference[0] - first) * (previous - second)
        previous = second
    return total


def _hypervolume_recursive(vectors: list[Vector], reference: Vector) -> float:
    """Sum exclusive contributions, which is exact in any dimension.

    The volume a point adds beyond every point after it is its own box minus
    the volume of that box already taken. The part already taken is itself a
    hypervolume: the one covered by each later point clipped to this point's
    box, which is the componentwise maximum of the two. The recursion therefore
    computes an exact answer without inclusion-exclusion over all subsets.
    """

    if not vectors:
        return 0.0
    if len(vectors) == 1:
        return _box(vectors[0], reference)
    total = 0.0
    for index, vector in enumerate(vectors):
        clipped = [
            tuple(max(a, b) for a, b in zip(vector, other, strict=True))
            for other in vectors[index + 1 :]
        ]
        total += _box(vector, reference) - _hypervolume_recursive(_nondominated(clipped), reference)
    return total


def hypervolume(vectors: Iterable[Sequence[float]], reference_point: Sequence[float]) -> float:
    """Return the volume dominated by `vectors` and bounded by `reference_point`.

    Every input is a minimization vector, as `Trial.objective_vector` is. A
    vector that is not strictly better than the reference point on every
    objective lies outside the box and contributes nothing; it is dropped
    rather than allowed to subtract volume.
    """

    reference = _validated([reference_point])[0]
    points = _validated(vectors, dimension=len(reference))
    inside = [
        point
        for point in points
        if all(value < bound for value, bound in zip(point, reference, strict=True))
    ]
    if not inside:
        return 0.0
    front = _nondominated(inside)
    if len(reference) == 1:
        return reference[0] - min(point[0] for point in front)
    if len(reference) == 2:
        return _hypervolume_2d(front, reference)
    limit = recursive_front_limit(len(reference))
    if len(front) > limit:
        raise ConfigurationError(
            f"exact hypervolume at {len(reference)} objectives is limited to {limit} "
            f"front points, received {len(front)}; measure a subset of the front or "
            f"state a reference point that excludes part of it"
        )
    return _hypervolume_recursive(front, reference)


def derive_reference_point(
    vectors: Iterable[Sequence[float]], *, margin: float = DEFAULT_REFERENCE_MARGIN
) -> Vector:
    """Place a reference point beyond the worst value of each objective.

    The margin is a fraction of the range the vectors themselves span, so the
    result carries no unit and no assumption about the problem. A point placed
    exactly on the worst value would give the solution holding it zero volume,
    which would score an extreme solution as worthless.

    A reference derived this way describes the set it was derived from. Two
    such numbers are comparable only when both came from the same derivation,
    which is why `compare_fronts` derives one for the union.
    """

    if margin < 0.0 or not math.isfinite(margin):
        raise ConfigurationError("reference margin must be finite and non-negative")
    points = _validated(vectors)
    if not points:
        raise ConfigurationError("a reference point cannot be derived from no vectors")
    reference: list[float] = []
    for index in range(len(points[0])):
        column = [point[index] for point in points]
        worst = max(column)
        span = worst - min(column)
        reference.append(worst + margin * (span if span > 0.0 else DEGENERATE_SPAN))
    return tuple(reference)


def spacing(vectors: Iterable[Sequence[float]]) -> float:
    """Return Schott's spacing: the spread of nearest-neighbour distances.

    Zero means the front is perfectly evenly distributed; a large value means
    it clusters. It says nothing about whether the front is good, only about
    how its points are laid out, so it is reported beside the hypervolume
    rather than folded into it.
    """

    points = _validated(vectors)
    if len(points) < 2:
        return 0.0
    distances = [
        min(math.dist(point, points[other]) for other in range(len(points)) if other != index)
        for index, point in enumerate(points)
    ]
    mean = math.fsum(distances) / len(distances)
    variance = math.fsum((distance - mean) ** 2 for distance in distances) / len(distances)
    return math.sqrt(variance)


def coverage(left: Iterable[Sequence[float]], right: Iterable[Sequence[float]]) -> float:
    """Return the fraction of `right` weakly dominated by some point of `left`.

    This is the C-metric of Zitzler and Thiele. It is not symmetric and the two
    directions are not complementary: both can be zero when neither front
    reaches the other, and both can be one when the fronts share every point.
    An empty `right` is reported as zero, since there is nothing to cover.
    """

    covering = _validated(left)
    covered = _validated(right, dimension=len(covering[0]) if covering else None)
    if not covered:
        return 0.0
    if not covering:
        return 0.0
    dominated = sum(
        any(all(a <= b for a, b in zip(point, target, strict=True)) for point in covering)
        for target in covered
    )
    return dominated / len(covered)


def epsilon_indicator(left: Iterable[Sequence[float]], right: Iterable[Sequence[float]]) -> float:
    """Return the additive shift that makes `left` weakly dominate `right`.

    A value at or below zero means every point of `right` is already matched by
    a point of `left`. A positive value is the worst single-objective distance,
    in normalized units, by which `left` falls short somewhere.
    """

    shifting = _validated(left)
    target = _validated(right, dimension=len(shifting[0]) if shifting else None)
    if not shifting or not target:
        raise ConfigurationError("the epsilon indicator needs a point in each front")
    return max(
        min(max(a - b for a, b in zip(point, other, strict=True)) for point in shifting)
        for other in target
    )


def _feasible_vectors(trials: Iterable[Trial]) -> list[Vector]:
    """Every feasible evaluated point, dominated ones included.

    This is what a reference point is derived from, because the box has to hold
    the worse fronts a run passed through as well as the one it ended on.
    """

    return [
        trial.objective_vector
        for trial in trials
        if trial.status is TrialStatus.SUCCESS and trial.feasible
    ]


def _front_vectors(trials: Iterable[Trial]) -> list[Vector]:
    """The feasible nondominated points, whatever was passed in.

    A caller handing over a whole run rather than its front would otherwise be
    told the front size, spacing and extent of the run. The hypervolume would
    survive it, which is worse: three numbers wrong beside one that is right.
    """

    return [trial.objective_vector for trial in pareto_front(list(trials))]


def _resolve_reference(
    vectors: list[Vector],
    reference_point: Sequence[float] | None,
    pool: Iterable[Sequence[float]] | None,
    margin: float,
) -> tuple[Vector, bool]:
    """Settle the box every number in one measurement is bounded by.

    An explicit reference wins. Otherwise the reference is derived from `pool`
    when one is given, and from the vectors being measured when it is not. The
    pool exists because a front and the curve that produced it derive different
    boxes on their own -- the front is tight, the curve must hold every worse
    front that preceded it -- and two hypervolumes taken against different
    boxes cannot appear in one report without contradicting each other.
    """

    if reference_point is not None:
        return _validated([reference_point])[0], False
    if pool is not None:
        return derive_reference_point(pool, margin=margin), True
    return derive_reference_point(vectors, margin=margin), True


def _measure(vectors: list[Vector], reference: Vector) -> FrontQuality:
    inside = [
        point
        for point in vectors
        if all(value < bound for value, bound in zip(point, reference, strict=True))
    ]
    extent = tuple(
        (max(point[index] for point in vectors) - min(point[index] for point in vectors))
        if vectors
        else 0.0
        for index in range(len(reference))
    )
    return FrontQuality(
        reference_point=reference,
        reference_derived=False,
        hypervolume=hypervolume(vectors, reference),
        front_size=len(vectors),
        contributing=len(inside),
        spacing=spacing(vectors),
        extent=extent,
    )


def _comparison_derivation(comparison: FrontComparison, derived: bool) -> FrontComparison:
    """Restate how a comparison got its reference point.

    `compare_fronts` is told the reference explicitly when the caller derived
    one for a wider set, and would otherwise report it as supplied by the user.
    Whoever reads the report supplied nothing, so the flag is corrected rather
    than left to describe the internal call.
    """

    if not derived:
        return comparison
    return FrontComparison(
        reference_point=comparison.reference_point,
        reference_derived=True,
        left=_with_derivation(comparison.left, True),
        right=_with_derivation(comparison.right, True),
        left_covers_right=comparison.left_covers_right,
        right_covers_left=comparison.right_covers_left,
        left_epsilon=comparison.left_epsilon,
        right_epsilon=comparison.right_epsilon,
    )


def _with_derivation(quality: FrontQuality, derived: bool) -> FrontQuality:
    if not derived:
        return quality
    return FrontQuality(
        reference_point=quality.reference_point,
        reference_derived=True,
        hypervolume=quality.hypervolume,
        front_size=quality.front_size,
        contributing=quality.contributing,
        spacing=quality.spacing,
        extent=quality.extent,
    )


def front_quality(
    frontier: Iterable[Trial],
    *,
    reference_point: Sequence[float] | None = None,
    pool: Iterable[Sequence[float]] | None = None,
    margin: float = DEFAULT_REFERENCE_MARGIN,
) -> FrontQuality:
    """Measure one feasible front, deriving a reference point if none is given.

    Infeasible and failed trials are excluded, matching the front the run
    itself reports. An empty front has no volume and no derivable reference, so
    the reference point must be supplied to measure one.
    """

    vectors = _front_vectors(frontier)
    if not vectors and reference_point is None and pool is None:
        raise ConfigurationError(
            "an empty front has no reference point to derive; supply reference_point"
        )
    reference, derived = _resolve_reference(vectors, reference_point, pool, margin)
    return _with_derivation(_measure(vectors, reference), derived)


def attainment_curve(
    problem: Problem,
    trials: Sequence[Trial],
    *,
    steps: int = 20,
    reference_point: Sequence[float] | None = None,
    pool: Iterable[Sequence[float]] | None = None,
    margin: float = DEFAULT_REFERENCE_MARGIN,
) -> Attainment:
    """Report the hypervolume of the front after each prefix of the run.

    The reference point is fixed once, from every feasible trial in the run, so
    the curve is monotone. Deriving one per prefix would let the curve fall
    when a later trial widened the front, which would describe the reference
    point rather than the search.

    Trials are read in the order given, which for a ledger is trial-ID order --
    the order the run committed them, and the order a resume reproduces.
    """

    if steps <= 0:
        raise ConfigurationError("an attainment curve needs at least one step")
    ordered = list(trials)
    feasible = _feasible_vectors(ordered)
    if not feasible and reference_point is None and pool is None:
        raise ConfigurationError(
            "a run with no feasible trial has no reference point to derive; supply reference_point"
        )
    reference, derived = _resolve_reference(feasible, reference_point, pool, margin)
    if len(reference) != len(problem.objectives):
        raise ConfigurationError(
            f"reference point has {len(reference)} components for "
            f"{len(problem.objectives)} objectives"
        )
    # The divisor is the number of cuts actually taken, not the number asked
    # for. Dividing by `steps` when it exceeds the run length rounds every cut
    # down to one, which would drop the final evaluation and with it both the
    # reported final hypervolume and the last improvement.
    taken = min(steps, len(ordered))
    cuts = sorted({round(len(ordered) * (index + 1) / taken) for index in range(taken)})
    points = []
    for cut in cuts:
        front = pareto_front(ordered[:cut])
        points.append(
            AttainmentPoint(
                evaluations=cut,
                hypervolume=hypervolume([trial.objective_vector for trial in front], reference),
                front_size=len(front),
            )
        )
    return Attainment(reference_point=reference, reference_derived=derived, points=tuple(points))


def compare_fronts(
    left: Iterable[Trial],
    right: Iterable[Trial],
    *,
    reference_point: Sequence[float] | None = None,
    pool: Iterable[Sequence[float]] | None = None,
    margin: float = DEFAULT_REFERENCE_MARGIN,
) -> FrontComparison:
    """Measure two fronts against one reference point derived from their union.

    Deriving a reference point for each front separately and comparing the two
    hypervolumes measures the reference points as much as the fronts. Taking
    the union removes that: both are bounded by the same box, so the difference
    is a statement about the fronts.
    """

    left_vectors = _front_vectors(left)
    right_vectors = _front_vectors(right)
    if not left_vectors and not right_vectors:
        raise ConfigurationError("neither front has a feasible point to compare")
    reference, derived = _resolve_reference(
        left_vectors + right_vectors, reference_point, pool, margin
    )
    return FrontComparison(
        reference_point=reference,
        reference_derived=derived,
        left=_with_derivation(_measure(left_vectors, reference), derived),
        right=_with_derivation(_measure(right_vectors, reference), derived),
        left_covers_right=coverage(left_vectors, right_vectors),
        right_covers_left=coverage(right_vectors, left_vectors),
        left_epsilon=(
            epsilon_indicator(left_vectors, right_vectors)
            if left_vectors and right_vectors
            else math.inf
        ),
        right_epsilon=(
            epsilon_indicator(right_vectors, left_vectors)
            if left_vectors and right_vectors
            else math.inf
        ),
    )


@dataclass(frozen=True, slots=True)
class RunQuality:
    """Everything one report says about a run, under one reference point."""

    reference_point: Vector
    reference_derived: bool
    front: FrontQuality
    attainment: Attainment | None
    comparison: FrontComparison | None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "reference_point": list(self.reference_point),
            "reference_derived": self.reference_derived,
            "front": self.front.as_dict(),
        }
        if self.attainment is not None:
            payload["attainment"] = self.attainment.as_dict()
        if self.comparison is not None:
            payload["comparison"] = self.comparison.as_dict()
        return payload


def measure_run(
    problem: Problem,
    trials: Sequence[Trial],
    *,
    compare: Sequence[Trial] | None = None,
    steps: int | None = None,
    reference_point: Sequence[float] | None = None,
    margin: float = DEFAULT_REFERENCE_MARGIN,
) -> RunQuality:
    """Measure a run, and optionally a second, against one reference point.

    The reference is derived from every feasible trial in every run involved,
    not from the fronts alone. A front derives a box it fills tightly, while an
    attainment curve needs a box wide enough for the worse fronts that preceded
    it, so measuring them separately produces two numbers that disagree by
    orders of magnitude and cannot both be called the hypervolume of the run.
    Deriving once removes the contradiction: the last point of the curve is the
    hypervolume of the front, because both are bounded by the same box.
    """

    pool = _feasible_vectors(trials) + _feasible_vectors(compare or ())
    if not pool and reference_point is None:
        raise ConfigurationError(
            "no run has a feasible trial to derive a reference point from; supply reference_point"
        )
    reference, derived = _resolve_reference(pool, reference_point, None, margin)
    frontier = pareto_front(list(trials))
    return RunQuality(
        reference_point=reference,
        reference_derived=derived,
        front=_with_derivation(
            _measure([trial.objective_vector for trial in frontier], reference), derived
        ),
        attainment=(
            attainment_curve(problem, trials, steps=steps, reference_point=reference)
            if steps is not None
            else None
        ),
        comparison=(
            _comparison_derivation(
                compare_fronts(frontier, pareto_front(list(compare)), reference_point=reference),
                derived,
            )
            if compare is not None
            else None
        ),
    )
