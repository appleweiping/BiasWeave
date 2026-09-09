"""Synchronous ask/tell contract shared by every optimizer strategy.

Strategies operate on the normalized unit hypercube.  Decoding remains the
single source of truth for real, integer, choice, and linked variables.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence

from biasweave.dominance import assess, failed_trial
from biasweave.encoding import make_point
from biasweave.errors import ConfigurationError
from biasweave.evaluator import validate_metrics
from biasweave.model import Point, Problem, Trial, TrialStatus
from biasweave.search_space import FiniteDomainEnumerator

MAX_POPULATION_SIZE = 512
_FALLBACK_RANDOM_ATTEMPTS = 512


def clamp_coordinate(value: float) -> float:
    """Project one finite normalized coordinate onto the closed unit interval."""
    if not math.isfinite(value):
        raise ConfigurationError("optimizer produced a non-finite coordinate")
    return min(1.0, max(0.0, value))


def clamp_vector(values: Sequence[float]) -> tuple[float, ...]:
    """Project a coordinate vector onto the normalized search domain."""
    return tuple(clamp_coordinate(value) for value in values)


def finite_parameter(value: object, name: str) -> float:
    """Normalize one public numeric option or raise a consistent error."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be numeric")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ConfigurationError(f"{name} must be finite") from error
    if not math.isfinite(numeric):
        raise ConfigurationError(f"{name} must be finite")
    return numeric


def population_size_parameter(value: object, algorithm: str) -> int:
    """Validate a population before an optimizer allocates proportional state."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("population_size must be an integer")
    if value < 4:
        raise ConfigurationError(f"{algorithm} population_size must be at least 4")
    if value > MAX_POPULATION_SIZE:
        raise ConfigurationError(
            f"{algorithm} population_size must be at most {MAX_POPULATION_SIZE}"
        )
    return value


def canonical_trial(problem: Problem, trial: Trial, point: Point) -> Trial:
    """Validate an ask/tell report and return a detached canonical snapshot."""
    canonical_point = make_point(problem, point.coordinates)
    if point != canonical_point or trial.point != canonical_point:
        raise ConfigurationError("tell trials do not match the canonical pending ask batch")
    if (
        not isinstance(trial.trial_id, int)
        or isinstance(trial.trial_id, bool)
        or trial.trial_id < 0
    ):
        raise ConfigurationError("tell trial IDs must be non-negative integers")
    if not isinstance(trial.status, TrialStatus):
        raise ConfigurationError("tell trial status is invalid")
    if trial.status is TrialStatus.FAILED:
        if not isinstance(trial.error, str) or not trial.error.strip():
            raise ConfigurationError("failed tell trial has inconsistent assessment fields")
        expected = failed_trial(trial.trial_id, canonical_point, trial.error)
    else:
        try:
            metrics = validate_metrics(problem, trial.metrics)
            expected = assess(problem, trial.trial_id, canonical_point, metrics)
        except Exception as error:
            raise ConfigurationError(
                f"successful tell trial metrics are invalid: {error}"
            ) from error
    if trial != expected:
        raise ConfigurationError(
            "tell trial has inconsistent assessment for its metrics and problem"
        )
    return expected


class AskTellOptimizer(ABC):
    """Strict synchronous ask/tell optimizer.

    An ``ask`` batch must be reported exactly once, in order, through ``tell``
    before another batch can be requested.  This deliberately small contract
    makes evaluation scheduling independent from algorithm state and keeps a
    seeded run reproducible across worker counts.
    """

    name: str

    def __init__(self, problem: Problem, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("seed must be an integer")
        self.problem = problem
        self.seed = seed
        # This seeded generator defines repeatable search, not cryptography.
        self.random = random.Random(seed)  # nosec B311
        self._pending: tuple[Point, ...] = ()
        self._last_trial_id = -1
        self._seen_keys: set[str] = set()
        self._finite_domain = FiniteDomainEnumerator(problem)
        self._finite_axes = self._finite_domain.axes
        self._empty_ask_reason: str | None = None

    @property
    def dimension(self) -> int:
        return len(self.problem.free_variables)

    @property
    def pending(self) -> tuple[Point, ...]:
        return self._pending

    @property
    def recommended_batch_size(self) -> int:
        """Natural batch size; callers may safely request a smaller batch."""
        return 1

    @property
    def empty_ask_reason(self) -> str | None:
        """Explain an empty ask without overstating stochastic proposal failure."""
        return self._empty_ask_reason

    def random_vector(self) -> tuple[float, ...]:
        return tuple(self.random.random() for _ in range(self.dimension))

    def point_if_new(
        self,
        coordinates: Sequence[float],
        seen_keys: Collection[str],
        local_keys: Collection[str],
    ) -> Point | None:
        point = make_point(self.problem, clamp_vector(coordinates))
        if point.key in seen_keys or point.key in local_keys:
            return None
        return point

    def fallback_point(
        self, seen_keys: Collection[str], local_keys: Collection[str]
    ) -> Point | None:
        """Return a deterministic finite-domain point or a bounded random immigrant."""
        if self._finite_domain.finite:
            point = self._finite_domain.next_unseen(self._seen_keys, set(local_keys))
            if point is not None:
                return point
            if self._finite_domain.exhausted:
                self._empty_ask_reason = "search_space_exhausted"
                return None

        for _attempt in range(_FALLBACK_RANDOM_ATTEMPTS):
            point = self.point_if_new(self.random_vector(), self._seen_keys, local_keys)
            if point is not None:
                return point
        self._empty_ask_reason = "proposal_stalled"
        return None

    def ask(self, count: int, seen_keys: Collection[str] = ()) -> tuple[Point, ...]:
        """Request at most ``count`` previously unseen evaluation points."""
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ConfigurationError("ask count must be a positive integer")
        if self._pending:
            raise ConfigurationError("tell must report the pending batch before ask")
        self._empty_ask_reason = None
        self._seen_keys.update(seen_keys)
        points = self._ask(count, self._seen_keys)
        if len(points) > count:
            raise RuntimeError(f"{self.name} returned more points than requested")
        canonical = tuple(make_point(self.problem, point.coordinates) for point in points)
        if tuple(points) != canonical:
            raise RuntimeError(f"{self.name} returned a non-canonical point")
        keys = [point.key for point in canonical]
        if len(keys) != len(set(keys)) or any(key in self._seen_keys for key in keys):
            raise RuntimeError(f"{self.name} returned a duplicate or previously seen point")
        if not canonical and self._empty_ask_reason is None:
            self._empty_ask_reason = "proposal_stalled"
        self._pending = canonical
        return self._pending

    def tell(self, trials: Sequence[Trial]) -> None:
        """Report the complete pending batch in the order returned by ``ask``."""
        if not self._pending:
            raise ConfigurationError("tell requires a pending ask batch")
        reported = tuple(trials)
        if len(reported) != len(self._pending):
            raise ConfigurationError("tell must report the complete pending batch")
        trial_ids = tuple(trial.trial_id for trial in reported)
        if any(
            isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 0
            for trial_id in trial_ids
        ):
            raise ConfigurationError("tell trial IDs must be non-negative integers")
        if trial_ids != tuple(sorted(set(trial_ids))) or trial_ids[0] <= self._last_trial_id:
            raise ConfigurationError("tell trial IDs must be unique and strictly increasing")
        canonical = tuple(
            canonical_trial(self.problem, trial, point)
            for trial, point in zip(reported, self._pending, strict=True)
        )
        self._tell(canonical)
        self._last_trial_id = trial_ids[-1]
        self._seen_keys.update(point.key for point in self._pending)
        self._pending = ()

    @abstractmethod
    def _ask(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        """Produce a validated candidate batch."""

    @abstractmethod
    def _tell(self, trials: tuple[Trial, ...]) -> None:
        """Update strategy state from a validated complete batch."""
