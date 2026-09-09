"""Bounded sequential Bayesian optimization over canonical mixed-variable features."""

from __future__ import annotations

import hashlib
import math
import random
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass
from fractions import Fraction

from biasweave.encoding import default_coordinates, encode
from biasweave.errors import ConfigurationError, ProblemError
from biasweave.model import Point, Problem, Trial, TrialStatus, VariableKind
from biasweave.optimizers.base import AskTellOptimizer, finite_parameter
from biasweave.optimizers.ranking import preference_key
from biasweave.surrogate import GaussianProcess, expected_improvement

MAX_FEATURES = 64
MAX_TRAINING_WINDOW = 128
MAX_CANDIDATE_POOL = 64
MAX_OBJECTIVES = 128
_PROPOSAL_ATTEMPTS_PER_POINT = 32
_MIN_SEED = -(2**63)
_MAX_SEED = 2**63 - 1


def _bounded_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _stable_unit(value: float, low: float, high: float) -> float:
    """Normalize one finite binary64 value without forming an overflowing width."""

    if value == low or high == low:
        return 0.0
    if value == high:
        return 1.0
    numerator = Fraction.from_float(value) - Fraction.from_float(low)
    denominator = Fraction.from_float(high) - Fraction.from_float(low)
    result = float(numerator / denominator)
    if not math.isfinite(result):
        raise ConfigurationError("Bayesian target normalization produced a non-finite value")
    return min(1.0, max(0.0, result))


def _normalize_rows(rows: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """Column-normalize finite rows to [0, 1] using exact binary-rational ratios."""

    if not rows:
        return ()
    dimensions = len(rows[0])
    if dimensions == 0 or any(len(row) != dimensions for row in rows):
        raise ConfigurationError("Bayesian target rows must have one consistent dimension")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ConfigurationError("Bayesian targets must be finite")
    lows = tuple(min(row[index] for row in rows) for index in range(dimensions))
    highs = tuple(max(row[index] for row in rows) for index in range(dimensions))
    return tuple(
        tuple(_stable_unit(value, lows[index], highs[index]) for index, value in enumerate(row))
        for row in rows
    )


@dataclass(frozen=True, slots=True)
class ScalarizationRecord:
    """One bounded diagnostic entry from the seeded simplex-weight schedule."""

    step: int
    weights: tuple[float, ...]


class BayesianOptimizer(AskTellOptimizer):
    """One-at-a-time expected-improvement search with strictly bounded GP state.

    Feasible observations model a seeded normalized-Chebyshev objective. Until
    feasibility exists, a separate model targets constraint violation. Failed
    evaluations remain part of the runner's ledger but never become numeric
    observations.
    """

    name = "bayes"
    feature_encoding = "canonical-unit-choice-one-hot-v1"
    scalarization_schedule = "seeded-exponential-simplex-v1"

    def __init__(
        self,
        problem: Problem,
        seed: int,
        *,
        initial_design_size: int = 8,
        training_window_size: int = 128,
        candidate_pool_size: int = 64,
        length_scale: float = 0.25,
        noise_variance: float = 1e-6,
        local_fraction: float = 0.5,
        local_radius: float = 0.15,
        exploration: float = 0.0,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("seed must be an integer")
        if not _MIN_SEED <= seed <= _MAX_SEED:
            raise ConfigurationError("Bayesian seed must be a signed 64-bit integer")
        super().__init__(problem, seed)
        self.initial_design_size = _bounded_integer(
            initial_design_size, "initial_design_size", minimum=1, maximum=128
        )
        self.training_window_size = _bounded_integer(
            training_window_size,
            "training_window_size",
            minimum=2,
            maximum=MAX_TRAINING_WINDOW,
        )
        self.candidate_pool_size = _bounded_integer(
            candidate_pool_size,
            "candidate_pool_size",
            minimum=4,
            maximum=MAX_CANDIDATE_POOL,
        )
        self.length_scale = finite_parameter(length_scale, "length_scale")
        self.noise_variance = finite_parameter(noise_variance, "noise_variance")
        self.local_fraction = finite_parameter(local_fraction, "local_fraction")
        self.local_radius = finite_parameter(local_radius, "local_radius")
        self.exploration = finite_parameter(exploration, "exploration")
        if not 1e-4 <= self.length_scale <= 10.0:
            raise ConfigurationError("length_scale must be between 1e-4 and 10")
        if not 1e-10 <= self.noise_variance <= 1.0:
            raise ConfigurationError("noise_variance must be between 1e-10 and 1")
        if not 0.0 <= self.local_fraction <= 1.0:
            raise ConfigurationError("local_fraction must be between 0 and 1")
        if not 0.0 < self.local_radius <= 1.0:
            raise ConfigurationError("local_radius must be in (0, 1]")
        if not 0.0 <= self.exploration <= 1.0:
            raise ConfigurationError("exploration must be between 0 and 1")

        feature_count = 0
        for variable in problem.free_variables:
            feature_count += len(variable.values) if variable.kind is VariableKind.CHOICE else 1
            if feature_count > MAX_FEATURES:
                raise ConfigurationError(
                    f"Bayesian canonical feature expansion supports at most {MAX_FEATURES} dimensions"
                )
        self.feature_count = feature_count
        self.feature_limit = MAX_FEATURES
        if not 2 <= len(problem.objectives) <= MAX_OBJECTIVES:
            raise ConfigurationError(
                f"Bayesian optimization requires at least 2 and supports at most "
                f"{MAX_OBJECTIVES} objectives"
            )
        self.scalarization_seed = seed

        recent_size = self.training_window_size - 1
        self._recent_feasible: deque[Trial] = deque(maxlen=recent_size)
        self._recent_infeasible: deque[Trial] = deque(maxlen=recent_size)
        self._best_feasible: Trial | None = None
        self._best_infeasible: Trial | None = None
        self._evaluations = 0
        self.successful_observations = 0
        self.failed_observations = 0
        self._scalarization_history: deque[ScalarizationRecord] = deque(
            maxlen=self.training_window_size
        )
        self.scalarization_step = 0

    @property
    def scalarization_history(self) -> tuple[ScalarizationRecord, ...]:
        """Return an immutable replay diagnostic for realized objective weights."""

        return tuple(self._scalarization_history)

    def _point_features(self, point: Point) -> tuple[float, ...]:
        try:
            canonical = encode(self.problem, point.values)
        except ProblemError as error:
            raise ConfigurationError(
                f"Bayesian canonical feature encoding failed: {error}"
            ) from error
        features: list[float] = []
        for variable, coordinate in zip(self.problem.free_variables, canonical, strict=True):
            if variable.kind is VariableKind.CHOICE:
                selected = point.values[variable.name]
                one_hot = tuple(
                    1.0 if type(selected) is type(value) and selected == value else 0.0
                    for value in variable.values
                )
                if sum(one_hot) != 1.0:
                    raise ConfigurationError(
                        f"Bayesian choice feature for {variable.name} is not canonical"
                    )
                features.extend(one_hot)
            else:
                features.append(coordinate)
        if len(features) != self.feature_count or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in features
        ):
            raise ConfigurationError("Bayesian canonical feature encoding is invalid")
        return tuple(features)

    def _training_trials(self) -> tuple[Trial, ...]:
        if self._best_feasible is not None:
            incumbent = self._best_feasible
            recent = tuple(self._recent_feasible)
        elif self._best_infeasible is not None:
            incumbent = self._best_infeasible
            recent = tuple(self._recent_infeasible)
        else:
            return ()
        if any(trial.trial_id == incumbent.trial_id for trial in recent):
            return recent
        return (incumbent, *recent)

    def _objective_weights(self) -> tuple[float, ...]:
        step = self.scalarization_step
        material = hashlib.sha256(
            f"biasweave:{self.scalarization_schedule}:{self.scalarization_seed}:{step}".encode(
                "ascii"
            )
        ).digest()
        weight_random = random.Random(int.from_bytes(material, "big"))  # nosec B311
        # Independent exponential draws normalized by their sum are uniform
        # on the positive simplex. Random.random() may return zero, so use one
        # explicitly representable 53-bit endpoint rather than log(0).
        raw = tuple(
            -math.log(max(weight_random.random(), 2.0**-53)) for _ in self.problem.objectives
        )
        total = math.fsum(raw)
        weights = tuple(value / total for value in raw)
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise RuntimeError("Bayesian scalarization weights are not finite and positive")
        self._scalarization_history.append(ScalarizationRecord(step, weights))
        self.scalarization_step += 1
        return weights

    def _model_targets(self, trials: tuple[Trial, ...]) -> tuple[float, ...]:
        if not trials:
            raise RuntimeError("Bayesian modeling requires successful observations")
        if trials[0].feasible:
            if any(
                not trial.feasible or trial.status is not TrialStatus.SUCCESS for trial in trials
            ):
                raise RuntimeError("Bayesian feasible model received an ineligible trial")
            weights = self._objective_weights()
            normalized = _normalize_rows(tuple(trial.objective_vector for trial in trials))
            targets = tuple(
                2.0 * max(weight * value for weight, value in zip(weights, row, strict=True)) - 1.0
                for row in normalized
            )
        else:
            if any(trial.feasible or trial.status is not TrialStatus.SUCCESS for trial in trials):
                raise RuntimeError("Bayesian recovery model received an ineligible trial")
            normalized = _normalize_rows(tuple((trial.violation,) for trial in trials))
            targets = tuple(2.0 * row[0] - 1.0 for row in normalized)
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in targets):
            raise RuntimeError("Bayesian normalized model target is invalid")
        return targets

    def _random_candidate(
        self,
        seen_keys: Collection[str],
        local_keys: Collection[str],
    ) -> Point | None:
        return self.point_if_new(self.random_vector(), seen_keys, local_keys)

    def _local_candidate(
        self,
        anchor: tuple[float, ...],
        seen_keys: Collection[str],
        local_keys: Collection[str],
    ) -> Point | None:
        sigma = self.local_radius / math.sqrt(max(1, self.dimension))
        coordinates: list[float] = []
        for variable, value in zip(self.problem.free_variables, anchor, strict=True):
            if variable.kind is VariableKind.CHOICE:
                # A categorical neighborhood has no left/right ordering. Keep
                # or uniformly resample its label instead of perturbing the
                # arbitrary declaration index as if it were a numeric axis.
                if len(variable.values) > 1 and self.random.random() < self.local_radius:
                    index = self.random.randrange(len(variable.values))
                    coordinates.append((index + 0.5) / len(variable.values))
                else:
                    coordinates.append(value)
            else:
                coordinates.append(value + self.random.gauss(0.0, sigma))
        return self.point_if_new(coordinates, seen_keys, local_keys)

    def _candidate_pool(self, seen_keys: set[str]) -> tuple[Point, ...]:
        incumbent = self._best_feasible or self._best_infeasible
        local_target = (
            min(
                self.candidate_pool_size - 1,
                max(1, round(self.candidate_pool_size * self.local_fraction)),
            )
            if incumbent is not None and self.local_fraction > 0.0
            else 0
        )
        global_target = self.candidate_pool_size - local_target
        points: list[Point] = []
        local_keys: set[str] = set()

        attempts = 0
        maximum_attempts = max(1, global_target) * _PROPOSAL_ATTEMPTS_PER_POINT
        while len(points) < global_target and attempts < maximum_attempts:
            point = self._random_candidate(seen_keys, local_keys)
            attempts += 1
            if point is not None:
                local_keys.add(point.key)
                points.append(point)

        if local_target:
            if incumbent is None:
                raise RuntimeError("Bayesian local candidate generation has no incumbent")
            try:
                anchor = encode(self.problem, incumbent.point.values)
            except ProblemError as error:
                raise ConfigurationError(
                    f"Bayesian incumbent feature encoding failed: {error}"
                ) from error
            local_added = 0
            attempts = 0
            maximum_attempts = local_target * _PROPOSAL_ATTEMPTS_PER_POINT
            while local_added < local_target and attempts < maximum_attempts:
                point = self._local_candidate(anchor, seen_keys, local_keys)
                attempts += 1
                if point is not None:
                    local_keys.add(point.key)
                    points.append(point)
                    local_added += 1

        # Decoded plateaus can prevent the requested local/global split. Spend
        # a final bounded global allowance without growing the advertised pool.
        attempts = 0
        maximum_attempts = self.candidate_pool_size * _PROPOSAL_ATTEMPTS_PER_POINT
        while len(points) < self.candidate_pool_size and attempts < maximum_attempts:
            point = self._random_candidate(seen_keys, local_keys)
            attempts += 1
            if point is not None:
                local_keys.add(point.key)
                points.append(point)
        return tuple(points)

    def _initial_point(self, seen_keys: set[str]) -> Point | None:
        if self._evaluations == 0:
            point = self.point_if_new(default_coordinates(self.problem), seen_keys, ())
            if point is not None:
                return point
        for _attempt in range(_PROPOSAL_ATTEMPTS_PER_POINT * 4):
            point = self._random_candidate(seen_keys, ())
            if point is not None:
                return point
        return self.fallback_point(seen_keys, ())

    def _ask(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        del count  # The posterior changes after every observation.
        if self._evaluations < self.initial_design_size:
            point = self._initial_point(seen_keys)
            return (point,) if point is not None else ()

        trials = self._training_trials()
        if not trials:
            point = self._initial_point(seen_keys)
            return (point,) if point is not None else ()
        candidates = self._candidate_pool(seen_keys)
        if not candidates:
            point = self.fallback_point(seen_keys, ())
            return (point,) if point is not None else ()

        training_points = tuple(self._point_features(trial.point) for trial in trials)
        targets = self._model_targets(trials)
        model = GaussianProcess(
            training_points,
            targets,
            length_scale=self.length_scale,
            noise_variance=self.noise_variance,
        )
        best = min(targets)
        scored = tuple(
            (
                expected_improvement(
                    model.predict(self._point_features(point)),
                    best,
                    exploration=self.exploration,
                ),
                point.key,
                point,
            )
            for point in candidates
        )
        selected = min(scored, key=lambda item: (-item[0], item[1]))[2]
        return (selected,)

    def _tell(self, trials: tuple[Trial, ...]) -> None:
        self._evaluations += len(trials)
        for trial in trials:
            if trial.status is TrialStatus.FAILED:
                self.failed_observations += 1
                continue
            self.successful_observations += 1
            if trial.feasible:
                self._recent_feasible.append(trial)
                if self._best_feasible is None or preference_key(trial) < preference_key(
                    self._best_feasible
                ):
                    self._best_feasible = trial
            else:
                self._recent_infeasible.append(trial)
                if self._best_infeasible is None or preference_key(trial) < preference_key(
                    self._best_infeasible
                ):
                    self._best_infeasible = trial
