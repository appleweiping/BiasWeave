"""A compact decomposition-based multi-objective evolutionary optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from biasweave.encoding import default_coordinates
from biasweave.errors import ConfigurationError
from biasweave.model import Point, Problem, Trial, TrialStatus
from biasweave.optimizers.base import (
    AskTellOptimizer,
    finite_parameter,
    population_size_parameter,
)
from biasweave.optimizers.differential_evolution import differential_trial
from biasweave.optimizers.ranking import preference_key

_MAX_WEIGHT_OBJECTIVES = 128


def _prime_bases(count: int) -> tuple[int, ...]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def simplex_weights(size: int, objectives: int) -> tuple[tuple[float, ...], ...]:
    """Construct unique deterministic Halton-simplex weights plus the extremes."""
    if size <= 0 or objectives <= 0:
        raise ConfigurationError("weight dimensions must be positive")
    if objectives > _MAX_WEIGHT_OBJECTIVES:
        raise ConfigurationError(f"MOEA/D supports at most {_MAX_WEIGHT_OBJECTIVES} objectives")
    weights: list[tuple[float, ...]] = []
    for objective in range(min(size, objectives)):
        weights.append(tuple(1.0 if index == objective else 0.0 for index in range(objectives)))
    seen = set(weights)
    bases = _prime_bases(objectives)
    index = 1
    while len(weights) < size:
        raw = tuple(_radical_inverse(index, base) for base in bases)
        total = math.fsum(raw)
        candidate = tuple(value / total for value in raw)
        if candidate not in seen:
            seen.add(candidate)
            weights.append(candidate)
        index += 1
    return tuple(weights)


def tchebycheff(trial: Trial, weight: tuple[float, ...], ideal: tuple[float, ...]) -> float:
    """Return the weighted Chebyshev distance from the current ideal point."""
    if len(trial.objective_vector) != len(weight) or len(weight) != len(ideal):
        raise ConfigurationError("scalarization vectors must have matching dimensions")
    return max(
        max(component, 1e-6) * abs(value - best)
        for value, component, best in zip(trial.objective_vector, weight, ideal, strict=True)
    )


@dataclass(frozen=True, slots=True)
class _Subproblem:
    index: int


class MOEADOptimizer(AskTellOptimizer):
    """MOEA/D with neighborhood DE variation and constraint-first replacement."""

    name = "moead"

    def __init__(
        self,
        problem: Problem,
        seed: int,
        *,
        population_size: int = 16,
        neighborhood_size: int | None = None,
        differential_weight: float = 0.6,
        crossover_rate: float = 0.9,
    ):
        super().__init__(problem, seed)
        population_size = population_size_parameter(population_size, "MOEA/D")
        if population_size < len(problem.objectives):
            raise ConfigurationError(
                "MOEA/D population_size must be at least the number of objectives"
            )
        neighborhood = (
            max(4, round(math.sqrt(population_size)))
            if neighborhood_size is None
            else neighborhood_size
        )
        if isinstance(neighborhood, bool) or not isinstance(neighborhood, int):
            raise ConfigurationError("neighborhood_size must be an integer")
        if not 4 <= neighborhood <= population_size:
            raise ConfigurationError("neighborhood_size must be in [4, population_size]")
        differential_weight = finite_parameter(differential_weight, "differential_weight")
        crossover_rate = finite_parameter(crossover_rate, "crossover_rate")
        if not 0.0 < differential_weight <= 2.0:
            raise ConfigurationError("differential_weight must be in (0, 2]")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ConfigurationError("crossover_rate must be in [0, 1]")
        self.population_size = population_size
        self.neighborhood_size = neighborhood
        self.differential_weight = differential_weight
        self.crossover_rate = crossover_rate
        self.weights = simplex_weights(population_size, len(problem.objectives))
        self.neighborhoods = tuple(
            tuple(
                sorted(
                    range(population_size),
                    key=lambda other: (
                        math.dist(self.weights[index], self.weights[other]),
                        other,
                    ),
                )[:neighborhood]
            )
            for index in range(population_size)
        )
        self.population: list[Trial] = []
        self.ideal = tuple(math.inf for _ in problem.objectives)
        self._cursor = 0
        self._subproblems: list[_Subproblem | None] = []

    @property
    def recommended_batch_size(self) -> int:
        return self.population_size

    def _initial_points(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        points: list[Point] = []
        local_keys: set[str] = set()
        attempts = 0
        while len(points) < count and attempts < max(300, count * 300):
            coordinates = (
                default_coordinates(self.problem)
                if not self.population and not points
                else self.random_vector()
            )
            point = self.point_if_new(coordinates, seen_keys, local_keys)
            attempts += 1
            if point is None:
                continue
            local_keys.add(point.key)
            points.append(point)
        while len(points) < count:
            point = self.fallback_point(seen_keys, local_keys)
            if point is None:
                break
            local_keys.add(point.key)
            points.append(point)
        self._subproblems = [None] * len(points)
        return tuple(points)

    def _candidate(self, index: int, seen_keys: set[str], local_keys: set[str]) -> Point | None:
        neighborhood = self.neighborhoods[index]
        alternatives = [other for other in neighborhood if other != index]
        for _attempt in range(250):
            base_index, addend_index, subtrahend_index = self.random.sample(alternatives, 3)
            coordinates = differential_trial(
                self.population[index].point.coordinates,
                self.population[base_index].point.coordinates,
                self.population[addend_index].point.coordinates,
                self.population[subtrahend_index].point.coordinates,
                differential_weight=self.differential_weight,
                crossover_rate=self.crossover_rate,
                crossover_draws=self.random_vector(),
                forced_dimension=self.random.randrange(self.dimension),
            )
            point = self.point_if_new(coordinates, seen_keys, local_keys)
            if point is not None:
                return point
        # Neighborhood replacement can intentionally place one strong solution
        # in several subproblems. Inject a uniform immigrant when that collapses
        # all differential vectors to the same decoded point.
        return self.fallback_point(seen_keys, local_keys)

    def _ask(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        if len(self.population) < self.population_size:
            missing = self.population_size - len(self.population)
            return self._initial_points(min(count, missing), seen_keys)

        points: list[Point] = []
        local_keys: set[str] = set()
        self._subproblems = []
        attempted = 0
        limit = min(count, self.population_size)
        while len(points) < limit and attempted < self.population_size:
            index = self._cursor
            self._cursor = (self._cursor + 1) % self.population_size
            attempted += 1
            point = self._candidate(index, seen_keys, local_keys)
            if point is not None:
                local_keys.add(point.key)
                points.append(point)
                self._subproblems.append(_Subproblem(index))
        return tuple(points)

    def _update_ideal(self, trial: Trial) -> None:
        if trial.status is TrialStatus.SUCCESS and trial.feasible:
            self.ideal = tuple(
                min(current, value)
                for current, value in zip(self.ideal, trial.objective_vector, strict=True)
            )

    def _better_for(self, candidate: Trial, incumbent: Trial, subproblem: int) -> bool:
        if candidate.status is TrialStatus.FAILED:
            return False
        if incumbent.status is TrialStatus.FAILED:
            return True
        if candidate.feasible != incumbent.feasible:
            return candidate.feasible
        if not candidate.feasible:
            return preference_key(candidate) < preference_key(incumbent)
        candidate_value = tchebycheff(candidate, self.weights[subproblem], self.ideal)
        incumbent_value = tchebycheff(incumbent, self.weights[subproblem], self.ideal)
        return (candidate_value, preference_key(candidate)) < (
            incumbent_value,
            preference_key(incumbent),
        )

    def _tell(self, trials: tuple[Trial, ...]) -> None:
        if len(self.population) < self.population_size:
            self.population.extend(trials)
            for trial in trials:
                self._update_ideal(trial)
            return
        for trial, subproblem in zip(trials, self._subproblems, strict=True):
            if subproblem is None:
                raise RuntimeError("evolutionary MOEA/D trial has no subproblem")
            self._update_ideal(trial)
            for neighbor in self.neighborhoods[subproblem.index]:
                if self._better_for(trial, self.population[neighbor], neighbor):
                    self.population[neighbor] = trial
        self._subproblems = []
