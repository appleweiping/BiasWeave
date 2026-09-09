"""Constraint-first differential evolution."""

from __future__ import annotations

from dataclasses import dataclass

from biasweave.encoding import default_coordinates
from biasweave.errors import ConfigurationError
from biasweave.model import Point, Problem, Trial
from biasweave.optimizers.base import (
    AskTellOptimizer,
    clamp_vector,
    finite_parameter,
    population_size_parameter,
)
from biasweave.optimizers.ranking import preferred


def differential_trial(
    target: tuple[float, ...],
    base: tuple[float, ...],
    addend: tuple[float, ...],
    subtrahend: tuple[float, ...],
    *,
    differential_weight: float,
    crossover_rate: float,
    crossover_draws: tuple[float, ...],
    forced_dimension: int,
) -> tuple[float, ...]:
    """Build one DE/rand/1/bin trial and project it into the unit box."""
    dimensions = len(target)
    vectors = (base, addend, subtrahend, crossover_draws)
    if any(len(vector) != dimensions for vector in vectors):
        raise ConfigurationError("DE vectors must have matching dimensions")
    if not 0 <= forced_dimension < dimensions:
        raise ConfigurationError("forced_dimension is outside the vector")
    donor = clamp_vector(
        tuple(
            base[index] + differential_weight * (addend[index] - subtrahend[index])
            for index in range(dimensions)
        )
    )
    return tuple(
        donor[index]
        if index == forced_dimension or crossover_draws[index] < crossover_rate
        else target[index]
        for index in range(dimensions)
    )


@dataclass(frozen=True, slots=True)
class _Target:
    index: int


class DifferentialEvolutionOptimizer(AskTellOptimizer):
    """DE/rand/1/bin with deterministic target order and feasibility-first survival."""

    name = "de"

    def __init__(
        self,
        problem: Problem,
        seed: int,
        *,
        population_size: int = 16,
        differential_weight: float = 0.8,
        crossover_rate: float = 0.9,
    ):
        super().__init__(problem, seed)
        population_size = population_size_parameter(population_size, "DE")
        differential_weight = finite_parameter(differential_weight, "differential_weight")
        crossover_rate = finite_parameter(crossover_rate, "crossover_rate")
        if not 0.0 < differential_weight <= 2.0:
            raise ConfigurationError("differential_weight must be in (0, 2]")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ConfigurationError("crossover_rate must be in [0, 1]")
        self.population_size = population_size
        self.differential_weight = differential_weight
        self.crossover_rate = crossover_rate
        self.population: list[Trial] = []
        self._cursor = 0
        self._targets: list[_Target | None] = []

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
        self._targets = [None] * len(points)
        return tuple(points)

    def _candidate(self, target_index: int, seen: set[str], local: set[str]) -> Point | None:
        choices = [index for index in range(self.population_size) if index != target_index]
        for _attempt in range(250):
            base_index, addend_index, subtrahend_index = self.random.sample(choices, 3)
            point_coordinates = differential_trial(
                self.population[target_index].point.coordinates,
                self.population[base_index].point.coordinates,
                self.population[addend_index].point.coordinates,
                self.population[subtrahend_index].point.coordinates,
                differential_weight=self.differential_weight,
                crossover_rate=self.crossover_rate,
                crossover_draws=self.random_vector(),
                forced_dimension=self.random.randrange(self.dimension),
            )
            point = self.point_if_new(point_coordinates, seen, local)
            if point is not None:
                return point
        return self.fallback_point(seen, local)

    def _ask(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        if len(self.population) < self.population_size:
            missing = self.population_size - len(self.population)
            return self._initial_points(min(count, missing), seen_keys)

        points: list[Point] = []
        local_keys: set[str] = set()
        self._targets = []
        attempted = 0
        limit = min(count, self.population_size)
        while len(points) < limit and attempted < self.population_size:
            target_index = self._cursor
            self._cursor = (self._cursor + 1) % self.population_size
            attempted += 1
            point = self._candidate(target_index, seen_keys, local_keys)
            if point is not None:
                local_keys.add(point.key)
                points.append(point)
                self._targets.append(_Target(target_index))
        return tuple(points)

    def _tell(self, trials: tuple[Trial, ...]) -> None:
        for trial, target in zip(trials, self._targets, strict=True):
            if target is None:
                self.population.append(trial)
            elif preferred(trial, self.population[target.index]):
                self.population[target.index] = trial
        self._targets = []
