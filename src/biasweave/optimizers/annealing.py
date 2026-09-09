"""Constraint-first simulated annealing."""

from __future__ import annotations

import math

from biasweave.encoding import default_coordinates
from biasweave.errors import ConfigurationError
from biasweave.model import Point, Problem, Trial
from biasweave.optimizers.base import AskTellOptimizer, finite_parameter
from biasweave.optimizers.ranking import annealing_energy, preferred


class SimulatedAnnealingOptimizer(AskTellOptimizer):
    """A seeded, single-chain Metropolis search with geometric cooling."""

    name = "sa"

    def __init__(
        self,
        problem: Problem,
        seed: int,
        *,
        initial_temperature: float = 1.0,
        cooling: float = 0.96,
        minimum_temperature: float = 1e-3,
    ):
        super().__init__(problem, seed)
        initial_temperature = finite_parameter(initial_temperature, "initial_temperature")
        cooling = finite_parameter(cooling, "cooling")
        minimum_temperature = finite_parameter(minimum_temperature, "minimum_temperature")
        if initial_temperature <= 0.0:
            raise ConfigurationError("initial_temperature must be positive and finite")
        if not 0.0 < cooling < 1.0:
            raise ConfigurationError("cooling must be finite and in (0, 1)")
        if minimum_temperature <= 0.0:
            raise ConfigurationError("minimum_temperature must be positive and finite")
        self.temperature = initial_temperature
        self.cooling = cooling
        self.minimum_temperature = minimum_temperature
        self.current: Trial | None = None

    def _ask(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        del count  # A single Markov chain has one natural pending transition.
        local_keys: set[str] = set()
        if self.current is None:
            point = self.point_if_new(default_coordinates(self.problem), seen_keys, local_keys)
            if point is None:
                point = self._unique_neighbor(self.random_vector(), seen_keys)
        else:
            point = self._unique_neighbor(self.current.point.coordinates, seen_keys)
        if point is None:
            point = self.fallback_point(seen_keys, ())
        return (point,) if point is not None else ()

    def _unique_neighbor(self, anchor: tuple[float, ...], seen_keys: set[str]) -> Point | None:
        local_keys: set[str] = set()
        radius = max(0.015, min(0.35, 0.35 * math.sqrt(self.temperature)))
        for _attempt in range(300):
            coordinates = list(anchor)
            dimension = self.random.randrange(self.dimension)
            coordinates[dimension] += self.random.gauss(0.0, radius)
            # Occasionally move more than one axis to escape decoded plateaus.
            if self.dimension > 1 and self.random.random() < 0.25:
                other = (dimension + 1 + self.random.randrange(self.dimension - 1)) % self.dimension
                coordinates[other] += self.random.gauss(0.0, radius)
            point = self.point_if_new(coordinates, seen_keys, local_keys)
            if point is not None:
                return point
        for _attempt in range(300):
            point = self.point_if_new(self.random_vector(), seen_keys, local_keys)
            if point is not None:
                return point
        return None

    def _tell(self, trials: tuple[Trial, ...]) -> None:
        candidate = trials[0]
        if self.current is None or preferred(candidate, self.current):
            self.current = candidate
        elif (
            candidate.status is self.current.status and candidate.feasible == self.current.feasible
        ):
            delta = max(0.0, annealing_energy(candidate) - annealing_energy(self.current))
            probability = math.exp(-delta / self.temperature)
            if self.random.random() < probability:
                self.current = candidate
        self.temperature = max(self.minimum_temperature, self.temperature * self.cooling)
