"""NSGA-II with binary tournament selection and real-coded variation."""

from __future__ import annotations

from biasweave.encoding import default_coordinates
from biasweave.errors import ConfigurationError
from biasweave.model import Point, Problem, Trial
from biasweave.optimizers.base import (
    AskTellOptimizer,
    clamp_coordinate,
    finite_parameter,
    population_size_parameter,
)
from biasweave.optimizers.ranking import (
    crowding_distance,
    non_dominated_sort,
    preference_key,
    select_population,
)


def simulated_binary_coordinate(left: float, right: float, draw: float, eta: float) -> float:
    """Return one bounded simulated-binary-crossover coordinate."""
    if draw <= 0.5:
        beta = (2.0 * draw) ** (1.0 / (eta + 1.0))
    else:
        beta = (1.0 / (2.0 * (1.0 - draw))) ** (1.0 / (eta + 1.0))
    return clamp_coordinate(0.5 * ((1.0 + beta) * left + (1.0 - beta) * right))


def polynomial_mutation_coordinate(value: float, draw: float, eta: float) -> float:
    """Mutate a coordinate using the bounded NSGA-II polynomial operator."""
    if draw < 0.5:
        delta = (2.0 * draw + (1.0 - 2.0 * draw) * (1.0 - value) ** (eta + 1.0)) ** (
            1.0 / (eta + 1.0)
        ) - 1.0
    else:
        delta = 1.0 - (2.0 * (1.0 - draw) + 2.0 * (draw - 0.5) * value ** (eta + 1.0)) ** (
            1.0 / (eta + 1.0)
        )
    return clamp_coordinate(value + delta)


class NSGA2Optimizer(AskTellOptimizer):
    """Elitist NSGA-II using exact non-dominated ranks and crowding."""

    name = "nsga2"

    def __init__(
        self,
        problem: Problem,
        seed: int,
        *,
        population_size: int = 16,
        crossover_probability: float = 0.9,
        crossover_eta: float = 15.0,
        mutation_eta: float = 20.0,
    ):
        super().__init__(problem, seed)
        population_size = population_size_parameter(population_size, "NSGA-II")
        crossover_probability = finite_parameter(crossover_probability, "crossover_probability")
        crossover_eta = finite_parameter(crossover_eta, "crossover_eta")
        mutation_eta = finite_parameter(mutation_eta, "mutation_eta")
        if not 0.0 <= crossover_probability <= 1.0:
            raise ConfigurationError("crossover_probability must be in [0, 1]")
        if crossover_eta <= 0.0:
            raise ConfigurationError("crossover_eta must be positive and finite")
        if mutation_eta <= 0.0:
            raise ConfigurationError("mutation_eta must be positive and finite")
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.population: list[Trial] = []
        self.offspring: list[Trial] = []

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
        return tuple(points)

    def _selection_keys(self) -> dict[str, tuple[object, ...]]:
        keys: dict[str, tuple[object, ...]] = {}
        for rank, front in enumerate(non_dominated_sort(self.population)):
            distances = crowding_distance(front)
            for trial in front:
                keys[trial.point.key] = (
                    rank,
                    -distances[trial.point.key],
                    preference_key(trial),
                )
        return keys

    def _parent(self, keys: dict[str, tuple[object, ...]]) -> Trial:
        left = self.population[self.random.randrange(len(self.population))]
        right = self.population[self.random.randrange(len(self.population))]
        return left if keys[left.point.key] <= keys[right.point.key] else right

    def _offspring_point(
        self,
        keys: dict[str, tuple[object, ...]],
        seen_keys: set[str],
        local_keys: set[str],
    ) -> Point | None:
        mutation_probability = 1.0 / self.dimension
        for _attempt in range(250):
            left = self._parent(keys).point.coordinates
            right = self._parent(keys).point.coordinates
            coordinates: list[float] = []
            for left_value, right_value in zip(left, right, strict=True):
                if self.random.random() < self.crossover_probability:
                    value = simulated_binary_coordinate(
                        left_value,
                        right_value,
                        self.random.random(),
                        self.crossover_eta,
                    )
                else:
                    value = left_value
                if self.random.random() < mutation_probability:
                    value = polynomial_mutation_coordinate(
                        value, self.random.random(), self.mutation_eta
                    )
                coordinates.append(value)
            point = self.point_if_new(coordinates, seen_keys, local_keys)
            if point is not None:
                return point
        # A uniform immigrant is still a standard diversity mechanism and is
        # essential when discrete decoding collapses genetic variation.
        return self.fallback_point(seen_keys, local_keys)

    def _ask(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        if len(self.population) < self.population_size:
            missing = self.population_size - len(self.population)
            return self._initial_points(min(count, missing), seen_keys)

        remaining = self.population_size - len(self.offspring)
        limit = min(count, remaining)
        keys = self._selection_keys()
        points: list[Point] = []
        local_keys: set[str] = set()
        while len(points) < limit:
            point = self._offspring_point(keys, seen_keys, local_keys)
            if point is None:
                break
            local_keys.add(point.key)
            points.append(point)
        return tuple(points)

    def _tell(self, trials: tuple[Trial, ...]) -> None:
        if len(self.population) < self.population_size:
            self.population.extend(trials)
            return
        self.offspring.extend(trials)
        if len(self.offspring) == self.population_size:
            self.population = list(
                select_population((*self.population, *self.offspring), self.population_size)
            )
            self.offspring = []
