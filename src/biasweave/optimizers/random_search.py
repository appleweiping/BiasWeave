"""Independent uniform-random search on BiasWeave's normalized domain."""

from __future__ import annotations

from biasweave.model import Point, Problem, Trial
from biasweave.optimizers.base import AskTellOptimizer


class RandomOptimizer(AskTellOptimizer):
    """Draw unique points from a seeded uniform stream."""

    name = "random"

    def __init__(self, problem: Problem, seed: int):
        super().__init__(problem, seed)

    def _ask(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        points: list[Point] = []
        local_keys: set[str] = set()
        attempts = 0
        maximum_attempts = max(200, count * 200)
        while len(points) < count and attempts < maximum_attempts:
            point = self.point_if_new(self.random_vector(), seen_keys, local_keys)
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

    def _tell(self, trials: tuple[Trial, ...]) -> None:
        # Random search intentionally has no adaptive state.
        del trials
