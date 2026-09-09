"""Particle swarm optimization with explicit unit-box boundary handling."""

from __future__ import annotations

import math
from dataclasses import dataclass

from biasweave.encoding import default_coordinates
from biasweave.errors import ConfigurationError
from biasweave.model import Point, Problem, Trial
from biasweave.optimizers.base import (
    AskTellOptimizer,
    finite_parameter,
    population_size_parameter,
)
from biasweave.optimizers.ranking import preference_key, preferred


def advance_particle(
    position: tuple[float, ...],
    velocity: tuple[float, ...],
    personal_best: tuple[float, ...],
    global_best: tuple[float, ...],
    *,
    inertia: float,
    cognitive: float,
    social: float,
    personal_draws: tuple[float, ...],
    social_draws: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Advance one particle and damp reflected velocity at the unit boundary."""
    dimensions = len(position)
    vectors = (velocity, personal_best, global_best, personal_draws, social_draws)
    if any(len(vector) != dimensions for vector in vectors):
        raise ConfigurationError("particle vectors must have matching dimensions")
    next_position: list[float] = []
    next_velocity: list[float] = []
    for index in range(dimensions):
        speed = (
            inertia * velocity[index]
            + cognitive * personal_draws[index] * (personal_best[index] - position[index])
            + social * social_draws[index] * (global_best[index] - position[index])
        )
        if not math.isfinite(speed):
            raise ConfigurationError("particle update produced a non-finite velocity")
        location = position[index]
        remaining = 1.0
        reflections = 0
        while speed != 0.0:
            boundary = 1.0 if speed > 0.0 else 0.0
            time_to_boundary = (boundary - location) / speed
            if time_to_boundary >= remaining:
                location += speed * remaining
                break
            location = boundary
            remaining -= max(0.0, time_to_boundary)
            speed = -speed * 0.5
            reflections += 1
            if reflections > 4096:  # Extreme finite coefficients must still be resource bounded.
                raise ConfigurationError("particle update exceeded the reflection limit")
        next_position.append(min(1.0, max(0.0, location)))
        next_velocity.append(speed)
    return tuple(next_position), tuple(next_velocity)


@dataclass(slots=True)
class _Particle:
    point: Point
    velocity: tuple[float, ...]
    current: Trial
    best: Trial


@dataclass(frozen=True, slots=True)
class _Move:
    index: int
    velocity: tuple[float, ...]


class ParticleSwarmOptimizer(AskTellOptimizer):
    """Canonical global-best PSO over the normalized mixed-variable encoding."""

    name = "pso"

    def __init__(
        self,
        problem: Problem,
        seed: int,
        *,
        population_size: int = 16,
        inertia: float = 0.72,
        cognitive: float = 1.49,
        social: float = 1.49,
    ):
        super().__init__(problem, seed)
        population_size = population_size_parameter(population_size, "PSO")
        inertia = finite_parameter(inertia, "inertia")
        cognitive = finite_parameter(cognitive, "cognitive")
        social = finite_parameter(social, "social")
        parameters = (inertia, cognitive, social)
        if not all(value >= 0.0 for value in parameters):
            raise ConfigurationError("PSO coefficients must be finite and non-negative")
        self.population_size = population_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.particles: list[_Particle] = []
        self._cursor = 0
        self._moves: list[_Move | None] = []

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
                if not self.particles and not points
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
        self._moves = [None] * len(points)
        return tuple(points)

    def _candidate(self, index: int, seen_keys: set[str], local_keys: set[str]) -> Point | None:
        particle = self.particles[index]
        global_best = min((item.best for item in self.particles), key=preference_key)
        for _attempt in range(200):
            personal_draws = self.random_vector()
            social_draws = self.random_vector()
            coordinates, velocity = advance_particle(
                particle.point.coordinates,
                particle.velocity,
                particle.best.point.coordinates,
                global_best.point.coordinates,
                inertia=self.inertia,
                cognitive=self.cognitive,
                social=self.social,
                personal_draws=personal_draws,
                social_draws=social_draws,
            )
            point = self.point_if_new(coordinates, seen_keys, local_keys)
            if point is not None:
                self._moves.append(_Move(index, velocity))
                return point
            # Decoding can collapse many normalized coordinates for integer and
            # categorical variables. A small independent kick avoids stalling.
            kicked = tuple(
                coordinate + self.random.uniform(-0.2, 0.2)
                for coordinate in particle.point.coordinates
            )
            point = self.point_if_new(kicked, seen_keys, local_keys)
            if point is not None:
                kick_velocity = tuple(
                    new - old
                    for new, old in zip(point.coordinates, particle.point.coordinates, strict=True)
                )
                self._moves.append(_Move(index, kick_velocity))
                return point
        point = self.fallback_point(seen_keys, local_keys)
        if point is not None:
            velocity = tuple(
                new - old
                for new, old in zip(point.coordinates, particle.point.coordinates, strict=True)
            )
            self._moves.append(_Move(index, velocity))
        return point

    def _ask(self, count: int, seen_keys: set[str]) -> tuple[Point, ...]:
        if len(self.particles) < self.population_size:
            missing = self.population_size - len(self.particles)
            return self._initial_points(min(count, missing), seen_keys)

        points: list[Point] = []
        local_keys: set[str] = set()
        self._moves = []
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
        return tuple(points)

    def _tell(self, trials: tuple[Trial, ...]) -> None:
        for trial, move in zip(trials, self._moves, strict=True):
            if move is None:
                velocity = tuple(self.random.uniform(-0.1, 0.1) for _ in range(self.dimension))
                self.particles.append(_Particle(trial.point, velocity, trial, trial))
                continue
            particle = self.particles[move.index]
            particle.point = trial.point
            particle.velocity = move.velocity
            particle.current = trial
            if preferred(trial, particle.best):
                particle.best = trial
        self._moves = []
