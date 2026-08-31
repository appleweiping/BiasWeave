"""Deterministic coverage, frontier, and feasibility-repair proposals."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping
from typing import Any

from biasweave.archive import Archive
from biasweave.encoding import default_coordinates, make_point
from biasweave.model import Point, Problem, Trial, TrialStatus

_STRIDES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
_MODULUS = 1009


class ProposalGenerator:
    """Generate unique points while carrying only deterministic search state."""

    def __init__(self, problem: Problem, seed: int):
        self.problem = problem
        self.seed = seed
        # This deterministic generator is for search coverage, not cryptography.
        self.random = random.Random(seed)  # nosec B311
        self.coverage_index = 0
        self.proposal_index = 0
        self.radius = 0.2

    def restore(self, completed_trials: int) -> None:
        """Use conservative counters when resuming a legacy checkpoint."""
        self.coverage_index = completed_trials
        self.proposal_index = completed_trials

    def snapshot(self) -> dict[str, Any]:
        return {
            "coverage_index": self.coverage_index,
            "proposal_index": self.proposal_index,
            "radius": self.radius,
            "random_state": self.random.getstate(),
        }

    def restore_snapshot(self, state: Mapping[str, Any]) -> None:
        def tuples(value: Any) -> Any:
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value

        required = {"coverage_index", "proposal_index", "radius", "random_state"}
        if set(state) != required:
            raise ValueError("generator snapshot has unexpected fields")
        coverage = state["coverage_index"]
        proposal = state["proposal_index"]
        radius = state["radius"]
        if isinstance(coverage, bool) or not isinstance(coverage, int) or coverage < 0:
            raise ValueError("coverage_index must be a non-negative integer")
        if isinstance(proposal, bool) or not isinstance(proposal, int) or proposal < 0:
            raise ValueError("proposal_index must be a non-negative integer")
        if isinstance(radius, bool) or not isinstance(radius, int | float):
            raise ValueError("radius must be numeric")
        numeric_radius = float(radius)
        if not math.isfinite(numeric_radius) or not 0.0 < numeric_radius <= 1.0:
            raise ValueError("radius must be finite and in (0, 1]")
        self.random.setstate(tuples(state["random_state"]))
        self.coverage_index = coverage
        self.proposal_index = proposal
        self.radius = numeric_radius

    def observe(self, frontier_grew: bool) -> None:
        if frontier_grew:
            self.radius = min(0.35, self.radius * 1.15)
        else:
            self.radius = max(0.015, self.radius * 0.82)

    def _coverage(self) -> tuple[float, ...]:
        index = self.coverage_index
        self.coverage_index += 1
        offset = self.seed % _MODULUS
        return tuple(
            (((index + 1) * _STRIDES[dimension % len(_STRIDES)] + offset) % _MODULUS + 0.5)
            / _MODULUS
            for dimension in range(len(self.problem.free_variables))
        )

    def _mutate(self, anchor: Trial) -> tuple[float, ...]:
        coordinates = list(anchor.point.coordinates)
        dimensions = len(coordinates)
        count = min(dimensions, 1 + self.proposal_index % min(3, dimensions))
        selected = self.random.sample(range(dimensions), count)
        for dimension in selected:
            step = self.random.uniform(-self.radius, self.radius)
            coordinates[dimension] = min(1.0, max(0.0, coordinates[dimension] + step))
        return tuple(coordinates)

    def _repair(self, archive: Archive, trials: tuple[Trial, ...]) -> tuple[float, ...]:
        anchor = archive.best_infeasible()
        if anchor is None:
            frontier_anchor = archive.sparse_anchor()
            return self._mutate(frontier_anchor) if frontier_anchor else self._coverage()

        slopes: list[tuple[float, int, float]] = []
        for dimension, anchor_coordinate in enumerate(anchor.point.coordinates):
            estimates = []
            for trial in trials:
                if trial.status is not TrialStatus.SUCCESS or trial is anchor:
                    continue
                delta = trial.point.coordinates[dimension] - anchor_coordinate
                if abs(delta) > 1e-9:
                    estimates.append((trial.violation - anchor.violation) / delta)
            if estimates:
                slope = statistics.median(estimates)
                slopes.append((abs(slope), dimension, slope))
        if not slopes:
            return self._mutate(anchor)
        _magnitude, dimension, slope = max(slopes)
        coordinates = list(anchor.point.coordinates)
        direction = -1.0 if slope > 0.0 else 1.0
        coordinates[dimension] = min(
            1.0,
            max(0.0, coordinates[dimension] + direction * self.radius),
        )
        return tuple(coordinates)

    def _strand(self, archive: Archive, trials: tuple[Trial, ...]) -> tuple[float, ...]:
        slot = self.proposal_index % 5
        self.proposal_index += 1
        if slot < 2:
            return self._coverage()
        if slot < 4 and archive.frontier:
            anchor = archive.sparse_anchor()
            if anchor is None:
                raise RuntimeError("frontier anchor disappeared")
            return self._mutate(anchor)
        return self._repair(archive, trials)

    def propose(
        self,
        count: int,
        archive: Archive,
        trials: tuple[Trial, ...],
        seen_keys: set[str],
    ) -> list[Point]:
        """Return at most `count` new decoded points."""
        points: list[Point] = []
        local_seen = set(seen_keys)
        attempts = 0
        maximum_attempts = max(100, count * 100)
        while len(points) < count and attempts < maximum_attempts:
            if not local_seen and not points:
                coordinates = default_coordinates(self.problem)
            else:
                coordinates = self._strand(archive, trials)
            point = make_point(self.problem, coordinates)
            attempts += 1
            if point.key in local_seen:
                continue
            local_seen.add(point.key)
            points.append(point)
        return points
