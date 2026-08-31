"""In-memory exact frontier and proposal-anchor selection."""

from __future__ import annotations

import math

from biasweave.dominance import pareto_front
from biasweave.model import Problem, Trial, TrialStatus


class Archive:
    """Track successful trials while preserving the exact feasible frontier."""

    def __init__(self, problem: Problem, trials: list[Trial] | None = None):
        self.problem = problem
        self._trials: list[Trial] = []
        self._frontier: tuple[Trial, ...] = ()
        for trial in trials or []:
            self.add(trial)

    @property
    def trials(self) -> tuple[Trial, ...]:
        return tuple(self._trials)

    @property
    def frontier(self) -> tuple[Trial, ...]:
        return self._frontier

    @property
    def infeasible(self) -> tuple[Trial, ...]:
        return tuple(
            trial
            for trial in self._trials
            if trial.status is TrialStatus.SUCCESS and not trial.feasible
        )

    @property
    def signature(self) -> tuple[int, ...]:
        return tuple(trial.trial_id for trial in self._frontier)

    def add(self, trial: Trial) -> bool:
        previous = self.signature
        if trial.status is TrialStatus.SUCCESS:
            self._trials.append(trial)
            self._frontier = pareto_front(self._trials)
        return self.signature != previous

    def best_infeasible(self) -> Trial | None:
        candidates = self.infeasible
        if not candidates:
            return None
        return min(
            candidates, key=lambda trial: (trial.violation, trial.max_violation, trial.trial_id)
        )

    def sparse_anchor(self) -> Trial | None:
        """Choose a frontier point in a low-occupancy objective epsilon cell."""
        if not self._frontier:
            return None
        if len(self._frontier) == 1:
            return self._frontier[0]
        cells: dict[tuple[int, ...], list[Trial]] = {}
        for trial in self._frontier:
            cell = tuple(
                math.floor(value / objective.epsilon)
                for value, objective in zip(
                    trial.objective_vector, self.problem.objectives, strict=True
                )
            )
            cells.setdefault(cell, []).append(trial)
        minimum_occupancy = min(len(items) for items in cells.values())
        candidates = [
            trial for items in cells.values() if len(items) == minimum_occupancy for trial in items
        ]

        def nearest_distance(trial: Trial) -> float:
            return min(
                math.dist(trial.objective_vector, other.objective_vector)
                for other in self._frontier
                if other is not trial
            )

        return max(candidates, key=lambda trial: (nearest_distance(trial), -trial.trial_id))
