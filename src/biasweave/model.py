"""Immutable optimizer domain objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeAlias

Scalar: TypeAlias = int | float | str


class VariableKind(StrEnum):
    REAL = "real"
    INTEGER = "integer"
    CHOICE = "choice"
    LINKED = "linked"


class VariableScale(StrEnum):
    LINEAR = "linear"
    LOG = "log"


class Goal(StrEnum):
    MIN = "min"
    MAX = "max"


class Relation(StrEnum):
    GE = "ge"
    LE = "le"
    BETWEEN = "between"


class TrialStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Variable:
    name: str
    kind: VariableKind
    low: float | int | None = None
    high: float | int | None = None
    scale: VariableScale = VariableScale.LINEAR
    quantum: float | None = None
    values: tuple[Scalar, ...] = ()
    default: Scalar | None = None
    source: str | None = None
    factor: float = 1.0
    offset: float = 0.0

    @property
    def free(self) -> bool:
        return self.kind is not VariableKind.LINKED


@dataclass(frozen=True, slots=True)
class Objective:
    metric: str
    goal: Goal
    scale: float
    epsilon: float
    reference: float = 0.0


@dataclass(frozen=True, slots=True)
class Constraint:
    metric: str
    relation: Relation
    scale: float
    tolerance: float = 0.0
    limit: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True, slots=True)
class Problem:
    variables: tuple[Variable, ...]
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...]
    source_hash: str
    source_path: str

    @property
    def free_variables(self) -> tuple[Variable, ...]:
        return tuple(variable for variable in self.variables if variable.free)

    @property
    def variable_map(self) -> dict[str, Variable]:
        return {variable.name: variable for variable in self.variables}


@dataclass(frozen=True, slots=True)
class Point:
    coordinates: tuple[float, ...]
    values: dict[str, Scalar]
    key: str


@dataclass(frozen=True, slots=True)
class Trial:
    trial_id: int
    point: Point
    status: TrialStatus
    metrics: dict[str, float]
    error: str | None
    feasible: bool
    violation: float
    max_violation: float
    objective_vector: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "point": {
                "coordinates": list(self.point.coordinates),
                "values": self.point.values,
                "key": self.point.key,
            },
            "status": self.status,
            "metrics": self.metrics,
            "error": self.error,
            "feasible": self.feasible,
            "violation": self.violation if math.isfinite(self.violation) else None,
            "max_violation": self.max_violation if math.isfinite(self.max_violation) else None,
            "objective_vector": list(self.objective_vector),
        }


@dataclass(frozen=True, slots=True)
class RunConfig:
    budget: int
    seed: int = 0
    workers: int = 1
    batch_size: int = 8
    max_stagnation: int = 0
    wall_time_seconds: float | None = None
    command_timeout_seconds: float = 300.0


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    problem: Problem
    trials: tuple[Trial, ...]
    frontier: tuple[Trial, ...]
    stop_reason: str
    evaluator_id: str
    seed: int

    @property
    def successful_trials(self) -> int:
        return sum(trial.status is TrialStatus.SUCCESS for trial in self.trials)

    @property
    def failed_trials(self) -> int:
        return len(self.trials) - self.successful_trials
