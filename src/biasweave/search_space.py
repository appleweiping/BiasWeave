"""Bounded, deterministic enumeration of provably finite decoded domains."""

from __future__ import annotations

import math
from fractions import Fraction

from biasweave.encoding import _log_coordinate, default_coordinates, make_point
from biasweave.model import Point, Problem, VariableKind, VariableScale

MAX_ENUMERABLE_POINTS = 100_000


def _quantized_axis_size(low: float, high: float, quantum: float) -> int | None:
    ratio = (Fraction.from_float(high) - Fraction.from_float(low)) / Fraction.from_float(quantum)
    return math.ceil(ratio) + 2


def finite_axes(problem: Problem) -> tuple[tuple[float, ...], ...] | None:
    """Return one exact decoded representative per value in a small finite domain."""

    potential = 1
    for variable in problem.free_variables:
        size: int | None
        if variable.kind is VariableKind.INTEGER:
            if not isinstance(variable.low, int) or not isinstance(variable.high, int):
                return None
            size = variable.high - variable.low + 1
        elif variable.kind is VariableKind.CHOICE:
            size = len(variable.values)
        elif variable.kind is VariableKind.REAL and variable.quantum is not None:
            if not isinstance(variable.low, int | float) or not isinstance(
                variable.high, int | float
            ):
                return None
            size = _quantized_axis_size(float(variable.low), float(variable.high), variable.quantum)
            if size is None:
                return None
        else:
            return None
        potential *= size
        if potential > MAX_ENUMERABLE_POINTS:
            return None

    defaults = default_coordinates(problem)
    axes: list[tuple[float, ...]] = []
    for variable_index, variable in enumerate(problem.free_variables):
        candidates: tuple[float, ...]
        if variable.kind is VariableKind.INTEGER:
            if not isinstance(variable.low, int) or not isinstance(variable.high, int):
                return None
            span = variable.high - variable.low
            if span == 0:
                candidates = (0.5,)
            else:
                # Fraction avoids first converting a large integer bound to a
                # lossy IEEE-754 value. The domain is bounded above, so every
                # offset is small even when the absolute bounds are not.
                candidates = tuple(float(Fraction(offset, span)) for offset in range(span + 1))
        elif variable.kind is VariableKind.CHOICE:
            candidates = tuple(
                (index + 0.5) / len(variable.values) for index in range(len(variable.values))
            )
        else:
            if (
                variable.kind is not VariableKind.REAL
                or variable.quantum is None
                or not isinstance(variable.low, int | float)
                or not isinstance(variable.high, int | float)
            ):
                return None
            low = float(variable.low)
            high = float(variable.high)
            low_fraction = Fraction.from_float(low)
            high_fraction = Fraction.from_float(high)
            quantum_fraction = Fraction.from_float(variable.quantum)
            ratio = (high_fraction - low_fraction) / quantum_fraction
            maximum_level = max(0, math.ceil(ratio + Fraction(1, 2)) - 1)
            candidates_list: list[float] = [0.0, 1.0]
            for level in range(maximum_level + 1):
                lower_ratio = max(Fraction(0), Fraction(2 * level - 1, 2))
                upper_ratio = min(ratio, Fraction(2 * level + 1, 2))
                if upper_ratio > lower_ratio:
                    representative = (lower_ratio + upper_ratio) / 2
                    exact_value = low_fraction + representative * quantum_fraction
                    if variable.scale is VariableScale.LOG:
                        value = float(exact_value)
                        coordinate = _log_coordinate(low, high, value)
                    else:
                        coordinate = float(
                            (exact_value - low_fraction) / (high_fraction - low_fraction)
                        )
                    candidates_list.append(min(1.0, max(0.0, coordinate)))
            candidates = tuple(candidates_list)

        representatives: dict[object, float] = {}
        for coordinate in candidates:
            coordinates = list(defaults)
            coordinates[variable_index] = coordinate
            point = make_point(problem, coordinates)
            representatives.setdefault(
                point.values[variable.name], point.coordinates[variable_index]
            )
        axes.append(tuple(representatives.values()))

    if math.prod(len(axis) for axis in axes) > MAX_ENUMERABLE_POINTS:
        return None
    return tuple(axes)


class FiniteDomainEnumerator:
    """Walk a small finite product exactly once and expose proof of exhaustion."""

    def __init__(self, problem: Problem) -> None:
        self._problem = problem
        self._axes = finite_axes(problem)
        self._cursor = 0

    @property
    def finite(self) -> bool:
        return self._axes is not None

    @property
    def axes(self) -> tuple[tuple[float, ...], ...] | None:
        """Expose immutable representatives for diagnostics and compatibility."""
        return self._axes

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def exhausted(self) -> bool:
        return self._axes is not None and self._cursor >= self.cardinality

    @property
    def cardinality(self) -> int:
        return 0 if self._axes is None else math.prod(len(axis) for axis in self._axes)

    def restore_cursor(self, cursor: object) -> None:
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("finite_cursor must be a non-negative integer")
        if self._axes is None:
            if cursor != 0:
                raise ValueError("finite_cursor must be zero for a non-enumerable domain")
        elif cursor > self.cardinality:
            raise ValueError("finite_cursor exceeds the enumerable domain")
        self._cursor = cursor

    def next_unseen(self, blocked: set[str], local: set[str]) -> Point | None:
        if self._axes is None:
            return None
        while self._cursor < self.cardinality:
            index = self._cursor
            self._cursor += 1
            coordinates: list[float] = []
            for axis in reversed(self._axes):
                coordinates.append(axis[index % len(axis)])
                index //= len(axis)
            point = make_point(self._problem, tuple(reversed(coordinates)))
            if point.key not in blocked and point.key not in local:
                return point
        return None
