"""Mixed-variable encoding, decoding, and stable point identity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from biasweave.errors import ProblemError
from biasweave.model import Point, Problem, Scalar, Variable, VariableKind, VariableScale


def _finite_float(value: int | float, context: str) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ProblemError(f"{context} must be finite and representable") from error
    if not math.isfinite(result):
        raise ProblemError(f"{context} must be finite and representable")
    return result


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        raise ProblemError("normalized coordinates must be finite")
    return min(1.0, max(0.0, value))


def _decode_variable(variable: Variable, coordinate: float) -> Scalar:
    unit = _clamp(coordinate)
    if variable.kind is VariableKind.REAL:
        if not isinstance(variable.low, float | int) or not isinstance(variable.high, float | int):
            raise ProblemError(f"real variable {variable.name} requires numeric bounds")
        low = _finite_float(variable.low, f"lower bound for {variable.name}")
        high = _finite_float(variable.high, f"upper bound for {variable.name}")
        if unit == 0.0:
            value = low
        elif unit == 1.0:
            value = high
        elif variable.scale is VariableScale.LOG:
            value = math.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
        else:
            value = low + unit * (high - low)
        if variable.quantum is not None and 0.0 < unit < 1.0:
            value = low + round((value - low) / variable.quantum) * variable.quantum
        value = min(high, max(low, value))
        return 0.0 if value == 0.0 else value
    if variable.kind is VariableKind.INTEGER:
        if not isinstance(variable.low, int) or not isinstance(variable.high, int):
            raise ProblemError(f"integer variable {variable.name} requires integer bounds")
        return min(variable.high, round(variable.low + unit * (variable.high - variable.low)))
    if variable.kind is VariableKind.CHOICE:
        index = min(len(variable.values) - 1, int(unit * len(variable.values)))
        return variable.values[index]
    raise ProblemError(f"linked variable {variable.name} cannot consume a coordinate")


def _linked_value(
    variable: Variable, variables: Mapping[str, Variable], values: dict[str, Scalar]
) -> float:
    if variable.source is None:
        raise ProblemError(f"linked variable {variable.name} requires a source")
    if variable.source not in values:
        source = variables[variable.source]
        if source.kind is not VariableKind.LINKED:
            raise ProblemError(f"source value was not decoded: {source.name}")
        values[source.name] = _linked_value(source, variables, values)
    source_value = values[variable.source]
    if isinstance(source_value, str):
        raise ProblemError(f"linked source {variable.source} is not numeric")
    value = (
        _finite_float(source_value, f"linked source {variable.source}") * variable.factor
        + variable.offset
    )
    if not math.isfinite(value):
        raise ProblemError(f"linked variable {variable.name} produced a non-finite value")
    return 0.0 if value == 0.0 else value


def decode(problem: Problem, coordinates: Sequence[float]) -> dict[str, Scalar]:
    """Decode normalized free-variable coordinates and then resolve links."""
    free = problem.free_variables
    if len(coordinates) != len(free):
        raise ProblemError(f"expected {len(free)} coordinates, received {len(coordinates)}")
    values = {
        variable.name: _decode_variable(variable, float(coordinate))
        for variable, coordinate in zip(free, coordinates, strict=True)
    }
    variables = problem.variable_map
    for variable in problem.variables:
        if variable.kind is VariableKind.LINKED:
            values[variable.name] = _linked_value(variable, variables, values)
    return {variable.name: values[variable.name] for variable in problem.variables}


def _encode_variable(variable: Variable, value: Scalar) -> float:
    if variable.kind is VariableKind.CHOICE:
        try:
            index = variable.values.index(value)
        except ValueError as error:
            raise ProblemError(f"{value!r} is not a choice for {variable.name}") from error
        return (index + 0.5) / len(variable.values)
    if isinstance(value, str):
        raise ProblemError(f"numeric variable {variable.name} received a string")
    numeric = _finite_float(value, f"value for {variable.name}")
    if not isinstance(variable.low, int | float) or not isinstance(variable.high, int | float):
        raise ProblemError(f"numeric variable {variable.name} requires numeric bounds")
    low = _finite_float(variable.low, f"lower bound for {variable.name}")
    high = _finite_float(variable.high, f"upper bound for {variable.name}")
    if not low <= numeric <= high:
        raise ProblemError(f"value for {variable.name} is outside [{low}, {high}]")
    if high == low:
        return 0.5
    if variable.scale is VariableScale.LOG:
        return (math.log(numeric) - math.log(low)) / (math.log(high) - math.log(low))
    return (numeric - low) / (high - low)


def encode(problem: Problem, values: Mapping[str, Scalar]) -> tuple[float, ...]:
    """Encode explicitly supplied free-variable values."""
    missing = [variable.name for variable in problem.free_variables if variable.name not in values]
    if missing:
        raise ProblemError(f"missing free-variable values: {', '.join(missing)}")
    return tuple(
        _encode_variable(variable, values[variable.name]) for variable in problem.free_variables
    )


def default_coordinates(problem: Problem) -> tuple[float, ...]:
    defaults = {variable.name: variable.default for variable in problem.free_variables}
    return encode(problem, defaults)  # type: ignore[arg-type]


def point_key(values: Mapping[str, Scalar]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def make_point(problem: Problem, coordinates: Sequence[float]) -> Point:
    normalized = tuple(_clamp(float(value)) for value in coordinates)
    values = decode(problem, normalized)
    return Point(normalized, values, point_key(values))
