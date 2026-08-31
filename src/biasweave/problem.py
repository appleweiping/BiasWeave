"""TOML problem definition and semantic validation."""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from biasweave.errors import ProblemError
from biasweave.model import (
    Constraint,
    Goal,
    Objective,
    Problem,
    Relation,
    Variable,
    VariableKind,
    VariableScale,
)


def _table(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProblemError(f"{context} must be a table")
    return value


def _unknown(table: Mapping[str, Any], allowed: set[str], context: str) -> None:
    extra = sorted(set(table) - allowed)
    if extra:
        raise ProblemError(f"{context} has unsupported fields: {', '.join(extra)}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProblemError(f"{context} must be a non-empty string")
    return value.strip()


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProblemError(f"{context} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ProblemError(f"{context} must be finite") from error
    if not math.isfinite(result):
        raise ProblemError(f"{context} must be finite")
    return result


def _positive(value: Any, context: str) -> float:
    result = _finite(value, context)
    if result <= 0.0:
        raise ProblemError(f"{context} must be greater than zero")
    return result


def _enum(enum_type: type[Any], value: Any, context: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        options = ", ".join(item.value for item in enum_type)
        raise ProblemError(f"{context} must be one of: {options}") from error


def _scalar(value: Any, context: str) -> int | float | str:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ProblemError(f"{context} must be a string or finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ProblemError(f"{context} must be finite")
    return value


def _bounded_default(default: Any, low: float, high: float, context: str) -> float:
    result = _finite(default, context)
    if not low <= result <= high:
        raise ProblemError(f"{context} must be within [{low}, {high}]")
    return result


def _variable(name: str, raw: Any) -> Variable:
    context = f"variables.{name}"
    table = _table(raw, context)
    kind = _enum(VariableKind, table.get("kind"), f"{context}.kind")

    if kind is VariableKind.REAL:
        _unknown(table, {"kind", "low", "high", "scale", "quantum", "default"}, context)
        low = _finite(table.get("low"), f"{context}.low")
        high = _finite(table.get("high"), f"{context}.high")
        if low >= high:
            raise ProblemError(f"{context}.low must be less than high")
        scale = _enum(VariableScale, table.get("scale", "linear"), f"{context}.scale")
        if scale is VariableScale.LOG and low <= 0.0:
            raise ProblemError(f"{context}.low must be positive for logarithmic scaling")
        quantum = table.get("quantum")
        quantum_value = _positive(quantum, f"{context}.quantum") if quantum is not None else None
        default = table.get(
            "default", math.sqrt(low * high) if scale is VariableScale.LOG else (low + high) / 2
        )
        default_value = _bounded_default(default, low, high, f"{context}.default")
        return Variable(name, kind, low, high, scale, quantum_value, default=default_value)

    if kind is VariableKind.INTEGER:
        _unknown(table, {"kind", "low", "high", "default"}, context)
        low_raw = table.get("low")
        high_raw = table.get("high")
        if isinstance(low_raw, bool) or not isinstance(low_raw, int):
            raise ProblemError(f"{context}.low must be an integer")
        if isinstance(high_raw, bool) or not isinstance(high_raw, int):
            raise ProblemError(f"{context}.high must be an integer")
        if low_raw > high_raw:
            raise ProblemError(f"{context}.low must not exceed high")
        default = table.get("default", (low_raw + high_raw) // 2)
        if isinstance(default, bool) or not isinstance(default, int):
            raise ProblemError(f"{context}.default must be an integer")
        if not low_raw <= default <= high_raw:
            raise ProblemError(f"{context}.default must be within bounds")
        return Variable(name, kind, low_raw, high_raw, default=default)

    if kind is VariableKind.CHOICE:
        _unknown(table, {"kind", "values", "default"}, context)
        raw_values = table.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ProblemError(f"{context}.values must be a non-empty array")
        values = tuple(_scalar(value, f"{context}.values") for value in raw_values)
        if len(set(values)) != len(values):
            raise ProblemError(f"{context}.values must be unique")
        default = _scalar(table.get("default", values[0]), f"{context}.default")
        if default not in values:
            raise ProblemError(f"{context}.default must be one of values")
        return Variable(name, kind, values=values, default=default)

    _unknown(table, {"kind", "source", "factor", "offset"}, context)
    source = _text(table.get("source"), f"{context}.source")
    factor = _finite(table.get("factor", 1.0), f"{context}.factor")
    offset = _finite(table.get("offset", 0.0), f"{context}.offset")
    return Variable(name, kind, source=source, factor=factor, offset=offset)


def _check_links(variables: tuple[Variable, ...]) -> None:
    by_name = {variable.name: variable for variable in variables}
    for variable in variables:
        if variable.kind is not VariableKind.LINKED:
            continue
        seen = {variable.name}
        source_name = variable.source
        while source_name is not None:
            if source_name not in by_name:
                raise ProblemError(
                    f"linked variable {variable.name} references unknown {source_name}"
                )
            if source_name in seen:
                raise ProblemError(f"linked-variable cycle includes {source_name}")
            seen.add(source_name)
            source = by_name[source_name]
            if source.kind is VariableKind.CHOICE and not all(
                isinstance(value, int | float) and not isinstance(value, bool)
                for value in source.values
            ):
                raise ProblemError(f"linked variable {variable.name} requires a numeric source")
            source_name = source.source if source.kind is VariableKind.LINKED else None


def _objective(raw: Any, index: int) -> Objective:
    context = f"objectives[{index}]"
    table = _table(raw, context)
    _unknown(table, {"metric", "goal", "scale", "epsilon", "reference"}, context)
    return Objective(
        metric=_text(table.get("metric"), f"{context}.metric"),
        goal=_enum(Goal, table.get("goal"), f"{context}.goal"),
        scale=_positive(table.get("scale"), f"{context}.scale"),
        epsilon=_positive(table.get("epsilon", 0.02), f"{context}.epsilon"),
        reference=_finite(table.get("reference", 0.0), f"{context}.reference"),
    )


def _constraint(raw: Any, index: int) -> Constraint:
    context = f"constraints[{index}]"
    table = _table(raw, context)
    relation = _enum(Relation, table.get("relation"), f"{context}.relation")
    base = {"metric", "relation", "scale", "tolerance"}
    scale = _positive(table.get("scale"), f"{context}.scale")
    tolerance = _finite(table.get("tolerance", 0.0), f"{context}.tolerance")
    if tolerance < 0.0:
        raise ProblemError(f"{context}.tolerance must not be negative")
    metric = _text(table.get("metric"), f"{context}.metric")
    if relation in (Relation.GE, Relation.LE):
        _unknown(table, base | {"limit"}, context)
        return Constraint(
            metric,
            relation,
            scale,
            tolerance,
            limit=_finite(table.get("limit"), f"{context}.limit"),
        )
    _unknown(table, base | {"lower", "upper"}, context)
    lower = _finite(table.get("lower"), f"{context}.lower")
    upper = _finite(table.get("upper"), f"{context}.upper")
    if lower > upper:
        raise ProblemError(f"{context}.lower must not exceed upper")
    return Constraint(metric, relation, scale, tolerance, lower=lower, upper=upper)


def parse_problem(
    data: Mapping[str, Any], *, source_hash: str = "", source_path: str = ""
) -> Problem:
    """Validate parsed TOML and return an immutable problem."""
    _unknown(data, {"schema_version", "variables", "objectives", "constraints"}, "problem")
    if data.get("schema_version") != 1:
        raise ProblemError("problem.schema_version must be 1")
    raw_variables = _table(data.get("variables"), "problem.variables")
    if not raw_variables:
        raise ProblemError("problem.variables must not be empty")
    variables = tuple(
        _variable(_text(name, "variable name"), raw) for name, raw in raw_variables.items()
    )
    _check_links(variables)
    if not any(variable.free for variable in variables):
        raise ProblemError("problem must contain at least one free variable")

    raw_objectives = data.get("objectives")
    if not isinstance(raw_objectives, list) or len(raw_objectives) < 2:
        raise ProblemError("problem.objectives must contain at least two entries")
    objectives = tuple(_objective(raw, index) for index, raw in enumerate(raw_objectives))
    objective_names = [objective.metric for objective in objectives]
    if len(objective_names) != len(set(objective_names)):
        raise ProblemError("objective metrics must be unique")

    raw_constraints = data.get("constraints", [])
    if not isinstance(raw_constraints, list):
        raise ProblemError("problem.constraints must be an array")
    constraints = tuple(_constraint(raw, index) for index, raw in enumerate(raw_constraints))
    return Problem(variables, objectives, constraints, source_hash, source_path)


def load_problem(path: str | Path) -> Problem:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ProblemError(f"cannot read problem {source}: {error}") from error
    try:
        data = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ProblemError(f"invalid TOML problem {source}: {error}") from error
    return parse_problem(
        data,
        source_hash=hashlib.sha256(payload).hexdigest(),
        source_path=str(source),
    )
