from __future__ import annotations

import copy

import pytest

from biasweave.errors import ProblemError
from biasweave.model import Goal, Relation, VariableKind, VariableScale
from biasweave.problem import load_problem, parse_problem
from tests.helpers import problem_data


def test_parses_all_variable_kinds_and_semantics():
    problem = parse_problem(problem_data())
    assert [variable.kind for variable in problem.variables] == [
        VariableKind.REAL,
        VariableKind.INTEGER,
        VariableKind.CHOICE,
        VariableKind.LINKED,
    ]
    assert problem.free_variables == problem.variables[:3]
    assert problem.objectives[0].goal is Goal.MIN
    assert problem.objectives[1].goal is Goal.MAX
    assert problem.constraints[0].relation is Relation.GE
    assert problem.constraints[1].relation is Relation.BETWEEN


def test_load_problem_records_bytes_digest_and_path(tmp_path):
    path = tmp_path / "problem.toml"
    path.write_text(
        """schema_version = 1
[variables.x]
kind = "real"
low = 1.0
high = 10.0
scale = "log"
[[objectives]]
metric = "a"
goal = "min"
scale = 1.0
[[objectives]]
metric = "b"
goal = "max"
scale = 1.0
""",
        encoding="utf-8",
    )
    problem = load_problem(path)
    assert len(problem.source_hash) == 64
    assert problem.source_path == str(path)
    assert problem.variables[0].scale is VariableScale.LOG


@pytest.mark.parametrize("schema", [None, False, True, 0, 1.0, 2, "1"])
def test_rejects_unsupported_schema(schema):
    data = problem_data()
    data["schema_version"] = schema
    with pytest.raises(ProblemError, match="schema_version"):
        parse_problem(data)


def test_rejects_unknown_problem_and_variable_fields():
    data = problem_data()
    data["typo"] = 1
    with pytest.raises(ProblemError, match="unsupported fields: typo"):
        parse_problem(data)
    data = problem_data()
    data["variables"]["x"]["typo"] = 1
    with pytest.raises(ProblemError, match="variables.x has unsupported"):
        parse_problem(data)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"low": 1.0, "high": 1.0}, "low must be less"),
        ({"low": 0.0, "high": 1.0, "scale": "log"}, "positive"),
        ({"low": 0.0, "high": 1.0, "quantum": 0.0}, "greater than zero"),
        ({"low": 0.0, "high": 1.0, "default": 2.0}, "within"),
        ({"scale": "curved"}, "linear, log"),
        ({"low": float("inf")}, "finite"),
    ],
)
def test_rejects_invalid_real_variables(patch, message):
    data = problem_data()
    data["variables"]["x"].update(patch)
    with pytest.raises(ProblemError, match=message):
        parse_problem(data)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"low": 1.5}, "must be an integer"),
        ({"high": True}, "must be an integer"),
        ({"low": 5, "high": 3}, "must not exceed"),
        ({"default": 2.5}, "default must be an integer"),
        ({"default": 20}, "within bounds"),
    ],
)
def test_rejects_invalid_integer_variables(patch, message):
    data = problem_data()
    data["variables"]["n"].update(patch)
    with pytest.raises(ProblemError, match=message):
        parse_problem(data)


@pytest.mark.parametrize(
    ("values", "default", "message"),
    [
        ([], None, "non-empty"),
        (["a", "a"], None, "unique"),
        (["a"], "b", "one of"),
        ([float("nan")], None, "finite"),
    ],
)
def test_rejects_invalid_choice_variables(values, default, message):
    data = problem_data()
    table = data["variables"]["mode"]
    table["values"] = values
    if default is not None:
        table["default"] = default
    else:
        table.pop("default", None)
    with pytest.raises(ProblemError, match=message):
        parse_problem(data)


def test_rejects_unknown_link_cycle_and_non_numeric_source():
    data = problem_data()
    data["variables"]["twice_x"]["source"] = "absent"
    with pytest.raises(ProblemError, match="unknown absent"):
        parse_problem(data)
    data = problem_data()
    data["variables"]["loop"] = {"kind": "linked", "source": "twice_x"}
    data["variables"]["twice_x"]["source"] = "loop"
    with pytest.raises(ProblemError, match="cycle"):
        parse_problem(data)
    data = problem_data()
    data["variables"]["twice_x"]["source"] = "mode"
    with pytest.raises(ProblemError, match="numeric source"):
        parse_problem(data)


def test_rejects_empty_variables_and_too_few_or_duplicate_objectives():
    data = problem_data()
    data["variables"] = {}
    with pytest.raises(ProblemError, match="must not be empty"):
        parse_problem(data)
    data = problem_data()
    data["objectives"] = data["objectives"][:1]
    with pytest.raises(ProblemError, match="at least two"):
        parse_problem(data)
    data = problem_data()
    data["objectives"][1]["metric"] = "loss"
    with pytest.raises(ProblemError, match="must be unique"):
        parse_problem(data)


@pytest.mark.parametrize(
    ("constraint", "message"),
    [
        ({"metric": "x", "relation": "ge", "scale": 1.0}, "limit"),
        (
            {"metric": "x", "relation": "between", "scale": 1.0, "lower": 2, "upper": 1},
            "must not exceed",
        ),
        ({"metric": "x", "relation": "le", "scale": 0.0, "limit": 1}, "greater than zero"),
        (
            {"metric": "x", "relation": "le", "scale": 1.0, "limit": 1, "tolerance": -1},
            "must not be negative",
        ),
    ],
)
def test_rejects_invalid_constraints(constraint, message):
    data = problem_data()
    data["constraints"] = [constraint]
    with pytest.raises(ProblemError, match=message):
        parse_problem(data)


def test_parser_does_not_mutate_input():
    data = problem_data()
    original = copy.deepcopy(data)
    parse_problem(data)
    assert data == original


def test_huge_integer_is_reported_as_problem_error():
    data = problem_data()
    data["objectives"][0]["scale"] = 10**1000
    with pytest.raises(ProblemError, match="finite"):
        parse_problem(data)


def test_load_problem_reports_file_and_toml_errors(tmp_path):
    with pytest.raises(ProblemError, match="cannot read problem"):
        load_problem(tmp_path / "missing.toml")
    path = tmp_path / "bad.toml"
    path.write_text("[[", encoding="utf-8")
    with pytest.raises(ProblemError, match="invalid TOML"):
        load_problem(path)


def test_load_problem_rejects_resource_exhaustion_inputs(tmp_path):
    oversized = tmp_path / "oversized.toml"
    oversized.write_bytes(b"#" + b" " * 1_048_576)
    with pytest.raises(ProblemError, match="byte input limit"):
        load_problem(oversized)

    deep = tmp_path / "deep.toml"
    deep.write_text("value = " + "[" * 65 + "0" + "]" * 65, encoding="utf-8")
    with pytest.raises(ProblemError, match=r"complexity|invalid TOML"):
        load_problem(deep)

    many = tmp_path / "many.toml"
    many.write_text("value = [" + ",".join("0" for _ in range(10_001)) + "]", encoding="utf-8")
    with pytest.raises(ProblemError, match="complexity"):
        load_problem(many)
