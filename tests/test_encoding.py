from __future__ import annotations

import math

import pytest

from biasweave.encoding import decode, default_coordinates, encode, make_point, point_key
from biasweave.errors import ProblemError
from biasweave.model import VariableScale
from biasweave.problem import parse_problem
from tests.helpers import make_problem, problem_data


def test_defaults_round_trip_and_link_resolves():
    problem = make_problem()
    coordinates = default_coordinates(problem)
    values = decode(problem, coordinates)
    assert values["x"] == pytest.approx(0.5)
    assert values["n"] == 2
    assert values["mode"] == "fast"
    assert values["twice_x"] == pytest.approx(1.0)
    assert encode(problem, values) == pytest.approx(coordinates)


def test_decode_clamps_real_integer_and_choice_coordinates():
    problem = make_problem()
    low = decode(problem, [-4.0, -1.0, -0.2])
    high = decode(problem, [4.0, 2.0, 1.4])
    assert low == {"x": 0.01, "n": 1, "mode": "fast", "twice_x": 0.02}
    assert high == {"x": 1.0, "n": 4, "mode": "quiet", "twice_x": 2.0}


def test_decode_rejects_wrong_length_and_non_finite_coordinate():
    problem = make_problem()
    with pytest.raises(ProblemError, match="expected 3 coordinates"):
        decode(problem, [0.5])
    with pytest.raises(ProblemError, match="must be finite"):
        decode(problem, [math.nan, 0.5, 0.5])


def test_logarithmic_real_decoding_and_encoding():
    data = problem_data()
    data["variables"]["x"].update({"low": 1e-3, "high": 1e3, "scale": "log", "default": 1.0})
    problem = parse_problem(data)
    variable = problem.variables[0]
    assert variable.scale is VariableScale.LOG
    assert decode(problem, [0.5, 0.5, 0.5])["x"] == pytest.approx(1.0)
    assert encode(problem, {"x": 1.0, "n": 2, "mode": "fast"})[0] == pytest.approx(0.5)


def test_real_quantum_snaps_relative_to_low_and_stays_bounded():
    data = problem_data()
    data["variables"]["x"].update({"low": 0.1, "high": 1.0, "quantum": 0.2, "default": 0.5})
    problem = parse_problem(data)
    assert decode(problem, [0.51, 0.5, 0.5])["x"] == pytest.approx(0.5)
    assert decode(problem, [1.0, 0.5, 0.5])["x"] == pytest.approx(1.0)


def test_encode_rejects_missing_out_of_range_and_bad_choice():
    problem = make_problem()
    with pytest.raises(ProblemError, match="missing free-variable"):
        encode(problem, {"x": 0.5})
    with pytest.raises(ProblemError, match="outside"):
        encode(problem, {"x": 2.0, "n": 2, "mode": "fast"})
    with pytest.raises(ProblemError, match="not a choice"):
        encode(problem, {"x": 0.5, "n": 2, "mode": "unknown"})
    with pytest.raises(ProblemError, match="received a string"):
        encode(problem, {"x": "bad", "n": 2, "mode": "fast"})
    with pytest.raises(ProblemError, match="representable"):
        encode(problem, {"x": 10**1000, "n": 2, "mode": "fast"})


def test_nested_linked_variables_resolve_in_declaration_order():
    data = problem_data()
    data["variables"]["four_x"] = {"kind": "linked", "source": "twice_x", "factor": 2.0}
    problem = parse_problem(data)
    values = decode(problem, default_coordinates(problem))
    assert values["twice_x"] == pytest.approx(1.0)
    assert values["four_x"] == pytest.approx(2.0)


def test_point_key_is_canonical_and_make_point_captures_values():
    first = point_key({"b": 2, "a": 1})
    second = point_key({"a": 1, "b": 2})
    assert first == second
    assert len(first) == 64
    point = make_point(make_problem(), [0.5, 0.5, 0.5])
    assert point.key == point_key(point.values)
    assert point.coordinates == (0.5, 0.5, 0.5)


def test_constant_integer_range_decodes_without_division():
    data = problem_data()
    data["variables"]["n"].update({"low": 3, "high": 3, "default": 3})
    problem = parse_problem(data)
    values = decode(problem, default_coordinates(problem))
    assert values["n"] == 3
    assert encode(problem, values)[1] == pytest.approx(0.5)


@pytest.mark.parametrize("low", [2**53, -(2**60)])
def test_large_integer_three_value_domain_round_trips_exactly(low: int) -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {
                "x": {"kind": "integer", "low": low, "high": low + 2, "default": low + 1}
            },
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )
    for value in range(low, low + 3):
        coordinate = encode(problem, {"x": value})
        assert decode(problem, coordinate)["x"] == value


def test_wide_integer_encoding_refuses_silent_float_coordinate_alias() -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": {"kind": "integer", "low": 0, "high": 2**60}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )

    defaults = default_coordinates(problem)
    assert defaults == (0.5,)
    assert decode(problem, defaults)["x"] == 2**59
    low_endpoint = encode(problem, {"x": 0})
    assert low_endpoint == (0.0,)
    assert decode(problem, low_endpoint)["x"] == 0
    with pytest.raises(ProblemError, match="cannot be represented exactly"):
        encode(problem, {"x": 2**60 - 1})
    endpoint = encode(problem, {"x": 2**60})
    assert endpoint == (1.0,)
    assert decode(problem, endpoint)["x"] == 2**60


@pytest.mark.parametrize(
    ("variable", "expected_default"),
    [
        ({"kind": "real", "low": -1e308, "high": 1e308}, 0.0),
        ({"kind": "real", "low": 1e200, "high": 1e300, "scale": "log"}, 1e250),
    ],
)
def test_extreme_finite_real_defaults_round_trip_without_intermediate_overflow(
    variable: dict[str, object], expected_default: float
) -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": variable},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )

    coordinates = default_coordinates(problem)
    assert coordinates == pytest.approx((0.5,))
    assert decode(problem, coordinates)["x"] == pytest.approx(expected_default)


def test_extreme_finite_linear_real_endpoints_and_center_round_trip() -> None:
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": {"kind": "real", "low": -1e308, "high": 1e308}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )

    for value, coordinate in ((-1e308, 0.0), (0.0, 0.5), (1e308, 1.0)):
        encoded = encode(problem, {"x": value})
        assert encoded == pytest.approx((coordinate,))
        assert decode(problem, encoded)["x"] == value


@pytest.mark.parametrize("low", [1e-200, 1e200, 1e308])
def test_ulp_narrow_log_real_endpoints_defaults_and_interior_round_trip(low: float) -> None:
    values = [low]
    for _ in range(8):
        values.append(math.nextafter(values[-1], math.inf))
    high = values[-1]
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {"x": {"kind": "real", "low": low, "high": high, "scale": "log"}},
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )

    default = problem.free_variables[0].default
    assert isinstance(default, float)
    assert low <= default <= high
    assert decode(problem, default_coordinates(problem))["x"] == default
    for value in (values[0], values[3], values[5], values[8]):
        coordinate = encode(problem, {"x": value})
        assert 0.0 <= coordinate[0] <= 1.0
        assert decode(problem, coordinate)["x"] == value


@pytest.mark.parametrize("low", [1e-200, 1e200, 1e308])
def test_ulp_narrow_log_explicit_lower_endpoint_encodes_to_zero(low: float) -> None:
    high = math.nextafter(math.nextafter(low, math.inf), math.inf)
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": {
                "x": {
                    "kind": "real",
                    "low": low,
                    "high": high,
                    "scale": "log",
                    "default": low,
                }
            },
            "objectives": [
                {"metric": "a", "goal": "min", "scale": 1.0},
                {"metric": "b", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )

    assert default_coordinates(problem) == (0.0,)
    assert decode(problem, (0.0,))["x"] == low
