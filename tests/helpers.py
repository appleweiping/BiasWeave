"""Small original fixtures shared by behavioral tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from biasweave.model import Scalar
from biasweave.problem import parse_problem


def problem_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "variables": {
            "x": {"kind": "real", "low": 0.01, "high": 1.0, "default": 0.5},
            "n": {"kind": "integer", "low": 1, "high": 4, "default": 2},
            "mode": {"kind": "choice", "values": ["fast", "quiet"], "default": "fast"},
            "twice_x": {"kind": "linked", "source": "x", "factor": 2.0},
        },
        "objectives": [
            {"metric": "loss", "goal": "min", "scale": 1.0, "epsilon": 0.05},
            {"metric": "score", "goal": "max", "scale": 2.0, "epsilon": 0.05},
        ],
        "constraints": [
            {"metric": "quality", "relation": "ge", "limit": 0.35, "scale": 0.5},
            {
                "metric": "window",
                "relation": "between",
                "lower": 0.0,
                "upper": 2.5,
                "scale": 1.0,
            },
        ],
    }


def make_problem():
    return parse_problem(problem_data(), source_hash="fixture-hash", source_path="fixture.toml")


def evaluator(point: Mapping[str, Scalar]) -> dict[str, float]:
    x = float(point["x"])
    n = float(point["n"])
    quiet = 1.0 if point["mode"] == "quiet" else 0.0
    return {
        "loss": (x - 0.25) ** 2 + 0.015 * n + 0.01 * quiet,
        "score": 2.0 * x + 0.08 * n - 0.04 * quiet,
        "quality": x + 0.12 * n + 0.1 * quiet,
        "window": float(point["twice_x"]),
    }


def failed_evaluator(_point: Mapping[str, Scalar]) -> dict[str, float]:
    raise RuntimeError("synthetic evaluator failure")
