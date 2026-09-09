"""Auditable equal-call acquisition check for the bounded Bayesian policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

import biasweave
from biasweave.engine import problem_fingerprint
from biasweave.model import OptimizationResult, RunConfig, Scalar
from biasweave.optimizers.catalog import create_optimizer, optimizer_parameters
from biasweave.problem import parse_problem
from biasweave.quality import hypervolume
from biasweave.strategy import optimize_strategy

_MIN_SEED = -(2**63)
_MAX_SEED = 2**63 - 1
_REFERENCE = (2.0, 2.0)


def _package_tree_sha256() -> str:
    if biasweave.__file__ is None:
        raise RuntimeError("cannot locate imported biasweave package")
    root = Path(biasweave.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _problem():
    return parse_problem(
        {
            "schema_version": 1,
            "variables": {
                "x": {"kind": "real", "low": 0.0, "high": 1.0, "default": 0.5},
                "nuisance": {
                    "kind": "real",
                    "low": 0.0,
                    "high": 1.0,
                    "default": 0.7,
                },
            },
            "objectives": [
                {"metric": "left", "goal": "min", "scale": 1.0},
                {"metric": "right", "goal": "min", "scale": 1.0},
            ],
            "constraints": [],
        }
    )


def _evaluate(point: Mapping[str, Scalar]) -> dict[str, float]:
    x = float(point["x"])
    nuisance = float(point["nuisance"])
    return {
        "left": (x - 0.2) ** 2 + nuisance**2,
        "right": (x - 0.8) ** 2 + nuisance**2,
    }


def _trajectory_sha256(keys: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(keys).encode("ascii")).hexdigest()


def _summary(result: OptimizationResult, strategy_parameters: dict[str, Any]) -> dict[str, object]:
    keys = tuple(trial.point.key for trial in result.trials)
    return {
        "evaluations": len(result.trials),
        "unique_points": len(set(keys)),
        "feasible": sum(trial.feasible for trial in result.trials),
        "frontier": len(result.frontier),
        "hypervolume": hypervolume(
            (trial.objective_vector for trial in result.frontier), _REFERENCE
        ),
        "stop_reason": result.stop_reason,
        "strategy_parameters": strategy_parameters,
        "trajectory_sha256": _trajectory_sha256(keys),
    }


def run(seeds: tuple[int, ...] = (0, 1, 2, 3), budget: int = 16) -> dict[str, object]:
    """Run fixed smooth trade-off searches and retain every per-seed result."""

    if (
        not isinstance(seeds, tuple)
        or not 1 <= len(seeds) <= 32
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not _MIN_SEED <= seed <= _MAX_SEED
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be 1 through 32 unique signed 64-bit integers")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 9 <= budget <= 128:
        raise ValueError("budget must be an integer from 9 through 128")

    problem = _problem()
    configuration = {
        "budget_per_strategy": budget,
        "batch_size": min(8, budget),
        "workers": 1,
        "evaluator_id": "analytic:bayesian-acquisition-v1",
        "reference_point": list(_REFERENCE),
    }
    rows: list[dict[str, object]] = []
    bayesian_volumes: list[float] = []
    random_volumes: list[float] = []
    for seed in seeds:
        algorithms: dict[str, dict[str, object]] = {}
        volumes: dict[str, float] = {}
        for strategy in ("bayes", "random"):
            parameters = optimizer_parameters(
                strategy, create_optimizer(problem, strategy, seed=seed)
            )
            result = optimize_strategy(
                problem,
                _evaluate,
                evaluator_id="analytic:bayesian-acquisition-v1",
                config=RunConfig(
                    budget,
                    seed=seed,
                    workers=1,
                    batch_size=min(8, budget),
                ),
                strategy=strategy,
            )
            summary = _summary(result, parameters)
            if summary["evaluations"] != budget or summary["unique_points"] != budget:
                raise RuntimeError("equal-call Bayesian acquisition invariant failed")
            algorithms[strategy] = summary
            volume = summary["hypervolume"]
            if not isinstance(volume, float):
                raise RuntimeError("acquisition hypervolume is invalid")
            volumes[strategy] = volume
        bayes_volume = volumes["bayes"]
        random_volume = volumes["random"]
        bayesian_volumes.append(bayes_volume)
        random_volumes.append(random_volume)
        rows.append(
            {
                "seed": seed,
                "budget_per_strategy": budget,
                "algorithms": algorithms,
                "hypervolume_difference_bayes_minus_random": bayes_volume - random_volume,
            }
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "biasweave-bayesian-acquisition-v1",
        "distribution_version": version("biasweave"),
        "package_tree_sha256": _package_tree_sha256(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "problem_sha256": problem_fingerprint(problem),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "seeds": list(seeds),
        "configuration": configuration,
        "results": rows,
        "aggregate": {
            "bayesian_hypervolume_sum": math.fsum(bayesian_volumes),
            "random_hypervolume_sum": math.fsum(random_volumes),
            "difference_sum": math.fsum(bayesian_volumes) - math.fsum(random_volumes),
        },
        "interpretation": (
            "Deterministic analytic acquisition regression at equal evaluator calls; "
            "not a universal ranking, circuit result, or state-of-the-art claim."
        ),
    }
    canonical = json.dumps(report, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    report["report_sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--budget", default=16, type=int)
    options = parser.parse_args()
    try:
        parsed_seeds = tuple(int(value) for value in options.seeds.split(","))
        output = run(parsed_seeds, options.budget)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(output, indent=2, sort_keys=True))
