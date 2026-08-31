"""Same-contract optimizer scaling measurements without speed claims."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import biasweave
from biasweave.benchmark import compare_with_random, load_analog_benchmark

BASE = Path(__file__).parents[1]
CONTRACT = BASE / "benchmarks" / "manifest.json"


def _package_tree_sha256() -> str:
    if biasweave.__file__ is None:
        raise RuntimeError("cannot locate imported biasweave package")
    root = Path(biasweave.__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _host() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def run(budgets: tuple[int, ...], repetitions: int, seed: int) -> dict[str, object]:
    """Benchmark complete deterministic comparisons over increasing budgets."""

    if not budgets or any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2_000
        for value in budgets
    ):
        raise ValueError("budgets must be integers from 1 through 2000")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or not 1 <= repetitions <= 50
    ):
        raise ValueError("repetitions must be an integer from 1 through 50")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not -(1 << 63) <= seed <= (1 << 63) - 1
    ):
        raise ValueError("seed must be a signed 64-bit integer")
    benchmark = load_analog_benchmark(CONTRACT)
    workload = sha256(CONTRACT.read_bytes())
    workload.update(seed.to_bytes(8, "big", signed=True))
    rows: list[dict[str, object]] = []
    for budget in budgets:
        workload.update(budget.to_bytes(4, "big"))
        elapsed: list[float] = []
        comparison = None
        for _ in range(repetitions):
            start = perf_counter()
            comparison = compare_with_random(benchmark, budget=budget, seed=seed)
            elapsed.append(perf_counter() - start)
        if comparison is None:
            raise RuntimeError("benchmark comparison was not produced")
        algorithms = comparison["algorithms"]
        invariants = {
            name: {
                "evaluations": summary["evaluations"],
                "unique_points": summary["unique_points"],
                "feasible": summary["feasible"],
                "frontier": summary["frontier"],
            }
            for name, summary in algorithms.items()
        }
        if any(value["evaluations"] != budget for value in invariants.values()):
            raise RuntimeError("same-budget comparison invariant failed")
        median = statistics.median(elapsed)
        rows.append(
            {
                "budget_per_algorithm": budget,
                "comparison_sha256": comparison["comparison_sha256"],
                "invariants": invariants,
                "timing": {
                    "median_seconds": round(median, 9),
                    "minimum_seconds": round(min(elapsed), 9),
                    "evaluations_per_second": round((2 * budget) / median, 3),
                },
            }
        )
    host = _host()
    host_json = json.dumps(host, separators=(",", ":"), sort_keys=True)
    return {
        "schema_version": 1,
        "benchmark": "biasweave-same-budget-scaling-v1",
        "distribution_version": version("biasweave"),
        "package_tree_sha256": _package_tree_sha256(),
        "harness_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": host,
        "environment_sha256": sha256(host_json.encode()).hexdigest(),
        "workload_sha256": workload.hexdigest(),
        "seed": seed,
        "repetitions": repetitions,
        "results": rows,
        "timing_policy": "Informational only; no timing value is an acceptance threshold.",
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--budgets", default="16,64,256")
    cli.add_argument("--repetitions", default=3, type=int)
    cli.add_argument("--seed", default=17, type=int)
    options = cli.parse_args()
    try:
        parsed_budgets = tuple(int(item) for item in options.budgets.split(","))
        report = run(parsed_budgets, options.repetitions, options.seed)
    except ValueError as error:
        cli.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
