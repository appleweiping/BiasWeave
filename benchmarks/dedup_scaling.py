"""Measure decoded-key bookkeeping over increasing evaluation counts."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from biasweave.benchmark import load_analog_benchmark
from biasweave.dominance import failed_trial
from biasweave.optimizers import create_optimizer

BASE = Path(__file__).parents[1]
CONTRACT = BASE / "benchmarks" / "manifest.json"


def run(budgets: tuple[int, ...], repetitions: int, seed: int) -> dict[str, object]:
    """Return descriptive timing plus exact uniqueness invariants."""

    if not budgets or any(
        isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 20_000
        for budget in budgets
    ):
        raise ValueError("budgets must be integers from 1 through 20000")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or not 1 <= repetitions <= 20
    ):
        raise ValueError("repetitions must be an integer from 1 through 20")
    if isinstance(seed, bool) or not isinstance(seed, int) or not -(1 << 63) <= seed < 1 << 63:
        raise ValueError("seed must be a signed 64-bit integer")

    benchmark = load_analog_benchmark(CONTRACT)
    rows: list[dict[str, object]] = []
    for budget in budgets:
        elapsed: list[float] = []
        keys: tuple[str, ...] = ()
        for _repeat in range(repetitions):
            start = perf_counter()
            optimizer = create_optimizer(benchmark.problem, "random", seed=seed)
            collected: list[str] = []
            while len(collected) < budget:
                points = optimizer.ask(min(64, budget - len(collected)))
                if not points:
                    raise RuntimeError("deduplication scaling proposal stream stalled")
                first_trial = len(collected)
                optimizer.tell(
                    tuple(
                        failed_trial(first_trial + offset, point, "dedup-scaling")
                        for offset, point in enumerate(points)
                    )
                )
                collected.extend(point.key for point in points)
            elapsed.append(perf_counter() - start)
            keys = tuple(collected)
        if len(keys) != budget or len(set(keys)) != budget:
            raise RuntimeError("deduplication scaling invariant failed")
        median = statistics.median(elapsed)
        rows.append(
            {
                "evaluations": budget,
                "unique_points": len(set(keys)),
                "trajectory_sha256": sha256("\n".join(keys).encode("ascii")).hexdigest(),
                "median_seconds": round(median, 9),
                "minimum_seconds": round(min(elapsed), 9),
                "microseconds_per_evaluation": round(1_000_000 * median / budget, 3),
            }
        )
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    return {
        "schema_version": 1,
        "benchmark": "biasweave-dedup-scaling-v1",
        "distribution_version": version("biasweave"),
        "contract_sha256": benchmark.contract_sha256,
        "harness_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": environment,
        "seed": seed,
        "repetitions": repetitions,
        "results": rows,
        "timing_policy": "Informational only; no timing value is an acceptance threshold.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", default="1000,5000,10000")
    parser.add_argument("--repetitions", default=3, type=int)
    parser.add_argument("--seed", default=17, type=int)
    options = parser.parse_args()
    try:
        parsed = tuple(int(item) for item in options.budgets.split(","))
        report = run(parsed, options.repetitions, options.seed)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
