from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("benchmarks/dedup_scaling.py")
    spec = importlib.util.spec_from_file_location("biasweave_dedup_scaling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load deduplication scaling harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deduplication_scaling_harness_records_unique_trajectories() -> None:
    report = _module().run((32, 128), 1, 17)
    assert [row["evaluations"] for row in report["results"]] == [32, 128]
    assert all(row["unique_points"] == row["evaluations"] for row in report["results"])
    assert all(len(row["trajectory_sha256"]) == 64 for row in report["results"])


@pytest.mark.parametrize("budgets", [(), (0,), (20_001,), (True,)])
def test_deduplication_scaling_harness_bounds_work(budgets) -> None:
    with pytest.raises(ValueError, match="budgets"):
        _module().run(budgets, 1, 0)
