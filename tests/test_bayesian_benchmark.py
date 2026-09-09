from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _module():
    path = Path("benchmarks/bayesian_acquisition.py")
    spec = importlib.util.spec_from_file_location("biasweave_bayesian_acquisition", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Bayesian acquisition harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bayesian_acquisition_harness_records_each_equal_call_seed() -> None:
    report = _module().run((0, 1, 2, 3), 16)
    assert report["seeds"] == [0, 1, 2, 3]
    assert report["configuration"]["budget_per_strategy"] == 16
    assert len(report["problem_sha256"]) == 64
    assert len(report["results"]) == 4
    for seed, row in enumerate(report["results"]):
        assert row["seed"] == seed
        assert row["budget_per_strategy"] == 16
        assert set(row["algorithms"]) == {"bayes", "random"}
        assert row["algorithms"]["bayes"]["strategy_parameters"]["candidate_pool_size"] == 64
        assert row["algorithms"]["bayes"]["strategy_parameters"]["scalarization_seed"] == seed
        assert row["algorithms"]["random"]["strategy_parameters"] == {}
        for summary in row["algorithms"].values():
            assert summary["evaluations"] == summary["unique_points"] == 16
            assert summary["hypervolume"] > 0.0
            assert summary["stop_reason"] == "budget"
            assert len(summary["trajectory_sha256"]) == 64
    assert report["aggregate"]["difference_sum"] > 0.1
    unsigned = dict(report)
    digest = unsigned.pop("report_sha256")
    canonical = json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    assert digest == hashlib.sha256(canonical.encode("ascii")).hexdigest()
    assert "not a universal ranking" in report["interpretation"]


def test_recorded_acquisition_report_is_self_and_source_bound() -> None:
    evidence = json.loads(
        Path("benchmarks/bayesian-acquisition-v1.json").read_text(encoding="utf-8")
    )
    unsigned = dict(evidence)
    digest = unsigned.pop("report_sha256")
    canonical = json.dumps(unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    assert digest == hashlib.sha256(canonical.encode("ascii")).hexdigest()
    assert (
        evidence["harness_sha256"]
        == hashlib.sha256(Path("benchmarks/bayesian_acquisition.py").read_bytes()).hexdigest()
    )
    assert evidence["package_tree_sha256"] == _module()._package_tree_sha256()
    assert [row["seed"] for row in evidence["results"]] == evidence["seeds"]
    assert all(row["budget_per_strategy"] == 16 for row in evidence["results"])


def test_bayesian_acquisition_harness_is_an_executable_json_command() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/bayesian_acquisition.py",
            "--seeds",
            "0",
            "--budget",
            "9",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    assert report["seeds"] == [0]
    assert report["results"][0]["budget_per_strategy"] == 9
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("seeds", "budget", "message"),
    [
        ((), 16, "seeds"),
        ((0, 0), 16, "seeds"),
        ((True,), 16, "seeds"),
        (([],), 16, "seeds"),
        (({},), 16, "seeds"),
        ((10**5000,), 16, "seeds"),
        ((0,), 8, "budget"),
        ((0,), 129, "budget"),
    ],
)
def test_bayesian_acquisition_harness_bounds_inputs(seeds, budget, message) -> None:
    with pytest.raises(ValueError, match=message):
        _module().run(seeds, budget)
