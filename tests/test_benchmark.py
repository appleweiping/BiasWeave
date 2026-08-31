from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import biasweave
from biasweave.benchmark import compare_with_random, load_analog_benchmark, sizing_decision
from biasweave.cli import entrypoint, main
from biasweave.errors import ProblemError

CONTRACT = Path("benchmarks/manifest.json")
PRODUCER_CONTRACT_SHA256 = "bd642866abe0b2ee7787c53d31c7b9082cec263dbe827a73936301f18c5e6422"
PRODUCER_SCHEMA_SHA256 = "9d5bbc50f9306ec33873cdeb3d4c0afa953ce3a03731c340dd66517a95b30d2b"


def _imported_tree_sha256() -> str:
    assert biasweave.__file__ is not None
    root = Path(biasweave.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _resign(data):
    body = dict(data)
    body.pop("contract_sha256", None)
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    data["contract_sha256"] = hashlib.sha256(payload.encode("ascii")).hexdigest()


def _resign_comparison(data):
    body = dict(data)
    body.pop("comparison_sha256", None)
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    data["comparison_sha256"] = hashlib.sha256(payload.encode("ascii")).hexdigest()


def test_contract_loads_and_evaluator_is_finite() -> None:
    benchmark = load_analog_benchmark(CONTRACT)
    metrics = benchmark.evaluate(
        {
            "topology_id": next(iter(benchmark.candidates)),
            "width_um": 4.0,
            "length_um": 0.5,
            "bias_ua": 100.0,
            "compensation_pf": 2.0,
        }
    )
    assert set(metrics) == {"gain_db", "phase_margin_deg", "power_mw", "area_um2", "bandwidth_mhz"}
    assert all(value > 0 for value in metrics.values())


def test_consumer_pins_byte_exact_topology_lantern_artifacts() -> None:
    schema = Path("docs/schemas/analog-sizing-benchmark-1.schema.json")
    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == PRODUCER_CONTRACT_SHA256
    assert hashlib.sha256(schema.read_bytes()).hexdigest() == PRODUCER_SCHEMA_SHA256


def test_same_budget_comparison_is_deterministic() -> None:
    benchmark = load_analog_benchmark(CONTRACT)
    first = compare_with_random(benchmark, budget=24, seed=19)
    second = compare_with_random(benchmark, budget=24, seed=19)
    assert first == second
    for summary in first["algorithms"].values():
        assert summary["evaluations"] == 24
        assert summary["unique_points"] == 24
    assert "state-of-the-art" in first["interpretation"]
    representative = first["algorithms"]["biasweave"]["representative"]
    assert representative["selection_policy"] == "normalized-l1-to-observed-ideal-v1"
    assert representative["values"]["topology_id"] in benchmark.candidates
    assert benchmark.evaluate(representative["values"]) == representative["metrics"]


def test_sizing_object_order_cannot_change_the_seeded_search(tmp_path: Path) -> None:
    original = json.loads(CONTRACT.read_text(encoding="utf-8"))
    original["sizing"] = dict(reversed(tuple(original["sizing"].items())))
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(original), encoding="utf-8")
    canonical = load_analog_benchmark(CONTRACT)
    equivalent = load_analog_benchmark(reordered)
    assert canonical.contract_sha256 == equivalent.contract_sha256
    assert compare_with_random(canonical, budget=12, seed=17) == compare_with_random(
        equivalent, budget=12, seed=17
    )


def test_sky130_reference_contract_exports_a_content_bound_point() -> None:
    benchmark = load_analog_benchmark("benchmarks/sky130-common-source.json")
    comparison = compare_with_random(benchmark, budget=64, seed=17)
    assert comparison["contract_sha256"] == benchmark.contract_sha256
    selected = comparison["algorithms"]["biasweave"]["representative"]
    assert selected is not None
    assert selected["values"]["topology_id"] == "TL-00052f7b8c5e"
    assert (
        selected["point_key"] == "5a307408195ba83bca3db375f4050c27eee1b50bff1ee00313f52b77dacce8ee"
    )
    decision = sizing_decision(benchmark, comparison)
    assert decision["source"]["topology_signature"] == (
        "086ce3f4158fa05ea8fddf2e7552af1d9774f95faffbe4dee11fae262d8857c8"
    )
    assert decision["variables"]["width_um"] == pytest.approx(4.4721359549995805)
    assert len(decision["decision_sha256"]) == 64
    assert decision == json.loads(
        Path("benchmarks/sky130-sizing-decision.json").read_text(encoding="utf-8")
    )


def test_public_schemas_accept_checked_in_interchange_documents() -> None:
    benchmark_schema = json.loads(
        Path("docs/schemas/analog-sizing-benchmark-1.schema.json").read_text(encoding="utf-8")
    )
    decision_schema = json.loads(
        Path("docs/schemas/sizing-decision-1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(benchmark_schema)
    Draft202012Validator.check_schema(decision_schema)
    benchmark_validator = Draft202012Validator(benchmark_schema)
    for name in ("manifest.json", "sky130-common-source.json"):
        benchmark_validator.validate(
            json.loads(Path("benchmarks", name).read_text(encoding="utf-8"))
        )
    Draft202012Validator(decision_schema).validate(
        json.loads(Path("benchmarks/sky130-sizing-decision.json").read_text(encoding="utf-8"))
    )


def test_scaling_benchmark_smoke_preserves_same_budget_invariants() -> None:
    completed = subprocess.run(  # nosec B603
        [
            sys.executable,
            "benchmarks/scaling.py",
            "--budgets",
            "3,9",
            "--repetitions",
            "1",
            "--seed",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == 1
    assert len(payload["environment_sha256"]) == 64
    assert payload["distribution_version"] == biasweave.__version__
    assert payload["package_tree_sha256"] == _imported_tree_sha256()
    assert (
        payload["harness_sha256"]
        == hashlib.sha256(Path("benchmarks/scaling.py").read_bytes()).hexdigest()
    )
    for row in payload["results"]:
        expected = row["budget_per_algorithm"]
        assert all(item["evaluations"] == expected for item in row["invariants"].values())


def test_scaling_benchmark_rejects_non_integer_limits_and_bounds_seeds() -> None:
    scaling_run = runpy.run_path("benchmarks/scaling.py")["run"]
    with pytest.raises(ValueError, match="budgets"):
        scaling_run((1.5,), 1, 0)
    with pytest.raises(ValueError, match="repetitions"):
        scaling_run((1,), 1.5, 0)
    minimum_seed = -(1 << 63)
    report = scaling_run((1,), 1, minimum_seed)
    assert report["seed"] == minimum_seed
    assert len(report["workload_sha256"]) == 64
    with pytest.raises(ValueError, match="signed 64-bit"):
        scaling_run((1,), 1, 1 << 63)


def test_scaling_cli_rejects_non_integer_budget_without_traceback() -> None:
    completed = subprocess.run(  # nosec B603
        [sys.executable, "benchmarks/scaling.py", "--budgets", "1.5"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "budgets" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_sizing_decision_rejects_a_comparison_from_another_contract() -> None:
    benchmark = load_analog_benchmark(CONTRACT)
    comparison = compare_with_random(benchmark, budget=8, seed=3)
    comparison["contract_sha256"] = "0" * 64
    _resign_comparison(comparison)
    with pytest.raises(ProblemError, match="does not belong"):
        sizing_decision(benchmark, comparison)


def test_sizing_decision_replays_the_claimed_representative_policy() -> None:
    benchmark = load_analog_benchmark(CONTRACT)
    comparison = compare_with_random(benchmark, budget=8, seed=3)
    comparison["algorithms"]["biasweave"]["representative"]["point_key"] = "0" * 64
    _resign_comparison(comparison)
    with pytest.raises(ProblemError, match="replay"):
        sizing_decision(benchmark, comparison)


def test_sizing_decision_rejects_json_types_that_compare_equal_in_python() -> None:
    benchmark = load_analog_benchmark("benchmarks/sky130-common-source.json")
    comparison = compare_with_random(benchmark, budget=64, seed=17)
    representative = comparison["algorithms"]["biasweave"]["representative"]
    assert representative["values"]["compensation_pf"] == 1.0
    representative["values"]["compensation_pf"] = 1
    _resign_comparison(comparison)
    with pytest.raises(ProblemError, match="replay"):
        sizing_decision(benchmark, comparison)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(extra=True),
        lambda data: data.update(version=True),
        lambda data: data.update(version=1.0),
        lambda data: data["source"].update(candidate_count=99),
    ],
)
def test_contract_rejects_unknown_wrong_type_and_inconsistent_data(tmp_path, mutation) -> None:
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutation(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProblemError):
        load_analog_benchmark(path)


def test_contract_rejects_digest_tampering(tmp_path) -> None:
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data["constraints"]["gain_db_min"] = 99.0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProblemError, match="digest"):
        load_analog_benchmark(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["source"].update(candidate_count=99),
        lambda data: data["source"].update(supply_voltage=True),
        lambda data: data["candidates"][0].update(device_count=True),
        lambda data: data["candidates"][1].update(
            candidate_id=data["candidates"][0]["candidate_id"]
        ),
        lambda data: data.update(expected_metrics=[]),
        lambda data: data.update(disclaimer=""),
        lambda data: data["candidates"][0].update(candidate_id="TL-NOT-LOWERCASE"),
        lambda data: data["provenance"].update(license=""),
        lambda data: data["source"].update(supply_voltage=0.0),
        lambda data: data["source"].update(spec_sha256="z" * 64),
        lambda data: data["candidates"][0].update(signature="z" * 64),
        lambda data: data["sizing"]["width_um"].update(low=0.0),
        lambda data: data["sizing"]["width_um"].update(high=0.1),
        lambda data: data["sizing"]["width_um"].update(scale="linear"),
        lambda data: data["constraints"].update(gain_db_min=0.0),
        lambda data: data["constraints"].update(phase_margin_deg_min=181.0),
    ],
)
def test_resigned_but_semantically_invalid_contract_is_rejected(tmp_path, mutation) -> None:
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mutation(data)
    _resign(data)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProblemError):
        load_analog_benchmark(path)


def test_evaluator_rejects_unknown_topology() -> None:
    benchmark = load_analog_benchmark(CONTRACT)
    with pytest.raises(ValueError, match="topology_id"):
        benchmark.evaluate(
            {
                "topology_id": "unknown",
                "width_um": 4.0,
                "length_um": 0.5,
                "bias_ua": 100.0,
                "compensation_pf": 2.0,
            }
        )


@pytest.mark.parametrize(
    "point",
    [
        {},
        {
            "topology_id": "TL-2a033a1ce04c",
            "width_um": True,
            "length_um": 0.5,
            "bias_ua": 100.0,
            "compensation_pf": 2.0,
        },
        {
            "topology_id": "TL-2a033a1ce04c",
            "width_um": float("nan"),
            "length_um": 0.5,
            "bias_ua": 100.0,
            "compensation_pf": 2.0,
        },
        {
            "topology_id": "TL-2a033a1ce04c",
            "width_um": 400.0,
            "length_um": 0.5,
            "bias_ua": 100.0,
            "compensation_pf": 2.0,
        },
    ],
)
def test_evaluator_rejects_malformed_direct_api_points(point) -> None:
    with pytest.raises(ValueError):
        load_analog_benchmark(CONTRACT).evaluate(point)


def test_contract_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ProblemError, match="duplicate key"):
        load_analog_benchmark(path)


def test_contract_rejects_deep_and_oversized_json(tmp_path: Path) -> None:
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 1_500 + "0" + "]" * 1_500, encoding="utf-8")
    with pytest.raises(ProblemError, match=r"cannot load|complexity limits"):
        load_analog_benchmark(deep)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * 1_048_577)
    with pytest.raises(ProblemError, match="byte input limit"):
        load_analog_benchmark(oversized)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1" + "0" * 400])
def test_contract_rejects_nonfinite_or_unrepresentable_numbers(tmp_path, token) -> None:
    text = CONTRACT.read_text(encoding="utf-8").replace(
        '"supply_voltage": 1.8', f'"supply_voltage": {token}'
    )
    path = tmp_path / "number.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProblemError):
        load_analog_benchmark(path)


def test_contract_rejects_resigned_integer_too_large_for_float(tmp_path) -> None:
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data["source"]["supply_voltage"] = 10**400
    _resign(data)
    path = tmp_path / "huge.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProblemError, match="numeric range"):
        load_analog_benchmark(path)


def test_candidate_mapping_is_immutable() -> None:
    benchmark = load_analog_benchmark(CONTRACT)
    with pytest.raises(TypeError):
        benchmark.candidates["new"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        next(iter(benchmark.candidates.values()))["device_count"] = 99  # type: ignore[index]


@pytest.mark.parametrize(
    ("budget", "seed"),
    [(True, 1), (0, 1), (1.5, 1), (1, True), (1, 1 << 63)],
)
def test_comparison_rejects_invalid_parameters(budget, seed) -> None:
    benchmark = load_analog_benchmark(CONTRACT)
    with pytest.raises(ProblemError):
        compare_with_random(benchmark, budget=budget, seed=seed)


def test_benchmark_cli_writes_comparison(tmp_path) -> None:
    output = tmp_path / "comparison.json"
    decision = tmp_path / "decision.json"
    assert (
        main(
            [
                "benchmark",
                "--contract",
                str(CONTRACT),
                "--budget",
                "12",
                "--seed",
                "7",
                "--output",
                str(output),
                "--decision-output",
                str(decision),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["budget"] == 12
    assert json.loads(decision.read_text())["schema"] == "org.biasweave.sizing-decision"


def test_benchmark_cli_rejects_colliding_outputs_without_writing(tmp_path) -> None:
    destination = tmp_path / "result.json"
    assert (
        main(
            [
                "benchmark",
                "--contract",
                str(CONTRACT),
                "--budget",
                "2",
                "--output",
                str(destination),
                "--decision-output",
                str(destination),
            ]
        )
        == 2
    )
    assert not destination.exists()


def test_benchmark_cli_stages_all_outputs_before_writing(tmp_path) -> None:
    comparison = tmp_path / "comparison.json"
    missing_decision = tmp_path / "missing" / "decision.json"
    assert (
        main(
            [
                "benchmark",
                "--contract",
                str(CONTRACT),
                "--budget",
                "2",
                "--output",
                str(comparison),
                "--decision-output",
                str(missing_decision),
            ]
        )
        == 3
    )
    assert not comparison.exists()


def test_console_entrypoint_propagates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["biasweave", "benchmark"])
    with pytest.raises(SystemExit) as raised:
        entrypoint()
    assert raised.value.code == 2
