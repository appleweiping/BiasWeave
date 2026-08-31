"""Strict interoperability and baseline runner for analog sizing benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from biasweave._strict_json import (
    JSONLimits,
    StrictJSONError,
    loads_strict_json,
    read_limited_bytes,
)
from biasweave.archive import Archive
from biasweave.dominance import assess
from biasweave.encoding import make_point
from biasweave.engine import optimize
from biasweave.errors import ProblemError
from biasweave.evaluator import validate_metrics
from biasweave.model import OptimizationResult, Problem, RunConfig, Scalar, Trial
from biasweave.problem import parse_problem

_SCHEMA = "org.topology-lantern.analog-sizing-benchmark"
_MODEL = "deterministic-analytic-proxy-v1"
_MAX_JSON_BYTES = 1_048_576
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 10_000
_BENCHMARK_JSON_LIMITS = JSONLimits(
    max_bytes=_MAX_JSON_BYTES,
    max_depth=_MAX_JSON_DEPTH,
    max_nodes=_MAX_JSON_NODES,
    max_number_characters=4_096,
)
_MIN_SEED = -(1 << 63)
_MAX_SEED = (1 << 63) - 1


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ProblemError("value is not finite canonical JSON") from error


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class AnalogBenchmark:
    contract_sha256: str
    problem: Problem
    candidates: Mapping[str, Mapping[str, int | str]]
    supply_voltage: float
    gain_min: float
    phase_margin_min: float

    def evaluate(self, point: Mapping[str, Scalar]) -> dict[str, float]:
        """Evaluate the documented deterministic proxy (not a circuit simulation)."""
        expected = {variable.name for variable in self.problem.free_variables}
        if set(point) != expected:
            raise ValueError("benchmark point has missing or unknown variables")
        topology_id = point["topology_id"]
        if not isinstance(topology_id, str) or topology_id not in self.candidates:
            raise ValueError("unknown topology_id")
        topology = self.candidates[topology_id]
        numeric: dict[str, float] = {}
        for name in expected - {"topology_id"}:
            value = point[name]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be numeric")
            converted = float(value)
            variable = self.problem.variable_map[name]
            if (
                not math.isfinite(converted)
                or not isinstance(variable.low, int | float)
                or not isinstance(variable.high, int | float)
                or not float(variable.low) <= converted <= float(variable.high)
            ):
                raise ValueError(f"{name} must be finite and within its bounds")
            numeric[name] = converted
        width = numeric["width_um"]
        length = numeric["length_um"]
        bias = numeric["bias_ua"]
        compensation = numeric["compensation_pf"]
        gm_proxy = math.sqrt(bias * width / length)
        devices = int(topology["device_count"])
        transistors = int(topology["transistor_count"])
        passives = int(topology["passive_count"])
        stages = int(topology["stage_count"])
        headroom = int(topology["headroom_units"])
        symmetry = int(topology["symmetry_penalty"])
        gain_db = 20.0 * math.log10(1.0 + 2.5 * gm_proxy / (1.0 + headroom + symmetry))
        bandwidth = 180.0 * gm_proxy / (devices * (1.0 + compensation))
        power = self.supply_voltage * bias * (1.0 + 0.12 * devices) / 1000.0
        area = width * length * max(1, transistors) + 8.0 * passives + 2.0 * devices
        phase_margin = 42.0 + 22.0 * compensation / (compensation + 0.8 * stages)
        return {
            "gain_db": gain_db,
            "phase_margin_deg": phase_margin,
            "power_mw": power,
            "area_um2": area,
            "bandwidth_mhz": bandwidth,
        }


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProblemError(f"{context} must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise ProblemError(f"{context} is outside the supported numeric range") from error
    if not math.isfinite(result):
        raise ProblemError(f"{context} must be finite")
    return result


def _exact(value: Any, fields: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProblemError(f"{context} has missing or unknown fields")
    return value


def _hex_digest(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProblemError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def load_analog_benchmark(path: str | Path) -> AnalogBenchmark:
    """Load and cryptographically verify a strict version-1 contract."""
    contract_path = Path(path)
    try:
        payload = read_limited_bytes(
            contract_path,
            max_bytes=_MAX_JSON_BYTES,
            context="benchmark contract",
        )
    except (OSError, StrictJSONError) as error:
        raise ProblemError(f"cannot load benchmark contract: {error}") from error
    try:
        raw = loads_strict_json(
            payload,
            limits=_BENCHMARK_JSON_LIMITS,
            context="benchmark JSON",
        )
    except StrictJSONError as error:
        raise ProblemError(f"cannot load benchmark contract: {error}") from error
    top = _exact(
        raw,
        {
            "schema",
            "version",
            "model",
            "disclaimer",
            "provenance",
            "source",
            "candidates",
            "sizing",
            "objectives",
            "constraints",
            "expected_metrics",
            "contract_sha256",
        },
        "benchmark",
    )
    if top["schema"] != _SCHEMA or top["model"] != _MODEL:
        raise ProblemError("unsupported benchmark schema or model")
    if (
        not isinstance(top["version"], int)
        or isinstance(top["version"], bool)
        or top["version"] != 1
    ):
        raise ProblemError("benchmark version must be integer 1")
    if not isinstance(top["disclaimer"], str) or not top["disclaimer"]:
        raise ProblemError("benchmark disclaimer must be a non-empty string")
    provenance = _exact(
        top["provenance"],
        {"data_source", "license", "generation_command"},
        "provenance",
    )
    if not all(isinstance(value, str) and value for value in provenance.values()):
        raise ProblemError("benchmark provenance values must be non-empty strings")
    digest = _hex_digest(top["contract_sha256"], "benchmark contract_sha256")
    body = dict(top)
    del body["contract_sha256"]
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if hashlib.sha256(encoded.encode("ascii")).hexdigest() != digest:
        raise ProblemError("benchmark contract digest does not match its content")
    source = _exact(top["source"], {"spec_sha256", "candidate_count", "supply_voltage"}, "source")
    _hex_digest(source["spec_sha256"], "source.spec_sha256")
    supply = _number(source["supply_voltage"], "source.supply_voltage")
    if supply <= 0.0:
        raise ProblemError("source.supply_voltage must be greater than zero")
    rows = top["candidates"]
    if not isinstance(rows, list) or not rows:
        raise ProblemError("candidates must be a non-empty array")
    candidate_fields = {
        "candidate_id",
        "signature",
        "device_count",
        "transistor_count",
        "passive_count",
        "stage_count",
        "headroom_units",
        "symmetry_penalty",
    }
    candidates: dict[str, Mapping[str, int | str]] = {}
    for index, item in enumerate(rows):
        row = _exact(item, candidate_fields, f"candidates[{index}]")
        identifier = row["candidate_id"]
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"TL-[0-9a-f]{12}", identifier) is None
            or identifier in candidates
        ):
            raise ProblemError("candidate IDs must be unique TL-prefixed lowercase identifiers")
        _hex_digest(row["signature"], "candidate signature")
        for field in candidate_fields - {"candidate_id", "signature"}:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProblemError(f"candidate {field} must be a non-negative integer")
        candidates[identifier] = MappingProxyType(dict(row))
    count = source["candidate_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(candidates):
        raise ProblemError("source.candidate_count disagrees with candidates")
    sizing = _exact(
        top["sizing"], {"width_um", "length_um", "bias_ua", "compensation_pf"}, "sizing"
    )
    variables: dict[str, Any] = {"topology_id": {"kind": "choice", "values": list(candidates)}}
    for name in ("width_um", "length_um", "bias_ua", "compensation_pf"):
        raw_bounds = sizing[name]
        bounds = _exact(raw_bounds, {"low", "high", "scale"}, f"sizing.{name}")
        low = _number(bounds["low"], f"sizing.{name}.low")
        high = _number(bounds["high"], f"sizing.{name}.high")
        if low <= 0.0 or low >= high:
            raise ProblemError(f"sizing.{name} must satisfy 0 < low < high")
        if bounds["scale"] != "log":
            raise ProblemError(f"sizing.{name}.scale must be log")
        variables[name] = {
            "kind": "real",
            "low": low,
            "high": high,
            "scale": "log",
        }
    if top["objectives"] != ["power_mw", "area_um2", "bandwidth_mhz"]:
        raise ProblemError("benchmark objectives are unsupported")
    if top["expected_metrics"] != [
        "gain_db",
        "phase_margin_deg",
        "power_mw",
        "area_um2",
        "bandwidth_mhz",
    ]:
        raise ProblemError("benchmark expected_metrics are unsupported")
    limits = _exact(top["constraints"], {"gain_db_min", "phase_margin_deg_min"}, "constraints")
    gain_min = _number(limits["gain_db_min"], "constraints.gain_db_min")
    phase_min = _number(limits["phase_margin_deg_min"], "constraints.phase_margin_deg_min")
    if gain_min <= 0.0:
        raise ProblemError("constraints.gain_db_min must be greater than zero")
    if not 0.0 < phase_min <= 180.0:
        raise ProblemError("constraints.phase_margin_deg_min must be in (0, 180]")
    problem = parse_problem(
        {
            "schema_version": 1,
            "variables": variables,
            "objectives": [
                {"metric": "power_mw", "goal": "min", "scale": 1.0},
                {"metric": "area_um2", "goal": "min", "scale": 100.0},
                {"metric": "bandwidth_mhz", "goal": "max", "scale": 100.0},
            ],
            "constraints": [
                {"metric": "gain_db", "relation": "ge", "limit": gain_min, "scale": 20.0},
                {"metric": "phase_margin_deg", "relation": "ge", "limit": phase_min, "scale": 20.0},
            ],
        },
        source_hash=digest,
        source_path=str(contract_path),
    )
    return AnalogBenchmark(
        digest, problem, MappingProxyType(candidates), supply, gain_min, phase_min
    )


def _summary(trials: tuple[Trial, ...], problem: Problem) -> dict[str, Any]:
    archive = Archive(problem, list(trials))
    feasible = [trial for trial in trials if trial.feasible]
    representative = _balanced_representative(archive.frontier)
    return {
        "evaluations": len(trials),
        "unique_points": len({trial.point.key for trial in trials}),
        "feasible": len(feasible),
        "frontier": len(archive.frontier),
        "best": {
            objective.metric: (min if objective.goal.value == "min" else max)(
                (trial.metrics[objective.metric] for trial in feasible), default=None
            )
            for objective in problem.objectives
        },
        "representative": None
        if representative is None
        else {
            "selection_policy": "normalized-l1-to-observed-ideal-v1",
            "trial_id": representative.trial_id,
            "point_key": representative.point.key,
            "values": dict(representative.point.values),
            "metrics": dict(sorted(representative.metrics.items())),
        },
    }


def _balanced_representative(frontier: tuple[Trial, ...]) -> Trial | None:
    """Choose one descriptive equal-coordinate compromise.

    Each already-minimized objective vector is normalized by the observed
    frontier range, which gives every non-constant normalized objective one
    equal L1 contribution.  The lowest distance to the observed ideal wins,
    with trial order as the deterministic tie-break.  A zero-range objective
    adds no distance.  This is an explicit selection convention, not a claim
    that a user's engineering preferences are equally weighted or optimal.
    """
    if not frontier:
        return None
    dimensions = len(frontier[0].objective_vector)
    minima = [
        min(trial.objective_vector[index] for trial in frontier) for index in range(dimensions)
    ]
    maxima = [
        max(trial.objective_vector[index] for trial in frontier) for index in range(dimensions)
    ]

    def distance(trial: Trial) -> tuple[float, int]:
        terms = []
        for index, value in enumerate(trial.objective_vector):
            span = maxima[index] - minima[index]
            terms.append(0.0 if span == 0.0 else (value - minima[index]) / span)
        return math.fsum(terms), trial.trial_id

    return min(frontier, key=distance)


def compare_with_random(benchmark: AnalogBenchmark, *, budget: int, seed: int) -> dict[str, Any]:
    """Compare BiasWeave with seeded uniform random search at identical budget."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ProblemError("benchmark budget must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not _MIN_SEED <= seed <= _MAX_SEED:
        raise ProblemError("benchmark seed must be a signed 64-bit integer")
    woven: OptimizationResult = optimize(
        benchmark.problem,
        benchmark.evaluate,
        evaluator_id=f"analytic:{benchmark.contract_sha256}",
        config=RunConfig(budget, seed=seed, batch_size=min(8, budget)),
    )
    # Pseudorandomness is intentional here: the seed defines the baseline sequence.
    randomizer = random.Random(seed)  # nosec B311
    random_trials: list[Trial] = []
    seen: set[str] = set()
    while len(random_trials) < budget:
        point = make_point(
            benchmark.problem, [randomizer.random() for _ in benchmark.problem.free_variables]
        )
        if point.key in seen:
            continue
        seen.add(point.key)
        metrics = validate_metrics(benchmark.problem, benchmark.evaluate(point.values))
        random_trials.append(assess(benchmark.problem, len(random_trials), point, metrics))
    body: dict[str, Any] = {
        "schema": "org.biasweave.benchmark-comparison",
        "version": 1,
        "contract_sha256": benchmark.contract_sha256,
        "budget": budget,
        "seed": seed,
        "algorithms": {
            "biasweave": _summary(woven.trials, benchmark.problem),
            "uniform_random_search": _summary(tuple(random_trials), benchmark.problem),
        },
        "interpretation": "Descriptive same-budget comparison; no statistical or state-of-the-art claim.",
    }
    body["comparison_sha256"] = _canonical_digest(body)
    return body


def sizing_decision(
    benchmark: AnalogBenchmark, comparison: Mapping[str, Any], *, algorithm: str = "biasweave"
) -> dict[str, Any]:
    """Export one content-bound proxy-sizing point for a downstream simulator.

    The decision carries the original topology signature and benchmark digest.
    Its measurements remain analytic proxies; a consumer must run and gate a
    real simulator independently.
    """
    expected_fields = {
        "schema",
        "version",
        "contract_sha256",
        "budget",
        "seed",
        "algorithms",
        "interpretation",
        "comparison_sha256",
    }
    if set(comparison) != expected_fields:
        raise ProblemError("comparison has missing or unknown fields")
    supplied_digest = _hex_digest(comparison.get("comparison_sha256"), "comparison_sha256")
    comparison_body = dict(comparison)
    del comparison_body["comparison_sha256"]
    if _canonical_digest(comparison_body) != supplied_digest:
        raise ProblemError("comparison digest does not match its content")
    if comparison.get("contract_sha256") != benchmark.contract_sha256:
        raise ProblemError("comparison does not belong to the supplied benchmark")
    comparison_version = comparison.get("version")
    if (
        comparison.get("schema") != "org.biasweave.benchmark-comparison"
        or isinstance(comparison_version, bool)
        or comparison_version != 1
    ):
        raise ProblemError("comparison schema or version is unsupported")
    algorithms = comparison.get("algorithms")
    if not isinstance(algorithms, Mapping) or algorithm not in algorithms:
        raise ProblemError("comparison does not contain the requested algorithm")
    summary = algorithms[algorithm]
    if not isinstance(summary, Mapping):
        raise ProblemError("comparison algorithm summary is invalid")
    selected = summary.get("representative")
    if not isinstance(selected, Mapping):
        raise ProblemError("comparison has no feasible representative")
    values = selected.get("values")
    metrics = selected.get("metrics")
    if not isinstance(values, Mapping) or not isinstance(metrics, Mapping):
        raise ProblemError("comparison representative is invalid")
    topology_id = values.get("topology_id")
    if not isinstance(topology_id, str) or topology_id not in benchmark.candidates:
        raise ProblemError("comparison representative topology is invalid")
    expected_values = {variable.name for variable in benchmark.problem.free_variables}
    if set(values) != expected_values or benchmark.evaluate(values) != dict(metrics):
        raise ProblemError("comparison representative does not replay against the benchmark")
    point_key = selected.get("point_key")
    policy = selected.get("selection_policy")
    budget = comparison.get("budget")
    seed = comparison.get("seed")
    if (
        not isinstance(point_key, str)
        or len(point_key) != 64
        or any(character not in "0123456789abcdef" for character in point_key)
        or policy != "normalized-l1-to-observed-ideal-v1"
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ProblemError("comparison representative provenance is invalid")
    replayed = compare_with_random(benchmark, budget=budget, seed=seed)
    if _canonical_json(comparison) != _canonical_json(replayed):
        raise ProblemError("comparison does not replay to the claimed search result")
    topology = benchmark.candidates[topology_id]
    body: dict[str, Any] = {
        "schema": "org.biasweave.sizing-decision",
        "version": 1,
        "source": {
            "benchmark_sha256": benchmark.contract_sha256,
            "comparison_sha256": supplied_digest,
            "topology_id": topology_id,
            "topology_signature": topology["signature"],
        },
        "selection": {
            "algorithm": algorithm,
            "budget": budget,
            "seed": seed,
            "policy": policy,
            "point_key": point_key,
        },
        "variables": {name: values[name] for name in sorted(values) if name != "topology_id"},
        "proxy_metrics": dict(sorted(metrics.items())),
        "disclaimer": ("Analytic proxy selection only; downstream SPICE simulation is required."),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    body["decision_sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return body
