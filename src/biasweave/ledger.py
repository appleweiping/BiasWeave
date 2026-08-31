"""Append-only trial persistence and atomic run metadata."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from biasweave.errors import CheckpointError
from biasweave.model import Point, Trial, TrialStatus


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CheckpointError(f"{context} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise CheckpointError(f"{context} must be finite") from error
    if not math.isfinite(result):
        raise CheckpointError(f"{context} must be finite")
    return result


def trial_from_dict(raw: Any) -> Trial:
    if not isinstance(raw, Mapping):
        raise CheckpointError("trial record must be an object")
    allowed = {
        "trial_id",
        "point",
        "status",
        "metrics",
        "error",
        "feasible",
        "violation",
        "max_violation",
        "objective_vector",
    }
    if set(raw) - allowed:
        raise CheckpointError("trial record contains unknown fields")
    point_raw = raw.get("point")
    if not isinstance(point_raw, Mapping):
        raise CheckpointError("trial.point must be an object")
    if set(point_raw) != {"coordinates", "values", "key"}:
        raise CheckpointError("trial.point fields are invalid")
    coordinates_raw = point_raw.get("coordinates")
    values = point_raw.get("values")
    key = point_raw.get("key")
    if (
        not isinstance(coordinates_raw, list)
        or not isinstance(values, dict)
        or not isinstance(key, str)
    ):
        raise CheckpointError("trial.point coordinates, values, and key are invalid")
    coordinates = tuple(_number(value, "trial coordinate") for value in coordinates_raw)
    status_raw = raw.get("status")
    if not isinstance(status_raw, str):
        raise CheckpointError("trial.status is invalid")
    try:
        status = TrialStatus(status_raw)
    except ValueError as error:
        raise CheckpointError("trial.status is invalid") from error
    trial_id = raw.get("trial_id")
    if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 0:
        raise CheckpointError("trial_id must be a non-negative integer")
    metrics_raw = raw.get("metrics")
    if not isinstance(metrics_raw, Mapping):
        raise CheckpointError("trial.metrics must be an object")
    if not all(isinstance(name, str) for name in metrics_raw):
        raise CheckpointError("trial.metrics keys must be strings")
    metrics = {name: _number(value, f"metric {name}") for name, value in metrics_raw.items()}
    objective_raw = raw.get("objective_vector")
    if not isinstance(objective_raw, list):
        raise CheckpointError("trial.objective_vector must be an array")
    objective = tuple(_number(value, "objective value") for value in objective_raw)
    feasible = raw.get("feasible")
    if not isinstance(feasible, bool):
        raise CheckpointError("trial.feasible must be boolean")
    error_text = raw.get("error")
    if error_text is not None and not isinstance(error_text, str):
        raise CheckpointError("trial.error must be a string or null")
    violation_raw = raw.get("violation")
    max_violation_raw = raw.get("max_violation")
    if status is TrialStatus.FAILED:
        if violation_raw is not None or max_violation_raw is not None:
            raise CheckpointError("failed trial violations must be null")
        violation = math.inf
        max_violation = math.inf
    else:
        violation = _number(violation_raw, "trial.violation")
        max_violation = _number(max_violation_raw, "trial.max_violation")
    return Trial(
        trial_id,
        Point(coordinates, dict(values), key),
        status,
        metrics,
        error_text,
        feasible,
        violation,
        max_violation,
        objective,
    )


class TrialLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, trials: Iterable[Trial]) -> None:
        records = [json.dumps(trial.as_dict(), sort_keys=True, allow_nan=False) for trial in trials]
        if not records:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(records) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise CheckpointError(f"cannot append trial ledger {self.path}: {error}") from error

    def read(self) -> list[Trial]:
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as error:
            raise CheckpointError(f"cannot read trial ledger {self.path}: {error}") from error
        trials: list[Trial] = []
        lines = payload.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                is_truncated_tail = index == len(lines) - 1 and not payload.endswith(b"\n")
                if is_truncated_tail:
                    break
                raise CheckpointError(f"invalid trial ledger line {index + 1}: {error}") from error
            trials.append(trial_from_dict(raw))
        expected_ids = list(range(len(trials)))
        actual_ids = [trial.trial_id for trial in trials]
        if actual_ids != expected_ids:
            raise CheckpointError("trial IDs must be contiguous and start at zero")
        return trials


def write_metadata(path: str | Path, data: Mapping[str, Any]) -> Path:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(dict(data), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except (OSError, TypeError, ValueError) as error:
        raise CheckpointError(f"cannot write run metadata {target}: {error}") from error
    return target


def read_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read run metadata {source}: {error}") from error
    if not isinstance(data, dict):
        raise CheckpointError("run metadata must be an object")
    return data
