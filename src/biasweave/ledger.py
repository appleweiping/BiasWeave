"""Append-only trial persistence and atomic run metadata."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

from biasweave._output import (
    WriterClaim,
    atomic_write_bytes,
    preflight_outputs,
    protect_open_descriptor,
)
from biasweave._strict_json import (
    JSONLimits,
    StrictJSONError,
    json_node_count,
    loads_strict_json,
    read_limited_bytes,
)
from biasweave.errors import CheckpointError
from biasweave.model import Point, Trial, TrialStatus

_MAX_LEDGER_BYTES = 67_108_864
_MAX_LEDGER_LINE_BYTES = 1_048_576
_MAX_LEDGER_LINES = 200_000
_MAX_LEDGER_RECORDS = 100_000
_MAX_LEDGER_TOTAL_NODES = 2_000_000
_MAX_METADATA_BYTES = 1_048_576
_MAX_FIELDS = 4_096
_MAX_METRICS = 4_096
_MAX_TEXT_CHARS = 65_536
_MAX_TOTAL_TEXT_CHARS = 262_144
_MAX_JSON_NUMBER_CHARACTERS = 129
_POINT_KEY = re.compile(r"[0-9a-f]{64}\Z")
_LEDGER_JSON_LIMITS = JSONLimits(
    max_bytes=_MAX_LEDGER_LINE_BYTES,
    max_depth=64,
    max_nodes=10_000,
    # Problem values may have 128 digits plus a leading minus sign.
    max_number_characters=_MAX_JSON_NUMBER_CHARACTERS,
)
_METADATA_JSON_LIMITS = JSONLimits(
    max_bytes=_MAX_METADATA_BYTES,
    max_depth=64,
    max_nodes=10_000,
    max_number_characters=_MAX_JSON_NUMBER_CHARACTERS,
)


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


def _validate_text_envelope(value: object, context: str) -> None:
    pending = [value]
    total = 0
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if len(item) > _MAX_TEXT_CHARS:
                raise CheckpointError(
                    f"{context} contains text longer than {_MAX_TEXT_CHARS} characters"
                )
            total += len(item)
            if total > _MAX_TOTAL_TEXT_CHARS:
                raise CheckpointError(
                    f"{context} exceeds {_MAX_TOTAL_TEXT_CHARS} total text characters"
                )
        elif isinstance(item, Mapping):
            if len(item) > _MAX_FIELDS:
                raise CheckpointError(f"{context} contains too many object fields")
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list | tuple):
            pending.extend(item)


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
    _validate_text_envelope(raw, "trial record")
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
    if len(coordinates_raw) > _MAX_FIELDS or len(values) > _MAX_FIELDS:
        raise CheckpointError("trial.point exceeds the variable-count limit")
    if _POINT_KEY.fullmatch(key) is None:
        raise CheckpointError("trial.point key must be a lowercase SHA-256 digest")
    for name, value in values.items():
        if not isinstance(name, str) or not name or len(name) > 256:
            raise CheckpointError("trial.point variable names are invalid")
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise CheckpointError(f"trial.point value {name!r} is not scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise CheckpointError(f"trial.point value {name!r} must be finite")
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
    if len(metrics_raw) > _MAX_METRICS:
        raise CheckpointError("trial.metrics exceeds the metric-count limit")
    if not all(isinstance(name, str) and 0 < len(name) <= 256 for name in metrics_raw):
        raise CheckpointError("trial.metrics keys must be bounded non-empty strings")
    metrics = {name: _number(value, f"metric {name}") for name, value in metrics_raw.items()}
    objective_raw = raw.get("objective_vector")
    if not isinstance(objective_raw, list):
        raise CheckpointError("trial.objective_vector must be an array")
    if len(objective_raw) > _MAX_METRICS:
        raise CheckpointError("trial.objective_vector exceeds the metric-count limit")
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

    def append(
        self,
        trials: Iterable[Trial],
        *,
        protected: Iterable[str | Path] = (),
        _claim: WriterClaim | None = None,
    ) -> None:
        protected_paths = tuple(protected)
        batch = tuple(trials)
        if not batch:
            return
        if _claim is None:
            with WriterClaim(self.path.parent) as claim:
                self.append(batch, protected=protected_paths, _claim=claim)
            return
        if not _claim.owns(self.path.parent, scope="run"):
            raise CheckpointError("trial ledger requires the live run writer claim")
        records: list[bytes] = []
        batch_nodes = 0
        for trial in batch:
            rendered, nodes = _trial_record_bytes(trial)
            batch_nodes += nodes
            if batch_nodes > _MAX_LEDGER_TOTAL_NODES:
                raise CheckpointError("trial ledger exceeds the total JSON node limit")
            records.append(rendered + b"\n")
        preflight_outputs((self.path,), force=True, protected=protected_paths)
        existing_payload = b""
        try:
            existing_payload = read_limited_bytes(
                self.path, max_bytes=_MAX_LEDGER_BYTES, context="trial ledger"
            )
        except FileNotFoundError:
            pass
        except (OSError, StrictJSONError) as error:
            raise CheckpointError(f"cannot inspect trial ledger {self.path}: {error}") from error
        existing_trials = self.read()
        append_base = existing_payload
        repair_truncated_tail = False
        if existing_payload and not existing_payload.endswith(b"\n"):
            prefix, _separator, tail = existing_payload.rpartition(b"\n")
            try:
                raw_tail = loads_strict_json(
                    tail, limits=_LEDGER_JSON_LIMITS, context="trial ledger final line"
                )
                trial_from_dict(raw_tail)
            except StrictJSONError as error:
                if not error.syntax_error:
                    raise CheckpointError(f"invalid trial ledger final line: {error}") from error
                append_base = prefix + (b"\n" if prefix else b"")
                repair_truncated_tail = True
        if [trial.trial_id for trial in batch] != list(
            range(len(existing_trials), len(existing_trials) + len(batch))
        ):
            raise CheckpointError("appended trial IDs must continue the durable ledger")
        existing_nodes = sum(
            json_node_count(
                trial.as_dict(),
                max_depth=_LEDGER_JSON_LIMITS.max_depth,
                max_nodes=_MAX_LEDGER_TOTAL_NODES,
                context="trial ledger",
            )
            for trial in existing_trials
        )
        if len(existing_trials) + len(batch) > _MAX_LEDGER_RECORDS:
            raise CheckpointError(f"trial ledger exceeds {_MAX_LEDGER_RECORDS} record limit")
        if existing_nodes + batch_nodes > _MAX_LEDGER_TOTAL_NODES:
            raise CheckpointError("trial ledger exceeds the total JSON node limit")
        physical_lines = append_base.count(b"\n") + bool(
            append_base and not append_base.endswith(b"\n")
        )
        if physical_lines + len(records) > _MAX_LEDGER_LINES:
            raise CheckpointError(f"trial ledger exceeds {_MAX_LEDGER_LINES} physical line limit")
        separator = b"" if not append_base or append_base.endswith(b"\n") else b"\n"
        addition = separator + b"".join(records)
        if len(append_base) + len(addition) > _MAX_LEDGER_BYTES:
            raise CheckpointError(f"trial ledger exceeds {_MAX_LEDGER_BYTES} byte limit")
        if repair_truncated_tail:
            atomic_write_bytes(
                self.path,
                append_base + addition,
                force=True,
                protected=protected_paths,
            )
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                protect_open_descriptor(descriptor, self.path, protected_paths)
                view = memoryview(addition)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("ledger append made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise CheckpointError(f"cannot append trial ledger {self.path}: {error}") from error

    def read(self) -> list[Trial]:
        try:
            payload = read_limited_bytes(
                self.path,
                max_bytes=_MAX_LEDGER_BYTES,
                context="trial ledger",
            )
        except FileNotFoundError:
            return []
        except StrictJSONError as error:
            raise CheckpointError(f"cannot read trial ledger {self.path}: {error}") from error
        except OSError as error:
            raise CheckpointError(f"cannot read trial ledger {self.path}: {error}") from error
        trials: list[Trial] = []
        physical_lines = 0
        total_nodes = 0
        stream = BytesIO(payload)
        for index, encoded_line in enumerate(stream):
            physical_lines += 1
            if physical_lines > _MAX_LEDGER_LINES:
                raise CheckpointError(
                    f"trial ledger exceeds {_MAX_LEDGER_LINES} physical line limit"
                )
            complete_line = encoded_line.endswith(b"\n")
            line = encoded_line[:-1] if complete_line else encoded_line
            if line.endswith(b"\r"):
                line = line[:-1]
            if len(line) > _MAX_LEDGER_LINE_BYTES:
                raise CheckpointError(
                    f"invalid trial ledger line {index + 1}: exceeds "
                    f"{_MAX_LEDGER_LINE_BYTES} byte line limit"
                )
            if not line.strip():
                continue
            if len(trials) >= _MAX_LEDGER_RECORDS:
                raise CheckpointError(f"trial ledger exceeds {_MAX_LEDGER_RECORDS} record limit")
            try:
                raw = loads_strict_json(
                    line,
                    limits=_LEDGER_JSON_LIMITS,
                    context=f"trial ledger line {index + 1}",
                )
                total_nodes += json_node_count(
                    raw,
                    max_depth=_LEDGER_JSON_LIMITS.max_depth,
                    max_nodes=_MAX_LEDGER_TOTAL_NODES - total_nodes,
                    context="trial ledger",
                )
            except StrictJSONError as error:
                is_truncated_tail = stream.tell() == len(payload) and not complete_line
                if is_truncated_tail and error.syntax_error:
                    break
                raise CheckpointError(f"invalid trial ledger line {index + 1}: {error}") from error
            trials.append(trial_from_dict(raw))
        expected_ids = list(range(len(trials)))
        actual_ids = [trial.trial_id for trial in trials]
        if actual_ids != expected_ids:
            raise CheckpointError("trial IDs must be contiguous and start at zero")
        return trials


def _trial_record_bytes(trial: Trial) -> tuple[bytes, int]:
    try:
        rendered = json.dumps(
            trial.as_dict(), sort_keys=True, allow_nan=False, ensure_ascii=True
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CheckpointError(f"cannot serialize trial ledger record: {error}") from error
    if len(rendered) > _MAX_LEDGER_LINE_BYTES:
        raise CheckpointError(
            f"trial ledger record exceeds {_MAX_LEDGER_LINE_BYTES} byte line limit"
        )
    try:
        raw = loads_strict_json(rendered, limits=_LEDGER_JSON_LIMITS, context="trial ledger record")
        nodes = json_node_count(
            raw,
            max_depth=_LEDGER_JSON_LIMITS.max_depth,
            max_nodes=_MAX_LEDGER_TOTAL_NODES,
            context="trial ledger",
        )
        trial_from_dict(raw)
    except StrictJSONError as error:
        raise CheckpointError(f"trial ledger record violates its read envelope: {error}") from error
    return rendered, nodes


def ledger_bytes(trials: Iterable[Trial]) -> bytes:
    """Serialize a complete contiguous ledger under the reader's exact envelope."""

    records: list[bytes] = []
    total_nodes = 0
    for expected_id, trial in enumerate(trials):
        if trial.trial_id != expected_id:
            raise CheckpointError("trial IDs must be contiguous and start at zero")
        rendered, nodes = _trial_record_bytes(trial)
        total_nodes += nodes
        if len(records) >= _MAX_LEDGER_RECORDS:
            raise CheckpointError(f"trial ledger exceeds {_MAX_LEDGER_RECORDS} record limit")
        if len(records) >= _MAX_LEDGER_LINES:
            raise CheckpointError(f"trial ledger exceeds {_MAX_LEDGER_LINES} physical line limit")
        if total_nodes > _MAX_LEDGER_TOTAL_NODES:
            raise CheckpointError("trial ledger exceeds the total JSON node limit")
        records.append(rendered + b"\n")
    payload = b"".join(records)
    if len(payload) > _MAX_LEDGER_BYTES:
        raise CheckpointError(f"trial ledger exceeds {_MAX_LEDGER_BYTES} byte limit")
    return payload


def write_metadata(
    path: str | Path,
    data: Mapping[str, Any],
    *,
    force: bool = False,
    protected: Iterable[str | Path] = (),
    _claim: WriterClaim | None = None,
) -> Path:
    target = Path(path)
    if _claim is None:
        with WriterClaim(target.parent) as claim:
            return write_metadata(
                target,
                data,
                force=force,
                protected=protected,
                _claim=claim,
            )
    if not _claim.owns(target.parent, scope="run"):
        raise CheckpointError("run metadata requires the live run writer claim")
    payload = metadata_bytes(data)
    return atomic_write_bytes(target, payload, force=force, protected=protected)


def metadata_bytes(data: Mapping[str, Any]) -> bytes:
    """Serialize metadata under the exact envelope enforced by the reader."""

    try:
        payload = (
            json.dumps(dict(data), indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True)
            + "\n"
        ).encode("ascii")
        raw = loads_strict_json(payload, limits=_METADATA_JSON_LIMITS, context="run metadata")
        _validate_text_envelope(raw, "run metadata")
    except (TypeError, ValueError, OverflowError, StrictJSONError) as error:
        raise CheckpointError(f"cannot serialize run metadata: {error}") from error
    return payload


def read_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = read_limited_bytes(
            source,
            max_bytes=_MAX_METADATA_BYTES,
            context="run metadata",
        )
        data = loads_strict_json(
            payload,
            limits=_METADATA_JSON_LIMITS,
            context="run metadata",
        )
        _validate_text_envelope(data, "run metadata")
    except (OSError, StrictJSONError) as error:
        raise CheckpointError(f"cannot read run metadata {source}: {error}") from error
    if not isinstance(data, dict):
        raise CheckpointError("run metadata must be an object")
    return data
