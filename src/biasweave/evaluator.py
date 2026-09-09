"""Trusted Python and no-shell JSON subprocess evaluator adapters."""

from __future__ import annotations

import importlib
import json
import math
import os
import queue
import subprocess  # nosec B404
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import BinaryIO, Protocol, cast

from biasweave._strict_json import JSONLimits, StrictJSONError, loads_strict_json
from biasweave.errors import EvaluationError
from biasweave.model import Problem, Scalar

_EVALUATOR_JSON_LIMITS = JSONLimits(
    max_bytes=1_048_576,
    max_depth=16,
    max_nodes=10_000,
    max_number_characters=128,
)
_MAX_EVALUATOR_STDERR_BYTES = 65_536
_PIPE_CHUNK_BYTES = 65_536
_MAX_METRICS = 4_096
_MAX_METRIC_NAME_CHARS = 256


class Evaluator(Protocol):
    def __call__(self, point: Mapping[str, Scalar]) -> Mapping[str, float]: ...


def validate_metrics(problem: Problem, raw: object) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise EvaluationError("evaluator result must be a metric mapping")
    if len(raw) > _MAX_METRICS:
        raise EvaluationError(f"evaluator returned more than {_MAX_METRICS} metrics")
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name or len(name) > _MAX_METRIC_NAME_CHARS:
            raise EvaluationError("metric names must be bounded non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise EvaluationError(f"metric {name!r} must be numeric")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as error:
            raise EvaluationError(f"metric {name!r} must be finite") from error
        if not math.isfinite(numeric):
            raise EvaluationError(f"metric {name!r} must be finite")
        metrics[name] = numeric
    required = {objective.metric for objective in problem.objectives}
    required.update(constraint.metric for constraint in problem.constraints)
    missing = sorted(required - set(metrics))
    if missing:
        raise EvaluationError(f"evaluator omitted required metrics: {', '.join(missing)}")
    return metrics


def load_python_evaluator(
    specification: str,
) -> Callable[[Mapping[str, Scalar]], Mapping[str, float]]:
    """Load a trusted `python:module.path:function` evaluator."""
    prefix, separator, target = specification.partition(":")
    if prefix != "python" or not separator:
        raise EvaluationError("evaluator must have python:module:function form")
    module_name, separator, attribute_name = target.rpartition(":")
    if not separator or not module_name or not attribute_name:
        raise EvaluationError("evaluator must have python:module:function form")
    try:
        module = importlib.import_module(module_name)
        evaluator = getattr(module, attribute_name)
    except (ImportError, AttributeError) as error:
        raise EvaluationError(f"cannot load evaluator {specification}: {error}") from error
    if not callable(evaluator):
        raise EvaluationError(f"evaluator target is not callable: {specification}")
    return cast(Callable[[Mapping[str, Scalar]], Mapping[str, float]], evaluator)


@dataclass(frozen=True, slots=True)
class CommandEvaluator:
    """Run one trusted executable per point with JSON stdin/stdout."""

    argv: tuple[str, ...]
    timeout_seconds: float = 300.0

    def __init__(self, argv: Sequence[str], timeout_seconds: float = 300.0):
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise EvaluationError("command argv must contain non-empty strings")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise EvaluationError("command timeout must be positive and finite")
        object.__setattr__(self, "argv", tuple(argv))
        object.__setattr__(self, "timeout_seconds", float(timeout_seconds))

    def __call__(self, point: Mapping[str, Scalar]) -> Mapping[str, float]:
        try:
            request = json.dumps(
                dict(point), sort_keys=True, ensure_ascii=True, allow_nan=False
            ).encode("ascii")
        except (TypeError, ValueError, OverflowError) as error:
            raise EvaluationError(f"evaluator point is not canonical JSON: {error}") from error
        if len(request) > _EVALUATOR_JSON_LIMITS.max_bytes:
            raise EvaluationError(
                f"evaluator point exceeds {_EVALUATOR_JSON_LIMITS.max_bytes} byte input limit"
            )
        try:
            process = subprocess.Popen(  # nosec B603
                self.argv,
                shell=False,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise EvaluationError(f"evaluator command failed to run: {error}") from error

        events: queue.Queue[tuple[str, str, object]] = queue.Queue()

        def drain(name: str, stream: BinaryIO, limit: int) -> None:
            chunks: list[bytes] = []
            size = 0
            try:
                while True:
                    # os.read returns currently available pipe data instead of
                    # asking BufferedReader to fill the entire chunk first.
                    chunk = os.read(stream.fileno(), _PIPE_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > limit:
                        events.put(("limit", name, limit))
                        with suppress(OSError):
                            process.kill()
                        return
                    chunks.append(chunk)
                events.put(("data", name, b"".join(chunks)))
            except OSError as error:
                events.put(("error", name, error))
                with suppress(OSError):
                    process.kill()
            finally:
                with suppress(OSError):
                    stream.close()

        if process.stdout is None or process.stderr is None or process.stdin is None:
            process.kill()
            process.wait()
            raise EvaluationError("evaluator command pipes could not be created")
        stdin = process.stdin
        readers = (
            threading.Thread(
                target=drain,
                args=("stdout", process.stdout, _EVALUATOR_JSON_LIMITS.max_bytes),
                daemon=True,
            ),
            threading.Thread(
                target=drain,
                args=("stderr", process.stderr, _MAX_EVALUATOR_STDERR_BYTES),
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()

        def feed() -> None:
            try:
                stdin.write(request)
                stdin.flush()
            except (BrokenPipeError, OSError):
                # An early process exit or output-limit kill closes stdin. The
                # exit/reader result below supplies the useful diagnostic.
                pass
            finally:
                with suppress(OSError):
                    stdin.close()

        writer = threading.Thread(target=feed, name="biasweave-stdin", daemon=True)
        writer.start()
        try:
            try:
                returncode = process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.wait()
                raise EvaluationError(
                    f"evaluator command exceeded {self.timeout_seconds:g} second timeout"
                ) from error
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            writer.join(timeout=5.0)
            for reader in readers:
                reader.join(timeout=5.0)

        outputs: dict[str, bytes] = {}
        failures: list[tuple[str, str, object]] = []
        while not events.empty():
            kind, name, value = events.get_nowait()
            if kind == "data":
                outputs[name] = cast(bytes, value)
            else:
                failures.append((kind, name, value))
        if writer.is_alive() or any(reader.is_alive() for reader in readers):
            raise EvaluationError("evaluator command pipe cleanup did not complete")
        if failures:
            kind, name, value = failures[0]
            if kind == "limit":
                raise EvaluationError(f"evaluator command {name} exceeds {value} byte limit")
            raise EvaluationError(f"evaluator command {name} could not be read: {value}")
        stdout = outputs.get("stdout", b"")
        stderr = outputs.get("stderr", b"")
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise EvaluationError(
                f"evaluator command exited with {returncode}: {detail or 'no stderr'}"
            )
        try:
            result = loads_strict_json(
                stdout,
                limits=_EVALUATOR_JSON_LIMITS,
                context="evaluator command JSON",
            )
        except StrictJSONError as error:
            raise EvaluationError(f"evaluator command returned invalid JSON: {error}") from error
        if not isinstance(result, Mapping):
            raise EvaluationError("evaluator command JSON must be an object")
        return result
