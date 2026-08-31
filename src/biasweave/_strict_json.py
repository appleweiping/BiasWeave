"""Resource-bounded JSON decoding for untrusted interoperability inputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class JSONLimits:
    """Hard limits applied before and after JSON decoding."""

    max_bytes: int = 1_048_576
    max_depth: int = 64
    max_nodes: int = 10_000
    max_number_characters: int = 128


class StrictJSONError(ValueError):
    """A JSON document is malformed or violates a resource/policy limit."""

    syntax_error: bool

    def __init__(self, message: str, *, syntax_error: bool = False) -> None:
        super().__init__(message)
        self.syntax_error = syntax_error


_DEFAULT_LIMITS = JSONLimits()


def read_limited_bytes(path: Path, *, max_bytes: int, context: str) -> bytes:
    """Read at most ``max_bytes`` and reject a larger regular or streamed file."""

    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise StrictJSONError(f"{context} exceeds {max_bytes} byte input limit")
    return payload


def json_node_count(value: object, *, max_depth: int, max_nodes: int, context: str) -> int:
    """Count JSON values iteratively while enforcing depth and node limits."""

    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > max_depth or nodes > max_nodes:
            raise StrictJSONError(f"{context} exceeds JSON complexity limits")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    return nodes


def loads_strict_json(
    payload: bytes | str,
    *,
    limits: JSONLimits = _DEFAULT_LIMITS,
    context: str = "JSON input",
) -> Any:
    """Decode strict JSON with duplicate, numeric, byte, and complexity checks."""

    if isinstance(payload, str):
        try:
            encoded = payload.encode("utf-8")
        except UnicodeEncodeError as error:
            raise StrictJSONError(f"invalid {context}: {error}", syntax_error=True) from error
    else:
        encoded = payload
    if len(encoded) > limits.max_bytes:
        raise StrictJSONError(f"{context} exceeds {limits.max_bytes} byte input limit")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise StrictJSONError(f"{context} contains duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise StrictJSONError(f"{context} contains non-finite number: {token}")

    def check_token(token: str) -> None:
        if len(token) > limits.max_number_characters:
            raise StrictJSONError(
                f"{context} contains a number longer than {limits.max_number_characters} characters"
            )

    def parse_integer(token: str) -> int:
        check_token(token)
        try:
            return int(token)
        except (ValueError, OverflowError) as error:
            raise StrictJSONError(f"{context} contains an invalid integer") from error

    def parse_floating(token: str) -> float:
        check_token(token)
        try:
            value = float(token)
        except (ValueError, OverflowError) as error:
            raise StrictJSONError(f"{context} contains an invalid number") from error
        if not math.isfinite(value):
            raise StrictJSONError(f"{context} contains non-finite number: {token}")
        return value

    try:
        text = encoded.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_int=parse_integer,
            parse_float=parse_floating,
        )
    except StrictJSONError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrictJSONError(f"invalid {context}: {error}", syntax_error=True) from error
    except (RecursionError, OverflowError, ValueError) as error:
        raise StrictJSONError(f"invalid {context}: {error}") from error

    json_node_count(
        value,
        max_depth=limits.max_depth,
        max_nodes=limits.max_nodes,
        context=context,
    )
    return value
