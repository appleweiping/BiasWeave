"""Small, total formatting boundary for untrusted evaluator exceptions."""

from __future__ import annotations

_MAX_EVALUATOR_ERROR_CHARS = 512
_MAX_EXCEPTION_NAME_CHARS = 96


def evaluator_failure(error: Exception) -> str:
    """Return a bounded single-line error even when ``str`` and ``repr`` fail."""

    try:
        detail = str(error)
    except BaseException:
        try:
            detail = repr(error)
        except BaseException:
            detail = "<unprintable exception>"
    # Bound hostile exception strings before normalization, which otherwise
    # could allocate another unbounded list and string through split/join.
    detail = detail[: _MAX_EVALUATOR_ERROR_CHARS * 2]
    # A surrogate cannot be emitted as UTF-8 by the ledger writer. Replacement
    # also makes terminal control characters visible as spaces, not new records.
    detail = detail.encode("utf-8", errors="replace").decode("utf-8")
    detail = " ".join(detail.split())
    try:
        raw_name = type(error).__name__
        name = raw_name if isinstance(raw_name, str) and raw_name else "Exception"
    except BaseException:
        name = "Exception"
    name = " ".join(name[:_MAX_EXCEPTION_NAME_CHARS].split()) or "Exception"
    prefix = f"{name}: " if detail else name
    message = prefix + detail if detail else prefix
    return message[:_MAX_EVALUATOR_ERROR_CHARS]
