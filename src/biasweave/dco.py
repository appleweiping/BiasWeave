"""Strict author-matching Developer Certificate of Origin verification."""

from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

_SIGNOFF = re.compile(
    r"^Signed-off-by:\s*(?P<name>[^<>\r\n]+?)\s*<(?P<email>[^<>\s]+)>\s*$",
    re.IGNORECASE,
)
_TRAILER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*:\s+\S.*$")
_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
_MAX_COMMITS = 250
_MAX_TEXT_CHARACTERS = 262_144


class DCOError(ValueError):
    """Pull-request commit metadata violates the DCO gate."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DCOError(f"pull-request commit metadata contains duplicate key: {key}")
        result[key] = value
    return result


def _constant(token: str) -> NoReturn:
    raise DCOError(f"pull-request commit metadata contains non-finite number: {token}")


def _integer(token: str) -> int:
    if len(token) > 128:
        raise DCOError("pull-request commit metadata number length exceeds 128 characters")
    return int(token)


def _floating(token: str) -> float:
    if len(token) > 128:
        raise DCOError("pull-request commit metadata number length exceeds 128 characters")
    value = float(token)
    if not math.isfinite(value):
        return _constant(token)
    return value


def _complexity(value: Any) -> None:
    # Keep auxiliary state proportional to depth, not the widest API array.
    frames: list[tuple[Iterator[Any], int]] = [(iter((value,)), 0)]
    count = 0
    while frames:
        items, depth = frames[-1]
        try:
            item = next(items)
        except StopIteration:
            frames.pop()
            continue
        count += 1
        if depth > 64 or count > 250_000:
            raise DCOError("pull-request commit metadata exceeds JSON complexity limits")
        if isinstance(item, dict):
            frames.append((iter(item.values()), depth + 1))
        elif isinstance(item, list):
            frames.append((iter(item), depth + 1))


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TEXT_CHARACTERS:
        raise DCOError(f"GitHub returned an invalid {label}")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise DCOError(f"GitHub returned an invalid {label}") from error
    return value


def _canonical_author(author: dict[str, Any]) -> tuple[str, str]:
    name = unicodedata.normalize("NFC", _text(author.get("name"), "author name").strip())
    email = _text(author.get("email"), "author email").strip().casefold()
    if any(ord(character) < 32 or ord(character) == 127 for character in name + email):
        raise DCOError("GitHub returned an invalid commit author")
    return name, email


def verify_commit_pages(
    pages: Any,
    *,
    expected_count: int | None = None,
    expected_head: str | None = None,
) -> int:
    """Verify immutable GitHub REST pull-commit pages and return their count."""

    if (
        not isinstance(pages, list)
        or not pages
        or any(not isinstance(page, list) for page in pages)
    ):
        raise DCOError("GitHub returned invalid pull-request commit pages")
    commits = [commit for page in pages for commit in page]
    if not commits:
        raise DCOError("the pull request has no commits")
    if len(commits) > _MAX_COMMITS:
        raise DCOError(f"pull request exceeds the {_MAX_COMMITS}-commit verification limit")
    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
        or expected_count > _MAX_COMMITS
    ):
        raise DCOError("the expected pull-request commit count is invalid")
    if expected_count is not None and len(commits) != expected_count:
        raise DCOError(
            f"GitHub returned {len(commits)} commits but the pull request declares {expected_count}"
        )

    failures: list[str] = []
    seen: set[str] = set()
    last_sha = ""
    for commit in commits:
        if not isinstance(commit, dict):
            raise DCOError("GitHub returned a malformed commit record")
        details = commit.get("commit")
        if not isinstance(details, dict):
            raise DCOError("GitHub returned a malformed commit record")
        author = details.get("author")
        if not isinstance(author, dict):
            raise DCOError("GitHub returned a malformed commit author")
        sha = _text(commit.get("sha"), "commit SHA")
        if _SHA.fullmatch(sha) is None:
            raise DCOError("GitHub returned an invalid commit SHA")
        if sha in seen:
            raise DCOError(f"GitHub returned duplicate commit metadata: {sha}")
        seen.add(sha)
        last_sha = sha
        name, email = _canonical_author(author)
        message = _text(details.get("message"), "commit message").rstrip()
        final_paragraph = message.rsplit("\n\n", maxsplit=1)[-1]
        trailer_lines = final_paragraph.splitlines()
        trailers = []
        if trailer_lines and all(_TRAILER.fullmatch(line) for line in trailer_lines):
            trailers = [
                match for line in trailer_lines if (match := _SIGNOFF.fullmatch(line)) is not None
            ]
        if not any(
            unicodedata.normalize("NFC", trailer["name"].strip()) == name
            and trailer["email"].casefold() == email
            for trailer in trailers
        ):
            failures.append(f"{sha[:12]} expected Signed-off-by: {name} <{email}>")
    if expected_head is not None and (
        _SHA.fullmatch(expected_head) is None or last_sha != expected_head
    ):
        raise DCOError("the commit list does not end at the immutable pull-request head")
    if failures:
        raise DCOError("DCO sign-off check failed:\n" + "\n".join(failures))
    return len(commits)


def verify_commit_file(
    path: str | Path,
    *,
    expected_count: int,
    expected_head: str,
) -> int:
    """Strictly load one bounded API response and verify its immutable anchors."""

    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise DCOError("pull-request commit metadata is not a regular file")
        with source.open("rb") as stream:
            payload = stream.read(_MAX_PAYLOAD_BYTES + 1)
        if len(payload) > _MAX_PAYLOAD_BYTES:
            raise DCOError("pull-request commit metadata exceeds the byte input limit")
        pages = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_int=_integer,
            parse_float=_floating,
        )
        _complexity(pages)
    except DCOError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError, OverflowError) as error:
        raise DCOError(f"cannot read pull-request commit metadata: {error}") from error
    return verify_commit_pages(
        pages,
        expected_count=expected_count,
        expected_head=expected_head,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI used exclusively by the trusted ``pull_request_target`` workflow."""

    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        print(
            "usage: python -I dco.py PULL_COMMITS.json EXPECTED_COUNT EXPECTED_HEAD",
            file=sys.stderr,
        )
        return 2
    try:
        if re.fullmatch(r"[1-9][0-9]{0,2}", arguments[1]) is None:
            raise DCOError("expected commit count is invalid")
        count = verify_commit_file(
            arguments[0],
            expected_count=int(arguments[1]),
            expected_head=arguments[2],
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"verified author-matching DCO sign-offs on {count} commit(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
