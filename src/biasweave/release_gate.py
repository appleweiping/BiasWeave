"""Fail-closed distribution archive validation."""

from __future__ import annotations

import re
import stat
import sys
import tarfile
import unicodedata
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

_MAX_ENTRY_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_ENTRIES = 100_000
_MAX_PATH_BYTES = 1_024
_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}\Z")


class ReleaseGateError(ValueError):
    """A distribution archive violates the release contract."""


def _version(value: str) -> str:
    if _VERSION.fullmatch(value) is None:
        raise ReleaseGateError("release version is invalid")
    return value


def _path(raw: str, *, directory: bool) -> PurePosixPath:
    name = raw[:-1] if directory and raw.endswith("/") else raw
    if (
        not name
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or PureWindowsPath(name).drive
    ):
        raise ReleaseGateError("archive path is empty or non-portable")
    try:
        encoded = name.encode("utf-8")
    except UnicodeError as error:
        raise ReleaseGateError("archive path is not valid UTF-8 text") from error
    if len(encoded) > _MAX_PATH_BYTES:
        raise ReleaseGateError("archive path exceeds the supported length")
    if unicodedata.normalize("NFC", name) != name:
        raise ReleaseGateError("archive path is not Unicode-normalized")
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseGateError("archive path is absolute or traverses a parent")
    if path.as_posix() != name:
        raise ReleaseGateError("archive path is not canonical")
    return path


def _key(path: PurePosixPath) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", part).casefold() for part in path.parts)


def _record_path(
    path: PurePosixPath,
    *,
    directory: bool,
    observed: set[tuple[str, ...]],
    files: set[tuple[str, ...]],
    parents: set[tuple[str, ...]],
) -> None:
    folded = _key(path)
    if folded in observed:
        raise ReleaseGateError("archive contains duplicate or normalized-casefold names")
    for index in range(1, len(folded)):
        if folded[:index] in files:
            raise ReleaseGateError("archive contains a file/directory prefix collision")
        parents.add(folded[:index])
    if not directory and folded in parents:
        raise ReleaseGateError("archive contains a file/directory prefix collision")
    observed.add(folded)
    if not directory:
        files.add(folded)


def _tar_is_sparse(member: tarfile.TarInfo) -> bool:
    return bool(
        member.type == getattr(tarfile, "GNUTYPE_SPARSE", b"S")
        or getattr(member, "sparse", None) is not None
        or any(key.startswith("GNU.sparse") for key in member.pax_headers)
    )


def audit_sdist(archive: tarfile.TarFile, *, expected_root: str) -> frozenset[str]:
    """Validate an sdist and return canonical regular-file names below its root."""

    observed: set[tuple[str, ...]] = set()
    files: set[tuple[str, ...]] = set()
    parents: set[tuple[str, ...]] = set()
    roots: set[str] = set()
    result: set[str] = set()
    total = 0
    for count, member in enumerate(archive, start=1):
        if count > _MAX_ENTRIES:
            raise ReleaseGateError("source distribution contains too many entries")
        directory = member.isdir()
        path = _path(member.name, directory=directory)
        if _tar_is_sparse(member):
            raise ReleaseGateError("source distribution contains a sparse entry")
        if not directory and not member.isreg():
            raise ReleaseGateError("source distribution contains a link or special entry")
        if member.size < 0 or member.size > _MAX_ENTRY_BYTES:
            raise ReleaseGateError("source distribution entry size is outside the limit")
        total += member.size
        if total > _MAX_TOTAL_BYTES:
            raise ReleaseGateError("source distribution exceeds the total size limit")
        _record_path(
            path,
            directory=directory,
            observed=observed,
            files=files,
            parents=parents,
        )
        roots.add(path.parts[0])
        if len(path.parts) > 1 and not directory:
            result.add(PurePosixPath(*path.parts[1:]).as_posix())
    if roots != {expected_root}:
        raise ReleaseGateError("source distribution does not have the exact canonical root")
    return frozenset(result)


def _zip_kind(member: zipfile.ZipInfo) -> tuple[bool, bool]:
    directory = member.is_dir()
    unix_mode = member.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    if directory:
        return True, file_type in {0, stat.S_IFDIR}
    return False, file_type in {0, stat.S_IFREG}


def audit_wheel(archive: zipfile.ZipFile) -> frozenset[str]:
    """Validate a wheel ZIP and return its canonical regular-file names."""

    observed: set[tuple[str, ...]] = set()
    files: set[tuple[str, ...]] = set()
    parents: set[tuple[str, ...]] = set()
    result: set[str] = set()
    total = 0
    members: Iterable[zipfile.ZipInfo] = archive.infolist()
    for count, member in enumerate(members, start=1):
        if count > _MAX_ENTRIES:
            raise ReleaseGateError("wheel contains too many entries")
        directory, permitted_type = _zip_kind(member)
        path = _path(member.filename, directory=directory)
        if not permitted_type or member.flag_bits & 1:
            raise ReleaseGateError("wheel contains a link, special, or encrypted entry")
        if member.file_size < 0 or member.file_size > _MAX_ENTRY_BYTES:
            raise ReleaseGateError("wheel entry size is outside the limit")
        total += member.file_size
        if total > _MAX_TOTAL_BYTES:
            raise ReleaseGateError("wheel exceeds the total size limit")
        _record_path(
            path,
            directory=directory,
            observed=observed,
            files=files,
            parents=parents,
        )
        if not directory:
            result.add(path.as_posix())
    corrupt = archive.testzip()
    if corrupt is not None:
        raise ReleaseGateError(f"wheel member failed its CRC check: {corrupt}")
    return frozenset(result)


def validate_distributions(directory: str | Path, version: str) -> tuple[Path, Path]:
    """Require one exact wheel/sdist pair and audit every archive member."""

    release_version = _version(version)
    root = Path(directory)
    wheel = root / f"biasweave-{release_version}-py3-none-any.whl"
    sdist = root / f"biasweave-{release_version}.tar.gz"
    try:
        observed = {path.name for path in root.iterdir()}
    except OSError as error:
        raise ReleaseGateError(f"cannot inspect distribution directory: {error}") from error
    expected = {wheel.name, sdist.name}
    if observed != expected:
        raise ReleaseGateError(f"distribution directory must contain exactly {sorted(expected)}")
    for artifact in (wheel, sdist):
        if artifact.is_symlink() or not artifact.is_file():
            raise ReleaseGateError(f"distribution is not a regular file: {artifact.name}")

    try:
        with tarfile.open(sdist, "r:gz") as archive:
            source_names = audit_sdist(archive, expected_root=f"biasweave-{release_version}")
        with zipfile.ZipFile(wheel) as archive:
            wheel_names = audit_wheel(archive)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise ReleaseGateError(f"cannot inspect distribution archive: {error}") from error

    required_source = {
        ".github/allowed_signers",
        ".github/workflows/ci.yml",
        ".github/workflows/dco.yml",
        ".github/workflows/release.yml",
        "CITATION.cff",
        "benchmarks/dedup_scaling.py",
        "benchmarks/manifest.json",
        "benchmarks/sky130-common-source.json",
        "benchmarks/sky130-sizing-decision.json",
        "docs/optimizers.md",
        "docs/schemas/analog-sizing-benchmark-1.schema.json",
        "docs/schemas/sizing-decision-1.schema.json",
        "src/biasweave/dco.py",
        "src/biasweave/release_artifacts.py",
        "src/biasweave/release_gate.py",
        "tests/data/optimizer-golden-v2.json",
        "tests/test_dco.py",
        "tests/test_release_artifacts.py",
        "tests/test_release_gate.py",
        "tests/test_standalone_workflows.py",
        "uv.lock",
    }
    if missing := required_source - source_names:
        raise ReleaseGateError(f"source distribution is missing: {sorted(missing)}")
    required_wheel = {
        "biasweave/_version.py",
        "biasweave/dco.py",
        "biasweave/py.typed",
        "biasweave/release_artifacts.py",
        "biasweave/release_gate.py",
        "biasweave/schemas/catalog-run-v1.schema.json",
        "biasweave/schemas/weave-checkpoint-v2.schema.json",
    }
    if missing := required_wheel - wheel_names:
        raise ReleaseGateError(f"wheel is missing: {sorted(missing)}")
    return wheel, sdist


def main(argv: list[str] | None = None) -> int:
    """Run the distribution gate from the trusted workflow."""

    arguments = sys.argv[1:] if argv is None else argv
    try:
        if len(arguments) != 3:
            raise ReleaseGateError(
                "usage: python -m biasweave.release_gate distributions PATH VERSION"
            )
        command, path, version = arguments
        if command == "distributions":
            validate_distributions(path, version)
        else:
            raise ReleaseGateError(f"unknown release gate: {command}")
    except (OSError, ReleaseGateError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"validated {arguments[0]} release gate for BiasWeave {arguments[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
