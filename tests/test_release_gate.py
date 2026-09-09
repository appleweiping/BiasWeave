from __future__ import annotations

import io
import stat
import tarfile
import unicodedata
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

import biasweave.release_gate as release_gate_module
from biasweave.release_gate import (
    ReleaseGateError,
    audit_sdist,
    audit_wheel,
    main,
    validate_distributions,
)

VERSION = "0.4.0"
REQUIRED_SOURCE = {
    ".github/allowed_signers",
    ".github/workflows/ci.yml",
    ".github/workflows/dco.yml",
    ".github/workflows/release.yml",
    "CITATION.cff",
    "benchmarks/bayesian-acquisition-v1.json",
    "benchmarks/bayesian_acquisition.py",
    "benchmarks/dedup_scaling.py",
    "benchmarks/manifest.json",
    "benchmarks/sky130-common-source.json",
    "benchmarks/sky130-sizing-decision.json",
    "docs/optimizers.md",
    "docs/surrogate-model.md",
    "docs/schemas/analog-sizing-benchmark-1.schema.json",
    "docs/schemas/sizing-decision-1.schema.json",
    "src/biasweave/dco.py",
    "src/biasweave/optimizers/bayesian.py",
    "src/biasweave/release_artifacts.py",
    "src/biasweave/release_gate.py",
    "src/biasweave/surrogate.py",
    "tests/data/optimizer-golden-v2.json",
    "tests/test_bayesian.py",
    "tests/test_bayesian_benchmark.py",
    "tests/test_dco.py",
    "tests/test_release_artifacts.py",
    "tests/test_release_gate.py",
    "tests/test_standalone_workflows.py",
    "tests/test_surrogate.py",
    "tests/test_surrogate_oracle.py",
    "uv.lock",
}
REQUIRED_WHEEL = {
    "biasweave/_version.py",
    "biasweave/dco.py",
    "biasweave/optimizers/bayesian.py",
    "biasweave/py.typed",
    "biasweave/release_artifacts.py",
    "biasweave/release_gate.py",
    "biasweave/surrogate.py",
    "biasweave/schemas/catalog-run-v1.schema.json",
    "biasweave/schemas/weave-checkpoint-v2.schema.json",
}


def make_tar(entries: Sequence[tuple[str, bytes, bytes | None]]) -> tarfile.TarFile:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, content, type_code in entries:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            if type_code is not None:
                member.type = type_code
                if type_code in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                    member.linkname = "target"
            archive.addfile(member, io.BytesIO(content) if member.isreg() else None)
    payload.seek(0)
    return tarfile.open(fileobj=payload, mode="r:gz")


def make_zip(entries: Sequence[tuple[str, bytes, int | None]]) -> zipfile.ZipFile:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w") as archive:
        for name, content, mode in entries:
            member = zipfile.ZipInfo(name)
            if mode is not None:
                member.create_system = 3
                member.external_attr = mode << 16
            archive.writestr(member, content)
    payload.seek(0)
    result = zipfile.ZipFile(payload)
    # ZipInfo normalizes host separators on Windows; restore the parsed name
    # so the auditor sees the hostile archive metadata a Linux runner sees.
    for member, (name, _content, _mode) in zip(result.infolist(), entries, strict=True):
        member.filename = name
    return result


def write_distribution_pair(
    directory: Path,
    *,
    source_names: set[str] | None = None,
    wheel_names: set[str] | None = None,
) -> None:
    root = f"biasweave-{VERSION}"
    selected_source = REQUIRED_SOURCE if source_names is None else source_names
    selected_wheel = REQUIRED_WHEEL if wheel_names is None else wheel_names
    with tarfile.open(directory / f"{root}.tar.gz", mode="w:gz") as archive:
        for name in sorted(selected_source):
            content = f"source:{name}".encode()
            member = tarfile.TarInfo(f"{root}/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    with zipfile.ZipFile(directory / f"biasweave-{VERSION}-py3-none-any.whl", mode="w") as archive:
        for name in sorted(selected_wheel):
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(member, f"wheel:{name}".encode())


def test_distribution_auditors_accept_only_canonical_regular_files() -> None:
    with make_tar(
        [
            ("biasweave-0.4.0/", b"", tarfile.DIRTYPE),
            ("biasweave-0.4.0/src/", b"", tarfile.DIRTYPE),
            ("biasweave-0.4.0/src/module.py", b"value = 1\n", None),
        ]
    ) as archive:
        assert audit_sdist(archive, expected_root="biasweave-0.4.0") == {"src/module.py"}
    with make_zip(
        [
            ("biasweave/", b"", stat.S_IFDIR | 0o755),
            ("biasweave/module.py", b"value = 1\n", stat.S_IFREG | 0o644),
        ]
    ) as archive:
        assert audit_wheel(archive) == {"biasweave/module.py"}


def test_distribution_pair_is_exact_and_contains_release_contract(tmp_path: Path) -> None:
    write_distribution_pair(tmp_path)
    wheel, sdist = validate_distributions(tmp_path, VERSION)
    assert wheel.name == "biasweave-0.4.0-py3-none-any.whl"
    assert sdist.name == "biasweave-0.4.0.tar.gz"
    (tmp_path / "extra").write_bytes(b"x")
    with pytest.raises(ReleaseGateError, match="exactly"):
        validate_distributions(tmp_path, VERSION)


def test_distribution_pair_rejects_missing_required_members(tmp_path: Path) -> None:
    write_distribution_pair(
        tmp_path,
        source_names=REQUIRED_SOURCE - {"src/biasweave/dco.py"},
    )
    with pytest.raises(ReleaseGateError, match="source distribution is missing"):
        validate_distributions(tmp_path, VERSION)
    for path in tmp_path.iterdir():
        path.unlink()
    write_distribution_pair(
        tmp_path,
        wheel_names=REQUIRED_WHEEL - {"biasweave/dco.py"},
    )
    with pytest.raises(ReleaseGateError, match="wheel is missing"):
        validate_distributions(tmp_path, VERSION)


def test_distribution_pair_rejects_invalid_version_directory_and_archive(tmp_path: Path) -> None:
    with pytest.raises(ReleaseGateError, match="version"):
        validate_distributions(tmp_path, "../bad")
    with pytest.raises(ReleaseGateError, match="inspect distribution directory"):
        validate_distributions(tmp_path / "missing", VERSION)
    write_distribution_pair(tmp_path)
    (tmp_path / f"biasweave-{VERSION}.tar.gz").write_bytes(b"not a tar")
    with pytest.raises(ReleaseGateError, match="cannot inspect distribution archive"):
        validate_distributions(tmp_path, VERSION)


@pytest.mark.parametrize(
    "name",
    [
        "/absolute.py",
        "../escape.py",
        "root/../escape.py",
        "C:/drive.py",
        "root\\windows.py",
        "root/control\x1f.py",
        f"root/{unicodedata.normalize('NFD', 'café')}.py",
    ],
)
def test_archive_auditors_reject_nonportable_paths(name: str) -> None:
    with (
        make_tar([(name, b"x", None)]) as archive,
        pytest.raises(ReleaseGateError, match="path|root"),
    ):
        audit_sdist(archive, expected_root="root")
    with (
        make_zip([(name, b"x", stat.S_IFREG | 0o644)]) as archive,
        pytest.raises(ReleaseGateError, match="path"),
    ):
        audit_wheel(archive)


@pytest.mark.parametrize(
    "entries",
    [
        [("root/A.py", b"a", None), ("root/a.py", b"b", None)],
        [("root/pkg", b"file", None), ("root/pkg/module.py", b"x", None)],
        [("root/a.py", b"a", None), ("root/a.py", b"b", None)],
    ],
)
def test_sdist_rejects_duplicate_casefold_and_prefix_collisions(entries) -> None:
    with make_tar(entries) as archive, pytest.raises(ReleaseGateError, match="duplicate|prefix"):
        audit_sdist(archive, expected_root="root")


def test_archive_auditors_reject_reverse_prefix_collision_and_wrong_root() -> None:
    with (
        make_tar([("root/pkg/a.py", b"a", None), ("root/pkg", b"file", None)]) as archive,
        pytest.raises(ReleaseGateError, match="prefix"),
    ):
        audit_sdist(archive, expected_root="root")
    with (
        make_tar([("other/a.py", b"a", None)]) as archive,
        pytest.raises(ReleaseGateError, match="canonical root"),
    ):
        audit_sdist(archive, expected_root="root")


def test_archive_resource_limits_and_crc_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_gate_module, "_MAX_ENTRIES", 1)
    with (
        make_tar([("root/a", b"a", None), ("root/b", b"b", None)]) as archive,
        pytest.raises(ReleaseGateError, match="too many entries"),
    ):
        audit_sdist(archive, expected_root="root")
    with (
        make_zip(
            [
                ("root/a", b"a", stat.S_IFREG | 0o644),
                ("root/b", b"b", stat.S_IFREG | 0o644),
            ]
        ) as archive,
        pytest.raises(ReleaseGateError, match="too many entries"),
    ):
        audit_wheel(archive)

    monkeypatch.setattr(release_gate_module, "_MAX_ENTRIES", 100_000)
    monkeypatch.setattr(release_gate_module, "_MAX_ENTRY_BYTES", 0)
    with (
        make_tar([("root/a", b"a", None)]) as archive,
        pytest.raises(ReleaseGateError, match="entry size"),
    ):
        audit_sdist(archive, expected_root="root")
    with (
        make_zip([("root/a", b"a", stat.S_IFREG | 0o644)]) as archive,
        pytest.raises(ReleaseGateError, match="entry size"),
    ):
        audit_wheel(archive)

    monkeypatch.setattr(release_gate_module, "_MAX_ENTRY_BYTES", 64)
    monkeypatch.setattr(release_gate_module, "_MAX_TOTAL_BYTES", 1)
    with (
        make_tar([("root/a", b"aa", None)]) as archive,
        pytest.raises(ReleaseGateError, match="total size"),
    ):
        audit_sdist(archive, expected_root="root")
    with (
        make_zip([("root/a", b"aa", stat.S_IFREG | 0o644)]) as archive,
        pytest.raises(ReleaseGateError, match="total size"),
    ):
        audit_wheel(archive)

    monkeypatch.setattr(release_gate_module, "_MAX_TOTAL_BYTES", 256)
    with (
        make_zip([("root/a", b"a", stat.S_IFREG | 0o644)]) as archive,
        pytest.raises(ReleaseGateError, match="CRC"),
    ):
        monkeypatch.setattr(archive, "testzip", lambda: "root/a")
        audit_wheel(archive)


def test_archive_path_rejects_surrogate_oversize_and_noncanonical_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ReleaseGateError, match="UTF-8"):
        release_gate_module._path("root/\ud800", directory=False)
    monkeypatch.setattr(release_gate_module, "_MAX_PATH_BYTES", 3)
    with pytest.raises(ReleaseGateError, match="length"):
        release_gate_module._path("root/a", directory=False)
    monkeypatch.setattr(release_gate_module, "_MAX_PATH_BYTES", 1_024)
    with pytest.raises(ReleaseGateError, match="canonical"):
        release_gate_module._path("root//a", directory=False)


@pytest.mark.parametrize(
    "type_code",
    [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.GNUTYPE_SPARSE],
)
def test_sdist_rejects_links_special_and_sparse_entries(type_code: bytes) -> None:
    with (
        make_tar([("root/hostile", b"", type_code)]) as archive,
        pytest.raises(ReleaseGateError, match="link|special|sparse"),
    ):
        audit_sdist(archive, expected_root="root")


def test_wheel_rejects_symlink_duplicate_prefix_and_encryption() -> None:
    with (
        make_zip([("biasweave/link", b"target", stat.S_IFLNK | 0o777)]) as archive,
        pytest.raises(ReleaseGateError, match="link|special"),
    ):
        audit_wheel(archive)
    with (
        make_zip(
            [
                ("biasweave/A.py", b"a", stat.S_IFREG | 0o644),
                ("biasweave/a.py", b"b", stat.S_IFREG | 0o644),
            ]
        ) as archive,
        pytest.raises(ReleaseGateError, match="duplicate"),
    ):
        audit_wheel(archive)
    with (
        make_zip(
            [
                ("biasweave/pkg", b"file", stat.S_IFREG | 0o644),
                ("biasweave/pkg/a.py", b"a", stat.S_IFREG | 0o644),
            ]
        ) as archive,
        pytest.raises(ReleaseGateError, match="prefix"),
    ):
        audit_wheel(archive)
    with make_zip([("biasweave/a.py", b"a", stat.S_IFREG | 0o644)]) as archive:
        archive.infolist()[0].flag_bits |= 1
        with pytest.raises(ReleaseGateError, match="encrypted"):
            audit_wheel(archive)


def test_release_gate_cli_rejects_unknown_gate(capsys) -> None:
    assert main(["unknown", ".", VERSION]) == 1
    assert "unknown release gate" in capsys.readouterr().err
    assert main([]) == 1
    assert "usage" in capsys.readouterr().err
