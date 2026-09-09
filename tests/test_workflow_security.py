from __future__ import annotations

import re
from pathlib import Path


def test_every_github_action_is_pinned_to_a_full_commit() -> None:
    workflow_root = Path(".github/workflows")
    references: list[tuple[Path, int, str]] = []
    for workflow in sorted(workflow_root.glob("*.yml")):
        for line_number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("uses:"):
                references.append((workflow, line_number, stripped.partition("uses:")[2].strip()))
    assert references
    for workflow, line_number, reference in references:
        assert re.match(r"[^\s@]+@[0-9a-f]{40}(?:\s+#.*)?\Z", reference), (
            workflow,
            line_number,
            reference,
        )


def test_source_distribution_contains_its_frozen_self_test_contract() -> None:
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")
    assert "include uv.lock" in manifest
    assert "recursive-include .github/workflows *.yml" in manifest
    assert "recursive-include tests *.py *.json" in manifest


def test_dco_runs_only_the_trusted_base_verifier() -> None:
    dco = Path(".github/workflows/dco.yml").read_text(encoding="utf-8")
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pull_request_target:" in dco
    assert "types: [opened, reopened, synchronize, edited]" in dco
    assert "cancel-in-progress: true" in dco
    assert "ref: ${{ github.event.pull_request.base.sha }}" in dco
    assert "repository: ${{ github.event.pull_request.base.repo.full_name }}" in dco
    assert "BASE_SHA: ${{ github.event.pull_request.base.sha }}" in dco
    assert "BASE_REF: ${{ github.event.pull_request.base.ref }}" in dco
    assert "BASE_REPOSITORY: ${{ github.event.pull_request.base.repo.full_name }}" in dco
    assert dco.count(".base.sha, .base.ref, .base.repo.full_name") == 3
    assert "state=pending" in dco
    assert "pull-commits.json" in dco
    assert "python -I src/biasweave/dco.py" in dco
    assert "PYTHONPATH:" not in dco
    assert "ref: ${{ github.event.pull_request.head.sha }}" not in dco
    assert "git rev-list" not in ci
    assert "Signed-off-by:" not in ci


def test_release_authenticates_source_before_any_source_execution() -> None:
    release = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    signature = release.index("verify-tag --raw")
    first_python = release.index("Set up exact Python")
    first_uv = release.index("Set up checksum-verified uv")
    quality = release.index("Run release quality gate")
    assert signature < first_python < first_uv < quality
    commit_signature = release.index("Require a verified signature on the exact source commit")
    assert signature < commit_signature < first_python
    assert "repos/$GITHUB_REPOSITORY/commits/$GITHUB_SHA" in release
    assert ".commit.verification.verified" in release
    assert 'if [ "$verified" != true ]; then' in release
    assert 'python-version: "3.11.16"' in release
    assert "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9" in release
    assert "checksum: 8c88519b0ef0af9801fcdee419bbb12116bd9e6b18e162ae093c932d8b264050" in release
    assert 'RELEASE_SYFT_VERSION: "1.51.1"' in release
    assert "syft-version: v${{ env.RELEASE_SYFT_VERSION }}" in release
    assert "refs/tags/v*) ;;" in release
    assert r".event == \"push\"" in release
    assert r".head_branch == \"main\"" in release
    assert r".head_sha == \"$GITHUB_SHA\"" in release
    assert "path: .release-smoke" in release
    assert "SYFT_FILE_METADATA_SELECTION: all" in release
    assert "output-file: dist/SBOM.spdx.json" in release
    assert "upload-artifact: false" in release
    assert "upload-release-assets: false" in release
    assert "bind-installed-wheel" in release
    assert '--expected-external-path "bin/biasweave"' in release
    assert release.count("release_artifacts.py verify-release") == 3
    assert release.count("ref: ${{ github.sha }}") == 3
    assert "subject-checksums: dist/SHA256SUMS" in release
    assert ".sbom-root" not in release
    assert "biasweave.spdx.json" not in release
    assert "subject-path: dist/*" not in release
    assert "sha256sum --check" not in release
    assert "python -m biasweave.release_artifacts" not in release
    assert 'archive.extractall(destination, filter="data")' in release
    assert 'cd "$source_root/biasweave-$RELEASE_PROJECT_VERSION"' in release
    assert release.count("uv sync --frozen --extra dev") == 2
    assert release.count("uv run --frozen pytest") == 2
    assert "dist/*\n          if-no-files-found" not in release
