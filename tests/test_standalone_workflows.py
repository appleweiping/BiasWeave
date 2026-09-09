from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "workflow,script,arguments,expected",
    [
        ("dco.yml", "dco.py", [], 2),
        ("release.yml", "release_artifacts.py", ["--help"], 0),
        ("release.yml", "release_gate.py", [], 1),
    ],
)
def test_actual_standalone_workflow_command_is_import_isolated(
    workflow: str,
    script: str,
    arguments: list[str],
    expected: int,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    body = (root / ".github/workflows" / workflow).read_text(encoding="utf-8")
    relative = f"src/biasweave/{script}"
    workflow_site_isolated = f"python -I -S {relative}" in body
    isolated = workflow_site_isolated or f"python -I {relative}" in body
    isolation = ["-I"] if isolated else []
    # The trusted-base DCO job and downstream release jobs do not install this
    # package. Disable site-packages so an editable dev install cannot hide an
    # accidental project dependency in a supposedly standalone helper.
    command = [sys.executable, *isolation, "-S", str(root / relative), *arguments]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == expected, result.stderr
    assert "usage:" in result.stdout + result.stderr
    assert isolated, "standalone trusted helpers must not import sibling or user-site modules"
    if workflow == "dco.yml":
        assert workflow_site_isolated, "trusted DCO execution must also disable site-packages"
