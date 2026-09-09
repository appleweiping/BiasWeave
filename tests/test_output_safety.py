from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import biasweave._output as output_module
import biasweave.ledger as ledger_module
from biasweave._output import WriterClaim, atomic_write_many, paths_alias
from biasweave._strict_json import JSONLimits, json_node_count
from biasweave.cli import main
from biasweave.demo import evaluate as demo_evaluator
from biasweave.dominance import failed_trial
from biasweave.engine import optimize
from biasweave.errors import CheckpointError
from biasweave.ledger import TrialLedger, write_metadata
from biasweave.model import RunConfig
from biasweave.problem import load_problem
from tests.helpers import evaluator
from tests.test_ledger_results import successful_trial

EXAMPLE = Path("examples/two_stage_ota/problem.toml")
CONTRACT = Path("benchmarks/manifest.json")


def test_atomic_output_is_race_safe_no_clobber(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "result.json"
    real_link = os.link

    def race(source: str | Path, target: str | Path, *args, **kwargs) -> None:
        Path(target).write_bytes(b"racer")
        real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(output_module.os, "link", race)
    with pytest.raises(CheckpointError, match="refusing to overwrite"):
        atomic_write_many(((destination, b"ours"),))
    assert destination.read_bytes() == b"racer"
    assert not tuple(tmp_path.glob("*.tmp"))


def test_forced_multi_output_failure_rolls_back_every_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_bytes(b"old-a")
    second.write_bytes(b"old-b")
    real_link = os.link

    def fail_second_install(source: str | Path, target: str | Path, *args, **kwargs) -> None:
        if Path(source).suffix == ".tmp" and Path(target) == second:
            raise OSError("injected install failure")
        real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(output_module.os, "link", fail_second_install)
    with pytest.raises(CheckpointError, match="injected install failure"):
        atomic_write_many(((first, b"new-a"), (second, b"new-b")), force=True)
    assert first.read_bytes() == b"old-a"
    assert second.read_bytes() == b"old-b"
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob(".*.bak"))


def test_no_clobber_rollback_never_removes_a_concurrent_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    real_link = os.link
    real_replace = os.replace
    calls = 0

    def race(source: str | Path, target: str | Path, *args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            real_link(source, target, *args, **kwargs)
            replacement = tmp_path / "attacker.tmp"
            replacement.write_bytes(b"concurrent-a")
            real_replace(replacement, first)
            return
        if calls == 2:
            Path(target).write_bytes(b"concurrent-b")
        real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(output_module.os, "link", race)
    with pytest.raises(CheckpointError, match="installed output identity changed"):
        atomic_write_many(((first, b"ours-a"), (second, b"ours-b")))
    assert first.read_bytes() == b"concurrent-a"
    assert not second.exists()


def test_forced_transaction_restores_old_files_after_base_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_bytes(b"old-a")
    second.write_bytes(b"old-b")
    real_replace = os.replace
    backup_calls = 0

    def interrupt_second_backup(source: str | Path, target: str | Path) -> None:
        nonlocal backup_calls
        if Path(target).suffix == ".bak":
            backup_calls += 1
            if backup_calls == 2:
                raise KeyboardInterrupt("injected cancellation")
        real_replace(source, target)

    monkeypatch.setattr(output_module.os, "replace", interrupt_second_backup)
    with pytest.raises(KeyboardInterrupt, match="injected cancellation"):
        atomic_write_many(((first, b"new-a"), (second, b"new-b")), force=True)
    assert first.read_bytes() == b"old-a"
    assert second.read_bytes() == b"old-b"
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob(".*.bak"))


@pytest.mark.parametrize("interrupt_phase", ["backup", "install"])
def test_forced_transaction_recovers_if_cancellation_arrives_after_filesystem_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, interrupt_phase: str
) -> None:
    target = tmp_path / "result.txt"
    target.write_bytes(b"old")
    real_replace = os.replace
    real_link = os.link

    def replace_then_interrupt(source: str | Path, destination: str | Path) -> None:
        destination_path = Path(destination)
        is_phase = interrupt_phase == "backup" and destination_path.suffix == ".bak"
        real_replace(source, destination)
        if is_phase:
            raise KeyboardInterrupt(f"after {interrupt_phase}")

    def link_then_interrupt(source: str | Path, destination: str | Path, *args, **kwargs) -> None:
        real_link(source, destination, *args, **kwargs)
        if (
            interrupt_phase == "install"
            and Path(source).suffix == ".tmp"
            and Path(destination) == target
        ):
            raise KeyboardInterrupt("after install")

    monkeypatch.setattr(output_module.os, "replace", replace_then_interrupt)
    monkeypatch.setattr(output_module.os, "link", link_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match=f"after {interrupt_phase}"):
        atomic_write_many(((target, b"new"),), force=True)
    assert target.read_bytes() == b"old"
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob(".*.bak"))


def test_transaction_cleans_staged_file_if_fsync_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_fsync = os.fsync
    calls = 0

    def interrupt_staging(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("staging interrupted")
        real_fsync(descriptor)

    monkeypatch.setattr(output_module.os, "fsync", interrupt_staging)
    with pytest.raises(KeyboardInterrupt, match="staging interrupted"):
        atomic_write_many(((tmp_path / "result.txt", b"new"),))
    assert not (tmp_path / "result.txt").exists()
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert not tuple(tmp_path.glob(".biasweave-writer-*"))


def test_writer_claim_cleans_its_own_partial_state_after_base_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claim = WriterClaim(tmp_path)

    def interrupt(_descriptor: int) -> None:
        raise KeyboardInterrupt("claim interrupted")

    monkeypatch.setattr(output_module.os, "fsync", interrupt)
    with pytest.raises(KeyboardInterrupt, match="claim interrupted"):
        claim.__enter__()
    assert not claim.directory.exists()


def test_writer_claim_never_removes_a_replaced_owner(tmp_path: Path) -> None:
    claim = WriterClaim(tmp_path)
    claim.__enter__()
    replacement = tmp_path / "replacement-owner"
    replacement.write_bytes(b"concurrent owner")
    os.replace(replacement, claim.owner)
    with pytest.raises(CheckpointError, match="ownership changed"):
        claim.__exit__(None, None, None)
    assert claim.owner.read_bytes() == b"concurrent owner"
    assert claim.directory.is_dir()


def test_writer_claim_finishes_known_empty_cleanup_after_release_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claim = WriterClaim(tmp_path)
    claim.__enter__()
    real_unlink = Path.unlink

    def unlink_then_interrupt(path: Path, *args, **kwargs) -> None:
        real_unlink(path, *args, **kwargs)
        if path == claim.owner:
            raise KeyboardInterrupt("after owner unlink")

    monkeypatch.setattr(Path, "unlink", unlink_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="after owner unlink"):
        claim.__exit__(None, None, None)
    assert not claim.directory.exists()


def test_writer_claim_excludes_a_second_writer_and_releases_cleanly(tmp_path: Path) -> None:
    first = WriterClaim(tmp_path, scope="same-output")
    with first:
        assert first.owns(tmp_path)
        with (
            pytest.raises(CheckpointError, match="another writer"),
            WriterClaim(tmp_path, scope="same-output"),
        ):
            raise AssertionError("unreachable")
    assert not first.directory.exists()


def test_run_writers_reject_a_live_claim_from_another_scope(tmp_path: Path) -> None:
    root = tmp_path / "run"
    ledger = TrialLedger(root / "trials.jsonl")
    with WriterClaim(root, scope="other") as claim:
        assert claim.owns(root)
        assert not claim.owns(root, scope="run")
        with pytest.raises(CheckpointError, match="run writer claim"):
            ledger.append((successful_trial(0),), _claim=claim)
        with pytest.raises(CheckpointError, match="run writer claim"):
            write_metadata(root / "run.json", {"schema_version": 1}, _claim=claim)
    assert not ledger.path.exists()


def test_force_detects_and_restores_a_replacement_displaced_during_backup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "result.txt"
    target.write_bytes(b"old")
    real_replace = os.replace
    raced = False

    def replace_with_race(source: str | Path, destination: str | Path) -> None:
        nonlocal raced
        if not raced and Path(source) == target and Path(destination).suffix == ".bak":
            raced = True
            replacement = tmp_path / "replacement.tmp"
            replacement.write_bytes(b"concurrent")
            real_replace(replacement, target)
        real_replace(source, destination)

    monkeypatch.setattr(output_module.os, "replace", replace_with_race)
    with pytest.raises(CheckpointError, match="identity changed while being backed up"):
        atomic_write_many(((target, b"ours"),), force=True)
    assert target.read_bytes() == b"concurrent"
    assert not tuple(tmp_path.glob(".*.bak"))


def test_post_commit_cancellation_is_rethrown_with_an_explicit_commit_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "result.txt"
    target.write_bytes(b"old")
    real_unlink = Path.unlink
    backup_unlinks = 0

    def unlink_then_interrupt(path: Path, *args, **kwargs) -> None:
        nonlocal backup_unlinks
        if path.suffix == ".bak":
            backup_unlinks += 1
            real_unlink(path, *args, **kwargs)
            if backup_unlinks == 2:
                raise KeyboardInterrupt("after committed backup deletion")
            return
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_then_interrupt)
    with pytest.raises(KeyboardInterrupt, match="after committed") as captured:
        atomic_write_many(((target, b"ours"),), force=True)
    assert target.read_bytes() == b"ours"
    assert not tuple(tmp_path.glob(".*.bak"))
    assert any("transaction committed" in note for note in captured.value.__notes__)


def test_input_aliases_remain_protected_even_with_force(tmp_path: Path) -> None:
    source = tmp_path / "Source.JSON"
    source.write_bytes(b"source")
    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    assert paths_alias(source, hardlink)
    with pytest.raises(CheckpointError, match="aliases an input"):
        atomic_write_many(((hardlink, b"replacement"),), force=True, protected=(source,))
    assert source.read_bytes() == b"source"

    with pytest.raises(CheckpointError, match="aliases an input"):
        atomic_write_many(
            ((tmp_path / "case.json", b"replacement"),),
            force=True,
            protected=(tmp_path / "CASE.JSON",),
        )


def test_symlink_input_alias_remains_protected_with_force(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(b"source")
    symlink = tmp_path / "symlink.json"
    try:
        symlink.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable on this platform: {error}")

    assert paths_alias(source, symlink)
    with pytest.raises(CheckpointError, match="aliases an input"):
        atomic_write_many(((symlink, b"replacement"),), force=True, protected=(source,))
    assert source.read_bytes() == b"source"


def test_weave_force_still_protects_problem_hardlink(tmp_path: Path) -> None:
    problem_path = tmp_path / "problem.toml"
    problem_path.write_bytes(EXAMPLE.read_bytes())
    before = problem_path.read_bytes()
    output = tmp_path / "run"
    output.mkdir()
    os.link(problem_path, output / "frontier.json")
    problem = load_problem(problem_path)
    with pytest.raises(CheckpointError, match="aliases an input"):
        optimize(
            problem,
            evaluator,
            evaluator_id="tests:protected",
            config=RunConfig(1),
            output_directory=output,
            force=True,
        )
    assert problem_path.read_bytes() == before
    assert not (output / "run.json").exists()


def test_weave_force_initializes_every_artifact_to_the_new_run_generation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    old = {
        "trials.jsonl": b"old-ledger\n",
        "run.json": b"old-run\n",
        "frontier.json": b"old-frontier\n",
        "summary.md": b"old-summary\n",
    }
    for name, payload in old.items():
        (output / name).write_bytes(payload)

    def interrupt(_point):
        raise KeyboardInterrupt("cancel new generation")

    with pytest.raises(KeyboardInterrupt, match="cancel new generation"):
        optimize(
            load_problem(EXAMPLE),
            interrupt,
            evaluator_id="tests:new-generation",
            config=RunConfig(2),
            output_directory=output,
            force=True,
        )

    assert (output / "trials.jsonl").read_bytes() == b""
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert metadata["evaluator_id"] == "tests:new-generation"
    assert len(metadata["pending_keys"]) == 2
    frontier = json.loads((output / "frontier.json").read_text(encoding="utf-8"))
    assert frontier["evaluator_id"] == "tests:new-generation"
    assert frontier["stop_reason"] == "in_progress"
    assert frontier["trial_count"] == 0
    assert "`in_progress`" in (output / "summary.md").read_text(encoding="utf-8")
    assert all((output / name).read_bytes() != payload for name, payload in old.items())


def test_ledger_append_rechecks_input_identity_after_open_race(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protected = tmp_path / "problem.toml"
    protected.write_bytes(b"do-not-change")
    ledger_path = tmp_path / "trials.jsonl"
    real_open = os.open

    def raced_open(path, flags, mode=0o777):
        os.link(protected, path)
        return real_open(path, flags, mode)

    monkeypatch.setattr(ledger_module.os, "open", raced_open)
    with pytest.raises(CheckpointError, match="aliases an input"):
        TrialLedger(ledger_path).append((successful_trial(0),), protected=(protected,))
    assert protected.read_bytes() == b"do-not-change"


def test_ledger_append_excludes_a_second_live_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = TrialLedger(tmp_path / "run" / "trials.jsonl")
    first_opened = threading.Event()
    release_first = threading.Event()
    real_protect = ledger_module.protect_open_descriptor
    call_count = 0
    call_guard = threading.Lock()

    def pause_first_writer(
        descriptor: int,
        target: str | Path,
        protected: tuple[str | Path, ...],
    ) -> None:
        nonlocal call_count
        real_protect(descriptor, target, protected)
        with call_guard:
            call_count += 1
            first = call_count == 1
        if first:
            first_opened.set()
            assert release_first.wait(5)

    monkeypatch.setattr(ledger_module, "protect_open_descriptor", pause_first_writer)

    def capture_append(trial_id: int) -> BaseException | None:
        try:
            ledger.append((successful_trial(trial_id),))
        except BaseException as error:  # The exact concurrent result is asserted below.
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(capture_append, 0)
        assert first_opened.wait(5)
        second = pool.submit(capture_append, 0)
        try:
            second_result = second.result(timeout=2)
        finally:
            release_first.set()
        assert first.result(timeout=5) is None

    assert isinstance(second_result, CheckpointError)
    assert "writer" in str(second_result).casefold()
    assert [trial.trial_id for trial in ledger.read()] == [0]


def test_cli_outputs_are_no_clobber_forceable_and_never_overwrite_inputs(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "contract.json"
    contract.write_bytes(CONTRACT.read_bytes())
    output = tmp_path / "comparison.json"
    output.write_bytes(b"sentinel")
    args = [
        "benchmark",
        "--contract",
        str(contract),
        "--budget",
        "4",
        "--output",
        str(output),
    ]
    assert main(args) == 2
    assert output.read_bytes() == b"sentinel"
    assert main([*args, "--force"]) == 0
    assert output.read_bytes() != b"sentinel"
    original = contract.read_bytes()
    assert main([*args[:-1], str(contract), "--force"]) == 2
    assert contract.read_bytes() == original


def test_quality_force_cannot_replace_a_ledger_input(tmp_path: Path) -> None:
    problem_path = tmp_path / "problem.toml"
    problem_path.write_bytes(EXAMPLE.read_bytes())
    output = tmp_path / "run"
    optimize(
        load_problem(problem_path),
        demo_evaluator,
        evaluator_id="tests:quality-alias",
        config=RunConfig(4),
        output_directory=output,
    )
    ledger = output / "trials.jsonl"
    before = ledger.read_bytes()
    assert (
        main(
            [
                "quality",
                "--problem",
                str(problem_path),
                "--ledger",
                str(ledger),
                "--output",
                str(ledger),
                "--force",
            ]
        )
        == 2
    )
    assert ledger.read_bytes() == before


def test_rejected_ledger_and_metadata_writes_preserve_old_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger_path = tmp_path / "trials.jsonl"
    ledger = TrialLedger(ledger_path)
    first = successful_trial(0)
    ledger.append((first,))
    before = ledger_path.read_bytes()

    too_many_metrics = {f"m{index}": 1.0 for index in range(4_097)}
    invalid = replace(successful_trial(1), metrics=too_many_metrics)
    with pytest.raises(CheckpointError, match="metric-count|complexity|too many object fields"):
        ledger.append((invalid,))
    assert ledger_path.read_bytes() == before

    huge_error = failed_trial(1, first.point, "x" * 65_537)
    with pytest.raises(CheckpointError, match="text longer"):
        ledger.append((huge_error,))
    assert ledger_path.read_bytes() == before

    metadata_path = tmp_path / "run.json"
    write_metadata(metadata_path, {"schema_version": 1, "state": "old"})
    metadata_before = metadata_path.read_bytes()
    monkeypatch.setattr(
        ledger_module,
        "_METADATA_JSON_LIMITS",
        JSONLimits(max_bytes=32, max_depth=64, max_nodes=10_000, max_number_characters=128),
    )
    with pytest.raises(CheckpointError, match="byte input limit"):
        write_metadata(
            metadata_path,
            {"schema_version": 1, "state": "x" * 100},
            force=True,
        )
    assert metadata_path.read_bytes() == metadata_before


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("_MAX_LEDGER_LINE_BYTES", 64, "line limit"),
        ("_MAX_LEDGER_RECORDS", 1, "record limit"),
        ("_MAX_LEDGER_LINES", 1, "physical line limit"),
    ],
)
def test_every_ledger_writer_envelope_rejects_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    limit_name: str,
    limit_value: int,
    message: str,
) -> None:
    path = tmp_path / f"{limit_name}.jsonl"
    ledger = TrialLedger(path)
    ledger.append((successful_trial(0),))
    before = path.read_bytes()
    monkeypatch.setattr(ledger_module, limit_name, limit_value)
    with pytest.raises(CheckpointError, match=message):
        ledger.append((successful_trial(1),))
    assert path.read_bytes() == before


def test_ledger_total_byte_and_node_envelopes_preserve_old_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    byte_path = tmp_path / "bytes.jsonl"
    byte_ledger = TrialLedger(byte_path)
    byte_ledger.append((successful_trial(0),))
    byte_before = byte_path.read_bytes()
    monkeypatch.setattr(ledger_module, "_MAX_LEDGER_BYTES", len(byte_before))
    with pytest.raises(CheckpointError, match="byte limit"):
        byte_ledger.append((successful_trial(1),))
    assert byte_path.read_bytes() == byte_before

    node_path = tmp_path / "nodes.jsonl"
    node_ledger = TrialLedger(node_path)
    first = successful_trial(0)
    node_ledger.append((first,))
    node_before = node_path.read_bytes()
    nodes = json_node_count(first.as_dict(), max_depth=64, max_nodes=10_000, context="test")
    monkeypatch.setattr(ledger_module, "_MAX_LEDGER_TOTAL_NODES", nodes)
    with pytest.raises(CheckpointError, match="total JSON node limit"):
        node_ledger.append((successful_trial(1),))
    assert node_path.read_bytes() == node_before


def test_metadata_field_and_text_envelopes_preserve_old_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "metadata.json"
    write_metadata(path, {"state": "old"})
    before = path.read_bytes()
    monkeypatch.setattr(ledger_module, "_MAX_FIELDS", 1)
    with pytest.raises(CheckpointError, match="too many object fields"):
        write_metadata(path, {"state": "new", "extra": 1}, force=True)
    assert path.read_bytes() == before
    monkeypatch.setattr(ledger_module, "_MAX_FIELDS", 4_096)
    monkeypatch.setattr(ledger_module, "_MAX_TOTAL_TEXT_CHARS", 8)
    with pytest.raises(CheckpointError, match="total text characters"):
        write_metadata(path, {"state": "too-long"}, force=True)
    assert path.read_bytes() == before


def test_append_repairs_only_an_incomplete_tail(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    first = successful_trial(0)
    path.write_text(json.dumps(first.as_dict()) + '\n{"trial_id":', encoding="utf-8")
    TrialLedger(path).append((successful_trial(1),))
    assert [trial.trial_id for trial in TrialLedger(path).read()] == [0, 1]
