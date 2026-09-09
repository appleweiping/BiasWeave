"""Deterministic optimization orchestration and checkpoint recovery."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from biasweave._failure import evaluator_failure
from biasweave._output import WriterClaim, atomic_write_many, preflight_outputs
from biasweave._version import __version__
from biasweave.archive import Archive
from biasweave.dominance import assess, failed_trial
from biasweave.encoding import make_point
from biasweave.errors import CheckpointError, ConfigurationError
from biasweave.evaluator import validate_metrics
from biasweave.ledger import TrialLedger, metadata_bytes, read_metadata, write_metadata
from biasweave.model import (
    OptimizationResult,
    Point,
    Problem,
    RunConfig,
    Scalar,
    Trial,
    TrialStatus,
)
from biasweave.proposal import ProposalGenerator
from biasweave.results import result_outputs, write_result

_SCHEMA_VERSION = 2
_STRATEGY_SCHEMA_VERSION = 1
_MAX_SEED_DIGITS = 128


def validate_run_config(config: RunConfig) -> None:
    """Reject settings that would make progress ambiguous or unsafe."""
    if isinstance(config.budget, bool) or not isinstance(config.budget, int) or config.budget <= 0:
        raise ConfigurationError("budget must be a positive integer")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise ConfigurationError("seed must be an integer")
    if len(str(abs(config.seed))) > _MAX_SEED_DIGITS:
        raise ConfigurationError(f"seed must be at most {_MAX_SEED_DIGITS} digits")
    if (
        isinstance(config.workers, bool)
        or not isinstance(config.workers, int)
        or config.workers <= 0
    ):
        raise ConfigurationError("workers must be a positive integer")
    if (
        isinstance(config.batch_size, bool)
        or not isinstance(config.batch_size, int)
        or config.batch_size <= 0
    ):
        raise ConfigurationError("batch_size must be a positive integer")
    if (
        isinstance(config.max_stagnation, bool)
        or not isinstance(config.max_stagnation, int)
        or config.max_stagnation < 0
    ):
        raise ConfigurationError("max_stagnation must be a non-negative integer")
    if config.wall_time_seconds is not None and (
        not math.isfinite(config.wall_time_seconds) or config.wall_time_seconds <= 0.0
    ):
        raise ConfigurationError("wall_time_seconds must be positive and finite")
    if not math.isfinite(config.command_timeout_seconds) or config.command_timeout_seconds <= 0.0:
        raise ConfigurationError("command_timeout_seconds must be positive and finite")


def problem_fingerprint(problem: Problem) -> str:
    """Return the source digest or a stable digest for an in-memory problem."""
    if problem.source_hash:
        return problem.source_hash
    data = {
        "variables": [asdict(variable) for variable in problem.variables],
        "objectives": [asdict(objective) for objective in problem.objectives],
        "constraints": [asdict(constraint) for constraint in problem.constraints],
    }
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata(
    problem_hash: str,
    evaluator_id: str,
    config: RunConfig,
    completed_trials: int,
    stagnation: int,
    generator: ProposalGenerator,
    batch_ends: list[int],
    pending_keys: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "package_version": __version__,
        "strategy": "weave",
        "strategy_schema_version": _STRATEGY_SCHEMA_VERSION,
        "strategy_parameters": {},
        "problem_sha256": problem_hash,
        "evaluator_id": evaluator_id,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "completed_trials": completed_trials,
        "stagnation": stagnation,
        "generator": generator.snapshot(),
        "batch_ends": batch_ends,
        "pending_keys": pending_keys or [],
    }


def _require_metadata(
    raw: Mapping[str, Any],
    *,
    problem_hash: str,
    evaluator_id: str,
    config: RunConfig,
    trial_count: int,
) -> tuple[int, int, Mapping[str, Any], list[int], list[str]]:
    expected_fields = {
        "schema_version",
        "package_version",
        "strategy",
        "strategy_schema_version",
        "strategy_parameters",
        "problem_sha256",
        "evaluator_id",
        "seed",
        "batch_size",
        "completed_trials",
        "stagnation",
        "generator",
        "batch_ends",
        "pending_keys",
    }
    if set(raw) != expected_fields:
        raise CheckpointError("run metadata fields are invalid")
    schema_version = raw.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _SCHEMA_VERSION
    ):
        raise CheckpointError("unsupported run metadata schema")
    expected = {
        "package_version": __version__,
        "strategy": "weave",
        "strategy_schema_version": _STRATEGY_SCHEMA_VERSION,
        "strategy_parameters": {},
        "problem_sha256": problem_hash,
        "evaluator_id": evaluator_id,
        "seed": config.seed,
        "batch_size": config.batch_size,
    }
    for field, value in expected.items():
        if raw.get(field) != value:
            raise CheckpointError(f"checkpoint {field} does not match this run")
    completed = raw.get("completed_trials")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise CheckpointError("checkpoint completed_trials counter is invalid")
    if completed > trial_count:
        raise CheckpointError("checkpoint metadata and trial ledger disagree")
    stagnation = raw.get("stagnation", 0)
    if isinstance(stagnation, bool) or not isinstance(stagnation, int) or stagnation < 0:
        raise CheckpointError("checkpoint stagnation counter is invalid")
    state = raw.get("generator")
    if not isinstance(state, Mapping):
        raise CheckpointError("checkpoint proposal-generator state is invalid")
    batch_ends_raw = raw.get("batch_ends")
    if (
        not isinstance(batch_ends_raw, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in batch_ends_raw)
        or batch_ends_raw != sorted(set(batch_ends_raw))
        or (batch_ends_raw and batch_ends_raw[-1] != completed)
        or any(item <= 0 for item in batch_ends_raw)
    ):
        raise CheckpointError("checkpoint batch boundaries disagree with completed trials")
    pending_raw = raw.get("pending_keys", [])
    if not isinstance(pending_raw, list) or not all(isinstance(item, str) for item in pending_raw):
        raise CheckpointError("checkpoint pending batch is invalid")
    return completed, stagnation, state, list(batch_ends_raw), list(pending_raw)


def _replay_ledger_tail(
    problem: Problem,
    trials: list[Trial],
    completed: int,
    batch_size: int,
    generator: ProposalGenerator,
    stagnation: int,
    proposed_count: int,
) -> tuple[int, list[Point], bool]:
    """Recover a durable ledger batch written just before its metadata snapshot."""
    archive = Archive(problem, trials[:completed])
    signature = archive.signature
    expected = generator.propose(proposed_count, archive, tuple(trials[:completed]), set())
    actual = trials[completed:]
    if len(actual) > len(expected) or [point.key for point in expected[: len(actual)]] != [
        trial.point.key for trial in actual
    ]:
        raise CheckpointError("checkpoint metadata and ledger disagree; tail cannot be replayed")
    for trial in actual:
        archive.add(trial)
    if len(actual) == len(expected):
        grew = archive.signature != signature
        generator.observe(grew)
        return (0 if grew else stagnation + len(actual)), [], False
    return stagnation, expected[len(actual) :], True


def _reconstruct_generator(
    problem: Problem,
    trials: list[Trial],
    completed: int,
    batch_ends: list[int],
    seed: int,
) -> tuple[ProposalGenerator, int]:
    generator = ProposalGenerator(problem, seed)
    archive = Archive(problem)
    stagnation = 0
    cursor = 0
    for end in batch_ends:
        count = end - cursor
        signature = archive.signature
        expected = generator.propose(count, archive, tuple(trials[:cursor]), set())
        actual = trials[cursor : cursor + count]
        if [point.key for point in expected] != [trial.point.key for trial in actual]:
            raise CheckpointError("checkpoint trial sequence is not reproducible")
        for trial in actual:
            archive.add(trial)
        grew = archive.signature != signature
        generator.observe(grew)
        stagnation = 0 if grew else stagnation + count
        cursor = end
    if cursor != completed:
        raise CheckpointError("checkpoint batch boundaries do not cover completed trials")
    return generator, stagnation


def _evaluate_one(
    problem: Problem,
    evaluator: Callable[[Mapping[str, Scalar]], Mapping[str, float]],
    trial_id: int,
    point: Point,
) -> Trial:
    try:
        raw = evaluator(dict(point.values))
        metrics = validate_metrics(problem, raw)
        return assess(problem, trial_id, point, metrics)
    except Exception as error:  # Evaluator boundaries must preserve the remaining run.
        return failed_trial(trial_id, point, evaluator_failure(error))


def _evaluate_batch(
    problem: Problem,
    evaluator: Callable[[Mapping[str, Scalar]], Mapping[str, float]],
    first_trial_id: int,
    points: list[Point],
    workers: int,
) -> list[Trial]:
    jobs = [(first_trial_id + offset, point) for offset, point in enumerate(points)]

    def run(job: tuple[int, Point]) -> Trial:
        trial_id, point = job
        return _evaluate_one(problem, evaluator, trial_id, point)

    if workers == 1 or len(jobs) == 1:
        return [run(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="biasweave") as executor:
        return list(executor.map(run, jobs))


def _checkpoint_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "trials.jsonl", directory / "run.json"


def validate_checkpoint_trials(problem: Problem, trials: list[Trial]) -> None:
    """Verify that restored records are coherent with the bound problem."""
    expected_names = {variable.name for variable in problem.variables}
    for trial in trials:
        if len(trial.point.coordinates) != len(problem.free_variables):
            raise CheckpointError(f"trial {trial.trial_id} has the wrong coordinate count")
        if set(trial.point.values) != expected_names:
            raise CheckpointError(f"trial {trial.trial_id} has incompatible variable values")
        try:
            decoded = make_point(problem, trial.point.coordinates)
        except Exception as error:
            raise CheckpointError(f"trial {trial.trial_id} cannot be decoded: {error}") from error
        if decoded.key != trial.point.key or decoded.values != trial.point.values:
            raise CheckpointError(f"trial {trial.trial_id} point identity is inconsistent")
        if trial.status is TrialStatus.FAILED:
            if not isinstance(trial.error, str) or not trial.error.strip():
                raise CheckpointError(f"failed trial {trial.trial_id} is not canonical")
            expected_failure = failed_trial(trial.trial_id, trial.point, trial.error)
            if trial != expected_failure:
                raise CheckpointError(f"failed trial {trial.trial_id} is not canonical")
            continue
        try:
            metrics = validate_metrics(problem, trial.metrics)
            expected = assess(problem, trial.trial_id, trial.point, metrics)
        except Exception as error:
            raise CheckpointError(f"trial {trial.trial_id} metrics are invalid: {error}") from error
        if trial != expected:
            raise CheckpointError(f"trial {trial.trial_id} assessment is not canonical")


def _optimize_coordinated(
    problem: Problem,
    evaluator: Callable[[Mapping[str, Scalar]], Mapping[str, float]],
    *,
    evaluator_id: str,
    config: RunConfig,
    output_directory: str | Path | None = None,
    resume: bool = False,
    clock: Callable[[], float] = time.monotonic,
    force: bool = False,
    protected_paths: Iterable[str | Path] = (),
    _claim: WriterClaim | None = None,
) -> OptimizationResult:
    """Search a problem and optionally persist an exactly resumable run.

    Evaluator exceptions become failed trials. SystemExit, KeyboardInterrupt, and
    other BaseException subclasses are deliberately not swallowed.
    """
    validate_run_config(config)
    if not isinstance(evaluator_id, str) or not evaluator_id.strip() or len(evaluator_id) > 4_096:
        raise ConfigurationError("evaluator_id must be a bounded non-empty string")
    problem_hash = problem_fingerprint(problem)
    generator = ProposalGenerator(problem, config.seed)
    trials: list[Trial] = []
    stagnation = 0
    ledger: TrialLedger | None = None
    metadata_path: Path | None = None
    pending_points: list[Point] = []
    pending_batch_start: int | None = None
    batch_ends: list[int] = []
    protected = tuple(protected_paths) + ((problem.source_path,) if problem.source_path else ())

    if resume and output_directory is None:
        raise ConfigurationError("resume requires an output directory")
    if output_directory is not None:
        directory = Path(output_directory)
        ledger_path, metadata_path = _checkpoint_paths(directory)
        ledger = TrialLedger(ledger_path)
        artifacts = (
            ledger_path,
            metadata_path,
            directory / "frontier.json",
            directory / "summary.md",
        )
        if resume:
            preflight_outputs(artifacts, force=True, protected=protected)
            if not metadata_path.is_file():
                raise CheckpointError(f"checkpoint metadata does not exist: {metadata_path}")
            trials = ledger.read()
            validate_checkpoint_trials(problem, trials)
            raw_metadata = read_metadata(metadata_path)
            completed, stagnation, generator_state, batch_ends, recorded_pending = (
                _require_metadata(
                    raw_metadata,
                    problem_hash=problem_hash,
                    evaluator_id=evaluator_id,
                    config=config,
                    trial_count=len(trials),
                )
            )
            generator, reconstructed_stagnation = _reconstruct_generator(
                problem, trials, completed, batch_ends, config.seed
            )
            if recorded_pending:
                prior_archive = Archive(problem, trials[:completed])
                proposed = generator.propose(
                    len(recorded_pending),
                    prior_archive,
                    tuple(trials[:completed]),
                    set(),
                )
                if [point.key for point in proposed] != recorded_pending:
                    raise CheckpointError("checkpoint pending batch is not reproducible")
            computed_state = json.dumps(generator.snapshot(), sort_keys=True)
            recorded_state = json.dumps(dict(generator_state), sort_keys=True)
            if computed_state != recorded_state or stagnation != reconstructed_stagnation:
                raise CheckpointError(
                    "checkpoint metadata and proposal-generator state disagree with the trial ledger"
                )
            if recorded_pending:
                if config.budget < completed + len(recorded_pending):
                    raise ConfigurationError(
                        "budget cannot be less than the committed pending batch"
                    )
                actual = trials[completed:]
                if len(actual) > len(proposed) or [trial.point.key for trial in actual] != [
                    point.key for point in proposed[: len(actual)]
                ]:
                    raise CheckpointError("checkpoint pending batch disagrees with ledger")
                incomplete = len(actual) < len(proposed)
                pending_points = proposed[len(actual) :]
                pending_batch_start = completed if incomplete else None
                if not incomplete:
                    before = prior_archive.signature
                    for trial in actual:
                        prior_archive.add(trial)
                    grew = prior_archive.signature != before
                    generator.observe(grew)
                    stagnation = 0 if grew else stagnation + len(actual)
                    batch_ends.append(len(trials))
                    write_metadata(
                        metadata_path,
                        _metadata(
                            problem_hash,
                            evaluator_id,
                            config,
                            len(trials),
                            stagnation,
                            generator,
                            batch_ends,
                        ),
                        force=True,
                        protected=protected,
                        _claim=_claim,
                    )
            elif completed < len(trials):
                proposed_count = min(config.batch_size, config.budget - completed)
                stagnation, pending_points, incomplete = _replay_ledger_tail(
                    problem,
                    trials,
                    completed,
                    config.batch_size,
                    generator,
                    stagnation,
                    proposed_count,
                )
                pending_batch_start = completed if incomplete else None
                if not incomplete:
                    batch_ends.append(len(trials))
                    write_metadata(
                        metadata_path,
                        _metadata(
                            problem_hash,
                            evaluator_id,
                            config,
                            len(trials),
                            stagnation,
                            generator,
                            batch_ends,
                        ),
                        force=True,
                        protected=protected,
                        _claim=_claim,
                    )
            if config.budget < len(trials):
                raise ConfigurationError("budget cannot be less than completed trials")
        else:
            preflight_outputs(artifacts, force=force, protected=protected)
            initial = _metadata(
                problem_hash, evaluator_id, config, 0, stagnation, generator, batch_ends
            )
            initial_results = result_outputs(
                OptimizationResult(
                    problem,
                    (),
                    (),
                    "in_progress",
                    evaluator_id,
                    config.seed,
                ),
                directory,
            )
            atomic_write_many(
                (
                    (ledger_path, b""),
                    (metadata_path, metadata_bytes(initial)),
                    *initial_results,
                ),
                force=force,
                protected=protected,
            )

    archive = Archive(problem, trials)
    start = clock()
    stop_reason = "budget"

    if pending_points:
        if pending_batch_start is None:
            raise CheckpointError("incomplete checkpoint batch has no start marker")
        before = Archive(problem, trials[:pending_batch_start]).signature
        batch = _evaluate_batch(problem, evaluator, len(trials), pending_points, config.workers)
        for trial in batch:
            trials.append(trial)
            archive.add(trial)
        grew = archive.signature != before
        generator.observe(grew)
        stagnation = 0 if grew else stagnation + len(trials) - pending_batch_start
        batch_ends.append(len(trials))
        if ledger is not None and metadata_path is not None:
            ledger.append(batch, protected=protected, _claim=_claim)
            write_metadata(
                metadata_path,
                _metadata(
                    problem_hash,
                    evaluator_id,
                    config,
                    len(trials),
                    stagnation,
                    generator,
                    batch_ends,
                ),
                force=True,
                protected=protected,
                _claim=_claim,
            )

    while len(trials) < config.budget:
        if config.wall_time_seconds is not None and clock() - start >= config.wall_time_seconds:
            stop_reason = "wall_time"
            break
        if config.max_stagnation and stagnation >= config.max_stagnation:
            stop_reason = "stagnation"
            break
        remaining = config.budget - len(trials)
        points = generator.propose(min(config.batch_size, remaining), archive, tuple(trials), set())
        if not points:
            stop_reason = generator.empty_reason or "proposal_stalled"
            break
        if metadata_path is not None:
            write_metadata(
                metadata_path,
                _metadata(
                    problem_hash,
                    evaluator_id,
                    config,
                    len(trials),
                    stagnation,
                    generator,
                    batch_ends,
                    [point.key for point in points],
                ),
                force=True,
                protected=protected,
                _claim=_claim,
            )
        signature = archive.signature
        batch = _evaluate_batch(problem, evaluator, len(trials), points, config.workers)
        for trial in batch:
            trials.append(trial)
            archive.add(trial)
        frontier_grew = archive.signature != signature
        generator.observe(frontier_grew)
        stagnation = 0 if frontier_grew else stagnation + len(batch)
        batch_ends.append(len(trials))
        if ledger is not None and metadata_path is not None:
            ledger.append(batch, protected=protected, _claim=_claim)
            write_metadata(
                metadata_path,
                _metadata(
                    problem_hash,
                    evaluator_id,
                    config,
                    len(trials),
                    stagnation,
                    generator,
                    batch_ends,
                ),
                force=True,
                protected=protected,
                _claim=_claim,
            )

    result = OptimizationResult(
        problem,
        tuple(trials),
        archive.frontier,
        stop_reason,
        evaluator_id,
        config.seed,
    )
    if output_directory is not None:
        write_result(
            result,
            output_directory,
            # A fresh persisted run installs an atomic in-progress generation
            # before evaluation starts. These two files are therefore always
            # an internal checkpoint transition at completion, even when the
            # public fresh-run preflight was no-clobber.
            force=True,
            protected=protected,
        )
    return result


def optimize(
    problem: Problem,
    evaluator: Callable[[Mapping[str, Scalar]], Mapping[str, float]],
    *,
    evaluator_id: str,
    config: RunConfig,
    output_directory: str | Path | None = None,
    resume: bool = False,
    clock: Callable[[], float] = time.monotonic,
    force: bool = False,
    protected_paths: Iterable[str | Path] = (),
) -> OptimizationResult:
    """Search a problem, coordinating every durable run transition."""

    if output_directory is None:
        return _optimize_coordinated(
            problem,
            evaluator,
            evaluator_id=evaluator_id,
            config=config,
            output_directory=None,
            resume=resume,
            clock=clock,
            force=force,
            protected_paths=protected_paths,
        )
    with WriterClaim(Path(output_directory)) as claim:
        return _optimize_coordinated(
            problem,
            evaluator,
            evaluator_id=evaluator_id,
            config=config,
            output_directory=output_directory,
            resume=resume,
            clock=clock,
            force=force,
            protected_paths=protected_paths,
            _claim=claim,
        )
