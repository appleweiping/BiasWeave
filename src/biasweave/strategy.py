"""Budget-exact execution for the optimizer catalog."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from biasweave._failure import evaluator_failure
from biasweave._output import WriterClaim, atomic_write_many, preflight_outputs
from biasweave._version import __version__
from biasweave.archive import Archive
from biasweave.dominance import assess, failed_trial
from biasweave.engine import problem_fingerprint, validate_run_config
from biasweave.errors import ConfigurationError
from biasweave.evaluator import validate_metrics
from biasweave.ledger import ledger_bytes, metadata_bytes
from biasweave.model import OptimizationResult, Point, Problem, RunConfig, Scalar, Trial
from biasweave.optimizers.base import canonical_trial
from biasweave.optimizers.catalog import (
    StrategyName,
    create_optimizer,
    optimizer_parameters,
    parse_strategy,
)
from biasweave.results import result_outputs

_STRATEGY_SCHEMA_VERSION = 1


def _evaluate_one(
    problem: Problem,
    evaluator: Callable[[Mapping[str, Scalar]], Mapping[str, float]],
    trial_id: int,
    point: Point,
) -> Trial:
    try:
        metrics = validate_metrics(problem, evaluator(dict(point.values)))
        return assess(problem, trial_id, point, metrics)
    except Exception as error:  # Keep evaluator failures local to their trial.
        return failed_trial(trial_id, point, evaluator_failure(error))


def _evaluate_batch(
    problem: Problem,
    evaluator: Callable[[Mapping[str, Scalar]], Mapping[str, float]],
    first_trial_id: int,
    points: tuple[Point, ...],
    workers: int,
) -> tuple[Trial, ...]:
    jobs = tuple((first_trial_id + offset, point) for offset, point in enumerate(points))

    def run(job: tuple[int, Point]) -> Trial:
        return _evaluate_one(problem, evaluator, job[0], job[1])

    if workers == 1 or len(jobs) == 1:
        return tuple(run(job) for job in jobs)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="biasweave") as executor:
        return tuple(executor.map(run, jobs))


def _prepare_output(
    directory: Path,
    *,
    force: bool,
    protected: Iterable[str | Path],
) -> tuple[Path, ...]:
    artifacts = tuple(
        directory / name
        for name in ("trials.jsonl", "catalog-run.json", "frontier.json", "summary.md")
    )
    return preflight_outputs(artifacts, force=force, protected=protected)


def _optimize_strategy_coordinated(
    problem: Problem,
    evaluator: Callable[[Mapping[str, Scalar]], Mapping[str, float]],
    *,
    evaluator_id: str,
    config: RunConfig,
    strategy: StrategyName | str,
    population_size: int | None = None,
    output_directory: str | Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    force: bool = False,
    protected_paths: Iterable[str | Path] = (),
) -> OptimizationResult:
    """Run one catalog optimizer with exact budgeting and deterministic scheduling.

    All optimizers consume the same normalized mixed-variable representation and
    constraint assessment. Results are told in proposal order, so changing worker
    count cannot change a seeded strategy's state trajectory.
    """
    validate_run_config(config)
    if not isinstance(evaluator_id, str) or not evaluator_id.strip() or len(evaluator_id) > 4_096:
        raise ConfigurationError("evaluator_id must be a bounded non-empty string")
    selected = parse_strategy(strategy)
    population_strategies = {
        StrategyName.PSO,
        StrategyName.DE,
        StrategyName.NSGA2,
        StrategyName.MOEAD,
    }
    effective_population = (
        (population_size if population_size is not None else 16)
        if selected in population_strategies
        else None
    )
    optimizer = create_optimizer(
        problem,
        selected,
        seed=config.seed,
        population_size=population_size,
    )
    parameters = optimizer_parameters(selected, optimizer)
    directory: Path | None = None
    artifact_paths: tuple[Path, ...] = ()
    protected = tuple(protected_paths) + ((problem.source_path,) if problem.source_path else ())
    if output_directory is not None:
        directory = Path(output_directory)
        artifact_paths = _prepare_output(directory, force=force, protected=protected)

    trials: list[Trial] = []
    archive = Archive(problem)
    stagnation = 0
    start = clock()
    stop_reason = "budget"

    while len(trials) < config.budget:
        if config.wall_time_seconds is not None and clock() - start >= config.wall_time_seconds:
            stop_reason = "wall_time"
            break
        if config.max_stagnation and stagnation >= config.max_stagnation:
            stop_reason = "stagnation"
            break
        count = min(config.batch_size, config.budget - len(trials))
        # The optimizer records every accepted tell. Avoid re-copying the full
        # monotonically growing key set on every batch (quadratic total work).
        points = optimizer.ask(count)
        if not points:
            stop_reason = optimizer.empty_ask_reason or "proposal_stalled"
            break
        signature = archive.signature
        evaluated = _evaluate_batch(problem, evaluator, len(trials), points, config.workers)
        batch = tuple(
            canonical_trial(problem, trial, point)
            for trial, point in zip(evaluated, points, strict=True)
        )
        optimizer.tell(batch)
        for trial in batch:
            trials.append(trial)
            archive.add(trial)
        frontier_grew = archive.signature != signature
        stagnation = 0 if frontier_grew else stagnation + len(batch)

    result = OptimizationResult(
        problem,
        tuple(trials),
        archive.frontier,
        stop_reason,
        evaluator_id,
        config.seed,
    )
    if directory is not None:
        metadata = {
            "schema_version": 1,
            "package_version": __version__,
            "strategy_schema_version": _STRATEGY_SCHEMA_VERSION,
            "problem_sha256": problem_fingerprint(problem),
            "evaluator_id": evaluator_id,
            "strategy": selected.value,
            "strategy_parameters": parameters,
            "population_size": effective_population,
            "seed": config.seed,
            "budget": config.budget,
            "batch_size": config.batch_size,
            "workers": config.workers,
            "max_stagnation": config.max_stagnation,
            "wall_time_seconds": config.wall_time_seconds,
            "command_timeout_seconds": config.command_timeout_seconds,
            "completed_trials": len(trials),
            "stop_reason": stop_reason,
        }
        result_artifacts = result_outputs(result, directory)
        outputs = (
            (artifact_paths[0], ledger_bytes(trials)),
            (artifact_paths[1], metadata_bytes(metadata)),
            *result_artifacts,
        )
        atomic_write_many(outputs, force=force, protected=protected)
    return result


def optimize_strategy(
    problem: Problem,
    evaluator: Callable[[Mapping[str, Scalar]], Mapping[str, float]],
    *,
    evaluator_id: str,
    config: RunConfig,
    strategy: StrategyName | str,
    population_size: int | None = None,
    output_directory: str | Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    force: bool = False,
    protected_paths: Iterable[str | Path] = (),
) -> OptimizationResult:
    """Run a catalog strategy while coordinating every durable output transition."""

    if output_directory is None:
        return _optimize_strategy_coordinated(
            problem,
            evaluator,
            evaluator_id=evaluator_id,
            config=config,
            strategy=strategy,
            population_size=population_size,
            output_directory=None,
            clock=clock,
            force=force,
            protected_paths=protected_paths,
        )
    with WriterClaim(Path(output_directory)):
        return _optimize_strategy_coordinated(
            problem,
            evaluator,
            evaluator_id=evaluator_id,
            config=config,
            strategy=strategy,
            population_size=population_size,
            output_directory=output_directory,
            clock=clock,
            force=force,
            protected_paths=protected_paths,
        )
