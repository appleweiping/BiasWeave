"""Command-line interface for validating, running, and inspecting searches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from biasweave._output import atomic_write_many, paths_alias, utf8
from biasweave._strict_json import JSONLimits, StrictJSONError, loads_strict_json
from biasweave._version import __version__
from biasweave.archive import Archive
from biasweave.benchmark import (
    compare_optimizer_catalog,
    compare_with_random,
    load_analog_benchmark,
    sizing_decision,
)
from biasweave.engine import optimize, validate_checkpoint_trials
from biasweave.errors import BiasWeaveError, ConfigurationError
from biasweave.evaluator import CommandEvaluator, load_python_evaluator
from biasweave.ledger import TrialLedger, read_metadata
from biasweave.model import Problem, RunConfig, Scalar, Trial
from biasweave.optimizers import StrategyName
from biasweave.problem import load_problem
from biasweave.quality import measure_run
from biasweave.results import attainment_lines, comparison_lines, frontier_table, quality_lines
from biasweave.strategy import optimize_strategy

_COMMAND_JSON_LIMITS = JSONLimits(
    max_bytes=65_536,
    max_depth=16,
    max_nodes=256,
    max_number_characters=64,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def _reference_point(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "reference point must be comma-separated numbers"
        ) from error
    if not parsed or not all(math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError("reference point must be finite and non-empty")
    return parsed


def _evaluator(
    specification: str, timeout: float
) -> Callable[[Mapping[str, Scalar]], Mapping[str, float]]:
    if specification.startswith("python:"):
        return load_python_evaluator(specification)
    if specification.startswith("command:"):
        payload = specification.removeprefix("command:")
        try:
            argv = loads_strict_json(
                payload,
                limits=_COMMAND_JSON_LIMITS,
                context="command evaluator JSON",
            )
        except StrictJSONError as error:
            raise ConfigurationError(
                f"command evaluator must contain a JSON argv array: {error}"
            ) from error
        if not isinstance(argv, list):
            raise ConfigurationError("command evaluator must contain a JSON argv array")
        return CommandEvaluator(argv, timeout)
    raise ConfigurationError("evaluator must start with python: or command:")


def _add_run_options(parser: argparse.ArgumentParser, *, resume: bool) -> None:
    parser.add_argument("--problem", required=True, type=Path, help="TOML problem definition")
    parser.add_argument(
        "--evaluator",
        required=True,
        help="trusted python:module:function or command:JSON_ARGV evaluator",
    )
    budget_name = "--additional-budget" if resume else "--budget"
    parser.add_argument(budget_name, required=True, type=_positive)
    parser.add_argument("--out", required=True, type=Path, help="checkpoint/output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=_positive, default=1)
    parser.add_argument("--batch-size", type=_positive, default=8)
    parser.add_argument("--max-stagnation", type=_non_negative, default=0)
    parser.add_argument("--wall-time", type=float)
    parser.add_argument("--command-timeout", type=float, default=300.0)
    if not resume:
        parser.add_argument(
            "--strategy",
            choices=tuple(strategy.value for strategy in StrategyName),
            default=StrategyName.WEAVE.value,
            help="independent search strategy (default: weave)",
        )
        parser.add_argument(
            "--population-size",
            type=_positive,
            help="population for pso, de, nsga2, or moead (default: 16)",
        )
        parser.add_argument(
            "--force", action="store_true", help="replace existing outputs, never input aliases"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biasweave",
        description="Deterministic constraint-first multi-objective sizing search.",
    )
    parser.add_argument("--version", action="version", version=f"BiasWeave {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a TOML problem")
    validate.add_argument("--problem", required=True, type=Path)

    run = commands.add_parser("run", help="start a new optimization run")
    _add_run_options(run, resume=False)

    resume = commands.add_parser("resume", help="continue an existing checkpoint")
    _add_run_options(resume, resume=True)

    front = commands.add_parser("front", help="print the feasible Pareto front")
    front.add_argument("--problem", required=True, type=Path)
    front.add_argument("--ledger", required=True, type=Path)

    quality = commands.add_parser(
        "quality", help="measure the front a ledger records, or compare two"
    )
    quality.add_argument("--problem", required=True, type=Path)
    quality.add_argument("--ledger", required=True, type=Path)
    quality.add_argument(
        "--compare",
        type=Path,
        help="a second ledger of the same problem, measured against one shared reference",
    )
    quality.add_argument(
        "--reference-point",
        type=_reference_point,
        help="comma-separated bound per objective, in normalized units; "
        "derived from the front when omitted",
    )
    quality.add_argument(
        "--steps",
        type=_positive,
        default=20,
        help="points on the attainment curve; 0 is not permitted, omit --curve to skip it",
    )
    quality.add_argument(
        "--curve", action="store_true", help="report hypervolume against evaluations spent"
    )
    quality.add_argument("--output", type=Path, help="write JSON instead of text")
    quality.add_argument("--force", action="store_true", help="replace an existing output")

    benchmark = commands.add_parser("benchmark", help="run an analog contract comparison")
    benchmark.add_argument("--contract", required=True, type=Path)
    benchmark.add_argument("--budget", required=True, type=_positive)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--force", action="store_true", help="replace existing outputs")
    benchmark.add_argument(
        "--decision-output",
        type=Path,
        help="write the content-bound BiasWeave representative for downstream simulation",
    )

    catalog_benchmark = commands.add_parser(
        "catalog-benchmark", help="run every optimizer against one analog contract"
    )
    catalog_benchmark.add_argument("--contract", required=True, type=Path)
    catalog_benchmark.add_argument("--budget", required=True, type=_positive)
    catalog_benchmark.add_argument("--seed", type=int, default=0)
    catalog_benchmark.add_argument("--population-size", type=_positive, default=16)
    catalog_benchmark.add_argument("--output", type=Path)
    catalog_benchmark.add_argument(
        "--force", action="store_true", help="replace an existing output"
    )
    return parser


def _config(arguments: argparse.Namespace, budget: int) -> RunConfig:
    return RunConfig(
        budget=budget,
        seed=arguments.seed,
        workers=arguments.workers,
        batch_size=arguments.batch_size,
        max_stagnation=arguments.max_stagnation,
        wall_time_seconds=arguments.wall_time,
        command_timeout_seconds=arguments.command_timeout,
    )


def _run(arguments: argparse.Namespace, *, resume: bool) -> int:
    problem = load_problem(arguments.problem)
    evaluator = _evaluator(arguments.evaluator, arguments.command_timeout)
    if resume:
        metadata = read_metadata(arguments.out / "run.json")
        completed = metadata.get("completed_trials")
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            raise ConfigurationError("checkpoint completed_trials is invalid")
        budget = completed + arguments.additional_budget
    else:
        budget = arguments.budget
    config = _config(arguments, budget)
    strategy = StrategyName.WEAVE if resume else StrategyName(arguments.strategy)
    if strategy is StrategyName.WEAVE:
        if not resume and arguments.population_size is not None:
            raise ConfigurationError("--population-size does not apply to weave")
        result = optimize(
            problem,
            evaluator,
            evaluator_id=arguments.evaluator,
            config=config,
            output_directory=arguments.out,
            resume=resume,
            force=False if resume else arguments.force,
            protected_paths=(arguments.problem,),
        )
    else:
        result = optimize_strategy(
            problem,
            evaluator,
            evaluator_id=arguments.evaluator,
            config=config,
            strategy=strategy,
            population_size=arguments.population_size,
            output_directory=arguments.out,
            force=arguments.force,
            protected_paths=(arguments.problem,),
        )
    prefix = "" if strategy is StrategyName.WEAVE else f"{strategy.value}: "
    print(
        f"{prefix}{result.stop_reason}: {len(result.trials)} evaluations, "
        f"{len(result.frontier)} feasible Pareto points, {result.failed_trials} failed"
    )
    print(f"Results: {arguments.out}")
    return 0


def dispatch(arguments: argparse.Namespace) -> int:
    if arguments.command == "validate":
        problem = load_problem(arguments.problem)
        print(
            f"Valid problem: {len(problem.free_variables)} free variables, "
            f"{len(problem.objectives)} objectives, {len(problem.constraints)} constraints"
        )
        return 0
    if arguments.command == "run":
        return _run(arguments, resume=False)
    if arguments.command == "resume":
        return _run(arguments, resume=True)
    if arguments.command == "front":
        problem = load_problem(arguments.problem)
        trials = TrialLedger(arguments.ledger).read()
        validate_checkpoint_trials(problem, trials)
        archive = Archive(problem, trials)
        print(frontier_table(archive.frontier))
        return 0
    if arguments.command == "quality":
        return _quality(arguments)
    if arguments.command == "benchmark":
        if (
            arguments.output
            and arguments.decision_output
            and paths_alias(arguments.output, arguments.decision_output)
        ):
            raise ConfigurationError("--output and --decision-output must be different paths")
        benchmark = load_analog_benchmark(arguments.contract)
        comparison = compare_with_random(benchmark, budget=arguments.budget, seed=arguments.seed)
        decision = sizing_decision(benchmark, comparison) if arguments.decision_output else None
        rendered = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        outputs: list[tuple[Path, str]] = []
        if arguments.output:
            outputs.append((arguments.output, rendered))
        if arguments.decision_output:
            outputs.append(
                (
                    arguments.decision_output,
                    json.dumps(decision, indent=2, sort_keys=True) + "\n",
                )
            )
        _write_outputs_atomically(
            outputs,
            force=arguments.force,
            protected=(arguments.contract,),
        )
        if not arguments.output:
            print(rendered, end="")
        return 0
    if arguments.command == "catalog-benchmark":
        benchmark = load_analog_benchmark(arguments.contract)
        comparison = compare_optimizer_catalog(
            benchmark,
            budget=arguments.budget,
            seed=arguments.seed,
            population_size=arguments.population_size,
        )
        rendered = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        if arguments.output:
            _write_outputs_atomically(
                [(arguments.output, rendered)],
                force=arguments.force,
                protected=(arguments.contract,),
            )
        else:
            print(rendered, end="")
        return 0
    raise ConfigurationError(f"unsupported command: {arguments.command}")


def _ledger_trials(problem: Problem, path: Path) -> list[Trial]:
    """Read a ledger and hold it to the problem it claims to belong to.

    Two fronts are only comparable when they answer the same question, so both
    ledgers of a comparison are validated against the one problem document
    rather than merely being the same length.
    """

    trials = TrialLedger(path).read()
    validate_checkpoint_trials(problem, trials)
    return trials


def _quality(arguments: argparse.Namespace) -> int:
    problem = load_problem(arguments.problem)
    reference = arguments.reference_point
    if reference is not None and len(reference) != len(problem.objectives):
        raise ConfigurationError(
            f"--reference-point has {len(reference)} components for "
            f"{len(problem.objectives)} objectives"
        )
    trials = _ledger_trials(problem, arguments.ledger)
    compared = _ledger_trials(problem, arguments.compare) if arguments.compare else None
    # One call, so the front, the curve and the comparison are bounded by the
    # same box and the numbers in one report can be read against each other.
    measured = measure_run(
        problem,
        trials,
        compare=compared,
        steps=arguments.steps if arguments.curve else None,
        reference_point=reference,
    )

    if arguments.output:
        payload: dict[str, object] = {
            "schema_version": 1,
            "problem_sha256": problem.source_hash,
            "ledger": str(arguments.ledger),
            **measured.as_dict(),
        }
        if arguments.compare:
            payload["compared_ledger"] = str(arguments.compare)
        _write_outputs_atomically(
            [(arguments.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")],
            force=arguments.force,
            protected=tuple(
                path
                for path in (arguments.problem, arguments.ledger, arguments.compare)
                if path is not None
            ),
        )
        return 0

    if measured.comparison is not None:
        lines = comparison_lines(measured.comparison, problem.objectives)
    else:
        lines = quality_lines(measured.front, problem.objectives)
    if measured.attainment is not None:
        lines = [*lines, *attainment_lines(measured.attainment, len(trials))]
    print("\n".join(lines))
    return 0


def _write_outputs_atomically(
    outputs: list[tuple[Path, str]],
    *,
    force: bool,
    protected: Sequence[Path],
) -> None:
    """Install a complete output set without clobbering any analyzed input."""

    atomic_write_many(
        tuple((destination, utf8(content)) for destination, content in outputs),
        force=force,
        protected=protected,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        return dispatch(parser.parse_args(argv))
    except BiasWeaveError as error:
        print(f"biasweave: error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"biasweave: I/O error: {error}", file=sys.stderr)
        return 3


def entrypoint() -> None:
    """Convert the library-friendly return value into a process status."""

    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
