"""Command-line interface for validating, running, and inspecting searches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from biasweave._strict_json import JSONLimits, StrictJSONError, loads_strict_json
from biasweave._version import __version__
from biasweave.archive import Archive
from biasweave.benchmark import compare_with_random, load_analog_benchmark, sizing_decision
from biasweave.engine import optimize, validate_checkpoint_trials
from biasweave.errors import BiasWeaveError, ConfigurationError
from biasweave.evaluator import CommandEvaluator, load_python_evaluator
from biasweave.ledger import TrialLedger, read_metadata
from biasweave.model import RunConfig, Scalar
from biasweave.problem import load_problem
from biasweave.results import frontier_table

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

    benchmark = commands.add_parser("benchmark", help="run an analog contract comparison")
    benchmark.add_argument("--contract", required=True, type=Path)
    benchmark.add_argument("--budget", required=True, type=_positive)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument(
        "--decision-output",
        type=Path,
        help="write the content-bound BiasWeave representative for downstream simulation",
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
    result = optimize(
        problem,
        evaluator,
        evaluator_id=arguments.evaluator,
        config=_config(arguments, budget),
        output_directory=arguments.out,
        resume=resume,
    )
    print(
        f"{result.stop_reason}: {len(result.trials)} evaluations, "
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
    if arguments.command == "benchmark":
        if (
            arguments.output
            and arguments.decision_output
            and arguments.output.resolve() == arguments.decision_output.resolve()
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
        _write_outputs_atomically(outputs)
        if not arguments.output:
            print(rendered, end="")
        return 0
    raise ConfigurationError(f"unsupported command: {arguments.command}")


def _write_outputs_atomically(outputs: list[tuple[Path, str]]) -> None:
    """Stage every benchmark artifact before making a destination visible."""

    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in outputs:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            staged.append((temporary, destination))
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)


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
