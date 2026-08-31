"""Command-line interface for validating, running, and inspecting searches."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from biasweave.archive import Archive
from biasweave.engine import optimize, validate_checkpoint_trials
from biasweave.errors import BiasWeaveError, ConfigurationError
from biasweave.evaluator import CommandEvaluator, load_python_evaluator
from biasweave.ledger import TrialLedger, read_metadata
from biasweave.model import RunConfig, Scalar
from biasweave.problem import load_problem
from biasweave.results import frontier_table


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
            argv = json.loads(payload)
        except json.JSONDecodeError as error:
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
    parser.add_argument("--version", action="version", version="BiasWeave 0.1.0")
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
    raise ConfigurationError(f"unsupported command: {arguments.command}")


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


if __name__ == "__main__":
    raise SystemExit(main())
