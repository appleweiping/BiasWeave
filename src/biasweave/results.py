"""Optimization result serialization and summaries."""

from __future__ import annotations

import json
from pathlib import Path

from biasweave.errors import CheckpointError
from biasweave.model import OptimizationResult, Trial


def result_data(result: OptimizationResult) -> dict[str, object]:
    return {
        "schema_version": 1,
        "problem_sha256": result.problem.source_hash,
        "evaluator_id": result.evaluator_id,
        "seed": result.seed,
        "stop_reason": result.stop_reason,
        "trial_count": len(result.trials),
        "successful_trials": result.successful_trials,
        "failed_trials": result.failed_trials,
        "frontier": [trial.as_dict() for trial in result.frontier],
    }


def summary_markdown(result: OptimizationResult) -> str:
    lines = [
        "# BiasWeave run summary",
        "",
        f"- Stop reason: `{result.stop_reason}`",
        f"- Evaluations: {len(result.trials)}",
        f"- Successful: {result.successful_trials}",
        f"- Failed: {result.failed_trials}",
        f"- Feasible Pareto points: {len(result.frontier)}",
        "",
    ]
    if result.frontier:
        variable_names = [variable.name for variable in result.problem.variables]
        objective_names = [objective.metric for objective in result.problem.objectives]
        headers = ["Trial", *variable_names, *objective_names]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
        for trial in result.frontier:
            values = [
                str(trial.trial_id),
                *(
                    f"{trial.point.values[name]:.8g}"
                    if isinstance(trial.point.values[name], float)
                    else str(trial.point.values[name])
                    for name in variable_names
                ),
                *(f"{trial.metrics[name]:.8g}" for name in objective_names),
            ]
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("No feasible point was found within the evaluation budget.")
    return "\n".join(lines) + "\n"


def write_result(result: OptimizationResult, directory: str | Path) -> tuple[Path, Path]:
    target = Path(directory)
    frontier_path = target / "frontier.json"
    summary_path = target / "summary.md"
    try:
        target.mkdir(parents=True, exist_ok=True)
        frontier_path.write_text(
            json.dumps(result_data(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(summary_markdown(result), encoding="utf-8")
    except OSError as error:
        raise CheckpointError(f"cannot write result under {target}: {error}") from error
    return frontier_path, summary_path


def frontier_table(trials: tuple[Trial, ...]) -> str:
    if not trials:
        return "No feasible Pareto points."
    lines = ["trial  objectives  variables"]
    for trial in trials:
        objectives = ", ".join(f"{value:.6g}" for value in trial.objective_vector)
        variables = json.dumps(trial.point.values, sort_keys=True, separators=(",", ":"))
        lines.append(f"{trial.trial_id:>5}  [{objectives}]  {variables}")
    return "\n".join(lines)
