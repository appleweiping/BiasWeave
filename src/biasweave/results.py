"""Optimization result serialization and summaries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from biasweave._output import atomic_write_many, utf8
from biasweave.errors import CheckpointError, ConfigurationError
from biasweave.model import Objective, OptimizationResult, Trial
from biasweave.quality import Attainment, FrontComparison, FrontQuality, measure_run


def result_data(result: OptimizationResult) -> dict[str, object]:
    quality = result_quality(result)
    return {
        "schema_version": 2,
        "quality": quality.as_dict() if quality else None,
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
    quality = result_quality(result)
    if quality is not None and quality.front_size:
        lines.append("## Front quality")
        lines.append("")
        lines.extend(f"- {line}" for line in quality_lines(quality, result.problem.objectives))
        lines.append("")
        lines.append(
            "The reference point is derived from every feasible trial this run "
            "recorded, which is what `biasweave quality` derives for the same "
            "ledger, so the two agree. Comparing two runs needs "
            "`biasweave quality --compare`, which widens that reference to cover "
            "both rather than measuring each against its own."
        )
        lines.append("")
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


def result_outputs(
    result: OptimizationResult, directory: str | Path
) -> tuple[tuple[Path, bytes], ...]:
    """Render both result artifacts without making either one visible."""

    target = Path(directory)
    frontier_path = target / "frontier.json"
    summary_path = target / "summary.md"
    try:
        frontier = utf8(
            json.dumps(result_data(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        summary = utf8(summary_markdown(result))
    except (TypeError, ValueError, OverflowError) as error:
        raise CheckpointError(f"cannot write result under {target}: {error}") from error
    return ((frontier_path, frontier), (summary_path, summary))


def write_result(
    result: OptimizationResult,
    directory: str | Path,
    *,
    force: bool = False,
    protected: Iterable[str | Path] = (),
) -> tuple[Path, Path]:
    outputs = result_outputs(result, directory)
    written = atomic_write_many(outputs, force=force, protected=protected)
    return written[0], written[1]


def reference_line(quality: FrontQuality, objectives: Sequence[Objective]) -> str:
    origin = "derived" if quality.reference_derived else "supplied"
    return f"Reference point ({origin}): " + ", ".join(
        f"{objective.metric}={value:.6g}"
        for objective, value in zip(objectives, quality.reference_point, strict=False)
    )


def quality_lines(
    quality: FrontQuality,
    objectives: Sequence[Objective],
    *,
    with_reference: bool = True,
) -> list[str]:
    """Render one measured front, naming the objective each number belongs to.

    `with_reference` is cleared when the caller has already printed the box, so
    a comparison does not repeat one shared reference point under each front as
    though the two had been measured separately.
    """

    lines = [
        f"Hypervolume: {quality.hypervolume:.6g}",
        *([reference_line(quality, objectives)] if with_reference else []),
        f"Front points: {quality.front_size} "
        f"({quality.contributing} inside the reference box, {quality.ignored} outside)",
        f"Spacing: {quality.spacing:.6g}",
        "Extent: "
        + ", ".join(
            f"{objective.metric}={value:.6g}"
            for objective, value in zip(objectives, quality.extent, strict=False)
        ),
    ]
    if quality.ignored:
        lines.append(
            f"Note: {quality.ignored} front point(s) lie outside the reference box and "
            "contribute no volume."
        )
    return lines


def attainment_lines(attainment: Attainment, total: int) -> list[str]:
    """Render the curve, and say what the budget bought.

    The share of the final hypervolume is printed beside each point because a
    run can keep improving to its last evaluation while the improvement is
    immaterial, and a column of raw volumes hides that.
    """

    final = attainment.final_hypervolume
    lines = ["", "evaluations  front  hypervolume  share of final"]
    for point in attainment.points:
        share = f"{point.hypervolume / final:.2%}" if final > 0.0 else "-"
        lines.append(
            f"{point.evaluations:>11}  {point.front_size:>5}  "
            f"{point.hypervolume:>11.6g}  {share:>14}"
        )
    last = attainment.last_improvement
    if not last:
        return lines
    reached = attainment.evaluations_for()
    if last < total:
        lines.append(
            f"The front last improved at evaluation {last} of {total}; the remaining "
            f"{total - last} evaluations did not extend it."
        )
    else:
        lines.append(f"The front was still improving at the final evaluation ({total}).")
    if reached and reached < total:
        lines.append(
            f"It was within 1% of its final hypervolume by evaluation {reached}, "
            f"which is {reached / total:.0%} of the run."
        )
    return lines


def comparison_lines(comparison: FrontComparison, objectives: Sequence[Objective]) -> list[str]:
    """Render two fronts measured against the one shared reference point."""

    lines = [reference_line(comparison.left, objectives), "", "Left front:"]
    lines += [
        f"  {line}" for line in quality_lines(comparison.left, objectives, with_reference=False)
    ]
    lines += ["", "Right front:"]
    lines += [
        f"  {line}" for line in quality_lines(comparison.right, objectives, with_reference=False)
    ]
    lines += [
        "",
        f"Hypervolume difference (left - right): {comparison.hypervolume_difference:.6g}",
        f"Left covers {comparison.left_covers_right:.1%} of the right front; "
        f"right covers {comparison.right_covers_left:.1%} of the left.",
        f"Additive epsilon, left to right: {comparison.left_epsilon:.6g}; "
        f"right to left: {comparison.right_epsilon:.6g}.",
    ]
    if comparison.left_covers_right == 0.0 and comparison.right_covers_left == 0.0:
        lines.append(
            "Neither front reaches any point of the other, so the hypervolume "
            "difference reflects where each is concentrated rather than a dominance "
            "relation between them."
        )
    return lines


def result_quality(result: OptimizationResult) -> FrontQuality | None:
    """Measure the run front, or report nothing if it cannot be measured.

    A quality indicator is a description of a finished run. Letting it raise
    would discard the run itself -- every evaluation already spent -- over a
    diagnostic, so an unmeasurable front is reported as absent instead.
    """

    try:
        return measure_run(result.problem, result.trials).front
    except (ConfigurationError, ArithmeticError, OverflowError):
        return None


def frontier_table(trials: tuple[Trial, ...]) -> str:
    if not trials:
        return "No feasible Pareto points."
    lines = ["trial  objectives  variables"]
    for trial in trials:
        objectives = ", ".join(f"{value:.6g}" for value in trial.objective_vector)
        variables = json.dumps(dict(trial.point.values), sort_keys=True, separators=(",", ":"))
        lines.append(f"{trial.trial_id:>5}  [{objectives}]  {variables}")
    return "\n".join(lines)
