"""The `quality` command, its rendering, and the one reference point it uses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biasweave.cli import main
from biasweave.model import Goal, Objective
from biasweave.quality import (
    DEFAULT_ATTAINMENT_FRACTION,
    attainment_curve,
    compare_fronts,
    derive_reference_point,
    front_quality,
    measure_run,
)
from biasweave.results import (
    attainment_lines,
    comparison_lines,
    quality_lines,
    result_quality,
    summary_markdown,
)
from tests.helpers import make_problem
from tests.test_quality import improving_run, trials

EXAMPLE = Path(__file__).parents[1] / "examples" / "two_stage_ota" / "problem.toml"

OBJECTIVES = (
    Objective(metric="loss", goal=Goal.MIN, scale=1.0, epsilon=0.05),
    Objective(metric="score", goal=Goal.MAX, scale=2.0, epsilon=0.05),
)


def run_search(tmp_path: Path, name: str, *, budget: int, seed: int) -> Path:
    directory = tmp_path / name
    assert (
        main(
            [
                "run",
                "--problem",
                str(EXAMPLE),
                "--evaluator",
                "python:biasweave.demo:evaluate",
                "--budget",
                str(budget),
                "--seed",
                str(seed),
                "--out",
                str(directory),
            ]
        )
        == 0
    )
    return directory


# ---------------------------------------------------------------------------
# One reference point per report.
# ---------------------------------------------------------------------------


def test_the_curve_ends_where_the_front_measurement_lands() -> None:
    """The property that makes a single report readable.

    Measured apart, the front derives a box it fills tightly and the curve
    derives one wide enough for every worse front before it, so the two
    hypervolumes differ by orders of magnitude while sharing a name.
    """

    run = improving_run()
    measured = measure_run(make_problem(), run, steps=8)
    assert measured.attainment is not None
    assert measured.attainment.final_hypervolume == pytest.approx(measured.front.hypervolume)

    apart = front_quality(run)
    assert apart.hypervolume != pytest.approx(measured.front.hypervolume)


def test_a_pool_widens_the_box_beyond_the_front_that_is_measured() -> None:
    front = trials((0.1, 0.1))
    wider = [(0.1, 0.1), (5.0, 5.0)]
    assert front_quality(front).hypervolume < front_quality(front, pool=wider).hypervolume
    assert front_quality(front, pool=wider).reference_point == derive_reference_point(wider)


def test_a_pooled_reference_is_still_reported_as_derived() -> None:
    quality = front_quality(trials((0.1, 0.1)), pool=[(0.0, 0.0), (2.0, 2.0)])
    assert quality.reference_derived


def test_an_explicit_reference_beats_a_pool() -> None:
    quality = front_quality(
        trials((0.1, 0.1)), reference_point=(1.0, 1.0), pool=[(9.0, 9.0)]
    )
    assert quality.reference_point == (1.0, 1.0)
    assert not quality.reference_derived


def test_a_pool_lets_an_empty_front_be_measured() -> None:
    assert front_quality([], pool=[(0.0, 0.0), (1.0, 1.0)]).hypervolume == 0.0
    curve = attainment_curve(make_problem(), [], pool=[(0.0, 0.0), (1.0, 1.0)])
    assert curve.points == ()


def test_a_compared_run_shares_the_reference_with_its_own_curve() -> None:
    measured = measure_run(
        make_problem(), improving_run(), compare=trials((0.2, 0.2)), steps=4
    )
    assert measured.comparison is not None
    assert measured.attainment is not None
    assert measured.comparison.reference_point == measured.reference_point
    assert measured.attainment.reference_point == measured.reference_point


def test_a_derived_reference_is_not_reported_as_supplied_inside_a_comparison() -> None:
    # `measure_run` passes the reference into `compare_fronts` explicitly, which
    # would otherwise record it as something the reader supplied.
    measured = measure_run(make_problem(), improving_run(), compare=trials((0.2, 0.2)))
    assert measured.comparison is not None
    assert measured.comparison.reference_derived
    assert measured.comparison.left.reference_derived
    assert measured.comparison.right.reference_derived


def test_a_supplied_reference_stays_supplied_through_a_comparison() -> None:
    measured = measure_run(
        make_problem(),
        improving_run(),
        compare=trials((0.2, 0.2)),
        reference_point=(2.0, 2.0),
    )
    assert measured.comparison is not None
    assert not measured.comparison.reference_derived
    assert not measured.comparison.left.reference_derived


def test_a_run_without_a_feasible_trial_cannot_derive_a_reference() -> None:
    from biasweave.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="no run has a feasible trial"):
        measure_run(make_problem(), [])


def test_a_measured_run_serializes_only_what_it_measured() -> None:
    payload = measure_run(make_problem(), improving_run()).as_dict()
    assert "attainment" not in payload
    assert "comparison" not in payload
    full = measure_run(
        make_problem(), improving_run(), compare=trials((0.2, 0.2)), steps=3
    ).as_dict()
    assert set(full) >= {"attainment", "comparison", "front", "reference_point"}


# ---------------------------------------------------------------------------
# How much of the budget bought how much of the front.
# ---------------------------------------------------------------------------


def test_the_curve_says_when_it_was_practically_finished() -> None:
    curve = attainment_curve(make_problem(), improving_run(), steps=8)
    # The run flattens at evaluation 4, so the first cut reaching the whole
    # final volume is that one, not the last cut of the curve.
    assert curve.evaluations_for(1.0) == 4
    assert curve.evaluations_for() <= curve.evaluations_for(1.0)
    assert curve.evaluations_for(1.0) <= curve.points[-1].evaluations


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_a_fraction_outside_the_unit_interval_is_refused(fraction: float) -> None:
    from biasweave.errors import ConfigurationError

    curve = attainment_curve(make_problem(), improving_run(), steps=4)
    with pytest.raises(ConfigurationError, match="fraction"):
        curve.evaluations_for(fraction)


def test_a_run_that_found_nothing_reached_no_share_of_nothing() -> None:
    curve = attainment_curve(
        make_problem(), trials((5.0, 5.0)), reference_point=(1.0, 1.0)
    )
    assert curve.final_hypervolume == 0.0
    assert curve.evaluations_for() == 0


def test_the_default_fraction_is_a_near_miss_not_a_whole_one() -> None:
    assert 0.5 < DEFAULT_ATTAINMENT_FRACTION < 1.0


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def test_the_rendering_names_each_objective_beside_its_number() -> None:
    lines = quality_lines(front_quality(trials((0.1, 0.9), (0.9, 0.1))), OBJECTIVES)
    text = "\n".join(lines)
    assert "loss=" in text
    assert "score=" in text
    assert "derived" in text


def test_the_rendering_says_when_a_reference_was_supplied() -> None:
    lines = quality_lines(
        front_quality(trials((0.1, 0.1)), reference_point=(1.0, 1.0)), OBJECTIVES
    )
    assert any("supplied" in line for line in lines)


def test_the_rendering_warns_about_points_it_could_not_see() -> None:
    lines = quality_lines(
        front_quality(trials((0.1, 0.1), (9.0, 0.05)), reference_point=(1.0, 1.0)),
        OBJECTIVES,
    )
    assert any("outside the reference box" in line for line in lines)


def test_a_comparison_prints_the_shared_box_once() -> None:
    text = "\n".join(
        comparison_lines(compare_fronts(trials((0.1, 0.1)), trials((0.6, 0.6))), OBJECTIVES)
    )
    assert text.count("Reference point") == 1
    assert "Left front:" in text and "Right front:" in text


def test_a_comparison_says_when_neither_front_reaches_the_other() -> None:
    text = "\n".join(
        comparison_lines(
            compare_fronts(trials((0.1, 0.9)), trials((0.9, 0.1))), OBJECTIVES
        )
    )
    assert "Neither front reaches" in text


def test_a_comparison_stays_quiet_when_one_front_dominates() -> None:
    text = "\n".join(
        comparison_lines(compare_fronts(trials((0.1, 0.1)), trials((0.6, 0.6))), OBJECTIVES)
    )
    assert "Neither front reaches" not in text


def test_the_curve_rendering_shows_the_share_of_the_final_volume() -> None:
    curve = attainment_curve(make_problem(), improving_run(), steps=8)
    text = "\n".join(attainment_lines(curve, 8))
    assert "share of final" in text
    assert "100.00%" in text


def test_the_curve_rendering_reports_a_run_that_stopped_improving() -> None:
    run = [*improving_run(), *(trials((0.9, 0.9)) * 4)]
    curve = attainment_curve(make_problem(), run, steps=6)
    text = "\n".join(attainment_lines(curve, len(run)))
    assert "did not extend it" in text


def test_the_curve_rendering_reports_a_run_still_improving() -> None:
    run = [
        trial
        for trial in improving_run()[:4]
    ]
    curve = attainment_curve(make_problem(), run, steps=4)
    assert "still improving" in "\n".join(attainment_lines(curve, len(run)))


def test_the_curve_rendering_survives_a_run_that_found_nothing() -> None:
    curve = attainment_curve(
        make_problem(), trials((5.0, 5.0)), reference_point=(1.0, 1.0)
    )
    text = "\n".join(attainment_lines(curve, 1))
    assert "share of final" in text
    assert "-" in text


# ---------------------------------------------------------------------------
# The run summary.
# ---------------------------------------------------------------------------


def test_a_run_summary_reports_the_quality_of_its_front(tmp_path: Path) -> None:
    directory = run_search(tmp_path, "summary", budget=32, seed=3)
    summary = (directory / "summary.md").read_text(encoding="utf-8")
    assert "## Front quality" in summary
    assert "Hypervolume:" in summary


def test_the_stored_result_carries_the_same_quality(tmp_path: Path) -> None:
    directory = run_search(tmp_path, "stored", budget=32, seed=3)
    data = json.loads((directory / "frontier.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["quality"]["hypervolume"] > 0.0
    assert data["quality"]["front_size"] == len(data["frontier"])


def test_an_unmeasurable_front_does_not_discard_a_finished_run() -> None:
    """A diagnostic must never destroy the run it describes.

    Every evaluation is already spent by the time the front is measured, so a
    front the indicator cannot handle is reported as absent rather than raised.
    """

    from biasweave.model import OptimizationResult

    problem = make_problem()
    result = OptimizationResult(
        problem=problem,
        trials=(),
        frontier=(),
        stop_reason="budget",
        evaluator_id="python:tests:none",
        seed=0,
    )
    assert result_quality(result) is None
    assert "No feasible point" in summary_markdown(result)


# ---------------------------------------------------------------------------
# The command.
# ---------------------------------------------------------------------------


def test_the_command_measures_a_ledger(tmp_path: Path, capsys) -> None:
    directory = run_search(tmp_path, "one", budget=48, seed=7)
    capsys.readouterr()
    assert (
        main(
            [
                "quality",
                "--problem",
                str(EXAMPLE),
                "--ledger",
                str(directory / "trials.jsonl"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Hypervolume:" in output
    assert "power_w=" in output


def test_the_command_reports_the_curve_on_request(tmp_path: Path, capsys) -> None:
    directory = run_search(tmp_path, "curve", budget=48, seed=7)
    capsys.readouterr()
    assert (
        main(
            [
                "quality",
                "--problem",
                str(EXAMPLE),
                "--ledger",
                str(directory / "trials.jsonl"),
                "--curve",
                "--steps",
                "4",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "share of final" in output
    assert output.count("\n") > 6


def test_the_command_compares_two_ledgers(tmp_path: Path, capsys) -> None:
    left = run_search(tmp_path, "left", budget=64, seed=17)
    right = run_search(tmp_path, "right", budget=64, seed=5)
    capsys.readouterr()
    assert (
        main(
            [
                "quality",
                "--problem",
                str(EXAMPLE),
                "--ledger",
                str(left / "trials.jsonl"),
                "--compare",
                str(right / "trials.jsonl"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Left front:" in output
    assert "covers" in output
    assert output.count("Reference point") == 1


def test_the_command_writes_json_instead_of_text(tmp_path: Path, capsys) -> None:
    directory = run_search(tmp_path, "json", budget=48, seed=7)
    target = tmp_path / "quality.json"
    capsys.readouterr()
    assert (
        main(
            [
                "quality",
                "--problem",
                str(EXAMPLE),
                "--ledger",
                str(directory / "trials.jsonl"),
                "--curve",
                "--steps",
                "3",
                "--output",
                str(target),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == ""
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["front"]["hypervolume"] > 0.0
    assert len(payload["attainment"]["points"]) == 3
    assert payload["problem_sha256"]


def test_the_json_names_both_ledgers_of_a_comparison(tmp_path: Path) -> None:
    left = run_search(tmp_path, "jl", budget=48, seed=17)
    right = run_search(tmp_path, "jr", budget=48, seed=5)
    target = tmp_path / "compare.json"
    assert (
        main(
            [
                "quality",
                "--problem",
                str(EXAMPLE),
                "--ledger",
                str(left / "trials.jsonl"),
                "--compare",
                str(right / "trials.jsonl"),
                "--output",
                str(target),
            ]
        )
        == 0
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["compared_ledger"].endswith("trials.jsonl")
    assert "comparison" in payload


def test_the_command_accepts_an_explicit_reference_point(tmp_path: Path, capsys) -> None:
    directory = run_search(tmp_path, "explicit", budget=48, seed=7)
    capsys.readouterr()
    assert (
        main(
            [
                "quality",
                "--problem",
                str(EXAMPLE),
                "--ledger",
                str(directory / "trials.jsonl"),
                "--reference-point",
                "5,5",
            ]
        )
        == 0
    )
    assert "supplied" in capsys.readouterr().out


def test_a_reference_point_of_the_wrong_width_is_refused(
    tmp_path: Path, capsys
) -> None:
    directory = run_search(tmp_path, "width", budget=16, seed=7)
    capsys.readouterr()
    assert (
        main(
            [
                "quality",
                "--problem",
                str(EXAMPLE),
                "--ledger",
                str(directory / "trials.jsonl"),
                "--reference-point",
                "5,5,5",
            ]
        )
        == 2
    )
    assert "3 components for 2 objectives" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", "a,b", "1,nan", "2,inf"])
def test_an_unparsable_reference_point_is_refused(value: str) -> None:
    with pytest.raises(SystemExit):
        main(["quality", "--problem", str(EXAMPLE), "--ledger", "x", "--reference-point", value])


def test_a_ledger_of_another_problem_is_refused(tmp_path: Path, capsys) -> None:
    """Two fronts are comparable only when they answer the same question."""

    directory = run_search(tmp_path, "mismatch", budget=16, seed=7)
    other = tmp_path / "other.toml"
    other.write_text(
        EXAMPLE.read_text(encoding="utf-8").replace("limit = 62.0", "limit = 40.0"),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert (
        main(
            [
                "quality",
                "--problem",
                str(other),
                "--ledger",
                str(directory / "trials.jsonl"),
            ]
        )
        == 2
    )
    assert capsys.readouterr().err
