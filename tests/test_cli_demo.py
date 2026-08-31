from __future__ import annotations

import json
from pathlib import Path

import pytest

from biasweave.cli import build_parser, main
from biasweave.demo import evaluate, ug_bw_ratio
from biasweave.problem import load_problem

EXAMPLE = Path(__file__).parents[1] / "examples" / "two_stage_ota" / "problem.toml"


def test_parser_help_and_version(capsys):
    parser = build_parser()
    assert "constraint-first" in parser.description
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert "BiasWeave 0.1.0" in capsys.readouterr().out


def test_validate_command_reports_problem_shape(capsys):
    assert main(["validate", "--problem", str(EXAMPLE)]) == 0
    output = capsys.readouterr().out
    assert "8 free variables" in output
    assert "2 objectives" in output
    assert "4 constraints" in output


def test_validate_command_reports_expected_error(tmp_path, capsys):
    path = tmp_path / "invalid.toml"
    path.write_text("schema_version = 9", encoding="utf-8")
    assert main(["validate", "--problem", str(path)]) == 2
    assert "schema_version" in capsys.readouterr().err


def test_run_resume_and_front_commands(tmp_path, capsys):
    output = tmp_path / "run with spaces"
    common = [
        "--problem",
        str(EXAMPLE),
        "--evaluator",
        "python:biasweave.demo:evaluate",
        "--workers",
        "2",
        "--batch-size",
        "4",
        "--out",
        str(output),
    ]
    assert main(["run", *common, "--budget", "12", "--seed", "9"]) == 0
    run_output = capsys.readouterr().out
    assert "12 evaluations" in run_output
    assert output.is_dir()
    assert main(["resume", *common, "--additional-budget", "4", "--seed", "9"]) == 0
    resume_output = capsys.readouterr().out
    assert "16 evaluations" in resume_output
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["completed_trials"] == 16
    assert (
        main(
            [
                "front",
                "--problem",
                str(EXAMPLE),
                "--ledger",
                str(output / "trials.jsonl"),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip()


def test_run_rejects_unknown_evaluator_transport(tmp_path, capsys):
    code = main(
        [
            "run",
            "--problem",
            str(EXAMPLE),
            "--evaluator",
            "remote:https://example.invalid",
            "--budget",
            "1",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == 2
    assert "must start with python: or command:" in capsys.readouterr().err


@pytest.mark.parametrize("payload", ["not-json", "{}", "[1]"])
def test_command_transport_requires_json_string_array(payload, tmp_path, capsys):
    code = main(
        [
            "run",
            "--problem",
            str(EXAMPLE),
            "--evaluator",
            f"command:{payload}",
            "--budget",
            "1",
            "--out",
            str(tmp_path / payload.replace("/", "_")),
        ]
    )
    assert code == 2
    error = capsys.readouterr().err
    assert "JSON argv array" in error or "non-empty strings" in error


def test_resume_reports_corrupt_completed_count(tmp_path, capsys):
    output = tmp_path / "checkpoint"
    output.mkdir()
    (output / "run.json").write_text('{"completed_trials": true}', encoding="utf-8")
    code = main(
        [
            "resume",
            "--problem",
            str(EXAMPLE),
            "--evaluator",
            "python:biasweave.demo:evaluate",
            "--additional-budget",
            "2",
            "--out",
            str(output),
        ]
    )
    assert code == 2
    assert "completed_trials" in capsys.readouterr().err


def test_demo_evaluator_returns_finite_metrics_for_example_defaults():
    problem = load_problem(EXAMPLE)
    point = {variable.name: variable.default for variable in problem.free_variables}
    point["mirror_width_m"] = float(point["input_width_m"]) * 0.5
    metrics = evaluate(point)
    assert set(metrics) == {
        "gain_db",
        "ugbw_hz",
        "phase_margin_deg",
        "slew_v_per_s",
        "power_w",
        "area_m2",
    }
    assert all(value > 0.0 for value in metrics.values())
    assert 0.0 < metrics["phase_margin_deg"] < 90.0


def test_demo_relationships_have_expected_direction():
    problem = load_problem(EXAMPLE)
    point = {variable.name: variable.default for variable in problem.free_variables}
    point["mirror_width_m"] = float(point["input_width_m"]) * 0.5
    baseline = evaluate(point)
    higher_current = dict(point)
    higher_current["bias_current_a"] = float(point["bias_current_a"]) * 2.0
    changed = evaluate(higher_current)
    assert changed["power_w"] > baseline["power_w"]
    assert changed["slew_v_per_s"] > baseline["slew_v_per_s"]
    larger_cap = dict(point)
    larger_cap["compensation_cap_f"] = float(point["compensation_cap_f"]) * 2.0
    compensated = evaluate(larger_cap)
    assert compensated["ugbw_hz"] < baseline["ugbw_hz"]
    assert compensated["area_m2"] > baseline["area_m2"]


def test_ug_bw_ratio_guards_zero_second_pole():
    assert ug_bw_ratio(10.0, 2.0) == pytest.approx(5.0)
    assert ug_bw_ratio(1.0, 0.0) == pytest.approx(1e30)


def test_demo_rejects_unknown_device_flavor():
    problem = load_problem(EXAMPLE)
    point = {variable.name: variable.default for variable in problem.free_variables}
    point["mirror_width_m"] = float(point["input_width_m"]) * 0.5
    point["device_flavor"] = "mystery"
    with pytest.raises(ValueError, match="device_flavor"):
        evaluate(point)
