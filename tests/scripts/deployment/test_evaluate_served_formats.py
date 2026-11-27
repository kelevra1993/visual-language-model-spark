"""
Tests of scripts/deployment/evaluate_served_formats.py, the test metric of the served formats of the serving benchmark:
test_metrics.csv keeps every evaluated row so that a relaunch skips it. The requests and conversions of its rows are
those of the test split parity of the export, tested in tests/deployment/test_export_parity.py. The script prints its
help when launched without arguments, reads the folders of the machine from its command line as the speed sweep does,
refuses a missing folder or a setting of the plan, and evaluates the models, precision settings and largest request
of the plan of benchmark_serving_formats.py, which still plan every row of the committed test metrics.
"""

import csv
import pytest

from pathlib import Path
from typing import List

from scripts.deployment import benchmark_serving_formats, evaluate_served_formats
from scripts.deployment.benchmark_serving_formats import (BENCHMARKED_MODELS, PRECISION_SETTINGS,
                                                          get_repository_formats, get_row_name,
                                                          parse_benchmark_arguments)
from scripts.deployment.evaluate_served_formats import (EAGER_ROW_NAME, METRIC_COLUMNS, append_row_metrics,
                                                        parse_evaluation_arguments, read_measured_rows)

# Committed copy of the test_metrics.csv of the evaluation, whose rows the plan of the speed sweep must still list
COMMITTED_METRICS_PATH = Path("documentation") / "architecture_benchmarking" / "serving_benchmark_test_metrics.csv"

# Folders of the machine given on the command line, never read by the parser
FOLDER_ARGUMENTS = ["--benchmark_folder", "serving_benchmark", "--runs_folder", "tutorial_runs",
                    "--data_folder", "data"]


def test_metrics_file_gives_back_the_evaluated_rows(tmp_path: Path) -> None:
    """
    Every scalar of an evaluated row is one line of test_metrics.csv, the headline metric flagged, and the row is
    found again at a relaunch.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    metrics_path = tmp_path / "test_metrics.csv"
    assert read_measured_rows(metrics_path=metrics_path) == set()
    results = {"scalars": {"Accuracy": 0.95, "Balanced Accuracy": 0.94}, "headline_metric": "Accuracy",
               "evaluated_samples": "50000 Test Images"}
    append_row_metrics(metrics_path=metrics_path, model_name="classification", row_name="triton_tensorrt_fp16",
                       results=results, duration_seconds=12.0)
    lines = metrics_path.read_text().splitlines()
    assert lines[0].split(",") == METRIC_COLUMNS
    assert len(lines) == 3
    assert lines[1].startswith("classification,triton_tensorrt_fp16,Accuracy,0.95,True,")
    assert read_measured_rows(metrics_path=metrics_path) == {("classification", "triton_tensorrt_fp16")}


def test_evaluation_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the evaluation driver launched without any argument prints the help of its folders and stops without
    evaluating, instead of failing on the missing folders.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_evaluation_arguments(arguments=[])
    assert exit_information.value.code == 0
    help_text = capsys.readouterr().out
    assert [argument for argument in ["--benchmark_folder", "--runs_folder", "--data_folder"]
            if argument not in help_text] == []


def test_evaluation_arguments_are_the_folders_of_the_speed_sweep(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the evaluation driver reads the same folders as the speed sweep from the same command line, a '~'
    the shell leaves after '=' expanded to the home folder, and that it evaluates the precision settings and largest
    request of the plan of the speed sweep instead of a copy of its own.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    arguments = ["--benchmark_folder=~/serving_benchmark", "--runs_folder=~/tutorial_runs", "--data_folder=~/data"]
    parsed_arguments = parse_evaluation_arguments(arguments=arguments)
    assert vars(parsed_arguments) == {"benchmark_folder": tmp_path / "serving_benchmark",
                                      "runs_folder": tmp_path / "tutorial_runs", "data_folder": tmp_path / "data"}
    assert {name: value for name, value in vars(parse_benchmark_arguments(arguments=arguments)).items()
            if name != "phases"} == vars(parsed_arguments)
    assert evaluate_served_formats.PRECISION_SETTINGS is benchmark_serving_formats.PRECISION_SETTINGS
    assert evaluate_served_formats.MAXIMUM_BATCH_SIZE == benchmark_serving_formats.MAXIMUM_BATCH_SIZE


@pytest.mark.parametrize(argnames="arguments", argvalues=[
    ["--benchmark_folder", "serving_benchmark", "--runs_folder", "tutorial_runs"],
    FOLDER_ARGUMENTS + ["--phases", "triton"],
    FOLDER_ARGUMENTS + ["--maximum_request_size", "8"]])
def test_evaluation_script_refuses_a_missing_folder_or_unknown_values(arguments: List[str],
                                                                      capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the evaluation driver stops with its usage and the error, before anything is evaluated, when a
    folder of the machine is missing, or a phase or a setting of the plan, kept in the constants of the speed sweep,
    is given.

    Args:
        arguments (List[str]): The command line arguments.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_evaluation_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_plan_lists_every_committed_evaluated_row() -> None:
    """
    Verifies that the models and precision settings of the plan of the speed sweep still give every (model, row) pair
    of the committed test_metrics.csv, the eager row and the served rows of every model.
    """
    with open(COMMITTED_METRICS_PATH, newline="") as metrics_file:
        committed_rows = {(line["model"], line["row"]) for line in csv.DictReader(metrics_file)}
    planned_rows = {(model["name"], row_name) for model in BENCHMARKED_MODELS
                    for row_name in [EAGER_ROW_NAME] + [
                        get_row_name(precision_setting=precision_setting, export_format=export_format)
                        for precision_setting in PRECISION_SETTINGS
                        for export_format in get_repository_formats(precision_setting=precision_setting)]}
    assert committed_rows - planned_rows == set()
