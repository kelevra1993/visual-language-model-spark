"""
Tests of scripts/evaluation/collect_experiment_results.py, which gathers the launches of a hyperparameter experiment
into the results file read by the experiment plans and their figures: the settings given as lists (the [layers,
channels] of the mask heads and the backbone blocks of their skips in the instance segmentation experiment) group the
launches by their values like the scalar settings, one mean row per setting, and are written to the file as they are
in the configuration. The script prints its help when launched without arguments, reads the task, the experiment
folder, the number of last evaluations and the results file from its command line, the folder of the task and its
results.csv for those not given, refuses a missing task or an unknown value, and writes the results file its arguments
name.
"""

import csv
import pytest
import runpy
import sys
import yaml

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List

from scripts.evaluation import collect_experiment_results
from scripts.evaluation.collect_experiment_results import (DEFAULT_NUMBER_LAST_EVALUATIONS, RESULTS_FILE_NAME,
                                                           TASK_EXPERIMENT_FOLDERS, TASK_METRIC_COLUMNS, TASK_METRICS,
                                                           TASK_SWEPT_SETTINGS, build_result_rows, make_hashable,
                                                           parse_collection_arguments, write_results)

# Every argument of the collection script, which its help must describe
COLLECTION_SCRIPT_ARGUMENTS = ["--task", "--experiment_folder", "--number_last_evaluations", "--output_path"]

# Swept settings of the fake experiment: a scalar one, the convolutions of the mask heads and the skips of their
# finer inputs, both lists, and the averaged weights flag every experiment reads
SWEPT_SETTINGS = {"learning_rate": ("ExperimentConfiguration", "learning_rate"),
                  "mask_convolutions": ("DetectionLevels", "4", "mask_head", "convolutions"),
                  "mask_skip_blocks_at_pool_size": ("MaskHead", "skip_blocks_at_pool_size"),
                  "exponential_moving_average": ("Optimization", "exponential_moving_average", "activated")}


def make_launch(launch_name: str, mask_convolutions: List[int], skip_blocks: List[int],
                evaluations: Dict[int, float]) -> Dict[str, Any]:
    """
    Builds a fake launch as collect_launches returns it, so that the grouping of build_result_rows can be checked
    without training folders.

    Args:
        launch_name (str): The launch, e.g. "run_7".
        mask_convolutions (List[int]): The [layers, channels] of its mask heads.
        skip_blocks (List[int]): The backbone blocks of its skips at the pool size, empty for none.
        evaluations (Dict[int, float]): Its test metric at every evaluated iteration.

    Returns:
        Dict[str, Any]: The launch, finished, with its trained weights.
    """
    settings = {"learning_rate": 0.001, "mask_convolutions": mask_convolutions,
                "mask_skip_blocks_at_pool_size": skip_blocks, "exponential_moving_average": True,
                "evaluated_weights": "trained"}
    return {"launch": launch_name, "evaluations": evaluations, "finished": True, "settings": settings,
            "run_folder": f"INS_{launch_name}"}


# Two launches of the reference mask heads, two of larger heads (the lists built anew, as every configuration file
# gives its own), and one of the larger heads with skips
LAUNCHES = [make_launch(launch_name="run_1", mask_convolutions=[3, 32], skip_blocks=[],
                        evaluations={1000: 0.70, 2000: 0.80}),
            make_launch(launch_name="run_2", mask_convolutions=[3, 32], skip_blocks=[],
                        evaluations={1000: 0.72, 2000: 0.82}),
            make_launch(launch_name="run_7", mask_convolutions=[4, 64], skip_blocks=[],
                        evaluations={1000: 0.74, 2000: 0.84}),
            make_launch(launch_name="run_8", mask_convolutions=[4, 64], skip_blocks=[],
                        evaluations={1000: 0.76, 2000: 0.86}),
            make_launch(launch_name="run_11", mask_convolutions=[4, 64], skip_blocks=[4],
                        evaluations={1000: 0.78, 2000: 0.88})]


def test_list_settings_compare_by_value() -> None:
    """
    The hashable form of a setting is equal for equal lists or mappings built separately, differs for other values,
    and leaves the scalar settings as they are, so that the grouping of the launches only changes for list settings.
    """
    assert make_hashable(value=[4, 64]) == make_hashable(value=[4, 64]) == (4, 64)
    assert hash(make_hashable(value=[[1, 2], [3]])) == hash(((1, 2), (3,)))
    assert make_hashable(value=[4, 64]) != make_hashable(value=[3, 32])
    assert make_hashable(value=[]) != make_hashable(value=[4])
    assert make_hashable(value={"convolutions": [4, 64], "roi_align_pool_size": 14}) == \
        make_hashable(value={"roi_align_pool_size": 14, "convolutions": [4, 64]})
    assert make_hashable(value=0.001) == 0.001 and make_hashable(value="batch") == "batch"
    assert make_hashable(value=None) is None and make_hashable(value=True) is True


def test_launches_with_list_settings_get_one_mean_row_per_setting() -> None:
    """
    The launches of the same list settings share one mean row whatever the list objects, those of other lists get
    their own, in the order of their first launch, with the mean of their launches and their settings unchanged.
    """
    columns, rows = build_result_rows(launches=LAUNCHES, metric_column="mask_map_50_95", number_last_evaluations=1,
                                      swept_settings=SWEPT_SETTINGS)
    assert [row["launch"] for row in rows] == ["run_1", "run_2", "mean", "run_7", "run_8", "mean", "run_11", "mean"]

    # Mean rows of the three settings, carrying the settings as lists
    mean_rows = [row for row in rows if row["launch"] == "mean"]
    assert [row["mask_convolutions"] for row in mean_rows] == [[3, 32], [4, 64], [4, 64]]
    assert [row["mask_skip_blocks_at_pool_size"] for row in mean_rows] == [[], [], [4]]
    assert [round(row["mask_map_50_95_2000"], 6) for row in mean_rows] == [0.81, 0.85, 0.88]
    assert [round(row["mask_map_50_95_last_1_mean"], 6) for row in mean_rows] == [0.81, 0.85, 0.88]
    assert columns[-5:] == [*SWEPT_SETTINGS, "run_folder"]


def test_results_file_writes_list_settings_as_in_the_configuration(tmp_path: Path) -> None:
    """
    The results file writes the list settings as the configuration gives them, so that a spreadsheet, the experiment
    plan and its figures read them back unchanged.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the results file.
    """
    columns, rows = build_result_rows(launches=LAUNCHES, metric_column="mask_map_50_95", number_last_evaluations=1,
                                      swept_settings=SWEPT_SETTINGS)
    output_path = tmp_path / "results.csv"
    write_results(columns=columns, rows=rows, output_path=output_path, swept_settings=SWEPT_SETTINGS)
    with output_path.open(encoding="utf-8", newline="") as results_file:
        written_rows = list(csv.DictReader(f=results_file))
    assert [row["mask_convolutions"] for row in written_rows] == ["[3, 32]"] * 3 + ["[4, 64]"] * 5
    assert [row["mask_skip_blocks_at_pool_size"] for row in written_rows] == ["[]"] * 6 + ["[4]"] * 2
    assert [row["mask_map_50_95_2000"] for row in written_rows][:3] == ["0.800000", "0.820000", "0.810000"]


def test_collection_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the collection script launched without any argument prints the help of every one of its arguments
    and stops without reading any launch, instead of failing on the missing task.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_collection_arguments(arguments=[])
    assert exit_information.value.code == 0
    help_text = capsys.readouterr().out
    assert [argument for argument in COLLECTION_SCRIPT_ARGUMENTS if argument not in help_text] == []


@pytest.mark.parametrize(argnames="task", argvalues=list(TASK_SWEPT_SETTINGS))
def test_collection_arguments_default_to_the_folder_of_the_task(task: str) -> None:
    """
    Verifies that the collection script given only the task reads the experiment folder of the task in the
    hyperparameter_experiments folder of the repository (TASK_EXPERIMENT_FOLDERS), averages the
    DEFAULT_NUMBER_LAST_EVALUATIONS last evaluations and writes the RESULTS_FILE_NAME of that folder, and that every
    task has its folder, metric and metric column.

    Args:
        task (str): The task of the experiment.
    """
    parsed_arguments = parse_collection_arguments(arguments=["--task", task])
    assert vars(parsed_arguments) == {"task": task, "experiment_folder": TASK_EXPERIMENT_FOLDERS[task],
                                      "number_last_evaluations": DEFAULT_NUMBER_LAST_EVALUATIONS,
                                      "output_path": TASK_EXPERIMENT_FOLDERS[task] / RESULTS_FILE_NAME}
    assert set(TASK_EXPERIMENT_FOLDERS) == set(TASK_METRICS) == set(TASK_METRIC_COLUMNS) == set(TASK_SWEPT_SETTINGS)
    repository_folder = Path(collect_experiment_results.__file__).absolute().parents[2]
    assert TASK_EXPERIMENT_FOLDERS[task].is_relative_to(repository_folder / "hyperparameter_experiments")


def test_collection_arguments_reach_every_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that every setting of the collection can be given on the command line, paths starting with a '~' the shell
    leaves after '=' being expanded to the home folder, and that a given experiment folder without a results file
    writes the RESULTS_FILE_NAME of that folder.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    parsed_arguments = parse_collection_arguments(arguments=[
        "--task", "classification", "--experiment_folder=~/experiment", "--number_last_evaluations", "2",
        "--output_path=~/stage_results.csv"])
    assert vars(parsed_arguments) == {"task": "classification", "experiment_folder": tmp_path / "experiment",
                                      "number_last_evaluations": 2, "output_path": tmp_path / "stage_results.csv"}
    parsed_arguments = parse_collection_arguments(arguments=["--task", "object_detection",
                                                             "--experiment_folder=~/experiment"])
    assert parsed_arguments.output_path == tmp_path / "experiment" / RESULTS_FILE_NAME


@pytest.mark.parametrize(argnames="arguments", argvalues=[
    ["--experiment_folder", "experiment"],
    ["--task", "instance_segmentation"],
    ["--task", "classification", "--number_last_evaluations", "four"]])
def test_collection_script_refuses_a_missing_task_or_unknown_values(arguments: List[str],
                                                                    capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the collection script stops with its usage and the error, before any launch is read, when the task
    is missing or is not one of TASK_SWEPT_SETTINGS, or the number of last evaluations is not a number.

    Args:
        arguments (List[str]): The command line arguments.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_collection_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_collection_script_writes_the_results_file_of_its_arguments(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the main block of the collection script passes every argument to the collection: two launches of
    the same settings of a detection experiment are written with their mean row and the mean of their last evaluation
    to the results file of --output_path, nothing being written in the experiment folder.

    Args:
        tmp_path (Path): The pytest temporary directory, holding the experiment and the results file.
        monkeypatch (pytest.MonkeyPatch): Sets the command line of the script.
    """
    # Two launches of the same settings, each with the configuration and the test curve of its run folder
    experiment_folder = tmp_path / "experiment"
    for launch_name, curve in [("run_1", {500: 0.25, 1000: 0.5}), ("run_2", {500: 0.75, 1000: 1.0})]:
        run_folder = experiment_folder / launch_name / f"DET_{launch_name}"
        run_folder.mkdir(parents=True)
        configuration = {"ExperimentConfiguration": {"learning_rate": 0.001, "number_iterations": 1000},
                         "Optimization": {"exponential_moving_average": {"activated": False}}}
        (run_folder / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=configuration),
                                                                encoding="utf-8")
        writer = SummaryWriter(log_dir=str(run_folder / "Tensorboard" / "Test"))
        for iteration, value in curve.items():
            writer.add_scalar(tag=TASK_METRICS["object_detection"], scalar_value=value, global_step=iteration)
        writer.close()

    # The script run as a module would run it, with every argument given
    output_path = tmp_path / "stage_results.csv"
    monkeypatch.setattr(target=sys, name="argv", value=[
        collect_experiment_results.__file__, "--task", "object_detection", "--experiment_folder",
        str(experiment_folder), "--number_last_evaluations", "1", "--output_path", str(output_path)])
    runpy.run_path(path_name=collect_experiment_results.__file__, run_name="__main__")

    with output_path.open(encoding="utf-8", newline="") as results_file:
        written_rows = list(csv.DictReader(f=results_file))
    assert [row["launch"] for row in written_rows] == ["run_1", "run_2", "mean"]
    assert [row["map_50_95_last_1_mean"] for row in written_rows] == ["0.500000", "1.000000", "0.750000"]
    assert [row["finished"] for row in written_rows] == ["True", "True", "True"]
    assert not (experiment_folder / RESULTS_FILE_NAME).exists()
