"""
Tests of scripts/evaluation/compare_best_evaluations.py, the Markdown tables comparing the best test evaluation of
several trained runs: the best evaluation over the trained and averaged weights, the columns expanded per class and
titled, the curves that are rates written as percentages ("90.00 %") and the others (ranks) with their decimals, and
the counts of data.json. The script prints its help when launched without arguments, reads every setting of
compare_best_evaluations from its command line, DEFAULT_COLUMNS (with the chosen best metric) and
DEFAULT_COLUMN_TITLES for those not given, and refuses missing experiment folders, an unknown value or a column given
two titles before anything is read.
"""

import hashlib
import inspect
import json
import pytest
import yaml

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional

from scripts.evaluation.compare_best_evaluations import (DEFAULT_BEST_METRIC, DEFAULT_COLUMN_TITLES, DEFAULT_COLUMNS,
                                                         compare_best_evaluations, parse_comparison_arguments)

# Classes of the fake runs, with their category identifiers in the labels files and the label mapping
FAKE_CATEGORIES = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]

# Columns of the default comparison of the tests, and their titles
COLUMNS = ["configuration:Data.image_settings.size", "best_iteration", "Balanced Accuracy", "Recall/{class}",
           "validation_images/{class}", "train_images"]
COLUMN_TITLES = {"configuration:Data.image_settings.size": "Image Size", "Recall/{class}": "Recall {class}",
                 "validation_images/{class}": "Validation {class}"}

# Every argument of the comparison script, which its help must describe
COMPARISON_SCRIPT_ARGUMENTS = ["--experiment_folders", "--best_metric", "--lower_is_better", "--evaluated_weights",
                               "--columns", "--column_titles", "--table_grouping", "--number_decimals",
                               "--rates_as_decimals", "--output_path"]


def write_test_curves(test_folder: Path, curves: Dict[str, Dict[int, float]]) -> None:
    """
    Writes test curves in a Tensorboard folder of a fake run, as the trainer writes them at every evaluation.

    Args:
        test_folder (Path): The Tensorboard/Test or Tensorboard/Test_EMA folder.
        curves (Dict[str, Dict[int, float]]): The value at every evaluated iteration of every curve.
    """
    writer = SummaryWriter(log_dir=str(test_folder))
    for curve_name, curve_values in curves.items():
        for iteration, value in curve_values.items():
            writer.add_scalar(tag=curve_name, scalar_value=value, global_step=iteration)
    writer.close()


def write_fake_run(run_folder: Path, data_folder: Path, image_size: int, trained_curves: Dict[str, Dict[int, float]],
                   averaged_curves: Optional[Dict[str, Dict[int, float]]], counted_classes: bool) -> None:
    """
    Writes a fake classification run folder: its training configuration, its test curves, and its data.json with the
    counted classes, or with the class names only as the records written before the classes were counted, its splits
    being labels files of 3 cats and 1 dog (train) and 1 cat and 2 dogs (validation, test).

    Args:
        run_folder (Path): The run folder.
        data_folder (Path): The folder receiving the train, validation and test splits and the label mapping.
        image_size (int): The image size of the configuration.
        trained_curves (Dict[str, Dict[int, float]]): The curves of the trained weights.
        averaged_curves (Optional[Dict[str, Dict[int, float]]]): The curves of the averaged weights, None without.
        counted_classes (bool): True for a data.json with 'classes', False for one with 'class_names'.
    """
    split_categories = {"train": [1, 1, 1, 2], "validation": [1, 2, 2], "test": [1, 2, 2]}
    split_records = {}
    for split_name, category_identifiers in split_categories.items():
        (data_folder / split_name).mkdir(parents=True, exist_ok=True)
        labels = {"categories": FAKE_CATEGORIES,
                  "samples": [{"id": index, "file_name": f"{index}.png", "category_id": category_identifier}
                              for index, category_identifier in enumerate(category_identifiers)]}
        (data_folder / split_name / "labels.json").write_text(data=json.dumps(obj=labels))
        split_records[split_name] = {
            "data_folder": str(data_folder / split_name),
            "labels_sha256": hashlib.sha256((data_folder / split_name / "labels.json").read_bytes()).hexdigest(),
            "number_images": len(category_identifiers),
            "images": [f"{index}.png" for index in range(len(category_identifiers))]}
    (data_folder / "label_mapping.json").write_text(data=json.dumps(obj={"1": 0, "2": 1}))

    # Configuration, counted classes or class names, and test curves of the run
    run_folder.mkdir(parents=True)
    configuration = {"ExperimentConfiguration": {"task": "classification"},
                     "Data": {"image_settings": {"size": image_size},
                              "label_mapping": {"mode": "multi_class",
                                                "mapping_file_path": str(data_folder / "label_mapping.json")}}}
    (run_folder / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=configuration))
    class_records = [
        {"class_index": 0, "class_name": "cat", "category_ids": [1],
         "images": {"train": 3, "validation": 1, "test": 1}},
        {"class_index": 1, "class_name": "dog", "category_ids": [2],
         "images": {"train": 1, "validation": 2, "test": 2}}]
    classes_entry = {"classes": class_records} if counted_classes else {"class_names": ["cat", "dog"]}
    (run_folder / "data.json").write_text(data=json.dumps(obj={**classes_entry, "splits": split_records}))
    write_test_curves(test_folder=run_folder / "Tensorboard" / "Test", curves=trained_curves)
    if averaged_curves is not None:
        write_test_curves(test_folder=run_folder / "Tensorboard" / "Test_EMA", curves=averaged_curves)


def build_curves(balanced_accuracies: Dict[int, float]) -> Dict[str, Dict[int, float]]:
    """
    Builds the test curves of a fake run from its balanced accuracy at every iteration, the recall of the cats being
    the balanced accuracy plus 0.05 and the one of the dogs the balanced accuracy minus 0.05.

    Args:
        balanced_accuracies (Dict[int, float]): The balanced accuracy at every evaluated iteration.

    Returns:
        Dict[str, Dict[int, float]]: The 'Balanced Accuracy', 'Recall/cat' and 'Recall/dog' curves.
    """
    return {"Balanced Accuracy": balanced_accuracies,
            "Recall/cat": {iteration: value + 0.05 for iteration, value in balanced_accuracies.items()},
            "Recall/dog": {iteration: value - 0.05 for iteration, value in balanced_accuracies.items()}}


def run_comparison(experiment_folders: List[Path], columns: List[str], table_grouping: str = "run",
                   evaluated_weights: Optional[List[str]] = None, curves_in_percent: bool = True) -> str:
    """
    Runs the comparison with the balanced accuracy as best metric and the titles of the tests.

    Args:
        experiment_folders (List[Path]): The experiment or run folders.
        columns (List[str]): The columns, in their order of appearance.
        table_grouping (str): "run" or "columns". Defaults to "run".
        evaluated_weights (Optional[List[str]]): The competing weights, trained and averaged when None.
        curves_in_percent (bool): Whether the curves are written as percentages, as for the classification, or with 3
            decimals. Defaults to True.

    Returns:
        str: The Markdown tables.
    """
    return compare_best_evaluations(experiment_folders=experiment_folders, best_metric="Balanced Accuracy",
                                    higher_is_better=True, evaluated_weights=evaluated_weights or ["trained", "ema"],
                                    columns=columns, column_titles=COLUMN_TITLES, table_grouping=table_grouping,
                                    number_decimals=3, curves_in_percent=curves_in_percent)


def test_best_evaluation_over_trained_and_averaged_weights(tmp_path: Path) -> None:
    """
    Verifies the table of one run: the averaged weights winning at iteration 400 give '400 (EMA)', the curves at that
    evaluation, the per-class columns expanded in class order and titled, the setting of the configuration, and the
    counts of data.json; with the trained weights only, the best trained evaluation, a tie keeping the earliest
    iteration.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    run_folder = tmp_path / "experiments" / "run_256"
    write_fake_run(run_folder=run_folder, data_folder=tmp_path / "data", image_size=256,
                   trained_curves=build_curves(balanced_accuracies={200: 0.8, 400: 0.85, 600: 0.85}),
                   averaged_curves=build_curves(balanced_accuracies={200: 0.75, 400: 0.9, 600: 0.7}),
                   counted_classes=True)

    assert run_comparison(experiment_folders=[tmp_path / "experiments"], columns=COLUMNS).splitlines() == [
        "### experiments · run_256", "",
        "| Image Size | best_iteration | Balanced Accuracy | Recall cat | Recall dog | Validation cat | Validation dog "
        "| train_images |",
        "|---|---|---|---|---|---|---|---|",
        "| 256 | 400 (EMA) | 90.00 % | 95.00 % | 85.00 % | 1 | 2 | 4 |"]
    trained_table = run_comparison(experiment_folders=[run_folder], columns=["best_iteration", "Balanced Accuracy"],
                                   evaluated_weights=["trained"])
    assert trained_table.splitlines()[-1] == "| 400 | 85.00 % |"

    # Every curve with its decimals when the percentages are turned off
    decimal_table = run_comparison(experiment_folders=[run_folder], columns=["best_iteration", "Balanced Accuracy"],
                                   evaluated_weights=["trained"], curves_in_percent=False)
    assert decimal_table.splitlines()[-1] == "| 400 | 0.850 |"


def test_curves_that_are_not_rates_keep_their_decimals(tmp_path: Path) -> None:
    """
    Verifies that the curves that are not rates keep their decimals next to the rates written as percentages: the mean
    rank of the true frames of the frame selection, global and per class, and the effective rank of the
    self-supervised task, whose values above 1 would otherwise be written "125.00 %".

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    run_folder = tmp_path / "experiments" / "run_256"
    trained_curves = {**build_curves(balanced_accuracies={200: 0.8, 400: 0.85}),
                      "Mean True Frame Rank": {200: 1.5, 400: 1.25}, "Mean True Frame Rank/cat": {200: 1.0, 400: 1.0},
                      "Mean True Frame Rank/dog": {200: 2.0, 400: 1.5}, "Effective Rank": {200: 10.5, 400: 12.34}}
    write_fake_run(run_folder=run_folder, data_folder=tmp_path / "data", image_size=256, trained_curves=trained_curves,
                   averaged_curves=None, counted_classes=True)

    table = run_comparison(experiment_folders=[run_folder], evaluated_weights=["trained"],
                           columns=["best_iteration", "Balanced Accuracy", "Recall/{class}", "Mean True Frame Rank",
                                    "Mean True Frame Rank/{class}", "Effective Rank"])
    assert table.splitlines()[-1] == "| 400 | 85.00 % | 90.00 % | 80.00 % | 1.250 | 1.000 | 1.500 | 12.340 |"


def test_comparison_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the comparison script launched without any argument prints the help of every one of its arguments
    and stops without comparing, instead of failing on the missing experiment folders.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_comparison_arguments(arguments=[])
    assert exit_information.value.code == 0
    help_text = capsys.readouterr().out
    assert [argument for argument in COMPARISON_SCRIPT_ARGUMENTS if argument not in help_text] == []


def test_comparison_arguments_default_to_the_default_columns() -> None:
    """
    Verifies that the comparison script given only the experiment folders compares them by DEFAULT_BEST_METRIC, the
    highest value winning, over the trained and averaged weights, with DEFAULT_COLUMNS and DEFAULT_COLUMN_TITLES, one
    table per run and the rates as percentages, printed only; that a chosen best metric takes the place of the default
    one in the default columns but not in columns given; and that every parsed argument but output_path is a parameter
    of compare_best_evaluations, so that the main block passes them all.
    """
    parsed_arguments = parse_comparison_arguments(arguments=["--experiment_folders", "experiments"])
    assert vars(parsed_arguments) == {"experiment_folders": [Path("experiments")], "best_metric": DEFAULT_BEST_METRIC,
                                      "higher_is_better": True, "evaluated_weights": ["trained", "ema"],
                                      "columns": DEFAULT_COLUMNS, "column_titles": DEFAULT_COLUMN_TITLES,
                                      "table_grouping": "run", "number_decimals": 3, "curves_in_percent": True,
                                      "output_path": None}
    assert set(vars(parsed_arguments)) - {"output_path"} == set(
        inspect.signature(compare_best_evaluations).parameters)

    # The chosen best metric in the place of the default one, in the default columns only
    macro_arguments = parse_comparison_arguments(arguments=["--experiment_folders", "experiments", "--best_metric",
                                                            "Macro F1"])
    assert macro_arguments.columns == ["Macro F1" if column == DEFAULT_BEST_METRIC else column
                                       for column in DEFAULT_COLUMNS]
    assert "Macro F1" in macro_arguments.columns and DEFAULT_BEST_METRIC not in macro_arguments.columns
    given_arguments = parse_comparison_arguments(arguments=["--experiment_folders", "experiments", "--best_metric",
                                                            "Macro F1", "--columns", "run", DEFAULT_BEST_METRIC])
    assert given_arguments.columns == ["run", DEFAULT_BEST_METRIC]


def test_comparison_arguments_reach_every_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that every setting of compare_best_evaluations and the output file can be given on the command line of
    the script: the columns in their order, titles given as COLUMN=TITLE replacing those of their columns only (a
    title holding '=' kept whole), the weights alone, and paths starting with a '~' the shell left unexpanded (after
    '=' or quoted) pointing to the home folder.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    parsed_arguments = parse_comparison_arguments(arguments=[
        "--experiment_folders", "~/experiments", "runs/run_256", "--best_metric", "Validation Loss",
        "--lower_is_better", "--evaluated_weights", "ema", "--columns", "run", "Validation Loss", "Recall/{class}",
        "--column_titles",
        "Recall/{class}=Recall {class}", "run=Run = Folder", "--table_grouping", "columns", "--number_decimals", "5",
        "--rates_as_decimals", "--output_path=~/tables.md"])
    assert vars(parsed_arguments) == {"experiment_folders": [tmp_path / "experiments", Path("runs/run_256")],
                                      "best_metric": "Validation Loss", "higher_is_better": False,
                                      "evaluated_weights": ["ema"], "columns": ["run", "Validation Loss",
                                                                               "Recall/{class}"],
                                      "column_titles": {**DEFAULT_COLUMN_TITLES, "Recall/{class}": "Recall {class}",
                                                        "run": "Run = Folder"},
                                      "table_grouping": "columns", "number_decimals": 5, "curves_in_percent": False,
                                      "output_path": tmp_path / "tables.md"}


def test_parsed_comparison_arguments_compare_the_runs(tmp_path: Path) -> None:
    """
    Verifies that the arguments read from a command line compare the runs of an experiment folder as the main block
    passes them, a title given replacing that of its column only, the other columns keeping their default titles.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_fake_run(run_folder=tmp_path / "experiments" / "run_256", data_folder=tmp_path / "data", image_size=256,
                   trained_curves=build_curves(balanced_accuracies={200: 0.8, 400: 0.85}),
                   averaged_curves=build_curves(balanced_accuracies={200: 0.75, 400: 0.9}), counted_classes=True)
    parsed_arguments = parse_comparison_arguments(arguments=[
        "--experiment_folders", str(tmp_path / "experiments"), "--evaluated_weights", "trained", "--columns",
        "best_iteration", "Balanced Accuracy", "Recall/{class}", "--column_titles", "best_iteration=Iteration"])

    table = compare_best_evaluations(**{name: value for name, value in vars(parsed_arguments).items()
                                        if name != "output_path"})

    assert table.splitlines()[2:] == ["| Iteration | Balanced Accuracy | Recall cat | Recall dog |",
                                      "|---|---|---|---|", "| 400 | 85.00 % | 90.00 % | 80.00 % |"]


@pytest.mark.parametrize(argnames="arguments", argvalues=[
    ["--best_metric", "Accuracy"],
    ["--experiment_folders"],
    ["--experiment_folders", "experiments", "--evaluated_weights", "averaged"],
    ["--experiment_folders", "experiments", "--table_grouping", "experiment"],
    ["--experiment_folders", "experiments", "--number_decimals", "three"],
    ["--experiment_folders", "experiments", "--column_titles", "Recall/{class}"],
    ["--experiment_folders", "experiments", "--column_titles", "=Run"],
    ["--experiment_folders", "experiments", "--column_titles", "run=Run", "run=Folder"]])
def test_comparison_script_refuses_missing_folders_or_unknown_values(arguments: List[str],
                                                                     capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the comparison script stops with its usage and the error, before anything is read, when the
    experiment folders are missing, weights, a grouping or a number is not one the comparison accepts, or a title is
    not given as COLUMN=TITLE or is given twice for a column.

    Args:
        arguments (List[str]): The command line arguments.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_comparison_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert "error:" in capsys.readouterr().err
