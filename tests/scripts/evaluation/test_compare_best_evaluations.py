"""
Tests of scripts/evaluation/compare_best_evaluations.py, the Markdown tables comparing the best test evaluation of
several trained runs: the best evaluation over the trained and averaged weights, the columns expanded per class and
titled, and the counts of data.json.
"""

import hashlib
import json
import yaml

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional

from scripts.evaluation.compare_best_evaluations import compare_best_evaluations

# Classes of the fake runs, with their category identifiers in the labels files and the label mapping
FAKE_CATEGORIES = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]

# Columns of the default comparison of the tests, and their titles
COLUMNS = ["configuration:Data.image_settings.size", "best_iteration", "Balanced Accuracy", "Recall/{class}",
           "validation_images/{class}", "train_images"]
COLUMN_TITLES = {"configuration:Data.image_settings.size": "Image Size", "Recall/{class}": "Recall {class}",
                 "validation_images/{class}": "Validation {class}"}


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
                   evaluated_weights: Optional[List[str]] = None) -> str:
    """
    Runs the comparison with the balanced accuracy as best metric and the titles of the tests.

    Args:
        experiment_folders (List[Path]): The experiment or run folders.
        columns (List[str]): The columns, in their order of appearance.
        table_grouping (str): "run" or "columns". Defaults to "run".
        evaluated_weights (Optional[List[str]]): The competing weights, trained and averaged when None.

    Returns:
        str: The Markdown tables.
    """
    return compare_best_evaluations(experiment_folders=experiment_folders, best_metric="Balanced Accuracy",
                                    higher_is_better=True, evaluated_weights=evaluated_weights or ["trained", "ema"],
                                    columns=columns, column_titles=COLUMN_TITLES, table_grouping=table_grouping,
                                    number_decimals=3)


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
        "| 256 | 400 (EMA) | 0.900 | 0.950 | 0.850 | 1 | 2 | 4 |"]
    trained_table = run_comparison(experiment_folders=[run_folder], columns=["best_iteration", "Balanced Accuracy"],
                                   evaluated_weights=["trained"])
    assert trained_table.splitlines()[-1] == "| 400 | 0.850 |"
