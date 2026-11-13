"""
Evaluation tables of the classification tutorials (the chapter judging and shipping a classifier), read at every build
by tutorials/figures/classification/figures.py from the evaluation folders of the classification run of
tutorials/tutorial_sources.yaml: the confusion matrix of every evaluated checkpoint (Evaluation/Iteration_<n>/
confusion_matrix.csv) as the row percentages the grids of the confusion scene draw, the main confusions with their
counts, the per-class metrics and the thresholds at the target precision of the last checkpoint (class_metrics.csv,
class_thresholds.json) as the values and labels of the bars, the headline metrics of every evaluation read from the
Test logs of TensorBoard, the size of the predictions file, and the temperatures of the slider of the temperature
explorer page with the port of the script serving it. Nothing is drawn: the scenes draw these values with the grid and
bars elements of the diagram engine. The labels of the bars write the F1 and recalls as the percentages of the reports
of the repository ("94.03 %"), the facts keeping every rate between 0 and 1.
"""

import csv
import json

import numpy

from pathlib import Path
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import round_fact
from tutorials.figures.foundations.code_values import read_main_block_value
from tutorials.figures.foundations.training_curves import read_scalars
from tutorial_builder.figure_context import find_repository_root
from vision_training.evaluation.classification.temperature_explorer import TEMPERATURE_GRID
from vision_training.evaluation.report_formatting import format_percentage

# Evaluated checkpoints of the classification run whose confusion matrices the scene shows, and the last one, whose
# per-class metrics and thresholds the bars show
CONFUSION_ITERATIONS = [500, 1000, 1500]
LAST_ITERATION = 1500

# Headline metrics of the Test logs of TensorBoard, in the order of the report block, with the names of their facts
HEADLINE_METRICS = {"Accuracy": "accuracy", "Balanced Accuracy": "balanced_accuracy", "Macro F1": "macro_f1",
                    "Top-2 Accuracy": "top_2_accuracy", "Macro ROC AUC": "macro_roc_auc",
                    "Macro Average Precision": "macro_average_precision"}

# Script serving the temperature explorer pages, whose main block sets the port of its local page
TEMPERATURE_EXPLORER_SCRIPT = Path("scripts") / "evaluation" / "serve_temperature_explorer.py"

# Pairs of true and predicted classes whose counts the narration quotes, the errors left at the last checkpoint
NAMED_CONFUSIONS = [("square", "rectangle"), ("rectangle", "square"), ("pentagon", "hexagon"), ("circle", "octagon"),
                    ("octagon", "hexagon")]

# Short names of the classes, the column labels of the confusion grids, too narrow for the full names
SHORT_CLASS_NAMES = {"square": "squ", "circle": "cir", "triangle": "tri", "rectangle": "rec", "pentagon": "pen",
                     "hexagon": "hex", "plus_sign": "plu", "star": "sta", "octagon": "oct"}

# Text of a per-class value that does not exist, the dash of the report block of the repository
MISSING_VALUE_TEXT = "—"


def read_confusion_matrix(evaluation_folder: Path) -> Tuple[List[str], numpy.ndarray]:
    """
    Reads the confusion_matrix.csv written by the evaluation of a checkpoint (save_confusion_matrix_table of the
    repository): the true classes in rows, the predicted classes in columns, in class index order.

    Args:
        evaluation_folder (Path): The Evaluation/Iteration_<n>/ folder.

    Returns:
        Tuple[List[str], numpy.ndarray]: The class names and the [K, K] counts.

    Raises:
        ValueError: If the rows and the columns do not name the same classes.
    """
    with open(file=evaluation_folder / "confusion_matrix.csv", encoding="utf-8", newline="") as table_file:
        rows = list(csv.reader(table_file))
    class_names = rows[0][1:]
    row_names = [row[0] for row in rows[1:]]
    if row_names != class_names:
        raise ValueError(f"The rows {row_names} and columns {class_names} of {evaluation_folder}/confusion_matrix.csv "
                         f"differ.")
    return class_names, numpy.array([[int(value) for value in row[1:]] for row in rows[1:]])


def read_class_metrics(evaluation_folder: Path) -> List[Dict[str, Any]]:
    """
    Reads the class_metrics.csv of an evaluation (write_class_metrics_file of the repository): one row per class in
    class index order, an empty cell for a value that does not exist (a threshold never reaching the target precision).

    Args:
        evaluation_folder (Path): The Evaluation/Iteration_<n>/ folder.

    Returns:
        List[Dict[str, Any]]: The rows, the numbers as floats and the missing values as None.
    """
    with open(file=evaluation_folder / "class_metrics.csv", encoding="utf-8", newline="") as table_file:
        rows = list(csv.DictReader(table_file))
    numeric_columns = ["precision", "recall", "f1", "roc_auc", "average_precision", "threshold_at_target_precision",
                       "recall_at_target_precision"]
    return [{**row, **{column: float(row[column]) if row[column] != "" else None for column in numeric_columns}}
            for row in rows]


def order_by_f1(row: Dict[str, Any]) -> Tuple[float, int]:
    """
    Gives the sort key of a class of the per-class table of the report block: its F1, the worst first, ties in class
    index order.

    Args:
        row (Dict[str, Any]): A row of class_metrics.csv read by read_class_metrics.

    Returns:
        Tuple[float, int]: The F1 and the class index.
    """
    return row["f1"], int(row["class_index"])


def describe_confusion_matrices(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the facts of the confusion matrices of every evaluated checkpoint: the share of every true class predicted
    as every class, in whole percents (the values of the grids), the recall and precision of every class, and the
    counts and shares of the named confusions of the last checkpoint.

    Args:
        run_folder (Path): The folder of the classification run.

    Returns:
        Dict[str, Any]: The facts, named confusion.*, with the evaluated checkpoints shown.
    """
    facts = {"confusion.iterations": CONFUSION_ITERATIONS}
    for iteration in CONFUSION_ITERATIONS:
        class_names, counts = read_confusion_matrix(evaluation_folder=run_folder / "Evaluation" /
                                                    f"Iteration_{iteration}")
        row_shares = counts / counts.sum(axis=1, keepdims=True)
        column_totals = counts.sum(axis=0)
        prefix = f"confusion.iteration_{iteration}"
        facts[f"{prefix}.row_percentages"] = numpy.rint(row_shares * 100).astype(int).tolist()
        facts[f"{prefix}.test_images"] = int(counts.sum())
        for class_index, class_name in enumerate(class_names):
            facts[f"{prefix}.recall.{class_name}"] = round_fact(value=row_shares[class_index, class_index])
            facts[f"{prefix}.precision.{class_name}"] = round_fact(
                value=counts[class_index, class_index] / column_totals[class_index] if column_totals[class_index] else 0)

        # Counts and shares of the true class of the named confusions
        for true_class, predicted_class in NAMED_CONFUSIONS:
            true_index, predicted_index = class_names.index(true_class), class_names.index(predicted_class)
            pair_name = f"{prefix}.{true_class}_to_{predicted_class}"
            facts[f"{pair_name}.count"] = int(counts[true_index, predicted_index])
            facts[f"{pair_name}.share"] = round_fact(value=row_shares[true_index, predicted_index])
    facts["confusion.class_names"] = class_names
    facts["confusion.short_class_names"] = [SHORT_CLASS_NAMES[class_name] for class_name in class_names]
    return facts


def describe_class_metrics(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the facts of the per-class metrics of the last checkpoint, the classes sorted by F1 from the worst like the
    table of the report block: their F1 and their recall at the target precision as bar values and labels (0 and a
    dash for a class never reaching the target precision), and the threshold and recall of every class.

    Args:
        run_folder (Path): The folder of the classification run.

    Returns:
        Dict[str, Any]: The facts, named class_metrics.*.

    Raises:
        ValueError: If class_thresholds.json and class_metrics.csv disagree on a threshold.
    """
    evaluation_folder = run_folder / "Evaluation" / f"Iteration_{LAST_ITERATION}"
    rows = read_class_metrics(evaluation_folder=evaluation_folder)
    thresholds = json.loads((evaluation_folder / "class_thresholds.json").read_text(encoding="utf-8"))

    # Classes from the worst F1, ties in class order, as the report block sorts them
    sorted_rows = sorted(rows, key=order_by_f1)
    facts = {"class_metrics.iteration": LAST_ITERATION, "class_metrics.target_precision": thresholds["target_precision"],
             "class_metrics.sorted_class_names": [row["class"] for row in sorted_rows],
             "class_metrics.sorted_f1": [round_fact(value=row["f1"]) for row in sorted_rows],
             "class_metrics.sorted_f1_labels": [format_percentage(value=row["f1"]) for row in sorted_rows],
             "class_metrics.sorted_recall_at_target": [round_fact(value=row["recall_at_target_precision"] or 0.0)
                                                       for row in sorted_rows],
             "class_metrics.sorted_recall_at_target_labels": [
                 format_percentage(value=row["recall_at_target_precision"])
                 if row["recall_at_target_precision"] is not None else MISSING_VALUE_TEXT for row in sorted_rows]}

    # Every class with its metrics, checked against the thresholds file the deployment reads
    never_reaching = []
    for row in rows:
        class_threshold = thresholds["classes"][row["class"]]
        if (class_threshold is None) != (row["threshold_at_target_precision"] is None) or (
                class_threshold is not None and abs(class_threshold["threshold"] -
                                                    row["threshold_at_target_precision"]) > 1e-5):
            raise ValueError(f"class_thresholds.json and class_metrics.csv of {evaluation_folder} disagree on the "
                             f"threshold of {row['class']}.")
        if class_threshold is None:
            never_reaching.append(row["class"])
        for column in ["samples", "precision", "recall", "f1", "roc_auc", "average_precision",
                       "threshold_at_target_precision", "recall_at_target_precision"]:
            value = row[column]
            facts[f"class_metrics.{row['class']}.{column}"] = int(value) if column == "samples" else (
                round_fact(value=value) if value is not None else None)
    facts["class_metrics.classes_never_reaching_target"] = never_reaching
    return facts


def describe_headline_metrics(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the headline metrics of every evaluated checkpoint, read from the Test logs of TensorBoard (the values of
    the report blocks of Results.txt), and the size of the predictions file of the last checkpoint.

    Args:
        run_folder (Path): The folder of the classification run.

    Returns:
        Dict[str, Any]: The facts, named headline.* and predictions_file.*.
    """
    facts = {}
    for metric_tag, fact_name in HEADLINE_METRICS.items():
        iterations, values = read_scalars(log_folder=run_folder / "Tensorboard" / "Test", tag=metric_tag)
        for iteration, value in zip(iterations, values):
            facts[f"headline.iteration_{int(iteration)}.{fact_name}"] = round_fact(value=value)

    # Rows of the predictions file: one per test image under its header line
    predictions_path = run_folder / "Evaluation" / f"Iteration_{LAST_ITERATION}" / "predictions.csv"
    with open(file=predictions_path, encoding="utf-8") as predictions_file:
        line_count = sum(1 for _ in predictions_file)
    facts.update({"predictions_file.rows": line_count - 1,
                  "predictions_file.megabytes": round_fact(value=predictions_path.stat().st_size / 1e6)})
    return facts


def make_evaluation_tables(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the evaluation tables of the classification run and gives their facts.

    Args:
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named confusion.*, class_metrics.*, headline.*, predictions_file.* and
            temperature_explorer.* (the lowest and highest temperatures of the slider of the explorer page, and the
            port of the local page of serve_temperature_explorer.py).
    """
    facts = describe_confusion_matrices(run_folder=run_folder)
    facts.update(describe_class_metrics(run_folder=run_folder))
    facts.update(describe_headline_metrics(run_folder=run_folder))

    # Temperatures of the slider of the explorer page, and the port its serving script sets in its main block
    facts.update({"temperature_explorer.minimum_temperature": min(TEMPERATURE_GRID),
                  "temperature_explorer.maximum_temperature": max(TEMPERATURE_GRID),
                  "temperature_explorer.port": read_main_block_value(
                      script_path=find_repository_root() / TEMPERATURE_EXPLORER_SCRIPT, variable_name="port")})
    return facts
