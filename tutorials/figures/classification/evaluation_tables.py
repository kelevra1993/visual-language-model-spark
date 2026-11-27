"""
Evaluation tables of the classification tutorials (the chapter judging and shipping a classifier), read at every build
by tutorials/figures/classification/figures.py from the evaluation folders of the classification run of
tutorials/tutorial_sources.yaml: the confusion matrix of every evaluated checkpoint (Evaluation/Iteration_<n>/
confusion_matrix.csv), of which the confusion scene draws the rows and columns of the classes the largest confusions
involve as row percentages, the largest confusions of the last checkpoint and the weakest classes of the first one,
named by their role so that another run names other classes without any edit, the per-class metrics and the thresholds
at the target precision of the last checkpoint (class_metrics.csv, class_thresholds.json) as the values and labels of
the bars of its worst and best classes, the headline metrics of every evaluation read from the Test logs of
TensorBoard, those of the standalone evaluation of the last checkpoint (classification_evaluation of
tutorial_sources.yaml, written by scripts/evaluation/evaluate_checkpoint.py), whose report block the report scene
shows, the size of the predictions file, and the temperatures of the slider of the temperature explorer page with the
default port of the script serving it (DEFAULT_PORT). Nothing is drawn: the scenes draw these values with the grid and
bars elements of the diagram engine. The labels of the bars write the F1 and recalls as the percentages of the reports
of the repository ("94.03 %"), the facts keeping every rate between 0 and 1. A build from the run snapshots reads the
reduced copy of the predictions file, the size of the original coming from the manifest of the series.
"""

import csv
import json

import numpy

from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.evaluation.serve_temperature_explorer import DEFAULT_PORT
from tutorials.figures.classification.snapshot_helpers import measure_predictions_bytes, open_predictions_file
from tutorials.figures.figure_utilities import round_fact
from tutorials.figures.foundations.training_curves import read_scalars
from vision_training.evaluation.classification.classification_metrics import TOP_K_ACCURACY_CLASSES
from vision_training.evaluation.classification.temperature_explorer import TEMPERATURE_GRID
from vision_training.evaluation.report_formatting import format_percentage

# Evaluated checkpoints of the classification run whose confusion matrices the scene shows (the first, one in the
# middle and the last), and the last one, whose per-class metrics, thresholds, predictions and standalone evaluation the
# chapter reads; the run must hold no evaluation after it
CONFUSION_ITERATIONS = [500, 1000, 2000]
LAST_ITERATION = 2000

# Headline metrics of the Test logs of TensorBoard, in the order of the report block, with the names of their facts:
# the top-k accuracy counts the TOP_K_ACCURACY_CLASSES most probable classes
HEADLINE_METRICS = {"Accuracy": "accuracy", "Balanced Accuracy": "balanced_accuracy", "Macro F1": "macro_f1",
                    f"Top-{TOP_K_ACCURACY_CLASSES} Accuracy": "top_k_accuracy", "Macro ROC AUC": "macro_roc_auc",
                    "Macro Average Precision": "macro_average_precision"}

# Largest confusions of the last checkpoint named by the facts (pairs of a true and a predicted class, the largest
# first), weakest classes of the first checkpoint (lowest recall first), and the most classes of the confusion grids,
# those of these pairs and weakest classes, in class index order
NAMED_PAIR_COUNT = 4
WEAKEST_CLASS_COUNT = 3
MAXIMUM_FOCUS_CLASSES = 10

# Classes of the bars of the per-class scene: the worst and the best by F1, the worst first like the report block
WORST_BAR_CLASSES = 8
BEST_BAR_CLASSES = 4

# Letters of a short class name per word of its name, the column labels of the grids, too narrow for the full names
SHORT_NAME_LETTERS = 3

# Shapes with four sides among the synthetic shapes (filled, or hollow with the prefix below): the narration calls the
# largest confusions and the worst classes quadrilaterals, which the facts check
FOUR_SIDED_SHAPES = ["square", "rectangle", "rhombus", "parallelogram", "trapezoid", "kite"]
HOLLOW_PREFIX = "hollow_"

# Text of a per-class value that does not exist, the dash of the report block of the repository
MISSING_VALUE_TEXT = "—"


def format_class_name(class_name: str) -> str:
    """
    Writes a class name of the synthetic shapes as the narration and the screen texts read it: words separated by
    spaces, a one-letter word in capitals ("l_shape" as "L shape", "hollow_pentagon" as "hollow pentagon").

    Args:
        class_name (str): The class name of the label dictionary.

    Returns:
        str: The name to read.
    """
    return " ".join(word.upper() if len(word) == 1 else word for word in class_name.split("_"))


def check_four_sided(class_names: List[str], role: str) -> None:
    """
    Checks that every class of a group the narration calls quadrilaterals (the largest confusions, the worst
    classes) is a four-sided shape, filled or hollow, so that another run stops the build instead of narrating a false
    sentence.

    Args:
        class_names (List[str]): The class names of the group.
        role (str): What the group is, for the error message.

    Raises:
        ValueError: If a class of the group is not a four-sided shape.
    """
    other_classes = [name for name in class_names if name.removeprefix(HOLLOW_PREFIX) not in FOUR_SIDED_SHAPES]
    if other_classes:
        raise ValueError(f"The {role} {class_names} hold shapes that are not four-sided, {other_classes}: reword the "
                         f"narration of 02_judging_and_shipping.yaml that calls them quadrilaterals.")


def join_class_names(class_names: List[str]) -> str:
    """
    Writes a list of classes as the narration reads it, "trapezoid, hollow parallelogram and hollow kite".

    Args:
        class_names (List[str]): The class names, as format_class_name writes them.

    Returns:
        str: The names joined by commas, the last one by "and".
    """
    if len(class_names) <= 1:
        return "".join(class_names)
    return ", ".join(class_names[:-1]) + " and " + class_names[-1]


def build_short_class_names(class_names: List[str]) -> List[str]:
    """
    Gives the column labels of the confusion grids: the first SHORT_NAME_LETTERS letters of the last word of every
    class, after the first letter of every word before it ("hollow_pentagon" as "hpen", "plus_sign" as "psig"), refused
    when two classes would get the same label.

    Args:
        class_names (List[str]): The class names.

    Returns:
        List[str]: The short names, in the same order.

    Raises:
        ValueError: If two classes get the same short name.
    """
    short_names = []
    for class_name in class_names:
        words = class_name.split("_")
        short_names.append("".join(word[0] for word in words[:-1]) + words[-1][:SHORT_NAME_LETTERS])
    repeated_names = sorted({name for name in short_names if short_names.count(name) > 1})
    if repeated_names:
        raise ValueError(f"The short class names {repeated_names} name several classes of {class_names}.")
    return short_names


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


def check_last_iteration(run_folder: Path) -> None:
    """
    Checks that LAST_ITERATION is the last evaluated checkpoint of the run and that every iteration of
    CONFUSION_ITERATIONS was evaluated, so that a run trained longer or cut elsewhere stops the build instead of
    narrating another checkpoint as the last one.

    Args:
        run_folder (Path): The folder of the classification run.

    Raises:
        ValueError: If the evaluated checkpoints of the run do not match these constants.
    """
    evaluated_iterations = sorted(int(folder.name.removeprefix("Iteration_"))
                                  for folder in (run_folder / "Evaluation").glob("Iteration_*") if folder.is_dir())
    missing_iterations = sorted(set(CONFUSION_ITERATIONS) - set(evaluated_iterations))
    if missing_iterations or not evaluated_iterations or evaluated_iterations[-1] != LAST_ITERATION:
        raise ValueError(f"{run_folder}/Evaluation holds the iterations {evaluated_iterations}: expected every one of "
                         f"{CONFUSION_ITERATIONS} and {LAST_ITERATION} as the last one.")


def find_largest_confusions(counts: numpy.ndarray, pair_count: int) -> List[Tuple[int, int]]:
    """
    Gives the largest confusions of a confusion matrix, the off-diagonal cells holding the most images, the largest
    first, ties in row then column order.

    Args:
        counts (numpy.ndarray): The [K, K] counts, true classes in rows.
        pair_count (int): The number of pairs given.

    Returns:
        List[Tuple[int, int]]: The true and predicted class indices of every pair.
    """
    off_diagonal = counts.copy()
    numpy.fill_diagonal(off_diagonal, -1)
    flat_order = numpy.argsort(-off_diagonal, axis=None, kind="stable")[:pair_count]
    return [(int(index) // counts.shape[1], int(index) % counts.shape[1]) for index in flat_order]


def describe_confusion_matrices(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the facts of the confusion matrices of the evaluated checkpoints: the recall and precision of every class,
    the largest confusions of the last checkpoint and the weakest classes of the first one by their role (names, counts,
    shares and recalls at every shown checkpoint), the class taking the most images of other classes at the last
    checkpoint, and the share of every true class predicted as every class among the classes these confusions involve,
    in whole percents (the values of the grids), with the rows and columns of the cells the scene marks.

    Args:
        run_folder (Path): The folder of the classification run.

    Returns:
        Dict[str, Any]: The facts, named confusion.*, with the evaluated checkpoints shown.

    Raises:
        ValueError: If the confusions to show involve more than MAXIMUM_FOCUS_CLASSES classes.
    """
    matrices = {}
    for iteration in CONFUSION_ITERATIONS:
        class_names, matrices[iteration] = read_confusion_matrix(evaluation_folder=run_folder / "Evaluation" /
                                                                 f"Iteration_{iteration}")
    facts = {"confusion.iterations": CONFUSION_ITERATIONS, "confusion.class_count": len(class_names)}

    # Recall and precision of every class at every shown checkpoint
    for iteration, counts in matrices.items():
        prefix = f"confusion.iteration_{iteration}"
        row_shares = counts / counts.sum(axis=1, keepdims=True)
        column_totals = counts.sum(axis=0)
        facts[f"{prefix}.test_images"] = int(counts.sum())
        for class_index, class_name in enumerate(class_names):
            facts[f"{prefix}.recall.{class_name}"] = round_fact(value=row_shares[class_index, class_index])
            facts[f"{prefix}.precision.{class_name}"] = round_fact(
                value=counts[class_index, class_index] / column_totals[class_index]
                if column_totals[class_index] else 0)

    # Largest confusions of the last checkpoint and weakest classes of the first one, by their role
    last_counts, first_counts = matrices[LAST_ITERATION], matrices[CONFUSION_ITERATIONS[0]]
    named_pairs = find_largest_confusions(counts=last_counts, pair_count=NAMED_PAIR_COUNT)
    check_four_sided(class_names=[class_names[index] for pair in named_pairs for index in pair],
                     role=f"classes of the {NAMED_PAIR_COUNT} largest confusions")
    first_recalls = numpy.diag(first_counts) / first_counts.sum(axis=1)
    weakest_indices = [int(index) for index in numpy.argsort(first_recalls, kind="stable")[:WEAKEST_CLASS_COUNT]]

    # Classes of the grids: those of the named pairs and the weakest classes, in class index order
    focus_indices = sorted({index for pair in named_pairs for index in pair} | set(weakest_indices))
    if len(focus_indices) > MAXIMUM_FOCUS_CLASSES:
        raise ValueError(f"The confusions of {run_folder} involve {len(focus_indices)} classes, above the "
                         f"{MAXIMUM_FOCUS_CLASSES} the confusion grids can show.")
    focus_names = [class_names[index] for index in focus_indices]
    facts.update({"confusion.focus_class_names": [format_class_name(class_name=name) for name in focus_names],
                  "confusion.focus_short_names": build_short_class_names(class_names=focus_names),
                  "confusion.focus_class_count": len(focus_indices)})
    for iteration, counts in matrices.items():
        row_shares = counts / counts.sum(axis=1, keepdims=True)
        facts[f"confusion.iteration_{iteration}.focus_row_percentages"] = numpy.rint(
            row_shares[numpy.ix_(focus_indices, focus_indices)] * 100).astype(int).tolist()

    # Every named pair: its classes, count and share of the true class at every shown checkpoint, and its cell
    for pair_number, (true_index, predicted_index) in enumerate(named_pairs, start=1):
        prefix = f"confusion.pair_{pair_number}"
        facts.update({f"{prefix}.true_class": format_class_name(class_name=class_names[true_index]),
                      f"{prefix}.predicted_class": format_class_name(class_name=class_names[predicted_index]),
                      f"{prefix}.row": focus_indices.index(true_index),
                      f"{prefix}.column": focus_indices.index(predicted_index)})
        for iteration, counts in matrices.items():
            facts[f"{prefix}.iteration_{iteration}.count"] = int(counts[true_index, predicted_index])
            facts[f"{prefix}.iteration_{iteration}.share"] = round_fact(
                value=counts[true_index, predicted_index] / counts[true_index].sum())

    # Every weakest class of the first checkpoint: its name, recall at every shown checkpoint and diagonal cell
    for weak_number, class_index in enumerate(weakest_indices, start=1):
        prefix = f"confusion.weakest_{weak_number}"
        facts.update({f"{prefix}.class_name": format_class_name(class_name=class_names[class_index]),
                      f"{prefix}.cell": focus_indices.index(class_index)})
        for iteration, counts in matrices.items():
            facts[f"{prefix}.iteration_{iteration}.recall"] = round_fact(
                value=counts[class_index, class_index] / counts[class_index].sum())

    # Class taking the most images of other classes at the last checkpoint, with its precision and recall
    wrong_column_counts = last_counts.sum(axis=0) - numpy.diag(last_counts)
    taking_index = int(numpy.argmax(wrong_column_counts))
    facts.update({"confusion.most_taking.class_name": format_class_name(class_name=class_names[taking_index]),
                  "confusion.most_taking.wrong_count": int(wrong_column_counts[taking_index]),
                  "confusion.most_taking.precision": facts[f"confusion.iteration_{LAST_ITERATION}.precision."
                                                           f"{class_names[taking_index]}"],
                  "confusion.most_taking.recall": facts[f"confusion.iteration_{LAST_ITERATION}.recall."
                                                        f"{class_names[taking_index]}"]})
    return facts


def describe_class_metrics(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the facts of the per-class metrics of the last checkpoint: the worst and the best classes by F1, sorted from
    the worst like the table of the report block, with their F1 and their recall at the target precision as bar values
    and labels (0 and a dash for a class never reaching the target precision), the threshold and recall of every class,
    and the classes never reaching the target precision.

    Args:
        run_folder (Path): The folder of the classification run.

    Returns:
        Dict[str, Any]: The facts, named class_metrics.*.

    Raises:
        ValueError: If class_thresholds.json and class_metrics.csv disagree on a threshold, or if no class or every
            class reaches the target precision.
    """
    evaluation_folder = run_folder / "Evaluation" / f"Iteration_{LAST_ITERATION}"
    rows = read_class_metrics(evaluation_folder=evaluation_folder)
    thresholds = json.loads((evaluation_folder / "class_thresholds.json").read_text(encoding="utf-8"))

    # Worst and best classes by F1, ties in class order, as the report block sorts them
    sorted_rows = sorted(rows, key=order_by_f1)
    bar_rows = sorted_rows[:WORST_BAR_CLASSES] + sorted_rows[-BEST_BAR_CLASSES:]
    facts = {"class_metrics.iteration": LAST_ITERATION,
             "class_metrics.target_precision": thresholds["target_precision"], "class_metrics.class_count": len(rows),
             "class_metrics.worst_bar_count": WORST_BAR_CLASSES, "class_metrics.best_bar_count": BEST_BAR_CLASSES,
             "class_metrics.bar_class_names": [format_class_name(class_name=row["class"]) for row in bar_rows],
             "class_metrics.bar_f1": [round_fact(value=row["f1"]) for row in bar_rows],
             "class_metrics.bar_f1_labels": [format_percentage(value=row["f1"]) for row in bar_rows],
             "class_metrics.bar_recall_at_target": [round_fact(value=row["recall_at_target_precision"] or 0.0)
                                                    for row in bar_rows],
             "class_metrics.bar_recall_at_target_labels": [
                 format_percentage(value=row["recall_at_target_precision"])
                 if row["recall_at_target_precision"] is not None else MISSING_VALUE_TEXT for row in bar_rows]}

    check_four_sided(class_names=[row["class"] for row in sorted_rows[:WORST_BAR_CLASSES]],
                     role=f"{WORST_BAR_CLASSES} worst classes by F1")

    # Classes by their rank of F1: the three worst and the best, with their F1 and recall at the target precision, and
    # their name as the report block writes it
    for rank_name, row in [("worst_1", sorted_rows[0]), ("worst_2", sorted_rows[1]), ("worst_3", sorted_rows[2]),
                           ("best", sorted_rows[-1])]:
        facts.update({f"class_metrics.{rank_name}.class_name": format_class_name(class_name=row["class"]),
                      f"class_metrics.{rank_name}.report_name": row["class"],
                      f"class_metrics.{rank_name}.f1": round_fact(value=row["f1"]),
                      f"class_metrics.{rank_name}.average_precision": round_fact(value=row["average_precision"]),
                      f"class_metrics.{rank_name}.recall_at_target_precision": round_fact(
                          value=row["recall_at_target_precision"]) if row["recall_at_target_precision"] is not None
                      else None})

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
            never_reaching.append(format_class_name(class_name=row["class"]))
        for column in ["samples", "precision", "recall", "f1", "roc_auc", "average_precision",
                       "threshold_at_target_precision", "recall_at_target_precision"]:
            value = row[column]
            facts[f"class_metrics.{row['class']}.{column}"] = int(value) if column == "samples" else (
                round_fact(value=value) if value is not None else None)
    # Classes never reaching the target precision in the order of the report block (the worst F1 first), and the
    # class reaching it with the lowest recall
    never_reaching_rows = [row for row in sorted_rows if row["threshold_at_target_precision"] is None]
    reaching_rows = [row for row in sorted_rows if row["recall_at_target_precision"] is not None]
    if not never_reaching_rows or not reaching_rows:
        raise ValueError(f"{evaluation_folder} has {len(never_reaching_rows)} classes never reaching the target "
                         f"precision and {len(reaching_rows)} reaching it: the report, threshold and pitfall scenes of "
                         f"02_judging_and_shipping.yaml name one of each.")
    lowest_reaching_row = min(reaching_rows, key=lambda row: row["recall_at_target_precision"])
    facts.update({"class_metrics.classes_never_reaching_target": never_reaching,
                  "class_metrics.never_reaching_count": len(never_reaching),
                  "class_metrics.never_reaching_text": join_class_names(
                      class_names=[format_class_name(class_name=row["class"]) for row in never_reaching_rows]),
                  "class_metrics.first_never_reaching.class_name": format_class_name(
                      class_name=never_reaching_rows[0]["class"]),
                  "class_metrics.first_never_reaching.report_name": never_reaching_rows[0]["class"],
                  "class_metrics.lowest_reaching.class_name": format_class_name(
                      class_name=lowest_reaching_row["class"]),
                  "class_metrics.lowest_reaching.recall_at_target_precision": round_fact(
                      value=lowest_reaching_row["recall_at_target_precision"])})
    return facts


def describe_headline_metrics(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the headline metrics of every evaluated checkpoint, read from the Test logs of TensorBoard (the values of
    the report blocks of Results.txt), and the size of the predictions file of the last checkpoint (its rows and the
    megabytes of the CSV, whose reduced copy in the snapshots gives the same numbers).

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
    with open_predictions_file(predictions_path=predictions_path, newline=None) as predictions_file:
        line_count = sum(1 for _ in predictions_file)
    facts.update({"predictions_file.rows": line_count - 1, "predictions_file.megabytes": round_fact(
        value=measure_predictions_bytes(predictions_path=predictions_path) / 1e6)})
    return facts


def describe_checkpoint_evaluation(evaluation_folder: Path) -> Dict[str, Any]:
    """
    Gives the headline metrics of the standalone evaluation of the last checkpoint, read from its Test logs of
    TensorBoard, the values of the report block of its Results.txt that the report scene shows.

    Args:
        evaluation_folder (Path): The '<run>/Checkpoint_Evaluation' folder of the classification run.

    Returns:
        Dict[str, Any]: The facts, named checkpoint_evaluation.*, with the evaluated iteration.

    Raises:
        ValueError: If the evaluation does not hold LAST_ITERATION alone.
    """
    facts = {"checkpoint_evaluation.iteration": LAST_ITERATION}
    for metric_tag, fact_name in HEADLINE_METRICS.items():
        iterations, values = read_scalars(log_folder=evaluation_folder / "Tensorboard" / "Test", tag=metric_tag)
        if iterations.tolist() != [LAST_ITERATION]:
            raise ValueError(f"The standalone evaluation {evaluation_folder} logs {metric_tag} at the iterations "
                             f"{iterations.tolist()}, expected [{LAST_ITERATION}] alone.")
        facts[f"checkpoint_evaluation.{fact_name}"] = round_fact(value=values[0])
    return facts


def make_evaluation_tables(run_folder: Path, evaluation_folder: Path) -> Dict[str, Any]:
    """
    Reads the evaluation tables of the classification run and of its standalone evaluation, and gives their facts.

    Args:
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.
        evaluation_folder (Path): The folder of its standalone evaluation, classification_evaluation of
            tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named confusion.*, class_metrics.*, headline.*, checkpoint_evaluation.*,
            predictions_file.* and temperature_explorer.* (the lowest and highest temperatures of the slider of the
            explorer page, and the default port of the local page of serve_temperature_explorer.py).
    """
    check_last_iteration(run_folder=run_folder)
    facts = describe_confusion_matrices(run_folder=run_folder)
    facts.update(describe_class_metrics(run_folder=run_folder))
    facts.update(describe_headline_metrics(run_folder=run_folder))
    facts.update(describe_checkpoint_evaluation(evaluation_folder=evaluation_folder))

    # Test images whose true class is not among the most probable classes of the top-k accuracy
    test_images = facts[f"confusion.iteration_{LAST_ITERATION}.test_images"]
    facts["checkpoint_evaluation.top_k_missed_images"] = round(
        (1 - facts["checkpoint_evaluation.top_k_accuracy"]) * test_images)

    # Temperatures of the slider of the explorer page, and the port its serving script takes without --port
    facts.update({"temperature_explorer.minimum_temperature": min(TEMPERATURE_GRID),
                  "temperature_explorer.maximum_temperature": max(TEMPERATURE_GRID),
                  "temperature_explorer.port": DEFAULT_PORT})
    return facts
