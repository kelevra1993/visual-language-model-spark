import csv
import pytest

from pathlib import Path
from typing import Dict, List


def read_predictions_file(evaluation_folder: Path) -> List[Dict[str, str]]:
    """
    Reads the predictions file written by the classification-like trainers in the evaluation folder of a checkpoint.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/'.

    Returns:
        List[Dict[str, str]]: The rows of 'predictions.csv', keyed by column name.
    """
    with open(file=evaluation_folder / "predictions.csv", mode="r", newline="") as predictions_file:
        return list(csv.DictReader(f=predictions_file))


def check_predictions_file(evaluation_folder: Path, results_lines: List[str], expected_rows: int,
                           class_names: List[str], identification_columns: List[str]) -> List[Dict[str, str]]:
    """
    Verifies the predictions file of a multi-class evaluation, shared by the trainer smoke tests: one row per evaluated
    sample (or labelled frame) starting with the columns naming it, the probabilities of every class summing to 1,
    the predicted class being the most probable one, and the accuracy recomputed from the file equal to the accuracy
    reported in Results.txt.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
        results_lines (List[str]): The lines of Results.txt, the first block being the evaluation of the checkpoint.
        expected_rows (int): The number of evaluated samples, or of labelled frames for the sequence labelling task.
        class_names (List[str]): The class names of the run, in class index order.
        identification_columns (List[str]): The columns naming the samples, in order.

    Returns:
        List[Dict[str, str]]: The rows of the predictions file.
    """
    rows = read_predictions_file(evaluation_folder=evaluation_folder)
    assert len(rows) == expected_rows
    assert list(rows[0]) == identification_columns + ["true_class", "predicted_class", "correct"] + [
        f"probability_{class_name}" for class_name in class_names]

    # Every row holds a distribution of probabilities whose most probable class is the predicted one
    for row in rows:
        probabilities = [float(row[f"probability_{class_name}"]) for class_name in class_names]
        assert sum(probabilities) == pytest.approx(expected=1.0, abs=1e-4)
        assert row["predicted_class"] == class_names[probabilities.index(max(probabilities))]
        assert row["correct"] == str(int(row["true_class"] == row["predicted_class"]))

    # The accuracy of the file is the reported one, e.g. '  Accuracy 0.8333 | Balanced Accuracy ...'
    reported_accuracy = float(results_lines[1].split("|")[0].split()[1])
    assert sum(int(row["correct"]) for row in rows) / len(rows) == pytest.approx(expected=reported_accuracy, abs=5e-5)
    return rows
