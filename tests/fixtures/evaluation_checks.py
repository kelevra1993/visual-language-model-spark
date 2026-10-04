"""
Checks of the evaluation outputs written by the trainers (the blocks of Results.txt, the per-class metrics file, the
predictions file and the error image folders), shared by the trainer smoke tests of every task.
"""

import csv
import re
import pytest

from pathlib import Path
from typing import Any, Dict, List

# Header starting every block of Results.txt: the iteration, the evaluated samples and the time of the evaluation
RESULTS_BLOCK_HEADER = re.compile(pattern=r"^Iteration (\d+) · (.+?) · (\d{4}-\d{2}-\d{2} \d{2}:\d{2})")


def read_results_blocks(results_path: Path) -> List[Dict[str, Any]]:
    """
    Splits the Results.txt file of a run into the blocks of its evaluated checkpoints, found by their header (the
    tables of a block hold empty lines, so the blocks cannot be split on empty lines).

    Args:
        results_path (Path): The Results.txt file of the run.

    Returns:
        List[Dict[str, Any]]: For every block, its 'iteration', 'evaluated_samples', 'timestamp' and 'text' (the lines
            below the header).
    """
    blocks = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        header_match = RESULTS_BLOCK_HEADER.match(string=line)
        if header_match is not None:
            blocks.append({"iteration": int(header_match.group(1)), "evaluated_samples": header_match.group(2),
                           "timestamp": header_match.group(3), "text": ""})
        elif blocks:
            blocks[-1]["text"] += line + "\n"
    return blocks


def check_class_metrics_file(evaluation_folder: Path, class_names: List[str], number_samples: int) -> None:
    """
    Verifies the per-class metrics file of a classification-like evaluation: one row per class in class index order,
    with the columns of the report, the samples of the classes summing to the evaluated samples (or labelled frames).

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
        class_names (List[str]): The class names of the run, in class index order.
        number_samples (int): The number of evaluated samples, or of labelled frames for the sequence labelling task.
    """
    with open(file=evaluation_folder / "class_metrics.csv", mode="r", newline="") as class_metrics_file:
        rows = list(csv.DictReader(f=class_metrics_file))
    assert [row["class"] for row in rows] == class_names
    assert [row["class_index"] for row in rows] == [str(class_index) for class_index in range(len(class_names))]
    assert list(rows[0]) == ["class_index", "class", "samples", "precision", "recall", "f1",
                             "threshold_at_target_precision", "recall_at_target_precision"]
    assert sum(int(row["samples"]) for row in rows) == number_samples


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


def check_predictions_file(evaluation_folder: Path, reported_accuracy: float, expected_rows: int,
                           class_names: List[str], identification_columns: List[str]) -> List[Dict[str, str]]:
    """
    Verifies the predictions file of a multi-class evaluation, shared by the trainer smoke tests: one row per evaluated
    sample (or labelled frame) starting with the columns naming it, the probabilities of every class summing to 1,
    the predicted class being the most probable one, and the accuracy recomputed from the file equal to the reported
    accuracy.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
        reported_accuracy (float): The accuracy of the evaluation results of the checkpoint.
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

    # The accuracy of the file is the reported one
    assert sum(int(row["correct"]) for row in rows) / len(rows) == pytest.approx(expected=reported_accuracy)
    return rows


def check_error_images(evaluation_folder: Path, rows: List[Dict[str, str]], notes: List[str],
                       maximum_per_class_pair: int) -> int:
    """
    Verifies the error images of a multi-class evaluation against its predictions file, shared by the trainer smoke
    tests: one '<TRUE>-TO-<PREDICTED>' folder per pair of classes with wrong predictions, holding as many images as the
    wrong predictions of the pair up to the maximum per pair, every image named after the score of a wrong prediction
    of its pair, and the number of images reported in the notes of the evaluation results.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
        rows (List[Dict[str, str]]): The rows of the predictions file of the checkpoint.
        notes (List[str]): The notes of the evaluation results of the checkpoint.
        maximum_per_class_pair (int): The largest number of images saved per class pair.

    Returns:
        int: The number of error images.
    """
    # Scores of the predicted class of the wrong predictions of every class pair
    pair_scores = {}
    for row in rows:
        if row["correct"] == "0":
            pair_name = f"{row['true_class']}-TO-{row['predicted_class']}"
            pair_scores.setdefault(pair_name, []).append(float(row[f"probability_{row['predicted_class']}"]))

    # One folder per pair with wrong predictions, as many images as its errors up to the maximum
    errors_folder = evaluation_folder / "Errors"
    pair_folders = sorted(errors_folder.iterdir()) if errors_folder.exists() else []
    assert sorted(pair_folder.name for pair_folder in pair_folders) == sorted(pair_scores)
    number_images = 0
    for pair_folder in pair_folders:
        image_paths = sorted(pair_folder.glob("*.png"))
        assert len(image_paths) == min(len(pair_scores[pair_folder.name]), maximum_per_class_pair)
        for image_path in image_paths:
            image_score = float(image_path.stem.split("_")[-1])
            assert any(abs(image_score - score) <= 6e-4 for score in pair_scores[pair_folder.name]), image_path
        number_images += len(image_paths)

    assert (f"Error Images : {number_images} in {len(pair_folders)} class pairs, at most {maximum_per_class_pair} "
            f"per pair") in notes
    return number_images
