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


def check_error_images(evaluation_folder: Path, rows: List[Dict[str, str]], results_lines: List[str],
                       maximum_per_class_pair: int) -> int:
    """
    Verifies the error images of a multi-class evaluation against its predictions file, shared by the trainer smoke
    tests: one '<TRUE>-TO-<PREDICTED>' folder per pair of classes with wrong predictions, holding as many images as the
    wrong predictions of the pair up to the maximum per pair, every image named after the score of a wrong prediction
    of its pair, and the number of images reported in Results.txt.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
        rows (List[Dict[str, str]]): The rows of the predictions file of the checkpoint.
        results_lines (List[str]): The lines of the block of the checkpoint in Results.txt.
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

    assert (f"  Error Images : {number_images} in {len(pair_folders)} class pairs, at most {maximum_per_class_pair} "
            f"per pair") in results_lines
    return number_images
