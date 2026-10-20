"""
Checks of the evaluation outputs written by the trainers (the blocks of Results.txt, the per-class metrics file, the
predictions file, the error image folders, the sample images, and the confusion matrix, class thresholds and error
images of the detection tasks), shared by the trainer smoke tests of every task.
"""

import csv
import json
import re
import cv2
import numpy as np
import pytest

from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_training.utilities.visualization import (CAPTION_BAR_HEIGHT, CORRECT_PREDICTION_COLOUR,
                                                     WRONG_PREDICTION_COLOUR)

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


def check_class_metrics_file(evaluation_folder: Path, class_names: List[str], number_samples: int,
                             per_class_report: bool) -> List[Dict[str, str]]:
    """
    Verifies the per-class metrics file of a classification-like evaluation: one row per class in class index order,
    with the columns of the report (the ROC AUC and average precision with the per-class report), the samples of the
    classes summing to the evaluated samples (or labelled frames).

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
        class_names (List[str]): The class names of the run, in class index order.
        number_samples (int): The number of evaluated samples, or of labelled frames for the sequence labelling task.
        per_class_report (bool): Whether the evaluation ran with the per-class report, adding the curve columns.

    Returns:
        List[Dict[str, str]]: The rows of 'class_metrics.csv', keyed by column name.
    """
    with open(file=evaluation_folder / "class_metrics.csv", mode="r", newline="") as class_metrics_file:
        rows = list(csv.DictReader(f=class_metrics_file))
    assert [row["class"] for row in rows] == class_names
    assert [row["class_index"] for row in rows] == [str(class_index) for class_index in range(len(class_names))]
    curve_columns = ["roc_auc", "average_precision"] if per_class_report else []
    assert list(rows[0]) == ["class_index", "class", "samples", "precision", "recall", "f1"] + curve_columns + [
        "threshold_at_target_precision", "recall_at_target_precision"]
    assert sum(int(row["samples"]) for row in rows) == number_samples
    return rows


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


def check_sample_images(evaluation_folder: Path, results: Dict[str, Any], number_sample_images: int, sample_kind: str,
                        correct_predictions: Optional[Dict[str, bool]]) -> None:
    """
    Verifies the sample images of an evaluation, shared by the trainer smoke tests: one RGB image per fixed test sample
    in the results under the TensorBoard tag 'Samples/<sample name>', each saved in BGR as '<sample name>.png' in the
    'Samples' folder of the checkpoint, with the number of images in the notes and the folder in the files, and, when
    the correctness of the samples is given, the caption of every image written in green when its prediction is right
    and in red when it is wrong.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
        results (Dict[str, Any]): The evaluation results of the checkpoint.
        number_sample_images (int): The expected number of sample images.
        sample_kind (str): The kind of the samples written in the note, e.g. 'images' or 'sequences'.
        correct_predictions (Optional[Dict[str, bool]]): Whether the prediction of every sample is right, keyed by
            sample name, or None to leave the caption colours unchecked.
    """
    samples_folder = evaluation_folder / "Samples"
    sample_names = sorted(image_name.removeprefix("Samples/") for image_name in results["sample_images"])
    assert len(sample_names) == number_sample_images
    assert sorted(image_path.stem for image_path in samples_folder.glob("*.png")) == sample_names
    assert f"Sample Images : {number_sample_images} fixed test {sample_kind}" in results["notes"]
    assert results["files"]["Sample Images"] == samples_folder

    for sample_name in sample_names:
        # The saved image is the BGR version of the RGB image sent to TensorBoard
        sample_image = results["sample_images"][f"Samples/{sample_name}"]
        assert sample_image.dtype == np.uint8 and sample_image.ndim == 3 and sample_image.shape[2] == 3
        saved_image = cv2.imread(filename=str(samples_folder / f"{sample_name}.png"))
        assert (cv2.cvtColor(src=saved_image, code=cv2.COLOR_BGR2RGB) == sample_image).all()

        # The caption is written in the colour of the correctness of the prediction, given in BGR
        if correct_predictions is not None:
            caption_colour = CORRECT_PREDICTION_COLOUR if correct_predictions[sample_name] else WRONG_PREDICTION_COLOUR
            caption_bar = sample_image[:CAPTION_BAR_HEIGHT]
            assert (caption_bar == np.array(object=caption_colour[::-1])).all(axis=-1).any(), sample_name


def check_detection_error_analysis(evaluation_folder: Path, results: Dict[str, Any], class_names: List[str],
                                   number_objects: int, maximum_per_class_pair: int) -> None:
    """
    Verifies the error analysis of a detection evaluation, shared by the object detection and instance segmentation
    smoke tests: confusion_matrix.csv with its background row and column, every object of the test images counted
    once (found or missed) and its background cell empty, the note counting the outcomes, class_thresholds.json with a
    threshold and recall within [0, 1] or None for every foreground class, and the error images, one folder per
    non-empty error cell of the matrix ('FALSE-POSITIVE/<class>', 'MISSED/<class>', '<TRUE>-TO-<PREDICTED>'), each
    holding at most the errors of its cell and the maximum per folder, the total in the notes.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint.
        results (Dict[str, Any]): The evaluation results of the checkpoint.
        class_names (List[str]): The name of every class, background first.
        number_objects (int): The number of objects of the test images.
        maximum_per_class_pair (int): The largest number of images saved per folder.
    """
    # Confusion matrix: rows true classes, columns predicted classes, the background first
    with open(file=evaluation_folder / "confusion_matrix.csv", mode="r", newline="") as matrix_file:
        matrix_rows = list(csv.reader(matrix_file))
    assert matrix_rows[0] == ["true \\ predicted"] + class_names
    assert [matrix_row[0] for matrix_row in matrix_rows[1:]] == class_names
    counts = np.array(object=[[int(count) for count in matrix_row[1:]] for matrix_row in matrix_rows[1:]])
    assert counts[0, 0] == 0 and int(counts[1:].sum()) == number_objects
    right, found = int(np.trace(counts[1:, 1:])), int(counts[1:, 1:].sum())
    assert any(note.startswith("Confusion Matrix : ") and note.endswith(
        f"{right} right, {found - right} wrong class, {int(counts[0].sum())} false positives, "
        f"{int(counts[:, 0].sum())} missed") for note in results["notes"])

    # Threshold of every foreground class, or None
    class_thresholds = json.loads(s=(evaluation_folder / "class_thresholds.json").read_text())
    assert class_thresholds["target_precision"] == 0.99 and list(class_thresholds["classes"]) == class_names[1:]
    for class_threshold in class_thresholds["classes"].values():
        assert class_threshold is None or (0.0 <= class_threshold["threshold"] <= 1.0 and
                                           0.0 <= class_threshold["recall"] <= 1.0)

    # One error folder per non-empty error cell of the matrix, at most its errors and the maximum per folder
    expected_folders = {}
    for class_index, class_name in enumerate(class_names[1:], start=1):
        expected_folders[f"FALSE-POSITIVE/{class_name}"] = int(counts[0, class_index])
        expected_folders[f"MISSED/{class_name}"] = int(counts[class_index, 0])
        for predicted_index, predicted_name in enumerate(class_names[1:], start=1):
            if predicted_index != class_index:
                expected_folders[f"{class_name}-TO-{predicted_name}"] = int(counts[class_index, predicted_index])
    errors_folder = evaluation_folder / "Errors"
    error_folders = {str(image_path.parent.relative_to(errors_folder)) for image_path in errors_folder.rglob("*.png")}
    assert error_folders == {folder_name for folder_name, number_errors in expected_folders.items() if number_errors}
    number_images = 0
    for folder_name in error_folders:
        image_paths = list((errors_folder / folder_name).glob("*.png"))
        assert 1 <= len(image_paths) <= min(expected_folders[folder_name], maximum_per_class_pair)
        number_images += len(image_paths)
    assert (f"Error Images : {number_images} in {len(error_folders)} folders, at most {maximum_per_class_pair} per "
            f"folder") in results["notes"]
