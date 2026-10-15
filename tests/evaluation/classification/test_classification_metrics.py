"""
Tests of the classification evaluation of vision_training/evaluation/classification: the confusion matrix and metrics
against hand-computed values, the top-k accuracy, the multi-label metrics, the class thresholds of threshold_finder.py
(with the recall of the detection classes) and the confusion matrix files and images (with the background of the
detection tasks).
"""

import csv
import sys
import cv2
import numpy as np
import pytest
import torch

from pathlib import Path

from vision_training.evaluation.classification.classification_metrics import (compute_confusion_matrix,
                                                                              compute_metrics_from_confusion_matrix,
                                                                              compute_multi_label_metrics,
                                                                              compute_top_k_accuracy)
from vision_training.evaluation.classification.confusion_matrix import (draw_confusion_matrix,
                                                                        save_confusion_matrix_image,
                                                                        save_confusion_matrix_table)
from vision_training.evaluation.classification.threshold_finder import find_class_thresholds, find_score_threshold


def test_confusion_matrix_and_metrics_match_hand_computed_values() -> None:
    """
    Verifies the confusion matrix and the multi-class metrics on 6 samples of 3 classes, against values computed by
    hand: class 0 has 3 samples (2 right, 1 predicted as class 1), class 1 has 2 samples (both right) and class 2 has
    1 sample (predicted as class 1).
    """
    labels = torch.tensor(data=[0, 0, 0, 1, 1, 2])
    predicted_labels = torch.tensor(data=[0, 0, 1, 1, 1, 1])

    confusion_matrix = compute_confusion_matrix(predicted_labels=predicted_labels, labels=labels, number_classes=3)
    assert confusion_matrix.tolist() == [[2, 1, 0], [0, 2, 0], [0, 1, 0]]

    metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)
    assert metrics["accuracy"] == pytest.approx(expected=4 / 6)
    assert metrics["precision"] == pytest.approx(expected=[1.0, 0.5, 0.0])
    assert metrics["recall"] == pytest.approx(expected=[2 / 3, 1.0, 0.0])
    assert metrics["f1"] == pytest.approx(expected=[0.8, 2 / 3, 0.0])
    assert metrics["support"] == [3, 2, 1]
    assert metrics["balanced_accuracy"] == pytest.approx(expected=(2 / 3 + 1.0 + 0.0) / 3)
    assert metrics["macro_f1"] == pytest.approx(expected=(0.8 + 2 / 3 + 0.0) / 3)


def test_class_absent_from_the_evaluation_is_left_out_of_the_averages() -> None:
    """
    Verifies that a class without any evaluated sample does not lower the balanced accuracy and the macro F1, which
    average over the classes present in the evaluated samples only.
    """
    confusion_matrix = compute_confusion_matrix(predicted_labels=torch.tensor(data=[0, 1]),
                                                labels=torch.tensor(data=[0, 1]), number_classes=3)
    metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)
    assert metrics["balanced_accuracy"] == metrics["macro_f1"] == metrics["accuracy"] == 1.0


def test_top_k_accuracy() -> None:
    """
    Verifies the top-k accuracy: a sample counts as right when its true class is among its top_k most probable
    classes, top_k being clipped to the number of classes.
    """
    probabilities = torch.tensor(data=[[0.5, 0.3, 0.2], [0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    labels = torch.tensor(data=[1, 2, 2])
    assert compute_top_k_accuracy(probabilities=probabilities, labels=labels, top_k=1) == pytest.approx(
        expected=1 / 3)
    assert compute_top_k_accuracy(probabilities=probabilities, labels=labels, top_k=2) == pytest.approx(
        expected=2 / 3)
    assert compute_top_k_accuracy(probabilities=probabilities, labels=labels, top_k=10) == 1.0


def test_multi_label_metrics() -> None:
    """
    Verifies the multi-label metrics against hand-computed values: 3 samples, 2 classes, classes predicted from a
    probability of 0.5.
    """
    probabilities = torch.tensor(data=[[0.9, 0.8], [0.7, 0.2], [0.1, 0.6]])
    labels = torch.tensor(data=[[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]])

    metrics = compute_multi_label_metrics(probabilities=probabilities, labels=labels, threshold=0.5)
    # Class 0: 1 true positive, 1 false positive. Class 1: 2 true positives
    assert metrics["precision"] == pytest.approx(expected=[0.5, 1.0])
    assert metrics["recall"] == pytest.approx(expected=[1.0, 1.0])
    assert metrics["support"] == [1, 2]
    assert metrics["subset_accuracy"] == pytest.approx(expected=2 / 3)
    assert metrics["micro_f1"] == pytest.approx(expected=2 * 0.75 * 1.0 / 1.75)
    assert metrics["macro_f1"] == pytest.approx(expected=(2 / 3 + 1.0) / 2)


def test_class_thresholds_reach_the_target_precision() -> None:
    """
    Verifies the per-class thresholds: the lowest threshold whose predictions reach the target precision is kept
    with its recall, a class whose target precision is never reached gets None, and so does a class without sample.
    """
    probabilities = torch.tensor(data=[[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.9, 0.0], [0.6, 0.3, 0.0],
                                       [0.4, 0.95, 0.0]])
    positive_masks = torch.tensor(data=[[True, False, False], [True, False, False], [False, True, False],
                                        [True, False, False], [False, False, False]])

    class_thresholds = find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                             target_precision=0.75)
    # Class 0: scores 0.9 (+), 0.8 (+), 0.7 (-), 0.6 (+), 0.4 (-): cutting at 0.6 keeps a precision of 3 / 4
    assert class_thresholds[0] == pytest.approx(expected=(0.6, 1.0))
    # Class 1: the highest score 0.95 is a negative, then 0.9 is the positive: the precision never exceeds 1 / 2
    assert class_thresholds[1] is None
    # Class 2 has no positive sample
    assert class_thresholds[2] is None


def test_tied_scores_are_cut_together() -> None:
    """
    Verifies that samples with the same score are kept or rejected together, since a threshold cannot separate them.
    """
    probabilities = torch.tensor(data=[[0.9], [0.5], [0.5]])
    positive_masks = torch.tensor(data=[[True], [True], [False]])
    assert find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                 target_precision=0.9)[0] == pytest.approx(expected=(0.9, 0.5))


def test_recall_counts_the_members_never_predicted() -> None:
    """
    Verifies the threshold of one class whose recall is counted over more members than its right predictions, the
    objects of a detection class found by no detection: the precision decides the cut, the recall divides by every
    member, and a class without prediction or without member gets None.
    """
    scores = torch.tensor(data=[0.9, 0.8, 0.3])
    positives = torch.tensor(data=[True, True, False])
    assert find_score_threshold(scores=scores, positives=positives, number_positives=4,
                                target_precision=0.99) == pytest.approx(expected=(0.8, 0.5))
    assert find_score_threshold(scores=torch.zeros(size=(0,)), positives=torch.zeros(size=(0,), dtype=torch.bool),
                                number_positives=3, target_precision=0.99) is None
    assert find_score_threshold(scores=scores, positives=positives, number_positives=0, target_precision=0.99) is None


def test_confusion_matrix_files(tmp_path: Path) -> None:
    """
    Verifies that the confusion matrix is saved as a CSV table, one row per true class, and as an image identical to
    the RGB array logged to TensorBoard (the PNG being written in BGR order).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    confusion_matrix = torch.tensor(data=[[5, 1], [0, 4]])
    table_path = save_confusion_matrix_table(confusion_matrix=confusion_matrix, class_names=["circle", "square"],
                                             output_folder=tmp_path / "Iteration_10")
    with open(table_path, newline="") as table_file:
        rows = list(csv.reader(table_file))
    assert rows == [["true \\ predicted", "circle", "square"], ["circle", "5", "1"], ["square", "0", "4"]]

    image_path, image = save_confusion_matrix_image(confusion_matrix=confusion_matrix, class_names=["circle", "square"],
                                                    output_folder=tmp_path / "Iteration_10", title="Confusion Matrix")
    assert image_path == tmp_path / "Iteration_10" / "confusion_matrix.png"
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    assert np.array_equal(cv2.cvtColor(src=cv2.imread(filename=str(image_path)), code=cv2.COLOR_BGR2RGB), image)


def test_confusion_matrix_image_grows_with_the_classes_and_stays_headless() -> None:
    """
    Verifies that the confusion matrix image grows with the number of classes up to a cap, that it is drawn without
    pyplot (no global figure state in the training process), and that a matrix without any sample is still drawn.
    """
    pyplot_loaded_before = "matplotlib.pyplot" in sys.modules
    small_image = draw_confusion_matrix(confusion_matrix=torch.eye(n=3, dtype=torch.int64) * 4,
                                        class_names=["circle", "square", "triangle"], title="Small")
    large_image = draw_confusion_matrix(confusion_matrix=torch.eye(n=50, dtype=torch.int64) * 4,
                                        class_names=[f"class_{class_index}" for class_index in range(50)],
                                        title="Large")
    capped_image = draw_confusion_matrix(confusion_matrix=torch.eye(n=200, dtype=torch.int64),
                                         class_names=[f"class_{class_index}" for class_index in range(200)],
                                         title="Capped")
    assert large_image.shape[0] > small_image.shape[0] and large_image.shape[1] > small_image.shape[1]
    assert capped_image.shape == large_image.shape
    assert pyplot_loaded_before or "matplotlib.pyplot" not in sys.modules

    empty_image = draw_confusion_matrix(confusion_matrix=torch.zeros(size=(3, 3), dtype=torch.int64),
                                        class_names=["circle", "square", "triangle"], title="Empty")
    assert empty_image.shape == small_image.shape


def test_detection_confusion_matrix_leaves_its_background_cell_blank() -> None:
    """
    Verifies the confusion matrix image of the detection tasks: with a background row and column, it is drawn like
    the others, the background cell (left blank, never counted) being the only difference from the same matrix drawn
    without background, whose background cell shows a bold 0 and whose background recall and precision are drawn.
    """
    confusion_matrix = torch.tensor(data=[[0, 3, 1], [2, 5, 0], [1, 1, 4]])
    class_names = ["background", "circle", "square"]
    detection_image = draw_confusion_matrix(confusion_matrix=confusion_matrix, class_names=class_names,
                                            title="Detection", background_index=0)
    plain_image = draw_confusion_matrix(confusion_matrix=confusion_matrix, class_names=class_names, title="Detection")
    assert detection_image.shape == plain_image.shape
    assert not np.array_equal(detection_image, plain_image)
