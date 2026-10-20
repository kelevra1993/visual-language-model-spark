"""
Tests of vision_training/evaluation/segmentation/segmentation_metrics.py and segmentation_report.py: the pixel confusion
matrix and the histograms of the pixel probabilities leave the ignored pixels out, the metrics computed from a known
confusion matrix match the values computed by hand with the absent classes left out of the means, the per-class rows,
table and scalars give no metric to a class without pixels, and the ROC AUC and average precision of the per-class
report join them.
"""

import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.evaluation.segmentation.segmentation_metrics import (compute_pixel_confusion_matrix,
                                                                          compute_pixel_score_histograms,
                                                                          compute_segmentation_metrics)
from vision_training.evaluation.segmentation.segmentation_report import (build_segmentation_class_rows,
                                                                         build_segmentation_class_scalars,
                                                                         build_segmentation_class_table)


def test_pixel_confusion_matrix_leaves_out_the_ignored_pixels() -> None:
    """
    Verifies that every labelled pixel is counted once at its true and predicted classes, whatever the prediction on
    the ignored pixels.
    """
    masks = torch.tensor(data=[[[0, 1], [2, IGNORED_LABEL_INDEX]]])
    class_maps = torch.tensor(data=[[[0, 2], [2, 1]]])

    confusion_matrix = compute_pixel_confusion_matrix(class_maps=class_maps, masks=masks, number_classes=3)
    assert confusion_matrix.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]


def test_pixel_score_histograms_count_the_labelled_pixels() -> None:
    """
    Verifies the histograms of the pixel probabilities of the curves: every labelled pixel is counted in the bin of its
    probability of every class, as a positive of its class and a negative of the others, a probability of 1 in the
    last bin, the ignored pixel being left out.
    """
    masks = torch.tensor(data=[[[0, 1], [IGNORED_LABEL_INDEX, 1]]])
    class_probabilities = torch.tensor(data=[[[[1.0, 0.0], [0.7, 0.5]], [[0.0, 1.0], [0.3, 0.5]]]])

    score_histograms = compute_pixel_score_histograms(class_probabilities=class_probabilities, masks=masks,
                                                      number_bins=4)
    assert score_histograms.tolist() == [[[1, 0, 1, 0], [0, 0, 0, 1]], [[1, 0, 0, 0], [0, 0, 1, 1]]]


def test_curve_metrics_join_the_class_rows_table_and_scalars() -> None:
    """
    Verifies the ROC AUC and average precision of the per-class report of the segmentation: their columns of
    class_metrics.csv and of the table after the recall, with their legend, and their scalars for the classes defining
    them.
    """
    metrics = compute_segmentation_metrics(confusion_matrix=torch.tensor(data=[[7, 0, 1], [1, 3, 0], [0, 0, 0]]))
    metrics.update({"boundary_iou": [0.5, None, None], "boundary_f1": [0.6, None, None]})
    class_curves = [{"roc_auc": 0.95, "average_precision": 0.9}, {"roc_auc": 0.8, "average_precision": 0.7},
                    {"roc_auc": None, "average_precision": None}]
    class_rows = build_segmentation_class_rows(class_names=["background", "circle", "square"], metrics=metrics,
                                               class_curves=class_curves)
    assert list(class_rows[0])[-3:] == ["recall", "roc_auc", "average_precision"]

    table = build_segmentation_class_table(class_rows=class_rows)
    assert table["columns"][-3:] == ["Recall", "ROC AUC", "Average Precision"]
    assert table["rows"][1][-2:] == [0.8, 0.7] and "no pixel of another class for the ROC AUC" in table["legend"]

    class_scalars = build_segmentation_class_scalars(class_rows=class_rows)
    assert [tag for tag in class_scalars if tag.startswith(("ROC AUC/", "Average Precision/"))] == [
        "ROC AUC/background", "Average Precision/background", "ROC AUC/circle", "Average Precision/circle"]


def test_metrics_of_a_known_confusion_matrix() -> None:
    """
    Verifies the pixel accuracy, the IoU and Dice of every class, their means over the classes present and the
    frequency-weighted IoU on a confusion matrix whose class 2 has no true pixel but is predicted once.
    """
    # Rows true classes, columns predicted classes: 8 background pixels (one predicted as class 2), 4 pixels of class 1
    # (one predicted as background), no pixel of class 2
    confusion_matrix = torch.tensor(data=[[7, 0, 1], [1, 3, 0], [0, 0, 0]])
    metrics = compute_segmentation_metrics(confusion_matrix=confusion_matrix)

    # IoU: background 7 / (8 + 8 - 7), class 1 3 / (4 + 3 - 3), class 2 0 (absent from the true masks)
    background_iou, class_iou = 7.0 / 9.0, 3.0 / 4.0
    assert metrics["pixel_accuracy"] == pytest.approx(expected=10.0 / 12.0)
    assert metrics["iou"] == pytest.approx(expected=[background_iou, class_iou, 0.0])
    assert metrics["mean_iou"] == pytest.approx(expected=(background_iou + class_iou) / 2.0)
    assert metrics["frequency_weighted_iou"] == pytest.approx(expected=(8.0 * background_iou + 4.0 * class_iou) / 12.0)

    # Dice: twice the right pixels over the true and predicted pixels, the mean leaving out the absent class
    background_dice, class_dice = 14.0 / 16.0, 6.0 / 7.0
    assert metrics["dice"] == pytest.approx(expected=[background_dice, class_dice, 0.0])
    assert metrics["mean_dice"] == pytest.approx(expected=(background_dice + class_dice) / 2.0)
    assert metrics["precision"] == pytest.approx(expected=[7.0 / 8.0, 1.0, 0.0])
    assert metrics["recall"] == pytest.approx(expected=[7.0 / 8.0, 3.0 / 4.0, 0.0])
    assert metrics["support"] == [8, 4, 0]


def test_metrics_without_labelled_pixel_are_zero() -> None:
    """
    Verifies that a confusion matrix without any labelled pixel gives metrics of 0 instead of failing.
    """
    metrics = compute_segmentation_metrics(confusion_matrix=torch.zeros(size=(3, 3), dtype=torch.int64))
    assert [metrics[metric_key] for metric_key in ["pixel_accuracy", "mean_iou", "frequency_weighted_iou",
                                                    "mean_dice"]] == [0.0, 0.0, 0.0, 0.0]


def test_class_rows_table_and_scalars_leave_out_the_absent_classes() -> None:
    """
    Verifies the per-class report: the rows of class_metrics.csv in class index order, a class without true pixels
    having no metric and a class without true edge no boundary metric, the table sorted by IoU with its legend, and
    the per-class scalars of the metrics the classes have.
    """
    metrics = compute_segmentation_metrics(confusion_matrix=torch.tensor(data=[[7, 0, 1], [1, 3, 0], [0, 0, 0]]))
    metrics.update({"boundary_iou": [0.5, None, None], "boundary_f1": [0.6, None, None]})
    class_rows = build_segmentation_class_rows(class_names=["background", "circle", "square"], metrics=metrics)

    assert [list(class_row) for class_row in class_rows] == [["class_index", "class", "pixels", "iou", "dice",
                                                              "boundary_iou", "boundary_f1", "precision",
                                                              "recall"]] * 3
    assert [class_row["pixels"] for class_row in class_rows] == [8, 4, 0]
    assert class_rows[1]["iou"] == pytest.approx(expected=0.75) and class_rows[1]["boundary_iou"] is None
    assert [class_rows[2][metric_key] for metric_key in ["iou", "dice", "boundary_iou", "boundary_f1", "precision",
                                                         "recall"]] == [None] * 6

    table = build_segmentation_class_table(class_rows=class_rows)
    assert table["columns"] == ["Class", "Pixels", "IoU", "Dice", "Boundary IoU", "Boundary F1", "Precision", "Recall"]
    assert table["sort_column"] == "IoU" and table["per_class"]
    assert table["rows"][2] == ["square", 0, None, None, None, None, None, None]

    class_scalars = build_segmentation_class_scalars(class_rows=class_rows)
    assert sorted(class_scalars) == sorted(
        [f"{metric_name}/{class_name}" for class_name in ["background", "circle"]
         for metric_name in ["IoU", "Dice", "Precision", "Recall"]] + ["Boundary IoU/background",
                                                                       "Boundary F1/background"])
