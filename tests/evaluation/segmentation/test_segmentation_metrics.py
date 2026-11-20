"""
Tests of vision_training/evaluation/segmentation/segmentation_metrics.py: the pixel confusion matrix and the per-image
pixel counts leave the ignored pixels out, and the metrics computed from a known confusion matrix match the values
computed by hand with the classes predicted without true pixel counted in the means and the classes neither present
nor predicted left out.
"""

import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.evaluation.segmentation.segmentation_metrics import (compute_image_class_pixel_counts,
                                                                          compute_pixel_confusion_matrix,
                                                                          compute_segmentation_metrics)


def test_pixel_confusion_matrix_leaves_out_the_ignored_pixels() -> None:
    """
    Verifies that every labelled pixel is counted once at its true and predicted classes, whatever the prediction on
    the ignored pixels.
    """
    masks = torch.tensor(data=[[[0, 1], [2, IGNORED_LABEL_INDEX]]])
    class_maps = torch.tensor(data=[[[0, 2], [2, 1]]])

    confusion_matrix = compute_pixel_confusion_matrix(class_maps=class_maps, masks=masks, number_classes=3)
    assert confusion_matrix.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]


def test_image_class_pixel_counts_add_up_to_the_confusion_matrix() -> None:
    """
    Verifies the per-image counts of the Korn-Graubard interval of the recall: the true pixels of every class in every
    image and those predicted as their class, the ignored pixels left out, adding up over the images to the rows and
    the diagonal of the confusion matrix.
    """
    masks = torch.tensor(data=[[[0, 1], [1, IGNORED_LABEL_INDEX]], [[2, 2], [0, 1]]])
    class_maps = torch.tensor(data=[[[0, 1], [0, 2]], [[2, 1], [0, 1]]])

    true_positive_pixels, true_pixels = compute_image_class_pixel_counts(class_maps=class_maps, masks=masks,
                                                                         number_classes=3)
    assert true_pixels.tolist() == [[1, 2, 0], [1, 1, 2]]
    assert true_positive_pixels.tolist() == [[1, 1, 0], [1, 1, 1]]
    confusion_matrix = compute_pixel_confusion_matrix(class_maps=class_maps, masks=masks, number_classes=3)
    assert true_pixels.sum(dim=0).tolist() == confusion_matrix.sum(dim=1).tolist()
    assert true_positive_pixels.sum(dim=0).tolist() == confusion_matrix.diagonal().tolist()


def test_metrics_of_a_known_confusion_matrix() -> None:
    """
    Verifies the pixel accuracy, the IoU and Dice of every class, their means over the classes present or predicted
    and the frequency-weighted IoU on a confusion matrix whose class 2 has no true pixel but is predicted once.
    """
    # Rows true classes, columns predicted classes: 8 background pixels (one predicted as class 2), 4 pixels of class 1
    # (one predicted as background), no pixel of class 2
    confusion_matrix = torch.tensor(data=[[7, 0, 1], [1, 3, 0], [0, 0, 0]])
    metrics = compute_segmentation_metrics(confusion_matrix=confusion_matrix)

    # IoU: background 7 / (8 + 8 - 7), class 1 3 / (4 + 3 - 3), class 2 0 (absent from the true masks)
    background_iou, class_iou = 7.0 / 9.0, 3.0 / 4.0
    assert metrics["pixel_accuracy"] == pytest.approx(expected=10.0 / 12.0)
    assert metrics["iou"] == pytest.approx(expected=[background_iou, class_iou, 0.0])
    # The mean counts class 2, predicted on a background pixel without any true pixel, with its IoU of 0
    assert metrics["mean_iou"] == pytest.approx(expected=(background_iou + class_iou + 0.0) / 3.0)
    assert metrics["frequency_weighted_iou"] == pytest.approx(expected=(8.0 * background_iou + 4.0 * class_iou) / 12.0)

    # Dice: twice the right pixels over the true and predicted pixels, the mean counting the predicted class 2
    background_dice, class_dice = 14.0 / 16.0, 6.0 / 7.0
    assert metrics["dice"] == pytest.approx(expected=[background_dice, class_dice, 0.0])
    assert metrics["mean_dice"] == pytest.approx(expected=(background_dice + class_dice + 0.0) / 3.0)
    assert metrics["precision"] == pytest.approx(expected=[7.0 / 8.0, 1.0, 0.0])
    assert metrics["recall"] == pytest.approx(expected=[7.0 / 8.0, 3.0 / 4.0, 0.0])
    assert metrics["support"] == [8, 4, 0]


def test_class_painted_without_true_pixel_lowers_the_mean_iou() -> None:
    """
    Verifies the case found by the audit of 2026-10-20: masks of classes 0 and 1, class 2 painted on a quarter of the pixels (half of
    class 0), give the IoU [0.5, 1, 0] and a mean IoU of 0.5 as in mmseg and Cityscapes (every class with a non-empty
    union) instead of the 0.75 of the classes present only, the mean Dice counting class 2 as well, and a class neither
    present nor predicted staying out of both means.
    """
    confusion_matrix = torch.tensor(data=[[2, 0, 2, 0], [0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    metrics = compute_segmentation_metrics(confusion_matrix=confusion_matrix)
    assert metrics["iou"] == pytest.approx(expected=[0.5, 1.0, 0.0, 0.0])
    assert metrics["mean_iou"] == pytest.approx(expected=0.5)
    assert metrics["dice"] == pytest.approx(expected=[2 / 3, 1.0, 0.0, 0.0])
    assert metrics["mean_dice"] == pytest.approx(expected=(2 / 3 + 1.0 + 0.0) / 3)
