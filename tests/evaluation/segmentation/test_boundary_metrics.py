"""
Tests of vision_training/evaluation/segmentation/boundary_metrics.py: the boundary IoU and boundary F1 of a perfect
prediction, of an object predicted a few pixels too wide (high pixel IoU, no right contour within a smaller
tolerance), of predictions on ignored pixels, and of a class without true edge.
"""

import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.evaluation.segmentation.boundary_metrics import (BOUNDARY_COUNT_COLUMNS,
                                                                      compute_boundary_counts,
                                                                      compute_boundary_metrics)


def build_square_masks(image_size: int, square_start: int, square_end: int) -> torch.Tensor:
    """
    Builds a [1, S, S] mask of a square object of class 1 on a background of class 0, the true masks and predicted
    class maps of the boundary metric tests.

    Args:
        image_size (int): The side S of the image.
        square_start (int): The first row and column of the square.
        square_end (int): The row and column after the last one of the square.

    Returns:
        torch.Tensor: The [1, S, S] int64 class index of every pixel.
    """
    masks = torch.zeros(size=(1, image_size, image_size), dtype=torch.int64)
    masks[0, square_start:square_end, square_start:square_end] = 1
    return masks


def test_perfect_prediction_has_boundary_metrics_of_one() -> None:
    """
    Verifies that a prediction equal to the masks gives a boundary IoU and boundary F1 of 1 for every class with edges,
    and that a class absent from the masks has no boundary metric and is left out of the means.
    """
    masks = build_square_masks(image_size=32, square_start=8, square_end=24)
    boundary_counts = compute_boundary_counts(class_maps=masks, masks=masks, number_classes=3, tolerance=2)
    assert boundary_counts.shape == (3, len(BOUNDARY_COUNT_COLUMNS)) and boundary_counts.dtype == torch.int64

    boundary_metrics = compute_boundary_metrics(boundary_counts=boundary_counts)
    assert boundary_metrics["mean_boundary_iou"] == pytest.approx(expected=1.0)
    assert boundary_metrics["mean_boundary_f1"] == pytest.approx(expected=1.0)
    assert boundary_metrics["boundary_iou"] == [1.0, 1.0, None] and boundary_metrics["boundary_f1"] == [1.0, 1.0, None]


def test_contours_further_than_the_tolerance_are_wrong() -> None:
    """
    Verifies that a square predicted 3 pixels too wide on every side keeps a high pixel IoU, but that none of its
    contour is right within a tolerance of 2 pixels (boundary IoU and F1 of 0), while all of it is right within a
    tolerance of 3 pixels (boundary F1 of 1).
    """
    masks = build_square_masks(image_size=64, square_start=8, square_end=56)
    class_maps = build_square_masks(image_size=64, square_start=5, square_end=59)
    pixel_iou = float(((class_maps == 1) & (masks == 1)).sum() / ((class_maps == 1) | (masks == 1)).sum())
    assert pixel_iou > 0.75

    strict_metrics = compute_boundary_metrics(boundary_counts=compute_boundary_counts(
        class_maps=class_maps, masks=masks, number_classes=2, tolerance=2))
    assert strict_metrics["mean_boundary_iou"] == 0.0 and strict_metrics["mean_boundary_f1"] == 0.0

    tolerant_metrics = compute_boundary_metrics(boundary_counts=compute_boundary_counts(
        class_maps=class_maps, masks=masks, number_classes=2, tolerance=3))
    assert tolerant_metrics["boundary_f1"] == [pytest.approx(expected=1.0), pytest.approx(expected=1.0)]


def test_ignored_pixels_draw_no_contour() -> None:
    """
    Verifies that the border between labelled and ignored pixels is no contour and that the predictions on the ignored
    pixels are left out: a prediction right on the labelled pixels and wrong on the ignored ones scores 1, and a mask
    whose only class touches nothing but ignored pixels has no boundary metric.
    """
    masks = build_square_masks(image_size=32, square_start=4, square_end=12)
    masks[0, 16:] = IGNORED_LABEL_INDEX
    class_maps = masks.clamp(min=0)
    class_maps[0, 16:] = 1
    boundary_metrics = compute_boundary_metrics(boundary_counts=compute_boundary_counts(
        class_maps=class_maps, masks=masks, number_classes=2, tolerance=1))
    assert boundary_metrics["boundary_iou"] == [1.0, 1.0] and boundary_metrics["boundary_f1"] == [1.0, 1.0]

    # Background on the top half and ignored pixels on the bottom half: no true edge at all
    background_masks = torch.zeros(size=(1, 32, 32), dtype=torch.int64)
    background_masks[0, 16:] = IGNORED_LABEL_INDEX
    empty_metrics = compute_boundary_metrics(boundary_counts=compute_boundary_counts(
        class_maps=class_maps, masks=background_masks, number_classes=2, tolerance=1))
    assert empty_metrics == {"mean_boundary_iou": 0.0, "mean_boundary_f1": 0.0, "boundary_iou": [None, None],
                             "boundary_f1": [None, None]}
