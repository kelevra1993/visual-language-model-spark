"""
Boundary metrics of the semantic segmentation task, which only score the pixels near the contours of the regions,
where the mean IoU hardly sees an error (an object predicted 3 pixels too wide on every side keeps a high IoU while
none of its predicted contour is right): the boundary IoU (IoU of the bands of the true and predicted regions within
the tolerance of their contours) and the boundary F1 (share of the predicted contour within the tolerance of the true
one, and the reverse). The counts of every class are added up on the device over the test split by the segmentation
trainer, then turned into the metrics of every class and their means.
"""

import torch

from typing import Any, Dict

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.utilities.mask_morphology import compute_contour_bands, dilate_masks

# Counts of every class added up over the test split, in the order of the columns of the boundary counts
BOUNDARY_COUNT_COLUMNS = ["band_intersection", "band_union", "predicted_edges", "matched_predicted_edges",
                          "true_edges", "matched_true_edges"]


def compute_boundary_counts(class_maps: torch.Tensor, masks: torch.Tensor, number_classes: int,
                            tolerance: int) -> torch.Tensor:
    """
    Counts the boundary pixels of every class in a batch, on the device of the masks, for the boundary metrics of an
    evaluated checkpoint: the intersection and union of the true and predicted bands within tolerance pixels of their
    contours, the predicted edge pixels and those within tolerance pixels of a true edge, and the true edge pixels and
    those within tolerance pixels of a predicted edge. The ignored pixels (padding, unlabelled) draw no contour and
    their predictions are left out.

    Args:
        class_maps (torch.Tensor): The [B, H, W] predicted class index of every pixel.
        masks (torch.Tensor): The [B, H, W] true class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.
        number_classes (int): The number of classes K, background included.
        tolerance (int): The distance in pixels within which a contour pixel is right, at least 1.

    Returns:
        torch.Tensor: The [K, 6] int64 counts of every class, in the order of BOUNDARY_COUNT_COLUMNS.
    """
    ignored_pixels = masks == IGNORED_LABEL_INDEX
    class_counts = []
    for class_index in range(number_classes):
        true_regions = masks == class_index
        predicted_regions = (class_maps == class_index) & ~ignored_pixels

        # Bands of the true and predicted regions within the tolerance of their contours, for the boundary IoU
        true_bands = compute_contour_bands(regions=true_regions, neutral_pixels=ignored_pixels, radius=tolerance)
        predicted_bands = compute_contour_bands(regions=predicted_regions, neutral_pixels=ignored_pixels,
                                                radius=tolerance)

        # Edge pixels of both regions, each matched when an edge of the other region lies within the tolerance
        true_edges = compute_contour_bands(regions=true_regions, neutral_pixels=ignored_pixels, radius=1)
        predicted_edges = compute_contour_bands(regions=predicted_regions, neutral_pixels=ignored_pixels, radius=1)
        matched_predicted_edges = predicted_edges & dilate_masks(masks=true_edges, radius=tolerance)
        matched_true_edges = true_edges & dilate_masks(masks=predicted_edges, radius=tolerance)

        class_counts.append(torch.stack(tensors=[(true_bands & predicted_bands).sum(),
                                                 (true_bands | predicted_bands).sum(), predicted_edges.sum(),
                                                 matched_predicted_edges.sum(), true_edges.sum(),
                                                 matched_true_edges.sum()]))
    return torch.stack(tensors=class_counts).to(dtype=torch.int64)


def compute_boundary_metrics(boundary_counts: torch.Tensor) -> Dict[str, Any]:
    """
    Computes the boundary metrics of a test split from its boundary counts: for every class with true edges, the
    boundary IoU (intersection over union of the true and predicted bands) and the boundary F1 (harmonic mean of the
    share of the predicted edges near a true edge and of the true edges near a predicted edge), and their means over
    these classes. A class without true edge (absent from the test masks, or covering whole images) has no boundary
    metric and is left out of the means.

    Args:
        boundary_counts (torch.Tensor): The [K, 6] counts of every class (compute_boundary_counts), summed over the
            test split.

    Returns:
        Dict[str, Any]: The 'mean_boundary_iou' and 'mean_boundary_f1' floats (0 without any true edge), and the
            per-class 'boundary_iou' and 'boundary_f1' lists, None for the classes without true edge.
    """
    counts = boundary_counts.to(dtype=torch.float64)
    band_intersections, band_unions, predicted_edges, matched_predicted_edges, true_edges, matched_true_edges = \
        counts.unbind(dim=1)

    # Boundary IoU of the bands, and precision and recall of the edges, 0 where nothing was counted
    boundary_iou = torch.where(condition=band_unions > 0, input=band_intersections / band_unions.clamp(min=1.0),
                               other=0.0)
    edge_precision = torch.where(condition=predicted_edges > 0,
                                 input=matched_predicted_edges / predicted_edges.clamp(min=1.0), other=0.0)
    edge_recall = torch.where(condition=true_edges > 0, input=matched_true_edges / true_edges.clamp(min=1.0),
                              other=0.0)
    precision_recall_sum = edge_precision + edge_recall
    boundary_f1 = torch.where(condition=precision_recall_sum > 0,
                              input=2.0 * edge_precision * edge_recall / precision_recall_sum.clamp(min=1e-12),
                              other=0.0)

    # Means over the classes with true edges, the other classes having no boundary metric
    scored_classes = true_edges > 0
    return {"mean_boundary_iou": float(boundary_iou[scored_classes].mean()) if scored_classes.any() else 0.0,
            "mean_boundary_f1": float(boundary_f1[scored_classes].mean()) if scored_classes.any() else 0.0,
            "boundary_iou": [float(value) if scored else None
                             for value, scored in zip(boundary_iou.tolist(), scored_classes.tolist())],
            "boundary_f1": [float(value) if scored else None
                            for value, scored in zip(boundary_f1.tolist(), scored_classes.tolist())]}
