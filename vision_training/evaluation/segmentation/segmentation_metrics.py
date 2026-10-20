"""
Metrics of the semantic segmentation task, computed from the confusion matrix of the labelled pixels (rows true
classes, columns predicted classes): the pixel accuracy, the IoU and Dice of every class with their means over the
classes present, the frequency-weighted IoU, and the precision and recall of every class. Used by the segmentation
trainer on every training and validation batch and on the whole test split of an evaluated checkpoint, which also
counts, with the per-class report, the probabilities of the labelled pixels in histograms giving the ROC and
precision-recall curves of every class (score_curves.py).
"""

import torch

from typing import Any, Dict

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.evaluation.classification.classification_metrics import (compute_confusion_matrix,
                                                                              compute_metrics_from_confusion_matrix)

# Number of bins of equal width over [0, 1] of the histograms of the pixel probabilities, every bin edge being a
# threshold of the ROC and precision-recall curves of the segmentation (every 0.0001 of probability)
SCORE_HISTOGRAM_BINS = 10000


def compute_pixel_confusion_matrix(class_maps: torch.Tensor, masks: torch.Tensor, number_classes: int) -> torch.Tensor:
    """
    Counts the labelled pixels of a batch for every pair of true and predicted classes, on the device of the masks,
    the ignored pixels (padding, unlabelled) being left out. The segmentation trainer adds up the matrices of every
    test batch on the device, so that no pixel has to be kept on the CPU during an evaluation.

    Args:
        class_maps (torch.Tensor): The [B, H, W] predicted class index of every pixel.
        masks (torch.Tensor): The [B, H, W] true class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.
        number_classes (int): The number of classes K, background included.

    Returns:
        torch.Tensor: The [K, K] int64 confusion matrix of the labelled pixels.
    """
    labelled_pixels = masks != IGNORED_LABEL_INDEX
    return compute_confusion_matrix(predicted_labels=class_maps[labelled_pixels], labels=masks[labelled_pixels],
                                    number_classes=number_classes)


def compute_pixel_score_histograms(class_probabilities: torch.Tensor, masks: torch.Tensor,
                                   number_bins: int) -> torch.Tensor:
    """
    Counts the labelled pixels of a batch in histograms of their class probabilities, on the device of the masks, for
    the ROC and precision-recall curves of the per-class report (compute_histogram_score_curves): for every class, the
    probabilities of the class of the pixels of the other classes (negatives) and of the pixels of the class
    (positives), the ignored pixels (padding, unlabelled) being left out like in the confusion matrix. The segmentation
    trainer adds up the histograms of every test batch on the device, the probabilities of all the pixels being far
    too many to keep. The classes are counted one by one so that no [pixels, classes] copy is made.

    Args:
        class_probabilities (torch.Tensor): The [B, K, H, W] class probabilities of every pixel.
        masks (torch.Tensor): The [B, H, W] true class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.
        number_bins (int): The number of bins of equal width over [0, 1], a probability of 1 falling in the last one.

    Returns:
        torch.Tensor: The [K, 2, number_bins] int64 histograms, the negatives first and the positives second.
    """
    labelled_pixels = masks != IGNORED_LABEL_INDEX
    pixel_labels = masks[labelled_pixels]
    number_classes = class_probabilities.shape[1]
    histograms = torch.zeros(size=(number_classes, 2, number_bins), dtype=torch.int64, device=masks.device)
    for class_index in range(number_classes):
        # Bin of the probability of the class of every labelled pixel, and whether the pixel is of the class
        pixel_bins = (class_probabilities[:, class_index][labelled_pixels].to(dtype=torch.float32) *
                      number_bins).long().clamp(min=0, max=number_bins - 1)
        is_positive = (pixel_labels == class_index).long()

        # Negatives in the first histogram and positives in the second, counted together
        histograms[class_index] = torch.bincount(input=is_positive * number_bins + pixel_bins,
                                                 minlength=2 * number_bins).view(2, number_bins)
    return histograms


def compute_segmentation_metrics(confusion_matrix: torch.Tensor) -> Dict[str, Any]:
    """
    Computes the segmentation metrics of a batch or of a whole test split from its pixel confusion matrix: the pixel
    accuracy (share of the labelled pixels given their class), the IoU (intersection over union of the predicted and
    true pixels) and Dice (twice the intersection over their sum, the F1 score of the pixels) of every class, their
    means over the classes present in the true masks, a class absent from them being left out like in the macro
    averages of the classifiers, and the frequency-weighted IoU (IoU of every class weighted by its share of the
    pixels).

    Args:
        confusion_matrix (torch.Tensor): The [K, K] pixel confusion matrix (rows true classes, columns predicted
            classes).

    Returns:
        Dict[str, Any]: The 'pixel_accuracy', 'mean_iou', 'frequency_weighted_iou' and 'mean_dice' floats (0 without
            labelled pixel), the per-class 'iou', 'dice', 'precision' and 'recall' lists (0 for a class neither present
            nor predicted), and the per-class 'support' list of true pixels.
    """
    # Accuracy, and precision, recall and F1 (the Dice) of every class with their mean over the classes present
    classification_metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)

    # Intersection over union of every class, from its right pixels, its true pixels and its predicted pixels
    pixel_counts = confusion_matrix.to(dtype=torch.float64)
    true_positives, true_counts = pixel_counts.diagonal(), pixel_counts.sum(dim=1)
    union_counts = true_counts + pixel_counts.sum(dim=0) - true_positives
    intersection_over_union = torch.where(condition=union_counts > 0,
                                          input=true_positives / union_counts.clamp(min=1.0), other=0.0)

    # Means over the classes present in the true masks, and the mean weighted by the share of the pixels
    present_classes = true_counts > 0
    number_pixels = float(true_counts.sum())
    return {"pixel_accuracy": classification_metrics["accuracy"],
            "mean_iou": float(intersection_over_union[present_classes].mean()) if present_classes.any() else 0.0,
            "frequency_weighted_iou": float((true_counts * intersection_over_union).sum()) / number_pixels
            if number_pixels else 0.0,
            "mean_dice": classification_metrics["macro_f1"],
            "iou": intersection_over_union.tolist(), "dice": classification_metrics["f1"],
            "precision": classification_metrics["precision"], "recall": classification_metrics["recall"],
            "support": classification_metrics["support"]}
