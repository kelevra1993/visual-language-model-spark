"""
Tests of vision_training/evaluation/classification/score_curves.py: the ROC AUC and average precision of every class
match scikit-learn on scores with ties, the threshold reaching every target precision is marked at its recall and
precision with a written value selecting the same samples, and the curves of the pixel histograms of
segmentation_metrics.py equal the exact curves of the same pixels.
"""

import pytest
import torch

from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Tuple

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.evaluation.classification.score_curves import (compute_class_score_curves,
                                                                     compute_histogram_score_curves,
                                                                     format_threshold_label)
from vision_training.evaluation.classification.threshold_finder import find_class_thresholds
from vision_training.evaluation.segmentation.segmentation_metrics import compute_pixel_score_histograms


def build_random_scores(number_samples: int, number_classes: int, number_levels: int,
                        seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds the probabilities and positives of a random one-against-the-rest evaluation for the tests of the curves,
    the scores rounded to a few levels so that many of them are tied, a sample being positive more often with a high
    score.

    Args:
        number_samples (int): The number of samples N.
        number_classes (int): The number of classes K.
        number_levels (int): The number of distinct score levels, a few for many ties.
        seed (int): The seed of the random generator.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [N, K] float64 probabilities and the [N, K] boolean positives.
    """
    generator = torch.Generator().manual_seed(seed)
    probabilities = (torch.rand(size=(number_samples, number_classes), generator=generator, dtype=torch.float64) *
                     number_levels).round() / number_levels
    positive_masks = torch.rand(size=(number_samples, number_classes), generator=generator,
                                dtype=torch.float64) < probabilities * 0.8
    return probabilities, positive_masks


def test_curve_metrics_match_scikit_learn_with_ties() -> None:
    """
    Verifies on 500 samples of 4 classes with tied scores that the ROC AUC and average precision of every class equal
    those of scikit-learn, and that the point marked on the precision-recall curve for every target precision has the
    recall of the threshold reaching it and the precision of the samples whose score reaches the threshold, its written
    value selecting the same samples as the exact threshold.
    """
    probabilities, positive_masks = build_random_scores(number_samples=500, number_classes=4, number_levels=50, seed=0)
    target_precisions = [0.7, 0.6]
    class_thresholds = [find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                              target_precision=target_precision)
                        for target_precision in target_precisions]
    class_curves = compute_class_score_curves(probabilities=probabilities, positive_masks=positive_masks,
                                              class_thresholds=class_thresholds)

    for class_index, curves in enumerate(class_curves):
        labels, scores = positive_masks[:, class_index], probabilities[:, class_index]
        assert curves["number_positives"] == int(labels.sum())
        assert curves["roc_auc"] == pytest.approx(expected=roc_auc_score(y_true=labels.numpy(), y_score=scores.numpy()),
                                                  abs=1e-12)
        assert curves["average_precision"] == pytest.approx(
            expected=average_precision_score(y_true=labels.numpy(), y_score=scores.numpy()), abs=1e-12)

        # Point of the threshold of every target precision: its recall, the precision of the samples accepted by it,
        # and its written value accepting the same samples
        assert len(curves["threshold_points"]) == len(target_precisions)
        for target_precision, target_thresholds, threshold_point in zip(target_precisions, class_thresholds,
                                                                        curves["threshold_points"]):
            threshold, threshold_recall = target_thresholds[class_index]
            accepted_samples = scores >= threshold
            assert threshold_point[:2] == pytest.approx(
                expected=(threshold_recall, float(labels[accepted_samples].double().mean())))
            assert threshold_point[1] >= target_precision
            assert torch.equal(scores >= float(threshold_point[2]), accepted_samples)


def test_histogram_curves_equal_the_exact_curves_of_the_same_pixels() -> None:
    """
    Verifies that the curves of the segmentation, from the histograms of the probabilities of the labelled pixels,
    equal the exact curves of the same pixels when every probability lies inside its own bin: same ROC AUC, average
    precision and curve points, the ignored pixels being left out.
    """
    number_bins = 50
    generator = torch.Generator().manual_seed(1)
    masks = torch.randint(low=0, high=3, size=(2, 16, 16), generator=generator)
    masks[:, :2] = IGNORED_LABEL_INDEX
    levels = torch.randint(low=0, high=number_bins, size=(2, 3, 16, 16), generator=generator)
    class_probabilities = (levels.to(dtype=torch.float32) + 0.5) / number_bins

    # Histograms of the labelled pixels, and the exact curves of the same pixels
    score_histograms = compute_pixel_score_histograms(class_probabilities=class_probabilities, masks=masks,
                                                      number_bins=number_bins)
    labelled_pixels = masks != IGNORED_LABEL_INDEX
    assert int(score_histograms.sum()) == 3 * int(labelled_pixels.sum())
    exact_curves = compute_class_score_curves(
        probabilities=class_probabilities.permute(0, 2, 3, 1)[labelled_pixels].to(dtype=torch.float64),
        positive_masks=torch.nn.functional.one_hot(masks[labelled_pixels], num_classes=3).to(dtype=torch.bool))
    histogram_curves = compute_histogram_score_curves(score_histograms=score_histograms)

    for exact, binned in zip(exact_curves, histogram_curves):
        assert binned["roc_auc"] == pytest.approx(expected=exact["roc_auc"], abs=1e-12)
        assert binned["average_precision"] == pytest.approx(expected=exact["average_precision"], abs=1e-12)
        for curve_key in ["false_positive_rates", "true_positive_rates", "recalls", "precisions"]:
            assert binned[curve_key] == pytest.approx(expected=exact[curve_key], abs=1e-12)


def test_threshold_labels_select_the_same_samples() -> None:
    """
    Verifies the written thresholds: rounded down to 4 decimals when the next lower score stays below, with more
    decimals otherwise (0.99988 and not 0.9998 under a score of 0.99986), and up to 9 significant digits for the tiny
    thresholds of a confident class, so that a threshold read from the image keeps the samples of the exact one.
    """
    assert format_threshold_label(threshold=0.999885, next_lower_score=None) == "0.9998"
    assert format_threshold_label(threshold=0.999885, next_lower_score=0.99986) == "0.99988"
    assert format_threshold_label(threshold=0.00052122, next_lower_score=0.0005211) == "0.0005212"
    assert format_threshold_label(threshold=3.215e-7, next_lower_score=3.21e-7) == "0.0000003215"
    assert format_threshold_label(threshold=0.29, next_lower_score=0.28) == "0.2900"
