"""
Tests of vision_training/evaluation/classification/score_curves.py: the ROC AUC and average precision of every class
match scikit-learn on scores with ties, the threshold reaching the target precision is marked at its recall and
precision, separated, reversed and undefined classes get their expected metrics, the macro means leave out the
undefined classes (NaN without any), the curves of the pixel histograms of segmentation_metrics.py equal the exact
curves of the same pixels, the drawn curves keep at most MAXIMUM_CURVE_POINTS points while their metrics use every
point, and the images of a few classes (single plot) and of many classes (grid of panels, the lowest classes above
MAXIMUM_GRID_PANELS) are saved identical to the returned ones.
"""

import cv2
import math
import pytest
import torch

from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Tuple

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.evaluation.classification.confusion_matrix import IMAGE_DOTS_PER_INCH
from vision_training.evaluation.classification.score_curves import (MAXIMUM_CURVE_POINTS, MAXIMUM_GRID_PANELS,
                                                                     OVERLAY_FIGURE_SIZE, compute_class_score_curves,
                                                                     compute_histogram_score_curves,
                                                                     compute_macro_curve_metrics,
                                                                     save_score_curve_images, select_drawn_classes)
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
    those of scikit-learn, and that the point marked on the precision-recall curve has the recall of the threshold
    reaching the target precision and the precision of the samples whose score reaches the threshold.
    """
    probabilities, positive_masks = build_random_scores(number_samples=500, number_classes=4, number_levels=50, seed=0)
    class_thresholds = find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                             target_precision=0.7)
    class_curves = compute_class_score_curves(probabilities=probabilities, positive_masks=positive_masks,
                                              class_thresholds=class_thresholds)

    for class_index, curves in enumerate(class_curves):
        labels, scores = positive_masks[:, class_index], probabilities[:, class_index]
        assert curves["number_positives"] == int(labels.sum())
        assert curves["roc_auc"] == pytest.approx(expected=roc_auc_score(y_true=labels.numpy(), y_score=scores.numpy()),
                                                  abs=1e-12)
        assert curves["average_precision"] == pytest.approx(
            expected=average_precision_score(y_true=labels.numpy(), y_score=scores.numpy()), abs=1e-12)

        # Point of the threshold: its recall, and the precision of the samples accepted by it
        threshold, threshold_recall = class_thresholds[class_index]
        accepted_samples = scores >= threshold
        assert curves["threshold_point"] == pytest.approx(
            expected=(threshold_recall, float(labels[accepted_samples].double().mean())))
        assert curves["threshold_point"][1] >= 0.7


def test_separated_reversed_and_undefined_classes() -> None:
    """
    Verifies the metrics of four classes: separated scores give 1 and 1, reversed scores a ROC AUC of 0 and the
    average precision of their cuts, a class without positive no metric and empty curves, a class without negative an
    average precision of 1 and no ROC AUC, and the macro means leave the undefined classes out, being NaN without any
    class defining them.
    """
    probabilities = torch.tensor(data=[[0.9, 0.1, 0.5, 0.5], [0.8, 0.2, 0.4, 0.6], [0.2, 0.8, 0.3, 0.7],
                                       [0.1, 0.9, 0.2, 0.8]], dtype=torch.float64)
    positive_masks = torch.tensor(data=[[True, True, False, True], [True, True, False, True],
                                        [False, False, False, True], [False, False, False, True]])
    class_curves = compute_class_score_curves(probabilities=probabilities, positive_masks=positive_masks)

    # Reversed: cuts at recall 0.5 (precision 1/3) and 1 (precision 2/4)
    assert [(curves["roc_auc"], curves["average_precision"]) for curves in class_curves] == [
        (1.0, 1.0), (0.0, pytest.approx(expected=0.5 / 3 + 0.5 * 0.5)), (None, None), (None, 1.0)]
    assert len(class_curves[2]["recalls"]) == 0 and len(class_curves[3]["false_positive_rates"]) == 0
    assert class_curves[0]["threshold_point"] is None

    # Macro means over the classes defining every metric, NaN without any
    macro_metrics = compute_macro_curve_metrics(class_curves=class_curves)
    assert macro_metrics == {"Macro ROC AUC": 0.5,
                             "Macro Average Precision": pytest.approx(expected=(2.0 + 0.5 / 3 + 0.5 * 0.5) / 3)}
    assert all(math.isnan(value) for value in compute_macro_curve_metrics(class_curves=class_curves[2:3]).values())


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


def test_drawn_curves_keep_at_most_the_maximum_points() -> None:
    """
    Verifies that the curves of 10,000 distinct scores keep at most MAXIMUM_CURVE_POINTS points to be drawn, from
    (0, 0) to (1, 1) for the ROC curve, while their metrics, computed on every point, still match scikit-learn.
    """
    probabilities, positive_masks = build_random_scores(number_samples=10000, number_classes=1,
                                                        number_levels=10 ** 9, seed=2)
    curves, = compute_class_score_curves(probabilities=probabilities, positive_masks=positive_masks)
    assert len(curves["false_positive_rates"]) <= MAXIMUM_CURVE_POINTS
    assert len(curves["recalls"]) <= MAXIMUM_CURVE_POINTS
    assert (curves["false_positive_rates"][0], curves["true_positive_rates"][0]) == (0.0, 0.0)
    assert (curves["false_positive_rates"][-1], curves["true_positive_rates"][-1]) == (1.0, 1.0)
    assert curves["roc_auc"] == pytest.approx(expected=roc_auc_score(y_true=positive_masks[:, 0].numpy(),
                                                                     y_score=probabilities[:, 0].numpy()), abs=1e-12)


def test_curve_images_of_a_few_and_of_many_classes(tmp_path: Path) -> None:
    """
    Verifies the curve images: 3 classes are drawn on a single plot of OVERLAY_FIGURE_SIZE, 12 classes on a grid of 4
    columns and 3 rows of panels, both images of every evaluation being saved under their file name identical to the
    returned RGB images, and above MAXIMUM_GRID_PANELS classes only the classes with the lowest metric are drawn.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    expected_shapes = {3: (int(OVERLAY_FIGURE_SIZE[1] * IMAGE_DOTS_PER_INCH),
                           int(OVERLAY_FIGURE_SIZE[0] * IMAGE_DOTS_PER_INCH), 3),
                       12: (int((3 * 2.0 + 0.8) * IMAGE_DOTS_PER_INCH), int(4 * 2.0 * IMAGE_DOTS_PER_INCH), 3)}
    for number_classes, expected_shape in expected_shapes.items():
        probabilities, positive_masks = build_random_scores(number_samples=200, number_classes=number_classes,
                                                            number_levels=20, seed=3)
        class_curves = compute_class_score_curves(probabilities=probabilities, positive_masks=positive_masks)
        image_paths, images = save_score_curve_images(
            class_curves=class_curves, class_names=[f"class_{index}" for index in range(number_classes)],
            output_folder=tmp_path / f"classes_{number_classes}", title_suffix=" · Iteration 4", target_precision=0.9)

        # Both images, saved under their names and identical to the returned ones
        assert list(images) == ["ROC Curves", "Precision Recall Curves"]
        assert [path.name for path in image_paths.values()] == ["roc_curves.png", "precision_recall_curves.png"]
        for image_name, image in images.items():
            assert image.shape == expected_shape and image.dtype.name == "uint8"
            saved_image = cv2.imread(filename=str(image_paths[f"{image_name} Image"]))
            assert (cv2.cvtColor(src=saved_image, code=cv2.COLOR_BGR2RGB) == image).all()

    # Above MAXIMUM_GRID_PANELS classes, the classes with the lowest metric, the classes without one left out
    many_class_curves = [{"roc_auc": float(class_index % 7) + class_index / 1000} for class_index in range(150)]
    many_class_curves[3]["roc_auc"] = None
    drawn_classes = select_drawn_classes(class_curves=many_class_curves, metric_key="roc_auc")
    assert len(drawn_classes) == MAXIMUM_GRID_PANELS and 3 not in drawn_classes
    assert drawn_classes == sorted(drawn_classes)
    lowest_values = sorted(curves["roc_auc"] for curves in many_class_curves if curves["roc_auc"] is not None)
    assert sorted(many_class_curves[class_index]["roc_auc"] for class_index in drawn_classes) == \
        lowest_values[:MAXIMUM_GRID_PANELS]


def test_curve_image_titles_stay_inside_the_image(tmp_path: Path) -> None:
    """
    Verifies that the titles of the curve images of a segmentation evaluation, the longest ones (pixels, iteration and
    macro mean), stay inside the single plot: the left and right columns of the title rows hold no ink, where a title
    on one line used to run past both edges.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    probabilities, positive_masks = build_random_scores(number_samples=200, number_classes=10, number_levels=20,
                                                        seed=5)
    class_curves = compute_class_score_curves(probabilities=probabilities, positive_masks=positive_masks)
    _, images = save_score_curve_images(class_curves=class_curves,
                                        class_names=[f"class_{index}" for index in range(10)],
                                        output_folder=tmp_path, title_suffix=" · Pixels · Iteration 20000")
    for image in images.values():
        # The title rows are the top tenth of the image, white outside the text
        title_rows = image[:image.shape[0] // 10]
        assert (title_rows[:, :3] > 200).all() and (title_rows[:, -3:] > 200).all()
