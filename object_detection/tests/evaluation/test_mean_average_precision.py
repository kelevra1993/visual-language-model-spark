import pytest
import torch

from typing import Dict, List

from evaluation.mean_average_precision import (compute_average_precision, compute_precision_recall,
                                               evaluate_detections, gather_class_detections,
                                               match_detections_to_ground_truth)


def make_detections(boxes: List[List[float]], scores: List[float], labels: List[int]) -> Dict[str, torch.Tensor]:
    """
    Builds the detections of one image in the format collected by the Trainer test evaluation.

    Args:
        boxes (List[List[float]]): The [x1, y1, x2, y2] detection boxes.
        scores (List[float]): The detection scores.
        labels (List[int]): The detection class indices.

    Returns:
        Dict[str, torch.Tensor]: The 'boxes' [N, 4], 'scores' [N] and 'labels' [N] of the image.
    """
    return {"boxes": torch.tensor(data=boxes, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.tensor(data=scores, dtype=torch.float32), "labels": torch.tensor(data=labels)}


def make_ground_truth(boxes: List[List[float]], labels: List[int]) -> Dict[str, torch.Tensor]:
    """
    Builds the ground truth of one image in the format collected by the Trainer test evaluation.

    Args:
        boxes (List[List[float]]): The [x1, y1, x2, y2] ground truth boxes.
        labels (List[int]): The ground truth class indices.

    Returns:
        Dict[str, torch.Tensor]: The 'boxes' [M, 4] and 'labels' [M] of the image.
    """
    return {"boxes": torch.tensor(data=boxes, dtype=torch.float32).reshape(-1, 4), "labels": torch.tensor(data=labels)}


def get_worked_example() -> tuple:
    """
    Builds a single-class example following the one of the video: 5 ground truth objects in 2 images and 5
    detections which are, by decreasing score, a true positive, a false positive (overlap of 0.4 only), then three
    true positives, so that one object of the second image is never found.

    Returns:
        tuple: The detections and ground truths of the two images.
    """
    ground_truths = [make_ground_truth(boxes=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], labels=[1, 1, 1]),
                     make_ground_truth(boxes=[[0, 0, 10, 10], [20, 20, 30, 30]], labels=[1, 1])]
    detections = [make_detections(boxes=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
                                  scores=[0.95, 0.85, 0.80], labels=[1, 1, 1]),
                  make_detections(boxes=[[0, 0, 10, 4], [20, 20, 30, 30]], scores=[0.90, 0.75], labels=[1, 1])]
    return detections, ground_truths


def test_worked_example_true_positives_and_curve() -> None:
    """
    Verifies on the worked example that the detections are marked true positive, false positive, then three true
    positives in score order, and that the third point of the curve has the precision 2 / 3 and recall 2 / 5 of a
    score threshold placed at the third detection.
    """
    detections, ground_truths = get_worked_example()
    gathered = gather_class_detections(detections=detections, ground_truths=ground_truths, class_label=1)
    true_positive_flags = match_detections_to_ground_truth(
        detection_image_indices=gathered["image_indices"],
        best_intersection_over_union=gathered["best_intersection_over_union"],
        best_ground_truth_indices=gathered["best_ground_truth_indices"], intersection_over_union_threshold=0.5)
    assert true_positive_flags.tolist() == [True, False, True, True, True]

    precisions, recalls = compute_precision_recall(true_positive_flags=true_positive_flags,
                                                   number_ground_truth=gathered["number_ground_truth"])
    assert gathered["number_ground_truth"] == 5
    assert precisions.tolist() == pytest.approx(expected=[1.0, 1 / 2, 2 / 3, 3 / 4, 4 / 5])
    assert recalls.tolist() == pytest.approx(expected=[0.2, 0.2, 0.4, 0.6, 0.8])


@pytest.mark.parametrize(argnames=("method", "expected_average_precision"),
                         argvalues=[("area", 0.68), ("11_point", 7.8 / 11), ("101_point", 69 / 101)])
def test_worked_example_average_precision(method: str, expected_average_precision: float) -> None:
    """
    Verifies the average precision of the worked example with every method, computed by hand from its precision
    envelope (1 up to a recall of 0.2, then 0.8 up to a recall of 0.8, then 0): the area 0.2 x 1 + 0.6 x 0.8, the
    11-point mean (3 x 1 + 6 x 0.8) / 11 and the 101-point mean (21 x 1 + 60 x 0.8) / 101.

    Args:
        method (str): The average precision method.
        expected_average_precision (float): The average precision computed by hand.
    """
    detections, ground_truths = get_worked_example()
    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=2)
    assert results[method]["mAP@0.5"] == pytest.approx(expected=expected_average_precision)
    assert results[method]["class_AP@0.5"][1] == pytest.approx(expected=expected_average_precision)


def test_perfect_detections_give_one() -> None:
    """
    Verifies that detections identical to the ground truth give an average precision of 1 with every method and at
    every intersection over union threshold.
    """
    ground_truths = [make_ground_truth(boxes=[[0, 0, 10, 10], [20, 20, 40, 30]], labels=[1, 2])]
    detections = [make_detections(boxes=[[0, 0, 10, 10], [20, 20, 40, 30]], scores=[0.9, 0.8], labels=[1, 2])]

    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=3)
    for method_results in results.values():
        assert method_results["mAP@0.5"] == pytest.approx(expected=1.0)
        assert method_results["mAP@0.75"] == pytest.approx(expected=1.0)
        assert method_results["mAP@[0.5:0.95]"] == pytest.approx(expected=1.0)


def test_duplicate_detection_is_a_false_positive() -> None:
    """
    Verifies that a second detection of an already matched object is a false positive, which leaves the average
    precision at 1 since it comes after the full recall is reached.
    """
    ground_truths = [make_ground_truth(boxes=[[0, 0, 10, 10]], labels=[1])]
    detections = [make_detections(boxes=[[0, 0, 10, 10], [0, 0, 10, 10]], scores=[0.9, 0.8], labels=[1, 1])]

    gathered = gather_class_detections(detections=detections, ground_truths=ground_truths, class_label=1)
    true_positive_flags = match_detections_to_ground_truth(
        detection_image_indices=gathered["image_indices"],
        best_intersection_over_union=gathered["best_intersection_over_union"],
        best_ground_truth_indices=gathered["best_ground_truth_indices"], intersection_over_union_threshold=0.5)
    assert true_positive_flags.tolist() == [True, False]

    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=2)
    assert results["area"]["mAP@0.5"] == pytest.approx(expected=1.0)


def test_no_detection_gives_zero() -> None:
    """
    Verifies that a class with ground truth but no detection has an average precision of 0 and still counts in the
    mean.
    """
    ground_truths = [make_ground_truth(boxes=[[0, 0, 10, 10], [20, 20, 30, 30]], labels=[1, 2])]
    detections = [make_detections(boxes=[[20, 20, 30, 30]], scores=[0.9], labels=[2])]

    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=3)
    for method_results in results.values():
        assert method_results["class_AP@0.5"] == pytest.approx(expected={1: 0.0, 2: 1.0})
        assert method_results["mAP@0.5"] == pytest.approx(expected=0.5)


def test_class_without_ground_truth_is_left_out() -> None:
    """
    Verifies that a class with detections but no ground truth anywhere is left out of the mean, instead of counting
    as an average precision of 0.
    """
    ground_truths = [make_ground_truth(boxes=[[0, 0, 10, 10]], labels=[1])]
    detections = [make_detections(boxes=[[0, 0, 10, 10], [50, 50, 60, 60]], scores=[0.9, 0.95], labels=[1, 2])]

    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=3)
    assert list(results["area"]["class_AP@0.5"]) == [1]
    assert results["area"]["mAP@0.5"] == pytest.approx(expected=1.0)


@pytest.mark.parametrize(argnames=("intersection_over_union_threshold", "expected_flags"),
                         argvalues=[(0.5, [False]), (0.3, [True])])
def test_overlap_below_threshold_is_a_false_positive(intersection_over_union_threshold: float,
                                                     expected_flags: List[bool]) -> None:
    """
    Verifies that a detection overlapping its object by 0.4 is a false positive at a threshold of 0.5 and a true
    positive at 0.3.

    Args:
        intersection_over_union_threshold (float): The matching threshold.
        expected_flags (List[bool]): The expected true positive flag of the detection.
    """
    ground_truths = [make_ground_truth(boxes=[[0, 0, 10, 10]], labels=[1])]
    detections = [make_detections(boxes=[[0, 0, 10, 4]], scores=[0.9], labels=[1])]

    gathered = gather_class_detections(detections=detections, ground_truths=ground_truths, class_label=1)
    true_positive_flags = match_detections_to_ground_truth(
        detection_image_indices=gathered["image_indices"],
        best_intersection_over_union=gathered["best_intersection_over_union"],
        best_ground_truth_indices=gathered["best_ground_truth_indices"],
        intersection_over_union_threshold=intersection_over_union_threshold)
    assert true_positive_flags.tolist() == expected_flags


def test_wrong_class_never_matches() -> None:
    """
    Verifies that a detection placed exactly on an object but with another class does not find it: the object class
    gets an average precision of 0, and the predicted class, without ground truth, is left out.
    """
    ground_truths = [make_ground_truth(boxes=[[0, 0, 10, 10]], labels=[1])]
    detections = [make_detections(boxes=[[0, 0, 10, 10]], scores=[0.9], labels=[2])]

    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=3)
    assert results["area"]["class_AP@0.5"] == {1: 0.0}
    assert results["area"]["mAP@0.5"] == 0.0


def test_coco_style_mean_over_thresholds() -> None:
    """
    Verifies that mAP@[0.5:0.95] is the mean over the ten thresholds: a detection overlapping its object by 0.82 is
    found at the 7 thresholds from 0.50 to 0.80 and missed at 0.85, 0.90 and 0.95, giving 7 / 10.
    """
    ground_truths = [make_ground_truth(boxes=[[0, 0, 10, 10]], labels=[1])]
    detections = [make_detections(boxes=[[0, 0, 10, 8.2]], scores=[0.9], labels=[1])]

    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=2)
    for method_results in results.values():
        assert method_results["mAP@0.75"] == pytest.approx(expected=1.0)
        assert method_results["mAP@[0.5:0.95]"] == pytest.approx(expected=0.7)


def test_unknown_method_is_rejected() -> None:
    """
    Verifies that an unknown average precision method raises an error instead of silently returning a value.
    """
    with pytest.raises(expected_exception=ValueError):
        compute_average_precision(precisions=torch.tensor(data=[1.0]), recalls=torch.tensor(data=[1.0]),
                                  method="trapezoid")
