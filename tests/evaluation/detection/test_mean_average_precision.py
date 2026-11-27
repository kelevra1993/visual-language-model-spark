"""
Tests of vision_training/evaluation/detection/mean_average_precision.py on a worked example and edge cases: the
average precision of the worked example, the per-class average precisions and ground truths, wrong-class detections,
the mean over the COCO overlap thresholds, the score threshold of every class, the mask average precision of the
instance segmentation, the average precision of the five object sizes relative to the image (their bounds included
below, an object changing size with the image), the mean average recall of all the objects and of every size, and the
vectorised 101-point average precision against its level by level definition.
"""

import pytest
import torch

from typing import Any, Dict, List

from vision_training.evaluation.detection.mean_average_precision import (compute_average_precision,
                                                                         compute_precision_recall, evaluate_detections,
                                                                         gather_class_detections)
from vision_training.evaluation.detection.object_sizes import compute_box_areas
from vision_training.models.detection.box_operations import get_intersection_over_union
from vision_training.models.instance_segmentation.mask_operations import compute_mask_intersection_over_union

# Side of the evaluated images in pixels, the 10 x 10 boxes of the examples having a side under 1/16 of it
IMAGE_SIDE = 800.0


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


def make_ground_truth(boxes: List[List[float]], labels: List[int], image_side: float = IMAGE_SIDE) -> Dict[str, Any]:
    """
    Builds the ground truth of one image in the format collected by the Trainer test evaluation.

    Args:
        boxes (List[List[float]]): The [x1, y1, x2, y2] ground truth boxes.
        labels (List[int]): The ground truth class indices.
        image_side (float): The side of the square image, which sizes its objects. Defaults to IMAGE_SIDE.

    Returns:
        Dict[str, Any]: The 'boxes' [M, 4], 'labels' [M] and 'image_area' of the image.
    """
    return {"boxes": torch.tensor(data=boxes, dtype=torch.float32).reshape(-1, 4), "labels": torch.tensor(data=labels),
            "image_area": image_side * image_side}


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


@pytest.mark.parametrize(argnames=("method", "expected_average_precision"), argvalues=[("area", 0.68)])
def test_worked_example_average_precision(method: str, expected_average_precision: float) -> None:
    """
    Verifies the average precision of the worked example with the area method, computed by hand from its precision
    envelope (1 up to a recall of 0.2, then 0.8 up to a recall of 0.8, then 0): the area 0.2 x 1 + 0.6 x 0.8.

    Args:
        method (str): The average precision method.
        expected_average_precision (float): The average precision computed by hand.
    """
    detections, ground_truths = get_worked_example()
    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=2)
    assert results[method]["mAP@0.5"] == pytest.approx(expected=expected_average_precision)
    assert results[method]["class_AP@0.5"][1] == pytest.approx(expected=expected_average_precision)


@pytest.mark.parametrize(argnames=("target_precision", "expected_threshold"), argvalues=[(0.99, (0.95, 0.2))])
def test_class_score_thresholds_of_the_worked_example(target_precision: float, expected_threshold: tuple) -> None:
    """
    Verifies the score threshold of the worked example, whose detections are by decreasing score a true positive, a
    false positive and three true positives among 5 objects: only its first detection reaches a precision of 0.99
    (recall 1 / 5). The thresholds are only given with a target precision.

    Args:
        target_precision (float): The precision the threshold must reach.
        expected_threshold (tuple): The expected threshold and recall.
    """
    detections, ground_truths = get_worked_example()
    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=2,
                                  target_precision=target_precision)
    assert results["class_score_thresholds"][1] == pytest.approx(expected=expected_threshold)
    assert "class_score_thresholds" not in evaluate_detections(detections=detections, ground_truths=ground_truths,
                                                               number_classes=2)


def test_per_class_average_precisions_and_ground_truths() -> None:
    """
    Verifies the per-class results of the report: a box found exactly (AP 1 at every threshold) and a box found with
    an intersection over union of 0.72 (AP 1 up to the threshold 0.7, 0 from 0.75, so 0.5 over the ten thresholds),
    the mean of the per-class averages over 0.5 to 0.95 being the mAP@[0.5:0.95], and the ground truth boxes of every
    foreground class, a class without any being counted as 0.
    """
    ground_truths = [make_ground_truth(boxes=[[0, 0, 10, 10], [20, 20, 30, 30]], labels=[1, 2])]
    detections = [make_detections(boxes=[[0, 0, 10, 10], [20, 20, 30, 27.2]], scores=[0.9, 0.8], labels=[1, 2])]

    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=4)
    for method_results in results.values():
        assert method_results["class_AP@0.75"] == pytest.approx(expected={1: 1.0, 2: 0.0})
        assert method_results["class_AP@[0.5:0.95]"] == pytest.approx(expected={1: 1.0, 2: 0.5})
        assert sum(method_results["class_AP@[0.5:0.95]"].values()) / 2 == pytest.approx(
            expected=method_results["mAP@[0.5:0.95]"])
        assert method_results["class_ground_truths"] == {1: 1, 2: 1, 3: 0}


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


def interpolate_average_precision_by_definition(precisions: torch.Tensor, recalls: torch.Tensor,
                                                number_points: int) -> float:
    """
    Reference of the 11 and 101-point average precision written as its definition, one recall level at a time: the
    maximum precision at a recall greater than or equal to the level, 0 beyond the maximum recall. The vectorised
    compute_average_precision of the evaluation must return exactly the same value.

    Args:
        precisions (torch.Tensor): The [D] precisions, from compute_precision_recall.
        recalls (torch.Tensor): The [D] non-decreasing recalls, from compute_precision_recall.
        number_points (int): The number of equally spaced recall levels, 11 or 101.

    Returns:
        float: The average precision.
    """
    recall_levels = torch.arange(end=number_points, dtype=torch.float64) / (number_points - 1)
    interpolated_precisions = [float(precisions[recalls >= recall_level].max()) if bool(
        (recalls >= recall_level).any()) else 0.0 for recall_level in recall_levels]
    return sum(interpolated_precisions) / number_points


@pytest.mark.parametrize(argnames=("method", "number_points"), argvalues=[("101_point", 101)])
def test_point_average_precision_matches_its_definition(method: str, number_points: int) -> None:
    """
    Verifies that the vectorised 101-point average precision equals, exactly, the level by level definition on
    random curves: a single detection, no true positive, recalls reaching 1 or stopping below it, and ground truth
    counts such as 5 or 20 whose recalls fall exactly on recall levels.

    Args:
        method (str): The average precision method.
        number_points (int): Its number of recall levels.
    """
    generator = torch.Generator().manual_seed(0)
    for number_detections in [1, 2, 7, 50, 300]:
        for true_positive_rate in [0.0, 0.3, 0.8, 1.0]:
            for number_missed in [0, 1, 4]:
                # Random true positive flags, the ground truths being the true positives plus the missed objects
                true_positive_flags = torch.rand(size=(number_detections,), generator=generator) < true_positive_rate
                number_ground_truth = max(int(true_positive_flags.sum()) + number_missed, 1)
                precisions, recalls = compute_precision_recall(true_positive_flags=true_positive_flags,
                                                               number_ground_truth=number_ground_truth)

                # The vectorised envelope lookup and the definition must agree to the last bit
                assert compute_average_precision(precisions=precisions, recalls=recalls, method=method) == \
                    interpolate_average_precision_by_definition(precisions=precisions, recalls=recalls,
                                                                number_points=number_points)


def test_average_precision_by_object_size() -> None:
    """
    Verifies the breakdown by object size on one class in an 800 x 800 image with objects of sides 10, 50, 100 and 400
    pixels (1/80, exactly 1/16, 1/8 and 1/2 of the image side): the 10 x 10 object is found after an unmatched
    detection of its size (AP 0.5 with the area method), the 50 x 50 one falls in the size starting at 1/16 and is
    never found (AP 0), the 100 x 100 and 400 x 400 ones are found before an unmatched detection of their size or alone
    (AP 1), and the size from 1/4 to 1/2 without object gets no mAP. The same objects in a 1600 x 1600 image halve
    their side shares and move down one size, the 10 x 10 and 50 x 50 objects sharing the size under 1/16.
    """
    ground_truth_boxes = [[0, 0, 10, 10], [100, 100, 150, 150], [200, 200, 300, 300], [0, 400, 400, 800]]
    ground_truths = [make_ground_truth(boxes=ground_truth_boxes, labels=[1, 1, 1, 1])]
    detections = [make_detections(boxes=[[400, 400, 410, 410], [0, 0, 10, 10], [200, 200, 300, 300],
                                         [500, 500, 600, 600], [0, 400, 400, 800]],
                                  scores=[0.95, 0.9, 0.8, 0.7, 0.6], labels=[1, 1, 1, 1, 1])]

    sizes = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=2)["area"]["sizes"]
    assert list(sizes) == ["under_1_16", "1_16_to_1_8", "1_8_to_1_4", "1_4_to_1_2", "1_2_and_above"]
    assert sizes["under_1_16"]["mAP@0.5"] == pytest.approx(expected=0.5)
    assert sizes["1_16_to_1_8"]["mAP@0.5"] == pytest.approx(expected=0.0)
    assert sizes["1_8_to_1_4"]["mAP@[0.5:0.95]"] == pytest.approx(expected=1.0)
    assert sizes["1_2_and_above"]["mAP@[0.5:0.95]"] == pytest.approx(expected=1.0)
    assert [size_results["class_ground_truths"] for size_results in sizes.values()] == [{1: 1}, {1: 1}, {1: 1},
                                                                                         {1: 0}, {1: 1}]

    # A size without any object gives no mAP and no class average precision
    assert sizes["1_4_to_1_2"]["mAP@0.5"] is None and sizes["1_4_to_1_2"]["mAP@0.75"] is None
    assert sizes["1_4_to_1_2"]["class_AP@0.5"] == {}

    # The same objects in an image twice as large move down one size
    larger_image_sizes = evaluate_detections(detections=detections, ground_truths=[make_ground_truth(
        boxes=ground_truth_boxes, labels=[1, 1, 1, 1], image_side=2.0 * IMAGE_SIDE)], number_classes=2)["area"]["sizes"]
    assert [size_results["class_ground_truths"] for size_results in larger_image_sizes.values()] == [
        {1: 2}, {1: 1}, {1: 0}, {1: 1}, {1: 0}]


def test_mean_average_recall_of_all_the_objects_and_of_every_size() -> None:
    """
    Verifies the mean average recall on two classes: class 1 is the worked example (4 of its 5 objects found exactly,
    whatever the score and the false positive, recall 0.8 at every threshold) and the only class 2 object is found
    with an overlap of 0.68 (recall 1 at the thresholds 0.5 to 0.65, 0 above). The recall is the same for every method,
    class 3 without ground truth is left out, and the 10 x 10 objects all have a side under 1/16 of the image side,
    the other sizes having no recall.
    """
    detections, ground_truths = get_worked_example()
    ground_truths[0] = make_ground_truth(boxes=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 70]],
                                         labels=[1, 1, 1, 2])
    detections[0] = make_detections(boxes=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50], [60, 60, 70, 66.8]],
                                    scores=[0.95, 0.85, 0.80, 0.5], labels=[1, 1, 1, 2])
    results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=4)

    # Means over the classes with ground truth, at 0.5, 0.75, 0.95 and over the ten thresholds, the same for all methods
    for method in ["area", "11_point", "101_point"]:
        assert results[method]["mAR@0.5"] == pytest.approx(expected=0.9)
        assert results[method]["mAR@0.75"] == pytest.approx(expected=0.4)
        assert results[method]["mAR@0.95"] == pytest.approx(expected=0.4)
        assert results[method]["mAR@[0.5:0.95]"] == pytest.approx(expected=0.6)
    assert results["area"]["class_AR@0.5"] == pytest.approx(expected={1: 0.8, 2: 1.0})
    assert results["area"]["class_AR@[0.5:0.95]"] == pytest.approx(expected={1: 0.8, 2: 0.4})

    # Every object is under 1/16 of the image side, the larger sizes have no recall
    sizes = results["area"]["sizes"]
    assert sizes["under_1_16"]["mAR@[0.5:0.95]"] == pytest.approx(expected=0.6)
    assert sizes["1_16_to_1_8"]["mAR@0.5"] is None and sizes["1_2_and_above"]["mAR@[0.5:0.95]"] is None
    assert sizes["1_2_and_above"]["class_AR@0.5"] == {}


def fill_box_masks(boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Builds the binary masks filling integer boxes exactly, whose mask intersection over union equals the box one.

    Args:
        boxes (torch.Tensor): The [N, 4] integer [x1, y1, x2, y2] boxes.
        image_size (int): The side of the square image.

    Returns:
        torch.Tensor: The [N, image_size, image_size] boolean masks.
    """
    masks = torch.zeros(size=(boxes.shape[0], image_size, image_size), dtype=torch.bool)
    for object_index, (x_1, y_1, x_2, y_2) in enumerate(boxes.int().tolist()):
        masks[object_index, y_1:y_2, x_1:x_2] = True
    return masks


def test_mask_average_precision_equals_box_average_precision_for_box_masks() -> None:
    """
    Verifies that when every detection and ground truth mask fills its box exactly, the mask overlaps equal the box
    overlaps exactly (both exact ratios of pixel counts, so that an exact ratio such as 3 / 4 reaches its threshold
    with both), that the mask average precision read from overlaps equal to the box ones gives the box average
    precision of every method, threshold and class (the rows and columns of every class being sliced from the
    overlaps of the whole image), and that an unknown overlap type is refused.
    """
    generator = torch.Generator().manual_seed(0)
    detections, ground_truths = [], []
    for _ in range(4):
        ground_truth_corners = torch.randint(low=0, high=40, size=(5, 2), generator=generator)
        ground_truth_sizes = torch.randint(low=4, high=20, size=(5, 2), generator=generator)
        ground_truth_boxes = torch.cat(tensors=[ground_truth_corners, ground_truth_corners + ground_truth_sizes], dim=1)
        detection_boxes = (ground_truth_boxes.repeat(2, 1) + torch.randint(low=-1, high=2, size=(10, 4),
                                                                           generator=generator)).clamp(min=0, max=64)
        ground_truth = {"boxes": ground_truth_boxes.float(), "image_area": 64.0 * 64.0,
                        "labels": torch.randint(low=1, high=3, size=(5,), generator=generator)}
        detection = {"boxes": detection_boxes.float(), "scores": torch.rand(size=(10,), generator=generator),
                     "labels": torch.randint(low=1, high=3, size=(10,), generator=generator)}
        mask_overlaps = compute_mask_intersection_over_union(
            predicted_masks=fill_box_masks(boxes=detection_boxes, image_size=64),
            ground_truth_masks=fill_box_masks(boxes=ground_truth_boxes, image_size=64), chunk_size=3)
        box_overlaps = get_intersection_over_union(boxes_1=detection_boxes.double(),
                                                   boxes_2=ground_truth_boxes.double())
        assert torch.equal(input=mask_overlaps, other=box_overlaps)
        detection["mask_overlaps"] = box_overlaps
        # Masks filling their boxes have the areas of their boxes, so the sizes of both evaluations match too
        detection["mask_areas"] = compute_box_areas(boxes=detection["boxes"])
        ground_truth["mask_areas"] = compute_box_areas(boxes=ground_truth["boxes"])
        detections.append(detection)
        ground_truths.append(ground_truth)

    box_results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=3)
    mask_results = evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=3,
                                       overlap_type="masks")
    for method, method_results in box_results.items():
        for metric_name in ["mAP@0.5", "mAP@0.75", "mAP@0.95", "mAP@[0.5:0.95]"]:
            assert mask_results[method][metric_name] == pytest.approx(expected=method_results[metric_name])
        assert mask_results[method]["class_AP@0.5"] == pytest.approx(expected=method_results["class_AP@0.5"])
        for object_size, size_results in method_results["sizes"].items():
            assert mask_results[method]["sizes"][object_size]["class_AP@[0.5:0.95]"] == pytest.approx(
                expected=size_results["class_AP@[0.5:0.95]"])
    assert 0.0 < box_results["area"]["mAP@0.5"] < 1.0

    with pytest.raises(ValueError, match="Unknown overlap type"):
        gather_class_detections(detections=detections, ground_truths=ground_truths, class_label=1,
                                overlap_type="polygons")
