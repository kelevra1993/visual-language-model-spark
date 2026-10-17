"""
Tests of the average precision of the keypoint detection, the object keypoint similarity taking the place of the
intersection over union: the objects grouped by group_image_objects (evaluation/keypoints/object_keypoint_similarity.py)
are evaluated by evaluate_detections with the 'keypoints' overlap (evaluation/detection/mean_average_precision.py) and
reported by summarise_average_precisions (evaluation/detection/average_precision_report.py), as the keypoint trainer
does on the test split: perfect keypoints, keypoints whose similarity passes 0.5 but not 0.75, the small, medium and
large objects sized by their box areas, and the mean average recall.
"""

import pytest
import torch

from typing import Dict, List, Tuple

from vision_training.evaluation.detection.average_precision_report import summarise_average_precisions
from vision_training.evaluation.detection.mean_average_precision import (AVERAGE_PRECISION_METHODS,
                                                                         INTERSECTION_OVER_UNION_THRESHOLDS,
                                                                         evaluate_detections)
from vision_training.evaluation.keypoints.object_keypoint_similarity import (build_keypoint_sigma_table,
                                                                             group_image_objects)

# Foreground classes of the keypoint task, both with the four corners of a square as keypoints
KEYPOINT_CATEGORIES = {1: {"name": "square", "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"]},
                       2: {"name": "diamond", "keypoints": ["top", "right", "bottom", "left"]}}

# Name of every class index in the report, background included
CLASS_NAMES = {0: "background", 1: "square", 2: "diamond"}

# Number of classes of the average precision, background included
NUMBER_CLASSES = 3

# Displacement of every keypoint of a 100 x 100 object giving a similarity of exp(-100 / (2 · 10000 · 0.1²)) =
# exp(-0.5) ≈ 0.61 with the default sigma of 0.05, matched at the thresholds 0.5 to 0.6 but not from 0.65 on
PARTIAL_SHIFT_PIXELS = 10.0


def make_square_objects(object_definitions: List[Tuple[int, int, float, float, float, float]]) -> Dict[
        str, torch.Tensor]:
    """
    Builds the evaluated objects of the test split, as gathered by the keypoint trainer, from squares whose four
    corners are the keypoints and whose predicted keypoints are all moved along x by a known shift.

    Args:
        object_definitions (List[Tuple[int, int, float, float, float, float]]): The image identifier, class index,
            left coordinate, side, keypoint shift in pixels and keypoint score of every object.

    Returns:
        Dict[str, torch.Tensor]: The 'image_identifiers', 'labels', 'bounding_boxes', 'keypoints', 'visibilities',
            'predicted_keypoints' and 'scores' of the objects.
    """
    bounding_boxes, keypoints, keypoint_shifts = [], [], []
    for _, _, left, side, shift, _ in object_definitions:
        bounding_boxes.append([left, 0.0, left + side, side])
        keypoints.append([[left, 0.0], [left + side, 0.0], [left + side, side], [left, side]])
        keypoint_shifts.append([[shift, 0.0]] * 4)
    keypoints_tensor = torch.tensor(data=keypoints, dtype=torch.float32)
    return {"image_identifiers": torch.tensor(data=[definition[0] for definition in object_definitions]),
            "labels": torch.tensor(data=[definition[1] for definition in object_definitions]),
            "bounding_boxes": torch.tensor(data=bounding_boxes, dtype=torch.float32), "keypoints": keypoints_tensor,
            "visibilities": torch.full(size=(len(object_definitions), 4), fill_value=2),
            "predicted_keypoints": keypoints_tensor + torch.tensor(data=keypoint_shifts, dtype=torch.float32),
            "scores": torch.tensor(data=[definition[5] for definition in object_definitions], dtype=torch.float32)}


def evaluate_keypoint_objects(objects: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
    """
    Evaluates the objects with the average precision on the object keypoint similarity, the default sigma of 0.05
    applying to every keypoint, as the keypoint trainer does at the end of its test evaluation.

    Args:
        objects (Dict[str, torch.Tensor]): The evaluated objects of make_square_objects.

    Returns:
        Dict[str, Dict]: The results of evaluate_detections for every average precision method.
    """
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=0.05,
                                             keypoint_sigmas={}, maximum_keypoints=4)
    detections, ground_truths = group_image_objects(objects=objects, sigma_table=sigma_table)
    return evaluate_detections(detections=detections, ground_truths=ground_truths, number_classes=NUMBER_CLASSES,
                               overlap_type="keypoints")


def test_perfect_keypoints_give_an_average_precision_of_one_at_every_threshold() -> None:
    """
    Verifies that keypoints predicted exactly on their ground truth, for two classes over two images, give an average
    precision and a recall of 1 with every method at every threshold, the objects of different classes or far apart
    never taking each other's ground truth.
    """
    objects = make_square_objects(object_definitions=[(0, 1, 0.0, 100.0, 0.0, 0.9), (0, 2, 300.0, 50.0, 0.0, 0.4),
                                                      (1, 1, 0.0, 20.0, 0.0, 0.7), (1, 1, 300.0, 100.0, 0.0, 0.2)])
    results = evaluate_keypoint_objects(objects=objects)
    for method in AVERAGE_PRECISION_METHODS:
        for metric_name in ["mAP@0.5", "mAP@0.75", "mAP@[0.5:0.95]", "mAR@0.5", "mAR@0.75", "mAR@[0.5:0.95]"]:
            assert results[method][metric_name] == pytest.approx(1.0), f"{method} {metric_name}"
    assert results["area"]["class_ground_truths"] == {1: 3, 2: 1}


def test_partially_correct_keypoints_lower_the_strict_thresholds_first() -> None:
    """
    Verifies that an object whose keypoints reach a similarity of about 0.61 counts as found up to the threshold 0.6
    and missed from 0.65 on: with a perfect object of higher score in the same image, the AP and the recall stay 1 at
    0.5 and fall to 0.5 at 0.75, and their mean over the ten thresholds is (3 · 1 + 7 · 0.5) / 10.
    """
    objects = make_square_objects(object_definitions=[(0, 1, 0.0, 100.0, 0.0, 0.9),
                                                      (0, 1, 300.0, 100.0, PARTIAL_SHIFT_PIXELS, 0.8)])
    results = evaluate_keypoint_objects(objects=objects)
    area_results = results["area"]
    assert area_results["mAP@0.5"] == pytest.approx(1.0) and area_results["mAP@0.75"] == pytest.approx(0.5)
    assert area_results["mAP@[0.5:0.95]"] == pytest.approx(0.65)
    assert area_results["class_AP@0.5"] == {1: pytest.approx(1.0)}
    assert area_results["mAR@0.5"] == pytest.approx(1.0) and area_results["mAR@0.75"] == pytest.approx(0.5)
    assert area_results["mAR@[0.5:0.95]"] == pytest.approx(0.65)
    assert INTERSECTION_OVER_UNION_THRESHOLDS[2:4] == [0.6, 0.65]

    # The same keypoints ranked first make a false positive come before the true positive at 0.75
    swapped_objects = make_square_objects(object_definitions=[(0, 1, 0.0, 100.0, 0.0, 0.8),
                                                              (0, 1, 300.0, 100.0, PARTIAL_SHIFT_PIXELS, 0.9)])
    swapped_results = evaluate_keypoint_objects(objects=swapped_objects)
    assert swapped_results["area"]["mAP@0.5"] == pytest.approx(1.0)
    assert swapped_results["area"]["mAP@0.75"] < area_results["mAP@0.75"]


def test_object_sizes_are_measured_by_the_box_areas() -> None:
    """
    Verifies that the objects are sorted into the COCO sizes by their box areas: a perfect 20 x 20 object is small with
    an AP of 1, a partially correct 100 x 100 object is large with an AP of 1 at 0.5 and 0 at 0.75, and no object being
    medium gives no medium mAP nor mAR.
    """
    objects = make_square_objects(object_definitions=[(0, 1, 0.0, 20.0, 0.0, 0.9),
                                                      (0, 1, 300.0, 100.0, PARTIAL_SHIFT_PIXELS, 0.8)])
    size_results = evaluate_keypoint_objects(objects=objects)["area"]["sizes"]
    assert size_results["small"]["mAP@0.5"] == pytest.approx(1.0)
    assert size_results["small"]["mAP@0.75"] == pytest.approx(1.0)
    assert size_results["large"]["mAP@0.5"] == pytest.approx(1.0)
    assert size_results["large"]["mAP@0.75"] == pytest.approx(0.0)
    assert size_results["large"]["mAR@0.75"] == pytest.approx(0.0)
    assert size_results["medium"]["mAP@0.5"] is None and size_results["medium"]["mAR@0.5"] is None


def test_keypoint_report_prefixes_its_tables_scalars_and_columns() -> None:
    """
    Verifies the report of the keypoint average precisions as called by the keypoint trainer: the 'OKS' table titles,
    the 'OKS ' scalars of every method, size and recall (none for the size without object), and the 'oks_' columns of
    the class rows.
    """
    objects = make_square_objects(object_definitions=[(0, 1, 0.0, 20.0, 0.0, 0.9),
                                                      (0, 2, 300.0, 100.0, PARTIAL_SHIFT_PIXELS, 0.8)])
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=0.05,
                                             keypoint_sigmas={}, maximum_keypoints=4)
    detections, ground_truths = group_image_objects(objects=objects, sigma_table=sigma_table)
    summary = summarise_average_precisions(detections=detections, ground_truths=ground_truths,
                                           number_classes=NUMBER_CLASSES, class_names=CLASS_NAMES,
                                           overlap_type="keypoints", title_prefix="OKS", name_prefix="OKS ",
                                           column_prefix="oks_")

    # Scalars of the mean average precisions, of the sizes with objects and of the recalls
    scalars = summary["scalars"]
    assert scalars["OKS mAP@0.5/area"] == pytest.approx(1.0) and scalars["OKS mAP@0.75/area"] == pytest.approx(0.5)
    assert scalars["OKS mAP@0.5 By Size/small"] == pytest.approx(1.0)
    assert "OKS mAP@0.5 By Size/medium" not in scalars
    assert scalars["OKS mAR@0.75/all"] == pytest.approx(0.5) and scalars["OKS mAR@0.75/large"] == pytest.approx(0.0)
    assert summary["class_scalars"] == {"OKS AP@0.5/square": pytest.approx(1.0),
                                        "OKS AP@0.5/diamond": pytest.approx(1.0)}

    # Class rows with the prefixed columns, the diamond missed at 0.75
    class_rows = {row["class"]: row for row in summary["class_rows"]}
    assert class_rows["square"]["oks_average_precision_75"] == pytest.approx(1.0)
    assert class_rows["diamond"]["oks_average_precision_75"] == pytest.approx(0.0)
    assert all(table["title"].startswith("OKS") for table in summary["tables"].values())
    assert summary["class_score_thresholds"] is None
