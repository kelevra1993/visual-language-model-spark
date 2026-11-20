"""
Tests of vision_training/evaluation/keypoints/object_keypoint_similarity.py, the keypoint evaluation of the keypoint
trainer on the test split: the sigma table built from the configured COCO sigmas (default, padding positions, merged
categories), the object keypoint similarity on a hand-computed case with the box-area scale, the one-to-one matching
of the ground truth objects of an image to the predicted ones by the largest total similarity over the pairs reaching
the minimum similarity (a far prediction leaving its object missed), the comparison of the predicted and ground truth
objects of every image into the detections and ground truths of the average precision and the matched objects of the
keypoint accuracy, the similarity of the predicted objects with the objects without labelled keypoint (the box
extended by its size on every side, as pycocotools), and the matched fraction, PCK and pixel errors of every class and
keypoint, with the missed objects and the classes and keypoints that have no labelled keypoint, the rates written as
percentages in the report.
"""

import math
import pytest
import torch

from typing import Any, Dict, List, Optional

from vision_training.evaluation.keypoints.object_keypoint_similarity import (CORRECT_KEYPOINTS_NAME,
                                                                             EVALUATED_OBJECT_KEYS,
                                                                             MINIMUM_MATCH_SIMILARITY,
                                                                             build_keypoint_sigma_table,
                                                                             compute_object_keypoint_similarity,
                                                                             compute_squared_extended_box_distances,
                                                                             evaluate_image_keypoints,
                                                                             match_ground_truth_objects,
                                                                             measure_keypoint_accuracy,
                                                                             summarise_keypoint_accuracy)
from vision_training.evaluation.report_formatting import build_table_renderables, render_report_text

# Foreground classes of the keypoint task as injected in 'Data.keypoint_categories': a triangle, a rectangle, and a
# class merging two categories that has no object in the evaluated objects below
KEYPOINT_CATEGORIES = {1: {"name": "triangle", "keypoints": ["top", "left", "right"]},
                       2: {"name": "rectangle", "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"]},
                       3: {"name": "square+diamond", "keypoints": ["centre", "corner"]}}

# Name of every class index in the report, background included
CLASS_NAMES = {0: "background", 1: "triangle", 2: "rectangle", 3: "square+diamond"}

# Keypoint positions of every object, the rectangle having the most keypoints
MAXIMUM_KEYPOINTS = 4

# Sigma of the keypoints without their own value, the default of the configuration
DEFAULT_SIGMA = 0.05


def make_objects(labels: List[int], bounding_boxes: List[List[float]], keypoints: List[List[List[float]]],
                 visibilities: List[List[int]], keypoint_errors: List[List[List[float]]],
                 matched: List[bool]) -> Dict[str, torch.Tensor]:
    """
    Builds the ground truth objects of the test split in the format of evaluate_image_keypoints before the keypoint
    accuracy, the keypoints of the predicted object matched to every object being its ground truth keypoints moved by a
    known error, so that every distance is known in advance, and NaN for the objects missed.

    Args:
        labels (List[int]): The class index of every object.
        bounding_boxes (List[List[float]]): The [x1, y1, x2, y2] box of every object.
        keypoints (List[List[List[float]]]): The [K, 2] ground truth keypoints of every object.
        visibilities (List[List[int]]): The [K] COCO visibilities of every object.
        keypoint_errors (List[List[List[float]]]): The [K, 2] displacement of the matched keypoints of every object.
        matched (List[bool]): Whether every object is matched to a predicted object.

    Returns:
        Dict[str, torch.Tensor]: The 'labels', 'bounding_boxes', 'keypoints', 'visibilities', 'predicted_keypoints'
            and 'matched' of the objects.
    """
    ground_truth_keypoints = torch.tensor(data=keypoints, dtype=torch.float32)
    predicted_keypoints = ground_truth_keypoints + torch.tensor(data=keypoint_errors, dtype=torch.float32)
    matched_tensor = torch.tensor(data=matched)
    predicted_keypoints[~matched_tensor] = float("nan")
    return {"labels": torch.tensor(data=labels),
            "bounding_boxes": torch.tensor(data=bounding_boxes, dtype=torch.float64),
            "keypoints": ground_truth_keypoints, "visibilities": torch.tensor(data=visibilities),
            "predicted_keypoints": predicted_keypoints, "matched": matched_tensor}


def make_accuracy_objects() -> Dict[str, torch.Tensor]:
    """
    Builds a triangle in a 100 x 50 box (correct within 5 pixels) and a rectangle in a 40 x 40 box (correct within 2
    pixels), whose matched keypoints are moved along x by known pixel errors, with unlabelled keypoints and padding
    positions whose large errors must be left out of every metric, and a second rectangle missed by the detector, whose
    two labelled keypoints count as wrong and are left out of the pixel errors.

    Returns:
        Dict[str, torch.Tensor]: The evaluated objects of make_objects.
    """
    triangle_keypoints = [[50.0, 5.0], [10.0, 45.0], [90.0, 45.0], [0.0, 0.0]]
    rectangle_keypoints = [[205.0, 5.0], [235.0, 5.0], [235.0, 35.0], [205.0, 35.0]]
    missed_keypoints = [[305.0, 5.0], [335.0, 5.0], [335.0, 35.0], [305.0, 35.0]]

    # The triangle has its 'right' keypoint unlabelled and a padding position, the rectangle its 'bottom_left'
    # unlabelled, the missed rectangle its two bottom keypoints unlabelled
    return make_objects(labels=[1, 2, 2],
                        bounding_boxes=[[0.0, 0.0, 100.0, 50.0], [200.0, 0.0, 240.0, 40.0], [300.0, 0.0, 340.0, 40.0]],
                        keypoints=[triangle_keypoints, rectangle_keypoints, missed_keypoints],
                        visibilities=[[2, 1, 0, 0], [2, 1, 2, 0], [2, 2, 0, 0]],
                        keypoint_errors=[[[3.0, 0.0], [6.0, 0.0], [50.0, 0.0], [70.0, 0.0]],
                                         [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [10.0, 0.0]], [[0.0, 0.0]] * 4],
                        matched=[True, True, False])


def make_image(labels: List[int], bounding_boxes: List[List[float]], keypoints: List[List[List[float]]],
               visibilities: List[List[int]], predicted_labels: List[int],
               predicted_keypoints: List[List[List[float]]], predicted_scores: List[float],
               ignored_bounding_boxes: Optional[List[List[float]]] = None,
               ignored_labels: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
    """
    Builds one test image in the format gathered by the keypoint trainer for evaluate_image_keypoints: its ground truth
    objects, its objects without labelled keypoint (none by default, none of them a crowd) and the objects grouped
    from the keypoint candidates of the detector, with 4 keypoint positions each.

    Args:
        labels (List[int]): The class index of every ground truth object.
        bounding_boxes (List[List[float]]): The [x1, y1, x2, y2] box of every ground truth object.
        keypoints (List[List[List[float]]]): The [K, 2] keypoints of every ground truth object.
        visibilities (List[List[int]]): The [K] COCO visibilities of every ground truth object.
        predicted_labels (List[int]): The class index of every predicted object.
        predicted_keypoints (List[List[List[float]]]): The [K, 2] keypoints of every predicted object, NaN where not
            found.
        predicted_scores (List[float]): The score of every predicted object.
        ignored_bounding_boxes (Optional[List[List[float]]]): The [x1, y1, x2, y2] box of every object without
            labelled keypoint, None for none. Defaults to None.
        ignored_labels (Optional[List[int]]): The class index of every object without labelled keypoint, None for
            none. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: The ground truth 'labels', 'bounding_boxes', 'keypoints' and 'visibilities', the
            'ignored_bounding_boxes', 'ignored_labels' and 'ignored_crowds', and the 'predicted_labels',
            'predicted_keypoints' and 'predicted_scores'.
    """
    # No object without labelled keypoint unless given
    ignored_bounding_boxes = [] if ignored_bounding_boxes is None else ignored_bounding_boxes
    ignored_labels = [] if ignored_labels is None else ignored_labels
    return {"labels": torch.tensor(data=labels, dtype=torch.int64),
            "bounding_boxes": torch.tensor(data=bounding_boxes, dtype=torch.float32).reshape(-1, 4),
            "keypoints": torch.tensor(data=keypoints, dtype=torch.float32).reshape(-1, MAXIMUM_KEYPOINTS, 2),
            "visibilities": torch.tensor(data=visibilities, dtype=torch.int64).reshape(-1, MAXIMUM_KEYPOINTS),
            "ignored_bounding_boxes": torch.tensor(data=ignored_bounding_boxes, dtype=torch.float32).reshape(-1, 4),
            "ignored_labels": torch.tensor(data=ignored_labels, dtype=torch.int64),
            "ignored_crowds": torch.zeros(size=(len(ignored_labels),), dtype=torch.bool),
            "predicted_labels": torch.tensor(data=predicted_labels, dtype=torch.int64),
            "predicted_keypoints": torch.tensor(data=predicted_keypoints, dtype=torch.float32).reshape(
                -1, MAXIMUM_KEYPOINTS, 2),
            "predicted_scores": torch.tensor(data=predicted_scores, dtype=torch.float32)}


def compute_expected_similarity(squared_distances: List[float], labelled: List[bool], area: float,
                                sigmas: List[float]) -> float:
    """
    Computes the COCO object keypoint similarity of one object from its definition, keypoint by keypoint, as the
    reference of the vectorised compute_object_keypoint_similarity.

    Args:
        squared_distances (List[float]): The squared distance of every keypoint, in squared pixels.
        labelled (List[bool]): Whether every keypoint is labelled.
        area (float): The box area of the object, in squared pixels.
        sigmas (List[float]): The sigma of every keypoint.

    Returns:
        float: The mean of exp(-d² / (2 · area · (2σ)²)) over the labelled keypoints, 0 without labelled keypoint.
    """
    keypoint_similarities = [math.exp(-squared_distance / (2.0 * area * (2.0 * sigma) ** 2))
                             for squared_distance, is_labelled, sigma in zip(squared_distances, labelled, sigmas)
                             if is_labelled]
    return sum(keypoint_similarities) / len(keypoint_similarities) if keypoint_similarities else 0.0


def find_row(rows: List[Dict[str, Any]], **expected_values: Any) -> Dict[str, Any]:
    """
    Finds the only row of a list of class or keypoint rows of the keypoint report that has the given values.

    Args:
        rows (List[Dict[str, Any]]): The rows of summarise_keypoint_accuracy.
        **expected_values (Any): The values identifying the row, such as its class index and keypoint name.

    Returns:
        Dict[str, Any]: The matching row.
    """
    matching_rows = [row for row in rows if all(row[key] == value for key, value in expected_values.items())]
    assert len(matching_rows) == 1, f"Expected one row with {expected_values}, found {len(matching_rows)}."
    return matching_rows[0]


def test_sigma_table_takes_the_configured_sigmas_and_the_default_elsewhere() -> None:
    """
    Verifies that the sigma table holds the configured sigma of a keypoint at its class and position, the default for
    the keypoints left out, for the background and the padding positions, and the value given under any of the names of
    the categories merged into one class.
    """
    keypoint_sigmas = {"triangle": {"top": 0.1}, "diamond": {"corner": 0.2}, "unused_category": {"top": 0.9}}
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas=keypoint_sigmas, maximum_keypoints=MAXIMUM_KEYPOINTS)

    # One row per class index, background included, and one column per keypoint position
    assert sigma_table.shape == (4, MAXIMUM_KEYPOINTS) and sigma_table.dtype == torch.float64
    expected_table = torch.full(size=(4, MAXIMUM_KEYPOINTS), fill_value=DEFAULT_SIGMA, dtype=torch.float64)
    expected_table[1, 0] = 0.1
    expected_table[3, 1] = 0.2
    assert torch.equal(sigma_table, expected_table)


def test_object_keypoint_similarity_matches_a_hand_computed_case() -> None:
    """
    Verifies the similarity of an object in a 10 x 10 box with a sigma of 0.05: a perfect keypoint gives 1, a keypoint
    2 pixels away gives exp(-4 / (2 · 100 · 0.1²)) = exp(-2), and the unlabelled keypoint, however far, is left out of
    the mean.
    """
    similarity = compute_object_keypoint_similarity(
        squared_distances=torch.tensor(data=[[0.0, 4.0, 1.0e6]]), visibilities=torch.tensor(data=[[2, 1, 0]]),
        areas=torch.tensor(data=[100.0]), sigmas=torch.full(size=(1, 3), fill_value=0.05))
    assert similarity.dtype == torch.float64 and similarity.shape == (1,)
    assert similarity.item() == pytest.approx((1.0 + math.exp(-2.0)) / 2.0)


def test_ground_truth_objects_are_matched_by_the_largest_total_similarity() -> None:
    """
    Verifies that the matching maximises the total similarity rather than taking the best pair first (prediction 0
    goes to ground truth 1 so that prediction 1 can take ground truth 0), that a pair of similarity 0 never matches even
    when it is the only one, and that images without prediction or without ground truth give no match.
    """
    total_matching = match_ground_truth_objects(keypoint_overlaps=torch.tensor(data=[[0.9, 0.8], [0.7, 0.0]],
                                                                               dtype=torch.float64))
    assert total_matching.dtype == torch.int64 and total_matching.tolist() == [1, 0]
    assert match_ground_truth_objects(keypoint_overlaps=torch.tensor(data=[[0.0, 0.5]])).tolist() == [-1, 0]
    assert match_ground_truth_objects(keypoint_overlaps=torch.tensor(data=[[0.0]])).tolist() == [-1]
    assert match_ground_truth_objects(keypoint_overlaps=torch.zeros(size=(0, 2))).tolist() == [-1, -1]
    assert match_ground_truth_objects(keypoint_overlaps=torch.zeros(size=(2, 0))).tolist() == []


def test_a_far_prediction_is_neither_matched_nor_measured() -> None:
    """
    Verifies the case found by the audit of 2026-10-20: a 20x20 rectangle whose only prediction lies 60 pixels away
    has a similarity above 0 (it only underflows about 4 box sides away) but far below the minimum, so the object is
    missed: Matched Objects 0, PCK 0, and no mean keypoint error instead of the 60 pixels of the far prediction.
    """
    rectangle = [[1.0, 1.0], [19.0, 1.0], [19.0, 19.0], [1.0, 19.0]]
    far_rectangle = [[keypoint_x + 60.0, keypoint_y] for keypoint_x, keypoint_y in rectangle]
    images = [make_image(labels=[2], bounding_boxes=[[0.0, 0.0, 20.0, 20.0]], keypoints=[rectangle],
                         visibilities=[[2, 2, 2, 2]], predicted_labels=[2], predicted_keypoints=[far_rectangle],
                         predicted_scores=[0.9])]
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas={}, maximum_keypoints=MAXIMUM_KEYPOINTS)
    evaluation = evaluate_image_keypoints(images=images, sigma_table=sigma_table)
    similarity = evaluation["detections"][0]["keypoint_overlaps"][0, 0].item()
    assert 0.0 < similarity < MINIMUM_MATCH_SIMILARITY
    assert evaluation["objects"]["matched"].tolist() == [False]

    summary = summarise_keypoint_accuracy(objects=evaluation["objects"], sigma_table=sigma_table,
                                          keypoint_categories=KEYPOINT_CATEGORIES, class_names=CLASS_NAMES)
    assert summary["metrics"]["Matched Objects"] == 0.0 and summary["metrics"][CORRECT_KEYPOINTS_NAME] == 0.0
    assert summary["metrics"]["Mean Keypoint Error (pixels)"] is None


def test_evaluate_image_keypoints_compares_the_predicted_and_ground_truth_objects_of_every_image() -> None:
    """
    Verifies the evaluation of three images: the first with two close triangles and a rectangle found in another order
    plus a far false positive, the second with a rectangle and no prediction, the third with a prediction and no ground
    truth. The detections take the boxes of their keypoints found, their scores and classes; the overlap of a
    prediction with a ground truth is the similarity of their keypoints (the visibilities, box area and class sigmas of
    the ground truth, a keypoint not found adding 0), 0 across classes; and every ground truth object gets the
    keypoints of its matched prediction, NaN when missed.
    """
    nan = float("nan")
    triangle = [[10.0, 2.0], [2.0, 18.0], [18.0, 18.0], [0.0, 0.0]]
    shifted_triangle = [[keypoint_x + 4.0, keypoint_y] for keypoint_x, keypoint_y in triangle]
    rectangle = [[1.0, 1.0], [19.0, 1.0], [19.0, 19.0], [1.0, 19.0]]

    # Predictions of the first image: the shifted triangle moved down by 2 pixels, the exact rectangle, the triangle
    # moved right by 1 pixel without its right keypoint, and a lone keypoint far from every object
    predicted_keypoints = [[[14.0, 4.0], [6.0, 20.0], [22.0, 20.0], [nan, nan]], rectangle,
                           [[11.0, 2.0], [3.0, 18.0], [nan, nan], [nan, nan]],
                           [[200.0, 200.0], [nan, nan], [nan, nan], [nan, nan]]]
    images = [make_image(labels=[1, 1, 2],
                         bounding_boxes=[[0.0, 0.0, 20.0, 20.0], [4.0, 0.0, 24.0, 20.0], [0.0, 0.0, 20.0, 20.0]],
                         keypoints=[triangle, shifted_triangle, rectangle],
                         visibilities=[[2, 2, 1, 0], [2, 0, 2, 0], [2, 2, 2, 2]], predicted_labels=[1, 2, 1, 1],
                         predicted_keypoints=predicted_keypoints, predicted_scores=[0.7, 0.6, 0.9, 0.3]),
              make_image(labels=[2], bounding_boxes=[[0.0, 0.0, 20.0, 20.0]], keypoints=[rectangle],
                         visibilities=[[2, 2, 2, 2]], predicted_labels=[], predicted_keypoints=[], predicted_scores=[]),
              make_image(labels=[], bounding_boxes=[], keypoints=[], visibilities=[], predicted_labels=[2],
                         predicted_keypoints=[rectangle], predicted_scores=[0.5])]
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas={"triangle": {"left": 0.1}},
                                             maximum_keypoints=MAXIMUM_KEYPOINTS)
    evaluation = evaluate_image_keypoints(images=images, sigma_table=sigma_table)
    detections, ground_truths = evaluation["detections"], evaluation["ground_truths"]

    # Detections boxed by their keypoints found, with their scores and classes, and the ground truth boxes
    assert len(detections) == len(ground_truths) == 3
    assert detections[0]["boxes"].tolist() == [[6.0, 4.0, 22.0, 20.0], [1.0, 1.0, 19.0, 19.0], [3.0, 2.0, 11.0, 18.0],
                                               [200.0, 200.0, 200.0, 200.0]]
    assert detections[0]["scores"].dtype == torch.float64
    assert detections[0]["scores"].tolist() == pytest.approx([0.7, 0.6, 0.9, 0.3])
    assert detections[0]["labels"].tolist() == [1, 2, 1, 1] and ground_truths[0]["labels"].tolist() == [1, 1, 2]
    assert ground_truths[0]["boxes"].dtype == torch.float64
    assert detections[1]["boxes"].shape == (0, 4) and detections[1]["keypoint_overlaps"].shape == (0, 1)
    assert detections[2]["keypoint_overlaps"].shape == (1, 0) and ground_truths[2]["boxes"].shape == (0, 4)

    # Every prediction of the first image against every ground truth, each ground truth bringing its own visibilities,
    # box area and sigmas, a keypoint not found adding 0, the rectangle never matching a triangle
    keypoint_overlaps = detections[0]["keypoint_overlaps"]
    assert keypoint_overlaps.shape == (4, 3)
    for prediction_index in range(4):
        for ground_truth_index in range(3):
            if images[0]["predicted_labels"][prediction_index] != images[0]["labels"][ground_truth_index]:
                assert keypoint_overlaps[prediction_index, ground_truth_index].item() == 0.0
                continue
            differences = (images[0]["predicted_keypoints"][prediction_index]
                           - images[0]["keypoints"][ground_truth_index])
            squared_distances = torch.nan_to_num(input=(differences ** 2).sum(dim=-1), nan=float("inf"))
            box = images[0]["bounding_boxes"][ground_truth_index]
            expected_similarity = compute_expected_similarity(
                squared_distances=squared_distances.tolist(), area=float((box[2] - box[0]) * (box[3] - box[1])),
                labelled=(images[0]["visibilities"][ground_truth_index] > 0).tolist(),
                sigmas=sigma_table[images[0]["labels"][ground_truth_index]].tolist())
            assert keypoint_overlaps[prediction_index, ground_truth_index].item() == pytest.approx(expected_similarity)
    assert keypoint_overlaps[3].tolist() == [0.0, 0.0, 0.0]

    # The ground truth objects of all the images with the keypoints of their matched prediction, NaN when missed
    objects = evaluation["objects"]
    assert sorted(objects) == sorted(EVALUATED_OBJECT_KEYS)
    assert objects["labels"].tolist() == [1, 1, 2, 2] and objects["matched"].tolist() == [True, True, True, False]
    expected_keypoints = torch.tensor(data=[predicted_keypoints[2], predicted_keypoints[0], rectangle,
                                            [[nan, nan]] * 4])
    assert torch.allclose(input=objects["predicted_keypoints"], other=expected_keypoints, equal_nan=True)
    assert objects["keypoints"].shape == (4, MAXIMUM_KEYPOINTS, 2) and objects["visibilities"].shape == (4, 4)


def test_objects_without_labelled_keypoint_are_scored_against_their_extended_box() -> None:
    """
    Verifies the similarity of the predicted objects with the objects without labelled keypoint, the rule of
    pycocotools for the COCO crowds and persons without keypoints: the squared distance of every keypoint found to the
    box extended by its width left and right and by its height above and below (0 inside, infinite for a keypoint not
    found), turned into a similarity with the box area and the sigmas of the class of the object and averaged over the
    keypoints found, 0 across classes; the detections carry these similarities, the ground truths the classes and crowd
    flags of these objects, and the ground truth objects of the keypoint accuracy leave them out.
    """
    nan = float("nan")

    # A 40 x 20 box extended to x 60 to 180 and y 80 to 140: inside, 10 pixels right of it, 3 left of and 4 below it
    ignored_box = [100.0, 100.0, 140.0, 120.0]
    predicted_rectangle = [[65.0, 85.0], [190.0, 85.0], [57.0, 144.0], [nan, nan]]
    squared_distances = compute_squared_extended_box_distances(
        predicted_keypoints=torch.tensor(data=[predicted_rectangle]), boxes=torch.tensor(data=[ignored_box]))
    assert squared_distances.dtype == torch.float64
    assert squared_distances.tolist() == [[0.0, 100.0, 25.0, float("inf")]]

    # The rectangle found there scores the mean similarity of its three keypoints found, a triangle at the same place 0
    triangle = [[10.0, 2.0], [2.0, 18.0], [18.0, 18.0], [0.0, 0.0]]
    images = [make_image(labels=[1], bounding_boxes=[[0.0, 0.0, 20.0, 20.0]], keypoints=[triangle],
                         visibilities=[[2, 2, 2, 0]], predicted_labels=[2, 1],
                         predicted_keypoints=[predicted_rectangle, predicted_rectangle], predicted_scores=[0.9, 0.8],
                         ignored_bounding_boxes=[ignored_box], ignored_labels=[2])]
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas={"rectangle": {"top_right": 0.1}},
                                             maximum_keypoints=MAXIMUM_KEYPOINTS)
    evaluation = evaluate_image_keypoints(images=images, sigma_table=sigma_table)
    expected_similarity = compute_expected_similarity(squared_distances=[0.0, 100.0, 25.0, float("inf")],
                                                      labelled=[True, True, True, False], area=800.0,
                                                      sigmas=sigma_table[2].tolist())
    ignored_overlaps = evaluation["detections"][0]["ignored_overlaps"]
    assert ignored_overlaps.shape == (2, 1) and ignored_overlaps.dtype == torch.float64
    assert ignored_overlaps[0, 0].item() == pytest.approx(expected_similarity) and ignored_overlaps[1, 0].item() == 0.0
    assert 0.0 < expected_similarity < 1.0
    assert evaluation["ground_truths"][0]["ignored_labels"].tolist() == [2]
    assert evaluation["ground_truths"][0]["ignored_crowds"].tolist() == [False]

    # The keypoint accuracy only measures the labelled triangle
    assert evaluation["objects"]["labels"].tolist() == [1]


def test_summarise_keypoint_accuracy_gives_the_global_class_and_keypoint_values() -> None:
    """
    Verifies the global fraction of matched objects, PCK and mean similarity over the labelled keypoints (the missed
    object counting as wrong) and mean pixel error over the keypoints found, the rows of every class and of every
    keypoint, None for a keypoint never labelled and for a class without object, the tables built from the rows, and
    the scalars; the matched fraction, the PCK and the mean OKS are written as percentages in the grid and the tables,
    the counts and pixel errors keeping their values.
    """
    objects = make_accuracy_objects()
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas={}, maximum_keypoints=MAXIMUM_KEYPOINTS)
    summary = summarise_keypoint_accuracy(objects=objects, sigma_table=sigma_table,
                                          keypoint_categories=KEYPOINT_CATEGORIES, class_names=CLASS_NAMES)
    similarities = measure_keypoint_accuracy(objects=objects, sigma_table=sigma_table)["similarities"]

    # Two of the three objects are matched, three of the seven labelled keypoints are correct, and the five keypoints
    # found have errors of 3, 6, 1, 2 and 3 pixels
    assert CORRECT_KEYPOINTS_NAME == "PCK@0.05"
    assert summary["metrics"]["Matched Objects"] == pytest.approx(2.0 / 3.0)
    assert summary["metrics"][CORRECT_KEYPOINTS_NAME] == pytest.approx(3.0 / 7.0)
    assert summary["metrics"]["Mean Keypoint Error (pixels)"] == pytest.approx(3.0)
    assert summary["metrics"]["Mean OKS"] == pytest.approx(float(similarities.mean()))
    assert summary["scalars"] == {f"Keypoint Accuracy/{name}": value for name, value in summary["metrics"].items()}

    # One row per class in class index order, the class without object having no mean
    class_rows = summary["class_rows"]
    assert [row["class_index"] for row in class_rows] == [1, 2, 3]
    assert class_rows[0] == {"class_index": 1, "class": "triangle", "objects": 1, "matched_objects": 1,
                             "labelled_keypoints": 2, "percentage_correct_keypoints": pytest.approx(0.5),
                             "mean_object_keypoint_similarity": pytest.approx(similarities[0].item()),
                             "mean_error_pixels": pytest.approx(4.5)}
    assert class_rows[1]["objects"] == 2 and class_rows[1]["matched_objects"] == 1
    assert class_rows[1]["labelled_keypoints"] == 5
    assert class_rows[1]["percentage_correct_keypoints"] == pytest.approx(2.0 / 5.0)
    assert class_rows[1]["mean_object_keypoint_similarity"] == pytest.approx(similarities[1].item() / 2.0)
    assert class_rows[1]["mean_error_pixels"] == pytest.approx(2.0)
    assert class_rows[2] == {"class_index": 3, "class": "square+diamond", "objects": 0, "matched_objects": 0,
                             "labelled_keypoints": 0, "percentage_correct_keypoints": None,
                             "mean_object_keypoint_similarity": None, "mean_error_pixels": None}

    # One row per keypoint of every class, the keypoints never labelled having no mean, those of the missed object
    # labelled but not found
    keypoint_rows = summary["keypoint_rows"]
    assert len(keypoint_rows) == 3 + 4 + 2
    assert find_row(rows=keypoint_rows, class_index=1, keypoint="top")["mean_error_pixels"] == pytest.approx(3.0)
    assert find_row(rows=keypoint_rows, class_index=1, keypoint="left")["percentage_correct_keypoints"] == 0.0
    assert find_row(rows=keypoint_rows, class_index=1, keypoint="right") == {
        "class_index": 1, "class": "triangle", "keypoint": "right", "labelled": 0, "found": 0,
        "percentage_correct_keypoints": None, "mean_error_pixels": None}
    assert find_row(rows=keypoint_rows, class_index=2, keypoint="top_right") == {
        "class_index": 2, "class": "rectangle", "keypoint": "top_right", "labelled": 2, "found": 1,
        "percentage_correct_keypoints": 0.5, "mean_error_pixels": pytest.approx(2.0)}
    assert find_row(rows=keypoint_rows, class_index=2, keypoint="bottom_right")["mean_error_pixels"] == pytest.approx(
        3.0)
    assert find_row(rows=keypoint_rows, class_index=2, keypoint="bottom_left")["labelled"] == 0

    # Tables with one line per row, sorted on the PCK column by the report
    class_table, keypoint_table = summary["tables"]["classes"], summary["tables"]["keypoints"]
    assert len(class_table["rows"]) == 3 and len(keypoint_table["rows"]) == 9
    assert class_table["sort_column"] == keypoint_table["sort_column"] == CORRECT_KEYPOINTS_NAME
    assert class_table["rows"][2] == ["square+diamond", 0, 0, 0, None, None, None]
    assert keypoint_table["rows"][2] == ["triangle", "right", 0, 0, None, None]
    assert len(class_table["columns"]) == len(class_table["rows"][0])
    assert len(keypoint_table["columns"]) == len(keypoint_table["rows"][0])

    # Rates of the grid and of the tables written as percentages, the counts and pixel errors as they are
    assert summary["percent_metrics"] == ["Matched Objects", CORRECT_KEYPOINTS_NAME, "Mean OKS"]
    assert class_table["percent_columns"] == [CORRECT_KEYPOINTS_NAME, "Mean\nOKS"]
    assert keypoint_table["percent_columns"] == [CORRECT_KEYPOINTS_NAME]
    class_lines = render_report_text(renderables=build_table_renderables(table=class_table)).splitlines()
    triangle_row = next(line for line in class_lines if line.startswith(" triangle")).split()
    assert triangle_row[1:] == ["1", "1", "2", "50.00", "%", f"{100.0 * similarities[0].item():.2f}", "%", "4.5000"]
