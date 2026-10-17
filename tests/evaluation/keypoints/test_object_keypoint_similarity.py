"""
Tests of vision_training/evaluation/keypoints/object_keypoint_similarity.py, the keypoint evaluation of the keypoint
trainer on the test split: the sigma table built from the configured COCO sigmas (default, padding positions, merged
categories and their conflict), the object keypoint similarity on a hand-computed case with the box-area scale, the
grouping of the objects by image into the detections and ground truths of the average precision, and the PCK and
pixel errors of every class and keypoint, with the classes and keypoints that have no labelled keypoint.
"""

import math
import pytest
import torch

from typing import Any, Dict, List

from vision_training.evaluation.keypoints.object_keypoint_similarity import (CORRECT_KEYPOINTS_NAME,
                                                                             build_keypoint_sigma_table,
                                                                             compute_object_keypoint_similarity,
                                                                             group_image_objects,
                                                                             measure_keypoint_accuracy,
                                                                             summarise_keypoint_accuracy)

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


def make_objects(image_identifiers: List[int], labels: List[int], bounding_boxes: List[List[float]],
                 keypoints: List[List[List[float]]], visibilities: List[List[int]],
                 keypoint_errors: List[List[List[float]]], scores: List[float]) -> Dict[str, torch.Tensor]:
    """
    Builds the evaluated objects of the test split in the format gathered by the keypoint trainer before the average
    precision and the keypoint accuracy, the predicted keypoints being the ground truth keypoints moved by a known
    error, so that every distance is known in advance.

    Args:
        image_identifiers (List[int]): The image of every object.
        labels (List[int]): The class index of every object.
        bounding_boxes (List[List[float]]): The [x1, y1, x2, y2] box of every object.
        keypoints (List[List[List[float]]]): The [K, 2] ground truth keypoints of every object.
        visibilities (List[List[int]]): The [K] COCO visibilities of every object.
        keypoint_errors (List[List[List[float]]]): The [K, 2] displacement of the predicted keypoints of every object.
        scores (List[float]): The keypoint score of every object.

    Returns:
        Dict[str, torch.Tensor]: The 'image_identifiers', 'labels', 'bounding_boxes', 'keypoints', 'visibilities',
            'predicted_keypoints' and 'scores' of the objects.
    """
    ground_truth_keypoints = torch.tensor(data=keypoints, dtype=torch.float32)
    return {"image_identifiers": torch.tensor(data=image_identifiers), "labels": torch.tensor(data=labels),
            "bounding_boxes": torch.tensor(data=bounding_boxes, dtype=torch.float32),
            "keypoints": ground_truth_keypoints, "visibilities": torch.tensor(data=visibilities),
            "predicted_keypoints": ground_truth_keypoints + torch.tensor(data=keypoint_errors, dtype=torch.float32),
            "scores": torch.tensor(data=scores, dtype=torch.float32)}


def make_accuracy_objects() -> Dict[str, torch.Tensor]:
    """
    Builds a triangle in a 100 x 50 box (correct within 5 pixels) and a rectangle in a 40 x 40 box (correct within 2
    pixels), whose predicted keypoints are moved along x by known pixel errors, with unlabelled keypoints and padding
    positions whose large errors must be left out of every metric.

    Returns:
        Dict[str, torch.Tensor]: The evaluated objects of make_objects.
    """
    triangle_keypoints = [[50.0, 5.0], [10.0, 45.0], [90.0, 45.0], [0.0, 0.0]]
    rectangle_keypoints = [[205.0, 5.0], [235.0, 5.0], [235.0, 35.0], [205.0, 35.0]]

    # The triangle has its 'right' keypoint unlabelled and a padding position, the rectangle its 'bottom_left'
    # unlabelled
    return make_objects(image_identifiers=[0, 0], labels=[1, 2],
                        bounding_boxes=[[0.0, 0.0, 100.0, 50.0], [200.0, 0.0, 240.0, 40.0]],
                        keypoints=[triangle_keypoints, rectangle_keypoints], visibilities=[[2, 1, 0, 0], [2, 1, 2, 0]],
                        keypoint_errors=[[[3.0, 0.0], [6.0, 0.0], [50.0, 0.0], [70.0, 0.0]],
                                         [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [10.0, 0.0]]],
                        scores=[0.9, 0.8])


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


def test_sigma_table_accepts_equal_sigmas_and_rejects_conflicting_ones_of_merged_categories() -> None:
    """
    Verifies that two categories merged into one class may give the same sigma to a keypoint, and that different sigmas
    raise an error naming the merged class.
    """
    equal_sigmas = {"square": {"centre": 0.3}, "diamond": {"centre": 0.3}}
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas=equal_sigmas, maximum_keypoints=MAXIMUM_KEYPOINTS)
    assert sigma_table[3, 0].item() == pytest.approx(0.3)

    # Different values of the same keypoint under both names cannot be resolved
    conflicting_sigmas = {"square": {"centre": 0.3}, "diamond": {"centre": 0.4}}
    with pytest.raises(ValueError, match="square\\+diamond"):
        build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                   keypoint_sigmas=conflicting_sigmas, maximum_keypoints=MAXIMUM_KEYPOINTS)


def test_object_keypoint_similarity_matches_a_hand_computed_case() -> None:
    """
    Verifies the similarity of an object in a 10 x 10 box with a sigma of 0.05: a perfect keypoint gives 1, a keypoint
    2 pixels away gives exp(-4 / (2 · 100 · 0.1²)) = exp(-2), and the unlabelled keypoint, however far, is left out of
    the mean.
    """
    similarity = compute_object_keypoint_similarity(squared_distances=torch.tensor(data=[[0.0, 4.0, 1.0e6]]),
                                                    visibilities=torch.tensor(data=[[2, 1, 0]]),
                                                    areas=torch.tensor(data=[100.0]),
                                                    sigmas=torch.full(size=(1, 3), fill_value=0.05))
    assert similarity.dtype == torch.float64 and similarity.shape == (1,)
    assert similarity.item() == pytest.approx((1.0 + math.exp(-2.0)) / 2.0)


def test_object_keypoint_similarity_of_perfect_and_unlabelled_objects() -> None:
    """
    Verifies that perfect keypoints give a similarity of 1 whatever the box, even a degenerate one, that an object
    without labelled keypoint gives 0 instead of a NaN, and that a larger box tolerates the same error better.
    """
    squared_distances = torch.tensor(data=[[0.0, 0.0], [0.0, 0.0], [9.0, 9.0], [25.0, 25.0], [25.0, 25.0]])
    visibilities = torch.tensor(data=[[2, 2], [2, 1], [0, 0], [2, 2], [2, 2]])
    areas = torch.tensor(data=[400.0, 0.0, 400.0, 400.0, 1600.0])
    similarities = compute_object_keypoint_similarity(squared_distances=squared_distances, visibilities=visibilities,
                                                      areas=areas, sigmas=torch.full(size=(5, 2), fill_value=0.05))
    assert similarities[:2].tolist() == [1.0, 1.0]
    assert similarities[2].item() == 0.0
    assert 0.0 < similarities[3].item() < similarities[4].item() < 1.0


def test_group_image_objects_builds_the_keypoint_overlaps_of_every_image() -> None:
    """
    Verifies that the objects are grouped by image in increasing identifier order, every ground truth giving one
    detection with its box, score and class, and that the overlap of a detection with a ground truth is the
    similarity of its keypoints with the keypoints of that ground truth (its visibilities, box area and class sigmas),
    0 across classes.
    """
    triangle_keypoints = [[10.0, 2.0], [2.0, 18.0], [18.0, 18.0], [0.0, 0.0]]
    shifted_triangle = [[keypoint_x + 4.0, keypoint_y] for keypoint_x, keypoint_y in triangle_keypoints]
    rectangle_keypoints = [[1.0, 1.0], [19.0, 1.0], [19.0, 19.0], [1.0, 19.0]]
    objects = make_objects(image_identifiers=[7, 3, 7, 7], labels=[1, 1, 1, 2],
                           bounding_boxes=[[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 20.0, 20.0],
                                           [4.0, 0.0, 24.0, 20.0], [0.0, 0.0, 20.0, 20.0]],
                           keypoints=[triangle_keypoints, triangle_keypoints, shifted_triangle, rectangle_keypoints],
                           visibilities=[[2, 2, 1, 0], [2, 2, 2, 0], [2, 0, 2, 0], [2, 2, 2, 2]],
                           keypoint_errors=[[[1.0, 0.0]] * 4, [[0.0, 0.0]] * 4, [[0.0, 2.0]] * 4, [[0.0, 0.0]] * 4],
                           scores=[0.9, 0.8, 0.7, 0.6])
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas={"triangle": {"left": 0.1}},
                                             maximum_keypoints=MAXIMUM_KEYPOINTS)
    detections, ground_truths = group_image_objects(objects=objects, sigma_table=sigma_table)

    # Image 3 first with its single object, then image 7 with its three objects in their original order
    assert len(detections) == len(ground_truths) == 2
    assert detections[0]["labels"].tolist() == [1] and detections[1]["labels"].tolist() == [1, 1, 2]
    assert detections[1]["scores"].tolist() == pytest.approx([0.9, 0.7, 0.6])
    assert torch.equal(detections[1]["boxes"], ground_truths[1]["boxes"])
    assert torch.equal(ground_truths[1]["labels"], detections[1]["labels"])
    assert detections[0]["keypoint_overlaps"].tolist() == [[1.0]]

    # Every detection of image 7 against every ground truth of image 7, each ground truth bringing its own visibilities,
    # box area and sigmas, the rectangle never matching a triangle
    image_objects = [0, 2, 3]
    keypoint_overlaps = detections[1]["keypoint_overlaps"]
    assert keypoint_overlaps.shape == (3, 3)
    for detection_position, detection_index in enumerate(image_objects):
        for ground_truth_position, ground_truth_index in enumerate(image_objects):
            if objects["labels"][detection_index] != objects["labels"][ground_truth_index]:
                assert keypoint_overlaps[detection_position, ground_truth_position].item() == 0.0
                continue
            differences = objects["predicted_keypoints"][detection_index] - objects["keypoints"][ground_truth_index]
            box = objects["bounding_boxes"][ground_truth_index]
            expected_similarity = compute_expected_similarity(
                squared_distances=(differences ** 2).sum(dim=-1).tolist(),
                labelled=(objects["visibilities"][ground_truth_index] > 0).tolist(),
                area=float((box[2] - box[0]) * (box[3] - box[1])),
                sigmas=sigma_table[objects["labels"][ground_truth_index]].tolist())
            assert keypoint_overlaps[detection_position, ground_truth_position].item() == pytest.approx(
                expected_similarity)


def test_measure_keypoint_accuracy_counts_the_keypoints_within_the_box_fraction() -> None:
    """
    Verifies the distances, the labelled and correct keypoints (within 0.05 of the longest box side, the limit itself
    included) and the similarity of every object, the unlabelled keypoints and padding positions never being correct.
    """
    objects = make_accuracy_objects()
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas={}, maximum_keypoints=MAXIMUM_KEYPOINTS)
    accuracy = measure_keypoint_accuracy(objects=objects, sigma_table=sigma_table)

    assert torch.allclose(input=accuracy["distances"], other=torch.tensor(
        data=[[3.0, 6.0, 50.0, 70.0], [1.0, 2.0, 3.0, 10.0]], dtype=torch.float64))
    assert accuracy["labelled"].tolist() == [[True, True, False, False], [True, True, True, False]]
    assert accuracy["correct"].tolist() == [[True, False, False, False], [True, True, False, False]]

    # Similarities from their definition, with the 100 x 50 and 40 x 40 box areas
    expected_similarities = [
        compute_expected_similarity(squared_distances=[9.0, 36.0, 2500.0, 4900.0], labelled=[True, True, False, False],
                                    area=5000.0, sigmas=[DEFAULT_SIGMA] * 4),
        compute_expected_similarity(squared_distances=[1.0, 4.0, 9.0, 100.0], labelled=[True, True, True, False],
                                    area=1600.0, sigmas=[DEFAULT_SIGMA] * 4)]
    assert accuracy["similarities"].tolist() == pytest.approx(expected_similarities)


def test_summarise_keypoint_accuracy_gives_the_global_class_and_keypoint_values() -> None:
    """
    Verifies the global PCK, mean similarity and mean pixel error over the labelled keypoints, the rows of every class
    and of every keypoint, None for a keypoint never labelled and for a class without object, the tables built from the
    rows, and the scalars.
    """
    objects = make_accuracy_objects()
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas={}, maximum_keypoints=MAXIMUM_KEYPOINTS)
    summary = summarise_keypoint_accuracy(objects=objects, sigma_table=sigma_table,
                                          keypoint_categories=KEYPOINT_CATEGORIES, class_names=CLASS_NAMES)
    similarities = measure_keypoint_accuracy(objects=objects, sigma_table=sigma_table)["similarities"]

    # Three of the five labelled keypoints are correct, with errors of 3, 6, 1, 2 and 3 pixels
    assert CORRECT_KEYPOINTS_NAME == "PCK@0.05"
    assert summary["metrics"][CORRECT_KEYPOINTS_NAME] == pytest.approx(0.6)
    assert summary["metrics"]["Mean Keypoint Error (pixels)"] == pytest.approx(3.0)
    assert summary["metrics"]["Mean OKS"] == pytest.approx(float(similarities.mean()))
    assert summary["scalars"] == {f"Keypoint Accuracy/{name}": value for name, value in summary["metrics"].items()}

    # One row per class in class index order, the class without object having no mean
    class_rows = summary["class_rows"]
    assert [row["class_index"] for row in class_rows] == [1, 2, 3]
    assert class_rows[0] == {"class_index": 1, "class": "triangle", "objects": 1, "labelled_keypoints": 2,
                             "percentage_correct_keypoints": pytest.approx(0.5),
                             "mean_object_keypoint_similarity": pytest.approx(similarities[0].item()),
                             "mean_error_pixels": pytest.approx(4.5)}
    assert class_rows[1]["labelled_keypoints"] == 3
    assert class_rows[1]["percentage_correct_keypoints"] == pytest.approx(2.0 / 3.0)
    assert class_rows[1]["mean_error_pixels"] == pytest.approx(2.0)
    assert class_rows[2] == {"class_index": 3, "class": "square+diamond", "objects": 0, "labelled_keypoints": 0,
                             "percentage_correct_keypoints": None, "mean_object_keypoint_similarity": None,
                             "mean_error_pixels": None}

    # One row per keypoint of every class, the keypoints never labelled having no mean
    keypoint_rows = summary["keypoint_rows"]
    assert len(keypoint_rows) == 3 + 4 + 2
    assert find_row(rows=keypoint_rows, class_index=1, keypoint="top")["mean_error_pixels"] == pytest.approx(3.0)
    assert find_row(rows=keypoint_rows, class_index=1, keypoint="left")["percentage_correct_keypoints"] == 0.0
    assert find_row(rows=keypoint_rows, class_index=1, keypoint="right") == {
        "class_index": 1, "class": "triangle", "keypoint": "right", "labelled": 0,
        "percentage_correct_keypoints": None, "mean_error_pixels": None}
    assert find_row(rows=keypoint_rows, class_index=2, keypoint="top_right")["percentage_correct_keypoints"] == 1.0
    assert find_row(rows=keypoint_rows, class_index=2, keypoint="bottom_right")["mean_error_pixels"] == pytest.approx(
        3.0)
    assert find_row(rows=keypoint_rows, class_index=2, keypoint="bottom_left")["labelled"] == 0

    # Tables with one line per row, sorted on the PCK column by the report
    class_table, keypoint_table = summary["tables"]["classes"], summary["tables"]["keypoints"]
    assert len(class_table["rows"]) == 3 and len(keypoint_table["rows"]) == 9
    assert class_table["sort_column"] == keypoint_table["sort_column"] == CORRECT_KEYPOINTS_NAME
    assert class_table["rows"][2] == ["square+diamond", 0, 0, None, None, None]
    assert keypoint_table["rows"][2] == ["triangle", "right", 0, None, None]


def test_summarise_keypoint_accuracy_without_labelled_keypoint_has_no_value() -> None:
    """
    Verifies that a split without any labelled keypoint gives no PCK, no pixel error and no mean similarity, neither in
    the metrics, the scalars nor the class rows, instead of a NaN or the meaningless similarity of 0 of such objects.
    """
    objects = make_accuracy_objects()
    objects["visibilities"] = torch.zeros_like(input=objects["visibilities"])
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=DEFAULT_SIGMA,
                                             keypoint_sigmas={}, maximum_keypoints=MAXIMUM_KEYPOINTS)
    summary = summarise_keypoint_accuracy(objects=objects, sigma_table=sigma_table,
                                          keypoint_categories=KEYPOINT_CATEGORIES, class_names=CLASS_NAMES)
    assert summary["metrics"][CORRECT_KEYPOINTS_NAME] is None
    assert summary["metrics"]["Mean Keypoint Error (pixels)"] is None
    assert summary["metrics"]["Mean OKS"] is None
    assert summary["scalars"] == {}
    assert all(row["mean_object_keypoint_similarity"] is None for row in summary["class_rows"])
    assert all(row["percentage_correct_keypoints"] is None for row in summary["keypoint_rows"])
