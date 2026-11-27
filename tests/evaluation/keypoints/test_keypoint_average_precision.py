"""
Tests of the average precision of the keypoint detection, the object keypoint similarity taking the place of the
intersection over union: the predicted and ground truth objects of every image compared by evaluate_image_keypoints
(evaluation/keypoints/object_keypoint_similarity.py) are evaluated by evaluate_detections with the 'keypoints' overlap
(evaluation/detection/mean_average_precision.py), as the keypoint trainer does on the test split: keypoints whose
similarity passes 0.5 but not 0.75, next to perfect ones, with the mean average recall, and the objects found on the
objects without labelled keypoint ignored as in pycocotools, a crowd taking any number of them.
"""

import pytest
import torch

from typing import Dict, List, Optional, Tuple

from vision_training.evaluation.detection.mean_average_precision import (INTERSECTION_OVER_UNION_THRESHOLDS,
                                                                         evaluate_detections)
from vision_training.evaluation.keypoints.object_keypoint_similarity import (build_keypoint_sigma_table,
                                                                             evaluate_image_keypoints)

# Foreground classes of the keypoint task, both with the four corners of a square as keypoints
KEYPOINT_CATEGORIES = {1: {"name": "square", "keypoints": ["top_left", "top_right", "bottom_right", "bottom_left"]},
                       2: {"name": "diamond", "keypoints": ["top", "right", "bottom", "left"]}}

# Number of classes of the average precision, background included
NUMBER_CLASSES = 3

# Displacement of every keypoint of a 100 x 100 object giving a similarity of exp(-100 / (2 · 10000 · 0.1²)) =
# exp(-0.5) ≈ 0.61 with the default sigma of 0.05, matched at the thresholds 0.5 to 0.6 but not from 0.65 on
PARTIAL_SHIFT_PIXELS = 10.0

# Box of the square without labelled keypoint of add_ignored_square, next to the labelled squares at the left
IGNORED_SQUARE_BOX = [300.0, 0.0, 400.0, 100.0]

# Area of the evaluated network input images, 512 x 512 pixels, which sizes their squares for the average precision
IMAGE_AREA = 512.0 * 512.0


def make_square_images(object_definitions: List[Tuple[int, int, float, float, float, float]]) -> List[
        Dict[str, torch.Tensor]]:
    """
    Builds the evaluated test images, as gathered by the keypoint trainer, from squares whose four corners are the
    keypoints, every square being found by the detector with its keypoints all moved along x by a known shift, or
    missed when its score is negative.

    Args:
        object_definitions (List[Tuple[int, int, float, float, float, float]]): The image index, class index, left
            coordinate, side, keypoint shift in pixels and score of every object, a negative score for a missed one.

    Returns:
        List[Dict[str, torch.Tensor]]: The ground truth 'labels', 'bounding_boxes', 'keypoints' and 'visibilities' of
            every image, its 'ignored_bounding_boxes', 'ignored_labels' and 'ignored_crowds' (none), its
            'predicted_labels', 'predicted_keypoints' and 'predicted_scores', and its 'image_area' (IMAGE_AREA).
    """
    number_images = max(definition[0] for definition in object_definitions) + 1
    images = [{"labels": [], "bounding_boxes": [], "keypoints": [], "predicted_labels": [], "predicted_keypoints": [],
               "predicted_scores": []} for _ in range(number_images)]
    for image_index, class_index, left, side, shift, score in object_definitions:
        image = images[image_index]
        corners = [[left, 0.0], [left + side, 0.0], [left + side, side], [left, side]]
        image["labels"].append(class_index)
        image["bounding_boxes"].append([left, 0.0, left + side, side])
        image["keypoints"].append(corners)

        # The detector finds the square with its keypoints shifted, unless it misses it
        if score >= 0.0:
            image["predicted_labels"].append(class_index)
            image["predicted_keypoints"].append([[corner_x + shift, corner_y] for corner_x, corner_y in corners])
            image["predicted_scores"].append(score)
    return [{"labels": torch.tensor(data=image["labels"], dtype=torch.int64),
             "bounding_boxes": torch.tensor(data=image["bounding_boxes"], dtype=torch.float32),
             "keypoints": torch.tensor(data=image["keypoints"], dtype=torch.float32),
             "visibilities": torch.full(size=(len(image["labels"]), 4), fill_value=2, dtype=torch.int64),
             "ignored_bounding_boxes": torch.zeros(size=(0, 4), dtype=torch.float32),
             "ignored_labels": torch.zeros(size=(0,), dtype=torch.int64),
             "ignored_crowds": torch.zeros(size=(0,), dtype=torch.bool),
             "predicted_labels": torch.tensor(data=image["predicted_labels"], dtype=torch.int64),
             "predicted_keypoints": torch.tensor(data=image["predicted_keypoints"], dtype=torch.float32).reshape(
                 -1, 4, 2),
             "predicted_scores": torch.tensor(data=image["predicted_scores"], dtype=torch.float32),
             "image_area": IMAGE_AREA} for image in images]


def add_ignored_square(image: Dict[str, torch.Tensor], ignored_crowds: List[bool],
                       prediction_scores: List[float]) -> Dict[str, torch.Tensor]:
    """
    Adds to an image of make_square_images squares of class 1 without labelled keypoint, all on IGNORED_SQUARE_BOX (as
    a COCO person without keypoints or a crowd), and predicted squares of class 1 found exactly on that box, so that
    the evaluation must ignore them; no ignored square gives the image of the loaders that removed these objects.

    Args:
        image (Dict[str, torch.Tensor]): An evaluated image of make_square_images.
        ignored_crowds (List[bool]): Whether every added square without labelled keypoint is a crowd.
        prediction_scores (List[float]): The score of every predicted square found on IGNORED_SQUARE_BOX.

    Returns:
        Dict[str, torch.Tensor]: The image with its ignored objects and its added predictions after its own.
    """
    left, top, right, bottom = IGNORED_SQUARE_BOX
    corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
    number_predictions = len(prediction_scores)
    return {**image,
            "ignored_bounding_boxes": torch.tensor(data=[IGNORED_SQUARE_BOX] * len(ignored_crowds),
                                                   dtype=torch.float32).reshape(-1, 4),
            "ignored_labels": torch.ones(size=(len(ignored_crowds),), dtype=torch.int64),
            "ignored_crowds": torch.tensor(data=ignored_crowds, dtype=torch.bool),
            "predicted_labels": torch.cat(tensors=[image["predicted_labels"],
                                                   torch.ones(size=(number_predictions,), dtype=torch.int64)]),
            "predicted_keypoints": torch.cat(tensors=[image["predicted_keypoints"], torch.tensor(
                data=[corners] * number_predictions, dtype=torch.float32).reshape(-1, 4, 2)]),
            "predicted_scores": torch.cat(tensors=[image["predicted_scores"],
                                                   torch.tensor(data=prediction_scores, dtype=torch.float32)])}


def evaluate_keypoint_images(images: List[Dict[str, torch.Tensor]],
                             target_precision: Optional[float] = None) -> Dict[str, Dict]:
    """
    Evaluates the images with the average precision on the object keypoint similarity, the default sigma of 0.05
    applying to every keypoint, as the keypoint trainer does at the end of its test evaluation.

    Args:
        images (List[Dict[str, torch.Tensor]]): The evaluated images of make_square_images.
        target_precision (Optional[float]): The precision of the score threshold of every class, None to skip the
            thresholds. Defaults to None.

    Returns:
        Dict[str, Dict]: The results of evaluate_detections for every average precision method.
    """
    sigma_table = build_keypoint_sigma_table(keypoint_categories=KEYPOINT_CATEGORIES, default_sigma=0.05,
                                             keypoint_sigmas={}, maximum_keypoints=4)
    evaluation = evaluate_image_keypoints(images=images, sigma_table=sigma_table)
    return evaluate_detections(detections=evaluation["detections"], ground_truths=evaluation["ground_truths"],
                               number_classes=NUMBER_CLASSES, overlap_type="keypoints",
                               target_precision=target_precision)


def test_partially_correct_keypoints_lower_the_strict_thresholds_first() -> None:
    """
    Verifies that an object whose keypoints reach a similarity of about 0.61 counts as found up to the threshold 0.6
    and missed from 0.65 on: with a perfect object of higher score in the same image, the AP and the recall stay 1 at
    0.5 and fall to 0.5 at 0.75 and 0.95, and their mean over the ten thresholds is (3 · 1 + 7 · 0.5) / 10.
    """
    images = make_square_images(object_definitions=[(0, 1, 0.0, 100.0, 0.0, 0.9),
                                                      (0, 1, 300.0, 100.0, PARTIAL_SHIFT_PIXELS, 0.8)])
    results = evaluate_keypoint_images(images=images)
    area_results = results["area"]
    assert area_results["mAP@0.5"] == pytest.approx(1.0) and area_results["mAP@0.75"] == pytest.approx(0.5)
    assert area_results["mAP@0.95"] == pytest.approx(0.5) and area_results["mAP@[0.5:0.95]"] == pytest.approx(0.65)
    assert area_results["class_AP@0.5"] == {1: pytest.approx(1.0)}
    assert area_results["mAR@0.5"] == pytest.approx(1.0) and area_results["mAR@0.75"] == pytest.approx(0.5)
    assert area_results["mAR@0.95"] == pytest.approx(0.5) and area_results["mAR@[0.5:0.95]"] == pytest.approx(0.65)
    assert INTERSECTION_OVER_UNION_THRESHOLDS[2:4] == [0.6, 0.65]

    # The same keypoints ranked first make a false positive come before the true positive at 0.75
    swapped_images = make_square_images(object_definitions=[(0, 1, 0.0, 100.0, 0.0, 0.8),
                                                              (0, 1, 300.0, 100.0, PARTIAL_SHIFT_PIXELS, 0.9)])
    swapped_results = evaluate_keypoint_images(images=swapped_images)
    assert swapped_results["area"]["mAP@0.5"] == pytest.approx(1.0)
    assert swapped_results["area"]["mAP@0.75"] < area_results["mAP@0.75"]


def test_objects_found_on_objects_without_labelled_keypoint_are_ignored() -> None:
    """
    Verifies that a predicted object found on an object without labelled keypoint (a COCO person without keypoints or
    a crowd) is neither a true nor a false positive, as in pycocotools: ranked before the true positive, it leaves the
    AP at 1, of all the objects and of their size (100 pixel squares, a side from 1/8 to 1/4 of the 512 pixel image),
    and lets the class reach a precision of 0.99 at the score of the true positive, where the same prediction with that
    object removed from the ground truth is a false positive that halves the AP and leaves no threshold; a crowd takes
    every object found on it, any other ignored object only the most confident one, the second one being a false
    positive unless a second ignored object it reaches is still free, even when both tie at a similarity of 1.
    """
    labelled_image = make_square_images(object_definitions=[(0, 1, 0.0, 100.0, 0.0, 0.5)])[0]

    # A confident object found on a person without keypoints is ignored
    ignored_results = evaluate_keypoint_images(images=[add_ignored_square(
        image=labelled_image, ignored_crowds=[False], prediction_scores=[0.9])], target_precision=0.99)
    assert ignored_results["area"]["mAP@0.5"] == pytest.approx(1.0)
    assert ignored_results["area"]["mAP@[0.5:0.95]"] == pytest.approx(1.0)
    assert ignored_results["area"]["sizes"]["1_8_to_1_4"]["mAP@0.5"] == pytest.approx(1.0)
    assert ignored_results["class_score_thresholds"][1] == pytest.approx((0.5, 1.0))

    # The same object found with the person removed from the ground truth is a false positive ranked first
    removed_results = evaluate_keypoint_images(images=[add_ignored_square(
        image=labelled_image, ignored_crowds=[], prediction_scores=[0.9])], target_precision=0.99)
    assert removed_results["area"]["mAP@0.5"] == pytest.approx(0.5)
    assert removed_results["class_score_thresholds"][1] is None

    # Two objects found on the same ignored object: a crowd takes both, a person only the first
    crowd_results = evaluate_keypoint_images(images=[add_ignored_square(
        image=labelled_image, ignored_crowds=[True], prediction_scores=[0.9, 0.8])])
    person_results = evaluate_keypoint_images(images=[add_ignored_square(
        image=labelled_image, ignored_crowds=[False], prediction_scores=[0.9, 0.8])])
    assert crowd_results["area"]["mAP@0.5"] == pytest.approx(1.0)
    assert person_results["area"]["mAP@0.5"] == pytest.approx(0.5)
    assert person_results["area"]["mAR@0.5"] == pytest.approx(1.0)

    # Two persons without keypoints on the same place, both at a similarity of 1: the second object found takes the
    # second person, the first one being taken
    two_person_results = evaluate_keypoint_images(images=[add_ignored_square(
        image=labelled_image, ignored_crowds=[False, False], prediction_scores=[0.9, 0.8])])
    assert two_person_results["area"]["mAP@0.5"] == pytest.approx(1.0)
