"""
Tests of vision_training/evaluation/detection/detection_confusion.py on hand-built images: the matching of the
detections to the objects (the right class first, the most confident detection first, the overlap threshold), the
confusion matrix with its background row and column, the grouping of the errors of an image into the folders of the
error images, the summary of the outcomes, and the box and mask overlaps.
"""

import pytest
import torch

from vision_training.evaluation.detection.detection_confusion import (add_to_confusion_matrix,
                                                                      compute_overlap_matrix, group_image_errors,
                                                                      match_image_detections,
                                                                      summarise_confusion_matrix)

# Names of the classes of the tests, the background first
CLASS_NAMES = {0: "background", 1: "circle", 2: "square", 3: "triangle"}


def test_right_class_is_matched_before_a_closer_wrong_class() -> None:
    """
    Verifies that an object found by a detection of its class counts as found even when a more confident detection of
    another class overlaps it more: the detection of the right class takes it in the first pass, and the other
    detection, left without object, is a false positive.
    """
    overlaps = torch.tensor(data=[[0.9], [0.6]], dtype=torch.float64)
    matched = match_image_detections(overlaps=overlaps, detection_scores=torch.tensor(data=[0.9, 0.5]),
                                     detection_labels=torch.tensor(data=[2, 1]),
                                     ground_truth_labels=torch.tensor(data=[1]), intersection_over_union_threshold=0.5)
    assert matched.tolist() == [-1, 0]


def test_wrong_class_detection_finds_an_object_left_unfound() -> None:
    """
    Verifies the confusions: a detection of another class overlapping an object that no detection of its class found
    takes it in the second pass, while a detection overlapping no object enough stays a false positive.
    """
    overlaps = torch.tensor(data=[[0.7, 0.0], [0.0, 0.4]], dtype=torch.float64)
    matched = match_image_detections(overlaps=overlaps, detection_scores=torch.tensor(data=[0.8, 0.9]),
                                     detection_labels=torch.tensor(data=[2, 3]),
                                     ground_truth_labels=torch.tensor(data=[1, 3]),
                                     intersection_over_union_threshold=0.5)
    assert matched.tolist() == [0, -1]


def test_most_confident_detection_takes_the_object() -> None:
    """
    Verifies that two detections of the same class on the same object share it in decreasing score order: the most
    confident one finds it, even with the smaller overlap, and the duplicate is a false positive.
    """
    overlaps = torch.tensor(data=[[0.95], [0.6]], dtype=torch.float64)
    matched = match_image_detections(overlaps=overlaps, detection_scores=torch.tensor(data=[0.3, 0.8]),
                                     detection_labels=torch.tensor(data=[1, 1]),
                                     ground_truth_labels=torch.tensor(data=[1]), intersection_over_union_threshold=0.5)
    assert matched.tolist() == [-1, 0]


@pytest.mark.parametrize(argnames=("overlap", "expected_match"), argvalues=[(0.49, -1), (0.5, 0)])
def test_overlap_threshold_is_inclusive(overlap: float, expected_match: int) -> None:
    """
    Verifies that a detection finds an object from an overlap equal to the threshold, and not below it.

    Args:
        overlap (float): The overlap of the detection with the object.
        expected_match (int): The expected object of the detection, -1 for none.
    """
    matched = match_image_detections(overlaps=torch.tensor(data=[[overlap]], dtype=torch.float64),
                                     detection_scores=torch.tensor(data=[0.9]), detection_labels=torch.tensor(data=[1]),
                                     ground_truth_labels=torch.tensor(data=[1]), intersection_over_union_threshold=0.5)
    assert matched.tolist() == [expected_match]


def test_confusion_matrix_counts_every_detection_and_object_once() -> None:
    """
    Verifies the cells of the confusion matrix of an image with a right detection, a confusion, a false positive and a
    missed object: the right detection on the diagonal, the confusion in the row of its object, the false positive in
    the background row, the missed object in the background column, the background cell empty, and the summary of
    these outcomes.
    """
    confusion_matrix = torch.zeros(size=(4, 4), dtype=torch.int64)
    add_to_confusion_matrix(confusion_matrix=confusion_matrix,
                            matched_ground_truth_indices=torch.tensor(data=[0, 1, -1]),
                            detection_labels=torch.tensor(data=[1, 3, 2]),
                            ground_truth_labels=torch.tensor(data=[1, 2, 3]))
    assert confusion_matrix.tolist() == [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    assert summarise_confusion_matrix(confusion_matrix=confusion_matrix) == {"right": 1, "wrong_class": 1,
                                                                             "false_positives": 1, "missed": 1}

    # An image without detection adds its objects to the background column
    add_to_confusion_matrix(confusion_matrix=confusion_matrix,
                            matched_ground_truth_indices=torch.zeros(size=(0,), dtype=torch.int64),
                            detection_labels=torch.zeros(size=(0,), dtype=torch.int64),
                            ground_truth_labels=torch.tensor(data=[2, 2]))
    assert int(confusion_matrix[2, 0]) == 2 and int(confusion_matrix.sum()) == 6


def test_errors_are_grouped_by_folder() -> None:
    """
    Verifies the grouping of the errors of an image: two false positives of one class in one folder with their
    highest score and their count, the missed objects of a class without score, the confusion with both of its boxes,
    and no folder for the right detection.
    """
    image_errors = group_image_errors(matched_ground_truth_indices=torch.tensor(data=[0, -1, -1, 2]),
                                      detection_scores=torch.tensor(data=[0.9, 0.4, 0.7, 0.8]),
                                      detection_labels=torch.tensor(data=[1, 2, 2, 1]),
                                      ground_truth_labels=torch.tensor(data=[1, 3, 3]), class_names=CLASS_NAMES)
    assert list(image_errors) == ["FALSE-POSITIVE/square", "MISSED/triangle", "triangle-TO-circle"]
    assert image_errors["FALSE-POSITIVE/square"]["detection_indices"] == [1, 2]
    assert image_errors["FALSE-POSITIVE/square"]["score"] == pytest.approx(expected=0.7)
    assert image_errors["FALSE-POSITIVE/square"]["description"] == "FALSE-POSITIVE square x2"
    assert image_errors["MISSED/triangle"] == {"detection_indices": [], "ground_truth_indices": [1], "score": None,
                                               "description": "MISSED triangle"}
    assert image_errors["triangle-TO-circle"]["detection_indices"] == [3]
    assert image_errors["triangle-TO-circle"]["ground_truth_indices"] == [2]
    assert image_errors["triangle-TO-circle"]["description"] == "triangle -> circle"


def test_overlap_matrix_of_boxes_and_masks() -> None:
    """
    Verifies the overlaps of the matching: the box intersection over union of every detection with every object,
    whatever their classes, the mask overlaps given by the instance segmentation trainer, and an unknown overlap type
    refused.
    """
    detection = {"boxes": torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                 "mask_overlaps": torch.tensor(data=[[0.25], [0.0]])}
    ground_truth = {"boxes": torch.tensor(data=[[0.0, 0.0, 10.0, 5.0]])}
    box_overlaps = compute_overlap_matrix(detection=detection, ground_truth=ground_truth, overlap_type="boxes")
    assert box_overlaps.dtype == torch.float64 and box_overlaps.shape == (2, 1)
    assert box_overlaps[:, 0].tolist() == pytest.approx(expected=[0.5, 0.0], abs=1e-6)
    mask_overlaps = compute_overlap_matrix(detection=detection, ground_truth=ground_truth, overlap_type="masks")
    assert mask_overlaps[:, 0].tolist() == [0.25, 0.0]
    with pytest.raises(ValueError, match="Unknown overlap type"):
        compute_overlap_matrix(detection=detection, ground_truth=ground_truth, overlap_type="pixels")
