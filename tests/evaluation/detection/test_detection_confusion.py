"""
Tests of vision_training/evaluation/detection/detection_confusion.py on hand-built images: the matching of the
detections to the objects (the right class first), the confusion matrix with its background row and column and
the summary of the outcomes, and the overlap of every detection with its object written in the captions of the sample
and error images.
"""

import math
import torch

from vision_training.evaluation.detection.detection_confusion import (add_to_confusion_matrix,
                                                                      gather_detection_overlaps,
                                                                      match_image_detections,
                                                                      summarise_confusion_matrix)


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


def test_confusion_matrix_counts_every_detection_and_object_once() -> None:
    """
    Verifies the cells of the confusion matrix of an image with a right detection, a confusion, a false positive and a
    missed object: the right detection on the diagonal, the confusion in the row of its object, the false positive in
    the background row, the missed object in the background column, the background cell empty, and the summary of
    these outcomes.
    """
    confusion_matrix = torch.zeros(size=(4, 4), dtype=torch.int64)
    add_to_confusion_matrix(
        confusion_matrix=confusion_matrix, matched_ground_truth_indices=torch.tensor(data=[0, 1, -1]),
        detection_labels=torch.tensor(data=[1, 3, 2]), ground_truth_labels=torch.tensor(data=[1, 2, 3]))
    assert confusion_matrix.tolist() == [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    assert summarise_confusion_matrix(confusion_matrix=confusion_matrix) == {"right": 1, "wrong_class": 1,
                                                                             "false_positives": 1, "missed": 1}

    # An image without detection adds its objects to the background column
    add_to_confusion_matrix(
        confusion_matrix=confusion_matrix, matched_ground_truth_indices=torch.zeros(size=(0,), dtype=torch.int64),
        detection_labels=torch.zeros(size=(0,), dtype=torch.int64), ground_truth_labels=torch.tensor(data=[2, 2]))
    assert int(confusion_matrix[2, 0]) == 2 and int(confusion_matrix.sum()) == 6


def test_every_detection_reads_the_overlaps_of_its_object() -> None:
    """
    Verifies the overlaps written in the caption of every detection: a matched detection reads the box and mask
    overlaps of its own object, even when it overlaps another object more by box, a detection left without object
    reads those of the object it overlaps the most by the matching overlap (the masks here), and every detection of
    an image without object gets NaN.
    """
    overlaps = {"boxes": torch.tensor(data=[[0.6, 0.7], [0.2, 0.55], [0.1, 0.3]], dtype=torch.float64),
                "masks": torch.tensor(data=[[0.8, 0.1], [0.4, 0.2], [0.05, 0.35]], dtype=torch.float64)}
    detection_overlaps = gather_detection_overlaps(overlaps=overlaps,
                                                   matched_ground_truth_indices=torch.tensor(data=[0, -1, -1]),
                                                   matching_overlap_type="masks")
    assert detection_overlaps["boxes"].tolist() == [0.6, 0.2, 0.3]
    assert detection_overlaps["masks"].tolist() == [0.8, 0.4, 0.35]

    # An image without object, and an image without detection
    empty_overlaps = gather_detection_overlaps(overlaps={"boxes": torch.zeros(size=(2, 0), dtype=torch.float64)},
                                               matched_ground_truth_indices=torch.tensor(data=[-1, -1]),
                                               matching_overlap_type="boxes")
    assert all(math.isnan(overlap) for overlap in empty_overlaps["boxes"].tolist())
    assert gather_detection_overlaps(overlaps={"boxes": torch.zeros(size=(0, 3), dtype=torch.float64)},
                                     matched_ground_truth_indices=torch.zeros(size=(0,), dtype=torch.int64),
                                     matching_overlap_type="boxes")["boxes"].shape == (0,)
