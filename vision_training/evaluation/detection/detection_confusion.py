"""
Error analysis of the object detection and instance segmentation tasks at the deployment score threshold: matching of
the detections of a test image to its objects, the confusion matrix with a background row (the false positives) and a
background column (the missed objects), the errors of an image grouped into the folders of the error images, and the
overlap of every detection with the object it is matched to, written in its caption on the sample and error images.
Used by the detection trainers on every evaluated test image, next to the mean average precision, which keeps the
detections down to the evaluation score threshold instead.
"""

import torch

from typing import Any, Dict

from vision_training.evaluation.detection.mean_average_precision import OVERLAP_TYPES
from vision_training.models.detection.box_operations import get_intersection_over_union

# Class index of the background in the detection models, the row of the false positives and the column of the missed
# objects in the confusion matrix
BACKGROUND_CLASS_INDEX = 0

# Overlap from which a detection finds an object in the confusion matrix and the error images, the one of the headline
# mAP@0.5
MATCHING_INTERSECTION_OVER_UNION_THRESHOLD = 0.5

# Folders of the error images of the detections finding no object and of the objects found by no detection, each
# holding one folder per class, the confusions going to '<TRUE>-TO-<PREDICTED>'
FALSE_POSITIVE_FOLDER = "FALSE-POSITIVE"
MISSED_FOLDER = "MISSED"


def compute_overlap_matrix(detection: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                           overlap_type: str) -> torch.Tensor:
    """
    Gives the overlap of every detection of a test image with every object of the image, whatever their classes, for
    the matching of the confusion matrix: the intersection over union of their boxes (object detection), or that of
    their masks computed beforehand by the instance segmentation trainer.

    Args:
        detection (Dict[str, torch.Tensor]): The detection 'boxes' [D, 4] of the image, and for the masks its
            'mask_overlaps' [D, G].
        ground_truth (Dict[str, torch.Tensor]): The ground truth 'boxes' [G, 4] of the image.
        overlap_type (str): "boxes" or "masks".

    Returns:
        torch.Tensor: The [D, G] float64 overlaps.

    Raises:
        ValueError: If the overlap type is not one of OVERLAP_TYPES.
    """
    if overlap_type not in OVERLAP_TYPES:
        raise ValueError(f"Unknown overlap type {overlap_type!r}, expected one of {OVERLAP_TYPES}.")
    if overlap_type == "masks":
        return detection["mask_overlaps"].to(dtype=torch.float64)
    return get_intersection_over_union(boxes_1=detection["boxes"].to(dtype=torch.float64),
                                       boxes_2=ground_truth["boxes"].to(dtype=torch.float64))


def match_image_detections(overlaps: torch.Tensor, detection_scores: torch.Tensor, detection_labels: torch.Tensor,
                           ground_truth_labels: torch.Tensor, intersection_over_union_threshold: float) -> torch.Tensor:
    """
    Matches the detections of a test image, those the deployed detector returns, to the objects of the image, one
    object per detection at most, for the confusion matrix and the error images. Two passes go through the detections
    from the most to the least confident, each detection taking the free object that overlaps it the most, at least by
    the threshold: the first pass among the objects of its class, so that an object found with its class counts as
    found even when a detection of another class overlaps it more, the second among the objects left of any class, the
    confusions. A detection left without object is a false positive, an object left without detection is missed.

    Args:
        overlaps (torch.Tensor): The [D, G] overlaps of the detections with the objects (compute_overlap_matrix).
        detection_scores (torch.Tensor): The [D] scores of the detections.
        detection_labels (torch.Tensor): The [D] class indices of the detections.
        ground_truth_labels (torch.Tensor): The [G] class indices of the objects.
        intersection_over_union_threshold (float): The minimum overlap for a detection to find an object.

    Returns:
        torch.Tensor: The [D] int64 index of the object found by every detection, -1 for a false positive.
    """
    matched_ground_truth_indices = torch.full(size=(overlaps.shape[0],), fill_value=-1, dtype=torch.int64)
    taken_ground_truths = torch.zeros(size=(overlaps.shape[1],), dtype=torch.bool)
    detection_order = torch.argsort(input=detection_scores, descending=True, stable=True).tolist()

    # Objects of the class of the detection first, then the objects left of any class
    for same_class_only in [True, False]:
        for detection_index in detection_order:
            if matched_ground_truth_indices[detection_index] >= 0:
                continue
            candidate_objects = (overlaps[detection_index] >= intersection_over_union_threshold) & ~taken_ground_truths
            if same_class_only:
                candidate_objects &= ground_truth_labels == detection_labels[detection_index]
            if not bool(candidate_objects.any()):
                continue

            # The candidate overlapping the detection the most
            candidate_overlaps = torch.where(candidate_objects, overlaps[detection_index], -1.0)
            ground_truth_index = int(candidate_overlaps.argmax())
            matched_ground_truth_indices[detection_index] = ground_truth_index
            taken_ground_truths[ground_truth_index] = True
    return matched_ground_truth_indices


def gather_detection_overlaps(overlaps: Dict[str, torch.Tensor], matched_ground_truth_indices: torch.Tensor,
                              matching_overlap_type: str) -> Dict[str, torch.Tensor]:
    """
    Gives every detection of a test image its overlap with the object the matching gave it (match_image_detections),
    for its caption on the sample and error images of the evaluation, by every overlap type given (the boxes, and the
    masks for the instance segmentation): a matched detection reads its overlaps with its object, a detection matched
    to no object those with the object it overlaps the most by the matching overlap type, so that a duplicate or a
    loose box shows how close it came, and every detection of an image without any object gets NaN.

    Args:
        overlaps (Dict[str, torch.Tensor]): The [D, G] overlaps of the detections with the objects, keyed by overlap
            type ("boxes", "masks").
        matched_ground_truth_indices (torch.Tensor): The [D] object found by every detection, -1 for none.
        matching_overlap_type (str): The overlap type of the matching, which picks the object of an unmatched
            detection.

    Returns:
        Dict[str, torch.Tensor]: The [D] float64 overlap of every detection with its object, keyed like the overlaps.
    """
    number_detections, number_objects = overlaps[matching_overlap_type].shape
    if number_objects == 0:
        return {overlap_type: torch.full(size=(number_detections,), fill_value=float("nan"), dtype=torch.float64)
                for overlap_type in overlaps}

    # Object of every detection: the matched one, or the one overlapping it the most by the matching overlap type
    object_indices = torch.where(matched_ground_truth_indices >= 0, matched_ground_truth_indices,
                                 overlaps[matching_overlap_type].argmax(dim=1))
    return {overlap_type: overlap_matrix.to(dtype=torch.float64).gather(
        dim=1, index=object_indices.unsqueeze(dim=1).to(dtype=torch.int64)).squeeze(dim=1)
        for overlap_type, overlap_matrix in overlaps.items()}


def add_to_confusion_matrix(confusion_matrix: torch.Tensor, matched_ground_truth_indices: torch.Tensor,
                            detection_labels: torch.Tensor, ground_truth_labels: torch.Tensor) -> None:
    """
    Counts the matching of a test image in the confusion matrix of the evaluation, rows the true classes and columns
    the predicted classes: a detection that found an object in the cell of the class of the object and of its own
    class (the diagonal when both agree), a false positive in the background row, and a missed object in the
    background column.

    Args:
        confusion_matrix (torch.Tensor): The [K, K] int64 confusion matrix, background included, filled in place.
        matched_ground_truth_indices (torch.Tensor): The [D] object found by every detection, -1 for none
            (match_image_detections).
        detection_labels (torch.Tensor): The [D] class indices of the detections.
        ground_truth_labels (torch.Tensor): The [G] class indices of the objects.
    """
    is_matched = matched_ground_truth_indices >= 0

    # Every detection in the row of the class of its object, the background row for a false positive
    true_rows = torch.full(size=detection_labels.shape, fill_value=BACKGROUND_CLASS_INDEX, dtype=torch.int64)
    true_rows[is_matched] = ground_truth_labels[matched_ground_truth_indices[is_matched]].to(dtype=torch.int64)
    confusion_matrix.index_put_(indices=(true_rows, detection_labels.to(dtype=torch.int64)),
                                values=torch.ones(size=true_rows.shape, dtype=torch.int64), accumulate=True)

    # Every object found by no detection in the background column
    is_missed = torch.ones(size=ground_truth_labels.shape, dtype=torch.bool)
    is_missed[matched_ground_truth_indices[is_matched]] = False
    missed_labels = ground_truth_labels[is_missed].to(dtype=torch.int64)
    confusion_matrix.index_put_(indices=(missed_labels, torch.full(size=missed_labels.shape,
                                                                   fill_value=BACKGROUND_CLASS_INDEX)),
                                values=torch.ones(size=missed_labels.shape, dtype=torch.int64), accumulate=True)


def group_image_errors(matched_ground_truth_indices: torch.Tensor, detection_scores: torch.Tensor,
                       detection_labels: torch.Tensor, ground_truth_labels: torch.Tensor,
                       class_names: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
    """
    Groups the errors of a test image by the folder of their error image: 'FALSE-POSITIVE/<class>' for the detections
    of a class finding no object, 'MISSED/<class>' for the objects of a class found by no detection, and
    '<TRUE>-TO-<PREDICTED>' for the objects found with another class, so that the image is saved once in every folder
    of its errors, with the boxes of these errors outlined and the errors described in its caption.

    Args:
        matched_ground_truth_indices (torch.Tensor): The [D] object found by every detection, -1 for none
            (match_image_detections).
        detection_scores (torch.Tensor): The [D] scores of the detections.
        detection_labels (torch.Tensor): The [D] class indices of the detections.
        ground_truth_labels (torch.Tensor): The [G] class indices of the objects.
        class_names (Dict[int, str]): The name of every class index.

    Returns:
        Dict[str, Dict[str, Any]]: For every folder of the errors of the image, in the order false positives, missed
            objects, confusions: the 'detection_indices' and 'ground_truth_indices' of the boxes in error, the highest
            'score' of its detections (None for the missed objects) and the 'description' of the errors, e.g.
            'FALSE-POSITIVE circle x2'.
    """
    false_positive_errors, missed_errors, confusion_errors = {}, {}, {}

    # Detections finding no object, and detections finding an object of another class
    for detection_index, ground_truth_index in enumerate(matched_ground_truth_indices.tolist()):
        predicted_name = class_names[int(detection_labels[detection_index])]
        if ground_truth_index < 0:
            image_error = false_positive_errors.setdefault(f"{FALSE_POSITIVE_FOLDER}/{predicted_name}", {
                "detection_indices": [], "ground_truth_indices": [], "score": 0.0,
                "description": f"{FALSE_POSITIVE_FOLDER} {predicted_name}"})
        elif int(ground_truth_labels[ground_truth_index]) != int(detection_labels[detection_index]):
            true_name = class_names[int(ground_truth_labels[ground_truth_index])]
            image_error = confusion_errors.setdefault(f"{true_name}-TO-{predicted_name}", {
                "detection_indices": [], "ground_truth_indices": [], "score": 0.0,
                "description": f"{true_name} -> {predicted_name}"})
            image_error["ground_truth_indices"].append(ground_truth_index)
        else:
            continue
        image_error["detection_indices"].append(detection_index)
        image_error["score"] = max(image_error["score"], float(detection_scores[detection_index]))

    # Objects found by no detection
    found_ground_truths = set(matched_ground_truth_indices[matched_ground_truth_indices >= 0].tolist())
    for ground_truth_index, ground_truth_label in enumerate(ground_truth_labels.tolist()):
        if ground_truth_index in found_ground_truths:
            continue
        true_name = class_names[int(ground_truth_label)]
        image_error = missed_errors.setdefault(f"{MISSED_FOLDER}/{true_name}", {
            "detection_indices": [], "ground_truth_indices": [], "score": None,
            "description": f"{MISSED_FOLDER} {true_name}"})
        image_error["ground_truth_indices"].append(ground_truth_index)

    # Number of errors of every folder in its description when there are several
    image_errors = {**false_positive_errors, **missed_errors, **confusion_errors}
    for image_error in image_errors.values():
        number_errors = max(len(image_error["detection_indices"]), len(image_error["ground_truth_indices"]))
        if number_errors > 1:
            image_error["description"] = f"{image_error['description']} x{number_errors}"
    return image_errors


def summarise_confusion_matrix(confusion_matrix: torch.Tensor) -> Dict[str, int]:
    """
    Counts the outcomes of the detections of the test split from its confusion matrix, for the note of the report:
    the objects found with their class, those found with another class, the false positives and the missed objects.

    Args:
        confusion_matrix (torch.Tensor): The [K, K] confusion matrix, background included (add_to_confusion_matrix).

    Returns:
        Dict[str, int]: The 'right', 'wrong_class', 'false_positives' and 'missed' counts.
    """
    # Detections that found an object, in the rows and columns of the classes after the background (index 0)
    object_cells = confusion_matrix[1:, 1:]
    right = int(torch.diagonal(input=object_cells).sum())
    return {"right": right, "wrong_class": int(object_cells.sum()) - right,
            "false_positives": int(confusion_matrix[BACKGROUND_CLASS_INDEX].sum()),
            "missed": int(confusion_matrix[:, BACKGROUND_CLASS_INDEX].sum())}

