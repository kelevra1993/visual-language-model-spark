"""
Mean average precision of the object detection task: matching of the detections to the ground truth boxes,
precision-recall curves and average precision (area, 11 or 101 points) over the IoU thresholds. Used by the
detection trainer to evaluate a checkpoint on the test split.
"""

import torch

from typing import Any, Dict, List, Tuple

from vision_training.models.detection.box_operations import get_intersection_over_union

# Methods computing the average precision from a precision-recall curve
AVERAGE_PRECISION_METHODS = ["area", "11_point", "101_point"]

# Intersection over union thresholds of the COCO-style mAP@[0.5:0.95], from 0.50 to 0.95 in steps of 0.05
INTERSECTION_OVER_UNION_THRESHOLDS = [round(0.5 + 0.05 * index, 2) for index in range(10)]


def find_best_ground_truth(detection_image_indices: torch.Tensor, detection_boxes: torch.Tensor,
                           ground_truth_boxes_per_image: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds, for every detection of a class, the ground truth box of the same class and the same image that overlaps
    it the most. This best match does not depend on the intersection over union threshold nor on which ground truths
    are already matched, so it is computed once per class and reused by the matching at every threshold.

    Args:
        detection_image_indices (torch.Tensor): The [D] index of the image of every detection.
        detection_boxes (torch.Tensor): The [D, 4] [x1, y1, x2, y2] detection boxes.
        ground_truth_boxes_per_image (List[torch.Tensor]): The [G_i, 4] ground truth boxes of the class in every image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [D] intersection over union with the best ground truth (0 when the image
            has none) and the [D] index of that ground truth within its image (-1 when the image has none).
    """
    best_intersection_over_union = torch.zeros(size=(len(detection_boxes),), dtype=torch.float64)
    best_ground_truth_indices = torch.full(size=(len(detection_boxes),), fill_value=-1, dtype=torch.int64)

    # Compare the detections of every image with the ground truth boxes of that image only
    for image_index in torch.unique(detection_image_indices).tolist():
        ground_truth_boxes = ground_truth_boxes_per_image[image_index]
        if len(ground_truth_boxes) == 0:
            continue
        image_mask = detection_image_indices == image_index
        intersection_over_union = get_intersection_over_union(boxes_1=detection_boxes[image_mask],
                                                              boxes_2=ground_truth_boxes)
        best_intersection_over_union[image_mask], best_ground_truth_indices[image_mask] = intersection_over_union.max(
            dim=1)

    return best_intersection_over_union, best_ground_truth_indices


def match_detections_to_ground_truth(detection_image_indices: torch.Tensor,
                                     best_intersection_over_union: torch.Tensor,
                                     best_ground_truth_indices: torch.Tensor,
                                     intersection_over_union_threshold: float) -> torch.Tensor:
    """
    Marks every detection of a class as a true or false positive, going through the detections from the most to the
    least confident. A detection is a true positive when its best ground truth overlaps it by at least the threshold
    and was not already taken by a more confident detection, otherwise it is a false positive (PASCAL VOC rule:
    a detection whose best ground truth is already matched is a false positive, even if another one overlaps it).

    Args:
        detection_image_indices (torch.Tensor): The [D] image index of every detection, sorted by decreasing score.
        best_intersection_over_union (torch.Tensor): The [D] overlap with the best ground truth, from
            find_best_ground_truth.
        best_ground_truth_indices (torch.Tensor): The [D] index of the best ground truth within its image, -1 if none.
        intersection_over_union_threshold (float): The minimum overlap for a detection to find its object.

    Returns:
        torch.Tensor: The [D] boolean true positive flags, in the same order as the detections.
    """
    true_positive_flags = torch.zeros(size=(len(detection_image_indices),), dtype=torch.bool)
    matched_ground_truths = set()

    # Detections are visited in decreasing score order, so the most confident one takes each ground truth
    candidate_flags = (best_intersection_over_union >= intersection_over_union_threshold).tolist()
    for detection_index, (image_index, ground_truth_index, is_candidate) in enumerate(
            zip(detection_image_indices.tolist(), best_ground_truth_indices.tolist(), candidate_flags)):
        if is_candidate and (image_index, ground_truth_index) not in matched_ground_truths:
            true_positive_flags[detection_index] = True
            matched_ground_truths.add((image_index, ground_truth_index))

    return true_positive_flags


def compute_precision_recall(true_positive_flags: torch.Tensor, number_ground_truth: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
    """
    Builds the precision-recall curve of a class from its true positive flags: every detection, taken in decreasing
    score order, is one point of the curve, the precision and recall obtained if the score threshold was its score.

    Args:
        true_positive_flags (torch.Tensor): The [D] boolean true positive flags, sorted by decreasing score.
        number_ground_truth (int): The number of ground truth boxes of the class, strictly positive.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [D] float64 precisions and recalls.
    """
    cumulative_true_positives = torch.cumsum(input=true_positive_flags.to(dtype=torch.float64), dim=0)
    cumulative_false_positives = torch.cumsum(input=(~true_positive_flags).to(dtype=torch.float64), dim=0)

    precisions = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
    recalls = cumulative_true_positives / number_ground_truth
    return precisions, recalls


def compute_average_precision(precisions: torch.Tensor, recalls: torch.Tensor, method: str) -> float:
    """
    Summarises the precision-recall curve of a class into its average precision, with one of the three usual
    definitions: the area under the curve ('area', PASCAL VOC 2010 and later), or the mean of the interpolated
    precision at 11 ('11_point', PASCAL VOC 2007) or 101 ('101_point', COCO) equally spaced recall levels.

    Args:
        precisions (torch.Tensor): The [D] precisions, from compute_precision_recall.
        recalls (torch.Tensor): The [D] non-decreasing recalls, from compute_precision_recall.
        method (str): 'area', '11_point' or '101_point'.

    Returns:
        float: The average precision, within [0, 1], 0 when there is no detection.

    Raises:
        ValueError: If the method is not one of AVERAGE_PRECISION_METHODS.
    """
    if method not in AVERAGE_PRECISION_METHODS:
        raise ValueError(f"Unknown average precision method {method!r}, expected one of {AVERAGE_PRECISION_METHODS}.")
    if len(precisions) == 0:
        return 0.0

    if method == "area":
        # Add the boundary points of the curve, at recall 0 and recall 1, both with precision 0
        extended_recalls = torch.cat(tensors=[torch.tensor([0.0], dtype=torch.float64), recalls,
                                              torch.tensor([1.0], dtype=torch.float64)])
        extended_precisions = torch.cat(tensors=[torch.tensor([0.0], dtype=torch.float64), precisions,
                                                 torch.tensor([0.0], dtype=torch.float64)])

        # Remove the zigzag of the curve, every precision becomes the maximum precision on its right
        extended_precisions = torch.flip(input=torch.cummax(input=torch.flip(input=extended_precisions, dims=[0]),
                                                            dim=0).values, dims=[0])

        # Sum the rectangles found where the recall changes, of width the recall step and height the precision
        change_indices = torch.nonzero(input=extended_recalls[1:] != extended_recalls[:-1]).flatten()
        return float(torch.sum((extended_recalls[change_indices + 1] - extended_recalls[change_indices]) *
                               extended_precisions[change_indices + 1]))

    # Equally spaced recall levels, computed as exact fractions so that a level such as 0.8 equals a recall of 4 / 5
    number_points = 11 if method == "11_point" else 101
    recall_levels = torch.arange(end=number_points, dtype=torch.float64) / (number_points - 1)

    # The interpolated precision of a level is the maximum precision at a recall greater than or equal to it, and 0
    # beyond the maximum recall reached
    interpolated_precisions = [float(precisions[recalls >= recall_level].max()) if bool(
        (recalls >= recall_level).any()) else 0.0 for recall_level in recall_levels]
    return sum(interpolated_precisions) / number_points


def gather_class_detections(detections: List[Dict[str, torch.Tensor]], ground_truths: List[Dict[str, torch.Tensor]],
                            class_label: int) -> Dict[str, Any]:
    """
    Gathers the detections of one class over all images, sorted by decreasing score, with their best matching ground
    truth, and counts the ground truth boxes of that class: everything the average precision of the class needs,
    independently of the intersection over union threshold.

    Args:
        detections (List[Dict[str, torch.Tensor]]): For every image, its detection 'boxes' [N, 4], 'scores' [N] and
            'labels' [N].
        ground_truths (List[Dict[str, torch.Tensor]]): For every image, its ground truth 'boxes' [M, 4] and
            'labels' [M].
        class_label (int): The class index.

    Returns:
        Dict[str, Any]: The sorted 'image_indices', 'best_intersection_over_union' and 'best_ground_truth_indices' of
            the detections, and the 'number_ground_truth' of the class.
    """
    ground_truth_boxes_per_image = [ground_truth["boxes"][ground_truth["labels"] == class_label].to(dtype=torch.float64)
                                    for ground_truth in ground_truths]

    # Concatenate the detections of the class of every image, remembering the image each one comes from
    image_indices, boxes, scores = [], [], []
    for image_index, detection in enumerate(detections):
        class_mask = detection["labels"] == class_label
        boxes.append(detection["boxes"][class_mask].to(dtype=torch.float64))
        scores.append(detection["scores"][class_mask].to(dtype=torch.float64))
        image_indices.append(torch.full(size=(int(class_mask.sum()),), fill_value=image_index, dtype=torch.int64))
    boxes, scores, image_indices = torch.cat(tensors=boxes), torch.cat(tensors=scores), torch.cat(tensors=image_indices)

    # Sort the detections by decreasing score, the order in which they take their ground truth
    sorted_indices = torch.argsort(input=scores, descending=True, stable=True)
    image_indices, boxes = image_indices[sorted_indices], boxes[sorted_indices]

    best_intersection_over_union, best_ground_truth_indices = find_best_ground_truth(
        detection_image_indices=image_indices, detection_boxes=boxes,
        ground_truth_boxes_per_image=ground_truth_boxes_per_image)
    return {"image_indices": image_indices, "best_intersection_over_union": best_intersection_over_union,
            "best_ground_truth_indices": best_ground_truth_indices,
            "number_ground_truth": sum(len(boxes) for boxes in ground_truth_boxes_per_image)}


def compute_mean_average_precision(class_detections: Dict[int, Dict[str, Any]],
                                   intersection_over_union_threshold: float, method: str) -> Dict[str, Any]:
    """
    Computes the average precision of every class at one intersection over union threshold and their mean, the mAP.
    Classes without any ground truth box are left out of the mean, since their recall is undefined.

    Args:
        class_detections (Dict[int, Dict[str, Any]]): For every class index, its gathered detections, from
            gather_class_detections.
        intersection_over_union_threshold (float): The minimum overlap for a detection to find its object.
        method (str): 'area', '11_point' or '101_point'.

    Returns:
        Dict[str, Any]: The 'mean_average_precision' (0 when no class has ground truth) and the
            'class_average_precisions' of every class with ground truth.
    """
    class_average_precisions = {}
    for class_label, gathered in class_detections.items():
        if gathered["number_ground_truth"] == 0:
            continue
        true_positive_flags = match_detections_to_ground_truth(
            detection_image_indices=gathered["image_indices"],
            best_intersection_over_union=gathered["best_intersection_over_union"],
            best_ground_truth_indices=gathered["best_ground_truth_indices"],
            intersection_over_union_threshold=intersection_over_union_threshold)
        precisions, recalls = compute_precision_recall(true_positive_flags=true_positive_flags,
                                                       number_ground_truth=gathered["number_ground_truth"])
        class_average_precisions[class_label] = compute_average_precision(precisions=precisions, recalls=recalls,
                                                                          method=method)

    mean_average_precision = (sum(class_average_precisions.values()) / len(class_average_precisions)
                              if class_average_precisions else 0.0)
    return {"mean_average_precision": mean_average_precision, "class_average_precisions": class_average_precisions}


def evaluate_detections(detections: List[Dict[str, torch.Tensor]], ground_truths: List[Dict[str, torch.Tensor]],
                        number_classes: int) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates the detections of a whole test split, as reported by the Trainer at every evaluated checkpoint: with
    every average precision method, the mAP at an intersection over union of 0.5 (PASCAL VOC), at 0.75, and averaged
    over the thresholds 0.5 to 0.95 (COCO), plus the average precision of every class at 0.5.

    Args:
        detections (List[Dict[str, torch.Tensor]]): For every image, its detection 'boxes' [N, 4], 'scores' [N] and
            'labels' [N], without padding.
        ground_truths (List[Dict[str, torch.Tensor]]): For every image, its ground truth 'boxes' [M, 4] and
            'labels' [M], in the same coordinates as the detections.
        number_classes (int): The number of classes, background (index 0) included.

    Returns:
        Dict[str, Dict[str, Any]]: For every method, its 'mAP@0.5', 'mAP@0.75', 'mAP@[0.5:0.95]' and 'class_AP@0.5'
            (the average precision of every class with ground truth, at 0.5).
    """
    # Gather the detections and best matches of every foreground class once, for all thresholds and methods
    class_detections = {class_label: gather_class_detections(detections=detections, ground_truths=ground_truths,
                                                             class_label=class_label)
                        for class_label in range(1, number_classes)}

    results = {}
    for method in AVERAGE_PRECISION_METHODS:
        threshold_results = {threshold: compute_mean_average_precision(
            class_detections=class_detections, intersection_over_union_threshold=threshold, method=method)
            for threshold in INTERSECTION_OVER_UNION_THRESHOLDS}

        # The COCO-style mAP is the mean of the mAPs at the ten thresholds
        results[method] = {"mAP@0.5": threshold_results[0.5]["mean_average_precision"],
                           "mAP@0.75": threshold_results[0.75]["mean_average_precision"],
                           "mAP@[0.5:0.95]": sum(result["mean_average_precision"]
                                                 for result in threshold_results.values()) / len(threshold_results),
                           "class_AP@0.5": threshold_results[0.5]["class_average_precisions"]}
    return results
