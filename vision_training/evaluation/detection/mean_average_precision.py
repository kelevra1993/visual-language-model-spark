"""
Mean average precision of the object detection and instance segmentation tasks: matching of the detections to the
ground truths by their boxes or masks, precision-recall curves and average precision (area, 11 or 101 points) over the
IoU thresholds, and the score threshold of every class reaching a target precision at IoU 0.5. Used by the detection
and instance segmentation trainers to evaluate a checkpoint on the test split.
"""

import torch

from typing import Any, Dict, List, Optional, Tuple

from vision_training.evaluation.classification.threshold_finder import find_score_threshold
from vision_training.models.detection.box_operations import get_intersection_over_union

# Methods computing the average precision from a precision-recall curve
AVERAGE_PRECISION_METHODS = ["area", "11_point", "101_point"]

# Intersection over union thresholds of the COCO-style mAP@[0.5:0.95], from 0.50 to 0.95 in steps of 0.05
INTERSECTION_OVER_UNION_THRESHOLDS = [round(0.5 + 0.05 * index, 2) for index in range(10)]

# Overlaps comparing a detection with a ground truth: their boxes, or their masks (instance segmentation)
OVERLAP_TYPES = ["boxes", "masks"]

# Intersection over union threshold of the score threshold of every class, the one of the headline mAP@0.5
CLASS_THRESHOLD_INTERSECTION_OVER_UNION = 0.5


def find_best_ground_truth(overlap_matrices: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds, for every detection of a class, the ground truth of the same class and the same image that overlaps it the
    most, from the overlap matrix of every image (box or mask intersection over union). This best match does not
    depend on the intersection over union threshold nor on which ground truths are already matched, so it is computed
    once per class and reused by the matching at every threshold.

    Args:
        overlap_matrices (List[torch.Tensor]): The [D_i, G_i] intersection over union of the detections and ground
            truths of the class in every image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The intersection over union with the best ground truth (0 when the image
            has none) and the index of that ground truth within the ground truths of the class in its image (-1 when
            the image has none), for the detections of every image one after the other.
    """
    best_intersection_over_union, best_ground_truth_indices = [], []

    # Best ground truth of every detection within its own image
    for overlap_matrix in overlap_matrices:
        overlap_matrix = overlap_matrix.to(dtype=torch.float64)
        if overlap_matrix.shape[1] == 0:
            best_intersection_over_union.append(torch.zeros(size=(overlap_matrix.shape[0],), dtype=torch.float64))
            best_ground_truth_indices.append(torch.full(size=(overlap_matrix.shape[0],), fill_value=-1,
                                                        dtype=torch.int64))
            continue
        image_best_overlaps, image_best_indices = overlap_matrix.max(dim=1)
        best_intersection_over_union.append(image_best_overlaps)
        best_ground_truth_indices.append(image_best_indices)

    # Keep the [0] layout without images
    if len(overlap_matrices) == 0:
        return torch.zeros(size=(0,), dtype=torch.float64), torch.zeros(size=(0,), dtype=torch.int64)
    return torch.cat(tensors=best_intersection_over_union), torch.cat(tensors=best_ground_truth_indices)


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
                            class_label: int, overlap_type: str = "boxes") -> Dict[str, Any]:
    """
    Gathers the detections of one class over all images, sorted by decreasing score, with their best matching ground
    truth, and counts the ground truths of that class: everything the average precision of the class needs,
    independently of the intersection over union threshold. The detections and ground truths are compared by their
    boxes (object detection, and the box AP of the instance segmentation) or by their masks (mask AP of the instance
    segmentation), whose overlaps are computed beforehand by the trainer.

    Args:
        detections (List[Dict[str, torch.Tensor]]): For every image, its detection 'boxes' [N, 4], 'scores' [N] and
            'labels' [N], and for the masks the 'mask_overlaps' [N, M], the mask intersection over union of every
            detection with every ground truth of the image.
        ground_truths (List[Dict[str, torch.Tensor]]): For every image, its ground truth 'boxes' [M, 4] and
            'labels' [M].
        class_label (int): The class index.
        overlap_type (str): "boxes" or "masks". Defaults to "boxes".

    Returns:
        Dict[str, Any]: The sorted 'scores', 'image_indices', 'best_intersection_over_union' and
            'best_ground_truth_indices' of the detections, and the 'number_ground_truth' of the class.

    Raises:
        ValueError: If the overlap type is not one of OVERLAP_TYPES.
    """
    if overlap_type not in OVERLAP_TYPES:
        raise ValueError(f"Unknown overlap type {overlap_type!r}, expected one of {OVERLAP_TYPES}.")

    # Overlap matrix of the detections and ground truths of the class in every image, with their scores and images
    image_indices, scores, overlap_matrices = [], [], []
    number_ground_truth = 0
    for image_index, (detection, ground_truth) in enumerate(zip(detections, ground_truths, strict=True)):
        class_mask = detection["labels"] == class_label
        ground_truth_mask = ground_truth["labels"] == class_label
        number_ground_truth += int(ground_truth_mask.sum())
        if overlap_type == "boxes":
            overlap_matrices.append(get_intersection_over_union(
                boxes_1=detection["boxes"][class_mask].to(dtype=torch.float64),
                boxes_2=ground_truth["boxes"][ground_truth_mask].to(dtype=torch.float64)))
        else:
            overlap_matrices.append(detection["mask_overlaps"][class_mask][:, ground_truth_mask])
        scores.append(detection["scores"][class_mask].to(dtype=torch.float64))
        image_indices.append(torch.full(size=(int(class_mask.sum()),), fill_value=image_index, dtype=torch.int64))

    # Best ground truth of every detection, then the detections sorted by decreasing score, the order in which they
    # take their ground truth
    best_intersection_over_union, best_ground_truth_indices = find_best_ground_truth(overlap_matrices=overlap_matrices)
    scores, image_indices = torch.cat(tensors=scores), torch.cat(tensors=image_indices)
    sorted_indices = torch.argsort(input=scores, descending=True, stable=True)
    return {"scores": scores[sorted_indices], "image_indices": image_indices[sorted_indices],
            "best_intersection_over_union": best_intersection_over_union[sorted_indices],
            "best_ground_truth_indices": best_ground_truth_indices[sorted_indices],
            "number_ground_truth": number_ground_truth}


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


def find_class_score_thresholds(class_detections: Dict[int, Dict[str, Any]],
                                target_precision: float) -> Dict[int, Optional[Tuple[float, float]]]:
    """
    Finds, for every class, the detection score from which its detections reach a target precision, saved by the
    detection trainers in class_thresholds.json so that a deployment can keep the detections class by class instead of
    with the single 'Detector.inference.score_threshold'. A detection of the class is right when it finds an object of
    the class at an intersection over union of 0.5, the matching of the AP@0.5 (the most confident detection takes the
    object), and the recall counts every object of the class, including those found by no detection. Only the
    detections kept for the evaluation ('evaluation.score_threshold') can give a threshold.

    Args:
        class_detections (Dict[int, Dict[str, Any]]): For every class index, its gathered detections, from
            gather_class_detections.
        target_precision (float): The precision the threshold must reach, e.g. 0.99.

    Returns:
        Dict[int, Optional[Tuple[float, float]]]: For every class index, its threshold and the recall it keeps, or None
            if no threshold reaches the target precision (or the class has no ground truth or no detection).
    """
    class_thresholds = {}
    for class_label, gathered in class_detections.items():
        true_positive_flags = match_detections_to_ground_truth(
            detection_image_indices=gathered["image_indices"],
            best_intersection_over_union=gathered["best_intersection_over_union"],
            best_ground_truth_indices=gathered["best_ground_truth_indices"],
            intersection_over_union_threshold=CLASS_THRESHOLD_INTERSECTION_OVER_UNION)
        class_thresholds[class_label] = find_score_threshold(scores=gathered["scores"], positives=true_positive_flags,
                                                             number_positives=gathered["number_ground_truth"],
                                                             target_precision=target_precision)
    return class_thresholds


def evaluate_detections(detections: List[Dict[str, torch.Tensor]], ground_truths: List[Dict[str, torch.Tensor]],
                        number_classes: int, overlap_type: str = "boxes",
                        target_precision: Optional[float] = None) -> Dict[str, Dict[Any, Any]]:
    """
    Evaluates the detections of a whole test split, as reported by the Trainer at every evaluated checkpoint: with
    every average precision method, the mAP at an intersection over union of 0.5 (PASCAL VOC), at 0.75, and averaged
    over the thresholds 0.5 to 0.95 (COCO), plus the average precision of every class at the same thresholds and the
    number of ground truth boxes of every class, for the per-class report. The detections are compared with the
    ground truths by their boxes, or by their masks for the mask AP of the instance segmentation. With a target
    precision, the score threshold of every class reaching it is added (find_class_score_thresholds), from the same
    gathered detections.

    Args:
        detections (List[Dict[str, torch.Tensor]]): For every image, its detection 'boxes' [N, 4], 'scores' [N] and
            'labels' [N], without padding, and for the masks the 'mask_overlaps' [N, M] with the ground truths.
        ground_truths (List[Dict[str, torch.Tensor]]): For every image, its ground truth 'boxes' [M, 4] and
            'labels' [M], in the same coordinates as the detections.
        number_classes (int): The number of classes, background (index 0) included.
        overlap_type (str): "boxes" or "masks". Defaults to "boxes".
        target_precision (Optional[float]): The precision of the score threshold of every class, None to skip the
            thresholds. Defaults to None.

    Returns:
        Dict[str, Dict[Any, Any]]: For every method, its 'mAP@0.5', 'mAP@0.75' and 'mAP@[0.5:0.95]', the average
            precision of every class with ground truth at 0.5, at 0.75 and averaged over 0.5 to 0.95 ('class_AP@0.5',
            'class_AP@0.75', 'class_AP@[0.5:0.95]'), and the 'class_ground_truths' boxes of every foreground class;
            with a target precision, the 'class_score_thresholds' of every foreground class.
    """
    # Gather the detections and best matches of every foreground class once, for all thresholds and methods
    class_detections = {class_label: gather_class_detections(detections=detections, ground_truths=ground_truths,
                                                             class_label=class_label, overlap_type=overlap_type)
                        for class_label in range(1, number_classes)}

    results = {}
    for method in AVERAGE_PRECISION_METHODS:
        threshold_results = {threshold: compute_mean_average_precision(
            class_detections=class_detections, intersection_over_union_threshold=threshold, method=method)
            for threshold in INTERSECTION_OVER_UNION_THRESHOLDS}

        # The COCO-style mAP is the mean of the mAPs at the ten thresholds, and the COCO-style average precision of a
        # class the mean of its average precisions at the ten thresholds (the same classes have ground truth at every
        # threshold)
        class_labels = list(threshold_results[0.5]["class_average_precisions"])
        results[method] = {"mAP@0.5": threshold_results[0.5]["mean_average_precision"],
                           "mAP@0.75": threshold_results[0.75]["mean_average_precision"],
                           "mAP@[0.5:0.95]": sum(result["mean_average_precision"]
                                                 for result in threshold_results.values()) / len(threshold_results),
                           "class_AP@0.5": threshold_results[0.5]["class_average_precisions"],
                           "class_AP@0.75": threshold_results[0.75]["class_average_precisions"],
                           "class_AP@[0.5:0.95]": {class_label: sum(
                               result["class_average_precisions"][class_label] for result in threshold_results.values())
                               / len(threshold_results) for class_label in class_labels},
                           "class_ground_truths": {class_label: gathered["number_ground_truth"]
                                                   for class_label, gathered in class_detections.items()}}

    # Score threshold of every class reaching the target precision, for the deployment
    if target_precision is not None:
        results["class_score_thresholds"] = find_class_score_thresholds(class_detections=class_detections,
                                                                        target_precision=target_precision)
    return results
