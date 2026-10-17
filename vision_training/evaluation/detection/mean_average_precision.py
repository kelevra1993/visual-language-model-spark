"""
Mean average precision of the object detection and instance segmentation tasks: matching of the detections to the
ground truths by their boxes or masks, precision-recall curves and average precision (area, 11 or 101 points) over the
IoU thresholds, for all the objects and for the small, medium and large ones (object_sizes.py), the recall of all the
detections at the same thresholds (mean average recall), and the score threshold of every class reaching a target
precision at IoU 0.5. Used by the detection and instance segmentation trainers to
evaluate a checkpoint on the test split, whose report is built by average_precision_report.py.
"""

import torch

from typing import Any, Dict, List, Optional, Tuple

from vision_training.evaluation.classification.threshold_finder import find_score_threshold
from vision_training.evaluation.detection.object_sizes import OBJECT_SIZE_RANGES, compute_box_areas, is_in_object_size
from vision_training.models.detection.box_operations import get_intersection_over_union

# Methods computing the average precision from a precision-recall curve
AVERAGE_PRECISION_METHODS = ["area", "11_point", "101_point"]

# Intersection over union thresholds of the COCO-style mAP@[0.5:0.95], from 0.50 to 0.95 in steps of 0.05
INTERSECTION_OVER_UNION_THRESHOLDS = [round(0.5 + 0.05 * index, 2) for index in range(10)]

# Overlaps comparing a detection with a ground truth: their boxes, or their masks (instance segmentation)
OVERLAP_TYPES = ["boxes", "masks"]

# Intersection over union threshold of the score threshold of every class, the one of the headline mAP@0.5
CLASS_THRESHOLD_INTERSECTION_OVER_UNION = 0.5

# Metrics summarised over the thresholds: the key of their mean and of their class values in the results of
# compute_mean_average_precision, and their name in the summary (mAP and class_AP, mAR and class_AR)
SUMMARISED_METRICS = [("mean_average_precision", "class_average_precisions", "AP"),
                      ("mean_recall", "class_recalls", "AR")]


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
    segmentation), whose overlaps are computed beforehand by the trainer. The areas of the detections, of their best
    ground truth and of every ground truth (box areas, or mask areas for the masks) are kept for the average precision
    by object size.

    Args:
        detections (List[Dict[str, torch.Tensor]]): For every image, its detection 'boxes' [N, 4], 'scores' [N] and
            'labels' [N], and for the masks the 'mask_overlaps' [N, M], the mask intersection over union of every
            detection with every ground truth of the image, and the 'mask_areas' [N] of the detection masks.
        ground_truths (List[Dict[str, torch.Tensor]]): For every image, its ground truth 'boxes' [M, 4] and
            'labels' [M], and for the masks the 'mask_areas' [M] of the ground truth masks.
        class_label (int): The class index.
        overlap_type (str): "boxes" or "masks". Defaults to "boxes".

    Returns:
        Dict[str, Any]: The sorted 'scores', 'image_indices', 'best_intersection_over_union',
            'best_ground_truth_indices', 'detection_areas' and 'best_ground_truth_areas' (-1 without ground truth in the
            image) of the detections, the 'ground_truth_areas' and the 'number_ground_truth' of the class.

    Raises:
        ValueError: If the overlap type is not one of OVERLAP_TYPES.
    """
    if overlap_type not in OVERLAP_TYPES:
        raise ValueError(f"Unknown overlap type {overlap_type!r}, expected one of {OVERLAP_TYPES}.")

    # Overlap matrix of the detections and ground truths of the class in every image, with their scores, images and
    # areas
    image_indices, scores, overlap_matrices = [], [], []
    detection_areas, ground_truth_areas = [], []
    for image_index, (detection, ground_truth) in enumerate(zip(detections, ground_truths, strict=True)):
        class_mask = detection["labels"] == class_label
        ground_truth_mask = ground_truth["labels"] == class_label
        if overlap_type == "boxes":
            overlap_matrices.append(get_intersection_over_union(
                boxes_1=detection["boxes"][class_mask].to(dtype=torch.float64),
                boxes_2=ground_truth["boxes"][ground_truth_mask].to(dtype=torch.float64)))
            detection_areas.append(compute_box_areas(boxes=detection["boxes"][class_mask].to(dtype=torch.float64)))
            ground_truth_areas.append(compute_box_areas(
                boxes=ground_truth["boxes"][ground_truth_mask].to(dtype=torch.float64)))
        else:
            overlap_matrices.append(detection["mask_overlaps"][class_mask][:, ground_truth_mask])
            detection_areas.append(detection["mask_areas"][class_mask].to(dtype=torch.float64))
            ground_truth_areas.append(ground_truth["mask_areas"][ground_truth_mask].to(dtype=torch.float64))
        scores.append(detection["scores"][class_mask].to(dtype=torch.float64))
        image_indices.append(torch.full(size=(int(class_mask.sum()),), fill_value=image_index, dtype=torch.int64))

    # Best ground truth of every detection, then the detections sorted by decreasing score, the order in which they
    # take their ground truth
    best_intersection_over_union, best_ground_truth_indices = find_best_ground_truth(overlap_matrices=overlap_matrices)
    scores, image_indices = torch.cat(tensors=scores), torch.cat(tensors=image_indices)
    sorted_indices = torch.argsort(input=scores, descending=True, stable=True)

    # Area of the best ground truth of every detection, found among the ground truths of the class of its image, -1
    # for a detection in an image without ground truth of the class
    ground_truth_offsets = torch.tensor(data=[0] + [len(areas) for areas in ground_truth_areas],
                                        dtype=torch.int64).cumsum(dim=0)[:-1]
    all_ground_truth_areas = torch.cat(tensors=ground_truth_areas + [torch.zeros(size=(1,), dtype=torch.float64)])
    best_ground_truth_positions = (ground_truth_offsets[image_indices] + best_ground_truth_indices).clamp(min=0)
    best_ground_truth_areas = torch.where(best_ground_truth_indices >= 0, all_ground_truth_areas[
        best_ground_truth_positions], torch.full_like(input=best_intersection_over_union, fill_value=-1.0))
    return {"scores": scores[sorted_indices], "image_indices": image_indices[sorted_indices],
            "best_intersection_over_union": best_intersection_over_union[sorted_indices],
            "best_ground_truth_indices": best_ground_truth_indices[sorted_indices],
            "detection_areas": torch.cat(tensors=detection_areas)[sorted_indices],
            "best_ground_truth_areas": best_ground_truth_areas[sorted_indices],
            "ground_truth_areas": all_ground_truth_areas[:-1], "number_ground_truth": len(all_ground_truth_areas) - 1}


def compute_class_true_positive_flags(class_detections: Dict[int, Dict[str, Any]],
                                      intersection_over_union_threshold: float) -> Dict[int, torch.Tensor]:
    """
    Marks the detections of every class as true or false positives at one intersection over union threshold, once for
    the three average precision methods, the object sizes and the score thresholds, since the matching depends on the
    threshold only.

    Args:
        class_detections (Dict[int, Dict[str, Any]]): For every class index, its gathered detections, from
            gather_class_detections.
        intersection_over_union_threshold (float): The minimum overlap for a detection to find its object.

    Returns:
        Dict[int, torch.Tensor]: The [D] boolean true positive flags of the detections of every class, in decreasing
            score order.
    """
    return {class_label: match_detections_to_ground_truth(
        detection_image_indices=gathered["image_indices"],
        best_intersection_over_union=gathered["best_intersection_over_union"],
        best_ground_truth_indices=gathered["best_ground_truth_indices"],
        intersection_over_union_threshold=intersection_over_union_threshold)
        for class_label, gathered in class_detections.items()}


def select_object_size_detections(gathered: Dict[str, Any], true_positive_flags: torch.Tensor,
                                  intersection_over_union_threshold: float,
                                  object_size: str) -> Tuple[torch.Tensor, int]:
    """
    Keeps the detections of a class that count for the average precision of one object size, with the COCO rule: the
    ground truths of the other sizes are ignored, and so are the detections matched to them (their best ground truth
    overlapping them by at least the threshold) and the unmatched detections whose own area is of another size. A
    detection whose best ground truth of the size was already taken stays a false positive, as without sizes.

    Args:
        gathered (Dict[str, Any]): The gathered detections of the class, from gather_class_detections.
        true_positive_flags (torch.Tensor): The [D] true positive flags of the detections at the threshold.
        intersection_over_union_threshold (float): The minimum overlap for a detection to find its object.
        object_size (str): "small", "medium" or "large".

    Returns:
        Tuple[torch.Tensor, int]: The true positive flags of the kept detections, in decreasing score order, and the
            number of ground truths of the class of that size.
    """
    # A detection reaching the threshold with its best ground truth counts when that ground truth is of the size, any
    # other one when its own area is of the size
    matched = gathered["best_intersection_over_union"] >= intersection_over_union_threshold
    kept_detections = torch.where(matched, is_in_object_size(areas=gathered["best_ground_truth_areas"],
                                                             object_size=object_size),
                                  is_in_object_size(areas=gathered["detection_areas"], object_size=object_size))
    number_ground_truth = int(is_in_object_size(areas=gathered["ground_truth_areas"], object_size=object_size).sum())
    return true_positive_flags[kept_detections], number_ground_truth


def compute_mean_average_precision(class_detections: Dict[int, Dict[str, Any]],
                                   class_true_positive_flags: Dict[int, torch.Tensor],
                                   intersection_over_union_threshold: float, method: str,
                                   object_size: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes the average precision of every class at one intersection over union threshold and their mean, the mAP,
    for all the objects or for the objects of one size, with the recall of all the detections of every class and their
    mean, the mAR (the objects found by any detection kept for the evaluation, whatever its score). Classes without any
    ground truth box (of the size) are left out of the means, since their recall is undefined.

    Args:
        class_detections (Dict[int, Dict[str, Any]]): For every class index, its gathered detections, from
            gather_class_detections.
        class_true_positive_flags (Dict[int, torch.Tensor]): The true positive flags of the detections of every class
            at the threshold, from compute_class_true_positive_flags.
        intersection_over_union_threshold (float): The minimum overlap for a detection to find its object.
        method (str): 'area', '11_point' or '101_point'.
        object_size (Optional[str]): "small", "medium" or "large" for the objects of one size, None for all of them.
            Defaults to None.

    Returns:
        Dict[str, Any]: The 'mean_average_precision' and 'mean_recall' (0 when no class has ground truth), the
            'class_average_precisions' and 'class_recalls' of every class with ground truth, with the
            'class_ground_truths' counted.
    """
    class_average_precisions, class_recalls, class_ground_truths = {}, {}, {}
    for class_label, gathered in class_detections.items():
        # Detections and ground truths of the class, of one size or all of them
        true_positive_flags, number_ground_truth = class_true_positive_flags[class_label], \
            gathered["number_ground_truth"]
        if object_size is not None:
            true_positive_flags, number_ground_truth = select_object_size_detections(
                gathered=gathered, true_positive_flags=true_positive_flags,
                intersection_over_union_threshold=intersection_over_union_threshold, object_size=object_size)
        class_ground_truths[class_label] = number_ground_truth
        if number_ground_truth == 0:
            continue

        # Precision-recall curve of the class, summarised by the method, and the recall of all its detections
        precisions, recalls = compute_precision_recall(true_positive_flags=true_positive_flags,
                                                       number_ground_truth=number_ground_truth)
        class_average_precisions[class_label] = compute_average_precision(precisions=precisions, recalls=recalls,
                                                                          method=method)
        class_recalls[class_label] = float(true_positive_flags.sum()) / number_ground_truth

    mean_average_precision = (sum(class_average_precisions.values()) / len(class_average_precisions)
                              if class_average_precisions else 0.0)
    mean_recall = sum(class_recalls.values()) / len(class_recalls) if class_recalls else 0.0
    return {"mean_average_precision": mean_average_precision, "class_average_precisions": class_average_precisions,
            "mean_recall": mean_recall, "class_recalls": class_recalls, "class_ground_truths": class_ground_truths}


def find_class_score_thresholds(class_detections: Dict[int, Dict[str, Any]],
                                class_true_positive_flags: Dict[int, torch.Tensor],
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
        class_true_positive_flags (Dict[int, torch.Tensor]): The true positive flags of the detections of every class
            at an intersection over union of 0.5, from compute_class_true_positive_flags.
        target_precision (float): The precision the threshold must reach, e.g. 0.99.

    Returns:
        Dict[int, Optional[Tuple[float, float]]]: For every class index, its threshold and the recall it keeps, or None
            if no threshold reaches the target precision (or the class has no ground truth or no detection).
    """
    return {class_label: find_score_threshold(
        scores=gathered["scores"], positives=class_true_positive_flags[class_label],
        number_positives=gathered["number_ground_truth"], target_precision=target_precision)
        for class_label, gathered in class_detections.items()}


def summarise_thresholds(threshold_results: Dict[float, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarises the average precisions of one method and the recalls of all the detections at the ten intersection
    over union thresholds, for all the objects or for the objects of one size: the mAP and the mAR at 0.5, at 0.75 and
    their COCO-style mean over the ten thresholds, and the same three values for every class with ground truth (the
    same classes have ground truth at every threshold). The recalls do not depend on the method.

    Args:
        threshold_results (Dict[float, Dict[str, Any]]): The results of compute_mean_average_precision at every
            threshold of INTERSECTION_OVER_UNION_THRESHOLDS.

    Returns:
        Dict[str, Any]: The 'mAP@0.5', 'mAP@0.75', 'mAP@[0.5:0.95]', 'class_AP@0.5', 'class_AP@0.75',
            'class_AP@[0.5:0.95]', the same six recalls ('mAR@0.5' to 'class_AR@[0.5:0.95]') and the
            'class_ground_truths' of every class.
    """
    summary = {}
    for mean_key, class_key, metric_name in SUMMARISED_METRICS:
        # Mean over the classes at 0.5, 0.75 and averaged over the thresholds, then the same values of every class
        class_labels = list(threshold_results[0.5][class_key])
        summary.update({
            f"m{metric_name}@0.5": threshold_results[0.5][mean_key],
            f"m{metric_name}@0.75": threshold_results[0.75][mean_key],
            f"m{metric_name}@[0.5:0.95]": sum(result[mean_key] for result in threshold_results.values())
            / len(threshold_results),
            f"class_{metric_name}@0.5": threshold_results[0.5][class_key],
            f"class_{metric_name}@0.75": threshold_results[0.75][class_key],
            f"class_{metric_name}@[0.5:0.95]": {class_label: sum(
                result[class_key][class_label] for result in threshold_results.values()) / len(threshold_results)
                for class_label in class_labels}})
    summary["class_ground_truths"] = threshold_results[0.5]["class_ground_truths"]
    return summary


def evaluate_detections(detections: List[Dict[str, torch.Tensor]], ground_truths: List[Dict[str, torch.Tensor]],
                        number_classes: int, overlap_type: str = "boxes",
                        target_precision: Optional[float] = None) -> Dict[str, Dict[Any, Any]]:
    """
    Evaluates the detections of a whole test split, as reported by the Trainer at every evaluated checkpoint: with
    every average precision method, the mAP at an intersection over union of 0.5 (PASCAL VOC), at 0.75, and averaged
    over the thresholds 0.5 to 0.95 (COCO), plus the average precision of every class at the same thresholds and the
    number of ground truth boxes of every class, for the per-class report, and the same values for the small, medium
    and large objects (COCO sizes, see select_object_size_detections), each with the recall of all the detections at
    the same thresholds, the mean average recall (the same for every method). The detections are compared with the
    ground truths by their boxes, or by their masks for the mask AP of the instance segmentation. With a target
    precision, the score threshold of every class reaching it is added (find_class_score_thresholds), from the same
    gathered detections.

    Args:
        detections (List[Dict[str, torch.Tensor]]): For every image, its detection 'boxes' [N, 4], 'scores' [N] and
            'labels' [N], without padding, and for the masks the 'mask_overlaps' [N, M] with the ground truths and the
            'mask_areas' [N].
        ground_truths (List[Dict[str, torch.Tensor]]): For every image, its ground truth 'boxes' [M, 4] and
            'labels' [M], in the same coordinates as the detections, and for the masks the 'mask_areas' [M].
        number_classes (int): The number of classes, background (index 0) included.
        overlap_type (str): "boxes" or "masks". Defaults to "boxes".
        target_precision (Optional[float]): The precision of the score threshold of every class, None to skip the
            thresholds. Defaults to None.

    Returns:
        Dict[str, Dict[Any, Any]]: For every method, its 'mAP@0.5', 'mAP@0.75' and 'mAP@[0.5:0.95]', the average
            precision of every class with ground truth at 0.5, at 0.75 and averaged over 0.5 to 0.95 ('class_AP@0.5',
            'class_AP@0.75', 'class_AP@[0.5:0.95]'), the same values for the recall ('mAR@0.5' to
            'class_AR@[0.5:0.95]'), the 'class_ground_truths' boxes of every foreground class, and the same values for
            every object size under 'sizes' (their mAPs and mARs None when no object has the size); with a target
            precision, the 'class_score_thresholds' of every foreground class.
    """
    # Gather the detections and best matches of every foreground class once, for all thresholds and methods
    class_detections = {class_label: gather_class_detections(detections=detections, ground_truths=ground_truths,
                                                             class_label=class_label, overlap_type=overlap_type)
                        for class_label in range(1, number_classes)}

    # Match the detections once per threshold, for every method, size and the score thresholds
    threshold_flags = {threshold: compute_class_true_positive_flags(class_detections=class_detections,
                                                                    intersection_over_union_threshold=threshold)
                       for threshold in INTERSECTION_OVER_UNION_THRESHOLDS}

    results = {}
    for method in AVERAGE_PRECISION_METHODS:
        # All the objects, then the objects of every size, a size without object giving no mAP and no mAR
        results[method] = summarise_thresholds(threshold_results={threshold: compute_mean_average_precision(
            class_detections=class_detections, class_true_positive_flags=threshold_flags[threshold],
            intersection_over_union_threshold=threshold, method=method)
            for threshold in INTERSECTION_OVER_UNION_THRESHOLDS})
        results[method]["sizes"] = {}
        for object_size in OBJECT_SIZE_RANGES:
            size_results = summarise_thresholds(threshold_results={threshold: compute_mean_average_precision(
                class_detections=class_detections, class_true_positive_flags=threshold_flags[threshold],
                intersection_over_union_threshold=threshold, method=method, object_size=object_size)
                for threshold in INTERSECTION_OVER_UNION_THRESHOLDS})
            if not size_results["class_AP@0.5"]:
                size_results.update({f"m{metric_name}@{threshold_name}": None for _, _, metric_name in
                                     SUMMARISED_METRICS for threshold_name in ["0.5", "0.75", "[0.5:0.95]"]})
            results[method]["sizes"][object_size] = size_results

    # Score threshold of every class reaching the target precision, for the deployment
    if target_precision is not None:
        results["class_score_thresholds"] = find_class_score_thresholds(
            class_detections=class_detections, class_true_positive_flags=threshold_flags[
                CLASS_THRESHOLD_INTERSECTION_OVER_UNION], target_precision=target_precision)
    return results
