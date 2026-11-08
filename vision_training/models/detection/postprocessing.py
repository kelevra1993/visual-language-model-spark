"""
Final postprocessing of the object detector: class probabilities, background removal, score threshold, per-class
batched non-maximum suppression, greedy on the IoU or DIoU-NMS ('Detector.nms_criterion'), and a fixed number of
detections per image, vectorised so that the greedy one exports. Shared by the TwoStageDetector at inference and by
the InferenceDetector of the deployment.
"""

import torch

from typing import Dict
from torchvision.ops import batched_nms

from vision_training.models.detection.box_operations import (clamp_boxes_to_image_boundaries,
                                                             get_distance_intersection_over_union)

# Criteria of the final per-class non-maximum suppression ('Detector.nms_criterion'): the IoU of the greedy NMS of
# torchvision, or the IoU minus the normalized distance of the centres of DIoU-NMS
# (documentation/detection_nms_design.md)
NMS_CRITERIA = ["iou", "distance_iou"]

# Largest number of box pairs whose criterion DIoU-NMS computes at once: the groups of image and class are processed
# in chunks of at most that many pairs, so that a batch with large groups keeps its memory bounded
MAXIMUM_DISTANCE_NMS_PAIRS = 2 ** 22


def keep_cluster_boxes(padded_boxes: torch.Tensor, valid_mask: torch.Tensor, iou_threshold: float,
                       distance_exponent: float) -> torch.Tensor:
    """
    Runs the iterations of the Cluster-NMS of batched_distance_nms on a chunk of groups (Zheng et al., IEEE TCYB,
    Algorithm 2): the criterion between every box of a group and every lower scoring one, then the kept boxes taken as
    those no kept box suppresses, recomputed until they stop changing. A box depends on the better boxes only, so after
    t iterations the t best boxes of every group are those of the greedy loop of the authors' code, and the loop stops
    at the latest after as many iterations as the largest group has boxes.

    Args:
        padded_boxes (torch.Tensor): The [G, M, 4] boxes of every group by decreasing score, padded with empty boxes.
        valid_mask (torch.Tensor): The [G, M] boolean mask of the boxes, False for the padding.
        iou_threshold (float): The Distance-IoU above which the lower scoring box of a pair is suppressed.
        distance_exponent (float): The positive exponent of the normalized centre distance.

    Returns:
        torch.Tensor: The [G, M] boolean mask of the kept boxes, False for the padding.
    """
    # Pairs whose better box suppresses the lower one when kept: the criterion strictly above the threshold, between
    # two boxes of the group, the better box first (strict upper triangle of the score order)
    number_slots = padded_boxes.shape[1]
    criterion = get_distance_intersection_over_union(boxes_1=padded_boxes, boxes_2=padded_boxes,
                                                     distance_exponent=distance_exponent)
    slot_ranks = torch.arange(end=number_slots, device=padded_boxes.device)
    better_first = slot_ranks.unsqueeze(dim=1) < slot_ranks.unsqueeze(dim=0)
    suppression_mask = ((criterion > iou_threshold) & better_first & valid_mask.unsqueeze(dim=-1) &
                        valid_mask.unsqueeze(dim=-2))

    # Every box kept at first, then the boxes suppressed by no kept box, until the kept boxes stop changing
    kept_mask = valid_mask
    for _ in range(number_slots):
        updated_kept_mask = valid_mask & ~(suppression_mask & kept_mask.unsqueeze(dim=-1)).any(dim=-2)
        if torch.equal(input=updated_kept_mask, other=kept_mask):
            break
        kept_mask = updated_kept_mask
    return kept_mask


def batched_distance_nms(boxes: torch.Tensor, scores: torch.Tensor, group_indices: torch.Tensor, iou_threshold: float,
                         distance_exponent: float) -> torch.Tensor:
    """
    Suppresses, inside every group of boxes (an image and a class of the final detections), the boxes whose
    Distance-IoU with a better kept box of the group exceeds the threshold: the DIoU-NMS of Zheng et al. (AAAI 2020,
    Eq. 13), which keeps two overlapping boxes whose centres are apart, likely two objects, where the greedy NMS keeps
    one. postprocess_detections calls it in place of the batched_nms of torchvision when 'Detector.nms_criterion' is
    "distance_iou", with the same arguments, and it keeps the same boxes when no centre distance separates them.

    It is the exact Cluster-NMS of the journal version (Zheng et al., IEEE TCYB, Algorithm 2, Section IV-C), for every
    group at once: the boxes of every group sorted by decreasing score and padded into [G, M] slots
    (keep_cluster_boxes), the groups taken by decreasing size in chunks of at most MAXIMUM_DISTANCE_NMS_PAIRS pairs,
    each padded to its largest group. A box is suppressed when its criterion exceeds the threshold strictly, as in
    torchvision and darknet, and equal scores keep the order of the input.

    Args:
        boxes (torch.Tensor): The [N, 4] boxes [x_min, y_min, x_max, y_max].
        scores (torch.Tensor): Their [N] scores.
        group_indices (torch.Tensor): Their [N] integer groups, boxes of different groups never suppressing each other.
        iou_threshold (float): The Distance-IoU above which the lower scoring box of a pair is suppressed.
        distance_exponent (float): The positive exponent of the normalized centre distance, 1.0 in the paper.

    Returns:
        torch.Tensor: The int64 indices of the kept boxes, sorted by decreasing score.
    """
    # No candidate, nothing to keep
    number_boxes = boxes.shape[0]
    if number_boxes == 0:
        return torch.empty(size=(0,), dtype=torch.int64, device=boxes.device)

    # Order the boxes by group, and by decreasing score inside every group, equal scores keeping the input order
    score_order = torch.sort(input=scores, descending=True, stable=True).indices
    box_order = score_order[torch.sort(input=group_indices[score_order], stable=True).indices]

    # Number of every box's group, the size of every group and the rank of every box in its group
    _, group_numbers, group_sizes = torch.unique_consecutive(input=group_indices[box_order], return_inverse=True,
                                                             return_counts=True)
    group_starts = torch.cumsum(input=group_sizes, dim=0) - group_sizes
    box_ranks = torch.arange(end=number_boxes, device=boxes.device) - group_starts[group_numbers]

    # Criterion computed in float32 at least, for the boxes of the bfloat16 mixed precision
    sorted_boxes = boxes[box_order].to(dtype=torch.promote_types(boxes.dtype, torch.float32))

    # Chunks of groups by decreasing size, every chunk padded to its largest group and holding as many groups as the
    # pair budget allows, one at least
    size_order = torch.sort(input=group_sizes, descending=True, stable=True).indices
    sorted_group_sizes = group_sizes[size_order].tolist()
    sorted_kept_mask = torch.zeros(size=(number_boxes,), dtype=torch.bool, device=boxes.device)
    chunk_start = 0
    while chunk_start < len(sorted_group_sizes):
        padded_size = sorted_group_sizes[chunk_start]
        chunk_end = min(len(sorted_group_sizes), chunk_start + max(1, MAXIMUM_DISTANCE_NMS_PAIRS // padded_size ** 2))

        # Slot of every group of the chunk, -1 for the groups of other chunks, and the slots of their boxes
        group_slots = torch.full(size=group_sizes.shape, fill_value=-1, dtype=torch.int64, device=boxes.device)
        group_slots[size_order[chunk_start:chunk_end]] = torch.arange(end=chunk_end - chunk_start, device=boxes.device)
        box_slots = group_slots[group_numbers]
        chunk_box_mask = box_slots >= 0
        chunk_slots, chunk_ranks = box_slots[chunk_box_mask], box_ranks[chunk_box_mask]

        # Boxes of the chunk padded into [G, M] slots, then the kept ones read back in the order of the sorted boxes
        padded_boxes = torch.zeros(size=(chunk_end - chunk_start, padded_size, 4), dtype=sorted_boxes.dtype,
                                   device=boxes.device)
        padded_boxes[chunk_slots, chunk_ranks] = sorted_boxes[chunk_box_mask]
        valid_mask = torch.zeros(size=(chunk_end - chunk_start, padded_size), dtype=torch.bool, device=boxes.device)
        valid_mask[chunk_slots, chunk_ranks] = True
        chunk_kept_mask = keep_cluster_boxes(padded_boxes=padded_boxes, valid_mask=valid_mask,
                                             iou_threshold=iou_threshold, distance_exponent=distance_exponent)
        sorted_kept_mask[chunk_box_mask] = chunk_kept_mask[chunk_slots, chunk_ranks]
        chunk_start = chunk_end

    # Kept boxes in the input numbering, by decreasing score like batched_nms
    kept_indices = box_order[sorted_kept_mask]
    return kept_indices[torch.sort(input=scores[kept_indices], descending=True, stable=True).indices]


def suppress_overlapping_detections(boxes: torch.Tensor, scores: torch.Tensor, group_indices: torch.Tensor,
                                    iou_threshold: float, nms_criterion: str,
                                    nms_distance_exponent: float) -> torch.Tensor:
    """
    Runs the final per-class non-maximum suppression of the eager training and evaluation on the candidates of
    postprocess_detections with the criterion of 'Detector.nms_criterion': the batched greedy NMS of torchvision on
    the IoU, or DIoU-NMS (batched_distance_nms) on the IoU minus the normalized distance of the centres.

    Args:
        boxes (torch.Tensor): The [N, 4] boxes of the candidates.
        scores (torch.Tensor): Their [N] scores.
        group_indices (torch.Tensor): Their [N] groups of image and class.
        iou_threshold (float): The criterion above which the lower scoring box of a pair is suppressed.
        nms_criterion (str): One of NMS_CRITERIA.
        nms_distance_exponent (float): The exponent of the distance term of "distance_iou", unused by "iou".

    Returns:
        torch.Tensor: The int64 indices of the kept candidates, sorted by decreasing score.

    Raises:
        ValueError: If the criterion is not one of NMS_CRITERIA.
    """
    if nms_criterion == "iou":
        return batched_nms(boxes=boxes, scores=scores, idxs=group_indices, iou_threshold=iou_threshold)
    if nms_criterion == "distance_iou":
        return batched_distance_nms(boxes=boxes, scores=scores, group_indices=group_indices,
                                    iou_threshold=iou_threshold, distance_exponent=nms_distance_exponent)
    raise ValueError(f"Unknown NMS criterion {nms_criterion!r}, expected one of {NMS_CRITERIA}.")


def postprocess_detections(classification_scores: torch.Tensor, detection_boxes: torch.Tensor,
                           valid_proposal_mask: torch.Tensor, score_threshold: float, nms_iou_threshold: float,
                           nms_criterion: str, nms_distance_exponent: float, maximum_detections: int,
                           input_image_size: int, suppress_candidates_only: bool) -> Dict[str, torch.Tensor]:
    """
    Turns the raw detection head outputs into the final detections returned at inference and by the deployed
    models: class probabilities, removal of the background class, a score threshold, per-class non-maximum
    suppression, and a fixed number of detections per image.

    Every step is vectorised over the whole batch (a single batched NMS call grouped by image and class, then a fixed
    top-k). The eager training and evaluation give the NMS the pairs reaching the score threshold only, with the greedy
    NMS or DIoU-NMS of 'Detector.nms_criterion', while the deployed detector gives every pair to the greedy NMS,
    without data-dependent Python control flow nor data-dependent sizes, so that the same code can be exported to
    TorchScript and ONNX. Both keep the same detections with the greedy NMS, DIoU-NMS being refused by the exports until
    a stage of the detection benchmark keeps it (documentation/detection_nms_design.md, Part B). The outputs have a
    fixed size; padded entries have a score of 0, a label of 0 and an empty box.

    Args:
        classification_scores (torch.Tensor): The class logits of every proposal, shape [B, P, C] (C includes the
            background class 0).
        detection_boxes (torch.Tensor): The per-class decoded [x1, y1, x2, y2] boxes, shape [B, P, C, 4].
        valid_proposal_mask (torch.Tensor): Boolean mask of shape [B, P], False for padding proposals that must never
            produce a detection (used by the exportable inference path, which pads with placeholder proposals).
        score_threshold (float): The minimum class probability of a returned detection.
        nms_iou_threshold (float): The criterion above which a lower scoring box of the same class is suppressed: the
            IoU, or the Distance-IoU of "distance_iou".
        nms_criterion (str): One of NMS_CRITERIA, "iou" for the greedy NMS of torchvision or "distance_iou" for
            DIoU-NMS (batched_distance_nms), in the eager path only.
        nms_distance_exponent (float): The exponent of the distance term of "distance_iou", unused by "iou".
        maximum_detections (int): The fixed number K of detections returned per image, padded when the image has
            fewer (proposal, foreground class) pairs, as in the validation, which classifies the sampled proposals only.
        input_image_size (int): The size of the square network input, used to clip the boxes to the image.
        suppress_candidates_only (bool): True for the NMS on the pairs reaching the score threshold (eager), False for
            the NMS on every pair (export).

    Returns:
        Dict[str, torch.Tensor]: The final detections:
            - "final_detection_boxes" (torch.Tensor): Shape [B, K, 4].
            - "final_detection_scores" (torch.Tensor): Shape [B, K], sorted in decreasing order.
            - "final_detection_labels" (torch.Tensor): Shape [B, K], int64 class indices (0 for padding).
            - "number_detections" (torch.Tensor): Shape [B], int64 number of valid detections per image.
            - "final_detection_proposal_indices" (torch.Tensor): Shape [B, K], int64 index of the proposal of every
              detection among the P proposals of its image (0 for padding), read by the instance segmenter to run
              the mask head of the level of that proposal.

    Raises:
        ValueError: If the criterion is not one of NMS_CRITERIA, or is not "iou" in the export path.
    """
    batch_size, number_proposals, number_classes = classification_scores.shape

    # Class probabilities without the background class, zero for padding proposals, flattened over proposals and
    # foreground classes
    probabilities = torch.softmax(input=classification_scores, dim=-1)[..., 1:]
    probabilities = (probabilities * valid_proposal_mask.unsqueeze(dim=-1)).reshape(batch_size, -1)
    boxes = clamp_boxes_to_image_boundaries(boxes=detection_boxes[..., 1:, :],
                                            input_image_size=input_image_size).reshape(batch_size, -1, 4)
    labels = torch.arange(start=1, end=number_classes, device=classification_scores.device)
    labels = labels.repeat(number_proposals).unsqueeze(dim=0).expand(batch_size, -1)

    # Candidates that can become detections: the pairs of valid proposals reaching the score threshold
    batch_indices = torch.arange(end=batch_size, device=classification_scores.device).unsqueeze(dim=1).expand_as(labels)
    group_indices = batch_indices * number_classes + labels
    valid_candidate_mask = valid_proposal_mask.unsqueeze(dim=-1).expand(-1, -1, number_classes - 1)
    candidate_mask = (probabilities >= score_threshold) & valid_candidate_mask.reshape(batch_size, -1)

    # Suppress overlapping boxes of the same class and image with a single NMS call over the whole batch. The eager
    # path only gives it the candidates: a box below the threshold never suppresses a higher scoring one, so the
    # detections are the same, without the per-group NMS loop torchvision runs on the hundreds of thousands of pairs
    # of a whole batch. The exports give every pair to the greedy NMS, the dynamo ONNX export refusing a
    # data-dependent number of boxes, and refuse DIoU-NMS, whose exportable form is not written yet
    if suppress_candidates_only:
        candidate_indices = torch.nonzero(input=candidate_mask.reshape(-1)).flatten()
        kept_indices = candidate_indices[suppress_overlapping_detections(
            boxes=boxes.reshape(-1, 4)[candidate_indices], scores=probabilities.reshape(-1)[candidate_indices],
            group_indices=group_indices.reshape(-1)[candidate_indices], iou_threshold=nms_iou_threshold,
            nms_criterion=nms_criterion, nms_distance_exponent=nms_distance_exponent)]
    else:
        if nms_criterion != "iou":
            raise ValueError(f"The NMS on every pair of the exports is the greedy NMS ('iou') only, got "
                             f"{nms_criterion!r}: DIoU-NMS runs in the eager evaluation, its exportable form being "
                             f"Part B of documentation/detection_nms_design.md.")
        kept_indices = batched_nms(boxes=boxes.reshape(-1, 4), scores=probabilities.reshape(-1),
                                   idxs=group_indices.reshape(-1), iou_threshold=nms_iou_threshold)

    # Keep the NMS survivors among the candidates, the other pairs get a score below any valid one
    kept_mask = torch.zeros_like(input=probabilities.reshape(-1), dtype=torch.bool)
    kept_mask[kept_indices] = True
    kept_mask = kept_mask.reshape(batch_size, -1) & candidate_mask
    candidate_scores = torch.where(kept_mask, probabilities, torch.full_like(input=probabilities, fill_value=-1.0))

    # Append maximum_detections invalid candidates (empty box, background label), so that an image with fewer
    # candidates than detections (few proposals and classes) still fills its fixed number of detections, without
    # any Python condition on the tensor sizes that the exports would freeze
    candidate_scores = torch.nn.functional.pad(input=candidate_scores, pad=(0, maximum_detections), value=-1.0)
    boxes = torch.nn.functional.pad(input=boxes, pad=(0, 0, 0, maximum_detections))
    labels = torch.nn.functional.pad(input=labels, pad=(0, maximum_detections))

    # Select the fixed number of best detections per image and zero out the padded entries
    top_scores, top_indices = torch.topk(input=candidate_scores, k=maximum_detections, dim=1)
    valid_mask = top_scores >= 0.0
    final_boxes = torch.gather(input=boxes, dim=1, index=top_indices.unsqueeze(dim=-1).expand(-1, -1, 4))

    final_labels = torch.gather(input=labels, dim=1, index=top_indices)

    # Proposal of every detection, the candidates being flattened proposal by proposal over the foreground classes
    final_proposal_indices = torch.div(input=top_indices, other=number_classes - 1, rounding_mode="floor")
    return {"final_detection_boxes": torch.where(valid_mask.unsqueeze(dim=-1), final_boxes,
                                                 torch.zeros_like(input=final_boxes)),
            "final_detection_scores": torch.where(valid_mask, top_scores, torch.zeros_like(input=top_scores)),
            "final_detection_labels": torch.where(valid_mask, final_labels, torch.zeros_like(input=final_labels)),
            "number_detections": valid_mask.sum(dim=1),
            "final_detection_proposal_indices": torch.where(valid_mask, final_proposal_indices,
                                                            torch.zeros_like(input=final_proposal_indices))}
