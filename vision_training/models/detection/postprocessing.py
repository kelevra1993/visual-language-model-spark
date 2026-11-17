"""
Final postprocessing of the object detector: class probabilities, background removal, score threshold, per-class
batched non-maximum suppression, greedy on the IoU or DIoU-NMS ('Detector.nms_criterion'), and a fixed number of
detections per image, vectorised so that both suppressions export, the greedy one of the exported detections and
proposals through the custom operator suppress_stacked_groups, which the ONNX export writes without coordinate offsets.
Shared by the TwoStageDetector at inference and by the InferenceDetector of the deployment.
"""

import torch

from typing import Dict
from torchvision.ops import batched_nms, nms

from vision_training.models.detection.box_operations import (clamp_boxes_to_image_boundaries,
                                                             get_distance_intersection_over_union)

# Criteria of the final per-class non-maximum suppression ('Detector.nms_criterion'): the IoU of the greedy NMS of
# torchvision, or the IoU minus the normalized distance of the centres of DIoU-NMS
# (Zheng et al., AAAI 2020, Eq. 13, https://arxiv.org/abs/1911.08287)
NMS_CRITERIA = ["iou", "distance_iou"]

# Largest number of box pairs whose criterion DIoU-NMS computes at once: the groups of image and class are processed
# in chunks of at most that many pairs, so that a batch with large groups keeps its memory bounded
MAXIMUM_DISTANCE_NMS_PAIRS = 2 ** 22


def suppress_groups_apart(boxes: torch.Tensor, scores: torch.Tensor, group_indices: torch.Tensor,
                          iou_threshold: float) -> torch.Tensor:
    """
    Runs the greedy non-maximum suppression of every group of boxes (an image, or an image and a class) in a single
    torchvision nms call for the exported detectors, the boxes of every group moved by its index times the largest
    coordinate plus one, so that boxes of different groups never overlap. This is the coordinate trick batched_nms
    runs on few boxes or under TorchScript tracing; called directly, it keeps one algorithm for any number of boxes,
    whereas batched_nms switches to a loop over the groups above 4000 coordinates on the CPU and 100,000 on the GPU,
    a choice that depends on the batch size, so that the dynamo ONNX export of a dynamic batch stops on it (a detector
    of 35 classes keeping 250 proposals gives 35,000 coordinates per image, beyond the limit from 3 images on).

    Args:
        boxes (torch.Tensor): The [N, 4] boxes, N above zero.
        scores (torch.Tensor): Their [N] scores.
        group_indices (torch.Tensor): Their [N] non-negative group indices.
        iou_threshold (float): The overlap above which the lower scoring box of a pair of the same group is suppressed.

    Returns:
        torch.Tensor: The int64 indices of the kept boxes, sorted by decreasing score, as batched_nms gives them.
    """
    # Offset of every group beyond the largest coordinate of all the boxes, then one suppression over all of them
    group_offsets = group_indices.to(dtype=boxes.dtype) * (boxes.max() + 1.0)
    return nms(boxes=boxes + group_offsets.unsqueeze(dim=1), scores=scores, iou_threshold=iou_threshold)


def suppress_stacked_groups_apart(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Runs the greedy non-maximum suppression of groups of the same number of boxes stacked on the first dimension, with
    the groups moved apart (suppress_groups_apart): the form of suppress_stacked_groups in the eager model and in the
    TorchScript export, served by libtorch in float32, where the moved coordinates keep their precision.

    Args:
        boxes (torch.Tensor): The [G, N, 4] boxes of every group.
        scores (torch.Tensor): Their [G, N] scores.
        iou_threshold (float): The overlap above which the lower scoring box of a pair of the same group is suppressed.

    Returns:
        torch.Tensor: The int64 flat indices (group times N plus box) of the kept boxes.
    """
    # Group of every box of the stack, its row
    number_boxes = scores.shape[1]
    group_indices = torch.arange(end=scores.shape[0], device=scores.device).unsqueeze(dim=1).expand(-1, number_boxes)
    return suppress_groups_apart(boxes=boxes.reshape(-1, 4), scores=scores.reshape(-1),
                                 group_indices=group_indices.reshape(-1), iou_threshold=iou_threshold)


@torch.library.custom_op("vision_training::suppress_stacked_groups", mutates_args=())
def suppress_stacked_groups_operator(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Custom operator of the stacked group suppression, recorded as one node by the dynamo ONNX export, where
    translate_stacked_group_suppression (vision_training/deployment/export_formats.py) writes it as the batched
    NonMaxSuppression of ONNX; run outside the export, it suppresses with the groups moved apart.

    Args:
        boxes (torch.Tensor): The [G, N, 4] boxes of every group.
        scores (torch.Tensor): Their [G, N] scores.
        iou_threshold (float): The overlap above which the lower scoring box of a pair of the same group is suppressed.

    Returns:
        torch.Tensor: The int64 flat indices (group times N plus box) of the kept boxes.
    """
    return suppress_stacked_groups_apart(boxes=boxes, scores=scores, iou_threshold=iou_threshold)


@suppress_stacked_groups_operator.register_fake
def suppress_stacked_groups_shape(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Gives the export the output of the stacked group suppression without running it: int64 indices whose number
    depends on the data.

    Args:
        boxes (torch.Tensor): The [G, N, 4] boxes of every group.
        scores (torch.Tensor): Their [G, N] scores.
        iou_threshold (float): The overlap above which the lower scoring box of a pair of the same group is suppressed.

    Returns:
        torch.Tensor: An empty int64 tensor of data-dependent length.
    """
    return boxes.new_empty(size=(torch.library.get_ctx().new_dynamic_size(),), dtype=torch.int64)


# Overload of the custom operator that the dynamo ONNX export records, the key of its translation in export_onnx
STACKED_GROUP_SUPPRESSION_OPERATOR = torch.ops.vision_training.suppress_stacked_groups.default


def suppress_stacked_groups(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Runs the greedy non-maximum suppression of the exported detectors on groups stacked on the first dimension (the
    images of the proposal suppression, the images and classes of the final detections), the boxes of a group never
    suppressing those of another. Under the dynamo ONNX export it is the custom operator suppress_stacked_groups,
    written as one NonMaxSuppression with one ONNX batch per group, so that no coordinate leaves the image: moved apart
    by about 1,025 pixels per group, the coordinates of the 64th group would pass 65,504, the largest float16 value, and
    a TensorRT FP16 engine computing them would keep the duplicates of every image of a request after the first. The
    eager model and the TorchScript trace keep suppress_groups_apart, libtorch loading no custom operator.

    Args:
        boxes (torch.Tensor): The [G, N, 4] boxes of every group, clipped to the image.
        scores (torch.Tensor): Their [G, N] scores.
        iou_threshold (float): The overlap above which the lower scoring box of a pair of the same group is suppressed.

    Returns:
        torch.Tensor: The int64 flat indices (group times N plus box) of the kept boxes, in any order.
    """
    if torch.compiler.is_exporting():
        return torch.ops.vision_training.suppress_stacked_groups(boxes=boxes, scores=scores,
                                                                 iou_threshold=iou_threshold)
    return suppress_stacked_groups_apart(boxes=boxes, scores=scores, iou_threshold=iou_threshold)


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


def suppress_every_pair_by_distance(boxes: torch.Tensor, scores: torch.Tensor, candidate_mask: torch.Tensor,
                                    iou_threshold: float, distance_exponent: float) -> torch.Tensor:
    """
    Runs DIoU-NMS on every (proposal, foreground class) pair of the exported detector, the form of batched_distance_nms
    that the TorchScript trace and the dynamo ONNX export keep, so that
    a deployed model with 'Detector.nms_criterion' "distance_iou" keeps the detections of its eager evaluation. Every
    group of an image and a class holds exactly the P pairs of its proposals, so the criterion matrix has the static
    shape [B, C - 1, P, P] without padding, and the greedy suppression is the exact scan of the P ranks by decreasing
    score, P steps of operations over all the groups at once, without control flow depending on the data. As in
    batched_distance_nms, only the candidates suppress or are suppressed, a box is suppressed when its criterion with a
    better kept box exceeds the threshold strictly, equal scores keep the proposal order, and the criterion is computed
    in float32 at least.

    Args:
        boxes (torch.Tensor): The [B, P, C - 1, 4] boxes of every proposal and foreground class, clipped to the image,
            in any unit: postprocess_detections gives them in fractions of the image, which TensorRT FP16 holds.
        scores (torch.Tensor): Their [B, P, C - 1] class probabilities.
        candidate_mask (torch.Tensor): The [B, P, C - 1] boolean mask of the pairs of valid proposals reaching the
            score threshold.
        iou_threshold (float): The Distance-IoU above which the lower scoring box of a pair is suppressed.
        distance_exponent (float): The positive exponent of the normalized centre distance.

    Returns:
        torch.Tensor: The [B, P, C - 1] boolean mask of the kept candidates, False for the other pairs.
    """
    # Groups of image and class first, as [B, C - 1, P] scores and candidates
    group_scores, group_candidates = scores.transpose(dim0=1, dim1=2), candidate_mask.transpose(dim0=1, dim1=2)
    number_proposals = scores.shape[1]
    proposal_indices = torch.arange(end=number_proposals, device=boxes.device)
    earlier_first = proposal_indices.unsqueeze(dim=1) < proposal_indices.unsqueeze(dim=0)

    # Rank of every proposal in its group: the number of proposals with a higher score, or the same score and an
    # earlier index, which is the order of the stable sort of batched_distance_nms. It is computed from comparisons
    # rather than with a stable sort, which the dynamo ONNX export does not translate
    row_scores, column_scores = group_scores.unsqueeze(dim=-1), group_scores.unsqueeze(dim=-2)
    better_mask = (row_scores > column_scores) | ((row_scores == column_scores) & earlier_first)
    proposal_ranks = better_mask.sum(dim=-2)
    score_order = torch.zeros_like(input=proposal_ranks).scatter(dim=-1, index=proposal_ranks,
                                                                 src=proposal_indices.expand_as(proposal_ranks))

    # Boxes and candidates of every group by decreasing score
    sorted_boxes = torch.gather(input=boxes.transpose(dim0=1, dim1=2), dim=2,
                                index=score_order.unsqueeze(dim=-1).expand(-1, -1, -1, 4))
    sorted_candidates = torch.gather(input=group_candidates, dim=2, index=score_order)

    # Pairs whose better candidate suppresses the lower one when kept: the criterion computed in float32 at least and
    # strictly above the threshold, between two candidates of the group, the better one first
    sorted_boxes = sorted_boxes.to(dtype=torch.promote_types(boxes.dtype, torch.float32))
    criterion = get_distance_intersection_over_union(boxes_1=sorted_boxes, boxes_2=sorted_boxes,
                                                     distance_exponent=distance_exponent)
    suppression_mask = ((criterion > iou_threshold) & earlier_first & sorted_candidates.unsqueeze(dim=-1) &
                        sorted_candidates.unsqueeze(dim=-2))

    # Greedy scan of the ranks: the candidate of every rank is kept unless a better kept candidate suppressed it, and
    # then suppresses the lower ones of its row, the strict upper triangle leaving the better ranks unchanged
    suppressed_mask = torch.zeros_like(input=sorted_candidates)
    for rank in range(number_proposals):
        rank_kept = sorted_candidates[..., rank] & ~suppressed_mask[..., rank]
        suppressed_mask = suppressed_mask | (suppression_mask[..., rank, :] & rank_kept.unsqueeze(dim=-1))

    # Kept candidates put back in the proposal order of every group, then in the [B, P, C - 1] layout of the pairs
    sorted_kept_mask = sorted_candidates & ~suppressed_mask
    kept_mask = torch.zeros_like(input=sorted_kept_mask).scatter(dim=-1, index=score_order, src=sorted_kept_mask)
    return kept_mask.transpose(dim0=1, dim1=2)


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
    NMS or DIoU-NMS of 'Detector.nms_criterion', while the deployed detector gives every pair to the same NMS (the
    greedy one of suppress_stacked_groups, one row per image and class, or suppress_every_pair_by_distance for
    DIoU-NMS), without data-dependent Python control flow nor data-dependent sizes, so that the same code can be
    exported to TorchScript and ONNX. Both paths keep the same detections with either criterion. The outputs have a
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
            DIoU-NMS (batched_distance_nms in the eager path, suppress_every_pair_by_distance in the export path).
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
        ValueError: If the criterion is not one of NMS_CRITERIA.
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
    # of a whole batch. The exports give every pair to the NMS, the dynamo ONNX export refusing a data-dependent number
    # of boxes: the greedy NMS of the groups stacked as rows (suppress_stacked_groups), or DIoU-NMS on the static
    # [B, C - 1, P, P] matrix of the groups
    kept_mask = torch.zeros_like(input=probabilities.reshape(-1), dtype=torch.bool)
    if suppress_candidates_only:
        candidate_indices = torch.nonzero(input=candidate_mask.reshape(-1)).flatten()
        kept_indices = candidate_indices[suppress_overlapping_detections(
            boxes=boxes.reshape(-1, 4)[candidate_indices], scores=probabilities.reshape(-1)[candidate_indices],
            group_indices=group_indices.reshape(-1)[candidate_indices], iou_threshold=nms_iou_threshold,
            nms_criterion=nms_criterion, nms_distance_exponent=nms_distance_exponent)]
        kept_mask[kept_indices] = True
    elif nms_criterion == "iou":
        # One row of P pairs per group of image and class, [B (C - 1), P], then the kept rows put back in the
        # [B, P, C - 1] layout of the pairs
        group_boxes = boxes.reshape(batch_size, number_proposals, number_classes - 1, 4).transpose(dim0=1, dim1=2)
        group_scores = probabilities.reshape(batch_size, number_proposals, number_classes - 1).transpose(dim0=1,
                                                                                                         dim1=2)
        kept_indices = suppress_stacked_groups(boxes=group_boxes.reshape(-1, number_proposals, 4),
                                               scores=group_scores.reshape(-1, number_proposals),
                                               iou_threshold=nms_iou_threshold)
        group_kept_mask = torch.zeros_like(input=group_scores.reshape(-1), dtype=torch.bool)
        group_kept_mask[kept_indices] = True
        kept_mask = group_kept_mask.reshape(batch_size, number_classes - 1, number_proposals).transpose(dim0=1, dim1=2)
    elif nms_criterion == "distance_iou":
        # Boxes in fractions of the image, which leave the IoU and the normalized centre distance unchanged, so that
        # the areas stay below 1 where a TensorRT FP16 engine overflowed the areas in pixels from 256 x 256 pixels on
        kept_mask = suppress_every_pair_by_distance(
            boxes=(boxes / input_image_size).reshape(batch_size, number_proposals, number_classes - 1, 4),
            scores=probabilities.reshape(batch_size, number_proposals, number_classes - 1),
            candidate_mask=candidate_mask.reshape(batch_size, number_proposals, number_classes - 1),
            iou_threshold=nms_iou_threshold, distance_exponent=nms_distance_exponent)
    else:
        raise ValueError(f"Unknown NMS criterion {nms_criterion!r}, expected one of {NMS_CRITERIA}.")

    # Keep the NMS survivors among the candidates, the other pairs get a score below any valid one
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
