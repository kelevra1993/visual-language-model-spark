"""
Training target assignment of the object detector: matching of the ground truth boxes to the anchors (region
proposal stage) and to the proposals (detection stage) by intersection over union, regression targets, and the
sampling of positive and negative examples. Every function handles all the images of a training batch at once, the
ground truths of the images being padded to the same number, so that a batch costs the same few GPU operations
whatever its size and the GPU never waits for the host. Used by the TwoStageDetector and
vision_training/losses/detection_losses.py.
"""

import functools
import torch

from typing import Callable, Dict, List, Optional, Tuple

from vision_training.models.detection.box_operations import (BOX_REGRESSION_WEIGHTS, get_box_dimensions_and_centers,
                                                             get_intersection_over_union)

# Most (image, object, anchor) overlaps the anchor target assignments hold at once, about 2.7 GB at the 40 bytes per
# overlap of get_intersection_over_union: the objects of a crowded batch are matched to the anchors in chunks below it
MAXIMUM_ANCHOR_OVERLAPS = 2 ** 26

# Inductor settings of the compiled anchor matching (compile_for_gpu) giving the rounding of the eager operations: no
# fused multiply-add, divisions rounded to nearest, subnormal numbers kept, so that the overlaps match bit for bit
EAGER_ROUNDING_OPTIONS = {"emulate_precision_casts": True, "eager_numerics.division_rounding": True,
                          "eager_numerics.disable_ftz": True}


def count_objects_per_chunk(batch_size: int, number_anchors: int) -> int:
    """
    Gives the number of objects that the anchor target assignments (batch_assign_targets_to_anchors and
    compute_best_anchor_overlaps) match against all the anchors at once, so that their overlap tensors stay under
    MAXIMUM_ANCHOR_OVERLAPS whatever the number of objects of the most crowded image of a training batch. An Open
    Images image holds up to hundreds of boxes, whose overlaps with the 330,240 anchors of its detectors would
    otherwise take tens of GB; a batch of few anchors or objects is matched in a single chunk.

    Args:
        batch_size (int): The number of images of the batch.
        number_anchors (int): The number of anchors every object is matched against.

    Returns:
        int: The objects of a chunk, at least 1.
    """
    return max(1, MAXIMUM_ANCHOR_OVERLAPS // max(1, batch_size * number_anchors))


def match_objects_to_anchors(chunk_boxes: torch.Tensor, chunk_valid_ground_truths: torch.Tensor,
                             anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matches a chunk of the objects of a training batch to all the anchors, the step of batch_assign_targets_to_anchors
    repeated for every chunk of objects: the best object of the chunk of every anchor with its overlap, and the anchors
    forced foreground as the best anchor of an object of the chunk. On the GPU it runs compiled (compile_for_gpu), its
    overlaps and their three reductions fused into a few kernels.

    Args:
        chunk_boxes (torch.Tensor): The [B, G_chunk, 4] padded ground truth boxes of the chunk.
        chunk_valid_ground_truths (torch.Tensor): The [B, G_chunk] boolean mask of the real objects of the chunk.
        anchors (torch.Tensor): The [B, A, 4] anchors of every image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The [B, A] best overlap of every anchor with an object of the
            chunk, the [B, A] int64 index of that object within the chunk (the first one on ties) and the [B, A]
            boolean mask of the anchors that are the best anchor of an object of the chunk.
    """
    # Shape: [batch_size, objects of the chunk, number_of_anchors], the padding rows overlapping nothing
    intersection_over_union_matrix = get_intersection_over_union(boxes_1=chunk_boxes, boxes_2=anchors)
    intersection_over_union_matrix = intersection_over_union_matrix.masked_fill(
        mask=~chunk_valid_ground_truths.unsqueeze(dim=-1), value=0.0)

    # For every anchor, find the ground truth box of the chunk that has the highest IoU with it
    chunk_best_iou, chunk_best_index = intersection_over_union_matrix.max(dim=1)

    # Forcing the best anchor of every ground truth box positive, even below the foreground threshold
    # Get the best iou value for each ground truth box across all the anchors of its image, shape [batch, chunk]
    best_match_anchor_iou, _ = intersection_over_union_matrix.max(dim=2)

    # This gives us all the anchors that tie for the highest iou for each ground truth box.
    # We use torch.isclose to safely handle float32 truncation artifacts where mathematically
    # identical bounding box areas might differ by ~1e-6.
    # We also enforce a strict > 0.0 threshold to prevent the "Zero-IoU Explosion",
    # preventing a ground truth box with no overlap (or a padding row) from resurrecting thousands of anchors.
    is_best_anchor = (torch.isclose(input=intersection_over_union_matrix,
                                    other=best_match_anchor_iou.unsqueeze(dim=-1), atol=1e-6)
                      & (best_match_anchor_iou.unsqueeze(dim=-1) > 0.0))
    return chunk_best_iou, chunk_best_index, is_best_anchor.any(dim=1)


def compute_chunk_best_overlaps(chunk_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Gives the best overlap of every object of a chunk with the anchors of one detection level, the step of
    compute_best_anchor_overlaps repeated for every chunk of objects. On the GPU it runs compiled (compile_for_gpu), the
    overlaps and their maximum fused into a few kernels.

    Args:
        chunk_boxes (torch.Tensor): The [B, G_chunk, 4] padded ground truth boxes of the chunk.
        anchors (torch.Tensor): The [A, 4] anchors of the level, of the type of the boxes.

    Returns:
        torch.Tensor: The [B, G_chunk] best overlap of every object of the chunk with the anchors of the level.
    """
    return get_intersection_over_union(boxes_1=chunk_boxes, boxes_2=anchors).amax(dim=-1)


@functools.lru_cache(maxsize=None)
def compile_for_gpu(function: Callable) -> Callable:
    """
    Gives a matching function of the anchor target assignments (match_objects_to_anchors, compute_chunk_best_overlaps)
    compiled by torch.compile, used for the batches on the GPU. Run eagerly, the overlaps of a chunk of objects with all
    the anchors and their reductions take about fifteen passes over [B, objects, anchors] tensors, a quarter of the GPU
    time of a training step of the Open Images detectors (330,240 anchors); compiled, they take a few fused kernels, 8
    to 14 times faster. The functions compile once per process, at the first training batch (about 10 s, plus a few
    seconds for the chunks of a single object and for the evaluation passes without gradients, which compile apart),
    their shapes staying dynamic so that the changing numbers of objects never compile again. The kernels compile with
    the rounding of the eager operations (EAGER_ROUNDING_OPTIONS), so that their overlaps and targets are those of the
    eager functions bit for bit, at the same speed. The CPU keeps the eager functions.

    Args:
        function (Callable): The matching function to compile.

    Returns:
        Callable: The compiled function, the same object for every call of the process.
    """
    return torch.compile(model=function, dynamic=True, options=EAGER_ROUNDING_OPTIONS)


def compute_best_anchor_overlaps(padded_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Gives the best overlap of every object of a training batch with the anchors of one detection level, read by
    add_ground_truth_proposals to give every ground truth box added to the proposals the level of its best anchor. The
    objects are matched in chunks (count_objects_per_chunk), so that a crowded image never holds the overlaps of all
    its objects with all the anchors at once; the overlaps are those of a single pass.

    Args:
        padded_boxes (torch.Tensor): The [B, G, 4] padded ground truth boxes of the batch (pad_ground_truths).
        anchors (torch.Tensor): The [A, 4] anchors of the level, of the type of the boxes.

    Returns:
        torch.Tensor: The [B, G] best overlap of every object with the anchors of the level.
    """
    objects_per_chunk = count_objects_per_chunk(batch_size=padded_boxes.shape[0], number_anchors=anchors.shape[0])

    # Best overlap of the objects of every chunk, of shape [B, objects of the chunk], joined in the order of the
    # objects, fused on the GPU (compile_for_gpu)
    chunk_function = compile_for_gpu(function=compute_chunk_best_overlaps) if padded_boxes.is_cuda else \
        compute_chunk_best_overlaps
    return torch.cat(tensors=[chunk_function(chunk_boxes=padded_boxes[:, chunk_start:chunk_start + objects_per_chunk],
                                             anchors=anchors)
                              for chunk_start in range(0, padded_boxes.shape[1], objects_per_chunk)], dim=1)


def pad_ground_truths(batched_ground_truth_boxes: List[torch.Tensor],
                      batched_ground_truth_labels: Optional[List[torch.Tensor]] = None
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Stacks the ground truth boxes (and labels) of the images of a training batch into padded tensors, so that the
    anchor and proposal target assignments match every image of the batch at once. The images have different numbers
    of objects: every image is completed with empty boxes [0, 0, 0, 0] (label 0) up to the largest number of objects
    of the batch, at least one, an empty box overlapping nothing.

    Args:
        batched_ground_truth_boxes (List[torch.Tensor]): The [G_i, 4] ground truth boxes of every image.
        batched_ground_truth_labels (Optional[List[torch.Tensor]]): The [G_i] class indices of every image, or None.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]: The [B, G, 4] padded boxes, the [B, G] int64
            padded labels (None without labels) and the [B, G] boolean mask of the real objects.
    """
    # Object counts of the images, known on the host from the tensor shapes without waiting for the GPU
    number_objects = [int(boxes.shape[0]) for boxes in batched_ground_truth_boxes]
    maximum_objects = max(max(number_objects), 1)
    device = batched_ground_truth_boxes[0].device

    # Boxes of every image followed by empty boxes, at least one row even when no image has an object
    padded_boxes = torch.nn.utils.rnn.pad_sequence(sequences=batched_ground_truth_boxes, batch_first=True)
    padded_boxes = torch.nn.functional.pad(input=padded_boxes, pad=(0, 0, 0, maximum_objects - padded_boxes.shape[1]))

    # Labels of every image followed by the background label
    padded_labels = None
    if batched_ground_truth_labels is not None:
        padded_labels = torch.nn.utils.rnn.pad_sequence(sequences=batched_ground_truth_labels, batch_first=True)
        padded_labels = torch.nn.functional.pad(input=padded_labels.to(dtype=torch.int64),
                                                pad=(0, maximum_objects - padded_labels.shape[1]))

    # Real objects of every image, the first ones of its padded rows
    object_counts = torch.tensor(data=number_objects, device=device)
    row_positions = torch.arange(end=maximum_objects, device=device).unsqueeze(dim=0)
    valid_ground_truths = row_positions < object_counts.unsqueeze(dim=1)
    return padded_boxes, padded_labels, valid_ground_truths


def add_ground_truth_proposals(batched_ground_truth_boxes: List[torch.Tensor], batched_proposals: torch.Tensor,
                               batched_proposal_levels: torch.Tensor, level_anchors: Dict[int, torch.Tensor]
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adds the ground truth boxes of every image to its filtered proposals before the proposal target assignment of
    the detection heads, as Mask R-CNN does, so that every object of a training batch gives at least one positive
    proposal (with a zero regression target) even when the region proposal network still misses it, early in the
    training or for the objects its anchors cover badly. The ground truth boxes overwrite the last slots of every
    image: the RegionProposalFilter returns the survivors of the suppression first, best first, then the padding
    copies of random survivors, so the boxes replace the padding before the least confident proposals, and the
    number of proposals per image stays the same, as the stacking of the detections per image needs. An image with
    more objects than proposal slots keeps its last objects only. Every ground truth box takes the level of its best
    anchor, the level whose detection head would have received it from the region proposal network.

    Args:
        batched_ground_truth_boxes (List[torch.Tensor]): The [G_i, 4] ground truth boxes of every image, in network
            input pixels.
        batched_proposals (torch.Tensor): The [B, P, 4] filtered proposals of every image.
        batched_proposal_levels (torch.Tensor): The [B, P] int64 detection level (block number) of every proposal.
        level_anchors (Dict[int, torch.Tensor]): The [A, 4] anchors of every detection level, keyed by block number.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The new [B, P, 4] proposals, their [B, P] levels and the
            [B, P] boolean mask of the slots holding a ground truth box, which the training target viewer
            ('view_training_targets') reads to tell these copies from the proposals, the input tensors being left
            unchanged.
    """
    padded_boxes, _, valid_ground_truths = pad_ground_truths(batched_ground_truth_boxes=batched_ground_truth_boxes)
    padded_boxes = padded_boxes.to(dtype=batched_proposals.dtype)
    number_proposals = batched_proposals.shape[1]

    # Objects written into every image, at most one per proposal slot, starting at its first written slot
    object_counts = valid_ground_truths.sum(dim=1, keepdim=True)
    written_counts = object_counts.clamp(max=number_proposals)
    first_written_slots = number_proposals - written_counts
    slots = torch.arange(end=number_proposals, device=batched_proposals.device).unsqueeze(dim=0)
    written_slots = slots >= first_written_slots

    # Object of every written slot, the last objects of an image having more objects than slots
    object_positions = (slots - first_written_slots + object_counts - written_counts).clamp(
        min=0, max=padded_boxes.shape[1] - 1)
    slot_boxes = torch.gather(input=padded_boxes, dim=1, index=object_positions.unsqueeze(dim=-1).expand(-1, -1, 4))

    # Best anchor overlap of every object in every level, of shape [B, G, levels], the levels in increasing order and
    # the objects matched in chunks (compute_best_anchor_overlaps)
    levels = sorted(level_anchors)
    best_level_overlaps = torch.stack(tensors=[compute_best_anchor_overlaps(
        padded_boxes=padded_boxes, anchors=level_anchors[level].to(dtype=padded_boxes.dtype)) for level in levels],
        dim=-1)

    # Level of the best anchor of every object, read for the object of every written slot
    level_numbers = torch.tensor(data=levels, dtype=torch.int64, device=batched_proposals.device)
    object_levels = level_numbers[best_level_overlaps.argmax(dim=-1)]
    slot_levels = torch.gather(input=object_levels, dim=1, index=object_positions)

    # Written slots take the ground truth boxes and their levels, the others keep their proposals
    proposals = torch.where(condition=written_slots.unsqueeze(dim=-1), input=slot_boxes, other=batched_proposals)
    proposal_levels = torch.where(condition=written_slots, input=slot_levels, other=batched_proposal_levels)
    return proposals, proposal_levels, written_slots


def batch_assign_targets_to_anchors(batched_ground_truth_boxes: List[torch.Tensor], batched_anchors: torch.Tensor,
                                    background_iou_threshold: Dict[str, float],
                                    foreground_iou_threshold: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assigns ground truth boxes to the anchors of every image of a batch to prepare the binary classification and
    regression targets for the Region Proposal Network.

    This function sits at the heart of the RPN training pipeline. It computes the IoU between every anchor and every
    ground truth box of its image, for all the images at once (the ground truths padded with empty boxes, which
    overlap nothing) and for chunks of objects (count_objects_per_chunk), so that a crowded image of hundreds of
    objects never holds all its overlaps at once, the targets being those of a single pass (match_objects_to_anchors,
    compiled on the GPU by compile_for_gpu with the rounding of the eager operations). It categorizes each
    anchor as foreground (positive object), background (negative), or grey zone (ignored during training). The best
    anchor of every ground truth box is also made positive, so that objects with no anchor above the foreground
    threshold still train, but it regresses to the box it overlaps the most, as in torchvision and Detectron2: a box
    whose best anchor overlaps another box more, or a box overlapping no anchor, gets no positive anchor. The anchors
    of an image without ground truth are all background.

    Args:
        batched_ground_truth_boxes (List[torch.Tensor]): A list of length batch_size, where each tensor contains the
            [Num_GT, 4] ground truth boxes of that image in [x_min, y_min, x_max, y_max] format.
        batched_anchors (torch.Tensor): A tensor of shape [batch_size, number_of_anchors, 4].
        background_iou_threshold (Dict[str, float]): Dictionary defining 'min' and 'max' IoU thresholds for background.
        foreground_iou_threshold (Dict[str, float]): Dictionary of the foreground IoU range, of which only 'min' is
            read: every anchor reaching it is foreground, 'max' being only checked by detection_checks.py.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - batched_target_boxes (torch.Tensor): The assigned ground truth box of each anchor, shape
              [batch_size, number_of_anchors, 4].
            - batched_labels (torch.Tensor): The float32 label of each anchor (1.0 = foreground, 0.0 = background,
              -1.0 = ignored), shape [batch_size, number_of_anchors].
    """
    padded_boxes, _, valid_ground_truths = pad_ground_truths(batched_ground_truth_boxes=batched_ground_truth_boxes)
    objects_per_chunk = count_objects_per_chunk(batch_size=padded_boxes.shape[0],
                                                number_anchors=batched_anchors.shape[-2])

    # The objects are matched to the anchors in chunks (count_objects_per_chunk), the best object of every anchor and
    # the anchors forced foreground being gathered over the chunks, so that a crowded batch never holds the overlaps of
    # all its objects with all the anchors at once; the matching of a chunk is fused on the GPU (compile_for_gpu)
    matching_function = compile_for_gpu(function=match_objects_to_anchors) if padded_boxes.is_cuda else \
        match_objects_to_anchors
    best_match_ground_truth_iou, best_match_ground_truth_index, resurrected_anchors = None, None, None
    for chunk_start in range(0, padded_boxes.shape[1], objects_per_chunk):
        chunk_end = chunk_start + objects_per_chunk

        # Best object of the chunk of every anchor with its overlap, and the anchors forced foreground by the chunk
        chunk_best_iou, chunk_best_index, chunk_resurrected_anchors = matching_function(
            chunk_boxes=padded_boxes[:, chunk_start:chunk_end],
            chunk_valid_ground_truths=valid_ground_truths[:, chunk_start:chunk_end], anchors=batched_anchors)

        # The first chunk starts the matches, a later one takes the anchors it overlaps strictly more, so that the
        # first best object stays on ties, as the max over all the objects at once returns the first one
        if best_match_ground_truth_iou is None:
            best_match_ground_truth_iou, best_match_ground_truth_index = chunk_best_iou, chunk_best_index
            resurrected_anchors = chunk_resurrected_anchors
            continue
        is_better_match = chunk_best_iou > best_match_ground_truth_iou
        best_match_ground_truth_iou = torch.where(condition=is_better_match, input=chunk_best_iou,
                                                  other=best_match_ground_truth_iou)
        best_match_ground_truth_index = torch.where(condition=is_better_match, input=chunk_best_index + chunk_start,
                                                    other=best_match_ground_truth_index)
        resurrected_anchors = resurrected_anchors | chunk_resurrected_anchors

    # Store a pristine copy of the best matches before we corrupt it with negative labels
    # It gives the box that the forced best anchors of the ground truths regress to
    best_match_ground_truth_index_before_thresholding = best_match_ground_truth_index.clone()

    # Create readable condition variables based on the IoU thresholds
    below_background_minimum = best_match_ground_truth_iou < background_iou_threshold['min']
    above_or_equal_background_minimum = best_match_ground_truth_iou >= background_iou_threshold['min']
    below_background_maximum = best_match_ground_truth_iou < background_iou_threshold['max']
    above_or_equal_background_maximum = best_match_ground_truth_iou >= background_iou_threshold['max']
    below_foreground_minimum = best_match_ground_truth_iou < foreground_iou_threshold['min']

    # Anchors that fall strictly within the background IoU limits are true negatives
    background_indices = above_or_equal_background_minimum & below_background_maximum

    # Anchors that fall between the maximum background limit and the minimum foreground limit,
    # or below the absolute minimum background limit, are ignored (grey zone)
    grey_zone_indices = (above_or_equal_background_maximum & below_foreground_minimum) | below_background_minimum

    # Label the background anchors -1 and the grey zone anchors -2, with masked fills that never wait for the GPU
    best_match_ground_truth_index = best_match_ground_truth_index.masked_fill(mask=background_indices, value=-1)
    best_match_ground_truth_index = best_match_ground_truth_index.masked_fill(mask=grey_zone_indices, value=-2)

    # We recover the positive labels for the anchors that had the highest IoU for each ground truth box.
    # IMPORTANT QUIRK: Notice that we are assigning the anchor to its absolute best match overall
    # (using best_match_ground_truth_index_before_thresholding) rather than strictly forcing it to
    # match the specific ground truth box that selected it. For example, if an anchor is forced
    # positive to cover Ground Truth A, but it actually overlaps more with Ground Truth B, it will
    # be assigned to predict Ground Truth B. This perfectly mirrors the official PyTorch RPN logic!
    best_match_ground_truth_index = torch.where(condition=resurrected_anchors,
                                                input=best_match_ground_truth_index_before_thresholding,
                                                other=best_match_ground_truth_index)

    # Get coordinates of best matching ground truth target boxes (so each anchor will have at least one target)
    # But we will not necessarily train on all the assigned targets because the -1 and -2 will be bogus targets.
    # The -1 and -2 are clamped to 0 to always have the same coordinates for background and grey zone, the empty
    # padding box for an image without ground truth
    target_indices = best_match_ground_truth_index.clamp(min=0).unsqueeze(dim=-1).expand(-1, -1, 4)
    batched_target_boxes = torch.gather(input=padded_boxes, dim=1, index=target_indices)

    # Now set all labels for training so there is no ambiguity
    # For classification loss we consider labels above 0, where 0 is background and 1 is foreground
    # For bounding box regression we only consider labels equal to 1 which are foreground
    # The -1 indices give 0 (background) and the -2 indices give -1 (ignored), shape [batch, anchors]
    batched_labels = (best_match_ground_truth_index >= 0).to(dtype=torch.float32)
    batched_labels = batched_labels.masked_fill(mask=best_match_ground_truth_index == -2, value=-1.0)

    # The anchors of an image without ground truth are all background, whatever the thresholds
    has_ground_truth = valid_ground_truths.any(dim=1, keepdim=True)
    batched_labels = torch.where(condition=has_ground_truth, input=batched_labels,
                                 other=torch.zeros_like(input=batched_labels))

    return batched_target_boxes, batched_labels


def turn_boxes_to_transformation_targets(ground_truth_boxes: torch.Tensor,
                                         predicted_boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes ideal regression targets (dx, dy, dw, dh) required to transform predicted boxes into ground truth boxes.

    In object detection pipelines, bounding box regression heads do not directly predict absolute coordinates.
    Instead, they predict parameterized offsets relative to a base box (like an anchor or a previous proposal).
    This function calculates those continuous targets by computing the scale-invariant center shifts (dx, dy)
    and the log-space dimension adjustments (dw, dh) so the network can safely regress them using Smooth L1 Loss.
    The offsets are multiplied by BOX_REGRESSION_WEIGHTS, which apply_regression_predictions divides out when decoding.

    Args:
        ground_truth_boxes (torch.Tensor): A tensor of shape (..., N, 4)
        containing the target bounding boxes in [x_min, y_min, x_max, y_max] format.
        predicted_boxes (torch.Tensor): A tensor of shape (..., N, 4)
        containing the base/reference bounding boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        torch.Tensor: A tensor of shape (..., N, 4) containing the regression targets [dx, dy, dw, dh].
    """

    # Get ground truth dimensions (Widths, Heights, Center X, Center Y)
    # Shapes for all returned variables: (..., N)
    (ground_truth_boxes_widths, ground_truth_boxes_heights,
     ground_truth_boxes_center_x, ground_truth_boxes_center_y) = get_box_dimensions_and_centers(
        boxes=ground_truth_boxes)

    # Get predicted (base) boxes dimensions (Widths, Heights, Center X, Center Y)
    # Shapes for all returned variables: (..., N)
    (predicted_boxes_widths, predicted_boxes_heights,
     predicted_boxes_center_x, predicted_boxes_center_y) = get_box_dimensions_and_centers(
        boxes=predicted_boxes)

    # Compute scale-invariant offsets for the center coordinates (with epsilon to prevent division by zero)
    # Shape: (..., N)
    target_dx = (ground_truth_boxes_center_x - predicted_boxes_center_x) / (predicted_boxes_widths + 1e-6)
    target_dy = (ground_truth_boxes_center_y - predicted_boxes_center_y) / (predicted_boxes_heights + 1e-6)

    # Compute log-space scale adjustments for width and height (clamped to prevent log(0) -> -inf)
    # Shape: (..., N)
    target_dw = torch.log(input=(ground_truth_boxes_widths / (predicted_boxes_widths + 1e-6)).clamp(min=1e-6))
    target_dh = torch.log(input=(ground_truth_boxes_heights / (predicted_boxes_heights + 1e-6)).clamp(min=1e-6))

    # Stack the individual parameterized targets into the final format [dx, dy, dw, dh], each scaled by its weight
    # Shape: (..., N, 4)
    regression_targets = torch.stack(tensors=[target_dx, target_dy, target_dw, target_dh], dim=-1)
    regression_targets = regression_targets * regression_targets.new_tensor(data=BOX_REGRESSION_WEIGHTS)

    return regression_targets


def sample_positive_and_negative_training_targets(labels: torch.Tensor, desired_positives: int,
                                                  desired_total: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly samples positive and negative targets (anchors or proposals) to maintain a fixed ratio during training.

    This utility is universally applicable to both the Region Proposal Network (sampling raw anchors) and the
    Detector Network (sampling NMS-filtered proposals). To prevent the classification loss from being completely
    overwhelmed by the massive number of background targets, this function constructs a balanced mini-batch
    for each image. It guarantees up to `desired_positives` foreground targets, and fills the remainder
    of the `desired_total` quota with background targets, more positives filling the slots of an image short of
    negatives. Every image is sampled at once: a single sort of random keys gives every target a random rank among
    the targets of its kind in its image, and the targets ranked below the count of their kind are kept, a uniform
    draw without replacement like a random permutation of every kind.

    Args:
        labels (torch.Tensor): A batched tensor of shape (B, N) containing target labels
                               (Values >= 1.0 = positive foreground, 0.0 = negative background).
        desired_positives (int): The maximum number of positive targets to sample per image.
        desired_total (int): The total combined number of targets (positive + negative) to sample per image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two boolean masks of shape (B, N):
            - sampled_positive_mask (torch.Tensor): True for selected positive targets.
            - sampled_negative_mask (torch.Tensor): True for selected negative targets.
    """
    is_positive = labels >= 1.0
    is_negative = labels == 0.0
    number_positives = is_positive.sum(dim=1)
    number_negatives = is_negative.sum(dim=1)

    # Calculate the actual number of positives of every image, capped by availability, and fill the remaining
    # slots with negatives, capped by availability
    possible_positives = number_positives.clamp(max=desired_positives)
    desired_negatives = desired_total - possible_positives
    possible_negatives = torch.minimum(input=number_negatives, other=desired_negatives)

    # CRITICAL FIX: If an image ran out of negatives, we must over-sample positives to maintain
    # a uniform number of samples across the batch!
    possible_positives = torch.minimum(input=number_positives,
                                       other=possible_positives + desired_negatives - possible_negatives)

    # Random rank of every target among the targets of its kind in its image: the random keys offset by kind sort the
    # positives first, then the negatives, then the ignored targets, each kind in a random order
    kind_offsets = (~is_positive).to(dtype=torch.float32) + (~is_positive & ~is_negative).to(dtype=torch.float32)
    random_order = torch.argsort(input=torch.rand(size=labels.shape, device=labels.device) + kind_offsets, dim=1)
    positions = torch.arange(end=labels.shape[1], device=labels.device).expand_as(random_order)
    ranks = torch.empty_like(input=random_order).scatter_(dim=1, index=random_order, src=positions)

    # Keep the positives ranked below their count, and the negatives ranked below theirs after the positives
    sampled_positive_mask = is_positive & (ranks < possible_positives.unsqueeze(dim=1))
    sampled_negative_mask = is_negative & (ranks - number_positives.unsqueeze(dim=1)
                                           < possible_negatives.unsqueeze(dim=1))

    return sampled_positive_mask, sampled_negative_mask


def batch_assign_targets_to_proposals(batched_ground_truth_boxes: List[torch.Tensor],
                                      batched_proposals: torch.Tensor,
                                      batched_ground_truth_labels: List[torch.Tensor], foreground_iou_threshold: float
                                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assigns ground truth bounding boxes and class labels to the region proposals of every image of a batch for
    Detection Head training.

    This function matches the predicted region proposals (from the Region Proposal Network) against the actual ground
    truth boxes of their image using Intersection over Union (IoU), for all the images at once (the ground truths
    padded with empty boxes, which overlap nothing). Proposals with an IoU greater than or equal to the foreground
    threshold are assigned the corresponding ground truth class. Proposals falling below the threshold, and all the
    proposals of an image without ground truth, are assigned the background class (class index 0). The index of the
    matched ground truth is returned too, so that the instance segmenter picks the ground truth mask of every positive
    proposal.

    Args:
        batched_ground_truth_boxes (List[torch.Tensor]): A list of length batch_size, where each tensor contains the
            [Num_GT, 4] ground truth boxes of that image in [x_min, y_min, x_max, y_max] format.
        batched_proposals (torch.Tensor): The proposed bounding boxes of every image, shape
            [batch_size, number_of_proposals, 4].
        batched_ground_truth_labels (List[torch.Tensor]): A list of length batch_size, containing the true class
            indices for the ground truth boxes of each image.
        foreground_iou_threshold (float): The minimum IoU required for a proposal to be considered foreground.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - batched_target_boxes (torch.Tensor): The assigned ground truth coordinates.
             Shape [batch_size, number_of_proposals, 4]
            - batched_labels (torch.Tensor): The assigned int64 class labels (0 for background).
             Shape [batch_size, number_of_proposals]
            - batched_matched_ground_truth_indices (torch.Tensor): The int64 index of the matched ground truth within
             its image, -1 for the background. Shape [batch_size, number_of_proposals]
    """
    padded_boxes, padded_labels, valid_ground_truths = pad_ground_truths(
        batched_ground_truth_boxes=batched_ground_truth_boxes, batched_ground_truth_labels=batched_ground_truth_labels)

    # Shape: [batch_size, number_of_ground_truths, number_of_proposals], the padding rows overlapping nothing
    intersection_over_union_matrix = get_intersection_over_union(boxes_1=padded_boxes, boxes_2=batched_proposals)
    intersection_over_union_matrix = intersection_over_union_matrix.masked_fill(
        mask=~valid_ground_truths.unsqueeze(dim=-1), value=0.0)

    # For every proposal, find the ground truth box that has the highest IoU with it
    best_match_ground_truth_iou, best_match_ground_truth_index = intersection_over_union_matrix.max(dim=1)

    # Proposals below the foreground threshold, and those of an image without ground truth, are background
    # They will never be considered for the regression loss, and keep the index -1 of their match
    background_indices = ((best_match_ground_truth_iou < foreground_iou_threshold)
                          | ~valid_ground_truths.any(dim=1, keepdim=True))
    best_match_ground_truth_index = best_match_ground_truth_index.masked_fill(mask=background_indices, value=-1)

    # Get coordinates of best matching ground truth target boxes (so each proposal will have at least one target)
    # But we will not necessarily train on all the assigned targets because the -1 are bogus targets.
    # The -1 are clamped to 0 to always have the same coordinates for background
    target_indices = best_match_ground_truth_index.clamp(min=0)
    batched_target_boxes = torch.gather(input=padded_boxes, dim=1,
                                        index=target_indices.unsqueeze(dim=-1).expand(-1, -1, 4))

    # For classification loss we get the labels of the best iou ground truth match for each proposal, the
    # background proposals getting the class 0
    # For bounding box regression we only consider labels larger than 0 which are foreground
    batched_target_labels = torch.gather(input=padded_labels, dim=1, index=target_indices)
    batched_target_labels = batched_target_labels.masked_fill(mask=background_indices, value=0)

    return batched_target_boxes, batched_target_labels, best_match_ground_truth_index
