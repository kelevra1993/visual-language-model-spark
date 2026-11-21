"""
Mask operations of the instance segmentation task: the mask targets of the positive proposals cropped from the ground
truth masks, the targets of the mask IoU head and the mask scores of Mask Scoring R-CNN ('MaskHead.mask_scoring'), the
pasting of the M x M predicted masks into the image inside their boxes, and the intersection over union of binary
masks. Used by the InstanceSegmenter for its mask and mask IoU losses and the mask scores of its detections, by the
InferenceInstanceSegmenter of the exports for the same mask scores, and by the instance segmentation trainer for the
mask evaluation and the sample images; the clients paste the exported masks the same way.
"""

import torch

from typing import List
from torchvision.ops import roi_align

# Pixels of the segments the flattened masks are cut into before the products of their intersections: a partial count
# never exceeds a segment, below 2048, the largest whole number float16 holds exactly, so the products are exact in
# float16; one product over a whole image (a few masks of a million pixels) would leave most of the GPU idle
MASK_PRODUCT_SEGMENT_PIXELS = 1024


def build_mask_targets(ground_truth_masks: List[torch.Tensor], proposal_boxes: torch.Tensor,
                       matched_ground_truth_indices: torch.Tensor, mask_size: int) -> torch.Tensor:
    """
    Crops the ground truth mask of every positive proposal to the proposal box, at the resolution of the predicted
    masks, giving the targets of the mask loss: RoIAlign (aligned, like the mask heads) of the matched ground truth
    mask of every proposal, binarised at 0.5 as in Detectron2.

    Args:
        ground_truth_masks (List[torch.Tensor]): The [N_i, H, W] binary masks of every image of the batch.
        proposal_boxes (torch.Tensor): The [R, 5] positive proposals, the image index of every proposal followed by its
            [x1, y1, x2, y2] corners in network input pixels.
        matched_ground_truth_indices (torch.Tensor): The [R] index of the matched ground truth of every proposal
            within its image.
        mask_size (int): The side M of the predicted masks.

    Returns:
        torch.Tensor: The [R, M, M] targets, 1 on the object and 0 elsewhere, in the dtype of the proposals.
    """
    mask_targets = torch.zeros(size=(proposal_boxes.shape[0], mask_size, mask_size), dtype=proposal_boxes.dtype,
                               device=proposal_boxes.device)

    # One image at a time, its masks being the batch of RoIAlign and the matched index the batch column of every box
    for image_index, image_masks in enumerate(ground_truth_masks):
        proposal_indices = torch.nonzero(input=proposal_boxes[:, 0] == image_index).flatten()
        if proposal_indices.numel() == 0:
            continue
        image_boxes = torch.cat(tensors=[matched_ground_truth_indices[proposal_indices].unsqueeze(dim=1).to(
            dtype=proposal_boxes.dtype), proposal_boxes[proposal_indices, 1:]], dim=1)
        cropped_masks = roi_align(input=image_masks.unsqueeze(dim=1).to(dtype=proposal_boxes.dtype),
                                  boxes=image_boxes, output_size=(mask_size, mask_size), spatial_scale=1.0,
                                  sampling_ratio=0, aligned=True)
        mask_targets[proposal_indices] = (cropped_masks[:, 0] >= 0.5).to(dtype=proposal_boxes.dtype)
    return mask_targets


def build_mask_iou_targets(predicted_masks: torch.Tensor, mask_targets: torch.Tensor,
                           ground_truth_masks: List[torch.Tensor], proposal_boxes: torch.Tensor,
                           matched_ground_truth_indices: torch.Tensor) -> torch.Tensor:
    """
    Gives the target of the mask IoU head for every positive proposal, the intersection over union of its binary
    predicted mask with its object, as in Mask Scoring R-CNN (maskiou targets of its official code): the intersection
    is counted on the M x M grid of the proposal (the predicted mask against the target cropped by build_mask_targets),
    while the union counts the whole object, its area in image pixels projected onto the grid of the proposal (M x M
    cells for the area of the box), so that the part of the object outside the proposal, which its mask cannot cover,
    lowers the IoU as it lowers the IoU of the pasted masks of the evaluation. The targets are detached.

    Args:
        predicted_masks (torch.Tensor): The [R, M, M] binary predicted masks of the ground truth class of every
            proposal, thresholded at the mask threshold.
        mask_targets (torch.Tensor): The [R, M, M] binary targets of every proposal (build_mask_targets).
        ground_truth_masks (List[torch.Tensor]): The [N_i, H, W] binary masks of every image of the batch.
        proposal_boxes (torch.Tensor): The [R, 5] positive proposals, the image index of every proposal followed by its
            [x1, y1, x2, y2] corners in network input pixels.
        matched_ground_truth_indices (torch.Tensor): The [R] index of the matched ground truth of every proposal
            within its image.

    Returns:
        torch.Tensor: The [R] float32 targets within [0, 1], 0 when the predicted mask and the object are both empty.
    """
    mask_size = mask_targets.shape[-1]
    predicted_masks, mask_targets = predicted_masks.detach().float(), mask_targets.detach().float()
    intersections = (predicted_masks * mask_targets).flatten(start_dim=1).sum(dim=1)
    predicted_areas = predicted_masks.flatten(start_dim=1).sum(dim=1)
    object_areas = torch.zeros(size=(proposal_boxes.shape[0],), dtype=torch.float32, device=proposal_boxes.device)

    # Pixels of the matched object of every proposal, one image at a time
    for image_index, image_masks in enumerate(ground_truth_masks):
        proposal_indices = torch.nonzero(input=proposal_boxes[:, 0] == image_index).flatten()
        if proposal_indices.numel() == 0:
            continue
        image_object_areas = (image_masks > 0).flatten(start_dim=1).sum(dim=1).float()
        object_areas[proposal_indices] = image_object_areas[matched_ground_truth_indices[proposal_indices]]

    # Object area in cells of the grid of the proposal, at least its cropped target against the rounding of the grid
    boxes = proposal_boxes[:, 1:].detach().float()
    box_areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=1e-6)
    object_cells = torch.maximum(object_areas * mask_size * mask_size / box_areas,
                                 mask_targets.flatten(start_dim=1).sum(dim=1))
    unions = predicted_areas + object_cells - intersections
    return torch.where(unions > 0, intersections / unions.clamp(min=1e-6), torch.zeros_like(input=unions))


def compute_mask_scores(detection_scores: torch.Tensor, predicted_mask_iou: torch.Tensor) -> torch.Tensor:
    """
    Gives the mask score of every detection, its class score times its predicted mask IoU as in Mask Scoring R-CNN,
    which ranks the masks of the mask average precision and is exported with the masks, the box scores still ranking
    the boxes. The predicted IoU is clipped to [0, 1], the head being unbounded, so that the mask scores stay within
    [0, 1] like the class scores and their thresholds.

    Args:
        detection_scores (torch.Tensor): The [B, K] class scores of the detections.
        predicted_mask_iou (torch.Tensor): The [B, K] predicted mask IoU of the detected class of every detection.

    Returns:
        torch.Tensor: The [B, K] mask scores, in the dtype of the class scores.
    """
    return detection_scores * predicted_mask_iou.clamp(min=0.0, max=1.0).to(dtype=detection_scores.dtype)


def build_interpolation_weights(box_starts: torch.Tensor, box_ends: torch.Tensor, mask_size: int,
                                output_size: int) -> torch.Tensor:
    """
    Gives the linear interpolation weights placing the M cells of every mask along one axis of the image, so that
    paste_instance_masks resamples the masks separably along both axes: every image pixel centre is mapped into its
    box, onto the mask cell centres, and is interpolated between its two nearest cells, with 0 beyond the first and
    last cells (the sampling of grid_sample with aligned corners off and zero padding).

    Args:
        box_starts (torch.Tensor): The [K] start of every box along the axis, in pixels.
        box_ends (torch.Tensor): The [K] end of every box along the axis, in pixels.
        mask_size (int): The number M of mask cells along the axis.
        output_size (int): The number of image pixels along the axis.

    Returns:
        torch.Tensor: The [K, output_size, M] weights.
    """
    pixel_centres = torch.arange(output_size, device=box_starts.device, dtype=box_starts.dtype) + 0.5
    box_lengths = (box_ends - box_starts).clamp(min=1e-6)

    # Position of every pixel centre in mask cells, the centre of cell j lying at j
    positions = (pixel_centres[None, :] - box_starts[:, None]) / box_lengths[:, None] * mask_size - 0.5

    # Tent weights of every cell, 1 at its centre and 0 one cell away, so that outside the mask every weight is 0
    cell_indices = torch.arange(mask_size, device=box_starts.device, dtype=box_starts.dtype)
    return (1.0 - (positions[:, :, None] - cell_indices[None, None, :]).abs()).clamp(min=0.0)


def paste_instance_masks(masks: torch.Tensor, boxes: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    """
    Pastes the M x M mask of every detection into the image inside its box, the masks being resampled at the image
    pixel centres (bilinear inside the box, falling to 0 just outside it), as Detectron2 pastes its masks. Used for
    the mask evaluation and the sample images at the network resolution, and by the clients on the original images
    with the boxes restored to them.

    Args:
        masks (torch.Tensor): The [K, M, M] mask probabilities.
        boxes (torch.Tensor): The [K, 4] [x1, y1, x2, y2] boxes in the pixels of the target image.
        image_height (int): The height of the target image.
        image_width (int): The width of the target image.

    Returns:
        torch.Tensor: The [K, image_height, image_width] mask probabilities, 0 outside the boxes.
    """
    mask_size = masks.shape[-1]
    working_dtype = torch.float64 if masks.dtype == torch.float64 else torch.float32
    boxes = boxes.to(dtype=working_dtype)

    # Separable resampling: rows of the mask placed along the image height, then its columns along the image width
    row_weights = build_interpolation_weights(box_starts=boxes[:, 1], box_ends=boxes[:, 3], mask_size=mask_size,
                                              output_size=image_height)
    column_weights = build_interpolation_weights(box_starts=boxes[:, 0], box_ends=boxes[:, 2], mask_size=mask_size,
                                                 output_size=image_width)
    return torch.bmm(input=torch.bmm(input=row_weights, mat2=masks.to(dtype=working_dtype)),
                     mat2=column_weights.transpose(1, 2))


def cut_mask_segments(masks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Cuts every flattened binary mask into segments of MASK_PRODUCT_SEGMENT_PIXELS pixels, the last one padded with
    zeros, for the segment products of compute_mask_intersection_over_union in the mask evaluation, so that no partial
    pixel count of a product exceeds a segment.

    Args:
        masks (torch.Tensor): The [K, H, W] binary masks.
        dtype (torch.dtype): The dtype of the products.

    Returns:
        torch.Tensor: The [K, S, MASK_PRODUCT_SEGMENT_PIXELS] masks cut into S segments, in the given dtype.
    """
    flattened_masks = masks.flatten(start_dim=1).to(dtype=dtype)
    number_segments = -(-flattened_masks.shape[1] // MASK_PRODUCT_SEGMENT_PIXELS)
    padding = number_segments * MASK_PRODUCT_SEGMENT_PIXELS - flattened_masks.shape[1]

    # Pad only images whose pixels do not fill whole segments (a 1024 x 1024 image fills them)
    if padding > 0:
        flattened_masks = torch.nn.functional.pad(input=flattened_masks, pad=(0, padding))
    return flattened_masks.reshape(masks.shape[0], number_segments, MASK_PRODUCT_SEGMENT_PIXELS)


def count_segment_pixels(mask_segments: torch.Tensor) -> torch.Tensor:
    """
    Counts the pixels of every mask cut by cut_mask_segments, for the unions of compute_mask_intersection_over_union:
    every segment is summed in float32 (exact, a segment holding fewer pixels than float32 counts exactly), then the
    segments in float64, so that the count of a mask stays exact at any image size.

    Args:
        mask_segments (torch.Tensor): The [K, S, MASK_PRODUCT_SEGMENT_PIXELS] binary mask segments.

    Returns:
        torch.Tensor: The [K] float64 pixel counts of the masks.
    """
    return mask_segments.sum(dim=2, dtype=torch.float32).to(dtype=torch.float64).sum(dim=1)


def compute_mask_intersection_over_union(predicted_masks: torch.Tensor, ground_truth_masks: torch.Tensor,
                                         chunk_size: int) -> torch.Tensor:
    """
    Computes the intersection over union of every pair of predicted and ground truth binary masks, which replaces the
    box overlap in the mask average precision of the instance segmentation evaluation. The masks are cut into segments
    of MASK_PRODUCT_SEGMENT_PIXELS pixels (cut_mask_segments), and every segment of a few predicted masks (bounding the
    memory) is multiplied with the same segment of every ground truth mask in one batched product on their device (the
    mask part of the evaluation of a 1024 x 1024 image with 100 detections taking 16 ms on the GPU instead of 96 with
    one product over the whole image). The partial counts stay below a segment, exact in float16 on the GPU (float32 on
    the CPU, where float16 products are slow); they are summed and their ratio taken in float64, so that an exact ratio
    is compared with the thresholds like the box overlaps (13 / 20 in float32 falls below 0.65).

    Args:
        predicted_masks (torch.Tensor): The [D, H, W] binary predicted masks.
        ground_truth_masks (torch.Tensor): The [G, H, W] binary ground truth masks.
        chunk_size (int): The number of predicted masks compared at once.

    Returns:
        torch.Tensor: The [D, G] float64 intersection over union, 0 for two empty masks.
    """
    product_dtype = torch.float16 if ground_truth_masks.device.type == "cuda" else torch.float32
    ground_truth_segments = cut_mask_segments(masks=ground_truth_masks, dtype=product_dtype)
    ground_truth_areas = count_segment_pixels(mask_segments=ground_truth_segments)
    ground_truth_segments = ground_truth_segments.permute(1, 2, 0)
    overlaps = []

    # Intersections of a chunk of predicted masks with every ground truth mask, one product per segment, summed
    for chunk_start in range(0, predicted_masks.shape[0], chunk_size):
        prediction_segments = cut_mask_segments(masks=predicted_masks[chunk_start:chunk_start + chunk_size],
                                                dtype=product_dtype)
        intersections = torch.bmm(input=prediction_segments.transpose(0, 1), mat2=ground_truth_segments).to(
            dtype=torch.float64).sum(dim=0)
        prediction_areas = count_segment_pixels(mask_segments=prediction_segments)
        unions = prediction_areas[:, None] + ground_truth_areas[None, :] - intersections
        overlaps.append(torch.where(unions > 0, intersections / unions.clamp(min=1.0), torch.zeros_like(unions)))

    # Keep the [0, G] layout without predicted masks
    if len(overlaps) == 0:
        return torch.zeros(size=(0, ground_truth_masks.shape[0]), dtype=torch.float64, device=ground_truth_masks.device)
    return torch.cat(tensors=overlaps, dim=0)
