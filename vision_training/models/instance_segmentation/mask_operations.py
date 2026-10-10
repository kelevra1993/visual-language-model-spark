"""
Mask operations of the instance segmentation task: the mask targets of the positive proposals cropped from the ground
truth masks, the pasting of the M x M predicted masks into the image inside their boxes, and the intersection over
union of binary masks. Used by the InstanceSegmenter for its mask loss, and by the instance segmentation trainer for
the mask evaluation and the sample images; the clients paste the exported masks the same way.
"""

import torch

from typing import List
from torchvision.ops import roi_align


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


def compute_mask_intersection_over_union(predicted_masks: torch.Tensor, ground_truth_masks: torch.Tensor,
                                         chunk_size: int) -> torch.Tensor:
    """
    Computes the intersection over union of every pair of predicted and ground truth binary masks, which replaces the
    box overlap in the mask average precision of the instance segmentation evaluation. The intersections are products
    of the flattened masks on their device, a few predicted masks at a time to bound the memory.

    Args:
        predicted_masks (torch.Tensor): The [D, H, W] binary predicted masks.
        ground_truth_masks (torch.Tensor): The [G, H, W] binary ground truth masks.
        chunk_size (int): The number of predicted masks compared at once.

    Returns:
        torch.Tensor: The [D, G] float32 intersection over union, 0 for two empty masks.
    """
    flattened_ground_truths = ground_truth_masks.flatten(start_dim=1).to(dtype=torch.float32)
    ground_truth_areas = flattened_ground_truths.sum(dim=1)
    overlaps = []

    # Intersections of a chunk of predicted masks with every ground truth mask, as one matrix product
    for chunk_start in range(0, predicted_masks.shape[0], chunk_size):
        flattened_predictions = predicted_masks[chunk_start:chunk_start + chunk_size].flatten(start_dim=1).to(
            dtype=torch.float32)
        intersections = flattened_predictions @ flattened_ground_truths.T
        unions = flattened_predictions.sum(dim=1)[:, None] + ground_truth_areas[None, :] - intersections
        overlaps.append(torch.where(unions > 0, intersections / unions.clamp(min=1.0), torch.zeros_like(unions)))

    # Keep the [0, G] layout without predicted masks
    if len(overlaps) == 0:
        return torch.zeros(size=(0, ground_truth_masks.shape[0]), dtype=torch.float32, device=ground_truth_masks.device)
    return torch.cat(tensors=overlaps, dim=0)
