"""
Losses of the bottom-up keypoint detection task (HigherHRNet), used by the keypoint detector
(models/keypoints/keypoint_detector.py) on the targets of data/keypoints/heatmap_targets.py: the focal loss of the
predicted heatmaps against the target Gaussian heatmaps at every stride (CornerNet, CenterNet), and the associative
embedding losses of the tags, which pull the tags of the keypoints of an object towards their mean and push apart the
mean tags of the objects of the same class in the same image, so that the keypoints can be grouped into objects by
their tags. The base trainer adds their 'heatmap_loss', 'tag_pull_loss' and 'tag_push_loss' to the total loss.
"""

import torch

from torch.nn import functional as torch_functional
from typing import Tuple

# Exponent of the focal loss lowering the weight of the cells already predicted well, 2 in CornerNet and CenterNet
FOCAL_EXPONENT = 2.0

# Exponent of the penalty reduction of the negative cells near a keypoint, (1 - target) ** 4 in CornerNet and CenterNet
PENALTY_REDUCTION_EXPONENT = 4.0


def compute_heatmap_loss(predicted_logits: torch.Tensor, target_heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Computes the heatmap loss of the keypoint detector at one stride, the penalty-reduced focal loss of CornerNet
    ("CornerNet: Detecting Objects as Paired Keypoints" https://arxiv.org/abs/1808.01244) and CenterNet ("Objects as
    Points" https://arxiv.org/abs/1904.07850). With p the sigmoid of a predicted logit and y its target, a cell at 1
    (the cell containing a keypoint) gives -(1 - p) ** 2 log(p), and any other cell -(1 - y) ** 4 p ** 2 log(1 - p),
    the cells near a keypoint being penalised less and the cells already near 0 hardly at all. The sum over the cells
    and channels of all the images is divided by the number of cells at 1, the trained keypoints, so that neither the
    number of channels nor the size of the heatmaps dilutes the keypoints among the many empty cells (a batch without
    keypoint being divided by 1). Every channel is trained in every image, the channels without keypoint towards 0. It
    is computed in float32 from the logits, log(p) and log(1 - p) being the log-sigmoids of the logit and of its
    opposite.

    Args:
        predicted_logits (torch.Tensor): The [B, C, h, w] predicted heatmap logits, before the sigmoid.
        target_heatmaps (torch.Tensor): The [B, C, h, w] target Gaussian heatmaps of build_keypoint_targets, 1 at the
            cell of every keypoint.

    Returns:
        torch.Tensor: The scalar loss.
    """
    logits = predicted_logits.to(dtype=torch.float32)
    targets = target_heatmaps.to(dtype=torch.float32)
    probabilities = torch.sigmoid(logits)
    positive_cells = targets == 1.0

    # Focal terms of the keypoint cells and of the other cells, the latter reduced near the keypoints
    positive_losses = -(1.0 - probabilities) ** FOCAL_EXPONENT * torch_functional.logsigmoid(logits)
    negative_losses = (-(1.0 - targets) ** PENALTY_REDUCTION_EXPONENT * probabilities ** FOCAL_EXPONENT *
                       torch_functional.logsigmoid(-logits))
    cell_losses = torch.where(positive_cells, positive_losses, negative_losses)

    # Sum over every cell, normalised by the number of keypoint cells
    number_positives = positive_cells.sum().clamp(min=1).to(dtype=torch.float32)
    return cell_losses.sum() / number_positives


def compute_tag_losses(tag_maps: torch.Tensor, tag_indices: torch.Tensor, tag_valid: torch.Tensor,
                       object_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the associative embedding losses of the keypoint detector, in float32. The tag of every trained keypoint
    is read in the tag map of its channel at its cell, and the reference tag of an object is the mean tag of its
    trained keypoints. The pull loss of an image is the mean over its objects of the mean squared distance of their
    keypoint tags to their reference tag; the push loss of an image is the mean of exp(-(h_m - h_n) ** 2) over the
    ordered pairs of distinct objects m, n of the same class, whose keypoints share the same channels, the objects of
    different classes never competing for a channel. Both are averaged over the images of the batch, an image without
    object (or without pair) giving 0.

    Args:
        tag_maps (torch.Tensor): The [B, C, h, w] predicted tags at the tag stride.
        tag_indices (torch.Tensor): The [B, M, K] int64 flat index of every keypoint in the [C x h x w] tag map of its
            image.
        tag_valid (torch.Tensor): The [B, M, K] booleans of the trained keypoints.
        object_labels (torch.Tensor): The [B, M] int64 class of every object, 0 for the padding objects.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The scalar pull and push losses.
    """
    batch_size, maximum_objects, maximum_keypoints = tag_indices.shape
    flat_tags = tag_maps.to(dtype=torch.float32).reshape(batch_size, -1)
    keypoint_tags = flat_tags.gather(dim=1, index=tag_indices.reshape(batch_size, -1)).reshape(
        batch_size, maximum_objects, maximum_keypoints)
    valid = tag_valid.to(dtype=torch.float32)

    # Reference tag of every object, the mean tag of its trained keypoints, the objects without any being absent
    keypoint_counts = valid.sum(dim=2)
    present = keypoint_counts > 0
    reference_tags = (keypoint_tags * valid).sum(dim=2) / keypoint_counts.clamp(min=1.0)

    # Pull loss: squared distance of the keypoint tags to the reference tag of their object, averaged over its
    # keypoints, then over the objects of the image
    object_pull = ((keypoint_tags - reference_tags[..., None]) ** 2 * valid).sum(dim=2) / keypoint_counts.clamp(min=1.0)
    present_objects = present.to(dtype=torch.float32)
    pull_losses = (object_pull * present_objects).sum(dim=1) / present_objects.sum(dim=1).clamp(min=1.0)

    # Push loss: every ordered pair of distinct present objects of the same class
    distinct_objects = ~torch.eye(n=maximum_objects, device=tag_maps.device, dtype=torch.bool)
    pairs = (present[:, :, None] & present[:, None, :] & distinct_objects[None] &
             (object_labels[:, :, None] == object_labels[:, None, :])).to(dtype=torch.float32)
    tag_differences = reference_tags[:, :, None] - reference_tags[:, None, :]
    push_losses = (torch.exp(-tag_differences ** 2) * pairs).sum(dim=(1, 2)) / pairs.sum(dim=(1, 2)).clamp(min=1.0)
    return pull_losses.mean(), push_losses.mean()
