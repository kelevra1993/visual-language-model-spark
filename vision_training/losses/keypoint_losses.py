"""
Loss of the keypoint detection task: the mean squared error between the predicted and the target Gaussian heatmaps of
the keypoints of every crop, weighted by the keypoint weights of the crop transform (labelled and inside the crop).
Used by the keypoint detector (models/keypoints/keypoint_detector.py), whose 'keypoint_loss' the base trainer adds to
the total loss.
"""

import torch


def compute_keypoint_heatmap_loss(predicted_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor,
                                  heatmap_weights: torch.Tensor) -> torch.Tensor:
    """
    Computes the heatmap loss of the keypoint detector on the channels of the class of every crop: the mean squared
    error of every heatmap over its cells, averaged over the trained keypoints, the unlabelled keypoints, the ones
    outside the crop and the padding positions (weight 0) giving no gradient. It is computed in float32 since the small
    differences of the Gaussian tails vanish in bfloat16.

    Args:
        predicted_heatmaps (torch.Tensor): The [B, K, h, w] predicted heatmaps, in the keypoint order of the class.
        target_heatmaps (torch.Tensor): The [B, K, h, w] target Gaussian heatmaps of build_heatmap_targets.
        heatmap_weights (torch.Tensor): The [B, K] weight of every keypoint, 1 for the trained ones and 0 otherwise.

    Returns:
        torch.Tensor: The scalar loss, 0 for a batch without any trained keypoint.
    """
    # Mean squared error of every heatmap over its cells
    squared_errors = (predicted_heatmaps.to(dtype=torch.float32) - target_heatmaps.to(dtype=torch.float32)) ** 2
    heatmap_errors = squared_errors.mean(dim=(2, 3))

    # Average over the trained keypoints of the batch
    heatmap_weights = heatmap_weights.to(dtype=torch.float32)
    return (heatmap_errors * heatmap_weights).sum() / heatmap_weights.sum().clamp(min=1.0)
