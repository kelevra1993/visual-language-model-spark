"""
Losses of the semantic segmentation task: the pixel cross entropy (ignored pixels, label smoothing, class weights and
an optional focal factor) and the soft Dice loss over the classes present in the batch. Used by the semantic
segmenter, which multiplies them by their weights of the 'SegmentationLoss' section.
"""

import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX


def compute_segmentation_cross_entropy(class_logits: torch.Tensor, masks: torch.Tensor, label_smoothing: float,
                                       class_weights: torch.Tensor, focal_gamma: float) -> torch.Tensor:
    """
    Computes the pixel cross entropy of the segmentation logits, the main loss of the semantic segmenter: the pixels
    at IGNORED_LABEL_INDEX (padding, unlabelled pixels) give no loss, the classes are weighted by 'class_weights'
    (e.g. the balanced weights of rare classes), and the focal factor (1 - p_t) ** focal_gamma lowers the loss of the
    pixels already well classified, so that the hard pixels (boundaries, small objects) weigh more. The loss is
    normalised by the weights of the labelled pixels, so that a focal gamma of 0 gives exactly the weighted cross
    entropy. It is computed in float32, stable under mixed precision.

    Args:
        class_logits (torch.Tensor): The [B, K, H, W] class logits.
        masks (torch.Tensor): The [B, H, W] int64 class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.
        label_smoothing (float): The label smoothing, 0 for none.
        class_weights (torch.Tensor): The [K] weight of every class in the loss.
        focal_gamma (float): The focusing parameter of the focal factor, 0 for the plain cross entropy.

    Returns:
        torch.Tensor: The scalar loss, 0 for a batch without labelled pixel.
    """
    class_logits = class_logits.float()
    class_weights = class_weights.float()

    # Weighted and smoothed cross entropy of every pixel, 0 on the ignored pixels
    pixel_losses = torch.nn.functional.cross_entropy(input=class_logits, target=masks, weight=class_weights,
                                                     ignore_index=IGNORED_LABEL_INDEX,
                                                     label_smoothing=label_smoothing, reduction="none")

    # Weight of every labelled pixel, the class weight of its class, normalising the loss like the weighted mean
    labelled_pixels = masks != IGNORED_LABEL_INDEX
    target_classes = masks.clamp(min=0)
    pixel_weights = class_weights[target_classes] * labelled_pixels

    # Lower the loss of the well classified pixels, from the probability of their true class
    if focal_gamma > 0:
        target_probabilities = torch.softmax(input=class_logits, dim=1).gather(
            dim=1, index=target_classes.unsqueeze(dim=1)).squeeze(dim=1)
        pixel_losses = pixel_losses * (1.0 - target_probabilities) ** focal_gamma

    return pixel_losses.sum() / pixel_weights.sum().clamp(min=torch.finfo(torch.float32).tiny)


def compute_dice_loss(class_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Computes the soft Dice loss of the segmentation logits, the loss of the semantic segmenter measuring the overlap
    of the predicted and true regions of every class, which keeps the small objects and rare classes as important as
    the large ones: one minus the mean Dice coefficient (2 x intersection / (predicted + true pixels)) of the softmax
    probabilities, over the whole batch, for every class present in its labelled pixels. The ignored pixels are left
    out, and the absent classes are left out of the mean, their Dice being undefined. It is computed in float32,
    stable under mixed precision.

    Args:
        class_logits (torch.Tensor): The [B, K, H, W] class logits.
        masks (torch.Tensor): The [B, H, W] int64 class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.

    Returns:
        torch.Tensor: The scalar loss, 0 for a batch without labelled pixel.
    """
    number_classes = class_logits.shape[1]
    labelled_pixels = (masks != IGNORED_LABEL_INDEX).unsqueeze(dim=1)

    # Probabilities and one-hot targets of the labelled pixels only, [B, K, H, W]
    probabilities = torch.softmax(input=class_logits.float(), dim=1) * labelled_pixels
    targets = torch.nn.functional.one_hot(masks.clamp(min=0), num_classes=number_classes).permute(0, 3, 1, 2)
    targets = targets.float() * labelled_pixels

    # Dice coefficient of every class over the batch, averaged over the classes with true pixels
    intersections = (probabilities * targets).sum(dim=(0, 2, 3))
    true_pixels = targets.sum(dim=(0, 2, 3))
    cardinalities = probabilities.sum(dim=(0, 2, 3)) + true_pixels
    present_classes = true_pixels > 0
    if not bool(present_classes.any()):
        return class_logits.sum() * 0.0
    dice_coefficients = 2.0 * intersections[present_classes] / cardinalities[present_classes]
    return 1.0 - dice_coefficients.mean()
