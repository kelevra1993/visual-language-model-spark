"""
Losses of the semantic segmentation task: the pixel cross entropy (ignored pixels, label smoothing, class weights and
an optional focal factor), the soft Dice loss and the Lovász-Softmax loss over the classes present in the batch, and
the boundary loss weighting the errors by their closeness to the true edges. Used by the semantic segmenter, which
multiplies them by their weights of the 'SegmentationLoss' section, for its final logits and its deep supervision
heads.
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


def compute_lovasz_softmax_loss(class_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Computes the Lovász-Softmax loss of the segmentation logits (Berman et al., CVPR 2018), a loss of the semantic
    segmenter optimising the IoU of every class directly, where the cross entropy is dominated by the large classes:
    for every class, the errors |1 - p| of its true pixels and |p| of the other pixels are sorted in decreasing order
    and weighted by the gradient of the Lovász extension of 1 - IoU, giving a convex surrogate of 1 - IoU. The loss
    is the mean over the classes present in the labelled pixels of the batch, the ignored pixels being left out. It is
    computed in float32, stable under mixed precision.

    Args:
        class_logits (torch.Tensor): The [B, K, H, W] class logits.
        masks (torch.Tensor): The [B, H, W] int64 class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.

    Returns:
        torch.Tensor: The scalar loss, 0 for a batch without labelled pixel.
    """
    number_classes = class_logits.shape[1]
    labelled_pixels = masks != IGNORED_LABEL_INDEX

    # Probabilities and true class of the labelled pixels only, [N, K] and [N]
    probabilities = torch.softmax(input=class_logits.float(), dim=1).permute(0, 2, 3, 1)[labelled_pixels]
    labels = masks[labelled_pixels]

    # Surrogate of 1 - IoU of every class, computed for every class so that the absent ones are masked without a
    # synchronisation of the device
    class_losses, present_classes = [], []
    for class_index in range(number_classes):
        foreground = (labels == class_index).float()
        errors = (foreground - probabilities[:, class_index]).abs()
        sorted_errors, error_order = torch.sort(input=errors, descending=True)
        sorted_foreground = foreground[error_order]

        # Gradient of the Lovász extension: the increase of 1 - IoU as the pixels are added in decreasing error order
        true_count = sorted_foreground.sum()
        intersections = true_count - sorted_foreground.cumsum(dim=0)
        unions = true_count + (1.0 - sorted_foreground).cumsum(dim=0)
        jaccard_losses = 1.0 - intersections / unions
        lovasz_gradient = torch.cat(tensors=[jaccard_losses[:1], jaccard_losses[1:] - jaccard_losses[:-1]])
        class_losses.append((sorted_errors * lovasz_gradient).sum())
        present_classes.append(true_count > 0)

    # Mean over the classes present in the batch
    present_classes = torch.stack(tensors=present_classes).float()
    return (torch.stack(tensors=class_losses) * present_classes).sum() / present_classes.sum().clamp(min=1.0)


def compute_boundary_loss(class_logits: torch.Tensor, masks: torch.Tensor, edge_weights: torch.Tensor) -> torch.Tensor:
    """
    Computes the boundary loss of the segmentation logits, a loss of the semantic segmenter for precise contours: the
    cross entropy of every labelled pixel weighted by its closeness to the nearest true edge (1 on the edges,
    decreasing to 0 beyond 'SegmentationLoss.boundary_distance' pixels, compute_edge_distance_weights), normalised by
    the sum of the weights, so that the errors near the edges count the most whatever the area of the regions. It is
    computed in float32, stable under mixed precision.

    Args:
        class_logits (torch.Tensor): The [B, K, H, W] class logits.
        masks (torch.Tensor): The [B, H, W] int64 class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.
        edge_weights (torch.Tensor): The [B, H, W] float32 weight of every pixel, 0 on the ignored ones.

    Returns:
        torch.Tensor: The scalar loss, 0 for a batch without edge.
    """
    pixel_losses = torch.nn.functional.cross_entropy(input=class_logits.float(), target=masks,
                                                     ignore_index=IGNORED_LABEL_INDEX, reduction="none")
    return (pixel_losses * edge_weights).sum() / edge_weights.sum().clamp(min=torch.finfo(torch.float32).tiny)
