"""
Morphology of the segmentation masks on the device, by max pooling: dilation of boolean masks, the band of a region
within a distance of its contour, and the weights of the pixels near the true edges of class-index masks. Shared by
the boundary loss of the semantic segmenter (losses/segmentation_losses.py) and the boundary metrics of its
evaluation (evaluation/segmentation/boundary_metrics.py). The distances are chessboard distances (a pixel and its 8
neighbours are 1 pixel apart), and the outside of the image never draws a contour.
"""

import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX


def dilate_masks(masks: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Dilates boolean masks by a square of 2 x radius + 1 pixels: every pixel within radius pixels of a True pixel
    becomes True. The max pooling pads with minus infinity, so the outside of the image dilates nothing.

    Args:
        masks (torch.Tensor): The [B, H, W] boolean masks.
        radius (int): The dilation radius in pixels, 0 for none.

    Returns:
        torch.Tensor: The [B, H, W] dilated boolean masks.
    """
    if radius == 0:
        return masks
    return torch.nn.functional.max_pool2d(input=masks.unsqueeze(dim=1).float(), kernel_size=2 * radius + 1, stride=1,
                                          padding=radius).squeeze(dim=1) > 0


def compute_contour_bands(regions: torch.Tensor, neutral_pixels: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Finds the band of every region within radius pixels of its contour, inside the region: the pixels of the region
    with a pixel outside of it at most radius pixels away, i.e. the pixels the erosion of the region would remove. A
    radius of 1 gives the edge pixels of the region. The neutral pixels (the ignored pixels of the masks: padding,
    unlabelled pixels) and the outside of the image count as part of the region, so that they draw no contour.

    Args:
        regions (torch.Tensor): The [B, H, W] boolean regions, e.g. the pixels of one class.
        neutral_pixels (torch.Tensor): The [B, H, W] boolean pixels drawing no contour.
        radius (int): The width of the band in pixels, at least 1.

    Returns:
        torch.Tensor: The [B, H, W] boolean band of the regions.
    """
    return regions & dilate_masks(masks=~(regions | neutral_pixels), radius=radius)


def compute_edge_distance_weights(masks: torch.Tensor, maximum_distance: int) -> torch.Tensor:
    """
    Weighs every labelled pixel of class-index masks by its closeness to the nearest true edge, an edge pixel being a
    labelled pixel with a labelled neighbour of another class: 1 on the edges, decreasing linearly with the distance
    to 1 / (maximum_distance + 1) at maximum_distance pixels, and 0 further away and on the ignored pixels. Used by the
    boundary loss of the semantic segmenter, so that the errors next to the true edges weigh the most.

    Args:
        masks (torch.Tensor): The [B, H, W] int64 class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.
        maximum_distance (int): The distance in pixels at which the weights stop, at least 0.

    Returns:
        torch.Tensor: The [B, H, W] float32 weights.
    """
    labelled_pixels = masks != IGNORED_LABEL_INDEX

    # Largest and smallest labelled class of the 3x3 neighbourhood of every pixel, the ignored pixels and the outside
    # of the image taking no part
    class_values = masks.unsqueeze(dim=1).float()
    highest_classes = torch.where(condition=labelled_pixels.unsqueeze(dim=1), input=class_values, other=-1.0)
    lowest_classes = torch.where(condition=labelled_pixels.unsqueeze(dim=1), input=class_values, other=float("inf"))
    neighbourhood_maximum = torch.nn.functional.max_pool2d(input=highest_classes, kernel_size=3, stride=1, padding=1)
    neighbourhood_minimum = -torch.nn.functional.max_pool2d(input=-lowest_classes, kernel_size=3, stride=1, padding=1)

    # Edge pixels, whose neighbourhood holds another labelled class than their own
    edge_pixels = labelled_pixels & ((neighbourhood_maximum != class_values) |
                                     (neighbourhood_minimum != class_values)).squeeze(dim=1)

    # Every pixel at distance k <= maximum_distance of an edge lies in maximum_distance - k + 1 of the growing bands
    current_band = edge_pixels
    band_counts = edge_pixels.float()
    for _ in range(maximum_distance):
        current_band = dilate_masks(masks=current_band, radius=1)
        band_counts = band_counts + current_band.float()
    return band_counts / (maximum_distance + 1) * labelled_pixels
