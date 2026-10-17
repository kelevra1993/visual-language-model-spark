"""
Targets of the bottom-up keypoint detector (HigherHRNet), built on the device by the keypoint trainer for every
training and validation batch from the keypoints of the objects of the whole images: the Gaussian heatmap of every
keypoint channel at every stride of the heatmap outputs of models/keypoints/keypoint_detector.py (4, and 2 with the
higher resolution branch), and the cell of the tag map read by every labelled keypoint for the associative embedding
(pull and push) losses of losses/keypoint_losses.py. All the coordinates are continuous pixel coordinates (the centre
of the first pixel at 0.5), the convention of the datasets and of the augmentations.
"""

import math
import torch

from typing import Any, Dict, List


def draw_gaussian_heatmaps(points: torch.Tensor, map_indices: torch.Tensor, number_maps: int, heatmap_height: int,
                           heatmap_width: int, heatmap_stride: int, heatmap_sigma: float) -> torch.Tensor:
    """
    Draws the Gaussian of every keypoint into its heatmap, for the heatmap targets of build_keypoint_targets: a
    Gaussian of standard deviation heatmap_sigma (in heatmap cells) centred on the keypoint, the heatmap cell j covering
    the pixels from j * stride to (j + 1) * stride, truncated at ceil(3 x sigma) cells from the cell containing the
    keypoint (below exp(-4.5) = 0.011). That cell is set to exactly 1, the positive cell of the focal heatmap loss
    (losses/keypoint_losses.py), its neighbours keeping the values of the Gaussian centred on the keypoint, which tell
    on which side of the cell it lies. Where the Gaussians of two keypoints of the same heatmap overlap (the same
    keypoint of two objects of the same class), the largest value is kept, so that both peaks stay at 1.

    Args:
        points (torch.Tensor): The [P, 2] float32 keypoints (x, y) in image pixels.
        map_indices (torch.Tensor): The [P] int64 index of the heatmap of every keypoint, among the number_maps.
        number_maps (int): The number of heatmaps, the images times the channels.
        heatmap_height (int): The number of rows h of the heatmaps.
        heatmap_width (int): The number of columns w of the heatmaps.
        heatmap_stride (int): The image pixels per heatmap cell.
        heatmap_sigma (float): The standard deviation of the Gaussians in heatmap cells.

    Returns:
        torch.Tensor: The [number_maps, h, w] float32 heatmaps, zero away from the keypoints.
    """
    device = points.device
    heatmaps = torch.zeros(size=(number_maps * heatmap_height * heatmap_width,), device=device, dtype=torch.float32)

    # Centre of every Gaussian in heatmap cell indices, the centre of cell j lying at pixel (j + 0.5) * stride, and the
    # cell containing the keypoint, the one its tag is read at
    centres = points.to(dtype=torch.float32) / heatmap_stride - 0.5
    radius = int(math.ceil(3.0 * heatmap_sigma))
    offsets = torch.arange(start=-radius, end=radius + 1, device=device, dtype=torch.int64)
    keypoint_cells = torch.div(points.to(dtype=torch.float32), heatmap_stride, rounding_mode="floor").to(
        dtype=torch.int64)

    # Square window of cells around the cell of every keypoint, [P, D, D] with D = 2 x radius + 1
    columns = keypoint_cells[:, 0, None, None] + offsets[None, None, :]
    rows = keypoint_cells[:, 1, None, None] + offsets[None, :, None]
    squared_distances = (columns - centres[:, 0, None, None]) ** 2 + (rows - centres[:, 1, None, None]) ** 2
    values = torch.exp(-squared_distances / (2.0 * heatmap_sigma ** 2))

    # The cell containing the keypoint, at the centre of its window, is the positive cell of the focal loss
    values[:, radius, radius] = 1.0

    # Cells of the window inside the heatmap written with the maximum of the overlapping Gaussians
    inside = (columns >= 0) & (columns < heatmap_width) & (rows >= 0) & (rows < heatmap_height)
    flat_indices = (map_indices[:, None, None] * heatmap_height + rows) * heatmap_width + columns
    heatmaps.scatter_reduce_(dim=0, index=flat_indices[inside], src=values[inside], reduce="amax", include_self=True)
    return heatmaps.view(number_maps, heatmap_height, heatmap_width)


def build_keypoint_targets(keypoints: List[torch.Tensor], labels: List[torch.Tensor], channel_table: torch.Tensor,
                           number_channels: int, image_height: int, image_width: int, heatmap_strides: List[int],
                           tag_stride: int, heatmap_sigma: float) -> Dict[str, Any]:
    """
    Builds the training targets of the bottom-up keypoint detector for a batch of whole images, on the device of the
    channel table: the objects of every image are padded to the most objects of an image, every keypoint is assigned
    the heatmap channel of its class and keypoint (build_keypoint_channel_table), and a keypoint is trained when it is
    labelled (visibility above 0), its class has this keypoint and it lies inside the image. Its Gaussian is drawn at
    every heatmap stride, and its tag is read at the cell of its channel containing it in the tag map.

    Args:
        keypoints (List[torch.Tensor]): One [N_i, K, 3] tensor per image of the (x, y, visibility) keypoints of its
            objects in image pixels, K being the most keypoints of a class.
        labels (List[torch.Tensor]): One [N_i] int64 tensor per image of the class of its objects.
        channel_table (torch.Tensor): The [number of classes + 1, K] int64 channel of every keypoint position of every
            class, -1 for the padding positions and the classes without keypoints.
        number_channels (int): The number of heatmap channels C, the keypoints of all the classes.
        image_height (int): The height H of the images in pixels.
        image_width (int): The width W of the images in pixels.
        heatmap_strides (List[int]): The image pixels per cell of every heatmap output, e.g. [4, 2].
        tag_stride (int): The image pixels per cell of the tag maps, 4.
        heatmap_sigma (float): The standard deviation of the Gaussians in heatmap cells, the same at every stride.

    Returns:
        Dict[str, Any]: The 'heatmaps', one [B, C, H / stride, W / stride] float32 tensor per stride; the
            'tag_indices' [B, M, K] int64 flat index of every keypoint in the [C x H / tag_stride x W / tag_stride]
            tag map of its image (0 when not trained); the 'tag_valid' [B, M, K] booleans of the trained keypoints; and
            the 'object_labels' [B, M] int64 class of every object (0 for the padding objects), M being the most
            objects of an image (at least 1).
    """
    device = channel_table.device
    batch_size = len(keypoints)
    maximum_keypoints = channel_table.shape[1]
    maximum_objects = max([image_keypoints.shape[0] for image_keypoints in keypoints] + [1])

    # Keypoints and classes of the objects padded to the most objects of an image, the padding objects of class 0
    # having no channel
    object_keypoints = torch.zeros(size=(batch_size, maximum_objects, maximum_keypoints, 3), device=device,
                                   dtype=torch.float32)
    object_labels = torch.zeros(size=(batch_size, maximum_objects), device=device, dtype=torch.int64)
    for image_index, (image_keypoints, image_labels) in enumerate(zip(keypoints, labels, strict=True)):
        number_objects = image_keypoints.shape[0]
        object_keypoints[image_index, :number_objects] = image_keypoints.to(device=device, dtype=torch.float32)
        object_labels[image_index, :number_objects] = image_labels.to(device=device, dtype=torch.int64)

    # Channel of every keypoint, trained when labelled, defined for its class and inside the image
    channels = channel_table[object_labels]
    positions = object_keypoints[..., :2]
    inside = ((positions[..., 0] >= 0.0) & (positions[..., 0] < image_width) & (positions[..., 1] >= 0.0) &
              (positions[..., 1] < image_height))
    valid = (object_keypoints[..., 2] > 0) & (channels >= 0) & inside

    # Gaussian heatmaps of the trained keypoints at every stride, one heatmap per image and channel
    image_indices = torch.arange(end=batch_size, device=device, dtype=torch.int64)[:, None, None].expand_as(channels)
    map_indices = (image_indices * number_channels + channels)[valid]
    valid_points = positions[valid]
    heatmaps = []
    for heatmap_stride in heatmap_strides:
        heatmap_height, heatmap_width = image_height // heatmap_stride, image_width // heatmap_stride
        stride_heatmaps = draw_gaussian_heatmaps(points=valid_points, map_indices=map_indices,
                                                 number_maps=batch_size * number_channels,
                                                 heatmap_height=heatmap_height, heatmap_width=heatmap_width,
                                                 heatmap_stride=heatmap_stride, heatmap_sigma=heatmap_sigma)
        heatmaps.append(stride_heatmaps.view(batch_size, number_channels, heatmap_height, heatmap_width))

    # Tag map cell of every keypoint: the cell of its channel containing it, where its Gaussian peaks
    tag_height, tag_width = image_height // tag_stride, image_width // tag_stride
    tag_columns = torch.div(positions[..., 0], tag_stride, rounding_mode="floor").to(dtype=torch.int64)
    tag_rows = torch.div(positions[..., 1], tag_stride, rounding_mode="floor").to(dtype=torch.int64)
    tag_indices = ((channels.clamp(min=0) * tag_height + tag_rows.clamp(min=0, max=tag_height - 1)) * tag_width +
                   tag_columns.clamp(min=0, max=tag_width - 1))
    tag_indices = torch.where(valid, tag_indices, torch.zeros_like(input=tag_indices))
    return {"heatmaps": heatmaps, "tag_indices": tag_indices, "tag_valid": valid, "object_labels": object_labels}
