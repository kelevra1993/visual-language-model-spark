"""
Crop transform of the keypoint detection task, run on the device by the keypoint trainer for whole batches: the crop
region of every object (its box enlarged and widened to the aspect ratio of the crop), the random scale, rotation,
horizontal flip and shift of the crop augmentation, the affine matrices between the resized image and the crop, the
bilinear warp of the crops, the matching transformation of the keypoints, which keep their names whatever the
augmentation (a flipped keypoint stays in its heatmap channel), and the Gaussian heatmap targets of the
keypoint detector. All the coordinates are continuous pixel coordinates (the centre of the first pixel at 0.5), the
convention of the datasets, so that a point keeps its exact place through the warp and the inverse warp of the
predictions.
"""

import math
import torch

from typing import Any, Dict, List, Tuple
from torch.nn import functional as torch_functional


def compute_crop_regions(bounding_boxes: torch.Tensor, crop_height: int, crop_width: int,
                         box_enlargement: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gives the region of the resized image every object crop covers before its augmentation, so that the object fills
    the crop without distortion: the box centred, enlarged by box_enlargement to keep some context around the object,
    and widened along its short side to the aspect ratio of the crop.

    Args:
        bounding_boxes (torch.Tensor): The [B, 4] (x1, y1, x2, y2) boxes of the objects in resized image pixels.
        crop_height (int): The height of the crops in pixels.
        crop_width (int): The width of the crops in pixels.
        box_enlargement (float): The factor applied to both sides of the region, e.g. 1.25.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [B, 2] (x, y) centres and [B, 2] (width, height) sizes of the regions.
    """
    centres = (bounding_boxes[:, :2] + bounding_boxes[:, 2:]) / 2.0

    # A degenerate box still gives a region of one pixel
    widths = (bounding_boxes[:, 2] - bounding_boxes[:, 0]).clamp(min=1.0)
    heights = (bounding_boxes[:, 3] - bounding_boxes[:, 1]).clamp(min=1.0)

    # Widen the short side of the box to the aspect ratio of the crop, then enlarge both sides
    aspect_ratio = crop_width / crop_height
    widths, heights = torch.maximum(widths, heights * aspect_ratio), torch.maximum(heights, widths / aspect_ratio)
    return centres, torch.stack(tensors=[widths, heights], dim=1) * box_enlargement


def create_identity_augmentations(number_samples: int) -> Dict[str, torch.Tensor]:
    """
    Gives the crop augmentation parameters of unaugmented crops (validation, test and deployment), the object centred
    in its region without scale, rotation, flip nor shift.

    Args:
        number_samples (int): The number of crops.

    Returns:
        Dict[str, torch.Tensor]: The [B] 'scales' (1), [B] 'rotations' in degrees (0), [B] bool 'flips' (False) and
            [B, 2] 'shifts' (0) of the crops.
    """
    return {"scales": torch.ones(size=(number_samples,)), "rotations": torch.zeros(size=(number_samples,)),
            "flips": torch.zeros(size=(number_samples,), dtype=torch.bool),
            "shifts": torch.zeros(size=(number_samples, 2))}


def draw_crop_augmentations(number_samples: int, augmentation_configuration: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Draws the random geometric augmentation of every training crop on the CPU, applied by build_crop_affines in the
    single warp that cuts the crop out of the image: a scale of the region (above 1 the object looks smaller), a
    rotation drawn with its probability within the configured limit, a horizontal flip with its probability, and a
    shift of the region centre as a fraction of its sides. The same limits apply to every class.

    Args:
        number_samples (int): The number of crops.
        augmentation_configuration (Dict[str, Any]): The 'Data.crop_augmentation' section: 'scale_range',
            'maximum_rotation_degrees', 'rotation_probability', 'flip_probability' and 'maximum_shift_fraction'.

    Returns:
        Dict[str, torch.Tensor]: The [B] 'scales', [B] 'rotations' in degrees, [B] bool 'flips' and [B, 2] 'shifts'
            (fractions of the region width and height) of the crops.
    """
    minimum_scale, maximum_scale = augmentation_configuration["scale_range"]
    scales = minimum_scale + (maximum_scale - minimum_scale) * torch.rand(size=(number_samples,))

    # Rotation within the configured limit, applied with its probability
    rotations = ((2.0 * torch.rand(size=(number_samples,)) - 1.0) *
                 float(augmentation_configuration["maximum_rotation_degrees"]))
    rotations = torch.where(torch.rand(size=(number_samples,)) < augmentation_configuration["rotation_probability"],
                            rotations, torch.zeros_like(input=rotations))

    # Horizontal flip and shift of the region centre
    flips = torch.rand(size=(number_samples,)) < augmentation_configuration["flip_probability"]
    shifts = (2.0 * torch.rand(size=(number_samples, 2)) - 1.0) * augmentation_configuration["maximum_shift_fraction"]
    return {"scales": scales, "rotations": rotations, "flips": flips, "shifts": shifts}


def build_crop_affines(centres: torch.Tensor, sizes: torch.Tensor, augmentations: Dict[str, torch.Tensor],
                       crop_height: int, crop_width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds the affine matrices between the resized image and every crop, the warp of the crops using the inverse one
    and the keypoints the direct one, and the predictions of the network going back to the image through the inverse
    one: the region (shifted, then scaled) is mapped onto the crop, turned around its centre by the rotation
    (anticlockwise on the screen for a positive angle) and mirrored by the flip.

    Args:
        centres (torch.Tensor): The [B, 2] (x, y) region centres given by compute_crop_regions.
        sizes (torch.Tensor): The [B, 2] (width, height) region sizes given by compute_crop_regions.
        augmentations (Dict[str, torch.Tensor]): The 'scales', 'rotations', 'flips' and 'shifts' of the crops.
        crop_height (int): The height of the crops in pixels.
        crop_width (int): The width of the crops in pixels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [B, 2, 3] image to crop matrices and their [B, 2, 3] inverses, mapping
            continuous pixel coordinates.
    """
    device, dtype = centres.device, centres.dtype
    scales = augmentations["scales"].to(device=device, dtype=dtype)
    angles = augmentations["rotations"].to(device=device, dtype=dtype) * (math.pi / 180.0)
    mirror = 1.0 - 2.0 * augmentations["flips"].to(device=device, dtype=dtype)

    # Shifted and scaled region, mapped onto the crop width
    region_centres = centres + augmentations["shifts"].to(device=device, dtype=dtype) * sizes
    zooms = crop_width / (sizes[:, 0] * scales)

    # Linear part: the x axis mirrored by the flip, then turned anticlockwise on the screen (y pointing down)
    cosines, sines = torch.cos(angles), torch.sin(angles)
    linear = torch.stack(tensors=[torch.stack(tensors=[cosines * mirror, sines], dim=1),
                                  torch.stack(tensors=[-sines * mirror, cosines], dim=1)], dim=1) * zooms[:, None, None]

    # Translation placing the region centre at the crop centre
    crop_centre = torch.tensor(data=[crop_width / 2.0, crop_height / 2.0], device=device, dtype=dtype)
    translations = crop_centre - (linear @ region_centres[:, :, None])[:, :, 0]
    image_to_crop = torch.cat(tensors=[linear, translations[:, :, None]], dim=2)

    # Inverse matrices, from the crop back to the image
    inverse_linear = torch.linalg.inv(linear)
    crop_to_image = torch.cat(tensors=[inverse_linear,
                                       (region_centres - (inverse_linear @ crop_centre[:, None])[:, :, 0])[:, :, None]],
                              dim=2)
    return image_to_crop, crop_to_image


def transform_points(points: torch.Tensor, affines: torch.Tensor) -> torch.Tensor:
    """
    Applies one affine matrix per sample to its points, moving the keypoints of the objects into their crops and the
    predicted keypoints of the crops back into the resized images.

    Args:
        points (torch.Tensor): The [B, K, 2] (x, y) points.
        affines (torch.Tensor): The [B, 2, 3] matrices.

    Returns:
        torch.Tensor: The [B, K, 2] moved points.
    """
    return points @ affines[:, :, :2].transpose(1, 2) + affines[:, None, :, 2]


def warp_crops(images: torch.Tensor, crop_to_image: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    """
    Cuts the crops out of the resized images on the device with a bilinear sampling of the image at the place of every
    crop pixel centre, the parts of a region outside the image being black like the letterbox padding.

    Args:
        images (torch.Tensor): The [B, C, H, W] float images of the objects.
        crop_to_image (torch.Tensor): The [B, 2, 3] crop to image matrices of build_crop_affines.
        crop_height (int): The height of the crops in pixels.
        crop_width (int): The width of the crops in pixels.

    Returns:
        torch.Tensor: The [B, C, crop_height, crop_width] crops, in the data type of the images.
    """
    image_height, image_width = images.shape[-2:]
    device = images.device

    # Continuous coordinates of the crop pixel centres, moved into the images
    rows = torch.arange(end=crop_height, device=device, dtype=torch.float32) + 0.5
    columns = torch.arange(end=crop_width, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    crop_points = torch.stack(tensors=[grid_x, grid_y], dim=-1).reshape(1, -1, 2).expand(images.shape[0], -1, -1)
    image_points = transform_points(points=crop_points, affines=crop_to_image.to(dtype=torch.float32))

    # Normalised sampling grid, -1 and 1 being the outer edges of the image (align_corners False)
    sizes = torch.tensor(data=[image_width, image_height], device=device, dtype=torch.float32)
    sampling_grid = (image_points / sizes * 2.0 - 1.0).reshape(images.shape[0], crop_height, crop_width, 2)
    crops = torch_functional.grid_sample(input=images.to(dtype=torch.float32), grid=sampling_grid, mode="bilinear",
                                         padding_mode="zeros", align_corners=False)
    return crops.to(dtype=images.dtype)


def build_heatmap_targets(crop_keypoints: torch.Tensor, visibilities: torch.Tensor, crop_height: int, crop_width: int,
                          heatmap_stride: int, heatmap_sigma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Builds the target heatmaps of the keypoint detector, one per keypoint position: a Gaussian of standard deviation
    heatmap_sigma (in heatmap cells) centred on the keypoint, the heatmap cell j covering the crop pixels from
    j * stride to (j + 1) * stride. A keypoint is trained (weight 1) when it is labelled (visibility above 0) and falls
    inside the crop; its heatmap is zero otherwise.

    Args:
        crop_keypoints (torch.Tensor): The [B, K, 2] keypoints in crop pixels.
        visibilities (torch.Tensor): The [B, K] visibilities (0 unlabelled, 1 occluded, 2 visible).
        crop_height (int): The height of the crops in pixels.
        crop_width (int): The width of the crops in pixels.
        heatmap_stride (int): The crop pixels per heatmap cell, 4 for the HRNet output.
        heatmap_sigma (float): The standard deviation of the Gaussians in heatmap cells.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [B, K, crop_height / stride, crop_width / stride] float32 heatmaps and
            the [B, K] float32 weights of the keypoints.
    """
    device = crop_keypoints.device
    heatmap_height, heatmap_width = crop_height // heatmap_stride, crop_width // heatmap_stride
    crop_keypoints = crop_keypoints.to(dtype=torch.float32)

    # Keypoints inside the crop and labelled are the only ones trained
    inside = ((crop_keypoints[..., 0] >= 0.0) & (crop_keypoints[..., 0] < crop_width) &
              (crop_keypoints[..., 1] >= 0.0) & (crop_keypoints[..., 1] < crop_height))
    weights = (inside & (visibilities > 0)).to(dtype=torch.float32)

    # Gaussian centred on the keypoint in heatmap cell indices, the centre of cell j lying at crop pixel
    # (j + 0.5) * stride
    centres = crop_keypoints / heatmap_stride - 0.5
    cell_x = torch.arange(end=heatmap_width, device=device, dtype=torch.float32)
    cell_y = torch.arange(end=heatmap_height, device=device, dtype=torch.float32)
    squared_distance_x = (cell_x[None, None, :] - centres[..., 0:1]) ** 2
    squared_distance_y = (cell_y[None, None, :] - centres[..., 1:2]) ** 2
    heatmaps = torch.exp(-(squared_distance_y[..., :, None] + squared_distance_x[..., None, :]) /
                         (2.0 * heatmap_sigma ** 2))
    return heatmaps * weights[..., None, None], weights


def describe_crop_augmentation(augmentations: Dict[str, torch.Tensor], sample_index: int) -> List[str]:
    """
    Writes the geometric augmentation of a crop in the information panel of the input viewer ('view_input_images'),
    above the photometric augmentations of the DataAugmenter report.

    Args:
        augmentations (Dict[str, torch.Tensor]): The 'scales', 'rotations', 'flips' and 'shifts' of the batch.
        sample_index (int): The index of the crop in its batch.

    Returns:
        List[str]: The lines of the crop augmentation.
    """
    # Adding zero turns the negative zeros (no shift, or no rotation) into zeros, so that they never print with a minus
    # sign
    shift_x, shift_y = (shift + 0.0 for shift in augmentations["shifts"][sample_index].tolist())
    rotation = float(augmentations["rotations"][sample_index]) + 0.0
    return ["Crop :", f"  scale : {float(augmentations['scales'][sample_index]):.2f}",
            f"  rotation : {rotation:.1f} degrees",
            f"  flip : {'yes' if bool(augmentations['flips'][sample_index]) else 'no'}",
            f"  shift : {shift_x:+.2f} width, {shift_y:+.2f} height"]
