"""
Image operations of torchvision and kornia ported so that they run on the device from numbers only, for the types of
DataAugmenter (vision_training/data/augmentations.py) that augment the training batches on the GPU: the rotation of
torchvision (rotate_on_device, used by random_rotation for the images and the masks), the contrast-limited adaptive
histogram equalization of kornia (equalize_clahe_on_device, used by clahe_augment) and the RGB to YUV conversions of
kornia (convert_colour_space with get_yuv_matrices, used by equalize_histogram_augment). The originals copy their
matrices, their fill values or the indices of their tiles from the host at every image, which makes the host wait for
the device; the ports give the same values, their constants being filled on the device or cached per image size,
dtype and device.
"""

import math
import torch

from functools import lru_cache
from torch.nn import functional as torch_functional
from typing import Tuple

# Matrices of kornia's rgb_to_yuv and yuv_to_rgb (BT.470 M/PAL values), applied to the channel vector of every pixel
RGB_TO_YUV_MATRIX = ((0.299, 0.587, 0.114), (-0.147, -0.289, 0.436), (0.615, -0.515, -0.100))
YUV_TO_RGB_MATRIX = ((1.0, 0.0, 1.14), (1.0, -0.396, -0.581), (1.0, 2.029, 0.0))

# Bins of the histograms of the tiles of the CLAHE, one per intensity level, as in kornia's equalize_clahe
CLAHE_HISTOGRAM_BINS = 256


@lru_cache(maxsize=16)
def get_rotation_grid_terms(height: int, width: int, dtype: torch.dtype,
                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gives the terms of the sampling grid of rotate_on_device for one image size, created from numbers once per image
    size, dtype and device and only read afterwards: the coordinates of every pixel centre from the image centre with a
    third coordinate of 1 (the base grid of torchvision's _affine_grid), and the half width and half height dividing
    the rotation, which torchvision copies from the host at every image.

    Args:
        height (int): The height of the images in pixels.
        width (int): The width of the images in pixels.
        dtype (torch.dtype): The floating dtype of the grid.
        device (torch.device): The device of the images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [1, H * W, 3] base grid and the [2] half width and half height.
    """
    # Pixel centres of every row and column, as the two linspaces of torchvision
    base_grid = torch.empty(size=(1, height, width, 3), dtype=dtype, device=device)
    base_grid[..., 0].copy_(torch.linspace(start=(1.0 - width) * 0.5, end=(width - 1.0) * 0.5, steps=width,
                                           device=device))
    base_grid[..., 1].copy_(torch.linspace(start=(1.0 - height) * 0.5, end=(height - 1.0) * 0.5, steps=height,
                                           device=device).unsqueeze_(dim=-1))
    base_grid[..., 2].fill_(1)

    # Half sizes filled from numbers, dividing the matrix of the rotation into the [-1, 1] range of grid_sample
    half_sizes = torch.empty(size=(2,), dtype=dtype, device=device)
    half_sizes[0].fill_(0.5 * width)
    half_sizes[1].fill_(0.5 * height)
    return base_grid.view(1, height * width, 3), half_sizes


def rotate_on_device(image: torch.Tensor, angle: float, interpolation: str, fill: float) -> torch.Tensor:
    """
    Rotates images or masks [..., C, H, W] by an angle in degrees, counter-clockwise on the displayed image, around
    their centre without expanding them, a port of torchvision's rotate_image giving the same values, for the
    random_rotation type of DataAugmenter: a multiple of 180 degrees (or of 90 degrees for a square image) is a pixel
    permutation, and any other angle samples the input at the rotated pixel centres, the uncovered pixels taking the
    fill value. torchvision copies its rotation matrix, its scales and its fill value from the host at every image,
    which makes the host wait for the device; here they reach the device as numbers.

    Args:
        image (torch.Tensor): The float images or the integer masks [..., C, H, W], the masks being sampled in float32
            and rounded back to their dtype.
        angle (float): The angle in degrees.
        interpolation (str): The sampling of the input, "bilinear" or "nearest".
        fill (float): The value of the pixels the rotation uncovers.

    Returns:
        torch.Tensor: The rotated images, with the shape and dtype of the input.
    """
    # Rotations by a multiple of 180 degrees, and of 90 degrees for a square image, as pixel permutations
    angle = angle % 360
    input_height, input_width = image.shape[-2], image.shape[-1]
    if angle == 0:
        return image.clone()
    if angle == 180:
        return torch.rot90(input=image, k=2, dims=(-2, -1))
    if input_height == input_width and angle in (90, 270):
        return torch.rot90(input=image, k=1 if angle == 90 else 3, dims=(-2, -1))
    if image.numel() == 0:
        return image.reshape(image.shape)

    # Matrix of the opposite angle taking every output pixel centre back into the input, filled on the device, divided
    # by the half sizes and applied to the pixel centres of the base grid, as torchvision's _affine_grid
    radians = math.radians(-angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    theta = torch.empty(size=(1, 2, 3), dtype=dtype, device=image.device)
    for value_index, value in enumerate([cosine, sine, 0.0, -sine, cosine, 0.0]):
        theta.view(-1)[value_index].fill_(value)
    base_grid, half_sizes = get_rotation_grid_terms(height=input_height, width=input_width, dtype=dtype,
                                                    device=image.device)
    grid = base_grid.bmm(theta.transpose(1, 2).div_(half_sizes)).view(1, input_height, input_width, 2)

    # Sampling of every image of the batch with a channel of ones, which tells how much of every output pixel the
    # input covers, as torchvision's _apply_grid_transform
    number_channels = image.shape[-3]
    flat_images = image.reshape(-1, number_channels, input_height, input_width)
    float_images = flat_images if flat_images.dtype == dtype else flat_images.to(dtype=dtype)
    if flat_images.shape[0] > 1:
        grid = grid.expand(flat_images.shape[0], -1, -1, -1)
    coverage = torch.ones(size=(flat_images.shape[0], 1, input_height, input_width), dtype=dtype,
                          device=image.device)
    sampled_images = torch_functional.grid_sample(input=torch.cat(tensors=(float_images, coverage), dim=1), grid=grid,
                                                  mode=interpolation, padding_mode="zeros", align_corners=False)
    sampled_images, coverage = torch.tensor_split(input=sampled_images, indices=(-1,), dim=-3)
    coverage = coverage.expand_as(sampled_images)

    # Uncovered pixels take the fill value: wholly with the nearest sampling, in proportion of their uncovered part
    # with the bilinear one
    if interpolation == "nearest":
        sampled_images = sampled_images.masked_fill(mask=coverage < 0.5, value=fill)
    else:
        sampled_images = sampled_images.sub_(fill).mul_(coverage).add_(fill)

    # Masks rounded back to their integer dtype
    if dtype != image.dtype:
        sampled_images = sampled_images.round_().to(dtype=image.dtype)
    return sampled_images.reshape(image.shape)


def compute_clahe_tiles(images: torch.Tensor, grid_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cuts every image of a batch into the grid of tiles of equalize_clahe_on_device whose histograms give the lookup
    tables, kornia's _compute_tiles with tiles of even sides: the images are padded by reflection at their bottom and
    right so that the grid of tiles covers them exactly.

    Args:
        images (torch.Tensor): The [B, C, H, W] images in [0, 1].
        grid_size (Tuple[int, int]): The number of tiles of every column and row (GH, GW).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [B, GH, GW, C, TH, TW] tiles and the [B, C, H', W'] padded images.

    Raises:
        ValueError: If the padding needed by the grid is larger than the images.
    """
    # Even sides of the tiles covering the images, the missing rows and columns padded by reflection
    image_height, image_width = images.shape[-2], images.shape[-1]
    tile_height, tile_width = math.ceil(image_height / grid_size[0]), math.ceil(image_width / grid_size[1])
    tile_height, tile_width = tile_height + tile_height % 2, tile_width + tile_width % 2
    padding_height, padding_width = tile_height * grid_size[0] - image_height, tile_width * grid_size[1] - image_width
    if padding_height > image_height or padding_width > image_width:
        raise ValueError(f"CLAHE cannot cut {image_height}x{image_width} images into {grid_size[0]}x{grid_size[1]} "
                         f"tiles of {tile_height}x{tile_width} pixels, the padding exceeding the images.")
    if padding_height > 0 or padding_width > 0:
        images = torch_functional.pad(input=images, pad=[0, padding_width, 0, padding_height], mode="reflect")

    # Tiles of every image, channels kept together
    number_channels = images.shape[-3]
    tiles = (images.unfold(dimension=1, size=number_channels, step=number_channels)
             .unfold(dimension=2, size=tile_height, step=tile_height)
             .unfold(dimension=3, size=tile_width, step=tile_width).squeeze(dim=1)).contiguous()
    return tiles, images


def compute_clahe_lookup_tables(tiles: torch.Tensor, clip_limit: float) -> torch.Tensor:
    """
    Computes the lookup table of every tile of equalize_clahe_on_device, kornia's _compute_luts (the OpenCV approach):
    the histogram of the tile over 256 bins of [0, 1], the counts above the clip limit spread over every bin, then the
    cumulated counts scaled to the 256 levels. The histograms of all the tiles are counted at once with the bins of
    torch.histc, which kornia calls tile by tile: a value x of [0, 1] falls in the bin trunc(x * 256), and 1 in the
    last one, the values outside [0, 1] being left out.

    Args:
        tiles (torch.Tensor): The [B, GH, GW, C, TH, TW] tiles in [0, 1].
        clip_limit (float): The clip limit of the counts, relative to their mean over the bins; 0 for no limit.

    Returns:
        torch.Tensor: The [B, GH, GW, C, 256] lookup tables, levels from 0 to 255 in the dtype of the tiles.
    """
    batch_size, grid_height, grid_width, number_channels, tile_height, tile_width = tiles.shape
    number_pixels = tile_height * tile_width

    # Counts of every bin of every tile, computed in float32 (or float64) like torch.histc, in the dtype of the tiles
    histogram_dtype = torch.float64 if tiles.dtype == torch.float64 else torch.float32
    tile_values = tiles.reshape(-1, number_pixels).to(dtype=histogram_dtype)
    bin_indices = (tile_values * CLAHE_HISTOGRAM_BINS).long().clamp_(min=0, max=CLAHE_HISTOGRAM_BINS - 1)
    counted_values = ((tile_values >= 0.0) & (tile_values <= 1.0)).to(dtype=histogram_dtype)
    histograms = torch.zeros(size=(tile_values.shape[0], CLAHE_HISTOGRAM_BINS), dtype=histogram_dtype,
                             device=tiles.device).scatter_add_(dim=1, index=bin_indices, src=counted_values)
    histograms = histograms.to(dtype=tiles.dtype)

    # Contrast limiting: the counts above the limit are cut and spread evenly over the bins, the remainder one count
    # per bin from the first one
    if clip_limit > 0.0:
        maximum_count = max(clip_limit * number_pixels // CLAHE_HISTOGRAM_BINS, 1)
        histograms.clamp_(max=maximum_count)
        clipped_counts = number_pixels - histograms.sum(dim=1)
        residual_counts = torch.remainder(input=clipped_counts, other=CLAHE_HISTOGRAM_BINS)
        histograms += (clipped_counts - residual_counts).div(CLAHE_HISTOGRAM_BINS)[None].transpose(0, 1)
        bin_range = torch.arange(end=CLAHE_HISTOGRAM_BINS, device=histograms.device)
        histograms += bin_range.repeat(histograms.shape[0], 1) < residual_counts[None].transpose(0, 1)

    # Cumulated counts scaled to the 256 levels and floored, as the integer tables of OpenCV
    lookup_tables = torch.cumsum(input=histograms, dim=1) * ((CLAHE_HISTOGRAM_BINS - 1) / number_pixels)
    lookup_tables = lookup_tables.clamp(min=0, max=CLAHE_HISTOGRAM_BINS - 1).floor()
    return lookup_tables.view(batch_size, grid_height, grid_width, number_channels, CLAHE_HISTOGRAM_BINS)


@lru_cache(maxsize=16)
def get_clahe_tile_indices(interpolation_rows: int, interpolation_columns: int,
                           device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gives the histogram tiles whose lookup tables the inner interpolation tiles of equalize_clahe_on_device blend, the
    indices of kornia's _map_luts, built on the device from numbers once per grid and device where kornia builds them
    on the host and copies them at every image: the inner interpolation tile r (r from 0, after the border tile)
    blends the rows r // 2 and r // 2 + 1 of the histogram tiles, given as [top, top, bottom, bottom], and the columns
    r // 2 and r // 2 + 1, given as [left, right, left, right], for its four neighbours.

    Args:
        interpolation_rows (int): The number of rows of interpolation tiles, twice the rows of the grid.
        interpolation_columns (int): The number of columns of interpolation tiles, twice the columns of the grid.
        device (torch.device): The device of the images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [rows - 2, 4] row indices and [columns - 2, 4] column indices.
    """
    neighbour_offsets = torch.arange(end=4, device=device)
    row_starts = torch.div(input=torch.arange(end=interpolation_rows - 2, device=device), other=2,
                           rounding_mode="floor")
    column_starts = torch.div(input=torch.arange(end=interpolation_columns - 2, device=device), other=2,
                              rounding_mode="floor")
    row_indices = row_starts.view(-1, 1) + torch.div(input=neighbour_offsets, other=2, rounding_mode="floor")
    column_indices = column_starts.view(-1, 1) + torch.remainder(input=neighbour_offsets, other=2)
    return row_indices, column_indices


def map_clahe_lookup_tables(interpolation_tiles: torch.Tensor, lookup_tables: torch.Tensor) -> torch.Tensor:
    """
    Gives every interpolation tile of equalize_clahe_on_device the lookup tables of the histogram tiles it blends,
    kornia's _map_luts with indices kept on the device: one table for the corner tiles, two for the border tiles and
    four for the inner tiles, the unused places holding -1. The indices of the inner tiles follow the rows and the
    columns of the grid, so that a grid that is not square works too, where kornia's indexing fails.

    Args:
        interpolation_tiles (torch.Tensor): The [B, 2GH, 2GW, C, TH / 2, TW / 2] interpolation tiles.
        lookup_tables (torch.Tensor): The [B, GH, GW, C, 256] lookup tables of the histogram tiles.

    Returns:
        torch.Tensor: The [B, 2GH, 2GW, 4, C, 256] lookup tables of every interpolation tile.
    """
    number_images, rows, columns, number_channels = interpolation_tiles.shape[:4]
    row_indices, column_indices = get_clahe_tile_indices(interpolation_rows=rows, interpolation_columns=columns,
                                                         device=interpolation_tiles.device)
    mapped_tables = torch.full(size=(number_images, rows, columns, 4, number_channels, lookup_tables.shape[-1]),
                               fill_value=-1, dtype=interpolation_tiles.dtype, device=interpolation_tiles.device)
    row_step, column_step = max(rows // 2 - 1, 1), max(columns // 2 - 1, 1)

    # Corner tiles take the table of their corner histogram tile
    mapped_tables[:, 0::rows - 1, 0::columns - 1, 0] = lookup_tables[:, 0::row_step, 0::column_step]

    # Tiles of the left and right borders take the tables above and below them
    mapped_tables[:, 1:-1, 0::columns - 1, 0] = lookup_tables[:, row_indices[:, 0], 0::column_step]
    mapped_tables[:, 1:-1, 0::columns - 1, 1] = lookup_tables[:, row_indices[:, 2], 0::column_step]

    # Tiles of the top and bottom borders take the tables on their left and right
    mapped_tables[:, 0::rows - 1, 1:-1, 0] = lookup_tables[:, 0::row_step, column_indices[:, 0]]
    mapped_tables[:, 0::rows - 1, 1:-1, 1] = lookup_tables[:, 0::row_step, column_indices[:, 1]]

    # Inner tiles take the tables of their four neighbouring histogram tiles
    inner_shape = (rows - 2, columns - 2, 4)
    mapped_tables[:, 1:-1, 1:-1, :] = lookup_tables[:, row_indices[:, None, :].expand(inner_shape),
                                                    column_indices[None, :, :].expand(inner_shape)]
    return mapped_tables


def compute_clahe_equalized_tiles(interpolation_tiles: torch.Tensor, lookup_tables: torch.Tensor) -> torch.Tensor:
    """
    Equalizes every interpolation tile of equalize_clahe_on_device, kornia's _compute_equalized_tiles: every pixel is
    mapped by the lookup tables of its tile (map_clahe_lookup_tables), then the mapped values are blended bilinearly
    by the distance of the pixel to the centres of the histogram tiles, the corner tiles keeping their only table. The
    right and bottom borders are read at the last column and the last row of the grid, which kornia, written for
    square grids, exchanges.

    Args:
        interpolation_tiles (torch.Tensor): The [B, 2GH, 2GW, C, TH / 2, TW / 2] interpolation tiles in [0, 1].
        lookup_tables (torch.Tensor): The [B, GH, GW, C, 256] lookup tables of the histogram tiles.

    Returns:
        torch.Tensor: The equalized [B, 2GH, 2GW, C, TH / 2, TW / 2] tiles in [0, 1].
    """
    mapped_tables = map_clahe_lookup_tables(interpolation_tiles=interpolation_tiles, lookup_tables=lookup_tables)
    number_images, rows, columns, number_channels, tile_height, tile_width = interpolation_tiles.shape
    dtype, device = interpolation_tiles.dtype, interpolation_tiles.device

    # Every pixel mapped by the four tables of its tile, its level indexing their 256 entries
    pixel_levels = (interpolation_tiles * 255).long().flatten(-2, -1).unsqueeze(dim=-3).expand(
        number_images, rows, columns, 4, number_channels, tile_height * tile_width)
    mapped_pixels = torch.gather(input=mapped_tables, dim=5, index=pixel_levels).to(interpolation_tiles).reshape(
        number_images, rows, columns, 4, number_channels, tile_height, tile_width)
    equalized_tiles = torch.zeros_like(input=interpolation_tiles)

    # Weights of the tables above and on the left, from 1 next to their centre down to 0 next to the other centre
    vertical_weights = (torch.arange(start=2 * tile_height - 1, end=-1, step=-1, dtype=dtype, device=device)
                        .div(2.0 * tile_height - 1)[None].transpose(-2, -1).expand(2 * tile_height, tile_width)
                        .unfold(dimension=0, size=tile_height, step=tile_height)
                        .unfold(dimension=1, size=tile_width, step=tile_width))
    horizontal_weights = (torch.arange(start=2 * tile_width - 1, end=-1, step=-1, dtype=dtype, device=device)
                          .div(2.0 * tile_width - 1).expand(tile_height, 2 * tile_width)
                          .unfold(dimension=0, size=tile_height, step=tile_height)
                          .unfold(dimension=1, size=tile_width, step=tile_width))
    column_weights = horizontal_weights.expand((columns - 2) // 2, 2, tile_height, tile_width).reshape(
        columns - 2, 1, tile_height, tile_width).unsqueeze(dim=0)
    row_weights = vertical_weights.repeat((rows - 2) // 2, 1, 1, 1).unsqueeze(dim=1)

    # Inner tiles blend their four tables, first along the rows, then along the columns
    top_left, top_right, bottom_left, bottom_right = mapped_pixels[:, 1:-1, 1:-1].unbind(dim=3)
    top = torch.addcmul(input=top_right, tensor1=column_weights, tensor2=torch.sub(input=top_left, other=top_right))
    bottom = torch.addcmul(input=bottom_right, tensor1=column_weights,
                           tensor2=torch.sub(input=bottom_left, other=bottom_right))
    equalized_tiles[:, 1:-1, 1:-1] = torch.addcmul(input=bottom, tensor1=row_weights,
                                                   tensor2=torch.sub(input=top, other=bottom))

    # Corner tiles keep their only table
    equalized_tiles[:, 0::rows - 1, 0::columns - 1] = mapped_pixels[:, 0::rows - 1, 0::columns - 1, 0]

    # Tiles of the left and right borders blend the tables above and below them
    for column_index in [0, columns - 1]:
        upper, lower, _, _ = mapped_pixels[:, 1:-1, column_index].unbind(dim=2)
        equalized_tiles[:, 1:-1, column_index] = torch.addcmul(input=lower, tensor1=row_weights.squeeze(dim=1),
                                                               tensor2=torch.sub(input=upper, other=lower))

    # Tiles of the top and bottom borders blend the tables on their left and right
    for row_index in [0, rows - 1]:
        left, right, _, _ = mapped_pixels[:, row_index, 1:-1].unbind(dim=2)
        equalized_tiles[:, row_index, 1:-1] = torch.addcmul(input=right, tensor1=column_weights,
                                                            tensor2=torch.sub(input=left, other=right))
    return equalized_tiles.div(255.0)


def equalize_clahe_on_device(images: torch.Tensor, clip_limit: float, grid_size: Tuple[int, int]) -> torch.Tensor:
    """
    Applies the contrast-limited adaptive histogram equalization of kornia's equalize_clahe to every channel of the
    images, for the clahe_augment type of DataAugmenter, with the same values: the images are cut into a grid of tiles
    whose clipped histograms give lookup tables (compute_clahe_tiles, compute_clahe_lookup_tables), and every pixel
    takes the blend of the tables of the tiles around it (compute_clahe_equalized_tiles). kornia copies the indices of
    its tiles from the host at every image, which makes the host wait for the device; here they stay on the device.

    Args:
        images (torch.Tensor): The images [..., C, H, W] in [0, 1].
        clip_limit (float): The clip limit of the histograms, relative to their mean over the bins; 0 for no limit.
        grid_size (Tuple[int, int]): The number of tiles of every column and row (GH, GW).

    Returns:
        torch.Tensor: The equalized images in [0, 1], with the shape of the input.
    """
    # Histogram tiles of the images seen as a batch, and the interpolation tiles, a quarter of a histogram tile each
    batch = images.reshape(-1, *images.shape[-3:])
    histogram_tiles, padded_images = compute_clahe_tiles(images=batch, grid_size=grid_size)
    half_tile_height, half_tile_width = histogram_tiles.shape[-2] // 2, histogram_tiles.shape[-1] // 2
    number_channels = padded_images.shape[-3]
    interpolation_tiles = (padded_images.unfold(dimension=1, size=number_channels, step=number_channels)
                           .unfold(dimension=2, size=half_tile_height, step=half_tile_height)
                           .unfold(dimension=3, size=half_tile_width, step=half_tile_width)
                           .squeeze(dim=1)).contiguous()

    # Lookup tables of the histogram tiles, applied and blended over the interpolation tiles
    lookup_tables = compute_clahe_lookup_tables(tiles=histogram_tiles, clip_limit=clip_limit)
    equalized_tiles = compute_clahe_equalized_tiles(interpolation_tiles=interpolation_tiles,
                                                    lookup_tables=lookup_tables)

    # Images put back together from their tiles, the padding removed
    equalized_images = equalized_tiles.permute(0, 3, 1, 4, 2, 5).reshape_as(padded_images)
    return equalized_images[..., :images.shape[-2], :images.shape[-1]].reshape(images.shape)


@lru_cache(maxsize=16)
def get_yuv_matrices(dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gives the [3, 3] matrices of kornia's rgb_to_yuv and yuv_to_rgb on a device, filled from numbers once per dtype and
    device, kornia copying them from the host at every call, for the equalize_histogram_augment type of DataAugmenter.

    Args:
        dtype (torch.dtype): The floating dtype of the images.
        device (torch.device): The device of the images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The RGB to YUV and the YUV to RGB matrices.
    """
    device_matrices = []
    for matrix in [RGB_TO_YUV_MATRIX, YUV_TO_RGB_MATRIX]:
        device_matrix = torch.empty(size=(3, 3), dtype=dtype, device=device)
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                device_matrix[row_index, column_index].fill_(value)
        device_matrices.append(device_matrix)
    return device_matrices[0], device_matrices[1]


def convert_colour_space(images: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Applies a [3, 3] colour matrix (get_yuv_matrices) to the channel vector of every pixel of float images, kornia's
    _apply_linear_transformation, so that the conversions give the values of kornia's rgb_to_yuv and yuv_to_rgb: an
    einsum on the CPU, a 1x1 convolution on the other devices.

    Args:
        images (torch.Tensor): The float images [..., 3, H, W].
        matrix (torch.Tensor): The colour matrix, of the dtype and device of the images.

    Returns:
        torch.Tensor: The converted images, with the shape of the input.
    """
    if images.device.type == "cpu":
        return torch.einsum("oi, ...ihw -> ...ohw", matrix, images).contiguous()
    image_shape = images.shape
    return torch_functional.conv2d(input=images.reshape(-1, 3, image_shape[-2], image_shape[-1]),
                                   weight=matrix.view(3, 3, 1, 1)).reshape(image_shape)
