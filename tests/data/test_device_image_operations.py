"""
Tests of the operations of torchvision and kornia ported to the device in
vision_training/data/device_image_operations.py for the augmentation types of DataAugmenter: the rotation gives the
values of torchvision's rotate for images, sequences and masks at every kind of angle, the CLAHE those of kornia's
equalize_clahe for padded and unpadded images, sequences, several grids and clip limits, its tile indices those of
kornia's _map_luts, and a grid that is not square, which kornia refuses, is equalized consistently with its
transposition, the YUV conversions give the values of kornia's rgb_to_yuv and yuv_to_rgb, every comparison being exact
on the CPU and on the GPU, and on the GPU (skipped without one) none of the operations makes the host wait for the
device.
"""

import pytest
import torch
import warnings

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as vision_functional
from torchvision.transforms.v2 import InterpolationMode
from typing import Tuple

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.device_image_operations import (convert_colour_space, equalize_clahe_on_device,
                                                          get_clahe_tile_indices, get_yuv_matrices, rotate_on_device)

# kornia decorates some of its functions with torch.jit.script when it is imported, a harmless deprecation notice
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
    from kornia.color import rgb_to_yuv, yuv_to_rgb
    from kornia.enhance import equalize_clahe

# Devices of the comparisons, the GPU when there is one
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# Angles of the rotation tests: no rotation, the pixel permutations (180 degrees, and 90 degrees for square images
# only), small positive and negative angles, an angle just below a full turn and one too small to move a pixel
ROTATION_ANGLES = [0.0, 180.0, 90.0, -90.0, 3.7, -17.25, 359.5, 1e-9]


def create_random_images(shape: Tuple[int, ...], seed: int) -> torch.Tensor:
    """
    Creates float images of random values in [0, 1] with saturated corners and a row of values at the edges of the
    histogram bins, so that the CLAHE comparisons cover the first and last bins and the bin boundaries.

    Args:
        shape (Tuple[int, ...]): The shape of the images, [..., C, H, W].
        seed (int): The seed of the generator.

    Returns:
        torch.Tensor: The images.
    """
    images = torch.rand(size=shape, generator=torch.Generator().manual_seed(seed))
    images[..., :4, :4] = 0.0
    images[..., -4:, -4:] = 1.0
    images[..., 5, :] = torch.arange(shape[-1]) / 256.0 % 1.0
    return images


@pytest.mark.parametrize(argnames="device", argvalues=DEVICES)
@pytest.mark.parametrize(argnames="shape", argvalues=[(3, 64, 64), (3, 48, 80), (2, 3, 32, 32)])
def test_rotation_gives_the_values_of_torchvision(device: str, shape: Tuple[int, ...]) -> None:
    """
    Verifies that rotate_on_device gives exactly the bilinear rotation of torchvision filled with black, for square and
    oblong images and for sequences, at every kind of angle, the right angles being pixel permutations for square
    images only.

    Args:
        device (str): The device of the images.
        shape (Tuple[int, ...]): The shape of the images.
    """
    images = (torch.rand(size=shape, generator=torch.Generator().manual_seed(0)) * 255.0).to(device=device)
    for angle in ROTATION_ANGLES:
        expected_images = vision_functional.rotate(images, angle=angle, interpolation=InterpolationMode.BILINEAR,
                                                   fill=0.0)
        rotated_images = rotate_on_device(image=images, angle=angle, interpolation="bilinear", fill=0.0)
        assert torch.equal(input=rotated_images, other=expected_images), angle


@pytest.mark.parametrize(argnames="device", argvalues=DEVICES)
def test_mask_rotation_gives_the_values_of_torchvision(device: str) -> None:
    """
    Verifies that rotate_on_device gives exactly the nearest rotation of the masks of torchvision, with their dtype: a
    class mask whose uncovered corners take the ignored label, binary instance masks whose corners stay background, and
    an image without any instance.

    Args:
        device (str): The device of the masks.
    """
    generator = torch.Generator().manual_seed(1)
    class_mask = torch.randint(low=0, high=5, size=(48, 80), generator=generator)
    instance_masks = torch.randint(low=0, high=2, size=(3, 64, 64), generator=generator).to(dtype=torch.uint8)
    no_instance_masks = torch.zeros(size=(0, 64, 64), dtype=torch.uint8)
    for mask, fill in [(class_mask, IGNORED_LABEL_INDEX), (instance_masks, 0), (no_instance_masks, 0)]:
        mask = mask.to(device=device)
        for angle in [7.3, -12.0, 90.0, 180.0]:
            expected_mask = vision_functional.rotate(tv_tensors.Mask(mask), angle=angle,
                                                     fill=fill).as_subclass(torch.Tensor)
            rotated_mask = rotate_on_device(image=mask.unsqueeze(dim=0) if mask.dim() < 3 else mask, angle=angle,
                                            interpolation="nearest", fill=fill)
            rotated_mask = rotated_mask.squeeze(dim=0) if mask.dim() < 3 else rotated_mask
            assert rotated_mask.dtype == mask.dtype and torch.equal(input=rotated_mask, other=expected_mask), angle


@pytest.mark.parametrize(argnames="device", argvalues=DEVICES)
@pytest.mark.parametrize(argnames="shape", argvalues=[(3, 64, 64), (3, 50, 70), (2, 3, 64, 96), (1, 33, 47)])
def test_clahe_gives_the_values_of_kornia(device: str, shape: Tuple[int, ...]) -> None:
    """
    Verifies that equalize_clahe_on_device gives exactly the values of kornia's equalize_clahe, for images that the
    grid covers exactly or after a padding, sequences and single channels, on square grids of 1 to 8 tiles a side,
    with the clip limit of the data augmentation, without limit and with kornia's default.

    Args:
        device (str): The device of the images.
        shape (Tuple[int, ...]): The shape of the images.
    """
    images = create_random_images(shape=shape, seed=2).to(device=device)
    for grid_size in [(4, 4), (2, 2), (1, 1), (8, 8)]:
        for clip_limit in [2.0, 0.0, 40.0]:
            expected_images = equalize_clahe(images, clip_limit=clip_limit, grid_size=grid_size)
            equalized_images = equalize_clahe_on_device(images=images, clip_limit=clip_limit, grid_size=grid_size)
            assert torch.equal(input=equalized_images, other=expected_images), (grid_size, clip_limit)


def test_clahe_tile_indices_are_those_of_kornia() -> None:
    """
    Verifies the indices of the histogram tiles blended by the inner interpolation tiles, built on the device by
    get_clahe_tile_indices, against the construction of kornia's _map_luts, for grids of 1 to 8 tiles a side.
    """
    for interpolation_tiles in [2, 4, 8, 16]:
        expected_row_indices = torch.empty(size=(0, 4), dtype=torch.long)
        expected_column_indices = torch.empty(size=(0, 4), dtype=torch.long)
        if interpolation_tiles > 2:
            floor = torch.arange(start=1, end=interpolation_tiles - 1).view(-1, 1).div(2, rounding_mode="trunc")
            repeats = (interpolation_tiles - 2) // 2
            expected_row_indices = torch.tensor(data=[[0, 0, 1, 1], [-1, -1, 0, 0]] * repeats) + floor
            expected_column_indices = torch.tensor(data=[[0, 1, 0, 1], [-1, 0, -1, 0]] * repeats) + floor
        row_indices, column_indices = get_clahe_tile_indices(interpolation_rows=interpolation_tiles,
                                                             interpolation_columns=interpolation_tiles,
                                                             device=torch.device("cpu"))
        assert torch.equal(input=row_indices, other=expected_row_indices)
        assert torch.equal(input=column_indices, other=expected_column_indices)


@pytest.mark.parametrize(argnames="grid_size", argvalues=[(2, 4), (4, 8), (4, 4)])
def test_clahe_of_a_grid_that_is_not_square_follows_its_transposition(grid_size: Tuple[int, int]) -> None:
    """
    Verifies that a grid that is not square, which kornia's indexing refuses, is equalized consistently: equalizing the
    image on a grid of GH x GW tiles gives, within float rounding, the transposition of the equalized transposed image
    on a grid of GW x GH tiles, which fails when the rows and the columns of the grid are exchanged anywhere.

    Args:
        grid_size (Tuple[int, int]): The number of tiles of every column and row.
    """
    images = create_random_images(shape=(3, 64, 128), seed=3)
    equalized_images = equalize_clahe_on_device(images=images, clip_limit=2.0, grid_size=grid_size)
    equalized_transposition = equalize_clahe_on_device(images=images.transpose(-2, -1).contiguous(), clip_limit=2.0,
                                                       grid_size=(grid_size[1], grid_size[0])).transpose(-2, -1)
    assert torch.allclose(input=equalized_images, other=equalized_transposition, atol=1e-6)
    assert float(equalized_images.min()) >= 0.0 and float(equalized_images.max()) <= 1.0


@pytest.mark.parametrize(argnames="device", argvalues=DEVICES)
def test_colour_conversions_give_the_values_of_kornia(device: str) -> None:
    """
    Verifies that convert_colour_space with the matrices of get_yuv_matrices gives exactly kornia's rgb_to_yuv and
    yuv_to_rgb, the einsum of the CPU and the convolution of the GPU.

    Args:
        device (str): The device of the images.
    """
    images = torch.rand(size=(2, 3, 40, 56), generator=torch.Generator().manual_seed(4)).to(device=device)
    rgb_to_yuv_matrix, yuv_to_rgb_matrix = get_yuv_matrices(dtype=images.dtype, device=images.device)
    assert torch.equal(input=convert_colour_space(images=images, matrix=rgb_to_yuv_matrix), other=rgb_to_yuv(images))
    assert torch.equal(input=convert_colour_space(images=images, matrix=yuv_to_rgb_matrix), other=yuv_to_rgb(images))


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The training batches are augmented on the GPU.")
def test_device_operations_on_the_gpu_never_make_the_host_wait() -> None:
    """
    Verifies that the rotation of images, class masks and instance masks, the CLAHE and the YUV conversions never make
    the host wait for the device on the GPU, which the synchronization debug mode of PyTorch turns into an error, their
    constants being filled on the device instead of copied from the host.
    """
    images = (torch.rand(size=(2, 3, 64, 64), generator=torch.Generator().manual_seed(5)) * 255.0).to(device="cuda")
    class_mask = torch.zeros(size=(1, 64, 64), dtype=torch.int64, device="cuda")
    instance_masks = torch.ones(size=(2, 64, 64), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    # Every operation with every synchronization of the host turned into an error
    torch.cuda.set_sync_debug_mode(debug_mode="error")
    try:
        rotated_images = rotate_on_device(image=images, angle=11.0, interpolation="bilinear", fill=0.0)
        rotated_class_mask = rotate_on_device(image=class_mask, angle=-6.0, interpolation="nearest",
                                              fill=IGNORED_LABEL_INDEX)
        rotated_instance_masks = rotate_on_device(image=instance_masks, angle=4.0, interpolation="nearest", fill=0)
        equalized_images = equalize_clahe_on_device(images=images / 255.0, clip_limit=2.0, grid_size=(4, 4))
        rgb_to_yuv_matrix, yuv_to_rgb_matrix = get_yuv_matrices(dtype=images.dtype, device=images.device)
        converted_images = convert_colour_space(images=convert_colour_space(images=images / 255.0,
                                                                            matrix=rgb_to_yuv_matrix),
                                                matrix=yuv_to_rgb_matrix)
    finally:
        torch.cuda.set_sync_debug_mode(debug_mode="default")
    for output in [rotated_images, rotated_class_mask, rotated_instance_masks, equalized_images, converted_images]:
        assert output.device.type == "cuda"
    assert rotated_class_mask.dtype == torch.int64 and rotated_instance_masks.dtype == torch.uint8
