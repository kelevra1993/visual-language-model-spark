"""
Tests of vision_training/models/segmentation/sliding_window.py: the tile positions covering a side of an image, and
the segmentation of a whole image tile by tile with the padding of the small images, the batches of tiles and the
average of the probabilities on the overlaps.
"""

import functools
import torch

from typing import List

from vision_training.models.segmentation.sliding_window import compute_tile_starts, segment_image_by_tiles


def predict_pixel_probabilities(tiles: torch.Tensor, batch_sizes: List[int]) -> torch.Tensor:
    """
    Stands in for the segmenter of the sliding-window tests with a prediction of every pixel from its own value only,
    so that the tiled segmentation of an image must equal the prediction of the whole image, and records the number
    of tiles of every call.

    Args:
        tiles (torch.Tensor): The [N, 3, S, S] batch of tiles in [0, 255].
        batch_sizes (List[int]): The number of tiles of every call, appended to.

    Returns:
        torch.Tensor: The [N, 2, S, S] probabilities, the first channel of the pixel over 255 for class 0.
    """
    batch_sizes.append(tiles.shape[0])
    first_class_probabilities = tiles[:, 0] / 255.0
    return torch.stack(tensors=[first_class_probabilities, 1.0 - first_class_probabilities], dim=1)


def predict_tile_brightness(tiles: torch.Tensor) -> torch.Tensor:
    """
    Stands in for the segmenter of the sliding-window tests with a prediction depending on the whole tile: the mean
    brightness of the tile is the probability of class 0 on all its pixels, so that the pixels covered by two tiles
    show the average of both.

    Args:
        tiles (torch.Tensor): The [N, 3, S, S] batch of tiles in [0, 255].

    Returns:
        torch.Tensor: The [N, 2, S, S] probabilities, constant over every tile.
    """
    tile_brightness = (tiles[:, 0].mean(dim=(1, 2)) / 255.0).view(-1, 1, 1).expand(-1, *tiles.shape[-2:])
    return torch.stack(tensors=[tile_brightness, 1.0 - tile_brightness], dim=1)


def test_tiles_cover_every_pixel_without_leaving_the_image() -> None:
    """
    Verifies that the tiles advance by the tile size minus the overlap, the last one ending on the last pixel, and
    that a side not longer than a tile gets a single tile.
    """
    assert compute_tile_starts(length=100, tile_size=40, overlap=10) == [0, 30, 60]
    assert compute_tile_starts(length=88, tile_size=32, overlap=8) == [0, 24, 48, 56]
    assert compute_tile_starts(length=40, tile_size=40, overlap=10) == [0]
    assert compute_tile_starts(length=25, tile_size=40, overlap=10) == [0]


def test_tiled_segmentation_equals_the_whole_image_prediction() -> None:
    """
    Verifies that a model predicting every pixel from its own value gives the same probabilities by tiles as on the
    whole image, for an image larger than a tile cut into 6 tiles given 4 at a time, and for an image smaller than a
    tile, padded then cropped back.
    """
    image = torch.rand(size=(3, 50, 70), generator=torch.Generator().manual_seed(0)) * 255.0
    batch_sizes = []
    probabilities = segment_image_by_tiles(
        image=image, predict_tile_probabilities=functools.partial(predict_pixel_probabilities, batch_sizes=batch_sizes),
        tile_size=32, overlap=8, tiles_per_batch=4)
    assert batch_sizes == [4, 2]
    assert torch.allclose(input=probabilities, other=predict_pixel_probabilities(tiles=image[None],
                                                                                 batch_sizes=[])[0])

    small_image = image[:, :20, :24]
    small_probabilities = segment_image_by_tiles(
        image=small_image, predict_tile_probabilities=functools.partial(predict_pixel_probabilities, batch_sizes=[]),
        tile_size=32, overlap=8, tiles_per_batch=4)
    assert small_probabilities.shape == (2, 20, 24)
    assert torch.allclose(input=small_probabilities, other=predict_pixel_probabilities(tiles=small_image[None],
                                                                                       batch_sizes=[])[0])


def test_overlapping_tiles_are_averaged() -> None:
    """
    Verifies that the probabilities of the pixels covered by two tiles are the mean of the predictions of both: an
    image black on its first 32 columns and white on its last 24 is cut into a black tile (columns 0 to 31) and a
    tile three quarters white (columns 24 to 55).
    """
    image = torch.zeros(size=(3, 32, 56))
    image[:, :, 32:] = 255.0
    probabilities = segment_image_by_tiles(image=image, predict_tile_probabilities=predict_tile_brightness,
                                           tile_size=32, overlap=8, tiles_per_batch=8)

    expected_row = torch.cat(tensors=[torch.zeros(size=(24,)), torch.full(size=(8,), fill_value=0.375),
                                      torch.full(size=(24,), fill_value=0.75)])
    assert torch.allclose(input=probabilities[0], other=expected_row.expand(32, -1))
    assert torch.allclose(input=probabilities.sum(dim=0), other=torch.ones(size=(32, 56)))
