"""
Sliding-window segmentation of the images larger than the network input: the image is cut into overlapping tiles of
the network size processed at their own resolution, and the class probabilities of the tiles are averaged on their
overlaps, so that a small object keeps its size instead of shrinking with the image resized to the network input.
Used by the segmentation trainer to evaluate the test images of a run trained on tiles ('Data.tiling'), and usable by
the inference clients with any function giving the probabilities of a batch of tiles (e.g. a request to Triton).
"""

import torch

from typing import Callable, List


def compute_tile_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    """
    Gives the first pixel of every tile along one side of an image, the tiles advancing by tile_size - overlap pixels
    and the last tile ending on the last pixel, so that every pixel is covered and the tiles never leave the image. A
    side shorter than a tile gets one tile, padded by segment_image_by_tiles.

    Args:
        length (int): The number of pixels of the side.
        tile_size (int): The size of the square tiles, the network input size.
        overlap (int): The number of pixels shared by two neighbouring tiles, smaller than tile_size.

    Returns:
        List[int]: The first pixel of every tile, in increasing order.
    """
    if length <= tile_size:
        return [0]
    tile_starts = list(range(0, length - tile_size, tile_size - overlap))
    return tile_starts + [length - tile_size]


def segment_image_by_tiles(image: torch.Tensor, predict_tile_probabilities: Callable[[torch.Tensor], torch.Tensor],
                           tile_size: int, overlap: int, tiles_per_batch: int) -> torch.Tensor:
    """
    Segments a whole image tile by tile: the image, padded with black pixels on its bottom and right sides when it is
    smaller than a tile, is cut into the overlapping tiles of compute_tile_starts, the tiles go through the model in
    batches, and the probabilities of every pixel are averaged over the tiles covering it.

    Args:
        image (torch.Tensor): The [3, H, W] float BGR image in [0, 255], on the device of the model.
        predict_tile_probabilities (Callable[[torch.Tensor], torch.Tensor]): The function giving the [N, K, S, S]
            class probabilities of a [N, 3, S, S] batch of tiles, e.g. the segmenter in evaluation mode.
        tile_size (int): The size S of the square tiles, the network input size.
        overlap (int): The number of pixels shared by two neighbouring tiles, smaller than tile_size.
        tiles_per_batch (int): The largest number of tiles given to the model at once.

    Returns:
        torch.Tensor: The [K, H, W] float32 class probabilities of the image.
    """
    image_height, image_width = image.shape[-2:]

    # Black padding of the sides shorter than a tile, cut off from the probabilities at the end
    padded_image = torch.nn.functional.pad(input=image, pad=(0, max(0, tile_size - image_width), 0,
                                                             max(0, tile_size - image_height)))
    padded_height, padded_width = padded_image.shape[-2:]
    tile_positions = [(tile_top, tile_left)
                      for tile_top in compute_tile_starts(length=padded_height, tile_size=tile_size, overlap=overlap)
                      for tile_left in compute_tile_starts(length=padded_width, tile_size=tile_size, overlap=overlap)]

    # Sum of the probabilities of every pixel over its tiles, and number of tiles covering it
    probability_sums = None
    tile_counts = torch.zeros(size=(padded_height, padded_width), dtype=torch.float32, device=image.device)
    for batch_start in range(0, len(tile_positions), tiles_per_batch):
        batch_positions = tile_positions[batch_start:batch_start + tiles_per_batch]
        tiles = torch.stack(tensors=[padded_image[:, tile_top:tile_top + tile_size, tile_left:tile_left + tile_size]
                                     for tile_top, tile_left in batch_positions])
        tile_probabilities = predict_tile_probabilities(tiles).float()

        # Add the probabilities of every tile at its place in the image
        if probability_sums is None:
            probability_sums = torch.zeros(size=(tile_probabilities.shape[1], padded_height, padded_width),
                                           dtype=torch.float32, device=image.device)
        for (tile_top, tile_left), probabilities in zip(batch_positions, tile_probabilities):
            probability_sums[:, tile_top:tile_top + tile_size, tile_left:tile_left + tile_size] += probabilities
            tile_counts[tile_top:tile_top + tile_size, tile_left:tile_left + tile_size] += 1.0

    # Mean over the covering tiles, on the pixels of the image only
    return (probability_sums / tile_counts)[:, :image_height, :image_width]
