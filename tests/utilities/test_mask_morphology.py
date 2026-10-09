"""
Tests of vision_training/utilities/mask_morphology.py: the dilation of boolean masks without effect of the outside of
the image, the bands of the regions within a distance of their contours ignoring the neutral pixels, and the weights
of the pixels decreasing with their distance to the true edges.
"""

import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.utilities.mask_morphology import (compute_contour_bands, compute_edge_distance_weights,
                                                       dilate_masks)


def test_dilation_reaches_the_chessboard_neighbours() -> None:
    """
    Verifies that a single pixel dilated by a radius of 1 covers its 3 x 3 neighbourhood, by 2 its 5 x 5
    neighbourhood, and that a radius of 0 changes nothing.
    """
    masks = torch.zeros(size=(1, 7, 7), dtype=torch.bool)
    masks[0, 3, 3] = True
    assert int(dilate_masks(masks=masks, radius=1).sum()) == 9
    assert int(dilate_masks(masks=masks, radius=2).sum()) == 25
    assert torch.equal(input=dilate_masks(masks=masks, radius=0), other=masks)


def test_contour_bands_ignore_the_image_border_and_the_neutral_pixels() -> None:
    """
    Verifies the band of a square region: its outer ring of pixels for a radius of 1 and its two outer rings for a
    radius of 2, the side touching the image border and the side touching neutral pixels drawing no contour.
    """
    regions = torch.zeros(size=(1, 8, 8), dtype=torch.bool)
    regions[0, 2:6, 2:6] = True
    no_neutral_pixels = torch.zeros(size=(1, 8, 8), dtype=torch.bool)

    # Outer ring of the 4 x 4 square, then the whole square for a radius of 2
    edges = compute_contour_bands(regions=regions, neutral_pixels=no_neutral_pixels, radius=1)
    assert int(edges.sum()) == 16 - 4 and not bool(edges[0, 3:5, 3:5].any())
    assert int(compute_contour_bands(regions=regions, neutral_pixels=no_neutral_pixels, radius=2).sum()) == 16

    # A region touching the left border, and neutral pixels on its right side, keeps only its top and bottom edges
    border_regions = torch.zeros(size=(1, 8, 8), dtype=torch.bool)
    border_regions[0, 2:6, 0:4] = True
    neutral_pixels = torch.zeros(size=(1, 8, 8), dtype=torch.bool)
    neutral_pixels[0, :, 4:] = True
    border_edges = compute_contour_bands(regions=border_regions, neutral_pixels=neutral_pixels, radius=1)
    assert border_edges[0].nonzero().tolist() == [[2, column] for column in range(4)] + [[5, column]
                                                                                       for column in range(4)]


def test_edge_weights_decrease_with_the_distance_to_the_true_edges() -> None:
    """
    Verifies the weights of a mask split into two classes at column 4: 1 on the two edge columns, decreasing by
    1 / (maximum distance + 1) per pixel away from them, 0 beyond the maximum distance and on the ignored pixels, whose
    border with a class draws no edge.
    """
    masks = torch.zeros(size=(1, 3, 12), dtype=torch.int64)
    masks[0, :, 4:] = 1
    masks[0, :, 10:] = IGNORED_LABEL_INDEX
    weights = compute_edge_distance_weights(masks=masks, maximum_distance=2)

    expected_row = torch.tensor(data=[0.0, 1 / 3, 2 / 3, 1.0, 1.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(input=weights[0], other=expected_row.expand(3, -1))
