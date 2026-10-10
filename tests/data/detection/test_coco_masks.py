"""
Tests of vision_training/data/detection/coco_masks.py, the decoding of the COCO segmentations into the instance masks
of the instance segmentation task: exact pixels of the rasterised polygons with and without letterbox padding, union of
overlapping polygon parts, uncompressed and compressed run length encodings, and polygons and encodings of the same
object giving the same mask at the network resolution.
"""

import numpy as np
import pytest

from typing import List

from vision_training.data.detection.coco_masks import (build_instance_mask, decode_run_length_encoding,
                                                        encode_run_length_encoding, rasterize_polygons)


def compress_counts(counts: List[int]) -> str:
    """
    Writes run lengths as the string of a compressed COCO run length encoding, the reference encoding of COCO, so that
    the tests feed decode_run_length_encoding with the compressed form of hand-made masks.

    Args:
        counts (List[int]): The run lengths, alternating background and object pixels.

    Returns:
        str: The compressed 'counts' string.
    """
    characters = []
    for count_index, count in enumerate(counts):
        # From the third run on, the difference with the run two places before is written
        value = count - counts[count_index - 2] if count_index > 2 else count

        # Groups of 5 bits, the sixth bit flagging another group, the fifth bit of the last group the sign
        has_more_groups = True
        while has_more_groups:
            group = value & 0x1F
            value >>= 5
            has_more_groups = value != -1 if group & 0x10 else value != 0
            characters.append(chr((group | 0x20 if has_more_groups else group) + 48))
    return "".join(characters)


def build_square_mask(size: int, first_pixel: int, end_pixel: int) -> np.ndarray:
    """
    Builds the expected mask of an axis-aligned square covering the pixels [first_pixel, end_pixel) on both axes.

    Args:
        size (int): The side of the mask.
        first_pixel (int): The first covered row and column.
        end_pixel (int): One past the last covered row and column.

    Returns:
        np.ndarray: The [size, size] uint8 mask.
    """
    mask = np.zeros(shape=(size, size), dtype=np.uint8)
    mask[first_pixel:end_pixel, first_pixel:end_pixel] = 1
    return mask


def test_square_polygon_gives_its_exact_pixels() -> None:
    """
    Verifies that a square polygon whose corners lie on pixel edges covers exactly the pixels inside it, without the
    extra row and column of cv2.fillPoly, at the original size, scaled, and letterboxed with padding above and below.
    """
    square = [[2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]]
    original_size = rasterize_polygons(polygons=square, image_width=10, image_height=10, image_size=10,
                                       keep_ratio=True)
    assert np.array_equal(original_size, build_square_mask(size=10, first_pixel=2, end_pixel=6))

    # Scaled twice, the square covers twice as many rows and columns
    scaled = rasterize_polygons(polygons=square, image_width=10, image_height=10, image_size=20, keep_ratio=False)
    assert np.array_equal(scaled, build_square_mask(size=20, first_pixel=4, end_pixel=12))

    # A 10 x 5 image letterboxed into 20 pixels is scaled twice and moved down by 5 rows of padding
    letterboxed = rasterize_polygons(polygons=square, image_width=10, image_height=5, image_size=20, keep_ratio=True)
    expected = np.zeros(shape=(20, 20), dtype=np.uint8)
    expected[9:17, 4:12] = 1
    assert np.array_equal(letterboxed, expected)


def test_overlapping_polygon_parts_are_united_and_short_parts_skipped() -> None:
    """
    Verifies that two overlapping parts of one object give the union of their pixels (a single even-odd fill would
    cut a hole where they overlap), and that a part of fewer than 3 vertices adds nothing.
    """
    parts = [[2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0], [4.0, 4.0, 8.0, 4.0, 8.0, 8.0, 4.0, 8.0], [0.0, 0.0, 9.0, 9.0]]
    mask = rasterize_polygons(polygons=parts, image_width=10, image_height=10, image_size=10, keep_ratio=True)

    expected = build_square_mask(size=10, first_pixel=2, end_pixel=6) | build_square_mask(size=10, first_pixel=4,
                                                                                          end_pixel=8)
    assert np.array_equal(mask, expected)


@pytest.mark.parametrize(argnames="compressed", argvalues=[False, True])
def test_run_length_encodings_decode_column_by_column(compressed: bool) -> None:
    """
    Verifies that hand-made masks, whose runs follow the pixels column by column, are decoded exactly from their
    uncompressed and compressed encodings, including a mask starting on the object, a mask with long runs (several
    groups per run) and an empty mask.

    Args:
        compressed (bool): Whether the encoding is given in its compressed form.
    """
    random_generator = np.random.default_rng(seed=0)
    first_pixel_mask = np.zeros(shape=(3, 4), dtype=np.uint8)
    first_pixel_mask[0, 0] = 1
    large_mask = np.zeros(shape=(60, 70), dtype=np.uint8)
    large_mask[5:50, 10:65] = 1
    masks = [(random_generator.random(size=(7, 5)) > 0.5).astype(np.uint8), first_pixel_mask, large_mask,
             np.zeros(shape=(4, 4), dtype=np.uint8)]

    for mask in masks:
        encoding = encode_run_length_encoding(mask=mask)
        if compressed:
            encoding = {"size": encoding["size"], "counts": compress_counts(counts=encoding["counts"])}
        assert np.array_equal(decode_run_length_encoding(segmentation=encoding), mask)

    # The first column is read before the second one
    column_mask = decode_run_length_encoding(segmentation={"size": [2, 2], "counts": [0, 2, 2]})
    assert column_mask.tolist() == [[1, 0], [1, 0]]


def test_run_length_encoding_must_cover_the_image() -> None:
    """
    Verifies that an encoding whose runs do not add up to the image size is refused instead of giving a shifted mask.
    """
    with pytest.raises(ValueError, match="covers 3 pixels"):
        decode_run_length_encoding(segmentation={"size": [2, 2], "counts": [1, 2]})


def test_polygon_and_encoding_of_one_object_give_the_same_mask() -> None:
    """
    Verifies that an object given as a polygon or as the encoding of its pixels gives the same instance mask at the
    network resolution, the encoding being decoded at the original size then resized and padded like the image.
    """
    square = [[2.0, 2.0, 6.0, 2.0, 6.0, 6.0, 2.0, 6.0]]
    original_mask = np.zeros(shape=(5, 10), dtype=np.uint8)
    original_mask[2:5, 2:6] = 1
    polygon = [[2.0, 2.0, 6.0, 2.0, 6.0, 5.0, 2.0, 5.0]]

    for keep_ratio in [True, False]:
        polygon_mask = build_instance_mask(segmentation=polygon, image_width=10, image_height=5, image_size=20,
                                           keep_ratio=keep_ratio)
        encoded_mask = build_instance_mask(segmentation=encode_run_length_encoding(mask=original_mask),
                                           image_width=10, image_height=5, image_size=20, keep_ratio=keep_ratio)
        assert polygon_mask.dtype == encoded_mask.dtype == np.uint8
        assert polygon_mask.sum() > 0 and np.array_equal(polygon_mask, encoded_mask)

    # A square image keeps the same mask with or without the aspect ratio
    assert np.array_equal(build_instance_mask(segmentation=square, image_width=10, image_height=10, image_size=10,
                                              keep_ratio=True), build_square_mask(size=10, first_pixel=2, end_pixel=6))
