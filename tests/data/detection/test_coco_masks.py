"""
Tests of vision_training/data/detection/coco_masks.py, the decoding of the COCO segmentations into the instance masks
of the instance segmentation task: exact pixels of the rasterised polygons with and without letterbox padding, union of
overlapping polygon parts, uncompressed and compressed run length encodings, polygons and encodings of the same
object giving the same mask and boxes at the network resolution, also once shrunk, and at the original resolution of a
non-square image (the mask evaluation at that resolution), the boxes read from the segmentations without decoding, and
the instance index maps ('Data.instance_index_maps') giving the masks, objects and boxes of the per-object masks, with
their refusals.
"""

import numpy as np
import pytest
import torch

from vision_training.data.detection.coco_masks import (MAXIMUM_INDEX_MAP_OBJECTS, build_instance_index_map,
                                                        build_instance_mask, compress_counts,
                                                        compute_mask_bounding_boxes, compute_segmentation_bounding_box,
                                                        decode_compressed_counts, decode_run_length_encoding,
                                                        encode_run_length_encoding, expand_instance_index_maps,
                                                        find_index_map_objects, rasterize_polygons)


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


def test_compressed_encoding_writes_the_coco_string_and_decodes_back() -> None:
    """
    Verifies the compressed encoding written by the synthetic instance segmentation generator: hand-worked strings of
    COCO (a run as one character, a negative difference with the run two places before, a long run over two
    characters), and small masks, an empty mask and a 1024 pixel mask with a hole, the size of the hard images,
    decoded back exactly from their compressed string.
    """
    # Runs 3 and 2 are the characters '3' and '2'; the fourth run 2 is written as 2 - 5 = -3, the character 'M'
    # (29 + 48, the sign bit set); a run of 40 takes two groups, 8 with the continuation bit ('X') then 1 ('1')
    assert compress_counts(counts=[3, 2]) == "32"
    assert compress_counts(counts=[0, 5, 1, 2]) == "051M"
    assert compress_counts(counts=[40]) == "X1"

    # Masks of several shapes decode back to themselves through the compressed string
    random_generator = np.random.default_rng(seed=0)
    first_pixel_mask = np.zeros(shape=(3, 4), dtype=np.uint8)
    first_pixel_mask[0, 0] = 1
    large_mask = np.zeros(shape=(1024, 1024), dtype=np.uint8)
    large_mask[100:900, 50:1000] = 1
    large_mask[400:420, 300:700] = 0
    masks = [(random_generator.random(size=(7, 5)) > 0.5).astype(np.uint8), first_pixel_mask, large_mask,
             np.zeros(shape=(4, 4), dtype=np.uint8)]
    for mask in masks:
        compressed_encoding = encode_run_length_encoding(mask=mask, compressed=True)
        assert isinstance(compressed_encoding["counts"], str)
        assert compressed_encoding["size"] == encode_run_length_encoding(mask=mask)["size"]
        assert np.array_equal(decode_run_length_encoding(segmentation=compressed_encoding), mask)



def decode_compressed_counts_one_character_at_a_time(compressed_counts: str) -> list:
    """
    Reference decoder of the compressed COCO counts, reading the string one character at a time as the COCO API does,
    against which the numpy decoder of the training loader is compared.

    Args:
        compressed_counts (str): The 'counts' string of the encoding.

    Returns:
        list: The run lengths.
    """
    counts = []
    position = 0
    while position < len(compressed_counts):
        value, shift, has_more_groups = 0, 0, True
        # Groups of 5 bits of one run, the last one carrying the sign bit
        while has_more_groups:
            group = ord(compressed_counts[position]) - 48
            value |= (group & 0x1F) << (5 * shift)
            has_more_groups = bool(group & 0x20)
            position, shift = position + 1, shift + 1
            if not has_more_groups and group & 0x10:
                value |= -1 << (5 * shift)
        # From the fourth run on, a difference with the run two places before
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts


def test_numpy_decoder_matches_the_character_decoder_on_non_square_masks() -> None:
    """
    Verifies the numpy decoding of the training loader: the run lengths of the compressed strings of random masks
    (long runs over several groups, negative differences, a single row and a single column, an empty string) equal
    those of the character by character decoder as an int64 array, and the masks of non-square images decode back to
    themselves through the OpenCV transpose of the column by column pixels.
    """
    random_generator = np.random.default_rng(seed=3)
    masks = [np.zeros(shape=(1, 9), dtype=np.uint8), np.ones(shape=(9, 1), dtype=np.uint8)]
    # Rectangles of random sizes in non-square images, giving runs from one pixel to thousands of pixels
    for height, width in [(37, 53), (300, 120), (64, 1000)]:
        mask = np.zeros(shape=(height, width), dtype=np.uint8)
        for _ in range(4):
            top, left = random_generator.integers(low=0, high=height), random_generator.integers(low=0, high=width)
            mask[top: top + random_generator.integers(low=1, high=height), left: left + 40] = 1
        masks.append(mask)
    masks.append((random_generator.random(size=(31, 17)) > 0.5).astype(np.uint8))

    # Same run lengths as the reference decoder, and the mask back from its compressed encoding
    for mask in masks:
        encoding = encode_run_length_encoding(mask=mask, compressed=True)
        counts = decode_compressed_counts(compressed_counts=encoding["counts"])
        assert isinstance(counts, np.ndarray) and counts.dtype == np.int64
        assert counts.tolist() == decode_compressed_counts_one_character_at_a_time(compressed_counts=encoding["counts"])
        decoded_mask = decode_run_length_encoding(segmentation=encoding)
        assert decoded_mask.shape == mask.shape and decoded_mask.flags["C_CONTIGUOUS"]
        assert np.array_equal(decoded_mask, mask)
    assert decode_compressed_counts(compressed_counts="").shape == (0,)

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

    # Shrunk by a non-integer factor, both forms give the boxes of the scaled rectangles, the encoding being resized on
    # the pixel centres like the image instead of shifted by half a pixel to the bottom right
    random_generator = np.random.default_rng(seed=0)
    box_differences = []
    for _ in range(40):
        x_start, y_start = (int(value) for value in random_generator.integers(low=0, high=[900, 650]))
        object_width, object_height = (int(value) for value in random_generator.integers(low=20, high=100, size=2))
        x_end, y_end = x_start + object_width, y_start + object_height
        original_mask = np.zeros(shape=(750, 1000), dtype=np.uint8)
        original_mask[y_start:y_end, x_start:x_end] = 1
        polygon = [[float(x_start), float(y_start), float(x_end), float(y_start), float(x_end), float(y_end),
                    float(x_start), float(y_end)]]
        polygon_mask = build_instance_mask(segmentation=polygon, image_width=1000, image_height=750, image_size=256,
                                           keep_ratio=True)
        encoded_mask = build_instance_mask(segmentation=encode_run_length_encoding(mask=original_mask),
                                           image_width=1000, image_height=750, image_size=256, keep_ratio=True)
        both_masks = torch.from_numpy(np.stack(arrays=[encoded_mask, polygon_mask]))
        mask_boxes = compute_mask_bounding_boxes(instance_masks=both_masks)
        box_differences.append((mask_boxes[0] - mask_boxes[1]).numpy())
    assert np.abs(np.stack(arrays=box_differences)).max() <= 1
    assert np.abs(np.mean(a=np.stack(arrays=box_differences), axis=0)).max() < 0.2


def test_masks_at_the_original_resolution_of_a_non_square_image() -> None:
    """
    Verifies that without image size the masks stay at the original resolution of a non-square image, for the mask
    evaluation at that resolution: a polygon covers exactly its pixels, also beyond the first rows and columns of a
    square of the image height, the encoding of the same pixels gives the same mask, and an encoding of another size
    than its image is refused instead of giving a misaligned mask.
    """
    # A polygon reaching the right part of a 7 x 12 image covers exactly the pixels inside it
    polygon = [[3.0, 1.0, 11.0, 1.0, 11.0, 6.0, 3.0, 6.0]]
    expected = np.zeros(shape=(7, 12), dtype=np.uint8)
    expected[1:6, 3:11] = 1
    polygon_mask = build_instance_mask(segmentation=polygon, image_width=12, image_height=7, image_size=None,
                                       keep_ratio=True)
    assert polygon_mask.dtype == np.uint8 and np.array_equal(polygon_mask, expected)

    # The encoding of the same pixels is returned as decoded, whatever the aspect ratio setting
    for keep_ratio in [True, False]:
        encoded_mask = build_instance_mask(segmentation=encode_run_length_encoding(mask=expected), image_width=12,
                                           image_height=7, image_size=None, keep_ratio=keep_ratio)
        assert np.array_equal(encoded_mask, expected)

    # An encoding whose size is not the one of its image is refused
    with pytest.raises(expected_exception=ValueError, match="instead of the 12 x 7"):
        build_instance_mask(segmentation=encode_run_length_encoding(mask=expected), image_width=7, image_height=12,
                            image_size=None, keep_ratio=True)


@pytest.mark.parametrize(argnames="compressed", argvalues=[False, True])
def test_segmentation_boxes_match_the_decoded_masks(compressed: bool) -> None:
    """
    Verifies that the box read from a run length encoding without decoding it is the tight box of its decoded mask
    (an object within one column, an object crossing columns, an object touching the borders, random masks), that an
    empty encoding gives an empty box, and that polygons give the extent of their valid parts, the short parts skipped.

    Args:
        compressed (bool): Whether the encoding is given in its compressed form.
    """
    random_generator = np.random.default_rng(seed=1)
    column_mask = np.zeros(shape=(9, 6), dtype=np.uint8)
    column_mask[2:5, 3] = 1
    crossing_mask = np.zeros(shape=(9, 6), dtype=np.uint8)
    crossing_mask[7:9, 1] = 1
    crossing_mask[0:2, 2] = 1
    border_mask = np.zeros(shape=(5, 7), dtype=np.uint8)
    border_mask[0, 0] = border_mask[4, 6] = 1
    masks = [column_mask, crossing_mask, border_mask] + [
        (random_generator.random(size=(8, 11)) > 0.9).astype(np.uint8) for _ in range(20)]

    for mask in masks:
        encoding = encode_run_length_encoding(mask=mask)
        if compressed:
            encoding = {"size": encoding["size"], "counts": compress_counts(counts=encoding["counts"])}
        rows, columns = np.nonzero(mask)
        expected_box = [float(columns.min()), float(rows.min()), float(columns.max() + 1), float(rows.max() + 1)]
        assert compute_segmentation_bounding_box(segmentation=encoding) == expected_box

    # An empty encoding gives an empty box
    empty_encoding = encode_run_length_encoding(mask=np.zeros(shape=(4, 4), dtype=np.uint8))
    assert compute_segmentation_bounding_box(segmentation=empty_encoding) == [0.0, 0.0, 0.0, 0.0]

    # Polygons give the extent of their parts of at least 3 vertices
    polygons = [[2.0, 3.0, 6.0, 3.0, 6.0, 8.0], [10.0, 1.0, 12.0, 1.0, 11.0, 4.0], [0.0, 0.0, 50.0, 50.0]]
    assert compute_segmentation_bounding_box(segmentation=polygons) == [2.0, 1.0, 12.0, 8.0]
    assert compute_segmentation_bounding_box(segmentation=[[1.0, 1.0, 2.0, 2.0]]) == [0.0, 0.0, 0.0, 0.0]



def build_disjoint_object_masks(image_height: int, image_width: int, seed: int) -> list:
    """
    Builds the visible masks of synthetic objects of a non-square image, as the synthetic instance segmentation
    generator writes them: random rectangles and blobs painted one over the other, every object keeping the pixels no
    later object covers, so that the masks never overlap; one object is a single pixel, which can vanish once the
    image shrinks, and the last one has no visible pixel, leaving an empty mask.

    Args:
        image_height (int): The height of the image.
        image_width (int): The width of the image.
        seed (int): The seed of the random shapes.

    Returns:
        list: The [image_height, image_width] uint8 masks of the objects, in their order.
    """
    random_generator = np.random.default_rng(seed=seed)
    owner = np.zeros(shape=(image_height, image_width), dtype=np.int64)
    number_objects = 9

    # Rectangles and random blobs painted in order, a later object hiding the earlier ones
    for object_number in range(1, number_objects - 1):
        top = random_generator.integers(low=0, high=image_height)
        left = random_generator.integers(low=0, high=image_width)
        height = random_generator.integers(low=1, high=image_height // 2)
        width = random_generator.integers(low=1, high=image_width // 2)
        if object_number % 3 == 0:
            blob = random_generator.random(size=(image_height, image_width)) > 0.8
            owner[blob] = object_number
        else:
            owner[top: top + height, left: left + width] = object_number

    # A single pixel object in the last corner, and the last object leaving no visible pixel (its square repainted by
    # the first object, as when a later object covers it whole)
    owner[image_height - 1, image_width - 1] = number_objects - 1
    owner[0:2, 0:2] = 1
    return [(owner == object_number).astype(np.uint8) for object_number in range(1, number_objects + 1)]


@pytest.mark.parametrize(argnames="keep_ratio", argvalues=[True, False])
@pytest.mark.parametrize(argnames="image_size", argvalues=[24, 64, 160])
def test_index_map_gives_the_per_object_masks_and_boxes(keep_ratio: bool, image_size: int) -> None:
    """
    Verifies that the instance index map of a non-square image, painted from compressed and uncompressed run length
    encodings, expands into exactly the masks build_instance_mask gives object by object, at a shrinking, a slightly
    enlarging and an enlarging network size, letterboxed or stretched, and that find_index_map_objects gives the
    objects still covering a pixel (an object fully hidden by the others never does) and the boxes
    compute_mask_bounding_boxes gives from their masks.

    Args:
        keep_ratio (bool): Whether the image is letterboxed or stretched.
        image_size (int): The size of the network input.
    """
    image_height, image_width = 37, 61
    object_masks = build_disjoint_object_masks(image_height=image_height, image_width=image_width, seed=image_size)
    segmentations = [encode_run_length_encoding(mask=mask, compressed=object_index % 2 == 0)
                     for object_index, mask in enumerate(object_masks)]

    # Map at the network resolution, against the masks of every object resized one by one
    index_map = build_instance_index_map(segmentations=segmentations, image_width=image_width,
                                         image_height=image_height, image_size=image_size, keep_ratio=keep_ratio)
    expected_masks = torch.from_numpy(np.stack(arrays=[
        build_instance_mask(segmentation=segmentation, image_width=image_width, image_height=image_height,
                            image_size=image_size, keep_ratio=keep_ratio) for segmentation in segmentations]))
    assert index_map.shape == (image_size, image_size) and index_map.dtype == np.uint8
    expanded_masks = expand_instance_index_maps(index_maps=torch.from_numpy(index_map)[None],
                                                object_counts=[len(segmentations)])[0]
    assert expanded_masks.dtype == torch.uint8 and torch.equal(expanded_masks, expected_masks)

    # Objects covering a pixel and their boxes, as the per-object masks give them
    is_present, boxes = find_index_map_objects(index_map=index_map, number_objects=len(segmentations))
    expected_present = (expected_masks.flatten(start_dim=1) > 0).any(dim=1).numpy()
    assert np.array_equal(is_present, expected_present) and not is_present[-1]
    assert np.array_equal(boxes, compute_mask_bounding_boxes(instance_masks=expected_masks[expected_present]).numpy())


def test_index_map_refuses_what_one_map_cannot_hold() -> None:
    """
    Verifies the refusals of the instance index maps, each naming the setting to turn off: two masks sharing a pixel, a
    polygon, an encoding of another size than its image or not covering it, and more objects than a uint8 map numbers,
    while an image without objects gives an empty map and expands into [0, S, S] masks.
    """
    first_mask = np.zeros(shape=(6, 8), dtype=np.uint8)
    first_mask[1:4, 1:4] = 1
    second_mask = np.zeros(shape=(6, 8), dtype=np.uint8)
    second_mask[3:5, 3:6] = 1
    arguments = {"image_width": 8, "image_height": 6, "image_size": 16, "keep_ratio": True}

    # Overlapping masks, a polygon, another size and an encoding missing pixels
    with pytest.raises(expected_exception=ValueError, match="share pixels"):
        build_instance_index_map(segmentations=[encode_run_length_encoding(mask=first_mask),
                                                encode_run_length_encoding(mask=second_mask)], **arguments)
    with pytest.raises(expected_exception=ValueError, match="polygon"):
        build_instance_index_map(segmentations=[[[1.0, 1.0, 4.0, 1.0, 4.0, 4.0]]], **arguments)
    with pytest.raises(expected_exception=ValueError, match=r"size \[8, 6\]"):
        build_instance_index_map(segmentations=[encode_run_length_encoding(mask=first_mask.T)], **arguments)
    with pytest.raises(expected_exception=ValueError, match="covers 47 pixels"):
        build_instance_index_map(segmentations=[{"size": [6, 8], "counts": [40, 7]}], **arguments)

    # More objects than a uint8 map holds
    empty_encoding = encode_run_length_encoding(mask=np.zeros(shape=(6, 8), dtype=np.uint8))
    with pytest.raises(expected_exception=ValueError, match="instance_index_maps"):
        build_instance_index_map(segmentations=[empty_encoding] * (MAXIMUM_INDEX_MAP_OBJECTS + 1), **arguments)

    # An image without objects
    empty_map = build_instance_index_map(segmentations=[], **arguments)
    assert empty_map.shape == (16, 16) and not empty_map.any()
    assert expand_instance_index_maps(index_maps=torch.from_numpy(empty_map)[None], object_counts=[0])[0].shape == \
        (0, 16, 16)
    is_present, boxes = find_index_map_objects(index_map=empty_map, number_objects=0)
    assert is_present.shape == (0,) and boxes.shape == (0, 4)
