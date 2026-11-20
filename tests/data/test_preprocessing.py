"""
Tests of vision_training/data/preprocessing.py: read_image reads the files of more than 8 bits per channel as before
with one warning, restore_boxes_to_original_image inverts preprocess_boxes, bringing the detected boxes back to the
original image coordinates, clipped to the image or kept whole for the pasting of the instance masks, a very elongated
image keeps a short side of one pixel in the letterbox, preprocess_boxes moves polygons like boxes, and
restore_prediction_to_original_image brings segmentation class maps and probabilities back to their original images.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from vision_training.data import preprocessing
from vision_training.data.preprocessing import (compute_letterbox_region, preprocess_boxes, preprocess_image,
                                                read_image, restore_boxes_to_original_image,
                                                restore_prediction_to_original_image)


def test_images_of_more_than_8_bits_are_read_as_before_with_one_warning(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that read_image gives the 8 bit pixels cv2.imread gave for a 16 bit PNG, the values divided by 256, and
    warns once for every file of that pixel type, since a 12 bit source stored in 16 bits keeps only 4 bits, while an
    8 bit file is read without any warning.

    Args:
        tmp_path (Path): The pytest fixture folder holding the written images.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture emptying the pixel types already reported.
    """
    monkeypatch.setattr(target=preprocessing, name="REPORTED_IMAGE_DEPTHS", value=set())

    # A 12 bit source stored in a 16 bit PNG, and the same scene in an 8 bit PNG
    deep_pixels = np.random.default_rng(seed=0).integers(low=0, high=4096, size=(6, 8, 3), dtype=np.uint16)
    deep_image_path, shallow_image_path = tmp_path / "deep.png", tmp_path / "shallow.png"
    cv2.imwrite(filename=str(deep_image_path), img=deep_pixels)
    cv2.imwrite(filename=str(shallow_image_path), img=(deep_pixels // 16).astype(np.uint8))

    # The 8 bit file gives its pixels without any warning
    assert np.array_equal(read_image(image_path=shallow_image_path), (deep_pixels // 16).astype(np.uint8))
    assert "Read With 8 Bits Per Channel" not in capsys.readouterr().out

    # The 16 bit file read twice gives the pixels of cv2.imread both times, with a single warning naming it
    first_image, second_image = read_image(image_path=deep_image_path), read_image(image_path=deep_image_path)
    console_output = capsys.readouterr().out
    assert first_image.dtype == np.uint8 and np.array_equal(first_image, cv2.imread(filename=str(deep_image_path)))
    assert np.array_equal(second_image, first_image)
    assert console_output.count("Image Of uint16 Pixels Read With 8 Bits Per Channel") == 1
    assert "deep.png" in console_output


def test_very_elongated_image_keeps_a_short_side_of_one_pixel() -> None:
    """
    Verifies that an image one pixel high and 2000 wide letterboxed into 512 pixels keeps a resized height of one
    pixel instead of rounding to none, which failed in the OpenCV resize, and that its boxes come back to the image.
    """
    # A 1 x 2000 image scaled by 0.256 rounds its height to 0.256 pixels, kept to one row in the middle of the input
    assert compute_letterbox_region(image_height=1, image_width=2000, image_size=512) == (255, 0, 1, 512)
    processed_image = preprocess_image(image=np.full(shape=(1, 2000, 3), fill_value=200, dtype=np.uint8),
                                       image_size=512, keep_ratio=True)
    assert processed_image.shape == (512, 512, 3) and np.all(processed_image[255] == 200)
    assert processed_image[:255].max() == 0 and processed_image[256:].max() == 0

    # A box of the image comes back to its original coordinates through the network input
    boxes = [[100.0, 0.0, 1000.0, 1.0]]
    network_boxes = preprocess_boxes(bounding_boxes=boxes, image_width=2000, image_height=1, image_size=512,
                                     keep_ratio=True)
    restored_boxes = restore_boxes_to_original_image(bounding_boxes=np.array(network_boxes), image_width=2000,
                                                     image_height=1, image_size=512, keep_ratio=True)
    assert np.allclose(restored_boxes, boxes)


@pytest.mark.parametrize(argnames=("image_width", "image_height"), argvalues=[(640, 480)])
@pytest.mark.parametrize(argnames="keep_ratio", argvalues=[True])
def test_restored_boxes_invert_the_preprocessing(image_width: int, image_height: int, keep_ratio: bool) -> None:
    """
    Verifies that mapping boxes from the network input back to the original image exactly inverts the box
    preprocessing, for a landscape image with aspect ratio preservation.

    Args:
        image_width (int): The width of the original image.
        image_height (int): The height of the original image.
        keep_ratio (bool): Whether the preprocessing keeps the aspect ratio.
    """
    boxes = [[10.0, 20.0, 300.0, 150.0], [200.0, 5.0, 470.0, 470.0]]

    preprocessed_boxes = preprocess_boxes(bounding_boxes=boxes, image_width=image_width, image_height=image_height,
                                          image_size=1024, keep_ratio=keep_ratio)
    restored_boxes = restore_boxes_to_original_image(bounding_boxes=np.array(preprocessed_boxes),
                                                     image_width=image_width, image_height=image_height,
                                                     image_size=1024, keep_ratio=keep_ratio)

    assert np.allclose(restored_boxes, np.array(boxes))


def test_restored_boxes_are_clipped_unless_kept_whole() -> None:
    """
    Verifies that a box predicted partly on the padding of a letterboxed image is clipped to the original image by
    default, and kept whole without clipping, as the instance segmentation evaluated at the original resolution needs
    to paste the mask of a detection into the whole box it was predicted in.
    """
    # A 200 x 100 image letterboxed into 400 pixels is scaled twice and moved down by 100 rows of padding
    network_box = np.array(object=[[-20.0, 60.0, 120.0, 140.0]])
    clipped_box = restore_boxes_to_original_image(bounding_boxes=network_box, image_width=200, image_height=100,
                                                  image_size=400, keep_ratio=True)
    whole_box = restore_boxes_to_original_image(bounding_boxes=network_box, image_width=200, image_height=100,
                                                image_size=400, keep_ratio=True, clip=False)

    assert np.allclose(clipped_box, [[0.0, 0.0, 60.0, 20.0]])
    assert np.allclose(whole_box, [[-10.0, -20.0, 60.0, 20.0]])


@pytest.mark.parametrize(argnames="keep_ratio", argvalues=[True])
def test_polygons_are_transformed_like_the_boxes(keep_ratio: bool) -> None:
    """
    Verifies that a flat polygon [x, y, x, y, ...] goes through exactly the scaling and padding of the boxes, every
    vertex landing where the corner of a box at the same place lands, so that the masks rasterised from polygons stay
    aligned with the boxes.

    Args:
        keep_ratio (bool): Whether the preprocessing keeps the aspect ratio.
    """
    polygon = [10.0, 20.0, 300.0, 20.0, 300.0, 150.0]
    transformed_polygon = preprocess_boxes(bounding_boxes=[polygon], image_width=640, image_height=480,
                                           image_size=512, keep_ratio=keep_ratio)[0]
    transformed_box = preprocess_boxes(bounding_boxes=[[10.0, 20.0, 300.0, 150.0]], image_width=640, image_height=480,
                                       image_size=512, keep_ratio=keep_ratio)[0]

    assert len(transformed_polygon) == 6
    assert transformed_polygon[:2] == transformed_box[:2] and transformed_polygon[4:] == transformed_box[2:]
    assert transformed_polygon[2:4] == [transformed_box[2], transformed_box[1]]


@pytest.mark.parametrize(argnames="keep_ratio", argvalues=[True])
def test_class_map_is_restored_to_the_original_image(keep_ratio: bool) -> None:
    """
    Verifies that a class map at the network resolution is brought back exactly to its original image when the
    resizing factors are whole numbers: the centred padding of the kept aspect ratio is cut off, and the nearest
    resizing undoes the nearest resizing of the preprocessing.

    Args:
        keep_ratio (bool): Whether the image was padded to keep its aspect ratio, or stretched.
    """
    original_class_map = np.random.default_rng(seed=0).integers(low=0, high=5, size=(16, 8)).astype(np.int64)
    network_class_map = preprocess_image(image=original_class_map.astype(np.uint8), image_size=32,
                                         keep_ratio=keep_ratio, interpolation=cv2.INTER_NEAREST,
                                         padding_value=255).astype(np.int64)

    restored_class_map = restore_prediction_to_original_image(prediction=torch.from_numpy(network_class_map),
                                                              original_height=16, original_width=8,
                                                              keep_ratio=keep_ratio)
    assert restored_class_map.dtype == torch.int64
    assert np.array_equal(restored_class_map.numpy(), original_class_map)


def test_probabilities_are_restored_without_the_padding() -> None:
    """
    Verifies that class probabilities at the network resolution are restored to a landscape original image from the
    part of the input holding the image only: a class predicted everywhere on the image and another one on the padding
    give the first class to every original pixel, the restored probabilities keeping the class axis.
    """
    start_y, start_x, resized_height, resized_width = compute_letterbox_region(image_height=30, image_width=60,
                                                                               image_size=32)
    assert (start_y, start_x, resized_height, resized_width) == (8, 0, 16, 32)

    # Class 1 on the image part of the input, class 0 on its padding
    probabilities = torch.zeros(size=(2, 32, 32))
    probabilities[0] = 1.0
    probabilities[:, start_y:start_y + resized_height, start_x:start_x + resized_width] = torch.tensor(
        data=[0.2, 0.8]).view(2, 1, 1)

    restored_probabilities = restore_prediction_to_original_image(prediction=probabilities, original_height=30,
                                                                  original_width=60, keep_ratio=True)
    assert restored_probabilities.shape == (2, 30, 60)
    assert torch.allclose(input=restored_probabilities[1], other=torch.full(size=(30, 60), fill_value=0.8))
    assert bool((restored_probabilities.argmax(dim=0) == 1).all())
