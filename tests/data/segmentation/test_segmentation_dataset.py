"""
Tests of SegmentationDataset of vision_training/data/segmentation/segmentation_dataset.py: the categories of the mask
files become class indices (binary or multi-class mapping, unmapped categories as background, unlabelled pixels
ignored), the masks are resized with the nearest interpolation and padded like their images, the tiles of the training
on tiles are cut from the scaled images, the original class masks are read back, the class pixels are counted, and
invalid masks are refused.
"""

import json
import cv2
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.segmentation.segmentation_dataset import IGNORED_MASK_VALUE, SegmentationDataset

# Size of the square network input of the tests
IMAGE_SIZE = 64

# Label dictionary of the multi-class tests: categories 2 and 5 are the classes 1 and 2, category 7 is not mapped
MULTI_CLASS_LABEL_DICTIONARY = {2: 1, 5: 2, "background": 0}

# Tiling settings of the datasets resizing their images to the network input
NO_TILING = {"activated": False, "scale": 1.0, "overlap": 0}


def write_split(split_folder: Path, mask: np.ndarray) -> None:
    """
    Writes a split of one sample in the layout read by the segmentation dataset: a grey image of the size of the mask
    in 'data/', the mask in 'masks/' and the 'labels.json' manifest naming the categories 2, 5 and 7.

    Args:
        split_folder (Path): The folder of the split.
        mask (np.ndarray): The [H, W] uint8 category identifier of every pixel.
    """
    (split_folder / "data").mkdir(parents=True)
    (split_folder / "masks").mkdir(parents=True)
    cv2.imwrite(filename=str(split_folder / "data" / "image.png"),
                img=np.full(shape=mask.shape + (3,), fill_value=200, dtype=np.uint8))
    cv2.imwrite(filename=str(split_folder / "masks" / "image.png"), img=mask)
    manifest = {"categories": [{"id": 2, "name": "road"}, {"id": 5, "name": "car"}, {"id": 7, "name": "tree"}],
                "samples": [{"id": 1, "file_name": "image.png", "mask_file_name": "image.png"}]}
    (split_folder / "labels.json").write_text(data=json.dumps(obj=manifest))


def build_dataset(split_folder: Path, label_mode: str, label_dictionary: Dict[Any, int], keep_ratio: bool,
                  tiling_configuration: Dict[str, Any], tile_sampling: str) -> SegmentationDataset:
    """
    Builds the segmentation dataset of a split written by write_split.

    Args:
        split_folder (Path): The folder of the split.
        label_mode (str): "binary" or "multi_class".
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
        keep_ratio (bool): Whether the images and masks keep their aspect ratio, padded to a square.
        tiling_configuration (Dict[str, Any]): The 'Data.tiling' settings.
        tile_sampling (str): How the split cuts its tiles with tiling: "random", "fixed" or "whole".

    Returns:
        SegmentationDataset: The dataset of the split.
    """
    return SegmentationDataset(data_directory=str(split_folder / "data"), masks_directory=str(split_folder / "masks"),
                               labels_file=str(split_folder / "labels.json"), image_size=IMAGE_SIZE,
                               keep_ratio=keep_ratio, label_mode=label_mode, label_dictionary=label_dictionary,
                               tiling_configuration=tiling_configuration, tile_sampling=tile_sampling)


def build_wide_mask() -> np.ndarray:
    """
    Builds a 64 x 128 mask (twice as wide as high): background on the left quarter, category 2 on the second quarter,
    category 5 on the third, the unmapped category 7 and unlabelled pixels on the last one.

    Returns:
        np.ndarray: The [64, 128] uint8 mask.
    """
    mask = np.zeros(shape=(64, 128), dtype=np.uint8)
    mask[:, 32:64] = 2
    mask[:, 64:96] = 5
    mask[:32, 96:] = 7
    mask[32:, 96:] = IGNORED_MASK_VALUE
    return mask


def test_multi_class_mask_with_padding(tmp_path: Path) -> None:
    """
    Verifies a multi-class mask resized with its aspect ratio kept: the categories become their classes, the unmapped
    category becomes background, the unlabelled pixels and the padding rows are ignored exactly where the image is
    padded with black, and no interpolated value appears.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_split(split_folder=tmp_path, mask=build_wide_mask())
    dataset = build_dataset(split_folder=tmp_path, label_mode="multi_class",
                            label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True,
                            tiling_configuration=NO_TILING, tile_sampling="random")
    sample = dataset[0]

    # The 64 x 128 sample becomes 32 x 64 pixels centred on a 64 x 64 input, with 16 padding rows above and below
    assert sample["images"].shape == (3, IMAGE_SIZE, IMAGE_SIZE) and sample["masks"].dtype == torch.int64
    padding_rows = torch.cat(tensors=[sample["masks"][:16], sample["masks"][48:]])
    assert torch.all(padding_rows == IGNORED_LABEL_INDEX)
    assert torch.all(sample["images"][:, :16] == 0) and torch.all(sample["images"][:, 16:48] == 200)

    # Every quarter of the resized sample holds its class, only class or ignored values being present
    image_rows = sample["masks"][16:48]
    assert set(sample["masks"].unique().tolist()) == {0, 1, 2, IGNORED_LABEL_INDEX}
    assert torch.all(image_rows[:, :16] == 0) and torch.all(image_rows[:, 16:32] == 1)
    assert torch.all(image_rows[:, 32:48] == 2)
    assert torch.all(image_rows[:16, 48:] == 0) and torch.all(image_rows[16:, 48:] == IGNORED_LABEL_INDEX)
    assert dataset.class_names == {0: "background", 1: "road", 2: "car"}


def test_binary_mask_without_padding(tmp_path: Path) -> None:
    """
    Verifies a binary mask stretched to the square input: every category becomes the foreground class 1, the
    background stays 0, and only the unlabelled pixels are ignored.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_split(split_folder=tmp_path, mask=build_wide_mask())
    dataset = build_dataset(split_folder=tmp_path, label_mode="binary",
                            label_dictionary={"background": 0, "foreground": 1}, keep_ratio=False,
                            tiling_configuration=NO_TILING, tile_sampling="random")
    mask = dataset[0]["masks"]

    assert torch.all(mask[:, :16] == 0) and torch.all(mask[:, 16:48] == 1) and torch.all(mask[:32, 48:] == 1)
    assert torch.all(mask[32:, 48:] == IGNORED_LABEL_INDEX)
    assert dataset.class_names == {0: "background", 1: "foreground"}


def test_class_pixels_are_counted_at_the_original_resolution(tmp_path: Path) -> None:
    """
    Verifies the pixel counts of the balanced class weights: the labelled pixels of every class of the original mask,
    the unmapped category counting as background and the unlabelled pixels not counting.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_split(split_folder=tmp_path, mask=build_wide_mask())
    dataset = build_dataset(split_folder=tmp_path, label_mode="multi_class",
                            label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True,
                            tiling_configuration=NO_TILING, tile_sampling="random")

    assert dataset.count_class_pixels() == [64 * 32 + 32 * 32, 64 * 32, 64 * 32]


def test_invalid_masks_are_refused(tmp_path: Path) -> None:
    """
    Verifies that a mask with colour channels, or of another size than its image, is refused with its file named.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_split(split_folder=tmp_path, mask=build_wide_mask())
    dataset = build_dataset(split_folder=tmp_path, label_mode="multi_class",
                            label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True,
                            tiling_configuration=NO_TILING, tile_sampling="random")

    # A colour mask
    cv2.imwrite(filename=str(tmp_path / "masks" / "image.png"), img=np.zeros(shape=(64, 128, 3), dtype=np.uint8))
    with pytest.raises(expected_exception=ValueError, match="single-channel"):
        dataset[0]

    # A mask of another size than its image
    cv2.imwrite(filename=str(tmp_path / "masks" / "image.png"), img=np.zeros(shape=(32, 128), dtype=np.uint8))
    with pytest.raises(expected_exception=ValueError, match="has the size"):
        dataset[0]


def test_tiles_are_cut_from_the_scaled_images(tmp_path: Path) -> None:
    """
    Verifies the samples of the training on tiles: at full resolution, the training tiles are squares of the network
    size taken anywhere in the 64 x 128 image, the validation tile of an image is the same at every read, and the test
    image is kept whole with its original classes; at half resolution, the 32 x 64 image is padded on its bottom side
    to give a full tile, the padding being black and ignored.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_split(split_folder=tmp_path, mask=build_wide_mask())
    full_resolution = {"activated": True, "scale": 1.0, "overlap": 16}

    # Training tiles of the network size, and the validation tile fixed for the image
    training_dataset = build_dataset(split_folder=tmp_path, label_mode="multi_class",
                                     label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True,
                                     tiling_configuration=full_resolution, tile_sampling="random")
    assert training_dataset[0]["masks"].shape == (IMAGE_SIZE, IMAGE_SIZE)
    validation_dataset = build_dataset(split_folder=tmp_path, label_mode="multi_class",
                                       label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True,
                                       tiling_configuration=full_resolution, tile_sampling="fixed")
    assert torch.equal(input=validation_dataset[0]["masks"], other=validation_dataset[0]["masks"])

    # Whole test image with its classes at their original places
    test_dataset = build_dataset(split_folder=tmp_path, label_mode="multi_class",
                                 label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True,
                                 tiling_configuration=full_resolution, tile_sampling="whole")
    test_sample = test_dataset[0]
    assert test_sample["images"].shape == (3, 64, 128)
    assert torch.equal(input=test_sample["masks"], other=torch.from_numpy(test_dataset.read_class_mask(
        sample_identifier=1)))

    # Half resolution: a 32 x 64 image padded with 32 black and ignored rows below it
    half_resolution_dataset = build_dataset(split_folder=tmp_path, label_mode="multi_class",
                                            label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True,
                                            tiling_configuration={**full_resolution, "scale": 0.5},
                                            tile_sampling="random")
    half_resolution_sample = half_resolution_dataset[0]
    assert torch.all(half_resolution_sample["masks"][32:] == IGNORED_LABEL_INDEX)
    assert torch.all(half_resolution_sample["images"][:, 32:] == 0)
    assert torch.all(half_resolution_sample["masks"][:32, :16] == 0) and torch.all(
        half_resolution_sample["masks"][:32, 16:32] == 1)


def test_original_class_mask_is_read_back(tmp_path: Path) -> None:
    """
    Verifies the ground truth of the evaluation at the original resolution: the mask of a sample read at its original
    size, its categories converted to class indices and its unlabelled pixels ignored.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_split(split_folder=tmp_path, mask=build_wide_mask())
    dataset = build_dataset(split_folder=tmp_path, label_mode="multi_class",
                            label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True,
                            tiling_configuration=NO_TILING, tile_sampling="random")

    class_mask = dataset.read_class_mask(sample_identifier=1)
    assert class_mask.shape == (64, 128) and class_mask.dtype == np.int64
    assert np.all(class_mask[:, 32:64] == 1) and np.all(class_mask[:, 64:96] == 2)
    assert np.all(class_mask[:32, 96:] == 0) and np.all(class_mask[32:, 96:] == IGNORED_LABEL_INDEX)
