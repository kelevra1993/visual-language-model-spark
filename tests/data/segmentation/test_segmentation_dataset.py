"""
Tests of SegmentationDataset of vision_training/data/segmentation/segmentation_dataset.py: the categories of the mask
files become class indices (binary or multi-class mapping, unmapped categories as background, unlabelled pixels
ignored), the masks are resized with the nearest interpolation and padded like their images, the class pixels are
counted, and invalid masks are refused.
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


def build_dataset(split_folder: Path, label_mode: str, label_dictionary: Dict[Any, int],
                  keep_ratio: bool) -> SegmentationDataset:
    """
    Builds the segmentation dataset of a split written by write_split.

    Args:
        split_folder (Path): The folder of the split.
        label_mode (str): "binary" or "multi_class".
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
        keep_ratio (bool): Whether the images and masks keep their aspect ratio, padded to a square.

    Returns:
        SegmentationDataset: The dataset of the split.
    """
    return SegmentationDataset(data_directory=str(split_folder / "data"), masks_directory=str(split_folder / "masks"),
                               labels_file=str(split_folder / "labels.json"), image_size=IMAGE_SIZE,
                               keep_ratio=keep_ratio, label_mode=label_mode, label_dictionary=label_dictionary)


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
                            label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True)
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
                            label_dictionary={"background": 0, "foreground": 1}, keep_ratio=False)
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
                            label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True)

    assert dataset.count_class_pixels() == [64 * 32 + 32 * 32, 64 * 32, 64 * 32]


def test_invalid_masks_are_refused(tmp_path: Path) -> None:
    """
    Verifies that a mask with colour channels, or of another size than its image, is refused with its file named.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_split(split_folder=tmp_path, mask=build_wide_mask())
    dataset = build_dataset(split_folder=tmp_path, label_mode="multi_class",
                            label_dictionary=MULTI_CLASS_LABEL_DICTIONARY, keep_ratio=True)

    # A colour mask
    cv2.imwrite(filename=str(tmp_path / "masks" / "image.png"), img=np.zeros(shape=(64, 128, 3), dtype=np.uint8))
    with pytest.raises(expected_exception=ValueError, match="single-channel"):
        dataset[0]

    # A mask of another size than its image
    cv2.imwrite(filename=str(tmp_path / "masks" / "image.png"), img=np.zeros(shape=(32, 128), dtype=np.uint8))
    with pytest.raises(expected_exception=ValueError, match="has the size"):
        dataset[0]
