"""
SegmentationDataset, the Native (OpenCV, CPU) dataset of the semantic segmentation task, reading the images and their
class-index masks of a split from its 'labels.json' manifest, with the label dictionary of the configuration, either
resized to the network input or, with 'Data.tiling', scaled and cut into tiles of the network size. Instantiated by
get_segmentation_dataloaders of data/loaders.py, and read again by the segmentation trainer for the original masks of
the evaluation at the original resolution.
"""

import json
import os
import random
import cv2
import numpy as np
import torch

from pathlib import PurePosixPath
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.label_mapping import get_class_names
from vision_training.data.preprocessing import preprocess_image

# Value of the pixels without class in the mask files (unlabelled pixels), also given to the padding of the resized
# masks; every other value of a mask is the category identifier of its pixel, 0 being the background
IGNORED_MASK_VALUE = 255

# Number of values of a uint8 mask, the size of the lookup table converting the categories to class indices
NUMBER_MASK_VALUES = 256

# Ways a split cuts its samples out of the scaled images with 'Data.tiling': a random tile at every read (training),
# a tile fixed per image so that the passes stay comparable (validation), or the whole image, segmented tile by tile
# by the trainer (test)
VALID_TILE_SAMPLINGS = ["random", "fixed", "whole"]


def build_class_lookup_table(label_mode: str, label_dictionary: Dict[Any, int]) -> np.ndarray:
    """
    Builds the table converting every value of a uint8 mask file to the class index predicted by the segmentation
    model, applied to whole masks at once by the dataset: the background (0) stays class 0, the ignored value (255)
    becomes IGNORED_LABEL_INDEX, and the categories 1 to 254 become class 1 ('binary' mode) or the class of the label
    dictionary ('multi_class' mode), an unmapped category becoming background like the unmapped objects of the
    detector.

    Args:
        label_mode (str): "binary" (every category is the foreground class 1) or "multi_class".
        label_dictionary (Dict[Any, int]): The class index of every raw category identifier.

    Returns:
        np.ndarray: The [256] int64 class index of every mask value.
    """
    # Every category is foreground in binary mode, background unless mapped in multi-class mode
    lookup_table = np.full(shape=(NUMBER_MASK_VALUES,), fill_value=1 if label_mode == "binary" else 0, dtype=np.int64)
    if label_mode == "multi_class":
        for category_identifier, class_index in label_dictionary.items():
            if isinstance(category_identifier, int) and 0 < category_identifier < IGNORED_MASK_VALUE:
                lookup_table[category_identifier] = class_index

    # The background and the ignored pixels keep their meaning in both modes
    lookup_table[0] = 0
    lookup_table[IGNORED_MASK_VALUE] = IGNORED_LABEL_INDEX
    return lookup_table


class SegmentationDataset(Dataset):
    """
    Dataset of the semantic segmentation task, read with OpenCV on the CPU (Native loader).

    A split is a folder holding its images in 'data/', their masks in 'masks/' and a 'labels.json' manifest listing
    the 'categories' and the 'samples', each sample giving its 'file_name' and its 'mask_file_name'. A mask is a
    single-channel uint8 PNG of the size of its image, whose every pixel holds the category identifier of the image
    pixel (0 for the background, 255 for the unlabelled pixels). The images are resized and padded exactly like the
    other loaders, the masks with the nearest interpolation and the ignored value on the padding, and their categories
    are converted to the class indices of the label dictionary.

    With 'Data.tiling' activated, the images and masks are scaled by 'Data.tiling.scale' instead (1 for their full
    resolution), then cut into a square tile of the network size at a random place (training) or at a place fixed per
    image (validation), or kept whole (test, segmented tile by tile by the trainer). An image smaller than a tile is
    padded on its bottom and right sides, black for the image and ignored for the mask.
    """

    def __init__(self, data_directory: str, masks_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_mode: str, label_dictionary: Dict[Any, int], tiling_configuration: Dict[str, Any],
                 tile_sampling: str, shuffle: bool = True) -> None:
        """
        Reads the manifest of the split and prepares the conversion of the mask categories to class indices.

        Args:
            data_directory (str): The folder holding the image files.
            masks_directory (str): The folder holding the mask files.
            labels_file (str): The 'labels.json' manifest of the split.
            image_size (int): The size of the square network input.
            keep_ratio (bool): Whether to keep the aspect ratio of the images and masks, padding them.
            label_mode (str): "binary" (every category is the foreground) or "multi_class".
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
            tiling_configuration (Dict[str, Any]): The 'Data.tiling' section: 'activated', 'scale' and 'overlap'.
            tile_sampling (str): How the split cuts its tiles with tiling, one of VALID_TILE_SAMPLINGS.
            shuffle (bool): Whether to shuffle the samples once at creation. Defaults to True.
        """
        self.data_directory = data_directory
        self.masks_directory = masks_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.tiling_activated = tiling_configuration["activated"]
        self.tiling_scale = tiling_configuration["scale"]
        self.tile_sampling = tile_sampling
        self.number_classes = len(set(label_dictionary.values()))
        self.class_lookup_table = build_class_lookup_table(label_mode=label_mode, label_dictionary=label_dictionary)

        with open(file=labels_file, mode="r") as labels_file_handler:
            manifest = json.load(fp=labels_file_handler)

        # Every sample of the manifest, with its image and mask files
        self.samples = [{"identifier": sample["id"], "file_name": sample["file_name"],
                         "mask_file_name": sample["mask_file_name"]} for sample in manifest["samples"]]
        if shuffle:
            random.shuffle(self.samples)

        # Every sample by identifier, to name the evaluated samples in the evaluation outputs
        self.sample_records = {sample["identifier"]: sample for sample in self.samples}

        # Name of every class, taken from the categories of the manifest and the background of the label dictionary
        self.class_names = get_class_names(categories=manifest["categories"], label_dictionary=label_dictionary)

    def __len__(self) -> int:
        """
        Gives the number of samples.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def read_mask(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        Reads the uint8 mask file of a sample, which must be a single-channel image of the size of its image.

        Args:
            sample (Dict[str, Any]): The sample, with its 'mask_file_name'.

        Returns:
            np.ndarray: The [H, W] uint8 category identifier of every pixel.

        Raises:
            ValueError: If the mask cannot be read or has several channels.
        """
        mask_path = os.path.join(self.masks_directory, sample["mask_file_name"])
        mask = cv2.imread(filename=mask_path, flags=cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError(f"The mask must be a single-channel uint8 image : file://{os.path.abspath(mask_path)}")
        return mask

    def read_class_mask(self, sample_identifier: int) -> np.ndarray:
        """
        Reads the mask of a sample at its original resolution as class indices, the ground truth of the evaluation of
        the segmenter at the original resolution of the test images.

        Args:
            sample_identifier (int): The identifier of the sample, as given in the 'sample_identifiers' of a batch.

        Returns:
            np.ndarray: The [H, W] int64 class index of every pixel of the original mask, IGNORED_LABEL_INDEX on the
                unlabelled pixels.
        """
        return self.class_lookup_table[self.read_mask(sample=self.sample_records[sample_identifier])]

    def cut_tile(self, image: np.ndarray, mask: np.ndarray, sample_identifier: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares an image and its mask for the training on tiles ('Data.tiling'): both are scaled by the tiling scale
        (the image by area averaging when shrinking, the mask with the nearest interpolation), then the test split keeps
        the whole scaled image, and the other splits pad it to at least a tile on its bottom and right sides and cut a
        tile of the network size, at a random place for the training split and at a place drawn from the identifier
        of the image for the validation split.

        Args:
            image (np.ndarray): The [H, W, 3] uint8 BGR image.
            mask (np.ndarray): The [H, W] uint8 category identifier of every pixel.
            sample_identifier (int): The identifier of the sample, seeding the place of its validation tile.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The [S, S, 3] image tile and [S, S] mask tile, or the whole scaled image and
                mask for the test split.
        """
        # Scale of the tiles, 1 keeping the full resolution
        if self.tiling_scale != 1.0:
            scaled_size = (max(1, int(round(image.shape[1] * self.tiling_scale))),
                           max(1, int(round(image.shape[0] * self.tiling_scale))))
            image = cv2.resize(src=image, dsize=scaled_size,
                               interpolation=cv2.INTER_AREA if self.tiling_scale < 1.0 else cv2.INTER_LINEAR)
            mask = cv2.resize(src=mask, dsize=scaled_size, interpolation=cv2.INTER_NEAREST)
        if self.tile_sampling == "whole":
            return image, mask

        # Bottom and right padding of the sides shorter than a tile, the padded mask pixels being ignored
        padding_height = max(0, self.image_size - image.shape[0])
        padding_width = max(0, self.image_size - image.shape[1])
        image = np.pad(array=image, pad_width=((0, padding_height), (0, padding_width), (0, 0)), mode="constant",
                       constant_values=0)
        mask = np.pad(array=mask, pad_width=((0, padding_height), (0, padding_width)), mode="constant",
                      constant_values=IGNORED_MASK_VALUE)

        # Place of the tile, random at every read, or fixed for the image so that the validation passes are comparable
        position_generator = random.Random(sample_identifier) if self.tile_sampling == "fixed" else random
        tile_top = position_generator.randint(0, image.shape[0] - self.image_size)
        tile_left = position_generator.randint(0, image.shape[1] - self.image_size)
        return (image[tile_top:tile_top + self.image_size, tile_left:tile_left + self.image_size],
                mask[tile_top:tile_top + self.image_size, tile_left:tile_left + self.image_size])

    def __getitem__(self, sample_index: int) -> Dict[str, Any]:
        """
        Reads and preprocesses one image and its mask.

        Args:
            sample_index (int): The index of the sample.

        Returns:
            Dict[str, Any]: The uint8 [3, image_size, image_size] BGR image, its int64 [image_size, image_size] class
                mask ('masks', IGNORED_LABEL_INDEX on the padding and the unlabelled pixels) and its sample identifier.
                With tiling, a tile of the network size, or the whole scaled image and mask for the test split.

        Raises:
            ValueError: If the mask does not have the size of its image.
        """
        sample = self.samples[sample_index]
        image = cv2.imread(filename=os.path.join(self.data_directory, sample["file_name"]))
        mask = self.read_mask(sample=sample)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"The mask {sample['mask_file_name']} has the size {mask.shape}, its image "
                             f"{sample['file_name']} has the size {image.shape[:2]}.")

        # Resize the image bilinearly and the mask with the nearest interpolation, with the same scale and padding, or
        # cut the tile of the split out of the scaled image and mask
        if self.tiling_activated:
            resized_image, resized_mask = self.cut_tile(image=image, mask=mask, sample_identifier=sample["identifier"])
        else:
            resized_image = preprocess_image(image=image, image_size=self.image_size, keep_ratio=self.keep_ratio)
            resized_mask = preprocess_image(image=mask, image_size=self.image_size, keep_ratio=self.keep_ratio,
                                            interpolation=cv2.INTER_NEAREST, padding_value=IGNORED_MASK_VALUE)

        # Convert the categories of every pixel to class indices
        image_tensor = torch.from_numpy(np.ascontiguousarray(resized_image)).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(self.class_lookup_table[resized_mask])
        return {"images": image_tensor, "masks": mask_tensor, "sample_identifiers": sample["identifier"]}

    def describe_sample(self, sample_identifier: int) -> Dict[str, str]:
        """
        Names an evaluated sample in the evaluation outputs, from its identifier in the batches.

        Args:
            sample_identifier (int): The identifier of the sample, as given in the 'sample_identifiers' of a batch.

        Returns:
            Dict[str, str]: The 'sample' column, the file name of the image relative to the 'data' folder.
        """
        return {"sample": self.sample_records[sample_identifier]["file_name"]}

    def get_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of an evaluated sample used in the file names of the evaluation outputs (sample grids,
        predicted masks): its file name without extension, its folders joined with underscores.

        Args:
            sample_identifier (int): The identifier of the sample, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the sample, e.g. 'street_image_000012' for 'street/image_000012.png'.
        """
        return str(PurePosixPath(self.sample_records[sample_identifier]["file_name"]).with_suffix("")).replace("/", "_")

    def count_class_pixels(self) -> List[int]:
        """
        Counts the labelled pixels of every class over the masks of the split, at their original resolution, used to
        weight the classes of the cross entropy inversely to their frequency.

        Returns:
            List[int]: The number of pixels of every class index.
        """
        class_pixel_counts = np.zeros(shape=(self.number_classes,), dtype=np.int64)
        labelled_values = self.class_lookup_table >= 0

        # Count every mask value, then gather the counts of the values of every class
        for sample in self.samples:
            value_counts = np.bincount(self.read_mask(sample=sample).ravel(), minlength=NUMBER_MASK_VALUES)
            np.add.at(class_pixel_counts, self.class_lookup_table[labelled_values], value_counts[labelled_values])
        return class_pixel_counts.tolist()
