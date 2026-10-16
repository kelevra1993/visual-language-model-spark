"""
ClassificationDataset, the Native (OpenCV, CPU) dataset of the classification task, reading the images and the
multi-class or multi-label categories of a split from its 'labels.json' manifest, with the label dictionary of the
configuration. It also reads the splits of the self-supervised task, which keeps the images without mapped category,
their labels only serving its k-nearest neighbour evaluation. Instantiated by get_classification_dataloaders of
data/loaders.py.
"""

import json
import os
import random
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from pathlib import PurePosixPath
from typing import Any, Dict, List

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.label_mapping import get_class_names
from vision_training.data.preprocessing import preprocess_image
from vision_training.utilities.os_utilities import format_path_link, print_color


class ClassificationDataset(Dataset):
    """
    Dataset of the classification task, read with OpenCV on the CPU (Native loader).

    A split is a folder holding its images in 'data/' and a 'labels.json' manifest listing the 'categories' and the
    'samples', each sample giving its 'file_name' and its 'category_id' (multi-class) or 'category_ids' (multi-label).
    The images are resized and padded exactly like the detection loaders, and the raw categories are converted to the
    class indices of the label dictionary; samples whose category is not mapped are left out, unless the unlabelled
    samples are kept (self-supervised task), the label of a multi-class sample without a single mapped category (none,
    or several in the 'category_ids' of a multi-label manifest) being then IGNORED_LABEL_INDEX.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_dictionary: Dict[Any, int], output_mode: str, shuffle: bool = True,
                 keep_unlabelled_samples: bool = False) -> None:
        """
        Reads the manifest of the split and keeps the samples of the mapped categories, or every sample when the
        unlabelled samples are kept.

        Args:
            data_directory (str): The folder holding the image files.
            labels_file (str): The 'labels.json' manifest of the split.
            image_size (int): The size of the square network input.
            keep_ratio (bool): Whether to keep the aspect ratio of the images, padding them.
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
            output_mode (str): "multi_class" (one category per sample) or "multi_label" (any number of categories).
            shuffle (bool): Whether to shuffle the samples once at creation. Defaults to True.
            keep_unlabelled_samples (bool): Whether to keep the samples without a single mapped category (no category,
                a category missing from the label dictionary, or several mapped categories in the 'category_ids' of a
                multi-label manifest) without class, for the self-supervised task that trains on every image and
                evaluates on single classes. Defaults to False.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.output_mode = output_mode
        self.number_classes = len(set(label_dictionary.values()))

        with open(file=labels_file, mode="r") as labels_file_handler:
            manifest = json.load(fp=labels_file_handler)

        # Keep the samples of the mapped categories with their class indices, and the others without class when the
        # unlabelled samples are kept, which also reads the categories of a multi-label manifest
        self.samples = []
        number_unlabelled_samples = 0
        for sample in manifest["samples"]:
            if output_mode == "multi_label" or (keep_unlabelled_samples and "category_ids" in sample):
                raw_categories = sample["category_ids"]
            elif "category_id" in sample or not keep_unlabelled_samples:
                raw_categories = [sample["category_id"]]
            else:
                raw_categories = []
            class_indices = [label_dictionary[category] for category in raw_categories if category in label_dictionary]
            if output_mode == "multi_class" and len(class_indices) != 1:
                number_unlabelled_samples += 1
                if not keep_unlabelled_samples:
                    continue
                class_indices = []
            self.samples.append({"identifier": sample["id"], "file_name": sample["file_name"],
                                 "class_indices": class_indices})

        # Report the samples without mapped category, left out or kept without class
        if number_unlabelled_samples and keep_unlabelled_samples:
            print_color(color="yellow", output=f"{number_unlabelled_samples} samples without a single mapped category "
                                               f"kept without class : {format_path_link(path=labels_file)}")
        elif number_unlabelled_samples:
            print_color(color="yellow", output=f"{number_unlabelled_samples} samples of unmapped categories left out "
                                               f": {format_path_link(path=labels_file)}")
        if shuffle:
            random.shuffle(x=self.samples)

        # Every kept sample by identifier, to name the evaluated samples in the evaluation outputs
        self.sample_records = {sample["identifier"]: sample for sample in self.samples}

        # Name of every class, taken from the categories of the manifest, which are kept for the trainers naming the
        # classes from the categories of every split
        self.categories = manifest["categories"]
        self.class_names = get_class_names(categories=self.categories, label_dictionary=label_dictionary)

    def __len__(self) -> int:
        """
        Gives the number of kept samples.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def __getitem__(self, sample_index: int) -> Dict[str, Any]:
        """
        Reads and preprocesses one image, with its target.

        Args:
            sample_index (int): The index of the sample.

        Returns:
            Dict[str, Any]: The uint8 [3, image_size, image_size] BGR image, its label (an int64 class index,
                IGNORED_LABEL_INDEX for a kept sample without mapped category, or a float32 multi-hot vector of the
                classes in multi-label mode) and its sample identifier.
        """
        sample = self.samples[sample_index]
        image = cv2.imread(filename=os.path.join(self.data_directory, sample["file_name"]))
        resized_image = preprocess_image(image=image, image_size=self.image_size, keep_ratio=self.keep_ratio)
        image_tensor = torch.from_numpy(np.ascontiguousarray(a=resized_image)).permute(dims=(2, 0, 1))

        # One class index per sample (the ignored label without class), or a multi-hot vector of its classes
        if self.output_mode == "multi_label":
            label = torch.zeros(size=(self.number_classes,), dtype=torch.float32)
            label[sample["class_indices"]] = 1.0
        else:
            class_index = sample["class_indices"][0] if sample["class_indices"] else IGNORED_LABEL_INDEX
            label = torch.tensor(data=class_index, dtype=torch.int64)

        return {"images": image_tensor, "labels": label, "sample_identifiers": sample["identifier"]}

    def describe_sample(self, sample_identifier: int) -> Dict[str, str]:
        """
        Names an evaluated sample in the predictions file written at every evaluation, from its identifier in the
        batches.

        Args:
            sample_identifier (int): The identifier of the sample, as given in the 'sample_identifiers' of a batch.

        Returns:
            Dict[str, str]: The 'sample' column, the file name of the image relative to the 'data' folder.
        """
        return {"sample": self.sample_records[sample_identifier]["file_name"]}

    def get_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of an evaluated sample used in the file names of the error images of the evaluation: its
        file name without extension, its folders joined with underscores.

        Args:
            sample_identifier (int): The identifier of the sample, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the sample, e.g. 'circle_image_000012' for 'circle/image_000012.png'.
        """
        return str(PurePosixPath(self.sample_records[sample_identifier]["file_name"]).with_suffix("")).replace("/", "_")

    def get_sample_class_indices(self) -> Dict[int, List[int]]:
        """
        Lists the classes of every sample, from which the classification and self-supervised trainers spread the
        fixed test samples drawn at every evaluation over the classes, the samples without class forming their own
        group.

        Returns:
            Dict[int, List[int]]: The class indices of every sample keyed by its identifier, one per sample in
                multi-class mode (none for a kept sample without mapped category), any number in multi-label mode.
        """
        return {sample["identifier"]: list(sample["class_indices"]) for sample in self.samples}

    def get_class_counts(self) -> List[int]:
        """
        Counts the samples of every class, used to weight the classes of the loss inversely to their frequency.

        Returns:
            List[int]: The number of samples of every class index.
        """
        class_counts = [0] * self.number_classes
        for sample in self.samples:
            for class_index in sample["class_indices"]:
                class_counts[class_index] += 1
        return class_counts
