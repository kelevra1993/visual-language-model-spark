"""
SelfSupervisedDataset, the Native (OpenCV, CPU) dataset of the self-supervised task: it reads the images of a split
from the 'labels.json' manifest of the classification layout and keeps every one of them, labelled or not, since the
self-supervised training needs no label. The category of a labelled image is only used by the k-nearest neighbour
evaluation of the representations on the test split. Instantiated by get_self_supervised_dataloaders of
data/loaders.py for the self-supervised trainer.
"""

import json
import os
import random
import cv2
import numpy as np
import torch

from pathlib import PurePosixPath
from torch.utils.data import Dataset
from typing import Any, Dict, List

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.label_mapping import get_class_names
from vision_training.data.preprocessing import preprocess_image


class SelfSupervisedDataset(Dataset):
    """
    Dataset of the self-supervised task, read with OpenCV on the CPU (Native loader).

    A split is a folder holding its images in 'data/' and a 'labels.json' manifest listing the 'categories' and the
    'samples', each sample giving its 'file_name' and, when it is labelled, its 'category_id'. Unlike the
    classification dataset, no sample is left out: the unlabelled samples and the samples of the categories missing
    from the label dictionary are kept with the label IGNORED_LABEL_INDEX, the training only needing their images.
    The images are resized and padded exactly like the classification images.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_dictionary: Dict[Any, int], shuffle: bool = True) -> None:
        """
        Reads the manifest of the split and keeps every sample, with its class index when it has a mapped category.

        Args:
            data_directory (str): The folder holding the image files.
            labels_file (str): The 'labels.json' manifest of the split.
            image_size (int): The size of the square network input.
            keep_ratio (bool): Whether to keep the aspect ratio of the images, padding them.
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier, possibly empty for a
                dataset without labels.
            shuffle (bool): Whether to shuffle the samples once at creation. Defaults to True.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.number_classes = len(set(label_dictionary.values()))

        with open(file=labels_file, mode="r") as labels_file_handler:
            manifest = json.load(fp=labels_file_handler)

        # Keep every sample, the class index of a sample without mapped category being the ignored label
        self.samples = []
        for sample in manifest["samples"]:
            category = sample["category_id"] if "category_id" in sample else None
            class_index = label_dictionary[category] if category in label_dictionary else IGNORED_LABEL_INDEX
            self.samples.append({"identifier": sample["id"], "file_name": sample["file_name"],
                                 "class_index": class_index})
        if shuffle:
            random.shuffle(self.samples)

        # Every sample by identifier, to name the evaluated samples in the evaluation outputs
        self.sample_records = {sample["identifier"]: sample for sample in self.samples}

        # Name of every class, taken from the categories of the manifest
        self.class_names = get_class_names(categories=manifest["categories"], label_dictionary=label_dictionary)

    def __len__(self) -> int:
        """
        Gives the number of samples, labelled or not.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def __getitem__(self, sample_index: int) -> Dict[str, Any]:
        """
        Reads and preprocesses one image, with its class index for the evaluation.

        Args:
            sample_index (int): The index of the sample.

        Returns:
            Dict[str, Any]: The uint8 [3, image_size, image_size] BGR image, its int64 class index (IGNORED_LABEL_INDEX
                when it has no mapped category) and its sample identifier.
        """
        sample = self.samples[sample_index]
        image = cv2.imread(filename=os.path.join(self.data_directory, sample["file_name"]))
        resized_image = preprocess_image(image=image, image_size=self.image_size, keep_ratio=self.keep_ratio)
        image_tensor = torch.from_numpy(np.ascontiguousarray(resized_image)).permute(2, 0, 1)
        return {"images": image_tensor, "labels": torch.tensor(data=sample["class_index"], dtype=torch.int64),
                "sample_identifiers": sample["identifier"]}

    def describe_sample(self, sample_identifier: int) -> Dict[str, str]:
        """
        Names an evaluated sample in the files written at every evaluation, from its identifier in the batches.

        Args:
            sample_identifier (int): The identifier of the sample, as given in the 'sample_identifiers' of a batch.

        Returns:
            Dict[str, str]: The 'sample' column, the file name of the image relative to the 'data' folder.
        """
        return {"sample": self.sample_records[sample_identifier]["file_name"]}

    def get_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of an evaluated sample used in the file names of the sample images of the evaluation: its
        file name without extension, its folders joined with underscores.

        Args:
            sample_identifier (int): The identifier of the sample, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the sample, e.g. 'circle_image_000012' for 'circle/image_000012.png'.
        """
        return str(PurePosixPath(self.sample_records[sample_identifier]["file_name"]).with_suffix("")).replace("/", "_")

    def get_sample_class_indices(self) -> Dict[int, List[int]]:
        """
        Lists the class of every sample, from which the self-supervised trainer spreads the fixed test samples drawn at
        every evaluation over the classes, the unlabelled samples forming their own group.

        Returns:
            Dict[int, List[int]]: The class index of every sample keyed by its identifier, an empty list for the
                samples without mapped category.
        """
        return {sample["identifier"]: [] if sample["class_index"] == IGNORED_LABEL_INDEX else [sample["class_index"]]
                for sample in self.samples}
