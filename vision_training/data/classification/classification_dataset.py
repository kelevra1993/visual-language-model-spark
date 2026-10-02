import json
import os
import random
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Any, Dict, List

from vision_training.data.preprocessing import preprocess_image
from vision_training.utilities.os_utilities import print_color


class ClassificationDataset(Dataset):
    """
    Dataset of the classification task, read with OpenCV on the CPU (Native loader).

    A split is a folder holding its images in 'data/' and a 'labels.json' manifest listing the 'categories' and the
    'samples', each sample giving its 'file_name' and its 'category_id' (multi-class) or 'category_ids' (multi-label).
    The images are resized and padded exactly like the detection loaders, and the raw categories are converted to the
    class indices of the label dictionary; samples whose category is not mapped are left out.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_dictionary: Dict[Any, int], output_mode: str, shuffle: bool = True) -> None:
        """
        Reads the manifest of the split and keeps the samples of the mapped categories.

        Args:
            data_directory (str): The folder holding the image files.
            labels_file (str): The 'labels.json' manifest of the split.
            image_size (int): The size of the square network input.
            keep_ratio (bool): Whether to keep the aspect ratio of the images, padding them.
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier.
            output_mode (str): "multi_class" (one category per sample) or "multi_label" (any number of categories).
            shuffle (bool): Whether to shuffle the samples once at creation. Defaults to True.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.output_mode = output_mode
        self.number_classes = len(set(label_dictionary.values()))

        with open(file=labels_file, mode="r") as labels_file_handler:
            manifest = json.load(fp=labels_file_handler)

        # Keep the samples of the mapped categories, with their class indices
        self.samples = []
        for sample in manifest["samples"]:
            raw_categories = sample["category_ids"] if output_mode == "multi_label" else [sample["category_id"]]
            class_indices = [label_dictionary[category] for category in raw_categories if category in label_dictionary]
            if output_mode == "multi_label" or len(class_indices) == 1:
                self.samples.append({"identifier": sample["id"], "file_name": sample["file_name"],
                                     "class_indices": class_indices})
        if len(self.samples) < len(manifest["samples"]):
            print_color(color="yellow", output=f"{len(manifest['samples']) - len(self.samples)} samples of unmapped "
                                               f"categories left out : file://{os.path.abspath(labels_file)}")
        if shuffle:
            random.shuffle(self.samples)

        # Name of every class, taken from the categories of the manifest, joined with '+' when the label dictionary
        # merges several categories into one class, and the class index when no category names it
        category_names = {}
        for category in manifest["categories"]:
            if category["id"] in label_dictionary:
                category_names.setdefault(label_dictionary[category["id"]], []).append(category["name"])
        self.class_names = {class_index: "+".join(category_names.get(class_index, [str(class_index)]))
                            for class_index in range(self.number_classes)}

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
            Dict[str, Any]: The uint8 [3, image_size, image_size] BGR image, its label (an int64 class index, or a
                float32 multi-hot vector of the classes in multi-label mode) and its sample identifier.
        """
        sample = self.samples[sample_index]
        image = cv2.imread(filename=os.path.join(self.data_directory, sample["file_name"]))
        resized_image = preprocess_image(image=image, image_size=self.image_size, keep_ratio=self.keep_ratio)
        image_tensor = torch.from_numpy(np.ascontiguousarray(resized_image)).permute(2, 0, 1)

        # One class index per sample, or a multi-hot vector of its classes
        if self.output_mode == "multi_label":
            label = torch.zeros(size=(self.number_classes,), dtype=torch.float32)
            label[sample["class_indices"]] = 1.0
        else:
            label = torch.tensor(data=sample["class_indices"][0], dtype=torch.int64)

        return {"images": image_tensor, "labels": label, "sample_identifiers": sample["identifier"]}

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
