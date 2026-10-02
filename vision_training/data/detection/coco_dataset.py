import random
import json
import os
import cv2
import numpy as np
import torch

from typing import Any, Dict, List
from torch.utils.data import Dataset

from vision_training.data.preprocessing import preprocess_image_and_boxes
from vision_training.utilities.os_utilities import print_color


def collate_function(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    Collates a list of dataset samples into a batched format dictionary.

    This function is used by the PyTorch DataLoader to aggregate individual data samples
    into mini-batches. It stacks images into a single tensor for efficient GPU processing,
    while preserving bounding boxes and labels as lists since they vary in quantity per image.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of dictionaries containing individual (image, boxes, labels).

    Returns:
        Dict[str, Any]: A dictionary containing the batched images, list of bounding boxes, list of labels, and the
            tensor of COCO image identifiers ('sample_identifiers').
    """
    images = [item["images"] for item in batch]
    bounding_boxes = [item["bounding_boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    sample_identifiers = torch.tensor(data=[item["sample_identifiers"] for item in batch], dtype=torch.int64)

    # Consolidate the image tensors into a single unified batch tensor for GPU acceleration
    # Keep bounding boxes and labels as standard lists to accommodate variable object counts per image
    return {"images": torch.stack(tensors=images), "bounding_boxes": bounding_boxes, "labels": labels,
            "sample_identifiers": sample_identifiers}


class CocoDetectionDataset(Dataset):
    """
    A PyTorch Dataset implementation for reading the raw image format.

    This class reads images directly from the standard filesystem and scales them dynamically
    on the CPU during each dataloader fetch iteration.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_mode: str, label_dictionary: Dict[str, Any], shuffle: bool = True) -> None:
        """
        Initializes the COCO detection dataset and parses the JSON file.

        Args:
            data_directory (str): The absolute path to the directory containing the physical image files.
            labels_file (str): The absolute path to the JSON file containing the annotations.
            image_size (int): The target height and width for the scaled images.
            keep_ratio (bool): Whether to maintain the aspect ratio during scaling by padding.
            label_mode (str): Mode for class assignment ("binary" or "multi_class").
            label_dictionary (Dict[str, Any]): Dictionary to map raw classes.
            shuffle (bool): Whether to randomly shuffle the dataset internally. Defaults to True.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.label_mode = label_mode
        self.label_dictionary = label_dictionary

        # Parse the JSON annotation file into memory for rapid preprocessing
        with open(file=labels_file, mode='r') as file_handler:
            self.data = json.load(fp=file_handler)

        self.images = self.data['images']
        if shuffle:
            random.shuffle(self.images)
        # Unlabelled splits (e.g. the COCO 2017 test set) have no 'annotations' key, their images have no boxes
        if 'annotations' in self.data:
            self.annotations = self.data['annotations']
        else:
            print_color(color="yellow", output=f"No annotations found, every image is treated as unlabelled : "
                                               f"file://{os.path.abspath(labels_file)}")
            self.annotations = []

        # Map image identifiers to their respective bounding box and label collections for rapid retrieval
        self.image_to_annotations = {}
        for annotation in self.annotations:
            image_identifier = annotation['image_id']
            if image_identifier not in self.image_to_annotations:
                self.image_to_annotations[image_identifier] = []
            self.image_to_annotations[image_identifier].append(annotation)

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total count of available images.
        """
        return len(self.images)

    def __getitem__(self, _index: int) -> dict:
        """
        Retrieves, reads, and dynamically preprocesses a single image and its annotations.

        Args:
            _index (int): The index of the image metadata to fetch.

        Returns:
            dict: A dictionary containing the preprocessed image tensor, bounding boxes tensor, labels tensor, and
                the COCO identifier of the image ('sample_identifiers').
        """
        image_information = self.images[_index]
        image_identifier = image_information['id']
        image_path = os.path.join(self.data_directory, image_information['file_name'])

        # Load the physical image directly from the filesystem into a dense matrix format
        image = cv2.imread(filename=image_path)

        if image_identifier in self.image_to_annotations:
            image_annotations = self.image_to_annotations[image_identifier]
        else:
            image_annotations = []

        bounding_boxes = []
        labels = []
        for annotation in image_annotations:
            raw_id = annotation['category_id']
            if self.label_mode == "binary":
                mapped_label = 1 if raw_id > 0 else 0
            else:
                mapped_label = self.label_dictionary.get(raw_id, self.label_dictionary.get(str(raw_id), 0))

            if mapped_label > 0:
                x_coordinate, y_coordinate, width, height = annotation['bbox']
                bounding_boxes.append([x_coordinate, y_coordinate, x_coordinate + width, y_coordinate + height])
                labels.append(mapped_label)

        if len(bounding_boxes) == 0:
            bounding_boxes_numpy = np.zeros(shape=(0, 4), dtype=np.float32)
        else:
            bounding_boxes_numpy = np.array(object=bounding_boxes, dtype=np.float32)

        # Dynamically scale the image and calculate the corresponding adjustments for the bounding boxes
        resized_image, resized_bounding_boxes = preprocess_image_and_boxes(image=image,
                                                                           bounding_boxes=bounding_boxes_numpy,
                                                                           image_size=self.image_size,
                                                                           keep_ratio=self.keep_ratio)
        # Convert the resized array into a PyTorch tensor without dtype/device casting
        # Note: The standard division by 255.0 has been intentionally omitted to preserve raw pixel scale
        # The consumer (trainer) handles the cast per-batch for maximum data pipeline throughput
        image_tensor = torch.from_numpy(np.array(object=resized_image)).permute(2, 0, 1)

        # Keep the [N, 4] layout even for images without boxes, since an empty list would become a [0] tensor
        bounding_boxes_tensor = torch.tensor(data=resized_bounding_boxes, dtype=torch.float32).reshape(-1, 4)

        return {"images": image_tensor, "bounding_boxes": bounding_boxes_tensor,
                "labels": torch.tensor(data=labels, dtype=torch.int64), "sample_identifiers": image_identifier}
