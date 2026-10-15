"""
CocoDetectionDataset, the dataset of the object detection and instance segmentation tasks reading a COCO 'labels.json'
split with OpenCV on the CPU workers, the selection of the objects of an image from its annotations (also used by the
anchor recall of the detection trainer), and the collate function batching its images and per-image boxes, labels and
instance masks. Used by get_detection_dataloader of data/loaders.py.
"""

import random
import json
import os
import cv2
import numpy as np
import torch

from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset

from vision_training.data.detection.coco_masks import build_instance_mask
from vision_training.data.preprocessing import preprocess_image_and_boxes
from vision_training.utilities.os_utilities import format_path_link, print_color


def collate_function(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    Collates a list of dataset samples into a batched format dictionary.

    This function is used by the PyTorch DataLoader to aggregate individual data samples
    into mini-batches. It stacks images into a single tensor for efficient GPU processing,
    while preserving bounding boxes, labels and instance masks (instance segmentation only) as lists since they vary in
    quantity per image.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of dictionaries containing individual (image, boxes, labels).

    Returns:
        Dict[str, Any]: A dictionary containing the batched images, list of bounding boxes, list of labels, the
            tensor of COCO image identifiers ('sample_identifiers'), and the list of instance masks when the samples
            have them.
    """
    images = [item["images"] for item in batch]
    bounding_boxes = [item["bounding_boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    sample_identifiers = torch.tensor(data=[item["sample_identifiers"] for item in batch], dtype=torch.int64)

    # Consolidate the image tensors into a single unified batch tensor for GPU acceleration
    # Keep bounding boxes and labels as standard lists to accommodate variable object counts per image
    batched_samples = {"images": torch.stack(tensors=images), "bounding_boxes": bounding_boxes, "labels": labels,
                       "sample_identifiers": sample_identifiers}

    # Keep the [N_i, S, S] instance masks of the instance segmentation task as a list, like the boxes
    if "instance_masks" in batch[0]:
        batched_samples["instance_masks"] = [item["instance_masks"] for item in batch]
    return batched_samples


def select_image_objects(annotations: List[Dict[str, Any]], label_mode: str,
                         label_dictionary: Dict[str, Any]) -> Tuple[List[List[float]], List[int],
                                                                    List[Dict[str, Any]]]:
    """
    Selects the objects of one image the object detection and instance segmentation networks learn from its COCO
    annotations, shared by CocoDetectionDataset, which loads them with the image, and by the anchor recall of the
    detection trainer at launch, which measures how well the anchors fit them without loading any image. The crowd
    annotations are left out, and every category is mapped to its class, a category mapped to the background (0) being
    left out too.

    Args:
        annotations (List[Dict[str, Any]]): The COCO annotations of the image.
        label_mode (str): "binary" (every category becomes class 1) or "multi_class" (label_dictionary).
        label_dictionary (Dict[str, Any]): The class of every category identifier, as an integer or a string key.

    Returns:
        Tuple[List[List[float]], List[int], List[Dict[str, Any]]]: The [x1, y1, x2, y2] box of every kept object in
            original image pixels, its class and its annotation.
    """
    bounding_boxes, labels, kept_annotations = [], [], []
    for annotation in annotations:
        # The crowd annotations cover several objects with one box and one mask, left out of the object detection and
        # the instance segmentation, whose objects are single ones
        if annotation.get('iscrowd', 0) == 1:
            continue
        raw_id = annotation['category_id']
        if label_mode == "binary":
            mapped_label = 1 if raw_id > 0 else 0
        else:
            mapped_label = label_dictionary.get(raw_id, label_dictionary.get(str(raw_id), 0))

        # Objects mapped to the background are not learnt
        if mapped_label > 0:
            x_coordinate, y_coordinate, width, height = annotation['bbox']
            bounding_boxes.append([x_coordinate, y_coordinate, x_coordinate + width, y_coordinate + height])
            labels.append(mapped_label)
            kept_annotations.append(annotation)
    return bounding_boxes, labels, kept_annotations


class CocoDetectionDataset(Dataset):
    """
    A PyTorch Dataset implementation for reading the raw image format.

    This class reads images directly from the standard filesystem and scales them dynamically
    on the CPU during each dataloader fetch iteration.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_mode: str, label_dictionary: Dict[str, Any], load_instance_masks: bool = False,
                 shuffle: bool = True) -> None:
        """
        Initializes the COCO detection dataset and parses the JSON file.

        Args:
            data_directory (str): The absolute path to the directory containing the physical image files.
            labels_file (str): The absolute path to the JSON file containing the annotations.
            image_size (int): The target height and width for the scaled images.
            keep_ratio (bool): Whether to maintain the aspect ratio during scaling by padding.
            label_mode (str): Mode for class assignment ("binary" or "multi_class").
            label_dictionary (Dict[str, Any]): Dictionary to map raw classes.
            load_instance_masks (bool): Whether to return the binary mask of every object (instance segmentation).
                Defaults to False (object detection). The crowd annotations are left out of both tasks.
            shuffle (bool): Whether to randomly shuffle the dataset internally. Defaults to True.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.label_mode = label_mode
        self.label_dictionary = label_dictionary
        self.load_instance_masks = load_instance_masks

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
                                               f"{format_path_link(path=labels_file)}")
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
            dict: A dictionary containing the preprocessed image tensor, bounding boxes tensor, labels tensor, the
                COCO identifier of the image ('sample_identifiers'), and with load_instance_masks the [N, S, S] uint8
                instance masks ('instance_masks').

        Raises:
            ValueError: If an annotation has no 'segmentation' while the instance masks are loaded.
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

        # Objects of the image the network learns, with their [x1, y1, x2, y2] boxes in original pixels
        bounding_boxes, labels, kept_annotations = select_image_objects(annotations=image_annotations,
                                                                        label_mode=self.label_mode,
                                                                        label_dictionary=self.label_dictionary)

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

        sample = {"images": image_tensor, "bounding_boxes": bounding_boxes_tensor,
                  "labels": torch.tensor(data=labels, dtype=torch.int64), "sample_identifiers": image_identifier}
        if self.load_instance_masks:
            sample = self.add_instance_masks(sample=sample, annotations=kept_annotations, image_height=image.shape[0],
                                             image_width=image.shape[1])
        return sample

    def add_instance_masks(self, sample: Dict[str, Any], annotations: List[Dict[str, Any]], image_height: int,
                           image_width: int) -> Dict[str, Any]:
        """
        Adds the binary mask of every object of a sample at the network resolution, for the mask targets and the mask
        evaluation of the instance segmentation task, and removes the objects whose mask is empty once resized (with
        their box and label), since they cannot be segmented.

        Args:
            sample (Dict[str, Any]): The sample built by __getitem__, whose boxes and labels follow the annotations.
            annotations (List[Dict[str, Any]]): The annotations of the kept objects, in the order of the boxes.
            image_height (int): The height of the original image.
            image_width (int): The width of the original image.

        Returns:
            Dict[str, Any]: The sample with its 'instance_masks' [N, S, S] uint8 and the objects of empty mask removed.

        Raises:
            ValueError: If an annotation has no 'segmentation'.
        """
        masks = []
        for annotation in annotations:
            if 'segmentation' not in annotation:
                raise ValueError(f"The annotation {annotation.get('id')} of image {annotation['image_id']} has no "
                                 f"'segmentation', which the instance segmentation needs for every object.")
            masks.append(build_instance_mask(segmentation=annotation['segmentation'], image_width=image_width,
                                             image_height=image_height, image_size=self.image_size,
                                             keep_ratio=self.keep_ratio))

        # Stack the masks, keeping the [0, S, S] layout for an image without objects
        if len(masks) == 0:
            instance_masks = torch.zeros(size=(0, self.image_size, self.image_size), dtype=torch.uint8)
        else:
            instance_masks = torch.from_numpy(np.stack(arrays=masks))

        # Remove the objects whose mask vanished at the network resolution, with their box and label
        non_empty = (instance_masks.flatten(start_dim=1) > 0).any(dim=1)
        sample["instance_masks"] = instance_masks[non_empty]
        sample["bounding_boxes"], sample["labels"] = sample["bounding_boxes"][non_empty], sample["labels"][non_empty]
        return sample
