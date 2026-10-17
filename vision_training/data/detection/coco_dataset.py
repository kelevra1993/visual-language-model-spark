"""
CocoDetectionDataset, the dataset of the object detection, instance segmentation and keypoint detection tasks reading
a COCO 'labels.json' split with OpenCV on the CPU workers, the selection of the objects of an image from its annotations
(also used by the anchor recall of the detection trainer), and the collate function batching its images and per-image
boxes, labels, instance masks and keypoints, with the masks and boxes of the objects at the original image resolution
for the evaluation of the instance segmentation at that resolution. Used by get_detection_dataloader and
get_keypoint_dataloaders of data/loaders.py.
"""

import random
import json
import os
import cv2
import numpy as np
import torch

from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset

from vision_training.data.detection.coco_masks import (build_instance_mask, compute_mask_bounding_boxes,
                                                       compute_segmentation_bounding_box)
from vision_training.data.preprocessing import preprocess_boxes, preprocess_image_and_boxes
from vision_training.utilities.os_utilities import format_path_link, print_color


def collate_function(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    Collates a list of dataset samples into a batched format dictionary.

    This function is used by the PyTorch DataLoader to aggregate individual data samples
    into mini-batches. It stacks images into a single tensor for efficient GPU processing,
    while preserving bounding boxes, labels, instance masks (instance segmentation only) and keypoints (keypoint
    detection only) as lists since they vary in quantity per image.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of dictionaries containing individual (image, boxes, labels).

    Returns:
        Dict[str, Any]: A dictionary containing the batched images, list of bounding boxes, list of labels, the
            tensor of COCO image identifiers ('sample_identifiers'), the list of instance masks when the samples
            have them, the lists of the original instance masks and boxes when the samples have them, and the list of
            keypoints when the samples have them.
    """
    images = [item["images"] for item in batch]
    bounding_boxes = [item["bounding_boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    sample_identifiers = torch.tensor(data=[item["sample_identifiers"] for item in batch], dtype=torch.int64)

    # Consolidate the image tensors into a single unified batch tensor for GPU acceleration
    # Keep bounding boxes and labels as standard lists to accommodate variable object counts per image
    batched_samples = {"images": torch.stack(tensors=images), "bounding_boxes": bounding_boxes, "labels": labels,
                       "sample_identifiers": sample_identifiers}

    # Keep the [N_i, S, S] instance masks of the instance segmentation task as a list, like the boxes, the masks and
    # boxes at the original resolution of every image, whose sizes differ, and the [N_i, K, 3] keypoints
    for key in ["instance_masks", "original_instance_masks", "original_bounding_boxes", "keypoints"]:
        if key in batch[0]:
            batched_samples[key] = [item[key] for item in batch]
    return batched_samples


def select_image_objects(annotations: List[Dict[str, Any]], label_mode: str, label_dictionary: Dict[str, Any],
                         boxes_from_segmentation: bool = False) -> Tuple[List[List[float]], List[int],
                                                                         List[Dict[str, Any]]]:
    """
    Selects the objects of one image the object detection and instance segmentation networks learn from its COCO
    annotations, shared by CocoDetectionDataset, which loads them with the image, and by the anchor recall of the
    detection trainer at launch, which measures how well the anchors fit them without loading any image. Every
    category is mapped to its class, a category mapped to the background (0) being left out, and the crowd annotations
    (iscrowd 1) are objects like the others. The object detection reads the 'bbox' of every object; the instance
    segmentation reads its box from its 'segmentation' instead (compute_segmentation_bounding_box), so that its
    annotations need no 'bbox' and its boxes always follow its masks, an object whose segmentation has no usable part
    being left out.

    Args:
        annotations (List[Dict[str, Any]]): The COCO annotations of the image.
        label_mode (str): "binary" (every category becomes class 1) or "multi_class" (label_dictionary).
        label_dictionary (Dict[str, Any]): The class of every category identifier, as an integer or a string key.
        boxes_from_segmentation (bool): Whether to read the box of every object from its 'segmentation' (instance
            segmentation) instead of its 'bbox' (object detection). Defaults to False.

    Returns:
        Tuple[List[List[float]], List[int], List[Dict[str, Any]]]: The [x1, y1, x2, y2] box of every kept object in
            original image pixels, its class and its annotation.

    Raises:
        ValueError: If a kept object has no 'segmentation' while the boxes are read from it.
    """
    bounding_boxes, labels, kept_annotations = [], [], []
    for annotation in annotations:
        raw_id = annotation['category_id']
        if label_mode == "binary":
            mapped_label = 1 if raw_id > 0 else 0
        else:
            mapped_label = label_dictionary.get(raw_id, label_dictionary.get(str(raw_id), 0))

        # Objects mapped to the background are not learnt
        if mapped_label > 0:
            # Instance segmentation: the box of the object comes from its mask, which every object needs
            if boxes_from_segmentation:
                if 'segmentation' not in annotation:
                    raise ValueError(f"The annotation {annotation.get('id')} of image {annotation['image_id']} has "
                                     f"no 'segmentation', which the instance segmentation needs for every object.")
                segmentation_box = compute_segmentation_bounding_box(segmentation=annotation['segmentation'])

                # A segmentation without any usable part gives an empty mask, which the loader drops: the object is
                # left out here too, so that the anchor recall counts the objects the network learns
                if segmentation_box[2] <= segmentation_box[0] or segmentation_box[3] <= segmentation_box[1]:
                    continue
                bounding_boxes.append(segmentation_box)

            # Object detection: the box of the annotation, [x, y, width, height]
            else:
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
                 load_original_instances: bool = False, load_keypoints: bool = False, maximum_keypoints: int = 0,
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
                Defaults to False (object detection). The crowd annotations are objects like the others in both
                tasks.
            load_original_instances (bool): Whether to also return the masks and boxes of the objects at the original
                image resolution, for the evaluation of the instance segmentation at that resolution (test split of
                'evaluation.original_resolution'), every object being then kept at the network resolution. Needs
                load_instance_masks. Defaults to False.
            load_keypoints (bool): Whether to return the keypoints of every object (keypoint detection), the objects
                without any labelled keypoint being left out. Defaults to False.
            maximum_keypoints (int): The most keypoints of a class, the number of keypoint positions of every object
                with load_keypoints. Defaults to 0.
            shuffle (bool): Whether to randomly shuffle the dataset internally. Defaults to True.

        Raises:
            ValueError: If the original instances are asked without the instance masks, or the keypoints without a
                positive maximum_keypoints.
        """
        if load_original_instances and not load_instance_masks:
            raise ValueError("The objects at the original resolution (load_original_instances) are those of the "
                             "instance segmentation, which needs load_instance_masks.")
        if load_keypoints and maximum_keypoints <= 0:
            raise ValueError(f"The keypoints (load_keypoints) need a positive maximum_keypoints, got "
                             f"{maximum_keypoints}.")
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.label_mode = label_mode
        self.label_dictionary = label_dictionary
        self.load_instance_masks = load_instance_masks
        self.load_original_instances = load_original_instances
        self.load_keypoints = load_keypoints
        self.maximum_keypoints = maximum_keypoints

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

        # Number of keypoints of every category, 0 for a category defining none, checked against every annotation
        self.keypoint_counts = {category['id']: len(category['keypoints']) if 'keypoints' in category else 0
                                for category in self.data['categories']}

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
                COCO identifier of the image ('sample_identifiers'), with load_instance_masks the [N, S, S] uint8
                instance masks ('instance_masks'), with load_original_instances the [N, H, W] uint8 masks
                ('original_instance_masks') and [N, 4] boxes ('original_bounding_boxes') at the original resolution,
                and with load_keypoints the [N, K, 3] float32 keypoints ('keypoints').

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
        bounding_boxes, labels, kept_annotations = select_image_objects(
            annotations=image_annotations, label_mode=self.label_mode, label_dictionary=self.label_dictionary,
            boxes_from_segmentation=self.load_instance_masks)

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
        if self.load_original_instances:
            sample = self.add_original_instances(sample=sample, annotations=kept_annotations,
                                                 image_height=image.shape[0], image_width=image.shape[1])
        elif self.load_instance_masks:
            sample = self.add_instance_masks(sample=sample, annotations=kept_annotations, image_height=image.shape[0],
                                             image_width=image.shape[1])
        if self.load_keypoints:
            sample = self.add_keypoints(sample=sample, annotations=kept_annotations, image_height=image.shape[0],
                                        image_width=image.shape[1])
        return sample

    def add_keypoints(self, sample: Dict[str, Any], annotations: List[Dict[str, Any]], image_height: int,
                      image_width: int) -> Dict[str, Any]:
        """
        Adds the keypoints of every object of a sample of the keypoint detection, moved to the pixels of the resized
        image like the boxes, for the heatmap and tag targets of the keypoint trainer and the object keypoint
        similarity of its evaluation. The objects without any labelled keypoint are removed with their box and label:
        they give no heatmap nor tag target, and the object keypoint similarity cannot score them.

        Args:
            sample (Dict[str, Any]): The sample built by __getitem__, whose boxes and labels follow the annotations.
            annotations (List[Dict[str, Any]]): The annotations of the kept objects, in the order of the boxes.
            image_height (int): The height of the original image.
            image_width (int): The width of the original image.

        Returns:
            Dict[str, Any]: The sample with its 'keypoints' [N, K, 3] float32 (x, y, visibility), K being
                maximum_keypoints, the coordinates being continuous pixel coordinates of the resized image (its first
                pixel centre at 0.5), the unlabelled keypoints and the padding positions all zero.

        Raises:
            ValueError: If an annotation has no 'keypoints', or a number of keypoints differing from its category.
        """
        keypoints = torch.zeros(size=(len(annotations), self.maximum_keypoints, 3), dtype=torch.float32)
        for object_index, annotation in enumerate(annotations):
            # Every object needs the keypoints its category defines
            if 'keypoints' not in annotation:
                raise ValueError(f"The annotation {annotation.get('id')} of image {annotation['image_id']} has no "
                                 f"'keypoints', which the keypoint detection needs for every object.")
            keypoint_count = self.keypoint_counts[annotation['category_id']]
            if len(annotation['keypoints']) != 3 * keypoint_count or keypoint_count > self.maximum_keypoints:
                raise ValueError(f"The annotation {annotation.get('id')} has {len(annotation['keypoints']) // 3} "
                                 f"keypoints, its category {annotation['category_id']} defining {keypoint_count} "
                                 f"(at most {self.maximum_keypoints}).")
            if keypoint_count == 0:
                continue

            # The keypoints go through the box transformation as one flat [x, y, x, y, ...] list
            keypoint_triplets = np.array(object=annotation['keypoints'], dtype=np.float32).reshape(-1, 3)
            moved_keypoints = preprocess_boxes(bounding_boxes=[keypoint_triplets[:, :2].reshape(-1).tolist()],
                                               image_width=image_width, image_height=image_height,
                                               image_size=self.image_size, keep_ratio=self.keep_ratio)[0]
            keypoints[object_index, :keypoint_count, :2] = torch.tensor(data=moved_keypoints).reshape(-1, 2)
            keypoints[object_index, :keypoint_count, 2] = torch.from_numpy(keypoint_triplets[:, 2])

        # The unlabelled keypoints carry no position, and the objects without any labelled keypoint are removed
        labelled = keypoints[..., 2] > 0
        keypoints[~labelled] = 0.0
        has_keypoints = labelled.any(dim=1)
        sample["keypoints"] = keypoints[has_keypoints]
        sample["bounding_boxes"] = sample["bounding_boxes"][has_keypoints]
        sample["labels"] = sample["labels"][has_keypoints]
        return sample

    def add_instance_masks(self, sample: Dict[str, Any], annotations: List[Dict[str, Any]], image_height: int,
                           image_width: int) -> Dict[str, Any]:
        """
        Adds the binary mask of every object of a sample at the network resolution, for the mask targets and the mask
        evaluation of the instance segmentation task, removes the objects whose mask is empty once resized (with
        their box and label), since they cannot be segmented, and replaces the box of every object by the tight box
        of its resized mask, so that the boxes and the masks of the instance segmentation always agree.

        Args:
            sample (Dict[str, Any]): The sample built by __getitem__, whose boxes and labels follow the annotations.
            annotations (List[Dict[str, Any]]): The annotations of the kept objects, in the order of the boxes.
            image_height (int): The height of the original image.
            image_width (int): The width of the original image.

        Returns:
            Dict[str, Any]: The sample with its 'instance_masks' [N, S, S] uint8, the boxes of its masks and the objects
                of empty mask removed.
        """
        instance_masks = self.build_instance_masks(annotations=annotations, image_height=image_height,
                                                   image_width=image_width, image_size=self.image_size)

        # Remove the objects whose mask vanished at the network resolution, with their box and label
        non_empty = (instance_masks.flatten(start_dim=1) > 0).any(dim=1)
        sample["instance_masks"] = instance_masks[non_empty]
        sample["labels"] = sample["labels"][non_empty]

        # Box of every object from its mask at the network resolution
        sample["bounding_boxes"] = compute_mask_bounding_boxes(instance_masks=sample["instance_masks"])
        return sample

    def add_original_instances(self, sample: Dict[str, Any], annotations: List[Dict[str, Any]], image_height: int,
                               image_width: int) -> Dict[str, Any]:
        """
        Adds the objects of a test sample at the original image resolution, for the evaluation of the instance
        segmentation at that resolution (as COCO evaluates it), along with their masks at the network resolution for
        the proposal recall and the sample and error images. Only the objects without any pixel in the original image
        are removed: an object whose mask vanishes once resized stays in the evaluation, with its empty mask and its
        scaled segmentation box at the network resolution, so that both resolutions list the same objects in the same
        order. The other objects take the tight boxes of their masks at both resolutions.

        Args:
            sample (Dict[str, Any]): The sample built by __getitem__, whose boxes (the scaled segmentation boxes) and
                labels follow the annotations.
            annotations (List[Dict[str, Any]]): The annotations of the kept objects, in the order of the boxes.
            image_height (int): The height of the original image.
            image_width (int): The width of the original image.

        Returns:
            Dict[str, Any]: The sample with its 'instance_masks' [N, S, S] uint8 and boxes at the network resolution,
                its 'original_instance_masks' [N, H, W] uint8 and 'original_bounding_boxes' [N, 4] at the original
                resolution, the objects without any pixel in the original image removed.
        """
        original_instance_masks = self.build_instance_masks(annotations=annotations, image_height=image_height,
                                                            image_width=image_width, image_size=None)
        instance_masks = self.build_instance_masks(annotations=annotations, image_height=image_height,
                                                   image_width=image_width, image_size=self.image_size)

        # Remove the objects without any pixel in the original image, at both resolutions
        has_pixels = (original_instance_masks.flatten(start_dim=1) > 0).any(dim=1)
        sample["original_instance_masks"] = original_instance_masks[has_pixels]
        sample["original_bounding_boxes"] = compute_mask_bounding_boxes(
            instance_masks=sample["original_instance_masks"])
        sample["instance_masks"], sample["labels"] = instance_masks[has_pixels], sample["labels"][has_pixels]

        # Box of every object from its mask at the network resolution, the scaled segmentation box for a vanished mask
        mask_boxes = compute_mask_bounding_boxes(instance_masks=sample["instance_masks"])
        vanished = ~(sample["instance_masks"].flatten(start_dim=1) > 0).any(dim=1)
        sample["bounding_boxes"] = torch.where(vanished[:, None], sample["bounding_boxes"][has_pixels], mask_boxes)
        return sample

    def build_instance_masks(self, annotations: List[Dict[str, Any]], image_height: int, image_width: int,
                             image_size: Optional[int]) -> torch.Tensor:
        """
        Builds the binary masks of the objects of an image, at the network resolution for the training and evaluation
        of the instance segmentation, or at the original resolution for the evaluation at that resolution, stacked in
        the order of the annotations.

        Args:
            annotations (List[Dict[str, Any]]): The annotations of the kept objects, select_image_objects having checked
                that every one has a 'segmentation'.
            image_height (int): The height of the original image.
            image_width (int): The width of the original image.
            image_size (Optional[int]): The size of the square network input, None for the original resolution.

        Returns:
            torch.Tensor: The [N, S, S], or [N, H, W] at the original resolution, uint8 masks, [0, S, S] or [0, H, W]
                for an image without objects.
        """
        masks = [build_instance_mask(segmentation=annotation['segmentation'], image_width=image_width,
                                     image_height=image_height, image_size=image_size, keep_ratio=self.keep_ratio)
                 for annotation in annotations]

        # Stack the masks, keeping the [0, S, S] or [0, H, W] layout for an image without objects
        if len(masks) == 0:
            mask_shape = (image_height, image_width) if image_size is None else (image_size, image_size)
            return torch.zeros(size=(0, *mask_shape), dtype=torch.uint8)
        return torch.from_numpy(np.stack(arrays=masks))
