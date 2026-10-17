"""
KeypointDataset, the dataset of the keypoint detection task: one sample per annotated object of a COCO 'labels.json'
split, made of its whole image resized to the network input on the CPU workers (with the letterbox of the other
tasks), its box, its class and its keypoints with their visibilities in the pixels of that resized image. The keypoint
trainer cuts the object crop out of the image on the GPU (data/keypoints/crop_transform.py), so that the scale,
rotation, flip and shift of the crop augmentation are drawn there in a single warp. Used by get_keypoint_dataloaders
of data/loaders.py.
"""

import json
import os
import random
import cv2
import numpy as np
import torch

from pathlib import PurePosixPath
from typing import Any, Dict, List
from torch.utils.data import Dataset

from vision_training.data.detection.coco_dataset import select_image_objects
from vision_training.data.preprocessing import preprocess_boxes, preprocess_image


class KeypointDataset(Dataset):
    """
    One sample per object of a COCO keypoint split: the resized image holding the object, its box, class, keypoints and
    visibilities, padded to the most keypoints of a class so that the default collate stacks them. An object whose
    category is mapped to the background is left out, as are the objects without any labelled keypoint, which give no
    heatmap loss in training and which the object keypoint similarity cannot score.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_dictionary: Dict[Any, int], maximum_keypoints: int, shuffle: bool = True) -> None:
        """
        Reads the annotations of the split and lists its objects.

        Args:
            data_directory (str): The folder of the image files.
            labels_file (str): The COCO 'labels.json' of the split, whose annotations hold 'keypoints' and whose
                categories hold the keypoint names.
            image_size (int): The side of the square network input the images are resized to.
            keep_ratio (bool): Whether to keep the aspect ratio of the images with a centred padding.
            label_dictionary (Dict[Any, int]): The class index of every raw category identifier ('multi_class' mode).
            maximum_keypoints (int): The most keypoints of a class, the number of keypoint positions of every sample.
            shuffle (bool): Whether to shuffle the objects. Defaults to True.

        Raises:
            ValueError: If an annotation has no 'keypoints', or a number of keypoints differing from its category.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.maximum_keypoints = maximum_keypoints

        with open(file=labels_file, mode="r") as file_handler:
            data = json.load(fp=file_handler)
        self.images = {image_information["id"]: image_information for image_information in data["images"]}
        keypoint_counts = {category["id"]: len(category["keypoints"]) for category in data["categories"]
                           if "keypoints" in category}

        # Group the annotations by image, as select_image_objects maps the categories of one image at a time
        image_annotations = {}
        for annotation in data["annotations"]:
            image_annotations.setdefault(annotation["image_id"], []).append(annotation)

        # One record per kept object: its image, box in original pixels, class and annotation
        self.object_records = []
        for image_identifier, annotations in image_annotations.items():
            bounding_boxes, labels, kept_annotations = select_image_objects(
                annotations=annotations, label_mode="multi_class", label_dictionary=label_dictionary)
            for bounding_box, label, annotation in zip(bounding_boxes, labels, kept_annotations):
                if "keypoints" not in annotation:
                    raise ValueError(f"The annotation {annotation['id']} of image {image_identifier} has no "
                                     f"'keypoints', which the keypoint detection needs for every object.")
                if len(annotation["keypoints"]) != 3 * keypoint_counts[annotation["category_id"]]:
                    raise ValueError(f"The annotation {annotation['id']} has {len(annotation['keypoints']) // 3} "
                                     f"keypoints, its category {annotation['category_id']} defining "
                                     f"{keypoint_counts[annotation['category_id']]}.")

                # Objects without any labelled keypoint have nothing to learn from nor to score
                if not any(annotation["keypoints"][2::3]):
                    continue
                self.object_records.append({"image_identifier": image_identifier, "bounding_box": bounding_box,
                                            "label": label, "annotation": annotation})

        # Annotation identifiers name the samples of the evaluation, indexed once the objects are shuffled
        if shuffle:
            random.shuffle(self.object_records)
        self.record_indices = {record["annotation"]["id"]: record_index
                               for record_index, record in enumerate(self.object_records)}

    def __len__(self) -> int:
        """
        Gives the number of objects of the split, the number of samples of an epoch.

        Returns:
            int: The number of kept objects.
        """
        return len(self.object_records)

    def __getitem__(self, _index: int) -> Dict[str, Any]:
        """
        Reads the image of an object, resizes it to the network input and moves the box and keypoints of the object
        with it.

        Args:
            _index (int): The index of the object.

        Returns:
            Dict[str, Any]: The uint8 BGR 'images' [3, S, S], the float32 'bounding_boxes' [4] (x1, y1, x2, y2), the
                int64 'labels', the float32 'keypoints' [K, 2] and int64 'visibilities' [K] (0 for the padding
                positions), the COCO 'image_identifiers' and the annotation identifier as 'sample_identifiers', the
                coordinates being continuous pixel coordinates of the resized image (its first pixel centre at 0.5).
        """
        record = self.object_records[_index]
        image_information = self.images[record["image_identifier"]]
        image = cv2.imread(filename=os.path.join(self.data_directory, image_information["file_name"]))
        image_height, image_width = image.shape[:2]

        # The keypoints go through the box transformation as one flat [x, y, x, y, ...] list
        keypoint_triplets = np.array(object=record["annotation"]["keypoints"], dtype=np.float32).reshape(-1, 3)
        moved_box, moved_keypoints = preprocess_boxes(
            bounding_boxes=[record["bounding_box"], keypoint_triplets[:, :2].reshape(-1).tolist()],
            image_width=image_width, image_height=image_height, image_size=self.image_size, keep_ratio=self.keep_ratio)

        # Pad the keypoints of the class to the most keypoints of a class, the padding unlabelled
        number_keypoints = keypoint_triplets.shape[0]
        keypoints = torch.zeros(size=(self.maximum_keypoints, 2), dtype=torch.float32)
        visibilities = torch.zeros(size=(self.maximum_keypoints,), dtype=torch.int64)
        keypoints[:number_keypoints] = torch.tensor(data=moved_keypoints, dtype=torch.float32).reshape(-1, 2)
        visibilities[:number_keypoints] = torch.from_numpy(keypoint_triplets[:, 2].astype(np.int64))

        # The unlabelled keypoints carry no position
        keypoints[visibilities == 0] = 0.0
        resized_image = preprocess_image(image=image, image_size=self.image_size, keep_ratio=self.keep_ratio)
        return {"images": torch.from_numpy(resized_image).permute(2, 0, 1),
                "bounding_boxes": torch.tensor(data=moved_box, dtype=torch.float32),
                "labels": torch.tensor(data=record["label"], dtype=torch.int64), "keypoints": keypoints,
                "visibilities": visibilities, "image_identifiers": record["image_identifier"],
                "sample_identifiers": record["annotation"]["id"]}

    def get_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of an evaluated object used in the file names of the evaluation outputs (sample images):
        the file name of its image without extension, its folders joined with underscores, followed by its annotation
        identifier.

        Args:
            sample_identifier (int): The annotation identifier of the object, as given in the 'sample_identifiers'.

        Returns:
            str: The name of the object, e.g. 'test_000012_object_57'.
        """
        record = self.object_records[self.record_indices[sample_identifier]]
        image_name = str(PurePosixPath(self.images[record["image_identifier"]]["file_name"]).with_suffix(""))
        return f"{image_name.replace('/', '_')}_object_{sample_identifier}"

    def list_objects(self) -> List[Dict[str, Any]]:
        """
        Lists the objects of the split with their class and labelled keypoints, without reading any image, for the
        choice of the fixed test samples drawn at every evaluation and the description of the splits.

        Returns:
            List[Dict[str, Any]]: The 'sample_identifier', 'label' and number of 'labelled_keypoints' of every object.
        """
        return [{"sample_identifier": record["annotation"]["id"], "label": record["label"],
                 "labelled_keypoints": int(sum(1 for visibility in record["annotation"]["keypoints"][2::3]
                                               if visibility > 0))} for record in self.object_records]
