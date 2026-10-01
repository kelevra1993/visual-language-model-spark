import random
from nvidia.dali import pipeline_def as pipeline_definition
import nvidia.dali.fn as dali_function
import nvidia.dali.types as dali_types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from typing import List
import numpy as np
import cv2
import json
import os
import torch

from pathlib import Path
from typing import Dict, Any, Tuple, Union
from torch.utils.data import DataLoader, Dataset

from utilities.data_utilities import preprocess_image_and_boxes, preprocess_boxes
from utilities.os_utilities import print_color, read_json


def collate_function(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    Collates a list of dataset samples into a batched format dictionary.

    This function is used by the PyTorch DataLoader to aggregate individual data samples
    into mini-batches. It stacks images into a single tensor for efficient GPU processing,
    while preserving bounding boxes and labels as lists since they vary in quantity per image.

    Args:
        batch (List[Dict[str, torch.Tensor]]): A list of dictionaries containing individual (image, boxes, labels).

    Returns:
        Dict[str, Any]: A dictionary containing the batched images, list of bounding boxes, and list of labels.
    """
    images = [item["images"] for item in batch]
    bounding_boxes = [item["bounding_boxes"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Consolidate the image tensors into a single unified batch tensor for GPU acceleration
    # Keep bounding boxes and labels as standard lists to accommodate variable object counts per image
    return {"images": torch.stack(tensors=images), "bounding_boxes": bounding_boxes, "labels": labels}


class NativeDataset(Dataset):
    """
    A PyTorch Dataset implementation for reading the raw image format.

    This class reads images directly from the standard filesystem and scales them dynamically
    on the CPU during each dataloader fetch iteration.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool,
                 label_mode: str, label_dictionary: Dict[str, Any], shuffle: bool = True) -> None:
        """
        Initializes the Native dataset and parses the JSON file.

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
            dict: A dictionary containing the preprocessed image tensor, bounding boxes tensor, and labels tensor.
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
                "labels": torch.tensor(data=labels, dtype=torch.int64)}


@pipeline_definition
def dali_pipeline(data_directory: str, annotations_file: str, image_size: int, keep_ratio: bool):
    """
    Defines the NVIDIA DALI processing pipeline for decoding and augmenting images directly on the GPU.
    
    This circumvents the CPU bottleneck by routing raw JPEG/PNG bytes directly to the GPU's hardware decoder
    and CUDA cores, maximizing throughput.
    
    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        annotations_file (str): The absolute path to the JSON file containing the annotations.
        image_size (int): The target spatial dimension to scale the images to.
        keep_ratio (bool): Whether to maintain the aspect ratio during scaling.
        
    Returns:
        tuple: A tuple of DALI Edge objects representing the processed images, padded bounding boxes, and padded labels.
    """
    # Load raw data and annotations
    inputs, bounding_boxes, labels, image_ids = dali_function.readers.coco(file_root=data_directory,
                                                                           annotations_file=annotations_file,
                                                                           polygon_masks=False,
                                                                           ratio=False,
                                                                           ltrb=True,
                                                                           random_shuffle=True,
                                                                           image_ids=True,
                                                                           name="Reader")

    # Decode directly on the GPU using hardware acceleration
    images = dali_function.decoders.image(inputs, device="mixed", output_type=dali_types.BGR)

    # We must retrieve the original shapes to un-normalize bounding boxes correctly in the wrapper
    shapes = dali_function.peek_image_shape(inputs)

    # Hardware accelerated resize mimicking PyTorch zero-padding NativeDataset
    if keep_ratio:
        # Resize such that the longest edge equals image_size
        images = dali_function.resize(images, resize_x=image_size, resize_y=image_size, mode="not_larger")

        # We do NOT divide by 255.0 here to maintain parity with the native datasets, keeping pixels in [0, 255]
        # By providing a crop larger than the image (since it was resized to have a max dimension of image_size) 
        # and setting crop_pos=0.5, DALI automatically applies symmetric center padding!
        images = dali_function.crop_mirror_normalize(images, dtype=dali_types.FLOAT, output_layout="CHW",
                                                     mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
                                                     crop=(image_size, image_size), crop_pos_x=0.5, crop_pos_y=0.5,
                                                     out_of_bounds_policy="pad", fill_values=0.0)
    else:
        # Direct squash resize
        images = dali_function.resize(images, resize_x=image_size, resize_y=image_size,
                                      interp_type=dali_types.INTERP_LINEAR)

        images = dali_function.crop_mirror_normalize(images, dtype=dali_types.FLOAT, output_layout="CHW",
                                                     mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    # Pad variable-length boxes and labels to uniform tensors for the batch
    bounding_boxes = dali_function.pad(bounding_boxes, axes=(0,), fill_value=-1.0)
    labels = dali_function.pad(labels, axes=(0,), fill_value=-1)

    return images, bounding_boxes, labels, shapes, image_ids


class DALIDataloaderWrapper:
    """
    Wraps the NVIDIA DALIGenericIterator to perfectly match the expected PyTorch dataloader output format.
    
    Since DALI pads bounding boxes and labels to uniform tensors to process them in a single batch,
    this wrapper dynamically unpads them back into lists of independent tensors, allowing seamless integration
    with the existing model architecture and loss functions that expect variable-sized lists.
    """

    def __init__(self, dali_iterator: DALIGenericIterator, image_size: int, keep_ratio: bool,
                 label_mode: str, label_dictionary: Dict[str, Any]):
        """
        Initializes the DALI wrapper.
        
        Args:
            dali_iterator (DALIGenericIterator): The underlying DALI hardware-accelerated iterator.
            image_size (int): The target spatial dimension images were scaled to.
            keep_ratio (bool): Whether images were padded to maintain aspect ratio.
            label_mode (str): Mode for class assignment ("binary" or "multi_class").
            label_dictionary (Dict[str, Any]): Dictionary to map raw classes.
        """
        self.dali_iterator = dali_iterator
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.label_mode = label_mode
        self.label_dictionary = label_dictionary

    def __iter__(self):
        """
        Yields the next batch from the DALI pipeline, unpacking the padded tensors into standard lists.
        
        Yields:
            Dict[str, Any]: A dictionary containing the batched images, and unpadded lists for bounding boxes and labels.
        """
        for batch in self.dali_iterator:
            # DALI returns a list of dictionaries (one for each pipeline instance)
            data = batch[0]
            images = data["images"]
            bounding_boxes_padded = data["bounding_boxes"]
            labels_padded = data["labels"]
            shapes = data["shapes"]
            image_ids = data["image_ids"].squeeze(-1)  # Squeeze 2D [batch_size, 1] to 1D [batch_size] if necessary

            bounding_boxes_list = []
            labels_list = []

            # Unpack the padded tensors back into variable-length lists per image
            for batch_index in range(images.shape[0]):
                # Identify the valid items by filtering out the padding fill value (-1)
                valid_mask = (labels_padded[batch_index] != -1).squeeze(-1) if labels_padded.dim() == 3 else (
                        labels_padded[batch_index] != -1)

                valid_boxes = bounding_boxes_padded[batch_index][valid_mask].clone()
                valid_labels = labels_padded[batch_index][valid_mask]

                # Apply dynamic label mapping and filter out backgrounds
                if self.label_mode == "binary":
                    valid_labels = torch.where(valid_labels > 0, 1, 0)
                else:
                    valid_labels_cpu = valid_labels.cpu().numpy()
                    # Iterate through each raw label and retrieve the updated mapped class index from the dictionary.
                    # To defend against JSON parsers loading integer keys as strings, we perform a dual lookup:
                    # first attempting an integer cast, then a string cast, before finally defaulting to the background class 0.
                    mapped_labels = [self.label_dictionary.get(int(l), self.label_dictionary.get(str(l), 0)) for l in valid_labels_cpu]
                    valid_labels = torch.tensor(mapped_labels, dtype=torch.int64, device=valid_labels.device)

                # Filter out background objects entirely
                non_background_mask = valid_labels > 0
                valid_labels = valid_labels[non_background_mask]
                valid_boxes = valid_boxes[non_background_mask]

                if valid_boxes.size(0) == 0:
                    valid_boxes = torch.zeros((0, 4), dtype=torch.float32, device=valid_boxes.device)

                labels_list.append(valid_labels)

                # Transform bounding boxes using the centralized math in preprocess_boxes
                original_height = shapes[batch_index][0].item()
                original_width = shapes[batch_index][1].item()

                # Convert the absolute original coordinate PyTorch tensor to a Python list for preprocess_boxes
                processed_boxes_list = preprocess_boxes(bounding_boxes=valid_boxes.tolist(),
                                                        image_width=original_width,
                                                        image_height=original_height,
                                                        image_size=self.image_size,
                                                        keep_ratio=self.keep_ratio)

                # Convert back to a PyTorch tensor
                valid_boxes = torch.tensor(data=processed_boxes_list, dtype=torch.float32, device=valid_boxes.device)
                if valid_boxes.numel() == 0:
                    valid_boxes = valid_boxes.reshape(0, 4)

                bounding_boxes_list.append(valid_boxes)

            yield {"images": images, "bounding_boxes": bounding_boxes_list, "labels": labels_list,
                   "image_ids": image_ids}

    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.
        
        Returns:
            int: The total number of batches.
        """
        return len(self.dali_iterator)


def get_dataloader(split_name: str, split_data_folder: str, model_configuration: Dict[str, Any], batch_size: int,
                   number_of_workers: int) -> Union[DataLoader, DALIDataloaderWrapper]:
    """
    Instantiates the DataLoader feeding one split (Train, Validation or Test) to the Trainer.
    The GPU-accelerated DALI pipeline is used when activated, otherwise the Native PyTorch dataset decodes the
    images with OpenCV and resizes them on the CPU. Both return the same batch dictionary format.

    Args:
        split_name (str): The name of the split (e.g., 'Train', 'Validation', 'Test').
        split_data_folder (str): The root folder for this split, containing a 'data/' subfolder and a 'labels.json'.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.
        number_of_workers (int): The number of parallel workers (DALI threads or DataLoader subprocesses).

    Returns:
        Union[DataLoader, DALIDataloaderWrapper]: The configured loader for the requested split.
    """
    data_configuration = model_configuration["Data"]
    use_dali = data_configuration["DALI"]["activated"]

    image_settings = data_configuration["image_settings"]
    image_size = image_settings["size"]
    keep_ratio = image_settings["keep_ratio"]

    label_mode = data_configuration["label_mode"]
    label_dictionary = data_configuration["label_dictionary"]

    # Derive the physical image directory and annotation file from the split folder convention
    data_directory = os.path.join(split_data_folder, "data")
    labels_file = os.path.join(split_data_folder, "labels.json")

    # If DALI is activated, initialize the GPU-accelerated pipeline
    if use_dali:
        pipeline = dali_pipeline(data_directory=data_directory, annotations_file=labels_file, image_size=image_size,
                                 keep_ratio=keep_ratio, batch_size=batch_size, num_threads=number_of_workers,
                                 device_id=0)
        pipeline.build()

        dali_iterator = DALIGenericIterator(pipelines=[pipeline],
                                            output_map=["images", "bounding_boxes", "labels", "shapes", "image_ids"],
                                            reader_name="Reader", auto_reset=True)

        # Wrap the DALI iterator so it behaves exactly like our standard PyTorch DataLoader outputs
        return DALIDataloaderWrapper(dali_iterator=dali_iterator, image_size=image_size, keep_ratio=keep_ratio,
                                     label_mode=label_mode, label_dictionary=label_dictionary)

    # Otherwise fall back to the Native PyTorch dataset, decoding and resizing the images on the CPU
    dataset = NativeDataset(data_directory=data_directory, labels_file=labels_file, image_size=image_size,
                            keep_ratio=keep_ratio, label_mode=label_mode, label_dictionary=label_dictionary)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=number_of_workers,
                      collate_fn=collate_function)


def count_split_images_and_annotations(split_data_folder: str | Path) -> Tuple[int, int]:
    """
    Counts the images and annotations of a dataset split from its COCO 'labels.json', printed by the Trainer at
    startup. The file is read directly because the DALI loader rounds its length up to a whole number of batches,
    so neither loader length gives the exact number of images.

    Args:
        split_data_folder (str | Path): The root folder of the split, containing a 'labels.json'.

    Returns:
        Tuple[int, int]: The number of images, and the number of raw annotations (before label mapping, 0 for an
            unlabelled split without an 'annotations' list).
    """
    labels = read_json(path=os.path.join(str(split_data_folder), "labels.json"))
    number_annotations = len(labels["annotations"]) if "annotations" in labels else 0
    return len(labels["images"]), number_annotations


def get_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any], batch_size: int,
                    number_of_workers: int = 4) -> Tuple[Union[DataLoader, DALIDataloaderWrapper],
                                                         Union[DataLoader, DALIDataloaderWrapper],
                                                         Union[DataLoader, DALIDataloaderWrapper]]:
    """
    Initializes the training, validation and testing loaders consumed by the Trainer.
    This function is the entry point of the data loading subsystem: it resolves each split folder from the
    experiment configuration and builds the matching loader (DALI or Native) for it.

    Args:
        experiment_configuration (Dict[str, Any]): The overall experiment configuration including data paths.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.
        number_of_workers (int): The number of parallel workers to allocate for data loading. Defaults to 4.

    Returns:
        Tuple[Union[DataLoader, DALIDataloaderWrapper], ...]: The loaders for the training, validation and
            testing splits respectively.
    """
    # Map each split name to its corresponding root folder containing images and labels.json
    split_folder_mapping = {"Train": str(experiment_configuration["train_data_folder"]),
                            "Validation": str(experiment_configuration["validation_data_folder"]),
                            "Test": str(experiment_configuration["test_data_folder"])}

    # Build the training, validation and test loaders with the same data loading settings
    train_loader = get_dataloader(split_name="Train", split_data_folder=split_folder_mapping["Train"],
                                  model_configuration=model_configuration, batch_size=batch_size,
                                  number_of_workers=number_of_workers)
    validation_loader = get_dataloader(split_name="Validation", split_data_folder=split_folder_mapping["Validation"],
                                       model_configuration=model_configuration, batch_size=batch_size,
                                       number_of_workers=number_of_workers)
    test_loader = get_dataloader(split_name="Test", split_data_folder=split_folder_mapping["Test"],
                                 model_configuration=model_configuration, batch_size=batch_size,
                                 number_of_workers=number_of_workers)

    return train_loader, validation_loader, test_loader
