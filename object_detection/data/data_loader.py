import os
import glob
import json
import torch
import cv2
import numpy as np
import tensorflow

from typing import Dict, Any, Tuple, List
from torch.utils.data import Dataset, DataLoader, IterableDataset
from utilities.data_utilities import preprocess_image_and_boxes

# Hide GPU from TensorFlow so it does not reserve all memory, leaving none for PyTorch
tensorflow.config.set_visible_devices(devices=[], device_type='GPU')


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


def parse_single_example(serialized_data: tensorflow.Tensor) -> tuple[
    tensorflow.Tensor, tensorflow.Tensor, tensorflow.Tensor]:
    """
    Parses a single serialized tensorflow.train.Example protobuf into distinct TensorFlow tensors.
    
    This function decodes the raw image bytes into pixel values and reconstructs the sparse
    bounding boxes and labels arrays for downstream PyTorch conversion.
    
    Args:
        serialized_data (tensorflow.Tensor): The raw serialized protocol buffer string from the TFRecord.
        
    Returns:
        tuple[tensorflow.Tensor, tensorflow.Tensor, tensorflow.Tensor]: A tuple containing the decoded image tensor, 
                                                                        bounding boxes tensor, and labels tensor.
    """
    features = tensorflow.io.parse_single_example(serialized=serialized_data, features={
        "images": tensorflow.io.FixedLenFeature(shape=[], dtype=tensorflow.string),
        "bounding_boxes": tensorflow.io.FixedLenFeature(shape=[], dtype=tensorflow.string),
        "labels": tensorflow.io.FixedLenFeature(shape=[], dtype=tensorflow.string)})

    image = tensorflow.io.decode_png(contents=features["images"], channels=3)
    bounding_boxes = tensorflow.io.decode_raw(input_bytes=features["bounding_boxes"], out_type=tensorflow.float32)
    labels = tensorflow.io.decode_raw(input_bytes=features["labels"], out_type=tensorflow.int64)

    # Reshape bounding boxes back to expected multi-dimensional format
    bounding_boxes = tensorflow.reshape(tensor=bounding_boxes, shape=[-1, 4])

    return image, bounding_boxes, labels


class DummyDataset(Dataset):
    def __init__(self, image_size: int = 1024):
        self.image_size = image_size

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return {
            "images": torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32),
            "bounding_boxes": torch.zeros((1, 4), dtype=torch.float32),
            "labels": torch.zeros((1,), dtype=torch.int64)
        }


class NativeCocoDataset(Dataset):
    """
    A PyTorch Dataset implementation for reading the raw COCO image format.

    This class reads images directly from the standard filesystem and scales them dynamically 
    on the CPU during each dataloader fetch iteration.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int = 1024, keep_ratio: bool = True) -> None:
        """
        Initializes the Native dataset and parses the monolithic COCO JSON file.

        Args:
            data_directory (str): The absolute path to the directory containing the physical image files.
            labels_file (str): The absolute path to the JSON file containing the COCO annotations.
            image_size (int): The target height and width for the scaled images.
            keep_ratio (bool): Whether to maintain the aspect ratio during scaling by padding.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio

        with open(file=labels_file, mode='r') as file_handler:
            self.coco_data = json.load(fp=file_handler)

        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

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

        image = cv2.imread(filename=image_path)

        image_annotations = self.image_to_annotations.get(image_identifier, [])

        bounding_boxes = []
        labels = []
        for annotation in image_annotations:
            x_coordinate, y_coordinate, width, height = annotation['bbox']
            bounding_boxes.append([x_coordinate, y_coordinate, x_coordinate + width, y_coordinate + height])
            labels.append(annotation['category_id'])

        bounding_boxes_numpy = np.array(object=bounding_boxes, dtype=np.float32)

        # Dynamically scale the image and calculate the corresponding adjustments for the bounding boxes
        resized_image, resized_bounding_boxes = preprocess_image_and_boxes(image=image,
                                                                           bounding_boxes=bounding_boxes_numpy,
                                                                           image_size=self.image_size,
                                                                           keep_ratio=self.keep_ratio)
        # Convert the resized array into a PyTorch float tensor
        # Note: The standard division by 255.0 has been intentionally omitted to preserve raw pixel scale
        image_tensor = torch.from_numpy(np.array(object=resized_image)).permute(2, 0, 1).float()

        return {"images": image_tensor,
                "bounding_boxes": torch.tensor(data=resized_bounding_boxes, dtype=torch.float32),
                "labels": torch.tensor(data=labels, dtype=torch.int64)}


class TFRecordCocoDataset(IterableDataset):
    """
    A PyTorch IterableDataset implementation for reading pre-processed TFRecord archives using TensorFlow.
    
    It streams records directly from the disk using tensorflow.data for maximum efficiency.
    """

    def __init__(self, tensorflow_record_path: str) -> None:
        """
        Initializes the TFRecord iterable dataset.
        
        Args:
            tensorflow_record_path (str): The absolute path to the TFRecord archive file.
        """
        self.tensorflow_record_path = tensorflow_record_path

    def __iter__(self):
        """
        Returns an iterator over the dataset using optimized tensorflow.data pipeline.
        """
        worker_info = torch.utils.data.get_worker_info()
        dataset = tensorflow.data.TFRecordDataset(filenames=[self.tensorflow_record_path], buffer_size=262144)

        # Partition the dataset appropriately if multiple workers are deployed to prevent data duplication
        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        dataset = dataset.map(map_func=parse_single_example, num_parallel_calls=tensorflow.data.AUTOTUNE)

        for image, bounding_boxes, labels in dataset:
            # Convert TensorFlow Tensors into native NumPy arrays to bridge the gap with PyTorch
            image_numpy = image.numpy()
            boxes_numpy = bounding_boxes.numpy()
            labels_numpy = labels.numpy()

            # Reorder channels from height/width/channel structure to channel/height/width for PyTorch
            image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).contiguous()

            yield {"images": image_tensor, "bounding_boxes": torch.tensor(data=boxes_numpy, dtype=torch.float32),
                   "labels": torch.tensor(data=labels_numpy, dtype=torch.int64)}


class TFRecordShardedCocoDataset(IterableDataset):
    """
    A PyTorch IterableDataset implementation for reading from multiple sharded TFRecord files.
    """

    def __init__(self, directory_pattern: str) -> None:
        """
        Initializes the sharded TFRecord iterable dataset.
        
        Args:
            directory_pattern (str): The wildcard pattern used to locate all TFRecord shards.
        """
        self.tensorflow_record_files = sorted(glob.glob(pathname=directory_pattern))

    def __iter__(self):
        """
        Returns an iterator over the dataset using optimized tensorflow.data pipeline.
        """
        worker_info = torch.utils.data.get_worker_info()
        dataset = tensorflow.data.TFRecordDataset(filenames=self.tensorflow_record_files, buffer_size=262144)

        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        dataset = dataset.map(map_func=parse_single_example, num_parallel_calls=tensorflow.data.AUTOTUNE)

        for image, bounding_boxes, labels in dataset:
            image_numpy = image.numpy()
            boxes_numpy = bounding_boxes.numpy()
            labels_numpy = labels.numpy()

            image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).contiguous()

            yield {"images": image_tensor, "bounding_boxes": torch.tensor(data=boxes_numpy, dtype=torch.float32),
                   "labels": torch.tensor(data=labels_numpy, dtype=torch.int64)}


def get_dataloaders(preprocessed_directory: str, experiment_configuration: Dict[str, Any],
                    model_configuration: Dict[str, Any], batch_size: int, number_of_workers: int = 4) -> Tuple[
    DataLoader, DataLoader, DataLoader]:
    data_config = model_configuration.get("Data", {})
    tfrecord_config = data_config.get("TFRecord", {})
    use_tfrecord = tfrecord_config.get("activated", False)
    is_sharded = tfrecord_config.get("sharded", False)

    image_settings = data_config.get("image_settings", {})
    image_size = image_settings.get("size", 1024)
    keep_ratio = image_settings.get("keep_ratio", True)

    data_directory = experiment_configuration.get("data_folder", "")
    train_labels = experiment_configuration.get("train_split_file", "")

    tfrecord_path = os.path.join(data_directory, 'coco-train.tfrecord')
    sharded_pattern = os.path.join(data_directory, 'sharded', 'coco-train-*.tfrecord')

    dataset = None
    if use_tfrecord:
        if is_sharded and len(glob.glob(pathname=sharded_pattern)) > 0:
            dataset = TFRecordShardedCocoDataset(directory_pattern=sharded_pattern)
        elif os.path.exists(path=tfrecord_path):
            dataset = TFRecordCocoDataset(tensorflow_record_path=tfrecord_path)

    if dataset is None:
        if not data_directory or not train_labels or not os.path.isfile(train_labels):
            print(f"Warning: Falling back to dummy dataloader because data path does not exist: {data_directory}")
            dataset = DummyDataset(image_size=image_size)
        else:
            dataset = NativeCocoDataset(data_directory=data_directory, labels_file=train_labels, image_size=image_size,
                                        keep_ratio=keep_ratio)

    is_iterable = isinstance(dataset, IterableDataset)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=not is_iterable,
                              num_workers=number_of_workers, collate_fn=collate_function)
    return train_loader, train_loader, train_loader
