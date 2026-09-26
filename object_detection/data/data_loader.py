import os
import glob
import json
import torch
import cv2
import numpy as np
import tensorflow

from typing import Dict, Any, Tuple, List, Iterator
from torch.utils.data import Dataset, DataLoader, IterableDataset
from utilities.data_utilities import preprocess_image_and_boxes
from utilities.os_utilities import print_blue

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


def parse_single_example(serialized_data: tensorflow.Tensor) -> Tuple[
    tensorflow.Tensor, tensorflow.Tensor, tensorflow.Tensor]:
    """
    Parses a single serialized tensorflow.train.Example protobuf into distinct TensorFlow tensors.
    
    This function decodes the raw image bytes into pixel values and reconstructs the sparse
    bounding boxes and labels arrays for downstream PyTorch conversion.
    
    Args:
        serialized_data (tensorflow.Tensor): The raw serialized protocol buffer string from the TFRecord.
        
    Returns:
        Tuple[tensorflow.Tensor, tensorflow.Tensor, tensorflow.Tensor]: A tuple containing the decoded image tensor, 
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


class NativeCocoDataset(Dataset):
    """
    A PyTorch Dataset implementation for reading the raw COCO image format.

    This class reads images directly from the standard filesystem and scales them dynamically 
    on the CPU during each dataloader fetch iteration.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool, device: torch.device,
                 dtype: torch.dtype) -> None:
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
        self.device = device
        self.dtype = dtype

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
        image_tensor = torch.from_numpy(np.array(object=resized_image)).permute(2, 0, 1).to(device=self.device,
                                                                                            dtype=self.dtype)

        return {"images": image_tensor,
                "bounding_boxes": torch.tensor(data=resized_bounding_boxes, dtype=self.dtype, device=self.device),
                "labels": torch.tensor(data=labels, dtype=torch.int64, device=self.device)}


class TFRecordCocoDataset(IterableDataset):
    """
    A PyTorch IterableDataset implementation for reading pre-processed TFRecord archives using TensorFlow.
    
    It streams records directly from the disk using tensorflow.data for maximum efficiency.
    """

    def __init__(self, tensorflow_record_path: str, device: torch.device, dtype: torch.dtype, buffer_size: int) -> None:
        """
        Initializes the TFRecord iterable dataset.
        
        Args:
            tensorflow_record_path (str): The absolute path to the TFRecord archive file.
            device (torch.device): The device on which to place the output tensors.
            dtype (torch.dtype): The data type for the output tensors.
            buffer_size (int): The number of bytes in the read buffer.
        """
        self.tensorflow_record_path = tensorflow_record_path
        self.device = device
        self.dtype = dtype
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Returns an iterator over the dataset using optimized tensorflow.data pipeline.

        Returns:
            Iterator[Dict[str, torch.Tensor]]: An iterator yielding dictionaries containing images, boxes, and labels.
        """
        worker_info = torch.utils.data.get_worker_info()
        # Specify the read buffer size in bytes to optimize I/O throughput when streaming from the disk
        # This allocates memory to fetch large chunks of the file at once, preventing disk thrashing and accelerating load times
        dataset = tensorflow.data.TFRecordDataset(filenames=[self.tensorflow_record_path], buffer_size=self.buffer_size)

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
            image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).contiguous().to(device=self.device,
                                                                                          dtype=self.dtype)

            yield {"images": image_tensor,
                   "bounding_boxes": torch.tensor(data=boxes_numpy, dtype=self.dtype, device=self.device),
                   "labels": torch.tensor(data=labels_numpy, dtype=torch.int64, device=self.device)}


class TFRecordShardedCocoDataset(IterableDataset):
    """
    A PyTorch IterableDataset implementation for reading from multiple sharded TFRecord files.
    """

    def __init__(self, directory_pattern: str, device: torch.device, dtype: torch.dtype, buffer_size: int) -> None:
        """
        Initializes the sharded TFRecord iterable dataset.
        
        Args:
            directory_pattern (str): The wildcard pattern used to locate all TFRecord shards.
            device (torch.device): The device on which to place the output tensors.
            dtype (torch.dtype): The data type for the output tensors.
            buffer_size (int): The number of bytes in the read buffer.
        """
        self.tensorflow_record_files = sorted(glob.glob(pathname=directory_pattern))
        self.device = device
        self.dtype = dtype
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Returns an iterator over the dataset using optimized tensorflow.data pipeline.

        Returns:
            Iterator[Dict[str, torch.Tensor]]: An iterator yielding dictionaries containing images, boxes, and labels.
        """
        worker_info = torch.utils.data.get_worker_info()
        # Specify the read buffer size in bytes to optimize I/O throughput when streaming from the disk
        # This allocates memory to fetch large chunks of the file at once, preventing disk thrashing and accelerating load times
        dataset = tensorflow.data.TFRecordDataset(filenames=self.tensorflow_record_files, buffer_size=self.buffer_size)

        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        dataset = dataset.map(map_func=parse_single_example, num_parallel_calls=tensorflow.data.AUTOTUNE)

        for image, bounding_boxes, labels in dataset:
            image_numpy = image.numpy()
            boxes_numpy = bounding_boxes.numpy()
            labels_numpy = labels.numpy()

            image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).contiguous().to(device=self.device,
                                                                                          dtype=self.dtype)

            yield {"images": image_tensor,
                   "bounding_boxes": torch.tensor(data=boxes_numpy, dtype=self.dtype, device=self.device),
                   "labels": torch.tensor(data=labels_numpy, dtype=torch.int64, device=self.device)}


def get_dataloaders(preprocessed_directory: str, experiment_configuration: Dict[str, Any],
                    model_configuration: Dict[str, Any], batch_size: int, device: torch.device, dtype: torch.dtype,
                    number_of_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes and returns the training, validation, and testing dataloaders for the object detection pipeline.

    This function dynamically resolves the appropriate dataset implementation (Native or TFRecord) based on
    the model configuration, injecting the globally specified tensor devices and data types.

    Args:
        preprocessed_directory (str): The directory containing preprocessed dataset files.
        experiment_configuration (Dict[str, Any]): The configuration dictionary containing data folder and split definitions.
        model_configuration (Dict[str, Any]): The configuration dictionary containing data loading parameters.
        batch_size (int): The number of images per batch.
        device (torch.device): The hardware device to allocate output tensors to.
        dtype (torch.dtype): The numerical precision for the image and bounding box tensors.
        number_of_workers (int): The number of subprocesses to use for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the DataLoaders for
         training, validation, and testing splits.
    """
    data_configuration = model_configuration["Data"]
    tfrecord_configuration = data_configuration["TFRecord"]
    use_tfrecord = tfrecord_configuration["activated"]
    is_sharded = tfrecord_configuration["sharded"]
    buffer_size = tfrecord_configuration["buffer_size"]

    image_settings = data_configuration["image_settings"]
    image_size = image_settings["size"]
    keep_ratio = image_settings["keep_ratio"]

    data_directory = experiment_configuration["data_folder"]
    train_labels = experiment_configuration["train_split_file"]

    # todo will need to be changed
    # should have conventional nemaes
    tfrecord_path = os.path.join(data_directory, 'coco-train.tfrecord')
    sharded_pattern = os.path.join(data_directory, 'sharded', 'coco-train-*.tfrecord')

    dataset = None
    if use_tfrecord:
        # todo we might need a comment because not clear how we deal with this if the sharded tfrectords are not there.
        if is_sharded and len(glob.glob(pathname=sharded_pattern)) > 0:
            dataset = TFRecordShardedCocoDataset(directory_pattern=sharded_pattern, device=device, dtype=dtype,
                                                 buffer_size=buffer_size)
        elif os.path.exists(path=tfrecord_path):
            dataset = TFRecordCocoDataset(tensorflow_record_path=tfrecord_path, device=device, dtype=dtype,
                                          buffer_size=buffer_size)
        else:
            # todo we might need to a message to say that we are using the NativeDataset and that the tfrecord or shareded tfrecord creation will run in the background
            #  and once it is done the code will stop and re-run to launch itself on the tfrecords that have been created.
            # logic_to_be_implemented_here.
            pass

    # TODO Need of comment
    if dataset is None:
        print_blue(f"We Are Using The Native Dataset For {put_something_clear}")
        dataset = NativeCocoDataset(data_directory=data_directory, labels_file=train_labels, image_size=image_size,
                                        keep_ratio=keep_ratio, device=device, dtype=dtype)

    is_iterable = isinstance(dataset, IterableDataset)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=not is_iterable,
                              num_workers=number_of_workers, collate_fn=collate_function)

    # todo this is not good should not have been implemented like this.
    return train_loader, train_loader, train_loader
