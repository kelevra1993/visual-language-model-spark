from tqdm import tqdm
from typing import List, Iterator
import numpy as np
import cv2
import json
import os
import sys
import glob
import torch
import subprocess
import tensorflow

from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader, IterableDataset, Dataset

from utilities.data_utilities import preprocess_image_and_boxes
from utilities.os_utilities import print_blue, print_yellow

# Hide GPU from TensorFlow so it does not reserve all memory, leaving none for PyTorch
tensorflow.config.set_visible_devices(devices=[], device_type='GPU')


def get_dataset_prefix(image_size: int, keep_ratio: bool) -> str:
    """
    Creates a standardized prefix string for dataset files based on the spatial image size and aspect ratio handling.
    This prefix ensures that TFRecord files for different preprocessing configurations are stored uniquely,
    preventing pipeline conflicts during concurrent hyperparameter sweeps.

    Args:
        image_size (int): The target spatial dimension to scale the images to.
        keep_ratio (bool): Whether the aspect ratio is maintained by padding during preprocessing.

    Returns:
        str: The generated prefix string to be prepended to dataset filenames.
    """
    return f"KAR-{image_size}-" if keep_ratio else f"{image_size}-"


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


def convert_to_bytes_feature(value: bytes) -> tensorflow.train.Feature:
    """
    Wraps a raw bytes value into a TensorFlow Feature protocol buffer.
    
    This helper function is essential during the serialization process of dataset generation,
    acting to encapsulate raw encoded image bytes and flattened tensor arrays into the structured
    Feature format required by TFRecords for downstream model consumption.
    
    Args:
        value (bytes): The raw bytes data to be encapsulated.
        
    Returns:
        tensorflow.train.Feature: The TensorFlow Feature protocol buffer containing the bytes list.
    """
    return tensorflow.train.Feature(bytes_list=tensorflow.train.BytesList(value=[value]))


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
    # Extract the strictly defined protocol buffer features from the raw serialized binary string
    features = tensorflow.io.parse_single_example(serialized=serialized_data, features={
        "images": tensorflow.io.FixedLenFeature(shape=[], dtype=tensorflow.string),
        "bounding_boxes": tensorflow.io.FixedLenFeature(shape=[], dtype=tensorflow.string),
        "labels": tensorflow.io.FixedLenFeature(shape=[], dtype=tensorflow.string)})

    # Decode the compressed PNG byte stream back into a dense three-channel tensor representation
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

        # Parse the monolithic COCO JSON annotation file into memory for rapid preprocessing
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

        # Load the physical image directly from the filesystem into a dense matrix format
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
        # Allocate memory to fetch large chunks of the file at once
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


def create_tfrecord(data_directory: str, labels_file: str, output_tensorflow_record: str, image_size: int,
                    keep_ratio: bool) -> None:
    """
    Generates a monolithic TFRecord archive file from the raw COCO dataset.

    This function processes images and bounding boxes preemptively by scaling them to the target spatial dimension
    and serializing them into a highly optimized binary format. This dramatically reduces disk I/O bottlenecks
    during the downstream training pipeline.

    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        labels_file (str): The absolute path to the JSON file containing the dataset annotations.
        output_tensorflow_record (str): The absolute path where the final TFRecord file will be saved.
        image_size (int): The target height and width for the scaled images.
        keep_ratio (bool): Whether to maintain the original aspect ratio by padding the images.
    """
    # Verify if the requested TFRecord file has already been completely generated to avoid redundant processing
    if os.path.exists(output_tensorflow_record):
        print(f"Skipping generation, final TFRecord already exists: {output_tensorflow_record}")
        return

    # Construct a temporary buffer file path to ensure atomic writing and prevent corruption
    buffer_record = os.path.join(os.path.dirname(output_tensorflow_record),
                                 "Buffer-" + os.path.basename(output_tensorflow_record))
    if os.path.exists(buffer_record):
        os.remove(buffer_record)

    print(f"Loading annotations from {labels_file}...")
    # Parse the monolithic COCO JSON annotation file into memory for rapid preprocessing
    with open(file=labels_file, mode='r') as file_handler:
        coco_data = json.load(fp=file_handler)

    # Construct lookup dictionaries mapping image identifiers to their metadata and annotations
    images_dictionary = {image['id']: image for image in tqdm(iterable=coco_data['images'], desc="Indexing Images")}
    image_to_annotations = {}

    for annotation in tqdm(iterable=coco_data['annotations'], desc="Indexing Annotations"):
        image_identifier = annotation['image_id']
        if image_identifier not in image_to_annotations:
            image_to_annotations[image_identifier] = []
        image_to_annotations[image_identifier].append(annotation)

    writer = tensorflow.io.TFRecordWriter(path=buffer_record)

    # Iterate systematically over every indexed image to perform spatial scaling and binary serialization
    for image_identifier, image_information in tqdm(iterable=images_dictionary.items(), desc="Creating TFRecord"):
        image_path = os.path.join(data_directory, image_information['file_name'])
        if not os.path.exists(path=image_path):
            continue

        # Load the physical image directly from the filesystem into a dense matrix format
        image = cv2.imread(filename=image_path)
        if image is None:
            continue

        # Convert the decoded image matrix from the default BGR format utilized by OpenCV into the standard RGB format
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
        image_annotations = image_to_annotations.get(image_identifier, [])
        bounding_boxes = []
        labels = []

        # Extract and format the bounding box coordinates and object categories associated with the current image
        for annotation in image_annotations:
            x_coordinate, y_coordinate, width, height = annotation['bbox']
            bounding_boxes.append([x_coordinate, y_coordinate, x_coordinate + width, y_coordinate + height])
            labels.append(annotation['category_id'])

        bounding_boxes_numpy = np.array(object=bounding_boxes, dtype=np.float32)

        # Dynamically scale the raw image and adjust the corresponding bounding box coordinates to match dimensions
        resized_image, resized_bounding_boxes = preprocess_image_and_boxes(image=image,
                                                                           bounding_boxes=bounding_boxes_numpy,
                                                                           image_size=image_size, keep_ratio=keep_ratio)

        # Compress the image matrix back into a PNG byte stream to dramatically reduce the final TFRecord file size
        success, encoded_image = cv2.imencode(ext='.png', img=resized_image)
        if not success:
            continue

        image_bytes = encoded_image.tobytes()

        # Flatten the multidimensional bounding box arrays into a one-dimensional list for protocol buffer serialization
        flattened_boxes = []
        for box in resized_bounding_boxes:
            flattened_boxes.extend(box)

        bounding_boxes_bytes = np.array(object=flattened_boxes, dtype=np.float32).tobytes()
        labels_bytes = np.array(object=labels, dtype=np.int64).tobytes()

        # Encapsulate the raw serialized byte streams into a strictly formatted TensorFlow Example protocol buffer
        example = tensorflow.train.Example(features=tensorflow.train.Features(
            feature={"images": convert_to_bytes_feature(value=image_bytes),
                     "bounding_boxes": convert_to_bytes_feature(value=bounding_boxes_bytes),
                     "labels": convert_to_bytes_feature(value=labels_bytes)}))
        writer.write(record=example.SerializeToString())

    writer.close()

    # Perform rename of the temporary buffer file to its final destination name to signal successful completion
    os.rename(buffer_record, output_tensorflow_record)
    print(f"Successfully generated {output_tensorflow_record}")


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
        # This allocates memory to fetch large chunks of the file at once,
        # preventing disk thrashing and accelerating load times
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


def create_tfrecord_sharded(data_directory: str, labels_file: str, output_directory: str, image_size: int,
                            keep_ratio: bool, number_of_shards: int = 10, prefix: str = "") -> None:
    """
    Generates a sharded sequence of TFRecord archive files from the raw COCO dataset.

    This function splits the dataset processing across multiple fragmented binary files to allow the data loading
    pipeline to interleave multiple shards concurrently, maximizing read throughput and minimizing disk thrashing
    during heavy multi-GPU or distributed training loops.

    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        labels_file (str): The absolute path to the JSON file containing the dataset annotations.
        output_directory (str): The absolute path to the directory where the shards will be saved.
        image_size (int): The target height and width for the scaled images.
        keep_ratio (bool): Whether to maintain the original aspect ratio by padding the images.
        number_of_shards (int): The exact number of shards to split the dataset into. Defaults to 10.
        prefix (str): An optional string prefix to prepend to each shard filename. Defaults to "".
    """
    # Ensure the destination directory exists to house the fragmented dataset shards
    os.makedirs(name=output_directory, exist_ok=True)

    # Scan the output directory to verify if the required number of completed shards already exist
    final_shards = glob.glob(os.path.join(output_directory, f"{prefix}coco-*.tfrecord"))
    final_shards = [f for f in final_shards if "Buffer-" not in f]
    if len(final_shards) == number_of_shards:
        print(f"Skipping sharded generation, already found {number_of_shards} shards in {output_directory}")
        return

    print(f"Loading annotations from {labels_file}...")
    with open(file=labels_file, mode='r') as file_handler:
        coco_data = json.load(fp=file_handler)

    images_dictionary = {image['id']: image for image in tqdm(iterable=coco_data['images'], desc="Indexing Images")}
    image_to_annotations = {}

    for annotation in tqdm(iterable=coco_data['annotations'], desc="Indexing Annotations"):
        image_identifier = annotation['image_id']
        if image_identifier not in image_to_annotations:
            image_to_annotations[image_identifier] = []
        image_to_annotations[image_identifier].append(annotation)

    # Calculate the exact distribution of images across the requested number of dataset shards
    image_keys = list(images_dictionary.keys())
    images_per_shard = len(image_keys) // number_of_shards + (1 if len(image_keys) % number_of_shards != 0 else 0)

    # Iterate through the calculated shard splits, independently serializing the partitioned image chunks
    for shard_index in range(number_of_shards):
        final_shard_path = os.path.join(output_directory,
                                        f"{prefix}coco-{shard_index:04d}-of-{number_of_shards:04d}.tfrecord")
        buffer_shard_path = os.path.join(output_directory,
                                         f"Buffer-{prefix}coco-{shard_index:04d}-of-{number_of_shards:04d}.tfrecord")

        if os.path.exists(final_shard_path):
            continue

        if os.path.exists(buffer_shard_path):
            os.remove(buffer_shard_path)

        writer = tensorflow.io.TFRecordWriter(path=buffer_shard_path)

        start_index = shard_index * images_per_shard
        end_index = min((shard_index + 1) * images_per_shard, len(image_keys))
        shard_keys = image_keys[start_index:end_index]

        # Process the designated chunk of images for the current shard, applying scaling and binary serialization
        for image_identifier in tqdm(iterable=shard_keys, desc=f"Creating Shard {shard_index + 1}/{number_of_shards}",
                                     leave=False):
            image_information = images_dictionary[image_identifier]
            image_path = os.path.join(data_directory, image_information['file_name'])
            if not os.path.exists(path=image_path):
                continue

            image = cv2.imread(filename=image_path)
            if image is None:
                continue

            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
            image_annotations = image_to_annotations.get(image_identifier, [])
            bounding_boxes = []
            labels = []
            for annotation in image_annotations:
                x_coordinate, y_coordinate, width, height = annotation['bbox']
                bounding_boxes.append([x_coordinate, y_coordinate, x_coordinate + width, y_coordinate + height])
                labels.append(annotation['category_id'])

            bounding_boxes_numpy = np.array(object=bounding_boxes, dtype=np.float32)
            resized_image, resized_bounding_boxes = preprocess_image_and_boxes(image=image,
                                                                               bounding_boxes=bounding_boxes_numpy,
                                                                               image_size=image_size,
                                                                               keep_ratio=keep_ratio)

            success, encoded_image = cv2.imencode(ext='.png', img=resized_image)
            if not success:
                continue

            image_bytes = encoded_image.tobytes()

            flattened_boxes = []
            for box in resized_bounding_boxes:
                flattened_boxes.extend(box)

            bounding_boxes_bytes = np.array(object=flattened_boxes, dtype=np.float32).tobytes()
            labels_bytes = np.array(object=labels, dtype=np.int64).tobytes()

            example = tensorflow.train.Example(features=tensorflow.train.Features(
                feature={"images": convert_to_bytes_feature(value=image_bytes),
                         "bounding_boxes": convert_to_bytes_feature(value=bounding_boxes_bytes),
                         "labels": convert_to_bytes_feature(value=labels_bytes)}))
            writer.write(record=example.SerializeToString())

        writer.close()
        # Atomically rename the completed shard buffer to confirm
        # successful generation and allow the dataloader to begin streaming
        os.rename(buffer_shard_path, final_shard_path)


def get_dataloader(split_name: str, split_file: str, dataset_prefix: str,
                   experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                   batch_size: int, device: torch.device, dtype: torch.dtype,
                   number_of_workers: int) -> Tuple[DataLoader, bool]:
    """
    Instantiates and retrieves the appropriate DataLoader for designated data within the machine learning pipeline.
    This function dynamically attempts to utilize highly-optimized TFRecords if requested, but gracefully falls back 
    to the standard Native PyTorch dataset if the records have not yet been generated.

    Args:
        split_name (str): The name of the split (e.g., 'Train', 'Validation', 'Test').
        split_file (str): The absolute path to the JSON file containing the dataset annotations for this split.
        dataset_prefix (str): The standardized prefix identifying the preprocessing configuration.
        experiment_configuration (Dict[str, Any]): The overall experiment configuration including data paths.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.
        device (torch.device): The hardware device where the output tensors will be located.
        dtype (torch.dtype): The numerical precision type for the tensors.
        number_of_workers (int): The number of parallel subprocesses to allocate for data loading.

    Returns:
        Tuple[DataLoader, bool]: A tuple containing the configured DataLoader and a boolean flag indicating if the 
                                 system fell back to the Native dataset because TFRecords were missing.
    """
    data_configuration = model_configuration["Data"]
    tfrecord_configuration = data_configuration["TFRecord"]
    use_tfrecord = tfrecord_configuration["activated"]
    is_sharded = tfrecord_configuration["sharded"]
    buffer_size = tfrecord_configuration["buffer_size"]
    number_of_shards = tfrecord_configuration["number_shards"]

    image_settings = data_configuration["image_settings"]
    image_size = image_settings["size"]
    keep_ratio = image_settings["keep_ratio"]

    data_directory = experiment_configuration["data_folder"]

    tfrecord_filename = f"{dataset_prefix}{split_name}.tfrecord"
    tfrecord_path = os.path.join(data_directory, tfrecord_filename)

    sharded_directory = os.path.join(data_directory, f"{dataset_prefix}Sharded-Records-{split_name}")
    sharded_pattern = os.path.join(sharded_directory, f"coco-*.tfrecord")

    dataset = None
    fallback_to_native = False

    # Attempt to locate and load the highly optimized TFRecord archives to maximize data throughput
    if use_tfrecord:
        if is_sharded:
            # Verify that all required shards have been completely generated and are not currently buffering
            final_shards = glob.glob(pathname=sharded_pattern)
            final_shards = [file_path for file_path in final_shards if "Buffer-" not in file_path]

            if len(final_shards) == number_of_shards:
                dataset = TFRecordShardedCocoDataset(directory_pattern=sharded_pattern, device=device, dtype=dtype,
                                                     buffer_size=buffer_size)
        else:
            # Verify the monolithic TFRecord exists and is not currently being written by a background process
            if os.path.exists(path=tfrecord_path) and "Buffer-" not in tfrecord_path:
                dataset = TFRecordCocoDataset(tensorflow_record_path=tfrecord_path, device=device, dtype=dtype,
                                              buffer_size=buffer_size)

        # Flag the fallback state if the TFRecords were requested but could not be safely located
        if dataset is None:
            fallback_to_native = True
            print_yellow(output=f"TFRecords for {split_name} not found. Falling back to Native dataset.")

    # Instantiate the standard PyTorch Native dataset if TFRecords are disabled or missing,
    # ensuring training can commence immediately
    if dataset is None:
        dataset = NativeCocoDataset(data_directory=data_directory, labels_file=split_file, image_size=image_size,
                                    keep_ratio=keep_ratio, device=device, dtype=dtype)

    # Determine if the dataset streams data to properly configure the shuffle parameter
    is_iterable = isinstance(dataset, IterableDataset)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=not is_iterable,
                        num_workers=number_of_workers, collate_fn=collate_function)

    return loader, fallback_to_native


def get_dataloaders(experiment_configuration: Dict[str, Any],
                    model_configuration: Dict[str, Any], batch_size: int, device: torch.device, dtype: torch.dtype,
                    number_of_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, bool]:
    """
    Initializes and returns the training, validation, and testing dataloaders for the object detection pipeline.
    This function acts as the central coordinator for the data loading subsystem, establishing the dataset 
    prefixes and orchestrating the background generation of TFRecords if they are missing.

    Args:
        experiment_configuration (Dict[str, Any]): The overall experiment configuration including data paths.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.
        device (torch.device): The hardware device where the output tensors will be located.
        dtype (torch.dtype): The numerical precision type for the tensors.
        number_of_workers (int): The number of parallel subprocesses to allocate for data loading. Defaults to 4.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, bool]: A tuple containing the configured DataLoaders for the 
                                                         training, validation, and testing splits respectively, 
                                                         along with a boolean flag indicating if the system is waiting 
                                                         for background TFRecords to be generated.
    """
    data_configuration = model_configuration["Data"]
    image_settings = data_configuration["image_settings"]
    image_size = image_settings["size"]
    keep_ratio = image_settings["keep_ratio"]

    dataset_prefix = get_dataset_prefix(image_size=image_size, keep_ratio=keep_ratio)

    train_loader, train_fallback = get_dataloader(split_name="Train",
                                                  split_file=experiment_configuration["train_split_file"],
                                                  dataset_prefix=dataset_prefix,
                                                  experiment_configuration=experiment_configuration,
                                                  model_configuration=model_configuration,
                                                  batch_size=batch_size,
                                                  device=device, dtype=dtype, number_of_workers=number_of_workers)

    validation_loader, validation_fallback = get_dataloader(split_name="Validation",
                                                            split_file=experiment_configuration[
                                                                "validation_split_file"],
                                                            dataset_prefix=dataset_prefix,
                                                            experiment_configuration=experiment_configuration,
                                                            model_configuration=model_configuration,
                                                            batch_size=batch_size,
                                                            device=device, dtype=dtype,
                                                            number_of_workers=number_of_workers)

    test_loader, test_fallback = get_dataloader(split_name="Test",
                                                split_file=experiment_configuration["test_split_file"],
                                                dataset_prefix=dataset_prefix,
                                                experiment_configuration=experiment_configuration,
                                                model_configuration=model_configuration,
                                                batch_size=batch_size,
                                                device=device, dtype=dtype, number_of_workers=number_of_workers)

    waiting_for_tfrecords = train_fallback or validation_fallback or test_fallback

    # Launch a detached background worker to generate the highly optimized TFRecords
    # without blocking the primary training loop
    if waiting_for_tfrecords:
        print_blue(output="TFRecord generation required. Spawning background process...")

        creator_script_path = os.path.join(os.path.dirname(__file__), "tfrecord_creator.py")

        # Build the exact argument list for the dedicated creator script to mirror the current configuration
        command = [
            sys.executable, creator_script_path,
            "--data_folder", experiment_configuration["data_folder"],
            "--image_size", str(image_size),
            "--keep_ratio", str(keep_ratio),
            "--is_sharded", str(data_configuration["TFRecord"]["sharded"]),
            "--number_of_shards", str(data_configuration["TFRecord"].get("number_shards", 10)),
            "--prefix", dataset_prefix,
            "--train_split", experiment_configuration["train_split_file"],
            "--validation_split", experiment_configuration["validation_split_file"],
            "--test_split", experiment_configuration["test_split_file"]
        ]

        subprocess.Popen(args=command, start_new_session=True)

    return train_loader, validation_loader, test_loader, waiting_for_tfrecords
