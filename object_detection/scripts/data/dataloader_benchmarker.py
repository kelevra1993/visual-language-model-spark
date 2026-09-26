import os
from pathlib import Path

from utilities.os_utilities import print_blue

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import csv
import time
import cv2
import numpy
import torch
import tensorflow
from tqdm import tqdm
import glob
from typing import List, Dict, Any, Callable
from torch.utils.data import Dataset, DataLoader, IterableDataset
from utilities.data_utilities import preprocess_image_and_boxes, view_input_data

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


def convert_to_bytes_feature(value: bytes) -> tensorflow.train.Feature:
    """
    Converts a standard byte string into a TensorFlow compatible Feature object.
    
    This function wraps the raw bytes so they can be securely serialized within the 
    tensorflow.train.Example protocol buffer, which is required by the TensorFlow data pipeline.
    
    Args:
        value (bytes): The raw byte string to be encapsulated.
        
    Returns:
        tensorflow.train.Feature: The TensorFlow Feature object containing the byte string.
    """
    return tensorflow.train.Feature(bytes_list=tensorflow.train.BytesList(value=[value]))


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


def clean_data(data_directory: str, labels_file: str) -> None:
    """
    Sanitizes the COCO dataset by removing images without annotations and annotations without images.

    This ensures that the dataloader does not crash when encountering missing files
    or empty labels during the training loop. It writes the filtered output back
    to the original JSON file path.

    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        labels_file (str): The absolute path to the JSON file containing the COCO annotations.

    Returns:
        None
    """
    print(f"Loading annotations from {labels_file}...")
    with open(file=labels_file, mode='r') as file_handler:
        coco_data = json.load(fp=file_handler)

    print("Checking for missing images...")
    filtered_images = []
    for image in tqdm(iterable=coco_data['images'], desc="Filtering Images", leave=False):
        image_path = os.path.join(data_directory, image['file_name'])
        if os.path.exists(path=image_path):
            filtered_images.append(image)

    valid_image_ids = {image['id'] for image in filtered_images}

    print("Filtering annotations for valid images...")
    filtered_annotations = []
    for annotation in tqdm(iterable=coco_data['annotations'], desc="Filtering Annotations", leave=False):
        if annotation['image_id'] in valid_image_ids:
            filtered_annotations.append(annotation)

    coco_data['images'] = filtered_images
    coco_data['annotations'] = filtered_annotations

    print(f"Writing cleaned annotations back to {labels_file}...")
    with open(file=labels_file, mode='w') as file_handler:
        json.dump(obj=coco_data, fp=file_handler, indent=4)
    print("Data cleaning complete.")


class NativeCocoDataset(Dataset):
    """
    A PyTorch Dataset implementation for reading the raw COCO image format.

    This class serves as the baseline for the data loading benchmark pipeline. It reads
    images directly from the standard filesystem and scales them dynamically on the CPU
    during each dataloader fetch iteration.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool) -> None:
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

        bounding_boxes_numpy = numpy.array(object=bounding_boxes, dtype=numpy.float32)

        # Dynamically scale the image and calculate the corresponding adjustments for the bounding boxes
        resized_image, resized_bounding_boxes = preprocess_image_and_boxes(image=image,
                                                                           bounding_boxes=bounding_boxes_numpy,
                                                                           image_size=self.image_size,
                                                                           keep_ratio=self.keep_ratio)
        # Convert the resized array into a PyTorch float tensor
        # Note: The standard division by 255.0 has been intentionally omitted to preserve raw pixel scale
        image_tensor = torch.from_numpy(numpy.array(object=resized_image)).permute(2, 0, 1).float()

        return {"images": image_tensor,
                "bounding_boxes": torch.tensor(data=resized_bounding_boxes, dtype=torch.float32),
                "labels": torch.tensor(data=labels, dtype=torch.int64)}


def benchmark_native(data_directory: str, labels_file: str, number_of_runs: int, batch_size: int, image_size: int,
                     keep_ratio: bool, view_images: bool) -> List[float]:
    """
    Benchmarks the Native PyTorch dataset loader.

    This function evaluates the baseline throughput of CPU decoding by loading images from the disk
    and scaling them preemptively in the data pipeline.

    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        labels_file (str): The absolute path to the JSON file containing the COCO annotations.
        number_of_runs (int): The number of full epoch passes to simulate.
        batch_size (int): The number of images per batch.
        image_size (int): The target height and width for the scaled images.
        keep_ratio (bool): Whether to maintain the aspect ratio during scaling by padding.
        view_images (bool): Whether to visualize the batches using OpenCV.

    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = NativeCocoDataset(data_directory=data_directory, labels_file=labels_file, image_size=image_size,
                                keep_ratio=keep_ratio)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_function)
    times = []

    # Iteratively drain the dataloader to precisely measure the wall-clock time required for full epoch traversals
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for batch_data_dictionary in tqdm(iterable=dataloader, desc=f"Running Epoch {_run_index + 1}", leave=False):
            if view_images:
                # Iterate through each item in the current batch explicitly
                for batch_index in range(batch_data_dictionary["images"].size(0)):
                    # Dispatch the individual item to the visualization utility to render the ground truth annotations
                    user_quit = view_input_data(image=batch_data_dictionary["images"][batch_index],
                                                bounding_boxes=batch_data_dictionary["bounding_boxes"][batch_index],
                                                labels=batch_data_dictionary["labels"][batch_index])
                    if user_quit:
                        return 0.0
        times.append(time.time() - start_time)

    return times


def create_tfrecord(data_directory: str, labels_file: str, output_tensorflow_record: str, image_size: int,
                    keep_ratio: bool) -> None:
    """
    Parses a COCO dataset and compiles it into an optimized TFRecord archive.
    
    This function accelerates downstream training by preemptively scaling all images 
    and encoding them directly into a continuous binary file format.
    
    Args:
        data_directory (str): The absolute path to the physical image files.
        labels_file (str): The absolute path to the JSON COCO annotations.
        output_tensorflow_record (str): The absolute destination path for the TFRecord archive.
        image_size (int): The target dimension to scale the images to.
        keep_ratio (bool): Whether to pad the scaled images to maintain aspect ratio.
        
    Returns:
        None
    """
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

    writer = tensorflow.io.TFRecordWriter(path=output_tensorflow_record)

    for image_identifier, image_information in tqdm(iterable=images_dictionary.items(), desc="Creating TFRecord"):
        image_path = os.path.join(data_directory, image_information['file_name'])
        if not os.path.exists(path=image_path):
            continue

        image = cv2.imread(filename=image_path)
        if image is None:
            continue

        # Convert BGR to RGB intentionally. OpenCV's imencode assumes input is BGR and swaps channels for PNG storage.
        # By providing an RGB array, imencode physically saves a BGR PNG file. When tf.io.decode_png reads it, 
        # it natively returns a BGR tensor, ensuring consistency with the raw cv2.imread Native pipeline.
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

        image_annotations = image_to_annotations.get(image_identifier, [])
        bounding_boxes = []
        labels = []
        for annotation in image_annotations:
            x_coordinate, y_coordinate, width, height = annotation['bbox']
            bounding_boxes.append([x_coordinate, y_coordinate, x_coordinate + width, y_coordinate + height])
            labels.append(annotation['category_id'])

        bounding_boxes_numpy = numpy.array(object=bounding_boxes, dtype=numpy.float32)

        # Preemptively process the image to eliminate redundant CPU cycles during the actual training loop
        resized_image, resized_bounding_boxes = preprocess_image_and_boxes(image=image,
                                                                           bounding_boxes=bounding_boxes_numpy,
                                                                           image_size=image_size, keep_ratio=keep_ratio)

        # Re-encode the image as a standard PNG format to drastically reduce the binary archive size
        success, encoded_image = cv2.imencode(ext='.png', img=resized_image)
        if not success:
            continue

        image_bytes = encoded_image.tobytes()

        flattened_boxes = []
        for box in resized_bounding_boxes:
            flattened_boxes.extend(box)

        bounding_boxes_bytes = numpy.array(object=flattened_boxes, dtype=numpy.float32).tobytes()
        labels_bytes = numpy.array(object=labels, dtype=numpy.int64).tobytes()

        example = tensorflow.train.Example(features=tensorflow.train.Features(
            feature={"images": convert_to_bytes_feature(value=image_bytes),
                     "bounding_boxes": convert_to_bytes_feature(value=bounding_boxes_bytes),
                     "labels": convert_to_bytes_feature(value=labels_bytes)}))
        writer.write(record=example.SerializeToString())

    writer.close()


class TFRecordCocoDataset(IterableDataset):
    """
    A PyTorch IterableDataset implementation for reading pre-processed TFRecord archives using TensorFlow.
    
    This dataset serves as a high-throughput format in the data loading benchmark pipeline. 
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


def benchmark_tfrecord(tensorflow_record_path: str, number_of_runs: int, batch_size: int, view_images: bool) -> List[
    float]:
    """
    Benchmarks the Pre-Resized TFRecord dataset loader.
    
    This function evaluates the absolute theoretical maximum throughput of CPU decoding by 
    combining binary archive storage with preemptive spatial scaling in the benchmark pipeline.
    
    Args:
        tensorflow_record_path (str): The absolute path to the pre-resized TFRecord archive file.
        number_of_runs (int): The number of full epoch passes to simulate.
        batch_size (int): The number of images per batch.
        view_images (bool): Whether to visualize the batches using OpenCV.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = TFRecordCocoDataset(tensorflow_record_path=tensorflow_record_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_function)
    times = []

    # Iteratively drain the dataloader to precisely measure the wall-clock time required for full epoch traversals
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for batch_data_dictionary in tqdm(iterable=dataloader, desc=f"TFRecord Epoch {_run_index + 1}", leave=False):
            if view_images:
                # Iterate through each item in the current batch explicitly
                for batch_index in range(batch_data_dictionary["images"].size(0)):
                    # Dispatch the individual item to the visualization utility to render the ground truth annotations
                    user_quit = view_input_data(image=batch_data_dictionary["images"][batch_index],
                                                bounding_boxes=batch_data_dictionary["bounding_boxes"][batch_index],
                                                labels=batch_data_dictionary["labels"][batch_index])
                    if user_quit:
                        return 0.0
        times.append(time.time() - start_time)

    return times


def create_tfrecord_sharded(data_directory: str, labels_file: str, output_directory: str, image_size: int,
                            keep_ratio: bool, number_of_shards: int = 10) -> None:
    """
    Parses a COCO dataset and compiles it into optimized sharded TFRecord archives.
    
    Args:
        data_directory (str): The absolute path to the physical image files.
        labels_file (str): The absolute path to the JSON COCO annotations.
        output_directory (str): The destination directory for the sharded TFRecord archives.
        image_size (int): The target dimension to scale the images to.
        keep_ratio (bool): Whether to pad the scaled images to maintain aspect ratio.
        number_of_shards (int): Number of shards to create.
        
    Returns:
        None
    """
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

    os.makedirs(name=output_directory, exist_ok=True)

    image_keys = list(images_dictionary.keys())
    images_per_shard = len(image_keys) // number_of_shards + (1 if len(image_keys) % number_of_shards != 0 else 0)

    for shard_index in range(number_of_shards):
        shard_path = os.path.join(output_directory, f"coco-train-{shard_index:04d}-of-{number_of_shards:04d}.tfrecord")
        writer = tensorflow.io.TFRecordWriter(path=shard_path)

        start_index = shard_index * images_per_shard
        end_index = min((shard_index + 1) * images_per_shard, len(image_keys))
        shard_keys = image_keys[start_index:end_index]

        for image_identifier in tqdm(iterable=shard_keys, desc=f"Creating Shard {shard_index + 1}/{number_of_shards}",
                                     leave=False):
            image_information = images_dictionary[image_identifier]
            image_path = os.path.join(data_directory, image_information['file_name'])
            if not os.path.exists(path=image_path):
                continue

            image = cv2.imread(filename=image_path)
            if image is None:
                continue

            # Convert BGR to RGB intentionally. OpenCV's imencode assumes input is BGR and swaps channels for PNG storage.
            # By providing an RGB array, imencode physically saves a BGR PNG file. When tf.io.decode_png reads it, 
            # it natively returns a BGR tensor, ensuring consistency with the raw cv2.imread Native pipeline.
            image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

            image_annotations = image_to_annotations.get(image_identifier, [])
            bounding_boxes = []
            labels = []
            for annotation in image_annotations:
                x_coordinate, y_coordinate, width, height = annotation['bbox']
                bounding_boxes.append([x_coordinate, y_coordinate, x_coordinate + width, y_coordinate + height])
                labels.append(annotation['category_id'])

            bounding_boxes_numpy = numpy.array(object=bounding_boxes, dtype=numpy.float32)

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

            bounding_boxes_bytes = numpy.array(object=flattened_boxes, dtype=numpy.float32).tobytes()
            labels_bytes = numpy.array(object=labels, dtype=numpy.int64).tobytes()

            example = tensorflow.train.Example(features=tensorflow.train.Features(
                feature={"images": convert_to_bytes_feature(value=image_bytes),
                         "bounding_boxes": convert_to_bytes_feature(value=bounding_boxes_bytes),
                         "labels": convert_to_bytes_feature(value=labels_bytes)}))
            writer.write(record=example.SerializeToString())

        writer.close()


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


def benchmark_tfrecord_sharded(directory_pattern: str, number_of_runs: int, batch_size: int, view_images: bool) -> List[
    float]:
    """
    Benchmarks the sharded TFRecord dataset loader.
    
    Args:
        directory_pattern (str): The wildcard pattern used to locate all TFRecord shards.
        number_of_runs (int): The number of full epoch passes to simulate.
        batch_size (int): The number of images per batch.
        view_images (bool): Whether to visualize the batches using OpenCV.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = TFRecordShardedCocoDataset(directory_pattern=directory_pattern)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_function)
    times = []

    for _run_index in range(number_of_runs):
        start_time = time.time()
        for batch_data_dictionary in tqdm(iterable=dataloader, desc=f"TFRecord Sharded Epoch {_run_index + 1}",
                                          leave=False):
            if view_images:
                for batch_index in range(batch_data_dictionary["images"].size(0)):
                    user_quit = view_input_data(image=batch_data_dictionary["images"][batch_index],
                                                bounding_boxes=batch_data_dictionary["bounding_boxes"][batch_index],
                                                labels=batch_data_dictionary["labels"][batch_index])
                    if user_quit:
                        return 0.0
        times.append(time.time() - start_time)

    return times


def get_dataset_paths(project_base_directory: str, image_size: int, keep_ratio: bool) -> Dict[str, str]:
    """
    Constructs and orchestrates the standardized filesystem paths required by the dataloader format benchmarker pipeline to locate annotations and output compiled archives.
    
    Args:
        project_base_directory (str): The root directory of the project.
        image_size (int): The target image dimension used to construct file prefixes.
        keep_ratio (bool): Whether the aspect ratio is maintained, used for prefix formatting.
        
    Returns:
        Dict[str, str]: A dictionary containing absolute paths for data, labels, and output formats.
    """
    base_directory = os.path.join(project_base_directory, 'coco-2017', 'train')
    original_data_directory = os.path.join(base_directory, 'data')
    original_labels = os.path.join(base_directory, 'labels.json')

    prefix = f"KAR-{image_size}-" if keep_ratio else f"{image_size}-"
    formats_directory = os.path.join(project_base_directory, 'formats')
    tensorflow_record_path = os.path.join(formats_directory, f'{prefix}coco-train.tfrecord')
    sharded_output_directory = os.path.join(formats_directory, f'{prefix}Sharded-Records')
    tfrecord_sharded_pattern = os.path.join(sharded_output_directory, 'coco-train-*.tfrecord')

    return {
        "data_directory": original_data_directory,
        "labels_file": original_labels,
        "tensorflow_record_path": tensorflow_record_path,
        "sharded_output_directory": sharded_output_directory,
        "tfrecord_sharded_pattern": tfrecord_sharded_pattern
    }


def prepare_tfrecords(paths_dictionary: Dict[str, str], image_size: int, keep_ratio: bool) -> None:
    """
    Validates the presence of the pre-resized and sharded TFRecord archives on disk, dynamically compiling them from the raw COCO dataset if missing, in order to guarantee data availability for the benchmarking pipeline.
    
    Args:
        paths_dictionary (Dict[str, str]): A dictionary containing dataset input and output paths.
        image_size (int): The target dimension to scale images.
        keep_ratio (bool): Whether to pad images during scaling to maintain aspect ratio.
        
    Returns:
        None
    """
    tensorflow_record_path = paths_dictionary["tensorflow_record_path"]
    original_data_directory = paths_dictionary["data_directory"]
    original_labels = paths_dictionary["labels_file"]

    if not os.path.exists(path=tensorflow_record_path):
        print(f"TFRecord file not found at {tensorflow_record_path}. Generating it now...")
        os.makedirs(name=os.path.dirname(p=tensorflow_record_path), exist_ok=True)
        create_tfrecord(data_directory=original_data_directory, labels_file=original_labels,
                        output_tensorflow_record=tensorflow_record_path, image_size=image_size, keep_ratio=keep_ratio)
        print("TFRecord generation complete!")
    else:
        print(f"Found existing TFRecord file at {tensorflow_record_path}.")

    sharded_output_directory = paths_dictionary["sharded_output_directory"]
    tfrecord_sharded_pattern = paths_dictionary["tfrecord_sharded_pattern"]

    if not os.path.exists(path=sharded_output_directory) or len(glob.glob(pathname=tfrecord_sharded_pattern)) == 0:
        print(f"Sharded TFRecords not found at {sharded_output_directory}. Generating them now...")
        create_tfrecord_sharded(data_directory=original_data_directory, labels_file=original_labels,
                                output_directory=sharded_output_directory, image_size=image_size, keep_ratio=keep_ratio,
                                number_of_shards=10)
        print("Sharded TFRecord generation complete!")
    else:
        print(f"Found existing sharded TFRecords at {sharded_output_directory}.")


def benchmark_method(method_name: str, method_function: Callable, benchmark_arguments: Dict[str, Any]) -> List[float]:
    """
    Wraps and executes a designated benchmark sequence.
    
    This function abstracts the execution mechanism away from the main logic. It intentionally 
    omits exception handling so that any dataloader failures halt execution immediately.
    
    Args:
        method_name (str): The descriptive name of the dataloader format being tested.
        method_function (Callable): The specific benchmark function to invoke.
        benchmark_arguments (Dict[str, Any]): The explicit arguments to pass into the method.
        
    Returns:
        List[float]: A list of epoch times in seconds.
    """
    print(f"Benchmarking {method_name}...")
    times = method_function(**benchmark_arguments)
    if times:
        print("")
        print_blue(f"[{method_name}] Average Time: {sum(times) / len(times):.2f} s", indent=1)
    return times


def save_benchmark_results(csv_file_path: str, methods_to_benchmark: List[str], results: Dict[str, List[float]], number_of_runs: int) -> None:
    """
    Persists the collected benchmark execution times to a CSV file for offline analysis.
    
    Args:
        csv_file_path (str): The absolute or relative path where the CSV report will be written.
        methods_to_benchmark (List[str]): The specific dataloader architectures that were actively tested.
        results (Dict[str, List[float]]): The recorded epoch execution times mapped by dataloader strategy.
        number_of_runs (int): The predefined number of simulated training epochs to iterate through.
        
    Returns:
        None
    """
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(methods_to_benchmark)

        # Iterate across each simulated epoch to aggregate the individual run times into standardized rows
        for _index in range(number_of_runs):
            row = []
            for method_name in methods_to_benchmark:
                if _index < len(results[method_name]):
                    value = results[method_name][_index]
                    row.append(f"{value:.4f}")
                else:
                    row.append('')
            writer.writerow(row)

        # Compute and append the final average throughput times across all successful runs to conclude the benchmark report
        average_row = []
        for method_name in methods_to_benchmark:
            if results[method_name]:
                average = sum(results[method_name]) / len(results[method_name])
                average_row.append(f"{average:.4f}")
            else:
                average_row.append('')
        writer.writerow(average_row)

    print(f"Results saved to {csv_file_path}")

def main() -> None:
    """
    Executes the comprehensive Dataloader Format Benchmarking suite.

    This overarching pipeline orchestrates the sequential testing of various data storage
    strategies to empirically determine the optimal throughput architecture for the downstream trainer.

    Returns:
        None
    """
    # Define the core benchmarking parameters that govern the scale and configuration of the dataset testing
    number_of_runs = 20
    batch_size = 20
    image_size = 1024
    keep_ratio = True
    view_images = False

    # Specify the dataloader architectures that should actively be executed during the current benchmark run
    methods_to_benchmark = [
        'Native',
        'TFRecord',
        'Sharded'
    ]

    # Construct the absolute system paths required to locate the underlying dataset and to output the compiled storage formats
    project_base_directory = str(Path(__file__).absolute().parents[3] / 'datasets')

    # CSV Benchmark File
    csv_file_path = os.path.join(project_base_directory, 'benchmark_results.csv')

    # Retrieve the dynamically resolved absolute paths for the original data and generated formats
    paths_dictionary = get_dataset_paths(project_base_directory=project_base_directory, image_size=image_size,
                                         keep_ratio=keep_ratio)

    # Validate the existence of the requisite TFRecord archives, dynamically regenerating them if they are missing to guarantee smooth execution
    prepare_tfrecords(paths_dictionary=paths_dictionary, image_size=image_size, keep_ratio=keep_ratio)

    # Map each dataloader strategy to its specific execution function and arguments to enable a clean, dynamic testing loop
    benchmark_argument_dictionary = {
        'Native': {
            'function': benchmark_native,
            'arguments': {
                'data_directory': paths_dictionary["data_directory"],
                'labels_file': paths_dictionary["labels_file"],
                'number_of_runs': number_of_runs,
                'batch_size': batch_size,
                'image_size': image_size,
                'keep_ratio': keep_ratio,
                'view_images': view_images
            }
        },
        'TFRecord': {
            'function': benchmark_tfrecord,
            'arguments': {
                'tensorflow_record_path': paths_dictionary["tensorflow_record_path"],
                'number_of_runs': number_of_runs,
                'batch_size': batch_size,
                'view_images': view_images
            }
        },
        'Sharded': {
            'function': benchmark_tfrecord_sharded,
            'arguments': {
                'directory_pattern': paths_dictionary["tfrecord_sharded_pattern"],
                'number_of_runs': number_of_runs,
                'batch_size': batch_size,
                'view_images': view_images
            }
        }
    }

    # Initialize a tracking dictionary tailored precisely to the active methods to store the execution times of each epoch
    results = {key: [] for key in methods_to_benchmark}

    # Iterate over the predefined strategies, selectively triggering the underlying method if it is flagged for active testing
    for method_name, method_details in benchmark_argument_dictionary.items():
        if method_name in methods_to_benchmark:
            results[method_name] = benchmark_method(
                method_name=method_name,
                method_function=method_details['function'],
                benchmark_arguments=method_details['arguments'])
            print("-" * 50)

    # Dispatch the accumulated execution results to be formalized and persisted into a CSV file
    save_benchmark_results(csv_file_path=csv_file_path, methods_to_benchmark=methods_to_benchmark, results=results, number_of_runs=number_of_runs)

if __name__ == "__main__":
    main()
