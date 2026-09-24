"""
Comprehensive Dataloader Format Benchmarking Suite

This script focuses solely on benchmarking the Native PyTorch dataloader.
It reads raw JPEG images directly from the filesystem and parses a monolithic 
JSON annotations file. Highly bottlenecked by disk I/O, this serves as the 
baseline for future optimizations.
"""

import os
import shutil
from datetime import datetime
import time
import json
import torch
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
from tqdm import tqdm
from typing import Tuple, List, Dict, Any
from utilities.data_utilities import preprocess_image_and_boxes, view_input_data

import shutil
from datetime import datetime


def clean_data(data_directory: str, labels_file: str) -> None:
    """
    Synchronizes the COCO annotations file with the physical images present on disk.

    This function scans the image directory for existing files, backs up the original
    JSON annotations file with a timestamp, and actively filters out any images and
    annotations from the JSON that do not have a corresponding physical file.

    Args:
        data_directory (str): The path to the directory containing raw JPEG images.
        labels_file (str): The path to the JSON file containing COCO-formatted annotations.
    """
    # Scan the physical directory to build a rapid-access set of all existing image filenames
    existing_files = set(os.listdir(data_directory))

    # Generate a timestamped backup file path to ensure no original data is permanently lost
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{labels_file}.{timestamp}.backup"

    # Copy the original file to the backup location
    shutil.copy2(src=labels_file, dst=backup_file)
    print(f"Created a backup of the original labels at: {backup_file}")

    # Load the monolithic JSON file completely into memory to parse all annotations
    with open(file=labels_file, mode='r') as file_handler:
        coco_data = json.load(fp=file_handler)

    # Filter the images list to explicitly retain only those present in the physical directory
    filtered_images = []
    for image in tqdm(coco_data['images'], desc="Filtering Images", leave=False):
        if image['file_name'] in existing_files:
            filtered_images.append(image)

    valid_image_ids = {image['id'] for image in filtered_images}

    # Filter the annotations to explicitly retain only those linked to the valid physical images
    filtered_annotations = []
    for annotation in tqdm(coco_data['annotations'], desc="Filtering Annotations", leave=False):
        if annotation['image_id'] in valid_image_ids:
            filtered_annotations.append(annotation)

    # Mutate the original dictionary with the filtered lists and write back to the filesystem
    coco_data['images'] = filtered_images
    coco_data['annotations'] = filtered_annotations

    with open(file=labels_file, mode='w') as file_handler:
        json.dump(obj=coco_data, fp=file_handler, indent=4)

    print(f"Cleaned labels.json: Retained {len(filtered_images)} images and {len(filtered_annotations)} annotations.")


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
    # Extract the individual components from the batched dictionaries
    images = [item["images"] for item in batch]
    bounding_boxes = [item["bounding_boxes"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Consolidate the image tensors into a single unified batch tensor for GPU acceleration
    # Keep bounding boxes and labels as standard lists to accommodate variable object counts per image
    # Return as a structured dictionary for clean downstream consumption
    return {
        "images": torch.stack(tensors=images),
        "bounding_boxes": bounding_boxes,
        "labels": labels
    }


# --- 1. Native PyTorch ---
class NativeCocoDataset(Dataset):
    """
    A native PyTorch Dataset implementation for reading COCO-format datasets from raw JPEGs.

    This dataset serves as the baseline in the data loading benchmark pipeline. It reads raw
    image files directly from the disk filesystem and parses a monolithic JSON annotations file,
    which is highly representative of standard, unoptimized data loading bottlenecks.
    """

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool):
        """
        Initializes the native COCO dataset.

        Args:
            data_directory (str): The path to the directory containing raw JPEG images.
            labels_file (str): The path to the JSON file containing COCO-formatted annotations.
        """
        self.data_directory = data_directory
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        # Load the monolithic JSON file completely into memory to parse all annotations
        with open(file=labels_file, mode='r') as file_handler:
            coco_data = json.load(fp=file_handler)

        # Build a rapid lookup dictionary for images using their unique identifier as the key
        self.images = {image['id']: image for image in coco_data['images']}
        self.image_identifiers = list(self.images.keys())

        # Map each image identifier to its corresponding list of annotations for immediate retrieval during the getitem call
        self.image_to_annotations = {}
        for annotation in coco_data['annotations']:
            image_identifier = annotation['image_id']
            if image_identifier not in self.image_to_annotations:
                self.image_to_annotations[image_identifier] = []
            self.image_to_annotations[image_identifier].append(annotation)

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total count of image identifiers.
        """
        return len(self.image_identifiers)

    def __getitem__(self, _index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves and processes a single image and its corresponding annotations.

        Args:
            _index (int): The index of the image to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the resized image tensor,
                                     bounding boxes tensor, and labels tensor.
        """
        image_identifier = self.image_identifiers[_index]
        image_information = self.images[image_identifier]
        image_path = os.path.join(self.data_directory, image_information['file_name'])

        # Read the raw bytes from the disk filesystem and decode them into a BGR NumPy array using OpenCV
        image_cv2 = cv2.imread(filename=image_path)

        # Retrieve all bounding boxes for the current image and convert them from COCO format to the required coordinate format
        annotations = self.image_to_annotations.get(image_identifier, [])
        bounding_boxes = []
        for annotation in annotations:
            box_x, box_y, box_w, box_h = annotation['bbox']
            bounding_boxes.append([box_x, box_y, box_x + box_w, box_y + box_h])
        labels = [annotation['category_id'] for annotation in annotations]

        # Scale the image and pad it to the target uniform size while adjusting the bounding box coordinates to match
        image_resized, resized_bounding_boxes = preprocess_image_and_boxes(image=image_cv2,
                                                                           bounding_boxes=bounding_boxes,
                                                                           image_size=self.image_size,
                                                                           keep_ratio=self.keep_ratio)
        # Cast the preprocessed NumPy array into a contiguous PyTorch tensor and transpose
        # the axes to the expected channel-first format without altering pixel scales
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous()

        # Return the parsed data as a structured dictionary to match downstream interface requirements
        return {"images": image_tensor,
                "bounding_boxes": torch.tensor(data=resized_bounding_boxes),
                "labels": torch.tensor(data=labels)}


def benchmark_native(data_directory: str, labels_file: str, number_of_runs: int, batch_size: int, image_size: int,
                     keep_ratio: bool, view_images: bool) -> float:
    """
    Benchmarks the Native PyTorch dataset loader.

    This function evaluates the performance of the NativeCocoDataset within the data loading
    benchmark pipeline, simulating real-world iterative epoch loops.

    Args:
        data_directory (str): The path to the directory containing raw JPEG images.
        labels_file (str): The path to the JSON file containing COCO-formatted annotations.
        number_of_runs (int, optional): The number of full epoch passes to simulate. Defaults to 10.

    Returns:
        float: The average time taken per run in seconds.
    """
    # Initialize the dataset and wrap it in a multi-processed dataloader to simulate real-world parallel fetching
    dataset = NativeCocoDataset(data_directory=data_directory, labels_file=labels_file, image_size=image_size,
                                keep_ratio=keep_ratio)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_function)
    times = []
    # Iteratively drain the dataloader to precisely measure the wall-clock time required for full epoch traversals
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for batch_data_dictionary in tqdm(dataloader, desc=f"Running Epoch {_run_index + 1}", leave=False):
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
    return sum(times) / number_of_runs


def create_tfrecord(data_directory: str, labels_file: str, output_tfrecord: str, image_size: int, keep_ratio: bool) -> None:
    """
    Parses the COCO dataset, resizes the images and bounding boxes, and writes 
    them into a pure-Python TFRecord archive.
    
    This function acts as the preprocessing step for the TFRecord benchmark pipeline,
    ensuring that the visual-language model downstream trainer has access to pre-scaled
    images for maximum I/O throughput.
    
    Args:
        data_directory (str): The path to the directory containing raw JPEG images.
        labels_file (str): The path to the JSON file containing COCO-formatted annotations.
        output_tfrecord (str): The destination path where the TFRecord file will be saved.
        image_size (int): The target uniform image size for resizing.
        keep_ratio (bool): Whether to maintain aspect ratio with padding during resizing.
    """
    print(f"Loading annotations from {labels_file}...")
    with open(file=labels_file, mode='r') as file_handler:
        coco_data = json.load(fp=file_handler)
        
    images_dictionary = {image['id']: image for image in tqdm(coco_data['images'], desc="Indexing Images")}
    image_to_annotations = {}
    
    for annotation in tqdm(coco_data['annotations'], desc="Indexing Annotations"):
        image_identifier = annotation['image_id']
        if image_identifier not in image_to_annotations:
            image_to_annotations[image_identifier] = []
        image_to_annotations[image_identifier].append(annotation)
        
    writer = tf.io.TFRecordWriter(output_tfrecord)
    
    for image_identifier, image_information in tqdm(images_dictionary.items(), desc="Creating TFRecord"):
        image_path = os.path.join(data_directory, image_information['file_name'])
        image_cv2 = cv2.imread(filename=image_path)
        
        if image_cv2 is None:
            continue
            
        annotations = image_to_annotations.get(image_identifier, [])
        bounding_boxes = []
        for annotation in annotations:
            box_x, box_y, box_w, box_h = annotation['bbox']
            bounding_boxes.append([box_x, box_y, box_x + box_w, box_y + box_h])
        labels = [annotation['category_id'] for annotation in annotations]
        
        image_resized, resized_bounding_boxes = preprocess_image_and_boxes(
            image=image_cv2,
            bounding_boxes=bounding_boxes,
            image_size=image_size,
            keep_ratio=keep_ratio
        )
        
        success, encoded_image = cv2.imencode(ext='.png', img=image_resized)
        image_bytes = encoded_image.tobytes()

        # to compare with
        # encoded_image_string = cv2.imencode(".png", processed_image)[1].tostring()
        # image_label = os.path.basename(os.path.dirname(image_path))
        # def _bytes_feature(value):
        #     """
        #     :param value: input value of type bytes
        #     :return: a feature
        #     """
        #     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        # # example stored in tf record
        # example = tf.train.Example(
        #     features=tf.train.Features(
        #         feature={
        #             "input_name": _bytes_feature(name_raw),
        #             "input": _bytes_feature(encoded_image_string),
        #             "label": _bytes_feature(label_raw),
        #         }
        #     )
        # )
        
        flattened_boxes = []
        for box in resized_bounding_boxes:
            flattened_boxes.extend(box)
            
        import numpy as np
        boxes_bytes = np.array(flattened_boxes, dtype=np.float32).tobytes()
        labels_bytes = np.array(labels, dtype=np.int64).tobytes()
        
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
            
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "images": _bytes_feature(image_bytes),
                    "bounding_boxes": _bytes_feature(boxes_bytes),
                    "labels": _bytes_feature(labels_bytes),
                }
            )
        )
        writer.write(example.SerializeToString())
        
    writer.close()


class TFRecordCocoDataset(IterableDataset):
    """
    A PyTorch IterableDataset implementation for reading pre-processed TFRecord archives using TensorFlow.
    
    This dataset serves as a high-throughput format in the data loading benchmark pipeline. 
    It streams records directly from the disk using tf.data for maximum efficiency.
    """

    def __init__(self, tfrecord_path: str):
        """
        Initializes the TFRecord iterable dataset.
        
        Args:
            tfrecord_path (str): The absolute path to the TFRecord archive file.
        """
        self.tfrecord_path = tfrecord_path

    def __iter__(self):
        """
        Returns an iterator over the dataset using optimized tf.data pipeline.
        """
        worker_info = torch.utils.data.get_worker_info()
        dataset = tf.data.TFRecordDataset([self.tfrecord_path])
        
        # If running with multiple workers in DataLoader, shard the data to avoid duplicates
        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            
        def read_and_decode(serialized_data):
            features = tf.io.parse_single_example(
                serialized_data,
                features={
                    "images": tf.io.FixedLenFeature([], tf.string),
                    "bounding_boxes": tf.io.FixedLenFeature([], tf.string),
                    "labels": tf.io.FixedLenFeature([], tf.string),
                }
            )
            
            image = tf.io.decode_png(features["images"], channels=3)
            bounding_boxes = tf.io.decode_raw(features["bounding_boxes"], out_type=tf.float32)
            labels = tf.io.decode_raw(features["labels"], out_type=tf.int64)
            
            # Reshape bounding boxes back to [N, 4] format
            bounding_boxes = tf.reshape(bounding_boxes, [-1, 4])
            
            return image, bounding_boxes, labels
            
        dataset = dataset.map(map_func=read_and_decode, num_parallel_calls=tf.data.AUTOTUNE)
        
        for image, bounding_boxes, labels in dataset:
            # Convert TF Tensors -> NumPy -> PyTorch Tensors
            image_numpy = image.numpy()
            boxes_numpy = bounding_boxes.numpy()
            labels_numpy = labels.numpy()
            
            # Reorder channels for PyTorch (H, W, C) -> (C, H, W)
            image_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).contiguous()
            
            yield {
                "images": image_tensor,
                "bounding_boxes": torch.tensor(data=boxes_numpy, dtype=torch.float32),
                "labels": torch.tensor(data=labels_numpy, dtype=torch.int64)
            }


def benchmark_tfrecord(tfrecord_path: str, number_of_runs: int, batch_size: int, view_images: bool) -> float:
    """
    Benchmarks the Pre-Resized TFRecord dataset loader.
    
    This function evaluates the absolute theoretical maximum throughput of CPU decoding by 
    combining binary archive storage with preemptive spatial scaling in the benchmark pipeline.
    
    Args:
        tfrecord_path (str): The absolute path to the pre-resized TFRecord archive file.
        number_of_runs (int): The number of full epoch passes to simulate.
        batch_size (int): The number of images per batch.
        view_images (bool): Whether to visualize the batches using OpenCV.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = TFRecordCocoDataset(tfrecord_path=tfrecord_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_function)
    times = []

    # Iteratively drain the dataloader to precisely measure the wall-clock time required for full epoch traversals
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for batch_data_dictionary in tqdm(dataloader, desc=f"TFRecord Epoch {_run_index + 1}", leave=False):
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

    return sum(times) / number_of_runs


def main() -> None:
    """
    Executes the comprehensive Dataloader Format Benchmarking suite.

    This overarching pipeline orchestrates the sequential testing of various data storage
    strategies (e.g. TFRecords, WebDataset, LMDB) to empirically determine the optimal I/O
    throughput architecture for the visual-language model downstream trainer.

    Returns:
        None
    """
    number_of_runs = 5
    batch_size = 20

    image_size = 1024
    keep_ratio = True
    view_images = False

    base_directory = '/home/robert_kelevra/Projects/visual-language-model-spark/datasets/coco-2017/train'
    original_data_directory = os.path.join(base_directory, 'data')
    original_labels = os.path.join(base_directory, 'labels.json')

    tfrecord_path = os.path.join('/home/robert_kelevra/Projects/visual-language-model-spark/datasets/formats',
                                 'coco-train-preresized.tfrecord')

    if not os.path.exists(path=tfrecord_path):
        print(f"TFRecord file not found at {tfrecord_path}. Generating it now...")
        # Make sure the parent directory exists
        os.makedirs(name=os.path.dirname(p=tfrecord_path), exist_ok=True)
        create_tfrecord(
            data_directory=original_data_directory, 
            labels_file=original_labels, 
            output_tfrecord=tfrecord_path, 
            image_size=image_size, 
            keep_ratio=keep_ratio
        )
        print("TFRecord generation complete!")
    else:
        print(f"Found existing TFRecord file at {tfrecord_path}.")

    print("Synchronizing dataset annotations with physical disk files...")
    # clean_data(data_directory=original_data_directory, labels_file=original_labels)

    print(f"Starting Comprehensive Benchmarks... ({number_of_runs} runs each)")
    print("-" * 50)

    # try:
    #     print(f"Benchmarking Native PyTorch...")
    #     average_time = benchmark_native(data_directory=original_data_directory, labels_file=original_labels,
    #                                     number_of_runs=number_of_runs, batch_size=batch_size,
    #                                     image_size=image_size, keep_ratio=keep_ratio, view_images=view_images)
    #     print(f"[Native PyTorch] Average Time: {average_time:.4f} s")
    # except Exception as error:
    #     print(f"[Native PyTorch] Skipped due to error: {error}")
    # print("-" * 50)

    try:
        print(f"Benchmarking TFRecord (Pre-Resized)...")
        average_time = benchmark_tfrecord(tfrecord_path=tfrecord_path, number_of_runs=number_of_runs,
                                          batch_size=batch_size, view_images=view_images)
        print(f"[TFRecord (Pre-Resized)] Average Time: {average_time:.4f} s")
    except Exception as error:
        print(f"[TFRecord (Pre-Resized)] Skipped due to error: {error}")
    print("-" * 50)


if __name__ == "__main__":
    main()
