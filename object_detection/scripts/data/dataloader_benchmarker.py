import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import cv2
import numpy as np
import csv
import time
import torch
import tensorflow
from tqdm import tqdm
import glob

from utilities.os_utilities import print_blue
from utilities.data_utilities import view_input_data

from pathlib import Path
from typing import List, Dict, Any, Callable
from torch.utils.data import Dataset, DataLoader, IterableDataset

from data.data_loader import NativeDataset, TFRecordDataset, TFRecordShardedDataset, collate_function, \
    create_tfrecord, create_tfrecord_sharded, dali_pipeline, DALIDataloaderWrapper

from nvidia.dali.plugin.pytorch import DALIGenericIterator

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


def clean_data(data_directory: str, labels_file: str) -> None:
    """
    Sanitizes the dataset by removing images without annotations and annotations without images.

    This ensures that the dataloader does not crash when encountering missing files
    or empty labels during the training loop. It writes the filtered output back
    to the original JSON file path.

    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        labels_file (str): The absolute path to the JSON file containing the annotations.

    Returns:
        None
    """
    print(f"Loading annotations from {labels_file}...")
    with open(file=labels_file, mode='r') as file_handler:
        data = json.load(fp=file_handler)

    print("Checking for missing images...")
    filtered_images = []
    for image in tqdm(iterable=data['images'], desc="Filtering Images", leave=False):
        image_path = os.path.join(data_directory, image['file_name'])
        if os.path.exists(path=image_path):
            filtered_images.append(image)

    valid_image_ids = {image['id'] for image in filtered_images}

    print("Filtering annotations for valid images...")
    filtered_annotations = []
    for annotation in tqdm(iterable=data['annotations'], desc="Filtering Annotations", leave=False):
        if annotation['image_id'] in valid_image_ids:
            filtered_annotations.append(annotation)

    data['images'] = filtered_images
    data['annotations'] = filtered_annotations

    print(f"Writing cleaned annotations back to {labels_file}...")
    with open(file=labels_file, mode='w') as file_handler:
        json.dump(obj=data, fp=file_handler, indent=4)
    print("Data cleaning complete.")


def benchmark_native(data_directory: str, labels_file: str, number_of_runs: int, batch_size: int, image_size: int,
                     keep_ratio: bool, view_images: bool) -> List[float]:
    """
    Benchmarks the Native PyTorch dataset loader.

    This function evaluates the baseline throughput of CPU decoding by loading images from the disk
    and scaling them preemptively in the data pipeline.

    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        labels_file (str): The absolute path to the JSON file containing the annotations.
        number_of_runs (int): The number of full epoch passes to simulate.
        batch_size (int): The number of images per batch.
        image_size (int): The target height and width for the scaled images.
        keep_ratio (bool): Whether to maintain the aspect ratio during scaling by padding.
        view_images (bool): Whether to visualize the batches using OpenCV.

    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = NativeDataset(data_directory=data_directory, labels_file=labels_file, image_size=image_size,
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


def benchmark_tfrecord(tensorflow_record_path: str, number_of_runs: int, batch_size: int, view_images: bool,
                       buffer_size: int) -> List[float]:
    """
    Benchmarks the Pre-Resized TFRecord dataset loader.
    
    This function evaluates the absolute theoretical maximum throughput of CPU decoding by 
    combining binary archive storage with preemptive spatial scaling in the benchmark pipeline.
    
    Args:
        tensorflow_record_path (str): The absolute path to the pre-resized TFRecord archive file.
        number_of_runs (int): The number of full epoch passes to simulate.
        batch_size (int): The number of images per batch.
        view_images (bool): Whether to visualize the batches using OpenCV.
        buffer_size (int): The buffer size for TFRecord loading.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = TFRecordDataset(tensorflow_record_path=tensorflow_record_path, buffer_size=buffer_size)
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


def benchmark_tfrecord_sharded(directory_pattern: str, number_of_runs: int, batch_size: int, view_images: bool,
                               buffer_size: int) -> List[float]:
    """
    Benchmarks the sharded TFRecord dataset loader.
    
    Args:
        directory_pattern (str): The wildcard pattern used to locate all TFRecord shards.
        number_of_runs (int): The number of full epoch passes to simulate.
        batch_size (int): The number of images per batch.
        view_images (bool): Whether to visualize the batches using OpenCV.
        buffer_size (int): The buffer size for TFRecord loading.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = TFRecordShardedDataset(directory_pattern=directory_pattern, buffer_size=buffer_size)
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


def compare_dali_and_native(dali_image_tensor: torch.Tensor, dali_boxes: torch.Tensor, dali_labels: torch.Tensor,
                            native_image_tensor: torch.Tensor, native_boxes: torch.Tensor, native_labels: torch.Tensor) -> bool:
    """
    Serves as a visual debugging utility within the dataloader benchmarking suite to evaluate precision alignment.
    
    This function computes and visualizes the pixel-level absolute difference between the hardware-accelerated 
    DALI data pipeline and the Native Python preprocessing pipeline. It ensures that the generated image tensors 
    fed into the object detection model during training are mathematically equivalent regardless of the active dataloader.
    
    Args:
        dali_image_tensor (torch.Tensor): The image tensor processed by DALI.
        dali_boxes (torch.Tensor): The bounding box tensor from DALI (padded).
        dali_labels (torch.Tensor): The labels from DALI (padded).
        native_image_tensor (torch.Tensor): The image tensor processed by Native Python.
        native_boxes (torch.Tensor): The bounding box tensor from Native Python.
        native_labels (torch.Tensor): The labels from Native Python.
        
    Returns:
        bool: True if the user pressed ENTER to quit, False otherwise.
    """

    dali_image = dali_image_tensor.permute(1, 2, 0).numpy().copy().astype(np.uint8)
    native_image = native_image_tensor.permute(1, 2, 0).numpy().copy().astype(np.uint8)

    # Compute absolute difference before drawing boxes
    raw_difference = cv2.absdiff(src1=dali_image, src2=native_image)
    average_pixel_difference = np.mean(raw_difference)

    # Enhance difference visibility (optional but helpful)
    difference_image = cv2.convertScaleAbs(src=raw_difference, alpha=10.0)

    # Add average difference text directly onto the difference image
    cv2.putText(img=difference_image, text=f"Avg Diff: {average_pixel_difference:.4f}",
                org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(0, 255, 0), thickness=2)

    # Draw boxes on DALI
    for box, label in zip(dali_boxes.numpy(), dali_labels.numpy()):
        box_x1, box_y1, box_x2, box_y2 = map(int, box)
        if box_x1 != -1:
            cv2.rectangle(img=dali_image, pt1=(box_x1, box_y1), pt2=(box_x2, box_y2), color=(0, 255, 0), thickness=2)
            cv2.putText(img=dali_image, text=str(label), org=(box_x1, max(box_y1 - 10, 0)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)

    # Draw boxes on Native
    for box, label in zip(native_boxes.numpy(), native_labels.numpy()):
        box_x1, box_y1, box_x2, box_y2 = map(int, box)
        if box_x1 != -1:
            cv2.rectangle(img=native_image, pt1=(box_x1, box_y1), pt2=(box_x2, box_y2), color=(0, 255, 0), thickness=2)
            cv2.putText(img=native_image, text=str(label), org=(box_x1, max(box_y1 - 10, 0)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)

    # Create a 3-channel version of raw difference so it can be stacked if it was single channel
    # (cv2.absdiff on 3-channel BGR returns 3-channel BGR, so it's already 3-channel, but just to be safe)

    # Add titles directly to each frame before stacking
    cv2.putText(img=raw_difference, text="Raw Difference", org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=(0, 255, 0), thickness=2)
    cv2.putText(img=difference_image, text="Scaled Difference (x10)", org=(10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=(0, 255, 0), thickness=2)

    # Stack images horizontally
    combined_image = np.hstack((dali_image, native_image, raw_difference, difference_image))

    cv2.putText(img=combined_image, text="DALI | Native | Raw Diff | Scaled Diff - SPACE for next, ENTER to quit",
                org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=(0, 0, 255), thickness=2)

    cv2.imshow(winname="DALI vs Native Comparison", mat=combined_image)

    while True:
        pressed_key = cv2.waitKey(delay=0) & 0xFF
        if pressed_key == 13:  # 13 represents the Enter/Return key
            cv2.destroyAllWindows()
            return True
        if pressed_key == ord(' '):
            return False


def benchmark_dali(data_directory: str, labels_file: str, number_of_runs: int, batch_size: int, image_size: int,
                   keep_ratio: bool, view_images: bool) -> List[float]:
    """
    Benchmarks the NVIDIA DALI hardware-accelerated dataloader.
    
    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        labels_file (str): The absolute path to the JSON file containing the annotations.
        number_of_runs (int): The number of full epoch passes to simulate.
        batch_size (int): The number of images per batch.
        image_size (int): The target height and width for the scaled images.
        keep_ratio (bool): Whether to maintain the aspect ratio during scaling by padding.
        view_images (bool): Whether to visualize the batches using OpenCV.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    data_pipeline = dali_pipeline(data_directory=data_directory, annotations_file=labels_file,
                                  image_size=image_size, keep_ratio=keep_ratio,
                                  batch_size=batch_size, num_threads=4, device_id=0)
    data_pipeline.build()
    dali_iterator = DALIGenericIterator([data_pipeline],
                                        ['images', 'bounding_boxes', 'labels', 'shapes', 'image_ids'],
                                        reader_name="Reader",
                                        auto_reset=True)
    dataloader = DALIDataloaderWrapper(dali_iterator=dali_iterator, image_size=image_size, keep_ratio=keep_ratio)

    if view_images:
        native_dataset = NativeDataset(data_directory=data_directory, labels_file=labels_file,
                                       image_size=image_size, keep_ratio=keep_ratio)
        image_id_to_index = {img['id']: idx for idx, img in enumerate(native_dataset.images)}

    times = []

    for _run_index in range(number_of_runs):
        start_time = time.time()
        for batch_data_dictionary in tqdm(iterable=dataloader, desc=f"DALI Epoch {_run_index + 1}", leave=False):
            if view_images:
                for batch_index in range(batch_data_dictionary["images"].size(0)):
                    dali_image_tensor = batch_data_dictionary["images"][batch_index].cpu()
                    dali_boxes = batch_data_dictionary["bounding_boxes"][batch_index].cpu()
                    dali_labels = batch_data_dictionary["labels"][batch_index].cpu()
                    image_id = batch_data_dictionary["image_ids"][batch_index].item()

                    native_data = native_dataset[image_id_to_index[image_id]]
                    native_image_tensor = native_data["images"].cpu()
                    native_boxes = native_data["bounding_boxes"].cpu()
                    native_labels = native_data["labels"].cpu()

                    user_quit = compare_dali_and_native(dali_image_tensor=dali_image_tensor,
                                                        dali_boxes=dali_boxes,
                                                        dali_labels=dali_labels,
                                                        native_image_tensor=native_image_tensor,
                                                        native_boxes=native_boxes,
                                                        native_labels=native_labels)

                    if user_quit:
                        return 0.0
        times.append(time.time() - start_time)

    return times


def get_dataset_paths(project_base_directory: str, image_size: int, keep_ratio: bool) -> Dict[str, str]:
    """
    Constructs and orchestrates the standardized filesystem paths required by the dataloader format benchmarker
     pipeline to locate annotations and output compiled archives.
    
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
    Validates the presence of the pre-resized and sharded TFRecord archives on disk, dynamically compiling them from
     the raw dataset if missing, in order to guarantee data availability for the benchmarking pipeline.
    
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


def save_benchmark_results(csv_file_path: str, methods_to_benchmark: List[str], results: Dict[str, List[float]],
                           number_of_runs: int) -> None:
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

        # Compute and append the final average throughput times
        # across all successful runs to conclude the benchmark report
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
    Serves as the primary entry point for the dataloader benchmarking suite.
    
    This function orchestrates the sequential evaluation of different data ingestion strategies 
    (Native PyTorch vs. TFRecord variants) to experimentally determine the optimal data pipeline 
    configuration for the object detection model. By executing simulated epoch loops and aggregating 
    the wall-clock times, it provides the empirical metrics necessary to configure the downstream 
    training loop with the fastest possible dataloader.
    """
    # Core execution configuration
    number_of_runs = 2
    batch_size = 20
    image_size = 1024
    keep_ratio = True
    view_images = False
    buffer_size = 262144  # 256 MB

    # Specify the dataloader architectures that should actively be executed during the current benchmark run
    methods_to_benchmark = [
        # 'Native',
        'TFRecord',
        'Sharded',
        'DALI'
    ]

    # Construct the absolute system paths required
    # to locate the underlying dataset and to output the compiled storage formats
    project_base_directory = str(Path(__file__).absolute().parents[3] / 'datasets')

    # CSV Benchmark File
    csv_file_path = os.path.join(project_base_directory, 'benchmark_results.csv')

    # Retrieve the dynamically resolved absolute paths for the original data and generated formats
    paths_dictionary = get_dataset_paths(project_base_directory=project_base_directory, image_size=image_size,
                                         keep_ratio=keep_ratio)

    # Validate the existence of the requisite TFRecord archives,
    # dynamically regenerating them if they are missing to guarantee smooth execution
    prepare_tfrecords(paths_dictionary=paths_dictionary, image_size=image_size, keep_ratio=keep_ratio)

    # Map each dataloader strategy to its specific execution function and arguments
    # to enable a clean, dynamic testing loop
    benchmark_argument_dictionary = {
        'Native': {'function': benchmark_native,
                   'arguments': {'data_directory': paths_dictionary["data_directory"],
                                 'labels_file': paths_dictionary["labels_file"], 'number_of_runs': number_of_runs,
                                 'batch_size': batch_size, 'image_size': image_size, 'keep_ratio': keep_ratio,
                                 'view_images': view_images}},
        'TFRecord': {'function': benchmark_tfrecord,
                     'arguments': {'tensorflow_record_path': paths_dictionary["tensorflow_record_path"],
                                   'number_of_runs': number_of_runs, 'batch_size': batch_size,
                                   'view_images': view_images,
                                   'buffer_size': buffer_size}},
        'Sharded': {'function': benchmark_tfrecord_sharded,
                    'arguments': {'directory_pattern': paths_dictionary["tfrecord_sharded_pattern"],
                                  'number_of_runs': number_of_runs, 'batch_size': batch_size,
                                  'view_images': view_images,
                                  'buffer_size': buffer_size}},
        'DALI': {'function': benchmark_dali,
                 'arguments': {'data_directory': paths_dictionary["data_directory"],
                               'labels_file': paths_dictionary["labels_file"], 'number_of_runs': number_of_runs,
                               'batch_size': batch_size, 'image_size': image_size, 'keep_ratio': keep_ratio,
                               'view_images': view_images}}
    }

    # Initialize a tracking dictionary tailored precisely
    # to the active methods to store the execution times of each epoch
    results = {key: [] for key in methods_to_benchmark}

    # Iterate over the predefined strategies, selectively
    # triggering the underlying method if it is flagged for active testing
    for method_name, method_details in benchmark_argument_dictionary.items():
        if method_name in methods_to_benchmark:
            results[method_name] = benchmark_method(method_name=method_name, method_function=method_details['function'],
                                                    benchmark_arguments=method_details['arguments'])
            print("-" * 50)

    # Dispatch the accumulated execution results to be formalized and persisted into a CSV file
    save_benchmark_results(csv_file_path=csv_file_path, methods_to_benchmark=methods_to_benchmark, results=results,
                           number_of_runs=number_of_runs)


if __name__ == "__main__":
    main()
