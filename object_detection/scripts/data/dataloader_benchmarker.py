import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import csv
import time
import torch
import tensorflow
from tqdm import tqdm
import glob

from utilities.os_utilities import print_blue

from pathlib import Path
from typing import List, Dict, Any, Callable
from torch.utils.data import Dataset, DataLoader, IterableDataset

from data.data_loader import NativeCocoDataset, TFRecordCocoDataset, TFRecordShardedCocoDataset, collate_function, create_tfrecord, create_tfrecord_sharded, view_input_data

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


def benchmark_native(data_directory: str, labels_file: str, number_of_runs: int, batch_size: int, image_size: int,
                     keep_ratio: bool, view_images: bool, device: torch.device, dtype: torch.dtype) -> List[float]:
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
                                keep_ratio=keep_ratio, device=device, dtype=dtype)
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
                       device: torch.device, dtype: torch.dtype, buffer_size: int) -> List[float]:
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
    dataset = TFRecordCocoDataset(tensorflow_record_path=tensorflow_record_path, device=device, dtype=dtype,
                                  buffer_size=buffer_size)
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
                               device: torch.device, dtype: torch.dtype, buffer_size: int) -> List[float]:
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
    dataset = TFRecordShardedCocoDataset(directory_pattern=directory_pattern, device=device, dtype=dtype,
                                         buffer_size=buffer_size)
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
     the raw COCO dataset if missing, in order to guarantee data availability for the benchmarking pipeline.
    
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
    # Core execution configuration
    number_of_runs = 1
    batch_size = 20
    image_size = 1024
    keep_ratio = True
    view_images = True
    buffer_size = 262144

    device = torch.device('cpu')
    dtype = torch.float32

    # Specify the dataloader architectures that should actively be executed during the current benchmark run
    methods_to_benchmark = [
        'Native',
        'TFRecord',
        'Sharded'
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
                                 'view_images': view_images, 'device': device, 'dtype': dtype}},
        'TFRecord': {'function': benchmark_tfrecord,
                     'arguments': {'tensorflow_record_path': paths_dictionary["tensorflow_record_path"],
                                   'number_of_runs': number_of_runs, 'batch_size': batch_size,
                                   'view_images': view_images, 'device': device, 'dtype': dtype,
                                   'buffer_size': buffer_size}},
        'Sharded': {'function': benchmark_tfrecord_sharded,
                    'arguments': {'directory_pattern': paths_dictionary["tfrecord_sharded_pattern"],
                                  'number_of_runs': number_of_runs, 'batch_size': batch_size,
                                  'view_images': view_images, 'device': device, 'dtype': dtype,
                                  'buffer_size': buffer_size}}}

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
