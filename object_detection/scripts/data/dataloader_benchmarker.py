import os

import json
import cv2
import numpy as np
import csv
import time
import torch
from tqdm import tqdm

from utilities.os_utilities import print_blue
from utilities.data_utilities import view_input_data

from pathlib import Path
from typing import List, Dict, Any, Callable
from torch.utils.data import DataLoader

from data.data_loader import NativeDataset, dali_pipeline, DALIDataloaderWrapper

from nvidia.dali.plugin.pytorch import DALIGenericIterator


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
                     keep_ratio: bool, label_mode: str, label_dictionary: Dict[str, Any],
                     view_images: bool) -> List[float]:
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
        label_mode (str): Mode for class assignment ("binary" or "multi_class").
        label_dictionary (Dict[str, Any]): Dictionary mapping raw dataset classes to model classes.
        view_images (bool): Whether to visualize the batches using OpenCV.

    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = NativeDataset(data_directory=data_directory, labels_file=labels_file, image_size=image_size,
                            keep_ratio=keep_ratio, label_mode=label_mode, label_dictionary=label_dictionary)
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
                    user_quit = view_input_data(images=[batch_data_dictionary["images"][batch_index]],
                                                bounding_boxes=[batch_data_dictionary["bounding_boxes"][batch_index]],
                                                labels=[batch_data_dictionary["labels"][batch_index]],
                                                window_titles=["Input Data Viewer"],
                                                display_input_size=image_size)
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
                   keep_ratio: bool, label_mode: str, label_dictionary: Dict[str, Any],
                   view_images: bool) -> List[float]:
    """
    Benchmarks the NVIDIA DALI hardware-accelerated dataloader.
    
    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        labels_file (str): The absolute path to the JSON file containing the annotations.
        number_of_runs (int): The number of full epoch passes to simulate.
        batch_size (int): The number of images per batch.
        image_size (int): The target height and width for the scaled images.
        keep_ratio (bool): Whether to maintain the aspect ratio during scaling by padding.
        label_mode (str): Mode for class assignment ("binary" or "multi_class").
        label_dictionary (Dict[str, Any]): Dictionary mapping raw dataset classes to model classes.
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
    dataloader = DALIDataloaderWrapper(dali_iterator=dali_iterator, image_size=image_size, keep_ratio=keep_ratio,
                                       label_mode=label_mode, label_dictionary=label_dictionary)

    if view_images:
        native_dataset = NativeDataset(data_directory=data_directory, labels_file=labels_file,
                                       image_size=image_size, keep_ratio=keep_ratio, label_mode=label_mode,
                                       label_dictionary=label_dictionary, shuffle=False)
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


def get_dataset_paths(project_base_directory: str) -> Dict[str, str]:
    """
    Constructs the standardized filesystem paths the dataloader benchmarker uses to locate the COCO training
    images and annotations shared by the Native and DALI loaders.

    Args:
        project_base_directory (str): The root datasets directory of the project.

    Returns:
        Dict[str, str]: A dictionary containing absolute paths for the image directory and the labels file.
    """
    base_directory = os.path.join(project_base_directory, 'coco-2017', 'train')
    original_data_directory = os.path.join(base_directory, 'data')
    original_labels = os.path.join(base_directory, 'labels.json')

    return {"data_directory": original_data_directory, "labels_file": original_labels}


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
    (Native PyTorch vs. DALI) to experimentally determine the optimal data pipeline 
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

    # Label mapping shared by both loaders, matching the binary mode of the configuration loader
    label_mode = "binary"
    label_dictionary = {"background": 0, "foreground": 1}

    # Specify the dataloader architectures that should actively be executed during the current benchmark run
    methods_to_benchmark = [
        'Native',
        'DALI'
    ]

    # Construct the absolute system path required to locate the underlying dataset
    project_base_directory = str(Path(__file__).absolute().parents[3] / 'datasets')

    # CSV Benchmark File
    csv_file_path = os.path.join(project_base_directory, 'benchmark_results.csv')

    # Retrieve the dynamically resolved absolute paths for the original data
    paths_dictionary = get_dataset_paths(project_base_directory=project_base_directory)

    # Map each dataloader strategy to its specific execution function and arguments
    # to enable a clean, dynamic testing loop
    benchmark_argument_dictionary = {
        'Native': {'function': benchmark_native,
                   'arguments': {'data_directory': paths_dictionary["data_directory"],
                                 'labels_file': paths_dictionary["labels_file"], 'number_of_runs': number_of_runs,
                                 'batch_size': batch_size, 'image_size': image_size, 'keep_ratio': keep_ratio,
                                 'label_mode': label_mode, 'label_dictionary': label_dictionary,
                                 'view_images': view_images}},
        'DALI': {'function': benchmark_dali,
                 'arguments': {'data_directory': paths_dictionary["data_directory"],
                               'labels_file': paths_dictionary["labels_file"], 'number_of_runs': number_of_runs,
                               'batch_size': batch_size, 'image_size': image_size, 'keep_ratio': keep_ratio,
                               'label_mode': label_mode, 'label_dictionary': label_dictionary,
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
