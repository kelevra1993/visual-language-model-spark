import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import time
import cv2
import numpy
import torch
import tensorflow
from tqdm import tqdm
from typing import List, Dict, Any
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

    def __init__(self, data_directory: str, labels_file: str, image_size: int, keep_ratio: bool):
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
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

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
        image_tensor = torch.from_numpy(numpy.array(object=resized_image)).permute(2, 0, 1).float() / 255.0

        return {"images": image_tensor,
                "bounding_boxes": torch.tensor(data=resized_bounding_boxes, dtype=torch.float32),
                "labels": torch.tensor(data=labels, dtype=torch.int64)}


def benchmark_native(data_directory: str, labels_file: str, number_of_runs: int, batch_size: int, image_size: int,
                     keep_ratio: bool, view_images: bool) -> float:
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

    return sum(times) / number_of_runs


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

    def __init__(self, tensorflow_record_path: str):
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


def benchmark_tfrecord(tensorflow_record_path: str, number_of_runs: int, batch_size: int, view_images: bool) -> float:
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

    return sum(times) / number_of_runs


def main() -> None:
    """
    Executes the comprehensive Dataloader Format Benchmarking suite.

    This overarching pipeline orchestrates the sequential testing of various data storage
    strategies to empirically determine the optimal throughput architecture for the downstream trainer.

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

    tensorflow_record_path = os.path.join('/home/robert_kelevra/Projects/visual-language-model-spark/datasets/formats',
                                          'coco-train-preresized.tfrecord')

    # if not os.path.exists(path=tensorflow_record_path):
    #     print(f"TFRecord file not found at {tensorflow_record_path}. Generating it now...")
    #     os.makedirs(name=os.path.dirname(p=tensorflow_record_path), exist_ok=True)
    #     create_tfrecord(data_directory=original_data_directory, labels_file=original_labels, output_tensorflow_record=tensorflow_record_path, image_size=image_size, keep_ratio=keep_ratio)
    #     print("TFRecord generation complete!")
    # else:
    #     print(f"Found existing TFRecord file at {tensorflow_record_path}.")

    print("Synchronizing dataset annotations with physical disk files...")

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

    # try:
    #     print("Benchmarking TFRecord (Pre-Resized)...")
    #     average_time = benchmark_tfrecord(tensorflow_record_path=tensorflow_record_path, number_of_runs=number_of_runs,
    #                                       batch_size=batch_size, view_images=view_images)
    #     print(f"[TFRecord (Pre-Resized)] Average Time: {average_time:.4f} s")
    # except Exception as error:
    #     print(f"[TFRecord (Pre-Resized)] Skipped due to error: {error}")
    # print("-" * 50)


if __name__ == "__main__":
    main()
