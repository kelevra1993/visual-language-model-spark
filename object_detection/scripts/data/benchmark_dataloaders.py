import os
import time
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import webdataset as wds
import tfrecord
import torchvision.transforms as T
from PIL import Image
import io
import lmdb
import pickle
import h5py
import csv
import glob
from typing import Tuple, List, Dict, Any

try:
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    pipeline_def = lambda *args, **kwargs: lambda f: f

def resize_image_and_boxes(image_pil: Image.Image, bounding_boxes: List[List[float]], target_size: Tuple[int, int] = (1024, 1024)) -> Tuple[Image.Image, List[List[float]]]:
    """
    Resizes a PIL image and its corresponding bounding boxes to a target spatial size.
    
    This function processes the image to fit the expected input dimensions of the object detection 
    pipeline. It computes the independent x and y scaling factors and adjusts all bounding box 
    coordinates accordingly to ensure spatial alignment is preserved for the Region Proposal Network.

    Args:
        image_pil (Image.Image): The original PIL image.
        bounding_boxes (List[List[float]]): A list of bounding boxes in [x_min, y_min, width, height] format.
        target_size (Tuple[int, int], optional): The desired output size (width, height). Defaults to (1024, 1024).

    Returns:
        Tuple[Image.Image, List[List[float]]]: The resized image and the scaled bounding boxes.
    """
    width, height = image_pil.size
    image_resized = image_pil.resize(target_size, Image.BILINEAR)
    scale_x = target_size[0] / width
    scale_y = target_size[1] / height
    
    resized_bounding_boxes = []
    for bounding_box in bounding_boxes:
        box_x, box_y, box_width, box_height = bounding_box
        new_x = box_x * scale_x
        new_y = box_y * scale_y
        new_width = box_width * scale_x
        new_height = box_height * scale_y
        resized_bounding_boxes.append([new_x, new_y, new_width, new_height])
    
    return image_resized, resized_bounding_boxes

def collate_function(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Collates a list of dataset samples into a batched format.
    
    This function is used by the PyTorch DataLoader to aggregate individual data samples 
    into mini-batches. It stacks images into a single tensor for efficient GPU processing, 
    while preserving bounding boxes and labels as lists since they vary in quantity per image.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): A list of tuples containing (image, boxes, labels).

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]: The batched images, list of boxes, and list of labels.
    """
    images = [item[0] for item in batch]
    bounding_boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return torch.stack(tensors=images), bounding_boxes, labels

# --- 1. Native PyTorch ---
class NativeCocoDataset(Dataset):
    """
    A native PyTorch Dataset implementation for reading COCO-format datasets from raw JPEGs.
    
    This dataset serves as the baseline in the data loading benchmark pipeline. It reads raw 
    image files directly from the disk filesystem and parses a monolithic JSON annotations file, 
    which is highly representative of standard, unoptimized data loading bottlenecks.
    """
    def __init__(self, data_directory: str, labels_file: str):
        """
        Initializes the native COCO dataset.
        
        Args:
            data_directory (str): The path to the directory containing raw JPEG images.
            labels_file (str): The path to the JSON file containing COCO-formatted annotations.
        """
        self.data_directory = data_directory
        with open(file=labels_file, mode='r') as file_handler:
            coco_data = json.load(fp=file_handler)
            
        self.images = {image['id']: image for image in coco_data['images']}
        self.image_identifiers = list(self.images.keys())
        
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
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves and processes a single image and its corresponding annotations.
        
        Args:
            _index (int): The index of the image to retrieve.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the resized image tensor, 
                                                             bounding boxes tensor, and labels tensor.
        """
        image_identifier = self.image_identifiers[_index]
        image_information = self.images[image_identifier]
        image_path = os.path.join(self.data_directory, image_information['file_name'])
        
        image_pil = Image.open(fp=image_path).convert(mode='RGB')
        
        annotations = self.image_to_annotations.get(image_identifier, [])
        bounding_boxes = [annotation['bbox'] for annotation in annotations]
        labels = [annotation['category_id'] for annotation in annotations]
        
        image_resized, resized_bounding_boxes = resize_image_and_boxes(image_pil=image_pil, bounding_boxes=bounding_boxes)
        image_tensor = T.ToTensor()(image_resized)
        
        return image_tensor, torch.tensor(data=resized_bounding_boxes), torch.tensor(data=labels)

def benchmark_native(data_directory: str, labels_file: str, number_of_runs: int = 10) -> float:
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
    dataset = NativeCocoDataset(data_directory=data_directory, labels_file=labels_file)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_function)
    times = []
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for _batch in dataloader:
            pass
        times.append(time.time() - start_time)
    return sum(times) / number_of_runs

# --- 2. WebDataset ---
def decode_webdataset(sample: Dict[str, bytes]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decodes a single sample from a WebDataset TAR archive.
    
    This function serves as the mapping stage in the WebDataset pipeline, extracting 
    and parsing the raw JPEG bytes and JSON annotation bytes directly from memory without 
    incurring individual filesystem read overhead.
    
    Args:
        sample (Dict[str, bytes]): A dictionary containing the raw bytes for the image ('jpg') 
                                   and annotations ('json').
                                   
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the resized image tensor, 
                                                         bounding boxes tensor, and labels tensor.
    """
    image_bytes = sample['jpg']
    image_pil = Image.open(fp=io.BytesIO(initial_bytes=image_bytes)).convert(mode='RGB')
    
    json_bytes = sample['json']
    json_data = json.loads(s=json_bytes.decode(encoding='utf-8'))
    bounding_boxes = json_data['bboxes']
    labels = json_data['labels']
    
    image_resized, resized_bounding_boxes = resize_image_and_boxes(image_pil=image_pil, bounding_boxes=bounding_boxes)
    image_tensor = T.ToTensor()(image_resized)
    return image_tensor, torch.tensor(data=resized_bounding_boxes), torch.tensor(data=labels)

def benchmark_webdataset(tar_url: str, number_of_runs: int = 10) -> float:
    """
    Benchmarks the WebDataset tar archive loader.
    
    This function evaluates the performance of sequential TAR archive streaming within the 
    data loading benchmark pipeline.
    
    Args:
        tar_url (str): The POSIX-style shell brace expansion path to the TAR archives.
        number_of_runs (int, optional): The number of full epoch passes to simulate. Defaults to 10.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = wds.WebDataset(urls=tar_url).map(f=decode_webdataset).batched(batchsize=16, collation_fn=collate_function)
    dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=4)
    times = []
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for _batch in dataloader:
            pass
        times.append(time.time() - start_time)
    return sum(times) / number_of_runs

# --- 3. TFRecord (Raw and Sharded) ---
def decode_tfrecord(features: Dict[str, Any], pre_resized: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decodes a single sample from a TFRecord archive.
    
    This function acts as the parsing mechanism for TFRecord datasets in the benchmarking pipeline, 
    efficiently unpacking batched bytes and avoiding file I/O bottlenecks.
    
    Args:
        features (Dict[str, Any]): A dictionary mapping feature names to their extracted raw values.
        pre_resized (bool, optional): If True, skips spatial resizing. Defaults to False.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the image tensor, 
                                                         bounding boxes tensor, and labels tensor.
    """
    image_bytes = features['image']
    bounding_boxes_flat = features['bboxes']
    labels = features['labels']
    
    image_pil = Image.open(fp=io.BytesIO(initial_bytes=image_bytes)).convert(mode='RGB')
    
    bounding_boxes = []
    for index in range(0, len(bounding_boxes_flat), 4):
        bounding_boxes.append(bounding_boxes_flat[index:index+4])
        
    if not pre_resized:
        image_resized, resized_bounding_boxes = resize_image_and_boxes(image_pil=image_pil, bounding_boxes=bounding_boxes)
    else:
        image_resized, resized_bounding_boxes = image_pil, bounding_boxes
        
    image_tensor = T.ToTensor()(image_resized)
    return image_tensor, torch.tensor(data=resized_bounding_boxes), torch.tensor(data=labels)

class CustomTFRecordDataset(Dataset):
    """
    A custom PyTorch Dataset implementation for iterating over a monolithic TFRecord archive.
    
    This class integrates into the benchmark pipeline to evaluate the latency of decoding 
    massive binary records natively in Python.
    """
    def __init__(self, tfrecord_path: str, pre_resized: bool = False):
        """
        Initializes the TFRecord dataset and preloads records.
        
        Args:
            tfrecord_path (str): The absolute path to the TFRecord archive file.
            pre_resized (bool, optional): Indicates if the embedded JPEGs are already sized correctly. Defaults to False.
        """
        self.pre_resized = pre_resized
        self.data_records = []
        iterator = tfrecord.tfrecord_loader(data_path=tfrecord_path, index_path=None, description={
            "image": "byte",
            "bboxes": "float",
            "labels": "int"
        })
        for record in iterator:
            self.data_records.append(record)
            
    def __len__(self) -> int:
        """
        Returns the total number of records loaded from the archive.
        
        Returns:
            int: The total count of loaded data records.
        """
        return len(self.data_records)
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves and decodes a single TFRecord entry.
        
        Args:
            _index (int): The index of the record to decode.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The decoded image, boxes, and labels.
        """
        return decode_tfrecord(features=self.data_records[_index], pre_resized=self.pre_resized)

def benchmark_tfrecord(tfrecord_path: str, number_of_runs: int = 10) -> float:
    """
    Benchmarks the single-file TFRecord dataset loader.
    
    This function evaluates monolithic TFRecord performance within the data loading benchmark pipeline.
    
    Args:
        tfrecord_path (str): The absolute path to the TFRecord archive file.
        number_of_runs (int, optional): The number of full epoch passes to simulate. Defaults to 10.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = CustomTFRecordDataset(tfrecord_path=tfrecord_path, pre_resized=False)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_function)
    times = []
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for _batch in dataloader:
            pass
        times.append(time.time() - start_time)
    return sum(times) / number_of_runs

class CustomShardedTFRecordDataset(Dataset):
    """
    A custom PyTorch Dataset implementation for reading from multiple sharded TFRecord files.
    
    This class supports distributed, highly parallel data storage strategies in the benchmarking 
    pipeline by seamlessly merging sequential shards into a unified virtual dataset.
    """
    def __init__(self, directory_pattern: str, pre_resized: bool = False):
        """
        Initializes the sharded TFRecord dataset by traversing matching patterns.
        
        Args:
            directory_pattern (str): The wildcard pattern used to locate all TFRecord shards.
            pre_resized (bool, optional): Indicates if the embedded JPEGs are already sized correctly. Defaults to False.
        """
        self.pre_resized = pre_resized
        self.data_records = []
        tfrecord_files = sorted(glob.glob(pathname=directory_pattern))
        for tfrecord_path in tfrecord_files:
            iterator = tfrecord.tfrecord_loader(data_path=tfrecord_path, index_path=None, description={
                "image": "byte",
                "bboxes": "float",
                "labels": "int"
            })
            for record in iterator:
                self.data_records.append(record)
                
    def __len__(self) -> int:
        """
        Returns the total number of records across all shards.
        
        Returns:
            int: The total count of combined data records.
        """
        return len(self.data_records)
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves and decodes a single TFRecord entry from the combined shards.
        
        Args:
            _index (int): The index of the record to decode.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The decoded image, boxes, and labels.
        """
        return decode_tfrecord(features=self.data_records[_index], pre_resized=self.pre_resized)

def benchmark_tfrecord_sharded(directory_pattern: str, number_of_runs: int = 10) -> float:
    """
    Benchmarks the sharded TFRecord dataset loader.
    
    This function evaluates multi-file partitioned TFRecord performance within the benchmark pipeline.
    
    Args:
        directory_pattern (str): The wildcard pattern used to locate all TFRecord shards.
        number_of_runs (int, optional): The number of full epoch passes to simulate. Defaults to 10.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = CustomShardedTFRecordDataset(directory_pattern=directory_pattern, pre_resized=False)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_function)
    times = []
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for _batch in dataloader:
            pass
        times.append(time.time() - start_time)
    return sum(times) / number_of_runs

# --- 4. LMDB ---
class LMDBDataset(Dataset):
    """
    A PyTorch Dataset implementation for interacting with Lightning Memory-Mapped Databases (LMDB).
    
    This class is evaluated in the benchmarking pipeline to test the efficiency of memory-mapped 
    key-value stores for massive dataset retrieval, avoiding traditional disk seeks.
    """
    def __init__(self, lmdb_path: str):
        """
        Initializes the LMDB dataset and maps the environment into memory.
        
        Args:
            lmdb_path (str): The absolute path to the LMDB database folder.
        """
        self.environment = lmdb.open(path=lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.environment.begin() as transaction:
            self.keys = pickle.loads(data=transaction.get(key=b'__keys__'))
            
    def __len__(self) -> int:
        """
        Returns the total number of keys registered in the LMDB database.
        
        Returns:
            int: The total count of items.
        """
        return len(self.keys)
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves, unpickles, and decodes an image sample from the memory-mapped environment.
        
        Args:
            _index (int): The index of the item to retrieve.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The decoded image, boxes, and labels.
        """
        key = self.keys[_index]
        with self.environment.begin() as transaction:
            data = pickle.loads(data=transaction.get(key=str(key).encode(encoding='ascii')))
            
        image_bytes = data['jpg']
        bounding_boxes = data['bboxes']
        labels = data['labels']
        
        image_pil = Image.open(fp=io.BytesIO(initial_bytes=image_bytes)).convert(mode='RGB')
        image_resized, resized_bounding_boxes = resize_image_and_boxes(image_pil=image_pil, bounding_boxes=bounding_boxes)
        image_tensor = T.ToTensor()(image_resized)
        
        return image_tensor, torch.tensor(data=resized_bounding_boxes), torch.tensor(data=labels)

def benchmark_lmdb(lmdb_path: str, number_of_runs: int = 10) -> float:
    """
    Benchmarks the LMDB dataset loader.
    
    This function evaluates memory-mapped database latency within the data loading benchmark pipeline.
    
    Args:
        lmdb_path (str): The absolute path to the LMDB database folder.
        number_of_runs (int, optional): The number of full epoch passes to simulate. Defaults to 10.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = LMDBDataset(lmdb_path=lmdb_path)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_function)
    times = []
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for _batch in dataloader:
            pass
        times.append(time.time() - start_time)
    return sum(times) / number_of_runs

# --- 5. HDF5 ---
class HDF5Dataset(Dataset):
    """
    A PyTorch Dataset implementation for interacting with Hierarchical Data Format (HDF5) files.
    
    This dataset assesses the viability of scientific array storage formats within the 
    data loading benchmark pipeline.
    """
    def __init__(self, h5_path: str):
        """
        Initializes the HDF5 dataset without eagerly opening the file handler.
        
        Args:
            h5_path (str): The absolute path to the HDF5 file.
        """
        self.h5_path = h5_path
        self.h5_file = None
        
    def __len__(self) -> int:
        """
        Calculates the length of the dataset by safely peeking into the HDF5 file structure.
        
        Returns:
            int: The total count of images inside the 'images' group.
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(name=self.h5_path, mode='r')
            self.length = len(self.h5_file['images'])
            self.h5_file.close()
            self.h5_file = None
            return self.length
        return len(self.h5_file['images'])
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves and decodes an image sample from the underlying HDF5 hierarchy.
        
        Args:
            _index (int): The index of the item to retrieve.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The decoded image, boxes, and labels.
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(name=self.h5_path, mode='r')
            
        image_bytes = self.h5_file['images'][_index]
        bounding_boxes_flat = self.h5_file['bboxes'][_index]
        labels = self.h5_file['labels'][_index]
        
        image_pil = Image.open(fp=io.BytesIO(initial_bytes=image_bytes.tobytes())).convert(mode='RGB')
        
        bounding_boxes = []
        for index in range(0, len(bounding_boxes_flat), 4):
            bounding_boxes.append(bounding_boxes_flat[index:index+4])
            
        image_resized, resized_bounding_boxes = resize_image_and_boxes(image_pil=image_pil, bounding_boxes=bounding_boxes)
        image_tensor = T.ToTensor()(image_resized)
        
        return image_tensor, torch.tensor(data=resized_bounding_boxes), torch.tensor(data=labels)

def benchmark_hdf5(h5_path: str, number_of_runs: int = 10) -> float:
    """
    Benchmarks the HDF5 dataset loader.
    
    This function evaluates scientific storage container latency within the benchmark pipeline.
    
    Args:
        h5_path (str): The absolute path to the HDF5 file.
        number_of_runs (int, optional): The number of full epoch passes to simulate. Defaults to 10.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = HDF5Dataset(h5_path=h5_path)
    dataset.__len__()
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_function)
    times = []
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for _batch in dataloader:
            pass
        times.append(time.time() - start_time)
    return sum(times) / number_of_runs

# --- 6. Pre-Resized JPEGs (Native) ---
class PreResizedDataset(Dataset):
    """
    A PyTorch Dataset that loads native JPEGs which have already been preemptively scaled.
    
    This class is utilized in the benchmark pipeline to isolate and quantify the exact CPU 
    overhead introduced by runtime PIL resizing operations.
    """
    def __init__(self, data_directory: str, labels_file: str):
        """
        Initializes the pre-resized JPEGs dataset.
        
        Args:
            data_directory (str): The directory containing the pre-scaled JPEG images.
            labels_file (str): The JSON file containing the correspondingly scaled bounding box coordinates.
        """
        self.data_directory = data_directory
        with open(file=labels_file, mode='r') as file_handler:
            self.data_items = json.load(fp=file_handler)
            
    def __len__(self) -> int:
        """
        Returns the total number of pre-scaled images.
        
        Returns:
            int: The total count of images.
        """
        return len(self.data_items)
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a pre-scaled image and its pre-scaled annotations directly into a tensor.
        
        Args:
            _index (int): The index of the item to retrieve.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The image tensor, boxes, and labels.
        """
        item = self.data_items[_index]
        image_path = os.path.join(self.data_directory, item['file_name'])
        image_pil = Image.open(fp=image_path).convert(mode='RGB')
        image_tensor = T.ToTensor()(image_pil)
        return image_tensor, torch.tensor(data=item['bboxes']), torch.tensor(data=item['labels'])

def benchmark_preresized(data_directory: str, labels_file: str, number_of_runs: int = 10) -> float:
    """
    Benchmarks the Pre-Resized JPEGs dataset loader.
    
    This function evaluates the performance of the Native PyTorch dataloader without CPU resizing 
    bottlenecks within the benchmark pipeline.
    
    Args:
        data_directory (str): The path to the directory containing pre-scaled JPEG images.
        labels_file (str): The path to the JSON file containing pre-scaled COCO-formatted annotations.
        number_of_runs (int, optional): The number of full epoch passes to simulate. Defaults to 10.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = PreResizedDataset(data_directory=data_directory, labels_file=labels_file)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_function)
    times = []
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for _batch in dataloader:
            pass
        times.append(time.time() - start_time)
    return sum(times) / number_of_runs

# --- 7. Pre-Resized TFRecord ---
def benchmark_tfrecord_preresized(tfrecord_path: str, number_of_runs: int = 10) -> float:
    """
    Benchmarks the Pre-Resized TFRecord dataset loader.
    
    This function evaluates the absolute theoretical maximum throughput of CPU decoding by 
    combining binary archive storage with preemptive spatial scaling in the benchmark pipeline.
    
    Args:
        tfrecord_path (str): The absolute path to the pre-resized TFRecord archive file.
        number_of_runs (int, optional): The number of full epoch passes to simulate. Defaults to 10.
        
    Returns:
        float: The average time taken per run in seconds.
    """
    dataset = CustomTFRecordDataset(tfrecord_path=tfrecord_path, pre_resized=True)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_function)
    times = []
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for _batch in dataloader:
            pass
        times.append(time.time() - start_time)
    return sum(times) / number_of_runs

# --- 8. NVIDIA DALI ---
@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def coco_dali_pipeline(data_directory: str, annotations_file: str):
    """
    Defines the computation graph for NVIDIA DALI hardware-accelerated processing.
    
    This pipeline fundamentally replaces standard PyTorch CPU dataloading by offloading JPEG 
    decoding and spatial resizing directly to the GPU's NVJPEG decoders for massive speedups 
    in the training pipeline.
    
    Args:
        data_directory (str): The path to the directory containing raw JPEG images.
        annotations_file (str): The path to the JSON file containing COCO-formatted annotations.
        
    Returns:
        Tuple[nvidia.dali.types.TensorList, nvidia.dali.types.TensorList, nvidia.dali.types.TensorList]: 
            The DALI execution graph nodes representing the output images, boxes, and labels.
    """
    inputs, bounding_boxes, labels = fn.readers.coco(
        file_root=data_directory,
        annotations_file=annotations_file,
        polygon_masks=False,
        ratio=True,
        ltrb=False,
        name="Reader"
    )
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, resize_x=1024, resize_y=1024, interp_type=types.INTERP_LINEAR)
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT, output_layout="CHW", mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0])
    bounding_boxes = fn.pad(bounding_boxes, axes=(0,), fill_value=-1)
    labels = fn.pad(labels, axes=(0,), fill_value=-1)
    return images, bounding_boxes, labels

def benchmark_dali(data_directory: str, annotations_file: str, number_of_runs: int = 10) -> float:
    """
    Benchmarks the NVIDIA DALI hardware-accelerated pipeline.
    
    This function wraps the DALI pipeline in a PyTorch compatibility layer to simulate 
    its integration within the overall training loop benchmarking process.
    
    Args:
        data_directory (str): The path to the directory containing raw JPEG images.
        annotations_file (str): The path to the JSON file containing COCO-formatted annotations.
        number_of_runs (int, optional): The number of full epoch passes to simulate. Defaults to 10.
        
    Returns:
        float: The average time taken per run in seconds.
        
    Raises:
        RuntimeError: If the underlying DALI package is not installed on the system.
    """
    if not HAS_DALI:
        raise RuntimeError("DALI is not installed.")
    pipe = coco_dali_pipeline(data_directory=data_directory, annotations_file=annotations_file)
    pipe.build()
    dataloader = DALIGenericIterator([pipe], ['images', 'bboxes', 'labels'], reader_name="Reader", auto_reset=True)
    times = []
    for _run_index in range(number_of_runs):
        start_time = time.time()
        for _batch in dataloader:
            pass
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
    base_directory = '/Users/Robert/Data/object_detection_data/coco-2017/train'
    original_data_directory = os.path.join(base_directory, 'data')
    original_labels = os.path.join(base_directory, 'labels.json')
    
    output_directory = '/Users/Robert/Data/object_detection_data/formats'
    os.makedirs(name=output_directory, exist_ok=True)
    webdataset_url = os.path.join(output_directory, 'coco-train-{000..003}.tar')
    tfrecord_path = os.path.join(output_directory, 'coco-train.tfrecord')
    tfrecord_sharded_pattern = os.path.join(output_directory, 'sharded', 'coco-train-*.tfrecord')
    lmdb_path = os.path.join(output_directory, 'coco-train.lmdb')
    hdf5_path = os.path.join(output_directory, 'coco-train.h5')
    
    preresized_directory = os.path.join(output_directory, 'pre-resized')
    preresized_labels = os.path.join(preresized_directory, 'labels.json')
    preresized_tfrecord = os.path.join(output_directory, 'coco-train-preresized.tfrecord')
    
    print("Starting Comprehensive Benchmarks... (10 runs each)")
    print("-" * 50)
    
    results = {}
    
    def run_benchmark(name: str, function: callable, *arguments: Any) -> None:
        """
        Wraps and executes a designated benchmark sequence safely.
        
        This internal utility isolates crashing benchmarks (due to missing datasets) from 
        the remainder of the pipeline, ensuring that available formats are successfully timed.
        
        Args:
            name (str): The descriptive display name of the benchmark iteration.
            function (callable): The target benchmark method to invoke.
            *arguments (Any): The corresponding parameter arguments forwarded to the method.
            
        Returns:
            None
        """
        try:
            print(f"Benchmarking {name}...")
            average_time = function(*arguments)
            results[name] = average_time
            print(f"[{name}] Average Time: {average_time:.4f} s")
        except Exception as error:
            print(f"[{name}] Skipped due to error: {error}")
            results[name] = None
        print("-" * 50)

    run_benchmark("Native PyTorch", benchmark_native, original_data_directory, original_labels)
    run_benchmark("WebDataset", benchmark_webdataset, webdataset_url)
    run_benchmark("TFRecord (Raw)", benchmark_tfrecord, tfrecord_path)
    run_benchmark("TFRecord (Sharded)", benchmark_tfrecord_sharded, tfrecord_sharded_pattern)
    run_benchmark("LMDB", benchmark_lmdb, lmdb_path)
    run_benchmark("HDF5", benchmark_hdf5, hdf5_path)
    run_benchmark("Pre-Resized JPEGs (Native)", benchmark_preresized, preresized_directory, preresized_labels)
    run_benchmark("Pre-Resized TFRecord", benchmark_tfrecord_preresized, preresized_tfrecord)
    run_benchmark("NVIDIA DALI (GPU Decoding+Resize)", benchmark_dali, original_data_directory, original_labels)
    
    csv_file_path = os.path.abspath(path=os.path.join(os.path.dirname(p=__file__), 'benchmark_results_all.csv'))
    with open(file=csv_file_path, mode='w', newline='') as file_handler:
        writer = csv.writer(file_handler)
        writer.writerow(['Method', 'Average Time (s) per 2000 images'])
        for method, average_time in results.items():
            if average_time is not None:
                writer.writerow([method, average_time])
            
    print(f"Results written to {csv_file_path}")
    
if __name__ == "__main__":
    main()
