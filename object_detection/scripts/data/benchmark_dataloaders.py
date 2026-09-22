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
    coordinates accordingly to ensure spatial alignment is preserved.

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
    
    This is used by the DataLoader to stack images into a single tensor while keeping 
    bounding boxes and labels as lists since they vary in size per image.

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
    def __init__(self, data_directory: str, labels_file: str):
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
        return len(self.image_identifiers)
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def __init__(self, tfrecord_path: str, pre_resized: bool = False):
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
        return len(self.data_records)
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return decode_tfrecord(features=self.data_records[_index], pre_resized=self.pre_resized)

def benchmark_tfrecord(tfrecord_path: str, number_of_runs: int = 10) -> float:
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
    def __init__(self, directory_pattern: str, pre_resized: bool = False):
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
        return len(self.data_records)
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return decode_tfrecord(features=self.data_records[_index], pre_resized=self.pre_resized)

def benchmark_tfrecord_sharded(directory_pattern: str, number_of_runs: int = 10) -> float:
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
    def __init__(self, lmdb_path: str):
        self.environment = lmdb.open(path=lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.environment.begin() as transaction:
            self.keys = pickle.loads(data=transaction.get(key=b'__keys__'))
            
    def __len__(self) -> int:
        return len(self.keys)
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5_file = None
        
    def __len__(self) -> int:
        if self.h5_file is None:
            self.h5_file = h5py.File(name=self.h5_path, mode='r')
            self.length = len(self.h5_file['images'])
            self.h5_file.close()
            self.h5_file = None
            return self.length
        return len(self.h5_file['images'])
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def __init__(self, data_directory: str, labels_file: str):
        self.data_directory = data_directory
        with open(file=labels_file, mode='r') as file_handler:
            self.data_items = json.load(fp=file_handler)
            
    def __len__(self) -> int:
        return len(self.data_items)
        
    def __getitem__(self, _index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data_items[_index]
        image_path = os.path.join(self.data_directory, item['file_name'])
        image_pil = Image.open(fp=image_path).convert(mode='RGB')
        image_tensor = T.ToTensor()(image_pil)
        return image_tensor, torch.tensor(data=item['bboxes']), torch.tensor(data=item['labels'])

def benchmark_preresized(data_directory: str, labels_file: str, number_of_runs: int = 10) -> float:
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
        writer = csv.writer(csvfile=file_handler)
        writer.writerow(['Method', 'Average Time (s) per 2000 images'])
        for method, average_time in results.items():
            if average_time is not None:
                writer.writerow([method, average_time])
            
    print(f"Results written to {csv_file_path}")
    
if __name__ == "__main__":
    main()
