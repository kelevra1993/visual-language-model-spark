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

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

def resize_image_and_boxes(image_pil, bboxes, target_size=(1024, 1024)):
    w, h = image_pil.size
    image_resized = image_pil.resize(target_size, Image.BILINEAR)
    scale_x = target_size[0] / w
    scale_y = target_size[1] / h
    
    resized_bboxes = []
    for bbox in bboxes:
        bx, by, bw, bh = bbox
        nx = bx * scale_x
        ny = by * scale_y
        nw = bw * scale_x
        nh = bh * scale_y
        resized_bboxes.append([nx, ny, nw, nh])
    
    return image_resized, resized_bboxes

def collate_fn(batch):
    images = [item[0] for item in batch]
    bboxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return torch.stack(images), bboxes, labels

# --- 1. Native PyTorch ---
class NativeCocoDataset(Dataset):
    def __init__(self, data_dir, labels_file):
        self.data_dir = data_dir
        with open(labels_file, 'r') as f:
            coco = json.load(f)
            
        self.images = {img['id']: img for img in coco['images']}
        self.image_ids = list(self.images.keys())
        
        self.img_to_anns = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        
        image_pil = Image.open(img_path).convert('RGB')
        
        anns = self.img_to_anns.get(img_id, [])
        bboxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]
        
        image_resized, resized_bboxes = resize_image_and_boxes(image_pil, bboxes)
        image_tensor = T.ToTensor()(image_resized)
        
        return image_tensor, torch.tensor(resized_bboxes), torch.tensor(labels)

def benchmark_native(data_dir, labels_file, num_runs=10):
    dataset = NativeCocoDataset(data_dir, labels_file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader: pass
        times.append(time.time() - start)
    return sum(times) / num_runs

# --- 2. WebDataset ---
def decode_wds(sample):
    img_bytes = sample['jpg']
    image_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    json_bytes = sample['json']
    json_data = json.loads(json_bytes.decode('utf-8'))
    bboxes = json_data['bboxes']
    labels = json_data['labels']
    
    image_resized, resized_bboxes = resize_image_and_boxes(image_pil, bboxes)
    image_tensor = T.ToTensor()(image_resized)
    return image_tensor, torch.tensor(resized_bboxes), torch.tensor(labels)

def benchmark_wds(tar_url, num_runs=10):
    dataset = wds.WebDataset(tar_url).map(decode_wds).batched(16, collation_fn=collate_fn)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader: pass
        times.append(time.time() - start)
    return sum(times) / num_runs

# --- 3. TFRecord (Raw) ---
def decode_tfrecord(features, pre_resized=False):
    img_bytes = features['image']
    bboxes_flat = features['bboxes']
    labels = features['labels']
    
    image_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    bboxes = []
    for i in range(0, len(bboxes_flat), 4):
        bboxes.append(bboxes_flat[i:i+4])
        
    if not pre_resized:
        image_resized, resized_bboxes = resize_image_and_boxes(image_pil, bboxes)
    else:
        image_resized, resized_bboxes = image_pil, bboxes
        
    image_tensor = T.ToTensor()(image_resized)
    return image_tensor, torch.tensor(resized_bboxes), torch.tensor(labels)

class CustomTFRecordDataset(Dataset):
    def __init__(self, tfrecord_path, pre_resized=False):
        self.pre_resized = pre_resized
        self.data = []
        iterator = tfrecord.tfrecord_loader(tfrecord_path, None, {
            "image": "byte",
            "bboxes": "float",
            "labels": "int"
        })
        for record in iterator:
            self.data.append(record)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return decode_tfrecord(self.data[idx], self.pre_resized)

def benchmark_tfrecord(tfrecord_path, num_runs=10):
    dataset = CustomTFRecordDataset(tfrecord_path, pre_resized=False)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader: pass
        times.append(time.time() - start)
    return sum(times) / num_runs

# --- 4. LMDB ---
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            self.keys = pickle.loads(txn.get(b'__keys__'))
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(key).encode('ascii')))
        img_bytes = data['jpg']
        bboxes = data['bboxes']
        labels = data['labels']
        image_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image_resized, resized_bboxes = resize_image_and_boxes(image_pil, bboxes)
        image_tensor = T.ToTensor()(image_resized)
        return image_tensor, torch.tensor(resized_bboxes), torch.tensor(labels)

def benchmark_lmdb(lmdb_path, num_runs=10):
    dataset = LMDBDataset(lmdb_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader: pass
        times.append(time.time() - start)
    return sum(times) / num_runs

# --- 5. HDF5 ---
class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5f = None
    def __len__(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, 'r')
            self.length = len(self.h5f['images'])
            self.h5f.close()
            self.h5f = None
            return self.length
        return len(self.h5f['images'])
    def __getitem__(self, idx):
        if self.h5f is None:
            self.h5f = h5py.File(self.h5_path, 'r')
        img_bytes = self.h5f['images'][idx]
        bboxes_flat = self.h5f['bboxes'][idx]
        labels = self.h5f['labels'][idx]
        image_pil = Image.open(io.BytesIO(img_bytes.tobytes())).convert('RGB')
        bboxes = []
        for i in range(0, len(bboxes_flat), 4):
            bboxes.append(bboxes_flat[i:i+4])
        image_resized, resized_bboxes = resize_image_and_boxes(image_pil, bboxes)
        image_tensor = T.ToTensor()(image_resized)
        return image_tensor, torch.tensor(resized_bboxes), torch.tensor(labels)

def benchmark_hdf5(h5_path, num_runs=10):
    dataset = HDF5Dataset(h5_path)
    dataset.__len__()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader: pass
        times.append(time.time() - start)
    return sum(times) / num_runs

# --- 6. Pre-Resized JPEGs (Native) ---
class PreResizedDataset(Dataset):
    def __init__(self, data_dir, labels_file):
        self.data_dir = data_dir
        with open(labels_file, 'r') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.data_dir, item['file_name'])
        image_pil = Image.open(img_path).convert('RGB')
        image_tensor = T.ToTensor()(image_pil)
        return image_tensor, torch.tensor(item['bboxes']), torch.tensor(item['labels'])

def benchmark_preresized(data_dir, labels_file, num_runs=10):
    dataset = PreResizedDataset(data_dir, labels_file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader: pass
        times.append(time.time() - start)
    return sum(times) / num_runs

# --- 7. Pre-Resized TFRecord ---
def benchmark_tfrecord_preresized(tfrecord_path, num_runs=10):
    dataset = CustomTFRecordDataset(tfrecord_path, pre_resized=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader: pass
        times.append(time.time() - start)
    return sum(times) / num_runs

# --- 8. NVIDIA DALI ---
@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def coco_dali_pipeline(data_dir, annotations_file):
    inputs, bboxes, labels = fn.readers.coco(
        file_root=data_dir,
        annotations_file=annotations_file,
        polygon_masks=False,
        ratio=True,
        ltrb=False,
        name="Reader"
    )
    images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, resize_x=1024, resize_y=1024, interp_type=types.INTERP_LINEAR)
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT, output_layout="CHW", mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0])
    bboxes = fn.pad(bboxes, axes=(0,), fill_value=-1)
    labels = fn.pad(labels, axes=(0,), fill_value=-1)
    return images, bboxes, labels

def benchmark_dali(data_dir, annotations_file, num_runs=10):
    pipe = coco_dali_pipeline(data_dir=data_dir, annotations_file=annotations_file)
    pipe.build()
    dataloader = DALIGenericIterator([pipe], ['images', 'bboxes', 'labels'], reader_name="Reader", auto_reset=True)
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader: pass
        times.append(time.time() - start)
    return sum(times) / num_runs


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/coco-2017/train'))
    original_data_dir = os.path.join(base_dir, 'data')
    original_labels = os.path.join(base_dir, 'labels.json')
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/formats'))
    wds_url = os.path.join(out_dir, 'coco-train-{000..003}.tar')
    tfr_path = os.path.join(out_dir, 'coco-train.tfrecord')
    lmdb_path = os.path.join(out_dir, 'coco-train.lmdb')
    h5_path = os.path.join(out_dir, 'coco-train.h5')
    
    preresized_dir = os.path.join(out_dir, 'pre-resized')
    preresized_labels = os.path.join(preresized_dir, 'labels.json')
    preresized_tfr = os.path.join(out_dir, 'coco-train-preresized.tfrecord')
    
    print("Starting Comprehensive Benchmarks... (10 runs each)")
    print("-" * 50)
    
    results = {}
    
    def run_b(name, func, *args):
        print(f"Benchmarking {name}...")
        avg_t = func(*args)
        results[name] = avg_t
        print(f"[{name}] Average Time: {avg_t:.4f} s")
        print("-" * 50)

    run_b("Native PyTorch", benchmark_native, original_data_dir, original_labels)
    run_b("WebDataset", benchmark_wds, wds_url)
    run_b("TFRecord (Raw)", benchmark_tfrecord, tfr_path)
    run_b("LMDB", benchmark_lmdb, lmdb_path)
    run_b("HDF5", benchmark_hdf5, h5_path)
    run_b("Pre-Resized JPEGs (Native)", benchmark_preresized, preresized_dir, preresized_labels)
    run_b("Pre-Resized TFRecord", benchmark_tfrecord_preresized, preresized_tfr)
    run_b("NVIDIA DALI (GPU Decoding+Resize)", benchmark_dali, original_data_dir, original_labels)
    
    csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'benchmark_results_all.csv'))
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Method', 'Average Time (s) per 2000 images'])
        for method, avg_t in results.items():
            writer.writerow([method, avg_t])
            
    print(f"Results written to {csv_file}")
    
if __name__ == "__main__":
    main()
