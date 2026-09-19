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

def resize_image_and_boxes(image_pil, bboxes, target_size=(1024, 1024)):
    w, h = image_pil.size
    image_resized = image_pil.resize(target_size, Image.BILINEAR)
    scale_x = target_size[0] / w
    scale_y = target_size[1] / h
    
    resized_bboxes = []
    for bbox in bboxes:
        # COCO bbox: [x, y, width, height]
        bx, by, bw, bh = bbox
        nx = bx * scale_x
        ny = by * scale_y
        nw = bw * scale_x
        nh = bh * scale_y
        resized_bboxes.append([nx, ny, nw, nh])
    
    return image_resized, resized_bboxes

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
        
        # Convert to tensor just to simulate actual dataloader pipeline
        image_tensor = T.ToTensor()(image_resized)
        
        return image_tensor, torch.tensor(resized_bboxes), torch.tensor(labels)

def collate_fn(batch):
    images = [item[0] for item in batch]
    bboxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return torch.stack(images), bboxes, labels

def benchmark_native(data_dir, labels_file, num_runs=10):
    dataset = NativeCocoDataset(data_dir, labels_file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader:
            pass
        times.append(time.time() - start)
        
    avg_time = sum(times) / num_runs
    print(f"[Native PyTorch] Average Time over {num_runs} runs: {avg_time:.4f} s")
    return avg_time

def decode_wds(sample):
    # Decode image bytes to PIL Image
    img_bytes = sample['jpg']
    image_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # Extract json
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
        for batch in dataloader:
            pass
        times.append(time.time() - start)
        
    avg_time = sum(times) / num_runs
    print(f"[WebDataset] Average Time over {num_runs} runs: {avg_time:.4f} s")
    return avg_time

def decode_tfrecord(features):
    img_bytes = features['image']
    bboxes_flat = features['bboxes']
    labels = features['labels']
    
    image_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    bboxes = []
    for i in range(0, len(bboxes_flat), 4):
        bboxes.append(bboxes_flat[i:i+4])
        
    image_resized, resized_bboxes = resize_image_and_boxes(image_pil, bboxes)
    image_tensor = T.ToTensor()(image_resized)
    
    return image_tensor, torch.tensor(resized_bboxes), torch.tensor(labels)

class CustomTFRecordDataset(Dataset):
    def __init__(self, tfrecord_path):
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
        return decode_tfrecord(self.data[idx])

def benchmark_tfrecord_custom(tfrecord_path, num_runs=10):
    dataset = CustomTFRecordDataset(tfrecord_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    times = []
    for run in range(num_runs):
        start = time.time()
        for batch in dataloader:
            pass
        times.append(time.time() - start)
        
    avg_time = sum(times) / num_runs
    print(f"[TFRecord] Average Time over {num_runs} runs: {avg_time:.4f} s")
    return avg_time

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/coco-2017/train'))
    labels_file = os.path.join(base_dir, 'labels.json')
    data_dir = os.path.join(base_dir, 'data')
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/formats'))
    wds_url = os.path.join(out_dir, 'coco-train-{000..003}.tar')
    tfr_path = os.path.join(out_dir, 'coco-train.tfrecord')
    
    print("Starting Benchmarks... (10 runs each)")
    print("-" * 50)
    benchmark_native(data_dir, labels_file, num_runs=10)
    print("-" * 50)
    benchmark_wds(wds_url, num_runs=10)
    print("-" * 50)
    benchmark_tfrecord_custom(tfr_path, num_runs=10)
    print("-" * 50)
    
if __name__ == "__main__":
    main()
