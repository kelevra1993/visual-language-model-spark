import os
import glob
import json
import torch
import io
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, List
import tfrecord

def resize_image_and_boxes(image_pil: Image.Image, bounding_boxes: List[List[float]], target_size: Tuple[int, int] = (1024, 1024)) -> Tuple[Image.Image, List[List[float]]]:
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
    images = [item[0] for item in batch]
    bounding_boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return torch.stack(tensors=images), bounding_boxes, labels


class DummyDataset(Dataset):
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        return torch.zeros((3, 1024, 1024)), torch.zeros((1, 4)), torch.zeros((1,))

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

def get_dataloaders(preprocessed_directory: str, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any], batch_size: int, number_of_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_config = model_configuration.get("Data", {})
    tfrecord_config = data_config.get("TFRecord", {})
    use_tfrecord = tfrecord_config.get("activated", False)
    is_sharded = tfrecord_config.get("sharded", False)

    data_directory = experiment_configuration.get("data_folder", "")
    train_labels = experiment_configuration.get("train_split_file", "")
    
    tfrecord_path = os.path.join(data_directory, 'coco-train.tfrecord')
    sharded_pattern = os.path.join(data_directory, 'sharded', 'coco-train-*.tfrecord')
    
    dataset = None
    if use_tfrecord:
        if is_sharded and len(glob.glob(pathname=sharded_pattern)) > 0:
            dataset = CustomShardedTFRecordDataset(directory_pattern=sharded_pattern)
        elif os.path.exists(path=tfrecord_path):
            dataset = CustomTFRecordDataset(tfrecord_path=tfrecord_path)
    
    if dataset is None:
        if not data_directory or not train_labels or not os.path.isfile(train_labels):
            print(f"Warning: Falling back to dummy dataloader because data path does not exist: {data_directory}")
            dataset = DummyDataset()
        else:
            dataset = NativeCocoDataset(data_directory=data_directory, labels_file=train_labels)
        
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=number_of_workers, collate_fn=collate_function)
    return train_loader, train_loader, train_loader

