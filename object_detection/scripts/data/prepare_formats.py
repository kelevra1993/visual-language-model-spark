import os
import json
import webdataset as wds
import tfrecord
import struct
from PIL import Image
import numpy as np

def prepare():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/coco-2017/train'))
    labels_file = os.path.join(base_dir, 'labels.json')
    data_dir = os.path.join(base_dir, 'data')
    
    with open(labels_file, 'r') as f:
        coco = json.load(f)
        
    images = {img['id']: img for img in coco['images']}
    annotations = coco['annotations']
    
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/formats'))
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Prepare WebDataset
    wds_path = os.path.join(out_dir, 'coco-train-%03d.tar')
    sink = wds.ShardWriter(wds_path, maxcount=500)
    
    print("Writing WebDataset...")
    for idx, img_id in enumerate(images):
        img_info = images[img_id]
        img_path = os.path.join(data_dir, img_info['file_name'])
        
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
            
        anns = img_to_anns.get(img_id, [])
        bboxes = [ann['bbox'] for ann in anns] # [x, y, w, h]
        labels = [ann['category_id'] for ann in anns]
        
        sample = {
            "__key__": str(img_id),
            "jpg": img_bytes,
            "json": {"bboxes": bboxes, "labels": labels}
        }
        sink.write(sample)
    sink.close()
    
    # 2. Prepare TFRecord
    tfr_path = os.path.join(out_dir, 'coco-train.tfrecord')
    print("Writing TFRecord...")
    writer = tfrecord.TFRecordWriter(tfr_path)
    
    for idx, img_id in enumerate(images):
        img_info = images[img_id]
        img_path = os.path.join(data_dir, img_info['file_name'])
        
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
            
        anns = img_to_anns.get(img_id, [])
        bboxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]
        
        # Flatten bboxes
        bboxes_flat = [val for bbox in bboxes for val in bbox]
        
        features = {
            "image": (img_bytes, "byte"),
            "bboxes": (bboxes_flat, "float"),
            "labels": (labels, "int")
        }
        writer.write(features)
    writer.close()
    print("Done preparing formats.")

if __name__ == "__main__":
    prepare()
