import os
import json
import tfrecord

def prepare():
    base_dir = '/Users/Robert/Data/object_detection_data/coco-2017/train'
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
        
    out_dir = '/Users/Robert/Data/object_detection_data/formats'
    os.makedirs(out_dir, exist_ok=True)
    sharded_dir = os.path.join(out_dir, 'sharded')
    os.makedirs(sharded_dir, exist_ok=True)
    
    # 1. Prepare Monolithic TFRecord
    tfr_path = os.path.join(out_dir, 'coco-train.tfrecord')
    print("Writing Monolithic TFRecord...")
    writer = tfrecord.TFRecordWriter(tfr_path)
    
    # 2. Prepare Sharded TFRecords
    print("Writing Sharded TFRecords...")
    shard_size = 500
    shard_idx = 0
    sharded_writer = tfrecord.TFRecordWriter(os.path.join(sharded_dir, f'coco-train-{shard_idx:03d}.tfrecord'))
    
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
        
        # Write to sharded
        if idx > 0 and idx % shard_size == 0:
            sharded_writer.close()
            shard_idx += 1
            sharded_writer = tfrecord.TFRecordWriter(os.path.join(sharded_dir, f'coco-train-{shard_idx:03d}.tfrecord'))
        sharded_writer.write(features)
        
    writer.close()
    sharded_writer.close()
    print("Done preparing formats.")

if __name__ == "__main__":
    prepare()
