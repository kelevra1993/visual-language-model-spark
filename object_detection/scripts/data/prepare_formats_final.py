import os
import json
import tfrecord

def prepare_preresized_tfrecord():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/formats'))
    preresized_dir = os.path.join(base_dir, 'pre-resized')
    labels_file = os.path.join(preresized_dir, 'labels.json')
    
    with open(labels_file, 'r') as f:
        data = json.load(f)
        
    tfr_path = os.path.join(base_dir, 'coco-train-preresized.tfrecord')
    print("Writing Pre-Resized TFRecord...")
    writer = tfrecord.TFRecordWriter(tfr_path)
    
    for item in data:
        img_id = item['id']
        img_path = os.path.join(preresized_dir, item['file_name'])
        
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
            
        bboxes = item['bboxes']
        labels = item['labels']
        
        bboxes_flat = [val for bbox in bboxes for val in bbox]
        
        features = {
            "image": (img_bytes, "byte"),
            "bboxes": (bboxes_flat, "float"),
            "labels": (labels, "int")
        }
        writer.write(features)
    writer.close()
    print("Done writing Pre-Resized TFRecord.")

if __name__ == "__main__":
    prepare_preresized_tfrecord()
