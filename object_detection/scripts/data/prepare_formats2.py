import os
import json
import numpy as np
import webdataset as wds
import tfrecord
from PIL import Image
import h5py
import lmdb
import pickle

def resize_image_and_boxes(image_pil, bboxes, target_size=(1024, 1024)):
    w, h = image_pil.size
    image_resized = image_pil.resize(target_size, Image.BILINEAR)
    scale_x = target_size[0] / w
    scale_y = target_size[1] / h
    
    resized_bboxes = []
    for bbox in bboxes:
        bx, by, bw, bh = bbox
        resized_bboxes.append([bx * scale_x, by * scale_y, bw * scale_x, bh * scale_y])
    return image_resized, resized_bboxes

def prepare_new_formats():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/coco-2017/train'))
    labels_file = os.path.join(base_dir, 'labels.json')
    data_dir = os.path.join(base_dir, 'data')
    
    with open(labels_file, 'r') as f:
        coco = json.load(f)
        
    images = {img['id']: img for img in coco['images']}
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/formats'))
    
    # 3. LMDB
    lmdb_path = os.path.join(out_dir, 'coco-train.lmdb')
    print("Writing LMDB...")
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for idx, img_id in enumerate(images):
            img_info = images[img_id]
            img_path = os.path.join(data_dir, img_info['file_name'])
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            anns = img_to_anns.get(img_id, [])
            bboxes = [ann['bbox'] for ann in anns]
            labels = [ann['category_id'] for ann in anns]
            
            data_dict = {"jpg": img_bytes, "bboxes": bboxes, "labels": labels}
            txn.put(str(img_id).encode('ascii'), pickle.dumps(data_dict))
            
            if idx == 0: # Save a list of keys
                pass
        txn.put(b'__keys__', pickle.dumps(list(images.keys())))
    env.close()

    # 4. HDF5
    h5_path = os.path.join(out_dir, 'coco-train.h5')
    print("Writing HDF5...")
    with h5py.File(h5_path, 'w') as h5f:
        dt = h5py.vlen_dtype(np.dtype('uint8'))
        img_ds = h5f.create_dataset('images', (len(images),), dtype=dt)
        bbox_ds = h5f.create_dataset('bboxes', (len(images),), dtype=h5py.vlen_dtype(np.float32))
        label_ds = h5f.create_dataset('labels', (len(images),), dtype=h5py.vlen_dtype(np.int32))
        
        for idx, img_id in enumerate(images.keys()):
            img_info = images[img_id]
            img_path = os.path.join(data_dir, img_info['file_name'])
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
                
            anns = img_to_anns.get(img_id, [])
            bboxes = [val for ann in anns for val in ann['bbox']]
            labels = [ann['category_id'] for ann in anns]
            
            img_ds[idx] = np.frombuffer(img_bytes, dtype=np.uint8)
            bbox_ds[idx] = np.array(bboxes, dtype=np.float32)
            label_ds[idx] = np.array(labels, dtype=np.int32)
            
    # 5. Pre-resized JPEGs
    preresized_dir = os.path.join(out_dir, 'pre-resized')
    os.makedirs(preresized_dir, exist_ok=True)
    print("Writing Pre-resized JPEGs...")
    preresized_labels = []
    
    for idx, img_id in enumerate(images):
        img_info = images[img_id]
        img_path = os.path.join(data_dir, img_info['file_name'])
        image_pil = Image.open(img_path).convert('RGB')
        
        anns = img_to_anns.get(img_id, [])
        bboxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]
        
        image_resized, resized_bboxes = resize_image_and_boxes(image_pil, bboxes)
        
        out_img_path = os.path.join(preresized_dir, f"{img_id}.jpg")
        image_resized.save(out_img_path, format='JPEG')
        
        preresized_labels.append({
            "id": img_id,
            "file_name": f"{img_id}.jpg",
            "bboxes": resized_bboxes,
            "labels": labels
        })
        
    with open(os.path.join(preresized_dir, 'labels.json'), 'w') as f:
        json.dump(preresized_labels, f)

    print("Done preparing new formats.")

if __name__ == "__main__":
    prepare_new_formats()
