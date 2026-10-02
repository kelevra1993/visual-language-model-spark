import torch
import nvidia.dali.fn as dali_function
import nvidia.dali.types as dali_types

from nvidia.dali import pipeline_def as pipeline_definition
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from typing import Any, Dict

from vision_training.data.preprocessing import preprocess_boxes


@pipeline_definition
def dali_pipeline(data_directory: str, annotations_file: str, image_size: int, keep_ratio: bool):
    """
    Defines the NVIDIA DALI processing pipeline for decoding and augmenting images directly on the GPU.
    
    This circumvents the CPU bottleneck by routing raw JPEG/PNG bytes directly to the GPU's hardware decoder
    and CUDA cores, maximizing throughput.
    
    Args:
        data_directory (str): The absolute path to the directory containing the physical image files.
        annotations_file (str): The absolute path to the JSON file containing the annotations.
        image_size (int): The target spatial dimension to scale the images to.
        keep_ratio (bool): Whether to maintain the aspect ratio during scaling.
        
    Returns:
        tuple: A tuple of DALI Edge objects representing the processed images, padded bounding boxes, and padded labels.
    """
    # Load raw data and annotations
    inputs, bounding_boxes, labels, image_ids = dali_function.readers.coco(file_root=data_directory,
                                                                           annotations_file=annotations_file,
                                                                           polygon_masks=False,
                                                                           ratio=False,
                                                                           ltrb=True,
                                                                           random_shuffle=True,
                                                                           image_ids=True,
                                                                           name="Reader")

    # Decode directly on the GPU using hardware acceleration
    images = dali_function.decoders.image(inputs, device="mixed", output_type=dali_types.BGR)

    # We must retrieve the original shapes to un-normalize bounding boxes correctly in the wrapper
    shapes = dali_function.peek_image_shape(inputs)

    # Hardware accelerated resize mimicking the zero-padding of CocoDetectionDataset
    if keep_ratio:
        # Resize such that the longest edge equals image_size
        images = dali_function.resize(images, resize_x=image_size, resize_y=image_size, mode="not_larger")

        # We do NOT divide by 255.0 here to maintain parity with the native datasets, keeping pixels in [0, 255]
        # By providing a crop larger than the image (since it was resized to have a max dimension of image_size) 
        # and setting crop_pos=0.5, DALI automatically applies symmetric center padding!
        images = dali_function.crop_mirror_normalize(images, dtype=dali_types.FLOAT, output_layout="CHW",
                                                     mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
                                                     crop=(image_size, image_size), crop_pos_x=0.5, crop_pos_y=0.5,
                                                     out_of_bounds_policy="pad", fill_values=0.0)
    else:
        # Direct squash resize
        images = dali_function.resize(images, resize_x=image_size, resize_y=image_size,
                                      interp_type=dali_types.INTERP_LINEAR)

        images = dali_function.crop_mirror_normalize(images, dtype=dali_types.FLOAT, output_layout="CHW",
                                                     mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

    # Pad variable-length boxes and labels to uniform tensors for the batch
    bounding_boxes = dali_function.pad(bounding_boxes, axes=(0,), fill_value=-1.0)
    labels = dali_function.pad(labels, axes=(0,), fill_value=-1)

    return images, bounding_boxes, labels, shapes, image_ids


class DALIDataloaderWrapper:
    """
    Wraps the NVIDIA DALIGenericIterator to perfectly match the expected PyTorch dataloader output format.
    
    Since DALI pads bounding boxes and labels to uniform tensors to process them in a single batch,
    this wrapper dynamically unpads them back into lists of independent tensors, allowing seamless integration
    with the existing model architecture and loss functions that expect variable-sized lists.
    """

    def __init__(self, dali_iterator: DALIGenericIterator, image_size: int, keep_ratio: bool,
                 label_mode: str, label_dictionary: Dict[str, Any]):
        """
        Initializes the DALI wrapper.
        
        Args:
            dali_iterator (DALIGenericIterator): The underlying DALI hardware-accelerated iterator.
            image_size (int): The target spatial dimension images were scaled to.
            keep_ratio (bool): Whether images were padded to maintain aspect ratio.
            label_mode (str): Mode for class assignment ("binary" or "multi_class").
            label_dictionary (Dict[str, Any]): Dictionary to map raw classes.
        """
        self.dali_iterator = dali_iterator
        self.image_size = image_size
        self.keep_ratio = keep_ratio
        self.label_mode = label_mode
        self.label_dictionary = label_dictionary

    def __iter__(self):
        """
        Yields the next batch from the DALI pipeline, unpacking the padded tensors into standard lists.
        
        Yields:
            Dict[str, Any]: A dictionary containing the batched images, and unpadded lists for bounding boxes and labels.
        """
        for batch in self.dali_iterator:
            # DALI returns a list of dictionaries (one for each pipeline instance)
            data = batch[0]
            images = data["images"]
            bounding_boxes_padded = data["bounding_boxes"]
            labels_padded = data["labels"]
            shapes = data["shapes"]
            image_ids = data["image_ids"].squeeze(-1)  # Squeeze 2D [batch_size, 1] to 1D [batch_size] if necessary

            bounding_boxes_list = []
            labels_list = []

            # Unpack the padded tensors back into variable-length lists per image
            for batch_index in range(images.shape[0]):
                # Identify the valid items by filtering out the padding fill value (-1)
                valid_mask = (labels_padded[batch_index] != -1).squeeze(-1) if labels_padded.dim() == 3 else (
                        labels_padded[batch_index] != -1)

                valid_boxes = bounding_boxes_padded[batch_index][valid_mask].clone()
                valid_labels = labels_padded[batch_index][valid_mask]

                # Apply dynamic label mapping and filter out backgrounds
                if self.label_mode == "binary":
                    valid_labels = torch.where(valid_labels > 0, 1, 0)
                else:
                    valid_labels_cpu = valid_labels.cpu().numpy()
                    # Iterate through each raw label and retrieve the updated mapped class index from the dictionary.
                    # To defend against JSON parsers loading integer keys as strings, we perform a dual lookup:
                    # first attempting an integer cast, then a string cast, before finally defaulting to the background class 0.
                    mapped_labels = [self.label_dictionary.get(int(l), self.label_dictionary.get(str(l), 0)) for l in valid_labels_cpu]
                    valid_labels = torch.tensor(mapped_labels, dtype=torch.int64, device=valid_labels.device)

                # Filter out background objects entirely
                non_background_mask = valid_labels > 0
                valid_labels = valid_labels[non_background_mask]
                valid_boxes = valid_boxes[non_background_mask]

                if valid_boxes.size(0) == 0:
                    valid_boxes = torch.zeros((0, 4), dtype=torch.float32, device=valid_boxes.device)

                labels_list.append(valid_labels)

                # Transform bounding boxes using the centralized math in preprocess_boxes
                original_height = shapes[batch_index][0].item()
                original_width = shapes[batch_index][1].item()

                # Convert the absolute original coordinate PyTorch tensor to a Python list for preprocess_boxes
                processed_boxes_list = preprocess_boxes(bounding_boxes=valid_boxes.tolist(),
                                                        image_width=original_width,
                                                        image_height=original_height,
                                                        image_size=self.image_size,
                                                        keep_ratio=self.keep_ratio)

                # Convert back to a PyTorch tensor
                valid_boxes = torch.tensor(data=processed_boxes_list, dtype=torch.float32, device=valid_boxes.device)
                if valid_boxes.numel() == 0:
                    valid_boxes = valid_boxes.reshape(0, 4)

                bounding_boxes_list.append(valid_boxes)

            yield {"images": images, "bounding_boxes": bounding_boxes_list, "labels": labels_list,
                   "image_ids": image_ids}

    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.
        
        Returns:
            int: The total number of batches.
        """
        return len(self.dali_iterator)
