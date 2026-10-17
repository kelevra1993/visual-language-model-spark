"""
Builds the training, validation and test data loaders of every task from the experiment and model configurations:
COCO detection, classification and self-supervised learning (same layout), semantic segmentation, and the sequence
and sequence comparison tasks, every one with a PyTorch DataLoader over its dataset. Called by the task trainers.
"""

import os
import cv2

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple
from torch.utils.data import DataLoader, Dataset

from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.detection.coco_dataset import CocoDetectionDataset, collate_function
from vision_training.data.segmentation.segmentation_dataset import SegmentationDataset
from vision_training.data.sequence.sequence_dataset import SequenceDataset, collate_longest_windows
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS, SequencePairDataset


def build_native_loader(dataset: Dataset, batch_size: int, loading_configuration: Dict[str, Any],
                        collate_function: Optional[Callable] = None, drop_last: bool = False) -> DataLoader:
    """
    Builds the DataLoader of a Native dataset with the 'Data.loading' settings, shared by the loaders of every task:
    shuffled samples, worker processes kept alive between epochs and evaluations (no restart at every pass), each
    decoding with a single OpenCV thread, and batches copied to pinned memory for faster transfers to the GPU. With 0
    workers, the samples are loaded in the main process, for debugging. The single OpenCV thread is set in this
    process, which then also runs OpenCV on one thread.

    Args:
        dataset (Dataset): The Native dataset of the split.
        batch_size (int): The number of samples of a batch.
        loading_configuration (Dict[str, Any]): The 'Data.loading' section: 'number_workers' and 'prefetch_batches'.
        collate_function (Optional[Callable]): The function batching the samples, None for the default one.
        drop_last (bool): Whether to leave out the last incomplete batch of every pass over the split, for the losses
            comparing the samples of a batch with each other (self-supervised task). Defaults to False.

    Returns:
        DataLoader: The loader of the split.
    """
    number_workers = loading_configuration["number_workers"]
    use_workers = number_workers > 0

    # One OpenCV thread per worker, set in this process before the workers are forked so that they inherit it: OpenCV
    # otherwise starts one thread per core in every worker, and setting it inside a forked worker can block forever
    # once OpenCV has run its thread pool in this process
    if use_workers:
        cv2.setNumThreads(nthreads=1)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=number_workers,
                      collate_fn=collate_function, pin_memory=True, drop_last=drop_last, persistent_workers=use_workers,
                      prefetch_factor=loading_configuration["prefetch_batches"] if use_workers else None)


def get_detection_dataloader(split_name: str, split_data_folder: str, model_configuration: Dict[str, Any],
                             batch_size: int, load_instance_masks: bool = False,
                             load_original_instances: bool = False) -> DataLoader:
    """
    Instantiates the DataLoader feeding one COCO split (Train, Validation or Test) to the object detection or instance
    segmentation trainer: the CocoDetectionDataset decodes the images with OpenCV and resizes them on the CPU workers,
    with the instance masks rasterised there too for the instance segmentation, also at the original image resolution
    for the test split evaluated at that resolution.

    Args:
        split_name (str): The name of the split (e.g., 'Train', 'Validation', 'Test').
        split_data_folder (str): The root folder for this split, containing a 'data/' subfolder and a 'labels.json'.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.
        load_instance_masks (bool): Whether the samples carry the mask of every object (instance segmentation).
            Defaults to False.
        load_original_instances (bool): Whether the samples also carry the masks and boxes of the objects at the
            original image resolution (test split of the instance segmentation evaluated at that resolution).
            Defaults to False.

    Returns:
        DataLoader: The configured loader for the requested split.
    """
    data_configuration = model_configuration["Data"]
    image_settings = data_configuration["image_settings"]

    # Derive the physical image directory and annotation file from the split folder convention
    data_directory = os.path.join(split_data_folder, "data")
    labels_file = os.path.join(split_data_folder, "labels.json")

    # Decode and resize the images on the CPU workers, the boxes and labels kept as per-image lists by the collate
    dataset = CocoDetectionDataset(data_directory=data_directory, labels_file=labels_file,
                                   image_size=image_settings["size"], keep_ratio=image_settings["keep_ratio"],
                                   label_mode=data_configuration["label_mode"],
                                   label_dictionary=data_configuration["label_dictionary"],
                                   load_instance_masks=load_instance_masks,
                                   load_original_instances=load_original_instances)
    return build_native_loader(dataset=dataset, batch_size=batch_size,
                               loading_configuration=data_configuration["loading"], collate_function=collate_function)


def get_detection_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                              batch_size: int, load_instance_masks: bool = False,
                              load_original_instances: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the training, validation and testing loaders of the object detection and instance segmentation tasks,
    consumed by the detection trainer and its subclass: it resolves each split folder from the experiment configuration
    and builds the matching COCO loader for it, the test loader alone giving the objects at the original image
    resolution when they are asked, since only the evaluation reads them.

    Args:
        experiment_configuration (Dict[str, Any]): The overall experiment configuration including data paths.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.
        load_instance_masks (bool): Whether the samples carry the mask of every object (instance segmentation).
            Defaults to False.
        load_original_instances (bool): Whether the test samples also carry the masks and boxes of the objects at
            the original image resolution (instance segmentation evaluated at that resolution). Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The loaders for the training, validation and testing splits
            respectively.
    """
    # Map each split name to its corresponding root folder containing images and labels.json
    split_folder_mapping = {"Train": str(experiment_configuration["train_data_folder"]),
                            "Validation": str(experiment_configuration["validation_data_folder"]),
                            "Test": str(experiment_configuration["test_data_folder"])}

    # Build the training, validation and test loaders with the same data loading settings
    train_loader = get_detection_dataloader(split_name="Train", split_data_folder=split_folder_mapping["Train"],
                                            model_configuration=model_configuration, batch_size=batch_size,
                                            load_instance_masks=load_instance_masks)
    validation_loader = get_detection_dataloader(split_name="Validation",
                                                 split_data_folder=split_folder_mapping["Validation"],
                                                 model_configuration=model_configuration, batch_size=batch_size,
                                                 load_instance_masks=load_instance_masks)
    test_loader = get_detection_dataloader(split_name="Test", split_data_folder=split_folder_mapping["Test"],
                                           model_configuration=model_configuration, batch_size=batch_size,
                                           load_instance_masks=load_instance_masks,
                                           load_original_instances=load_original_instances)

    return train_loader, validation_loader, test_loader


def get_classification_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                                   batch_size: int, self_supervised: bool = False
                                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the training, validation and testing loaders of the classification task, read from the
    'labels.json' manifest of every split with the Native loader (OpenCV decoding and CPU resizing). The
    self-supervised task reads the same layout: every image is kept, labelled or not, the labels of the test images
    only serving its k-nearest neighbour evaluation, and the training and validation loaders leave out the last
    incomplete batch of every pass, since its losses compare the views of the images of a batch with each other (a
    batch of a single image would give a loss without gradient).

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, giving the split folders.
        model_configuration (Dict[str, Any]): The model configuration, giving the image settings, the label dictionary
            and the output mode of the classification head.
        batch_size (int): The number of samples of a batch.
        self_supervised (bool): Whether the loaders feed the self-supervised trainer, which has no classification
            head (multi-class labels). Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test loaders.

    Raises:
        ValueError: If a self-supervised training or validation split holds fewer images than a batch.
    """
    data_configuration = model_configuration["Data"]
    output_mode = "multi_class" if self_supervised else model_configuration["ClassificationHead"]["output_mode"]
    loaders = []

    # Build the loader of every split with the same settings
    for split_key in ["train_data_folder", "validation_data_folder", "test_data_folder"]:
        split_data_folder = str(experiment_configuration[split_key])
        dataset = ClassificationDataset(data_directory=os.path.join(split_data_folder, "data"),
                                        labels_file=os.path.join(split_data_folder, "labels.json"),
                                        image_size=data_configuration["image_settings"]["size"],
                                        keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                        label_dictionary=data_configuration["label_dictionary"],
                                        output_mode=output_mode, keep_unlabelled_samples=self_supervised)

        # Full batches only for the self-supervised losses, the test split being encoded image by image
        drop_last = self_supervised and split_key != "test_data_folder"
        if drop_last and len(dataset) < batch_size:
            raise ValueError(f"The '{split_key}' split holds {len(dataset)} images, fewer than the batch size "
                             f"{batch_size}: the self-supervised training and validation batches must be full.")
        loaders.append(build_native_loader(dataset=dataset, batch_size=batch_size,
                                           loading_configuration=data_configuration["loading"], drop_last=drop_last))

    return loaders[0], loaders[1], loaders[2]


def get_segmentation_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                                 batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the training, validation and testing loaders of the semantic segmentation task, read from the
    'labels.json' manifest, the 'data' images and the 'masks' class masks of every split with the Native loader
    (OpenCV decoding and CPU resizing). Every image and mask is resized to the square network input, so the default
    collate stacks them into [B, 3, H, W] images and [B, H, W] masks. With 'Data.tiling', the training and validation
    samples are tiles of the network size cut from the scaled images (at a random place, or at a place fixed per
    image), and the test images are kept whole at their scaled size, one per batch, to be segmented tile by tile.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, giving the split folders.
        model_configuration (Dict[str, Any]): The model configuration, giving the image settings, the label mode, the
            label dictionary and the loading settings.
        batch_size (int): The number of images of a batch.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test loaders.
    """
    data_configuration = model_configuration["Data"]
    tiling_configuration = data_configuration["tiling"]
    loaders = []

    # Build the loader of every split with the same settings, each split cutting its tiles its own way with tiling, the
    # whole test images of different sizes coming one per batch
    for split_key, tile_sampling in [("train_data_folder", "random"), ("validation_data_folder", "fixed"),
                                     ("test_data_folder", "whole")]:
        split_data_folder = str(experiment_configuration[split_key])
        dataset = SegmentationDataset(data_directory=os.path.join(split_data_folder, "data"),
                                      masks_directory=os.path.join(split_data_folder, "masks"),
                                      labels_file=os.path.join(split_data_folder, "labels.json"),
                                      image_size=data_configuration["image_settings"]["size"],
                                      keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                      label_mode=data_configuration["label_mode"],
                                      label_dictionary=data_configuration["label_dictionary"],
                                      tiling_configuration=tiling_configuration, tile_sampling=tile_sampling)
        split_batch_size = 1 if tiling_configuration["activated"] and tile_sampling == "whole" else batch_size
        loaders.append(build_native_loader(dataset=dataset, batch_size=split_batch_size,
                                           loading_configuration=data_configuration["loading"]))

    return loaders[0], loaders[1], loaders[2]


def get_sequence_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                             batch_size: int, label_type: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the training, validation and testing loaders of the sequence tasks, read from the 'labels.json'
    manifest of every split with the Native loader (OpenCV decoding and CPU resizing). Every window of frames is
    padded to the sequence length T by the datasets, so the default collate stacks them into [B, T, C, H, W] batches,
    or into [B, 2T, C, H, W] batches of pairs (the before window followed by the after window) for the comparison task.
    With 'Data.sequence_settings.batch_padding: "longest"', the windows of every batch are cut to its longest window
    instead (collate_longest_windows), T then being the longest window, or the longest member of a pair, of the batch.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, giving the split folders.
        model_configuration (Dict[str, Any]): The model configuration, giving the image settings, the label dictionary
            and the sequence settings.
        batch_size (int): The number of sequences (or pairs of sequences) of a batch.
        label_type (str): "sequence" (one class per sequence), "frame" (one class per frame), "selection" (one class
            per frame of a whole sequence, checked for the frame selection with the 'Selection' section) or "pair"
            (one class per pair of sequences, read from a 'pairs' manifest).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test loaders.
    """
    data_configuration = model_configuration["Data"]
    loaders = []

    # Unselected category and frames per wanted class of the frame selection, read by its sequence check
    selection_settings = model_configuration["Selection"] if label_type == "selection" else None

    # Batches padded to their longest window, a pair holding two member windows, or every window padded to the sequence
    # length by the default collate
    collate_function = None
    if data_configuration["sequence_settings"]["batch_padding"] == "longest":
        collate_function = partial(collate_longest_windows,
                                   members_per_sample=len(PAIR_MEMBERS) if label_type == "pair" else 1)

    # Build the loader of every split with the same settings
    for split_key in ["train_data_folder", "validation_data_folder", "test_data_folder"]:
        split_data_folder = str(experiment_configuration[split_key])

        # Pairs of before and after sequences, or sequences cut into windows
        if label_type == "pair":
            dataset = SequencePairDataset(data_directory=os.path.join(split_data_folder, "data"),
                                          labels_file=os.path.join(split_data_folder, "labels.json"),
                                          image_size=data_configuration["image_settings"]["size"],
                                          keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                          label_dictionary=data_configuration["label_dictionary"],
                                          sequence_settings=data_configuration["sequence_settings"])
        else:
            dataset = SequenceDataset(data_directory=os.path.join(split_data_folder, "data"),
                                      labels_file=os.path.join(split_data_folder, "labels.json"),
                                      image_size=data_configuration["image_settings"]["size"],
                                      keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                      label_dictionary=data_configuration["label_dictionary"],
                                      sequence_settings=data_configuration["sequence_settings"],
                                      label_type=label_type, selection_settings=selection_settings)
        loaders.append(build_native_loader(dataset=dataset, batch_size=batch_size,
                                           loading_configuration=data_configuration["loading"],
                                           collate_function=collate_function))

    return loaders[0], loaders[1], loaders[2]
