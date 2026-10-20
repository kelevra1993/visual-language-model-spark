"""
Builds the training, validation and test data loaders of every task from the experiment and model configurations:
COCO detection and keypoint detection (one sample per image), classification and self-supervised learning (same
layout), semantic segmentation, and the sequence and sequence comparison tasks, every one with a PyTorch DataLoader over
its dataset. Called by the task trainers.
"""

import os
import cv2

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, Sampler

from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.detection.coco_dataset import CocoDetectionDataset, collate_function
from vision_training.data.grouped_batch_sampler import build_grouped_batch_sampler
from vision_training.data.segmentation.segmentation_dataset import SegmentationDataset
from vision_training.data.sequence.sequence_dataset import SequenceDataset, collate_padded_sequences
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS, SequencePairDataset


def build_native_loader(dataset: Dataset, batch_size: int, loading_configuration: Dict[str, Any],
                        collate_function: Optional[Callable] = None, drop_last: bool = False,
                        shuffle: bool = True, batch_sampler: Optional[Sampler[List[int]]] = None) -> DataLoader:
    """
    Builds the DataLoader of a Native dataset with the 'Data.loading' settings, shared by the loaders of every task:
    shuffled samples (in their order for the embedding extraction of scripts/embedding/cluster_image_embeddings.py,
    which writes every batch to its rows), or the batches of a batch sampler (the grouped batches of the
    self-supervised task), worker processes kept alive between epochs and evaluations (no restart at every pass), each
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
        shuffle (bool): Whether to shuffle the samples at every pass, False to read them in their order. Defaults to
            True.
        batch_sampler (Optional[Sampler[List[int]]]): The sampler giving the dataset indices of every batch, given to
            the DataLoader instead of the batch size, the shuffling and drop_last, or None for batches of batch_size
            samples. Defaults to None.

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

    # The batches of a batch sampler replace the batch size, the shuffling and the last incomplete batch setting, which
    # the DataLoader refuses next to it
    batching_arguments = {"batch_sampler": batch_sampler} if batch_sampler is not None else {
        "batch_size": batch_size, "shuffle": shuffle, "drop_last": drop_last}
    return DataLoader(dataset=dataset, num_workers=number_workers, collate_fn=collate_function, pin_memory=True,
                      persistent_workers=use_workers,
                      prefetch_factor=loading_configuration["prefetch_batches"] if use_workers else None,
                      **batching_arguments)


def get_detection_dataloader(split_name: str, split_data_folder: str, model_configuration: Dict[str, Any],
                             batch_size: int, load_instance_masks: bool = False,
                             load_original_instances: bool = False, load_keypoints: bool = False,
                             maximum_keypoints: int = 0, load_ignored_regions: bool = False) -> DataLoader:
    """
    Instantiates the DataLoader feeding one COCO split (Train, Validation or Test) to the object detection, instance
    segmentation or keypoint detection trainer: the CocoDetectionDataset decodes the images with OpenCV and resizes
    them on the CPU workers, with the instance masks rasterised there too for the instance segmentation, also at the
    original image resolution for the test split evaluated at that resolution, and the keypoints moved with the image
    for the keypoint detection.

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
        load_keypoints (bool): Whether the samples carry the keypoints of every object (keypoint detection), the
            objects without any labelled keypoint being left out. Defaults to False.
        maximum_keypoints (int): The most keypoints of a class, with load_keypoints. Defaults to 0.
        load_ignored_regions (bool): Whether the samples of the keypoint detection also carry the pixels of the
            objects left out for having no labelled keypoint, which the heatmap loss leaves out. Defaults to False.

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
                                   load_original_instances=load_original_instances,
                                   load_keypoints=load_keypoints, maximum_keypoints=maximum_keypoints,
                                   load_ignored_regions=load_ignored_regions)
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


def get_keypoint_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                             batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the training, validation and testing loaders of the keypoint detection task, read from the COCO
    'labels.json' of every split like the object detection: every sample is one whole image resized on the CPU
    workers, with the boxes, classes and keypoints of all its objects, the keypoints being padded to the most keypoints
    of a class and kept as per-image lists by the collate. Every split leaves out the objects without any labelled
    keypoint, which give no heatmap nor tag target and cannot be scored. With 'KeypointHead.ignore_unlabelled_objects',
    the training and validation samples also carry the pixels of these objects, left out of their heatmap loss; the
    test split computes no loss and does not read them.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, giving the split folders.
        model_configuration (Dict[str, Any]): The model configuration, giving the image settings, the label dictionary,
            the keypoint categories injected by the configuration loader, the loading settings and the
            'KeypointHead.ignore_unlabelled_objects' setting.
        batch_size (int): The number of images of a batch.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test loaders.
    """
    maximum_keypoints = max(len(definition["keypoints"])
                            for definition in model_configuration["Data"]["keypoint_categories"].values())
    ignore_unlabelled_objects = model_configuration["KeypointHead"]["ignore_unlabelled_objects"]
    loaders = []

    # Build the loader of every split with the same settings, the ignored regions only where a loss is computed
    for split_name, split_key in [("Train", "train_data_folder"), ("Validation", "validation_data_folder"),
                                  ("Test", "test_data_folder")]:
        load_ignored_regions = ignore_unlabelled_objects and split_name != "Test"
        loaders.append(get_detection_dataloader(split_name=split_name,
                                                split_data_folder=str(experiment_configuration[split_key]),
                                                model_configuration=model_configuration, batch_size=batch_size,
                                                load_keypoints=True, maximum_keypoints=maximum_keypoints,
                                                load_ignored_regions=load_ignored_regions))

    return loaders[0], loaders[1], loaders[2]


def get_classification_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                                   batch_size: int, self_supervised: bool = False
                                   ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the training, validation and testing loaders of the classification task, read from the
    'labels.json' manifest of every split with the Native loader (OpenCV decoding and CPU resizing). The
    self-supervised task reads the same layout: every image is kept, labelled or not, the labels of the test images
    only serving its k-nearest neighbour evaluation, and the training and validation loaders leave out the last
    incomplete batch of every pass, since its losses compare the views of the images of a batch with each other (a
    batch of a single image would give a loss without gradient). With 'Data.grouped_batches' on, their batches gather
    blocks of images sharing an attribute of their "groups" (GroupedBatchSampler), the hard negatives of each other.

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
        ValueError: If a self-supervised training or validation split holds fewer images than a batch, or if only some
            of its samples hold the keys of 'Data.grouped_batches' when the grouped batches are on.
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

        # Batches grouped by the attributes of the images for the self-supervised training and validation splits, when
        # 'Data.grouped_batches' is on and the split has groups, random batches otherwise
        batch_sampler = build_grouped_batch_sampler(
            samples=dataset.samples, grouped_batches_configuration=data_configuration["grouped_batches"],
            batch_size=batch_size, split_name=split_key) if drop_last else None
        loaders.append(build_native_loader(dataset=dataset, batch_size=batch_size,
                                           loading_configuration=data_configuration["loading"], drop_last=drop_last,
                                           batch_sampler=batch_sampler))

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
    manifest of every split with the Native loader (OpenCV decoding and CPU resizing). Every sequence is given whole
    by the datasets, those longer than 'Data.sequence_settings.maximum_sequence_length' being left out with a warning,
    and the collate pads the sequences of every batch to its longest one (collate_padded_sequences), giving
    [B, T, C, H, W] batches, or [B, 2T, C, H, W] batches of pairs (the before member followed by the after member) for
    the comparison task, T being the longest sequence, or the longest member of a pair, of the batch.

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

    # Batches padded to their longest sequence, a pair holding two members
    collate_function = partial(collate_padded_sequences,
                               members_per_sample=len(PAIR_MEMBERS) if label_type == "pair" else 1)

    # Build the loader of every split with the same settings, its name given to the warnings of the left out sequences
    for split_key, split_name in [("train_data_folder", "Training"), ("validation_data_folder", "Validation"),
                                  ("test_data_folder", "Test")]:
        split_data_folder = str(experiment_configuration[split_key])

        # Pairs of before and after sequences, or whole sequences
        if label_type == "pair":
            dataset = SequencePairDataset(data_directory=os.path.join(split_data_folder, "data"),
                                          labels_file=os.path.join(split_data_folder, "labels.json"),
                                          image_size=data_configuration["image_settings"]["size"],
                                          keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                          label_dictionary=data_configuration["label_dictionary"],
                                          sequence_settings=data_configuration["sequence_settings"],
                                          split_name=split_name)
        else:
            dataset = SequenceDataset(data_directory=os.path.join(split_data_folder, "data"),
                                      labels_file=os.path.join(split_data_folder, "labels.json"),
                                      image_size=data_configuration["image_settings"]["size"],
                                      keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                      label_dictionary=data_configuration["label_dictionary"],
                                      sequence_settings=data_configuration["sequence_settings"],
                                      label_type=label_type, split_name=split_name,
                                      selection_settings=selection_settings)
        loaders.append(build_native_loader(dataset=dataset, batch_size=batch_size,
                                           loading_configuration=data_configuration["loading"],
                                           collate_function=collate_function))

    return loaders[0], loaders[1], loaders[2]
