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
from vision_training.utilities.os_utilities import format_path_link, print_color

# Worker processes reading and resizing the images of every loader of the trainings, the training, validation and test
# loaders each having their own, and the batches prepared in advance by every worker
NUMBER_LOADER_WORKERS = 2
PREFETCH_BATCHES = 2


def build_native_loader(dataset: Dataset, batch_size: int, collate_function: Optional[Callable] = None,
                        drop_last: bool = False, shuffle: bool = True,
                        batch_sampler: Optional[Sampler[List[int]]] = None,
                        number_workers: int = NUMBER_LOADER_WORKERS) -> DataLoader:
    """
    Builds the DataLoader of a Native dataset, shared by the loaders of every task: shuffled samples (in their order
    for the embedding extraction of scripts/embedding/cluster_image_embeddings.py, which writes every batch to its
    rows), or the batches of a batch sampler (the grouped batches of the self-supervised task), NUMBER_LOADER_WORKERS
    worker processes each preparing PREFETCH_BATCHES batches in advance and kept alive between epochs and evaluations
    (no restart at every pass), each decoding with a single OpenCV thread, and batches copied to pinned memory for
    faster transfers to the GPU. With 0 workers, the samples are loaded in the main process, for debugging. The single
    OpenCV thread is set in this process, which then also runs OpenCV on one thread.

    Args:
        dataset (Dataset): The Native dataset of the split.
        batch_size (int): The number of samples of a batch.
        collate_function (Optional[Callable]): The function batching the samples, None for the default one.
        drop_last (bool): Whether to leave out the last incomplete batch of every pass over the split, for the losses
            comparing the samples of a batch with each other (self-supervised task) and the training of the layers
            normalized by batch statistics (classification heads). Defaults to False.
        shuffle (bool): Whether to shuffle the samples at every pass, False to read them in their order. Defaults to
            True.
        batch_sampler (Optional[Sampler[List[int]]]): The sampler giving the dataset indices of every batch, given to
            the DataLoader instead of the batch size, the shuffling and drop_last, or None for batches of batch_size
            samples. Defaults to None.
        number_workers (int): The number of worker processes, 0 loading in the main process, for the embedding
            extraction of scripts/embedding/cluster_image_embeddings.py and the tests. Defaults to
            NUMBER_LOADER_WORKERS, the number of every training.

    Returns:
        DataLoader: The loader of the split.
    """
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
                      prefetch_factor=PREFETCH_BATCHES if use_workers else None,
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
        model_configuration (Dict[str, Any]): The model-specific configuration, giving the image settings and the
            label dictionary.
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
    return build_native_loader(dataset=dataset, batch_size=batch_size, collate_function=collate_function)


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
        model_configuration (Dict[str, Any]): The model-specific configuration, giving the image settings and the
            label dictionary.
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

    # Build the training, validation and test loaders, each with NUMBER_LOADER_WORKERS worker processes
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
            the keypoint categories injected by the configuration loader and the
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
                                   batch_size: int, self_supervised: bool = False,
                                   drop_last_training_batch: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the training, validation and testing loaders of the classification task, read from the
    'labels.json' manifest of every split with the Native loader (OpenCV decoding and CPU resizing). The
    self-supervised task reads the same layout: every image is kept, labelled or not, the labels of the test images
    only serving its k-nearest neighbour evaluation, and the training and validation loaders leave out the last
    incomplete batch of every pass, since its losses compare the views of the images of a batch with each other (a
    batch of a single image would give a loss without gradient). With 'Data.grouped_batches' on, their batches gather
    blocks of images sharing an attribute of their "groups" (GroupedBatchSampler), the hard negatives of each other.
    Once every split is built, the samples without a single mapped category of the splits, then the keys and batches
    of their grouped batches, are printed in one block each, one line per split and key. The classifier leaves out the
    last incomplete training batch when its training needs batch statistics (drop_last_training_batch), which a last
    batch of a single image cannot give.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, giving the split folders.
        model_configuration (Dict[str, Any]): The model configuration, giving the image settings, the label dictionary
            and the output mode of the classification head.
        batch_size (int): The number of samples of a batch.
        self_supervised (bool): Whether the loaders feed the self-supervised trainer, which has no classification
            head (multi-class labels). Defaults to False.
        drop_last_training_batch (bool): Whether the training loader leaves out the last incomplete batch of every
            pass, for a batch-normalized classification head, which a last batch of a single image would break
            (ClassificationTrainer.drops_last_training_batch). Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test loaders.

    Raises:
        ValueError: If a split whose batches must be full (self-supervised training and validation splits, training
            split with drop_last_training_batch) holds fewer images than a batch, or if only some of the samples of a
            self-supervised split hold the keys of 'Data.grouped_batches' when the grouped batches are on.
    """
    data_configuration = model_configuration["Data"]
    output_mode = "multi_class" if self_supervised else model_configuration["ClassificationHead"]["output_mode"]
    loaders = []

    # Console lines of the three splits, printed once every split is built so that the samples without a single mapped
    # category and the grouped batches come in one block each instead of interleaved split by split
    unlabelled_sample_lines = []
    grouped_batch_lines = []

    # Build the loader of every split with the same settings
    for split_key, split_label in [("train_data_folder", "Train"), ("validation_data_folder", "Validation"),
                                   ("test_data_folder", "Test")]:
        split_data_folder = str(experiment_configuration[split_key])
        labels_file = os.path.join(split_data_folder, "labels.json")
        dataset = ClassificationDataset(data_directory=os.path.join(split_data_folder, "data"), labels_file=labels_file,
                                        image_size=data_configuration["image_settings"]["size"],
                                        keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                        label_dictionary=data_configuration["label_dictionary"],
                                        output_mode=output_mode, keep_unlabelled_samples=self_supervised)
        if dataset.number_unlabelled_samples:
            unlabelled_sample_lines.append(f"{split_label} : {dataset.number_unlabelled_samples} samples "
                                           f"({format_path_link(path=labels_file)})")

        # Full batches for the self-supervised losses, the test split being encoded image by image, and for the training
        # batches of a classifier needing batch statistics
        self_supervised_split = self_supervised and split_key != "test_data_folder"
        drop_last = self_supervised_split or (drop_last_training_batch and split_key == "train_data_folder")
        if drop_last and len(dataset) < batch_size:
            raise ValueError(f"The '{split_key}' split holds {len(dataset)} images, fewer than the batch size "
                             f"{batch_size}: its batches must be full, for the self-supervised losses or for the batch "
                             f"statistics of the classifier.")

        # Batches grouped by the attributes of the images for the self-supervised training and validation splits, when
        # 'Data.grouped_batches' is on and the split has groups, random batches otherwise
        batch_sampler, sampler_report_lines = build_grouped_batch_sampler(
            samples=dataset.samples, grouped_batches_configuration=data_configuration["grouped_batches"],
            batch_size=batch_size, split_name=split_key) if self_supervised_split else (None, [])
        grouped_batch_lines.extend(f"{split_label} {report_line}" for report_line in sampler_report_lines)
        loaders.append(build_native_loader(dataset=dataset, batch_size=batch_size, drop_last=drop_last,
                                           batch_sampler=batch_sampler))

    # Report the samples without a single mapped category of every split, kept without class by the self-supervised
    # task and left out by the classifier, then the keys and batches of the grouped batches, one line each
    if unlabelled_sample_lines:
        unlabelled_sample_title = ("Samples Without A Single Mapped Category, Kept Without Class" if self_supervised
                                   else "Samples Of Unmapped Categories, Left Out")
        print_color(color="yellow", output=f"{unlabelled_sample_title} :")
        print_color(color="yellow", output="\n".join(unlabelled_sample_lines), indent=1)
    if grouped_batch_lines:
        print_color(color="blue", output="Grouped Batches :")
        print_color(color="blue", output="\n".join(grouped_batch_lines), indent=1)

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
        model_configuration (Dict[str, Any]): The model configuration, giving the image settings, the label mode and
            the label dictionary.
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
        loaders.append(build_native_loader(dataset=dataset, batch_size=split_batch_size))

    return loaders[0], loaders[1], loaders[2]


def get_sequence_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                             batch_size: int, label_type: str,
                             drop_last_training_batch: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
        drop_last_training_batch (bool): Whether the training loader leaves out the last incomplete batch of every
            pass, for a batch-normalized classification head, which a last batch of a single sequence would break
            (ClassificationTrainer.drops_last_training_batch). Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test loaders.

    Raises:
        ValueError: If the training split holds fewer samples than a batch with drop_last_training_batch.
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
            # Whole sequences labelled per sequence, per frame or for the frame selection
            dataset = SequenceDataset(data_directory=os.path.join(split_data_folder, "data"),
                                      labels_file=os.path.join(split_data_folder, "labels.json"),
                                      image_size=data_configuration["image_settings"]["size"],
                                      keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                      label_dictionary=data_configuration["label_dictionary"],
                                      sequence_settings=data_configuration["sequence_settings"],
                                      label_type=label_type, split_name=split_name,
                                      selection_settings=selection_settings)
        # The training loader of a head needing batch statistics leaves out its last incomplete batch, so its split
        # must hold at least one full batch
        drop_last = drop_last_training_batch and split_key == "train_data_folder"
        if drop_last and len(dataset) < batch_size:
            raise ValueError(f"The '{split_key}' split holds {len(dataset)} samples, fewer than the batch size "
                             f"{batch_size}: its batches must be full, for the batch statistics of the classifier.")

        # Wrap the dataset in a Native loader padding every batch to its longest sequence
        loaders.append(build_native_loader(dataset=dataset, batch_size=batch_size,
                                           collate_function=collate_function, drop_last=drop_last))

    return loaders[0], loaders[1], loaders[2]
