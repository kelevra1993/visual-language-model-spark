"""
Builds the training, validation and test data loaders of every task from the experiment and model configurations:
COCO detection with the DALI or Native loader, classification, and the sequence and sequence comparison tasks with
the Native loader. Called by the task trainers.
"""

import os
import cv2

from nvidia.dali.plugin.pytorch import DALIGenericIterator
from typing import Any, Callable, Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset

from vision_training.data.detection.coco_dali_pipeline import DALIDataloaderWrapper, dali_pipeline
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.detection.coco_dataset import CocoDetectionDataset, collate_function
from vision_training.data.sequence.sequence_dataset import SequenceDataset
from vision_training.data.sequence.sequence_pair_dataset import SequencePairDataset


def build_native_loader(dataset: Dataset, batch_size: int, loading_configuration: Dict[str, Any],
                        collate_function: Optional[Callable] = None) -> DataLoader:
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
                      collate_fn=collate_function, pin_memory=True, persistent_workers=use_workers,
                      prefetch_factor=loading_configuration["prefetch_batches"] if use_workers else None)


def get_detection_dataloader(split_name: str, split_data_folder: str, model_configuration: Dict[str, Any],
                             batch_size: int) -> Union[DataLoader, DALIDataloaderWrapper]:
    """
    Instantiates the DataLoader feeding one COCO split (Train, Validation or Test) to the object detection trainer.
    The GPU-accelerated DALI pipeline is used when activated, otherwise the Native PyTorch dataset decodes the
    images with OpenCV and resizes them on the CPU. Both return the same batch dictionary format.

    Args:
        split_name (str): The name of the split (e.g., 'Train', 'Validation', 'Test').
        split_data_folder (str): The root folder for this split, containing a 'data/' subfolder and a 'labels.json'.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.

    Returns:
        Union[DataLoader, DALIDataloaderWrapper]: The configured loader for the requested split.
    """
    data_configuration = model_configuration["Data"]
    use_dali = data_configuration["DALI"]["activated"]
    loading_configuration = data_configuration["loading"]

    image_settings = data_configuration["image_settings"]
    image_size = image_settings["size"]
    keep_ratio = image_settings["keep_ratio"]

    label_mode = data_configuration["label_mode"]
    label_dictionary = data_configuration["label_dictionary"]

    # Derive the physical image directory and annotation file from the split folder convention
    data_directory = os.path.join(split_data_folder, "data")
    labels_file = os.path.join(split_data_folder, "labels.json")

    # If DALI is activated, initialize the GPU-accelerated pipeline
    if use_dali:
        pipeline = dali_pipeline(data_directory=data_directory, annotations_file=labels_file, image_size=image_size,
                                 keep_ratio=keep_ratio, batch_size=batch_size,
                                 num_threads=loading_configuration["number_workers"],
                                 prefetch_queue_depth=loading_configuration["prefetch_batches"], device_id=0)
        pipeline.build()

        dali_iterator = DALIGenericIterator(pipelines=[pipeline],
                                            output_map=["images", "bounding_boxes", "labels", "shapes", "image_ids"],
                                            reader_name="Reader", auto_reset=True)

        # Wrap the DALI iterator so it behaves exactly like our standard PyTorch DataLoader outputs
        return DALIDataloaderWrapper(dali_iterator=dali_iterator, image_size=image_size, keep_ratio=keep_ratio,
                                     label_mode=label_mode, label_dictionary=label_dictionary)

    # Otherwise fall back to the Native PyTorch dataset, decoding and resizing the images on the CPU
    dataset = CocoDetectionDataset(data_directory=data_directory, labels_file=labels_file, image_size=image_size,
                            keep_ratio=keep_ratio, label_mode=label_mode, label_dictionary=label_dictionary)

    return build_native_loader(dataset=dataset, batch_size=batch_size, loading_configuration=loading_configuration,
                               collate_function=collate_function)


def get_detection_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                              batch_size: int) -> Tuple[Union[DataLoader, DALIDataloaderWrapper],
                                                        Union[DataLoader, DALIDataloaderWrapper],
                                                        Union[DataLoader, DALIDataloaderWrapper]]:
    """
    Initializes the training, validation and testing loaders of the object detection task, consumed by the detection
    trainer: it resolves each split folder from the experiment configuration and builds the matching COCO loader
    (DALI or Native) for it.

    Args:
        experiment_configuration (Dict[str, Any]): The overall experiment configuration including data paths.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.

    Returns:
        Tuple[Union[DataLoader, DALIDataloaderWrapper], ...]: The loaders for the training, validation and
            testing splits respectively.
    """
    # Map each split name to its corresponding root folder containing images and labels.json
    split_folder_mapping = {"Train": str(experiment_configuration["train_data_folder"]),
                            "Validation": str(experiment_configuration["validation_data_folder"]),
                            "Test": str(experiment_configuration["test_data_folder"])}

    # Build the training, validation and test loaders with the same data loading settings
    train_loader = get_detection_dataloader(split_name="Train", split_data_folder=split_folder_mapping["Train"],
                                            model_configuration=model_configuration, batch_size=batch_size)
    validation_loader = get_detection_dataloader(split_name="Validation",
                                                 split_data_folder=split_folder_mapping["Validation"],
                                                 model_configuration=model_configuration, batch_size=batch_size)
    test_loader = get_detection_dataloader(split_name="Test", split_data_folder=split_folder_mapping["Test"],
                                           model_configuration=model_configuration, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


def get_classification_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                                   batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the training, validation and testing loaders of the classification task, read from the
    'labels.json' manifest of every split with the Native loader (OpenCV decoding and CPU resizing).

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, giving the split folders.
        model_configuration (Dict[str, Any]): The model configuration, giving the image settings, the label dictionary
            and the output mode of the classification head.
        batch_size (int): The number of samples of a batch.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test loaders.
    """
    data_configuration = model_configuration["Data"]
    loaders = []

    # Build the loader of every split with the same settings
    for split_key in ["train_data_folder", "validation_data_folder", "test_data_folder"]:
        split_data_folder = str(experiment_configuration[split_key])
        dataset = ClassificationDataset(data_directory=os.path.join(split_data_folder, "data"),
                                        labels_file=os.path.join(split_data_folder, "labels.json"),
                                        image_size=data_configuration["image_settings"]["size"],
                                        keep_ratio=data_configuration["image_settings"]["keep_ratio"],
                                        label_dictionary=data_configuration["label_dictionary"],
                                        output_mode=model_configuration["ClassificationHead"]["output_mode"])
        loaders.append(build_native_loader(dataset=dataset, batch_size=batch_size,
                                           loading_configuration=data_configuration["loading"]))

    return loaders[0], loaders[1], loaders[2]


def get_sequence_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                             batch_size: int, label_type: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the training, validation and testing loaders of the sequence tasks, read from the 'labels.json'
    manifest of every split with the Native loader (OpenCV decoding and CPU resizing). Every window of frames is
    padded to the sequence length, so the default collate stacks them into [B, T, C, H, W] batches, or into
    [B, 2T, C, H, W] batches of pairs (the before window followed by the after window) for the comparison task.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, giving the split folders.
        model_configuration (Dict[str, Any]): The model configuration, giving the image settings, the label dictionary
            and the sequence settings.
        batch_size (int): The number of sequences (or pairs of sequences) of a batch.
        label_type (str): "sequence" (one class per sequence), "frame" (one class per frame) or "pair" (one class per
            pair of sequences, read from a 'pairs' manifest).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: The training, validation and test loaders.
    """
    data_configuration = model_configuration["Data"]
    loaders = []

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
                                      label_type=label_type)
        loaders.append(build_native_loader(dataset=dataset, batch_size=batch_size,
                                           loading_configuration=data_configuration["loading"]))

    return loaders[0], loaders[1], loaders[2]
