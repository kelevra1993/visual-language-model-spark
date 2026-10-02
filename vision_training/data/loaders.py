import os

from nvidia.dali.plugin.pytorch import DALIGenericIterator
from typing import Any, Dict, Tuple, Union
from torch.utils.data import DataLoader

from vision_training.data.detection.coco_dali_pipeline import DALIDataloaderWrapper, dali_pipeline
from vision_training.data.detection.coco_dataset import CocoDetectionDataset, collate_function


def get_dataloader(split_name: str, split_data_folder: str, model_configuration: Dict[str, Any], batch_size: int,
                   number_of_workers: int) -> Union[DataLoader, DALIDataloaderWrapper]:
    """
    Instantiates the DataLoader feeding one split (Train, Validation or Test) to the Trainer.
    The GPU-accelerated DALI pipeline is used when activated, otherwise the Native PyTorch dataset decodes the
    images with OpenCV and resizes them on the CPU. Both return the same batch dictionary format.

    Args:
        split_name (str): The name of the split (e.g., 'Train', 'Validation', 'Test').
        split_data_folder (str): The root folder for this split, containing a 'data/' subfolder and a 'labels.json'.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.
        number_of_workers (int): The number of parallel workers (DALI threads or DataLoader subprocesses).

    Returns:
        Union[DataLoader, DALIDataloaderWrapper]: The configured loader for the requested split.
    """
    data_configuration = model_configuration["Data"]
    use_dali = data_configuration["DALI"]["activated"]

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
                                 keep_ratio=keep_ratio, batch_size=batch_size, num_threads=number_of_workers,
                                 device_id=0)
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

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=number_of_workers,
                      collate_fn=collate_function)


def get_dataloaders(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any], batch_size: int,
                    number_of_workers: int = 4) -> Tuple[Union[DataLoader, DALIDataloaderWrapper],
                                                         Union[DataLoader, DALIDataloaderWrapper],
                                                         Union[DataLoader, DALIDataloaderWrapper]]:
    """
    Initializes the training, validation and testing loaders consumed by the Trainer.
    This function is the entry point of the data loading subsystem: it resolves each split folder from the
    experiment configuration and builds the matching loader (DALI or Native) for it.

    Args:
        experiment_configuration (Dict[str, Any]): The overall experiment configuration including data paths.
        model_configuration (Dict[str, Any]): The model-specific configuration detailing data loading preferences.
        batch_size (int): The number of samples to include in each batch.
        number_of_workers (int): The number of parallel workers to allocate for data loading. Defaults to 4.

    Returns:
        Tuple[Union[DataLoader, DALIDataloaderWrapper], ...]: The loaders for the training, validation and
            testing splits respectively.
    """
    # Map each split name to its corresponding root folder containing images and labels.json
    split_folder_mapping = {"Train": str(experiment_configuration["train_data_folder"]),
                            "Validation": str(experiment_configuration["validation_data_folder"]),
                            "Test": str(experiment_configuration["test_data_folder"])}

    # Build the training, validation and test loaders with the same data loading settings
    train_loader = get_dataloader(split_name="Train", split_data_folder=split_folder_mapping["Train"],
                                  model_configuration=model_configuration, batch_size=batch_size,
                                  number_of_workers=number_of_workers)
    validation_loader = get_dataloader(split_name="Validation", split_data_folder=split_folder_mapping["Validation"],
                                       model_configuration=model_configuration, batch_size=batch_size,
                                       number_of_workers=number_of_workers)
    test_loader = get_dataloader(split_name="Test", split_data_folder=split_folder_mapping["Test"],
                                 model_configuration=model_configuration, batch_size=batch_size,
                                 number_of_workers=number_of_workers)

    return train_loader, validation_loader, test_loader
