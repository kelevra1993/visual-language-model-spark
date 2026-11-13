"""
Sources shared by the figure families of the keypoint detection series of the tutorial videos
(tutorials/figures/keypoint_detection/figures.py): the synthetic keypoint run of tutorials/tutorial_sources.yaml and
the repository configuration sharing its model, the test image the chapters follow (test_000000.png, two hollow
capsules side by side, a trapezoid and a circle without keypoints), read by the COCO dataset of the repository with its
keypoints, the label dictionary and the channel table of the trained detector, and the colours the figures give to the
objects of that image. The test split of the run (a copy of the first 1024 test images) was removed, so the figures
read the test split and the settings of the repository configuration, refused unless its model sections, image
settings and label mapping are those of the training_configuration.yaml of the run.
"""

import json

import numpy
import torch

from pathlib import Path
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import SERIES_COLOURS
from tutorial_builder.figure_context import find_repository_root
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.data.keypoints.keypoint_categories import read_keypoint_categories
from vision_training.data.label_mapping import build_label_dictionary
from vision_training.models.keypoints.keypoint_detector import build_keypoint_channel_table
from vision_training.trainers.checkpointing import get_checkpoint_path

# Configuration of the repository the synthetic keypoint run was trained with, and the sections that define its model,
# compared with the configuration saved in the run folder before the model is rebuilt from it
REPOSITORY_CONFIGURATION_PATH = (find_repository_root() / "vision_training" / "configurations" / "keypoint_detection" /
                                 "configuration_synthetic_shapes.yaml")
MODEL_SECTIONS = ["Backbone", "HRNet", "KeypointHead"]
TASK_NAME = "keypoint_detection"

# Test image followed by the chapters, the first one of the test split and of the sample images of the run
SAMPLE_IMAGE_NAME = "test_000000.png"

# Checkpoint whose trained weights give the heatmaps, tags and candidates of the figures, and every evaluated one
CHECKPOINT_ITERATION = 6000
EVALUATED_ITERATIONS = [2000, 4000, 6000]

# Colour of every object of the test image in the figures, by its position among the objects with keypoints
OBJECT_COLOURS = [SERIES_COLOURS[0], SERIES_COLOURS[1], SERIES_COLOURS[2], SERIES_COLOURS[3], SERIES_COLOURS[4]]


def read_repository_configuration(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the repository configuration of the synthetic keypoint run and checks that it defines the model of the run,
    so that the model rebuilt from it for the figures is the trained one: the Backbone, HRNet and KeypointHead
    sections, the image settings and the label mapping must be those of the training_configuration.yaml of the run.

    Args:
        run_folder (Path): The folder of the keypoint run.

    Returns:
        Dict[str, Any]: The repository configuration, as plain YAML.

    Raises:
        ValueError: If a model section of the repository configuration differs from the one of the run.
    """
    repository_configuration = load_configuration(configuration_path=REPOSITORY_CONFIGURATION_PATH)
    run_configuration = load_configuration(configuration_path=run_folder / "training_configuration.yaml")

    # Every section defining the model, and the settings of the images and classes it was trained on
    compared_values = {section: (repository_configuration[section], run_configuration[section])
                       for section in MODEL_SECTIONS}
    compared_values["Data.image_settings"] = (repository_configuration["Data"]["image_settings"],
                                              run_configuration["Data"]["image_settings"])
    compared_values["Data.label_mapping"] = (repository_configuration["Data"]["label_mapping"],
                                             run_configuration["Data"]["label_mapping"])
    differing_sections = [name for name, (repository_value, run_value) in compared_values.items()
                          if repository_value != run_value]
    if differing_sections:
        raise ValueError(f"The sections {differing_sections} of {REPOSITORY_CONFIGURATION_PATH} differ from those of "
                         f"{run_folder / 'training_configuration.yaml'}: the figures would not show the trained model.")
    return repository_configuration


def find_test_split(configuration: Dict[str, Any]) -> Path:
    """
    Finds the test split of the synthetic keypoint dataset named by the repository configuration, whose first image
    is the first sample image of the run (the 1024 test images evaluated by the run were the first ones of this split,
    copied to a temporary folder since removed).

    Args:
        configuration (Dict[str, Any]): The repository configuration.

    Returns:
        Path: The test split folder, holding data/ and labels.json.

    Raises:
        FileNotFoundError: If the split or its labels.json is missing.
    """
    test_split = Path(configuration["ExperimentConfiguration"]["test_data_folder"])
    if not (test_split / "labels.json").exists():
        raise FileNotFoundError(f"No labels.json in the test split {test_split} of {REPOSITORY_CONFIGURATION_PATH}: "
                                f"generate the synthetic keypoint dataset first.")
    return test_split


def read_label_dictionary(configuration: Dict[str, Any]) -> Dict[Any, int]:
    """
    Builds the label dictionary of the keypoint run with the function of the configuration loader: the class index
    of every category of the mapping file, the four distractors without keypoints being absent from it.

    Args:
        configuration (Dict[str, Any]): The repository configuration.

    Returns:
        Dict[Any, int]: The class index of every category identifier, with 'background': 0.
    """
    return build_label_dictionary(label_mapping_configuration=configuration["Data"]["label_mapping"], task=TASK_NAME)


def read_channel_table(run_folder: Path, test_split: Path,
                       label_dictionary: Dict[Any, int]) -> Tuple[numpy.ndarray, List[str], Dict[int, Dict[str, Any]]]:
    """
    Reads the channel table of the trained detector, saved as a buffer in its checkpoint, and checks it against the one
    build_keypoint_channel_table gives for the keypoint categories of the test split, which names its channels.

    Args:
        run_folder (Path): The folder of the keypoint run.
        test_split (Path): The test split, whose labels.json defines the keypoints of every category.
        label_dictionary (Dict[Any, int]): The class index of every category.

    Returns:
        Tuple[numpy.ndarray, List[str], Dict[int, Dict[str, Any]]]: The [classes + 1, 18] channel of every keypoint
            of every class (-1 for none), the 'class.keypoint' name of every channel and the keypoint definition of
            every class.

    Raises:
        ValueError: If the channel table of the checkpoint differs from the one of the categories.
    """
    keypoint_categories = read_keypoint_categories(labels_file=test_split / "labels.json",
                                                   label_dictionary=label_dictionary)
    channel_table, channel_names = build_keypoint_channel_table(keypoint_categories=keypoint_categories)

    # The table registered in the checkpoint by the trained detector
    checkpoint_path = get_checkpoint_path(experiment_folder=run_folder, iteration=CHECKPOINT_ITERATION)
    checkpoint = torch.load(f=checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_table = checkpoint["model_state"]["channel_table"]
    if not torch.equal(checkpoint_table, channel_table):
        raise ValueError(f"The channel table of {checkpoint_path} differs from the one of the categories of "
                         f"{test_split / 'labels.json'}.")
    return channel_table.numpy(), channel_names, keypoint_categories


def load_test_sample(configuration: Dict[str, Any], test_split: Path, label_dictionary: Dict[Any, int],
                     maximum_keypoints: int) -> Dict[str, Any]:
    """
    Reads the test image of the chapters with the COCO dataset of the keypoint trainer (resized to the network input,
    the objects without labelled keypoint left out), and its raw records in labels.json, which the first chapter shows.

    Args:
        configuration (Dict[str, Any]): The repository configuration, with its image settings.
        test_split (Path): The test split.
        label_dictionary (Dict[Any, int]): The class index of every category.
        maximum_keypoints (int): The most keypoints of a class, the keypoint positions of every object.

    Returns:
        Dict[str, Any]: The 'sample' of the dataset (images [3, 512, 512] uint8, labels [N], keypoints [N, 18, 3]...),
            the 'image_record', its 'annotations' and the 'categories' of labels.json.

    Raises:
        FileNotFoundError: If the test image is not in the split.
    """
    image_settings = configuration["Data"]["image_settings"]
    dataset = CocoDetectionDataset(data_directory=str(test_split / "data"), labels_file=str(test_split / "labels.json"),
                                   image_size=image_settings["size"], keep_ratio=image_settings["keep_ratio"],
                                   label_mode=configuration["Data"]["label_mapping"]["mode"],
                                   label_dictionary=label_dictionary, load_keypoints=True,
                                   maximum_keypoints=maximum_keypoints, shuffle=False)

    # The image by its file name, and its records of the manifest
    image_index = next((image_index for image_index in range(len(dataset))
                        if dataset.get_image_record(image_index=image_index)[0]["file_name"] == SAMPLE_IMAGE_NAME),
                       None)
    if image_index is None:
        raise FileNotFoundError(f"No image {SAMPLE_IMAGE_NAME} in {test_split / 'labels.json'}.")
    image_record, image_annotations = dataset.get_image_record(image_index=image_index)
    return {"sample": dataset[image_index], "image_record": image_record, "annotations": image_annotations,
            "categories": dataset.categories, "image_path": test_split / "data" / SAMPLE_IMAGE_NAME}


def describe_sample_objects(sample: Dict[str, Any], keypoint_categories: Dict[int, Dict[str, Any]],
                            channel_table: numpy.ndarray) -> List[Dict[str, Any]]:
    """
    Describes the objects with keypoints of the test image for the figures and the narration: the class, the name and
    channel of every labelled keypoint, and its position in network input pixels.

    Args:
        sample (Dict[str, Any]): The dataset sample of the test image.
        keypoint_categories (Dict[int, Dict[str, Any]]): The keypoint definition of every class.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.

    Returns:
        List[Dict[str, Any]]: For every object, its 'class_index', 'class_name', 'keypoint_names', 'channels' and
            'positions' ([K, 2] list of its labelled keypoints).
    """
    objects = []
    for object_index, class_index in enumerate(sample["labels"].tolist()):
        definition = keypoint_categories[class_index]
        object_keypoints = sample["keypoints"][object_index].numpy()
        number_keypoints = len(definition["keypoints"])
        objects.append({"class_index": int(class_index), "class_name": definition["name"],
                        "keypoint_names": list(definition["keypoints"]),
                        "channels": channel_table[class_index][:number_keypoints].tolist(),
                        "positions": object_keypoints[:number_keypoints, :2].astype(float).tolist(),
                        "visibilities": object_keypoints[:number_keypoints, 2].astype(int).tolist()})
    return objects


def write_json(data: Any, output_path: Path) -> None:
    """
    Writes the data of an HTML scene (the channel table, the tags, the grouping steps) as a JSON file of the figures
    folder, which the storyboards name as the data of their page.

    Args:
        data (Any): The data, made of lists, dictionaries, numbers and texts.
        output_path (Path): The JSON file, its folder created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data=json.dumps(obj=data, indent=1), encoding="utf-8")
