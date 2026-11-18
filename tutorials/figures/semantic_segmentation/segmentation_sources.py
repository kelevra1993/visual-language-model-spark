"""
Sources shared by the figure families of the semantic segmentation series of the tutorial videos
(tutorials/figures/semantic_segmentation/figures.py): the synthetic segmentation run of tutorials/tutorial_sources.yaml
and its saved configuration, its test split, the test image the chapters follow (test_000028.png, a sample image of
every evaluation of the run: a circle, a rectangle, a pentagon and an octagon), read with the segmentation dataset of
the repository, the label dictionary and class names of the run, and the class colours of its sample images. The saved
configuration is read with read_run_configuration of figure_utilities.py, which points the dataset paths of a run read
from the committed snapshots (tutorials/run_snapshots) at the dataset files of the snapshots; such a run has no
checkpoint (has_checkpoint), its inference outputs being committed with the snapshots.
"""

import json

import numpy

from pathlib import Path
from typing import Any, Dict

from vision_training.data.label_mapping import build_label_dictionary
from vision_training.data.segmentation.segmentation_dataset import SegmentationDataset
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.trainers.checkpointing import get_checkpoint_file
from vision_training.utilities.visualization import build_class_colours

# Task of the run, and the test image followed by the chapters, one of the eight sample images of the run
TASK_NAME = "semantic_segmentation"
SAMPLE_IMAGE_NAME = "test_000028.png"

# Checkpoints evaluated by the run, the last one giving the activations of the figures
EVALUATED_ITERATIONS = [1000, 2000]
CHECKPOINT_ITERATION = 2000

# Folder of the evaluations the figures read, inside the run folder: both checkpoints evaluated again with the current
# code by scripts/evaluation/evaluate_checkpoint.py, the Evaluation folder of the training having been written by an
# older evaluation (no recall interval, the title of the curve images on one line)
EVALUATION_FOLDER_NAME = CHECKPOINT_EVALUATION_FOLDER_NAME

# Configuration of the task shipped with the repository, relative to its root, whose decoder section the decoder scene
# of chapter 1 shows next to the shortcuts it would get
SYNTHETIC_CONFIGURATION_PATH = ("vision_training/configurations/semantic_segmentation/"
                                "configuration_synthetic_shapes.yaml")


def has_checkpoint(run_folder: Path) -> bool:
    """
    Tells whether the segmentation run holds the checkpoint of CHECKPOINT_ITERATION, which the inference of the model
    figures reads; a run read from the committed snapshots has none, its inference outputs being committed.

    Args:
        run_folder (Path): The folder of the segmentation run.

    Returns:
        bool: Whether the checkpoint file exists.
    """
    return get_checkpoint_file(weights_directory=run_folder / "Weights", iteration=CHECKPOINT_ITERATION).exists()


def find_test_split(configuration: Dict[str, Any]) -> Path:
    """
    Finds the test split of the segmentation run, named by its configuration.

    Args:
        configuration (Dict[str, Any]): The configuration of the run.

    Returns:
        Path: The test split folder, holding data/, masks/ and labels.json.

    Raises:
        FileNotFoundError: If the split or its labels.json is missing.
    """
    test_split = Path(configuration["ExperimentConfiguration"]["test_data_folder"])
    if not (test_split / "labels.json").exists():
        raise FileNotFoundError(f"No labels.json in the test split {test_split} of the segmentation run: generate the "
                                f"synthetic segmentation dataset first.")
    return test_split


def read_label_dictionary(configuration: Dict[str, Any]) -> Dict[Any, int]:
    """
    Builds the label dictionary of the segmentation run with the function of the configuration loader: the class of
    every category value of the masks, 'background' being class 0.

    Args:
        configuration (Dict[str, Any]): The configuration of the run.

    Returns:
        Dict[Any, int]: The class index of every category value.
    """
    return build_label_dictionary(label_mapping_configuration=configuration["Data"]["label_mapping"], task=TASK_NAME)


def load_test_sample(configuration: Dict[str, Any], test_split: Path,
                     label_dictionary: Dict[Any, int]) -> Dict[str, Any]:
    """
    Reads the test image of the chapters with the segmentation dataset of the trainer (the image letterboxed to the
    network input, its mask converted to class indices), with its manifest entry and the categories of the split.

    Args:
        configuration (Dict[str, Any]): The configuration of the run, with its image settings.
        test_split (Path): The test split.
        label_dictionary (Dict[Any, int]): The class of every category value.

    Returns:
        Dict[str, Any]: The 'sample' of the dataset (images [3, 512, 512] uint8, masks [512, 512] int64), the
            'dataset', its 'manifest_entry', the 'categories' and the paths of the image and of its mask file.

    Raises:
        FileNotFoundError: If the test image is not in the split.
    """
    image_settings = configuration["Data"]["image_settings"]
    dataset = SegmentationDataset(data_directory=str(test_split / "data"), masks_directory=str(test_split / "masks"),
                                  labels_file=str(test_split / "labels.json"), image_size=image_settings["size"],
                                  keep_ratio=image_settings["keep_ratio"],
                                  label_mode=configuration["Data"]["label_mapping"]["mode"],
                                  label_dictionary=label_dictionary,
                                  tiling_configuration=configuration["Data"]["tiling"], tile_sampling="whole",
                                  shuffle=False)
    manifest = json.loads((test_split / "labels.json").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["samples"] if entry["file_name"] == SAMPLE_IMAGE_NAME]
    if not entries:
        raise FileNotFoundError(f"No image {SAMPLE_IMAGE_NAME} in {test_split / 'labels.json'}.")
    sample_index = next(index for index, sample in enumerate(dataset.samples)
                        if sample["file_name"] == SAMPLE_IMAGE_NAME)
    return {"sample": dataset[sample_index], "dataset": dataset, "manifest_entry": entries[0],
            "categories": manifest["categories"], "number_samples": len(manifest["samples"]),
            "image_path": test_split / "data" / SAMPLE_IMAGE_NAME,
            "mask_path": test_split / "masks" / entries[0]["mask_file_name"]}


def build_class_palette(number_classes: int) -> numpy.ndarray:
    """
    Gives the RGB colour of every class, those of the sample images of the trainer (build_class_colours, BGR).

    Args:
        number_classes (int): The number of classes, the background included.

    Returns:
        numpy.ndarray: The [K, 3] uint8 RGB colours.
    """
    return numpy.asarray(build_class_colours(number_classes=number_classes))[:, ::-1].astype(numpy.uint8)
