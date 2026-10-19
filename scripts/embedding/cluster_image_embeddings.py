"""
When run, embeds the images of datasets with a trained run of vision_training/main.py and clusters them, to pick
diverse representatives of datasets made of video frames, many of them nearly identical, and train a smaller model on
them. The settings are the variables of the main block, and the script runs three steps:
- Checks, before any inference: the asked convolution blocks exist in the backbone of the run (any task) and give at
  most MAXIMUM_BLOCK_VALUES values per image, the asked hidden fully connected layers exist in its classification head
  (classification runs), both counted from 1, the clustering settings are valid, and every data source exists and
  holds images.
- Extraction: one pass of the model over every data source (the train, validation and test splits of the run by
  default, the splits of another configuration file, a split folder of any task or a folder of plain images), the
  whole output of every asked block before its pooling and the output of every asked hidden layer stored as one
  float32 matrix per layer in '<output_root>/<run name>/Iteration_<n>[_EMA]/<source>/'. Its 'extraction.json' records
  what produced them, so that a relaunch loads them instead of running the model again, computes only the layers not
  stored yet, continues an interrupted extraction, and refuses the features of another checkpoint or of changed images.
- Clustering of every layer of every source, per category for the classification manifests: the vectors normalised to
  length 1, reduced with PCA, clustered with HDBSCAN, DBSCAN or k-means of scikit-learn, and written in
  'clusters/<layer>/<clustering>/' as CSV files, a folder of links and a mosaic per cluster, and the subset of the
  representatives of every cluster with the noise images, next to a random subset and a subset evenly spaced in the
  groups of the images (e.g. the frames of every video) of the same size, ready to train the smaller model on each.
Every image read (extraction, mosaics) and every feature matrix of a clustered source is released from the page cache
of Linux, which would otherwise fill the memory the GPU shares on the DGX Spark and stop other processes from starting.

Timings on the DGX Spark (2026-10-19), to extrapolate to other datasets: the ficha empty bins train split (11,772
empty_bin and 31,805 not_empty frames of 1920 x 1080 pixels) embedded by its classification run at 512 pixels
(iteration 200), a training running on the same machine. The extraction of block 6 took 812 Seconds (53.7 Images Per
Second, bound by the decoding of the frames). HDBSCAN (min_cluster_size 10, min_samples 5, eom) builds its spanning
tree on a single core over all the pairs of images, its time growing with the square of the number of images and with
the number of dimensions: about 0.6 Seconds per dimension for 31,805 images, 4 times more for twice the images. The
three layers at PCA 200 took 12.7 minutes, the mosaics, links and subsets included. Times are in Seconds, a dash
marking a layer clustered as it is, with fewer values than asked (512 for fully_connected_1, 256 for
fully_connected_2), and the agreement is the adjusted Rand index against the clustering at PCA 800 (1 for the same
clusters, the large clusters weighing the most). PCA 100 and 50 were not measured.

| Layer             | Category  | PCA | Variance | PCA Time | HDBSCAN Time | Clusters | Noise  | Subset | Agreement |
|-------------------|-----------|-----|----------|----------|--------------|----------|--------|--------|-----------|
| block_6           | empty_bin | 800 | 95.1 %   | 81       | 61           | 66       | 5.5 %  | 975    | 1.000     |
| block_6           | empty_bin | 400 | 92.6 %   | 44       | 28           | 80       | 6.4 %  | 1,151  | 0.832     |
| block_6           | empty_bin | 200 | 89.8 %   | 28       | 14           | 84       | 6.1 %  | 1,133  | 0.812     |
| block_6           | not_empty | 800 | 93.4 %   | 372      | 487          | 358      | 5.4 %  | 3,513  | 1.000     |
| block_6           | not_empty | 400 | 90.3 %   | 85       | 240          | 329      | 3.8 %  | 2,856  | 0.863     |
| block_6           | not_empty | 200 | 86.1 %   | 78       | 116          | 319      | 4.3 %  | 2,967  | 0.827     |
| fully_connected_1 | empty_bin | 800 | -        | -        | 38           | 54       | 9.2 %  | 1,354  | 1.000     |
| fully_connected_1 | empty_bin | 400 | 100.0 %  | 1        | 29           | 54       | 9.1 %  | 1,345  | 0.998     |
| fully_connected_1 | empty_bin | 200 | 99.9 %   | 3        | 16           | 54       | 9.2 %  | 1,356  | 0.985     |
| fully_connected_1 | not_empty | 800 | -        | -        | 358          | 254      | 9.1 %  | 4,176  | 1.000     |
| fully_connected_1 | not_empty | 400 | 100.0 %  | 2        | 233          | 255      | 9.2 %  | 4,201  | 0.995     |
| fully_connected_1 | not_empty | 200 | 99.8 %   | 3        | 116          | 249      | 8.7 %  | 4,006  | 0.965     |
| fully_connected_2 | empty_bin | 800 | -        | -        | 17           | 50       | 10.7 % | 1,515  | 1.000     |
| fully_connected_2 | empty_bin | 400 | -        | -        | 17           | 50       | 10.7 % | 1,515  | 1.000     |
| fully_connected_2 | empty_bin | 200 | 100.0 %  | < 1      | 16           | 49       | 10.6 % | 1,489  | 0.993     |
| fully_connected_2 | not_empty | 800 | -        | -        | 151          | 244      | 12.1 % | 5,064  | 1.000     |
| fully_connected_2 | not_empty | 400 | -        | -        | 150          | 244      | 12.1 % | 5,064  | 1.000     |
| fully_connected_2 | not_empty | 200 | 100.0 %  | < 1      | 126          | 245      | 12.1 % | 5,061  | 0.994     |
"""

import csv
import hashlib
import json
import os
import re
import shutil
import time

import cv2
import numpy as np
import torch

from operator import itemgetter
from pathlib import Path
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

from vision_training.configuration_processing.configuration_loader import (load_configuration,
                                                                           load_experiment_configuration)
from vision_training.data.data_record import list_manifest_images
from vision_training.data.loaders import build_native_loader
from vision_training.data.preprocessing import preprocess_image
from vision_training.losses.classification_losses import convert_logits_to_probabilities
from vision_training.models.components.sequence_normalization import SequenceBatchNormalization
from vision_training.models.model_loading import load_trained_model
from vision_training.models.model_transfer import read_source_class_names
from vision_training.trainers.checkpointing import get_checkpoint_file
from vision_training.utilities.os_utilities import (format_path_link, get_image_paths, link_folder, print_color,
                                                    read_json)
from vision_training.utilities.tensor_utilities import get_device
from vision_training.utilities.visualization import add_caption_bar, build_frame_grid, convert_image_to_array

# Task whose head gives the hidden fully connected layers that can be embedded
CLASSIFICATION_TASK = "classification"

# Data source standing for the train, validation and test splits of the embedding run
RUN_DATA_SOURCE = "run"

# Keys of the split folders in the 'ExperimentConfiguration' section, with the role of every split
SPLIT_ROLES = {"train_data_folder": "train", "validation_data_folder": "validation", "test_data_folder": "test"}

# Number of trailing components of the path of a data source naming its extraction folder, e.g.
# 'empty_bin_classification_data-classification_dataset_v1-train', so that the splits of two datasets differ
SOURCE_NAME_COMPONENTS = 3

# Files of the extraction folder of a data source
EXTRACTION_RECORD_FILE_NAME = "extraction.json"
IMAGES_FILE_NAME = "images.csv"

# Output holding the most probable class and its probability for every image, for the classification runs
PREDICTIONS_OUTPUT = "predictions"

# Largest number of values per image of an asked block (channels x height x width of its output before its pooling),
# so that the matrix of a category and the PCA fitted on it fit in memory: block 6 of the ficha runs at 512 pixels
# (128 x 16 x 16 = 32,768 values) is allowed, block 5 (128 x 32 x 32 = 131,072, 16.7 Gigabytes for 31,805 images) is
# refused
MAXIMUM_BLOCK_VALUES = 32768

# Entries of the extraction record that must be the same for stored features to be reused
FINGERPRINT_KEYS = ["checkpoint_sha256", "use_averaged_weights", "image_size", "keep_ratio", "source_folder",
                    "images_sha256", "number_images"]

# Clustering algorithms, with the parameters every one of them requires in the settings
CLUSTERING_PARAMETERS = {"hdbscan": ["min_cluster_size", "min_samples", "cluster_selection_method"],
                         "dbscan": ["eps", "min_samples"], "kmeans": ["number_clusters", "number_initializations"]}

# Ways HDBSCAN selects its clusters in its cluster tree: the most stable ones, or the leaves (smaller, more uniform)
HDBSCAN_SELECTION_METHODS = ["eom", "leaf"]

# Noise images kept in the subset, besides a number of noise images drawn at random
NOISE_SELECTIONS = ["all", "none"]

# Cluster label of the images in no cluster
NOISE_CLUSTER = -1

# Category of every image of a source without one category per image, clustered as a whole
ALL_IMAGES_CATEGORY = "all"

# Group of the images the group pattern does not match
NO_GROUP = "(no group)"

# Subsets written next to every clustering: the cluster representatives and the two baselines of the same size
CLUSTER_SUBSET_FOLDER_NAME = "subset"
RANDOM_SUBSET_FOLDER_NAME = "subset_random"
GROUP_SUBSET_FOLDER_NAME = "subset_evenly_spaced_per_group"

# File listing the chosen images of a subset, whatever the format of its source
SELECTED_IMAGES_FILE_NAME = "selected_images.csv"

# Height of an image in the mosaics of the clusters, in pixels, its width following the ratio of the first image of
# the cluster within a quarter and four times the height
MOSAIC_THUMBNAIL_HEIGHT = 160


def is_positive_integer(value: Any) -> bool:
    """
    Tells whether a setting is a positive integer, a boolean not counting as one, for the checks of the layer indices
    and of the clustering settings.

    Args:
        value (Any): The value of the setting.

    Returns:
        bool: True for an integer of at least 1.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def compute_file_sha256(file_path: Path) -> str:
    """
    Computes the SHA-256 of a file, which changes with its content even at the same path, for the record of an
    extraction: the checkpoint of the embedding run and the 'labels.json' of a split.

    Args:
        file_path (Path): The file.

    Returns:
        str: The hexadecimal SHA-256 of the file.
    """
    with open(file=file_path, mode="rb") as file_handler:
        return hashlib.file_digest(file_handler, "sha256").hexdigest()


def load_embedder(experiment_folder: Path, iteration: int, use_averaged_weights: bool,
                  device: torch.device) -> Dict[str, Any]:
    """
    Rebuilds the model of the embedding run at one of its checkpoints and describes what the checks and the
    extraction need: the preprocessing of its images, the shape of the output of every convolution block of its
    backbone (measured on one blank image), the hidden layers of its classification head, its class names and the
    SHA-256 of its checkpoint.

    Args:
        experiment_folder (Path): The run folder of the embedding run.
        iteration (int): The iteration of its checkpoint.
        use_averaged_weights (bool): Whether to load the moving average of its weights instead of its trained weights.
        device (torch.device): The device running the model.

    Returns:
        Dict[str, Any]: The 'model' in evaluation mode, its 'device', 'task', 'experiment_configuration',
            'image_size' and 'keep_ratio', the [channels, height, width] 'block_shapes' of the outputs of its blocks
            before their pooling (block 1 first), the 'hidden_layer_sizes' and 'output_mode' of its classification head
            (None for the other tasks), 'has_sequence_normalization', the 'class_names' of its record (None without
            one), 'checkpoint_path', 'checkpoint_sha256' and 'use_averaged_weights'.

    Raises:
        FileNotFoundError: If the configuration or the checkpoint of the run is missing.
        ValueError: If the backbone does not read 3-channel images.
    """
    configuration_path = experiment_folder / "training_configuration.yaml"
    checkpoint_path = get_checkpoint_file(weights_directory=experiment_folder / "Weights", iteration=iteration)
    for required_path in [configuration_path, checkpoint_path]:
        if not required_path.is_file():
            raise FileNotFoundError(f"The embedding run has no {required_path.name} : "
                                    f"{format_path_link(path=required_path)}")

    # The configuration of the run and its model at the checkpoint, in evaluation mode
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    model = load_trained_model(configuration_path=configuration_path, checkpoint_path=checkpoint_path, device=device,
                               use_averaged_weights=use_averaged_weights)

    # The images are read as BGR images, the input of the backbones of the framework
    if model_configuration["Backbone"]["input_channels"] != 3:
        raise ValueError(f"The backbone of the embedding run reads {model_configuration['Backbone']['input_channels']} "
                         f"channels, but the images are read as 3-channel BGR images.")
    image_size = model_configuration["Data"]["image_settings"]["size"]

    # Shape of the output of every block before its pooling, measured on one blank image
    with torch.no_grad():
        _, block_outputs = model.backbone(input_tensor=torch.zeros(size=(1, 3, image_size, image_size), device=device),
                                          return_pre_pooling_outputs=True)
    block_shapes = [list(block_outputs[str(block_index)].shape[1:]) for block_index in range(1, len(block_outputs) + 1)]

    # Hidden layers and output mode of the classification head, which the other tasks do not have
    task = experiment_configuration["task"]
    head_configuration = model_configuration["ClassificationHead"] if task == CLASSIFICATION_TASK else None

    return {"model": model, "device": device, "task": task, "experiment_configuration": experiment_configuration,
            "image_size": image_size, "keep_ratio": model_configuration["Data"]["image_settings"]["keep_ratio"],
            "block_shapes": block_shapes,
            "hidden_layer_sizes": None if head_configuration is None else list(head_configuration["fully_connected"]),
            "output_mode": None if head_configuration is None else head_configuration["output_mode"],
            "has_sequence_normalization": any(isinstance(module, SequenceBatchNormalization)
                                              for module in model.backbone.modules()),
            "class_names": read_source_class_names(source_folder=experiment_folder),
            "checkpoint_path": checkpoint_path, "checkpoint_sha256": compute_file_sha256(file_path=checkpoint_path),
            "use_averaged_weights": use_averaged_weights}


def check_layer_requests(embedder: Dict[str, Any], convolution_block_indices: List[int],
                         fully_connected_layer_indices: List[int]) -> None:
    """
    Verifies, before any inference, that the asked layers exist in the embedding run, every index being counted from
    1: the convolution blocks in its backbone, each giving at most MAXIMUM_BLOCK_VALUES values per image, and the
    hidden fully connected layers in the head of a classification run (the class logits layer excluded). A backbone
    normalising every frame with the other frames of its sequence is refused, a single image having no defined
    embedding with it.

    Args:
        embedder (Dict[str, Any]): The description of the embedding run given by load_embedder.
        convolution_block_indices (List[int]): The asked convolution blocks.
        fully_connected_layer_indices (List[int]): The asked hidden fully connected layers.

    Raises:
        ValueError: If any asked layer is invalid, listing every problem with the valid range.
    """
    errors = []

    # At least one layer
    if not convolution_block_indices and not fully_connected_layer_indices:
        errors.append("No layer asked: give at least one index in convolution_block_indices or "
                      "fully_connected_layer_indices.")

    # Convolution blocks of the backbone, counted from 1, each with an output small enough to be stored whole
    number_blocks = len(embedder["block_shapes"])
    if len(set(convolution_block_indices)) != len(convolution_block_indices):
        errors.append(f"convolution_block_indices holds duplicates: {convolution_block_indices}.")
    for block_index in dict.fromkeys(convolution_block_indices):
        if not is_positive_integer(value=block_index) or block_index > number_blocks:
            errors.append(f"Convolution block {block_index!r} asked, the backbone has {number_blocks} blocks "
                          f"(1 to {number_blocks}).")
            continue
        channels, height, width = embedder["block_shapes"][block_index - 1]
        if channels * height * width > MAXIMUM_BLOCK_VALUES:
            errors.append(f"Convolution block {block_index} gives {channels * height * width:,} values per image "
                          f"({channels} channels of {height} x {width}), more than the {MAXIMUM_BLOCK_VALUES:,} "
                          f"allowed: ask a later block.")

    # Hidden fully connected layers of a classification head, counted from 1, the class logits layer excluded
    if len(set(fully_connected_layer_indices)) != len(fully_connected_layer_indices):
        errors.append(f"fully_connected_layer_indices holds duplicates: {fully_connected_layer_indices}.")
    if fully_connected_layer_indices and embedder["hidden_layer_sizes"] is None:
        errors.append(f"Fully connected layers {fully_connected_layer_indices} asked, but the embedding run is a "
                      f"{embedder['task']} run: only the hidden layers of a classification head can be embedded.")
    elif fully_connected_layer_indices:
        number_hidden_layers = len(embedder["hidden_layer_sizes"])
        for layer_index in dict.fromkeys(fully_connected_layer_indices):
            if not is_positive_integer(value=layer_index) or layer_index > number_hidden_layers:
                errors.append(f"Fully connected layer {layer_index!r} asked, the classification head has "
                              f"{number_hidden_layers} hidden layers (1 to {number_hidden_layers}), the class logits "
                              f"layer excluded.")

    # A frame normalised with the other frames of its sequence has no embedding of its own
    if embedder["has_sequence_normalization"]:
        errors.append("The backbone normalises every frame with the other frames of its sequence (feature map "
                      "normalization 'sequence'): a single image has no defined embedding.")

    if errors:
        raise ValueError("Invalid layers asked:\n- " + "\n- ".join(errors))


def check_clustering_settings(clustering_algorithm: str, clustering_parameters: Dict[str, Any],
                              pca_dimensions: int, number_representatives_per_cluster: int,
                              noise_selection: str | int, group_pattern: Optional[str],
                              mosaic_grid_size: int) -> None:
    """
    Verifies the clustering and selection settings before any inference, so that a wrong value does not stop the
    script after a long extraction: the algorithm and its parameters, the PCA dimensions, the representatives and
    noise images of the subset, the group pattern and the mosaic grid.

    Args:
        clustering_algorithm (str): "hdbscan", "dbscan" or "kmeans".
        clustering_parameters (Dict[str, Any]): The parameters of the algorithm (see CLUSTERING_PARAMETERS).
        pca_dimensions (int): The number of PCA dimensions every layer is reduced to.
        number_representatives_per_cluster (int): The images of every cluster kept in the subset.
        noise_selection (str | int): "all", "none" or the number of noise images drawn for the subset.
        group_pattern (Optional[str]): The regular expression giving the group of an image, or None.
        mosaic_grid_size (int): The number of images per row of the mosaic of a cluster, and its maximum number of rows.

    Raises:
        ValueError: If any setting is invalid, listing every problem.
    """
    errors = []

    # The algorithm and the parameters it requires
    missing_parameters = []
    if clustering_algorithm not in CLUSTERING_PARAMETERS:
        errors.append(f"clustering_algorithm must be one of {list(CLUSTERING_PARAMETERS)}, got "
                      f"{clustering_algorithm!r}.")
    else:
        missing_parameters = [name for name in CLUSTERING_PARAMETERS[clustering_algorithm]
                              if name not in clustering_parameters]
        if missing_parameters:
            errors.append(f"The {clustering_algorithm} parameters miss {missing_parameters}.")

    # The values of the parameters given
    if clustering_algorithm == "hdbscan" and not missing_parameters:
        if not is_positive_integer(value=clustering_parameters["min_cluster_size"]) or \
                clustering_parameters["min_cluster_size"] < 2:
            errors.append(f"min_cluster_size must be an integer of at least 2, got "
                          f"{clustering_parameters['min_cluster_size']!r}.")
        if clustering_parameters["min_samples"] is not None and \
                not is_positive_integer(value=clustering_parameters["min_samples"]):
            errors.append(f"min_samples must be a positive integer or None, got "
                          f"{clustering_parameters['min_samples']!r}.")
        if clustering_parameters["cluster_selection_method"] not in HDBSCAN_SELECTION_METHODS:
            errors.append(f"cluster_selection_method must be one of {HDBSCAN_SELECTION_METHODS}, got "
                          f"{clustering_parameters['cluster_selection_method']!r}.")
    elif clustering_algorithm == "dbscan" and not missing_parameters:
        if isinstance(clustering_parameters["eps"], bool) or \
                not isinstance(clustering_parameters["eps"], (int, float)) or clustering_parameters["eps"] <= 0:
            errors.append(f"eps must be a positive number, got {clustering_parameters['eps']!r}.")
        if not is_positive_integer(value=clustering_parameters["min_samples"]):
            errors.append(f"min_samples must be a positive integer, got {clustering_parameters['min_samples']!r}.")
    elif clustering_algorithm == "kmeans" and not missing_parameters:
        for name in CLUSTERING_PARAMETERS["kmeans"]:
            if not is_positive_integer(value=clustering_parameters[name]):
                errors.append(f"{name} must be a positive integer, got {clustering_parameters[name]!r}.")

    # The reduction, the subset, the groups and the mosaics
    if not is_positive_integer(value=pca_dimensions):
        errors.append(f"pca_dimensions must be a positive integer, got {pca_dimensions!r}.")
    if not is_positive_integer(value=number_representatives_per_cluster):
        errors.append(f"number_representatives_per_cluster must be a positive integer, got "
                      f"{number_representatives_per_cluster!r}.")
    if noise_selection not in NOISE_SELECTIONS and not is_positive_integer(value=noise_selection):
        errors.append(f"noise_selection must be one of {NOISE_SELECTIONS} or a positive integer, got "
                      f"{noise_selection!r}.")
    if group_pattern is not None:
        try:
            re.compile(pattern=group_pattern)
        except re.error as error:
            errors.append(f"group_pattern is not a valid regular expression: {error}.")
    if not is_positive_integer(value=mosaic_grid_size):
        errors.append(f"mosaic_grid_size must be a positive integer, got {mosaic_grid_size!r}.")

    if errors:
        raise ValueError("Invalid clustering settings:\n- " + "\n- ".join(errors))


def build_layer_requests(convolution_block_indices: List[int],
                         fully_connected_layer_indices: List[int]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Names every asked layer, the name keying its features in the extraction record, its matrix file and its
    clustering folders: 'block_<index>' for a convolution block and 'fully_connected_<index>' for a hidden layer.

    Args:
        convolution_block_indices (List[int]): The asked convolution blocks.
        fully_connected_layer_indices (List[int]): The asked hidden fully connected layers.

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: The index of every asked block and of every asked hidden layer, keyed by
            layer name.
    """
    block_requests = {f"block_{block_index}": block_index for block_index in convolution_block_indices}
    fully_connected_requests = {f"fully_connected_{layer_index}": layer_index
                                for layer_index in fully_connected_layer_indices}
    return block_requests, fully_connected_requests


def describe_source_folder(folder: Path, role: str) -> Dict[str, Any]:
    """
    Describes one folder to embed: a split folder of any task ('data' and 'labels.json') or a folder of plain images,
    named after the last components of its path.

    Args:
        folder (Path): The folder.
        role (str): What the folder is for the settings: a split role ('train', 'validation', 'test') or 'folder'.

    Returns:
        Dict[str, Any]: Its 'name', its resolved 'folder', its 'data_folder' holding the images, its 'labels_path'
            (None for a folder of plain images) and its 'roles'.

    Raises:
        FileNotFoundError: If the folder, or the 'data' folder of a split, does not exist.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"The data source folder does not exist : {format_path_link(path=folder)}")
    resolved_folder = folder.resolve()

    # A split of the framework holds its 'labels.json' next to its 'data' folder, a folder of plain images neither
    labels_path = resolved_folder / "labels.json"
    if labels_path.is_file():
        data_folder = resolved_folder / "data"
        if not data_folder.is_dir():
            raise FileNotFoundError(f"The split has a labels.json but no data folder : "
                                    f"{format_path_link(path=resolved_folder)}")
    else:
        labels_path = None
        data_folder = resolved_folder

    return {"name": "-".join(resolved_folder.parts[-SOURCE_NAME_COMPONENTS:]), "folder": resolved_folder,
            "data_folder": data_folder, "labels_path": labels_path, "roles": [role]}


def resolve_data_sources(data_sources: List[str | Path],
                         embedder_experiment_configuration: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turns the data sources of the settings into the folders to embed, each once: "run" for the train, validation and
    test splits of the embedding run, a configuration file of any task for its three splits, a split folder or a
    folder of plain images. A folder given twice, e.g. a test split that is the validation folder, is embedded once
    with both roles.

    Args:
        data_sources (List[str | Path]): The data sources of the settings.
        embedder_experiment_configuration (Dict[str, Any]): The 'ExperimentConfiguration' of the embedding run.

    Returns:
        List[Dict[str, Any]]: Every folder to embed, described by describe_source_folder, in the order of the settings.

    Raises:
        ValueError: If no data source is given, or two folders get the same name.
        FileNotFoundError: If a configuration file or a folder does not exist.
    """
    if not data_sources:
        raise ValueError("No data source given: data_sources must list 'run', configuration files or folders.")
    sources_by_folder = {}
    for data_source in data_sources:
        # The three splits of the embedding run or of another configuration file, or one folder
        if data_source == RUN_DATA_SOURCE:
            split_folders = {role: Path(embedder_experiment_configuration[key]) for key, role in SPLIT_ROLES.items()}
        elif Path(data_source).suffix in [".yaml", ".yml"]:
            source_configuration = load_configuration(configuration_path=data_source)["ExperimentConfiguration"]
            split_folders = {role: Path(source_configuration[key]) for key, role in SPLIT_ROLES.items()}
        else:
            split_folders = {"folder": Path(data_source)}

        # Every folder once, a folder met again only gaining its new role
        for role, folder in split_folders.items():
            source = describe_source_folder(folder=folder, role=role)
            if source["folder"] in sources_by_folder:
                known_roles = sources_by_folder[source["folder"]]["roles"]
                known_roles += [] if role in known_roles else [role]
            else:
                sources_by_folder[source["folder"]] = source

    # Every extraction folder named after one source only
    sources = list(sources_by_folder.values())
    source_names = [source["name"] for source in sources]
    duplicated_names = sorted({name for name in source_names if source_names.count(name) > 1})
    if duplicated_names:
        raise ValueError(f"Several data sources would share the extraction folders {duplicated_names}: their last "
                         f"{SOURCE_NAME_COMPONENTS} path components are the same.")
    return sources


def list_source_images(source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lists the images of a data source in their row order, with the category of every image when the source gives one
    per image (the samples of a classification manifest), and the SHA-256 identifying its images in the extraction
    record: the SHA-256 of the 'labels.json' of a split, or of the list of the images of a folder with their sizes and
    modification dates.

    Args:
        source (Dict[str, Any]): The data source, described by describe_source_folder.

    Returns:
        Dict[str, Any]: The 'file_names' relative to the data folder (each image once, in the order of the manifest or
            sorted), the 'categories' (None without one category per image), the 'manifest' (None for a folder of
            plain images) and the 'images_sha256'.

    Raises:
        ValueError: If the source holds no image.
    """
    # A folder of plain images, with its subfolders, identified by the names, sizes and dates of its images
    if source["labels_path"] is None:
        image_paths = get_image_paths(folder=source["data_folder"], search_subfolders=True)
        file_names = [image_path.relative_to(source["data_folder"]).as_posix() for image_path in image_paths]
        listing_hash = hashlib.sha256()
        for file_name, image_path in zip(file_names, image_paths):
            file_status = image_path.stat()
            listing_hash.update(f"{file_name}\t{file_status.st_size}\t{file_status.st_mtime_ns}\n".encode())
        return {"file_names": file_names, "categories": None, "manifest": None,
                "images_sha256": listing_hash.hexdigest()}

    # A split of any task, its images listed by the manifest, each once (overlapping sequences share frames)
    manifest = read_json(path=str(source["labels_path"]))
    file_names = list(dict.fromkeys(list_manifest_images(labels=manifest)))
    if not file_names:
        raise ValueError(f"The manifest lists no image : {format_path_link(path=source['labels_path'])}")

    # One category per image for the samples of a classification manifest, none for the other formats
    categories = None
    if "samples" in manifest and all("category_id" in sample for sample in manifest["samples"]):
        category_names = {category["id"]: category["name"] for category in manifest["categories"]}
        image_categories = {sample["file_name"]: category_names[sample["category_id"]]
                            for sample in manifest["samples"]}
        categories = [image_categories[file_name] for file_name in file_names]

    return {"file_names": file_names, "categories": categories, "manifest": manifest,
            "images_sha256": compute_file_sha256(file_path=source["labels_path"])}


def release_file_from_page_cache(path: Path) -> None:
    """
    Asks Linux to drop a file the script has read from its page cache, the copy of the recently read files it keeps
    in memory. On the unified memory of the DGX Spark, the frames read for the extraction and the mosaics (about 20
    Gigabytes for the ficha train split) and the feature matrices would otherwise fill the memory the GPU shares and
    stop other processes from creating their CUDA context. Only the pages of this file are released, without sudo and
    without touching the files of other processes; the pages still mapped or not yet written to the disk stay. Does
    nothing on a system without posix_fadvise (macOS).

    Args:
        path (Path): The file read by the script.
    """
    # Nothing to release on a system without the call
    if not hasattr(os, "posix_fadvise"):
        return

    # The whole file (from offset 0, a length of 0 reaching its end) marked as not needed, its pages dropped at once
    file_descriptor = os.open(path=path, flags=os.O_RDONLY)
    try:
        os.posix_fadvise(file_descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd=file_descriptor)


class ImageListDataset(Dataset):
    """
    Images of a data source read in their row order for the extraction, from a first row to the last, so that an
    interrupted extraction continues where it stopped. Every image is preprocessed like the images of the embedding
    run: resized to its square input, padded to keep its ratio when the run does.
    """

    def __init__(self, image_paths: List[Path], first_row: int, image_size: int, keep_ratio: bool) -> None:
        """
        Keeps the images and the preprocessing settings.

        Args:
            image_paths (List[Path]): The image of every row.
            first_row (int): The row of the first image read.
            image_size (int): The size of the square input of the embedding run.
            keep_ratio (bool): Whether the embedding run pads its images to keep their ratio.
        """
        self.image_paths = image_paths
        self.first_row = first_row
        self.image_size = image_size
        self.keep_ratio = keep_ratio

    def __len__(self) -> int:
        """
        Gives the number of images read, from the first row to the last.

        Returns:
            int: The number of images.
        """
        return len(self.image_paths) - self.first_row

    def __getitem__(self, sample_index: int) -> Dict[str, Any]:
        """
        Reads and preprocesses one image, released from the page cache once decoded.

        Args:
            sample_index (int): The index of the image among the read ones.

        Returns:
            Dict[str, Any]: The uint8 [3, image_size, image_size] BGR image and its row.

        Raises:
            ValueError: If the image cannot be read.
        """
        # The image decoded, then released from the page cache, and preprocessed like the images of the run
        row = self.first_row + sample_index
        image = cv2.imread(filename=str(self.image_paths[row]))
        if image is None:
            raise ValueError(f"The image cannot be read : {format_path_link(path=self.image_paths[row])}")
        release_file_from_page_cache(path=self.image_paths[row])
        resized_image = preprocess_image(image=image, image_size=self.image_size, keep_ratio=self.keep_ratio)
        return {"images": torch.from_numpy(np.ascontiguousarray(a=resized_image)).permute(dims=(2, 0, 1)), "rows": row}


class LayerOutputRecorder:
    """
    Forward hook keeping a copy of the output of a module of the classification head, the ReLU of an asked hidden
    layer, when the head runs during the extraction.
    """

    def __init__(self, outputs: Dict[str, torch.Tensor], output_key: str) -> None:
        """
        Keeps the dictionary receiving the output and the name of the layer.

        Args:
            outputs (Dict[str, torch.Tensor]): The outputs of the batch, filled in place.
            output_key (str): The name of the layer.
        """
        self.outputs = outputs
        self.output_key = output_key

    def __call__(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """
        Copies the output of the module, which the in-place ReLU would otherwise share with the next layers.

        Args:
            module (nn.Module): The hooked module.
            inputs (Tuple[torch.Tensor, ...]): The inputs of the module.
            output (torch.Tensor): The output of the module.
        """
        self.outputs[self.output_key] = output.detach().clone()


def compute_batch_outputs(model: nn.Module, images: torch.Tensor, block_requests: Dict[str, int],
                          fully_connected_requests: Dict[str, int],
                          output_mode: Optional[str]) -> Dict[str, torch.Tensor]:
    """
    Runs the embedding model on a batch of images and gives the asked outputs: the whole output of every asked block
    before its pooling, flattened channel after channel (channels x height x width), the output of every asked hidden
    layer of the classification head after its ReLU, and, with an output mode, the most probable class of every image
    and its probability.

    Args:
        model (nn.Module): The embedding model in evaluation mode.
        images (torch.Tensor): The [B, 3, H, W] float BGR images in [0, 255].
        block_requests (Dict[str, int]): The asked blocks, keyed by layer name.
        fully_connected_requests (Dict[str, int]): The asked hidden layers, keyed by layer name.
        output_mode (Optional[str]): The output mode of the classification head to give the predictions, or None.

    Returns:
        Dict[str, torch.Tensor]: The [B, D] output of every asked layer, and the [B, 2] class index and probability of
            every image under PREDICTIONS_OUTPUT with an output mode.
    """
    final_feature_map, block_outputs = model.backbone(input_tensor=images, return_pre_pooling_outputs=True)

    # Every asked block kept whole, its pixels flattened after its channels
    outputs = {output_key: block_outputs[str(block_index)].flatten(start_dim=1)
               for output_key, block_index in block_requests.items()}
    if not fully_connected_requests and output_mode is None:
        return outputs

    # The head run once, the ReLU of every asked hidden layer recording its output
    relu_modules = [module for module in model.classification_head.fully_connected_block
                    if isinstance(module, nn.ReLU)]
    hook_handles = [relu_modules[layer_index - 1].register_forward_hook(
        hook=LayerOutputRecorder(outputs=outputs, output_key=output_key))
        for output_key, layer_index in fully_connected_requests.items()]
    try:
        class_logits = model.classification_head(features=model.global_pooling(input=final_feature_map))
    finally:
        for hook_handle in hook_handles:
            hook_handle.remove()

    # The most probable class of every image and its probability
    if output_mode is not None:
        class_probabilities = convert_logits_to_probabilities(class_logits=class_logits, output_mode=output_mode)
        top_probabilities, top_classes = class_probabilities.max(dim=1)
        outputs[PREDICTIONS_OUTPUT] = torch.stack(tensors=[top_classes.to(dtype=top_probabilities.dtype),
                                                           top_probabilities], dim=1)
    return outputs


def get_output_path(extraction_folder: Path, output_key: str) -> Path:
    """
    Gives the matrix file of an output in the extraction folder of a source: 'predictions.npy' for the predictions,
    'features_<layer name>.npy' for a layer.

    Args:
        extraction_folder (Path): The extraction folder of the source.
        output_key (str): The name of the output.

    Returns:
        Path: The matrix file.
    """
    if output_key == PREDICTIONS_OUTPUT:
        return extraction_folder / f"{PREDICTIONS_OUTPUT}.npy"
    return extraction_folder / f"features_{output_key}.npy"


def write_json_atomically(path: Path, content: Dict[str, Any]) -> None:
    """
    Writes a JSON file through a temporary file replaced in one step, so that the extraction record is never left
    half written when the extraction is interrupted.

    Args:
        path (Path): The JSON file.
        content (Dict[str, Any]): Its content.
    """
    temporary_path = path.with_name(f"{path.name}.temporary")
    temporary_path.write_text(data=json.dumps(content, indent=2), encoding="utf-8")
    os.replace(src=temporary_path, dst=path)


def write_csv(path: Path, field_names: List[str], rows: List[Dict[str, Any]]) -> None:
    """
    Writes rows of values as a CSV file with a header, for the image lists, the clusters and their summaries.

    Args:
        path (Path): The CSV file.
        field_names (List[str]): The columns, in order.
        rows (List[Dict[str, Any]]): The value of every column of every row.
    """
    with open(file=path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(f=csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def build_extraction_fingerprint(embedder: Dict[str, Any], source: Dict[str, Any],
                                 source_images: Dict[str, Any]) -> Dict[str, Any]:
    """
    Describes what produces the features of a source, recorded in its 'extraction.json' and compared at every
    relaunch on FINGERPRINT_KEYS: the checkpoint by its SHA-256 (its path only informs), the weights, the
    preprocessing, the source folder and its images.

    Args:
        embedder (Dict[str, Any]): The description of the embedding run given by load_embedder.
        source (Dict[str, Any]): The data source.
        source_images (Dict[str, Any]): The images of the source given by list_source_images.

    Returns:
        Dict[str, Any]: The entries of the record identifying the features.
    """
    return {"checkpoint_path": str(embedder["checkpoint_path"]), "checkpoint_sha256": embedder["checkpoint_sha256"],
            "use_averaged_weights": embedder["use_averaged_weights"], "image_size": embedder["image_size"],
            "keep_ratio": embedder["keep_ratio"], "source_folder": str(source["folder"]),
            "images_sha256": source_images["images_sha256"], "number_images": len(source_images["file_names"])}


def extract_source_features(embedder: Dict[str, Any], source: Dict[str, Any], source_images: Dict[str, Any],
                            block_requests: Dict[str, int], fully_connected_requests: Dict[str, int],
                            extraction_folder: Path, batch_size: int, number_workers: int) -> Dict[str, np.ndarray]:
    """
    Gives the features of every asked layer for every image of a source (and the predictions of a classification
    run), running the model only for what is not stored yet in the extraction folder. Its 'extraction.json' records
    the fingerprint of the features (build_extraction_fingerprint) and the number of finished images of every output:
    - A new folder: the record and 'images.csv' are written, then every output is computed.
    - The same fingerprint: the finished outputs are loaded, the missing ones computed in one pass, and an
      interrupted output continues from its last finished batch, every matrix being filled batch after batch on the
      disk.
    - Another fingerprint (another checkpoint, changed images, another folder): nothing is overwritten.

    Args:
        embedder (Dict[str, Any]): The description of the embedding run given by load_embedder.
        source (Dict[str, Any]): The data source.
        source_images (Dict[str, Any]): The images of the source given by list_source_images.
        block_requests (Dict[str, int]): The asked blocks, keyed by layer name.
        fully_connected_requests (Dict[str, int]): The asked hidden layers, keyed by layer name.
        extraction_folder (Path): The extraction folder of the source.
        batch_size (int): The number of images run together.
        number_workers (int): The number of loader worker processes reading the images.

    Returns:
        Dict[str, np.ndarray]: The read-only [number of images, D] matrix of every asked layer, and of the predictions
            of a classification run, keyed by output name.

    Raises:
        ValueError: If the folder holds the features of another checkpoint or of other images, or the model gives
            values that are not finite.
    """
    extraction_folder.mkdir(parents=True, exist_ok=True)
    record_path = extraction_folder / EXTRACTION_RECORD_FILE_NAME
    fingerprint = build_extraction_fingerprint(embedder=embedder, source=source, source_images=source_images)
    number_images = fingerprint["number_images"]

    # A previous record must describe the same checkpoint and images, a new folder starts with the list of its images
    if record_path.is_file():
        record = read_json(path=str(record_path))
        changed_entries = [key for key in FINGERPRINT_KEYS if record[key] != fingerprint[key]]
        if changed_entries:
            changes = "\n".join(f"- {key}: stored {record[key]!r}, now {fingerprint[key]!r}"
                                for key in changed_entries)
            raise ValueError(f"The extraction folder holds the features of another checkpoint or of other images, "
                             f"nothing is overwritten: delete the folder or give another output_root to compute "
                             f"them again : {format_path_link(path=extraction_folder)}\n{changes}")
    else:
        record = {**fingerprint, "outputs": {}}
        write_csv(path=extraction_folder / IMAGES_FILE_NAME, field_names=["row", "image", "category"],
                  rows=[{"row": row, "image": file_name,
                         "category": "" if source_images["categories"] is None else source_images["categories"][row]}
                        for row, file_name in enumerate(source_images["file_names"])])
        write_json_atomically(path=record_path, content=record)

    # Width of every output: the channels times the height times the width of a block, the units of a hidden layer,
    # and the class index and probability of the predictions
    output_dimensions = {output_key: int(np.prod(embedder["block_shapes"][block_index - 1]))
                         for output_key, block_index in block_requests.items()}
    output_dimensions.update({output_key: embedder["hidden_layer_sizes"][layer_index - 1]
                              for output_key, layer_index in fully_connected_requests.items()})
    if embedder["output_mode"] is not None:
        output_dimensions[PREDICTIONS_OUTPUT] = 2

    # Outputs not stored yet, or stopped before the last image
    missing_outputs = [output_key for output_key in output_dimensions
                       if output_key not in record["outputs"]
                       or record["outputs"][output_key]["finished_images"] < number_images]
    if missing_outputs:
        run_extraction_pass(embedder=embedder, source=source, source_images=source_images, record=record,
                            record_path=record_path, output_dimensions=output_dimensions,
                            missing_outputs=missing_outputs, block_requests=block_requests,
                            fully_connected_requests=fully_connected_requests, extraction_folder=extraction_folder,
                            batch_size=batch_size, number_workers=number_workers)
    else:
        print_color(color="green", output=f"Features of {source['name']} loaded, no inference : "
                                          f"{format_path_link(path=extraction_folder)}")

    return {output_key: np.load(file=get_output_path(extraction_folder=extraction_folder, output_key=output_key),
                                mmap_mode="r")
            for output_key in output_dimensions}


def run_extraction_pass(embedder: Dict[str, Any], source: Dict[str, Any], source_images: Dict[str, Any],
                        record: Dict[str, Any], record_path: Path, output_dimensions: Dict[str, int],
                        missing_outputs: List[str], block_requests: Dict[str, int],
                        fully_connected_requests: Dict[str, int], extraction_folder: Path, batch_size: int,
                        number_workers: int) -> None:
    """
    Computes the missing outputs of a source in one pass of the model over its images, from the first image one of
    them misses, for extract_source_features: every matrix is a float32 '.npy' file on the disk filled batch after
    batch, and the record gives the number of finished images of every output after every batch, so that an
    interrupted pass continues from its last finished batch.

    Args:
        embedder (Dict[str, Any]): The description of the embedding run given by load_embedder.
        source (Dict[str, Any]): The data source.
        source_images (Dict[str, Any]): The images of the source given by list_source_images.
        record (Dict[str, Any]): The extraction record, updated in place.
        record_path (Path): The extraction record file.
        output_dimensions (Dict[str, int]): The width of every output.
        missing_outputs (List[str]): The outputs to compute.
        block_requests (Dict[str, int]): The asked blocks, keyed by layer name.
        fully_connected_requests (Dict[str, int]): The asked hidden layers, keyed by layer name.
        extraction_folder (Path): The extraction folder of the source.
        batch_size (int): The number of images run together.
        number_workers (int): The number of loader worker processes reading the images.

    Raises:
        ValueError: If a stored matrix has another shape, the loader skips rows, or the model gives values that are
            not finite.
    """
    number_images = record["number_images"]

    # The matrix of every missing output, continued when an interrupted pass left it, created otherwise
    matrices = {}
    for output_key in missing_outputs:
        output_path = get_output_path(extraction_folder=extraction_folder, output_key=output_key)
        if output_key in record["outputs"] and output_path.is_file():
            matrices[output_key] = np.load(file=output_path, mmap_mode="r+")
            if matrices[output_key].shape != (number_images, output_dimensions[output_key]):
                raise ValueError(f"The stored matrix of {output_key} has the shape {matrices[output_key].shape} "
                                 f"instead of {(number_images, output_dimensions[output_key])} : "
                                 f"{format_path_link(path=output_path)}")
        else:
            matrices[output_key] = np.lib.format.open_memmap(filename=output_path, mode="w+", dtype=np.float32,
                                                              shape=(number_images, output_dimensions[output_key]))
            record["outputs"][output_key] = {"dimension": output_dimensions[output_key], "finished_images": 0}
    write_json_atomically(path=record_path, content=record)

    # The images from the first one an output misses, in their row order
    first_row = min(record["outputs"][output_key]["finished_images"] for output_key in missing_outputs)
    dataset = ImageListDataset(image_paths=[source["data_folder"] / file_name
                                            for file_name in source_images["file_names"]],
                               first_row=first_row, image_size=embedder["image_size"],
                               keep_ratio=embedder["keep_ratio"])
    loader = build_native_loader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                 loading_configuration={"number_workers": number_workers, "prefetch_batches": 2})
    missing_block_requests = {output_key: block_index for output_key, block_index in block_requests.items()
                              if output_key in missing_outputs}
    missing_fully_connected_requests = {output_key: layer_index
                                        for output_key, layer_index in fully_connected_requests.items()
                                        if output_key in missing_outputs}
    output_mode = embedder["output_mode"] if PREDICTIONS_OUTPUT in missing_outputs else None
    print_color(color="blue", output=f"Embedding {source['name']} from image {first_row} of {number_images} : "
                                     f"{', '.join(missing_outputs)}")

    # Every batch run, written to its rows, which must follow the rows of the previous batch, then recorded as finished
    start_time = time.perf_counter()
    batch_end = first_row
    with torch.no_grad():
        for batch in tqdm(iterable=loader, desc=f"Embedding {source['name']}", unit="batch"):
            rows = batch["rows"]
            batch_start, batch_end = batch_end, batch_end + len(rows)
            if rows.tolist() != list(range(batch_start, batch_end)):
                raise ValueError(f"The loader gave the rows {rows.tolist()} instead of the rows {batch_start} to "
                                 f"{batch_end - 1}.")
            outputs = compute_batch_outputs(model=embedder["model"],
                                            images=batch["images"].to(device=embedder["device"], dtype=torch.float32),
                                            block_requests=missing_block_requests,
                                            fully_connected_requests=missing_fully_connected_requests,
                                            output_mode=output_mode)
            for output_key, output in outputs.items():
                values = output.to(dtype=torch.float32).cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"The model gives values that are not finite for {output_key} on the images "
                                     f"{batch_start} to {batch_end - 1} of {source['name']}.")
                matrices[output_key][batch_start:batch_end] = values
            for output_key in missing_outputs:
                matrices[output_key].flush()
                record["outputs"][output_key]["finished_images"] = max(
                    record["outputs"][output_key]["finished_images"], batch_end)
            write_json_atomically(path=record_path, content=record)
    del loader

    # Throughput of the pass
    elapsed_seconds = time.perf_counter() - start_time
    number_embedded = number_images - first_row
    print_color(color="green", output=f"{number_embedded} images of {source['name']} embedded in "
                                      f"{elapsed_seconds:.1f} Seconds "
                                      f"({number_embedded / max(elapsed_seconds, 1e-9):.1f} Images Per Second) : "
                                      f"{format_path_link(path=extraction_folder)}")


def prepare_vectors(vectors: np.ndarray, normalize_vectors: bool, pca_dimensions: int,
                    seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Prepares the features of the images of one category for the clustering: every vector divided by its length, so
    that the distances compare which features are active rather than how strongly (a zero vector, with no active
    feature, staying zero), then reduced by a PCA to its main directions of variation, which makes the clustering
    faster. Vectors with no more dimensions or images than asked are clustered as they are, the PCA having nothing to
    reduce. Everything stays in float32, the type of the stored features, so that the matrix of the largest allowed
    block (MAXIMUM_BLOCK_VALUES values per image) and its PCA hold in memory.

    Args:
        vectors (np.ndarray): The [N, D] features.
        normalize_vectors (bool): Whether to divide every vector by its length.
        pca_dimensions (int): The number of PCA dimensions kept.
        seed (int): The seed of the randomised PCA solver.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: The [N, D'] float32 vectors to cluster, and the 'zero_vectors', the
            'dimensions' clustered, the 'kept_variance' share of the PCA and its 'pca_seconds' (both None when the
            vectors are not reduced).
    """
    # A float32 copy of the stored features, normalised in place to length 1, a zero vector staying zero
    prepared_vectors = np.array(vectors, dtype=np.float32)
    preparation = {"zero_vectors": 0, "dimensions": prepared_vectors.shape[1], "kept_variance": None,
                   "pca_seconds": None}
    if normalize_vectors:
        lengths = np.linalg.norm(prepared_vectors, axis=1, keepdims=True)
        preparation["zero_vectors"] = int((lengths == 0).sum())
        np.divide(prepared_vectors, lengths, out=prepared_vectors, where=lengths > 0)

    # Main directions of variation, when there are more dimensions and images than asked, timed to compare the PCA
    # settings
    if pca_dimensions < min(prepared_vectors.shape):
        start_time = time.perf_counter()
        pca = PCA(n_components=pca_dimensions, random_state=seed)
        prepared_vectors = pca.fit_transform(X=prepared_vectors).astype(dtype=np.float32, copy=False)
        preparation.update({"dimensions": pca_dimensions, "kept_variance": float(pca.explained_variance_ratio_.sum()),
                            "pca_seconds": time.perf_counter() - start_time})
    return prepared_vectors, preparation


def cluster_vectors(vectors: np.ndarray, clustering_algorithm: str, clustering_parameters: Dict[str, Any],
                    seed: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Clusters the prepared vectors of one category with HDBSCAN (dense regions of any shape, the isolated images
    being noise), DBSCAN (dense regions at a fixed distance, the others being noise) or k-means (a given number of
    clusters, every image in one).

    Args:
        vectors (np.ndarray): The [N, D] prepared vectors.
        clustering_algorithm (str): "hdbscan", "dbscan" or "kmeans".
        clustering_parameters (Dict[str, Any]): The parameters of the algorithm (see CLUSTERING_PARAMETERS).
        seed (int): The seed of k-means.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The [N] cluster of every vector (NOISE_CLUSTER for noise), and the
            [N] membership probability of HDBSCAN (None for the other algorithms).

    Raises:
        ValueError: If k-means asks more clusters than there are images.
    """
    number_vectors = vectors.shape[0]
    if clustering_algorithm == "hdbscan":
        # Fewer images than a cluster needs: all of them are noise, which HDBSCAN refuses to compute
        minimum_samples = clustering_parameters["min_samples"] if clustering_parameters["min_samples"] is not None \
            else clustering_parameters["min_cluster_size"]
        if number_vectors < max(clustering_parameters["min_cluster_size"], minimum_samples):
            return np.full(shape=number_vectors, fill_value=NOISE_CLUSTER), np.zeros(shape=number_vectors)
        clustering = HDBSCAN(min_cluster_size=clustering_parameters["min_cluster_size"],
                             min_samples=clustering_parameters["min_samples"],
                             cluster_selection_method=clustering_parameters["cluster_selection_method"], copy=True,
                             n_jobs=-1).fit(X=vectors)
        return clustering.labels_, clustering.probabilities_

    if clustering_algorithm == "dbscan":
        clustering = DBSCAN(eps=clustering_parameters["eps"], min_samples=clustering_parameters["min_samples"],
                            n_jobs=-1).fit(X=vectors)
        return clustering.labels_, None

    # k-means, every image in one of the asked clusters
    if number_vectors < clustering_parameters["number_clusters"]:
        raise ValueError(f"k-means asks {clustering_parameters['number_clusters']} clusters for {number_vectors} "
                         f"images: lower number_clusters below the number of images of every category.")
    clustering = KMeans(n_clusters=clustering_parameters["number_clusters"],
                        n_init=clustering_parameters["number_initializations"], random_state=seed).fit(X=vectors)
    return clustering.labels_, None


def compute_centre_distances(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Gives the distance of every clustered image to the centre of its cluster (the mean of its vectors), the closest
    images being the representatives of the cluster and the first of its links and mosaic.

    Args:
        vectors (np.ndarray): The [N, D] prepared vectors.
        labels (np.ndarray): The [N] cluster of every vector.

    Returns:
        np.ndarray: The [N] distances, NaN for the noise images.
    """
    distances = np.full(shape=labels.shape[0], fill_value=np.nan)
    for cluster in np.unique(labels[labels != NOISE_CLUSTER]):
        is_member = labels == cluster
        distances[is_member] = np.linalg.norm(vectors[is_member] - vectors[is_member].mean(axis=0), axis=1)
    return distances


def natural_sort_key(name: str) -> List[Any]:
    """
    Gives the sort key of a file name comparing its digit runs as numbers, so that 'frame_10' comes after 'frame_9',
    for the time order of the images of a group and the order of the noise images.

    Args:
        name (str): The file name.

    Returns:
        List[Any]: Its text parts and its digit runs as integers, alternately.
    """
    return [int(part) if part.isdigit() else part for part in re.split(pattern=r"(\d+)", string=name)]


def rank_cluster_members(labels: np.ndarray, distances: np.ndarray, file_names: List[str]) -> np.ndarray:
    """
    Ranks the images of every cluster from 1, the closest to its centre first, and the noise images in the order of
    their names, the rank ordering the links and the mosaic of a cluster and choosing its representatives.

    Args:
        labels (np.ndarray): The [N] cluster of every image.
        distances (np.ndarray): The [N] distance of every image to the centre of its cluster.
        file_names (List[str]): The [N] file names.

    Returns:
        np.ndarray: The [N] rank of every image in its cluster or in the noise.
    """
    ranks = np.zeros(shape=labels.shape[0], dtype=int)
    for cluster in np.unique(labels):
        member_positions = np.flatnonzero(labels == cluster)
        if cluster == NOISE_CLUSTER:
            ordered_positions = np.array([position for _, position in sorted(
                (natural_sort_key(name=file_names[position]), position) for position in member_positions)])
        else:
            ordered_positions = member_positions[np.argsort(distances[member_positions], kind="stable")]
        ranks[ordered_positions] = np.arange(start=1, stop=len(ordered_positions) + 1)
    return ranks


def select_cluster_representatives(labels: np.ndarray, ranks: np.ndarray, number_representatives_per_cluster: int,
                                   noise_selection: str | int, random_generator: np.random.Generator) -> Dict[int, str]:
    """
    Chooses the images of the cluster subset of one category: the representatives of every cluster (its images
    closest to its centre) and the noise images, all of them, none, or a number drawn at random, the noise holding
    the rare images that resemble no other.

    Args:
        labels (np.ndarray): The [N] cluster of every image.
        ranks (np.ndarray): The [N] rank of every image in its cluster.
        number_representatives_per_cluster (int): The images kept per cluster.
        noise_selection (str | int): "all", "none" or the number of noise images drawn.
        random_generator (np.random.Generator): The generator of the draw.

    Returns:
        Dict[int, str]: The position of every chosen image among the images of the category, with why it is chosen.
    """
    selection = {}

    # The images of every cluster closest to its centre
    for position in np.flatnonzero((labels != NOISE_CLUSTER) & (ranks <= number_representatives_per_cluster)):
        selection[int(position)] = f"representative of cluster {labels[position]}, rank {ranks[position]}"

    # The noise images, all, none or drawn
    noise_positions = np.flatnonzero(labels == NOISE_CLUSTER)
    if noise_selection == "all":
        chosen_noise_positions = noise_positions
    elif noise_selection == "none":
        chosen_noise_positions = noise_positions[:0]
    else:
        chosen_noise_positions = random_generator.choice(noise_positions, size=min(noise_selection,
                                                                                   len(noise_positions)),
                                                         replace=False)
    for position in chosen_noise_positions:
        selection[int(position)] = "noise"
    return selection


def select_random_baseline(number_images: int, budget: int, random_generator: np.random.Generator) -> np.ndarray:
    """
    Draws the images of the random baseline of one category, as many as its cluster subset, to tell whether the
    clustering chooses better images than chance.

    Args:
        number_images (int): The number of images of the category.
        budget (int): The number of images to draw.
        random_generator (np.random.Generator): The generator of the draw.

    Returns:
        np.ndarray: The sorted positions of the drawn images among the images of the category.
    """
    return np.sort(random_generator.choice(number_images, size=budget, replace=False))


def find_image_groups(file_names: List[str], group_pattern: Optional[str]) -> Optional[List[str]]:
    r"""
    Gives the group of every image from its file name with the group pattern of the settings, e.g. the video a frame
    comes from: the first capturing group of the pattern taking part in the match (its whole match without capturing
    group), so that every alternative of a pattern such as r"^(video_\d+)_|^(camera_\d+)/" gives its own groups,
    searched in the file name relative to the data folder, the images it does not match forming the group NO_GROUP.
    The groups feed the counts of the cluster summary and the baseline evenly spaced per group, without
    dataset-specific code.

    Args:
        file_names (List[str]): The file names of the images.
        group_pattern (Optional[str]): The regular expression, or None.

    Returns:
        Optional[List[str]]: The group of every image, None without pattern.
    """
    if group_pattern is None:
        return None
    compiled_pattern = re.compile(pattern=group_pattern)
    groups = []
    for file_name in file_names:
        match = compiled_pattern.search(string=file_name)
        if match is None:
            groups.append(NO_GROUP)
        else:
            groups.append(next((group for group in match.groups() if group is not None), match.group(0)))
    return groups


def allocate_budget_over_groups(group_sizes: Dict[str, int], budget: int) -> Dict[str, int]:
    """
    Spreads a number of images as evenly as possible over groups, for the baseline evenly spaced per group: a group
    smaller than its share gives all its images, and what it leaves goes to the larger groups, e.g. 10 images over
    groups of 2, 50 and 50 images give 2, 4 and 4.

    Args:
        group_sizes (Dict[str, int]): The number of images of every group.
        budget (int): The number of images to spread, at most the sum of the sizes.

    Returns:
        Dict[str, int]: The number of images taken from every group, summing to the budget.

    Raises:
        ValueError: If the budget exceeds the images of the groups.
    """
    if budget > sum(group_sizes.values()):
        raise ValueError(f"{budget} images asked from groups holding {sum(group_sizes.values())} images.")

    # Smallest groups first, each taking at most an even share of what is left, the largest group taking the rest,
    # which never exceeds its size since every earlier group takes its share or all its images
    ordered_groups = sorted(group_sizes.items(), key=itemgetter(1, 0))
    allocation = {}
    remaining_budget = budget
    for position, (group, group_size) in enumerate(ordered_groups):
        share = remaining_budget // (len(ordered_groups) - position)
        allocation[group] = min(group_size, share)
        remaining_budget -= allocation[group]
    return allocation


def select_evenly_spaced_per_group(file_names: List[str], groups: List[str], budget: int) -> Tuple[np.ndarray,
                                                                                                    List[str]]:
    """
    Chooses the images of the baseline evenly spaced per group of one category, as many as its cluster subset, the
    simple selection the clustering should beat: the budget spread evenly over the groups, the images of a group
    ordered by their file names (digit runs compared as numbers, the time order of names carrying the time of the
    frame) and taken at the centres of equal segments of that order.

    Args:
        file_names (List[str]): The file names of the images of the category.
        groups (List[str]): Their groups.
        budget (int): The number of images to choose.

    Returns:
        Tuple[np.ndarray, List[str]]: The sorted positions of the chosen images among the images of the category, and
            the group of every chosen image in the same order.
    """
    positions_by_group = {}
    for position, group in enumerate(groups):
        positions_by_group.setdefault(group, []).append(position)
    allocation = allocate_budget_over_groups(group_sizes={group: len(positions)
                                                          for group, positions in positions_by_group.items()},
                                             budget=budget)

    # The images of every group in the order of their names, taken at the centres of equal segments
    chosen_positions = []
    for group, positions in positions_by_group.items():
        number_chosen = allocation[group]
        if number_chosen == 0:
            continue
        ordered_positions = [position for _, position in sorted((natural_sort_key(name=file_names[position]),
                                                                 position) for position in positions)]
        segment_centres = ((np.arange(stop=number_chosen) + 0.5) * len(ordered_positions) / number_chosen).astype(int)
        chosen_positions += [ordered_positions[index] for index in segment_centres]
    chosen_positions = np.sort(np.array(chosen_positions, dtype=int))
    return chosen_positions, [groups[position] for position in chosen_positions]


def build_clustering_name(clustering_algorithm: str, clustering_parameters: Dict[str, Any], normalize_vectors: bool,
                          pca_dimensions: int) -> str:
    """
    Names the folder of a clustering after its settings, so that clusterings with other settings sit next to each
    other, e.g. 'hdbscan_min_cluster_size_10_min_samples_5_cluster_selection_method_eom_normalized_pca_100', a
    parameter set to None (min_samples) being written 'none'.

    Args:
        clustering_algorithm (str): "hdbscan", "dbscan" or "kmeans".
        clustering_parameters (Dict[str, Any]): The parameters of the algorithm.
        normalize_vectors (bool): Whether the vectors are divided by their length.
        pca_dimensions (int): The number of PCA dimensions every layer is reduced to.

    Returns:
        str: The folder name.
    """
    parameter_text = "_".join(f"{name}_{clustering_parameters[name]}"
                              for name in CLUSTERING_PARAMETERS[clustering_algorithm])
    vector_text = "normalized" if normalize_vectors else "raw"
    return f"{clustering_algorithm}_{parameter_text}_{vector_text}_pca_{pca_dimensions}".replace("None", "none")


def fit_image_in_cell(image: np.ndarray, cell_height: int, cell_width: int) -> np.ndarray:
    """
    Resizes an image to the largest size fitting a cell of a mosaic without changing its ratio, centred on a black
    cell, so that the images of a cluster keep their proportions in its mosaic.

    Args:
        image (np.ndarray): The [H, W, 3] uint8 BGR image.
        cell_height (int): The height of the cell, in pixels.
        cell_width (int): The width of the cell, in pixels.

    Returns:
        np.ndarray: The [cell_height, cell_width, 3] uint8 BGR cell.
    """
    scale = min(cell_height / image.shape[0], cell_width / image.shape[1])
    resized_height, resized_width = max(1, round(image.shape[0] * scale)), max(1, round(image.shape[1] * scale))
    resized_image = cv2.resize(src=image, dsize=(resized_width, resized_height), interpolation=cv2.INTER_AREA)

    # The resized image in the middle of the black cell
    cell = np.zeros(shape=(cell_height, cell_width, 3), dtype=np.uint8)
    top, left = (cell_height - resized_height) // 2, (cell_width - resized_width) // 2
    cell[top:top + resized_height, left:left + resized_width] = resized_image
    return cell


def build_mosaic(image_paths: List[Path], grid_size: int, caption: str) -> np.ndarray:
    """
    Draws the first images of a cluster under a caption, to look at a whole cluster at once: at most grid_size rows
    of grid_size images, numbered by their rank as the links of the cluster folder (build_frame_grid), every image
    fitted without changing its ratio in a cell MOSAIC_THUMBNAIL_HEIGHT pixels high and as wide as the ratio of the
    first image gives.

    Args:
        image_paths (List[Path]): The images in rank order, the first ones drawn first, row after row.
        grid_size (int): The number of images per row, and the maximum number of rows.
        caption (str): The text of the caption bar.

    Returns:
        np.ndarray: The uint8 BGR mosaic with its caption bar.

    Raises:
        ValueError: If an image cannot be read.
    """
    shown_images = []
    for image_path in image_paths[:grid_size * grid_size]:
        # Decoded at half its size, four times faster for the JPEG frames, and again at its full size when it is then
        # lower than a cell, then released from the page cache
        image = cv2.imread(filename=str(image_path), flags=cv2.IMREAD_REDUCED_COLOR_2)
        if image is not None and image.shape[0] < MOSAIC_THUMBNAIL_HEIGHT:
            image = cv2.imread(filename=str(image_path))
        if image is None:
            raise ValueError(f"The image cannot be read : {format_path_link(path=image_path)}")
        release_file_from_page_cache(path=image_path)
        shown_images.append(image)

    # Cells with the ratio of the first image, bounded so that a panorama or a narrow strip keeps a usable mosaic
    first_height, first_width = shown_images[0].shape[:2]
    cell_width = int(np.clip(a=round(MOSAIC_THUMBNAIL_HEIGHT * first_width / first_height),
                             a_min=MOSAIC_THUMBNAIL_HEIGHT // 4, a_max=4 * MOSAIC_THUMBNAIL_HEIGHT))
    thumbnails = np.stack(arrays=[fit_image_in_cell(image=image, cell_height=MOSAIC_THUMBNAIL_HEIGHT,
                                                    cell_width=cell_width) for image in shown_images])

    # The thumbnails numbered from rank 1 on the grid, under the caption
    grid = build_frame_grid(frames=torch.from_numpy(thumbnails).permute(dims=(0, 3, 1, 2)),
                            sequence_mask=torch.ones(size=(len(shown_images),), dtype=torch.bool), frame_captions=None,
                            number_columns=grid_size, first_frame_number=1)
    return add_caption_bar(image=convert_image_to_array(image=grid), caption=caption)


def link_image(link_path: Path, image_path: Path) -> None:
    """
    Points a link of a cluster folder or of a subset of plain images to an image of the source, so that the clusters
    and the subsets are browsed without copying any image.

    Args:
        link_path (Path): The path of the link, its folder being created.
        image_path (Path): The image the link points to.
    """
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target=image_path.absolute())


def sanitize_folder_name(name: str) -> str:
    """
    Turns a category name into a folder name, the path separators it may hold being replaced.

    Args:
        name (str): The category name.

    Returns:
        str: The folder name.
    """
    return re.sub(pattern=r"[\\/]", repl="_", string=name) or "_"


def write_subset(subset_folder: Path, source: Dict[str, Any], source_images: Dict[str, Any],
                 selected_rows: Dict[int, str]) -> None:
    """
    Writes a subset of the images of a source: 'selected_images.csv' listing every chosen image with why it is
    chosen, and, when every image of the source is an independent sample, the subset ready to train: a 'data' link to
    the images of the source and a 'labels.json' keeping the chosen samples (classification, semantic segmentation)
    or the chosen COCO images with their annotations, or links to the chosen images for a folder of plain images. The
    frames of sequences and pairs are only listed, being tied to their sequence.

    Args:
        subset_folder (Path): The folder of the subset, created.
        source (Dict[str, Any]): The data source.
        source_images (Dict[str, Any]): The images of the source given by list_source_images.
        selected_rows (Dict[int, str]): The row of every chosen image with why it is chosen.
    """
    subset_folder.mkdir(parents=True)
    file_names = source_images["file_names"]
    write_csv(path=subset_folder / SELECTED_IMAGES_FILE_NAME, field_names=["row", "image", "category", "reason"],
              rows=[{"row": row, "image": file_names[row],
                     "category": "" if source_images["categories"] is None else source_images["categories"][row],
                     "reason": selected_rows[row]} for row in sorted(selected_rows)])

    # A folder of plain images: links keeping the subfolders of the images
    manifest = source_images["manifest"]
    if manifest is None:
        for row in sorted(selected_rows):
            link_image(link_path=subset_folder / file_names[row], image_path=source["data_folder"] / file_names[row])
        return

    # The chosen COCO images with their annotations, or the chosen samples, in the order list_manifest_images reads
    # the manifest formats, the other entries kept
    selected_names = {file_names[row] for row in selected_rows}
    if "images" in manifest:
        kept_images = [image for image in manifest["images"] if image["file_name"] in selected_names]
        kept_image_identifiers = {image["id"] for image in kept_images}
        subset_manifest = {**manifest, "images": kept_images,
                           "annotations": [annotation for annotation in manifest["annotations"]
                                           if annotation["image_id"] in kept_image_identifiers]}
    elif "samples" in manifest:
        subset_manifest = {**manifest, "samples": [sample for sample in manifest["samples"]
                                                   if sample["file_name"] in selected_names]}
    else:
        return
    link_folder(link_path=subset_folder / "data", target_folder=source["data_folder"])
    (subset_folder / "labels.json").write_text(data=json.dumps(subset_manifest), encoding="utf-8")


def write_cluster_folders(clustering_folder: Path, source: Dict[str, Any], source_images: Dict[str, Any],
                          ordered_cluster_rows: Dict[Tuple[str, int], np.ndarray], mosaic_grid_size: int) -> None:
    """
    Writes, for every cluster of every category, a folder of links to its images named by their rank
    ('<category>/cluster_<k>/<rank>_<image>', the noise in '<category>/noise/') and a mosaic of its first images
    ('<category>/cluster_<k>.jpg'), to look at what every cluster gathers.

    Args:
        clustering_folder (Path): The folder of the clustering.
        source (Dict[str, Any]): The data source.
        source_images (Dict[str, Any]): The images of the source given by list_source_images.
        ordered_cluster_rows (Dict[Tuple[str, int], np.ndarray]): The rows of every cluster of every category, in
            rank order.
        mosaic_grid_size (int): The number of images per row of a mosaic, and its maximum number of rows.
    """
    file_names = source_images["file_names"]
    for (category, cluster), rows in tqdm(iterable=ordered_cluster_rows.items(), desc="Writing the clusters",
                                          unit="cluster"):
        category_folder = clustering_folder / sanitize_folder_name(name=category)
        cluster_name = "noise" if cluster == NOISE_CLUSTER else f"cluster_{cluster:04d}"

        # Links named by rank, the subfolders of an image folder flattened into the name
        for rank, row in enumerate(rows, start=1):
            link_image(link_path=category_folder / cluster_name / f"{rank:05d}_{file_names[row].replace('/', '__')}",
                       image_path=source["data_folder"] / file_names[row])

        # The first images on a grid
        mosaic = build_mosaic(image_paths=[source["data_folder"] / file_names[row] for row in rows],
                              grid_size=mosaic_grid_size, caption=f"{category} - {cluster_name} - {len(rows)} images")
        cv2.imwrite(filename=str(category_folder / f"{cluster_name}.jpg"), img=mosaic)


def cluster_layer(features: np.ndarray, layer_key: str, source: Dict[str, Any], source_images: Dict[str, Any],
                  predictions: Optional[np.ndarray], class_names: Optional[List[str]], groups: Optional[List[str]],
                  clustering_folder: Path, clustering_algorithm: str, clustering_parameters: Dict[str, Any],
                  normalize_vectors: bool, pca_dimensions: int, number_representatives_per_cluster: int,
                  noise_selection: str | int, seed: int, mosaic_grid_size: int) -> List[Dict[str, Any]]:
    """
    Clusters the features of one layer of a source, per category when the source gives one per image, and writes the
    clustering folder, replaced when it exists:
    - 'clusters.csv': every image with its category, cluster, membership probability (HDBSCAN), distance to the
      centre of its cluster, rank, choice, predicted class and group.
    - 'cluster_summary.csv': every cluster with its images, its chosen images, its mean membership probability, its
      error rate (when the embedding run classifies the categories of the source) and its groups.
    - The folders of links and the mosaics of the clusters (write_cluster_folders).
    - The cluster subset and the random and evenly spaced per group baselines of the same size (write_subset), the
      last one only with a group pattern.

    Args:
        features (np.ndarray): The [number of images, D] features of the layer.
        layer_key (str): The name of the layer.
        source (Dict[str, Any]): The data source.
        source_images (Dict[str, Any]): The images of the source given by list_source_images.
        predictions (Optional[np.ndarray]): The [number of images, 2] class index and probability of every image, None
            for the runs without classification head.
        class_names (Optional[List[str]]): The class names of the embedding run, None without record.
        groups (Optional[List[str]]): The group of every image, None without group pattern.
        clustering_folder (Path): The folder of the clustering.
        clustering_algorithm (str): "hdbscan", "dbscan" or "kmeans".
        clustering_parameters (Dict[str, Any]): The parameters of the algorithm.
        normalize_vectors (bool): Whether to divide every vector by its length.
        pca_dimensions (int): The number of PCA dimensions every layer is reduced to.
        number_representatives_per_cluster (int): The images kept per cluster in the subset.
        noise_selection (str | int): "all", "none" or the number of noise images drawn for the subset.
        seed (int): The seed of the draws and of the clustering.
        mosaic_grid_size (int): The number of images per row of a mosaic, and its maximum number of rows.

    Returns:
        List[Dict[str, Any]]: The summary of every category: its images, clusters, noise images, chosen images,
            clustered dimensions, kept variance, and times of the PCA and of the clustering of the vectors.
    """
    if clustering_folder.exists():
        shutil.rmtree(path=clustering_folder)
    clustering_folder.mkdir(parents=True)
    file_names = source_images["file_names"]
    number_images = len(file_names)
    categories = source_images["categories"] if source_images["categories"] is not None else \
        [ALL_IMAGES_CATEGORY] * number_images
    category_array = np.array(categories)

    # Predicted class names, and the error rate when the embedding run classifies the categories of the source
    predicted_names = None
    if predictions is not None:
        predicted_names = [class_names[int(class_index)] if class_names is not None else f"class_{int(class_index)}"
                           for class_index in predictions[:, 0]]
    errors_known = predicted_names is not None and source_images["categories"] is not None and \
        class_names is not None and set(categories) <= set(class_names)

    image_rows = {}
    summary_rows = []
    category_summaries = []
    ordered_cluster_rows = {}
    subsets = {CLUSTER_SUBSET_FOLDER_NAME: {}, RANDOM_SUBSET_FOLDER_NAME: {}, GROUP_SUBSET_FOLDER_NAME: {}}
    for category_position, category in enumerate(sorted(set(categories))):
        category_rows = np.flatnonzero(category_array == category)
        category_file_names = [file_names[row] for row in category_rows]

        # The vectors of the category prepared then clustered, the time of the PCA and of the clustering kept to compare
        # the PCA settings
        prepared_vectors, preparation = prepare_vectors(vectors=features[category_rows],
                                                        normalize_vectors=normalize_vectors,
                                                        pca_dimensions=pca_dimensions, seed=seed)
        start_time = time.perf_counter()
        labels, probabilities = cluster_vectors(vectors=prepared_vectors, clustering_algorithm=clustering_algorithm,
                                                clustering_parameters=clustering_parameters, seed=seed)
        clustering_seconds = time.perf_counter() - start_time
        distances = compute_centre_distances(vectors=prepared_vectors, labels=labels)
        ranks = rank_cluster_members(labels=labels, distances=distances, file_names=category_file_names)

        # The cluster subset, then the two baselines of the same size, every draw seeded per category
        random_generator = np.random.default_rng(seed=[seed, category_position])
        representatives = select_cluster_representatives(
            labels=labels, ranks=ranks, number_representatives_per_cluster=number_representatives_per_cluster,
            noise_selection=noise_selection, random_generator=random_generator)
        subsets[CLUSTER_SUBSET_FOLDER_NAME].update({int(category_rows[position]): reason
                                                    for position, reason in representatives.items()})
        budget = len(representatives)
        for position in select_random_baseline(number_images=len(category_rows), budget=budget,
                                               random_generator=random_generator):
            subsets[RANDOM_SUBSET_FOLDER_NAME][int(category_rows[position])] = "random"
        if groups is not None:
            group_positions, chosen_groups = select_evenly_spaced_per_group(
                file_names=category_file_names, groups=[groups[row] for row in category_rows], budget=budget)
            for position, group in zip(group_positions, chosen_groups):
                subsets[GROUP_SUBSET_FOLDER_NAME][int(category_rows[position])] = f"evenly spaced in {group}"

        # Every image with its cluster, and every cluster with its rows in rank order
        for position, row in enumerate(category_rows):
            image_rows[int(row)] = {
                "row": int(row), "image": file_names[row], "category": category, "cluster": int(labels[position]),
                "probability": "" if probabilities is None else f"{probabilities[position]:.4f}",
                "distance_to_centre": "" if np.isnan(distances[position]) else f"{distances[position]:.6f}",
                "rank": int(ranks[position]), "selected": representatives.get(position, ""),
                "predicted_class": "" if predicted_names is None else predicted_names[row],
                "predicted_probability": "" if predictions is None else f"{predictions[row, 1]:.4f}",
                "group": "" if groups is None else groups[row]}
        for cluster in np.unique(labels):
            member_positions = np.flatnonzero(labels == cluster)
            ordered_cluster_rows[(category, int(cluster))] = category_rows[member_positions[np.argsort(
                ranks[member_positions])]]

            # The summary of the cluster: its choices, its probability, its errors and its groups
            member_rows = category_rows[member_positions]
            summary_row = {"category": category, "cluster": int(cluster), "images": len(member_positions),
                           "selected": sum(position in representatives for position in member_positions),
                           "mean_probability": "" if probabilities is None else
                           f"{probabilities[member_positions].mean():.4f}",
                           "error_rate": "", "groups": "", "largest_group": "", "largest_group_share": ""}
            if errors_known:
                summary_row["error_rate"] = f"{np.mean([predicted_names[row] != category for row in member_rows]):.4f}"
            if groups is not None:
                member_groups = [groups[row] for row in member_rows]
                largest_group = max(set(member_groups), key=member_groups.count)
                largest_group_share = member_groups.count(largest_group) / len(member_groups)
                summary_row.update({"groups": len(set(member_groups)), "largest_group": largest_group,
                                    "largest_group_share": f"{largest_group_share:.4f}"})
            summary_rows.append(summary_row)

        # The summary of the category for the console
        number_noise = int((labels == NOISE_CLUSTER).sum())
        category_summaries.append({"layer": layer_key, "category": category, "images": len(category_rows),
                                   "clusters": len(set(labels.tolist()) - {NOISE_CLUSTER}), "noise": number_noise,
                                   "selected": budget, "dimensions": preparation["dimensions"],
                                   "kept_variance": preparation["kept_variance"],
                                   "zero_vectors": preparation["zero_vectors"],
                                   "pca_seconds": preparation["pca_seconds"],
                                   "clustering_seconds": clustering_seconds})

    # The tables of the images and of the clusters
    write_csv(path=clustering_folder / "clusters.csv",
              field_names=["row", "image", "category", "cluster", "probability", "distance_to_centre", "rank",
                           "selected", "predicted_class", "predicted_probability", "group"],
              rows=[image_rows[row] for row in sorted(image_rows)])
    write_csv(path=clustering_folder / "cluster_summary.csv",
              field_names=["category", "cluster", "images", "selected", "mean_probability", "error_rate", "groups",
                           "largest_group", "largest_group_share"], rows=summary_rows)

    # The links and mosaics of the clusters, then the three subsets
    write_cluster_folders(clustering_folder=clustering_folder, source=source, source_images=source_images,
                          ordered_cluster_rows=ordered_cluster_rows, mosaic_grid_size=mosaic_grid_size)
    for subset_name, selected_rows in subsets.items():
        if subset_name == GROUP_SUBSET_FOLDER_NAME and groups is None:
            continue
        write_subset(subset_folder=clustering_folder / subset_name, source=source, source_images=source_images,
                     selected_rows=selected_rows)
    return category_summaries


def print_category_summaries(category_summaries: List[Dict[str, Any]], clustering_folder: Path) -> None:
    """
    Prints the summary of every category of a clustering, to compare settings at a glance: images, clusters, noise
    images and their share, chosen images, clustered dimensions with the variance kept by the PCA and its time (or
    the vectors clustered as they are when the PCA had nothing to reduce), and the time of the clustering.

    Args:
        category_summaries (List[Dict[str, Any]]): The summaries given by cluster_layer.
        clustering_folder (Path): The folder of the clustering.
    """
    print_color(color="green", output=f"Clustering written : {format_path_link(path=clustering_folder)}")
    for summary in category_summaries:
        variance_text = " not reduced by the PCA (no more dimensions or images than asked)" \
            if summary["kept_variance"] is None else \
            f" ({100 * summary['kept_variance']:.1f} % of the variance, PCA in {summary['pca_seconds']:.1f} Seconds)"
        zero_text = "" if summary["zero_vectors"] == 0 else f", {summary['zero_vectors']} zero vectors"
        print_color(color="blue", indent=1,
                    output=f"{summary['layer']} - {summary['category']}: {summary['images']} images, "
                           f"{summary['clusters']} clusters, {summary['noise']} noise images "
                           f"({100 * summary['noise'] / summary['images']:.1f} %), {summary['selected']} chosen, "
                           f"{summary['dimensions']} dimensions{variance_text}{zero_text}, clustered in "
                           f"{summary['clustering_seconds']:.1f} Seconds")


def embed_and_cluster_sources(experiment_folder: Path, iteration: int, use_averaged_weights: bool,
                              data_sources: List[str | Path], convolution_block_indices: List[int],
                              fully_connected_layer_indices: List[int], batch_size: int, number_workers: int,
                              normalize_vectors: bool, pca_dimensions: int,
                              clustering_algorithm: str, clustering_parameters: Dict[str, Any],
                              number_representatives_per_cluster: int, noise_selection: str | int,
                              group_pattern: Optional[str], seed: int, mosaic_grid_size: int, output_root: Path,
                              device: torch.device) -> List[Path]:
    """
    Runs the three steps of the script with the settings of its main block: the checks of the layers and of the
    clustering settings, the extraction of every data source (loaded when stored), and the clustering of every layer
    of every source.

    Args:
        experiment_folder (Path): The run folder of the embedding run.
        iteration (int): The iteration of its checkpoint.
        use_averaged_weights (bool): Whether to embed with the moving average of its weights.
        data_sources (List[str | Path]): "run", configuration files, split folders or folders of plain images.
        convolution_block_indices (List[int]): The asked convolution blocks, counted from 1.
        fully_connected_layer_indices (List[int]): The asked hidden fully connected layers, counted from 1.
        batch_size (int): The number of images run together.
        number_workers (int): The number of loader worker processes reading the images.
        normalize_vectors (bool): Whether to divide every vector by its length before the clustering.
        pca_dimensions (int): The number of PCA dimensions every layer is reduced to.
        clustering_algorithm (str): "hdbscan", "dbscan" or "kmeans".
        clustering_parameters (Dict[str, Any]): The parameters of the algorithm.
        number_representatives_per_cluster (int): The images kept per cluster in the subset.
        noise_selection (str | int): "all", "none" or the number of noise images drawn for the subset.
        group_pattern (Optional[str]): The regular expression giving the group of an image, or None.
        seed (int): The seed of the draws and of the clustering.
        mosaic_grid_size (int): The number of images per row of a mosaic, and its maximum number of rows.
        output_root (Path): The root folder of the extractions.
        device (torch.device): The device running the model.

    Returns:
        List[Path]: The folder of every clustering, per source then per layer.
    """
    # Every setting checked before the model runs on any image
    check_clustering_settings(clustering_algorithm=clustering_algorithm, clustering_parameters=clustering_parameters,
                              pca_dimensions=pca_dimensions,
                              number_representatives_per_cluster=number_representatives_per_cluster,
                              noise_selection=noise_selection, group_pattern=group_pattern,
                              mosaic_grid_size=mosaic_grid_size)
    embedder = load_embedder(experiment_folder=experiment_folder, iteration=iteration,
                             use_averaged_weights=use_averaged_weights, device=device)
    check_layer_requests(embedder=embedder, convolution_block_indices=convolution_block_indices,
                         fully_connected_layer_indices=fully_connected_layer_indices)
    block_requests, fully_connected_requests = build_layer_requests(
        convolution_block_indices=convolution_block_indices,
        fully_connected_layer_indices=fully_connected_layer_indices)
    sources = resolve_data_sources(data_sources=data_sources,
                                   embedder_experiment_configuration=embedder["experiment_configuration"])
    clustering_name = build_clustering_name(clustering_algorithm=clustering_algorithm,
                                            clustering_parameters=clustering_parameters,
                                            normalize_vectors=normalize_vectors, pca_dimensions=pca_dimensions)
    weights_folder_name = f"Iteration_{iteration}" + ("_EMA" if use_averaged_weights else "")

    # The images of every source listed before the first extraction, so that a source without images stops the script
    # before any inference
    images_of_sources = [list_source_images(source=source) for source in sources]

    clustering_folders = []
    for source, source_images in zip(sources, images_of_sources):
        # The categories and groups of the images of the source, and the features of every asked layer
        number_categories = "no" if source_images["categories"] is None else len(set(source_images["categories"]))
        print_color(color="blue", add_separators=True,
                    output=f"Source {source['name']} ({', '.join(source['roles'])}): "
                           f"{len(source_images['file_names'])} images, {number_categories} categories : "
                           f"{format_path_link(path=source['folder'])}")
        groups = find_image_groups(file_names=source_images["file_names"], group_pattern=group_pattern)
        if groups is not None:
            print_color(color="blue", indent=1, output=f"{len(set(groups))} groups, {groups.count(NO_GROUP)} images "
                                                       f"matching no group")
        extraction_folder = output_root / experiment_folder.name / weights_folder_name / source["name"]
        outputs = extract_source_features(embedder=embedder, source=source, source_images=source_images,
                                          block_requests=block_requests,
                                          fully_connected_requests=fully_connected_requests,
                                          extraction_folder=extraction_folder, batch_size=batch_size,
                                          number_workers=number_workers)

        # Every layer clustered on its own
        for layer_key in [*block_requests, *fully_connected_requests]:
            clustering_folder = extraction_folder / "clusters" / layer_key / clustering_name
            category_summaries = cluster_layer(
                features=outputs[layer_key], layer_key=layer_key, source=source, source_images=source_images,
                predictions=outputs[PREDICTIONS_OUTPUT] if PREDICTIONS_OUTPUT in outputs else None,
                class_names=embedder["class_names"], groups=groups, clustering_folder=clustering_folder,
                clustering_algorithm=clustering_algorithm, clustering_parameters=clustering_parameters,
                normalize_vectors=normalize_vectors, pca_dimensions=pca_dimensions,
                number_representatives_per_cluster=number_representatives_per_cluster,
                noise_selection=noise_selection, seed=seed, mosaic_grid_size=mosaic_grid_size)
            print_category_summaries(category_summaries=category_summaries, clustering_folder=clustering_folder)
            clustering_folders.append(clustering_folder)

        # The feature matrices of the source released from the page cache once clustered, their mappings closed first
        output_paths = [get_output_path(extraction_folder=extraction_folder, output_key=output_key)
                        for output_key in outputs]
        del outputs
        for output_path in output_paths:
            release_file_from_page_cache(path=output_path)
    return clustering_folders


if __name__ == "__main__":
    repository_folder = Path(__file__).absolute().parents[2]
    # Trained run embedding the images, of any task for the convolution blocks and a classification run for the fully
    # connected layers, with the iteration of its checkpoint, and True to embed with the moving average of its weights
    experiment_folder = (repository_folder / "ficha-experiments-empty-bins" /
                         "CLS_IN-KAR-512_SE-RES-BN-B1x64.2x64.2x64.16x128.20x128.20x128-PT_HFLAT.512.256-HLN-CWB_MC2")
    iteration = 200
    use_averaged_weights = False
    # Images to embed, a list of:
    # - "run": the train, validation and test splits of the training_configuration.yaml of the run
    # - a configuration file (.yaml) of any task: its train, validation and test splits
    # - a split folder of any task, holding 'data' and 'labels.json' (COCO images, samples, frames of the sequences
    #   and of the pairs)
    # - a folder of plain images, searched with its subfolders
    # A folder given twice (a test split that is the validation folder) is embedded once
    data_sources = ["run"]
    # Layers stored and clustered, counted from 1: the convolution blocks of the backbone (their whole output before
    # their pooling, channels x height x width values per image, at most MAXIMUM_BLOCK_VALUES = 32,768: block 6 of the
    # ficha runs at 512 pixels, 128 x 16 x 16, and not block 5, 128 x 32 x 32) and the hidden fully connected layers of
    # a classification head (their output after their ReLU, the class logits layer excluded)
    convolution_block_indices = [6]
    fully_connected_layer_indices = [1, 2]
    # Images run together and loader worker processes reading them
    batch_size = 32
    number_workers = 2
    # Vectors divided by their length before the clustering, so that the distances compare which features are active
    # rather than how strongly, and number of PCA dimensions every layer is reduced to (its main directions of
    # variation), a layer with fewer dimensions being clustered as it is: the clustering time grows with the
    # dimensions (see the timings at the top of the file: 200 dimensions halve the HDBSCAN time of 400 on block 6)
    normalize_vectors = True
    pca_dimensions = 200
    # Clustering algorithm and its parameters, every category of a classification manifest clustered on its own:
    # - "hdbscan": dense regions of any shape, the isolated images being noise; min_cluster_size (images of the
    #   smallest cluster), min_samples (neighbours making an image dense, None for min_cluster_size, higher giving more
    #   noise) and cluster_selection_method ("eom" for the most stable clusters, "leaf" for smaller, more uniform ones)
    # - "dbscan": dense regions at a fixed distance eps between the prepared vectors (at most 2 between normalised
    #   vectors), min_samples neighbours within eps making an image dense, the others being noise
    # - "kmeans": number_clusters clusters per category, every image in one, best of number_initializations runs
    clustering_algorithm = "hdbscan"
    clustering_parameters = {"hdbscan": {"min_cluster_size": 10, "min_samples": 5, "cluster_selection_method": "eom"},
                             "dbscan": {"eps": 0.2, "min_samples": 5},
                             "kmeans": {"number_clusters": 200, "number_initializations": 3}}[clustering_algorithm]
    # Images of the cluster subset: the representatives of every cluster (its images closest to its centre) and the
    # noise images, "all", "none" or a number drawn at random. The random baseline and the baseline evenly spaced per
    # group take as many images per category
    number_representatives_per_cluster = 5
    noise_selection = "all"
    # Regular expression giving the group of every image, searched in its file name relative to its data folder: the
    # first capturing group taking part in the match (the whole match without capturing group), the images it does not
    # match forming the group "(no group)". Any pattern fits the file names of a dataset, e.g.
    # r"^(video_t\d+_[\d-]+_[\d-]+)_" for the video of the ficha frames 'video_t0053_2026-09-25_05-51-30_0037018ms.jpg',
    # r"^([^/]+)/" for the first subfolder of a folder of plain images, or r"^(video_\d+)_|^(camera_\d+)/" for one group
    # per alternative. The cluster summary counts the groups of every cluster, and the baseline evenly spaced per group
    # spreads its images evenly over the groups, the images of a group ordered by their names (digit runs compared as
    # numbers). None for no group and no such baseline
    group_pattern = r"^(video_t\d+_[\d-]+_[\d-]+)_"
    # Seed of the draws (noise images, random baseline) and of k-means
    seed = 0
    # Images per row of the mosaic of a cluster, which shows at most as many rows, the images closest to the centre
    # first
    mosaic_grid_size = 6
    # Root folder of the extractions, '<output_root>/<run name>/Iteration_<n>[_EMA]/<source>/'
    output_root = experiment_folder.parent / "embeddings"

    embed_and_cluster_sources(experiment_folder=experiment_folder, iteration=iteration,
                              use_averaged_weights=use_averaged_weights, data_sources=data_sources,
                              convolution_block_indices=convolution_block_indices,
                              fully_connected_layer_indices=fully_connected_layer_indices, batch_size=batch_size,
                              number_workers=number_workers, normalize_vectors=normalize_vectors,
                              pca_dimensions=pca_dimensions, clustering_algorithm=clustering_algorithm,
                              clustering_parameters=clustering_parameters,
                              number_representatives_per_cluster=number_representatives_per_cluster,
                              noise_selection=noise_selection, group_pattern=group_pattern, seed=seed,
                              mosaic_grid_size=mosaic_grid_size, output_root=output_root, device=get_device())
