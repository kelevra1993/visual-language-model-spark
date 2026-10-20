"""
Tests of scripts/embedding/cluster_image_embeddings.py, the embedding and clustering of the images of datasets with a
trained run to pick diverse representatives of video frames: the checks of the asked layers and of the clustering
settings, the features of the asked layers in the row order of the images, the relaunch that loads the stored features
and computes only the missing layers, the extraction continued after an interruption, the refusal of the features of
another checkpoint or of changed images, the data sources, the images of every manifest format, the three clustering
algorithms, the images kept by the clustering and by the baselines, the release of the files read from the page
cache, the clusters, mosaics and kept images written for classification splits, COCO splits and folders of plain
images, and the table of the kept images printed at the end of a run. The embedding run is a classification run
trained 2 iterations on the synthetic shapes.
"""

import csv
import ctypes
import json
import mmap
import os
import re
import shutil

import cv2
import numpy as np
import pytest
import torch
import yaml

from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from scripts.embedding.cluster_image_embeddings import (ALL_IMAGES_CATEGORY, EXTRACTION_RECORD_FILE_NAME,
                                                        FINGERPRINT_KEYS, IMAGES_FILE_NAME, ISOLATED_CLUSTER,
                                                        ISOLATED_IMAGE_REASON, ISOLATED_IMAGES_NAME,
                                                        KEPT_AT_RANDOM_FOLDER_NAME, KEPT_BY_CLUSTERING_FOLDER_NAME,
                                                        KEPT_EVENLY_SPACED_FOLDER_NAME, KEPT_IMAGES_FILE_NAME,
                                                        MOSAIC_THUMBNAIL_HEIGHT, NO_GROUP, PREDICTIONS_OUTPUT,
                                                        TOTAL_CATEGORY, allocate_budget_over_groups,
                                                        build_kept_images_table, build_layer_requests,
                                                        check_clustering_settings, build_mosaic, check_layer_requests,
                                                        cluster_vectors, compute_batch_outputs,
                                                        describe_source_folder, embed_and_cluster_sources,
                                                        extract_source_features, find_image_groups,
                                                        list_source_images, load_embedder, natural_sort_key,
                                                        prepare_vectors, rank_cluster_members,
                                                        release_file_from_page_cache, resolve_data_sources,
                                                        select_evenly_spaced_per_group,
                                                        select_kept_images_by_clustering, select_random_baseline)
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.preprocessing import preprocess_image
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.utilities.tensor_utilities import get_device
from vision_training.utilities.visualization import CAPTION_BAR_HEIGHT
from tests.trainers.test_classification_trainer import SMOKE_IMAGE_SIZE, build_trainer, write_smoke_configuration

# Iteration of the checkpoint of the embedding run, trained 2 iterations
EMBEDDING_ITERATION = 2

# Hidden layers of the classification head of the embedding run
HIDDEN_LAYER_SIZES = [32, 16]

# Shape of the output of every block of the synthetic shapes backbone before its pooling, for 64 pixel images: block 1
# gives 65,536 values per image, more than MAXIMUM_BLOCK_VALUES, and block 2 exactly 32,768
EXPECTED_BLOCK_SHAPES = [[16, 64, 64], [32, 32, 32], [32, 16, 16], [64, 8, 8], [64, 4, 4], [128, 2, 2]]

# Number of images of the test split of the synthetic shapes, run by batches of 4 in the extraction tests (rows 0 to 3,
# 4 to 7, then 8 and 9)
NUMBER_TEST_IMAGES = 10
EXTRACTION_BATCH_SIZE = 4

# BGR colours of the videos of the generated video splits, every video having its own colour
VIDEO_COLOURS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

# Pattern giving the video of a frame of the generated video splits, e.g. 'video_square_2' for
# 'video_square_2_frame_4.png'
VIDEO_GROUP_PATTERN = r"^(video_[a-z]+_\d+)_frame"

# HDBSCAN settings of the end-to-end tests, small enough for videos of 6 frames
HDBSCAN_PARAMETERS = {"min_cluster_size": 3, "min_samples": 2, "cluster_selection_method": "eom"}


class BatchComputationSpy:
    """
    Stand-in of compute_batch_outputs recording what every batch computes (the asked layers and the number of images)
    and raising an error at a chosen batch, to simulate an interrupted extraction.
    """

    def __init__(self, failing_call: Optional[int] = None) -> None:
        """
        Keeps the batch at which to raise.

        Args:
            failing_call (Optional[int]): The number of the call raising the simulated interruption, counted from 1,
                or None to never raise.
        """
        self.failing_call = failing_call
        self.calls = []

    def __call__(self, **keyword_arguments: Any) -> Dict[str, torch.Tensor]:
        """
        Records the call, then raises at the chosen batch or computes the outputs with the real function.

        Args:
            **keyword_arguments (Any): The keyword arguments of compute_batch_outputs.

        Returns:
            Dict[str, torch.Tensor]: The outputs of the batch.

        Raises:
            RuntimeError: At the chosen batch.
        """
        self.calls.append({"layers": sorted([*keyword_arguments["block_requests"],
                                             *keyword_arguments["fully_connected_requests"]]),
                           "output_mode": keyword_arguments["output_mode"],
                           "images": keyword_arguments["images"].shape[0]})
        if len(self.calls) == self.failing_call:
            raise RuntimeError("Simulated interruption of the extraction.")
        return compute_batch_outputs(**keyword_arguments)


@pytest.fixture(scope="module")
def embedding_run(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    """
    Trains the embedding run of the tests once for the module: a classification run with two hidden layers trained 2
    iterations on a generated synthetic shapes dataset with 64 pixel images.

    Args:
        tmp_path_factory (pytest.TempPathFactory): The pytest factory of the module temporary directories.

    Returns:
        Dict[str, Path]: The 'experiment_folder' of the run and the 'dataset_folder' of its splits.
    """
    temporary_directory = tmp_path_factory.mktemp(basename="embedding_run")
    dataset_folder = temporary_directory / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 12, "validation": 6, "test": NUMBER_TEST_IMAGES}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=temporary_directory, dataset_folder=dataset_folder, number_iterations=EMBEDDING_ITERATION,
        head_settings={"fully_connected": HIDDEN_LAYER_SIZES}, evaluation_settings={}))
    trainer.run_training_loop()
    experiment_folder = trainer.experiment_folder
    del trainer
    return {"experiment_folder": experiment_folder, "dataset_folder": dataset_folder}


@pytest.fixture(scope="module")
def cpu_embedder(embedding_run: Dict[str, Path]) -> Dict[str, Any]:
    """
    Loads the embedding run on the CPU, whose float32 computations do not depend on the batch composition, for the
    tests comparing features.

    Args:
        embedding_run (Dict[str, Path]): The embedding run of the module.

    Returns:
        Dict[str, Any]: The description of the embedding run given by load_embedder.
    """
    return load_embedder(experiment_folder=embedding_run["experiment_folder"], iteration=EMBEDDING_ITERATION,
                         use_averaged_weights=False, device=torch.device("cpu"))


def extract_split_features(embedder: Dict[str, Any], split_folder: Path, convolution_block_indices: List[int],
                           fully_connected_layer_indices: List[int], extraction_folder: Path,
                           number_workers: int = 0) -> Dict[str, np.ndarray]:
    """
    Extracts the features of the asked layers of a split folder by batches of EXTRACTION_BATCH_SIZE images, as the
    script does for one source.

    Args:
        embedder (Dict[str, Any]): The description of the embedding run.
        split_folder (Path): The split folder.
        convolution_block_indices (List[int]): The asked convolution blocks.
        fully_connected_layer_indices (List[int]): The asked hidden fully connected layers.
        extraction_folder (Path): The extraction folder.
        number_workers (int): The number of loader worker processes. Defaults to 0.

    Returns:
        Dict[str, np.ndarray]: The matrix of every output, keyed by output name.
    """
    source = describe_source_folder(folder=split_folder, role="test")
    block_requests, fully_connected_requests = build_layer_requests(
        convolution_block_indices=convolution_block_indices,
        fully_connected_layer_indices=fully_connected_layer_indices)
    return extract_source_features(embedder=embedder, source=source, source_images=list_source_images(source=source),
                                   block_requests=block_requests, fully_connected_requests=fully_connected_requests,
                                   extraction_folder=extraction_folder, batch_size=EXTRACTION_BATCH_SIZE,
                                   number_workers=number_workers)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """
    Reads the rows of a CSV file written by the script.

    Args:
        path (Path): The CSV file.

    Returns:
        List[Dict[str, str]]: The value of every column of every row.
    """
    with open(file=path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(f=csv_file))


def count_cached_pages(path: Path) -> int:
    """
    Counts the pages of a file held in the page cache of Linux, with mincore on a mapping of the file that reads none
    of its pages, to check that the script releases the files it has read.

    Args:
        path (Path): The file.

    Returns:
        int: The number of its pages in the page cache.

    Raises:
        OSError: If the file cannot be mapped or its pages queried.
    """
    libc = ctypes.CDLL(name=None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    file_size = path.stat().st_size
    residency = (ctypes.c_ubyte * ((file_size + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()

    # The file mapped without reading it, then the kernel asked which of its pages are in memory
    file_descriptor = os.open(path=path, flags=os.O_RDONLY)
    try:
        address = libc.mmap(None, file_size, mmap.PROT_READ, mmap.MAP_SHARED, file_descriptor, 0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), f"The file cannot be mapped : {path}")
        try:
            if libc.mincore(address, file_size, residency) != 0:
                raise OSError(ctypes.get_errno(), f"The pages of the file cannot be queried : {path}")
        finally:
            libc.munmap(address, file_size)
    finally:
        os.close(fd=file_descriptor)
    return sum(page & 1 for page in residency)


def draw_video_frame(video_index: int, frame_index: int) -> np.ndarray:
    """
    Draws one frame of a generated video: the shape of the video (a disc, a square or a triangle, by video index) in
    the colour of the video on a gray background, moved by one pixel per frame, so that the frames of a video are near
    duplicates and the videos differ.

    Args:
        video_index (int): The index of the video, giving its shape and colour.
        frame_index (int): The index of the frame in the video.

    Returns:
        np.ndarray: The [64, 48, 3] uint8 BGR frame, taller than wide so that the keep ratio padding is used.
    """
    frame = np.full(shape=(64, 48, 3), fill_value=96, dtype=np.uint8)
    colour = VIDEO_COLOURS[video_index]
    centre_x, centre_y = 16 + frame_index, 24 + frame_index
    if video_index % 3 == 0:
        cv2.circle(img=frame, center=(centre_x, centre_y), radius=10, color=colour, thickness=-1)
    elif video_index % 3 == 1:
        cv2.rectangle(img=frame, pt1=(centre_x - 10, centre_y - 10), pt2=(centre_x + 10, centre_y + 10), color=colour,
                      thickness=-1)
    else:
        triangle = np.array([[centre_x, centre_y - 12], [centre_x - 12, centre_y + 10], [centre_x + 12, centre_y + 10]])
        cv2.fillPoly(img=frame, pts=[triangle], color=colour)
    return frame


def write_video_split(split_folder: Path, category_names: List[str], number_videos: int,
                      number_frames: int) -> Dict[str, Any]:
    """
    Writes a classification split of generated videos, the kind of dataset the script reduces: every category holds
    several videos of near duplicate frames named 'video_<category>_<video>_frame_<frame>.png', every video of the
    split having its own shape and colour.

    Args:
        split_folder (Path): The split folder, receiving 'data' and 'labels.json'.
        category_names (List[str]): The categories of the split.
        number_videos (int): The number of videos of every category.
        number_frames (int): The number of frames of every video.

    Returns:
        Dict[str, Any]: The manifest of the split.
    """
    data_folder = split_folder / "data"
    data_folder.mkdir(parents=True)
    samples = []
    for category_index, category_name in enumerate(category_names):
        for video_index in range(number_videos):
            for frame_index in range(number_frames):
                file_name = f"video_{category_name}_{video_index}_frame_{frame_index}.png"
                cv2.imwrite(filename=str(data_folder / file_name),
                            img=draw_video_frame(video_index=category_index * number_videos + video_index,
                                                 frame_index=frame_index))
                samples.append({"id": len(samples) + 1, "file_name": file_name, "category_id": category_index + 1})
    manifest = {"categories": [{"id": category_index + 1, "name": category_name}
                               for category_index, category_name in enumerate(category_names)],
                "samples": samples}
    (split_folder / "labels.json").write_text(data=json.dumps(manifest))
    return manifest


def run_embedding_script(embedding_run: Dict[str, Path], data_sources: List[str | Path], output_root: Path,
                         clustering_algorithm: str, clustering_parameters: Dict[str, Any],
                         group_pattern: Optional[str], number_representatives_per_cluster: int = 2,
                         isolated_image_selection: str | int = "all") -> List[Path]:
    """
    Runs the whole script on the embedding run with the whole output of block 4 (64 x 8 x 8 values) and the first
    hidden layer (32 values), normalised vectors reduced to 8 PCA dimensions when there are more than 8 images, and
    mosaics of 3 x 3 images.

    Args:
        embedding_run (Dict[str, Path]): The embedding run of the module.
        data_sources (List[str | Path]): The data sources.
        output_root (Path): The root folder of the extractions.
        clustering_algorithm (str): The clustering algorithm.
        clustering_parameters (Dict[str, Any]): Its parameters.
        group_pattern (Optional[str]): The regular expression giving the group of an image.
        number_representatives_per_cluster (int): The images kept per cluster. Defaults to 2.
        isolated_image_selection (str | int): The isolated images kept. Defaults to "all".

    Returns:
        List[Path]: The folder of every clustering.
    """
    return embed_and_cluster_sources(experiment_folder=embedding_run["experiment_folder"],
                                     iteration=EMBEDDING_ITERATION, use_averaged_weights=False,
                                     data_sources=data_sources, convolution_block_indices=[4],
                                     fully_connected_layer_indices=[1], batch_size=8, number_workers=0,
                                     normalize_vectors=True, pca_dimensions=8,
                                     clustering_algorithm=clustering_algorithm,
                                     clustering_parameters=clustering_parameters,
                                     number_representatives_per_cluster=number_representatives_per_cluster,
                                     isolated_image_selection=isolated_image_selection, group_pattern=group_pattern,
                                     seed=0,
                                     mosaic_grid_size=3, output_root=output_root, device=get_device())


def test_layer_requests_are_checked_against_the_run(cpu_embedder: Dict[str, Any]) -> None:
    """
    Verifies that the blocks and hidden layers asked exist in the embedding run, counted from 1, every asked block
    giving at most 32,768 values per image (block 2 exactly, block 1 twice as many), every problem being listed with
    the valid range before any inference, that the fully connected layers are refused for the runs of another task,
    whose blocks remain usable, and that a backbone normalising the frames of a sequence together is refused.

    Args:
        cpu_embedder (Dict[str, Any]): The embedding run loaded on the CPU.
    """
    assert cpu_embedder["block_shapes"] == EXPECTED_BLOCK_SHAPES
    assert cpu_embedder["hidden_layer_sizes"] == HIDDEN_LAYER_SIZES
    assert cpu_embedder["class_names"] == list(SHAPE_CATEGORIES)
    check_layer_requests(embedder=cpu_embedder, convolution_block_indices=[2, 6], fully_connected_layer_indices=[1, 2])

    # Blocks outside the backbone, a block too large to store whole, a hidden layer outside the head and duplicates
    with pytest.raises(expected_exception=ValueError) as error_information:
        check_layer_requests(embedder=cpu_embedder, convolution_block_indices=[0, 1, 7, 1],
                             fully_connected_layer_indices=[3])
    assert str(error_information.value).splitlines() == [
        "Invalid layers asked:", "- convolution_block_indices holds duplicates: [0, 1, 7, 1].",
        "- Convolution block 0 asked, the backbone has 6 blocks (1 to 6).",
        "- Convolution block 1 gives 65,536 values per image (16 channels of 64 x 64), more than the 32,768 allowed: "
        "ask a later block.",
        "- Convolution block 7 asked, the backbone has 6 blocks (1 to 6).",
        "- Fully connected layer 3 asked, the classification head has 2 hidden layers (1 to 2), the class logits "
        "layer excluded."]

    # No layer
    with pytest.raises(expected_exception=ValueError, match="No layer asked"):
        check_layer_requests(embedder=cpu_embedder, convolution_block_indices=[], fully_connected_layer_indices=[])

    # A run of another task has blocks but no hidden layer to embed
    detection_embedder = {**cpu_embedder, "task": "object_detection", "hidden_layer_sizes": None, "output_mode": None}
    check_layer_requests(embedder=detection_embedder, convolution_block_indices=[3], fully_connected_layer_indices=[])
    with pytest.raises(expected_exception=ValueError, match="is a object_detection run: only the hidden layers"):
        check_layer_requests(embedder=detection_embedder, convolution_block_indices=[3],
                             fully_connected_layer_indices=[1])

    # A frame normalised with the other frames of its sequence has no embedding of its own
    with pytest.raises(expected_exception=ValueError, match="a single image has no defined embedding"):
        check_layer_requests(embedder={**cpu_embedder, "has_sequence_normalization": True},
                             convolution_block_indices=[3], fully_connected_layer_indices=[])


def test_clustering_settings_are_checked() -> None:
    """
    Verifies that the valid settings of the three algorithms are accepted and that every invalid clustering or
    selection setting is listed before any inference: an unknown algorithm, missing parameters, out of range values,
    the PCA dimensions, the representatives, the isolated image selection, the group pattern and the mosaic grid.
    """
    valid_settings = {"pca_dimensions": 100, "number_representatives_per_cluster": 1,
                      "isolated_image_selection": "all", "group_pattern": VIDEO_GROUP_PATTERN, "mosaic_grid_size": 1}
    for clustering_algorithm, clustering_parameters in [("hdbscan", {**HDBSCAN_PARAMETERS, "min_samples": None}),
                                                        ("dbscan", {"eps": 0.5, "min_samples": 1}),
                                                        ("kmeans", {"number_clusters": 2,
                                                                    "number_initializations": 1})]:
        check_clustering_settings(clustering_algorithm=clustering_algorithm,
                                  clustering_parameters=clustering_parameters, **valid_settings)
    check_clustering_settings(clustering_algorithm="kmeans",
                              clustering_parameters={"number_clusters": 2, "number_initializations": 1},
                              pca_dimensions=10, number_representatives_per_cluster=3, isolated_image_selection=4,
                              group_pattern=None, mosaic_grid_size=6)

    # Every invalid value of one call listed
    with pytest.raises(expected_exception=ValueError) as error_information:
        check_clustering_settings(clustering_algorithm="hdbscan",
                                  clustering_parameters={"min_cluster_size": 1, "min_samples": 0,
                                                         "cluster_selection_method": "best"},
                                  pca_dimensions=0, number_representatives_per_cluster=0,
                                  isolated_image_selection="some", group_pattern="video_(", mosaic_grid_size=True)
    assert str(error_information.value).splitlines()[:8] == [
        "Invalid clustering settings:", "- min_cluster_size must be an integer of at least 2, got 1.",
        "- min_samples must be a positive integer or None, got 0.",
        "- cluster_selection_method must be one of ['eom', 'leaf'], got 'best'.",
        "- pca_dimensions must be a positive integer, got 0.",
        "- number_representatives_per_cluster must be a positive integer, got 0.",
        "- isolated_image_selection must be one of ['all', 'none'] or a positive integer, got 'some'.",
        "- group_pattern is not a valid regular expression: missing ), unterminated subpattern at position 6."]
    assert str(error_information.value).splitlines()[8] == "- mosaic_grid_size must be a positive integer, got True."

    # An unknown algorithm, missing parameters, and the values of DBSCAN and k-means
    with pytest.raises(expected_exception=ValueError, match="clustering_algorithm must be one of"):
        check_clustering_settings(clustering_algorithm="optics", clustering_parameters={}, **valid_settings)
    with pytest.raises(expected_exception=ValueError, match=r"The dbscan parameters miss \['min_samples'\]"):
        check_clustering_settings(clustering_algorithm="dbscan", clustering_parameters={"eps": 0.5}, **valid_settings)
    with pytest.raises(expected_exception=ValueError, match="eps must be a positive number, got 0"):
        check_clustering_settings(clustering_algorithm="dbscan", clustering_parameters={"eps": 0, "min_samples": 2},
                                  **valid_settings)
    with pytest.raises(expected_exception=ValueError, match="number_clusters must be a positive integer, got 2.5"):
        check_clustering_settings(clustering_algorithm="kmeans",
                                  clustering_parameters={"number_clusters": 2.5, "number_initializations": 1},
                                  **valid_settings)


def test_features_are_the_outputs_of_the_asked_layers_in_row_order(embedding_run: Dict[str, Path],
                                                                   cpu_embedder: Dict[str, Any],
                                                                   tmp_path: Path) -> None:
    """
    Verifies that the extraction, by batches read by worker processes, stores for every image of a split in the order
    of its manifest the whole output of every asked block before its pooling, flattened channel after channel, the
    output of every asked hidden layer after its ReLU and the most probable class with its probability, as computed
    on all the images at once through the layers of the model, and records the images and the fingerprint of the
    features.

    Args:
        embedding_run (Dict[str, Path]): The embedding run of the module.
        cpu_embedder (Dict[str, Any]): The embedding run loaded on the CPU.
        tmp_path (Path): The pytest temporary directory.
    """
    split_folder = embedding_run["dataset_folder"] / "test"
    extraction_folder = tmp_path / "extraction"
    outputs = extract_split_features(embedder=cpu_embedder, split_folder=split_folder,
                                     convolution_block_indices=[2, 5], fully_connected_layer_indices=[1, 2],
                                     extraction_folder=extraction_folder, number_workers=2)

    # The images of the manifest, in its order, preprocessed like the images of the run
    manifest = json.loads(s=(split_folder / "labels.json").read_text())
    file_names = [sample["file_name"] for sample in manifest["samples"]]
    images = torch.stack(tensors=[torch.from_numpy(preprocess_image(
        image=cv2.imread(filename=str(split_folder / "data" / file_name)), image_size=SMOKE_IMAGE_SIZE,
        keep_ratio=True)).permute(dims=(2, 0, 1)) for file_name in file_names]).to(dtype=torch.float32)

    # The expected outputs, computed layer after layer on all the images at once
    model = cpu_embedder["model"]
    with torch.no_grad():
        final_feature_map, block_outputs = model.backbone(input_tensor=images, return_pre_pooling_outputs=True)
        hidden_outputs = []
        current_features = model.global_pooling(input=final_feature_map)
        for module in model.classification_head.fully_connected_block:
            current_features = module(current_features)
            if isinstance(module, nn.ReLU):
                hidden_outputs.append(current_features.clone())
        class_probabilities = model(input_tensor=images)["class_probabilities"]
    expected_outputs = {f"block_{block_index}": block_outputs[str(block_index)].flatten(start_dim=1)
                        for block_index in [2, 5]}
    expected_outputs.update({"fully_connected_1": hidden_outputs[0], "fully_connected_2": hidden_outputs[1]})
    top_probabilities, top_classes = class_probabilities.max(dim=1)
    expected_outputs[PREDICTIONS_OUTPUT] = torch.stack(tensors=[top_classes.to(dtype=torch.float32),
                                                                top_probabilities], dim=1)

    # Channels times height times width for the blocks, units for the hidden layers, class and probability for the
    # predictions
    assert {output_key: output.shape for output_key, output in outputs.items()} == {
        "block_2": (NUMBER_TEST_IMAGES, 32 * 32 * 32), "block_5": (NUMBER_TEST_IMAGES, 64 * 4 * 4),
        "fully_connected_1": (NUMBER_TEST_IMAGES, 32), "fully_connected_2": (NUMBER_TEST_IMAGES, 16),
        PREDICTIONS_OUTPUT: (NUMBER_TEST_IMAGES, 2)}
    for output_key, expected_output in expected_outputs.items():
        assert outputs[output_key].dtype == np.float32
        np.testing.assert_allclose(actual=outputs[output_key], desired=expected_output.numpy(), rtol=1e-4, atol=1e-5)
    assert (outputs["fully_connected_1"] >= 0).all() and (outputs["fully_connected_1"] > 0).any()

    # The list of the images and the record of the features
    category_names = {category["id"]: category["name"] for category in manifest["categories"]}
    assert read_csv_rows(path=extraction_folder / IMAGES_FILE_NAME) == [
        {"row": str(row), "image": sample["file_name"], "category": category_names[sample["category_id"]]}
        for row, sample in enumerate(manifest["samples"])]
    record = json.loads(s=(extraction_folder / EXTRACTION_RECORD_FILE_NAME).read_text())
    assert set(FINGERPRINT_KEYS) <= set(record)
    assert record["checkpoint_sha256"] == cpu_embedder["checkpoint_sha256"]
    assert (record["image_size"], record["keep_ratio"], record["number_images"]) == (SMOKE_IMAGE_SIZE, True,
                                                                                    NUMBER_TEST_IMAGES)
    assert record["source_folder"] == str(split_folder.resolve())
    assert record["outputs"] == {output_key: {"dimension": output.shape[1], "finished_images": NUMBER_TEST_IMAGES}
                                 for output_key, output in outputs.items()}


def test_relaunch_loads_the_stored_features_and_computes_only_the_missing_layers(embedding_run: Dict[str, Path],
                                                                                 cpu_embedder: Dict[str, Any],
                                                                                 tmp_path: Path) -> None:
    """
    Verifies that a relaunch with the same layers loads the stored features without running the model, and that a
    relaunch asking one more block runs the model for that block only, leaving the stored files untouched.

    Args:
        embedding_run (Dict[str, Path]): The embedding run of the module.
        cpu_embedder (Dict[str, Any]): The embedding run loaded on the CPU.
        tmp_path (Path): The pytest temporary directory.
    """
    split_folder = embedding_run["dataset_folder"] / "test"
    extraction_folder = tmp_path / "extraction"
    first_outputs = extract_split_features(embedder=cpu_embedder, split_folder=split_folder,
                                           convolution_block_indices=[5], fully_connected_layer_indices=[1],
                                           extraction_folder=extraction_folder)
    stored_files = {path.name: path.read_bytes() for path in extraction_folder.glob("*.npy")}
    assert sorted(stored_files) == ["features_block_5.npy", "features_fully_connected_1.npy",
                                    "predictions.npy"]

    # Same layers: no inference
    with patch(target="scripts.embedding.cluster_image_embeddings.compute_batch_outputs",
               side_effect=AssertionError("The model ran on stored features.")) as compute_mock:
        reloaded_outputs = extract_split_features(embedder=cpu_embedder, split_folder=split_folder,
                                                  convolution_block_indices=[5], fully_connected_layer_indices=[1],
                                                  extraction_folder=extraction_folder)
    compute_mock.assert_not_called()
    for output_key, output in first_outputs.items():
        assert np.array_equal(reloaded_outputs[output_key], output)

    # One more block: only that block computed, by batches over every image
    spy = BatchComputationSpy()
    with patch(target="scripts.embedding.cluster_image_embeddings.compute_batch_outputs", new=spy):
        extended_outputs = extract_split_features(embedder=cpu_embedder, split_folder=split_folder,
                                                  convolution_block_indices=[3, 5], fully_connected_layer_indices=[1],
                                                  extraction_folder=extraction_folder)
    assert spy.calls == [{"layers": ["block_3"], "output_mode": None, "images": images}
                         for images in [4, 4, 2]]
    assert {path.name: path.read_bytes() for path in extraction_folder.glob("*.npy")
            if path.name in stored_files} == stored_files
    assert extended_outputs["block_3"].shape == (NUMBER_TEST_IMAGES, 32 * 16 * 16)
    record = json.loads(s=(extraction_folder / EXTRACTION_RECORD_FILE_NAME).read_text())
    assert record["outputs"]["block_3"] == {"dimension": 32 * 16 * 16, "finished_images": NUMBER_TEST_IMAGES}


def test_interrupted_extraction_continues_from_its_last_finished_batch(embedding_run: Dict[str, Path],
                                                                       cpu_embedder: Dict[str, Any],
                                                                       tmp_path: Path) -> None:
    """
    Verifies that an extraction interrupted at its third batch records the two finished batches, and that the
    relaunch runs the model on the remaining images only, giving the features of an uninterrupted extraction.

    Args:
        embedding_run (Dict[str, Path]): The embedding run of the module.
        cpu_embedder (Dict[str, Any]): The embedding run loaded on the CPU.
        tmp_path (Path): The pytest temporary directory.
    """
    split_folder = embedding_run["dataset_folder"] / "test"
    extraction_folder = tmp_path / "interrupted_extraction"

    # Interrupted at the third batch, after the rows 0 to 7
    with patch(target="scripts.embedding.cluster_image_embeddings.compute_batch_outputs",
               new=BatchComputationSpy(failing_call=3)):
        with pytest.raises(expected_exception=RuntimeError, match="Simulated interruption"):
            extract_split_features(embedder=cpu_embedder, split_folder=split_folder, convolution_block_indices=[4],
                                   fully_connected_layer_indices=[2], extraction_folder=extraction_folder)
    record = json.loads(s=(extraction_folder / EXTRACTION_RECORD_FILE_NAME).read_text())
    assert {output_key: output_record["finished_images"]
            for output_key, output_record in record["outputs"].items()} == {
        "block_4": 8, "fully_connected_2": 8, PREDICTIONS_OUTPUT: 8}

    # The relaunch runs the last two images only
    spy = BatchComputationSpy()
    with patch(target="scripts.embedding.cluster_image_embeddings.compute_batch_outputs", new=spy):
        resumed_outputs = extract_split_features(embedder=cpu_embedder, split_folder=split_folder,
                                                 convolution_block_indices=[4], fully_connected_layer_indices=[2],
                                                 extraction_folder=extraction_folder)
    assert spy.calls == [{"layers": ["block_4", "fully_connected_2"],
                          "output_mode": cpu_embedder["output_mode"], "images": 2}]

    # The same features as an uninterrupted extraction
    complete_outputs = extract_split_features(embedder=cpu_embedder, split_folder=split_folder,
                                              convolution_block_indices=[4], fully_connected_layer_indices=[2],
                                              extraction_folder=tmp_path / "complete_extraction")
    for output_key, output in complete_outputs.items():
        np.testing.assert_allclose(actual=resumed_outputs[output_key], desired=output, rtol=1e-5, atol=1e-6)


def test_features_of_another_checkpoint_or_of_changed_images_are_refused(embedding_run: Dict[str, Path],
                                                                         cpu_embedder: Dict[str, Any],
                                                                         tmp_path: Path) -> None:
    """
    Verifies that stored features are never mixed with those of other images or of another checkpoint: a relaunch
    after a change of the labels of the split, or with another checkpoint, stops with an error naming the changed
    entries, and leaves the extraction folder unchanged.

    Args:
        embedding_run (Dict[str, Path]): The embedding run of the module.
        cpu_embedder (Dict[str, Any]): The embedding run loaded on the CPU.
        tmp_path (Path): The pytest temporary directory.
    """
    split_folder = tmp_path / "test_split"
    shutil.copytree(src=embedding_run["dataset_folder"] / "test", dst=split_folder)
    extraction_folder = tmp_path / "extraction"
    extract_split_features(embedder=cpu_embedder, split_folder=split_folder, convolution_block_indices=[6],
                           fully_connected_layer_indices=[], extraction_folder=extraction_folder)
    stored_files = {path.name: path.read_bytes() for path in extraction_folder.iterdir()}

    # Another checkpoint
    with pytest.raises(expected_exception=ValueError, match="holds the features of another checkpoint") as \
            error_information:
        extract_split_features(embedder={**cpu_embedder, "checkpoint_sha256": 64 * "0"}, split_folder=split_folder,
                               convolution_block_indices=[6], fully_connected_layer_indices=[],
                               extraction_folder=extraction_folder)
    assert "- checkpoint_sha256: stored" in str(error_information.value)

    # One sample less in the labels of the split
    manifest = json.loads(s=(split_folder / "labels.json").read_text())
    manifest["samples"] = manifest["samples"][:-1]
    (split_folder / "labels.json").write_text(data=json.dumps(manifest))
    with pytest.raises(expected_exception=ValueError, match="holds the features of another checkpoint") as \
            error_information:
        extract_split_features(embedder=cpu_embedder, split_folder=split_folder, convolution_block_indices=[6],
                               fully_connected_layer_indices=[], extraction_folder=extraction_folder)
    assert [line.split(":")[0] for line in str(error_information.value).splitlines()[1:]] == [
        "- images_sha256", "- number_images"]
    assert {path.name: path.read_bytes() for path in extraction_folder.iterdir()} == stored_files


def test_data_sources_are_resolved_once_each(embedding_run: Dict[str, Path], cpu_embedder: Dict[str, Any],
                                             tmp_path: Path) -> None:
    """
    Verifies that the data sources give every folder to embed once, named after its last three path components: the
    train, validation and test splits of the run, the splits of another configuration file, a split folder and a
    folder of plain images, a folder given twice gaining its new role, and that an empty list, two folders sharing a
    name, a missing folder and a split without its data folder are refused.

    Args:
        embedding_run (Dict[str, Path]): The embedding run of the module.
        cpu_embedder (Dict[str, Any]): The embedding run loaded on the CPU.
        tmp_path (Path): The pytest temporary directory.
    """
    dataset_folder = embedding_run["dataset_folder"]
    experiment_configuration = cpu_embedder["experiment_configuration"]

    # A configuration file whose train split is the validation split of the run, and a folder of plain images
    other_split_folder = tmp_path / "other_dataset" / "version_2" / "train"
    shutil.copytree(src=dataset_folder / "test", dst=other_split_folder)
    configuration_path = tmp_path / "other_configuration.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data={"ExperimentConfiguration": {
        "train_data_folder": str(dataset_folder / "validation"), "validation_data_folder": str(other_split_folder),
        "test_data_folder": str(other_split_folder)}}))
    image_folder = tmp_path / "camera" / "plain" / "images"
    image_folder.mkdir(parents=True)
    cv2.imwrite(filename=str(image_folder / "frame.png"), img=draw_video_frame(video_index=0, frame_index=0))

    sources = resolve_data_sources(data_sources=["run", str(dataset_folder / "test"), configuration_path, image_folder],
                                   embedder_experiment_configuration=experiment_configuration)
    dataset_prefix = "-".join(dataset_folder.resolve().parts[-2:])
    assert [(source["name"], source["roles"]) for source in sources] == [
        (f"{dataset_prefix}-train", ["train"]), (f"{dataset_prefix}-validation", ["validation", "train"]),
        (f"{dataset_prefix}-test", ["test", "folder"]), ("other_dataset-version_2-train", ["validation", "test"]),
        ("camera-plain-images", ["folder"])]
    assert sources[0]["data_folder"] == (dataset_folder / "train" / "data").resolve()
    assert sources[0]["labels_path"] == (dataset_folder / "train" / "labels.json").resolve()
    assert (sources[-1]["data_folder"], sources[-1]["labels_path"]) == (image_folder.resolve(), None)

    # No source, two folders with the same last three components, a missing folder, a split without data
    with pytest.raises(expected_exception=ValueError, match="No data source given"):
        resolve_data_sources(data_sources=[], embedder_experiment_configuration=experiment_configuration)
    same_name_folder = tmp_path / "copy" / "camera" / "plain" / "images"
    shutil.copytree(src=image_folder, dst=same_name_folder)
    with pytest.raises(expected_exception=ValueError, match=r"share the extraction folders \['camera-plain-images'\]"):
        resolve_data_sources(data_sources=[image_folder, same_name_folder],
                             embedder_experiment_configuration=experiment_configuration)
    with pytest.raises(expected_exception=FileNotFoundError, match="does not exist"):
        resolve_data_sources(data_sources=[tmp_path / "missing"],
                             embedder_experiment_configuration=experiment_configuration)
    shutil.rmtree(path=other_split_folder / "data")
    with pytest.raises(expected_exception=FileNotFoundError, match="has a labels.json but no data folder"):
        resolve_data_sources(data_sources=[other_split_folder],
                             embedder_experiment_configuration=experiment_configuration)


def test_images_of_every_manifest_format_and_of_plain_folders_are_listed(tmp_path: Path) -> None:
    """
    Verifies the images listed for every kind of source, each once in its row order: the samples of a classification
    manifest with their categories, the COCO images, the frames of overlapping sequences, the frames of both members
    of the pairs, and the images of a folder of plain images with its subfolders, whose fingerprint changes when an
    image changes.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    manifests = {
        "classification": {"categories": [{"id": 1, "name": "empty"}, {"id": 2, "name": "full"}],
                           "samples": [{"id": 1, "file_name": "b.png", "category_id": 2},
                                       {"id": 2, "file_name": "a.png", "category_id": 1}]},
        "coco": {"categories": [{"id": 1, "name": "shape"}], "images": [{"id": 4, "file_name": "c.png"}],
                 "annotations": [{"id": 1, "image_id": 4, "category_id": 1}]},
        "sequences": {"categories": [], "sequences": [{"frames": ["s1.png", "s2.png"]},
                                                      {"frames": ["s2.png", "s3.png"]}]},
        "pairs": {"categories": [], "pairs": [{"before": ["p1.png"], "after": ["p2.png", "p1.png"]}]}}
    expected_images = {"classification": (["b.png", "a.png"], ["full", "empty"]), "coco": (["c.png"], None),
                       "sequences": (["s1.png", "s2.png", "s3.png"], None), "pairs": (["p1.png", "p2.png"], None)}
    for manifest_name, manifest in manifests.items():
        split_folder = tmp_path / manifest_name
        (split_folder / "data").mkdir(parents=True)
        (split_folder / "labels.json").write_text(data=json.dumps(manifest))
        source_images = list_source_images(source=describe_source_folder(folder=split_folder, role="folder"))
        assert (source_images["file_names"], source_images["categories"]) == expected_images[manifest_name]
        assert source_images["manifest"] == manifest

    # A folder of plain images with a subfolder, its fingerprint following the content of its images
    image_folder = tmp_path / "plain_images"
    (image_folder / "camera_1").mkdir(parents=True)
    for image_path in [image_folder / "top.png", image_folder / "camera_1" / "frame_2.png"]:
        cv2.imwrite(filename=str(image_path), img=draw_video_frame(video_index=0, frame_index=0))
    source = describe_source_folder(folder=image_folder, role="folder")
    source_images = list_source_images(source=source)
    assert source_images["file_names"] == ["camera_1/frame_2.png", "top.png"]
    assert (source_images["categories"], source_images["manifest"]) == (None, None)
    cv2.imwrite(filename=str(image_folder / "top.png"), img=draw_video_frame(video_index=1, frame_index=3))
    assert list_source_images(source=source)["images_sha256"] != source_images["images_sha256"]


def test_clustering_algorithms_separate_distinct_groups() -> None:
    """
    Verifies on three distant groups of vectors and two isolated vectors that HDBSCAN and DBSCAN give one cluster per
    group and leave the isolated vectors in no cluster, that k-means gives one cluster per group, that HDBSCAN leaves
    every image isolated only when there are fewer images than a cluster needs, and that k-means refuses more clusters
    than images.
    """
    random_generator = np.random.default_rng(seed=0)
    group_centres = 10 * np.eye(N=3, M=8)
    grouped_vectors = np.concatenate([centre + random_generator.normal(scale=0.1, size=(20, 8))
                                      for centre in group_centres])
    vectors = np.concatenate([grouped_vectors, np.full(shape=(1, 8), fill_value=30.0),
                              np.full(shape=(1, 8), fill_value=-30.0)])
    for clustering_algorithm, clustering_parameters in [("hdbscan", {"min_cluster_size": 5, "min_samples": 3,
                                                                     "cluster_selection_method": "eom"}),
                                                        ("dbscan", {"eps": 1.0, "min_samples": 3})]:
        labels, probabilities = cluster_vectors(vectors=vectors, clustering_algorithm=clustering_algorithm,
                                                clustering_parameters=clustering_parameters, seed=0)
        group_labels = [set(labels[group_index * 20:(group_index + 1) * 20].tolist()) for group_index in range(3)]
        assert all(len(labels_of_group) == 1 for labels_of_group in group_labels)
        assert len(set.union(*group_labels) - {ISOLATED_CLUSTER}) == 3
        assert labels[60:].tolist() == [ISOLATED_CLUSTER, ISOLATED_CLUSTER]
        assert (probabilities is not None) == (clustering_algorithm == "hdbscan")

    # k-means puts every vector in one of the asked clusters
    labels, probabilities = cluster_vectors(vectors=grouped_vectors, clustering_algorithm="kmeans",
                                            clustering_parameters={"number_clusters": 3, "number_initializations": 2},
                                            seed=0)
    assert sorted(len(set(labels[group_index * 20:(group_index + 1) * 20].tolist())) for group_index in range(3)) == \
        [1, 1, 1]
    assert len(set(labels.tolist())) == 3 and probabilities is None

    # Too few images for a cluster, or for the clusters asked
    labels, probabilities = cluster_vectors(vectors=grouped_vectors[:4], clustering_algorithm="hdbscan",
                                            clustering_parameters={"min_cluster_size": 5, "min_samples": None,
                                                                   "cluster_selection_method": "eom"}, seed=0)
    assert labels.tolist() == 4 * [ISOLATED_CLUSTER] and probabilities.tolist() == 4 * [0.0]
    with pytest.raises(expected_exception=ValueError, match="k-means asks 5 clusters for 4 images"):
        cluster_vectors(vectors=grouped_vectors[:4], clustering_algorithm="kmeans",
                        clustering_parameters={"number_clusters": 5, "number_initializations": 1}, seed=0)


def test_vectors_are_normalised_and_reduced() -> None:
    """
    Verifies that the vectors are divided by their length in float32 without changing the stored features, a zero
    vector staying zero and being counted, that the PCA keeps the asked dimensions with the share of the variance
    they hold and its time, still in float32, and that the vectors are clustered as they are when they have no more
    dimensions or images than asked.
    """
    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    vectors.setflags(write=False)
    prepared_vectors, preparation = prepare_vectors(vectors=vectors, normalize_vectors=True, pca_dimensions=3, seed=0)
    np.testing.assert_allclose(actual=prepared_vectors, desired=[[0.6, 0.8, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert prepared_vectors.dtype == np.float32 and vectors.tolist()[0] == [3.0, 4.0, 0.0]
    assert preparation == {"zero_vectors": 1, "dimensions": 3, "kept_variance": None, "pca_seconds": None}
    raw_vectors, preparation = prepare_vectors(vectors=vectors, normalize_vectors=False, pca_dimensions=3, seed=0)
    assert np.array_equal(raw_vectors, vectors) and preparation["zero_vectors"] == 0

    # Three main directions of twenty vectors of ten dimensions, and no PCA for ten dimensions or twenty images
    random_vectors = np.random.default_rng(seed=0).normal(size=(20, 10))
    reduced_vectors, preparation = prepare_vectors(vectors=random_vectors, normalize_vectors=True, pca_dimensions=3,
                                                   seed=0)
    assert reduced_vectors.shape == (20, 3) and reduced_vectors.dtype == np.float32
    assert preparation["dimensions"] == 3 and 0.3 < preparation["kept_variance"] < 1.0
    assert preparation["pca_seconds"] >= 0.0
    for pca_dimensions in [10, 20]:
        unreduced_vectors, preparation = prepare_vectors(vectors=random_vectors, normalize_vectors=False,
                                                         pca_dimensions=pca_dimensions, seed=0)
        assert np.array_equal(unreduced_vectors, random_vectors.astype(dtype=np.float32))
        assert preparation["dimensions"] == 10 and preparation["kept_variance"] is preparation["pca_seconds"] is None


def test_representatives_and_baselines_are_kept() -> None:
    """
    Verifies the images kept by the clustering and by the baselines: the images of every cluster ranked by their
    distance to its centre and the isolated images by their names, the representatives closest to the centres with
    all, none or some isolated images, the images kept at random, the budget spread evenly over the groups, the images
    evenly spaced in the name order of every group, and the groups given by a pattern with or without capturing group
    or with alternatives, including the documented ficha pattern.
    """
    # Two clusters and three isolated images, whose names are ordered with their digit runs compared as numbers
    labels = np.array([0, 0, 0, 1, 1, ISOLATED_CLUSTER, ISOLATED_CLUSTER, ISOLATED_CLUSTER])
    distances = np.array([0.3, 0.1, 0.2, 0.5, 0.4, np.nan, np.nan, np.nan])
    file_names = ["a_1.png", "a_2.png", "a_3.png", "b_1.png", "b_2.png", "isolated_10.png", "isolated_9.png",
                  "isolated_1.png"]
    ranks = rank_cluster_members(labels=labels, distances=distances, file_names=file_names)
    assert ranks.tolist() == [3, 1, 2, 2, 1, 3, 2, 1]
    assert sorted(file_names[5:], key=natural_sort_key) == ["isolated_1.png", "isolated_9.png", "isolated_10.png"]

    random_generator = np.random.default_rng(seed=0)
    assert select_kept_images_by_clustering(labels=labels, ranks=ranks, number_representatives_per_cluster=2,
                                            isolated_image_selection="all", random_generator=random_generator) == {
        1: "representative of cluster 0, rank 1", 2: "representative of cluster 0, rank 2",
        3: "representative of cluster 1, rank 2", 4: "representative of cluster 1, rank 1", 5: ISOLATED_IMAGE_REASON,
        6: ISOLATED_IMAGE_REASON, 7: ISOLATED_IMAGE_REASON}
    assert sorted(select_kept_images_by_clustering(labels=labels, ranks=ranks, number_representatives_per_cluster=1,
                                                   isolated_image_selection="none",
                                                   random_generator=random_generator)) == [1, 4]
    drawn_kept_images = select_kept_images_by_clustering(labels=labels, ranks=ranks,
                                                         number_representatives_per_cluster=1,
                                                         isolated_image_selection=2, random_generator=random_generator)
    assert len(drawn_kept_images) == 4 and len({5, 6, 7} & set(drawn_kept_images)) == 2
    random_positions = select_random_baseline(number_images=8, budget=5, random_generator=random_generator)
    assert len(set(random_positions.tolist())) == 5 and random_positions.tolist() == sorted(random_positions.tolist())
    assert 0 <= random_positions.min() and random_positions.max() < 8

    # The budget spread over the groups, a small group giving all its images
    assert allocate_budget_over_groups(group_sizes={"b": 50, "a": 2, "c": 50}, budget=10) == {"a": 2, "b": 4, "c": 4}
    assert allocate_budget_over_groups(group_sizes={"a": 3, "b": 3}, budget=5) == {"a": 2, "b": 3}
    for _ in range(200):
        group_sizes = {f"group_{index}": int(size) for index, size in enumerate(random_generator.integers(
            low=1, high=30, size=random_generator.integers(low=1, high=6)))}
        budget = int(random_generator.integers(low=0, high=sum(group_sizes.values()) + 1))
        allocation = allocate_budget_over_groups(group_sizes=group_sizes, budget=budget)
        assert sum(allocation.values()) == budget
        assert all(0 <= allocation[group] <= size for group, size in group_sizes.items())
        unsaturated_allocations = [allocation[group] for group, size in group_sizes.items()
                                   if allocation[group] < size]
        assert not unsaturated_allocations or max(unsaturated_allocations) - min(unsaturated_allocations) <= 1
    with pytest.raises(expected_exception=ValueError, match="7 images asked from groups holding 6 images"):
        allocate_budget_over_groups(group_sizes={"a": 3, "b": 3}, budget=7)

    # Twenty frames of a video in shuffled manifest order and two frames of another, six images kept
    frame_names = [f"video_a_frame_{index}.png" for index in random_generator.permutation(np.arange(1, 21))]
    frame_names += ["video_b_frame_2.png", "video_b_frame_1.png"]
    frame_groups = find_image_groups(file_names=frame_names, group_pattern=r"^(video_[a-z])_frame")
    kept_positions, kept_groups = select_evenly_spaced_per_group(file_names=frame_names, groups=frame_groups,
                                                                 budget=6)
    assert sorted(frame_names[position] for position in kept_positions) == sorted(
        [f"video_a_frame_{index}.png" for index in [3, 8, 13, 18]] + ["video_b_frame_1.png", "video_b_frame_2.png"])
    assert kept_positions.tolist() == sorted(kept_positions.tolist())
    assert kept_groups == [frame_groups[position] for position in kept_positions]

    # The groups of a pattern with and without capturing group, the ficha example, and no pattern
    assert find_image_groups(file_names=["video_1_frame_3.png", "video_12_frame_1.png", "other.png"],
                             group_pattern=r"^(video_\d+)_") == ["video_1", "video_12", NO_GROUP]
    assert find_image_groups(file_names=["camera_2/video_7_frame_1.png"], group_pattern=r"video_\d+") == ["video_7"]
    assert find_image_groups(file_names=["video_t0053_2026-09-25_05-51-30_0037018ms.jpg"],
                             group_pattern=r"^(video_t\d+_[\d-]+_[\d-]+)_") == ["video_t0053_2026-09-25_05-51-30"]
    assert find_image_groups(file_names=["camera_1/frame.png"], group_pattern=r"^([^/]+)/") == ["camera_1"]
    assert find_image_groups(file_names=["video_3_frame_1.png", "camera_2/frame_1.png", "other.png"],
                             group_pattern=r"^(video_\d+)_|^(camera_\d+)/") == ["video_3", "camera_2", NO_GROUP]
    assert find_image_groups(file_names=["frame.png"], group_pattern=None) is None


@pytest.mark.skipif(condition=not hasattr(os, "posix_fadvise"), reason="posix_fadvise is a Linux call")
def test_released_files_leave_the_page_cache(tmp_path: Path) -> None:
    """
    Verifies that a file read then released by the script leaves the page cache of Linux: all its pages are cached
    after its reading, and none after its release.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # A file of 64 pages written to the disk, then read
    file_path = tmp_path / "frame.bin"
    with open(file=file_path, mode="wb") as written_file:
        written_file.write(np.random.default_rng(seed=0).bytes(64 * mmap.PAGESIZE))
        written_file.flush()
        os.fsync(written_file.fileno())
    file_path.read_bytes()
    assert count_cached_pages(path=file_path) == 64

    # None of its pages left in memory once released
    release_file_from_page_cache(path=file_path)
    assert count_cached_pages(path=file_path) == 0


def test_mosaics_keep_the_ratio_of_the_images(tmp_path: Path) -> None:
    """
    Verifies that the mosaic of a cluster lays out at most grid size rows of grid size images under its caption, in
    cells with the ratio of the first image, a large frame being read at half its size and a small image of another
    ratio fitted in its cell between black bands, the empty cells left gray, and that an unreadable image stops it.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # A large wide frame and a small tall image, on a grid of 2 images per row
    cv2.imwrite(filename=str(tmp_path / "large.jpg"), img=np.full(shape=(1080, 1920, 3), fill_value=200,
                                                                   dtype=np.uint8))
    cv2.imwrite(filename=str(tmp_path / "small.png"), img=np.full(shape=(64, 32, 3), fill_value=255, dtype=np.uint8))
    mosaic = build_mosaic(image_paths=[tmp_path / "large.jpg", tmp_path / "small.png", tmp_path / "large.jpg"],
                          grid_size=2, caption="cluster")
    cell_width = round(MOSAIC_THUMBNAIL_HEIGHT * 1920 / 1080)
    assert mosaic.shape == (CAPTION_BAR_HEIGHT + 2 * (MOSAIC_THUMBNAIL_HEIGHT + 4) + 4, 2 * (cell_width + 4) + 4, 3)

    # The middle row of the cells: the gray frame filling its cell, the small image 2.5 times larger between black
    # bands, and the empty cell of the second row left gray
    first_row_middle = CAPTION_BAR_HEIGHT + 4 + MOSAIC_THUMBNAIL_HEIGHT // 2
    second_cell_left = 4 + cell_width + 4
    assert abs(int(mosaic[first_row_middle, 4 + cell_width // 2, 0]) - 200) <= 2
    assert mosaic[first_row_middle, second_cell_left + cell_width // 2].tolist() == [255, 255, 255]
    assert mosaic[first_row_middle, second_cell_left + cell_width // 2 - 41].tolist() == [0, 0, 0]
    assert mosaic[first_row_middle + MOSAIC_THUMBNAIL_HEIGHT + 4, second_cell_left + cell_width // 2].tolist() == \
        [128, 128, 128]

    # At most 2 rows of 2 images, and an unreadable image
    assert build_mosaic(image_paths=5 * [tmp_path / "large.jpg"], grid_size=2, caption="cluster").shape == mosaic.shape
    (tmp_path / "broken.jpg").write_text(data="not an image")
    with pytest.raises(expected_exception=ValueError, match="The image cannot be read"):
        build_mosaic(image_paths=[tmp_path / "broken.jpg"], grid_size=2, caption="cluster")


def test_clustering_writes_clusters_links_mosaics_and_kept_images(embedding_run: Dict[str, Path], tmp_path: Path,
                                                                  capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole script on a classification split of near duplicate video frames, the case it is made for:
    every category clustered on its own, every cluster holding the frames of a single video, the tables of the images
    and of the clusters with the error rates of the shape categories, the links and mosaics of every cluster, the
    images kept by the clustering, at random and evenly spaced over the videos, as many per category and ready to
    train, and the table of the kept images printed at the end of the run. A relaunch reuses the features and replaces
    the clustering folder.

    Args:
        embedding_run (Dict[str, Path]): The embedding run of the module.
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    category_names = list(SHAPE_CATEGORIES)[:2]
    split_folder = tmp_path / "videos" / "dataset" / "train"
    manifest = write_video_split(split_folder=split_folder, category_names=category_names, number_videos=3,
                                 number_frames=6)
    output_root = tmp_path / "embeddings"
    with patch(target="scripts.embedding.cluster_image_embeddings.release_file_from_page_cache",
               wraps=release_file_from_page_cache) as release_spy:
        clustering_folders = run_embedding_script(embedding_run=embedding_run, data_sources=[split_folder],
                                                  output_root=output_root, clustering_algorithm="hdbscan",
                                                  clustering_parameters=HDBSCAN_PARAMETERS,
                                                  group_pattern=VIDEO_GROUP_PATTERN)
    extraction_folder = (output_root / embedding_run["experiment_folder"].name / f"Iteration_{EMBEDDING_ITERATION}" /
                         "videos-dataset-train")
    clustering_name = "hdbscan_min_cluster_size_3_min_samples_2_cluster_selection_method_eom_normalized_pca_8"
    assert clustering_folders == [extraction_folder / "clusters" / layer_key / clustering_name
                                  for layer_key in ["block_4", "fully_connected_1"]]
    console_output = capsys.readouterr().out
    assert "videos-dataset-train (folder): 36 images, 2 categories" in console_output
    assert "6 groups, 0 images matching no group" in console_output

    # Every layer of every category reduced from its stored width to 8 dimensions, with the time of the PCA and of the
    # clustering
    for layer_key in ["block_4", "fully_connected_1"]:
        for category_name in category_names:
            assert re.search(pattern=rf"{layer_key} - {category_name}: 18 initial images, \d+ image clusters, \d+ "
                                     rf"isolated images \(\d+\.\d %\), \d+ kept images, 8 dimensions \(\d+\.\d % of "
                                     rf"the variance, PCA in \d+\.\d Seconds\), clustered in \d+\.\d Seconds",
                             string=console_output)

    # The table of the kept images of the source printed once every layer is clustered, every layer with its
    # categories of 18 images and their total of 36
    assert "Kept Images · videos-dataset-train (folder)" in console_output
    assert console_output.index(f"Clustering {clustering_name}") > console_output.rindex("Clustering written")
    for layer_key in ["block_4", "fully_connected_1"]:
        for category_name in category_names:
            assert re.search(pattern=rf"{layer_key}\s+{category_name}\s+18\s", string=console_output)
        assert re.search(pattern=rf"{layer_key}\s+{TOTAL_CATEGORY}\s+36\s", string=console_output)

    # Every frame released from the page cache after its extraction and after every mosaic showing it (at most 9 per
    # cluster), and the feature matrices once the source is clustered
    released_paths = [Path(call.kwargs["path"]).resolve() for call in release_spy.call_args_list]
    frame_paths = {(split_folder / "data" / sample["file_name"]).resolve() for sample in manifest["samples"]}
    feature_paths = {extraction_folder.resolve() / file_name
                     for file_name in ["features_block_4.npy", "features_fully_connected_1.npy", "predictions.npy"]}
    number_mosaic_images = sum(min(int(row["images"]), 9) for folder in clustering_folders
                               for row in read_csv_rows(path=folder / "cluster_summary.csv"))
    assert set(released_paths) == frame_paths | feature_paths
    assert sum(path in frame_paths for path in released_paths) == len(frame_paths) + number_mosaic_images
    assert set(released_paths[-len(feature_paths):]) == feature_paths

    clustering_folder = clustering_folders[0]
    image_rows = read_csv_rows(path=clustering_folder / "clusters.csv")
    category_names_by_identifier = {category["id"]: category["name"] for category in manifest["categories"]}
    assert [(row["image"], row["category"]) for row in image_rows] == [
        (sample["file_name"], category_names_by_identifier[sample["category_id"]]) for sample in manifest["samples"]]
    assert all(row["group"] == row["image"].rsplit("_frame", 1)[0] for row in image_rows)

    # Every cluster holds the frames of a single video, and every category is clustered on its own
    summary_rows = read_csv_rows(path=clustering_folder / "cluster_summary.csv")
    clustered_rows = [row for row in summary_rows if int(row["cluster"]) != ISOLATED_CLUSTER]
    assert clustered_rows and all(row["largest_group_share"] == "1.0000" and row["groups"] == "1"
                                  for row in clustered_rows)
    for category_name in category_names:
        category_summary_rows = [row for row in summary_rows if row["category"] == category_name]
        assert sum(int(row["images"]) for row in category_summary_rows) == 18
        assert all(row["largest_group"].startswith(f"video_{category_name}_") for row in category_summary_rows
                   if int(row["cluster"]) != ISOLATED_CLUSTER)
        assert all(0.0 <= float(row["error_rate"]) <= 1.0 for row in category_summary_rows)

    # Links named by rank to the images of every cluster, and a mosaic per cluster of at most 3 rows of 3 images
    # under its caption, the 64 x 48 frames in cells of their ratio separated by the 4 pixel gaps of the frame grid
    for row in image_rows:
        cluster_name = ISOLATED_IMAGES_NAME if int(row["cluster"]) == ISOLATED_CLUSTER else \
            f"cluster_{int(row['cluster']):04d}"
        link_path = clustering_folder / row["category"] / cluster_name / f"{int(row['rank']):05d}_{row['image']}"
        assert link_path.is_symlink()
        assert link_path.resolve() == (split_folder / "data" / row["image"]).resolve()
    cell_width = round(MOSAIC_THUMBNAIL_HEIGHT * 48 / 64)
    for row in summary_rows:
        cluster_name = ISOLATED_IMAGES_NAME if int(row["cluster"]) == ISOLATED_CLUSTER else \
            f"cluster_{int(row['cluster']):04d}"
        mosaic = cv2.imread(filename=str(clustering_folder / row["category"] / f"{cluster_name}.jpg"))
        number_rows = (min(int(row["images"]), 9) + 2) // 3
        assert mosaic.shape[0] == CAPTION_BAR_HEIGHT + number_rows * (MOSAIC_THUMBNAIL_HEIGHT + 4) + 4
        assert mosaic.shape[1] >= 3 * (cell_width + 4) + 4

    # The clustering keeps the representatives and the isolated images, the baselines as many images per category,
    # every folder of kept images being a split with the kept samples
    kept_reasons = {int(row["row"]): row["kept_reason"] for row in image_rows if row["kept_reason"]}
    assert all(int(row["rank"]) <= 2 for row in image_rows if row["kept_reason"].startswith("representative"))
    assert all(row["kept_reason"] == ISOLATED_IMAGE_REASON for row in image_rows
               if int(row["cluster"]) == ISOLATED_CLUSTER)
    # Every folder of kept images lists the same samples in its CSV and its labels.json, its data linked to the split
    kept_counts = {}
    for folder_name in [KEPT_BY_CLUSTERING_FOLDER_NAME, KEPT_AT_RANDOM_FOLDER_NAME, KEPT_EVENLY_SPACED_FOLDER_NAME]:
        kept_images_folder = clustering_folder / folder_name
        kept_rows = read_csv_rows(path=kept_images_folder / KEPT_IMAGES_FILE_NAME)
        kept_manifest = json.loads(s=(kept_images_folder / "labels.json").read_text())
        assert [sample["file_name"] for sample in kept_manifest["samples"]] == [row["image"] for row in kept_rows]
        assert kept_manifest["categories"] == manifest["categories"]
        assert (kept_images_folder / "data").resolve() == (split_folder / "data").resolve()
        kept_counts[folder_name] = sorted((category_name, sum(row["category"] == category_name for row in kept_rows))
                                          for category_name in category_names)
        # The clustering gives the reasons of clusters.csv, the evenly spaced baseline as many images of every video
        if folder_name == KEPT_BY_CLUSTERING_FOLDER_NAME:
            assert {int(row["row"]): row["reason"] for row in kept_rows} == kept_reasons
        if folder_name == KEPT_EVENLY_SPACED_FOLDER_NAME:
            for category_name in category_names:
                video_counts = [sum(row["reason"] == f"evenly spaced in video_{category_name}_{video_index}"
                                    for row in kept_rows) for video_index in range(3)]
                assert max(video_counts) - min(video_counts) <= 1
    assert kept_counts[KEPT_AT_RANDOM_FOLDER_NAME] == kept_counts[KEPT_BY_CLUSTERING_FOLDER_NAME] == \
        kept_counts[KEPT_EVENLY_SPACED_FOLDER_NAME]

    # A relaunch reuses the features and writes the clustering folder again
    (clustering_folder / "stale_file.txt").write_text(data="left by a previous launch")
    with patch(target="scripts.embedding.cluster_image_embeddings.compute_batch_outputs",
               side_effect=AssertionError("The model ran on stored features.")):
        run_embedding_script(embedding_run=embedding_run, data_sources=[split_folder], output_root=output_root,
                             clustering_algorithm="hdbscan", clustering_parameters=HDBSCAN_PARAMETERS,
                             group_pattern=VIDEO_GROUP_PATTERN)
    assert not (clustering_folder / "stale_file.txt").exists()
    assert read_csv_rows(path=clustering_folder / "clusters.csv") == image_rows


def test_kept_images_of_plain_image_folders_and_coco_splits(embedding_run: Dict[str, Path], tmp_path: Path) -> None:
    """
    Verifies the kept images of the sources without one category per image, clustered as a whole with k-means: links
    to the kept images keeping the subfolders of a folder of plain images, without labels.json, with the baseline
    evenly spaced over its subfolders, and the kept COCO images of a split with their annotations only.

    Args:
        embedding_run (Dict[str, Path]): The embedding run of the module.
        tmp_path (Path): The pytest temporary directory.
    """
    # Two subfolders of a folder of plain images, and a COCO split of the same frames
    image_folder = tmp_path / "plain" / "camera" / "images"
    coco_folder = tmp_path / "coco" / "dataset" / "train"
    (coco_folder / "data").mkdir(parents=True)
    coco_images, coco_annotations = [], []
    for video_index in range(2):
        (image_folder / f"camera_{video_index}").mkdir(parents=True)
        for frame_index in range(4):
            frame = draw_video_frame(video_index=video_index, frame_index=frame_index)
            cv2.imwrite(filename=str(image_folder / f"camera_{video_index}" / f"frame_{frame_index}.png"), img=frame)
            file_name = f"video_{video_index}_frame_{frame_index}.png"
            cv2.imwrite(filename=str(coco_folder / "data" / file_name), img=frame)
            coco_images.append({"id": len(coco_images) + 10, "file_name": file_name, "width": 48, "height": 64})
            coco_annotations += [{"id": len(coco_annotations) + 1, "image_id": coco_images[-1]["id"],
                                  "category_id": 1, "bbox": [0, 0, 4, 4]} for _ in range(2)]
    coco_manifest = {"categories": [{"id": 1, "name": "shape"}], "images": coco_images,
                     "annotations": coco_annotations}
    (coco_folder / "labels.json").write_text(data=json.dumps(coco_manifest))

    clustering_folders = run_embedding_script(
        embedding_run=embedding_run, data_sources=[image_folder, coco_folder], output_root=tmp_path / "embeddings",
        clustering_algorithm="kmeans", clustering_parameters={"number_clusters": 2, "number_initializations": 2},
        group_pattern=r"^([^/]+)/", number_representatives_per_cluster=1)
    assert [folder.parts[-4] for folder in clustering_folders] == 2 * ["plain-camera-images"] + \
        2 * ["coco-dataset-train"]

    # The folder of plain images: one category, two clusters, links to the kept images in their subfolders
    plain_clustering_folder = clustering_folders[0]
    image_rows = read_csv_rows(path=plain_clustering_folder / "clusters.csv")
    assert {row["category"] for row in image_rows} == {ALL_IMAGES_CATEGORY}
    assert {row["predicted_class"] for row in image_rows} <= set(SHAPE_CATEGORIES)
    assert len({row["cluster"] for row in image_rows}) == 2
    kept_rows = read_csv_rows(path=plain_clustering_folder / KEPT_BY_CLUSTERING_FOLDER_NAME / KEPT_IMAGES_FILE_NAME)
    assert len(kept_rows) == 2
    for row in kept_rows:
        link_path = plain_clustering_folder / KEPT_BY_CLUSTERING_FOLDER_NAME / row["image"]
        assert row["image"].startswith("camera_") and link_path.is_symlink()
        assert link_path.resolve() == (image_folder / row["image"]).resolve()
    assert not (plain_clustering_folder / KEPT_BY_CLUSTERING_FOLDER_NAME / "labels.json").exists()
    group_rows = read_csv_rows(path=plain_clustering_folder / KEPT_EVENLY_SPACED_FOLDER_NAME / KEPT_IMAGES_FILE_NAME)
    assert sorted(row["reason"] for row in group_rows) == ["evenly spaced in camera_0", "evenly spaced in camera_1"]

    # The COCO split: the kept images with their two annotations each, the categories kept, and its images, which the
    # pattern does not match, forming a single group for the baseline evenly spaced per group
    coco_clustering_folder = clustering_folders[2]
    kept_manifest = json.loads(s=(coco_clustering_folder / KEPT_BY_CLUSTERING_FOLDER_NAME / "labels.json").read_text())
    kept_rows = read_csv_rows(path=coco_clustering_folder / KEPT_BY_CLUSTERING_FOLDER_NAME / KEPT_IMAGES_FILE_NAME)
    assert [image["file_name"] for image in kept_manifest["images"]] == [row["image"] for row in kept_rows]
    kept_identifiers = {image["id"] for image in kept_manifest["images"]}
    assert len(kept_identifiers) == 2 and kept_manifest["categories"] == coco_manifest["categories"]
    assert sorted(annotation["image_id"] for annotation in kept_manifest["annotations"]) == sorted(
        2 * list(kept_identifiers))
    group_rows = read_csv_rows(path=coco_clustering_folder / KEPT_EVENLY_SPACED_FOLDER_NAME / KEPT_IMAGES_FILE_NAME)
    assert [row["reason"] for row in group_rows] == 2 * [f"evenly spaced in {NO_GROUP}"]


def test_kept_images_table_gives_every_layer_and_category_with_their_total() -> None:
    """
    Verifies the table of the kept images printed at the end of a run: its title naming the source, its roles and the
    clustering, one row per layer and category with the counts written with thousands separators and the shares of the
    initial images, a total row after the categories of a layer that has several, and none after a single category.
    """
    # A layer with two categories, whose counts are those of the ficha empty bins run, and a layer with a single one
    source = {"name": "dataset-classification-train", "roles": ["validation", "test"]}
    summaries = [{"layer": "block_6", "category": "empty_bin", "initial_images": 11772, "image_clusters": 96,
                  "isolated_images": 1573, "kept_representatives": 476, "kept_isolated_images": 1573,
                  "kept_images": 2049},
                 {"layer": "block_6", "category": "not_empty", "initial_images": 31805, "image_clusters": 310,
                  "isolated_images": 2600, "kept_representatives": 1547, "kept_isolated_images": 1000,
                  "kept_images": 2547},
                 {"layer": "fully_connected_1", "category": "all", "initial_images": 8, "image_clusters": 2,
                  "isolated_images": 0, "kept_representatives": 2, "kept_isolated_images": 0, "kept_images": 2}]
    table = build_kept_images_table(source=source, clustering_name="dbscan_eps_0.16_min_samples_5_normalized_pca_200",
                                    category_summaries=summaries)
    # The title names the source, its roles and the clustering, above the columns of the counts and shares
    assert table["title"] == ("Kept Images · dataset-classification-train (validation, test)\n"
                              "Clustering dbscan_eps_0.16_min_samples_5_normalized_pca_200")
    assert table["columns"] == ["Layer", "Category", "Initial\nImages", "Image\nClusters", "Isolated\nImages",
                                "Isolated\nShare", "Kept\nRepresentatives", "Kept Isolated\nImages", "Kept\nImages",
                                "Kept\nShare"]
    # One row per layer and category, a total row after the two categories of block_6 and none after the single one
    assert table["rows"] == [
        ["block_6", "empty_bin", "11,772", "96", "1,573", "13.4 %", "476", "1,573", "2,049", "17.4 %"],
        ["block_6", "not_empty", "31,805", "310", "2,600", "8.2 %", "1,547", "1,000", "2,547", "8.0 %"],
        ["block_6", TOTAL_CATEGORY, "43,577", "406", "4,173", "9.6 %", "2,023", "2,573", "4,596", "10.5 %"],
        ["fully_connected_1", "all", "8", "2", "0", "0.0 %", "2", "0", "2", "25.0 %"]]
    assert table["sort_column"] is None and not table["per_class"] and table["legend"]
