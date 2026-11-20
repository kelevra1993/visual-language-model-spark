"""
Tests of the input contract of the exported models (vision_training/deployment/input_contract.py), written by
vision_training/deployment/export_model.py: a tiny classifier exported with keep_ratio False and True, and a tiny
segmenter trained on tiles, carry their contract in the metadata of their ONNX file and in the preprocessing.json next
to their exported models, and the inputs built from that contract alone for photos that are not square equal the inputs
of the training pixel for pixel; a run using an interpolation, a padding value, a channel order or a padding placement
the contract cannot describe is refused with the name of the setting, before the export writes any file; the sequence
models get the NTCHW layout and the segmenters trained on tiles the interpolation of their scale; and a contract read
back with a missing, unknown or wrong entry is refused.
"""

import json
import re
import shutil
import cv2
import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple

import vision_training.deployment.input_contract as input_contract_module

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.preprocessing import compute_letterbox_region, preprocess_image, read_image
from vision_training.data.segmentation.segmentation_dataset import SegmentationDataset
from vision_training.deployment.export_model import export_experiment, hide_known_export_warnings
from vision_training.deployment.export_specifications import (SEQUENCE_TASKS, TASK_INPUT_NAMES,
                                                              build_input_specification)
from vision_training.deployment.input_contract import (INPUT_CONTRACT_KEYS, PREPROCESSING_FILE_NAME, TileRecorder,
                                                       build_input_contract, check_input_contract)
from vision_training.models.model_registry import MODEL_DTYPE, TASK_MODEL_CLASSES
from vision_training.models.segmentation.sliding_window import segment_image_by_tiles
from vision_training.trainers.checkpointing import get_checkpoint_file
from tests.deployment.test_deployment_utilities import TEST_IMAGE_SIZE, write_test_experiment

# Exported iteration of the test runs and largest batch size of their exports
EXPORTED_ITERATION = 1
TEST_MAXIMUM_BATCH_SIZE = 4

# Tiling of the test segmenter trained on tiles: photos halved, then cut into tiles sharing 8 pixels
TEST_TILING = {"activated": True, "scale": 0.5, "overlap": 8}

# Channels of a photo decoded to RGB in the order of every value of input_color_order
CONTRACT_CHANNEL_ORDERS = {"BGR": [2, 1, 0]}

# OpenCV interpolation of every value of input_interpolation that resizes the photo
CONTRACT_INTERPOLATIONS = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


def build_expected_contract(changed_values: Dict[str, str]) -> Dict[str, str]:
    """
    Gives the expected input contract of a test run of 32 pixel inputs: the contract of a stretched image model,
    with the values of the run that differ from it.

    Args:
        changed_values (Dict[str, str]): The values of the run that differ from those of a stretched image model.

    Returns:
        Dict[str, str]: The expected contract, in the key order of INPUT_CONTRACT_KEYS.
    """
    stretched_contract = {"input_contract_version": "2", "input_color_order": "BGR", "input_resize": "stretch",
                          "input_size": "32", "input_interpolation": "linear", "input_padding_value": "0",
                          "input_padding_placement": "centred", "input_tile_scale": "none",
                          "input_tile_overlap": "none", "input_layout": "NCHW", "input_value_range": "0-255"}
    return stretched_contract | changed_values


def write_trained_model(experiment_folder: Path, task: str, keep_ratio: bool,
                        tiling_configuration: Optional[Dict[str, Any]]) -> None:
    """
    Writes the run folder of a tiny trained model as export_experiment reads it: the synthetic configuration of the
    task with 32 pixel inputs, 3 classes, the resize and tiling asked for, and a checkpoint of seeded weights at the
    exported iteration.

    Args:
        experiment_folder (Path): The run folder to create.
        task (str): "classification" or "semantic_segmentation".
        keep_ratio (bool): The 'Data.image_settings.keep_ratio' of the run.
        tiling_configuration (Optional[Dict[str, Any]]): The 'Data.tiling' of a segmentation run, None to keep the one
            of the synthetic configuration.
    """
    write_test_experiment(experiment_folder=experiment_folder, task=task, keep_ratio=keep_ratio,
                          tiling_configuration=tiling_configuration)
    _, model_configuration = load_experiment_configuration(
        configuration_path=experiment_folder / "training_configuration.yaml")

    # Seeded weights saved in the checkpoint layout of the trainers
    torch.manual_seed(seed=0)
    model = TASK_MODEL_CLASSES[task](configuration=model_configuration, device=torch.device("cpu"), dtype=MODEL_DTYPE)
    checkpoint_path = get_checkpoint_file(weights_directory=experiment_folder / "Weights", iteration=EXPORTED_ITERATION)
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(obj={"model_state": model.state_dict()}, f=checkpoint_path)


def export_test_run(experiment_folder: Path, repository: Path, formats: List[str]) -> None:
    """
    Exports a test run with the settings of the tests: its exported iteration, the trained weights, a temperature of
    1, no TensorRT and batches of up to 4 samples.

    Args:
        experiment_folder (Path): The run folder, from write_trained_model.
        repository (Path): The model repository receiving the exports.
        formats (List[str]): The export formats, among 'torchscript' and 'onnx'.
    """
    export_experiment(experiment_folder=experiment_folder, iteration=EXPORTED_ITERATION, use_averaged_weights=False,
                      formats=formats, repository=repository, model_version=1,
                      maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, tensorrt_precision=None, use_saved_temperature=False,
                      saved_thresholds_path=None)


def write_test_photo(photo_path: Path, photo_height: int, photo_width: int) -> None:
    """
    Writes a photo of random colours as a lossless PNG file with Pillow, a library that the training does not use, so
    that every decoder gives the same pixels and the random colours expose any difference of the resize.

    Args:
        photo_path (Path): The PNG file to write.
        photo_height (int): The height of the photo in pixels.
        photo_width (int): The width of the photo in pixels.
    """
    rgb_pixels = np.random.default_rng(seed=photo_height * photo_width).integers(
        low=0, high=256, size=(photo_height, photo_width, 3), dtype=np.uint8)
    Image.fromarray(obj=rgb_pixels).save(fp=photo_path)


def build_input_from_contract(photo_path: Path, input_contract: Dict[str, str]) -> np.ndarray:
    """
    Builds the network inputs of a photo from the input contract alone, as a server reading the contract of a model
    would: the photo is decoded by Pillow rather than by the OpenCV reader of the training, and none of
    preprocess_image, compute_letterbox_region, SegmentationDataset.cut_tile and segment_image_by_tiles is called, so
    that the comparison with the inputs of the training checks the contract itself.

    Args:
        photo_path (Path): The photo, a PNG file.
        input_contract (Dict[str, str]): The contract read back from the exported model.

    Returns:
        np.ndarray: The [N, 3, S, S] float32 network inputs: one for the stretch and keep_ratio resizes, one per tile
            for the tiles, row by row.
    """
    # Values of the contract this reference implements
    expected_placements = {"stretch": "centred", "keep_ratio": "centred", "tiles": "bottom_right"}
    assert input_contract["input_padding_placement"] == expected_placements[input_contract["input_resize"]]
    assert (input_contract["input_layout"], input_contract["input_value_range"]) == ("NCHW", "0-255")

    # Photo decoded to RGB, then put in the channel order of the contract
    with Image.open(fp=photo_path) as photo_file:
        rgb_photo = np.asarray(a=photo_file.convert(mode="RGB"))
    photo = np.ascontiguousarray(a=rgb_photo[:, :, CONTRACT_CHANNEL_ORDERS[input_contract["input_color_order"]]])
    photo_height, photo_width = photo.shape[:2]
    input_size, padding_value = int(input_contract["input_size"]), int(input_contract["input_padding_value"])

    # Stretch: both sides to the input size, whatever the aspect ratio of the photo
    if input_contract["input_resize"] == "stretch":
        network_photos = cv2.resize(src=photo, dsize=(input_size, input_size),
                                    interpolation=CONTRACT_INTERPOLATIONS[input_contract["input_interpolation"]])[None]

    # Keep ratio: one scale for both sides, rounded half to even and kept to one pixel at least, then the photo centred
    # on the padding
    elif input_contract["input_resize"] == "keep_ratio":
        scale = input_size / max(photo_height, photo_width)
        resized_height, resized_width = max(1, round(photo_height * scale)), max(1, round(photo_width * scale))
        top, left = (input_size - resized_height) // 2, (input_size - resized_width) // 2
        network_photos = np.full(shape=(1, input_size, input_size, 3), fill_value=padding_value, dtype=np.uint8)
        network_photos[0, top:top + resized_height, left:left + resized_width] = cv2.resize(
            src=photo, dsize=(resized_width, resized_height),
            interpolation=CONTRACT_INTERPOLATIONS[input_contract["input_interpolation"]])

    # Tiles: the photo scaled with its sides rounded half to even, padded on its bottom and right sides to at least a
    # tile, then cut into the overlapping tiles, row by row
    else:
        tile_scale, overlap = float(input_contract["input_tile_scale"]), int(input_contract["input_tile_overlap"])
        if input_contract["input_interpolation"] != "none":
            photo = cv2.resize(src=photo, dsize=(max(1, round(photo_width * tile_scale)),
                                                 max(1, round(photo_height * tile_scale))),
                               interpolation=CONTRACT_INTERPOLATIONS[input_contract["input_interpolation"]])
        padded_photo = np.pad(array=photo, mode="constant", constant_values=padding_value,
                              pad_width=((0, max(0, input_size - photo.shape[0])),
                                         (0, max(0, input_size - photo.shape[1])), (0, 0)))
        tile_starts = [[0] if length <= input_size else
                       list(range(0, length - input_size, input_size - overlap)) + [length - input_size]
                       for length in padded_photo.shape[:2]]
        network_photos = np.stack(arrays=[padded_photo[tile_top:tile_top + input_size,
                                                       tile_left:tile_left + input_size]
                                          for tile_top in tile_starts[0] for tile_left in tile_starts[1]])

    # Channels first, the uint8 values cast to float32 without scaling
    return network_photos.transpose(0, 3, 1, 2).astype(np.float32)


def build_training_tiles(photo_path: Path, tiling_configuration: Dict[str, Any], tile_size: int) -> np.ndarray:
    """
    Builds the tiles of a photo the way the evaluation of a segmenter trained on tiles does: the test split of a
    SegmentationDataset ('whole' tile sampling) holding the photo and an empty mask gives the scaled photo, which
    segment_image_by_tiles cuts into the tiles given to the model, recorded by a TileRecorder.

    Args:
        photo_path (Path): The photo, a PNG file.
        tiling_configuration (Dict[str, Any]): The 'Data.tiling' of the run.
        tile_size (int): The side of the square tiles, 'Data.image_settings.size'.

    Returns:
        np.ndarray: The [N, 3, S, S] float32 tiles, in the order the model receives them.
    """
    # Split of one photo and its empty mask, with the manifest of the segmentation datasets
    split_folder = photo_path.parent / "segmentation_split"
    (split_folder / "data").mkdir(parents=True)
    (split_folder / "masks").mkdir()
    shutil.copyfile(src=photo_path, dst=split_folder / "data" / photo_path.name)
    cv2.imwrite(filename=str(split_folder / "masks" / "mask.png"),
                img=np.zeros(shape=read_image(image_path=photo_path).shape[:2], dtype=np.uint8))
    (split_folder / "labels.json").write_text(data=json.dumps(obj={
        "samples": [{"id": 0, "file_name": photo_path.name, "mask_file_name": "mask.png"}],
        "categories": [{"id": 1, "name": "shape"}]}))

    # Scaled photo of the test split, then the tiles of the sliding window
    dataset = SegmentationDataset(data_directory=str(split_folder / "data"),
                                  masks_directory=str(split_folder / "masks"),
                                  labels_file=str(split_folder / "labels.json"), image_size=tile_size, keep_ratio=True,
                                  label_mode="binary", label_dictionary={1: 1},
                                  tiling_configuration=tiling_configuration, tile_sampling="whole", shuffle=False)
    tile_recorder = TileRecorder()
    segment_image_by_tiles(image=dataset[0]["images"].float(), predict_tile_probabilities=tile_recorder,
                           tile_size=tile_size, overlap=tiling_configuration["overlap"], tiles_per_batch=2)
    return torch.cat(tensors=tile_recorder.tiles).numpy()


def build_contract_inputs(task: str, keep_ratio: bool,
                          tiling_configuration: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds the model configuration and the export specification read by build_input_contract for a run of a task,
    without a model: the image settings with 32 pixel inputs (and the tiling of the segmentation), and the inputs of
    the task from build_input_specification, sequences of up to 5 frames for the sequence tasks.

    Args:
        task (str): The task of the run.
        keep_ratio (bool): The 'Data.image_settings.keep_ratio' of the run.
        tiling_configuration (Optional[Dict[str, Any]]): The 'Data.tiling' of a segmentation run, None for the other
            tasks.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The model configuration and the export specification.
    """
    model_configuration = {"Data": {"image_settings": {"size": TEST_IMAGE_SIZE, "keep_ratio": keep_ratio}}}
    if tiling_configuration is not None:
        model_configuration["Data"]["tiling"] = tiling_configuration
    maximum_sequence_length = 5 if task in SEQUENCE_TASKS else None
    specification = {"inputs": [build_input_specification(input_name=input_name, image_size=TEST_IMAGE_SIZE,
                                                          maximum_sequence_length=maximum_sequence_length,
                                                          input_index=input_index)
                                for input_index, input_name in enumerate(TASK_INPUT_NAMES[task])]}
    return model_configuration, specification


def preprocess_image_with_cubic_interpolation(image: np.ndarray, image_size: int, keep_ratio: bool = True,
                                              interpolation: int = cv2.INTER_CUBIC,
                                              padding_value: int = 0) -> np.ndarray:
    """
    Stand-in for a preprocess_image resizing the images with the bicubic interpolation by default, which the input
    contract cannot describe.

    Args:
        image (np.ndarray): The image to resize.
        image_size (int): The side of the square network input.
        keep_ratio (bool): Whether the aspect ratio is kept. Defaults to True.
        interpolation (int): The OpenCV interpolation. Defaults to cv2.INTER_CUBIC.
        padding_value (int): The value of the padding pixels. Defaults to 0.

    Returns:
        np.ndarray: The resized image.
    """
    return preprocess_image(image=image, image_size=image_size, keep_ratio=keep_ratio, interpolation=interpolation,
                            padding_value=padding_value)


def preprocess_image_with_grey_padding(image: np.ndarray, image_size: int, keep_ratio: bool = True,
                                       interpolation: int = cv2.INTER_LINEAR,
                                       padding_value: int = 114) -> np.ndarray:
    """
    Stand-in for a preprocess_image padding the images in grey by default, which the input contract cannot describe.

    Args:
        image (np.ndarray): The image to resize.
        image_size (int): The side of the square network input.
        keep_ratio (bool): Whether the aspect ratio is kept. Defaults to True.
        interpolation (int): The OpenCV interpolation. Defaults to cv2.INTER_LINEAR.
        padding_value (int): The value of the padding pixels. Defaults to 114.

    Returns:
        np.ndarray: The resized image.
    """
    return preprocess_image(image=image, image_size=image_size, keep_ratio=keep_ratio, interpolation=interpolation,
                            padding_value=padding_value)


def read_image_as_rgb(image_path: str | Path) -> np.ndarray:
    """
    Stand-in for a read_image converting the photos to RGB, which the input contract cannot describe.

    Args:
        image_path (str | Path): The image file.

    Returns:
        np.ndarray: The uint8 [H, W, 3] RGB image.
    """
    return cv2.cvtColor(src=read_image(image_path=image_path), code=cv2.COLOR_BGR2RGB)


def compute_top_left_letterbox_region(image_height: int, image_width: int,
                                      image_size: int) -> Tuple[int, int, int, int]:
    """
    Stand-in for a compute_letterbox_region placing the resized photo at the top left of the padding, which the input
    contract cannot describe.

    Args:
        image_height (int): The height of the photo in pixels.
        image_width (int): The width of the photo in pixels.
        image_size (int): The side of the square network input.

    Returns:
        Tuple[int, int, int, int]: The top row (0), left column (0), height and width of the resized photo.
    """
    _, _, resized_height, resized_width = compute_letterbox_region(image_height=image_height, image_width=image_width,
                                                                   image_size=image_size)
    return 0, 0, resized_height, resized_width


class CubicScalingDataset:
    """
    Stand-in for a SegmentationDataset scaling the photos of its tiles with the bicubic interpolation, which the input
    contract cannot describe.
    """

    def cut_tile(self, image: np.ndarray, mask: np.ndarray, sample_identifier: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scales an image and its mask by the tiling scale of the dataset, in the 'whole' tile sampling.

        Args:
            image (np.ndarray): The [H, W, 3] uint8 BGR image.
            mask (np.ndarray): The [H, W] uint8 mask.
            sample_identifier (int): The identifier of the sample, unused.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The scaled image and mask.
        """
        scaled_size = (max(1, int(round(image.shape[1] * self.tiling_scale))),
                       max(1, int(round(image.shape[0] * self.tiling_scale))))
        return (cv2.resize(src=image, dsize=scaled_size, interpolation=cv2.INTER_CUBIC),
                cv2.resize(src=mask, dsize=scaled_size, interpolation=cv2.INTER_NEAREST_EXACT))


def segment_image_by_grey_padded_tiles(image: torch.Tensor,
                                       predict_tile_probabilities: Callable[[torch.Tensor], torch.Tensor],
                                       tile_size: int, overlap: int, tiles_per_batch: int) -> torch.Tensor:
    """
    Stand-in for a segment_image_by_tiles padding the photos smaller than a tile in grey, which the input contract
    cannot describe.

    Args:
        image (torch.Tensor): The [3, H, W] float BGR image.
        predict_tile_probabilities (Callable[[torch.Tensor], torch.Tensor]): The model of the tiles.
        tile_size (int): The side of the square tiles.
        overlap (int): The pixels shared by two neighbouring tiles.
        tiles_per_batch (int): The largest number of tiles given to the model at once.

    Returns:
        torch.Tensor: The [K, H, W] class probabilities of the image.
    """
    image_height, image_width = image.shape[-2:]
    padded_image = torch.nn.functional.pad(input=image, value=114.0, pad=(0, max(0, tile_size - image_width), 0,
                                                                          max(0, tile_size - image_height)))
    return segment_image_by_tiles(image=padded_image, predict_tile_probabilities=predict_tile_probabilities,
                                  tile_size=tile_size, overlap=overlap,
                                  tiles_per_batch=tiles_per_batch)[:, :image_height, :image_width]


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
@pytest.mark.parametrize(argnames=("keep_ratio", "expected_resize", "expected_padding_lines"),
                         argvalues=[(False, "stretch", 0), (True, "keep_ratio", 21)])
def test_exported_classifier_carries_its_input_contract(tmp_path: Path, keep_ratio: bool, expected_resize: str,
                                                        expected_padding_lines: int) -> None:
    """
    Verifies that a tiny classifier exported with keep_ratio False and True carries its input contract in the metadata
    of its ONNX file, as written and as ONNX Runtime reads it, and in the preprocessing.json next to its TorchScript and
    ONNX models, and that the input built from that contract alone for a landscape and a portrait photo equals the
    input of the training (read_image, then preprocess_image) pixel for pixel and runs through the exported model. The
    keep_ratio inputs of the 46 x 133 and 133 x 46 photos hold 21 padding rows or columns, 32 minus their resized side
    of 11 pixels, the stretched inputs none.

    Args:
        tmp_path (Path): The pytest temporary directory.
        keep_ratio (bool): The 'Data.image_settings.keep_ratio' of the exported run.
        expected_resize (str): The input_resize of its contract.
        expected_padding_lines (int): The number of padding rows or columns of the inputs of both photos.
    """
    hide_known_export_warnings()
    experiment_folder, repository = tmp_path / "CLS_TEST_MC3", tmp_path / "served_models"
    write_trained_model(experiment_folder=experiment_folder, task="classification", keep_ratio=keep_ratio,
                        tiling_configuration=None)
    export_test_run(experiment_folder=experiment_folder, repository=repository, formats=["torchscript", "onnx"])
    expected_contract = build_expected_contract(changed_values={"input_resize": expected_resize})

    # Metadata of the ONNX file in the key order of the contract, and as ONNX Runtime gives it to a server
    onnx_path = repository / "image_classifier_onnx" / "1" / "model.onnx"
    metadata_entries = [(entry.key, entry.value)
                        for entry in onnx.load(f=str(onnx_path), load_external_data=False).metadata_props]
    assert metadata_entries == list(expected_contract.items())
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_modelmeta().custom_metadata_map == expected_contract

    # The same entries in preprocessing.json, next to the model of both formats
    for export_format in ["torchscript", "onnx"]:
        preprocessing_path = repository / f"image_classifier_{export_format}" / "1" / PREPROCESSING_FILE_NAME
        assert list(json.loads(s=preprocessing_path.read_text()).items()) == list(expected_contract.items())

    # Input of a landscape and a portrait photo built from the contract read back, against the training input
    for photo_height, photo_width in [(46, 133), (133, 46)]:
        photo_path = tmp_path / f"photo_{photo_height}x{photo_width}.png"
        write_test_photo(photo_path=photo_path, photo_height=photo_height, photo_width=photo_width)
        contract_input = build_input_from_contract(photo_path=photo_path, input_contract=dict(metadata_entries))
        training_input = preprocess_image(image=read_image(image_path=photo_path), image_size=TEST_IMAGE_SIZE,
                                          keep_ratio=keep_ratio).transpose(2, 0, 1)[None].astype(np.float32)
        assert (contract_input.dtype, contract_input.shape) == (training_input.dtype, (1, 3, 32, 32))
        assert np.array_equal(a1=contract_input, a2=training_input)

        # Padding rows of the landscape photo or columns of the portrait one, every channel at 0
        black_lines = (contract_input[0] == 0).all(axis=(0, 2)).sum() + (contract_input[0] == 0).all(axis=(0, 1)).sum()
        assert int(black_lines) == expected_padding_lines

        # The exported model takes the input built from the contract
        assert session.run(None, {"images": contract_input})[0].shape == (1, 3)


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
def test_exported_tiled_segmenter_carries_its_input_contract(tmp_path: Path) -> None:
    """
    Verifies that a tiny segmenter trained on tiles of its photos halved, sharing 8 pixels, carries the tiles contract
    (area shrink, bottom and right padding, its scale and overlap) in its ONNX metadata and preprocessing.json, and that
    the tiles built from that contract alone for a 46 x 133 photo equal the tiles its evaluation gives the model
    (SegmentationDataset, then segment_image_by_tiles) pixel for pixel and run through the exported model. The photo is
    scaled to 23 x 66 pixels, its width 66.5 rounded half to even, padded to 32 x 66 and cut into the tiles starting at
    the columns 0, 24 and 34.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    experiment_folder, repository = tmp_path / "SEG_TEST_MC3", tmp_path / "served_models"
    write_trained_model(experiment_folder=experiment_folder, task="semantic_segmentation", keep_ratio=True,
                        tiling_configuration=TEST_TILING)
    export_test_run(experiment_folder=experiment_folder, repository=repository, formats=["onnx"])
    expected_contract = build_expected_contract(changed_values={
        "input_resize": "tiles", "input_interpolation": "area", "input_padding_placement": "bottom_right",
        "input_tile_scale": "0.5", "input_tile_overlap": "8"})

    # Metadata of the ONNX file and preprocessing.json next to it
    model_folder = repository / "semantic_segmenter_onnx" / "1"
    session = onnxruntime.InferenceSession(str(model_folder / "model.onnx"), providers=["CPUExecutionProvider"])
    input_contract = session.get_modelmeta().custom_metadata_map
    assert input_contract == expected_contract
    assert list(json.loads(s=(model_folder / PREPROCESSING_FILE_NAME).read_text()).items()) == list(
        expected_contract.items())

    # Tiles of the photo built from the contract read back, against the tiles of the evaluation
    photo_path = tmp_path / "photo_46x133.png"
    write_test_photo(photo_path=photo_path, photo_height=46, photo_width=133)
    contract_tiles = build_input_from_contract(photo_path=photo_path, input_contract=input_contract)
    training_tiles = build_training_tiles(photo_path=photo_path, tiling_configuration=TEST_TILING,
                                          tile_size=TEST_IMAGE_SIZE)
    assert (contract_tiles.dtype, contract_tiles.shape) == (training_tiles.dtype, (3, 3, 32, 32))
    assert np.array_equal(a1=contract_tiles, a2=training_tiles)

    # The 9 bottom rows of every tile are padding, and the exported model takes the tiles
    assert (contract_tiles[:, :, 23:] == 0).all() and not (contract_tiles[:, :, 22] == 0).all()
    assert session.run(None, {"images": contract_tiles})[0].shape == (3, 32, 32)


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
def test_export_of_an_input_the_contract_cannot_describe_writes_nothing(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the export of a run whose photos are read as RGB stops with the name of the setting before writing
    any file in the model repository.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing read_image in the input contract module.
    """
    experiment_folder, repository = tmp_path / "CLS_TEST_MC3", tmp_path / "served_models"
    write_trained_model(experiment_folder=experiment_folder, task="classification", keep_ratio=True,
                        tiling_configuration=None)
    monkeypatch.setattr(target=input_contract_module, name="read_image", value=read_image_as_rgb)
    with pytest.raises(expected_exception=ValueError,
                       match=re.escape(pattern="input_color_order 'RGB', from the channel order of read_image")):
        export_test_run(experiment_folder=experiment_folder, repository=repository, formats=["torchscript", "onnx"])
    assert not repository.exists()


@pytest.mark.parametrize(argnames=("task", "keep_ratio", "tiling_configuration", "changed_values"), argvalues=[
    ("semantic_segmentation", False, {"activated": False, "scale": 0.5, "overlap": 8}, {}),
    ("sequence_comparison", True, None, {"input_resize": "keep_ratio", "input_layout": "NTCHW"}),
    ("semantic_segmentation", False, {"activated": True, "scale": 0.25, "overlap": 8},
     {"input_resize": "tiles", "input_interpolation": "area", "input_padding_placement": "bottom_right",
      "input_tile_scale": "0.25", "input_tile_overlap": "8"}),
    ("semantic_segmentation", False, {"activated": True, "scale": 1.0, "overlap": 0},
     {"input_resize": "tiles", "input_interpolation": "none", "input_padding_placement": "bottom_right",
      "input_tile_scale": "1.0", "input_tile_overlap": "0"}),
    ("semantic_segmentation", False, {"activated": True, "scale": 2, "overlap": 16},
     {"input_resize": "tiles", "input_interpolation": "linear", "input_padding_placement": "bottom_right",
      "input_tile_scale": "2.0", "input_tile_overlap": "16"})])
def test_contract_follows_the_image_settings_the_tiling_and_the_inputs_of_the_task(
        task: str, keep_ratio: bool, tiling_configuration: Optional[Dict[str, Any]],
        changed_values: Dict[str, str]) -> None:
    """
    Verifies the contract of a stretched segmenter trained on whole images, of a comparison model keeping the aspect
    ratio, whose two inputs are sequences of frames (NTCHW), and of segmenters trained on tiles of their photos
    shrunk (area), kept (no resize) and enlarged (linear), every key written in the order of INPUT_CONTRACT_KEYS.

    Args:
        task (str): The task of the run.
        keep_ratio (bool): The 'Data.image_settings.keep_ratio' of the run.
        tiling_configuration (Optional[Dict[str, Any]]): The 'Data.tiling' of a segmenter, None for the other task.
        changed_values (Dict[str, str]): The expected values that differ from those of a stretched image model.
    """
    model_configuration, specification = build_contract_inputs(task=task, keep_ratio=keep_ratio,
                                                               tiling_configuration=tiling_configuration)
    input_contract = build_input_contract(task=task, model_configuration=model_configuration,
                                          specification=specification)
    assert list(input_contract) == list(INPUT_CONTRACT_KEYS)
    assert input_contract == build_expected_contract(changed_values=changed_values)


@pytest.mark.parametrize(argnames=("replaced_function", "replacement", "task", "expected_message"), argvalues=[
    ("preprocess_image", preprocess_image_with_cubic_interpolation, "classification",
     "input_interpolation 'OpenCV interpolation 2', from the 'interpolation' default of preprocess_image, or the "
     "scaling of SegmentationDataset.cut_tile for the tiles, is not allowed (allowed: 'linear' or 'area' or 'none')"),
    ("preprocess_image", preprocess_image_with_grey_padding, "classification",
     "input_padding_value '114', from the 'padding_value' default of preprocess_image"),
    ("read_image", read_image_as_rgb, "classification",
     "input_color_order 'RGB', from the channel order of read_image, is not allowed"),
    ("compute_letterbox_region", compute_top_left_letterbox_region, "classification",
     "input_padding_placement 'region (top, left, height, width) (0, 0, 18, 32) of a 1080 x 1920 photo instead of "
     "(7, 0, 18, 32)', from compute_letterbox_region"),
    ("SegmentationDataset", CubicScalingDataset, "semantic_segmentation",
     "input_interpolation 'a scaling matched by no interpolation on the probe photos', from"),
    ("segment_image_by_tiles", segment_image_by_grey_padded_tiles, "semantic_segmentation",
     "input_padding_value '114', from the 'padding_value' default of preprocess_image, or the padding of "
     "segment_image_by_tiles for the tiles, is not allowed")])
def test_preprocessing_the_contract_cannot_describe_is_refused(monkeypatch: pytest.MonkeyPatch, replaced_function: str,
                                                               replacement: Callable[..., Any], task: str,
                                                               expected_message: str) -> None:
    """
    Verifies that a preprocessing resizing with another interpolation, padding with another value, reading the photos
    as RGB or placing the resized photo at the top left is refused, and so are a scaling of the tiles with another
    interpolation and a padding of the tiles with another value, the error naming the contract key, the value found
    and the setting it comes from.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing a preprocessing function or class in the input
            contract module.
        replaced_function (str): The name of the replaced preprocessing function or class.
        replacement (Callable[..., Any]): The stand-in preprocessing function or class.
        task (str): "classification", or "semantic_segmentation" for a segmenter trained on tiles (TEST_TILING).
        expected_message (str): The expected part of the error.
    """
    monkeypatch.setattr(target=input_contract_module, name=replaced_function, value=replacement)
    model_configuration, specification = build_contract_inputs(
        task=task, keep_ratio=True, tiling_configuration=TEST_TILING if task == "semantic_segmentation" else None)
    with pytest.raises(expected_exception=ValueError, match=re.escape(pattern=expected_message)):
        build_input_contract(task=task, model_configuration=model_configuration, specification=specification)


@pytest.mark.parametrize(argnames=("removed_key", "changed_entries", "expected_message"), argvalues=[
    ("input_layout", {}, "missing key input_layout"),
    (None, {"input_mean": "0.5"}, "unknown key input_mean"),
    (None, {"input_size": "0"}, "input_size '0', from Data.image_settings.size, is not allowed (allowed: a positive "
                                "integer)"),
    (None, {"input_tile_scale": "-0.5"}, "input_tile_scale '-0.5', from Data.tiling.scale, is not allowed (allowed: "
                                         "'none' or a positive number)"),
    (None, {"input_contract_version": "1"}, "input_contract_version '1', from INPUT_CONTRACT_VERSION, is not allowed "
                                            "(allowed: '2')")])
def test_contract_read_back_with_a_wrong_entry_is_refused(removed_key: Optional[str], changed_entries: Dict[str, str],
                                                          expected_message: str) -> None:
    """
    Verifies that check_input_contract, run by a server on the contract it reads back, accepts the contract of a run
    and refuses one missing a key, holding an unknown key, an input size that is not a positive integer, a tile scale
    that is not a positive number, or another contract version.

    Args:
        removed_key (Optional[str]): The key removed from the contract, None to keep every key.
        changed_entries (Dict[str, str]): The entries added to the contract or replacing its values.
        expected_message (str): The expected part of the error.
    """
    model_configuration, specification = build_contract_inputs(task="classification", keep_ratio=False,
                                                               tiling_configuration=None)
    input_contract = build_input_contract(task="classification", model_configuration=model_configuration,
                                          specification=specification)
    check_input_contract(input_contract=input_contract)
    read_back_contract = {key: value for key, value in input_contract.items() if key != removed_key} | changed_entries
    with pytest.raises(expected_exception=ValueError, match=re.escape(pattern=expected_message)):
        check_input_contract(input_contract=read_back_contract)
