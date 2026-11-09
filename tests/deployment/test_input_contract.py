"""
Tests of the input contract of the exported models (vision_training/deployment/input_contract.py), written by
vision_training/deployment/export_model.py: a tiny classifier exported with keep_ratio False and True carries its
contract in the metadata of its ONNX file and in the preprocessing.json next to both exported models, and the input
built from that contract alone for photos that are not square equals the input of the training pixel for pixel; a run
using an interpolation, a padding value, a channel order or a padding placement the contract cannot describe, or a
segmenter trained on tiles, is refused with the name of the setting, before the export writes any file; the sequence
models get the NTCHW layout; and a contract read back with a missing, unknown or wrong entry is refused.
"""

import json
import re
import cv2
import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple

import vision_training.deployment.input_contract as input_contract_module

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.preprocessing import compute_letterbox_region, preprocess_image, read_image
from vision_training.deployment.export_model import export_experiment, hide_known_export_warnings
from vision_training.deployment.export_specifications import (SEQUENCE_TASKS, TASK_INPUT_NAMES,
                                                              build_input_specification)
from vision_training.deployment.input_contract import (INPUT_CONTRACT_KEYS, PREPROCESSING_FILE_NAME,
                                                       build_input_contract, check_input_contract)
from vision_training.models.model_registry import MODEL_DTYPE, TASK_MODEL_CLASSES
from vision_training.trainers.checkpointing import get_checkpoint_file
from tests.deployment.test_deployment_utilities import TEST_IMAGE_SIZE, write_test_experiment

# Exported iteration of the test runs and largest batch size of their exports
EXPORTED_ITERATION = 1
TEST_MAXIMUM_BATCH_SIZE = 4

# Channels of a photo decoded to RGB in the order of every value of input_color_order
CONTRACT_CHANNEL_ORDERS = {"BGR": [2, 1, 0]}

# OpenCV interpolation of every value of input_interpolation
CONTRACT_INTERPOLATIONS = {"linear": cv2.INTER_LINEAR}


def write_trained_classifier(experiment_folder: Path, keep_ratio: bool) -> None:
    """
    Writes the run folder of a tiny trained classifier as export_experiment reads it: the synthetic classification
    configuration with 32 pixel inputs, 3 classes and the resize asked for, and a checkpoint of seeded weights at the
    exported iteration.

    Args:
        experiment_folder (Path): The run folder to create.
        keep_ratio (bool): The 'Data.image_settings.keep_ratio' of the run.
    """
    write_test_experiment(experiment_folder=experiment_folder, task="classification", keep_ratio=keep_ratio)
    _, model_configuration = load_experiment_configuration(
        configuration_path=experiment_folder / "training_configuration.yaml")

    # Seeded weights saved in the checkpoint layout of the trainers
    torch.manual_seed(seed=0)
    model = TASK_MODEL_CLASSES["classification"](configuration=model_configuration, device=torch.device("cpu"),
                                                 dtype=MODEL_DTYPE)
    checkpoint_path = get_checkpoint_file(weights_directory=experiment_folder / "Weights", iteration=EXPORTED_ITERATION)
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(obj={"model_state": model.state_dict()}, f=checkpoint_path)


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
    Builds the network input of a photo from the input contract alone, as a server reading the contract of a model
    would: the photo is decoded by Pillow rather than by the OpenCV reader of the training, and neither preprocess_image
    nor compute_letterbox_region is called, so that the comparison with the input of the training checks the contract
    itself.

    Args:
        photo_path (Path): The photo, a PNG file.
        input_contract (Dict[str, str]): The contract read back from the exported model.

    Returns:
        np.ndarray: The [1, 3, S, S] float32 network input.
    """
    # Values of the contract this reference implements
    assert input_contract["input_resize"] in ["stretch", "keep_ratio"]
    assert (input_contract["input_padding_placement"], input_contract["input_layout"],
            input_contract["input_value_range"]) == ("centred", "NCHW", "0-255")

    # Photo decoded to RGB, then put in the channel order of the contract
    with Image.open(fp=photo_path) as photo_file:
        rgb_photo = np.asarray(a=photo_file.convert(mode="RGB"))
    photo = np.ascontiguousarray(a=rgb_photo[:, :, CONTRACT_CHANNEL_ORDERS[input_contract["input_color_order"]]])
    photo_height, photo_width = photo.shape[:2]
    input_size = int(input_contract["input_size"])
    interpolation = CONTRACT_INTERPOLATIONS[input_contract["input_interpolation"]]

    # Stretch: both sides to the input size, whatever the aspect ratio of the photo
    if input_contract["input_resize"] == "stretch":
        network_photo = cv2.resize(src=photo, dsize=(input_size, input_size), interpolation=interpolation)
    else:
        # Keep ratio: one scale for both sides, rounded half to even, then the photo centred on the padding
        scale = input_size / max(photo_height, photo_width)
        resized_height, resized_width = round(photo_height * scale), round(photo_width * scale)
        top, left = (input_size - resized_height) // 2, (input_size - resized_width) // 2
        network_photo = np.full(shape=(input_size, input_size, 3), dtype=np.uint8,
                                fill_value=int(input_contract["input_padding_value"]))
        network_photo[top:top + resized_height, left:left + resized_width] = cv2.resize(
            src=photo, dsize=(resized_width, resized_height), interpolation=interpolation)

    # Channels first with a batch of one, the uint8 values cast to float32 without scaling
    return network_photo.transpose(2, 0, 1)[None].astype(np.float32)


def build_contract_inputs(task: str, keep_ratio: bool,
                          tiling_activated: Optional[bool]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds the model configuration and the export specification read by build_input_contract for a run of a task,
    without a model: the image settings with 32 pixel inputs (and the tiling of the segmentation), and the inputs of
    the task from build_input_specification, sequences of up to 5 frames for the sequence tasks.

    Args:
        task (str): The task of the run.
        keep_ratio (bool): The 'Data.image_settings.keep_ratio' of the run.
        tiling_activated (Optional[bool]): The 'Data.tiling.activated' of a segmentation run, None for the other tasks.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The model configuration and the export specification.
    """
    model_configuration = {"Data": {"image_settings": {"size": TEST_IMAGE_SIZE, "keep_ratio": keep_ratio}}}
    if tiling_activated is not None:
        model_configuration["Data"]["tiling"] = {"activated": tiling_activated}
    maximum_sequence_length = 5 if task in SEQUENCE_TASKS else None
    specification = {"inputs": [build_input_specification(input_name=input_name, image_size=TEST_IMAGE_SIZE,
                                                          maximum_sequence_length=maximum_sequence_length,
                                                          input_index=input_index)
                                for input_index, input_name in enumerate(TASK_INPUT_NAMES[task])]}
    return model_configuration, specification


def preprocess_image_with_area_interpolation(image: np.ndarray, image_size: int, keep_ratio: bool = True,
                                             interpolation: int = cv2.INTER_AREA,
                                             padding_value: int = 0) -> np.ndarray:
    """
    Stand-in for a preprocess_image resizing the images with the area interpolation by default, which the input
    contract cannot describe.

    Args:
        image (np.ndarray): The image to resize.
        image_size (int): The side of the square network input.
        keep_ratio (bool): Whether the aspect ratio is kept. Defaults to True.
        interpolation (int): The OpenCV interpolation. Defaults to cv2.INTER_AREA.
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
    experiment_folder = tmp_path / "CLS_TEST_MC3"
    write_trained_classifier(experiment_folder=experiment_folder, keep_ratio=keep_ratio)
    repository = tmp_path / "served_models"
    export_experiment(experiment_folder=experiment_folder, iteration=EXPORTED_ITERATION, use_averaged_weights=False,
                      formats=["torchscript", "onnx"], repository=repository, model_version=1,
                      maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, tensorrt_precision=None, use_saved_temperature=False)
    expected_contract = {"input_contract_version": "1", "input_color_order": "BGR", "input_resize": expected_resize,
                         "input_size": "32", "input_interpolation": "linear", "input_padding_value": "0",
                         "input_padding_placement": "centred", "input_layout": "NCHW", "input_value_range": "0-255"}

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
def test_export_of_an_input_the_contract_cannot_describe_writes_nothing(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the export of a run whose photos are read as RGB stops with the name of the setting before writing
    any file in the model repository.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing read_image in the input contract module.
    """
    experiment_folder = tmp_path / "CLS_TEST_MC3"
    write_trained_classifier(experiment_folder=experiment_folder, keep_ratio=True)
    monkeypatch.setattr(target=input_contract_module, name="read_image", value=read_image_as_rgb)
    repository = tmp_path / "served_models"
    with pytest.raises(expected_exception=ValueError,
                       match=re.escape(pattern="input_color_order 'RGB', from the channel order of read_image")):
        export_experiment(experiment_folder=experiment_folder, iteration=EXPORTED_ITERATION,
                          use_averaged_weights=False, formats=["torchscript", "onnx"], repository=repository,
                          model_version=1, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, tensorrt_precision=None,
                          use_saved_temperature=False)
    assert not repository.exists()


@pytest.mark.parametrize(argnames=("task", "keep_ratio", "tiling_activated", "expected_resize", "expected_layout"),
                         argvalues=[("semantic_segmentation", False, False, "stretch", "NCHW"),
                                    ("sequence_comparison", True, None, "keep_ratio", "NTCHW")])
def test_contract_follows_the_image_settings_and_the_inputs_of_the_task(
        task: str, keep_ratio: bool, tiling_activated: Optional[bool], expected_resize: str,
        expected_layout: str) -> None:
    """
    Verifies the contract of a stretched segmenter trained on whole images, and of a comparison model keeping the
    aspect ratio, whose two inputs are sequences of frames (NTCHW), every key written in the order of
    INPUT_CONTRACT_KEYS.

    Args:
        task (str): The task of the run.
        keep_ratio (bool): The 'Data.image_settings.keep_ratio' of the run.
        tiling_activated (Optional[bool]): The 'Data.tiling.activated' of the segmenter, None for the other task.
        expected_resize (str): The expected input_resize.
        expected_layout (str): The expected input_layout.
    """
    model_configuration, specification = build_contract_inputs(task=task, keep_ratio=keep_ratio,
                                                               tiling_activated=tiling_activated)
    input_contract = build_input_contract(task=task, model_configuration=model_configuration,
                                          specification=specification)
    assert list(input_contract) == list(INPUT_CONTRACT_KEYS)
    assert input_contract == {"input_contract_version": "1", "input_color_order": "BGR",
                              "input_resize": expected_resize, "input_size": "32", "input_interpolation": "linear",
                              "input_padding_value": "0", "input_padding_placement": "centred",
                              "input_layout": expected_layout, "input_value_range": "0-255"}


@pytest.mark.parametrize(argnames=("replaced_function", "replacement", "expected_message"), argvalues=[
    ("preprocess_image", preprocess_image_with_area_interpolation,
     "input_interpolation 'OpenCV interpolation 3', from the 'interpolation' default of preprocess_image, is not "
     "allowed (allowed: 'linear')"),
    ("preprocess_image", preprocess_image_with_grey_padding,
     "input_padding_value '114', from the 'padding_value' default of preprocess_image, is not allowed"),
    ("read_image", read_image_as_rgb, "input_color_order 'RGB', from the channel order of read_image, is not allowed"),
    ("compute_letterbox_region", compute_top_left_letterbox_region,
     "input_padding_placement 'region (top, left, height, width) (0, 0, 18, 32) of a 1080 x 1920 photo instead of "
     "(7, 0, 18, 32)', from compute_letterbox_region, is not allowed")])
def test_preprocessing_the_contract_cannot_describe_is_refused(monkeypatch: pytest.MonkeyPatch, replaced_function: str,
                                                               replacement: Callable[..., Any],
                                                               expected_message: str) -> None:
    """
    Verifies that a preprocessing resizing with another interpolation, padding with another value, reading the photos
    as RGB or placing the resized photo at the top left is refused, the error naming the contract key, the value found
    and the setting it comes from.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing a preprocessing function in the input contract
            module.
        replaced_function (str): The name of the replaced preprocessing function.
        replacement (Callable[..., Any]): The stand-in preprocessing function.
        expected_message (str): The expected part of the error.
    """
    monkeypatch.setattr(target=input_contract_module, name=replaced_function, value=replacement)
    model_configuration, specification = build_contract_inputs(task="classification", keep_ratio=True,
                                                               tiling_activated=None)
    with pytest.raises(expected_exception=ValueError, match=re.escape(pattern=expected_message)):
        build_input_contract(task="classification", model_configuration=model_configuration,
                             specification=specification)


def test_segmenter_trained_on_tiles_is_refused() -> None:
    """
    Verifies that a segmenter trained on tiles of the scaled images is refused, the error naming
    'Data.tiling.activated'.
    """
    model_configuration, specification = build_contract_inputs(task="semantic_segmentation", keep_ratio=True,
                                                               tiling_activated=True)
    with pytest.raises(expected_exception=ValueError, match=re.escape(pattern="'Data.tiling.activated' is True")):
        build_input_contract(task="semantic_segmentation", model_configuration=model_configuration,
                             specification=specification)


@pytest.mark.parametrize(argnames=("removed_key", "changed_entries", "expected_message"), argvalues=[
    ("input_layout", {}, "missing key input_layout"),
    (None, {"input_mean": "0.5"}, "unknown key input_mean"),
    (None, {"input_size": "0"}, "input_size '0', from Data.image_settings.size, is not allowed (allowed: a positive "
                                "integer)"),
    (None, {"input_contract_version": "2"}, "input_contract_version '2', from INPUT_CONTRACT_VERSION, is not allowed")])
def test_contract_read_back_with_a_wrong_entry_is_refused(removed_key: Optional[str], changed_entries: Dict[str, str],
                                                          expected_message: str) -> None:
    """
    Verifies that check_input_contract, run by a server on the contract it reads back, accepts the contract of a run
    and refuses one missing a key, holding an unknown key, an input size that is not a positive integer or another
    contract version.

    Args:
        removed_key (Optional[str]): The key removed from the contract, None to keep every key.
        changed_entries (Dict[str, str]): The entries added to the contract or replacing its values.
        expected_message (str): The expected part of the error.
    """
    model_configuration, specification = build_contract_inputs(task="classification", keep_ratio=False,
                                                               tiling_activated=None)
    input_contract = build_input_contract(task="classification", model_configuration=model_configuration,
                                          specification=specification)
    check_input_contract(input_contract=input_contract)
    read_back_contract = {key: value for key, value in input_contract.items() if key != removed_key} | changed_entries
    with pytest.raises(expected_exception=ValueError, match=re.escape(pattern=expected_message)):
        check_input_contract(input_contract=read_back_contract)
