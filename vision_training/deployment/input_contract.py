"""
Input contract of the exported models: how a server (such as a Triton deployment querying the exported classifiers)
builds the network input from a photo, so that it never has to guess the channel order, the resize, the padding, the
layout, the value range of the training or the image windows a detector takes with its images.
deployment/export_model.py builds it once per export with build_input_contract and writes it into the 'metadata_props'
of every ONNX file and into 'preprocessing.json' next to every exported model. Every value comes from the training
configuration of the run (Data.image_settings, Data.tiling) or from the preprocessing code run by the data loaders
(data/preprocessing.py: read_image, preprocess_image, compute_letterbox_region, compute_image_window; for a segmenter
trained on tiles, SegmentationDataset.cut_tile and the sliding window of models/segmentation/sliding_window.py), never
from a copy written by hand, and a run whose input the contract cannot describe stops the export with the name of the
setting.
"""

import inspect
import json
import re
import tempfile
import cv2
import numpy as np
import onnx
import torch

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

from vision_training.data.preprocessing import (compute_image_window, compute_letterbox_region, preprocess_image,
                                                read_image)
from vision_training.data.segmentation.segmentation_dataset import SegmentationDataset
from vision_training.deployment.export_specifications import IMAGE_WINDOWS_INPUT_NAME
from vision_training.models.segmentation.sliding_window import segment_image_by_tiles

# Version of the keys and values below, raised whenever a key is added or a value changes meaning, so that a server
# refuses a model whose contract it does not know: version 2 describes the segmenters trained on tiles, version 3 the
# image windows of the detectors and instance segmenters
INPUT_CONTRACT_VERSION = "3"

# File written next to every exported model, holding the same keys and values as the metadata of the ONNX file
PREPROCESSING_FILE_NAME = "preprocessing.json"

# Every key of the input contract, in the order of the written files, with its allowed values ('allowed_values'), the
# kind of number it may hold instead ('allowed_number', None for none) and the setting the export reads it from
# ('source', named by the error of a run the contract cannot describe). Every key is always written: the padding keys
# apply to the keep_ratio and tiles resizes, the tile keys hold 'none' for the stretch and keep_ratio resizes
INPUT_CONTRACT_KEYS = {
    # Version of the keys and values of the contract
    "input_contract_version": {"allowed_values": [INPUT_CONTRACT_VERSION], "allowed_number": None,
                               "source": "INPUT_CONTRACT_VERSION"},
    # Channel order of the decoded photo, as cv2.imread decodes it (3 channels of 8 bits, EXIF orientation applied)
    "input_color_order": {"allowed_values": ["BGR"], "allowed_number": None,
                          "source": "the channel order of read_image"},
    # Photo stretched to the square input whatever its aspect ratio, resized with its aspect ratio kept (the longest
    # side to the input size, the other one to max(1, round(side * size / longest side)), half to even) and padded
    # to the square, or scaled by the tile scale and cut into overlapping square tiles of the input size, one input per
    # tile
    "input_resize": {"allowed_values": ["stretch", "keep_ratio", "tiles"], "allowed_number": None,
                     "source": "Data.image_settings.keep_ratio and Data.tiling.activated"},
    # Side of the square network input in pixels, the side of a tile for the tiles
    "input_size": {"allowed_values": [], "allowed_number": "positive integer", "source": "Data.image_settings.size"},
    # OpenCV resize of the uint8 photo before its conversion to float32: bilinear (cv2.INTER_LINEAR), area averaging
    # (cv2.INTER_AREA), or none for the tiles of a scale of 1
    "input_interpolation": {"allowed_values": ["linear", "area", "none"], "allowed_number": None,
                            "source": "the 'interpolation' default of preprocess_image, or the scaling of "
                                      "SegmentationDataset.cut_tile for the tiles"},
    # Value of every channel of the padding pixels of the keep_ratio and tiles resizes
    "input_padding_value": {"allowed_values": ["0"], "allowed_number": None,
                            "source": "the 'padding_value' default of preprocess_image, or the padding of "
                                      "segment_image_by_tiles for the tiles"},
    # Keep ratio: resized photo centred in the square input, at the top row (size - resized height) // 2 and the left
    # column (size - resized width) // 2, an odd padding pixel going to the bottom or the right. Tiles: scaled photo
    # padded on its bottom and right sides to at least a tile
    "input_padding_placement": {"allowed_values": ["centred", "bottom_right"], "allowed_number": None,
                                "source": "compute_letterbox_region, or segment_image_by_tiles for the tiles"},
    # Factor scaling the photo before it is cut into tiles, each scaled side being max(1, round(side * scale)), half to
    # even, or none for the other resizes
    "input_tile_scale": {"allowed_values": ["none"], "allowed_number": "positive number",
                         "source": "Data.tiling.scale"},
    # Pixels shared by two neighbouring tiles: the tiles of a padded side of length L start at 0, size - overlap,
    # 2 (size - overlap)... below L - size, then at L - size, row by row, or none for the other resizes
    "input_tile_overlap": {"allowed_values": ["none"], "allowed_number": "non-negative integer",
                           "source": "Data.tiling.overlap"},
    # Dimensions of every input: batch, channels, height and width, with the frames after the batch for the sequences
    "input_layout": {"allowed_values": ["NCHW", "NTCHW"], "allowed_number": None,
                     "source": "the network inputs of the export specification"},
    # Pixel values from 0 to 255, the uint8 photo cast to float32 without scaling, the normalization being in the graph
    "input_value_range": {"allowed_values": ["0-255"], "allowed_number": None,
                          "source": "the pixel type of read_image"},
    # Second input 'image_windows' of the detectors and instance segmenters, which clip their boxes to it: the float32
    # [B, 4] [left, top, right, bottom] of the resized photo in the input, left and top being the left column and top
    # row of the keep_ratio placement and right and bottom adding the resized width and height, [0, 0, size, size] for
    # the stretch; none for the models without that input
    "input_image_windows": {"allowed_values": ["none", "resized_photo"], "allowed_number": None,
                            "source": "the 'image_windows' input of the export specification, checked on "
                                      "compute_image_window"},
}

# Contract value of the OpenCV interpolations the contract describes
INTERPOLATION_VALUES = {cv2.INTER_LINEAR: "linear", cv2.INTER_AREA: "area"}

# Decimal number written by repr for a positive float, such as '0.5', '2.0' or '1e-05'
DECIMAL_NUMBER_PATTERN = re.compile(pattern=r"\d+(\.\d+)?([eE][-+]?\d+)?")

# Blue, green and red pixels in the BGR order OpenCV writes from, read back through read_image to find its channel order
COLOR_PROBE_PIXELS = np.array(object=[[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)


def find_preprocessing_defaults() -> Dict[str, Any]:
    """
    Reads the default arguments of preprocess_image, which give the interpolation and the padding value of the images
    of every dataset (only the segmentation masks pass their own), so that the contract follows the preprocessing of
    the training rather than a copy of its values.

    Returns:
        Dict[str, Any]: The default value of every argument of preprocess_image that has one, keyed by argument name.
    """
    return {argument_name: parameter.default
            for argument_name, parameter in inspect.signature(obj=preprocess_image).parameters.items()
            if parameter.default is not inspect.Parameter.empty}


def find_decoded_photo_description() -> Tuple[str, str]:
    """
    Finds the channel order and the value range of the photos as the data loaders read them: blue, green and red pixels
    are written to a PNG file by OpenCV, which takes BGR arrays, and read back through read_image, unchanged pixels
    meaning BGR and reversed channels RGB. The range is the one of the pixel type of the decoded photo, cast to float32
    without scaling before the network.

    Returns:
        Tuple[str, str]: The channel order ('BGR', 'RGB', or a description of the decoded pixels) and the value range
            ('0-255' for uint8 pixels, or a description of a pixel type that is not an integer).
    """
    with tempfile.TemporaryDirectory() as probe_folder:
        probe_path = Path(probe_folder) / "color_probe.png"
        cv2.imwrite(filename=str(probe_path), img=COLOR_PROBE_PIXELS)
        decoded_pixels = read_image(image_path=probe_path)

    # Channel order of the decoded pixels against the written ones
    if np.array_equal(a1=decoded_pixels, a2=COLOR_PROBE_PIXELS):
        color_order = "BGR"
    elif np.array_equal(a1=decoded_pixels, a2=COLOR_PROBE_PIXELS[:, :, ::-1]):
        color_order = "RGB"
    else:
        color_order = f"pixels {decoded_pixels.tolist()} read back from the BGR pixels {COLOR_PROBE_PIXELS.tolist()}"

    # Range of the integer pixel type
    if not np.issubdtype(arg1=decoded_pixels.dtype, arg2=np.integer):
        return color_order, f"{decoded_pixels.dtype} pixels"
    type_information = np.iinfo(int_type=decoded_pixels.dtype)
    return color_order, f"{type_information.min}-{type_information.max}"


def find_padding_placement(image_size: int) -> str:
    """
    Checks the region where compute_letterbox_region places the resized photo of the keep_ratio preprocessing against
    the keep_ratio resize and the centred placement of the contract, on probe photo sizes: one pixel short of the input
    on either axis, an odd padding telling a centre rounded down from one rounded up, the landscape and portrait
    camera frames of 1920 x 1080 pixels, resized, and a photo one pixel high and four inputs wide, whose short side
    stays one pixel instead of rounding to none.

    Args:
        image_size (int): The side of the square network input.

    Returns:
        str: 'centred', or the description of the first region placed otherwise.
    """
    probe_shapes = [(image_size - 1, image_size), (image_size, image_size - 1), (1080, 1920), (1920, 1080),
                    (1, 4 * image_size)]
    for image_height, image_width in probe_shapes:
        letterbox_region = compute_letterbox_region(image_height=image_height, image_width=image_width,
                                                    image_size=image_size)

        # Region of the contract: one scale for both sides, each side kept to one pixel at least, then the resized photo
        # centred, rounded down
        scale = image_size / max(image_height, image_width)
        resized_height = max(1, int(round(image_height * scale)))
        resized_width = max(1, int(round(image_width * scale)))
        centred_region = ((image_size - resized_height) // 2, (image_size - resized_width) // 2, resized_height,
                          resized_width)
        if tuple(letterbox_region) != centred_region:
            return (f"region (top, left, height, width) {tuple(letterbox_region)} of a {image_height} x {image_width} "
                    f"photo instead of {centred_region}")
    return "centred"


def find_image_window_description(specification: Dict[str, Any], image_size: int, keep_ratio: bool) -> str:
    """
    Describes the image windows input of the detectors and instance segmenters for the contract, checking
    compute_image_window, which gives them to the evaluation and the clients, against the window the contract describes
    on the probe photo sizes of find_padding_placement: the resized photo placed by compute_letterbox_region with
    keep_ratio, the whole input for the stretch.

    Args:
        specification (Dict[str, Any]): The export specification of the task, from get_export_specification.
        image_size (int): The side of the square network input.
        keep_ratio (bool): Whether the photos are resized keeping their aspect ratio ('Data.image_settings.keep_ratio').

    Returns:
        str: 'none' for the models without the input, 'resized_photo', or the description of the first window
            computed otherwise.
    """
    if IMAGE_WINDOWS_INPUT_NAME not in [input_specification["name"] for input_specification in specification["inputs"]]:
        return "none"
    probe_shapes = [(image_size - 1, image_size), (image_size, image_size - 1), (1080, 1920), (1920, 1080),
                    (1, 4 * image_size)]
    for image_height, image_width in probe_shapes:
        # Window of the contract: the region of the resized photo, or the whole input for the stretch
        top, left, resized_height, resized_width = compute_letterbox_region(image_height=image_height,
                                                                            image_width=image_width,
                                                                            image_size=image_size)
        expected_window = ([left, top, left + resized_width, top + resized_height] if keep_ratio else
                           [0, 0, image_size, image_size])
        image_window = compute_image_window(image_height=image_height, image_width=image_width, image_size=image_size,
                                            keep_ratio=keep_ratio).tolist()
        if image_window != [float(coordinate) for coordinate in expected_window]:
            return f"window {image_window} of a {image_height} x {image_width} photo instead of {expected_window}"
    return "resized_photo"


def find_input_layout(specification: Dict[str, Any], image_size: int) -> str:
    """
    Finds the layout of the network inputs from the export specification: images of [3, S, S] after the batch
    dimension (NCHW), or sequences of such frames (NTCHW) for the sequence tasks, both inputs of the comparison model
    sharing it. S must be the configured input size, so that 'input_size' is the side the exported graph accepts. The
    image windows of the detectors are described by 'input_image_windows' instead.

    Args:
        specification (Dict[str, Any]): The export specification of the task, from get_export_specification.
        image_size (int): The configured side of the square network input, 'Data.image_settings.size'.

    Returns:
        str: 'NCHW', 'NTCHW', or the description of inputs of another layout.
    """
    # Dimensions after the batch one of every layout, the number of frames of a sequence being dynamic (-1)
    layout_dimensions = {"NCHW": [3, image_size, image_size], "NTCHW": [-1, 3, image_size, image_size]}
    image_inputs = [input_specification for input_specification in specification["inputs"]
                    if input_specification["name"] != IMAGE_WINDOWS_INPUT_NAME]
    input_layouts = [next((layout for layout, dimensions in layout_dimensions.items()
                           if dimensions == input_specification["dimensions"]), None)
                     for input_specification in image_inputs]
    if None in input_layouts or len(set(input_layouts)) != 1:
        return (", ".join(f"input '{input_specification['name']}' of dimensions {input_specification['dimensions']}"
                          for input_specification in image_inputs) +
                f" for the size {image_size}")
    return input_layouts[0]


class TileRecorder:
    """
    Stand-in for the model given to segment_image_by_tiles by find_tile_padding: it records every batch of tiles it is
    given and returns zero probabilities of one class, so that the tiles of the sliding window are compared with those
    of the contract without a model.
    """

    def __init__(self) -> None:
        """
        Starts with no recorded tile.
        """
        self.tiles = []

    def __call__(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Records a batch of tiles.

        Args:
            tiles (torch.Tensor): The [N, 3, S, S] batch of tiles.

        Returns:
            torch.Tensor: The [N, 1, S, S] zero probabilities of the tiles.
        """
        self.tiles.append(tiles.clone())
        return torch.zeros(size=(tiles.shape[0], 1) + tuple(tiles.shape[2:]), dtype=torch.float32)


def find_tile_interpolation(tiling_scale: float) -> str:
    """
    Finds the interpolation with which SegmentationDataset.cut_tile scales the photos of a segmenter trained on tiles,
    by running it on random probe photos through a stand-in dataset holding the settings it reads, in the 'whole' tile
    sampling of the test split, the one of the evaluation and of the deployment. The scaled photo must have the sides
    max(1, round(side * scale)) and equal the photo itself ('none') or its OpenCV resize with one value of
    INTERPOLATION_VALUES, the same value on every probe: a small photo of odd sides, which a scale of a half does not
    divide into whole pixels (where the linear and area shrinks of OpenCV can agree), and a camera frame of 1920 x 1080
    pixels, whose sides change even with a scale close to 1.

    Args:
        tiling_scale (float): The scale of the photos before the tiles are cut, 'Data.tiling.scale'.

    Returns:
        str: 'none', a value of INTERPOLATION_VALUES, or the description of a scaling that no value or several values
            describe.
    """
    random_generator = np.random.default_rng(seed=0)
    stand_in_dataset = SimpleNamespace(tiling_scale=tiling_scale, tile_sampling="whole")
    matching_values = {"none"} | set(INTERPOLATION_VALUES.values())
    for probe_height, probe_width in [(37, 53), (1080, 1920)]:
        probe_photo = random_generator.integers(low=0, high=256, size=(probe_height, probe_width, 3), dtype=np.uint8)
        scaled_photo, _ = SegmentationDataset.cut_tile(
            self=stand_in_dataset, image=probe_photo, mask=np.zeros(shape=(probe_height, probe_width), dtype=np.uint8),
            sample_identifier=0)

        # Sides of the contract, then every description of the scaled photo, the photo itself when its sides are kept
        scaled_size = (max(1, int(round(probe_width * tiling_scale))), max(1, int(round(probe_height * tiling_scale))))
        if scaled_photo.shape[:2] != (scaled_size[1], scaled_size[0]):
            return (f"sides {scaled_photo.shape[0]} x {scaled_photo.shape[1]} of a {probe_height} x {probe_width} "
                    f"photo scaled by {tiling_scale}, instead of {scaled_size[1]} x {scaled_size[0]}")
        candidate_photos = {value: cv2.resize(src=probe_photo, dsize=scaled_size, interpolation=interpolation)
                            for interpolation, value in INTERPOLATION_VALUES.items()}
        if scaled_photo.shape == probe_photo.shape:
            candidate_photos["none"] = probe_photo
        matching_values &= {value for value, candidate_photo in candidate_photos.items()
                            if np.array_equal(a1=scaled_photo, a2=candidate_photo)}

    # No resize describes the photos of a scale of 1 best, every interpolation keeping them unchanged
    if "none" in matching_values:
        return "none"
    if len(matching_values) == 1:
        return next(iter(matching_values))
    return f"a scaling matched by {sorted(matching_values) or 'no interpolation'} on the probe photos"


def find_tile_padding(image_size: int, overlap: int) -> Tuple[str, str]:
    """
    Finds the padding value of the tiles of a segmenter trained on tiles and checks their padding and placement, by
    running segment_image_by_tiles on random probe photos with a stand-in model recording the tiles (TileRecorder): a
    photo one pixel shorter than a tile and longer than two tiles, then the same photo turned. The padding value is the
    bottom right pixel of the first tile, a padding pixel with the placement of the contract, whose tiles the recorded
    ones must equal: the photo padded with that value on its bottom and right sides to at least a tile, and cut at the
    tile starts of the contract (input_tile_overlap), row by row.

    Args:
        image_size (int): The side of the square tiles, the network input size.
        overlap (int): The pixels shared by two neighbouring tiles, 'Data.tiling.overlap'.

    Returns:
        Tuple[str, str]: The padding value, and 'bottom_right' or the description of the first probe cut otherwise.
    """
    random_generator = np.random.default_rng(seed=0)
    padding_values = []
    for probe_height, probe_width in [(image_size - 1, 2 * image_size + 1), (2 * image_size + 1, image_size - 1)]:
        # Photo pixels from 1 to 255, so that a padding of 0 stands apart from them
        probe_photo = random_generator.integers(low=1, high=256, size=(probe_height, probe_width, 3)).astype(np.float32)
        tile_recorder = TileRecorder()
        segment_image_by_tiles(image=torch.from_numpy(probe_photo).permute(dims=(2, 0, 1)),
                               predict_tile_probabilities=tile_recorder, tile_size=image_size, overlap=overlap,
                               tiles_per_batch=4)
        recorded_tiles = torch.cat(tensors=tile_recorder.tiles).permute(dims=(0, 2, 3, 1)).numpy()
        padding_values.append(float(recorded_tiles[0, -1, -1, 0]))

        # Tiles of the contract: bottom and right padding to at least a tile, then the tile starts along both sides
        padded_photo = np.pad(array=probe_photo, mode="constant", constant_values=padding_values[-1],
                              pad_width=((0, max(0, image_size - probe_height)), (0, max(0, image_size - probe_width)),
                                         (0, 0)))
        tile_starts = [[0] if length <= image_size else
                       list(range(0, length - image_size, image_size - overlap)) + [length - image_size]
                       for length in padded_photo.shape[:2]]
        expected_tiles = np.stack(arrays=[padded_photo[tile_top:tile_top + image_size,
                                                       tile_left:tile_left + image_size]
                                          for tile_top in tile_starts[0] for tile_left in tile_starts[1]])
        if recorded_tiles.shape != expected_tiles.shape or not np.array_equal(a1=recorded_tiles, a2=expected_tiles):
            return (f"{padding_values[-1]:g}",
                    f"{recorded_tiles.shape[0]} tiles of a {probe_height} x {probe_width} photo cut otherwise than "
                    f"the {expected_tiles.shape[0]} tiles padded on the bottom and right sides")

    # One padding value for every probe, written as an integer when it is one
    if len(set(padding_values)) != 1:
        return f"padding values {padding_values} on the probe photos", "bottom_right"
    return f"{padding_values[0]:g}", "bottom_right"


def is_allowed_number(value: Any, number_kind: str) -> bool:
    """
    Tells whether a value of the input contract is a number of the kind a key allows, written as the export writes it:
    a decimal integer without leading zero for the integers, a decimal number as written by repr for the scale.

    Args:
        value (Any): The value read from the contract, a string when it was written by the export.
        number_kind (str): 'positive integer', 'non-negative integer' or 'positive number'.

    Returns:
        bool: True when the value is a number of that kind.
    """
    if not isinstance(value, str):
        return False
    if number_kind == "positive number":
        return DECIMAL_NUMBER_PATTERN.fullmatch(string=value) is not None and float(value) > 0
    minimum = 1 if number_kind == "positive integer" else 0
    return value.isdecimal() and value == str(int(value)) and int(value) >= minimum


def check_input_contract(input_contract: Dict[str, str]) -> None:
    """
    Checks an input contract against the keys and allowed values of INPUT_CONTRACT_KEYS: the export checks the contract
    of the run before writing it, and a server can check the contract it reads back from a model before building its
    inputs.

    Args:
        input_contract (Dict[str, str]): The value of every key of the contract.

    Raises:
        ValueError: If a key is missing or unknown, or a value is not allowed, the error listing each of them with the
            setting its value comes from.
    """
    # Every key of the contract, and no other
    errors = [f"missing key {key}" for key in INPUT_CONTRACT_KEYS if key not in input_contract]
    errors += [f"unknown key {key}" for key in input_contract if key not in INPUT_CONTRACT_KEYS]

    # Every value among the allowed ones, or a number of the kind the key allows
    for key, key_definition in INPUT_CONTRACT_KEYS.items():
        if key not in input_contract:
            continue
        value, number_kind = input_contract[key], key_definition["allowed_number"]
        is_allowed = value in key_definition["allowed_values"] or (
            number_kind is not None and is_allowed_number(value=value, number_kind=number_kind))
        allowed_text = " or ".join([f"'{allowed_value}'" for allowed_value in key_definition["allowed_values"]] +
                                   ([f"a {number_kind}"] if number_kind is not None else []))
        if not is_allowed:
            errors.append(f"{key} {value!r}, from {key_definition['source']}, is not allowed (allowed: {allowed_text})")

    if errors:
        raise ValueError(f"The input contract version {INPUT_CONTRACT_VERSION} cannot describe this input:\n" +
                         "\n".join(errors))


def build_input_contract(task: str, model_configuration: Dict[str, Any],
                         specification: Dict[str, Any]) -> Dict[str, str]:
    """
    Builds the input contract of the models exported from a run, called by export_experiment before any file is written:
    the resize and size from the image settings of the run, the interpolation and padding value from the defaults of
    preprocess_image, the placement checked on compute_letterbox_region, the channel order and value range from
    read_image, and the layout and the image windows of the detectors from the export specification
    (find_image_window_description). A segmenter trained on tiles takes its scale and overlap from 'Data.tiling', its
    interpolation from SegmentationDataset.cut_tile and its padding from segment_image_by_tiles instead. The contract is
    checked by check_input_contract, so that a run the contract cannot describe stops the export with the name of the
    setting.

    Args:
        task (str): The task of the run.
        model_configuration (Dict[str, Any]): The model configuration of the run, holding 'Data.image_settings' (and
            'Data.tiling' for the semantic segmentation).
        specification (Dict[str, Any]): The export specification of the task, from get_export_specification.

    Returns:
        Dict[str, str]: The value of every key of INPUT_CONTRACT_KEYS, in its order.

    Raises:
        ValueError: If a value of the run is not allowed by the contract (check_input_contract).
    """
    image_settings = model_configuration["Data"]["image_settings"]
    image_size = image_settings["size"]
    color_order, value_range = find_decoded_photo_description()

    # A segmenter trained on tiles takes the tiles of the scaled photo, the other models the photo resized to the input
    if task == "semantic_segmentation" and model_configuration["Data"]["tiling"]["activated"]:
        tiling_configuration = model_configuration["Data"]["tiling"]
        padding_value, padding_placement = find_tile_padding(image_size=image_size,
                                                             overlap=tiling_configuration["overlap"])
        resize_values = {"input_resize": "tiles",
                         "input_interpolation": find_tile_interpolation(tiling_scale=tiling_configuration["scale"]),
                         "input_padding_value": padding_value, "input_padding_placement": padding_placement,
                         "input_tile_scale": repr(float(tiling_configuration["scale"])),
                         "input_tile_overlap": str(tiling_configuration["overlap"])}
    else:
        # Interpolation and padding value with which every dataset resizes its images
        preprocessing_defaults = find_preprocessing_defaults()
        interpolation = preprocessing_defaults["interpolation"]
        resize_values = {"input_resize": "keep_ratio" if image_settings["keep_ratio"] else "stretch",
                         "input_interpolation": (INTERPOLATION_VALUES[interpolation]
                                                 if interpolation in INTERPOLATION_VALUES else
                                                 f"OpenCV interpolation {interpolation}"),
                         "input_padding_value": str(preprocessing_defaults["padding_value"]),
                         "input_padding_placement": find_padding_placement(image_size=image_size),
                         "input_tile_scale": "none", "input_tile_overlap": "none"}

    # Every value in the key order of the contract
    contract_values = {"input_contract_version": INPUT_CONTRACT_VERSION, "input_color_order": color_order,
                       "input_size": str(image_size), "input_value_range": value_range,
                       "input_layout": find_input_layout(specification=specification, image_size=image_size),
                       "input_image_windows": find_image_window_description(
                           specification=specification, image_size=image_size,
                           keep_ratio=image_settings["keep_ratio"]),
                       **resize_values}
    input_contract = {key: contract_values[key] for key in INPUT_CONTRACT_KEYS}
    check_input_contract(input_contract=input_contract)
    return input_contract


def write_onnx_input_contract(model_path: Path, input_contract: Dict[str, str]) -> None:
    """
    Writes the input contract into the 'metadata_props' of an exported ONNX file, which ONNX Runtime gives as the
    custom metadata of the model (InferenceSession.get_modelmeta().custom_metadata_map). The entries of an earlier
    contract are replaced and the other entries kept. The graph is loaded without its external weights, which stay
    untouched in their 'model.onnx.data' file.

    Args:
        model_path (Path): The exported 'model.onnx' file.
        input_contract (Dict[str, str]): The contract of the run, from build_input_contract.
    """
    model_proto = onnx.load(f=str(model_path), load_external_data=False)

    # Other metadata kept, then the contract in its key order
    kept_entries = [(entry.key, entry.value) for entry in model_proto.metadata_props
                    if entry.key not in INPUT_CONTRACT_KEYS]
    del model_proto.metadata_props[:]
    for key, value in kept_entries + list(input_contract.items()):
        model_proto.metadata_props.add(key=key, value=value)
    onnx.save(proto=model_proto, f=str(model_path))


def write_preprocessing_file(model_folder: Path, input_contract: Dict[str, str]) -> Path:
    """
    Writes the input contract as 'preprocessing.json' in the version folder of an exported model, next to the model
    file (and to 'class_thresholds.json' for a classifier), for the servers reading files rather than ONNX metadata,
    and for the TorchScript models, which carry no metadata.

    Args:
        model_folder (Path): The version folder of the exported model, '<model name>/<version>/'.
        input_contract (Dict[str, str]): The contract of the run, from build_input_contract.

    Returns:
        Path: The written file.
    """
    preprocessing_path = model_folder / PREPROCESSING_FILE_NAME
    preprocessing_path.write_text(data=json.dumps(obj=input_contract, indent=2) + "\n")
    return preprocessing_path
