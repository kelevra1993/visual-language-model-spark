"""
Input contract of the exported models: how a server (such as a Triton deployment querying the exported classifiers)
builds the network input from a photo, so that it never has to guess the channel order, the resize, the padding, the
layout or the value range of the training. deployment/export_model.py builds it once per export with
build_input_contract and writes it into the 'metadata_props' of every ONNX file and into 'preprocessing.json' next to
every exported model. Every value comes from the training configuration of the run (Data.image_settings) or from the
preprocessing code run by the data loaders (data/preprocessing.py: read_image, preprocess_image,
compute_letterbox_region), never from a copy written by hand, and a run whose input the contract cannot describe stops
the export with the name of the setting.
"""

import inspect
import json
import tempfile
import cv2
import numpy as np
import onnx

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.data.preprocessing import compute_letterbox_region, preprocess_image, read_image

# Version of the keys and values below, raised whenever a key is added or a value changes meaning, so that a server
# refuses a model whose contract it does not know
INPUT_CONTRACT_VERSION = "1"

# File written next to every exported model, holding the same keys and values as the metadata of the ONNX file
PREPROCESSING_FILE_NAME = "preprocessing.json"

# Every key of the input contract, in the order of the written files, with its allowed values ('allowed_values', None
# for any positive integer written in decimal) and the setting the export reads it from ('source', named by the error
# of a run the contract cannot describe). Every key is always written, the two padding keys applying to keep_ratio only
INPUT_CONTRACT_KEYS = {
    # Version of the keys and values of the contract
    "input_contract_version": {"allowed_values": [INPUT_CONTRACT_VERSION], "source": "INPUT_CONTRACT_VERSION"},
    # Channel order of the decoded photo, as cv2.imread decodes it (3 channels of 8 bits, EXIF orientation applied)
    "input_color_order": {"allowed_values": ["BGR"], "source": "the channel order of read_image"},
    # Photo stretched to the square input whatever its aspect ratio, or resized with its aspect ratio kept (the longest
    # side to the input size, the other one to round(side * size / longest side), half to even) and padded to the square
    "input_resize": {"allowed_values": ["stretch", "keep_ratio"], "source": "Data.image_settings.keep_ratio"},
    # Side of the square network input in pixels
    "input_size": {"allowed_values": None, "source": "Data.image_settings.size"},
    # OpenCV bilinear resize, cv2.resize with cv2.INTER_LINEAR, of the uint8 photo before its conversion to float32
    "input_interpolation": {"allowed_values": ["linear"], "source": "the 'interpolation' default of preprocess_image"},
    # Value of every channel of the padding pixels of the keep_ratio resize
    "input_padding_value": {"allowed_values": ["0"], "source": "the 'padding_value' default of preprocess_image"},
    # Resized photo centred in the square input, at the top row (size - resized height) // 2 and the left column
    # (size - resized width) // 2, an odd padding pixel going to the bottom or the right
    "input_padding_placement": {"allowed_values": ["centred"], "source": "compute_letterbox_region"},
    # Dimensions of every input: batch, channels, height and width, with the frames after the batch for the sequences
    "input_layout": {"allowed_values": ["NCHW", "NTCHW"], "source": "the network inputs of the export specification"},
    # Pixel values from 0 to 255, the uint8 photo cast to float32 without scaling, the normalization being in the graph
    "input_value_range": {"allowed_values": ["0-255"], "source": "the pixel type of read_image"},
}

# Contract value of the OpenCV interpolations the contract describes
INTERPOLATION_VALUES = {cv2.INTER_LINEAR: "linear"}

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
    on either axis, an odd padding telling a centre rounded down from one rounded up, and the landscape and portrait
    camera frames of 1920 x 1080 pixels, resized.

    Args:
        image_size (int): The side of the square network input.

    Returns:
        str: 'centred', or the description of the first region placed otherwise.
    """
    probe_shapes = [(image_size - 1, image_size), (image_size, image_size - 1), (1080, 1920), (1920, 1080)]
    for image_height, image_width in probe_shapes:
        letterbox_region = compute_letterbox_region(image_height=image_height, image_width=image_width,
                                                    image_size=image_size)

        # Region of the contract: one scale for both sides, then the resized photo centred, rounded down
        scale = image_size / max(image_height, image_width)
        resized_height, resized_width = int(round(image_height * scale)), int(round(image_width * scale))
        centred_region = ((image_size - resized_height) // 2, (image_size - resized_width) // 2, resized_height,
                          resized_width)
        if tuple(letterbox_region) != centred_region:
            return (f"region (top, left, height, width) {tuple(letterbox_region)} of a {image_height} x {image_width} "
                    f"photo instead of {centred_region}")
    return "centred"


def find_input_layout(specification: Dict[str, Any], image_size: int) -> str:
    """
    Finds the layout of the network inputs from the export specification: images of [3, S, S] after the batch
    dimension (NCHW), or sequences of such frames (NTCHW) for the sequence tasks, both inputs of the comparison model
    sharing it. S must be the configured input size, so that 'input_size' is the side the exported graph accepts.

    Args:
        specification (Dict[str, Any]): The export specification of the task, from get_export_specification.
        image_size (int): The configured side of the square network input, 'Data.image_settings.size'.

    Returns:
        str: 'NCHW', 'NTCHW', or the description of inputs of another layout.
    """
    # Dimensions after the batch one of every layout, the number of frames of a sequence being dynamic (-1)
    layout_dimensions = {"NCHW": [3, image_size, image_size], "NTCHW": [-1, 3, image_size, image_size]}
    input_layouts = [next((layout for layout, dimensions in layout_dimensions.items()
                           if dimensions == input_specification["dimensions"]), None)
                     for input_specification in specification["inputs"]]
    if None in input_layouts or len(set(input_layouts)) != 1:
        return (", ".join(f"input '{input_specification['name']}' of dimensions {input_specification['dimensions']}"
                          for input_specification in specification["inputs"]) +
                f" for the size {image_size}")
    return input_layouts[0]


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

    # Every value among the allowed ones, or a positive integer written in decimal for the input size
    for key, key_definition in INPUT_CONTRACT_KEYS.items():
        if key not in input_contract:
            continue
        value, allowed_values = input_contract[key], key_definition["allowed_values"]
        if allowed_values is None:
            is_allowed = isinstance(value, str) and value.isdecimal() and value == str(int(value)) and int(value) > 0
            allowed_text = "a positive integer"
        else:
            is_allowed = value in allowed_values
            allowed_text = " or ".join(f"'{allowed_value}'" for allowed_value in allowed_values)
        if not is_allowed:
            errors.append(f"{key} {value!r}, from {key_definition['source']}, is not allowed (allowed: {allowed_text})")

    if errors:
        raise ValueError(f"The input contract version {INPUT_CONTRACT_VERSION} cannot describe this input:\n" +
                         "\n".join(errors))


def build_input_contract(task: str, model_configuration: Dict[str, Any],
                         specification: Dict[str, Any]) -> Dict[str, str]:
    """
    Builds the input contract of the models exported from a run, called by export_experiment before any file is
    written: the resize and size from the image settings of the run, the interpolation and padding value from the
    defaults of preprocess_image, the placement checked on compute_letterbox_region, the channel order and value range
    from read_image, and the layout from the export specification. The contract is checked by check_input_contract, so
    that a run the contract cannot describe stops the export with the name of the setting.

    Args:
        task (str): The task of the run.
        model_configuration (Dict[str, Any]): The model configuration of the run, holding 'Data.image_settings' (and
            'Data.tiling' for the semantic segmentation).
        specification (Dict[str, Any]): The export specification of the task, from get_export_specification.

    Returns:
        Dict[str, str]: The value of every key of INPUT_CONTRACT_KEYS, in its order.

    Raises:
        ValueError: If the segmenter is trained on tiles ('Data.tiling.activated'), or if a value of the run is not
            allowed by the contract (check_input_contract).
    """
    image_settings = model_configuration["Data"]["image_settings"]

    # A segmenter trained on tiles takes tiles cut from the scaled image, which no resize of the contract describes
    if task == "semantic_segmentation" and model_configuration["Data"]["tiling"]["activated"]:
        raise ValueError(f"The input contract version {INPUT_CONTRACT_VERSION} cannot describe this input: "
                         f"'Data.tiling.activated' is True, the segmenter taking tiles cut from the image scaled by "
                         f"'Data.tiling.scale' instead of the whole resized image.")

    # Interpolation and padding value of the resize, channel order and value range of the decoded photos
    preprocessing_defaults = find_preprocessing_defaults()
    interpolation = preprocessing_defaults["interpolation"]
    color_order, value_range = find_decoded_photo_description()

    input_contract = {
        "input_contract_version": INPUT_CONTRACT_VERSION, "input_color_order": color_order,
        "input_resize": "keep_ratio" if image_settings["keep_ratio"] else "stretch",
        "input_size": str(image_settings["size"]),
        "input_interpolation": (INTERPOLATION_VALUES[interpolation] if interpolation in INTERPOLATION_VALUES else
                                f"OpenCV interpolation {interpolation}"),
        "input_padding_value": str(preprocessing_defaults["padding_value"]),
        "input_padding_placement": find_padding_placement(image_size=image_settings["size"]),
        "input_layout": find_input_layout(specification=specification, image_size=image_settings["size"]),
        "input_value_range": value_range}
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
