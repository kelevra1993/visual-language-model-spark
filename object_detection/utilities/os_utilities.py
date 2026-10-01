import os
import sys
import json
from json import JSONDecodeError

import torch
import numpy as np
import yaml

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import inspect

from utilities.model.configuration_checker import check_configuration, ConfigurationError

# No printing of scientific notations
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000, threshold=np.inf)
torch.set_printoptions(linewidth=1000)


def read_json(path: str) -> dict[str, Any] | list[Any]:
    """
    Reads and parses a JSON file.

    This function is used throughout the project to load records, manifests,
    and other metadata stored in JSON format.

    Args:
        path (str): The file path to the .json file.

    Returns:
        Dict[str, Any] | List[Any]: The parsed JSON data.
    """
    json_data = None
    try:
        with open(path, "r") as file:
            json_data = json.loads(file.read())
        return json_data
    except JSONDecodeError:
        return json_data


def load_configuration(configuration_path: str | Path) -> Dict[str, Any]:
    """
    Parses a YAML configuration file into a dictionary.

    Args:
        configuration_path (str | Path): The file path to the YAML configuration.

    Returns:
        Dict[str, Any]: The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    path = Path(configuration_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as file:
        try:
            configuration = yaml.safe_load(file)
            return configuration if configuration is not None else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file at {path}:\n{e}")


def load_experiment_configuration(configuration_path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads a single YAML configuration file and splits it into experiment
    and model configurations. Archiving the yaml into the experiment folder is handled by the Trainer.

    This function extracts the 'ExperimentConfiguration' section, processes
    its paths and types, and returns it alongside the rest of the configuration
    parameters.

    Args:
        configuration_path (str | Path): Path to the consolidated YAML file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - experiment_configuration: The processed experiment settings.
            - model_configuration: The remaining configuration (all other keys).

    Raises:
        KeyError: If 'ExperimentConfiguration' is missing from the file.
        SystemExit: If the configuration is incomplete or inconsistent (see 'check_configuration'), after printing
            each problem in red.
    """
    configuration = load_configuration(configuration_path)
    if "ExperimentConfiguration" not in configuration:
        raise KeyError(f"Key 'ExperimentConfiguration' not found in {configuration_path}")

    experiment_configuration = configuration.pop("ExperimentConfiguration")
    model_configuration = configuration

    # Convert paths
    path_keys = ["experiment_folder", "train_data_folder", "validation_data_folder", "test_data_folder"]
    for key in path_keys:
        if key in experiment_configuration:
            experiment_configuration[key] = Path(experiment_configuration[key])

    # Convert numerics
    int_keys = ["information_dump", "weight_saving_iterations", "intermediate_weight_saving_iterations",
                "number_iterations"]
    for key in int_keys:
        if key in experiment_configuration:
            experiment_configuration[key] = int(float(experiment_configuration[key]))

    if "learning_rate" in experiment_configuration:
        experiment_configuration["learning_rate"] = float(experiment_configuration["learning_rate"])

    # Convert dtype
    if "dtype" in experiment_configuration:
        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        experiment_configuration["dtype"] = dtype_map[experiment_configuration["dtype"]]

    # Convert beta values from fractions to floats
    for section_key in ["RegionProposal", "Detector"]:
        if section_key in model_configuration:
            beta_value = model_configuration[section_key]["localisation_loss_beta"]
            if isinstance(beta_value, str) and "/" in beta_value:
                numerator, denominator = beta_value.split("/")
                model_configuration[section_key]["localisation_loss_beta"] = float(numerator) / float(denominator)
            elif beta_value is not None:
                model_configuration[section_key]["localisation_loss_beta"] = float(beta_value)

    # Process label mapping dynamically to inject label_dictionary
    if "Data" in model_configuration and "label_mapping" in model_configuration["Data"]:
        label_mapping = model_configuration["Data"]["label_mapping"]
        mode = label_mapping["mode"]
        model_configuration["Data"]["label_mode"] = mode

        if mode == "binary":
            model_configuration["Data"]["label_dictionary"] = {"background": 0, "foreground": 1}
        elif mode == "multi_class":
            mapping_path = label_mapping["mapping_file_path"]
            if mapping_path and os.path.exists(mapping_path):
                with open(mapping_path, "r") as file:
                    raw_mapping = json.load(file)
                    # Convert string keys to int if they are digits for easier mapping during data loading
                    parsed_mapping = {int(k) if str(k).isdigit() else k: v for k, v in raw_mapping.items()}
                    # Ensure background is accounted for
                    if "background" not in parsed_mapping and 0 not in parsed_mapping.values():
                        parsed_mapping["background"] = 0
                    model_configuration["Data"]["label_dictionary"] = parsed_mapping
            else:
                raise FileNotFoundError(f"Label mapping file not found at: {mapping_path}")

    # Reject incomplete or inconsistent configurations before any model or data loader is built,
    # listing each problem as a short red message rather than a full traceback
    try:
        check_configuration(experiment_configuration=experiment_configuration, model_configuration=model_configuration)
    except ConfigurationError as configuration_error:
        print_color(color="red",
                    output=f"Invalid Configuration File : file://{Path(configuration_path).absolute()}",
                    add_separators=True)
        for error in configuration_error.errors:
            print_color(color="red", output=error, indent=1)
        sys.exit(1)

    return experiment_configuration, model_configuration


# Run name code of every feature map normalization strategy
NORMALIZATION_CODES = {"batch": "BN", "layer": "LN", "none": "NN"}


def join_level_tokens(level_tokens: List[str]) -> str:
    """
    Joins the per-level descriptions of a head type (region proposal heads, detector heads, anchor counts) for the
    run name: written once when every detection level uses the same settings, otherwise listed in level order and
    joined with '+'.

    Args:
        level_tokens (List[str]): The description of every detection level, in level order.

    Returns:
        str: The joined description.
    """
    return level_tokens[0] if len(set(level_tokens)) == 1 else "+".join(level_tokens)


def build_run_name(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> str:
    """
    Builds the name of the folder a training run is saved in, from the settings that define the run, so that a run
    always trains in a folder matching its configuration: the same settings resume in the same folder, and changed
    settings, in particular every setting changing the model size, start a new one instead of mixing with a folder
    whose saved configuration no longer matches.

    The name is made of '_' separated sections of '-' separated tokens, a flag token being left out when its setting
    is off, e.g. 'IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_
    DSE-DRES-DBN-D7.2x64.256.128_BIN_AUG-BS40-LR2e-4' (without the line break):
    - input: IN (input normalization), PSN (per-sample normalization), KAR (keep aspect ratio), the input size;
    - backbone: modules (SE, RES, INC<starting block>), feature map normalization (BN, LN or NN), then
      B<layers>x<filters> of every block;
    - L<blocks feeding the detection heads>;
    - anchors: A<anchors per position>.<smallest scale>-<largest scale>;
    - region proposal heads: R<normalization>, then R<layers>x<filters>;
    - detector heads: modules (DSE, DRES, DINC), D<normalization>, then D<RoIAlign pool size>.<layers>x<filters>
      .<fully connected sizes>;
    - classes: BIN or MC<foreground classes>;
    - training: AUG, BS<batch size>, LR<learning rate>.
    Per-level settings are written once when identical on every level, otherwise joined with '+' in level order.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model configuration, with its label dictionary.

    Returns:
        str: The run name.
    """
    data_configuration = model_configuration["Data"]
    backbone_configuration = model_configuration["Backbone"]
    region_proposal_configuration = model_configuration["RegionProposal"]
    detector_configuration = model_configuration["Detector"]
    level_names = [str(index) for index in backbone_configuration["detection_convolution_block_indices"]]

    # Input: normalization of the pixel values, resizing mode and input size
    input_tokens = []
    if backbone_configuration["normalization"]["input_normalization"]:
        input_tokens.append("IN")
    if backbone_configuration["normalization"]["per_sample_normalization"]:
        input_tokens.append("PSN")
    if data_configuration["image_settings"]["keep_ratio"]:
        input_tokens.append("KAR")
    input_tokens.append(str(data_configuration["image_settings"]["size"]))

    # Backbone: modules, feature map normalization, then the layers and filters of every block in order
    backbone_modules = backbone_configuration["modules"]
    backbone_tokens = []
    if backbone_modules["squeeze_and_excite"]:
        backbone_tokens.append("SE")
    if backbone_modules["residual_nets"]:
        backbone_tokens.append("RES")
    if backbone_modules["inception"]:
        backbone_tokens.append(f"INC{backbone_modules['inception_starting_block']}")
    backbone_tokens.append(NORMALIZATION_CODES[
        str(backbone_configuration["normalization"]["feature_map_normalization"]).lower().strip()])
    convolutions = backbone_configuration["convolutions"]
    backbone_tokens.append("B" + ".".join(f"{convolutions[block][0]}x{convolutions[block][1]}"
                                          for block in sorted(convolutions, key=int)))

    # Detection levels, and the number of anchors per position (scales x ratios) with the range of anchor scales
    levels_token = "L" + "".join(level_names)
    scales_and_ratios = model_configuration["Anchors"]["scales_and_ratios"]
    anchor_counts = [str(len(scales_and_ratios[level]["scales"]) * len(scales_and_ratios[level]["aspect_ratios"]))
                     for level in level_names]
    anchor_scales = [scale for level in level_names for scale in scales_and_ratios[level]["scales"]]
    anchors_token = f"A{join_level_tokens(level_tokens=anchor_counts)}.{min(anchor_scales)}-{max(anchor_scales)}"

    # Region proposal heads: normalization, then the layers and filters of the head of every level
    region_proposal_heads = region_proposal_configuration["head_configurations"]
    region_proposal_tokens = [
        "R" + NORMALIZATION_CODES[
            str(region_proposal_configuration["normalization"]["feature_map_normalization"]).lower().strip()],
        "R" + join_level_tokens(level_tokens=[f"{region_proposal_heads[level][0]}x{region_proposal_heads[level][1]}"
                                              for level in level_names])]

    # Detector heads: modules, normalization, then the RoIAlign pool size, convolutions and fully connected layers
    detector_modules = detector_configuration["modules"]
    detector_tokens = []
    if detector_modules["squeeze_and_excite"]:
        detector_tokens.append("DSE")
    if detector_modules["residual_nets"]:
        detector_tokens.append("DRES")
    if detector_modules["inception"]:
        detector_tokens.append("DINC")
    detector_tokens.append("D" + NORMALIZATION_CODES[
        str(detector_configuration["normalization"]["feature_map_normalization"]).lower().strip()])
    detector_heads = detector_configuration["head_configurations"]
    detector_tokens.append("D" + join_level_tokens(level_tokens=[
        f"{detector_heads[level]['roi_align_pool_size']}.{detector_heads[level]['convolutions'][0]}x"
        f"{detector_heads[level]['convolutions'][1]}." + ".".join(str(size) for size in
                                                                   detector_heads[level]["fully_connected"])
        for level in level_names]))

    # Classes, the label dictionary including the background class
    if data_configuration["label_mode"] == "binary":
        classes_token = "BIN"
    else:
        classes_token = f"MC{len(data_configuration['label_dictionary']) - 1}"

    # Training: augmentation, batch size and learning rate in short scientific notation (e.g. 2e-4, 1.5e-3)
    training_tokens = ["AUG"] if data_configuration["data_augmentation"]["activated"] else []
    training_tokens.append(f"BS{experiment_configuration['batch_size']}")
    mantissa, exponent = f"{experiment_configuration['learning_rate']:e}".split("e")
    training_tokens.append(f"LR{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}")

    sections = [input_tokens, backbone_tokens, [levels_token], [anchors_token], region_proposal_tokens,
                detector_tokens, [classes_token], training_tokens]
    return "_".join("-".join(section) for section in sections)


# ANSI escape codes of the console colours available to print_color
COLOR_CODES = {"blue": "\033[94m", "green": "\033[32m", "yellow": "\033[93m", "red": "\033[91m"}


def print_color(color: str, output: str, add_separators: bool = False, indent: int = 0) -> None:
    """
    Prints a string to the console in a bold colour, the single console output helper of the project.

    The colour carries the meaning of the message across the pipeline: blue for information and progress, green
    for successful operations (saved weights, completed steps), yellow for warnings that need attention without
    stopping the run, and red for errors and critical failures.

    Args:
        color (str): The colour of the message, one of "blue", "green", "yellow" or "red".
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal separators for better visibility.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.

    Raises:
        ValueError: If the colour is not one of the supported colours.
    """
    if color not in COLOR_CODES:
        raise ValueError(f"Unsupported color {color!r}, expected one of {list(COLOR_CODES)}.")
    color_code = COLOR_CODES[color]

    # Prefix every line with the indentation and a bullet when an indentation level is requested
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    indented_output = "\n".join(f"{indentation_string}{line}" for line in output.split("\n"))

    # Print the message in bold colour, optionally framed by separators as long as its longest line
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print(indentation_string + color_code + "\033[1m" + str(length * "-") + "\033[0m")
        print(color_code + "\033[1m" + indented_output + "\033[0m")
        print(indentation_string + color_code + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print(color_code + "\033[1m" + indented_output + "\033[0m")


def print_dictionary(dictionary: Dict[str, Any], indent: int = 4) -> None:
    """
    Prints a dictionary to the console in a formatted JSON-like style.

    This utility is used to display configuration parameters, experiment
    summaries, or manifest data in a readable format during execution.

    Args:
        dictionary (Dict[str, Any]): The dictionary to be printed.
        indent (int): The number of spaces to use for indentation.
    """
    print(json.dumps(dictionary, indent=indent))


def get_random_color() -> Tuple[int, int, int]:
    """
    Generates a random RGB color tuple.
    
    This utility is useful for randomly assigning colors to different bounding boxes
    or classes during visualization, ensuring distinct visual representations.
    
    Args:
        None
        
    Returns:
        Tuple[int, int, int]: A tuple containing random (B, G, R) values from 0 to 255.
    """
    return (int(np.random.randint(low=0, high=256)),
            int(np.random.randint(low=0, high=256)),
            int(np.random.randint(low=0, high=256)))


def get_variable_name(target_variable: Any, namespace: Optional[Dict[str, Any]] = None) -> str:
    """
    Retrieves the string name of a variable by checking for its exact memory identity in a given namespace.
    If no namespace is provided, it automatically inspects the caller's frame (two levels up) to find 
    the original variable name, removing the need for manual typing.
    
    Args:
        target_variable (Any): The variable whose name we want to find.
        namespace (Optional[Dict[str, Any]]): The dictionary mapping variable names to values. 
                                              Defaults to None, triggering automatic frame inspection.
        
    Returns:
        str: The name of the variable as a string, or 'Unknown_Variable' if not found.
    """
    if namespace is None:
        try:
            # We go back 2 frames: 1 frame for this function, 1 frame for the caller (e.g., print_tensor_shape)
            # This allows us to access the locals dictionary of the script/function that actually passed the variable.
            frame = inspect.currentframe().f_back.f_back
            namespace = frame.f_locals
        except Exception:
            namespace = {}

    for name, value in namespace.items():
        # Using 'is' checks for exact memory identity, not just equal value
        if value is target_variable:
            return name

    return "Unknown_Variable"


def get_image_paths(folder: Path) -> List[Path]:
    """
    Lists the images of a folder, e.g. the 'data' folder of a dataset split, to feed them to the deployed detector
    in the inference client.

    Args:
        folder (Path): The folder holding the images, which is not searched recursively.

    Returns:
        List[Path]: The sorted image paths.

    Raises:
        FileNotFoundError: If the folder holds no image, e.g. a split folder given instead of its 'data' subfolder.
    """
    # Image file extensions listed by get_image_paths
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    image_paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    # Fail loudly instead of running on nothing
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image found in : file://{folder.absolute()}")
    return image_paths
