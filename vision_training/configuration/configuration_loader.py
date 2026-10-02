import sys
import torch
import yaml

from pathlib import Path
from typing import Any, Dict, Tuple

from vision_training.configuration.configuration_checker import check_configuration
from vision_training.configuration.configuration_values import ConfigurationError
from vision_training.data.label_mapping import build_label_dictionary
from vision_training.utilities.os_utilities import print_color


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

    # Convert the optimization numbers that YAML reads as text when written without a dot (e.g. 1e-4 or 1e5), the
    # values of other types being left for the configuration checker to report
    if "Optimization" in model_configuration:
        optimization_configuration = model_configuration["Optimization"]
        for key in ["weight_decay", "minimum_learning_rate_ratio", "gradient_clipping_norm"]:
            if key in optimization_configuration and isinstance(optimization_configuration[key], str):
                optimization_configuration[key] = float(optimization_configuration[key])
        for key in ["warmup_iterations", "decay_iterations"]:
            if key in optimization_configuration and isinstance(optimization_configuration[key], (str, float)):
                optimization_configuration[key] = int(float(optimization_configuration[key]))

    # Process label mapping dynamically to inject label_dictionary
    if "Data" in model_configuration and "label_mapping" in model_configuration["Data"]:
        label_mapping = model_configuration["Data"]["label_mapping"]
        model_configuration["Data"]["label_mode"] = label_mapping["mode"]

        # The dictionary depends on the task; a missing task, an unknown mode or a mode the task does not support
        # give no dictionary, the configuration checker then reports the problem
        if "task" in experiment_configuration:
            label_dictionary = build_label_dictionary(label_mapping_configuration=label_mapping,
                                                      task=experiment_configuration["task"])
            if label_dictionary is not None:
                model_configuration["Data"]["label_dictionary"] = label_dictionary

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
