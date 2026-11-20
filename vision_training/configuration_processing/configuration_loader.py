"""
Loading of the YAML configuration files: parses a file, splits it into the experiment and model configurations,
processes the experiment paths and types and runs the configuration checks. Used by main.py, the scripts and the
tests to obtain the configurations given to the trainers, which stop on an invalid file, and by
vision_training/check_configuration.py, which reports the problems of several files without stopping.
"""

import sys
import yaml

from pathlib import Path
from typing import Any, Dict, List, Tuple

from vision_training.configuration_processing.configuration_checker import check_configuration
from vision_training.configuration_processing.configuration_values import ConfigurationError
from vision_training.data.keypoints.keypoint_categories import read_keypoint_categories
from vision_training.data.label_mapping import build_label_dictionary, read_run_label_dictionary
from vision_training.utilities.os_utilities import format_path_link, print_color


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
        except yaml.YAMLError as error:
            raise ValueError(f"Error parsing YAML file at {path}:\n{error}")


def inject_keypoint_categories(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Adds 'Data.keypoint_categories' to the configuration of the keypoint detection, the keypoint names and skeleton of
    every class read from the 'categories' of the training split: they size the heatmap channels of the keypoint
    detector, and the evaluation and viewers name and draw the keypoints. A missing training split or label dictionary
    gives no definitions, the configuration checker then reports the problem.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration, giving the training split.
        model_configuration (Dict[str, Any]): The model configuration, whose 'Data' section holds the injected label
            dictionary; updated in place.
    """
    # Every input of the definitions must be there, otherwise the checker reports the missing one
    if "train_data_folder" not in experiment_configuration or "Data" not in model_configuration:
        return
    data_configuration = model_configuration["Data"]
    if "label_dictionary" not in data_configuration:
        return

    # The definitions are read from the training split only, shared by the validation and test splits
    labels_file = Path(experiment_configuration["train_data_folder"]) / "labels.json"
    if labels_file.is_file():
        data_configuration["keypoint_categories"] = read_keypoint_categories(
            labels_file=labels_file, label_dictionary=data_configuration["label_dictionary"])


def prepare_experiment_configuration(configuration_path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads a single YAML configuration file, splits it into experiment and model configurations and checks them,
    raising the problems instead of stopping the program, so that load_experiment_configuration stops a training on
    them and vision_training/check_configuration.py reports those of every file it is given. Archiving the yaml into
    the experiment folder is handled by the Trainer.

    This function extracts the 'ExperimentConfiguration' section, processes its paths and types, adds the
    'tracking_root', 'run_name' (None) and 'standalone_evaluation_folder' (None) runtime keys read by the trainer, and
    returns it alongside the rest of the configuration parameters. The label dictionary is built from the mapping file,
    or read from 'label_dictionary.json' when the file sits in a run folder holding one (the saved configuration of a
    run, loaded by the evaluation, export and model loading tools).

    Args:
        configuration_path (str | Path): Path to the consolidated YAML file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - experiment_configuration: The processed experiment settings.
            - model_configuration: The remaining configuration (all other keys).

    Raises:
        FileNotFoundError: If the file, or the mapping file of the 'multi_class' label mapping, does not exist.
        ValueError: If the file or the mapping file cannot be parsed, if a number converted here (an iteration count,
            the learning rate, an optimization setting) is a text that is not a number, or if the keypoint categories of
            the training split are inconsistent.
        KeyError: If 'ExperimentConfiguration' is missing from the file.
        TypeError: If a folder or a number converted here has a value the conversion refuses, e.g. null.
        ConfigurationError: If the configuration is incomplete or inconsistent (see 'check_configuration'), listing
            every problem.
    """
    configuration = load_configuration(configuration_path=configuration_path)
    if "ExperimentConfiguration" not in configuration:
        raise KeyError(f"Key 'ExperimentConfiguration' not found in {configuration_path}")

    experiment_configuration = configuration.pop("ExperimentConfiguration")
    model_configuration = configuration

    # Convert paths
    path_keys = ["experiment_folder", "train_data_folder", "validation_data_folder", "test_data_folder"]
    for key in path_keys:
        if key in experiment_configuration:
            experiment_configuration[key] = Path(experiment_configuration[key])

    # Root folder of the MLflow backend of the run, the experiments root folder, kept apart from 'experiment_folder'
    # so that a standalone evaluation writing its files in another folder is still tracked in, and reports, the
    # backend of the evaluated experiment
    if "experiment_folder" in experiment_configuration:
        experiment_configuration["tracking_root"] = experiment_configuration["experiment_folder"]

    # Name of the run folder inside 'experiment_folder', None to name it after the run settings (build_run_name). A
    # standalone evaluation gives the name of the evaluated run folder, so that it finds the run and keeps its name,
    # with the tokens of the settings it turns off (the transfer of pretrained components)
    experiment_configuration["run_name"] = None

    # Folder inside the run folder where a trainer that only evaluates the checkpoints of the run writes, set by
    # scripts/evaluation/evaluate_checkpoint.py ('Checkpoint_Evaluation'): its iteration folders then sit directly in
    # it, '<run>/Checkpoint_Evaluation/Iteration_<n>/', and it saves no weights. None for a training
    experiment_configuration["standalone_evaluation_folder"] = None

    # Convert numerics
    integer_keys = ["information_dump", "weight_saving_iterations", "intermediate_weight_saving_iterations",
                    "number_iterations"]
    for key in integer_keys:
        if key in experiment_configuration:
            experiment_configuration[key] = int(float(experiment_configuration[key]))

    if "learning_rate" in experiment_configuration:
        experiment_configuration["learning_rate"] = float(experiment_configuration["learning_rate"])

    # Convert the optimization numbers that YAML reads as text when written without a dot (e.g. 1e-4 or 1e5), the
    # values of other types being left for the configuration checker to report
    if "Optimization" in model_configuration:
        optimization_configuration = model_configuration["Optimization"]
        for key in ["weight_decay", "minimum_learning_rate_ratio"]:
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
        # give no dictionary, the configuration checker then reports the problem. The configuration of a run folder
        # (evaluation, export, model loading) takes the label dictionary saved with the run, so that its classes keep
        # their meaning when the mapping file is edited or moved
        if "task" in experiment_configuration:
            label_dictionary = read_run_label_dictionary(run_folder=Path(configuration_path).parent)
            if label_dictionary is None:
                label_dictionary = build_label_dictionary(label_mapping_configuration=label_mapping,
                                                          task=experiment_configuration["task"])
            if label_dictionary is not None:
                model_configuration["Data"]["label_dictionary"] = label_dictionary

    # Keypoint definitions of the classes of the keypoint detection, read from the categories of the training split
    if "task" in experiment_configuration and experiment_configuration["task"] == "keypoint_detection":
        inject_keypoint_categories(experiment_configuration=experiment_configuration,
                                   model_configuration=model_configuration)

    # Reject incomplete or inconsistent configurations before any model or data loader is built, with every problem
    check_configuration(experiment_configuration=experiment_configuration, model_configuration=model_configuration)

    return experiment_configuration, model_configuration


def print_configuration_problems(configuration_path: str | Path, errors: List[str]) -> None:
    """
    Prints the problems of an invalid configuration file in red under a link to the file, each as a short message
    rather than a full traceback: the block a training prints before it stops (load_experiment_configuration), also
    printed for every invalid file by vision_training/check_configuration.py.

    Args:
        configuration_path (str | Path): The invalid configuration file.
        errors (List[str]): One short description per problem.
    """
    print_color(color="red", output=f"Invalid Configuration File : {format_path_link(path=configuration_path)}",
                add_separators=True)
    for error in errors:
        print_color(color="red", output=error, indent=1)


def load_experiment_configuration(configuration_path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads a configuration file for the trainings and the tools reading one (main.py, the evaluation, export and model
    loading tools, the scripts and the tests): the experiment and model configurations of
    prepare_experiment_configuration, an incomplete or inconsistent file stopping the program with its problems printed
    in red (print_configuration_problems).

    Args:
        configuration_path (str | Path): Path to the consolidated YAML file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - experiment_configuration: The processed experiment settings.
            - model_configuration: The remaining configuration (all other keys).

    Raises:
        KeyError: If 'ExperimentConfiguration' is missing from the file.
        SystemExit: If the configuration is incomplete or inconsistent (see 'check_configuration'), with code 1 after
            printing each problem in red.
    """
    try:
        return prepare_experiment_configuration(configuration_path=configuration_path)
    except ConfigurationError as configuration_error:
        print_configuration_problems(configuration_path=configuration_path, errors=configuration_error.errors)
        sys.exit(1)
