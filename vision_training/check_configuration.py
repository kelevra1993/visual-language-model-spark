"""
Command line check of configuration files, without training, for the pipelines that vet a configuration before they
launch a training with it (a continuous integration job, the step before a training pod): every file given is loaded
and checked as a training loads it (prepare_experiment_configuration), its problems printed in the red list a training
prints before it stops, or the name of the run folder it would train in (run_name.py) printed in green. With
--check_paths, the data folders, the label mapping file and the pretrained runs a new training reads must exist too,
the experiment folder being created by the training. Exits with 0 when every file is valid, 1 otherwise.
example :
uv run python -m vision_training.check_configuration \
    --configuration_path="vision_training/configurations/classification/configuration_synthetic_shapes.yaml" \
    --check_paths
"""

import argparse
import sys

from pathlib import Path
from typing import Any, Dict, List, Tuple

from vision_training.configuration_processing.common_checks import PRETRAINED_COMPONENT_NAMES
from vision_training.configuration_processing.configuration_loader import (prepare_experiment_configuration,
                                                                         print_configuration_problems)
from vision_training.configuration_processing.configuration_values import ConfigurationError
from vision_training.configuration_processing.run_name import build_run_name
from vision_training.data.data_record import DATA_RECORD_SPLITS
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.utilities.os_utilities import format_path_link, print_color

# Errors of the loading itself, raised before the checks run and reported as the problem of their file, so that the
# next files are still checked: a missing configuration or label mapping file, a YAML or JSON file that cannot be
# parsed, a missing 'ExperimentConfiguration' section, or a folder, count or rate the loader cannot convert (e.g. null)
LOADING_ERRORS = (FileNotFoundError, KeyError, TypeError, ValueError)

# Folders the loaders read next to the 'labels.json' of every split: the images or frames of every task, and the class
# masks of the semantic segmentation
SPLIT_IMAGE_FOLDER_NAME = "data"
SPLIT_MASK_FOLDER_NAME = "masks"


def parse_command_line() -> Tuple[List[Path], bool]:
    """
    Reads the configuration files to check from the command line (--configuration_path, repeated for several files,
    absolute or relative to the current folder) and whether the paths they name must exist (--check_paths), so that a
    pipeline checks any file before it launches a training with it.

    Returns:
        Tuple[List[Path], bool]: The absolute path of every configuration file, in the order given, and whether the
            paths are checked.

    Raises:
        SystemExit: With code 2 if no configuration file is given, after printing the usage.
    """
    parser = argparse.ArgumentParser(description="Checks configuration files without training: exits with 0 when every "
                                                 "file is valid, 1 otherwise.")
    parser.add_argument("--configuration_path", type=Path, action="append", required=True,
                        help="Path of a configuration file, absolute or relative to the current folder, e.g. "
                             "vision_training/configurations/classification/configuration_synthetic_shapes.yaml; "
                             "repeat the option to check several files.")
    parser.add_argument("--check_paths", action="store_true",
                        help="Also require the paths a new training reads: the training, validation and test folders "
                             "with their labels.json and data folder (and masks folder for the semantic "
                             "segmentation), the label mapping file of the multi_class mode, and the run folder and "
                             "checkpoint of every pretrained section. The experiment folder may not exist yet.")
    arguments = parser.parse_args()
    return ([configuration_path.expanduser().absolute() for configuration_path in arguments.configuration_path],
            arguments.check_paths)


def find_missing_split_paths(experiment_configuration: Dict[str, Any], task: str) -> List[str]:
    """
    Lists what the training, validation and test splits of a configuration lack for a training to read them, with
    --check_paths: every training reads the three splits (DATA_RECORD_SPLITS, recorded in its 'data.json'), each split
    folder holding its 'labels.json' and the 'data' folder of its images or frames, and the 'masks' folder of its class
    masks for the semantic segmentation, as the loaders of vision_training/data/loaders.py read them.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration, its split folders turned
            into paths by the loader.
        task (str): The task of the configuration, giving the folders of a split.

    Returns:
        List[str]: The description of every missing key, folder or file.
    """
    required_folder_names = [SPLIT_IMAGE_FOLDER_NAME]
    if task == "semantic_segmentation":
        required_folder_names.append(SPLIT_MASK_FOLDER_NAME)

    errors = []
    for split_key in DATA_RECORD_SPLITS.values():
        # The checks of the loading never read the split folders, so a missing key is reported here
        key_path = f"ExperimentConfiguration.{split_key}"
        if split_key not in experiment_configuration:
            errors.append(f"Missing configuration key '{key_path}'.")
            continue
        split_folder = experiment_configuration[split_key]
        if not split_folder.is_dir():
            errors.append(f"'{key_path}' is not a folder : {format_path_link(path=split_folder)}")
            continue

        # The manifest of the split, then the folders of its images and masks
        if not (split_folder / "labels.json").is_file():
            errors.append(f"'{key_path}' holds no 'labels.json' : {format_path_link(path=split_folder)}")
        for folder_name in required_folder_names:
            if not (split_folder / folder_name).is_dir():
                errors.append(f"'{key_path}' holds no '{folder_name}' folder : {format_path_link(path=split_folder)}")
    return errors


def find_missing_label_mapping_file(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Tells whether the label mapping file of the 'multi_class' mode exists, with --check_paths: a new training builds its
    classes from it, while the loading reads it only when the configuration is not the copy of a run folder holding
    its own 'label_dictionary.json'. The 'binary' mode reads no file.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with its 'Data.label_mapping'.

    Returns:
        List[str]: A single description when the file is missing, otherwise an empty list.
    """
    label_mapping_configuration = model_configuration["Data"]["label_mapping"]
    if label_mapping_configuration["mode"] != "multi_class":
        return []

    # The file named by the configuration, which a run folder holding its classes may no longer have
    mapping_file_path = label_mapping_configuration["mapping_file_path"]
    if mapping_file_path is None:
        return ["'Data.label_mapping.mapping_file_path' must name the mapping file of the 'multi_class' mode, got "
                "None."]
    if not Path(mapping_file_path).is_file():
        return [f"'Data.label_mapping.mapping_file_path' is not a file : {format_path_link(path=mapping_file_path)}"]
    return []


def find_missing_pretrained_runs(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists what the source runs of the transferred components lack for a new training to load them, with --check_paths,
    for every section of the configuration whose 'pretrained.experiment_folder' is set (the sections of
    PRETRAINED_COMPONENT_NAMES, as BaseTrainer.initialise_pretrained_components reads them): the run folder, its
    'training_configuration.yaml' compared with the new run by the transfer, and the checkpoint of
    'pretrained.iteration', found as the transfer finds it (get_checkpoint_path: the iteration given, or when it is null
    the last one of the registry of the source run, else its newest complete checkpoint folder). A run resuming from its
    own checkpoint does not load them again, but whether the run will resume is not known here, so every source run is
    required.

    Args:
        model_configuration (Dict[str, Any]): The processed model configuration, with the 'pretrained' settings of its
            sections, already checked.

    Returns:
        List[str]: The description of every missing folder, file or checkpoint.
    """
    errors = []
    for section_name in PRETRAINED_COMPONENT_NAMES:
        # Sections the configuration holds, a component trained from random weights reading no other run
        if section_name not in model_configuration:
            continue
        pretrained_configuration = model_configuration[section_name]["pretrained"]
        if pretrained_configuration["experiment_folder"] is None:
            continue

        # The source run folder, then the configuration the transfer compares with the new run
        key_path = f"{section_name}.pretrained.experiment_folder"
        source_folder = Path(pretrained_configuration["experiment_folder"])
        if not source_folder.is_dir():
            errors.append(f"'{key_path}' is not a folder : {format_path_link(path=source_folder)}")
            continue
        if not (source_folder / "training_configuration.yaml").is_file():
            errors.append(f"'{key_path}' holds no 'training_configuration.yaml' : "
                          f"{format_path_link(path=source_folder)}")

        # The checkpoint of the requested iteration, or the last one when it is null
        iteration = pretrained_configuration["iteration"]
        try:
            get_checkpoint_path(experiment_folder=source_folder, iteration=iteration)
        except FileNotFoundError as missing_checkpoint_error:
            iteration_description = "null, the last checkpoint" if iteration is None else str(iteration)
            errors.append(f"'{section_name}.pretrained.iteration' ({iteration_description}): "
                          f"{missing_checkpoint_error}")
    return errors


def check_configuration_file(configuration_path: Path, check_paths: bool) -> bool:
    """
    Checks one configuration file of the command line as a training loads it, then, with --check_paths, the paths a new
    training reads, those only once the settings are valid, and prints the outcome: the problems in the red list the
    trainings print (print_configuration_problems), or the run name the file would train under in green.

    Args:
        configuration_path (Path): The absolute path of the configuration file.
        check_paths (bool): Whether the data folders, the label mapping file and the pretrained runs must exist.

    Returns:
        bool: Whether the file is valid.
    """
    # Load and check the file as a training does, an error of the loading itself being listed like the problems of the
    # checks
    try:
        experiment_configuration, model_configuration = prepare_experiment_configuration(
            configuration_path=configuration_path)
    except ConfigurationError as configuration_error:
        print_configuration_problems(configuration_path=configuration_path, errors=configuration_error.errors)
        return False
    except LOADING_ERRORS as loading_error:
        # One line per error, a YAML parsing error spanning several lines otherwise reading as several problems
        loading_description = " ".join(f"{type(loading_error).__name__}: {loading_error}".split())
        print_configuration_problems(configuration_path=configuration_path, errors=[loading_description])
        return False

    # Paths a new training reads, the experiment folder being created by the training itself
    if check_paths:
        path_errors = find_missing_split_paths(experiment_configuration=experiment_configuration,
                                               task=experiment_configuration["task"])
        path_errors += find_missing_label_mapping_file(model_configuration=model_configuration)
        path_errors += find_missing_pretrained_runs(model_configuration=model_configuration)
        if path_errors:
            print_configuration_problems(configuration_path=configuration_path, errors=path_errors)
            return False

    # Name of the run folder the file would train in, inside its experiment folder
    run_name = build_run_name(experiment_configuration=experiment_configuration,
                              model_configuration=model_configuration)
    print_color(color="green", output=f"Valid Configuration File : {format_path_link(path=configuration_path)} | "
                                      f"Run Name : {run_name}")
    return True


def main() -> int:
    """
    Entry point of the command line check of configuration files, for the pipelines that vet a file before a training:
    every file of the command line is checked, the next files still being checked after an invalid one, so that one
    call reports every problem.

    Returns:
        int: The exit code, 0 when every file is valid, 1 otherwise.
    """
    configuration_paths, check_paths = parse_command_line()
    valid_files = [check_configuration_file(configuration_path=configuration_path, check_paths=check_paths)
                   for configuration_path in configuration_paths]
    return 0 if all(valid_files) else 1


if __name__ == "__main__":
    sys.exit(main())
