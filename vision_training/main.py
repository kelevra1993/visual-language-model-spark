"""
Entry point of the training pipeline: reads the task folder and the name of an experiment configuration file from the
command line (e.g. --task="classification" --name="run-1.yaml"), loads that file from
vision_training/configurations/<task>/ and runs the training loop of the trainer of its task, found in the task
registry.
"""

import argparse

from pathlib import Path
from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.core.task_registry import get_trainer_class
from vision_training.utilities.os_utilities import format_path_link

# Folder holding one folder of configuration files per task
CONFIGURATIONS_FOLDER = Path(__file__).parent / "configurations"


def parse_configuration_path() -> Path:
    """
    Reads the configuration file of a training launch from the command line, its task folder in
    vision_training/configurations/ given by --task and its file name by --name, so that a series of runs (e.g. the
    launches of a hyperparameter sweep, one configuration file each) is started without editing this file.

    Returns:
        Path: The path of the configuration file.

    Raises:
        SystemExit: If an argument is missing, the task folder does not exist or the task folder holds no file of that
            name, after printing the usage, or the configuration files of the task folder.
    """
    # Task folders of the configurations, the only values accepted by --task
    task_folders = sorted(folder.name for folder in CONFIGURATIONS_FOLDER.iterdir() if folder.is_dir())
    parser = argparse.ArgumentParser(description="Trains a model from a configuration file of "
                                                 "vision_training/configurations/.")
    parser.add_argument("--task", type=str, required=True, choices=task_folders,
                        help="Folder of the configuration file in vision_training/configurations/, e.g. "
                             "classification.")
    parser.add_argument("--name", type=str, required=True,
                        help="Name of the configuration file in the task folder, e.g. run-1.yaml.")
    arguments = parser.parse_args()

    # Stop with the configuration files of the task folder when the requested one does not exist
    configuration_path = CONFIGURATIONS_FOLDER / arguments.task / arguments.name
    if not configuration_path.is_file():
        available_files = ", ".join(sorted(path.name for path in configuration_path.parent.glob(pattern="*.yaml")))
        parser.error(message=f"No configuration file {arguments.name} in "
                             f"{format_path_link(path=configuration_path.parent)}, available files : {available_files}")
    return configuration_path


def main() -> None:
    """
    Entry point of the training pipeline, for every task.
    This function loads the experiment and model configurations of the YAML file given on the command line, and
    initializes the trainer of the task of the configuration ('ExperimentConfiguration.task') to begin the model
    training.
    """
    # Configuration file given on the command line
    experiment_configuration_path = parse_configuration_path()

    # Load configuration
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=experiment_configuration_path)

    # Initialize the trainer of the task of the configuration
    trainer_class = get_trainer_class(task=experiment_configuration["task"])
    trainer = trainer_class(experiment_configuration=experiment_configuration,
                            model_configuration=model_configuration,
                            configuration_path=experiment_configuration_path)

    # trainer.run_benchmarking_loop(benchmarking_iterations=1000)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()
