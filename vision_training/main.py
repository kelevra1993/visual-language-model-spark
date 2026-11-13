"""
Entry point of the training pipeline: reads the path of an experiment configuration file from the command line (the
command above trains the large classification model), loads that file and runs the training loop of the trainer of
its task ('ExperimentConfiguration.task'), found in the task registry.
example :
uv run python -m vision_training.main \
    --configuration_path="vision_training/configurations/classification/configuration_large_model.yaml"
"""

import argparse

from pathlib import Path
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.core.task_registry import get_trainer_class
from vision_training.data.manifest_cache import read_manifests_once
from vision_training.utilities.os_utilities import enable_expandable_cuda_segments, format_path_link


def parse_configuration_path() -> Path:
    """
    Reads the configuration file of a training launch from the command line, given by its path (--configuration_path,
    absolute or relative to the current folder), so that a series of runs (e.g. the launches of a hyperparameter
    sweep, one configuration file each) is started without editing this file, whatever folder holds the files. The
    task trained is the one of the configuration file itself.

    Returns:
        Path: The absolute path of the configuration file.

    Raises:
        SystemExit: If the argument is missing or the path is not a file, after printing the usage.
    """
    parser = argparse.ArgumentParser(description="Trains a model from an experiment configuration file.")
    parser.add_argument("--configuration_path", type=Path, required=True,
                        help="Path of the configuration file, absolute or relative to the current folder, e.g. "
                             "vision_training/configurations/classification/configuration_large_model.yaml.")
    arguments = parser.parse_args()

    # Stop before anything starts when the path does not lead to a file
    configuration_path = arguments.configuration_path.expanduser().absolute()
    if not configuration_path.is_file():
        parser.error(message=f"No configuration file at {format_path_link(path=configuration_path)}")
    return configuration_path


def main() -> None:
    """
    Entry point of the training pipeline, for every task.
    This function loads the experiment and model configurations of the YAML file given on the command line, and
    initializes the trainer of the task of the configuration ('ExperimentConfiguration.task') to begin the model
    training. The expandable segments of the CUDA allocator are turned on first, before anything reaches the GPU.
    """
    # Keep the GPU memory flat over the batches of changing shapes, before the first CUDA call
    enable_expandable_cuda_segments()

    # Configuration file given on the command line
    experiment_configuration_path = parse_configuration_path()

    # Load configuration and build the trainer of its task, every 'labels.json' parsed once for all its readers and
    # freed before the training starts
    with read_manifests_once():
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
