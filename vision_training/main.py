"""
Entry point of the training pipeline: loads an experiment configuration file and runs the training loop of the
trainer of its task, found in the task registry.
"""

from pathlib import Path
from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.core.task_registry import get_trainer_class


def main() -> None:
    """
    Entry point of the training pipeline, for every task.
    This function loads the user-specified experiment and model configurations from the YAML file, and initializes
    the trainer of the task of the configuration ('ExperimentConfiguration.task') to begin the model training.
    """
    # Define configuration path
    experiment_configuration_path = (Path(__file__).parent / "configurations" / "classification" /
                                     "configuration_large_model.yaml")

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
