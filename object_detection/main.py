from pathlib import Path
from utilities.os_utilities import load_experiment_configuration, print_yellow
from utilities.tensor_utilities import get_device
from trainer.trainer import Trainer


def main():
    # Define configuration path
    # Todo will be put back after all the code has been written.
    # experiment_configuration_path = Path(__file__).parent / "configurations" / f"{get_device()}_configuration.yaml"
    experiment_configuration_path = Path(__file__).parent / "configurations" / f"configuration_example.yaml"

    # Load configuration
    experiment_configuration, model_configuration = load_experiment_configuration(experiment_configuration_path)

    # Initialize Trainer
    trainer = Trainer(
        experiment_folder=experiment_configuration["experiment_folder"],
        model_configuration=model_configuration,
        data_folder=experiment_configuration["data_folder"],
        train_split_file=experiment_configuration["train_split_file"],
        validation_split_file=experiment_configuration["validation_split_file"],
        test_split_file=experiment_configuration["test_split_file"],
        number_iterations=experiment_configuration["number_iterations"],
        weight_saving_iterations=experiment_configuration["weight_saving_iterations"],
        compute_validation_iteration=experiment_configuration["compute_validation_iteration"],
        learning_rate=experiment_configuration["learning_rate"],
        dtype=experiment_configuration["dtype"],
        information_dump=experiment_configuration["information_dump"],
        resume_training=experiment_configuration["resume_training"])


if __name__ == "__main__":
    main()
