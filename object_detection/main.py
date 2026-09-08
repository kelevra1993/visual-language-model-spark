from pathlib import Path
from utilities.os_utilities import load_experiment_configuration, print_yellow
from utilities.tensor_utilities import get_device
from trainer.trainer import Trainer


def main() -> None:
    """
    Entry point for the object detection training pipeline.
    This function orchestrates the setup phase by determining the appropriate hardware device,
    loading the user-specified experiment and model configurations from the YAML file, and 
    initializing the Trainer class to begin the model training or evaluation lifecycle.
    """
    # Define configuration path
    # Todo will be put back after all the code has been written.
    # experiment_configuration_path = Path(__file__).parent / "configurations" / f"{get_device()}_configuration.yaml"
    experiment_configuration_path = Path(__file__).parent / "configurations" / f"configuration_example.yaml"

    # Load configuration
    experiment_configuration, model_configuration = load_experiment_configuration(experiment_configuration_path)

    # Initialize Trainer
    trainer = Trainer(experiment_configuration=experiment_configuration,
                      model_configuration=model_configuration,
                      configuration_path=experiment_configuration_path)


if __name__ == "__main__":
    main()
