import torch

from pathlib import Path
from typing import Dict, Tuple, List, Union, Any
from shutil import copyfile

from model.model import Model
from utilities.tensor_utilities import get_device
from utilities.os_utilities import print_blue, print_yellow


class Trainer:
    """
    Orchestrates the entire training pipeline for the object detection and segmentation models.
    This class handles the initialization of the neural network model, the setup of local storage
    for weights and logs, and manages the iterative optimization process over the provided datasets.
    """

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Initializes the training environment by setting up parameters, devices, models, and optimization tools.
        This constructor prepares the runtime state by extracting necessary settings from the configuration dictionaries,
        allocating the correct device tensors, creating directory structures for artifacts, and instantiating 
        the core neural network model and its optimizer.

        Args:
            experiment_configuration (Dict[str, Any]): Dictionary containing training-specific hyperparameters and paths.
            model_configuration (Dict[str, Any]): Dictionary containing the architectural settings for the neural network.
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Load Training Configuration containing all the training parameters
        self.experiment_configuration = experiment_configuration
        self.model_configuration = model_configuration

        # Get specific configurations
        self.data_configuration = model_configuration.get("Data")

        # Get device and dtype
        self.device = get_device()
        self.dtype = self.experiment_configuration["dtype"]

        print(f"For Training We Will Be Using device: {self.device}")

        # Basic training parameters
        self.experiment_folder = self.experiment_configuration["experiment_folder"]
        self.training_iterations = self.experiment_configuration["number_iterations"]
        self.weight_saving_iterations = self.experiment_configuration["weight_saving_iterations"]
        self.compute_validation_iteration = self.experiment_configuration["compute_validation_iteration"]
        self.information_dump = self.experiment_configuration["information_dump"]
        self.learning_rate = self.experiment_configuration["learning_rate"]

        self.input_image_size = self.data_configuration.get("image_settings").get("size")
        self.batch_size = self.experiment_configuration["batch_size"]
        self.compute_validation_iteration = self.experiment_configuration["compute_validation_iteration"]
        self.resume_training = self.experiment_configuration["resume_training"]

        # Setup paths and tensorboard
        self.tensorboard_directory, self.weights_directory, self.configuration_path = self.setup_training_paths(
            initial_configuration_path=Path(configuration_path))

        # Initialize Model and Optimizer
        self.model = Model(configuration=self.model_configuration, device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)

        # Setting Up The Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Print experiment information to user so that they can know everything about the experiment
        # as well as input and output shapes of the model.
        self.print_experiment_information()
        self.model.print_summary(expected_image_size=(self.input_image_size,
                                                      self.input_image_size))

    def setup_training_paths(self, initial_configuration_path: Path) -> tuple[Path, Path, Path]:
        """
        Sets up the directory structure and persistent files for training outputs.

        This method acts as the initialization step for experiment storage, ensuring that
        the `Tensorboard` directory, the `Weights` directory, and the `metrics_evolution.csv`
        file are properly initialized within the global `experiment_folder` before the training
        loop commences.

        Args:
            initial_configuration_path (Path): The original path of the experiment configuration file.

        Returns:
            tuple[Path, Path, Path]: A tuple containing:
                - tensorboard_directory (Path): Path to the TensorBoard logs folder.
                - weights_directory (Path): Path to the saved model weights folder.
                - configuration_path (Path): Path to the archived configuration file.
        """
        tensorboard_directory = self.experiment_folder / "Tensorboard"
        weights_directory = self.experiment_folder / "Weights"

        # Create directories
        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        tensorboard_directory.mkdir(exist_ok=True, parents=True)
        weights_directory.mkdir(exist_ok=True, parents=True)

        # Save a copy of the configuration file
        new_configuration_path = self.experiment_folder / "training_configuration.yaml"
        copyfile(src=str(initial_configuration_path), dst=str(new_configuration_path))

        return tensorboard_directory, weights_directory, new_configuration_path

    def print_experiment_information(self) -> None:
        """
        Prints out a summary of the experiment's configurations and paths to the console before training begins.
        This function ensures that the user is aware of where important outputs (like Tensorboard logs and weights)
        and inputs (datasets) are located, providing clickable links to easily navigate the project structure.

        Args:
            None
        """

        dataset_path = Path(self.experiment_configuration["data_folder"])

        print_yellow("=== Experiment Launch Information ===", add_separators=True)
        print_blue(f"- Tensorboard Directory : file://{self.tensorboard_directory.absolute()}")
        print_blue(f"- Weights Directory     : file://{self.weights_directory.absolute()}")
        print_blue(f"- Dataset Root Folder   : file://{dataset_path.absolute()}")
        print_blue(f"- Configuration File    : file://{self.configuration_path.absolute()}")
        print_blue(f"- Train Dataset         : file://{self.experiment_configuration.get('train_split_file')}")
        print_blue(f"- Validation Dataset    : file://{self.experiment_configuration.get('validation_split_file')}")
        print_blue(f"- Test Dataset          : file://{self.experiment_configuration.get('test_split_file')}")
        print_blue(f"- Batch Size            : {self.batch_size}")
        print_blue(f"- Total Iterations      : {self.training_iterations}")
        print_blue(f"- Learning Rate         : {self.learning_rate}")
        print_blue(f"- Resume Training       : {self.resume_training}")
