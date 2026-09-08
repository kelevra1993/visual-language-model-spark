import torch

from pathlib import Path
from typing import Dict, Tuple, List, Union, Any
from model.model import Model
from utilities.tensor_utilities import get_device


class Trainer:
    """
    todo to later be documented
    """

    def __init__(self, experiment_folder: Path, model_configuration: Dict[str, Any], data_folder: Path,
                 train_split_file: Path, validation_split_file: Path, test_split_file: Path,
                 number_iterations: int, weight_saving_iterations: int,
                 compute_validation_iteration: bool,
                 learning_rate: float,
                 dtype: torch.dtype,
                 information_dump: int,
                 resume_training: bool):

        # Get device and dtype
        self.device = get_device()
        self.dtype = dtype

        print(f"For Training We Will Be Using device: {self.device}")

        # Basic training parameters
        self.experiment_folder = experiment_folder
        self.training_iterations = number_iterations
        self.weight_saving_iterations = weight_saving_iterations
        self.compute_validation_iteration = compute_validation_iteration
        self.information_dump = information_dump
        self.learning_rate = learning_rate

        # Load Training Configuration containing all the training parameters
        self.model_configuration = model_configuration

        # Initialize Model and Optimizer
        self.model = Model(configuration=self.model_configuration, device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)

        # # TODO Later call the optimizer after an initial model configuration.
        # # Setting Up The Optimizer
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


        # # TODO Will Be Later Implemented Step By Step starting with our backbone layer
        # # Print experiment information to user so that they can know everything about the experiment
        # # as well as input and output shapes of the model.
        # self.print_experiment_information()
        # self.model.print_summary(expected_image_size=(self.expected_input_size, self.expected_input_size))