import os
import signal
import time
import torch
import torchvision

from pathlib import Path
from typing import Dict, Tuple, List, Union, Any, Optional, Iterator
from shutil import copyfile
from data.data_loader import get_dataloaders
from utilities.data_utilities import view_input_data

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import Model
from utilities.tensor_utilities import get_device, print_tensor_list
from utilities.os_utilities import print_color


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
        self.data_configuration = model_configuration["Data"]

        # Get device and dtype
        self.device = get_device()
        self.dtype = self.experiment_configuration["dtype"]

        print(f"For Training We Will Be Using device: {self.device}")

        # Basic training parameters
        self.experiment_folder = self.experiment_configuration["experiment_folder"]
        self.training_iterations = self.experiment_configuration["number_iterations"]
        self.weight_saving_iterations = self.experiment_configuration["weight_saving_iterations"]
        self.validation_iteration_ratio = self.experiment_configuration["validation_iteration_ratio"]
        self.information_dump = self.experiment_configuration["information_dump"]
        self.learning_rate = self.experiment_configuration["learning_rate"]
        self.diagnosis_modes = self.experiment_configuration["diagnosis_modes"]

        self.label_dictionary = self.data_configuration["label_dictionary"]
        self.input_image_size = self.data_configuration["image_settings"]["size"]
        self.batch_size = self.experiment_configuration["batch_size"]
        self.resume_training = self.experiment_configuration["resume_training"]

        # Setup paths and tensorboard
        self.tensorboard_directory, self.weights_directory, self.configuration_path = self.setup_training_paths(
            initial_configuration_path=Path(configuration_path))

        # We use lazy initialization for the tensorboard writers to avoid prematurely creating
        # empty log files on disk during the Trainer instantiation.
        self.training_writer = None
        self.validation_writer = None

        # Setting up dataloaders
        self.train_dataloader, self.validation_dataloader, self.test_dataloader = self.get_trainer_data_loaders()

        # Initialize Model and Optimizer
        self.model = Model(configuration=self.model_configuration, device=self.device, dtype=self.dtype,
                           verbose=self.diagnosis_modes["verbose"])
        self.model.to(device=self.device, dtype=self.dtype)

        # Setting Up The Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # TODO Add more tracking to this.
        # Metric tracking
        self.tracked_metrics_mapping = {
            "total_loss": "Total Loss",
            "region_proposal_classification_loss": "RPN Classification Loss",
            "region_proposal_localisation_loss": "RPN Regression Loss",
            "detection_classification_loss": "Detection Classification Loss",
            "detection_regression_loss": "Detection Regression Loss"}

        # Print experiment information to user so that they can know everything about the experiment
        # as well as input and output shapes of the model.
        # TODO Try to print out the number of data that we have for train, validation, and test
        self.print_experiment_information()
        self.model.print_summary()

        # Optional architecture diagnostics, printed before training continues
        if self.diagnosis_modes["compute_receptive_field"]:
            self.model.print_receptive_field_summary()
        if self.diagnosis_modes["compute_model_size"]:
            self.model.print_model_size_summary()

        # Optional viewer of the training and validation images fed to the network, closed before training continues
        if self.diagnosis_modes["view_input_images"]:
            self.view_training_and_validation_images()

        # Restoration logic before starting any training or inference
        self.start_iteration = 1
        if self.resume_training:
            self.start_iteration = self.restore_last_model()

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

        print_color(color="yellow", output="=== Experiment Launch Information ===", add_separators=True)
        print_color(color="blue", output=f"- Tensorboard Directory   : file://{self.tensorboard_directory.absolute()}")
        print_color(color="blue", output=f"- Weights Directory       : file://{self.weights_directory.absolute()}")
        print_color(color="blue", output=f"- Configuration File      : file://{self.configuration_path.absolute()}")
        print_color(color="blue",
                    output=f"- Train Data Folder       : file://{self.experiment_configuration['train_data_folder']}")
        print_color(color="blue",
                    output=f"- Validation Data Folder  : "
                           f"file://{self.experiment_configuration['validation_data_folder']}")
        print_color(color="blue",
                    output=f"- Test Data Folder        : file://{self.experiment_configuration['test_data_folder']}")
        print_color(color="blue", output=f"- Batch Size              : {self.batch_size}")
        print_color(color="blue", output=f"- Total Iterations        : {self.training_iterations}")
        print_color(color="blue", output=f"- Learning Rate           : {self.learning_rate}")
        print_color(color="blue", output=f"- Resume Training         : {self.resume_training}")
        print_color(color="blue", output=f"- Validation Iter. Ratio  : {self.validation_iteration_ratio}")

    def setup_tensorboard_writers(self) -> Tuple[SummaryWriter, Optional[SummaryWriter]]:
        """
        Initializes TensorBoard SummaryWriters for logging training and validation metrics.

        Returns:
            Tuple[SummaryWriter, Optional[SummaryWriter]]: A tuple containing the training writer
            and the validation writer (if validation is enabled).
        """
        training_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Train"))
        if self.validation_iteration_ratio > 0:
            validation_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Validation"))
        else:
            validation_writer = None

        return training_writer, validation_writer

    def get_trainer_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Initializes and returns the loaders (DALI or Native, depending on the configuration) for the training,
        validation, and test phases.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: The loaders for the training, validation and testing splits.
        """
        print_color(color="blue",
                    output="Initializing Training, Validation And Test DataLoaders...", add_separators=True)

        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(
            experiment_configuration=self.experiment_configuration,
            model_configuration=self.model_configuration,
            batch_size=self.batch_size,
            number_of_workers=4)

        return train_dataloader, validation_dataloader, test_dataloader

    def view_training_and_validation_images(self) -> None:
        """
        Shows the training and validation images exactly as the data loaders feed them to the network, each with its
        ground truth boxes and class names, in two windows side by side. Enabled by the 'view_input_images'
        diagnosis mode, it lets the user check the preprocessing (resizing, padding, box rescaling and label mapping)
        before training starts. Pressing ENTER closes the viewer and training continues.
        """
        print_color(color="red", output="View Of The Training And Validation Images That Are Fed To The Neural Network",
                    add_separators=True)
        print_color(color="green", output="Press SPACE To Navigate, Press ENTER To Stop Viewing And Start Training")

        # Invert the label dictionary so that each box is captioned with its class name instead of its index
        label_names = {class_index: str(class_name) for class_name, class_index in self.label_dictionary.items()}

        # Step through the training and validation loaders together, until the user stops or either one runs out
        for training_batch, validation_batch in zip(self.train_dataloader, self.validation_dataloader):
            number_images = min(training_batch["images"].shape[0], validation_batch["images"].shape[0])

            # Display the training and validation images at the same batch index next to each other
            for batch_index in range(number_images):
                user_quit = view_input_data(
                    images=[training_batch["images"][batch_index], validation_batch["images"][batch_index]],
                    bounding_boxes=[training_batch["bounding_boxes"][batch_index],
                                    validation_batch["bounding_boxes"][batch_index]],
                    labels=[training_batch["labels"][batch_index], validation_batch["labels"][batch_index]],
                    window_titles=["Training", "Validation"], display_input_size=self.input_image_size,
                    label_names=label_names)
                if user_quit:
                    return

    @staticmethod
    def get_next_batch(dataloader_iterator: _BaseDataLoaderIter, dataloader: DataLoader,
                       maximum_consecutive_errors: int = 10) -> Tuple[Dict[str, Any], _BaseDataLoaderIter]:
        """
        Retrieves the next batch of elements from the dataloader iterator for the training or evaluation pipeline.
        
        If the iterator is exhausted, it gracefully re-initializes it from the original dataloader.
        If an exception is encountered during data loading, it continues attempting to fetch the 
        next available batch, up to a maximum number of consecutive failures.

        Args:
            dataloader_iterator (_BaseDataLoaderIter): The current iterator for the dataloader.
            dataloader (DataLoader): The original dataloader object used to reset the iterator.
            maximum_consecutive_errors (int): The maximum number of consecutive exceptions allowed before exiting. Defaults to 10.

        Returns:
            Tuple[Dict[str, Any], _BaseDataLoaderIter]: A tuple containing the batch data dictionary,
                and the potentially refreshed dataloader iterator.

        Raises:
            SystemExit: If more than maximum_consecutive_errors consecutive exceptions occur.
        """
        error_counts = 0
        while True:
            try:
                data_dictionary = next(dataloader_iterator)
                break
            except StopIteration:
                dataloader_iterator = iter(dataloader)
            except Exception as exception:
                error_counts += 1
                if error_counts > maximum_consecutive_errors:
                    print_color(color="red",
                                output=f"Critical Error: Over {maximum_consecutive_errors} consecutive errors "
                                       f"encountered during the fetch of the next batch. Exiting.",
                                add_separators=True)
                    print(f"Last exception: {exception}")
                    exit(1)
                continue

        return data_dictionary, dataloader_iterator

    def prepare_input_data(self, data_dictionary: Dict[str, Any]) -> Tuple[
        torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Extracts and casts the batched input tensors to the designated target hardware device and precision type.
        
        This method ensures all components of the data dictionary (images, bounding boxes, and labels)
        are properly formatted and placed on the correct device (e.g., GPU) before executing the forward pass.
        
        Args:
            data_dictionary (Dict[str, Any]): The raw dictionary output from the dataloader.
            
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]: A tuple containing the casted images, 
                bounding boxes, and labels respectively.
        """
        images = data_dictionary["images"].to(device=self.device, dtype=self.dtype)

        bounding_boxes = [box_tensor.to(device=self.device, dtype=self.dtype) for box_tensor in
                          data_dictionary["bounding_boxes"]]
        labels = [label_tensor.to(device=self.device, dtype=torch.int64) for label_tensor in data_dictionary["labels"]]

        return images, bounding_boxes, labels

    def run_benchmarking_loop(self, benchmarking_iterations: int = 1e5) -> None:
        """
        Executes a performance benchmarking loop to assess model throughput and training speed.

        This function iterates through the dataset for a specified number of iterations without
        running a full training epoch. It is primarily used to isolate and measure the raw
        performance of the forward/backward passes and data loading bottleneck within the training pipeline.

        Args:
            benchmarking_iterations (int): The total number of iterations to run the benchmark. Defaults to 100,000.
        """

        # Get dataloader
        training_dataloader_iterator = iter(self.train_dataloader)
        validation_dataloader_iterator = iter(self.validation_dataloader)

        # Determine whether validation is enabled based on the configured ratio
        validation_enabled = self.validation_iteration_ratio > 0

        for training_iteration in tqdm(range(int(benchmarking_iterations)), desc="Benchmarking Loop"):
            # Get next training elements
            training_data_dictionary, training_dataloader_iterator = self.get_next_batch(
                dataloader_iterator=training_dataloader_iterator, dataloader=self.train_dataloader)

            self.model.train()
            self.optimizer.zero_grad()

            # Forward pass For Training
            training_loss, _ = self.run_model_iteration(data_dictionary=training_data_dictionary,
                                                        writer=self.training_writer,
                                                        iteration=training_iteration,
                                                        tracker_dictionary=None,
                                                        tracker_divisor=0)

            # Backward and Step
            training_loss.backward()
            self.optimizer.step()

            # Validation phase gated by validation_iteration_ratio
            if validation_enabled and training_iteration % self.validation_iteration_ratio == 0:
                # Get next validation elements
                validation_data_dictionary, validation_dataloader_iterator = self.get_next_batch(
                    dataloader_iterator=validation_dataloader_iterator, dataloader=self.validation_dataloader)

                self.model.eval()
                with torch.no_grad():
                    # Forward pass For Validation
                    _, _ = self.run_model_iteration(data_dictionary=validation_data_dictionary,
                                                    writer=self.validation_writer,
                                                    iteration=training_iteration,
                                                    tracker_dictionary=None,
                                                    tracker_divisor=0)

    def initialize_training_loop_state(self) -> Tuple[
        Iterator, Iterator, bool, Dict[str, float], Optional[Dict[str, float]], int, int]:
        """
        Initializes and returns the state variables required for the main training loop.
        
        This method sets up the initial data loaders, determines the validation gating configuration,
        and creates the metric trackers necessary for logging. Extracting this logic improves the 
        readability of the main training loop.
        
        Returns:
            Tuple: A tuple containing:
                - training_dataloader_iterator (Iterator): Iterator for the training dataset.
                - validation_dataloader_iterator (Iterator): Iterator for the validation dataset.
                - validation_enabled (bool): Flag indicating if validation is enabled.
                - training_trackers (Dict[str, float]): Dictionary for accumulating training metrics.
                - validation_trackers (Optional[Dict[str, float]]): Dictionary for validation metrics, or None.
                - validation_interval_count (int): Counter for the number of validation passes in the current interval.
                - consecutive_errors (int): Counter for consecutive errors encountered during training.
        """
        # Get dataloader
        training_dataloader_iterator = iter(self.train_dataloader)
        validation_dataloader_iterator = iter(self.validation_dataloader)

        # Determine whether validation is enabled based on the configured ratio
        validation_enabled = self.validation_iteration_ratio > 0

        # Get Metric Trackers
        # Currently just Cross Entropy Loss
        training_trackers = self.get_metric_trackers()
        validation_trackers = self.get_metric_trackers() if validation_enabled else None

        # Counter tracking how many validation forward passes occurred within the current console-log interval.
        # This is needed to compute the correct rolling average divisor for validation metrics, since
        # validation may run less frequently than training iterations depending on validation_iteration_ratio.
        validation_interval_count = 0

        consecutive_errors = 0

        return (training_dataloader_iterator, validation_dataloader_iterator, validation_enabled,
                training_trackers, validation_trackers, validation_interval_count, consecutive_errors)

    def run_training_loop(self, maximum_consecutive_errors: int = 5) -> None:
        """
        Executes the main training loop for the models.

        Args:
            maximum_consecutive_errors (int): The maximum number of consecutive exceptions allowed before the training loop exits.
        """
        # Initialize tensorboard writers right before training starts
        self.training_writer, self.validation_writer = self.setup_tensorboard_writers()

        # Initialize core loop state variables, dataloader iterators, and metric trackers
        (training_dataloader_iterator, validation_dataloader_iterator, validation_enabled,
         training_trackers, validation_trackers, validation_interval_count,
         consecutive_errors) = self.initialize_training_loop_state()

        for training_iteration in range(self.start_iteration, self.training_iterations + self.start_iteration, 1):
            try:
                if training_iteration % self.weight_saving_iterations == 0:
                    self.save_model(iteration=training_iteration)
                    # todo commented out for now because it is not properly implemented
                    # self.run_test_evaluation(iteration=training_iteration)

                # Get next training elements
                training_data_dictionary, training_dataloader_iterator = self.get_next_batch(
                    dataloader_iterator=training_dataloader_iterator,
                    dataloader=self.train_dataloader)

                self.model.train()
                self.optimizer.zero_grad()

                # Forward pass For Training
                training_loss, _ = self.run_model_iteration(data_dictionary=training_data_dictionary,
                                                            writer=self.training_writer,
                                                            iteration=training_iteration,
                                                            tracker_dictionary=training_trackers,
                                                            tracker_divisor=self.information_dump)

                # Backward and Step
                training_loss.backward()
                self.optimizer.step()

                # Validation phase gated by the configurable validation_iteration_ratio.
                # A ratio of 1 runs validation every iteration, 2 runs every 2nd, N runs every Nth.
                if validation_enabled and training_iteration % self.validation_iteration_ratio == 0:
                    self.model.eval()
                    with torch.no_grad():
                        validation_data_dictionary, validation_dataloader_iterator = self.get_next_batch(
                            dataloader_iterator=validation_dataloader_iterator, dataloader=self.validation_dataloader)
                        # Forward pass for validation — accumulates raw metric sums (divisor=1) into
                        # validation_trackers. The actual averaging by validation_interval_count is
                        # deferred to the console log dump so the moving average is correct.
                        _, _ = self.run_model_iteration(data_dictionary=validation_data_dictionary,
                                                        writer=self.validation_writer,
                                                        iteration=training_iteration,
                                                        tracker_dictionary=validation_trackers,
                                                        tracker_divisor=1)
                        validation_interval_count += 1

                # Console log dump
                if training_iteration % self.information_dump == 0:
                    # Normalize accumulated validation metrics by the actual validation count in this interval
                    if validation_enabled and validation_trackers is not None and validation_interval_count > 0:
                        for metric_key in self.tracked_metrics_mapping.keys():
                            validation_trackers[metric_key] /= validation_interval_count

                    training_trackers = self.console_log_update_tracker(
                        iterations=training_iteration,
                        training_tracker_dictionary=training_trackers,
                        validation_tracker_dictionary=validation_trackers,
                        validation_interval_count=validation_interval_count)

                    if validation_enabled:
                        validation_trackers = self.get_metric_trackers()
                        validation_interval_count = 0

                # Reset the consecutive errors counter after a fully successful iteration
                consecutive_errors = 0

            except KeyboardInterrupt:
                print_color(color="red",
                            output=f"Training Interrupted by User at iteration {training_iteration}.",
                            add_separators=True)
                break
            except Exception as error:
                # Increment error counter when an exception occurs inside the training iteration loop
                consecutive_errors += 1

                if consecutive_errors > maximum_consecutive_errors:
                    print_color(color="red",
                                output=f"Critical Error: Over {maximum_consecutive_errors} consecutive errors "
                                       f"encountered during training iterations. Exiting.", add_separators=True)
                    print(f"Last exception: {error}")
                    break
                else:
                    print_color(color="yellow", output=f"Error encountered during iteration {training_iteration}."
                                                       f" Continuing... (Consecutive Errors: "
                                                       f"{consecutive_errors}/{maximum_consecutive_errors})",
                                add_separators=True)
                    print(f"Exception: {error}")
                    continue

        self.save_model(iteration=training_iteration)
        print_color(color="green",
                    output=f"Model successfully saved at iteration {training_iteration}. Exiting Training.",
                    add_separators=True)
        if self.training_writer is not None:
            self.training_writer.close()
        if validation_enabled and self.validation_writer is not None:
            self.validation_writer.close()

    def run_test_evaluation(self, iteration: int) -> None:
        """
        Performs a full evaluation on the test dataset and logs results to a file.

        Args:
            iteration (int): The current training iteration index.
        """
        # Run the model in evaluation mode
        self.model.eval()

        total_test_loss = 0.0
        number_batches = len(self.test_dataloader)

        # Initialize dictionary to get all sorts of metrics
        testing_trackers = self.get_metric_trackers()
        total_metrics = {}

        # TODO MAKE SURE THIS IS THE WAY TO DO IT
        with torch.no_grad():
            for test_data_dictionary in tqdm(self.test_dataloader, total=number_batches,
                                             desc=f"Test Evaluation Iteration {iteration}"):
                testing_loss, test_model_outputs = self.run_model_iteration(data_dictionary=test_data_dictionary,
                                                                            writer=None,
                                                                            iteration=-1,
                                                                            tracker_dictionary=testing_trackers)

                # Aggregate metrics for the current set
                total_metrics = self.compute_and_accumulate_metrics(model_outputs=test_model_outputs,
                                                                    ground_truth_data_dictionary=test_data_dictionary,
                                                                    total_metrics=total_metrics)

        # Log to the results to a single file, for all the saved iterations of the given architecture.
        self.save_test_evaluation_csv(experiment_folder=self.project_root,
                                      iteration=iteration,
                                      loss=total_metrics)

        # Run sample predictions for visualization
        self.run_sample_predictions(iteration=iteration, number_samples=20)

    # todo to be properly implemented !!!
    # def run_sample_predictions(self, iteration: int, number_samples: int = 20) -> None:
    #     """
    #     Runs inference on a subset of the test dataset and saves the predicted
    #     segmentation masks as PNG images for visual inspection.
    #
    #     Args:
    #         iteration (int): The current training iteration index.
    #         number_samples (int): The maximum number of test samples to process.
    #     """
    #     print(f"\nRunning {number_samples} Sample Predictions For Iteration {iteration}...")
    #
    #     output_directory = self.weights_directory / f"Iteration_{iteration}" / "test_sample_predictions"
    #     output_directory.mkdir(exist_ok=True, parents=True)
    #
    #     self.model.eval()
    #     samples_processed = 0
    #     test_dataloader_iterator = iter(self.test_dataloader)
    #
    #     with torch.no_grad():
    #         while samples_processed < number_samples:
    #             try:
    #                 test_images, test_masks = next(test_dataloader_iterator)
    #             except StopIteration:
    #                 break
    #
    #             test_images = test_images.to(device=self.device, dtype=self.dtype)
    #             test_masks = test_masks.to(device=self.device, dtype=self.dtype)
    #
    #             model_outputs = self.model(test_images)
    #
    #             # For mutually exclusive multi-channel classification, the channel with the max
    #             # logit is the predicted class. This avoids a computationally expensive softmax.
    #             max_logits = model_outputs.max(dim=1, keepdim=True).values
    #             predicted_masks = (model_outputs == max_logits).to(self.dtype)
    #
    #             batch_size_current = test_images.size(0)
    #             for sample_index in range(batch_size_current):
    #
    #                 if samples_processed >= number_samples:
    #                     break
    #
    #                 # Extract the current input image from the batch
    #                 input_image_tensor = test_images[sample_index:sample_index + 1]
    #
    #                 # Convert RGB images to grayscale by averaging across the channel dimension
    #                 # This ensures the input image can be concatenated into the same visualization grid
    #                 # alongside the single-channel ground truth and predicted masks without shape mismatches
    #                 if input_image_tensor.shape[1] == 3:
    #                     input_image_tensor = input_image_tensor.mean(dim=1, keepdim=True)
    #
    #                 # Create a comprehensive evaluation grid for this sample, stacking rows for each channel
    #                 all_components = []
    #                 for channel_index in range(test_masks.shape[1]):
    #                     ground_truth_channel = test_masks[
    #                         sample_index:sample_index + 1, channel_index:channel_index + 1, :, :]
    #                     prediction_channel = predicted_masks[
    #                         sample_index:sample_index + 1, channel_index:channel_index + 1, :, :]
    #
    #                     # Generate the visualization row (Ground Truth, Prediction, Error Map, Boundary Overlay)
    #                     row_components = create_evaluation_row_for_channel(
    #                         original_image=input_image_tensor,
    #                         ground_truth=ground_truth_channel,
    #                         prediction=prediction_channel)
    #                     all_components.extend(row_components)
    #
    #                 comparison_grid = torch.cat(all_components, dim=0)
    #
    #                 # Determine the number of columns to wrap properly (5 images per channel row)
    #                 number_of_columns = 5
    #
    #                 output_path = output_directory / f"sample_{samples_processed:04d}.png"
    #                 torchvision.utils.save_image(comparison_grid, output_path, nrow=number_of_columns)
    #                 samples_processed += 1
    #
    #     print_color(color="green", output=f"- file://{output_directory}")

    # todo to be properly implemented !!!
    def run_model_iteration(self, data_dictionary: Dict[str, torch.Tensor | Any],
                            writer: Optional[SummaryWriter], iteration: int,
                            tracker_dictionary: dict | None,
                            tracker_divisor: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        # todo to be updated
        Executes a single forward pass of the models, computes the loss,
        and updates performance trackers.

        Args:
            data_dictionary (Dict[str, torch.Tensor | Any]): Dictionary containing batched 'images', 'bounding_boxes', and 'labels'.
            writer (Optional[SummaryWriter]): TensorBoard writer for logging. If None, logging is skipped.
            iteration (int): The current training iteration step.
            tracker_dictionary (Dict[str, Any] | None): Dictionary tracking rolling average metrics.
            tracker_divisor (int): The number of iterations contributing to the rolling average within a
                console log interval. For training this equals information_dump; for validation this equals
                the actual number of validation runs in the interval so that the moving average is correct.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the total_loss and the model predictions.
        """

        # Extract and cast the inputs to the correct hardware device and precision
        images, bounding_boxes, labels = self.prepare_input_data(data_dictionary=data_dictionary)

        # Execute the forward pass through the object detection model
        model_outputs = self.model(input_tensor=images,
                                   ground_truth_bounding_boxes=bounding_boxes,
                                   ground_truth_labels=labels)

        # Extract all specific losses dynamically from the model outputs using dictionary comprehension
        metric_dictionary = {key: value for key, value in model_outputs.items() if "_loss" in key}

        # Aggregate the individual loss components into a single total loss for backpropagation
        total_loss = sum(metric_dictionary.values())

        # Insert the total loss into the metrics for TensorBoard logging and console tracking
        metric_dictionary["total_loss"] = total_loss
        # TODO : Think about list of True Positives
        # TODO : Think about list of False Positives
        # TODO : Think about list of True Negatives
        # TODO : Think about list of False Negatives

        # Update tracker dictionary (rolling average accumulation)
        # Each metric value is divided by the tracker_divisor to compute a running mean over the interval
        if tracker_dictionary is not None and tracker_divisor > 0:
            for metric_key, metric_value in metric_dictionary.items():
                if metric_key in tracker_dictionary:
                    tracker_dictionary[metric_key] += metric_value.item() / tracker_divisor

        # Log data to tensorboard (per-iteration)
        self.log_metrics_to_tensorboard(writer=writer,
                                        iteration=iteration,
                                        metric_dictionary=metric_dictionary)

        return total_loss, model_outputs

    def save_model(self, iteration: int) -> None:
        """
        Persists the current model weights and optimizer state to disk, then records them in the checkpoint registry
        used to resume training.

        The save cannot be interrupted: Ctrl+C is ignored until both the weight file and the registry are written,
        since copying every GPU tensor to the CPU can take a few seconds and a second Ctrl+C during that time would
        otherwise leave a corrupted checkpoint. The weights are first written to a temporary file and then renamed,
        so a checkpoint on disk is always either complete or absent.

        Args:
            iteration (int): The current training iteration, used for naming the weight file.
        """
        model_directory = self.weights_directory / f"Iteration_{iteration}"
        model_directory.mkdir(exist_ok=True, parents=True)

        checkpoint_path = model_directory / f"model_{iteration:06}.pt"
        temporary_checkpoint_path = model_directory / f"model_{iteration:06}.pt.tmp"

        print_color(color="yellow", output="Saving model, please wait (Ctrl+C is ignored until the save completes)...")

        # Ignore keyboard interrupts while saving, and restore the previous handler once the save is complete
        previous_interrupt_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            # Write to a temporary file, then atomically move it to its final name
            torch.save(obj={'iteration': iteration, 'model_state': self.model.state_dict(),
                            'optimizer_state': self.optimizer.state_dict()}, f=temporary_checkpoint_path)
            os.replace(src=temporary_checkpoint_path, dst=checkpoint_path)

            # Update the full-checkpoint registry only once the weight file is complete
            self.dump_in_checkpoint(iteration=iteration)
        finally:
            signal.signal(signal.SIGINT, previous_interrupt_handler)

        print(f"Checkpoint Folder : file://{model_directory}")
        print(f" - Model Saved In : model_{iteration:06}.pt")

    def log_metrics_to_tensorboard(self, writer: Optional[SummaryWriter], iteration: int,
                                   metric_dictionary: Dict[str, torch.Tensor]):
        """
        Logs individual metric components and total loss to TensorBoard.

        Args:
            writer (Optional[SummaryWriter]): The TensorBoard writer to use. If None, writing is skipped.
            iteration (int): The current iteration index.
            metric_dictionary (Dict[str, torch.Tensor]): Dictionary containing current iteration metrics.
        """
        if writer is None:
            return

        for metric_key, display_name in self.tracked_metrics_mapping.items():
            writer.add_scalar(display_name, metric_dictionary[metric_key].item(), iteration)

    def extract_last_model_iteration(self) -> int:
        """
        Retrieves the iteration number of the most recently saved model checkpoint.

        This method reads the 'full-checkpoint' registry file located in the weights
        directory to determine the latest available checkpoint. This is crucial for
        seamlessly resuming training after an interruption without manual intervention.

        Returns:
            int: The iteration number of the last saved model, or 0 if no checkpoint exists.
        """
        last_iteration = 0

        full_checkpoint_logger = self.weights_directory / "full-checkpoint"

        if not full_checkpoint_logger.exists():
            return last_iteration

        with open(full_checkpoint_logger, 'r') as file:
            first_line = file.readline().strip()

        last_iteration_string = first_line.split('"')[1]
        last_iteration = int(last_iteration_string.split("_")[-1])

        return last_iteration

    def restore_last_model(self, index_iteration: Optional[int] = None) -> int:
        """
        Restores the model and optimizer states to a specific or the most recent checkpoint.

        If `index_iteration` is provided, it restores that exact checkpoint (useful for
        running isolated evaluation or inference on a specific model state). If not provided,
        it automatically finds and loads the latest checkpoint to resume training.

        Args:
            index_iteration (Optional[int]): The specific iteration to restore. If None,
                it resolves to the last saved iteration.

        Returns:
            int: The iteration number from which training should commence. If a model was
                loaded, it returns `loaded_iteration + 1`. If no model was found, it returns 1.
        """

        if not index_iteration:
            index_iteration = self.extract_last_model_iteration()

        # Despite trying to get the last model None was found
        if not index_iteration:
            print_color(color="green",
                        output="No Initiation Model Weights Will Be Used..."
                               "\nWe Generate A New Model That Will Be Trained From Scratch.",
                        add_separators=True)
            return 1
        else:
            self.load_model(iteration=index_iteration)
            print_color(color="green",
                        output=f"We Loaded A Model That Was Previously Saved At Iteration {index_iteration}",
                        add_separators=True)
            # Resume from the next iteration
            return index_iteration + 1

    def load_model(self, iteration: int) -> None:
        """
        Loads the model weights and optimizer state from a specified iteration checkpoint.

        This method reads the serialized `.pt` file from disk and maps the tensors to
        the currently active device (CPU, CUDA, or MPS).

        Args:
            iteration (int): The exact iteration number identifying the checkpoint to load.
        """

        model_path = self.weights_directory / f"Iteration_{iteration}" / f"model_{iteration:06}.pt"

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def dump_in_checkpoint(self, iteration: int) -> None:
        """
        Updates the checkpoint registry file to track the most recently saved model.

        The `full-checkpoint` file acts as a manifest, keeping a historical record of
        all saved checkpoints and explicitly marking the latest one. This ensures the
        resumption logic (`extract_last_model_iteration`) always knows where to start.

        Args:
            iteration (int): The iteration number of the newly saved checkpoint.
        """
        checkpoint_file = self.weights_directory / "full-checkpoint"

        # Saved the iteration in a checkpoint file
        try:
            with open(checkpoint_file, "r") as f:
                d = f.readlines()
        except FileNotFoundError:
            d = []

        with open(checkpoint_file, "w") as f:
            if len(d) == 0:
                d.append(f'model_checkpoint_path: "Iteration_{iteration}"\n')
                d.append(f'all_model_checkpoint_paths: "Iteration_{iteration}"\n')
            else:
                d[0] = f'model_checkpoint_path: "Iteration_{iteration}"\n'
                d.append(f'all_model_checkpoint_paths: "Iteration_{iteration}"\n')
            for line in d:
                f.write(line)

    @staticmethod
    def format_time_interval(duration_seconds: float) -> str:
        """
        Formats a time duration in seconds into a human-readable string with hours, minutes, and seconds.
        
        Args:
            duration_seconds (float): The time duration in seconds.
            
        Returns:
            str: The formatted time string.
        """
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = duration_seconds % 60

        # Format the time into a consistent HH:MM:SS layout
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"

    def console_log_update_tracker(self, iterations: int,
                                   training_tracker_dictionary: Dict[str, Any],
                                   validation_tracker_dictionary: Optional[Dict[str, Any]] = None,
                                   validation_interval_count: int = 0):
        """
        Prints the rolling average of training and validation metrics to the console at regular intervals.
        This method provides the primary training feedback loop, showing throughput and per-metric averages
        so the user can monitor convergence during the object detection training pipeline.

        Args:
            iterations (int): Current global training iteration.
            training_tracker_dictionary (Dict[str, Any]): Rolling average trackers for training.
            validation_tracker_dictionary (Optional[Dict[str, Any]]): Rolling average trackers for validation.
            validation_interval_count (int): The number of validation forward passes that occurred in this
                console-log interval. Used to inform the user and to gate validation metric display.

        Returns:
            Dict[str, Any]: A fresh training tracker dictionary for the next interval.
        """
        print_length = 100
        print("=" * print_length)

        # Calculate image scale and throughput metrics
        total_images = iterations * self.batch_size
        interval_images = self.information_dump * self.batch_size
        duration = time.time() - training_tracker_dictionary['start_time']
        images_per_second = interval_images / max(duration, 0.001)

        # Display the formatted summary header
        header = (f"| Iteration: {iterations:<6} | Total Images: {total_images:<8} "
                  f"| Interval Images: {interval_images:<6} | Throughput: {images_per_second:.1f} imgs/sec |")
        print(header)
        print("-" * print_length)

        for metric_key, display_name in self.tracked_metrics_mapping.items():
            train_value = training_tracker_dictionary[metric_key]
            message = f"Moving Average of Train {display_name:30} : {train_value:.4f}"
            print_color(color="blue", output=message)

            # Only display validation metrics if at least one validation iteration ran in this interval
            if validation_tracker_dictionary is not None and validation_interval_count > 0:
                validation_value = validation_tracker_dictionary[metric_key]
                validation_message = f"Moving Average of Valid {display_name:30} : {validation_value:.4f}"
                print_color(color="yellow", output=validation_message)

        print("-" * print_length)
        formatted_duration = self.format_time_interval(duration_seconds=duration)
        print(f"Interval Of {self.information_dump} Iterations Took {formatted_duration}.")

        # Show how many validation iterations contributed to the reported averages
        if self.validation_iteration_ratio > 0:
            print(f"Validation Iterations In This Interval: {validation_interval_count} "
                  f"(Ratio: Every {self.validation_iteration_ratio} Training Iterations).")

        print("=" * print_length)

        return self.get_metric_trackers()

    def get_metric_trackers(self) -> Dict[str, Any]:
        """
        Initializes a dictionary to track rolling averages of metrics.

        Returns:
            Dict[str, Any]: A dictionary with metric keys set to 0.0 and a start_time.
        """
        tracker_dictionary = {"start_time": time.time()}
        for metric_key in self.tracked_metrics_mapping.keys():
            tracker_dictionary[metric_key] = 0.0

        return tracker_dictionary

    def compute_and_accumulate_metrics(self, model_outputs, ground_truth_data_dictionary, total_metrics):
        pass

    def save_test_evaluation_csv(self, iteration, loss):
        pass
