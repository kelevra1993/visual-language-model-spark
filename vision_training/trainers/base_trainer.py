"""
General trainer shared by every task: the run folder, the training loop with its validation passes, the optimizer
and learning rate schedule, the checkpoints and the resume, the evaluation of the saved checkpoints on the test split,
the MLflow tracking of the run (vision_training/tracking) and the input batch viewer. The task trainers subclass it
and implement its abstract hooks, and vision_training/main.py builds them through the task registry.
"""

import math
import cv2
import numpy as np
import torch

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from vision_training.configuration_processing.configuration_changes import (ConfigurationChanges, find_changed_settings,
                                                                            format_setting_value)
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.configuration_processing.run_name import build_run_name
from vision_training.core.contracts import LOSS_KEY_SUFFIX, SAMPLE_IDENTIFIERS_KEY, EvaluationResults
from vision_training.data.augmentations import DataAugmenter
from vision_training.evaluation.evaluation_report import report_evaluation_results
from vision_training.evaluation.sample_selection import choose_sample_identifiers
from vision_training.models.model_registry import MODEL_DTYPE
from vision_training.models.model_transfer import (TRANSFERABLE_COMPONENTS, freeze_batch_normalization_statistics,
                                                   freeze_component, freeze_normalization_parameters,
                                                   get_component_name, load_pretrained_component)
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import get_last_checkpoint_iteration, load_checkpoint, save_checkpoint
from vision_training.trainers.learning_rate_schedule import compute_learning_rate_factor
from vision_training.trainers.metric_tracking import INTERVAL_BATCHES_KEY, MetricTracker
from vision_training.tracking.experiment_tracker import (FAILED_STATUS, FINISHED_STATUS, KILLED_STATUS,
                                                         RUN_RECORD_FILE_NAME, DisabledTracker,
                                                         create_experiment_tracker, resolve_tracking_uri)
from vision_training.utilities.os_utilities import format_path_link, print_color
from vision_training.utilities.tensor_utilities import get_device

# Evaluation files of a checkpoint sent to the MLflow run, small whatever the size of the test split: the confusion
# matrix image (per-class report only), the metrics of every class and the class thresholds of the classification-like
# tasks. The predictions, the error images and the sample images grow with the test split and stay in the run folder.
TRACKED_EVALUATION_FILES = ["Confusion Matrix Image", "Class Metrics", "Class Thresholds"]


class BaseTrainer(ABC):
    """
    General training pipeline shared by every task of the repository (object detection, classification, semantic
    segmentation, sequence classification, sequence labelling and sequence comparison).

    It owns everything that does not depend on the task: the run folder named after the settings, the training loop
    with its validation passes and checkpoint schedule, the metric tracking (console, TensorBoard and MLflow), the
    checkpoints and the resume, the evaluation loop of the saved checkpoints on the test split and its report, and
    the viewer of the input batches. A task subclasses it and implements the hooks that depend on its data and its
    model (marked abstract), and may override the optional ones (augmentation, batch metrics, evaluation context,
    optimizer...).
    """

    # Name of the task trained by the subclass, stored in its checkpoints
    TASK_NAME = ""

    # Kind of sample of a batch, "Images" for the image tasks, "Sequences" for the sequence tasks or "Pairs" for the
    # sequence comparison task
    SAMPLE_NAME = "Images"

    # Whether the real frames of the training batches are counted for a frame throughput, True for the sequence tasks
    COUNTS_FRAMES = False

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Prepares a training run: reads the settings, creates the run folder (not when only viewing the input images,
        which exits without training), builds the loaders, the model and the optimizer through the task hooks, prints
        the experiment information and resumes from the last checkpoint of the run folder when asked.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone and the task sections).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # A trainer only trains the task it implements, a configuration of another task is a mistake
        if experiment_configuration["task"] != self.TASK_NAME:
            raise ValueError(f"{type(self).__name__} trains the task {self.TASK_NAME!r}, but the configuration is for "
                             f"the task {experiment_configuration['task']!r}.")

        # Load Training Configuration containing all the training parameters
        self.experiment_configuration = experiment_configuration
        self.model_configuration = model_configuration

        # Get specific configurations
        self.data_configuration = model_configuration["Data"]

        # Get the device, the models always being built in float32
        self.device = get_device()
        self.dtype = MODEL_DTYPE

        print(f"For Training We Will Be Using device: {self.device}")

        # Every run trains in a folder named after its settings, inside the experiments root folder of the
        # configuration, so that the same settings resume in the same folder and changed settings start a new one
        self.run_name = self.get_run_name()
        self.experiment_folder = self.experiment_configuration["experiment_folder"] / self.run_name

        # Basic training parameters
        self.training_iterations = self.experiment_configuration["number_iterations"]
        self.weight_saving_iterations = self.experiment_configuration["weight_saving_iterations"]
        self.intermediate_weight_saving_iterations = self.experiment_configuration[
            "intermediate_weight_saving_iterations"]
        self.validation_iteration_ratio = self.experiment_configuration["validation_iteration_ratio"]
        self.information_dump = self.experiment_configuration["information_dump"]
        self.learning_rate = self.experiment_configuration["learning_rate"]
        self.diagnosis_modes = self.experiment_configuration["diagnosis_modes"]

        # Whether the evaluation reports show every class (table, TensorBoard scalars, confusion matrix image) or only
        # the global metrics, for the runs with many classes
        self.per_class_report = self.experiment_configuration["evaluation"]["per_class_report"]

        # Number of fixed test samples drawn with their targets and predictions at every evaluation, 0 for none
        self.number_sample_images = self.experiment_configuration["evaluation"]["number_sample_images"]

        # MLflow tracking of the run, opened at the start of the training loop: until then, and for the input viewer,
        # the benchmarking loop and the trainer of a standalone evaluation, the tracker does nothing
        self.tracking_configuration = self.experiment_configuration["tracking"]
        self.experiment_tracker = DisabledTracker()

        self.label_dictionary = self.data_configuration["label_dictionary"]

        # Random augmentation of the training batches, applied on their device right before the forward pass
        self.data_augmenter = DataAugmenter(augmentation_configuration=self.data_configuration["data_augmentation"])
        self.input_image_size = self.data_configuration["image_settings"]["size"]
        self.batch_size = self.experiment_configuration["batch_size"]
        self.resume_training = self.experiment_configuration["resume_training"]

        # Paths of the run folder, created with the copy of the configuration only for a training: the input viewer
        # exits without training, so it leaves no empty run folder and never overwrites the configuration of a run. A
        # resumed run also gets the settings changed since its last start, printed with the experiment information
        (self.tensorboard_directory, self.weights_directory, self.configuration_path,
         self.configuration_changes) = self.setup_training_paths(
            initial_configuration_path=Path(configuration_path),
            create_run_folder=not self.diagnosis_modes["view_input_images"])

        # We use lazy initialization for the tensorboard writers to avoid prematurely creating
        # empty log files on disk during the Trainer instantiation.
        self.training_writer = None
        self.validation_writer = None
        self.test_writer = None

        # Setting up dataloaders
        print_color(color="blue",
                    output="Initializing Training, Validation And Test DataLoaders...", add_separators=True)
        self.train_dataloader, self.validation_dataloader, self.test_dataloader = self.build_dataloaders()

        # Fixed test samples drawn at every evaluation, chosen once from the classes of the test samples so that the
        # same samples are followed over the training and its resumptions
        self.sample_image_identifiers = choose_sample_identifiers(sample_classes=self.get_test_sample_classes(),
                                                                  number_samples=self.number_sample_images)

        # Initialize Model and Optimizer, the backbone, the temporal module, the classification head, the
        # segmentation decoder and the region proposal and detection heads being possibly initialised by the ones of
        # other runs, through the 'pretrained' settings of the sections the task has
        self.pretrained_configurations = {section_name: self.model_configuration[section_name]["pretrained"]
                                          for section_name in TRANSFERABLE_COMPONENTS
                                          if section_name in self.model_configuration}
        self.optimization_configuration = self.model_configuration["Optimization"]
        self.mixed_precision = self.optimization_configuration["mixed_precision"]
        self.model = self.build_model()
        self.model.to(device=self.device, dtype=self.dtype)
        self.pretrained_checkpoint_paths = self.initialise_pretrained_components()
        self.optimizer = self.build_optimizer()

        # Learning rate of every parameter group before the schedule, which multiplies it at every step
        self.base_learning_rates = [parameter_group["lr"] for parameter_group in self.optimizer.param_groups]

        # Metric tracking of the losses and batch metrics chosen by the task
        self.tracked_metrics_mapping = self.get_tracked_metrics()
        self.metric_tracker = MetricTracker(tracked_metrics_mapping=self.tracked_metrics_mapping,
                                            batch_size=self.batch_size,
                                            validation_iteration_ratio=self.validation_iteration_ratio,
                                            sample_name=self.SAMPLE_NAME, counts_frames=self.COUNTS_FRAMES)

        # Print experiment information to user so that they can know everything about the experiment
        # as well as input and output shapes of the model.
        self.print_experiment_information()
        self.model.print_summary()

        # Optional architecture diagnostics, printed before training continues
        if self.diagnosis_modes["compute_receptive_field"]:
            self.model.print_receptive_field_summary()
        if self.diagnosis_modes["compute_model_size"]:
            self.model.print_model_size_summary()

        # Optional viewer of the training and validation samples fed to the network, the program exits once it is closed
        # without having created the run folder
        if self.diagnosis_modes["view_input_images"]:
            self.view_input_batches()
            exit()

        # Restoration logic before starting any training or inference
        self.start_iteration = 1
        if self.resume_training:
            self.start_iteration = self.restore_last_model()

    # ------------------------------------------------------------------------------------------------------------------
    # Task hooks: every task implements them
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the loaders of the training, validation and test splits of the task.

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders, yielding batch dictionaries.
        """

    @abstractmethod
    def build_model(self) -> TaskModel:
        """
        Builds the network of the task from the model configuration.

        Returns:
            TaskModel: The untrained network (moved to the device and precision of the run by the trainer).
        """

    @abstractmethod
    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the metrics tracked on every training and validation batch (console and TensorBoard): the losses of the
        model, the total loss and the batch metrics of compute_batch_metrics.

        Returns:
            Dict[str, str]: The display name of every tracked metric key.
        """

    @abstractmethod
    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the samples of a dataset split for the experiment information printed at startup.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of samples (giving the iterations per epoch of the training split) and the
                description printed next to the split folder.
        """

    @abstractmethod
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Moves the tensors of a loader batch to the device and precision of the model.

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a loader.

        Returns:
            Dict[str, Any]: The batch ready for the model, keeping its sample identifiers.
        """

    @abstractmethod
    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the model on a prepared training or validation batch, with its targets, so that it returns its losses.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The model outputs, every key ending in '_loss' being a scalar loss.
        """

    @abstractmethod
    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the model on a prepared test batch as in deployment, to get the predictions evaluated by the task.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The model predictions.
        """

    @abstractmethod
    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the empty collection filled by accumulate_evaluation_batch during an evaluation of the test split.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, which names its evaluation folder.

        Returns:
            Dict[str, Any]: The empty accumulator.
        """

    @abstractmethod
    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Adds the predictions and targets of a test batch to the evaluation accumulator.

        Args:
            accumulator (Dict[str, Any]): The accumulator, modified in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The predictions of predict_batch.
            sample_indices (List[int]): The indices of the batch samples to keep (samples already evaluated, repeated
                by a loader to fill its last batch, are left out).
        """

    @abstractmethod
    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the metrics of the task over the whole test split: the global metrics, notes and tables of the
        report, its scalars, and its per-class outputs (per-class tables, scalars and images), which the reporter only
        shows with the per-class report.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and targets.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results reported by the trainer.
        """

    @abstractmethod
    def get_test_sample_classes(self) -> Dict[int, List[int]]:
        """
        Lists the classes of every test sample, from which the fixed test samples drawn at every evaluation are chosen
        (evaluation/sample_selection.py): the samples are spread over their classes, and evenly over the test split when
        no sample has a class.

        Returns:
            Dict[int, List[int]]: The class indices of every test sample, keyed by sample identifier, empty lists for
                the tasks whose samples are not spread over the classes.
        """

    @abstractmethod
    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws one fixed test sample with its target and its prediction, saved and sent to TensorBoard at every
        evaluation so that the predictions of the same samples can be followed over the training. Its caption only
        describes the prediction: the iteration names the evaluation folder and the TensorBoard step, and the sample
        name the file and the TensorBoard tag, so that a short caption keeps the image at its own width.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The predictions of predict_batch.
            sample_index (int): The index of the sample in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sample.
        """

    @abstractmethod
    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays one training sample (augmented) next to one validation sample, with their targets, for the
        'view_input_images' diagnosis mode.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed samples in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Optional hooks: the default behaviour suits most tasks
    # ------------------------------------------------------------------------------------------------------------------

    def get_run_name(self) -> str:
        """
        Builds the name of the run folder from the settings of the run.

        Returns:
            str: The run name.
        """
        return build_run_name(experiment_configuration=self.experiment_configuration,
                              model_configuration=self.model_configuration)

    def get_test_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of a test sample, naming its sample image file and its TensorBoard tag. Given by the test
        dataset by default; the detection trainer overrides it with the file names of the test labels.json.

        Args:
            sample_identifier (int): The identifier of the sample, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the sample.
        """
        return self.test_dataloader.dataset.get_sample_name(sample_identifier=sample_identifier)

    def build_optimizer(self) -> torch.optim.Optimizer:
        """
        Builds the optimizer of 'Optimization.optimizer' with its weight decay: Adam (weight decay added to the
        gradients), AdamW (weight decay decoupled from the gradients) or SGD with Nesterov momentum 0.9. Every component
        loaded from another run (backbone, temporal module, classification head, segmentation decoder, region proposal
        heads, detection heads) that is not frozen gets its own parameter group at the learning rate times its
        'pretrained.learning_rate_multiplier', so that it can be fine-tuned at another speed than the new layers. The
        group is created even with a multiplier of 1, so that the layout of the groups only depends on the transfers and
        freezes written in the run name, and the multiplier can change between two resumptions of a run without changing
        the layout of its checkpoints. The first group holds every other parameter: without such a component, every
        parameter shares this single group, the layout of the checkpoints of the runs without transfer, and a
        transferred backbone comes second, the layout of the runs that only transferred their backbone.

        Returns:
            torch.optim.Optimizer: The optimizer at the configured learning rate.
        """
        # One group per transferred component that learns, in the order of TRANSFERABLE_COMPONENTS
        component_groups = []
        component_parameter_identifiers = set()
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            if pretrained_configuration["experiment_folder"] is None or pretrained_configuration["freeze"]:
                continue
            component_parameters = list(getattr(self.model, TRANSFERABLE_COMPONENTS[section_name]).parameters())
            component_parameter_identifiers.update(id(parameter) for parameter in component_parameters)
            component_groups.append({"params": component_parameters, "lr": self.learning_rate *
                                     pretrained_configuration["learning_rate_multiplier"]})

        # Every other parameter learns at the learning rate of the run, in the first group
        other_parameters = [parameter for parameter in self.model.parameters()
                            if id(parameter) not in component_parameter_identifiers]
        parameter_groups = [{"params": other_parameters}] + component_groups

        optimizer_name = self.optimization_configuration["optimizer"]
        weight_decay = self.optimization_configuration["weight_decay"]
        if optimizer_name == "adamw":
            return torch.optim.AdamW(params=parameter_groups, lr=self.learning_rate, weight_decay=weight_decay)
        if optimizer_name == "sgd":
            return torch.optim.SGD(params=parameter_groups, lr=self.learning_rate, momentum=0.9, nesterov=True,
                                   weight_decay=weight_decay)
        return torch.optim.Adam(params=parameter_groups, lr=self.learning_rate, weight_decay=weight_decay)

    def set_training_mode(self) -> None:
        """
        Puts the model in training mode before every training step, except the batch normalization layers of the
        frozen transferred components, kept in evaluation mode so that their statistics stay those of their source
        runs (see freeze_batch_normalization_statistics).
        """
        self.model.train()
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            if pretrained_configuration["freeze"]:
                freeze_batch_normalization_statistics(
                    component=getattr(self.model, TRANSFERABLE_COMPONENTS[section_name]))

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the images of a training batch. Tasks whose targets move with the images (e.g. bounding
        boxes, segmentation masks) override it to transform them too.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The batch with its augmented images.
        """
        augmented_images, _, _, _, _ = self.data_augmenter.augment_batch(images=batch["images"])
        return {**batch, "images": augmented_images}

    def compute_batch_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Computes metrics of a training or validation batch from the outputs of its forward pass (e.g. the accuracy of
        a classifier), tracked with the losses.

        Args:
            batch (Dict[str, Any]): The prepared batch.
            model_outputs (Dict[str, Any]): The outputs of forward_pass.

        Returns:
            Dict[str, torch.Tensor]: The scalar metrics of the batch, none by default.
        """
        return {}

    @contextmanager
    def evaluation_context(self) -> Iterator[None]:
        """
        Temporary settings applied to the model during the evaluation of the test split (e.g. a lower detection score
        threshold), restored afterwards. None by default.

        Yields:
            None: Control to the evaluation loop.
        """
        yield

    # ------------------------------------------------------------------------------------------------------------------
    # Generic pipeline
    # ------------------------------------------------------------------------------------------------------------------

    def step_optimizer(self, iteration: int) -> None:
        """
        Updates the model parameters after the backward pass of a training step: the gradients are clipped to
        'Optimization.gradient_clipping_norm' when it is set, the learning rate of every parameter group is set to
        its value of the schedule at this iteration, then the optimizer steps. The learning rate of the first group
        is written to TensorBoard.

        Args:
            iteration (int): The training iteration, giving the learning rate of the schedule.
        """
        # Clip the norm of all the gradients together, when asked
        if self.optimization_configuration["gradient_clipping_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                           max_norm=self.optimization_configuration["gradient_clipping_norm"])

        # Learning rate of the schedule, relative to the learning rate of every parameter group
        learning_rate_factor = compute_learning_rate_factor(iteration=iteration,
                                                            optimization_configuration=self.optimization_configuration)
        for parameter_group, base_learning_rate in zip(self.optimizer.param_groups, self.base_learning_rates):
            parameter_group["lr"] = base_learning_rate * learning_rate_factor
        self.optimizer.step()

        if self.training_writer is not None:
            self.training_writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], iteration)

    def initialise_pretrained_components(self) -> Dict[str, Optional[Path]]:
        """
        Loads the trained components of other runs into the model, for every section whose
        'pretrained.experiment_folder' is set (backbone, temporal module, classification head, segmentation decoder,
        region proposal heads, detection heads), and freezes them when asked, or only the scale and shift of the
        normalization layers of the backbone. A run resuming from its own checkpoints does not load them, its
        checkpoint holding the components it was training, but keeps them frozen.

        Returns:
            Dict[str, Optional[Path]]: The checkpoint file every component was loaded from, keyed by the section names
                of the task, None for a component that is not transferred or not loaded again.

        Raises:
            ValueError: If every parameter of the model is frozen, so that the training would change nothing.
        """
        pretrained_checkpoint_paths = {}
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            pretrained_checkpoint_paths[section_name] = None
            if pretrained_configuration["experiment_folder"] is None:
                continue
            component = getattr(self.model, TRANSFERABLE_COMPONENTS[section_name])

            # Load the source component for a new run only
            if self.resume_training and get_last_checkpoint_iteration(weights_directory=self.weights_directory):
                print_color(color="yellow", output=f"The Run Resumes From Its Own Checkpoint, The Pretrained "
                                                   f"{get_component_name(section_name=section_name).title()} Is Not "
                                                   f"Loaded Again.")
            else:
                pretrained_checkpoint_paths[section_name] = load_pretrained_component(
                    component=component, section_name=section_name, model_configuration=self.model_configuration,
                    device=self.device)

            # Freeze the whole component, or only the scale and shift of the normalization layers of the backbone
            if pretrained_configuration["freeze"]:
                freeze_component(component=component)
            elif section_name == "Backbone" and pretrained_configuration["freeze_normalization_parameters"]:
                freeze_normalization_parameters(component=component)

        # A model whose every parameter is frozen would not learn anything
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Every Parameter Of The Model Is Frozen By The 'pretrained.freeze' Settings, The Training "
                             "Would Change Nothing: Leave At Least One Component Trainable.")
        return pretrained_checkpoint_paths

    def setup_training_paths(self, initial_configuration_path: Path, create_run_folder: bool
                             ) -> Tuple[Path, Path, Path, Optional[ConfigurationChanges]]:
        """
        Sets up the directory structure and persistent files for training outputs.

        This method acts as the initialization step for experiment storage: it gives the paths of the `Tensorboard`
        and `Weights` directories and of the archived configuration within the run folder and, for a training, creates
        the directories and archives the configuration file of the run there before the training loop commences. A
        resumed run first compares the launched configuration with the saved one (see
        compare_with_saved_configuration), which the copy replaces. The input viewer, which exits without training,
        only needs the paths.

        Args:
            initial_configuration_path (Path): The original path of the experiment configuration file.
            create_run_folder (bool): True to create the directories and archive the configuration (a training), False
                to only give their paths (the input viewer).

        Returns:
            Tuple[Path, Path, Path, Optional[ConfigurationChanges]]: A tuple containing:
                - tensorboard_directory (Path): Path to the TensorBoard logs folder.
                - weights_directory (Path): Path to the saved model weights folder.
                - configuration_path (Path): Path to the archived configuration file.
                - configuration_changes (Optional[ConfigurationChanges]): The settings changed since the last start
                  of a resumed run, None for a run that does not resume from a checkpoint and for the input viewer.
        """
        tensorboard_directory = self.experiment_folder / "Tensorboard"
        weights_directory = self.experiment_folder / "Weights"
        new_configuration_path = self.experiment_folder / "training_configuration.yaml"
        if not create_run_folder:
            return tensorboard_directory, weights_directory, new_configuration_path, None

        # Create directories
        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        tensorboard_directory.mkdir(exist_ok=True, parents=True)
        weights_directory.mkdir(exist_ok=True, parents=True)

        # Settings changed since the last start of a resumed run, before the copy replaces the saved configuration
        configuration_changes = None
        if self.resume_training:
            configuration_changes = self.compare_with_saved_configuration(
                initial_configuration_path=initial_configuration_path, saved_configuration_path=new_configuration_path,
                weights_directory=weights_directory)

        # Save a copy of the configuration file
        copyfile(src=str(initial_configuration_path), dst=str(new_configuration_path))

        return tensorboard_directory, weights_directory, new_configuration_path, configuration_changes

    def compare_with_saved_configuration(self, initial_configuration_path: Path, saved_configuration_path: Path,
                                         weights_directory: Path) -> Optional[ConfigurationChanges]:
        """
        Compares the configuration a run is resumed with against the one saved in its folder by the previous start, so
        that the settings changed between two resumptions (learning rate, learning rate multipliers, batch size,
        dropout, augmentation...), which keep the run name, are printed at launch and stay traceable. When they differ,
        the saved file is kept as 'training_configuration_until_iteration_<n>.yaml', the configuration that trained the
        run up to the checkpoint of iteration n it resumes from. A file of that name is never replaced: a run resumed
        twice from the same checkpoint keeps the configuration that really trained it.

        Args:
            initial_configuration_path (Path): The configuration file the run is launched with.
            saved_configuration_path (Path): The configuration file saved in the run folder by the previous start.
            weights_directory (Path): The 'Weights' folder of the run, giving the checkpoint it resumes from.

        Returns:
            Optional[ConfigurationChanges]: The changed settings with the resumed iteration and the archived file, None
                when the run has no checkpoint or no saved configuration to resume from.
        """
        resumed_iteration = get_last_checkpoint_iteration(weights_directory=weights_directory)
        if not resumed_iteration or not saved_configuration_path.is_file():
            return None

        # Every setting whose value changed, was added or was removed since the previous start
        changed_settings = find_changed_settings(
            previous_configuration=load_configuration(configuration_path=saved_configuration_path),
            current_configuration=load_configuration(configuration_path=initial_configuration_path))
        if not changed_settings:
            return ConfigurationChanges(resumed_iteration=resumed_iteration, changed_settings=[],
                                        archived_configuration_path=None)

        # Keep the configuration that trained the run up to the resumed checkpoint, unless an earlier resume kept it
        archived_configuration_path = (saved_configuration_path.parent /
                                       f"training_configuration_until_iteration_{resumed_iteration}.yaml")
        if not archived_configuration_path.exists():
            copyfile(src=str(saved_configuration_path), dst=str(archived_configuration_path))
        return ConfigurationChanges(resumed_iteration=resumed_iteration, changed_settings=changed_settings,
                                    archived_configuration_path=archived_configuration_path)

    def print_experiment_information(self) -> None:
        """
        Prints out a summary of the experiment's configurations and paths to the console before training begins.
        This function ensures that the user is aware of where important outputs (like Tensorboard logs and weights)
        and inputs (datasets) are located, providing clickable links to easily navigate the project structure.
        """
        print_color(color="yellow", output="=== Experiment Launch Information ===", add_separators=True)
        print_color(color="blue", output=f"- Run Name                : {self.run_name}")
        print_color(color="blue", output=f"- Experiment Folder       : {format_path_link(path=self.experiment_folder)}")
        print_color(color="blue", output=f"- Tensorboard Directory   : "
                                         f"{format_path_link(path=self.tensorboard_directory)}")
        print_color(color="blue", output=f"- Weights Directory       : {format_path_link(path=self.weights_directory)}")
        print_color(color="blue", output=f"- Configuration File      : "
                                         f"{format_path_link(path=self.configuration_path)}")
        if self.diagnosis_modes["view_input_images"]:
            print_color(color="yellow", output="- Run Folder Not Created  : The Input Viewer Exits Without Training")

        # Settings changed since the last start of a resumed run, then every component initialised by another run
        self.print_configuration_changes()
        self.print_pretrained_components()

        # Print every split folder with its number of samples, described by the task, kept for the MLflow parameters
        self.number_samples_per_split = {}
        for split_label, split_key in [("Train Data Folder", "train_data_folder"),
                                       ("Validation Data Folder", "validation_data_folder"),
                                       ("Test Data Folder", "test_data_folder")]:
            number_samples, description = self.describe_split(
                split_data_folder=self.experiment_configuration[split_key])
            self.number_samples_per_split[split_key] = number_samples
            print_color(color="blue", output=f"- {split_label:<23} : "
                                             f"{format_path_link(path=self.experiment_configuration[split_key])} "
                                             f"({description})")

        # One epoch is a full pass over the training samples
        self.iterations_per_epoch = math.ceil(self.number_samples_per_split["train_data_folder"] / self.batch_size)
        print_color(color="blue", output=f"- Iterations Per Epoch    : {self.iterations_per_epoch}")
        print_color(color="blue", output=f"- Batch Size              : {self.batch_size}")
        print_color(color="blue", output=f"- Total Iterations        : {self.training_iterations}")
        print_color(color="blue", output=f"- Save And Evaluate Every : {self.weight_saving_iterations} iterations")
        print_color(color="blue", output=f"- Save Only Every         : {self.intermediate_weight_saving_iterations} "
                                         f"iterations")
        print_color(color="blue", output=f"- Learning Rate           : {self.learning_rate}")
        print_color(color="blue", output=f"- Optimizer               : {self.optimization_configuration['optimizer']} "
                                         f"(weight decay {self.optimization_configuration['weight_decay']})")
        print_color(color="blue", output=f"- Learning Rate Schedule  : "
                                         f"{self.optimization_configuration['learning_rate_schedule']} (warm-up "
                                         f"{self.optimization_configuration['warmup_iterations']} iterations)")
        print_color(color="blue", output=f"- Gradient Clipping Norm  : "
                                         f"{self.optimization_configuration['gradient_clipping_norm']}")
        print_color(color="blue", output=f"- Mixed Precision         : {self.mixed_precision}")
        print_color(color="blue", output=f"- Resume Training         : {self.resume_training}")
        print_color(color="blue", output=f"- Validation Iter. Ratio  : {self.validation_iteration_ratio}")

        # Backend of the MLflow tracking, the local database of the experiments root folder unless a server is set
        tracking_description = "Off"
        if self.tracking_configuration["activated"]:
            tracking_description, _ = resolve_tracking_uri(
                tracking_configuration=self.tracking_configuration,
                tracking_root=self.experiment_configuration["experiment_folder"])
        print_color(color="blue", output=f"- MLflow Tracking         : {tracking_description}")

    def print_configuration_changes(self) -> None:
        """
        Prints, in the experiment information of a resumed run, the settings changed since its last start (see
        compare_with_saved_configuration), so that a changed learning rate, learning rate multiplier or any other
        setting keeping the run name is visible at launch, with the file keeping the previous configuration. Nothing is
        printed for a run that does not resume from a checkpoint.
        """
        if self.configuration_changes is None:
            return
        resumed_iteration = self.configuration_changes["resumed_iteration"]
        changed_settings = self.configuration_changes["changed_settings"]
        if not changed_settings:
            print_color(color="blue", output=f"- Changed Settings        : None Since Iteration {resumed_iteration}")
            return

        # File keeping the previous configuration, then every changed setting aligned on the longest key
        archived_configuration_path = self.configuration_changes["archived_configuration_path"]
        print_color(color="yellow", output=f"- Previous Configuration  : "
                                           f"{format_path_link(path=archived_configuration_path)}")
        print_color(color="yellow", output=f"- Changed Settings        : Since Iteration {resumed_iteration}")
        key_width = max(len(setting_key) for setting_key, _, _ in changed_settings)
        for setting_key, previous_value, current_value in changed_settings:
            print_color(color="yellow", output=f"    - {setting_key:<{key_width}} : "
                                               f"{format_setting_value(value=previous_value)} -> "
                                               f"{format_setting_value(value=current_value)}")

    def print_pretrained_components(self) -> None:
        """
        Prints, in the experiment information, one block per component initialised by the trained one of another run
        (backbone, temporal module, classification head, segmentation decoder, region proposal heads, detection
        heads): its source run folder, then the
        checkpoint iteration it comes from, whether it is frozen, whether the normalization layers of the backbone are
        frozen, and the learning rate it learns at with its multiplier, which gives no token in the run name. A frozen
        component does not learn, so it has no learning rate line. Components trained from random weights print
        nothing.
        """
        pretrained_labels = {"Backbone": "Pretrained Backbone", "TemporalModel": "Pretrained Temp. Module",
                             "ClassificationHead": "Pretrained Class. Head",
                             "SegmentationDecoder": "Pretrained Seg. Decoder", "RegionProposal": "Pretrained RPN Heads",
                             "Detector": "Pretrained Det. Heads"}
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            if pretrained_configuration["experiment_folder"] is None:
                continue

            # Source run and checkpoint of the component, and whether it is frozen
            source_iteration = pretrained_configuration["iteration"]
            source_iteration_description = "last saved checkpoint" if source_iteration is None else source_iteration
            print_color(color="blue", output=f"- {pretrained_labels[section_name]:<23} : "
                                             f"{format_path_link(path=pretrained_configuration['experiment_folder'])}")
            print_color(color="blue", output=f"    - Source Iteration    : {source_iteration_description}")
            print_color(color="blue", output=f"    - Frozen              : {pretrained_configuration['freeze']}")

            # Scale and shift of the normalization layers kept fixed, a setting of the backbone only
            if section_name == "Backbone":
                print_color(color="blue", output=f"    - Norm. Layers Frozen : "
                                                 f"{pretrained_configuration['freeze_normalization_parameters']}")

            # Learning rate of the parameter group of a component that learns, the learning rate of the run times its
            # multiplier
            if not pretrained_configuration["freeze"]:
                learning_rate_multiplier = pretrained_configuration["learning_rate_multiplier"]
                print_color(color="blue", output=f"    - Learning Rate       : "
                                                 f"{self.learning_rate * learning_rate_multiplier:g} (multiplier "
                                                 f"{learning_rate_multiplier:g} x {self.learning_rate:g})")

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

    def view_input_batches(self) -> None:
        """
        Shows the training and validation samples exactly as the data loaders feed them to the network, the training
        ones augmented as during training, each with its targets, in two windows side by side. Enabled by the
        'view_input_images' diagnosis mode, it lets the user check the preprocessing, the label mapping and the data
        augmentation. Pressing ENTER closes the viewer, after which the program exits without training.
        """
        print_color(color="red", output="View Of The Training And Validation Images That Are Fed To The Neural Network",
                    add_separators=True)
        print_color(color="green", output="Press SPACE To Navigate, Press ENTER To Stop Viewing And Exit")

        # Step through the training and validation loaders together, until the user stops or either one runs out
        for training_batch, validation_batch in zip(self.train_dataloader, self.validation_dataloader):
            number_samples = min(training_batch["images"].shape[0], validation_batch["images"].shape[0])

            # Augment the training batch exactly as during training, so the viewer shows what the network receives
            training_batch = self.augment_batch(batch={**training_batch,
                                                       "images": training_batch["images"].to(dtype=torch.float32)})

            # Display the training and validation samples at the same batch index next to each other
            for sample_index in range(number_samples):
                if self.show_input_samples(training_batch=training_batch, validation_batch=validation_batch,
                                           sample_index=sample_index):
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
            maximum_consecutive_errors (int): The maximum number of consecutive exceptions allowed before exiting.
                Defaults to 10.

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
            training_batch, training_dataloader_iterator = self.get_next_batch(
                dataloader_iterator=training_dataloader_iterator, dataloader=self.train_dataloader)

            self.set_training_mode()
            self.optimizer.zero_grad()

            # Forward pass For Training
            training_loss, _ = self.run_model_iteration(batch=training_batch, writer=self.training_writer,
                                                        apply_augmentation=True, iteration=training_iteration,
                                                        tracker_dictionary=None)

            # Backward and Step
            training_loss.backward()
            self.step_optimizer(iteration=training_iteration)

            # Validation phase gated by validation_iteration_ratio
            if validation_enabled and training_iteration % self.validation_iteration_ratio == 0:
                # Get next validation elements
                validation_batch, validation_dataloader_iterator = self.get_next_batch(
                    dataloader_iterator=validation_dataloader_iterator, dataloader=self.validation_dataloader)

                self.model.eval()
                with torch.no_grad():
                    # Forward pass For Validation
                    _, _ = self.run_model_iteration(batch=validation_batch, writer=self.validation_writer,
                                                    apply_augmentation=False, iteration=training_iteration,
                                                    tracker_dictionary=None)

    def initialize_training_loop_state(self) -> Tuple[
            Iterator, Iterator, bool, Dict[str, float], Optional[Dict[str, float]], int]:
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
                - consecutive_errors (int): Counter for consecutive errors encountered during training.
        """
        # Get dataloader
        training_dataloader_iterator = iter(self.train_dataloader)
        validation_dataloader_iterator = iter(self.validation_dataloader)

        # Determine whether validation is enabled based on the configured ratio
        validation_enabled = self.validation_iteration_ratio > 0

        # Get the metric trackers of the training and validation batches, which count the batches of their interval
        training_trackers = self.metric_tracker.create_trackers()
        validation_trackers = self.metric_tracker.create_trackers() if validation_enabled else None

        consecutive_errors = 0

        return (training_dataloader_iterator, validation_dataloader_iterator, validation_enabled,
                training_trackers, validation_trackers, consecutive_errors)

    def run_training_loop(self, maximum_consecutive_errors: int = 5) -> None:
        """
        Trains the model (see run_training_iterations) and closes its MLflow run whatever ends the training: FINISHED
        once the training ends normally, KILLED when the user stops it (Ctrl+C), FAILED after too many consecutive
        errors or on an unexpected error, which is raised again once the run is closed.

        Args:
            maximum_consecutive_errors (int): The maximum number of consecutive exceptions allowed before the training
                loop exits.
        """
        run_status = FAILED_STATUS
        try:
            run_status = self.run_training_iterations(maximum_consecutive_errors=maximum_consecutive_errors)
        except KeyboardInterrupt:
            run_status = KILLED_STATUS
            raise
        finally:
            self.finish_experiment_tracking(run_status=run_status)

    def run_training_iterations(self, maximum_consecutive_errors: int) -> str:
        """
        Executes the main training loop: training steps, validation passes, console summaries, checkpoints (with the
        evaluation of the test split at every weight_saving_iterations checkpoint) and a final checkpoint, evaluated
        too when the run completes on a weight_saving_iterations iteration. The MLflow run of the training is opened
        once the loader workers exist, and receives the interval averages of every console summary and the test
        evaluations.

        Args:
            maximum_consecutive_errors (int): The maximum number of consecutive exceptions allowed before the training
                loop exits.

        Returns:
            str: The status of the MLflow run: FINISHED_STATUS, or KILLED_STATUS when the user stopped the training,
                or FAILED_STATUS when it stopped after too many consecutive errors.
        """
        # Initialize tensorboard writers right before training starts
        self.training_writer, self.validation_writer = self.setup_tensorboard_writers()

        # Initialize core loop state variables, dataloader iterators, and metric trackers
        (training_dataloader_iterator, validation_dataloader_iterator, validation_enabled,
         training_trackers, validation_trackers, consecutive_errors) = self.initialize_training_loop_state()

        # MLflow run of the training, opened after the loader worker processes are forked from this process so that
        # they never inherit the thread of the tracker
        self.start_experiment_tracking()

        # Status of the MLflow run set by a stop of the training before its last iteration
        stopped_status = None

        # The checkpoint of the last iteration is the final one, saved after its training step, so that a checkpoint
        # saved at the start of the last iteration is never overwritten after being evaluated
        last_iteration = self.training_iterations + self.start_iteration - 1
        training_completed = False

        for training_iteration in range(self.start_iteration, last_iteration + 1, 1):
            try:
                # Save and evaluate the model at every weight saving iteration, and only save it at the more
                # frequent intermediate weight saving iterations in between. The last iteration is left to the final
                # save after the loop
                is_last_iteration = training_iteration == last_iteration
                if not is_last_iteration and training_iteration % self.weight_saving_iterations == 0:
                    self.save_model(iteration=training_iteration)
                    # The checkpoint of the first iteration of a run is not evaluated: it is either the untrained
                    # model of a new run, or the already evaluated checkpoint a resumed run starts from
                    if training_iteration != self.start_iteration:
                        self.run_test_evaluation(iteration=training_iteration)
                elif not is_last_iteration and training_iteration % self.intermediate_weight_saving_iterations == 0:
                    self.save_model(iteration=training_iteration)

                # Get next training elements
                training_batch, training_dataloader_iterator = self.get_next_batch(
                    dataloader_iterator=training_dataloader_iterator, dataloader=self.train_dataloader)

                # Real frames of the training batch, counted for the frame throughput of the sequence tasks
                self.metric_tracker.count_frames(tracker_dictionary=training_trackers, batch=training_batch)

                self.set_training_mode()
                self.optimizer.zero_grad()

                # Forward pass For Training
                training_loss, _ = self.run_model_iteration(batch=training_batch, writer=self.training_writer,
                                                            apply_augmentation=True, iteration=training_iteration,
                                                            tracker_dictionary=training_trackers)

                # Backward and Step
                training_loss.backward()
                self.step_optimizer(iteration=training_iteration)

                # Validation phase gated by the configurable validation_iteration_ratio.
                # A ratio of 1 runs validation every iteration, 2 runs every 2nd, N runs every Nth.
                if validation_enabled and training_iteration % self.validation_iteration_ratio == 0:
                    self.model.eval()
                    with torch.no_grad():
                        validation_batch, validation_dataloader_iterator = self.get_next_batch(
                            dataloader_iterator=validation_dataloader_iterator, dataloader=self.validation_dataloader)
                        # Forward pass for validation, its metrics added to the sums of the interval
                        _, _ = self.run_model_iteration(batch=validation_batch, writer=self.validation_writer,
                                                        apply_augmentation=False, iteration=training_iteration,
                                                        tracker_dictionary=validation_trackers)

                # Console log dump, on the multiples of information_dump
                if training_iteration % self.information_dump == 0:
                    # Divide the sums of both splits by the batches the interval really ran: fewer than
                    # information_dump training batches in the first interval of a run resumed between two summaries
                    self.metric_tracker.average_trackers(tracker_dictionary=training_trackers)
                    if validation_trackers is not None:
                        self.metric_tracker.average_trackers(tracker_dictionary=validation_trackers)

                    # Interval averages sent to MLflow before the summary starts the trackers of the next interval
                    self.log_interval_metrics(iteration=training_iteration, training_trackers=training_trackers,
                                              validation_trackers=validation_trackers)

                    training_trackers = self.metric_tracker.print_interval_summary(
                        iterations=training_iteration, training_tracker_dictionary=training_trackers,
                        validation_tracker_dictionary=validation_trackers)

                    if validation_enabled:
                        validation_trackers = self.metric_tracker.create_trackers()

                # Reset the consecutive errors counter after a fully successful iteration, the run being completed once
                # its last iteration succeeds
                consecutive_errors = 0
                training_completed = is_last_iteration

            except KeyboardInterrupt:
                print_color(color="red",
                            output=f"Training Interrupted by User at iteration {training_iteration}.",
                            add_separators=True)
                stopped_status = KILLED_STATUS
                break
            except Exception as error:
                # Increment error counter when an exception occurs inside the training iteration loop
                consecutive_errors += 1

                if consecutive_errors > maximum_consecutive_errors:
                    print_color(color="red",
                                output=f"Critical Error: Over {maximum_consecutive_errors} consecutive errors "
                                       f"encountered during training iterations. Exiting.", add_separators=True)
                    print(f"Last exception: {error}")
                    stopped_status = FAILED_STATUS
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

        # Evaluate the final checkpoint of a completed run on a weight saving iteration, once it holds the weights
        # after the last training step
        if training_completed and training_iteration % self.weight_saving_iterations == 0:
            self.run_test_evaluation(iteration=training_iteration)

        if self.training_writer is not None:
            self.training_writer.close()
        if validation_enabled and self.validation_writer is not None:
            self.validation_writer.close()
        return FINISHED_STATUS if stopped_status is None else stopped_status

    def build_tracking_parameters(self) -> Dict[str, Any]:
        """
        Gathers the settings of the run logged as the parameters of its MLflow run, searched and compared in the
        MLflow table: every configuration section, and the values derived from them (number of parameters, samples of
        every split, iterations per epoch, device and library versions). Also used by the MLflow run of a standalone
        evaluation.

        Returns:
            Dict[str, Any]: The nested settings, flattened by the tracker.
        """
        # Size of the model, samples of the splits and environment of the training, derived from the settings
        model_parameters = list(self.model.parameters())
        device_name = torch.cuda.get_device_name(device=self.device) if self.device.type == "cuda" else str(
            self.device)
        derived_values = {"number_parameters": sum(parameter.numel() for parameter in model_parameters),
                          "number_trainable_parameters": sum(parameter.numel() for parameter in model_parameters
                                                             if parameter.requires_grad),
                          "number_training_samples": self.number_samples_per_split["train_data_folder"],
                          "number_validation_samples": self.number_samples_per_split["validation_data_folder"],
                          "number_test_samples": self.number_samples_per_split["test_data_folder"],
                          "iterations_per_epoch": self.iterations_per_epoch, "device": device_name,
                          "torch_version": torch.__version__, "cuda_version": torch.version.cuda}
        return {"ExperimentConfiguration": self.experiment_configuration, **self.model_configuration,
                "Derived": derived_values}

    def build_tracking_tags(self) -> Dict[str, str]:
        """
        Gives the tags identifying the MLflow run of the training: its task, its run name, its run folder (to find its
        checkpoints, which are not sent to MLflow) and the source run of every transferable component, as stored in
        its checkpoints. Also used by the MLflow run of a standalone evaluation.

        Returns:
            Dict[str, str]: The value of every tag.
        """
        # Identity of the run, and the source run of every transferable component, 'None' without transfer
        tags = {"task": self.TASK_NAME, "run_name": self.run_name,
                "run_folder": str(Path(self.experiment_folder).absolute())}
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            tags[f"pretrained_{TRANSFERABLE_COMPONENTS[section_name]}"] = str(pretrained_configuration[
                "experiment_folder"])
        return tags

    def start_experiment_tracking(self) -> None:
        """
        Opens the MLflow run of the training when 'tracking.activated' is True, at the start of the training loop: a
        resumed training continues the MLflow run recorded in its run folder ('mlflow_run.json'). The configuration of
        the run is sent as an artifact, under its own name at every resumption since its settings may have changed.
        """
        # MLflow tracker of the run, a DisabledTracker when the tracking is off or unavailable
        self.experiment_tracker = create_experiment_tracker(
            tracking_configuration=self.tracking_configuration,
            tracking_root=self.experiment_configuration["experiment_folder"], run_name=self.run_name,
            task=self.TASK_NAME, run_record_path=self.experiment_folder / RUN_RECORD_FILE_NAME)

        # A run resumed from its checkpoint of iteration n starts at n + 1
        resumed_iteration = self.start_iteration - 1 if self.start_iteration > 1 else None
        self.experiment_tracker.start(parameters=self.build_tracking_parameters(), tags=self.build_tracking_tags(),
                                      resumed_iteration=resumed_iteration)
        artifact_name = None if resumed_iteration is None else \
            f"training_configuration_resume_{resumed_iteration}.yaml"
        self.experiment_tracker.log_artifact(file_path=self.configuration_path, artifact_folder="configuration",
                                             artifact_name=artifact_name)

    def log_interval_metrics(self, iteration: int, training_trackers: Dict[str, Any],
                             validation_trackers: Optional[Dict[str, Any]]) -> None:
        """
        Sends the averages of a console log interval to the MLflow run of the training, at its last iteration: the
        training and validation losses and batch metrics under their display names ('train/<name>',
        'validation/<name>'), the throughput and duration of the interval (and its real frame throughput for the
        sequence tasks) and the learning rate of the first parameter group. The per-iteration curves stay in
        TensorBoard.

        Args:
            iteration (int): The last training iteration of the interval.
            training_trackers (Dict[str, Any]): The training averages of the interval, with its start time.
            validation_trackers (Optional[Dict[str, Any]]): The validation averages of the interval, None without
                validation; they are left out when no validation batch ran in the interval.
        """
        samples_per_second, duration = self.metric_tracker.compute_interval_statistics(
            training_tracker_dictionary=training_trackers)

        # Averages of the tracked metrics under their display names, then the speed and the learning rate
        metrics = {f"train/{display_name}": training_trackers[metric_key]
                   for metric_key, display_name in self.tracked_metrics_mapping.items()}
        if validation_trackers is not None and validation_trackers[INTERVAL_BATCHES_KEY] > 0:
            metrics.update({f"validation/{display_name}": validation_trackers[metric_key]
                            for metric_key, display_name in self.tracked_metrics_mapping.items()})
        metrics.update({"train/throughput": samples_per_second, "train/interval_seconds": duration,
                        "learning_rate": self.optimizer.param_groups[0]["lr"]})
        frames_per_second = self.metric_tracker.compute_frame_throughput(training_tracker_dictionary=training_trackers,
                                                                         duration=duration)
        if frames_per_second is not None:
            metrics["train/frame_throughput"] = frames_per_second
        self.experiment_tracker.log_metrics(metrics=metrics, step=iteration)

    def track_evaluation(self, iteration: int, results: EvaluationResults) -> None:
        """
        Sends the evaluation of a checkpoint to the MLflow run: the global scalars ('test/<scalar>'), the scalars of
        every class with the per-class report only ('test_class/<scalar>'), the best value so far of the headline
        metric with its iteration, and the small evaluation files of TRACKED_EVALUATION_FILES under
        'evaluation/iteration_<n>/'.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.
            results (EvaluationResults): The results of the evaluation.
        """
        # Global scalars, and the scalars of every class only with the per-class report (thousands for many classes)
        metrics = {f"test/{scalar_name}": value for scalar_name, value in results["scalars"].items()}
        if self.per_class_report:
            metrics.update({f"test_class/{scalar_name}": value
                            for scalar_name, value in results["class_scalars"].items()})
        self.experiment_tracker.log_metrics(metrics=metrics, step=iteration)
        self.experiment_tracker.update_best_metric(metric_name=results["headline_metric"],
                                                   value=results["scalars"][results["headline_metric"]],
                                                   step=iteration)
        # Small evaluation files written by the task for this checkpoint
        for file_key in TRACKED_EVALUATION_FILES:
            if file_key in results["files"]:
                self.experiment_tracker.log_artifact(file_path=results["files"][file_key],
                                                     artifact_folder=f"evaluation/iteration_{iteration}")

    def finish_experiment_tracking(self, run_status: str) -> None:
        """
        Closes the MLflow run of the training at its end, after sending Results.txt, the whole readable report of the
        run.

        Args:
            run_status (str): FINISHED_STATUS, KILLED_STATUS or FAILED_STATUS.
        """
        self.experiment_tracker.log_artifact(file_path=Path(self.experiment_folder) / "Results.txt",
                                             artifact_folder=None)
        self.experiment_tracker.close(status=run_status)

    def run_model_iteration(self, batch: Dict[str, Any], writer: Optional[SummaryWriter], iteration: int,
                            tracker_dictionary: Optional[Dict[str, Any]],
                            apply_augmentation: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes one forward pass of the model on a training or validation batch, the same for every task: the batch
        is prepared (and augmented for training), the model returns its losses, which are summed into the total loss,
        and the tracked metrics are accumulated and written to TensorBoard.

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a loader.
            writer (Optional[SummaryWriter]): TensorBoard writer for logging. If None, logging is skipped.
            iteration (int): The current training iteration step.
            tracker_dictionary (Optional[Dict[str, Any]]): The trackers of the interval, summing the metrics and
                counting the batches, or None to track nothing (benchmarking).
            apply_augmentation (bool): Whether to augment the batch, True for training batches only (validation and
                test batches must be evaluated on the original images). Augmentation also requires
                'Data.data_augmentation.activated'.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: A tuple containing the total loss and the model outputs.
        """
        # Move the batch to the device and precision of the model
        batch = self.prepare_batch(batch=batch)

        # Randomly augment the training batches, together with the targets that move with the images
        if apply_augmentation:
            batch = self.augment_batch(batch=batch)

        # Execute the forward pass through the model of the task, with the targets so that it returns its losses, in
        # bfloat16 when mixed precision is enabled
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.mixed_precision):
            model_outputs = self.forward_pass(batch=batch)

        # Extract all the losses of the model, and sum them into the total loss used for backpropagation
        metric_dictionary = {key: value for key, value in model_outputs.items() if key.endswith(LOSS_KEY_SUFFIX)}
        total_loss = sum(metric_dictionary.values())

        # Insert the total loss and the batch metrics of the task into the tracked metrics
        metric_dictionary["total_loss"] = total_loss
        metric_dictionary.update(self.compute_batch_metrics(batch=batch, model_outputs=model_outputs))

        # Add the metrics to the sums of the interval and log them to tensorboard (per-iteration)
        self.metric_tracker.accumulate(tracker_dictionary=tracker_dictionary, metric_dictionary=metric_dictionary)
        self.metric_tracker.log_to_tensorboard(writer=writer, iteration=iteration, metric_dictionary=metric_dictionary)

        return total_loss, model_outputs

    def run_test_evaluation(self, iteration: int) -> EvaluationResults:
        """
        Evaluates a saved checkpoint on the whole test split, at every weight_saving_iterations checkpoint, and
        reports it to the console, TensorBoard and the Results.txt file of the run. The model runs as in deployment,
        through the predict_batch hook and within the evaluation context of the task, and every test sample is
        evaluated exactly once. The images of the fixed test samples are drawn on the way and added to the results
        of the task, with a note and their folder.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation, for the tools comparing the checkpoints of a run.
        """
        self.model.eval()
        accumulator = self.create_evaluation_accumulator(iteration=iteration)
        evaluated_sample_identifiers = set()
        samples_folder = self.experiment_folder / "Evaluation" / f"Iteration_{iteration}" / "Samples"
        sample_images = {}

        # The evaluation settings of the task are always restored, even when the evaluation fails
        with self.evaluation_context(), torch.no_grad():
            for test_batch in tqdm(self.test_dataloader, total=len(self.test_dataloader),
                                   desc=f"Test Evaluation Iteration {iteration}"):
                test_batch = self.prepare_batch(batch=test_batch)
                predictions = self.predict_batch(batch=test_batch)

                # A loader filling its last batch with samples already given must not have them evaluated twice (the
                # loaders of the tasks give a smaller last batch instead)
                sample_indices = []
                for sample_index in range(test_batch["images"].shape[0]):
                    if SAMPLE_IDENTIFIERS_KEY in test_batch:
                        sample_identifier = int(test_batch[SAMPLE_IDENTIFIERS_KEY][sample_index])
                        if sample_identifier in evaluated_sample_identifiers:
                            continue
                        evaluated_sample_identifiers.add(sample_identifier)
                    sample_indices.append(sample_index)

                self.accumulate_evaluation_batch(accumulator=accumulator, batch=test_batch, predictions=predictions,
                                                 sample_indices=sample_indices)

                # Images of the fixed test samples of the batch, drawn while their inputs and predictions are at hand
                if self.sample_image_identifiers and SAMPLE_IDENTIFIERS_KEY in test_batch:
                    self.save_sample_images(batch=test_batch, predictions=predictions, sample_indices=sample_indices,
                                            samples_folder=samples_folder, sample_images=sample_images)

        # Results of the task, with the images of the fixed test samples reported at every evaluation
        results = self.compute_evaluation_results(accumulator=accumulator, iteration=iteration)
        results["sample_images"] = sample_images
        if sample_images:
            results["notes"].append(f"Sample Images : {len(sample_images)} fixed test {self.SAMPLE_NAME.lower()}")
            results["files"]["Sample Images"] = samples_folder
        self.report_evaluation(iteration=iteration, results=results)
        return results

    def save_sample_images(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_indices: List[int],
                           samples_folder: Path, sample_images: Dict[str, np.ndarray]) -> None:
        """
        Draws the fixed test samples of an evaluated test batch with the build_sample_image hook of the task, saves
        every image in the 'Samples' folder of the evaluated checkpoint, '<run>/Evaluation/Iteration_<n>/Samples/', and
        keeps it in RGB under the TensorBoard tag 'Samples/<sample name>', so that the same samples can be followed over
        the training in the Test TensorBoard writer.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The predictions of predict_batch.
            sample_indices (List[int]): The indices of the samples evaluated in this batch.
            samples_folder (Path): The 'Samples' folder of the evaluated checkpoint, created with its first image.
            sample_images (Dict[str, np.ndarray]): The RGB sample images keyed by TensorBoard tag, extended in place.
        """
        for sample_index in sample_indices:
            sample_identifier = int(batch[SAMPLE_IDENTIFIERS_KEY][sample_index])
            if sample_identifier not in self.sample_image_identifiers:
                continue

            # Image of the sample drawn by the task, saved in BGR and kept in RGB for TensorBoard
            sample_name = self.get_test_sample_name(sample_identifier=sample_identifier)
            sample_image = self.build_sample_image(batch=batch, predictions=predictions, sample_index=sample_index)
            samples_folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(filename=str(samples_folder / f"{sample_name}.png"), img=sample_image)
            sample_images[f"Samples/{sample_name}"] = cv2.cvtColor(src=sample_image, code=cv2.COLOR_BGR2RGB)

    def report_evaluation(self, iteration: int, results: EvaluationResults) -> None:
        """
        Reports the evaluation of a checkpoint to the console, the Test TensorBoard writer (created at the first
        evaluation) and the Results.txt file of the run, with its per-class outputs when the per-class report is on,
        then sends it to the MLflow run.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.
            results (EvaluationResults): The results of the evaluation.
        """
        if self.test_writer is None:
            self.test_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Test"))
        report_evaluation_results(results=results, iteration=iteration, test_writer=self.test_writer,
                                  results_path=Path(self.experiment_folder) / "Results.txt",
                                  per_class_report=self.per_class_report)
        self.track_evaluation(iteration=iteration, results=results)

    def save_model(self, iteration: int) -> None:
        """
        Saves a checkpoint of the model and optimizer, with the task, the run name, the identifier of the MLflow run
        ('mlflow_run_id', None without tracking or while the run is being opened) and the source run of every
        transferable component of the task ('pretrained_backbone', 'pretrained_temporal_module',
        'pretrained_classification_head', 'pretrained_segmentation_decoder', 'pretrained_region_proposer_dictionary',
        'pretrained_detector_heads_dictionary', None without transfer), and records it in
        the checkpoint registry of the run used to resume the training.

        Args:
            iteration (int): The current training iteration, used for naming the weight file.
        """
        # Task, run name, MLflow run and source run of every transferable component, stored with the weights
        metadata = {"task": self.TASK_NAME, "run_name": self.run_name,
                    "mlflow_run_id": self.experiment_tracker.get_run_id()}
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            metadata[f"pretrained_{TRANSFERABLE_COMPONENTS[section_name]}"] = pretrained_configuration[
                "experiment_folder"]
        model_directory = save_checkpoint(weights_directory=self.weights_directory, iteration=iteration,
                                          model=self.model, optimizer=self.optimizer, metadata=metadata)

        print("Checkpoint Folder :")
        print(f" - {format_path_link(path=model_directory)}")
        print(f" - Model Saved In : model_{iteration:06}.pt")

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
            index_iteration = get_last_checkpoint_iteration(weights_directory=self.weights_directory)

        # Despite trying to get the last model None was found
        if not index_iteration:
            # Components already loaded from other runs by initialise_pretrained_components, which the new model holds
            loaded_component_names = [get_component_name(section_name=section_name).title()
                                      for section_name, checkpoint_path in self.pretrained_checkpoint_paths.items()
                                      if checkpoint_path is not None]
            if loaded_component_names:
                message = (f"No Checkpoint Of This Run Found, The New Model Starts From Pretrained Components Of "
                           f"Other Runs : {', '.join(loaded_component_names)}."
                           f"\nEvery Other Layer Is Trained From Scratch.")
            else:
                message = ("No Initiation Model Weights Will Be Used..."
                           "\nWe Generate A New Model That Will Be Trained From Scratch.")
            print_color(color="green", output=message, add_separators=True)
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
        Loads the model weights and optimizer state from a specified iteration checkpoint of the run, with the tensors
        mapped to the currently active device (CPU, CUDA, or MPS).

        Args:
            iteration (int): The exact iteration number identifying the checkpoint to load.
        """
        checkpoint = load_checkpoint(weights_directory=self.weights_directory, iteration=iteration, device=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
