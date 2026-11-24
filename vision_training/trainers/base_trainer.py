"""
General trainer shared by every task: the run folder with the copy of the configuration and the record of the images of
every split, the training loop with its validation passes, the starting weights, the optimizer and learning rate
schedule, the moving average of the weights, the data augmentation with the strengths of the task, the checkpoints and
the resume, the evaluation of the saved checkpoints on the test split (trained and averaged weights), the MLflow
tracking of the run (vision_training/tracking), the input batch viewer and the input samples written at the start of
every training launch (trainers/input_samples.py). The task trainers subclass it and implement its abstract hooks, and
vision_training/main.py builds them through the task registry.
"""

import json
import math
import sys
import tempfile
import time
import traceback
import cv2
import numpy as np
import torch

from abc import ABC, abstractmethod
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from shutil import copyfile, rmtree
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from vision_training.configuration_processing.configuration_changes import (ConfigurationChanges, find_changed_settings,
                                                                            format_setting_value)
from vision_training.configuration_processing.configuration_loader import (inject_keypoint_categories,
                                                                           load_configuration)
from vision_training.configuration_processing.run_name import build_run_name
from vision_training.core.contracts import LOSS_KEY_SUFFIX, SAMPLE_IDENTIFIERS_KEY, EvaluationResults
from vision_training.data.augmentations import DataAugmenter
from vision_training.data.data_record import (CLASSES_RECORD_KEY, COMPARED_SPLIT_ENTRIES, DATA_RECORD_FILE_NAME,
                                              DATA_RECORD_SPLITS, RELABELLED_SPLIT_ENTRIES, DataRecordChanges,
                                              build_class_records, build_split_record, find_changed_splits)
from vision_training.data.label_mapping import (RUN_LABEL_DICTIONARY_FILE_NAME, read_run_label_dictionary,
                                                write_run_label_dictionary)
from vision_training.data.loaders import uses_pinned_memory
from vision_training.deployment.export_size_estimates import estimate_export_sizes, print_export_size_summary
from vision_training.deployment.export_specifications import (INFERENCE_MODEL_CLASSES, build_inference_model,
                                                              get_export_specification)
from vision_training.evaluation.evaluation_curves import (AVERAGED_WEIGHTS_COLUMN_PREFIX,
                                                          AVERAGED_WEIGHTS_UPDATES_COLUMN,
                                                          EVALUATION_CURVES_FIGURE_NAME, EVALUATION_SUMMARY_FILE_NAME,
                                                          build_summary_columns, draw_summary_curves,
                                                          seed_evaluation_summary, write_summary_columns)
from vision_training.evaluation.evaluation_report import report_evaluation_results
from vision_training.evaluation.report_formatting import HEADER_SEPARATOR
from vision_training.evaluation.sample_selection import choose_sample_identifiers
from vision_training.models.model_registry import MODEL_DTYPE
from vision_training.models.model_transfer import (LOADED_BACKBONE_BLOCKS_METADATA_KEY, TRANSFERABLE_COMPONENTS,
                                                   count_loaded_backbone_blocks, describe_block_range,
                                                   freeze_batch_normalization_statistics, freeze_component,
                                                   freeze_normalization_parameters, get_component_name,
                                                   load_pretrained_component, load_segmentation_class_layers,
                                                   select_transferred_part, transfers_class_layers)
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import (EXPONENTIAL_MOVING_AVERAGE_STATE_KEY, get_last_checkpoint_iteration,
                                                    load_checkpoint, read_checkpoint_value, save_checkpoint)
from vision_training.trainers.input_samples import (INPUT_SAMPLES_FOLDER_NAME, NUMBER_INPUT_SAMPLES,
                                                    list_input_sample_artifacts, prepare_input_samples_folder,
                                                    read_input_samples, write_input_sample_image,
                                                    write_input_samples_record)
from vision_training.trainers.learning_rate_schedule import compute_learning_rate_factor
from vision_training.trainers.metric_tracking import INTERVAL_BATCHES_KEY, MetricTracker
from vision_training.trainers.speed_benchmark import (BYTES_PER_GIGABYTE, FORMAT_FAILED_STATUS, INFERENCE_FORMATS,
                                                      MAXIMUM_BENCHMARK_BATCH_SIZE, MEASURED_STATUS,
                                                      NOT_AVAILABLE_STATUS, OUT_OF_MEMORY_STATUS, PROBE_ITERATIONS,
                                                      BenchmarkRow, MemoryBudget, build_inference_table,
                                                      build_training_table, describe_error, export_benchmark_formats,
                                                      find_largest_batch_size, is_out_of_memory_error,
                                                      join_header_parts, limit_process_memory,
                                                      list_other_gpu_processes, measure_iteration_time, move_batch,
                                                      open_export_format, print_speed_benchmark_report,
                                                      release_cached_memory, resize_batch)
from vision_training.trainers.weight_averaging import ExponentialMovingAverage
from vision_training.tracking.experiment_tracker import (FAILED_STATUS, FINISHED_STATUS, KILLED_STATUS,
                                                         RUN_RECORD_FILE_NAME, TRAINED_WEIGHTS_METRIC_PREFIX,
                                                         DisabledTracker, create_experiment_tracker,
                                                         resolve_tracking_uri)
from vision_training.utilities.os_utilities import format_path_link, print_color, read_json, release_freed_heap_memory
from vision_training.utilities.tensor_utilities import fork_random_state, get_device
from vision_training.utilities.visualization import ViewerWindow, view_input_data

# Evaluation files of a checkpoint sent to the MLflow run, small whatever the size of the test split: the confusion
# matrix image and the ROC and precision-recall curve images (per-class report only, with the zoom on the thresholds
# of the classification-like tasks), the metrics of every class, the class thresholds of the classification-like tasks
# and the proposal recall of the detection tasks. The predictions, the error images and the sample images grow with
# the test split and stay in the run folder.
TRACKED_EVALUATION_FILES = ["Confusion Matrix Image", "ROC Curves Image", "Precision Recall Curves Image",
                            "Precision Recall Thresholds Image", "Class Metrics", "Class Thresholds",
                            "Proposal Recall"]

# MLflow artifact folder of the files written at launch by the analysis of the training data (analyse_training_data)
TRAINING_DATA_ANALYSIS_ARTIFACT_FOLDER = "analysis"

# Outputs of the evaluation of the moving average of the weights, next to those of the trained weights: its folder in
# the evaluation folder of a checkpoint, its TensorBoard writer folder (its curves overlaid on the Test ones), the
# label of its Results.txt blocks and the prefix of its MLflow metrics
AVERAGED_WEIGHTS_FOLDER_NAME = "EMA"
AVERAGED_WEIGHTS_TENSORBOARD_FOLDER = "Test_EMA"

# Folder of the evaluations of the checkpoints of a training inside its run folder, '<run>/Evaluation/Iteration_<n>/';
# a standalone evaluation (scripts/evaluation/evaluate_checkpoint.py) keeps the iteration folders directly in its own
# folder instead, '<run>/Checkpoint_Evaluation/Iteration_<n>/'
EVALUATION_FOLDER_NAME = "Evaluation"
AVERAGED_WEIGHTS_LABEL = "EMA Weights"
AVERAGED_WEIGHTS_METRIC_PREFIX = "test_ema"

# Passes over a split ending without any batch after which the fetch of a batch gives up, an empty loader restarting
# forever otherwise (get_next_batch)
MAXIMUM_EMPTY_PASSES = 2

# Last fetches of a split over which its dropped batches are counted, and the number of them dropped above which the
# training stops (get_next_batch): more than 20 of the last 200 batches failing to load (e.g. 1 % of unreadable images
# in batches of 64 drop about 47 % of them) mean broken data rather than a few bad files
DROPPED_BATCH_WINDOW = 200
MAXIMUM_DROPPED_BATCHES_IN_WINDOW = 20


class NonFiniteLossError(RuntimeError):
    """
    Error of a training or validation batch whose total loss is NaN or infinite (run_model_iteration), raised before
    its metrics are tracked and before its backward pass, so that a diverging step or a corrupted batch never spreads
    NaN into the weights, the optimizer state and the checkpoints. The training loop counts it with the failing
    iterations (run_training_iterations): the batch is skipped, and too many in a row stop the training.
    """


class TrainingStopError(RuntimeError):
    """
    Error after which the training cannot go on, raised from its original error: a checkpoint that cannot be saved or
    a test evaluation that fails (stop_training_on_error, e.g. a full disk or an unreadable test image), or a split
    dropping too many batches (get_next_batch). The training loop stops at once on it with status FAILED instead of
    counting it with the failing iterations (run_training_iterations), which would otherwise go on training without
    checkpoints or evaluations.
    """


@contextmanager
def stop_training_on_error(failure_description: str) -> Iterator[None]:
    """
    Runs a step of the training loop whose failure must stop the training (the save of a checkpoint, the evaluation of
    the test split), turning any error it raises into a TrainingStopError that names the step, chained to the original
    error so that its traceback is printed. A Ctrl+C is left as it is.

    Args:
        failure_description (str): What failed, opening the message of the error, e.g. 'The checkpoint of iteration 4
            could not be saved'.

    Yields:
        None: Control to the step of the training loop.

    Raises:
        TrainingStopError: If the step raises an error.
    """
    try:
        yield
    except Exception as error:
        raise TrainingStopError(f"{failure_description}: {type(error).__name__}: {error}") from error


class BaseTrainer(ABC):
    """
    General training pipeline shared by every task of the repository (object detection, classification, semantic
    segmentation, sequence classification, sequence labelling and sequence comparison).

    It owns everything that does not depend on the task: the run folder named after the settings, the training loop
    with its validation passes and checkpoint schedule, the metric tracking (console, TensorBoard and MLflow), the
    checkpoints and the resume, the evaluation loop of the saved checkpoints on the test split and its report, the
    viewer of the input batches and the input samples written at the start of every training launch. A task
    subclasses it and implements the hooks that depend on its data and its model (marked abstract), and may override
    the optional ones (augmentation, batch metrics, evaluation context, optimizer...).
    """

    # Name of the task trained by the subclass, stored in its checkpoints
    TASK_NAME = ""

    # Kind of sample of a batch, "Images" for the image tasks, "Sequences" for the sequence tasks or "Pairs" for the
    # sequence comparison task
    SAMPLE_NAME = "Images"

    # Whether the training loader leaves out the last incomplete batch of every epoch, True for the self-supervised
    # task whose losses compare the samples of a full batch, so that an epoch has one iteration less; the
    # classification trainer decides it from its settings instead (drops_last_training_batch)
    DROPS_LAST_TRAINING_BATCH = False

    # Whether the real frames of the training batches are counted for a frame throughput, True for the sequence tasks
    COUNTS_FRAMES = False

    # Whether the data augmentation draws the views of the self-supervised task, with the stronger strengths of
    # SELF_SUPERVISED_AUGMENTATION_STRENGTHS, instead of augmenting the training batches of a supervised task
    SELF_SUPERVISED_AUGMENTATION = False

    # Whether the data augmentation reads the sequence settings of the configuration ('independent_frames' and
    # 'shuffle_frames'), held by the sequence tasks only, True for the sequence tasks
    SEQUENCE_AUGMENTATION = False

    # Splits whose input samples (save_input_samples) are augmented as the network receives them: the training split,
    # the self-supervised task adding the validation split, whose passes draw the two views of every image as well
    AUGMENTED_INPUT_SPLITS = ["Training"]

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

        # Every run trains in a folder named after its settings and its run name suffix, inside the experiments root
        # folder of the configuration, so that the same settings resume in the same folder and changed settings start a
        # new one. A standalone evaluation gives the name of the evaluated run folder instead ('run_name')
        self.run_name = self.experiment_configuration["run_name"]
        if self.run_name is None:
            self.run_name = self.get_run_name()
        self.experiment_folder = self.experiment_configuration["experiment_folder"] / self.run_name

        # A standalone evaluation writes in its own folder inside the evaluated run ('<run>/Checkpoint_Evaluation/'),
        # its iteration folders directly in it; a training keeps them in the 'Evaluation' folder of its run folder
        standalone_evaluation_folder = self.experiment_configuration["standalone_evaluation_folder"]
        self.standalone_evaluation = standalone_evaluation_folder is not None
        if self.standalone_evaluation:
            self.experiment_folder = self.experiment_folder / standalone_evaluation_folder
        self.evaluation_root = (self.experiment_folder if self.standalone_evaluation
                                else self.experiment_folder / EVALUATION_FOLDER_NAME)

        # Basic training parameters
        self.training_iterations = self.experiment_configuration["number_iterations"]
        self.weight_saving_iterations = self.experiment_configuration["weight_saving_iterations"]
        self.intermediate_weight_saving_iterations = self.experiment_configuration[
            "intermediate_weight_saving_iterations"]
        self.validation_iteration_ratio = self.experiment_configuration["validation_iteration_ratio"]
        self.information_dump = self.experiment_configuration["information_dump"]
        self.learning_rate = self.experiment_configuration["learning_rate"]
        self.diagnosis_modes = self.experiment_configuration["diagnosis_modes"]

        # Diagnosis modes ending the program before the training (the input viewer, the anchor coverage of the
        # detection tasks): the run folder, its data record and the resume checks are then left untouched
        self.exits_after_diagnosis = self.uses_exiting_diagnosis_mode()

        # The validation loader is built only when the training reads validation batches, or for a diagnosis mode
        # ending the program, which shows or measures the validation split: otherwise no validation batch is loaded
        # and no worker process holds validation samples (build_dataloaders gives None in its place)
        self.builds_validation_loader = self.validation_iteration_ratio > 0 or self.exits_after_diagnosis

        # Whether the evaluation reports show every class (table, TensorBoard scalars, confusion matrix image) or only
        # the global metrics, for the runs with many classes
        self.per_class_report = self.experiment_configuration["evaluation"]["per_class_report"]

        # Number of fixed test samples drawn with their targets and predictions at every evaluation, 0 for none
        self.number_sample_images = self.experiment_configuration["evaluation"]["number_sample_images"]

        # MLflow tracking of the run, opened at the start of the training loop: until then, and for the input viewer and
        # the trainer of a standalone evaluation, the tracker does nothing
        self.tracking_configuration = self.experiment_configuration["tracking"]
        self.experiment_tracker = DisabledTracker()

        # Random augmentation of the training batches, or views of the self-supervised task, applied on their device
        # right before the forward pass with the strengths of the task
        self.data_augmenter = DataAugmenter(augmentation_configuration=self.data_configuration["data_augmentation"],
                                            self_supervised=self.SELF_SUPERVISED_AUGMENTATION,
                                            sequence_augmentation=self.SEQUENCE_AUGMENTATION)
        self.input_image_size = self.data_configuration["image_settings"]["size"]
        self.batch_size = self.experiment_configuration["batch_size"]
        self.allow_data_change_on_resume = self.experiment_configuration["allow_data_change_on_resume"]

        # Classes of the run: those saved in its folder for a resumed run, the mapping file giving them otherwise
        self.restore_run_label_dictionary()
        self.label_dictionary = self.data_configuration["label_dictionary"]

        # A resume on changed labels is refused before the run folder is touched, changed data asking for a new run
        if not self.exits_after_diagnosis:
            self.check_resumed_data()

        # Paths of the run folder, created with the copy of the configuration only for a training: the diagnosis modes
        # exit without training, so they leave no empty run folder and never overwrite the configuration of a run. A
        # resumed run also gets the settings changed since its last start, printed with the experiment information
        (self.tensorboard_directory, self.weights_directory, self.configuration_path,
         self.configuration_changes) = self.setup_training_paths(
            initial_configuration_path=Path(configuration_path), create_run_folder=not self.exits_after_diagnosis)

        # We use lazy initialization for the tensorboard writers to avoid prematurely creating
        # empty log files on disk during the Trainer instantiation.
        self.training_writer = None
        self.validation_writer = None
        self.test_writer = None
        self.averaged_test_writer = None

        # Batches dropped since the last console summary because their loading failed, by split, and the errors
        # already printed, every distinct one being printed once in the run (get_next_batch)
        self.dropped_batch_counts = Counter()
        self.reported_fetch_errors = set()

        # Outcomes of the last DROPPED_BATCH_WINDOW fetches of every split, True for a dropped batch, which stop the
        # training when too many of them were dropped (get_next_batch)
        self.recent_fetch_drops: Dict[str, deque] = {}

        # Errors of failed training iterations whose traceback was already printed, every distinct one being printed
        # whole once in the run (report_iteration_error)
        self.reported_iteration_errors = set()

        # Setting up dataloaders
        print_color(color="blue",
                    output="Initializing Training, Validation And Test DataLoaders...", add_separators=True)
        self.train_dataloader, self.validation_dataloader, self.test_dataloader = self.build_dataloaders()
        if not self.builds_validation_loader:
            print_color(color="blue", output="- Validation Loader Not Built : 'validation_iteration_ratio' Is 0, No "
                                             "Validation Batch Is Loaded")
        self.check_loaders_give_batches()

        # Record of the images of every split next to the configuration, written for a training only like the run
        # folder, a resumed run whose data changed keeping the record of its previous data
        self.data_record_path = self.experiment_folder / DATA_RECORD_FILE_NAME
        self.data_record_changes = None
        if not self.exits_after_diagnosis:
            self.data_record_changes = self.save_data_record(data_record_path=self.data_record_path)

        # MLflow artifact folder of the input samples written by this launch (save_input_samples), None until they are
        # written and when the launch keeps those of its previous start
        self.input_samples_artifact_folder = None

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
        self.model = self.build_model()
        self.model.to(device=self.device, dtype=self.dtype)

        # Components transferred from other runs, replacing the PyTorch initialisation of their layers; the blocks of a
        # backbone deeper than its source backbone loaded from it are counted there (None for a whole backbone)
        self.loaded_backbone_blocks = None
        self.pretrained_checkpoint_paths = self.initialise_pretrained_components()
        self.optimizer = self.build_optimizer()

        # Moving average of the weights, started from the weights of the model once the transferred components are
        # loaded (a resume replaces it by the one of its checkpoint), and evaluated next to the trained weights
        self.exponential_moving_average_configuration = self.optimization_configuration["exponential_moving_average"]
        self.exponential_moving_average = None
        if self.exponential_moving_average_configuration["activated"]:
            self.exponential_moving_average = ExponentialMovingAverage(
                model=self.model, decay=self.exponential_moving_average_configuration["decay"])

        # Whether the test evaluation runs on the averaged weights (see use_averaged_weights) or the trained ones
        self.evaluating_averaged_weights = False

        # Learning rate of every parameter group before the schedule, which multiplies it at every step
        self.base_learning_rates = [parameter_group["lr"] for parameter_group in self.optimizer.param_groups]

        # Weight of every loss of the model in the total loss, None when every loss counts once
        self.loss_weights = self.get_loss_weights()

        # Metric tracking of the losses and batch metrics chosen by the task, its rates printed as percentages
        self.tracked_metrics_mapping = self.get_tracked_metrics()
        self.metric_tracker = MetricTracker(tracked_metrics_mapping=self.tracked_metrics_mapping,
                                            batch_size=self.batch_size,
                                            validation_iteration_ratio=self.validation_iteration_ratio,
                                            sample_name=self.SAMPLE_NAME, counts_frames=self.COUNTS_FRAMES,
                                            percent_metric_keys=self.get_percent_tracked_metrics())

        # Print experiment information to user so that they can know everything about the experiment
        # as well as input and output shapes of the model.
        self.print_experiment_information()
        self.model.print_summary()

        # Architecture diagnostics printed before training continues: the optional receptive field, the model size, then
        # the size of every file the model is exported to for the tasks Triton serves
        if self.diagnosis_modes["compute_receptive_field"]:
            self.model.print_receptive_field_summary()
        self.model.print_model_size_summary()
        if self.TASK_NAME in INFERENCE_MODEL_CLASSES:
            print_export_size_summary(export_sizes=estimate_export_sizes(
                task=self.TASK_NAME, model=self.model, optimizer=self.optimizer,
                keeps_moving_average=self.exponential_moving_average is not None))

        # Optional diagnoses ending the program without training nor creating the run folder: the analysis of every
        # split against the model (the anchor coverage of the detection tasks), the viewer of the training and
        # validation samples fed to the network, then the speed benchmark of the training step and of the inference
        if self.exits_after_diagnosis:
            self.diagnose_data_splits()
            if self.diagnosis_modes["view_input_images"]:
                self.view_input_batches()
            if self.diagnosis_modes["benchmark_speed"]:
                self.benchmark_speed()
            sys.exit()

        # Analysis of the training data against the model printed before the training (the anchor recall of the
        # detection tasks), its files written to the run folder at every launch and sent to MLflow with the
        # configuration
        self.training_data_analysis_files = self.analyse_training_data()

        # Restoration of the last checkpoint of the run folder before starting any training or inference, the first
        # iteration for a new run
        self.start_iteration = self.restore_last_model()

        # Heap memory freed during the launch given back before the loader workers are forked, the manifests shared
        # by read_manifests_once (main.py) being given back when it ends, right after this construction
        release_freed_heap_memory()

    # ------------------------------------------------------------------------------------------------------------------
    # Task hooks: every task implements them
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def build_dataloaders(self) -> Tuple[Iterable, Optional[Iterable], Iterable]:
        """
        Builds the loaders of the training, validation and test splits of the task, the validation one only when
        builds_validation_loader is True (a training reading validation batches, or a diagnosis mode ending the
        program), None otherwise, so that no validation batch is loaded and no worker process started for it
        (check_loaders_give_batches refuses any other case).

        Returns:
            Tuple[Iterable, Optional[Iterable], Iterable]: The training, validation (None when it is not built) and
                test loaders, yielding batch dictionaries.
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

    def get_loss_weights(self) -> Optional[Dict[str, float]]:
        """
        Gives the weight of every loss of the model in the total loss of run_model_iteration, keyed by the name of the
        loss output, read once when the trainer is built. Every loss counts once by default; the object detection and
        instance segmentation trainers read 'Optimization.loss_weights'.

        Returns:
            Optional[Dict[str, float]]: The weight of every loss output, or None when every loss counts once.
        """
        return None

    def get_percent_tracked_metrics(self) -> List[str]:
        """
        Lists the tracked batch metrics that are rates written as percentages in the console summary ("94.03 %", their
        changes "+0.52 %"), read once when the trainer is built. None by default, for the trainers tracking only
        losses, every tracked value keeping 4 decimals; every trainer tracking a rate lists it, such as the accuracy of
        the classification trainer, inherited by the sequence and frame selection trainers.

        Returns:
            List[str]: The tracked keys (keys of get_tracked_metrics) written as percentages.
        """
        return []

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
    def build_input_windows(self, batch: Dict[str, Any], sample_index: int, split_name: str) -> List[ViewerWindow]:
        """
        Draws one sample as the network receives it, for the windows of the input viewer ('view_input_images',
        show_input_samples) and the images of the input samples written at the start of every training launch
        (save_input_samples): the image with its target drawn on it (boxes, mask, keypoints, frame grid...), next to
        an information panel holding its target and its augmentations (describe_input_augmentations). One window for
        most tasks, one per view for the self-supervised task.

        Args:
            batch (Dict[str, Any]): The batch holding the sample, augmented for an augmented split, on the CPU.
            sample_index (int): The index of the sample in the batch.
            split_name (str): "Training", "Validation" or "Test", the title of the window.

        Returns:
            List[ViewerWindow]: The windows of the sample.
        """

    # ------------------------------------------------------------------------------------------------------------------
    # Optional hooks: the default behaviour suits most tasks
    # ------------------------------------------------------------------------------------------------------------------

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays one training sample (augmented) next to one validation sample for the 'view_input_images' diagnosis
        mode, each drawn by build_input_windows next to an information panel with its targets and its augmentations.
        The self-supervised trainer shows the two views of the training image only.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed samples in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        windows = [window for split_name, batch in [("Training", training_batch), ("Validation", validation_batch)]
                   for window in self.build_input_windows(batch=batch, sample_index=sample_index,
                                                          split_name=split_name)]
        return view_input_data(windows=windows)

    def get_run_name(self) -> str:
        """
        Builds the name of the run folder from the settings of the run, followed by the run name suffix of the
        configuration when it is not null.

        Returns:
            str: The run name.
        """
        return build_run_name(experiment_configuration=self.experiment_configuration,
                              model_configuration=self.model_configuration)

    def record_split(self, split_data_folder: Path) -> Dict[str, Any]:
        """
        Describes one split for the record of the data of the run ('data.json', see save_data_record): its data
        folder, the SHA-256 of its 'labels.json' and its images, read from the labels file whatever the task. A task
        whose splits have no labels file overrides it.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Dict[str, Any]: The 'data_folder', 'labels_sha256', 'number_images' and 'images' of the split.
        """
        return build_split_record(split_data_folder=split_data_folder)

    def show_input_batch(self, training_batch: Dict[str, Any], batch_index: int) -> bool:
        """
        Shows a whole training batch in the input viewer ('view_input_images') before its samples, as the loader gives
        it: the self-supervised task shows the mosaic of a batch built by its GroupedBatchSampler, the other tasks
        show nothing.

        Args:
            training_batch (Dict[str, Any]): The training batch as the loader gives it, on the CPU.
            batch_index (int): The position of the batch in the pass of the viewer over the training split.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        return False

    def list_class_names(self) -> Optional[List[str]]:
        """
        Gives the class names of the run in class index order (background first for the detection and segmentation
        tasks), written in the record of its data ('data.json', see save_data_record) and compared with those of the
        source run when a component with class layers is transferred, so that a run can add classes after those of
        its source run or hold them at other indices under the same names. Built by the tasks after their data
        loaders; None for a task without class names.

        Returns:
            Optional[List[str]]: The class names, None when the task has none.
        """
        return None

    def get_unmapped_class_index(self) -> Optional[int]:
        """
        Gives the class the loaders give to the categories the label mapping does not map, so that the counts of the
        classes of the record of the data ('data.json', see save_data_record) are those the model trains on. Only the
        frame selection gives them a class (its unselected class); the other tasks leave them out.

        Returns:
            Optional[int]: The class index of the unmapped categories, None when they are left out.
        """
        return None

    def analyse_training_data(self) -> Dict[str, Path]:
        """
        Analyses the training split against the built model at every launch of a training, before the first
        iteration, and writes its findings to the run folder: the detection tasks measure how many training objects
        their anchors can cover (anchor_recall.csv), so that a badly sized anchor setting is seen before training on
        it. Nothing for the other tasks.

        Returns:
            Dict[str, Path]: The written files keyed by display name, sent to MLflow by start_experiment_tracking.
        """
        return {}

    def uses_exiting_diagnosis_mode(self) -> bool:
        """
        Tells whether a diagnosis mode of 'ExperimentConfiguration.diagnosis_modes' ends the program before the
        training, so that the trainer creates no run folder and writes nothing to an existing one: the input viewer
        ('view_input_images') and the speed benchmark ('benchmark_speed') for every task, the detection tasks adding
        their anchor coverage ('compute_anchor_coverage').

        Returns:
            bool: Whether the program exits after the diagnosis.
        """
        return self.diagnosis_modes["view_input_images"] or self.diagnosis_modes["benchmark_speed"]

    def diagnose_data_splits(self) -> None:
        """
        Analyses every split against the built model in a diagnosis mode ending the program, printed before the input
        viewer when it is also on: the detection tasks print the anchor coverage of their training, validation and
        test objects ('compute_anchor_coverage'). Nothing for the other tasks.
        """
        return None

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
        heads, detection heads, mask heads) that is not frozen gets its own parameter group at the learning rate times
        its 'pretrained.learning_rate_multiplier', so that it can be fine-tuned at another speed than the new layers.
        The group is created even with a multiplier of 1, so that the layout of the groups only depends on the transfers
        and freezes written in the run name, and the multiplier can change between two resumptions of a run without
        changing the layout of its checkpoints. The first group holds every other parameter: without such a component,
        every parameter shares this single group, the layout of the checkpoints of the runs without transfer, and a
        transferred backbone comes second, the layout of the runs that only transferred their backbone. A backbone
        deeper than its source backbone puts only its loaded blocks in its group (see get_transferred_part), its next
        blocks learning at the learning rate of the run in the first group, as the other new layers, and so does a
        classification head loaded without its class layer ('ClassificationHead.pretrained.class_layer' False), its
        class layer learning in the first group.

        Returns:
            torch.optim.Optimizer: The optimizer at the configured learning rate.
        """
        # One group per transferred component that learns, in the order of TRANSFERABLE_COMPONENTS
        component_groups = []
        component_parameter_identifiers = set()
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            if pretrained_configuration["experiment_folder"] is None or pretrained_configuration["freeze"]:
                continue
            component_parameters = list(self.get_transferred_part(section_name=section_name).parameters())
            component_parameter_identifiers.update(id(parameter) for parameter in component_parameters)
            component_groups.append({"params": component_parameters, "lr": self.learning_rate *
                                     pretrained_configuration["learning_rate_multiplier"]})

        # Every other parameter learns at the learning rate of the run, in the first group
        other_parameters = [parameter for parameter in self.model.parameters()
                            if id(parameter) not in component_parameter_identifiers]
        parameter_groups = [{"params": other_parameters}] + component_groups
        return self.create_optimizer(parameter_groups=parameter_groups)

    def create_optimizer(self, parameter_groups: List[Dict[str, Any]]) -> torch.optim.Optimizer:
        """
        Creates the optimizer of 'Optimization.optimizer' with its weight decay over parameter groups, at the learning
        rate of the run for the groups without their own: Adam (weight decay added to the gradients), AdamW (weight
        decay decoupled from the gradients) or SGD with Nesterov momentum 0.9. Used for the model by build_optimizer.

        Args:
            parameter_groups (List[Dict[str, Any]]): The parameter groups, each with its 'params' and optionally its
                own 'lr'.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
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
        runs (see freeze_batch_normalization_statistics); only the loaded blocks of a backbone deeper than its source
        backbone are frozen (see get_transferred_part).
        """
        self.model.train()
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            if pretrained_configuration["freeze"]:
                freeze_batch_normalization_statistics(component=self.get_transferred_part(section_name=section_name))

    def set_evaluation_mode(self) -> None:
        """
        Puts the model in evaluation mode before every validation pass, its batch normalization layers using their
        running statistics.
        """
        self.model.eval()

    def zero_gradients(self) -> None:
        """
        Clears the gradients of the model before every training step.
        """
        self.optimizer.zero_grad()

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the images of a training batch. Tasks whose targets move with the images (e.g. bounding
        boxes, segmentation masks) override it to transform them too. The augmentation report of every sample is kept
        in the batch for the information panels of the input viewer.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The batch with its augmented images and their 'augmentation_reports'.
        """
        augmented_images, _, _, _, _, _, augmentation_reports = self.data_augmenter.augment_batch(
            images=batch["images"])
        return {**batch, "images": augmented_images, "augmentation_reports": augmentation_reports}

    def augment_input_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augments a batch shown by the input viewer ('view_input_images') or drawn as the input samples of a training
        launch (save_input_samples) as the network receives it: through augment_batch, as the training step does. The
        self-supervised trainer, whose augment_batch leaves the batch unchanged, draws the two views of its images
        instead, as its forward pass does.

        Args:
            batch (Dict[str, Any]): The batch of an augmented split (AUGMENTED_INPUT_SPLITS).

        Returns:
            Dict[str, Any]: The augmented batch, with the augmentation reports of its samples.
        """
        return self.augment_batch(batch=batch)

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

    def build_inference_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Gives the inputs of the inference model of the task (deployment/export_specifications.py) from a prepared
        batch, keyed by the names of TASK_INPUT_NAMES, so that the speed benchmark (benchmark_speed) times the eager
        inference model and every exported format on the batch of the training loader: its 'images' by default, the
        float32 BGR images (or frames) in [0, 255] the exports receive. The sequence comparison overrides it with the
        two members of its pairs.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, torch.Tensor]: The input of every argument of the inference model.
        """
        return {"images": batch["images"]}

    # ------------------------------------------------------------------------------------------------------------------
    # Generic pipeline
    # ------------------------------------------------------------------------------------------------------------------

    def step_optimizer(self, iteration: int) -> None:
        """
        Updates the model parameters after the backward pass of a training step, with the gradients as the backward
        pass left them: the learning rate of every parameter group is set to its value of the schedule at this
        iteration, then the optimizer steps and the moving average of the weights, when the run keeps one, moves
        towards the updated weights. The learning rate of the first group is written to TensorBoard.

        Args:
            iteration (int): The training iteration, giving the learning rate of the schedule.
        """
        # Learning rate of the schedule, relative to the learning rate of every parameter group
        learning_rate_factor = compute_learning_rate_factor(iteration=iteration,
                                                            optimization_configuration=self.optimization_configuration)
        for parameter_group, base_learning_rate in zip(self.optimizer.param_groups, self.base_learning_rates):
            parameter_group["lr"] = base_learning_rate * learning_rate_factor
        self.optimizer.step()

        # Move the moving average of the weights towards the updated weights
        if self.exponential_moving_average is not None:
            self.exponential_moving_average.update(model=self.model)

        if self.training_writer is not None:
            self.training_writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], iteration)

    def initialise_pretrained_components(self) -> Dict[str, Optional[Path]]:
        """
        Loads the trained components of other runs into the model, for every section whose
        'pretrained.experiment_folder' is set (backbone, temporal module, classification head, segmentation decoder,
        region proposal heads, detection heads, mask heads), and freezes them when asked, or only the scale and shift of
        the normalization layers of the backbone. The class names of the run are given to the transfer, so that a run
        may add classes after those of its source run or hold them at other indices under the same names, and the
        class layers of the semantic segmentation are loaded
        with its decoder when 'SegmentationDecoder.pretrained.load_class_layers' is True (they are never frozen, the
        new classes having to learn). A run resuming from its own checkpoints does not load them, its checkpoint holding
        the components it was training, but keeps them frozen. A backbone deeper than its source backbone loads its
        first blocks only, counted in loaded_backbone_blocks (from the configuration of the source run for a new run,
        from the 'pretrained_backbone_blocks' of its last checkpoint for a resumed run, whose source run may be gone):
        only these blocks are frozen, their next blocks learning (see get_transferred_part). A classification head
        loaded without its class layer ('ClassificationHead.pretrained.class_layer' False) freezes its hidden layers
        only, its random class layer learning.

        Returns:
            Dict[str, Optional[Path]]: The checkpoint file every component was loaded from, keyed by the section names
                of the task, None for a component that is not transferred or not loaded again.

        Raises:
            ValueError: If every parameter of the model is frozen, so that the training would change nothing.
        """
        pretrained_checkpoint_paths = {}
        last_checkpoint_iteration = get_last_checkpoint_iteration(weights_directory=self.weights_directory)
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            pretrained_checkpoint_paths[section_name] = None
            if pretrained_configuration["experiment_folder"] is None:
                continue
            component = getattr(self.model, TRANSFERABLE_COMPONENTS[section_name])

            # Load the source component for a new run only, a resumed run reading the backbone blocks it loaded from
            # its own checkpoint
            if last_checkpoint_iteration:
                print_color(color="yellow", output=f"The Run Resumes From Its Own Checkpoint, The Pretrained "
                                                   f"{get_component_name(section_name=section_name).title()} Is Not "
                                                   f"Loaded Again.")
                if section_name == "Backbone":
                    self.loaded_backbone_blocks = read_checkpoint_value(
                        weights_directory=self.weights_directory, iteration=last_checkpoint_iteration,
                        key_name=LOADED_BACKBONE_BLOCKS_METADATA_KEY)
            else:
                pretrained_checkpoint_paths[section_name] = load_pretrained_component(
                    component=component, section_name=section_name, model_configuration=self.model_configuration,
                    device=self.device, class_names=self.list_class_names())
                if section_name == "Backbone":
                    self.loaded_backbone_blocks = count_loaded_backbone_blocks(
                        model_configuration=self.model_configuration)

            # Freeze the loaded part of the component, or only the scale and shift of the normalization layers of the
            # loaded part of the backbone
            transferred_part = self.get_transferred_part(section_name=section_name)
            if pretrained_configuration["freeze"]:
                freeze_component(component=transferred_part)
            elif section_name == "Backbone" and pretrained_configuration["freeze_normalization_parameters"]:
                freeze_normalization_parameters(component=transferred_part)

        # Class layers of the semantic segmentation, outside its decoder, loaded with the decoder of a new run only
        if "SegmentationDecoder" in self.pretrained_configurations and \
                self.pretrained_configurations["SegmentationDecoder"]["load_class_layers"] and \
                pretrained_checkpoint_paths["SegmentationDecoder"] is not None:
            load_segmentation_class_layers(segmenter=self.model, model_configuration=self.model_configuration,
                                           device=self.device, class_names=self.list_class_names())

        # A model whose every parameter is frozen would not learn anything
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Every Parameter Of The Model Is Frozen By The 'pretrained.freeze' Settings, The Training "
                             "Would Change Nothing: Leave At Least One Component Trainable.")
        return pretrained_checkpoint_paths

    def get_transferred_part(self, section_name: str) -> nn.Module:
        """
        Gives the part of a transferred component holding the weights of its source run, which is frozen, keeps its
        batch statistics and learns at the learning rate multiplier of the transfer: the whole component, the loaded
        blocks of a backbone deeper than its source backbone (loaded_backbone_blocks), whose next blocks start random
        and learn at the learning rate of the run as the other new layers, or the layers of a classification head
        loaded without its class layer ('ClassificationHead.pretrained.class_layer' False), whose class layer does the
        same.

        Args:
            section_name (str): The configuration section of the component, a key of TRANSFERABLE_COMPONENTS.

        Returns:
            nn.Module: The component, or a module list of the loaded blocks of the backbone or of the layers of the
                head outside its class layer.
        """
        number_loaded_blocks = self.loaded_backbone_blocks if section_name == "Backbone" else None
        with_class_layers = transfers_class_layers(section_name=section_name,
                                                   pretrained_configuration=self.pretrained_configurations[section_name])
        return select_transferred_part(component=getattr(self.model, TRANSFERABLE_COMPONENTS[section_name]),
                                       section_name=section_name, number_loaded_blocks=number_loaded_blocks,
                                       with_class_layers=with_class_layers)

    def setup_training_paths(self, initial_configuration_path: Path, create_run_folder: bool
                             ) -> Tuple[Path, Path, Path, Optional[ConfigurationChanges]]:
        """
        Sets up the directory structure and persistent files for training outputs.

        This method acts as the initialization step for experiment storage: it gives the paths of the `Tensorboard`
        and `Weights` directories and of the archived configuration within the run folder and, for a training, creates
        the directories and archives the configuration file and the label dictionary of the run there
        ('label_dictionary.json', read back by restore_run_label_dictionary and by the configuration loader of the
        evaluation, export and model loading tools) before the training loop commences. A resumed run first compares
        the launched configuration with the saved one (see compare_with_saved_configuration), which the copy replaces.
        The input viewer, which exits without training, only needs the paths.

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

        # Create directories, a standalone evaluation reading the weights of the evaluated run and saving none
        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        tensorboard_directory.mkdir(exist_ok=True, parents=True)
        if not self.standalone_evaluation:
            weights_directory.mkdir(exist_ok=True, parents=True)

        # Settings changed since the last start of a resumed run, before the copy replaces the saved configuration
        configuration_changes = self.compare_with_saved_configuration(
            initial_configuration_path=initial_configuration_path, saved_configuration_path=new_configuration_path,
            weights_directory=weights_directory)

        # Save a copy of the configuration file
        copyfile(src=str(initial_configuration_path), dst=str(new_configuration_path))

        # Save the classes of the run, those of the mapping file or, for a resumed run, those it kept
        write_run_label_dictionary(label_dictionary=self.label_dictionary, run_folder=self.experiment_folder)

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

    def drops_last_training_batch(self) -> bool:
        """
        Tells whether the training loader of the task leaves out the last incomplete batch of every epoch, for the
        iterations per epoch printed at launch: DROPS_LAST_TRAINING_BATCH, True for the self-supervised task, and
        overridden by the classification trainer, whose training needs full batches with batch statistics.

        Returns:
            bool: True when the last incomplete training batch is left out.
        """
        return self.DROPS_LAST_TRAINING_BATCH

    def check_loaders_give_batches(self) -> None:
        """
        Refuses a run whose training or test loader, or validation loader when the validation is on, gives no batch,
        before anything is trained: an empty split, or a split whose every sample has a category left out by the label
        mapping, would make the training loop wait for a batch that never comes. A task whose build_dataloaders gives
        a validation loader without builds_validation_loader, or none with it, is refused as well, so that no task
        loads validation batches in a training without validation.

        Raises:
            ValueError: If a checked loader gives no batch, or if the validation loader does not follow
                builds_validation_loader.
        """
        # The validation loader exists exactly when the training or a diagnosis mode ending the program reads it
        if (self.validation_dataloader is not None) != self.builds_validation_loader:
            built_state = "built" if self.validation_dataloader is not None else "not built"
            raise ValueError(f"The validation loader of {type(self).__name__}.build_dataloaders is {built_state} while "
                             f"builds_validation_loader is {self.builds_validation_loader} "
                             f"('validation_iteration_ratio' {self.validation_iteration_ratio}, diagnosis mode ending "
                             f"the program {self.exits_after_diagnosis}): the task must pass builds_validation_loader "
                             f"to its loaders.")
        checked_loaders = {"train_data_folder": self.train_dataloader, "test_data_folder": self.test_dataloader}
        if self.validation_iteration_ratio > 0:
            checked_loaders["validation_data_folder"] = self.validation_dataloader

        # The number of batches of a pass, which counts the samples kept by the dataset after the label mapping
        for split_key, loader in checked_loaders.items():
            if len(loader) == 0:
                raise ValueError(f"The '{split_key}' split gives no batch: its dataset keeps {len(loader.dataset)} "
                                 f"samples of {format_path_link(path=self.experiment_configuration[split_key])}, check "
                                 f"the folder and the label mapping.")

    def restore_run_label_dictionary(self) -> None:
        """
        Keeps the classes of a resumed run: the label dictionary saved in its folder ('label_dictionary.json', written
        by setup_training_paths) replaces the one built from the mapping file, which may have been edited since with
        as many classes (another number of classes gives another run name, hence another folder), so that the classes
        of its checkpoints keep their meaning. The change is printed in yellow, and the keypoint definitions of the
        keypoint detection, indexed by class, are read again for the restored classes. A new run, a run without
        checkpoint and a run started before the file existed take the mapping file.
        """
        if not get_last_checkpoint_iteration(weights_directory=self.experiment_folder / "Weights"):
            return
        run_label_dictionary = read_run_label_dictionary(run_folder=self.experiment_folder)
        if run_label_dictionary is None or run_label_dictionary == self.data_configuration["label_dictionary"]:
            return
        print_color(color="yellow",
                    output=f"The Label Mapping Changed Since The Run Started, The Run Keeps Its Own Classes : "
                           f"{format_path_link(path=self.experiment_folder / RUN_LABEL_DICTIONARY_FILE_NAME)}",
                    add_separators=True)
        self.data_configuration["label_dictionary"] = run_label_dictionary

        # Keypoint definitions indexed by the classes of the mapping file, read again for the classes of the run
        if "keypoint_categories" in self.data_configuration:
            inject_keypoint_categories(experiment_configuration=self.experiment_configuration,
                                       model_configuration=self.model_configuration)

    def check_resumed_data(self) -> None:
        """
        Refuses to resume a run on labels that changed since the record of its data ('data.json') was written, before
        the copy of the configuration and the record are replaced, so that the run folder stays the one of the data
        that trained it: changed data asks for a new run. 'ExperimentConfiguration.allow_data_change_on_resume' lets
        the run resume anyway, save_data_record then keeping the previous record. A moved data folder holding the same
        labels file resumes.

        Raises:
            ValueError: If the labels file of a split changed since the last start of a resumed run and the change is
                not allowed.
        """
        data_record_path = self.experiment_folder / DATA_RECORD_FILE_NAME
        resumed_iteration = get_last_checkpoint_iteration(weights_directory=self.experiment_folder / "Weights")
        if self.allow_data_change_on_resume or not resumed_iteration or not data_record_path.is_file():
            return

        # Labels files of the splits compared with those of the saved record
        current_splits = {split_name: self.record_split(split_data_folder=self.experiment_configuration[split_key])
                          for split_name, split_key in DATA_RECORD_SPLITS.items()}
        changed_splits = find_changed_splits(previous_record=read_json(path=str(data_record_path)),
                                             current_record={"splits": current_splits},
                                             compared_entries=RELABELLED_SPLIT_ENTRIES)
        if changed_splits:
            raise ValueError(f"The labels.json of the {', '.join(changed_splits)} split(s) changed since iteration "
                             f"{resumed_iteration} of the run {format_path_link(path=self.experiment_folder)} (their "
                             f"SHA-256 differs from {DATA_RECORD_FILE_NAME}): changed data needs a new run. Set "
                             f"another 'ExperimentConfiguration.run_name_suffix' or "
                             f"'ExperimentConfiguration.experiment_folder', or move the run folder away, to train "
                             f"from scratch, or set 'ExperimentConfiguration.allow_data_change_on_resume' to true to "
                             f"resume on the new data.")

    def save_data_record(self, data_record_path: Path) -> Optional[DataRecordChanges]:
        """
        Writes the record of the data of the run ('data.json') next to its configuration file: the classes of the run
        with their names (list_class_names), categories and counts in every split (build_class_records, null for a task
        without class names), then the data folder, the SHA-256 of the 'labels.json' and the images of every split
        (record_split), so that the classes and the images a model was trained and evaluated on stay known when the
        data folders change. A resumed run compares the data it is launched with against the record saved by its
        previous start: an unchanged record is kept with its date, and a changed one (a moved data folder, or changed
        labels allowed by 'allow_data_change_on_resume', see check_resumed_data) is kept as
        'data_until_iteration_<n>.json', the data that trained the run up to the checkpoint of iteration n it resumes
        from, before the new record replaces it. As for the configurations, a file of that name is never replaced.

        Args:
            data_record_path (Path): The record of the data in the run folder.

        Returns:
            Optional[DataRecordChanges]: The changed splits with the resumed iteration and the archived file, None for
                a run that does not resume from a checkpoint or has no saved record.
        """
        split_data_folders = {split_name: self.experiment_configuration[split_key]
                              for split_name, split_key in DATA_RECORD_SPLITS.items()}

        # Classes of the run counted in the manifests of every split, null for a task without class names
        class_names = self.list_class_names()
        class_records = None if class_names is None else build_class_records(
            class_names=class_names, label_dictionary=self.label_dictionary,
            label_mode=self.data_configuration["label_mapping"]["mode"],
            unmapped_class_index=self.get_unmapped_class_index(), split_data_folders=split_data_folders)
        current_record = {"created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), CLASSES_RECORD_KEY: class_records,
                          "splits": {split_name: self.record_split(split_data_folder=split_data_folder)
                                     for split_name, split_data_folder in split_data_folders.items()}}

        # A resumed run compares its data with the record of its previous start, an unchanged record being kept
        data_record_changes = None
        resumed_iteration = get_last_checkpoint_iteration(weights_directory=self.weights_directory)
        if resumed_iteration and data_record_path.is_file():
            changed_splits = find_changed_splits(previous_record=read_json(path=str(data_record_path)),
                                                 current_record=current_record, compared_entries=COMPARED_SPLIT_ENTRIES)
            if not changed_splits:
                return DataRecordChanges(resumed_iteration=resumed_iteration, changed_splits=[],
                                         archived_data_record_path=None)

            # Keep the record of the data that trained the run up to the resumed checkpoint, unless an earlier resume
            # kept it
            archived_data_record_path = data_record_path.parent / f"data_until_iteration_{resumed_iteration}.json"
            if not archived_data_record_path.exists():
                copyfile(src=str(data_record_path), dst=str(archived_data_record_path))
            data_record_changes = DataRecordChanges(resumed_iteration=resumed_iteration, changed_splits=changed_splits,
                                                    archived_data_record_path=archived_data_record_path)

        # Written for a new run, for a run started before the record existed and for a resumed run whose data changed
        data_record_path.write_text(data=json.dumps(obj=current_record, indent=2))
        return data_record_changes

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
        print_color(color="blue", output=f"- Data Record             : {format_path_link(path=self.data_record_path)}")
        if self.exits_after_diagnosis:
            print_color(color="yellow", output="- Run Folder Not Created  : The Diagnosis Mode Exits Without Training")

        # Settings and data changed since the last start of a resumed run, then every component initialised by another
        # run
        self.print_configuration_changes()
        self.print_data_record_changes()
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

        # One epoch is a full pass over the training samples, without the last incomplete batch when the training
        # loader leaves it out
        self.iterations_per_epoch = math.ceil(self.number_samples_per_split["train_data_folder"] / self.batch_size)
        if self.drops_last_training_batch():
            self.iterations_per_epoch = self.number_samples_per_split["train_data_folder"] // self.batch_size
        # The length of the run and the iterations at which it saves and evaluates its weights
        print_color(color="blue", output=f"- Iterations Per Epoch    : {self.iterations_per_epoch}")
        print_color(color="blue", output=f"- Batch Size              : {self.batch_size}")
        print_color(color="blue", output=f"- Total Iterations        : {self.training_iterations}")
        print_color(color="blue", output=f"- Save And Evaluate Every : {self.weight_saving_iterations} iterations")
        # An intermediate weight saving of 0 saves no checkpoint between the evaluated ones
        intermediate_saving_description = f"{self.intermediate_weight_saving_iterations} iterations" if \
            self.intermediate_weight_saving_iterations > 0 else "Turned Off"
        print_color(color="blue", output=f"- Save Only Every         : {intermediate_saving_description}")
        # The optimization settings: learning rate, optimizer, schedule and weight moving average
        print_color(color="blue", output=f"- Learning Rate           : {self.learning_rate}")
        print_color(color="blue", output=f"- Optimizer               : {self.optimization_configuration['optimizer']} "
                                         f"(weight decay {self.optimization_configuration['weight_decay']})")
        print_color(color="blue", output=f"- Learning Rate Schedule  : "
                                         f"{self.optimization_configuration['learning_rate_schedule']} (warm-up "
                                         f"{self.optimization_configuration['warmup_iterations']} iterations)")
        moving_average_description = "Off"
        if self.exponential_moving_average is not None:
            moving_average_description = (f"decay {self.exponential_moving_average.decay}, evaluated next to the "
                                          f"trained weights")
        print_color(color="blue", output=f"- Weight Moving Average   : {moving_average_description}")
        print_color(color="blue", output=f"- Validation Iter. Ratio  : {self.validation_iteration_ratio}")

        # Backend of the MLflow tracking, the local database of the experiments root folder unless a server is set
        tracking_description = "Off"
        if self.tracking_configuration["activated"]:
            tracking_description, _ = resolve_tracking_uri(
                tracking_configuration=self.tracking_configuration,
                tracking_root=self.experiment_configuration["tracking_root"])
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

    def print_data_record_changes(self) -> None:
        """
        Prints, in the experiment information of a resumed run, whether its data changed since its last start (see
        save_data_record): the splits whose data folder or labels file changed, with the file keeping the previous
        record of the data. Nothing is printed for a run that does not resume from a checkpoint.
        """
        if self.data_record_changes is None:
            return
        resumed_iteration = self.data_record_changes["resumed_iteration"]
        changed_splits = self.data_record_changes["changed_splits"]
        if not changed_splits:
            print_color(color="blue", output=f"- Changed Data Splits     : None Since Iteration {resumed_iteration}")
            return

        # File keeping the previous record, then the changed splits
        archived_data_record_path = self.data_record_changes["archived_data_record_path"]
        print_color(color="yellow", output=f"- Previous Data Record    : "
                                           f"{format_path_link(path=archived_data_record_path)}")
        print_color(color="yellow", output=f"- Changed Data Splits     : {', '.join(changed_splits)} Since Iteration "
                                           f"{resumed_iteration}")

    def print_pretrained_components(self) -> None:
        """
        Prints, in the experiment information, one block per component initialised by the trained one of another run
        (backbone, temporal module, classification head, segmentation decoder, region proposal heads, detection
        heads, mask heads): its source run folder, then the checkpoint iteration it comes from, whether it is frozen,
        whether the normalization layers of the backbone are frozen, the loaded blocks of a backbone deeper than its
        source backbone (the others starting random), whether the class layer of the classification head is loaded
        with it, whether the class layers of the semantic segmentation are loaded with its decoder, and the learning
        rate it learns at with its multiplier, which gives no token in the run name.
        A frozen component does not learn, so it has no learning rate line. Components trained from random weights
        print nothing; the classes kept, added or moved by name are printed by the transfer itself
        (prepare_class_layers).
        """
        pretrained_labels = {"Backbone": "Pretrained Backbone", "TemporalModel": "Pretrained Temp. Module",
                             "ClassificationHead": "Pretrained Class. Head",
                             "SegmentationDecoder": "Pretrained Seg. Decoder", "RegionProposal": "Pretrained RPN Heads",
                             "Detector": "Pretrained Det. Heads", "MaskHead": "Pretrained Mask Heads"}
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

            # Scale and shift of the normalization layers kept fixed, a setting of the backbone only, and the blocks of
            # a backbone deeper than its source backbone loaded from it, the others starting random
            if section_name == "Backbone":
                print_color(color="blue", output=f"    - Norm. Layers Frozen : "
                                                 f"{pretrained_configuration['freeze_normalization_parameters']}")
            if section_name == "Backbone" and self.loaded_backbone_blocks is not None:
                number_blocks = len(self.model_configuration["Backbone"]["convolutions"])
                random_blocks = describe_block_range(first_block=self.loaded_backbone_blocks + 1,
                                                     last_block=number_blocks)
                print_color(color="blue", output=f"    - Loaded Blocks       : {self.loaded_backbone_blocks} Of "
                                                 f"{number_blocks}, {random_blocks} Random At The Run Learning Rate")

            # Class layer of the classification head, loaded with the head or left random, and class layers of the
            # semantic segmentation loaded with the decoder
            if section_name == "ClassificationHead":
                print_color(color="blue", output=f"    - Class Layer Loaded  : "
                                                 f"{pretrained_configuration['class_layer']}")
            if section_name == "SegmentationDecoder":
                print_color(color="blue", output=f"    - Class Layers Loaded : "
                                                 f"{pretrained_configuration['load_class_layers']}")

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
        ones augmented as during training, in two windows centred side by side on the screen, each image next to a
        black information panel with its targets and, for the training sample, the augmentations drawn for it with
        their values, every training batch being first shown whole when the task shows it (show_input_batch).
        Enabled by the 'view_input_images' diagnosis mode, it lets the user check the preprocessing, the label mapping
        and the data augmentation. Pressing ENTER closes the viewer, after which the program exits without training.
        """
        print_color(color="red", output="View Of The Training And Validation Images That Are Fed To The Neural Network",
                    add_separators=True)
        print_color(color="green", output="Press SPACE To Navigate, Press ENTER To Stop Viewing And Exit")

        # Step through the training and validation loaders together, until the user stops or either one runs out
        for batch_index, (training_batch, validation_batch) in enumerate(zip(self.train_dataloader,
                                                                             self.validation_dataloader)):
            number_samples = min(training_batch["images"].shape[0], validation_batch["images"].shape[0])

            # The whole training batch as the loader gives it, for the tasks showing it
            if self.show_input_batch(training_batch=training_batch, batch_index=batch_index):
                return

            # Augment the training batch exactly as during training, so the viewer shows what the network receives
            training_batch = self.augment_input_batch(
                batch={**training_batch, "images": training_batch["images"].to(dtype=torch.float32)})

            # Display the training and validation samples at the same batch index next to each other
            for sample_index in range(number_samples):
                if self.show_input_samples(training_batch=training_batch, validation_batch=validation_batch,
                                           sample_index=sample_index):
                    return

    def describe_input_augmentations(self, batch: Dict[str, Any], sample_index: int, split_name: str,
                                     label_names: Optional[Dict[int, str]] = None,
                                     real_frames: Optional[List[int]] = None) -> List[str]:
        """
        Writes the augmentation section of the information panel of a sample in the input viewer ('view_input_images')
        and the input samples of a training launch, placed by every build_input_windows under the target of the
        sample: the augmentations drawn for a training sample with their values, read from the 'augmentation_reports'
        stored in the batch by augment_batch, or a note for a validation or test sample, which is never augmented.

        Args:
            batch (Dict[str, Any]): The augmented training batch, or the validation or test batch.
            sample_index (int): The index of the displayed sample in its batch.
            split_name (str): "Training", "Validation" or "Test", named in the note of a sample without augmentation.
            label_names (Optional[Dict[int, str]]): The class name of every label index, to name the objects removed
                by the geometric augmentations. Defaults to None, in which case the label indices are written.
            real_frames (Optional[List[int]]): The indices of the real frames of a sequence, the only ones written in
                its frame by frame lines. Defaults to None, in which case every frame is written.

        Returns:
            List[str]: The lines of the section, starting with an empty line that separates it from the target.
        """
        # Validation and test batches never go through augment_batch, so they carry no augmentation report
        if "augmentation_reports" not in batch:
            return ["", f"Augmentations : none ({split_name.lower()} sample)"]
        return ["", "Augmentations"] + self.data_augmenter.describe_report(
            report=batch["augmentation_reports"][sample_index], label_names=label_names, real_frames=real_frames)

    def save_input_samples(self) -> None:
        """
        Writes the input samples of a training launch before its first iteration (trainers/input_samples.py): the
        first NUMBER_INPUT_SAMPLES samples by identifier of the training, validation and test splits, the same at every
        launch, drawn as the input viewer draws them (build_input_windows) into 'Input_Samples/Training/sample_00.png'
        and on, with their record 'samples.json'. The training samples are augmented on the device as by the training
        step (draw_input_split), on copies of the random states, so that the training draws what it would have drawn
        without them. Written for a new run, for a run started before the folder existed, and for a resumed run whose
        configuration or data changed, its previous folder being kept as 'Input_Samples_until_iteration_<n>'; a resumed
        run with the same configuration and data keeps its folder. A training without validation loader has no
        Validation folder, which the record and the console say. A split that cannot be drawn is reported with its
        traceback and recorded, the training going on.
        """
        # The decisions of the configuration and data record comparisons of the launch (setup_training_paths,
        # save_data_record), a resumed run writing its samples again only when one of them changed
        resumed_iteration = self.start_iteration - 1 if self.start_iteration > 1 else None
        launch_changed = ((self.configuration_changes is not None and bool(self.configuration_changes[
            "changed_settings"])) or (self.data_record_changes is not None and bool(self.data_record_changes[
                "changed_splits"])))
        input_samples_folder = self.experiment_folder / INPUT_SAMPLES_FOLDER_NAME
        writes_samples, archived_folder = prepare_input_samples_folder(
            run_folder=self.experiment_folder, resumed_iteration=resumed_iteration, launch_changed=launch_changed)
        if not writes_samples:
            print_color(color="blue", output=f"- Input Samples           : "
                                             f"{format_path_link(path=input_samples_folder)} (kept, configuration "
                                             f"and data unchanged since iteration {resumed_iteration})")
            return

        # Every split drawn on copies of the random states, a split without loader or failing being recorded as such
        start_time = time.perf_counter()
        split_records, sample_records = {}, []
        with fork_random_state(device=self.device):
            for split_name, dataloader in [("Training", self.train_dataloader),
                                           ("Validation", self.validation_dataloader), ("Test", self.test_dataloader)]:
                if dataloader is None:
                    split_records[split_name] = {"written_samples": 0, "augmented": False,
                                                 "note": "No validation loader: 'ExperimentConfiguration."
                                                         "validation_iteration_ratio' is 0."}
                    print_color(color="blue", output=f"- Input Samples           : No {split_name} Images, "
                                                     f"'validation_iteration_ratio' Is 0 (No Validation Loader)")
                    continue

                # A split failing to draw prints its traceback and is recorded with its error, the training going on
                try:
                    split_records[split_name], split_sample_records = self.draw_input_split(
                        split_name=split_name, dataloader=dataloader, split_folder=input_samples_folder / split_name)
                except Exception as error:
                    print_color(color="red", output=f"- Input Samples Of The {split_name} Split Not Drawn : "
                                                    f"{type(error).__name__}: {error}")
                    print("".join(traceback.format_exception(error)), end="")
                    split_records[split_name] = {"written_samples": 0,
                                                 "augmented": split_name in self.AUGMENTED_INPUT_SPLITS,
                                                 "error": f"{type(error).__name__}: {error}"}
                    split_sample_records = []
                sample_records += split_sample_records

        # Record of the samples, then one line with the folder, the images written and the time taken
        input_samples_folder.mkdir(parents=True, exist_ok=True)
        write_input_samples_record(input_samples_folder=input_samples_folder, first_iteration=self.start_iteration,
                                   split_records=split_records, sample_records=sample_records)
        self.input_samples_artifact_folder = INPUT_SAMPLES_FOLDER_NAME if resumed_iteration is None else \
            f"{INPUT_SAMPLES_FOLDER_NAME}_resume_{resumed_iteration}"
        number_images = sum(split_record["written_samples"] for split_record in split_records.values())
        archive_description = "" if archived_folder is None else \
            f", the previous ones kept in {format_path_link(path=archived_folder)}"
        print_color(color="blue", output=f"- Input Samples           : {format_path_link(path=input_samples_folder)} "
                                         f"({number_images} images written in {time.perf_counter() - start_time:.2f} "
                                         f"seconds{archive_description})")

    def draw_input_split(self, split_name: str, dataloader: DataLoader,
                         split_folder: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Draws the input samples of one split for save_input_samples: its first samples by identifier, each read in the
        main process as a batch of its own (read_input_samples), augmented for an augmented split
        (AUGMENTED_INPUT_SPLITS) on the device exactly as by the training step (prepare_batch then augment_input_batch)
        and brought back to the CPU, then drawn by build_input_windows into its 'sample_<position>.png', a sample that
        could not be read being recorded with its error.

        Args:
            split_name (str): "Training", "Validation" or "Test".
            dataloader (DataLoader): The loader of the split, giving its dataset and collate function.
            split_folder (Path): The folder of the images of the split, created with the first image.

        Returns:
            Tuple[Dict[str, Any], List[Dict[str, Any]]]: The summary of the split (images written, whether they are
                augmented) and the record of every listed sample.
        """
        augmented = split_name in self.AUGMENTED_INPUT_SPLITS
        sample_records = []
        for position, input_sample in enumerate(read_input_samples(dataloader=dataloader,
                                                                   number_samples=NUMBER_INPUT_SAMPLES)):
            sample_record = {"split": split_name, "file": None, "sample_identifier": input_sample.sample_identifier,
                             "source": input_sample.source}
            sample_records.append(sample_record)
            if input_sample.error is not None:
                sample_record["error"] = input_sample.error
                print_color(color="yellow", output=f"- Input Sample Not Read   : {split_name} sample {position:02d} "
                                                   f"{input_sample.source} : {input_sample.error}")
                continue

            # Augmented as by the training step, on the device of the model, then drawn on the CPU
            batch = input_sample.batch
            if augmented:
                batch = move_batch(batch=self.augment_input_batch(batch=self.prepare_batch(batch=batch)),
                                   device=torch.device("cpu"), pin_memory=False)
            image_path = split_folder / f"sample_{position:02d}.png"
            windows = self.build_input_windows(batch=batch, sample_index=0, split_name=split_name)
            sample_record.update(write_input_sample_image(windows=windows, image_path=image_path))
            sample_record["file"] = image_path.relative_to(split_folder.parent).as_posix()
            sample_record["image_size"] = list(batch["images"][0].shape[-2:])
        number_written_samples = sum(sample_record["file"] is not None for sample_record in sample_records)
        return {"written_samples": number_written_samples, "augmented": augmented}, sample_records

    def benchmark_speed(self) -> None:
        """
        Measures, for the 'benchmark_speed' diagnosis mode, how fast the configured architecture trains and infers when
        the data never waits, so that its training time and serving speed are known before any training: one batch of
        the training loader, kept in memory and tiled to every batch size (speed_benchmark.resize_batch), is timed
        through the inference of every format (PyTorch evaluation, PyTorch inference model, TorchScript, ONNX Runtime
        CUDA and TensorRT in FP32 and FP16) at a batch of 1, at the configured batch size and at the largest inference
        batch size that fits, then through the training step of the training loop (run_training_step) at the configured
        batch size and at the largest training batch size that fits. The largest batch sizes are searched up to 2048 in
        the memory left once a margin is kept free for the other processes (40 GB of an integrated GPU, 2 GB of a
        discrete one). The inference runs first, with the weights of the last checkpoint of the run folder when it has
        one (read only: the post-processing of the detection tasks depends on the detections of the weights), the
        training steps then changing the weights in memory only. Nothing is written to the run folder; the report is
        printed at the end.

        Raises:
            RuntimeError: If the trainer does not run on a CUDA GPU, or if less memory than the margin is free.
        """
        if self.device.type != "cuda":
            raise RuntimeError(f"The speed benchmark times CUDA GPUs only, the trainer runs on {self.device}.")
        print_color(color="blue", output="Speed Benchmark Of The Training Step And Of The Inference",
                    add_separators=True)

        # Other work on the GPU, announced at once and in the report, since it slows the timed iterations down
        warning_lines = []
        other_gpu_processes = list_other_gpu_processes()
        if other_gpu_processes is None:
            warning_lines.append("nvidia-smi could not list the GPU processes : check that the GPU runs nothing else")
        elif other_gpu_processes:
            warning_lines.append(f"Other processes on the GPU : {', '.join(other_gpu_processes)}; the times are those "
                                 f"of a shared GPU")
        for warning_line in warning_lines:
            print_color(color="yellow", output=warning_line)

        # One batch as the training loader gives it, the worker processes of the loader stopped at once so that their
        # memory is free before the budget is measured
        loader_batch = next(iter(self.train_dataloader))
        self.shutdown_loader_workers()
        source_batch_size = loader_batch["images"].shape[0]
        stored_batch = move_batch(batch=loader_batch, device=self.get_benchmark_batch_device(), pin_memory=False)
        del loader_batch

        # Memory capped for the whole benchmark, the inference measured first with the weights of the run
        with limit_process_memory(device=self.device) as memory_budget:
            checkpoint_iteration = self.load_benchmark_weights()
            inference_rows, inference_notes = self.benchmark_inference(stored_batch=stored_batch,
                                                                       memory_budget=memory_budget)
            training_rows, training_notes = self.benchmark_training(stored_batch=stored_batch)

        self.print_speed_benchmark(memory_budget=memory_budget, source_batch_size=source_batch_size,
                                   checkpoint_iteration=checkpoint_iteration, inference_rows=inference_rows,
                                   inference_notes=inference_notes, training_rows=training_rows,
                                   training_notes=training_notes, warning_lines=warning_lines)

    def get_benchmark_batch_device(self) -> torch.device:
        """
        Gives where the speed benchmark keeps its batches, where the loaders hand them to prepare_batch: in host memory
        for a discrete GPU (pinned by tile_benchmark_batch, as the loaders pin them), on the device for an integrated
        GPU, whose device memory is the host memory, so that the memory cap of the benchmark covers the batch.

        Returns:
            torch.device: The CPU for a discrete GPU, the device of the trainer otherwise.
        """
        return torch.device("cpu") if uses_pinned_memory() else self.device

    def tile_benchmark_batch(self, stored_batch: Dict[str, Any], batch_size: int) -> Dict[str, Any]:
        """
        Tiles the batch kept by the speed benchmark to a batch size, its samples repeated in their order, pinned in host
        memory for a discrete GPU as the loaders pin the batches they hand over.

        Args:
            stored_batch (Dict[str, Any]): The batch of the training loader kept by benchmark_speed.
            batch_size (int): The number of samples of the tiled batch.

        Returns:
            Dict[str, Any]: The tiled batch, as a loader would hand it to prepare_batch.
        """
        return move_batch(batch=resize_batch(batch=stored_batch, batch_size=batch_size),
                          device=self.get_benchmark_batch_device(), pin_memory=uses_pinned_memory())

    def count_benchmark_frames(self, batch: Dict[str, Any]) -> Optional[int]:
        """
        Counts the real frames of a batch of the speed benchmark, for its frame throughput, as the console summaries of
        the training count them (padding frames left out), or None for the image tasks.

        Args:
            batch (Dict[str, Any]): The tiled batch, with its 'sequence_mask' for the sequence tasks.

        Returns:
            Optional[int]: The real frames of the batch, None for the image tasks.
        """
        return int(batch["sequence_mask"].sum()) if self.COUNTS_FRAMES else None

    def load_benchmark_weights(self) -> Optional[int]:
        """
        Loads the trained weights of the last checkpoint of the run folder into the model for the speed benchmark, the
        optimizer state and the run folder left untouched, so that the post-processing of the detection tasks (NMS,
        masks, keypoints) is timed on the detections of trained weights.

        Returns:
            Optional[int]: The iteration of the loaded checkpoint, None when the run has none (the model keeping its
                initial weights).
        """
        checkpoint_iteration = get_last_checkpoint_iteration(weights_directory=self.weights_directory)
        if not checkpoint_iteration:
            return None
        checkpoint = load_checkpoint(weights_directory=self.weights_directory, iteration=checkpoint_iteration,
                                     device=self.device)
        self.model.load_state_dict(state_dict=checkpoint["model_state"])
        return checkpoint_iteration

    def run_evaluation_prediction(self, prepared_batch: Dict[str, Any],
                                  inference_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Runs one inference pass of the test evaluation for the speed benchmark: predict_batch on a prepared batch, in
        float32 without gradients and within the evaluation context of the task, as evaluate_test_split runs it.

        Args:
            prepared_batch (Dict[str, Any]): The prepared batch.
            inference_inputs (Dict[str, torch.Tensor]): The inputs of the inference model, unused by the evaluation.

        Returns:
            Dict[str, Any]: The predictions of the batch.
        """
        with self.evaluation_context(), torch.no_grad():
            return self.predict_batch(batch=prepared_batch)

    def probe_inference_batch(self, stored_batch: Dict[str, Any], batch_size: int) -> None:
        """
        Runs the test evaluation passes of one batch size tried by the search of the largest inference batch size,
        which raises an out of memory error when the batch does not fit.

        Args:
            stored_batch (Dict[str, Any]): The batch of the training loader kept by benchmark_speed.
            batch_size (int): The tried batch size.
        """
        prepared_batch = self.prepare_batch(batch=self.tile_benchmark_batch(stored_batch=stored_batch,
                                                                            batch_size=batch_size))
        for _ in range(PROBE_ITERATIONS):
            self.run_evaluation_prediction(prepared_batch=prepared_batch, inference_inputs={})
        torch.cuda.synchronize(device=self.device)

    def probe_training_batch(self, stored_batch: Dict[str, Any], batch_size: int) -> None:
        """
        Runs the training steps of one batch size tried by the search of the largest training batch size, which raises
        an out of memory error when the batch does not fit, the gradients and the optimizer state included.

        Args:
            stored_batch (Dict[str, Any]): The batch of the training loader kept by benchmark_speed.
            batch_size (int): The tried batch size.
        """
        batch = self.tile_benchmark_batch(stored_batch=stored_batch, batch_size=batch_size)
        for _ in range(PROBE_ITERATIONS):
            self.run_training_step(batch=batch, iteration=1, tracker_dictionary=None)
        torch.cuda.synchronize(device=self.device)

    def fits_benchmark_batch(self, batch_size: int, stored_batch: Dict[str, Any], training: bool) -> bool:
        """
        Tells the search of the largest batch size (speed_benchmark.find_largest_batch_size) whether a batch size fits
        in the memory budget of the speed benchmark, from its probe passes (probe_training_batch or
        probe_inference_batch), printing the answer. Only an out of memory error means that it does not fit, any other
        error being raised. The memory of the probe is given back before the answer.

        Args:
            batch_size (int): The tried batch size.
            stored_batch (Dict[str, Any]): The batch of the training loader kept by benchmark_speed.
            training (bool): True for the training step, False for the inference of the test evaluation.

        Returns:
            bool: True when the probe passes fit in memory.
        """
        try:
            if training:
                self.probe_training_batch(stored_batch=stored_batch, batch_size=batch_size)
            else:
                self.probe_inference_batch(stored_batch=stored_batch, batch_size=batch_size)
            fits = True
        except Exception as error:
            if not is_out_of_memory_error(error=error):
                raise
            fits = False

        # Gradients of a failed step dropped, then the memory of the probe given back
        self.zero_gradients()
        release_cached_memory()
        print_color(color="green" if fits else "yellow",
                    output=f"Batch Size {batch_size} : {'Fits' if fits else 'Out Of Memory'}", indent=1)
        return fits

    def benchmark_inference(self, stored_batch: Dict[str, Any],
                            memory_budget: MemoryBudget) -> Tuple[List[BenchmarkRow], List[str]]:
        """
        Times the inference of the speed benchmark: the largest batch size of the test evaluation is searched first,
        then every format of INFERENCE_FORMATS is opened in turn (the eager inference model and the exported files
        written once into a temporary folder, the ONNX file only when ONNX Runtime has a GPU provider) and timed at a
        batch of 1, at the configured batch size and at the largest one, on inputs already on the GPU. A format that
        cannot be opened here is reported with its reason; an exported format failing or running out of memory at a
        batch size is reported at that batch size.

        Args:
            stored_batch (Dict[str, Any]): The batch of the training loader kept by benchmark_speed.
            memory_budget (MemoryBudget): The memory cap of the benchmark, of which ONNX Runtime may take what PyTorch
                leaves.

        Returns:
            Tuple[List[BenchmarkRow], List[str]]: The rows of the inference table, and the lines of its legend.
        """
        self.set_evaluation_mode()
        print_color(color="blue", output="Searching The Largest Inference Batch Size (Test Evaluation, Float32)")
        largest_batch_size, smallest_failing_batch_size = find_largest_batch_size(
            fits=partial(self.fits_benchmark_batch, stored_batch=stored_batch, training=False),
            starting_batch_size=self.batch_size)
        batch_sizes = sorted({batch_size for batch_size in [1, self.batch_size, largest_batch_size]
                              if 0 < batch_size <= largest_batch_size})
        notes = [self.describe_largest_batch_size(largest_batch_size=largest_batch_size,
                                                  smallest_failing_batch_size=smallest_failing_batch_size,
                                                  step_name="inference")]

        # Inference model of the task and its exported files, for the tasks Triton serves
        inference_model, exported_files = None, None
        rows = []
        unavailable_reasons = {}
        with tempfile.TemporaryDirectory(prefix="speed_benchmark_") as export_folder:
            if self.TASK_NAME in INFERENCE_MODEL_CLASSES and batch_sizes:
                print_color(color="blue", output="Exporting The Inference Model (TorchScript, ONNX)")
                inference_model = build_inference_model(task=self.TASK_NAME, model=self.model)
                specification = get_export_specification(task=self.TASK_NAME, inference_model=inference_model)
                exported_files = export_benchmark_formats(inference_model=inference_model, specification=specification,
                                                          maximum_batch_size=max(batch_sizes),
                                                          export_folder=Path(export_folder), device=self.device)

            # Every format opened, timed at every batch size, then closed before the next one
            for inference_format in INFERENCE_FORMATS:
                format_rows, unavailable_reason = self.benchmark_inference_format(
                    inference_format=inference_format, stored_batch=stored_batch, batch_sizes=batch_sizes,
                    inference_model=inference_model, exported_files=exported_files, memory_budget=memory_budget,
                    export_folder=Path(export_folder))
                rows += format_rows
                if unavailable_reason is not None:
                    format_label = f"{inference_format['name']} {inference_format['precision']}"
                    unavailable_reasons.setdefault(unavailable_reason, []).append(format_label)
                release_cached_memory()

        # Formats that could not be measured, grouped by reason
        notes += [f"{', '.join(format_labels)} : {reason}" for reason, format_labels in unavailable_reasons.items()]
        return rows, notes

    def benchmark_inference_format(self, inference_format: Dict[str, Any], stored_batch: Dict[str, Any],
                                   batch_sizes: List[int], inference_model: Optional[nn.Module],
                                   exported_files: Optional[Dict[str, Dict[str, Any]]], memory_budget: MemoryBudget,
                                   export_folder: Path) -> Tuple[List[BenchmarkRow], Optional[str]]:
        """
        Opens one inference format of the speed benchmark and times it at every inference batch size: the test
        evaluation of the trainer (run_evaluation_prediction) for 'PyTorch Evaluation', the formats of
        speed_benchmark.open_export_format otherwise. A format that cannot be opened gives one row without batch size
        and its reason, the error of a failed opening being written as the reason.

        Args:
            inference_format (Dict[str, Any]): The format, from INFERENCE_FORMATS.
            stored_batch (Dict[str, Any]): The batch of the training loader kept by benchmark_speed.
            batch_sizes (List[int]): The inference batch sizes, empty when not even one sample fits.
            inference_model (Optional[nn.Module]): The inference model of the task, None for a task without export.
            exported_files (Optional[Dict[str, Dict[str, Any]]]): The files of export_benchmark_formats, None for a
                task without export.
            memory_budget (MemoryBudget): The memory cap of the benchmark.
            export_folder (Path): The temporary folder of the exported files.

        Returns:
            Tuple[List[BenchmarkRow], Optional[str]]: The rows of the format, and the reason why it could not be
                measured (None when it was opened).
        """
        unavailable_row = {"label": inference_format["name"], "precision": inference_format["precision"],
                           "batch_size": None, "measurement": None, "status": NOT_AVAILABLE_STATUS,
                           "frames_per_batch": None}

        # Function of one pass of the format, None with the reason when it cannot be opened here
        if inference_format["name"] == "PyTorch Evaluation":
            run_pass, unavailable_reason = self.run_evaluation_prediction, None
        elif inference_model is None:
            run_pass, unavailable_reason = None, f"no export specification for the {self.TASK_NAME} task"
        else:
            try:
                run_pass, unavailable_reason = open_export_format(
                    inference_format=inference_format, inference_model=inference_model, exported_files=exported_files,
                    device=self.device, export_folder=export_folder,
                    memory_limit_bytes=memory_budget["allowed_bytes"] - torch.cuda.memory_reserved(device=self.device))
            except Exception as error:
                return [{**unavailable_row, "status": FORMAT_FAILED_STATUS}], describe_error(error=error)
        if run_pass is None:
            return [unavailable_row], unavailable_reason

        # Every batch size of the inference, the row of a failing batch size keeping its status
        print_color(color="blue", output=f"Timing {inference_format['name']} {inference_format['precision']}")
        rows = []
        for batch_size in batch_sizes:
            rows.append(self.measure_inference_batch(inference_format=inference_format, run_pass=run_pass,
                                                     stored_batch=stored_batch, batch_size=batch_size))
        return rows, None

    def measure_inference_batch(self, inference_format: Dict[str, Any], run_pass: Callable[..., Any],
                                stored_batch: Dict[str, Any], batch_size: int) -> BenchmarkRow:
        """
        Times one inference format at one batch size for the speed benchmark, on the tiled batch prepared on the GPU and
        the inputs of the inference model built from it once, before the timed passes. An out of memory error marks the
        row; another error of an exported format marks it as failed, the error being printed; an error of the test
        evaluation is raised.

        Args:
            inference_format (Dict[str, Any]): The format, from INFERENCE_FORMATS.
            run_pass (Callable[..., Any]): Runs one pass of the format on (prepared_batch, inference_inputs).
            stored_batch (Dict[str, Any]): The batch of the training loader kept by benchmark_speed.
            batch_size (int): The batch size.

        Returns:
            BenchmarkRow: The row of the format at the batch size.
        """
        row = {"label": inference_format["name"], "precision": inference_format["precision"], "batch_size": batch_size,
               "measurement": None, "status": MEASURED_STATUS, "frames_per_batch": None}

        # Batch tiled to the batch size and prepared, and inference inputs built once, before the timed passes
        try:
            tiled_batch = self.tile_benchmark_batch(stored_batch=stored_batch, batch_size=batch_size)
            row["frames_per_batch"] = self.count_benchmark_frames(batch=tiled_batch)
            prepared_batch = self.prepare_batch(batch=tiled_batch)
            inference_inputs = {input_name: input_tensor.contiguous() for input_name, input_tensor
                                in self.build_inference_inputs(batch=prepared_batch).items()}
            row["measurement"] = measure_iteration_time(
                run_iteration=partial(run_pass, prepared_batch=prepared_batch, inference_inputs=inference_inputs),
                device=self.device)
        except Exception as error:
            if is_out_of_memory_error(error=error):
                row["status"] = OUT_OF_MEMORY_STATUS
            elif inference_format["name"] != "PyTorch Evaluation":
                row["status"] = FORMAT_FAILED_STATUS
                print_color(color="yellow", output=f"{inference_format['name']} {inference_format['precision']} "
                                                   f"Failed At Batch Size {batch_size} : {describe_error(error=error)}")
            else:
                raise

        # The memory of ONNX Runtime is outside PyTorch, which cannot report its peak
        if row["measurement"] is not None and inference_format["provider"] is not None:
            row["measurement"]["peak_memory_bytes"] = None
        status_color = "green" if row["status"] == MEASURED_STATUS else "yellow"
        print_color(color=status_color, output=f"Batch Size {batch_size} : {row['status']}", indent=1)
        return row

    def benchmark_training(self, stored_batch: Dict[str, Any]) -> Tuple[List[BenchmarkRow], List[str]]:
        """
        Times the training step of the speed benchmark (run_training_step, with the trackers of the training loop) at
        the configured batch size and at the largest batch size whose training step fits in the memory budget,
        searched first, from the batch in host memory as the loader hands it over (on the device of an integrated GPU),
        and gives the time of the training of the run at the configured batch size, a lower bound since the loading,
        the validation passes, the checkpoints and the test evaluations are left out.

        Args:
            stored_batch (Dict[str, Any]): The batch of the training loader kept by benchmark_speed.

        Returns:
            Tuple[List[BenchmarkRow], List[str]]: The rows of the training table, and the lines of its legend.
        """
        print_color(color="blue", output="Searching The Largest Training Batch Size")
        largest_batch_size, smallest_failing_batch_size = find_largest_batch_size(
            fits=partial(self.fits_benchmark_batch, stored_batch=stored_batch, training=True),
            starting_batch_size=self.batch_size)
        notes = [self.describe_largest_batch_size(largest_batch_size=largest_batch_size,
                                                  smallest_failing_batch_size=smallest_failing_batch_size,
                                                  step_name="training")]

        # The configured batch size, then the largest one when it differs
        measured_batches = [("Configured", self.batch_size)]
        if 0 < largest_batch_size != self.batch_size:
            measured_batches.append(("Largest Found", largest_batch_size))
        print_color(color="blue", output="Timing The Training Step")
        rows = []
        for label, batch_size in measured_batches:
            row = {"label": label, "precision": "BF16", "batch_size": batch_size, "measurement": None,
                   "status": OUT_OF_MEMORY_STATUS, "frames_per_batch": None}

            # A batch size above the largest one found keeps its out of memory row, the others are timed
            if batch_size <= largest_batch_size:
                batch = self.tile_benchmark_batch(stored_batch=stored_batch, batch_size=batch_size)
                row["frames_per_batch"] = self.count_benchmark_frames(batch=batch)
                row["measurement"] = measure_iteration_time(
                    run_iteration=partial(self.run_training_step, batch=batch, iteration=1,
                                          tracker_dictionary=self.metric_tracker.create_trackers()),
                    device=self.device)
                row["status"] = MEASURED_STATUS
                del batch
            rows.append(row)
            print_color(color="green" if row["status"] == MEASURED_STATUS else "yellow",
                        output=f"Batch Size {batch_size} : {row['status']}", indent=1)
            release_cached_memory()

        # Training time of the run at the configured batch size, when it fits
        if rows[0]["measurement"] is not None:
            training_seconds = self.training_iterations * rows[0]["measurement"]["seconds_per_iteration"]
            notes.append(f"{self.training_iterations:,} iterations (number_iterations) at the configured batch size : "
                         f"{MetricTracker.format_time_interval(duration_seconds=training_seconds)} at least, the "
                         f"loading, validation passes, checkpoints and test evaluations left out")
        return rows, notes

    def describe_largest_batch_size(self, largest_batch_size: int, smallest_failing_batch_size: Optional[int],
                                    step_name: str) -> str:
        """
        Writes the line of the report giving the largest batch size the search of the speed benchmark found for the
        training step or the inference, with the smallest batch size found not to fit, or the maximum of the search.

        Args:
            largest_batch_size (int): The largest batch size that fits, 0 when not even one sample fits.
            smallest_failing_batch_size (Optional[int]): The smallest batch size that does not fit, None when the
                maximum fits.
            step_name (str): "training" or "inference".

        Returns:
            str: The line of the report.
        """
        if largest_batch_size == 0:
            return f"Largest {step_name} batch size : none, a single sample does not fit in the memory budget"
        if smallest_failing_batch_size is None:
            return (f"Largest {step_name} batch size : {largest_batch_size}, the largest the search tries "
                    f"({MAXIMUM_BENCHMARK_BATCH_SIZE})")
        return (f"Largest {step_name} batch size : {largest_batch_size} ({smallest_failing_batch_size} does not fit in "
                f"the memory budget)")

    def print_speed_benchmark(self, memory_budget: MemoryBudget, source_batch_size: int,
                              checkpoint_iteration: Optional[int], inference_rows: List[BenchmarkRow],
                              inference_notes: List[str], training_rows: List[BenchmarkRow],
                              training_notes: List[str], warning_lines: List[str]) -> None:
        """
        Prints the report of the speed benchmark: the GPU, the memory budget, the batch and the weights used, the
        warnings (other processes on the GPU, initial weights), then the inference table and the training table with
        their legends.

        Args:
            memory_budget (MemoryBudget): The memory cap of the benchmark.
            source_batch_size (int): The number of samples of the batch drawn from the training loader.
            checkpoint_iteration (Optional[int]): The iteration of the checkpoint whose weights were loaded, None for
                the initial weights.
            inference_rows (List[BenchmarkRow]): The rows of the inference table.
            inference_notes (List[str]): The lines of the legend of the inference table.
            training_rows (List[BenchmarkRow]): The rows of the training table.
            training_notes (List[str]): The lines of the legend of the training table.
            warning_lines (List[str]): The warnings found when the benchmark started (other processes on the GPU).
        """
        gpu_kind = "integrated GPU sharing the host memory" if memory_budget["integrated"] else "discrete GPU"
        batch_location = "the shared memory of the GPU" if memory_budget["integrated"] else "pinned host memory"
        information_lines = [
            f"GPU : {torch.cuda.get_device_name(device=self.device)} ({gpu_kind})",
            f"Memory : {memory_budget['free_bytes'] / BYTES_PER_GIGABYTE:.1f} GB free at the start, "
            f"{memory_budget['margin_bytes'] / BYTES_PER_GIGABYTE:.0f} GB kept free for the other processes, "
            f"{memory_budget['budget_bytes'] / BYTES_PER_GIGABYTE:.1f} GB of budget",
            f"Batch : one training batch of {source_batch_size} {self.SAMPLE_NAME.lower()} of "
            f"{self.input_image_size} pixels, its samples repeated to every batch size",
            f"Training : from the batch where the loader hands it over ({batch_location}), moved, augmented, bfloat16 "
            f"forward pass, backward pass, optimizer step and moving average of the weights",
            "Inference : from inputs already on the GPU, float32 unless written otherwise"]
        warning_lines = list(warning_lines)
        if checkpoint_iteration is None:
            warning_lines.append("No checkpoint in the run folder : the inference runs on the initial weights, the "
                                 "post-processing of the detection tasks then timing the detections of an untrained "
                                 "model")
        else:
            information_lines.append(f"Weights : checkpoint of iteration {checkpoint_iteration} of the run folder")

        # Tables of the inference and of the training, with their legends
        inference_legend = "\n".join(inference_notes + [
            "PyTorch Evaluation is the predict_batch of the test evaluation; the peak memory of the ONNX Runtime "
            "formats is outside PyTorch and not measured"])
        training_legend = "\n".join(training_notes + [
            "Window Spread : gap between the slowest and the fastest timing windows, relative to their median"])
        tables = [build_inference_table(rows=inference_rows, sample_name=self.SAMPLE_NAME,
                                        counts_frames=self.COUNTS_FRAMES, legend=inference_legend),
                  build_training_table(rows=training_rows, sample_name=self.SAMPLE_NAME,
                                       counts_frames=self.COUNTS_FRAMES, legend=training_legend)]
        print_speed_benchmark_report(header=join_header_parts(parts=["Speed Benchmark", self.run_name]),
                                     information_lines=information_lines, warning_lines=warning_lines, tables=tables)

    def get_next_batch(self, dataloader_iterator: _BaseDataLoaderIter, dataloader: DataLoader, split_name: str,
                       maximum_consecutive_errors: int = 10) -> Tuple[Dict[str, Any], _BaseDataLoaderIter]:
        """
        Retrieves the next training or validation batch for the training loop, restarting the loader
        at the end of every pass over its split, which reshuffles it. A batch whose loading fails (e.g. an unreadable
        image, named by its dataset) is dropped and the next one taken, so that a few bad files never stop a training:
        every distinct error is printed the first time it occurs, and the dropped batches of every split are counted
        and printed with the next console summary (report_dropped_batches). More than
        MAXIMUM_DROPPED_BATCHES_IN_WINDOW batches dropped among the last DROPPED_BATCH_WINDOW fetches of the split
        stop the training, the data being broken rather than holding a few bad files.

        Args:
            dataloader_iterator (_BaseDataLoaderIter): The current iterator for the dataloader.
            dataloader (DataLoader): The original dataloader object used to reset the iterator.
            split_name (str): The split of the loader, "Training" or "Validation", naming it in the messages and in
                the counts of the dropped batches.
            maximum_consecutive_errors (int): The number of failed fetches in a row after which the loading is taken
                as broken (e.g. a data folder that disappeared). Defaults to 10.

        Returns:
            Tuple[Dict[str, Any], _BaseDataLoaderIter]: A tuple containing the batch data dictionary,
                and the potentially refreshed dataloader iterator.

        Raises:
            RuntimeError: If maximum_consecutive_errors fetches fail in a row, or if MAXIMUM_EMPTY_PASSES passes over
                the split end without any batch, an error which the training loop counts like the other errors of an
                iteration and after which it still saves its last checkpoint.
            TrainingStopError: If more than MAXIMUM_DROPPED_BATCHES_IN_WINDOW of the last DROPPED_BATCH_WINDOW
                fetches of the split were dropped, which stops the training at once, its last checkpoint still saved.
        """
        # Outcomes of the last fetches of the split, kept over the whole launch
        if split_name not in self.recent_fetch_drops:
            self.recent_fetch_drops[split_name] = deque(maxlen=DROPPED_BATCH_WINDOW)
        recent_drops = self.recent_fetch_drops[split_name]

        consecutive_errors = 0
        empty_passes = 0
        while True:
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                # A pass ends: the loader restarts, unless the passes keep ending without any batch, which would
                # otherwise loop forever on an empty split
                empty_passes += 1
                if empty_passes >= MAXIMUM_EMPTY_PASSES:
                    raise RuntimeError(f"The {split_name} loader gives no batch: {empty_passes} passes over its split "
                                       f"ended without any.")
                dataloader_iterator = iter(dataloader)
            except Exception as exception:
                # The batch is dropped and counted for the next console summary, its error printed the first time
                consecutive_errors += 1
                self.dropped_batch_counts[split_name] += 1
                recent_drops.append(True)
                self.report_fetch_error(exception=exception, split_name=split_name)

                # Too many dropped batches among the last fetches of the split stop the training, which would
                # otherwise go on with what is left of its data
                number_recent_drops = sum(recent_drops)
                if number_recent_drops > MAXIMUM_DROPPED_BATCHES_IN_WINDOW:
                    raise TrainingStopError(
                        f"{number_recent_drops} of the last {len(recent_drops)} {split_name} batches failed to load "
                        f"and were dropped, above the limit of {MAXIMUM_DROPPED_BATCHES_IN_WINDOW} in "
                        f"{DROPPED_BATCH_WINDOW} ({100 * MAXIMUM_DROPPED_BATCHES_IN_WINDOW / DROPPED_BATCH_WINDOW:.0f} "
                        f"%): fix or remove the files named above, then launch the run again to resume it.") \
                        from exception
                if consecutive_errors >= maximum_consecutive_errors:
                    raise RuntimeError(f"{consecutive_errors} fetches of a {split_name} batch failed in a row, the "
                                       f"last one with: {exception}") from exception
            else:
                recent_drops.append(False)
                return batch, dataloader_iterator

    def report_fetch_error(self, exception: Exception, split_name: str) -> None:
        """
        Prints the error of a dropped batch (get_next_batch) the first time it occurs in the run, so that a bad file
        is named once instead of at every epoch: its last line, the error raised by the dataset (e.g. the unreadable
        image), in yellow, and the whole message with the traceback of the loader worker for the first dropped batch
        of the run.

        Args:
            exception (Exception): The error raised by the fetch of the batch.
            split_name (str): The split of the loader, "Training" or "Validation".
        """
        message_lines = [line for line in str(exception).splitlines() if line.strip()]
        error_line = message_lines[-1] if message_lines else type(exception).__name__
        if error_line in self.reported_fetch_errors:
            return

        # The traceback of the first dropped batch, then the error of every new one
        if not self.reported_fetch_errors:
            print(f"{type(exception).__name__}: {exception}")
        self.reported_fetch_errors.add(error_line)
        print_color(color="yellow", output=f"{split_name} Batch Dropped : {error_line}")

    def report_iteration_error(self, error: Exception) -> None:
        """
        Prints the error of a failed training iteration that the training loop (run_training_iterations) goes on
        after: its whole traceback the first time this error occurs in the run, so that the line raising it is known,
        then one line at every later occurrence, so that an error coming back every few iterations (e.g. an out of
        memory error on the largest batches) does not flood the console.

        Args:
            error (Exception): The error raised inside the training iteration.
        """
        error_text = f"{type(error).__name__}: {error}"
        if error_text in self.reported_iteration_errors:
            print(f"Exception: {error}")
            return
        self.reported_iteration_errors.add(error_text)
        print("".join(traceback.format_exception(error)), end="")

    def report_dropped_batches(self) -> None:
        """
        Prints the number of batches of every split dropped since the previous console summary because their loading
        failed (get_next_batch), with the summary of the training loop, then starts the counts again. Prints nothing
        when no batch was dropped.
        """
        if sum(self.dropped_batch_counts.values()) == 0:
            return
        dropped_batches = ", ".join(f"{split_name} {dropped_count}"
                                    for split_name, dropped_count in self.dropped_batch_counts.items() if dropped_count)
        print_color(color="yellow", output=f"- Dropped Batches Since The Last Summary : {dropped_batches} (their "
                                           f"loading failed, see the errors printed above)")
        self.dropped_batch_counts.clear()

    def initialize_training_loop_state(self) -> Tuple[
            Iterator, Optional[Iterator], bool, Dict[str, float], Optional[Dict[str, float]], int]:
        """
        Initializes and returns the state variables required for the main training loop.

        This method sets up the initial data loaders, determines the validation gating configuration,
        and creates the metric trackers necessary for logging. Extracting this logic improves the
        readability of the main training loop.

        Returns:
            Tuple: A tuple containing:
                - training_dataloader_iterator (Iterator): Iterator for the training dataset.
                - validation_dataloader_iterator (Optional[Iterator]): Iterator for the validation dataset, None
                  without validation.
                - validation_enabled (bool): Flag indicating if validation is enabled.
                - training_trackers (Dict[str, float]): Dictionary for accumulating training metrics.
                - validation_trackers (Optional[Dict[str, float]]): Dictionary for validation metrics, or None.
                - consecutive_errors (int): Counter for consecutive errors encountered during training.
        """
        # Determine whether validation is enabled based on the configured ratio
        validation_enabled = self.validation_iteration_ratio > 0

        # Get dataloader, the validation iterator only with validation, so that its worker process never starts
        # otherwise
        training_dataloader_iterator = iter(self.train_dataloader)
        validation_dataloader_iterator = iter(self.validation_dataloader) if validation_enabled else None

        # Get the metric trackers of the training and validation batches, which count the batches of their interval
        training_trackers = self.metric_tracker.create_trackers()
        validation_trackers = self.metric_tracker.create_trackers() if validation_enabled else None

        consecutive_errors = 0

        return (training_dataloader_iterator, validation_dataloader_iterator, validation_enabled,
                training_trackers, validation_trackers, consecutive_errors)

    def run_training_loop(self, maximum_consecutive_errors: int = 5) -> None:
        """
        Trains the model (see run_training_iterations), then stops the worker processes of its loaders
        (shutdown_loader_workers) and closes its MLflow run whatever ends the training: FINISHED once the training ends
        normally, KILLED when the user stops it (Ctrl+C), FAILED after too many consecutive errors, a failed save or
        test evaluation, a split dropping too many batches, or on an unexpected error (e.g. the final save failing
        again), which is raised again once the run is closed.

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
            # The loader workers are stopped first, the MLflow run being closed even if their shutdown fails
            try:
                self.shutdown_loader_workers()
            finally:
                self.finish_experiment_tracking(run_status=run_status)

    def list_dataloaders(self) -> List[Iterable]:
        """
        Lists the loaders whose worker processes are stopped at the end of the training (shutdown_loader_workers): the
        training, validation (when built) and test loaders, to which a task adds its own (the reference images of the
        k-nearest neighbour evaluation of the self-supervised task).

        Returns:
            List[Iterable]: The loaders of the trainer.
        """
        return [dataloader for dataloader in [self.train_dataloader, self.validation_dataloader, self.test_dataloader]
                if dataloader is not None]

    def shutdown_loader_workers(self) -> None:
        """
        Stops the worker processes of the training, validation and test loaders, and of the other loaders of the task
        (list_dataloaders), once the training ends (run_training_loop), whatever ended it, so that they free their
        memory at once instead of at the end of the process. The workers are kept alive between the passes and
        evaluations of the training (persistent_workers of build_native_loader). Left to the garbage collection of the
        trainer, the queues of a loader may be freed before the loader itself, losing the stop message of its workers,
        which then wait the whole join timeout of PyTorch (5 seconds) before being terminated: 30 seconds for the 6
        workers of a trainer, in a process running several trainings such as the tests. A loader iterated again
        afterwards starts new workers.
        """
        for dataloader in self.list_dataloaders():
            # Only a DataLoader has workers, the loaders of a task being any iterable of batches (build_dataloaders)
            if not isinstance(dataloader, DataLoader):
                continue

            # The iterator kept by a loader with persistent workers holds them, None before its first pass or without
            # workers. PyTorch has no public method stopping them: the shutdown its own exit hook runs is called
            iterator = dataloader._iterator
            if isinstance(iterator, _MultiProcessingDataLoaderIter):
                iterator._shutdown_workers()
            dataloader._iterator = None

    def run_training_iterations(self, maximum_consecutive_errors: int) -> str:
        """
        Executes the main training loop up to iteration number_iterations, the total of the run (a resumed run trains
        the iterations left, and one already there trains no more): training steps, checkpoints saved right after the
        step of their iteration (Iteration_k holding k steps, so that a resume from it starts at k + 1), validation
        passes, console summaries, the evaluation of the test split at every weight_saving_iterations checkpoint, last
        of its iteration, and a final checkpoint of the steps taken since the last one, whatever ends the loop. The
        MLflow run of the training is opened once the loader workers exist, and receives the interval averages of
        every console summary and the test evaluations. A failing iteration (e.g. a batch with a non-finite loss,
        skipped without step) is printed and skipped, more than maximum_consecutive_errors of them in a row ending the
        loop; a failed save or test evaluation (e.g. a full disk) and a split dropping too many batches end it at once,
        since the training would otherwise go on without checkpoints, evaluations or most of its data.

        Args:
            maximum_consecutive_errors (int): The maximum number of consecutive exceptions allowed before the training
                loop exits.

        Returns:
            str: The status of the MLflow run: FINISHED_STATUS, or KILLED_STATUS when the user stopped the training,
                or FAILED_STATUS when it stopped after too many consecutive errors, a failed save or test evaluation,
                or a split dropping too many batches.
        """
        # A run resumed from a checkpoint at or beyond its total of iterations has nothing left to train
        if self.start_iteration > self.training_iterations:
            print_color(color="green",
                        output=f"The Run Is Already Complete: Its Last Checkpoint Is At Iteration "
                               f"{self.start_iteration - 1} And 'number_iterations' Is {self.training_iterations}, The "
                               f"Total Of The Run. Raise 'number_iterations' To Train It Further.", add_separators=True)
            return FINISHED_STATUS

        # First samples of every split drawn as the network receives them, before the loader workers start
        self.save_input_samples()

        # Initialize tensorboard writers right before training starts
        self.training_writer, self.validation_writer = self.setup_tensorboard_writers()

        # Initialize core loop state variables, dataloader iterators, and metric trackers
        (training_dataloader_iterator, validation_dataloader_iterator, validation_enabled,
         training_trackers, validation_trackers, consecutive_errors) = self.initialize_training_loop_state()

        # MLflow run of the training, opened after the training and validation loaders fork their worker processes
        # (their iterators above) so that these never inherit the thread of the tracker; the test loader forks its
        # workers later, at the first test evaluation while the tracker runs, and keeps them for the next evaluations
        self.start_experiment_tracking()

        # Status of the MLflow run set by a stop of the training before its last iteration
        stopped_status = None

        # Last iteration whose step was taken, saved and evaluated: the checkpoint a resumed run starts from holds the
        # steps up to the iteration before its first one, so that a stop before the first step saves nothing again
        last_stepped_iteration = self.start_iteration - 1
        last_saved_iteration = self.start_iteration - 1
        last_evaluated_iteration = None

        for training_iteration in range(self.start_iteration, self.training_iterations + 1, 1):
            try:
                # Get next training elements
                training_batch, training_dataloader_iterator = self.get_next_batch(
                    dataloader_iterator=training_dataloader_iterator, dataloader=self.train_dataloader,
                    split_name="Training")

                # Real frames of the training batch, counted for the frame throughput of the sequence tasks
                self.metric_tracker.count_frames(tracker_dictionary=training_trackers, batch=training_batch)

                # Forward pass, backward pass and optimizer step of the training batch
                self.run_training_step(batch=training_batch, iteration=training_iteration,
                                       tracker_dictionary=training_trackers)
                last_stepped_iteration = training_iteration

                # Save the model right after the step of every weight saving iteration (evaluated at the end of the
                # iteration) and of the more frequent intermediate weight saving iterations in between (none when
                # intermediate_weight_saving_iterations is 0), so that the checkpoint of iteration k holds its k steps.
                # A failed save (e.g. a full disk) stops the training instead of letting it go on without checkpoints
                is_evaluated_iteration = training_iteration % self.weight_saving_iterations == 0
                is_intermediate_iteration = self.intermediate_weight_saving_iterations > 0 and \
                    training_iteration % self.intermediate_weight_saving_iterations == 0
                if is_evaluated_iteration or is_intermediate_iteration:
                    with stop_training_on_error(
                            failure_description=f"The checkpoint of iteration {training_iteration} could not be saved"):
                        self.save_model(iteration=training_iteration)
                    last_saved_iteration = training_iteration

                # Validation phase gated by the configurable validation_iteration_ratio.
                # A ratio of 1 runs validation every iteration, 2 runs every 2nd, N runs every Nth.
                if validation_enabled and training_iteration % self.validation_iteration_ratio == 0:
                    self.set_evaluation_mode()
                    with torch.no_grad():
                        validation_batch, validation_dataloader_iterator = self.get_next_batch(
                            dataloader_iterator=validation_dataloader_iterator, dataloader=self.validation_dataloader,
                            split_name="Validation")
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
                    self.report_dropped_batches()

                    if validation_enabled:
                        validation_trackers = self.metric_tracker.create_trackers()

                # Evaluate the checkpoint of a weight saving iteration on the test split, last in the iteration so that
                # a failing evaluation (e.g. an unreadable test image or a full disk) finds the rest of the iteration
                # done; it stops the training instead of failing again at every later evaluation
                if is_evaluated_iteration:
                    with stop_training_on_error(failure_description=f"The evaluation of the checkpoint of iteration "
                                                                    f"{training_iteration} failed"):
                        self.run_test_evaluation(iteration=training_iteration)
                    last_evaluated_iteration = training_iteration

                # Reset the consecutive errors counter after a fully successful iteration
                consecutive_errors = 0

            except KeyboardInterrupt:
                print_color(color="red", output=f"Training Interrupted by User at iteration {training_iteration}.",
                            add_separators=True)
                stopped_status = KILLED_STATUS
                break
            except TrainingStopError as error:
                # A failed save or test evaluation, or a split dropping too many batches, ends the training at once
                # instead of being counted with the failing iterations, with the whole traceback of its error
                print_color(color="red", output=f"Critical Error At Iteration {training_iteration}, Exiting: {error}",
                            add_separators=True)
                print("Last exception:\n" + "".join(traceback.format_exception(error)), end="")
                stopped_status = FAILED_STATUS
                break
            except Exception as error:
                # Increment error counter when an exception occurs inside the training iteration loop
                consecutive_errors += 1

                if consecutive_errors > maximum_consecutive_errors:
                    print_color(color="red",
                                output=f"Critical Error: Over {maximum_consecutive_errors} consecutive errors "
                                       f"encountered during training iterations. Exiting.", add_separators=True)
                    # The whole traceback of the error ending the training, to find the line raising it
                    print("Last exception:\n" + "".join(traceback.format_exception(error)), end="")
                    stopped_status = FAILED_STATUS
                    break
                elif isinstance(error, NonFiniteLossError):
                    # A batch with a non-finite loss, skipped without step, is printed in red on one line
                    print_color(color="red", output=f"{error} (Consecutive Errors: {consecutive_errors}/"
                                                    f"{maximum_consecutive_errors})", add_separators=True)
                    continue
                else:
                    print_color(color="yellow", output=f"Error encountered during iteration {training_iteration}."
                                                       f" Continuing... (Consecutive Errors: "
                                                       f"{consecutive_errors}/{maximum_consecutive_errors})",
                                add_separators=True)
                    self.report_iteration_error(error=error)
                    continue

        # Final checkpoint of the steps taken since the last save, whatever ended the loop (its last iteration, a stop
        # by the user, too many errors, a failed save, tried once more here, or a failed evaluation), a checkpoint
        # already saved after the last step being kept as it is
        if last_stepped_iteration > last_saved_iteration:
            self.save_model(iteration=last_stepped_iteration)
            last_saved_iteration = last_stepped_iteration

        # A new run stopped before its first step has no checkpoint
        saved_checkpoint_text = f"Last Checkpoint Saved At Iteration {last_saved_iteration}" if last_saved_iteration \
            else "No Checkpoint Saved"
        print_color(color="green", output=f"{saved_checkpoint_text}. Exiting Training.", add_separators=True)

        # A last checkpoint of a weight saving iteration saved by this launch but left unevaluated by a stop is
        # announced, to be evaluated apart
        if last_saved_iteration >= self.start_iteration and last_saved_iteration % self.weight_saving_iterations == 0 \
                and last_saved_iteration != last_evaluated_iteration:
            print_color(color="yellow",
                        output=f"The Checkpoint Of Iteration {last_saved_iteration} Was Not Evaluated: "
                               f"scripts/evaluation/evaluate_checkpoint.py Evaluates It.")

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
        the run is sent as an artifact, under its own name at every resumption since its settings may have changed, and
        so are the files of the analysis of the training data (analyse_training_data), under 'analysis'. The record of
        the data and the input samples (save_input_samples) are sent when this start wrote them.
        """
        # MLflow tracker of the run, a DisabledTracker when the tracking is off or unavailable
        self.experiment_tracker = create_experiment_tracker(
            tracking_configuration=self.tracking_configuration,
            tracking_root=self.experiment_configuration["tracking_root"], run_name=self.run_name,
            task=self.TASK_NAME, run_record_path=self.experiment_folder / RUN_RECORD_FILE_NAME)

        # A run resumed from its checkpoint of iteration n starts at n + 1
        resumed_iteration = self.start_iteration - 1 if self.start_iteration > 1 else None
        self.experiment_tracker.start(parameters=self.build_tracking_parameters(), tags=self.build_tracking_tags(),
                                      resumed_iteration=resumed_iteration)
        artifact_name = None if resumed_iteration is None else \
            f"training_configuration_resume_{resumed_iteration}.yaml"
        self.experiment_tracker.log_artifact(file_path=self.configuration_path, artifact_folder="configuration",
                                             artifact_name=artifact_name)

        # Files of the analysis of the training data, rewritten at every launch, under their own name at a resumption
        for analysis_file_path in self.training_data_analysis_files.values():
            analysis_artifact_name = None if resumed_iteration is None else \
                f"{analysis_file_path.stem}_resume_{resumed_iteration}{analysis_file_path.suffix}"
            self.experiment_tracker.log_artifact(file_path=analysis_file_path,
                                                 artifact_folder=TRAINING_DATA_ANALYSIS_ARTIFACT_FOLDER,
                                                 artifact_name=analysis_artifact_name)

        # The record of the data is sent when this start wrote it: a new run, or a resumed run whose data changed
        if self.data_record_changes is None or self.data_record_changes["changed_splits"]:
            data_record_name = None if self.data_record_changes is None else \
                f"data_resume_{self.data_record_changes['resumed_iteration']}.json"
            self.experiment_tracker.log_artifact(file_path=self.data_record_path, artifact_folder="data",
                                                 artifact_name=data_record_name)

        # The input samples are sent when this start wrote them (save_input_samples), their split folders kept under
        # 'Input_Samples', or 'Input_Samples_resume_<n>' at a resumption
        if self.input_samples_artifact_folder is not None:
            for file_path, artifact_folder in list_input_sample_artifacts(
                    input_samples_folder=self.experiment_folder / INPUT_SAMPLES_FOLDER_NAME,
                    artifact_folder=self.input_samples_artifact_folder):
                self.experiment_tracker.log_artifact(file_path=file_path, artifact_folder=artifact_folder)

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
        'evaluation/iteration_<n>/'. The averaged weights send theirs as 'test_ema/<scalar>' and
        'test_ema_class/<scalar>', with their files under 'evaluation/iteration_<n>/ema/' and their own best value
        ('test_ema/best_<metric>', tagged 'best_iteration_test_ema').

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.
            results (EvaluationResults): The results of the evaluation.
        """
        # Metric prefix and artifact folder of the evaluated weights
        metric_prefix = (AVERAGED_WEIGHTS_METRIC_PREFIX if self.evaluating_averaged_weights
                         else TRAINED_WEIGHTS_METRIC_PREFIX)
        artifact_folder = f"evaluation/iteration_{iteration}"
        if self.evaluating_averaged_weights:
            artifact_folder = f"{artifact_folder}/{AVERAGED_WEIGHTS_FOLDER_NAME.lower()}"

        # Global scalars, and the scalars of every class only with the per-class report (thousands for many classes)
        metrics = {f"{metric_prefix}/{scalar_name}": value for scalar_name, value in results["scalars"].items()}
        if self.per_class_report:
            metrics.update({f"{metric_prefix}_class/{scalar_name}": value
                            for scalar_name, value in results["class_scalars"].items()})
        self.experiment_tracker.log_metrics(metrics=metrics, step=iteration)

        # Best value of the headline metric, followed separately for the trained and the averaged weights
        self.experiment_tracker.update_best_metric(metric_name=results["headline_metric"],
                                                   value=results["scalars"][results["headline_metric"]],
                                                   step=iteration, metric_prefix=metric_prefix)
        # Small evaluation files written by the task for this checkpoint
        for file_key in TRACKED_EVALUATION_FILES:
            if file_key in results["files"]:
                self.experiment_tracker.log_artifact(file_path=results["files"][file_key],
                                                     artifact_folder=artifact_folder)

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

    def run_training_step(self, batch: Dict[str, Any], iteration: int,
                          tracker_dictionary: Optional[Dict[str, Any]]) -> torch.Tensor:
        """
        Runs one training step on a batch of the training loader, the same in the training loop
        (run_training_iterations) and in the speed benchmark (benchmark_speed), so that the benchmark times exactly what
        the training runs: the model in training mode with its gradients cleared, the forward pass of
        run_model_iteration (batch moved to the device, augmented, bfloat16 losses tracked), the backward pass, then
        the optimizer step with the learning rate of the iteration and the update of the moving average of the weights.
        A batch whose total loss is not finite stops in run_model_iteration, before the backward pass and the step.

        Args:
            batch (Dict[str, Any]): The batch yielded by the training loader.
            iteration (int): The training iteration, giving the learning rate of the schedule and the TensorBoard step.
            tracker_dictionary (Optional[Dict[str, Any]]): The training trackers of the interval, or None to track
                nothing.

        Returns:
            torch.Tensor: The total training loss of the batch.

        Raises:
            NonFiniteLossError: If the total loss of the batch is NaN or infinite, the weights being left unchanged.
        """
        self.set_training_mode()
        self.zero_gradients()

        # Forward pass of the augmented batch, its losses tracked and written to TensorBoard
        training_loss, _ = self.run_model_iteration(batch=batch, writer=self.training_writer, apply_augmentation=True,
                                                    iteration=iteration, tracker_dictionary=tracker_dictionary)

        # Backward pass and optimizer step
        training_loss.backward()
        self.step_optimizer(iteration=iteration)
        return training_loss

    def run_model_iteration(self, batch: Dict[str, Any], writer: Optional[SummaryWriter], iteration: int,
                            tracker_dictionary: Optional[Dict[str, Any]],
                            apply_augmentation: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes one forward pass of the model on a training or validation batch, the same for every task: the batch
        is prepared (and augmented for training), the model returns its losses, which are summed into the total loss,
        each multiplied by its weight when the task weights its losses (get_loss_weights), and the tracked metrics are
        accumulated and written to TensorBoard, the losses unweighted and the total loss weighted. A total loss that is
        NaN or infinite raises before anything is tracked, so that the interval averages stay finite and the training
        step never back-propagates it.

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a loader.
            writer (Optional[SummaryWriter]): TensorBoard writer for logging. If None, logging is skipped.
            iteration (int): The current training iteration step.
            tracker_dictionary (Optional[Dict[str, Any]]): The trackers of the interval, summing the metrics and
                counting the batches, or None to track nothing.
            apply_augmentation (bool): Whether to augment the batch, True for training batches only (validation and
                test batches must be evaluated on the original images). Augmentation also requires
                'Data.data_augmentation.activated'.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: A tuple containing the total loss and the model outputs.

        Raises:
            NonFiniteLossError: If the total loss of the batch is NaN or infinite, with the value of every loss.
        """
        # Move the batch to the device and precision of the model
        batch = self.prepare_batch(batch=batch)

        # Randomly augment the training batches, together with the targets that move with the images
        if apply_augmentation:
            batch = self.augment_batch(batch=batch)

        # Execute the forward pass through the model of the task, with the targets so that it returns its losses, in
        # bfloat16 (mixed precision, the weights staying in float32)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            model_outputs = self.forward_pass(batch=batch)

        # Extract all the losses of the model, and sum them into the total loss used for backpropagation, each
        # multiplied by its weight when the task weights its losses
        metric_dictionary = {key: value for key, value in model_outputs.items() if key.endswith(LOSS_KEY_SUFFIX)}
        total_loss = sum(loss_value if self.loss_weights is None else self.loss_weights[loss_key] * loss_value
                         for loss_key, loss_value in metric_dictionary.items())

        # A NaN or infinite total loss (a diverging step, a corrupted batch) stops the batch before it is tracked or
        # back-propagated, which would turn every weight into NaN; the tracked metrics synchronise with the device
        # right after anyway, so the check costs no extra wait
        if not torch.isfinite(total_loss).item():
            loss_values = ", ".join(f"{loss_key} {loss_value.item():.4g}" for loss_key, loss_value in
                                    [("total_loss", total_loss)] + list(metric_dictionary.items()))
            raise NonFiniteLossError(f"The loss of the batch of iteration {iteration} is not finite ({loss_values}): "
                                     f"the batch is skipped, without backward pass nor optimizer step, the weights "
                                     f"staying those of the previous step.")

        # Insert the total loss and the batch metrics of the task into the tracked metrics
        metric_dictionary["total_loss"] = total_loss
        metric_dictionary.update(self.compute_batch_metrics(batch=batch, model_outputs=model_outputs))

        # Add the metrics to the sums of the interval and log them to tensorboard (per-iteration)
        self.metric_tracker.accumulate(tracker_dictionary=tracker_dictionary, metric_dictionary=metric_dictionary)
        self.metric_tracker.log_to_tensorboard(writer=writer, iteration=iteration, metric_dictionary=metric_dictionary)

        return total_loss, model_outputs

    def run_test_evaluation(self, iteration: int) -> Tuple[EvaluationResults, Optional[EvaluationResults]]:
        """
        Evaluates a saved checkpoint on the whole test split, at every weight_saving_iterations checkpoint: its trained
        weights, then, for a run keeping a moving average of its weights updated at least once, its averaged weights,
        each reported on its own (see evaluate_test_split).

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            Tuple[EvaluationResults, Optional[EvaluationResults]]: The results of the trained weights and those of the
                averaged weights (None without them), for the tools comparing the checkpoints of a run.
        """
        results = self.evaluate_test_split(iteration=iteration)

        # Averaged weights of the same checkpoint, reported next to the trained ones (an average restarted from the
        # loaded weights of a checkpoint saved without one would only repeat them)
        averaged_weights_results = None
        if self.exponential_moving_average is not None and self.exponential_moving_average.number_updates > 0:
            with self.use_averaged_weights():
                averaged_weights_results = self.evaluate_test_split(iteration=iteration)
        return results, averaged_weights_results

    @contextmanager
    def use_averaged_weights(self) -> Iterator[None]:
        """
        Puts the averaged model in place of the trained model for the evaluation of the moving average of the weights,
        so that the hooks of the task (predict_batch, evaluation_context) run it unchanged, and routes the outputs of
        the evaluation to their own folder, TensorBoard writer, Results.txt label and MLflow metrics. The trained model
        is always restored, even when the evaluation fails.

        Yields:
            None: Control to the evaluation of the averaged weights.
        """
        trained_model = self.model
        self.model = self.exponential_moving_average.averaged_model
        self.evaluating_averaged_weights = True
        try:
            yield
        finally:
            self.model = trained_model
            self.evaluating_averaged_weights = False

    def get_evaluation_folder(self, iteration: int) -> Path:
        """
        Gives the folder of the files of the evaluation of a checkpoint, written by the base trainer (sample images)
        and the task trainers (class metrics, thresholds, confusion matrix, predictions, error images...):
        '<run>/Evaluation/Iteration_<n>/' for a training, '<run>/Checkpoint_Evaluation/Iteration_<n>/' for a standalone
        evaluation (evaluation_root), and its 'EMA' subfolder for the averaged weights.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            Path: The evaluation folder of the weights being evaluated.
        """
        evaluation_folder = self.evaluation_root / f"Iteration_{iteration}"
        if self.evaluating_averaged_weights:
            return evaluation_folder / AVERAGED_WEIGHTS_FOLDER_NAME
        return evaluation_folder

    def reserve_error_folder(self, accumulator: Dict[str, Any], folder_name: str) -> Optional[Path]:
        """
        Counts one more error image in a folder of the 'Errors' folder of the evaluation (a pair of classes, a wanted
        class for the frame selection, or the false positives, missed objects or confusions of a class for the
        detection), and gives that folder, or None when it already holds 'maximum_per_class_pair' images, so that the
        image is only built when it is saved. Used by the trainers saving error images, which read the
        'evaluation.error_dumps' settings into error_dumps_configuration.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, holding the 'errors_folder' and 'error_counts'.
            folder_name (str): The name of the folder of the error image, possibly nested ('MISSED/<class>').

        Returns:
            Optional[Path]: The folder, created, or None when it is full.
        """
        error_counts = accumulator["error_counts"]
        number_images = error_counts[folder_name] if folder_name in error_counts else 0
        if number_images >= self.error_dumps_configuration["maximum_per_class_pair"]:
            return None
        error_counts[folder_name] = number_images + 1
        error_folder = accumulator["errors_folder"] / folder_name
        error_folder.mkdir(parents=True, exist_ok=True)
        return error_folder

    def evaluate_test_split(self, iteration: int) -> EvaluationResults:
        """
        Evaluates the weights in the model (the trained ones, or the averaged ones within use_averaged_weights) on the
        whole test split and reports them to the console, TensorBoard and the Results.txt file of the run. The model
        runs as in deployment, through the predict_batch hook and within the evaluation context of the task, and every
        test sample is evaluated exactly once. The images of the fixed test samples are drawn on the way and added to
        the results of the task, with a note and their folder. An evaluation saving error images first removes those
        an earlier evaluation of the same checkpoint and weights left in its 'Errors' folder.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        self.model.eval()
        accumulator = self.create_evaluation_accumulator(iteration=iteration)
        evaluated_sample_identifiers = set()
        samples_folder = self.get_evaluation_folder(iteration=iteration) / "Samples"
        sample_images = {}
        weights_description = f" ({AVERAGED_WEIGHTS_LABEL})" if self.evaluating_averaged_weights else ""

        # Error images of an earlier evaluation of the same weights removed before this one saves its own (the tasks
        # saving error images give their folder), so that the 'Errors' folder never mixes two evaluations, e.g. the
        # standalone evaluations of a checkpoint with two limits per pair of classes
        if "errors_folder" in accumulator and self.error_dumps_configuration["activated"]:
            if accumulator["errors_folder"].exists():
                rmtree(path=accumulator["errors_folder"])

        # The evaluation settings of the task are always restored, even when the evaluation fails
        with self.evaluation_context(), torch.no_grad():
            for test_batch in tqdm(self.test_dataloader, total=len(self.test_dataloader),
                                   desc=f"Test Evaluation Iteration {iteration}{weights_description}"):
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

        # Results of the task, with the images of the fixed test samples reported at every evaluation, the header of
        # the block of the averaged weights naming them
        results = self.compute_evaluation_results(accumulator=accumulator, iteration=iteration)
        results["sample_images"] = sample_images
        if self.evaluating_averaged_weights:
            results["evaluated_samples"] = f"{results['evaluated_samples']}{HEADER_SEPARATOR}{AVERAGED_WEIGHTS_LABEL}"
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
        evaluation, Test_EMA for the averaged weights, with the same tags so that TensorBoard overlays both curves) and
        the Results.txt file of the run, with its per-class outputs when the per-class report is on, sends it to the
        MLflow run, then writes it into the evaluation summary and redraws the evaluation curves
        (update_evaluation_curves).

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.
            results (EvaluationResults): The results of the evaluation.
        """
        if self.evaluating_averaged_weights:
            if self.averaged_test_writer is None:
                self.averaged_test_writer = SummaryWriter(
                    log_dir=str(self.tensorboard_directory / AVERAGED_WEIGHTS_TENSORBOARD_FOLDER))
            test_writer = self.averaged_test_writer
        else:
            if self.test_writer is None:
                self.test_writer = SummaryWriter(log_dir=str(self.tensorboard_directory / "Test"))
            test_writer = self.test_writer
        report_evaluation_results(results=results, iteration=iteration, test_writer=test_writer,
                                  results_path=Path(self.experiment_folder) / "Results.txt",
                                  per_class_report=self.per_class_report)
        self.track_evaluation(iteration=iteration, results=results)
        self.update_evaluation_curves(iteration=iteration, results=results)

    def update_evaluation_curves(self, iteration: int, results: EvaluationResults) -> None:
        """
        Keeps the evaluation summary and the evaluation curves of the run up to date after every test evaluation, the
        trained weights then the averaged ones, so that the test results can be read at a glance without TensorBoard:
        the scalars of the evaluated weights go into the row of the iteration of 'evaluation_summary.csv'
        (evaluation_root, '<run>/Evaluation/' for a training, '<run>/Checkpoint_Evaluation/' for a standalone
        evaluation; the averaged weights in 'ema_' columns with their number of updates 'ema_number_updates', an
        iteration evaluated again replacing its row), then 'evaluation_curves.png', next to Results.txt, is drawn again
        from the summary: the headline metric of both weights, the best iteration of each ringed, and the windows of
        the averaged weights of the highest evaluated iteration, given by their number of updates. A summary missing
        while the TensorBoard test folders hold evaluations (a run evaluated before the summary existed) is first
        started from them. With the tracking on, both files are sent to the MLflow run, replacing their previous
        version.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.
            results (EvaluationResults): The results of the evaluation.
        """
        # Earlier evaluations of a run evaluated before the summary existed, read once from its TensorBoard test curves
        summary_path = self.evaluation_root / EVALUATION_SUMMARY_FILE_NAME
        if not summary_path.is_file():
            seed_evaluation_summary(summary_path=summary_path, test_folders={
                "": self.tensorboard_directory / "Test",
                AVERAGED_WEIGHTS_COLUMN_PREFIX: self.tensorboard_directory / AVERAGED_WEIGHTS_TENSORBOARD_FOLDER},
                current_iteration=iteration)

        # Row of the iteration started by the trained weights and completed by the averaged ones, with their number of
        # updates (restored with the checkpoint), which gives the warm-up of their decay
        column_prefix = AVERAGED_WEIGHTS_COLUMN_PREFIX if self.evaluating_averaged_weights else ""
        columns = build_summary_columns(results=results, prefix=column_prefix, per_class_report=self.per_class_report)
        if self.evaluating_averaged_weights:
            columns[AVERAGED_WEIGHTS_UPDATES_COLUMN] = self.exponential_moving_average.number_updates
        write_summary_columns(summary_path=summary_path, iteration=iteration, columns=columns,
                              starts_row=not self.evaluating_averaged_weights)

        # Figure of the headline metric drawn again from every row, with the windows of the moving average when it is
        # on
        figure_path = Path(self.experiment_folder) / EVALUATION_CURVES_FIGURE_NAME
        moving_average_decay = (self.exponential_moving_average.decay if self.exponential_moving_average is not None
                                else None)
        draw_summary_curves(summary_path=summary_path, figure_path=figure_path, metric_name=results["headline_metric"],
                            in_percent=results["headline_metric"] in results["percent_metrics"],
                            moving_average_decay=moving_average_decay)
        print_color(color="green", output=f"Evaluation Curves : {format_path_link(path=figure_path)}", indent=1)

        # Both files sent to the MLflow run, at its root next to Results.txt
        self.experiment_tracker.log_artifact(file_path=summary_path, artifact_folder=None)
        self.experiment_tracker.log_artifact(file_path=figure_path, artifact_folder=None)

    def save_model(self, iteration: int) -> None:
        """
        Saves a checkpoint of the model and optimizer, with the task, the run name, the label dictionary of the run
        ('label_dictionary', its keys written as text as in 'label_dictionary.json'), the identifier of the MLflow run
        ('mlflow_run_id', None without tracking or while the run is being opened), the source run of every
        transferable component of the task ('pretrained_backbone', 'pretrained_temporal_module',
        'pretrained_classification_head', 'pretrained_segmentation_decoder', 'pretrained_region_proposer_dictionary',
        'pretrained_detector_heads_dictionary', 'pretrained_mask_heads_dictionary', None without transfer), the
        backbone blocks loaded from a shorter source backbone ('pretrained_backbone_blocks', None when the whole
        backbone or none of it comes from another run, read back when the run resumes) and the moving average of the
        weights of a run keeping one, and records it in the checkpoint registry of the run used to resume the training.

        Args:
            iteration (int): The current training iteration, used for naming the weight file.
        """
        # Task, run name, classes, MLflow run and source run of every transferable component, stored with the weights
        metadata = {"task": self.TASK_NAME, "run_name": self.run_name,
                    "label_dictionary": {str(key): class_index for key, class_index in self.label_dictionary.items()},
                    "mlflow_run_id": self.experiment_tracker.get_run_id()}
        for section_name, pretrained_configuration in self.pretrained_configurations.items():
            metadata[f"pretrained_{TRANSFERABLE_COMPONENTS[section_name]}"] = pretrained_configuration[
                "experiment_folder"]

        # Backbone blocks loaded from a shorter source backbone, read back when the run resumes
        metadata[LOADED_BACKBONE_BLOCKS_METADATA_KEY] = self.loaded_backbone_blocks
        exponential_moving_average_state = None
        if self.exponential_moving_average is not None:
            exponential_moving_average_state = self.exponential_moving_average.state_dict()
        model_directory = save_checkpoint(weights_directory=self.weights_directory, iteration=iteration,
                                          model=self.model, optimizer=self.optimizer, metadata=metadata,
                                          exponential_moving_average_state=exponential_moving_average_state)

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
        mapped to the currently active device (CPU, CUDA, or MPS), and the moving average of the weights of a run
        keeping one.

        Args:
            iteration (int): The exact iteration number identifying the checkpoint to load.
        """
        checkpoint = load_checkpoint(weights_directory=self.weights_directory, iteration=iteration, device=self.device)

        self.model.load_state_dict(state_dict=checkpoint["model_state"])
        self.optimizer.load_state_dict(state_dict=checkpoint["optimizer_state"])
        self.load_exponential_moving_average(checkpoint=checkpoint, iteration=iteration)

    def load_exponential_moving_average(self, checkpoint: Dict[str, Any], iteration: int) -> None:
        """
        Restores the moving average of the weights saved with a checkpoint, once its trained weights are in the model,
        for a resumed run and for the evaluation of the saved checkpoints (vision_training/evaluation/
        checkpoint_evaluation.py). A checkpoint saved without average (a run that turns it on when it resumes) restarts
        the average from its trained weights. Nothing is done for a run without average.

        Args:
            checkpoint (Dict[str, Any]): The loaded checkpoint.
            iteration (int): The iteration of the checkpoint, for the message.
        """
        if self.exponential_moving_average is None:
            return
        if EXPONENTIAL_MOVING_AVERAGE_STATE_KEY in checkpoint:
            self.exponential_moving_average.load_state_dict(
                state_dictionary=checkpoint[EXPONENTIAL_MOVING_AVERAGE_STATE_KEY])
            return
        self.exponential_moving_average.reset(model=self.model)
        print_color(color="yellow", output=f"The Checkpoint Of Iteration {iteration} Holds No Moving Average Of The "
                                           f"Weights, The Average Restarts From Its Trained Weights.")
