import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision_training.core.contracts import SequenceBatch
from vision_training.data.loaders import get_sequence_dataloaders
from vision_training.data.split_description import count_split_sequences
from vision_training.trainers.classification_trainer import ClassificationTrainer
from vision_training.utilities.visualization import build_frame_grid, view_input_data

# Frames per row of the frame grids of the input viewer, and longest side in pixels of their windows
FRAME_GRID_COLUMNS = 4
FRAME_GRID_DISPLAY_SIZE = 800


class SequenceClassificationTrainer(ClassificationTrainer):
    """
    Trainer of the sequence classification task (many to one): the sequence classifier trained on windows of frames
    cut from the sequences of 'labels.json' manifests (Native loader), one class per window. The losses, the accuracy
    tracked on every batch and the evaluation of the checkpoints are those of the classification task, a window
    being classified as a whole like an image.

    The sequence labelling trainer extends it with one class per frame, and the sequence comparison trainer with one
    class per pair of sequences.
    """

    TASK_NAME = "sequence_classification"
    SAMPLE_NAME = "Sequences"

    # Labels read from the manifests: one class per window (many to one)
    LABEL_TYPE = "sequence"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the sequence settings used by the hooks, then prepares the training run through the classification
        trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data with its sequence settings,
                Backbone, TemporalModel, ClassificationHead).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Window of frames fed to the model, cut from the sequences of the manifests
        self.sequence_settings = model_configuration["Data"]["sequence_settings"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the sequence loaders of the three splits, and takes the class names from the categories of the
        training manifest.

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders.
        """
        loaders = get_sequence_dataloaders(experiment_configuration=self.experiment_configuration,
                                           model_configuration=self.model_configuration, batch_size=self.batch_size,
                                           number_of_workers=4, label_type=self.LABEL_TYPE)
        training_dataset = loaders[0].dataset
        self.class_names = [training_dataset.class_names[class_index]
                            for class_index in range(training_dataset.number_classes)]
        return loaders

    def get_frames_per_sample(self) -> Optional[int]:
        """
        Gives the number of frames of a window, so that the frame throughput is printed with the sequence one.

        Returns:
            Optional[int]: The number of frames of a window.
        """
        return self.sequence_settings["sequence_length"]

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the sequences, frames and windows of a sequence split from its labels.json manifest.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of windows (the samples of an epoch) and their description with the number of
                sequences and frames.
        """
        number_sequences, number_frames, number_windows = count_split_sequences(
            split_data_folder=split_data_folder, sequence_settings=self.sequence_settings)
        return number_windows, (f"{number_sequences} sequences, {number_frames} frames, {number_windows} windows of "
                                f"{self.sequence_settings['sequence_length']} frames")

    def prepare_batch(self, batch: Dict[str, Any]) -> SequenceBatch:
        """
        Moves the frames, labels and sequence mask of a sequence batch to the device, the frames in the precision of
        the model.

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a sequence loader.

        Returns:
            SequenceBatch: The batch ready for the sequence model.
        """
        prepared_batch = super().prepare_batch(batch=batch)
        prepared_batch["sequence_mask"] = batch["sequence_mask"].to(device=self.device, dtype=torch.bool)
        return prepared_batch

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the sequence model with the labels of the batch, so that it returns its loss.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The model outputs, with its loss.
        """
        return self.model(input_tensor=batch["images"], sequence_mask=batch["sequence_mask"], labels=batch["labels"])

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the sequence model without labels, as in deployment.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The class logits and probabilities.
        """
        return self.model(input_tensor=batch["images"], sequence_mask=batch["sequence_mask"])

    def get_frame_captions(self, label: torch.Tensor, sequence_mask: torch.Tensor) -> Optional[List[str]]:
        """
        Gives the captions written on the frames of a window in the input viewer: none with one class per window,
        whose class is the caption of the whole window.

        Args:
            label (torch.Tensor): The label of the window.
            sequence_mask (torch.Tensor): The [T] booleans, True for the real frames.

        Returns:
            Optional[List[str]]: The caption of every real frame, or None.
        """
        return None

    def get_window_caption(self, label: torch.Tensor) -> str:
        """
        Gives the caption of a window in the input viewer: its class with one class per window.

        Args:
            label (torch.Tensor): The label of the window.

        Returns:
            str: The caption of the window.
        """
        return f"Class : {self.get_label_caption(label=label)}"

    def build_sample_frame_grid(self, frames: torch.Tensor, sequence_mask: torch.Tensor,
                                label: torch.Tensor) -> torch.Tensor:
        """
        Lays out the real frames of a sample on the grid shown by the input viewer, each frame with its caption. The
        sequence comparison trainer overrides it to stack the grids of the two members of a pair.

        Args:
            frames (torch.Tensor): The [T, 3, H, W] frames of the sample.
            sequence_mask (torch.Tensor): The [T] booleans, True for the real frames.
            label (torch.Tensor): The label of the sample.

        Returns:
            torch.Tensor: The [3, height, width] uint8 grid.
        """
        return build_frame_grid(frames=frames, sequence_mask=sequence_mask,
                                frame_captions=self.get_frame_captions(label=label, sequence_mask=sequence_mask),
                                number_columns=FRAME_GRID_COLUMNS)

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays the frames of an augmented training window next to those of a validation window, each laid out on a
        grid and captioned with its classes.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed windows in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        frame_grids, captions = [], []
        for batch in [training_batch, validation_batch]:
            label, sequence_mask = batch["labels"][sample_index], batch["sequence_mask"][sample_index]
            frame_grids.append(self.build_sample_frame_grid(frames=batch["images"][sample_index],
                                                            sequence_mask=sequence_mask, label=label))
            captions.append(self.get_window_caption(label=label))
        return view_input_data(images=frame_grids, bounding_boxes=None, labels=None,
                               window_titles=["Training", "Validation"], display_input_size=FRAME_GRID_DISPLAY_SIZE,
                               captions=captions)
