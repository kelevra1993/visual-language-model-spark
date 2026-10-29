"""
Trainer of the sequence classification task (M2O): the loaders of the whole sequences of the 'labels.json'
manifests, the sequence classifier, the sequence augmentations and the frame grids of the input viewer,
the error images and the sample images, the losses and evaluation being those of the classification trainer. Registered
in the task registry and extended by the sequence labelling and sequence comparison trainers.
"""

import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision_training.core.contracts import SequenceBatch
from vision_training.data.loaders import get_sequence_dataloaders
from vision_training.data.split_description import count_split_sequences
from vision_training.models.sequence.temporal_modules import ORDERED_TEMPORAL_TYPES
from vision_training.trainers.classification_trainer import ClassificationTrainer
from vision_training.utilities.os_utilities import print_color
from vision_training.utilities.visualization import build_frame_grid, convert_image_to_array, view_input_data

# Frames per row of the frame grids of the input viewer, and longest side in pixels of their windows
FRAME_GRID_COLUMNS = 4
FRAME_GRID_DISPLAY_SIZE = 800


class SequenceClassificationTrainer(ClassificationTrainer):
    """
    Trainer of the sequence classification task (many to one): the sequence classifier trained on the whole sequences
    of 'labels.json' manifests (Native loader), one class per sequence. The losses, the accuracy tracked on every batch
    and the evaluation of the checkpoints are those of the classification task, a sequence being classified as a
    whole like an image.

    The sequence labelling trainer extends it with one class per frame, and the sequence comparison trainer with one
    class per pair of sequences.
    """

    TASK_NAME = "sequence_classification"
    SAMPLE_NAME = "Sequences"

    # Labels read from the manifests: one class per sequence (many to one)
    LABEL_TYPE = "sequence"

    # Real frames of the training batches counted for the frame throughput of the console summaries
    COUNTS_FRAMES = True

    # Sequence settings of the data augmentation, the photometric types drawn frame by frame and the shuffled frames
    SEQUENCE_AUGMENTATION = True

    # Sequences of a sample stored one after the other along its frame axis: one sequence, two members for the pairs
    MEMBERS_PER_SAMPLE = 1

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
        # Longest sequence fed to the model, the longer sequences of the manifests being left out
        self.sequence_settings = model_configuration["Data"]["sequence_settings"]

        # Shuffled frames carry no order, which an ordered temporal module would otherwise learn from
        augmentation_configuration = model_configuration["Data"]["data_augmentation"]
        temporal_type = model_configuration["TemporalModel"]["type"]
        if augmentation_configuration["activated"] and augmentation_configuration["shuffle_frames"]["activated"] and \
                temporal_type in ORDERED_TEMPORAL_TYPES:
            print_color(color="yellow", output=f"The frames of the training sequences are shuffled "
                                               f"('Data.data_augmentation.shuffle_frames') while the {temporal_type!r} "
                                               f"temporal module models their order.")
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
                                           label_type=self.LABEL_TYPE,
                                           drop_last_training_batch=self.drops_last_training_batch())
        training_dataset = loaders[0].dataset
        self.class_names = [training_dataset.class_names[class_index]
                            for class_index in range(training_dataset.number_classes)]
        return loaders

    def get_member_length(self, batch: Dict[str, Any]) -> int:
        """
        Gives the number of frames of the sequences of a batch, along which the training augmentation shuffles their
        real frames: the longest sequence of the batch, to which the collate pads the others, a pair holding
        MEMBERS_PER_SAMPLE members of that length.

        Args:
            batch (Dict[str, Any]): The batch, with its [B, MEMBERS_PER_SAMPLE x T] sequence mask.

        Returns:
            int: The number of frames T of a sequence, or of a member of a pair, of the batch.
        """
        return batch["sequence_mask"].shape[1] // self.MEMBERS_PER_SAMPLE

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the sequences, frames and kept sequences of a sequence split from its labels.json manifest.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of kept sequences (the samples of an epoch) and their description with the
                number of sequences and frames.
        """
        number_sequences, number_frames, number_kept_sequences = count_split_sequences(
            split_data_folder=split_data_folder, sequence_settings=self.sequence_settings)
        return number_kept_sequences, (f"{number_sequences} sequences, {number_frames} frames, "
                                       f"{number_kept_sequences} sequences of at most "
                                       f"{self.sequence_settings['maximum_sequence_length']} frames")

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
        Gives the captions written on the frames of a sequence in the input viewer: none with one class per sequence,
        whose class is the caption of the whole sequence.

        Args:
            label (torch.Tensor): The label of the sequence.
            sequence_mask (torch.Tensor): The [T] booleans, True for the real frames.

        Returns:
            Optional[List[str]]: The caption of every real frame, or None.
        """
        return None

    def get_sample_caption(self, label: torch.Tensor) -> str:
        """
        Gives the caption of a sequence in the input viewer: its class with one class per sequence.

        Args:
            label (torch.Tensor): The label of the sequence.

        Returns:
            str: The caption of the sequence.
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

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the frames of a training batch of sequences (one draw per sequence for every type, or per
        frame for the photometric types with 'independent_frames'), then shuffles the real frames of the sequences when
        'shuffle_frames' is on, the frame labels of the sequence labelling task following their frames. The
        comparison trainer inherits it, each member of a pair being shuffled on its own. The sequences have the length
        of the longest sequence of the batch.

        Args:
            batch (Dict[str, Any]): The training batch, with its sequence mask.

        Returns:
            Dict[str, Any]: The batch with its augmented and possibly shuffled frames and frame labels, the
                'augmentation_reports' of its sequences and their [B, T] 'frame_orders' for the input viewer.
        """
        augmented_batch = super().augment_batch(batch=batch)
        shuffled_images, frame_orders = self.data_augmenter.shuffle_frames(
            images=augmented_batch["images"], sequence_mask=augmented_batch["sequence_mask"],
            frame_block_length=self.get_member_length(batch=augmented_batch))
        augmented_batch["images"] = shuffled_images
        augmented_batch["frame_orders"] = frame_orders

        # One class per frame follows its frame, the sequence mask being unchanged since the real frames stay first
        if augmented_batch["labels"].dim() == 2:
            augmented_batch["labels"] = torch.gather(input=augmented_batch["labels"], dim=1,
                                                     index=frame_orders.to(device=augmented_batch["labels"].device))
        return augmented_batch

    def build_input_image(self, batch: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Builds the image of a test sequence shown by the error images and the sample images of the evaluation: its real
        frames laid out on a grid, the before member above the after member for a pair.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            sample_index (int): The index of the sequence (or pair) in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR frame grid of the sample.
        """
        frame_grid = self.build_sample_frame_grid(frames=batch["images"][sample_index],
                                                  sequence_mask=batch["sequence_mask"][sample_index],
                                                  label=batch["labels"][sample_index])
        return convert_image_to_array(image=frame_grid)

    def describe_frame_order(self, batch: Dict[str, Any], sample_index: int) -> List[str]:
        """
        Writes the frame order of a sequence in its information panel of the input viewer, when the training
        augmentation shuffles the frames ('Data.data_augmentation.shuffle_frames'): the original frame shown at every
        position of the grid, so that the frame by frame augmentation lines, numbered before the shuffle, can be
        matched with the frames on the grid. The validation sequences are never shuffled.

        Args:
            batch (Dict[str, Any]): The augmented training batch, with its 'frame_orders', or the validation batch.
            sample_index (int): The index of the displayed sequence (or pair) in its batch.

        Returns:
            List[str]: The frame order line, or no line when the frames are not shuffled.
        """
        if "frame_orders" not in batch or not (self.data_augmenter.activated and
                                               self.data_augmenter.shuffle_frames_activated):
            return []

        # The original index of the real frames of every member, the members of a pair separated by a bar
        frame_order = batch["frame_orders"][sample_index]
        sequence_mask = batch["sequence_mask"][sample_index].to(dtype=torch.bool)
        member_length = self.get_member_length(batch=batch)
        member_orders = []
        for block_start in range(0, sequence_mask.shape[0], member_length):
            block_order = frame_order[block_start:block_start + member_length]
            member_orders.append(block_order[sequence_mask[block_start:block_start + member_length]].tolist())
        if all(order == sorted(order) for order in member_orders):
            return ["Frame order : kept"]
        return ["Frame order : shuffled, showing frames " +
                " | ".join(", ".join(str(frame_index) for frame_index in order) for order in member_orders)]

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays the frames of an augmented training sequence next to those of a validation sequence, each laid out on a
        grid next to an information panel holding its classes, its number of real frames, the augmentations drawn for
        every real frame and the order of its frames.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed sequences in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        frame_grids, panel_lines = [], []
        for batch in [training_batch, validation_batch]:
            label, sequence_mask = batch["labels"][sample_index], batch["sequence_mask"][sample_index]
            frame_grids.append(self.build_sample_frame_grid(frames=batch["images"][sample_index],
                                                            sequence_mask=sequence_mask, label=label))

            # The real frames of every member, which stay first in their member after a shuffle
            member_length = self.get_member_length(batch=batch)
            member_frame_counts = [str(int(sequence_mask[block_start:block_start + member_length].sum()))
                                   for block_start in range(0, sequence_mask.shape[0], member_length)]
            real_frames = torch.nonzero(input=sequence_mask).flatten().tolist()

            # The classes and the real frames of the sequence, then its augmentations and its frame order
            panel_lines.append([self.get_sample_caption(label=label),
                                f"Real frames : {' + '.join(member_frame_counts)}"] +
                               self.describe_input_augmentations(batch=batch, sample_index=sample_index,
                                                                 real_frames=real_frames) +
                               self.describe_frame_order(batch=batch, sample_index=sample_index))
        return view_input_data(images=frame_grids, bounding_boxes=None, labels=None,
                               window_titles=["Training", "Validation"], display_input_size=FRAME_GRID_DISPLAY_SIZE,
                               panel_lines=panel_lines)
