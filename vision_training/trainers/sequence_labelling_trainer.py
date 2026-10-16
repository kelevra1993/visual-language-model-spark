"""
Trainer of the sequence labelling task (M2M): one class per frame of the windows, the accuracy and the evaluation of
the checkpoints counting only the labelled frames (frame metrics, error frames, predictions file and the share of fully
correct windows), plus the segment metrics of the whole test sequences rebuilt from their windows (edit score and
segment F1), with the sample images captioned frame by frame, the loaders being those of the sequence classification
trainer. Registered in the task registry.
"""

import cv2
import numpy as np
import torch

from typing import Any, Dict, List, Optional

from vision_training.core.contracts import IGNORED_LABEL_INDEX, SAMPLE_IDENTIFIERS_KEY
from vision_training.evaluation.sequence.segment_metrics import compute_segment_metrics, merge_window_predictions
from vision_training.evaluation.sequence.sequence_metrics import compute_frame_accuracy, compute_sequence_exact_matches
from vision_training.trainers.sequence_classification_trainer import FRAME_GRID_COLUMNS, SequenceClassificationTrainer
from vision_training.utilities.visualization import (CORRECT_PREDICTION_COLOUR, WRONG_PREDICTION_COLOUR,
                                                     add_caption_bar, build_frame_grid, convert_image_to_array)


class SequenceLabellingTrainer(SequenceClassificationTrainer):
    """
    Trainer of the sequence labelling task (many to many): the sequence labeller trained on windows of frames cut
    from the sequences of 'labels.json' manifests, one class per frame. The loss and the accuracy only count the
    labelled frames (not the padding frames nor the frames of unmapped categories), and every evaluated checkpoint is
    scored on the labelled frames of the test windows with the classification metrics (frame accuracy, per-class
    precision, recall and F1, confusion matrix, class thresholds), plus the share of windows whose every labelled
    frame is right, and on the whole test sequences rebuilt from their windows with the segment metrics (edit score,
    segment F1 at 10, 25 and 50 % overlap).
    """

    TASK_NAME = "sequence_labelling"

    # Labels read from the manifests: one class per frame (many to many)
    LABEL_TYPE = "frame"

    # Name of the column of the evaluated samples of every class in the per-class table of the report: its labelled
    # frames, on which the classification metrics are computed
    CLASS_SUPPORT_COLUMN = "Frames"

    def compute_batch_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Computes the accuracy of a training or validation batch over its labelled frames, from the logits of its
        forward pass.

        Args:
            batch (Dict[str, Any]): The prepared batch.
            model_outputs (Dict[str, Any]): The sequence labeller outputs.

        Returns:
            Dict[str, torch.Tensor]: The 'accuracy' of the batch.
        """
        with torch.no_grad():
            return {"accuracy": compute_frame_accuracy(predicted_labels=model_outputs["class_logits"].argmax(dim=-1),
                                                       labels=batch["labels"])}

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the lists of class probabilities and labels of the labelled frames, of the window and position of
        every labelled frame, and of the exact matches of the windows, filled during the evaluation of the test split,
        with the folder and the counts of the error images.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, which names its evaluation folder.

        Returns:
            Dict[str, Any]: The empty 'probabilities', 'labels', 'frame_window_identifiers', 'frame_positions' and
                'exact_matches' lists, the 'errors_folder' of the error images and their empty 'error_counts'.
        """
        return {"probabilities": [], "labels": [], "frame_window_identifiers": [], "frame_positions": [],
                "exact_matches": [],
                "errors_folder": self.get_evaluation_folder(iteration=iteration) / "Errors",
                "error_counts": {}}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the class probabilities and the labels of the labelled frames of the kept windows of a test batch on the
        CPU, with the window and the position of every labelled frame, which name the frames in the predictions file,
        and whether every labelled frame of each window is right.

        Args:
            accumulator (Dict[str, Any]): The 'probabilities', 'labels', 'frame_window_identifiers', 'frame_positions'
                and 'exact_matches' lists, extended in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The sequence labeller outputs.
            sample_indices (List[int]): The indices of the windows to keep.
        """
        probabilities = predictions["class_probabilities"][sample_indices].detach()
        labels = batch["labels"][sample_indices]
        labelled_frames = labels != IGNORED_LABEL_INDEX
        accumulator["probabilities"].append(probabilities[labelled_frames].cpu().double())
        accumulator["labels"].append(labels[labelled_frames].cpu())

        # Window and position of every labelled frame, in the order of the frames kept above
        window_identifiers = batch[SAMPLE_IDENTIFIERS_KEY][sample_indices].to(device=labels.device)
        frame_positions = torch.arange(labels.shape[1], device=labels.device).expand(labels.shape[0], -1)
        accumulator["frame_window_identifiers"].append(
            window_identifiers[:, None].expand(-1, labels.shape[1])[labelled_frames].cpu())
        accumulator["frame_positions"].append(frame_positions[labelled_frames].cpu())

        # Whether every labelled frame of each window is right
        accumulator["exact_matches"].append(compute_sequence_exact_matches(
            predicted_labels=probabilities.argmax(dim=-1), labels=labels).cpu())

        # Misclassified frames saved as images while their windows are still in the batch
        if self.error_dumps_configuration["activated"]:
            self.dump_errors(accumulator=accumulator, batch=batch, predictions=predictions,
                             sample_indices=sample_indices)

    def dump_errors(self, accumulator: Dict[str, Any], batch: Dict[str, Any], predictions: Dict[str, Any],
                    sample_indices: List[int]) -> None:
        """
        Saves every misclassified frame of the windows of a test batch as an image: the frame grid of its window, every
        real frame captioned with its true and predicted classes and the wrong ones marked with 'X', saved as
        'Errors/<TRUE>-TO-<PREDICTED>/<window>_frame<index>_<score>.png', the score being the probability of the
        predicted class of the frame. At most 'maximum_per_class_pair' frames are saved per pair of classes.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, holding the 'errors_folder' and 'error_counts'.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The sequence labeller outputs.
            sample_indices (List[int]): The indices of the windows kept for the evaluation.
        """
        probabilities = predictions["class_probabilities"].detach()
        test_dataset = self.test_dataloader.dataset
        for sample_index in sample_indices:
            frame_labels = batch["labels"][sample_index].tolist()
            predicted_labels = probabilities[sample_index].argmax(dim=-1).tolist()
            sequence_mask = batch["sequence_mask"][sample_index]
            number_real_frames = int(sequence_mask.sum())
            wrong_frames = [frame_index for frame_index in range(number_real_frames)
                            if frame_labels[frame_index] not in [IGNORED_LABEL_INDEX, predicted_labels[frame_index]]]
            if not wrong_frames:
                continue

            # Frame grid of the window, built once, every real frame captioned with its true and predicted classes
            frame_captions = self.build_frame_prediction_captions(frame_labels=frame_labels,
                                                                  predicted_labels=predicted_labels,
                                                                  number_real_frames=number_real_frames)
            frame_grid = None
            window_identifier = int(batch[SAMPLE_IDENTIFIERS_KEY][sample_index])
            window_name = test_dataset.get_sample_name(sample_identifier=window_identifier)

            # One image per wrong frame, in the folder of its class pair unless it is full
            for frame_index in wrong_frames:
                pair_folder = self.reserve_error_image(accumulator=accumulator,
                                                       true_class_index=frame_labels[frame_index],
                                                       predicted_class_index=predicted_labels[frame_index])
                if pair_folder is None:
                    continue
                if frame_grid is None:
                    frame_grid = convert_image_to_array(image=build_frame_grid(
                        frames=batch["images"][sample_index], sequence_mask=sequence_mask,
                        frame_captions=frame_captions, number_columns=FRAME_GRID_COLUMNS))
                score = float(probabilities[sample_index, frame_index, predicted_labels[frame_index]])
                caption = (f"frame {frame_index} : {self.class_names[frame_labels[frame_index]]} -> "
                           f"{self.class_names[predicted_labels[frame_index]]} {score:.3f}")
                cv2.imwrite(filename=str(pair_folder / f"{window_name}_frame{frame_index}_{score:.3f}.png"),
                            img=add_caption_bar(image=frame_grid, caption=caption))

    def build_frame_prediction_captions(self, frame_labels: List[int], predicted_labels: List[int],
                                        number_real_frames: int) -> List[str]:
        """
        Captions every real frame of a test window with its true and predicted classes, 'true/predicted', the wrong
        frames being marked with 'X' and the frames of unmapped categories named 'none', for the frame grids of the
        error images and of the sample images.

        Args:
            frame_labels (List[int]): The class index of every frame of the window, IGNORED_LABEL_INDEX for the
                unlabelled and padding frames.
            predicted_labels (List[int]): The predicted class index of every frame of the window.
            number_real_frames (int): The number of real frames of the window, which come first.

        Returns:
            List[str]: The caption of every real frame.
        """
        frame_captions = []
        for frame_index in range(number_real_frames):
            true_label, predicted_label = frame_labels[frame_index], predicted_labels[frame_index]
            wrong_frame = true_label not in [IGNORED_LABEL_INDEX, predicted_label]
            true_class_name = self.class_names[true_label] if true_label != IGNORED_LABEL_INDEX else "none"
            frame_captions.append(("X " if wrong_frame else "") +
                                  f"{true_class_name}/{self.class_names[predicted_label]}")
        return frame_captions

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws a fixed test window for the sample images of the evaluation: the frame grid of the window, every real
        frame captioned with its true and predicted classes and the wrong ones marked with 'X', under a caption giving
        the number of labelled frames that are right, in green when they all are and in red otherwise.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The sequence labeller outputs.
            sample_index (int): The index of the window in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the window.
        """
        frame_labels = batch["labels"][sample_index].tolist()
        predicted_labels = predictions["class_probabilities"][sample_index].detach().argmax(dim=-1).tolist()
        sequence_mask = batch["sequence_mask"][sample_index]
        number_real_frames = int(sequence_mask.sum())

        # Labelled real frames of the window and those predicted right
        labelled_frames = [frame_index for frame_index in range(number_real_frames)
                           if frame_labels[frame_index] != IGNORED_LABEL_INDEX]
        number_correct_frames = sum(frame_labels[frame_index] == predicted_labels[frame_index]
                                    for frame_index in labelled_frames)

        # Frame grid with the classes of every frame, under the caption of the window
        frame_captions = self.build_frame_prediction_captions(frame_labels=frame_labels,
                                                              predicted_labels=predicted_labels,
                                                              number_real_frames=number_real_frames)
        frame_grid = convert_image_to_array(image=build_frame_grid(
            frames=batch["images"][sample_index], sequence_mask=sequence_mask, frame_captions=frame_captions,
            number_columns=FRAME_GRID_COLUMNS))
        return add_caption_bar(image=frame_grid,
                               caption=f"{number_correct_frames}/{len(labelled_frames)} labelled frames right "
                                       f"(true/predicted)",
                               caption_colour=CORRECT_PREDICTION_COLOUR
                               if number_correct_frames == len(labelled_frames) else WRONG_PREDICTION_COLOUR)

    def build_prediction_rows(self, accumulator: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Builds the rows of the predictions file, one per labelled frame of the evaluated test windows: its sequence and
        its frame file, followed by its prediction.

        Args:
            accumulator (Dict[str, Any]): The class probabilities and labels of the labelled frames, with their windows
                and positions.

        Returns:
            List[Dict[str, Any]]: The rows of the predictions file, in the order of the evaluation.
        """
        probabilities = torch.cat(tensors=accumulator["probabilities"], dim=0)
        labels = torch.cat(tensors=accumulator["labels"], dim=0)
        window_identifiers = torch.cat(tensors=accumulator["frame_window_identifiers"], dim=0)
        frame_positions = torch.cat(tensors=accumulator["frame_positions"], dim=0)
        test_dataset = self.test_dataloader.dataset

        # The sequence and the file of every frame, then its prediction
        rows = []
        for row_index in range(len(labels)):
            window_identifier = int(window_identifiers[row_index])
            frame_names = test_dataset.get_window_frame_names(sample_identifier=window_identifier)
            row = {"sequence": test_dataset.describe_sample(sample_identifier=window_identifier)["sequence"],
                   "frame": frame_names[int(frame_positions[row_index])]}
            row.update(self.describe_prediction(probabilities=probabilities[row_index], label=labels[row_index]))
            rows.append(row)
        return rows

    def describe_evaluated_samples(self, accumulator: Dict[str, Any]) -> str:
        """
        Describes the evaluated test samples in the header of the report: the windows and their labelled frames, on
        which the classification metrics are computed.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split.

        Returns:
            str: The description of the evaluated samples.
        """
        number_windows = sum(len(exact_matches) for exact_matches in accumulator["exact_matches"])
        number_frames = sum(len(labels) for labels in accumulator["labels"])
        return f"{number_windows} Test Sequences, {number_frames} Labelled Frames"

    def compute_extra_headline_metrics(self, accumulator: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes the share of test windows whose every labelled frame is right and the segment metrics of the whole
        test sequences, reported with the frame metrics.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split.

        Returns:
            Dict[str, float]: The 'Sequence Exact Match' of the test windows, and the 'Edit Score' and the
                'Segment F1@<overlap>' of the test sequences.
        """
        exact_matches = torch.cat(tensors=accumulator["exact_matches"], dim=0)
        extra_metrics = {"Sequence Exact Match": float(exact_matches.to(dtype=torch.float64).mean())
                         if len(exact_matches) else 0.0}

        # Segment metrics of the whole sequences, the F1 being named after the overlap in percent
        segment_results = self.compute_segment_results(accumulator=accumulator)
        extra_metrics["Edit Score"] = segment_results["edit_score"]
        extra_metrics.update({f"Segment F1@{round(overlap_threshold * 100)}": segment_f1
                              for overlap_threshold, segment_f1 in segment_results["segment_f1"].items()})
        return extra_metrics

    def describe_extra_notes(self, accumulator: Dict[str, Any]) -> List[str]:
        """
        Counts the whole test sequences and their true and predicted segments behind the segment metrics, the windows
        being merged back into their sequences.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split.

        Returns:
            List[str]: The note of the segments.
        """
        segment_results = self.compute_segment_results(accumulator=accumulator)
        return [f"Segments : {segment_results['number_true_segments']} true and "
                f"{segment_results['number_predicted_segments']} predicted in {segment_results['number_sequences']} "
                f"whole test sequences, their windows being merged and an unlabelled frame ending a segment"]

    def compute_segment_results(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes once per evaluation the segment metrics of the test sequences, read by the headline metrics and the
        notes of the report: every labelled frame is placed in its sequence from the start of its window, the windows
        of a sequence are merged, then the segments of every sequence are scored.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split, which keeps the
                results under 'segment_results'.

        Returns:
            Dict[str, Any]: The segment metrics and counts of compute_segment_metrics.
        """
        if "segment_results" in accumulator:
            return accumulator["segment_results"]
        test_dataset = self.test_dataloader.dataset
        window_identifiers = torch.cat(tensors=accumulator["frame_window_identifiers"], dim=0)

        # Sequence index and first frame of every evaluated window, the sequences being numbered by name
        sequence_numbers, window_sequence_indices, window_starts = {}, {}, {}
        for window_identifier in torch.unique(input=window_identifiers).tolist():
            sequence_name = test_dataset.describe_sample(sample_identifier=window_identifier)["sequence"]
            window_sequence_indices[window_identifier] = sequence_numbers.setdefault(sequence_name,
                                                                                     len(sequence_numbers))
            window_starts[window_identifier] = test_dataset.get_window_start(sample_identifier=window_identifier)

        # Sequence and position in it of every labelled frame, then the whole sequences scored
        frame_window_identifiers = window_identifiers.tolist()
        sequences = merge_window_predictions(
            sequence_indices=torch.tensor(data=[window_sequence_indices[window_identifier]
                                                for window_identifier in frame_window_identifiers]),
            frame_indices=torch.tensor(data=[window_starts[window_identifier]
                                             for window_identifier in frame_window_identifiers]) +
            torch.cat(tensors=accumulator["frame_positions"], dim=0),
            probabilities=torch.cat(tensors=accumulator["probabilities"], dim=0),
            labels=torch.cat(tensors=accumulator["labels"], dim=0))
        accumulator["segment_results"] = compute_segment_metrics(sequences=sequences)
        return accumulator["segment_results"]

    def get_frame_captions(self, label: torch.Tensor, sequence_mask: torch.Tensor) -> Optional[List[str]]:
        """
        Gives the captions written on the frames of a window in the input viewer: the class of every real frame, or
        'none' for the frames of unmapped categories.

        Args:
            label (torch.Tensor): The [T] class index of every frame of the window.
            sequence_mask (torch.Tensor): The [T] booleans, True for the real frames.

        Returns:
            Optional[List[str]]: The caption of every real frame.
        """
        return [self.class_names[int(frame_label)] if int(frame_label) != IGNORED_LABEL_INDEX else "none"
                for frame_label in label[sequence_mask]]

    def get_window_caption(self, label: torch.Tensor) -> str:
        """
        Gives the caption of a window in the input viewer: the successive classes of its labelled frames, every run of
        frames of the same class being written once.

        Args:
            label (torch.Tensor): The [T] class index of every frame of the window.

        Returns:
            str: The caption of the window.
        """
        class_runs = []
        for frame_label in label.tolist():
            if frame_label != IGNORED_LABEL_INDEX and (not class_runs or class_runs[-1] != frame_label):
                class_runs.append(frame_label)
        return "Classes : " + " > ".join(self.class_names[class_index] for class_index in class_runs)
