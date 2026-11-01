"""
Trainer of the sequence labelling task (M2M): one class per frame of the sequences, the accuracy and the evaluation of
the checkpoints counting only the labelled frames (frame metrics, error frames, predictions file and the share of fully
correct sequences), plus the segment metrics of the test sequences (edit score and segment F1), with the sample images
captioned frame by frame, the loaders being those of the sequence classification trainer. Registered in the task
registry.
"""

import cv2
import numpy as np
import torch

from typing import Any, Dict, List, Optional, Tuple

from vision_training.core.contracts import IGNORED_LABEL_INDEX, SAMPLE_IDENTIFIERS_KEY
from vision_training.evaluation.recall_intervals import (compute_bootstrap_recall_intervals,
                                                         describe_bootstrap_interval, sum_counts_by_group)
from vision_training.evaluation.sequence.segment_metrics import compute_segment_metrics, group_sequence_predictions
from vision_training.evaluation.sequence.sequence_metrics import compute_frame_accuracy, compute_sequence_exact_matches
from vision_training.trainers.sequence_classification_trainer import FRAME_GRID_COLUMNS, SequenceClassificationTrainer
from vision_training.utilities.visualization import (CORRECT_PREDICTION_COLOUR, WRONG_PREDICTION_COLOUR,
                                                     add_caption_bar, build_frame_grid, convert_image_to_array)


class SequenceLabellingTrainer(SequenceClassificationTrainer):
    """
    Trainer of the sequence labelling task (many to many): the sequence labeller trained on the whole sequences of
    'labels.json' manifests, one class per frame. The loss and the accuracy only count the labelled frames (not the
    padding frames nor the frames of unmapped categories), and every evaluated checkpoint is scored on the labelled
    frames of the test sequences with the classification metrics (frame accuracy, per-class precision, recall and F1,
    confusion matrix, class thresholds), plus the share of sequences whose every labelled frame is right, and with the
    segment metrics (edit score, segment F1 at 10, 25 and 50 % overlap).
    """

    TASK_NAME = "sequence_labelling"

    # Labels read from the manifests: one class per frame (many to many)
    LABEL_TYPE = "frame"

    # Name of the column of the evaluated samples of every class in the per-class table of the report: its labelled
    # frames, on which the classification metrics are computed
    CLASS_SUPPORT_COLUMN = "Frames"

    # Confidence interval of the frame recall of every class: a Korn-Graubard interval from a bootstrap over the test
    # sequences, the frames of a sequence being correlated (compute_recall_intervals)
    RECALL_INTERVAL_DESCRIPTION = describe_bootstrap_interval(group_name="test sequences")

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
        Creates the lists of class probabilities and labels of the labelled frames, of the sequence and position of
        every labelled frame, and of the exact matches of the sequences, filled during the evaluation of the test split,
        with the folder and the counts of the error images.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, which names its evaluation folder.

        Returns:
            Dict[str, Any]: The empty 'probabilities', 'labels', 'frame_sequence_identifiers', 'frame_positions' and
                'exact_matches' lists, the 'errors_folder' of the error images and their empty 'error_counts'.
        """
        return {"probabilities": [], "labels": [], "frame_sequence_identifiers": [], "frame_positions": [],
                "exact_matches": [],
                "errors_folder": self.get_evaluation_folder(iteration=iteration) / "Errors",
                "error_counts": {}}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the class probabilities and the labels of the labelled frames of the kept sequences of a test batch on
        the CPU, with the sequence and the position of every labelled frame, which name the frames in the predictions
        file and place them for the segment metrics, and whether every labelled frame of each sequence is right.

        Args:
            accumulator (Dict[str, Any]): The 'probabilities', 'labels', 'frame_sequence_identifiers', 'frame_positions'
                and 'exact_matches' lists, extended in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The sequence labeller outputs.
            sample_indices (List[int]): The indices of the sequences to keep.
        """
        probabilities = predictions["class_probabilities"][sample_indices].detach()
        labels = batch["labels"][sample_indices]
        labelled_frames = labels != IGNORED_LABEL_INDEX
        accumulator["probabilities"].append(probabilities[labelled_frames].cpu().double())
        accumulator["labels"].append(labels[labelled_frames].cpu())

        # Sequence and position of every labelled frame, in the order of the frames kept above
        sequence_identifiers = batch[SAMPLE_IDENTIFIERS_KEY][sample_indices].to(device=labels.device)
        frame_positions = torch.arange(labels.shape[1], device=labels.device).expand(labels.shape[0], -1)
        accumulator["frame_sequence_identifiers"].append(
            sequence_identifiers[:, None].expand(-1, labels.shape[1])[labelled_frames].cpu())
        accumulator["frame_positions"].append(frame_positions[labelled_frames].cpu())

        # Whether every labelled frame of each sequence is right
        accumulator["exact_matches"].append(compute_sequence_exact_matches(
            predicted_labels=probabilities.argmax(dim=-1), labels=labels).cpu())

        # Misclassified frames saved as images while their sequences are still in the batch
        if self.error_dumps_configuration["activated"]:
            self.dump_errors(accumulator=accumulator, batch=batch, predictions=predictions,
                             sample_indices=sample_indices)

    def dump_errors(self, accumulator: Dict[str, Any], batch: Dict[str, Any], predictions: Dict[str, Any],
                    sample_indices: List[int]) -> None:
        """
        Saves every misclassified frame of the sequences of a test batch as an image: the frame grid of its sequence,
        every real frame captioned with its true and predicted classes and the wrong ones marked with 'X', saved as
        'Errors/<TRUE>-TO-<PREDICTED>/<sequence>_frame<index>_<score>.png', the score being the probability of the
        predicted class of the frame. At most 'maximum_per_class_pair' frames are saved per pair of classes.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, holding the 'errors_folder' and 'error_counts'.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The sequence labeller outputs.
            sample_indices (List[int]): The indices of the sequences kept for the evaluation.
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

            # Frame grid of the sequence, built once, every real frame captioned with its true and predicted classes
            frame_captions = self.build_frame_prediction_captions(frame_labels=frame_labels,
                                                                  predicted_labels=predicted_labels,
                                                                  number_real_frames=number_real_frames)
            frame_grid = None
            sequence_identifier = int(batch[SAMPLE_IDENTIFIERS_KEY][sample_index])
            sequence_name = test_dataset.get_sample_name(sample_identifier=sequence_identifier)

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
                cv2.imwrite(filename=str(pair_folder / f"{sequence_name}_frame{frame_index}_{score:.3f}.png"),
                            img=add_caption_bar(image=frame_grid, caption=caption))

    def build_frame_prediction_captions(self, frame_labels: List[int], predicted_labels: List[int],
                                        number_real_frames: int) -> List[str]:
        """
        Captions every real frame of a test sequence with its true and predicted classes, 'true/predicted', the wrong
        frames being marked with 'X' and the frames of unmapped categories named 'none', for the frame grids of the
        error images and of the sample images.

        Args:
            frame_labels (List[int]): The class index of every frame of the sequence, IGNORED_LABEL_INDEX for the
                unlabelled and padding frames.
            predicted_labels (List[int]): The predicted class index of every frame of the sequence.
            number_real_frames (int): The number of real frames of the sequence, which come first.

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
        Draws a fixed test sequence for the sample images of the evaluation: the frame grid of the sequence, every real
        frame captioned with its true and predicted classes and the wrong ones marked with 'X', under a caption giving
        the number of labelled frames that are right, in green when they all are and in red otherwise.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The sequence labeller outputs.
            sample_index (int): The index of the sequence in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sequence.
        """
        frame_labels = batch["labels"][sample_index].tolist()
        predicted_labels = predictions["class_probabilities"][sample_index].detach().argmax(dim=-1).tolist()
        sequence_mask = batch["sequence_mask"][sample_index]
        number_real_frames = int(sequence_mask.sum())

        # Labelled real frames of the sequence and those predicted right
        labelled_frames = [frame_index for frame_index in range(number_real_frames)
                           if frame_labels[frame_index] != IGNORED_LABEL_INDEX]
        number_correct_frames = sum(frame_labels[frame_index] == predicted_labels[frame_index]
                                    for frame_index in labelled_frames)

        # Frame grid with the classes of every frame, under the caption of the sequence
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
        Builds the rows of the predictions file, one per labelled frame of the evaluated test sequences: its sequence
        and its frame file, followed by its prediction.

        Args:
            accumulator (Dict[str, Any]): The class probabilities and labels of the labelled frames, with their
                sequences and positions.

        Returns:
            List[Dict[str, Any]]: The rows of the predictions file, in the order of the evaluation.
        """
        probabilities = torch.cat(tensors=accumulator["probabilities"], dim=0)
        labels = torch.cat(tensors=accumulator["labels"], dim=0)
        sequence_identifiers = torch.cat(tensors=accumulator["frame_sequence_identifiers"], dim=0)
        frame_positions = torch.cat(tensors=accumulator["frame_positions"], dim=0)
        test_dataset = self.test_dataloader.dataset

        # The sequence and the file of every frame, then its prediction
        rows = []
        for row_index in range(len(labels)):
            sequence_identifier = int(sequence_identifiers[row_index])
            frame_names = test_dataset.get_sequence_frame_names(sample_identifier=sequence_identifier)
            row = {"sequence": test_dataset.describe_sample(sample_identifier=sequence_identifier)["sequence"],
                   "frame": frame_names[int(frame_positions[row_index])]}
            row.update(self.describe_prediction(probabilities=probabilities[row_index], label=labels[row_index]))
            rows.append(row)
        return rows

    def describe_evaluated_samples(self, accumulator: Dict[str, Any]) -> str:
        """
        Describes the evaluated test samples in the header of the report: the sequences and their labelled frames, on
        which the classification metrics are computed.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split.

        Returns:
            str: The description of the evaluated samples.
        """
        number_sequences = sum(len(exact_matches) for exact_matches in accumulator["exact_matches"])
        number_frames = sum(len(labels) for labels in accumulator["labels"])
        return f"{number_sequences} Test Sequences, {number_frames} Labelled Frames"

    def compute_recall_intervals(self, accumulator: Dict[str, Any], probabilities: torch.Tensor, labels: torch.Tensor,
                                 metrics: Dict[str, Any]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """
        Computes the confidence interval of the frame recall of every class of the test split, a Korn-Graubard interval
        from a bootstrap over the test sequences, written next to the recall in class_metrics.csv and shown in the
        recall table: the frames of a sequence share its content and are predicted together, so an exact interval over
        the frames would be far too narrow, while resampling whole sequences keeps their correlation in the variance
        that sets the effective sample size of the interval. The true positive and true frames of every class are
        added up per sequence from the most probable class of every labelled frame, as in the confusion matrix of the
        metrics.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, with the 'frame_sequence_identifiers' of the
                labelled frames.
            probabilities (torch.Tensor): The [F, K] class probabilities of the labelled frames.
            labels (torch.Tensor): The [F] true class of the labelled frames.
            metrics (Dict[str, Any]): The metrics of the evaluation, with the per-class 'support' of labelled frames.

        Returns:
            Tuple[List[Optional[float]], List[Optional[float]]]: The lower and upper bounds of every class, None for a
                class without labelled frames.

        Raises:
            ValueError: If the frames added up per sequence differ from the frames of the metrics.
        """
        # True frames of every class, and those whose most probable class is the true one, for every labelled frame
        true_frames = torch.nn.functional.one_hot(input=labels, num_classes=len(self.class_names))
        found_frames = true_frames * (probabilities.argmax(dim=1) == labels).unsqueeze(dim=1)

        # Both counts added up per test sequence, checked against the frames of the metrics
        frame_sequence_identifiers = torch.cat(tensors=accumulator["frame_sequence_identifiers"], dim=0).numpy()
        sequence_true_frames = sum_counts_by_group(group_identifiers=frame_sequence_identifiers,
                                                   counts=true_frames.numpy())
        sequence_found_frames = sum_counts_by_group(group_identifiers=frame_sequence_identifiers,
                                                    counts=found_frames.numpy())
        if sequence_true_frames.sum(axis=0).tolist() != [int(support) for support in metrics["support"]]:
            raise ValueError("Expected the labelled frames added up per test sequence to match the frames of the "
                             "classification metrics.")
        return compute_bootstrap_recall_intervals(group_true_positives=sequence_found_frames,
                                                  group_trials=sequence_true_frames)

    def compute_extra_headline_metrics(self, accumulator: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes the share of test sequences whose every labelled frame is right and the segment metrics of the test
        sequences, reported with the frame metrics.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split.

        Returns:
            Dict[str, float]: The 'Sequence Exact Match' of the test sequences, and the 'Edit Score' and the
                'Segment F1@<overlap>' of the test sequences.
        """
        exact_matches = torch.cat(tensors=accumulator["exact_matches"], dim=0)
        extra_metrics = {"Sequence Exact Match": float(exact_matches.to(dtype=torch.float64).mean())
                         if len(exact_matches) else 0.0}

        # Segment metrics of the sequences, the F1 being named after the overlap in percent
        segment_results = self.compute_segment_results(accumulator=accumulator)
        extra_metrics["Edit Score"] = segment_results["edit_score"]
        extra_metrics.update({f"Segment F1@{round(overlap_threshold * 100)}": segment_f1
                              for overlap_threshold, segment_f1 in segment_results["segment_f1"].items()})
        return extra_metrics

    def describe_extra_notes(self, accumulator: Dict[str, Any]) -> List[str]:
        """
        Counts the test sequences and their true and predicted segments behind the segment metrics.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split.

        Returns:
            List[str]: The note of the segments.
        """
        segment_results = self.compute_segment_results(accumulator=accumulator)
        return [f"Segments : {segment_results['number_true_segments']} true and "
                f"{segment_results['number_predicted_segments']} predicted in {segment_results['number_sequences']} "
                f"test sequences, an unlabelled frame ending a segment"]

    def compute_segment_results(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes once per evaluation the segment metrics of the test sequences, read by the headline metrics and the
        notes of the report: the labelled frames are grouped by sequence at their position in it, then the segments of
        every sequence are scored.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split, which keeps the
                results under 'segment_results'.

        Returns:
            Dict[str, Any]: The segment metrics and counts of compute_segment_metrics.
        """
        if "segment_results" in accumulator:
            return accumulator["segment_results"]

        # Sequence and position in it of every labelled frame, then the sequences scored
        sequences = group_sequence_predictions(
            sequence_indices=torch.cat(tensors=accumulator["frame_sequence_identifiers"], dim=0),
            frame_indices=torch.cat(tensors=accumulator["frame_positions"], dim=0),
            probabilities=torch.cat(tensors=accumulator["probabilities"], dim=0),
            labels=torch.cat(tensors=accumulator["labels"], dim=0))
        accumulator["segment_results"] = compute_segment_metrics(sequences=sequences)
        return accumulator["segment_results"]

    def get_frame_captions(self, label: torch.Tensor, sequence_mask: torch.Tensor) -> Optional[List[str]]:
        """
        Gives the captions written on the frames of a sequence in the input viewer: the class of every real frame, or
        'none' for the frames of unmapped categories.

        Args:
            label (torch.Tensor): The [T] class index of every frame of the sequence.
            sequence_mask (torch.Tensor): The [T] booleans, True for the real frames.

        Returns:
            Optional[List[str]]: The caption of every real frame.
        """
        return [self.class_names[int(frame_label)] if int(frame_label) != IGNORED_LABEL_INDEX else "none"
                for frame_label in label[sequence_mask]]

    def get_sample_caption(self, label: torch.Tensor) -> str:
        """
        Gives the caption of a sequence in the input viewer: the successive classes of its labelled frames, every run
        of frames of the same class being written once.

        Args:
            label (torch.Tensor): The [T] class index of every frame of the sequence.

        Returns:
            str: The caption of the sequence.
        """
        class_runs = []
        for frame_label in label.tolist():
            if frame_label != IGNORED_LABEL_INDEX and (not class_runs or class_runs[-1] != frame_label):
                class_runs.append(frame_label)
        return "Classes : " + " > ".join(self.class_names[class_index] for class_index in class_runs)
