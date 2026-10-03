import torch

from typing import Any, Dict, List, Optional

from vision_training.core.contracts import IGNORED_LABEL_INDEX, SAMPLE_IDENTIFIERS_KEY
from vision_training.evaluation.sequence.sequence_metrics import compute_frame_accuracy, compute_sequence_exact_matches
from vision_training.trainers.sequence_classification_trainer import SequenceClassificationTrainer


class SequenceLabellingTrainer(SequenceClassificationTrainer):
    """
    Trainer of the sequence labelling task (many to many): the sequence labeller trained on windows of frames cut
    from the sequences of 'labels.json' manifests, one class per frame. The loss and the accuracy only count the
    labelled frames (not the padding frames nor the frames of unmapped categories), and every evaluated checkpoint is
    scored on the labelled frames of the test windows with the classification metrics (frame accuracy, per-class
    precision, recall and F1, confusion matrix, class thresholds), plus the share of windows whose every labelled
    frame is right.
    """

    TASK_NAME = "sequence_labelling"

    # Labels read from the manifests: one class per frame (many to many)
    LABEL_TYPE = "frame"

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

    def create_evaluation_accumulator(self) -> Dict[str, Any]:
        """
        Creates the lists of class probabilities and labels of the labelled frames, of the window and position of
        every labelled frame, and of the exact matches of the windows, filled during the evaluation of the test split.

        Returns:
            Dict[str, Any]: The empty 'probabilities', 'labels', 'frame_window_identifiers', 'frame_positions' and
                'exact_matches' lists.
        """
        return {"probabilities": [], "labels": [], "frame_window_identifiers": [], "frame_positions": [],
                "exact_matches": []}

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
        accumulator["exact_matches"].append(compute_sequence_exact_matches(
            predicted_labels=probabilities.argmax(dim=-1), labels=labels).cpu())

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
        Describes the evaluated test samples in the report: the windows and their labelled frames, on which the
        classification metrics are computed.

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
        Computes the share of test windows whose every labelled frame is right, reported with the frame metrics.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split.

        Returns:
            Dict[str, float]: The 'Sequence Exact Match' of the test windows.
        """
        exact_matches = torch.cat(tensors=accumulator["exact_matches"], dim=0)
        return {"Sequence Exact Match": float(exact_matches.to(dtype=torch.float64).mean()) if len(exact_matches)
                else 0.0}

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
