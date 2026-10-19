"""
Trainer of the frame selection task (SEL): the frame selector trained on whole sequences of the 'labels.json'
manifests (every wanted class present 'frames_per_class' times in every sequence), every evaluated checkpoint
being scored with both decodings of the selection (Hungarian and greedy) on the same probabilities, with the ranks of
the true frames, the per-frame metrics, the confidence thresholds of the chosen frames, the selections file and the
sequence grids of the sample and error images. Built on the sequence labelling trainer and registered in the task
registry.
"""

import csv
import json
import cv2
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision_training.core.contracts import SAMPLE_IDENTIFIERS_KEY, EvaluationResults, EvaluationTable
from vision_training.data.split_description import count_split_sequences
from vision_training.evaluation.classification.classification_metrics import (compute_confusion_matrix,
                                                                              compute_metrics_from_confusion_matrix)
from vision_training.evaluation.classification.confusion_matrix import (save_confusion_matrix_image,
                                                                        save_confusion_matrix_table)
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.evaluation.sequence.frame_selection_metrics import (compute_ranking_metrics,
                                                                         compute_selection_metrics,
                                                                         find_selection_thresholds)
from vision_training.models.sequence.frame_selection_decoding import (NO_FRAME, VALID_DECODINGS,
                                                                      convert_selection_to_frame_labels,
                                                                      decode_frame_selection, flag_low_confidence,
                                                                      gather_frame_probabilities,
                                                                      get_wanted_class_indices)
from vision_training.models.task_model import TaskModel
from vision_training.trainers.sequence_labelling_trainer import SequenceLabellingTrainer
from vision_training.utilities.visualization import build_frame_selection_image, convert_image_to_array

# Title of the per-class table of the report
SELECTION_TABLE_TITLE = "Per Class Selection"


class FrameSelectionTrainer(SequenceLabellingTrainer):
    """
    Trainer of the frame selection task: the frame selector trained on whole sequences, one class per frame (a wanted
    class or the unselected class), with the per-frame loss and the selection loss. Every evaluated checkpoint chooses
    the frames of every wanted class of the test sequences with both decodings, scored side by side (selection
    accuracy of every class, sequences fully right, unselected precision and recall), with the top-k accuracy and mean
    rank of the true frames before decoding and the per-frame metrics. The decoding of 'Selection.decoding' gives the
    confidence thresholds, the decoded confusion matrix and the sample and error images.
    """

    TASK_NAME = "frame_selection"

    # Labels read from the manifests: one class per frame of a whole sequence, the frames of unmapped categories being
    # unselected
    LABEL_TYPE = "selection"

    # Name of the column of the true frames of every wanted class in the per-class table of the report
    CLASS_SUPPORT_COLUMN = "True Frames"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the selection settings used by the hooks, then prepares the training run through the sequence labelling
        trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data with its sequence settings,
                Backbone, TemporalModel, ClassificationHead, Selection).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Decoding drawn in the images and giving the thresholds, and number of frames chosen for every wanted class
        self.decoding = model_configuration["Selection"]["decoding"]
        self.frames_per_class = model_configuration["Selection"]["frames_per_class"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the loaders of the whole sequences of the three splits, takes the class names from the categories of
        the training manifest, and the unselected class from its category, which must be the same class in every
        split.

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders.

        Raises:
            ValueError: If the unselected category gives another class in the validation or test manifest.
        """
        loaders = super().build_dataloaders()
        unselected_class_indices = [loader.dataset.unselected_class_index for loader in loaders]
        if len(set(unselected_class_indices)) != 1:
            raise ValueError(f"The unselected category gives the classes {unselected_class_indices} in the training, "
                             f"validation and test manifests, which must agree.")
        self.unselected_class_index = unselected_class_indices[0]
        self.wanted_class_indices = get_wanted_class_indices(number_classes=len(self.class_names),
                                                             unselected_class_index=self.unselected_class_index)
        return loaders

    def get_unmapped_class_index(self) -> Optional[int]:
        """
        Gives the unselected class, to which the loaders give the frames of the categories the label mapping does not
        map, so that the record of the data counts them as unselected frames.

        Returns:
            Optional[int]: The class index of the unselected frames.
        """
        return self.unselected_class_index

    def build_model(self) -> TaskModel:
        """
        Builds the frame selector through the classification trainer (with the balanced class weights when asked),
        and marks its unselected class, so that the selection loss only covers the wanted classes and the exports know
        the unselected class.

        Returns:
            TaskModel: The untrained frame selector.
        """
        model = super().build_model()
        model.wanted_class_mask[self.unselected_class_index] = 0.0
        return model

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the losses and the frame accuracy tracked on every training and validation batch.

        Returns:
            Dict[str, str]: The display name of every tracked metric.
        """
        return {"total_loss": "Total Loss", "classification_loss": "Classification Loss",
                "selection_loss": "Selection Loss", "accuracy": "Accuracy"}

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the sequences, frames and kept sequences of a frame selection split from its labels.json manifest,
        every kept sequence being given whole.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of kept sequences (the samples of an epoch) and their description.
        """
        number_sequences, number_frames, number_kept_sequences = count_split_sequences(
            split_data_folder=split_data_folder, sequence_settings=self.sequence_settings)
        return number_kept_sequences, (f"{number_sequences} sequences, {number_frames} frames, "
                                       f"{number_kept_sequences} sequences of at most "
                                       f"{self.sequence_settings['maximum_sequence_length']} frames")

    def get_sequence_predictions(self, batch: Dict[str, Any], predictions: Dict[str, Any],
                                 sample_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads the class probabilities and the true classes of the real frames of one test sequence of a batch, the
        inputs of the decodings, of the metrics and of the images.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The frame selector outputs.
            sample_index (int): The index of the sequence in the batch.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The [T, K] float64 class probabilities and the [T] int64 true classes of
                the real frames.
        """
        number_real_frames = int(batch["sequence_mask"][sample_index].sum())
        probabilities = predictions["class_probabilities"][sample_index, :number_real_frames].detach().double().cpu()
        frame_labels = batch["labels"][sample_index, :number_real_frames].detach().cpu()
        return probabilities.numpy(), frame_labels.numpy().astype(np.int64)

    def decode_sequence(self, probabilities: np.ndarray, decoding: str) -> np.ndarray:
        """
        Chooses the frames of every wanted class of one sequence with a decoding, with the settings of the run.

        Args:
            probabilities (np.ndarray): The [T, K] class probabilities of the frames of the sequence.
            decoding (str): "hungarian" or "greedy".

        Returns:
            np.ndarray: The [W, N] chosen frames of every wanted class.
        """
        return decode_frame_selection(probabilities=probabilities, unselected_class_index=self.unselected_class_index,
                                      frames_per_class=self.frames_per_class, decoding=decoding)

    def build_selection_image(self, batch: Dict[str, Any], sample_index: int, probabilities: np.ndarray,
                              frame_labels: np.ndarray, selected_frames: np.ndarray) -> np.ndarray:
        """
        Draws one test sequence with its selection for the sample and error images: the grid of its frames with the
        selection borders, the probability heat map and the number of right selections.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            sample_index (int): The index of the sequence in the batch.
            probabilities (np.ndarray): The [T, K] class probabilities of the real frames of the sequence.
            frame_labels (np.ndarray): The [T] true classes of the real frames.
            selected_frames (np.ndarray): The [W, N] chosen frames of every wanted class.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sequence.
        """
        frames = convert_image_to_array(image=batch["images"][sample_index, :len(frame_labels)])
        return build_frame_selection_image(frames=frames, probabilities=probabilities, frame_labels=frame_labels,
                                           selected_frames=selected_frames,
                                           wanted_class_indices=self.wanted_class_indices,
                                           unselected_class_index=self.unselected_class_index,
                                           class_names=self.class_names)

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the lists of class probabilities and true classes of the frames of every test sequence, and of the
        sequence identifiers, filled during the evaluation of the test split, with the folder and the counts of the
        error images.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, which names its evaluation folder.

        Returns:
            Dict[str, Any]: The empty 'probabilities', 'frame_labels' and 'sequence_identifiers' lists, the
                'errors_folder' of the error images and their empty 'error_counts' per class.
        """
        return {"probabilities": [], "frame_labels": [], "sequence_identifiers": [],
                "errors_folder": self.get_evaluation_folder(iteration=iteration) / "Errors",
                "error_counts": {}}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the class probabilities and the true classes of the real frames of every kept test sequence on the CPU,
        decoded once the whole test split is evaluated, and saves the sequences with a wrong selection as images when
        the error images are on.

        Args:
            accumulator (Dict[str, Any]): The 'probabilities', 'frame_labels' and 'sequence_identifiers' lists,
                extended in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The frame selector outputs.
            sample_indices (List[int]): The indices of the sequences to keep.
        """
        for sample_index in sample_indices:
            probabilities, frame_labels = self.get_sequence_predictions(batch=batch, predictions=predictions,
                                                                        sample_index=sample_index)
            accumulator["probabilities"].append(probabilities)
            accumulator["frame_labels"].append(frame_labels)
            accumulator["sequence_identifiers"].append(int(batch[SAMPLE_IDENTIFIERS_KEY][sample_index]))

        # Sequences with a wrong selection saved as images while their frames are still in the batch
        if self.error_dumps_configuration["activated"]:
            self.dump_errors(accumulator=accumulator, batch=batch, predictions=predictions,
                             sample_indices=sample_indices)

    def dump_errors(self, accumulator: Dict[str, Any], batch: Dict[str, Any], predictions: Dict[str, Any],
                    sample_indices: List[int]) -> None:
        """
        Saves every test sequence whose selection of the decoding of the run is wrong for a wanted class as an image,
        'Errors/<CLASS>/<sequence>.png' for every class picked wrong, the image being the one of the sample images. At
        most 'maximum_per_class_pair' images are saved per class.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, holding the 'errors_folder' and 'error_counts'.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The frame selector outputs.
            sample_indices (List[int]): The indices of the sequences kept for the evaluation.
        """
        test_dataset = self.test_dataloader.dataset
        for sample_index in sample_indices:
            probabilities, frame_labels = self.get_sequence_predictions(batch=batch, predictions=predictions,
                                                                        sample_index=sample_index)
            selected_frames = self.decode_sequence(probabilities=probabilities, decoding=self.decoding)

            # Wanted classes whose chosen frames are not exactly their true frames
            wrong_class_indices = []
            for wanted_index, class_index in enumerate(self.wanted_class_indices):
                chosen_frames = set(selected_frames[wanted_index][selected_frames[wanted_index] != NO_FRAME].tolist())
                if chosen_frames != set(np.flatnonzero(frame_labels == class_index).tolist()):
                    wrong_class_indices.append(class_index)
            if not wrong_class_indices:
                continue

            # One image of the sequence, built once and saved in the folder of every class picked wrong unless full
            sequence_image = None
            sequence_name = test_dataset.get_sample_name(sample_identifier=int(batch[SAMPLE_IDENTIFIERS_KEY][
                sample_index]))
            for class_index in wrong_class_indices:
                class_folder = self.reserve_error_folder(accumulator=accumulator,
                                                         folder_name=self.class_names[class_index])
                if class_folder is None:
                    continue
                if sequence_image is None:
                    sequence_image = self.build_selection_image(batch=batch, sample_index=sample_index,
                                                                probabilities=probabilities,
                                                                frame_labels=frame_labels,
                                                                selected_frames=selected_frames)
                cv2.imwrite(filename=str(class_folder / f"{sequence_name}.png"), img=sequence_image)

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws a fixed test sequence for the sample images of the evaluation, with the selection of the decoding of the
        run: the grid of its frames with the selection borders, the probability heat map and the number of right
        selections, in green when they all are right and in red otherwise.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The frame selector outputs.
            sample_index (int): The index of the sequence in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sequence.
        """
        probabilities, frame_labels = self.get_sequence_predictions(batch=batch, predictions=predictions,
                                                                    sample_index=sample_index)
        return self.build_selection_image(batch=batch, sample_index=sample_index, probabilities=probabilities,
                                          frame_labels=frame_labels,
                                          selected_frames=self.decode_sequence(probabilities=probabilities,
                                                                               decoding=self.decoding))

    def describe_evaluated_samples(self, accumulator: Dict[str, Any]) -> str:
        """
        Describes the evaluated test samples in the header of the report: the sequences and their frames.

        Args:
            accumulator (Dict[str, Any]): The accumulated probabilities and true classes of the test sequences.

        Returns:
            str: The description of the evaluated samples.
        """
        number_frames = sum(len(frame_labels) for frame_labels in accumulator["frame_labels"])
        return f"{len(accumulator['frame_labels'])} Test Sequences, {number_frames} Frames"

    def build_selection_table(self, class_rows: List[Dict[str, Any]]) -> EvaluationTable:
        """
        Builds the per-class table of the report, shown with the per-class report and sorted by ascending selection
        accuracy of the decoding of the run so that the worst classes come first: the true frames of every wanted
        class, the selection accuracy of both decodings, the top-k accuracy and mean rank of the true frames, and the
        confidence threshold with the share of the true frames it keeps.

        Args:
            class_rows (List[Dict[str, Any]]): The metrics of every wanted class, as saved in class_metrics.csv.

        Returns:
            EvaluationTable: The per-class table.
        """
        sort_column = f"{self.decoding.capitalize()}\nAccuracy"
        rows = [[class_row["class"], class_row["true_frames"], class_row["hungarian_accuracy"],
                 class_row["greedy_accuracy"], class_row["top_k_accuracy"], class_row["mean_rank"],
                 class_row["threshold_at_target_precision"], class_row["recall_at_target_precision"]]
                for class_row in class_rows]
        return {"title": SELECTION_TABLE_TITLE,
                "columns": ["Class", self.CLASS_SUPPORT_COLUMN, "Hungarian\nAccuracy", "Greedy\nAccuracy",
                            f"Top-{self.top_k}\nAccuracy", "Mean\nRank", f"Threshold\n@ P{self.target_precision:g}",
                            f"Recall\n@ P{self.target_precision:g}"],
                "rows": rows, "sort_column": sort_column, "per_class": True,
                "legend": f"Worst {self.decoding} accuracy first, ties in class order. Accuracy : share of the true "
                          f"frames of the class chosen for it. Rank : mean rank of the true frames among the frames "
                          f"ordered by their probability for the class, 1 being the best. Threshold : probability "
                          f"from which the {self.decoding} choices reach the precision {self.target_precision:g}, "
                          f"with the share of the true frames kept. — : never reached."}

    def write_selections_file(self, accumulator: Dict[str, Any], selections: Dict[str, List[np.ndarray]],
                              frame_ranks: List[np.ndarray], class_thresholds: List[Optional[float]],
                              evaluation_folder: Path) -> Path:
        """
        Saves the selections of every evaluated test sequence to 'selections.csv' in the evaluation folder of the
        checkpoint, one row per sequence and wanted class, so that the choices of both decodings can be studied
        without running the model: the true frames with their probabilities and ranks, the frames chosen by each
        decoding with their probabilities and the number of right ones, and the low-confidence flags of the decoding
        of the run. The frames are named by their file, several frames being separated by ';'.

        Args:
            accumulator (Dict[str, Any]): The probabilities, true classes and identifiers of the test sequences.
            selections (Dict[str, List[np.ndarray]]): The [W, N] chosen frames of every sequence, keyed by decoding.
            frame_ranks (List[np.ndarray]): The [T, W] rank of every frame of every sequence for every wanted class.
            class_thresholds (List[Optional[float]]): The confidence threshold of every wanted class, None when the
                target precision was never reached.
            evaluation_folder (Path): The evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/'.

        Returns:
            Path: The written selections file.
        """
        test_dataset = self.test_dataloader.dataset
        rows = []
        for sequence_index, sequence_identifier in enumerate(accumulator["sequence_identifiers"]):
            probabilities = accumulator["probabilities"][sequence_index]
            frame_labels = accumulator["frame_labels"][sequence_index]
            frame_names = test_dataset.get_sequence_frame_names(sample_identifier=sequence_identifier)
            sequence_name = test_dataset.describe_sample(sample_identifier=sequence_identifier)["sequence"]

            # Low-confidence flags of the frames chosen by the decoding of the run
            run_selection = selections[self.decoding][sequence_index]
            low_confidence = flag_low_confidence(
                selected_frames=run_selection, class_thresholds=class_thresholds,
                selected_probabilities=gather_frame_probabilities(probabilities=probabilities, frames=run_selection,
                                                                  wanted_class_indices=self.wanted_class_indices))

            for wanted_index, class_index in enumerate(self.wanted_class_indices):
                # True frames of the class, with their probabilities and ranks
                true_frames = np.flatnonzero(frame_labels == class_index)
                row = {"sequence": sequence_name, "class": self.class_names[class_index],
                       "true_frames": ";".join(frame_names[frame_index] for frame_index in true_frames),
                       "true_probabilities": ";".join(f"{probabilities[frame_index, class_index]:.6f}"
                                                      for frame_index in true_frames),
                       "true_ranks": ";".join(str(frame_ranks[sequence_index][frame_index, wanted_index])
                                              for frame_index in true_frames)}

                # Frames chosen by every decoding, with their probabilities and the number of right ones
                for decoding in VALID_DECODINGS:
                    chosen_row = selections[decoding][sequence_index][wanted_index]
                    chosen_frames = chosen_row[chosen_row != NO_FRAME]
                    row[f"{decoding}_frames"] = ";".join(frame_names[frame_index] for frame_index in chosen_frames)
                    row[f"{decoding}_probabilities"] = ";".join(f"{probabilities[frame_index, class_index]:.6f}"
                                                                for frame_index in chosen_frames)
                    row[f"{decoding}_right_selections"] = len(set(chosen_frames.tolist()) & set(true_frames.tolist()))
                row[f"{self.decoding}_low_confidence"] = ";".join(
                    str(int(flag)) for flag in low_confidence[wanted_index][run_selection[wanted_index] != NO_FRAME])
                rows.append(row)

        selections_path = evaluation_folder / "selections.csv"
        with open(file=selections_path, mode="w", newline="") as selections_file:
            writer = csv.DictWriter(f=selections_file, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        return selections_path

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Decodes the test sequences with both decodings and organises the metrics for the report: for every decoding
        its selection accuracy, its share of fully right sequences and its unselected precision and recall; before
        decoding the top-k selection accuracy and the mean rank of the true frames; and the per-frame accuracy,
        balanced accuracy and macro F1 of the most probable class of every frame. The confidence thresholds of the
        decoding of the run, the metrics of every wanted class, the decoded confusion matrix table and, when asked,
        the selections of every sequence are saved in the evaluation folder of the checkpoint whatever the per-class
        report; the confusion matrix image only with it, also logged to TensorBoard.

        Args:
            accumulator (Dict[str, Any]): The probabilities, true classes and identifiers of the test sequences.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        probabilities, frame_labels = accumulator["probabilities"], accumulator["frame_labels"]
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        evaluation_folder.mkdir(parents=True, exist_ok=True)
        number_classes = len(self.class_names)
        files, images, headline_metrics = {}, {}, {}

        # Both decodings of every sequence, scored on the same probabilities
        selections, decoded_frame_labels, selection_metrics = {}, {}, {}
        for decoding in VALID_DECODINGS:
            selections[decoding] = [self.decode_sequence(probabilities=sequence_probabilities, decoding=decoding)
                                    for sequence_probabilities in probabilities]
            decoded_frame_labels[decoding] = [
                convert_selection_to_frame_labels(selected_frames=sequence_selection,
                                                  wanted_class_indices=self.wanted_class_indices,
                                                  number_frames=len(sequence_labels),
                                                  unselected_class_index=self.unselected_class_index)
                for sequence_selection, sequence_labels in zip(selections[decoding], frame_labels)]
            selection_metrics[decoding] = compute_selection_metrics(
                selected_frames=selections[decoding], frame_labels=frame_labels,
                decoded_frame_labels=decoded_frame_labels[decoding], wanted_class_indices=self.wanted_class_indices,
                unselected_class_index=self.unselected_class_index)
            decoding_name = decoding.capitalize()
            headline_metrics.update({
                f"{decoding_name} Selection Accuracy": selection_metrics[decoding]["selection_accuracy"],
                f"{decoding_name} Fully Right Sequences": selection_metrics[decoding]["fully_right_sequences"],
                f"{decoding_name} Unselected Precision": selection_metrics[decoding]["unselected_precision"],
                f"{decoding_name} Unselected Recall": selection_metrics[decoding]["unselected_recall"]})

        # Ranks of the true frames before decoding, then the metrics of the most probable class of every frame
        ranking_metrics = compute_ranking_metrics(probabilities=probabilities, frame_labels=frame_labels,
                                                  wanted_class_indices=self.wanted_class_indices, top_k=self.top_k)
        all_labels = torch.from_numpy(np.concatenate(frame_labels))
        frame_metrics = compute_metrics_from_confusion_matrix(confusion_matrix=compute_confusion_matrix(
            predicted_labels=torch.from_numpy(np.concatenate(probabilities)).argmax(dim=1), labels=all_labels,
            number_classes=number_classes))
        headline_metrics.update({f"Top-{self.top_k} Selection Accuracy": ranking_metrics["top_k_accuracy"],
                                 "Mean True Frame Rank": ranking_metrics["mean_rank"],
                                 "Frame Accuracy": frame_metrics["accuracy"],
                                 "Frame Balanced Accuracy": frame_metrics["balanced_accuracy"],
                                 "Frame Macro F1": frame_metrics["macro_f1"]})

        # Confidence threshold of every wanted class for the decoding of the run, saved for the deployment
        class_thresholds = find_selection_thresholds(probabilities=probabilities, frame_labels=frame_labels,
                                                     selected_frames=selections[self.decoding],
                                                     wanted_class_indices=self.wanted_class_indices,
                                                     target_precision=self.target_precision)
        thresholds_path = evaluation_folder / "class_thresholds.json"
        thresholds_path.write_text(data=json.dumps(obj={
            "target_precision": self.target_precision, "decoding": self.decoding,
            "unselected_class": self.class_names[self.unselected_class_index],
            "classes": {self.class_names[class_index]: None if class_threshold is None else
                        {"threshold": class_threshold[0], "recall": class_threshold[1]}
                        for class_index, class_threshold in zip(self.wanted_class_indices, class_thresholds)}},
            indent=4))
        files["Class Thresholds"] = thresholds_path

        # Metrics of every wanted class, saved whatever the per-class report
        class_rows = []
        for wanted_index, class_index in enumerate(self.wanted_class_indices):
            class_threshold = class_thresholds[wanted_index]
            class_rows.append({
                "class_index": class_index, "class": self.class_names[class_index],
                "true_frames": selection_metrics[self.decoding]["class_true_frames"][wanted_index],
                "hungarian_accuracy": selection_metrics["hungarian"]["class_accuracies"][wanted_index],
                "greedy_accuracy": selection_metrics["greedy"]["class_accuracies"][wanted_index],
                "top_k_accuracy": ranking_metrics["class_top_k_accuracies"][wanted_index],
                "mean_rank": ranking_metrics["class_mean_ranks"][wanted_index],
                "threshold_at_target_precision": None if class_threshold is None else class_threshold[0],
                "recall_at_target_precision": None if class_threshold is None else class_threshold[1]})
        files["Class Metrics"] = write_class_metrics_file(class_rows=class_rows, evaluation_folder=evaluation_folder)

        # Confusion matrix of the decoded classes of the frames, the image only with the per-class report
        decoded_confusion_matrix = compute_confusion_matrix(
            predicted_labels=torch.from_numpy(np.concatenate(decoded_frame_labels[self.decoding])), labels=all_labels,
            number_classes=number_classes)
        files["Confusion Matrix Table"] = save_confusion_matrix_table(confusion_matrix=decoded_confusion_matrix,
                                                                      class_names=self.class_names,
                                                                      output_folder=evaluation_folder)
        if self.per_class_report:
            files["Confusion Matrix Image"], images["Confusion Matrix"] = save_confusion_matrix_image(
                confusion_matrix=decoded_confusion_matrix, class_names=self.class_names,
                output_folder=evaluation_folder,
                title=f"Decoded Frames · Iteration {iteration} · {self.decoding.capitalize()} Selection Accuracy "
                      f"{selection_metrics[self.decoding]['selection_accuracy']:.4f}")

        # Selections of every evaluated sequence, when asked
        if self.save_predictions:
            files["Selections"] = self.write_selections_file(
                accumulator=accumulator, selections=selections, frame_ranks=ranking_metrics["frame_ranks"],
                class_thresholds=[None if class_threshold is None else class_threshold[0]
                                  for class_threshold in class_thresholds], evaluation_folder=evaluation_folder)

        # Decoding of the images and thresholds, and number of error images with their folder when some were saved
        notes = [f"Images, Thresholds And Confusion Matrix : {self.decoding} decoding ('Selection.decoding')"]
        if self.error_dumps_configuration["activated"]:
            error_counts = accumulator["error_counts"]
            notes.append(f"Error Images : {sum(error_counts.values())} in {len(error_counts)} classes, at most "
                         f"{self.error_dumps_configuration['maximum_per_class_pair']} per class")
            if error_counts:
                files["Error Images"] = accumulator["errors_folder"]

        # Selection accuracy of every wanted class for both decodings, and mean rank of its true frames
        class_scalars = {}
        for class_row in class_rows:
            for scalar_name, column_name in [("Hungarian Selection Accuracy", "hungarian_accuracy"),
                                             ("Greedy Selection Accuracy", "greedy_accuracy"),
                                             ("Mean True Frame Rank", "mean_rank")]:
                if class_row[column_name] is not None:
                    class_scalars[f"{scalar_name}/{class_row['class']}"] = class_row[column_name]

        return {"evaluated_samples": self.describe_evaluated_samples(accumulator=accumulator),
                "metrics": headline_metrics, "notes": notes,
                "tables": [self.build_selection_table(class_rows=class_rows)], "scalars": dict(headline_metrics),
                "class_scalars": class_scalars, "images": images, "files": files,
                "headline_metric": f"{self.decoding.capitalize()} Selection Accuracy"}

    def get_frame_captions(self, label: torch.Tensor, sequence_mask: torch.Tensor) -> Optional[List[str]]:
        """
        Gives the captions written on the frames of a sequence in the input viewer: the wanted class of every frame to
        select, nothing on the unselected frames.

        Args:
            label (torch.Tensor): The [T] class index of every frame of the sequence.
            sequence_mask (torch.Tensor): The [T] booleans, True for the real frames.

        Returns:
            Optional[List[str]]: The caption of every real frame.
        """
        return [self.class_names[int(frame_label)] if int(frame_label) != self.unselected_class_index else ""
                for frame_label in label[sequence_mask]]

    def get_sample_caption(self, label: torch.Tensor) -> str:
        """
        Gives the caption of a sequence in the input viewer: the frames of every wanted class, e.g. 'Selected : A 3,
        B 0, C 5'.

        Args:
            label (torch.Tensor): The [T] class index of every frame of the sequence.

        Returns:
            str: The caption of the sequence.
        """
        frame_labels = label.tolist()
        class_frames = [f"{self.class_names[class_index]} " +
                        ",".join(str(frame_index) for frame_index, frame_label in enumerate(frame_labels)
                                 if frame_label == class_index) for class_index in self.wanted_class_indices]
        return "Selected : " + ", ".join(class_frames)
