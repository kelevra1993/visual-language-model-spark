"""
Trainer of the object detection task: the COCO-format loaders, the two-stage detector with its four
losses, the geometric augmentations moving the bounding boxes, the anchor recall of the training objects printed at
every launch (and of every split by the 'compute_anchor_coverage' diagnosis mode, which then exits), and the evaluation
of the checkpoints on the test split with the mean average precision and the mean
average recall (all the objects and every object size) and the average precision of every class, the proposal recall,
the score threshold of every class reaching the target precision, the confusion matrix and the error images of the
detections the deployed detector returns, and the sample images of fixed test images with their true and detected
boxes. Registered for the object detection task in the task registry, and extended by the instance segmentation
trainer.
"""

import json
import cv2
import numpy as np
import torch

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from vision_training.core.contracts import SAMPLE_IDENTIFIERS_KEY, DetectionBatch, EvaluationResults
from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.data.loaders import get_detection_dataloaders
from vision_training.evaluation.classification.confusion_matrix import (save_confusion_matrix_image,
                                                                        save_confusion_matrix_table)
from vision_training.evaluation.detection.average_precision_report import summarise_average_precisions
from vision_training.evaluation.detection.detection_confusion import (BACKGROUND_CLASS_INDEX,
                                                                      MATCHING_INTERSECTION_OVER_UNION_THRESHOLD,
                                                                      add_to_confusion_matrix, compute_overlap_matrix,
                                                                      group_image_errors, match_image_detections,
                                                                      summarise_confusion_matrix)
from vision_training.evaluation.detection.object_sizes import compute_box_areas
from vision_training.evaluation.detection.region_recall import (
    accumulate_proposal_recall, compute_best_anchor_overlaps, create_proposal_recall_accumulator, gather_object_boxes,
    summarise_anchor_recall, summarise_proposal_recall)
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.evaluation.report_formatting import build_table_renderables, print_report
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.trainers.coco_split_trainer import CocoSplitTrainer
from vision_training.utilities.os_utilities import print_color
from vision_training.utilities.visualization import (build_detection_sample_grid, convert_image_to_array,
                                                     view_input_data)

# Files of the recalls of the anchors, in the run folder, and of the proposals, in the evaluation folder of every
# checkpoint
ANCHOR_RECALL_FILE_NAME = "anchor_recall.csv"
PROPOSAL_RECALL_FILE_NAME = "proposal_recall.csv"


class DetectionTrainer(CocoSplitTrainer):
    """
    Trainer of the object detection task: the two-stage detector trained on COCO-format datasets, with its four
    losses tracked, bounding boxes moved by the geometric augmentations, and every evaluated checkpoint scored with
    the mean average precision on the test split.
    """

    TASK_NAME = "object_detection"

    # Whether the loaders give the binary mask of every object, for the instance segmentation subclass
    LOAD_INSTANCE_MASKS = False

    # Whether the test loader also gives the objects at the original resolution of the images, set by the instance
    # segmentation subclass from 'evaluation.original_resolution' to score its boxes and masks at that resolution
    original_resolution = False

    # Overlap matching the detections to the objects for the class thresholds, the confusion matrix and the error
    # images, the one of the headline metric: the boxes, or the masks for the instance segmentation subclass
    MATCHING_OVERLAP_TYPE = "boxes"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the detection evaluation settings (score thresholds, target precision of the class thresholds, error
        images), then prepares the training run through the general trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, DetectionLevels,
                RegionProposal, Detector).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Score threshold of the detections kept for the mAP, lower than the deployment one so that the
        # precision-recall curves keep their low-confidence end
        self.evaluation_score_threshold = experiment_configuration["evaluation"]["score_threshold"]

        # Score threshold of the deployment, keeping the detections of the sample images, the confusion matrix and the
        # error images, so that they show what the deployed detector returns
        self.deployment_score_threshold = model_configuration["Detector"]["inference"]["score_threshold"]

        # Precision targeted by the score threshold of every class, and the error images of the evaluation
        self.target_precision = experiment_configuration["evaluation"]["target_precision"]
        self.error_dumps_configuration = experiment_configuration["evaluation"]["error_dumps"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the COCO detection loaders of the three splits, names the classes after the categories of the test
        split for the per-class report, with a colour per class for the sample images, and keeps the file name of
        every test image, which names the sample images. The test loader also gives the objects at the original
        resolution of the images when the evaluation is made at that resolution (instance segmentation only).

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders.
        """
        # Class names, class colours and test image names from the test manifest, then the loaders of the three splits
        self.read_test_manifest()
        return get_detection_dataloaders(experiment_configuration=self.experiment_configuration,
                                         model_configuration=self.model_configuration, batch_size=self.batch_size,
                                         load_instance_masks=self.LOAD_INSTANCE_MASKS,
                                         load_original_instances=self.original_resolution)

    def analyse_training_data(self) -> Dict[str, Path]:
        """
        Measures at every launch how well the anchors of the detector fit the objects of the training split, before
        training on them: the share of the objects with an anchor overlapping them by at least the foreground threshold
        of the region proposal network, in every level and in all of them, for all the objects and every object size.
        Printed in the console and saved to anchor_recall.csv in the run folder, the objects being read from the
        annotations without loading any image.

        Returns:
            Dict[str, Path]: The 'Anchor Recall' file.
        """
        # Recall at the foreground threshold of the region proposal network, its rows saved in the run folder
        anchor_recall = self.measure_anchor_recall(
            dataset=self.train_dataloader.dataset, split_name="Training",
            intersection_over_union_threshold=self.model_configuration["RegionProposal"]["foreground_iou_threshold"][
                "min"], threshold_setting="RegionProposal.foreground_iou_threshold.min", show_progress=False)
        return {"Anchor Recall": write_class_metrics_file(class_rows=anchor_recall["rows"],
                                                          evaluation_folder=self.experiment_folder,
                                                          file_name=ANCHOR_RECALL_FILE_NAME)}

    def uses_exiting_diagnosis_mode(self) -> bool:
        """
        Tells whether a diagnosis mode ends the program before the training: the input viewer of every task, or the
        anchor coverage of the splits ('diagnosis_modes.compute_anchor_coverage'), which checks the anchor setting
        without creating the run folder.

        Returns:
            bool: Whether the program exits after the diagnosis.
        """
        return super().uses_exiting_diagnosis_mode() or self.diagnosis_modes["compute_anchor_coverage"]["activated"]

    def diagnose_data_splits(self) -> None:
        """
        Prints the anchor coverage of the training, validation and test splits when
        'diagnosis_modes.compute_anchor_coverage' is activated: for every split, the share of its objects with an anchor
        overlapping them by at least the 'iou_threshold' of the mode, in every level and in all of them, for all the
        objects and every object size, so that the anchor setting can be checked against every split before a launch.
        Nothing is written; the trainer exits afterwards.
        """
        anchor_coverage_configuration = self.diagnosis_modes["compute_anchor_coverage"]
        if not anchor_coverage_configuration["activated"]:
            return None

        # One table per split, each read with a progress bar over its images
        for split_name, data_loader in [("Training", self.train_dataloader), ("Validation", self.validation_dataloader),
                                        ("Test", self.test_dataloader)]:
            self.measure_anchor_recall(
                dataset=data_loader.dataset, split_name=split_name,
                intersection_over_union_threshold=anchor_coverage_configuration["iou_threshold"],
                threshold_setting="ExperimentConfiguration.diagnosis_modes.compute_anchor_coverage.iou_threshold",
                show_progress=True)
        return None

    def measure_anchor_recall(self, dataset: CocoDetectionDataset, split_name: str,
                              intersection_over_union_threshold: float, threshold_setting: str,
                              show_progress: bool) -> Dict[str, Any]:
        """
        Measures and prints the anchor recall of one split, shared by the check of the training split at every launch
        (analyse_training_data) and the anchor coverage diagnosis mode (diagnose_data_splits): the boxes of the objects
        read from the annotations without loading any image, the best anchor of every object in every level found from
        the anchor grid of the level, then the recall of every level and of all of them at the threshold.

        Args:
            dataset (CocoDetectionDataset): The dataset of the split.
            split_name (str): The name of the split in the console, e.g. 'Training'.
            intersection_over_union_threshold (float): The minimum overlap for an object to be covered.
            threshold_setting (str): The configuration setting giving the threshold, named in the legend of the table.
            show_progress (bool): Whether a progress bar follows the images of the split.

        Returns:
            Dict[str, Any]: The 'rows' and 'table' of the anchor recall (summarise_anchor_recall).
        """
        # Best anchor of every object of the split in every level
        object_boxes = gather_object_boxes(dataset=dataset, progress_description=(
            f"Gathering The {split_name} Objects" if show_progress else None))
        best_anchor_overlaps = compute_best_anchor_overlaps(
            object_boxes=object_boxes, level_anchor_generators=self.model.get_level_anchor_generators())
        anchor_recall = summarise_anchor_recall(
            best_anchor_overlaps=best_anchor_overlaps, object_areas=compute_box_areas(boxes=object_boxes),
            intersection_over_union_threshold=intersection_over_union_threshold, split_name=split_name,
            threshold_setting=threshold_setting)

        # Table printed with the layout of the report
        print_color(color="blue", output=f"Anchor Recall Of The {object_boxes.shape[0]} {split_name} Objects...",
                    add_separators=True)
        print_report(renderables=build_table_renderables(table=anchor_recall["table"]))
        return anchor_recall

    def build_model(self) -> TwoStageDetector:
        """
        Builds the two-stage detector from the model configuration.

        Returns:
            TwoStageDetector: The untrained detector.
        """
        return TwoStageDetector(configuration=self.model_configuration, device=self.device, dtype=self.dtype)

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the total loss and the four losses of the detector, tracked on every training and validation batch.

        Returns:
            Dict[str, str]: The display name of every tracked loss.
        """
        return {"total_loss": "Total Loss",
                "region_proposal_classification_loss": "RPN Classification Loss",
                "region_proposal_localisation_loss": "RPN Regression Loss",
                "detection_classification_loss": "Detection Classification Loss",
                "detection_regression_loss": "Detection Regression Loss"}

    def get_loss_weights(self) -> Optional[Dict[str, float]]:
        """
        Gives the weight of every loss of the detector in the total loss, 'Optimization.loss_weights': the four losses
        of the region proposal and detection heads, and the mask loss of the instance segmentation, which inherits it.

        Returns:
            Optional[Dict[str, float]]: The weight of every loss output of the model.
        """
        return self.optimization_configuration["loss_weights"]

    def prepare_batch(self, batch: Dict[str, Any]) -> DetectionBatch:
        """
        Moves the images and the per-image ground truth boxes and labels of a detection batch to the device, in the
        precision of the model (labels as int64 class indices).

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a COCO loader.

        Returns:
            DetectionBatch: The batch ready for the detector.
        """
        return {"images": batch["images"].to(device=self.device, dtype=self.dtype),
                "bounding_boxes": [box_tensor.to(device=self.device, dtype=self.dtype)
                                   for box_tensor in batch["bounding_boxes"]],
                "labels": [label_tensor.to(device=self.device, dtype=torch.int64) for label_tensor in batch["labels"]],
                "sample_identifiers": batch["sample_identifiers"]}

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the training images, moving their boxes with the geometric augmentations and dropping the
        boxes (with their labels) that leave the image.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The batch with its augmented images, boxes and labels, and the 'augmentation_reports' of
                its samples for the input viewer.
        """
        images, bounding_boxes, labels, _, _, _, augmentation_reports = self.data_augmenter.augment_batch(
            images=batch["images"], bounding_boxes=batch["bounding_boxes"], labels=batch["labels"])
        return {**batch, "images": images, "bounding_boxes": bounding_boxes, "labels": labels,
                "augmentation_reports": augmentation_reports}

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the detector with the ground truth of the batch, so that it samples its training targets and returns its
        four losses.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The detector outputs, with its four losses.
        """
        return self.model(input_tensor=batch["images"], ground_truth_bounding_boxes=batch["bounding_boxes"],
                          ground_truth_labels=batch["labels"])

    @contextmanager
    def evaluation_context(self) -> Iterator[None]:
        """
        Lowers the detection score threshold of the model to the evaluation one during the evaluation of the test
        split, so that the precision-recall curves keep their low-confidence end, and restores it afterwards, even
        when the evaluation fails.

        Yields:
            None: Control to the evaluation loop.
        """
        model_score_threshold = self.model.detection_score_threshold
        self.model.detection_score_threshold = self.evaluation_score_threshold
        try:
            yield
        finally:
            self.model.detection_score_threshold = model_score_threshold

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the detector without ground truth, as in deployment: given ground truth, the detector would only classify
        the proposals sampled around it for training, which would inflate the mAP.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The detector outputs, with the final detections of every image.
        """
        return self.model(input_tensor=batch["images"])

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the lists of detections and ground truths filled during the evaluation of the test split, with the
        confusion matrix of the detections the deployed detector returns, the folder and the counts of the error
        images, and the overlaps of the objects with the proposals for the proposal recall.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, which names its evaluation folder.

        Returns:
            Dict[str, Any]: The empty 'detections' and 'ground_truths' lists, the [K, K] zero 'confusion_matrix'
                (background included), the 'errors_folder' of the error images and their empty 'error_counts' per
                folder, and the empty 'proposal_recall' accumulator of the detection levels.
        """
        number_classes = len(self.class_names)
        return {"detections": [], "ground_truths": [],
                "confusion_matrix": torch.zeros(size=(number_classes, number_classes), dtype=torch.int64),
                "errors_folder": self.get_evaluation_folder(iteration=iteration) / "Errors", "error_counts": {},
                "proposal_recall": create_proposal_recall_accumulator(levels=list(self.model.get_level_anchors()))}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the valid detections and the ground truth of every kept image of a test batch on the CPU, both expressed
        in network input pixels (in original image pixels for the instance segmentation evaluated at the original
        resolution), adds the overlaps of its objects with the proposals to the proposal recall, matches the
        detections the deployed detector returns (score reaching 'Detector.inference.score_threshold') to the objects
        of the image at an overlap of 0.5 for the confusion matrix, and saves the errors of the image as images when the
        error images are on, while the image is still in the batch.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, its lists extended and its confusion matrix
                filled in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The detector outputs, with the filtered proposals of every image.
            sample_indices (List[int]): The indices of the images to keep.
        """
        for image_index in sample_indices:
            detection = self.gather_image_detections(batch=batch, predictions=predictions, image_index=image_index)
            ground_truth = self.gather_image_ground_truth(batch=batch, image_index=image_index)
            accumulator["detections"].append(detection)
            accumulator["ground_truths"].append(ground_truth)

            # Best overlap of every object with the proposals given to the detection heads, on the device of the model
            accumulate_proposal_recall(proposal_accumulator=accumulator["proposal_recall"],
                                       proposal_boxes=predictions["proposal_boxes"][image_index],
                                       proposal_levels=predictions["proposal_levels"][image_index],
                                       valid_proposal_mask=predictions["valid_proposal_mask"][image_index],
                                       ground_truth_boxes=batch["bounding_boxes"][image_index])

            # Detections of the deployed detector matched to the objects of the image, counted in the confusion matrix
            kept_detections = detection["scores"] >= self.deployment_score_threshold
            detection_scores = detection["scores"][kept_detections]
            detection_labels = detection["labels"][kept_detections]
            overlaps = compute_overlap_matrix(detection=detection, ground_truth=ground_truth,
                                              overlap_type=self.MATCHING_OVERLAP_TYPE)[kept_detections]
            matched_ground_truth_indices = match_image_detections(
                overlaps=overlaps, detection_scores=detection_scores, detection_labels=detection_labels,
                ground_truth_labels=ground_truth["labels"],
                intersection_over_union_threshold=MATCHING_INTERSECTION_OVER_UNION_THRESHOLD)
            add_to_confusion_matrix(confusion_matrix=accumulator["confusion_matrix"],
                                    matched_ground_truth_indices=matched_ground_truth_indices,
                                    detection_labels=detection_labels, ground_truth_labels=ground_truth["labels"])

            # Errors of the image saved as images while the image is still in the batch
            if self.error_dumps_configuration["activated"]:
                self.dump_errors(accumulator=accumulator, batch=batch, predictions=predictions, image_index=image_index,
                                 matched_ground_truth_indices=matched_ground_truth_indices,
                                 detection_scores=detection_scores, detection_labels=detection_labels,
                                 ground_truth_labels=ground_truth["labels"])

    def gather_image_detections(self, batch: Dict[str, Any], predictions: Dict[str, Any],
                                image_index: int) -> Dict[str, torch.Tensor]:
        """
        Gives the valid detections of one test image on the CPU, without the padding of the batch, for the mean
        average precision and the matching of the confusion matrix. The instance segmentation trainer adds the mask
        overlaps of the detections with the objects of the image.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The detector outputs.
            image_index (int): The index of the image in the batch.

        Returns:
            Dict[str, torch.Tensor]: The detection 'boxes' [D, 4], 'scores' [D] and 'labels' [D] of the image.
        """
        number_detections = int(predictions["number_detections"][image_index])
        return {"boxes": predictions["final_detection_boxes"][image_index, :number_detections].cpu(),
                "scores": predictions["final_detection_scores"][image_index, :number_detections].cpu(),
                "labels": predictions["final_detection_labels"][image_index, :number_detections].cpu()}

    def gather_image_ground_truth(self, batch: Dict[str, Any], image_index: int) -> Dict[str, torch.Tensor]:
        """
        Gives the objects of one test image on the CPU, for the mean average precision (by box area for the object
        sizes) and the matching of the confusion matrix. The instance segmentation trainer adds the areas of their
        masks, which size the objects of the mask average precision.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            image_index (int): The index of the image in the batch.

        Returns:
            Dict[str, torch.Tensor]: The ground truth 'boxes' [G, 4] and 'labels' [G] of the image.
        """
        return {"boxes": batch["bounding_boxes"][image_index].cpu(), "labels": batch["labels"][image_index].cpu()}

    def dump_errors(self, accumulator: Dict[str, Any], batch: Dict[str, Any], predictions: Dict[str, Any],
                    image_index: int, matched_ground_truth_indices: torch.Tensor, detection_scores: torch.Tensor,
                    detection_labels: torch.Tensor, ground_truth_labels: torch.Tensor) -> None:
        """
        Saves the errors of a test image as images, to look at the errors of the checkpoint: the image is saved once in
        the folder of every kind of error it holds, 'Errors/FALSE-POSITIVE/<class>', 'Errors/MISSED/<class>' or
        'Errors/<TRUE>-TO-<PREDICTED>' in its evaluation folder, drawn like the sample images with the boxes of these
        errors outlined with red dashes and the errors described in the caption. It is named after the image and, but
        for the missed objects, the highest score of its detections in error. At most 'maximum_per_class_pair' images
        are saved per folder, the first ones met.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, holding the 'errors_folder' and 'error_counts'.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The detector outputs.
            image_index (int): The index of the image in the batch.
            matched_ground_truth_indices (torch.Tensor): The [D] object found by every detection of the deployed
                detector, -1 for none (match_image_detections).
            detection_scores (torch.Tensor): The [D] scores of the detections of the deployed detector.
            detection_labels (torch.Tensor): The [D] class indices of the detections of the deployed detector.
            ground_truth_labels (torch.Tensor): The [G] class indices of the objects of the image.
        """
        image_errors = group_image_errors(matched_ground_truth_indices=matched_ground_truth_indices,
                                          detection_scores=detection_scores, detection_labels=detection_labels,
                                          ground_truth_labels=ground_truth_labels, class_names=self.class_names)
        image_name = self.get_test_sample_name(sample_identifier=int(batch[SAMPLE_IDENTIFIERS_KEY][image_index]))
        for folder_name, image_error in image_errors.items():
            # The folder of the error, unless it is full
            error_folder = self.reserve_error_folder(accumulator=accumulator, folder_name=folder_name)
            if error_folder is None:
                continue

            # Boxes in error outlined on the image drawn like a sample image, named after the image and its score
            highlighted_ground_truths = np.zeros(shape=(len(ground_truth_labels),), dtype=bool)
            highlighted_ground_truths[image_error["ground_truth_indices"]] = True
            highlighted_detections = np.zeros(shape=(len(detection_labels),), dtype=bool)
            highlighted_detections[image_error["detection_indices"]] = True
            error_image = self.draw_test_image(batch=batch, predictions=predictions, sample_index=image_index,
                                               error_description=image_error["description"],
                                               highlighted_ground_truths=highlighted_ground_truths,
                                               highlighted_detections=highlighted_detections)
            file_name = image_name if image_error["score"] is None else f"{image_name}_{image_error['score']:.3f}"
            cv2.imwrite(filename=str(error_folder / f"{file_name}.png"), img=error_image)

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the mean average precision of the test split with the three average precision methods (the area
        method being the headline one, its mAP@0.5 comparing the checkpoints), the mean average precision of every
        object size, the mean average recall of all the objects and of every size, the average precisions of every
        class, the score threshold of every class reaching the target precision and the proposal recall, and organises
        them for the report: the tables of the methods, of the object sizes, of the mean average recall and of the
        proposal recall, always shown, and the per-class table (shown with the per-class report, worst
        AP@0.5 first), with the same values as TensorBoard scalars (the AP@0.5 of every class only with the per-class
        report). The average precisions and thresholds of every class are saved to class_metrics.csv, the proposal
        recall to proposal_recall.csv, the thresholds to class_thresholds.json and the confusion matrix to
        confusion_matrix.csv in the evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/', whatever
        the per-class report (save_error_analysis).

        Args:
            accumulator (Dict[str, Any]): The detections and ground truths of the test split, with the confusion matrix,
                the error images and the proposal recall.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        summary = summarise_average_precisions(
            detections=accumulator["detections"], ground_truths=accumulator["ground_truths"],
            number_classes=len(set(self.label_dictionary.values())), class_names=self.class_names,
            target_precision=self.target_precision,
            resolution_name="original image" if self.original_resolution else "network input")
        proposal_recall = self.save_proposal_recall(accumulator=accumulator, iteration=iteration)

        # Average precisions and thresholds of every class, saved whatever the per-class report, and the proposal
        # recall
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        files = {"Class Metrics": write_class_metrics_file(class_rows=summary["class_rows"],
                                                           evaluation_folder=evaluation_folder),
                 "Proposal Recall": proposal_recall["file"]}

        # Class thresholds, confusion matrix and error images of the detections of the deployed detector
        error_analysis = self.save_error_analysis(accumulator=accumulator, iteration=iteration,
                                                  class_score_thresholds=summary["class_score_thresholds"])
        files.update(error_analysis["files"])

        # Tables always shown first, the per-class table and the per-class recall table last
        tables = [summary["tables"]["methods"], summary["tables"]["sizes"], summary["tables"]["recalls"],
                  proposal_recall["table"], summary["tables"]["classes"], summary["tables"]["class_recalls"]]
        return {"evaluated_samples": f"{len(accumulator['detections'])} Test Images", "metrics": {},
                "notes": error_analysis["notes"], "tables": tables,
                "scalars": {**summary["scalars"], **proposal_recall["scalars"]},
                "class_scalars": summary["class_scalars"], "images": error_analysis["images"], "files": files,
                "headline_metric": "mAP@0.5/area"}

    def save_proposal_recall(self, accumulator: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """
        Summarises the proposal recall of the test split for the report of an evaluated checkpoint, in the object
        detection and the instance segmentation: its table, always shown, its TensorBoard scalars, and its rows saved
        to proposal_recall.csv in the evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/', which is
        sent to MLflow.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, with the filled 'proposal_recall'.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            Dict[str, Any]: The 'table', the 'scalars' and the written 'file'.
        """
        proposal_recall = summarise_proposal_recall(proposal_accumulator=accumulator["proposal_recall"])
        proposal_recall_path = write_class_metrics_file(
            class_rows=proposal_recall["rows"], evaluation_folder=self.get_evaluation_folder(iteration=iteration),
            file_name=PROPOSAL_RECALL_FILE_NAME)
        return {"table": proposal_recall["table"], "scalars": proposal_recall["scalars"], "file": proposal_recall_path}

    def save_error_analysis(self, accumulator: Dict[str, Any], iteration: int,
                            class_score_thresholds: Dict[int, Optional[Tuple[float, float]]]) -> Dict[str, Any]:
        """
        Saves the outputs of the evaluation of a checkpoint telling where the detector fails, in its evaluation
        folder, '<run>/Evaluation/Iteration_<n>/', for compute_evaluation_results: the score threshold of every class
        reaching the target precision in class_thresholds.json (the format of the classification tasks, for the
        deployment), and the confusion matrix of the detections the deployed detector returns in confusion_matrix.csv,
        both whatever the per-class report, the confusion matrix image only with it, also logged to TensorBoard. The
        report gets a note counting the outcomes of the detections and, when the error images are on, a note counting
        them with their folder.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, with the confusion matrix and the error images.
            iteration (int): The training iteration of the evaluated checkpoint.
            class_score_thresholds (Dict[int, Optional[Tuple[float, float]]]): The threshold and recall of every
                foreground class, or None (evaluate_detections with a target precision).

        Returns:
            Dict[str, Any]: The 'files' (Class Thresholds, Confusion Matrix Table, Confusion Matrix Image, Error
                Images), the 'images' (Confusion Matrix) and the 'notes' of the report.
        """
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        evaluation_folder.mkdir(parents=True, exist_ok=True)
        files, images, notes = {}, {}, []

        # Threshold of every class reaching the target precision, saved for the deployment of the checkpoint
        thresholds_path = evaluation_folder / "class_thresholds.json"
        thresholds_path.write_text(data=json.dumps(obj={
            "target_precision": self.target_precision,
            "classes": {self.class_names[class_index]: None if class_threshold is None else
                        {"threshold": class_threshold[0], "recall": class_threshold[1]}
                        for class_index, class_threshold in class_score_thresholds.items()}}, indent=4))
        files["Class Thresholds"] = thresholds_path

        # Confusion matrix with its background row and column, the image only with the per-class report
        confusion_matrix = accumulator["confusion_matrix"]
        class_names = [self.class_names[class_index] for class_index in range(len(self.class_names))]
        files["Confusion Matrix Table"] = save_confusion_matrix_table(confusion_matrix=confusion_matrix,
                                                                      class_names=class_names,
                                                                      output_folder=evaluation_folder)
        if self.per_class_report:
            files["Confusion Matrix Image"], images["Confusion Matrix"] = save_confusion_matrix_image(
                confusion_matrix=confusion_matrix, class_names=class_names, output_folder=evaluation_folder,
                title=f"Confusion Matrix · Iteration {iteration} · Score >= {self.deployment_score_threshold:.2f} · "
                      f"IoU >= {MATCHING_INTERSECTION_OVER_UNION_THRESHOLD:g}", background_index=BACKGROUND_CLASS_INDEX)

        # Outcomes of the detections of the deployed detector
        outcomes = summarise_confusion_matrix(confusion_matrix=confusion_matrix)
        notes.append(f"Confusion Matrix : score >= {self.deployment_score_threshold:.2f}, IoU >= "
                     f"{MATCHING_INTERSECTION_OVER_UNION_THRESHOLD:g} ({self.MATCHING_OVERLAP_TYPE}), "
                     f"{outcomes['right']} right, {outcomes['wrong_class']} wrong class, {outcomes['false_positives']} "
                     f"false positives, {outcomes['missed']} missed")

        # Number of error images and their folder when some were saved
        if self.error_dumps_configuration["activated"]:
            error_counts = accumulator["error_counts"]
            notes.append(f"Error Images : {sum(error_counts.values())} in {len(error_counts)} folders, at most "
                         f"{self.error_dumps_configuration['maximum_per_class_pair']} per folder")
            if error_counts:
                files["Error Images"] = accumulator["errors_folder"]
        return {"files": files, "images": images, "notes": notes}

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws a fixed test image for the sample images of the evaluation: its ground truth boxes next to its
        detections whose score reaches the deployment score threshold ('Detector.inference.score_threshold'), the
        evaluation keeping lower scores for the mAP, under a caption counting both (draw_test_image).

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The detector outputs, with the final detections of every image.
            sample_index (int): The index of the image in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sample.
        """
        return self.draw_test_image(batch=batch, predictions=predictions, sample_index=sample_index,
                                    error_description=None, highlighted_ground_truths=None,
                                    highlighted_detections=None)

    def draw_test_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int,
                        error_description: Optional[str], highlighted_ground_truths: Optional[np.ndarray],
                        highlighted_detections: Optional[np.ndarray]) -> np.ndarray:
        """
        Draws a test image for the sample images and the error images of the evaluation: its ground truth boxes next to
        its detections whose score reaches the deployment score threshold ('Detector.inference.score_threshold'), with
        the masks of the instance segmentation painted under them, under a caption counting both. An error image
        outlines the boxes of its errors and describes them first in the caption.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The detector outputs, with the final detections of every image.
            sample_index (int): The index of the image in the batch.
            error_description (Optional[str]): The errors written first in the caption, None for a sample image.
            highlighted_ground_truths (Optional[np.ndarray]): The [G] flags of the true boxes in error, None for a
                sample image.
            highlighted_detections (Optional[np.ndarray]): The [D] flags of the drawn detections in error (those
                reaching the deployment score threshold), None for a sample image.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image.
        """
        image = convert_image_to_array(image=batch["images"][sample_index])

        # Valid detections of the image, kept from the deployment score threshold
        number_detections = int(predictions["number_detections"][sample_index])
        scores = predictions["final_detection_scores"][sample_index, :number_detections].detach().float().cpu().numpy()
        kept_detections = scores >= self.deployment_score_threshold
        predicted_boxes = predictions["final_detection_boxes"][sample_index, :number_detections].detach().float()
        predicted_labels = predictions["final_detection_labels"][sample_index, :number_detections].detach()
        ground_truth_boxes = batch["bounding_boxes"][sample_index].detach().float().cpu().numpy()
        ground_truth_masks, predicted_masks = self.get_sample_masks(batch=batch, predictions=predictions,
                                                                    sample_index=sample_index,
                                                                    kept_detections=kept_detections)

        # Caption counting the boxes, after the errors of an error image
        caption = (f"{len(ground_truth_boxes)} ground truth boxes | {int(kept_detections.sum())} detections with "
                   f"score >= {self.deployment_score_threshold:.2f}")
        if error_description is not None:
            caption = f"{error_description} | {caption}"

        return build_detection_sample_grid(
            image=image, ground_truth_boxes=ground_truth_boxes,
            ground_truth_labels=batch["labels"][sample_index].detach().cpu().numpy(),
            predicted_boxes=predicted_boxes.cpu().numpy()[kept_detections],
            predicted_labels=predicted_labels.cpu().numpy()[kept_detections], predicted_scores=scores[kept_detections],
            class_names=self.class_names, class_colours=self.class_colours, caption=caption,
            ground_truth_masks=ground_truth_masks, predicted_masks=predicted_masks,
            highlighted_ground_truths=highlighted_ground_truths, highlighted_detections=highlighted_detections)

    def get_sample_masks(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int,
                         kept_detections: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Gives the masks painted on a sample image under its boxes: none for the object detection, the instance
        segmentation trainer returning the masks of the true objects and of the kept detections.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The model outputs, with the final detections of every image.
            sample_index (int): The index of the image in the batch.
            kept_detections (np.ndarray): The [K] flags of the valid detections drawn on the image.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: The ground truth and predicted masks, None and None.
        """
        return None, None

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays an augmented training image next to a validation image, each with its ground truth boxes captioned
        with their class names and an information panel holding its number of objects per class, its augmentations
        and the objects they removed.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed images in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        # Invert the label dictionary so that each box is captioned with its class name instead of its index
        label_names = {class_index: str(class_name) for class_name, class_index in self.label_dictionary.items()}

        # The objects of every image counted per class, in the order of the label dictionary, then its augmentations
        panel_lines = []
        for batch in [training_batch, validation_batch]:
            sample_labels = batch["labels"][sample_index].tolist()
            class_counts = [f"{class_name} {sample_labels.count(class_index)}"
                            for class_index, class_name in label_names.items() if class_index in sample_labels]
            object_line = f"Objects : {len(sample_labels)} ({', '.join(class_counts)})" if sample_labels \
                else "Objects : none"
            panel_lines.append([object_line] + self.describe_input_augmentations(
                batch=batch, sample_index=sample_index, label_names=label_names))
        return view_input_data(
            images=[self.get_input_sample_image(batch=training_batch, sample_index=sample_index),
                    self.get_input_sample_image(batch=validation_batch, sample_index=sample_index)],
            bounding_boxes=[training_batch["bounding_boxes"][sample_index],
                            validation_batch["bounding_boxes"][sample_index]],
            labels=[training_batch["labels"][sample_index], validation_batch["labels"][sample_index]],
            window_titles=["Training", "Validation"], display_input_size=self.input_image_size,
            label_names=label_names, panel_lines=panel_lines)

    def get_input_sample_image(self, batch: Dict[str, Any], sample_index: int) -> torch.Tensor:
        """
        Gives the image of a batch shown by the input viewer, under its boxes: the image itself for the object
        detection, the instance segmentation trainer painting the masks of its objects on it.

        Args:
            batch (Dict[str, Any]): The training or validation batch, on the CPU.
            sample_index (int): The index of the image in the batch.

        Returns:
            torch.Tensor: The [3, H, W] image in [0, 255].
        """
        return batch["images"][sample_index]
