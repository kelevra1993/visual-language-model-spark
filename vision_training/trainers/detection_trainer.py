"""
Trainer of the object detection task: the COCO-format loaders, the two-stage detector with its four losses, the
geometric augmentations moving the bounding boxes, the anchor recall of the training and test objects measured at the
first launch of a run into anchor_recall.csv and anchor_recall.png (and of every split by the
'compute_anchor_coverage' diagnosis mode, its tables coloured, which then exits), and the evaluation of the
checkpoints on the test split with the mean average precision and the mean average recall (all the objects and every
object size) and the average precision of every class, the proposal recall, the score threshold of every class reaching
the target precision, the confusion matrix and the error images of the detections the deployed detector returns, and the
sample images of fixed test images with their true and detected boxes. Registered for the object detection task in the
task registry, and extended by the instance segmentation trainer.
"""

import csv
import json
import math
import cv2
import numpy as np
import torch

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from vision_training.core.contracts import SAMPLE_IDENTIFIERS_KEY, DetectionBatch, EvaluationResults
from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.data.loaders import get_detection_dataloaders
from vision_training.evaluation.classification.confusion_matrix import (save_confusion_matrix_image,
                                                                        save_confusion_matrix_table)
from vision_training.evaluation.detection.anchor_recall_figure import draw_anchor_recall_figure
from vision_training.evaluation.detection.average_precision_report import summarise_average_precisions
from vision_training.evaluation.detection.detection_confusion import (BACKGROUND_CLASS_INDEX,
                                                                      MATCHING_INTERSECTION_OVER_UNION_THRESHOLD,
                                                                      add_to_confusion_matrix, compute_overlap_matrix,
                                                                      group_image_errors, match_image_detections,
                                                                      summarise_confusion_matrix)
from vision_training.evaluation.detection.object_sizes import OBJECT_SIZE_NAMES, compute_box_areas
from vision_training.evaluation.detection.region_recall import (
    accumulate_proposal_recall, compute_best_anchor_overlaps, create_proposal_recall_accumulator, gather_object_boxes,
    summarise_anchor_recall, summarise_proposal_recall)
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.evaluation.report_formatting import build_table_renderables, format_report_value, print_report
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.trainers.coco_split_trainer import CocoSplitTrainer
from vision_training.trainers.metric_tracking import MetricTracker
from vision_training.utilities.os_utilities import format_path_link, print_color
from vision_training.utilities.visualization import (PanelTable, ViewerWindow, build_detection_sample_grid,
                                                     convert_image_to_array)

# Files of the recalls of the anchors and their figure, in the run folder, and of the proposals, in the evaluation
# folder of every checkpoint
ANCHOR_RECALL_FILE_NAME = "anchor_recall.csv"
ANCHOR_RECALL_FIGURE_NAME = "anchor_recall.png"
PROPOSAL_RECALL_FILE_NAME = "proposal_recall.csv"

# Splits whose anchor recall the first launch of a run measures and saves, with the setting giving their folder: the
# objects the anchors are trained on, and those the evaluation scores
ANCHOR_RECALL_SPLITS = {"Training": "train_data_folder", "Test": "test_data_folder"}


def read_anchor_recall_settings(file_path: Path) -> Optional[Tuple[Dict[str, str], float]]:
    """
    Reads what the anchor_recall.csv of an earlier launch of the run measured, so that a resumed launch keeps the file
    and its figure instead of measuring the same splits again, and measures them again when the split folders or the
    threshold changed.

    Args:
        file_path (Path): The anchor_recall.csv of the run folder.

    Returns:
        Optional[Tuple[Dict[str, str], float]]: The folder of every measured split keyed by split name, and the
            intersection over union threshold, or None without file or for a file written before the splits were
            recorded (no 'split' column).
    """
    if not file_path.exists():
        return None
    with open(file=file_path, newline="", encoding="utf-8") as anchor_recall_file:
        rows = list(csv.DictReader(anchor_recall_file))
    if not rows or "split" not in rows[0]:
        return None
    return ({row["split"]: row["split_folder"] for row in rows}, float(rows[0]["iou_threshold"]))


def describe_anchor_recall(rows: List[Dict[str, Any]], split_name: str) -> str:
    """
    Writes the console line of the anchor recall of one split at the first launch of a run, its table being drawn in
    anchor_recall.png instead: the objects, their recall in all the levels, and that of the smallest size holding
    objects, the objects the anchors miss first.

    Args:
        rows (List[Dict[str, Any]]): The rows of summarise_anchor_recall, all the objects first, then every size.
        split_name (str): The name of the split, e.g. "Training".

    Returns:
        str: The line, e.g. "Anchor Recall Of The 703 879 Training Objects: 64.04 % In All Levels, 8.11 % For The
            Smallest Objects (Under 1/16)", without the smallest objects for a split without objects.
    """
    all_objects_row = rows[0]
    line = (f"Anchor Recall Of The {MetricTracker.format_grouped_number(value=all_objects_row['count'])} {split_name} "
            f"Objects: {format_report_value(value=all_objects_row['recall_all_levels'], in_percent=True)} In All "
            f"Levels")

    # Recall of the smallest size holding objects, the sizes being in increasing order after all the objects
    sized_rows = [row for row in rows[1:] if row["count"] > 0]
    if not sized_rows:
        return line
    return (f"{line}, {format_report_value(value=sized_rows[0]['recall_all_levels'], in_percent=True)} For The "
            f"Smallest Objects ({OBJECT_SIZE_NAMES[sized_rows[0]['objects']]})")


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

    # Whether the loaders give the masks as one instance index map per image, set by the instance segmentation
    # subclass from 'Data.instance_index_maps'
    instance_index_maps = False

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
        # precision-recall curves keep their low-confidence end (check_detection_evaluation_configuration refuses it
        # above the deployment one, whose detections the confusion matrix and the images keep among these)
        self.evaluation_score_threshold = experiment_configuration["evaluation"]["score_threshold"]

        # Score threshold of the deployment, keeping the detections of the sample images, the confusion matrix and the
        # error images, so that they show what the deployed detector returns
        self.deployment_score_threshold = model_configuration["Detector"]["inference"]["score_threshold"]

        # Precision targeted by the score threshold of every class, and the error images of the evaluation
        self.target_precision = experiment_configuration["evaluation"]["target_precision"]
        self.error_dumps_configuration = experiment_configuration["evaluation"]["error_dumps"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Optional[Iterable], Iterable]:
        """
        Builds the COCO detection loaders of the three splits, names the classes after the categories of the test
        split for the per-class report, with a colour per class for the sample images, and keeps the file name of
        every test image, which names the sample images. The test loader also gives the objects at the original
        resolution of the images when the evaluation is made at that resolution, and the loaders give one instance
        index map per image instead of its masks with 'Data.instance_index_maps' (instance segmentation only).

        Returns:
            Tuple[Iterable, Optional[Iterable], Iterable]: The training, validation (None when it is not
                built) and test loaders.
        """
        # Class names, class colours and test image names from the test manifest, then the loaders of the three splits
        self.read_test_manifest()
        return get_detection_dataloaders(experiment_configuration=self.experiment_configuration,
                                         model_configuration=self.model_configuration, batch_size=self.batch_size,
                                         load_instance_masks=self.LOAD_INSTANCE_MASKS,
                                         load_instance_index_maps=self.instance_index_maps,
                                         load_original_instances=self.original_resolution,
                                         build_validation_loader=self.builds_validation_loader)

    def analyse_training_data(self) -> Dict[str, Path]:
        """
        Measures at the first launch of a run how well the anchors of the detector fit the objects of the training and
        test splits, before training on them: the share of the objects with an anchor overlapping them by at least the
        foreground threshold of the region proposal network, in every level and in all of them, for all the objects
        and every object size. Saved to anchor_recall.csv and drawn in anchor_recall.png in the run folder, the console
        giving one line per split; a later launch keeps both files when they measured the same split folders at the
        same threshold, and measures again otherwise. The objects are read from the annotations without loading any
        image.

        Returns:
            Dict[str, Path]: The 'Anchor Recall' file and the 'Anchor Recall Figure' written by this launch, sent to
                MLflow, or none when the files of an earlier launch are kept.
        """
        intersection_over_union_threshold = self.model_configuration["RegionProposal"]["foreground_iou_threshold"][
            "min"]
        split_folders = {split_name: str(self.experiment_configuration[folder_setting])
                         for split_name, folder_setting in ANCHOR_RECALL_SPLITS.items()}
        file_path = self.experiment_folder / ANCHOR_RECALL_FILE_NAME
        figure_path = self.experiment_folder / ANCHOR_RECALL_FIGURE_NAME

        # Files of an earlier launch kept when they measured the same split folders at the same threshold, the
        # threshold read back from the 6 decimals of the file
        recorded_settings = read_anchor_recall_settings(file_path=file_path)
        if figure_path.exists() and recorded_settings is not None and recorded_settings[0] == split_folders and (
                math.isclose(recorded_settings[1], intersection_over_union_threshold, abs_tol=1e-6)):
            print_color(color="blue", output=f"Anchor recall measured at the first launch of the run: "
                                             f"{format_path_link(path=figure_path)}")
            return {}

        # Recall of every split at the foreground threshold of the region proposal network, one console line each
        rows, tables = [], []
        for split_name, folder_setting in ANCHOR_RECALL_SPLITS.items():
            data_loader = self.train_dataloader if folder_setting == "train_data_folder" else self.test_dataloader
            anchor_recall = self.measure_anchor_recall(
                dataset=data_loader.dataset, split_name=split_name,
                intersection_over_union_threshold=intersection_over_union_threshold,
                threshold_setting="RegionProposal.foreground_iou_threshold.min", show_progress=False,
                print_table=False)
            rows += [{"split": split_name, "split_folder": split_folders[split_name],
                      "iou_threshold": intersection_over_union_threshold, **row} for row in anchor_recall["rows"]]
            tables.append(anchor_recall["table"])
            print_color(color="blue", output=describe_anchor_recall(rows=anchor_recall["rows"], split_name=split_name))

        # Rows of every split in the file, and their tables drawn one under the other
        write_class_metrics_file(class_rows=rows, evaluation_folder=self.experiment_folder,
                                 file_name=ANCHOR_RECALL_FILE_NAME)
        draw_anchor_recall_figure(tables=tables, output_path=figure_path)
        print_color(color="blue", output=f"Anchor recall of every object size: {format_path_link(path=figure_path)}")
        return {"Anchor Recall": file_path, "Anchor Recall Figure": figure_path}

    def uses_exiting_diagnosis_mode(self) -> bool:
        """
        Tells whether a diagnosis mode ends the program before the training: the input viewer and the speed benchmark
        of every task (BaseTrainer.uses_exiting_diagnosis_mode), or the anchor coverage of the splits
        ('diagnosis_modes.compute_anchor_coverage'), which checks the anchor setting without creating the run folder.

        Returns:
            bool: Whether the program exits after the diagnosis.
        """
        return super().uses_exiting_diagnosis_mode() or self.diagnosis_modes["compute_anchor_coverage"]["activated"]

    def diagnose_data_splits(self) -> None:
        """
        Prints the anchor coverage of the training, validation and test splits when
        'diagnosis_modes.compute_anchor_coverage' is activated: for every split, the share of its objects with an anchor
        overlapping them by at least the 'iou_threshold' of the mode, in every level and in all of them, for all the
        objects and every object size, its recalls coloured from red to green, so that the anchor setting can be
        checked against every split before a launch. Nothing is written; the trainer exits afterwards.
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
                show_progress=True, print_table=True)
        return None

    def measure_anchor_recall(self, dataset: CocoDetectionDataset, split_name: str,
                              intersection_over_union_threshold: float, threshold_setting: str,
                              show_progress: bool, print_table: bool) -> Dict[str, Any]:
        """
        Measures the anchor recall of one split, shared by the check of the training and test splits at the first
        launch of a run (analyse_training_data) and the anchor coverage diagnosis mode (diagnose_data_splits), which
        prints its table: the boxes of the objects read from the annotations without loading any image, the best
        anchor of every object in every level found from the anchor grid of the level, then the recall of every level
        and of all of them at the threshold, for every object size relative to the network input.

        Args:
            dataset (CocoDetectionDataset): The dataset of the split.
            split_name (str): The name of the split in the console, e.g. 'Training'.
            intersection_over_union_threshold (float): The minimum overlap for an object to be covered.
            threshold_setting (str): The configuration setting giving the threshold, named in the legend of the table.
            show_progress (bool): Whether a progress bar follows the images of the split.
            print_table (bool): Whether the table is printed in the console, with the layout of the report.

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
            image_side=dataset.image_size, intersection_over_union_threshold=intersection_over_union_threshold,
            split_name=split_name, threshold_setting=threshold_setting)

        # Table printed with the layout of the report, its recalls coloured
        if print_table:
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
        Moves the images, the per-image ground truth boxes and labels and the window of every image of a detection batch
        to the device, in the precision of the model (labels as int64 class indices).

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a COCO loader.

        Returns:
            DetectionBatch: The batch ready for the detector.
        """
        return {"images": batch["images"].to(device=self.device, dtype=self.dtype),
                "bounding_boxes": [box_tensor.to(device=self.device, dtype=self.dtype)
                                   for box_tensor in batch["bounding_boxes"]],
                "labels": [label_tensor.to(device=self.device, dtype=torch.int64) for label_tensor in batch["labels"]],
                "sample_identifiers": batch["sample_identifiers"],
                "image_windows": batch["image_windows"].to(device=self.device, dtype=torch.float32)}

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
        four losses, with the windows of the images, to which the detections of the validation are clipped.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The detector outputs, with its four losses.
        """
        return self.model(input_tensor=batch["images"], image_windows=batch["image_windows"],
                          ground_truth_bounding_boxes=batch["bounding_boxes"], ground_truth_labels=batch["labels"])

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
        the proposals sampled around it for training, which would inflate the mAP. The detections are clipped to the
        window of their image, as the exported detector clips them.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The detector outputs, with the final detections of every image.
        """
        return self.model(input_tensor=batch["images"], image_windows=batch["image_windows"])

    def build_inference_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Gives the inputs of the inference detector (and instance segmenter) from a prepared batch for the speed
        benchmark: the float32 BGR images in [0, 255] and the window of every image in the network input, the two
        inputs of the exports (TASK_INPUT_NAMES).

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, torch.Tensor]: The 'images' and the [B, 4] float32 'image_windows'.
        """
        return {"images": batch["images"], "image_windows": batch["image_windows"]}

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the lists of detections and ground truths filled during the evaluation of the test split, from which
        the confusion matrix of the deployed detector is built once the class thresholds are found, the folder and the
        counts of the error images, and the overlaps of the objects with the proposals for the proposal recall.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, which names its evaluation folder.

        Returns:
            Dict[str, Any]: The empty 'detections' and 'ground_truths' lists, the 'errors_folder' of the error images
                and their empty 'error_counts' per folder, and the empty 'proposal_recall' accumulator of the detection
                levels.
        """
        return {"detections": [], "ground_truths": [],
                "errors_folder": self.get_evaluation_folder(iteration=iteration) / "Errors", "error_counts": {},
                "proposal_recall": create_proposal_recall_accumulator(levels=list(self.model.get_level_anchors()),
                                                                       image_side=self.input_image_size)}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the valid detections and the ground truth of every kept image of a test batch on the CPU, both expressed
        in network input pixels (in original image pixels for the instance segmentation evaluated at the original
        resolution), adds the overlaps of its objects with the proposals to the proposal recall, and, when the error
        images are on, matches the detections reaching 'Detector.inference.score_threshold' to the objects of the
        image at an overlap of 0.5 and saves the errors of the image as images, while the image is still in the batch.
        The error images keep that threshold for every class, the class thresholds of the confusion matrix being found
        once the whole split is evaluated (save_error_analysis).

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, its lists extended in place.
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

            # Errors of the detections reaching the configured threshold saved as images while the image is still in
            # the batch
            if self.error_dumps_configuration["activated"]:
                kept_detections = detection["scores"] >= self.deployment_score_threshold
                detection_scores = detection["scores"][kept_detections]
                detection_labels = detection["labels"][kept_detections]
                matched_ground_truth_indices = self.match_kept_detections(
                    detection=detection, ground_truth=ground_truth, kept_detections=kept_detections)
                self.dump_errors(accumulator=accumulator, batch=batch, predictions=predictions, image_index=image_index,
                                 matched_ground_truth_indices=matched_ground_truth_indices,
                                 detection_scores=detection_scores, detection_labels=detection_labels,
                                 ground_truth_labels=ground_truth["labels"])

    def match_kept_detections(self, detection: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                              kept_detections: torch.Tensor) -> torch.Tensor:
        """
        Matches the kept detections of one test image to its objects at an overlap of 0.5 (boxes, or masks for the
        instance segmentation), for the confusion matrix and the error images.

        Args:
            detection (Dict[str, torch.Tensor]): The detections of the image (gather_image_detections).
            ground_truth (Dict[str, torch.Tensor]): The objects of the image (gather_image_ground_truth).
            kept_detections (torch.Tensor): The [D] flags of the kept detections.

        Returns:
            torch.Tensor: The object found by every kept detection, -1 for none (match_image_detections).
        """
        overlaps = compute_overlap_matrix(detection=detection, ground_truth=ground_truth,
                                          overlap_type=self.MATCHING_OVERLAP_TYPE)[kept_detections]
        return match_image_detections(overlaps=overlaps, detection_scores=detection["scores"][kept_detections],
                                      detection_labels=detection["labels"][kept_detections],
                                      ground_truth_labels=ground_truth["labels"],
                                      intersection_over_union_threshold=MATCHING_INTERSECTION_OVER_UNION_THRESHOLD)

    def build_deployed_confusion_matrix(self, accumulator: Dict[str, Any],
                                        class_score_thresholds: Dict[int, Optional[Tuple[float, float]]]
                                        ) -> torch.Tensor:
        """
        Builds the confusion matrix of the detector deployed with the class thresholds of the evaluation, once they are
        found, from the detections kept for the evaluation: every detection whose score reaches the threshold of its
        class ('class_thresholds.json', the threshold the export builds into the model,
        deployment/detection_deployment.py), or 'Detector.inference.score_threshold' for a class that never reaches the
        target precision, matched to the objects of its image at an overlap of 0.5.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, with the 'detections' and 'ground_truths' of every
                test image.
            class_score_thresholds (Dict[int, Optional[Tuple[float, float]]]): The threshold and recall of every
                foreground class, or None.

        Returns:
            torch.Tensor: The [K, K] int64 confusion matrix, background included.
        """
        number_classes = len(self.class_names)
        score_thresholds = torch.full(size=(number_classes,), fill_value=float(self.deployment_score_threshold))
        for class_index, class_threshold in class_score_thresholds.items():
            if class_threshold is not None:
                score_thresholds[class_index] = class_threshold[0]

        # Kept detections of every image matched to its objects, the background threshold being unused
        confusion_matrix = torch.zeros(size=(number_classes, number_classes), dtype=torch.int64)
        for detection, ground_truth in zip(accumulator["detections"], accumulator["ground_truths"]):
            kept_detections = detection["scores"] >= score_thresholds[detection["labels"]]
            matched_ground_truth_indices = self.match_kept_detections(detection=detection, ground_truth=ground_truth,
                                                                      kept_detections=kept_detections)
            add_to_confusion_matrix(confusion_matrix=confusion_matrix,
                                    matched_ground_truth_indices=matched_ground_truth_indices,
                                    detection_labels=detection["labels"][kept_detections],
                                    ground_truth_labels=ground_truth["labels"])
        return confusion_matrix

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
        sizes, relative to the area of the network input) and the matching of the confusion matrix. The instance
        segmentation trainer adds the areas of their masks, which size the objects of the mask average precision.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            image_index (int): The index of the image in the batch.

        Returns:
            Dict[str, torch.Tensor]: The ground truth 'boxes' [G, 4] and 'labels' [G] of the image, and the
                'image_area' of the network input.
        """
        image_height, image_width = batch["images"][image_index].shape[-2:]
        return {"boxes": batch["bounding_boxes"][image_index].cpu(), "labels": batch["labels"][image_index].cpu(),
                "image_area": float(image_height * image_width)}

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
        confusion_matrix.csv in the evaluation folder of the checkpoint,
        '<run>/[Checkpoint_]Evaluation/Iteration_<n>/', whatever the per-class report (save_error_analysis). The
        report writes the rates as percentages, the headline mAP@0.5 included in the best checkpoint line of the
        evaluation of several checkpoints, while the files and scalars keep them between 0 and 1.

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
            resolution_name="original image" if self.original_resolution else "network input",
            image_side=None if self.original_resolution else self.input_image_size)
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
                "headline_metric": "mAP@0.5/area", "percent_metrics": ["mAP@0.5/area"]}

    def save_proposal_recall(self, accumulator: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """
        Summarises the proposal recall of the test split for the report of an evaluated checkpoint, in the object
        detection and the instance segmentation: its table, always shown, its TensorBoard scalars, and its rows saved
        to proposal_recall.csv in the evaluation folder of the checkpoint,
        '<run>/[Checkpoint_]Evaluation/Iteration_<n>/', which is sent to MLflow.

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
        Saves the outputs of the evaluation of a checkpoint telling where the detector fails, in its evaluation folder,
        '<run>/[Checkpoint_]Evaluation/Iteration_<n>/', for compute_evaluation_results: the score threshold of every
        class reaching the target precision in class_thresholds.json (the format of the classification tasks, read by
        the export, deployment/detection_deployment.py), and the confusion matrix of the detections the detector
        exported with these thresholds returns (build_deployed_confusion_matrix) in confusion_matrix.csv, both whatever
        the per-class report, the confusion matrix image only with it, its shares written as percentages, also logged
        to TensorBoard. The report gets a note counting the outcomes of the detections and, when the error images are
        on, a note counting them with their folder, the error images keeping 'Detector.inference.score_threshold'.

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

        # Confusion matrix of the detector deployed with the class thresholds, with its background row and column, the
        # image only with the per-class report
        confusion_matrix = self.build_deployed_confusion_matrix(
            accumulator=accumulator, class_score_thresholds=class_score_thresholds)
        number_class_thresholds = sum(class_threshold is not None
                                      for class_threshold in class_score_thresholds.values())
        threshold_text = (f"Score >= {self.deployment_score_threshold:.2f}" if number_class_thresholds == 0 else
                          f"Score >= Class Thresholds ({number_class_thresholds} Of {len(class_score_thresholds)}), "
                          f"{self.deployment_score_threshold:.2f} Otherwise")
        class_names = [self.class_names[class_index] for class_index in range(len(self.class_names))]
        files["Confusion Matrix Table"] = save_confusion_matrix_table(
            confusion_matrix=confusion_matrix, class_names=class_names, output_folder=evaluation_folder)
        if self.per_class_report:
            files["Confusion Matrix Image"], images["Confusion Matrix"] = save_confusion_matrix_image(
                confusion_matrix=confusion_matrix, class_names=class_names, output_folder=evaluation_folder,
                title=f"Confusion Matrix · Iteration {iteration} · {threshold_text} · "
                      f"IoU >= {MATCHING_INTERSECTION_OVER_UNION_THRESHOLD:g}", background_index=BACKGROUND_CLASS_INDEX,
                in_percent=True)

        # Outcomes of the detections of the deployed detector
        outcomes = summarise_confusion_matrix(confusion_matrix=confusion_matrix)
        notes.append(f"Confusion Matrix : {threshold_text.lower()}, IoU >= "
                     f"{MATCHING_INTERSECTION_OVER_UNION_THRESHOLD:g} ({self.MATCHING_OVERLAP_TYPE}), "
                     f"{outcomes['right']} right, {outcomes['wrong_class']} wrong class, {outcomes['false_positives']} "
                     f"false positives, {outcomes['missed']} missed")

        # Number of error images and their folder when some were saved
        if self.error_dumps_configuration["activated"]:
            error_counts = accumulator["error_counts"]
            notes.append(f"Error Images : {sum(error_counts.values())} in {len(error_counts)} folders, at most "
                         f"{self.error_dumps_configuration['maximum_per_class_pair']} per folder, score >= "
                         f"{self.deployment_score_threshold:.2f} for every class")
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

    def build_input_windows(self, batch: Dict[str, Any], sample_index: int, split_name: str) -> List[ViewerWindow]:
        """
        Draws one image for the input viewer and the input samples of a training launch: the image as the network
        receives it with its ground truth boxes captioned with their class names, next to an information panel holding
        its number of objects, a table of its classes (Class row) over their numbers of objects (Count row), its
        augmentations and the objects they removed.

        Args:
            batch (Dict[str, Any]): The batch holding the image, augmented for the training split, on the CPU.
            sample_index (int): The index of the image in the batch.
            split_name (str): "Training", "Validation" or "Test", the title of the window.

        Returns:
            List[ViewerWindow]: The window of the image.
        """
        # The class names of the test manifest, so that the boxes, the table and the removed objects read "square"
        # instead of the category identifier of the label dictionary
        label_names = self.class_names

        # The objects of the image: their number, then a table of their classes in class order over their counts
        class_counts = Counter(batch["labels"][sample_index].tolist())
        if class_counts:
            object_lines = [f"Objects : {sum(class_counts.values())}",
                            PanelTable(row_titles=("Class", "Count"),
                                       columns=[(label_names[class_index], str(class_counts[class_index]))
                                                for class_index in sorted(class_counts)])]
        else:
            object_lines = ["Objects : none"]
        # Then the augmentations of the image and the objects they removed
        augmentation_lines = self.describe_input_augmentations(batch=batch, sample_index=sample_index,
                                                               split_name=split_name, label_names=label_names)
        return [ViewerWindow(image=self.get_input_sample_image(batch=batch, sample_index=sample_index),
                             bounding_boxes=batch["bounding_boxes"][sample_index], labels=batch["labels"][sample_index],
                             label_names=label_names, title=split_name, target_lines=object_lines,
                             augmentation_lines=augmentation_lines, display_input_size=self.input_image_size)]

    def get_input_sample_image(self, batch: Dict[str, Any], sample_index: int) -> torch.Tensor:
        """
        Gives the image of a batch shown by the input viewer and the input samples of a training launch, under its
        boxes: the image itself for the object detection, the instance segmentation trainer painting the masks of its
        objects on it.

        Args:
            batch (Dict[str, Any]): The training, validation or test batch, on the CPU.
            sample_index (int): The index of the image in the batch.

        Returns:
            torch.Tensor: The [3, H, W] image in [0, 255].
        """
        return batch["images"][sample_index]
