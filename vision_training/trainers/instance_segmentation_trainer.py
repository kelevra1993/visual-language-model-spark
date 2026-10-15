"""
Trainer of the instance segmentation task: the detection trainer with the instance masks of the COCO loaders, the
instance segmenter with its mask loss, the masks moved by the geometric augmentations with their boxes recomputed, and
the evaluation of the checkpoints with the box and mask average precisions, the class thresholds, confusion matrix and
error images of the detection trainer matched by the masks, the sample and error images painting the masks under the
boxes. Registered for the instance segmentation task in the task registry.
"""

import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from vision_training.core.contracts import EvaluationResults, InstanceSegmentationBatch
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.models.instance_segmentation.instance_segmenter import InstanceSegmenter
from vision_training.models.instance_segmentation.mask_operations import (compute_mask_intersection_over_union,
                                                                          paste_instance_masks)
from vision_training.trainers.detection_trainer import DetectionTrainer
from vision_training.utilities.visualization import overlay_instance_masks

# Number of detection masks pasted into the image and compared with the ground truth masks at once, bounding the
# memory of the evaluation (a 1024 pixel float mask takes 4 MB)
MASK_OVERLAP_CHUNK_SIZE = 16

# Legends of the tables of the methods, the mask mAP@0.5 comparing the checkpoints
MASK_TABLE_LEGEND = "The mask mAP@0.5 of the area method compares the checkpoints of the run."
BOX_TABLE_LEGEND = "Box average precisions of the same detections, comparable with the object detection runs."


class InstanceSegmentationTrainer(DetectionTrainer):
    """
    Trainer of the instance segmentation task: the instance segmenter trained on COCO-format datasets with
    segmentations, its five losses tracked, its instance masks moved by the geometric augmentations, and every
    evaluated checkpoint scored with the box and mask mean average precisions on the test split.
    """

    TASK_NAME = "instance_segmentation"
    LOAD_INSTANCE_MASKS = True

    # The class thresholds, the confusion matrix and the error images match the detections to the objects by their
    # masks, like the headline mask mAP
    MATCHING_OVERLAP_TYPE = "masks"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the mask threshold of the evaluation and the sample images, then prepares the run like the detection
        trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, RegionProposal, Anchors,
                Detector, MaskHead).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Probability from which a pixel of a pasted mask belongs to its object, for the mask overlaps and the images
        self.mask_threshold = model_configuration["MaskHead"]["inference"]["mask_threshold"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_model(self) -> InstanceSegmenter:
        """
        Builds the instance segmenter from the model configuration.

        Returns:
            InstanceSegmenter: The untrained instance segmenter.
        """
        return InstanceSegmenter(configuration=self.model_configuration, device=self.device, dtype=self.dtype,
                                 verbose=self.diagnosis_modes["verbose"])

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the losses of the detector and the mask loss, tracked on every training and validation batch.

        Returns:
            Dict[str, str]: The display name of every tracked loss.
        """
        return {**super().get_tracked_metrics(), "mask_loss": "Mask Loss"}

    def prepare_batch(self, batch: Dict[str, Any]) -> InstanceSegmentationBatch:
        """
        Moves a batch to the device like the detection trainer, with the uint8 instance masks of every image.

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a COCO loader with instance masks.

        Returns:
            InstanceSegmentationBatch: The batch ready for the instance segmenter.
        """
        return {**super().prepare_batch(batch=batch),
                "instance_masks": [masks.to(device=self.device) for masks in batch["instance_masks"]]}

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the training images, moving the instance masks with the geometric augmentations, the boxes
        becoming the tight boxes of the moved masks and the objects leaving the image being removed with their label
        and mask.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The batch with its augmented images, boxes, labels and instance masks.
        """
        images, bounding_boxes, labels, _, instance_masks = self.data_augmenter.augment_batch(
            images=batch["images"], bounding_boxes=batch["bounding_boxes"], labels=batch["labels"],
            instance_masks=batch["instance_masks"])
        return {**batch, "images": images, "bounding_boxes": bounding_boxes, "labels": labels,
                "instance_masks": instance_masks}

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the instance segmenter with the ground truth boxes, labels and masks of the batch, giving its five losses.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The outputs of the instance segmenter, with its five losses.
        """
        return self.model(input_tensor=batch["images"], ground_truth_bounding_boxes=batch["bounding_boxes"],
                          ground_truth_labels=batch["labels"], ground_truth_masks=batch["instance_masks"])

    def gather_image_detections(self, batch: Dict[str, Any], predictions: Dict[str, Any],
                                image_index: int) -> Dict[str, torch.Tensor]:
        """
        Gives the valid detections of one test image like the detection trainer, with the mask intersection over union
        of every valid detection with every ground truth mask, computed on the device of the model from the detection
        masks pasted at the network resolution, for the mask average precision and the matching of the confusion
        matrix. The masks themselves are not kept, since those of a whole test split would not fit in memory.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The outputs of the instance segmenter.
            image_index (int): The index of the image in the batch.

        Returns:
            Dict[str, torch.Tensor]: The detection 'boxes' [D, 4], 'scores' [D] and 'labels' [D] of the image, and
                their 'mask_overlaps' [D, G] with its objects.
        """
        detection = super().gather_image_detections(batch=batch, predictions=predictions, image_index=image_index)
        detection["mask_overlaps"] = self.compute_mask_overlaps(predictions=predictions,
                                                                ground_truth_masks=batch["instance_masks"][image_index],
                                                                image_index=image_index).cpu()
        return detection

    def compute_mask_overlaps(self, predictions: Dict[str, Any], ground_truth_masks: torch.Tensor,
                              image_index: int) -> torch.Tensor:
        """
        Computes the mask intersection over union of the valid detections of an image with its ground truth masks,
        the detection masks being pasted into their boxes and thresholded at the mask threshold, a few at a time.

        Args:
            predictions (Dict[str, Any]): The outputs of the instance segmenter, with the masks of the detections.
            ground_truth_masks (torch.Tensor): The [G, H, W] binary ground truth masks of the image.
            image_index (int): The index of the image in the batch.

        Returns:
            torch.Tensor: The [D, G] mask intersection over union of the D valid detections.
        """
        number_detections = int(predictions["number_detections"][image_index])
        detection_masks = predictions["final_detection_masks"][image_index, :number_detections]
        detection_boxes = predictions["final_detection_boxes"][image_index, :number_detections]
        image_height, image_width = ground_truth_masks.shape[-2], ground_truth_masks.shape[-1]
        binary_ground_truth_masks = ground_truth_masks > 0
        overlaps = [torch.zeros(size=(0, ground_truth_masks.shape[0]), dtype=torch.float64,
                                device=ground_truth_masks.device)]

        # Paste and compare a chunk of detections at a time
        for chunk_start in range(0, number_detections, MASK_OVERLAP_CHUNK_SIZE):
            chunk_end = chunk_start + MASK_OVERLAP_CHUNK_SIZE
            pasted_masks = paste_instance_masks(masks=detection_masks[chunk_start:chunk_end],
                                                boxes=detection_boxes[chunk_start:chunk_end],
                                                image_height=image_height, image_width=image_width)
            overlaps.append(compute_mask_intersection_over_union(
                predicted_masks=pasted_masks >= self.mask_threshold, ground_truth_masks=binary_ground_truth_masks,
                chunk_size=MASK_OVERLAP_CHUNK_SIZE))
        return torch.cat(tensors=overlaps, dim=0)

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the mask and box mean average precisions of the test split with the three average precision methods,
        the mask mAP@0.5 of the area method comparing the checkpoints, and the score threshold of every class reaching
        the target precision with the masks, and organises them for the report: the mask tables then the box tables
        (methods always shown, classes with the per-class report, the thresholds in the mask table), the box scalars
        under the names of the object detection and the mask scalars under 'Mask' names. The box and mask average
        precisions and the thresholds of every class are saved together to class_metrics.csv in
        '<run>/Evaluation/Iteration_<n>/', with the class thresholds, the confusion matrix and the error images of the
        detection trainer (save_error_analysis).

        Args:
            accumulator (Dict[str, Any]): The detections, with their mask overlaps, and ground truths of the test split,
                with the confusion matrix and the error images.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        mask_summary = self.summarise_average_precisions(accumulator=accumulator, overlap_type="masks",
                                                         title_prefix="Mask", name_prefix="Mask ",
                                                         column_prefix="mask_", legend=MASK_TABLE_LEGEND,
                                                         target_precision=self.target_precision)
        box_summary = self.summarise_average_precisions(accumulator=accumulator, overlap_type="boxes",
                                                        title_prefix="Box", legend=BOX_TABLE_LEGEND)

        # Box and mask average precisions of every class in one row per class, saved whatever the per-class report
        class_rows = [{**box_row, **mask_row} for box_row, mask_row in zip(box_summary["class_rows"],
                                                                           mask_summary["class_rows"], strict=True)]
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        files = {"Class Metrics": write_class_metrics_file(class_rows=class_rows, evaluation_folder=evaluation_folder)}

        # Class thresholds, confusion matrix and error images of the detections of the deployed instance segmenter
        error_analysis = self.save_error_analysis(accumulator=accumulator, iteration=iteration,
                                                  class_score_thresholds=mask_summary["class_score_thresholds"])
        files.update(error_analysis["files"])

        # Tables of the methods first, then the per-class tables, the masks before the boxes
        tables = [mask_summary["tables"][0], box_summary["tables"][0], mask_summary["tables"][1],
                  box_summary["tables"][1]]
        return {"evaluated_samples": f"{len(accumulator['detections'])} Test Images", "metrics": {},
                "notes": error_analysis["notes"], "tables": tables,
                "scalars": {**box_summary["scalars"], **mask_summary["scalars"]},
                "class_scalars": {**box_summary["class_scalars"], **mask_summary["class_scalars"]},
                "images": error_analysis["images"], "files": files, "headline_metric": "Mask mAP@0.5/area"}

    def get_sample_masks(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int,
                         kept_detections: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Gives the masks painted on a sample image under its boxes: the ground truth masks, and the masks of the kept
        detections pasted into their boxes at the network resolution and thresholded at the mask threshold.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The outputs of the instance segmenter.
            sample_index (int): The index of the image in the batch.
            kept_detections (np.ndarray): The [K] flags of the valid detections drawn on the image.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: The [N, H, W] ground truth and [M, H, W] predicted binary
                masks.
        """
        ground_truth_masks = batch["instance_masks"][sample_index]
        number_detections = kept_detections.shape[0]
        kept_indices = torch.from_numpy(np.flatnonzero(kept_detections)).to(device=ground_truth_masks.device)
        detection_masks = predictions["final_detection_masks"][sample_index, :number_detections][kept_indices]
        detection_boxes = predictions["final_detection_boxes"][sample_index, :number_detections][kept_indices]
        pasted_masks = paste_instance_masks(masks=detection_masks.detach(), boxes=detection_boxes.detach(),
                                            image_height=ground_truth_masks.shape[-2],
                                            image_width=ground_truth_masks.shape[-1])
        return ((ground_truth_masks > 0).cpu().numpy(), (pasted_masks >= self.mask_threshold).cpu().numpy())

    def get_input_sample_image(self, batch: Dict[str, Any], sample_index: int) -> torch.Tensor:
        """
        Gives the image of a batch shown by the input viewer with the masks of its objects painted in the colour of
        their class, so that the user checks the masks and their augmentations along with the boxes.

        Args:
            batch (Dict[str, Any]): The training or validation batch, on the CPU.
            sample_index (int): The index of the image in the batch.

        Returns:
            torch.Tensor: The [3, H, W] image in [0, 255] with its masks.
        """
        image = np.ascontiguousarray(batch["images"][sample_index].detach().cpu().clamp(min=0.0, max=255.0)
                                     .to(dtype=torch.uint8).permute(1, 2, 0).numpy())
        painted_image = overlay_instance_masks(image=image,
                                               instance_masks=batch["instance_masks"][sample_index].cpu().numpy(),
                                               labels=batch["labels"][sample_index].cpu().numpy(),
                                               class_colours=self.class_colours)
        return torch.from_numpy(painted_image).permute(2, 0, 1).float()
