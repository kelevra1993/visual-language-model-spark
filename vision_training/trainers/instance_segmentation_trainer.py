"""
Trainer of the instance segmentation task: the detection trainer with the instance masks of the COCO loaders, the
instance segmenter with its mask loss, the masks moved by the geometric augmentations with their boxes recomputed, and
the evaluation of the checkpoints with the box and mask average precisions and mean average recalls (the mask ones
sizing the objects by their mask areas), at the network resolution or, with 'evaluation.original_resolution', at the
original resolution of the test images (the detections restored to them, scored against the objects of the original
annotations), the proposal recall, the class thresholds, confusion matrix and error images of the detection trainer
matched by the masks, the sample and error images painting the masks under the boxes at the network resolution. With
'Data.instance_index_maps', the loaders give one instance index map per image, expanded into the masks on the device
by prepare_batch (and on the CPU for the input viewer). Registered for the instance segmentation task in the task
registry.
"""

import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from vision_training.core.contracts import EvaluationResults, InstanceSegmentationBatch
from vision_training.data.detection.coco_masks import expand_instance_index_maps
from vision_training.data.preprocessing import restore_boxes_to_original_image
from vision_training.evaluation.detection.average_precision_report import summarise_average_precisions
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.models.instance_segmentation.instance_segmenter import InstanceSegmenter
from vision_training.models.instance_segmentation.mask_operations import (compute_mask_intersection_over_union,
                                                                          paste_instance_masks)
from vision_training.trainers.detection_trainer import DetectionTrainer
from vision_training.utilities.visualization import convert_image_to_array, overlay_instance_masks

# Number of detection masks of a 1024 x 1024 image pasted and compared with the ground truth masks at once, scaled
# inversely to the number of pixels of the evaluated image, bounding the memory of the pasted detection masks at any
# resolution (a 1024 x 1024 float mask takes 4 MB); the ground truth masks of the image are compared all at once
MASK_OVERLAP_CHUNK_SIZE = 16
MASK_OVERLAP_REFERENCE_PIXELS = 1024 * 1024

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
        Reads the mask threshold of the evaluation and the sample images, the resolution of the evaluation and whether
        the loaders give instance index maps, then prepares the run like the detection trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, DetectionLevels,
                RegionProposal, Detector, MaskHead).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Probability from which a pixel of a pasted mask belongs to its object, for the mask overlaps and the images
        self.mask_threshold = model_configuration["MaskHead"]["inference"]["mask_threshold"]

        # Boxes and masks scored at the original resolution of the test images, and the masks of every image loaded
        # as one instance index map, read before the loaders are built
        self.original_resolution = experiment_configuration["evaluation"]["original_resolution"]
        self.instance_index_maps = model_configuration["Data"]["instance_index_maps"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_model(self) -> InstanceSegmenter:
        """
        Builds the instance segmenter from the model configuration.

        Returns:
            InstanceSegmenter: The untrained instance segmenter.
        """
        return InstanceSegmenter(configuration=self.model_configuration, device=self.device, dtype=self.dtype)

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the losses of the detector and the mask loss, tracked on every training and validation batch.

        Returns:
            Dict[str, str]: The display name of every tracked loss.
        """
        return {**super().get_tracked_metrics(), "mask_loss": "Mask Loss"}

    def prepare_batch(self, batch: Dict[str, Any]) -> InstanceSegmentationBatch:
        """
        Moves a batch to the device like the detection trainer, with the uint8 instance masks of every image: the
        instance index maps of the loaders ('Data.instance_index_maps') are moved to the device and expanded there
        into the masks, so that the rest of the training and the evaluation read the same masks either way. The
        masks and boxes of the objects at the original resolution, given by the test loader with
        'evaluation.original_resolution', stay on the CPU and are moved one image at a time by the evaluation, the
        original masks of a whole batch being possibly too large for the device.

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a COCO loader with instance masks.

        Returns:
            InstanceSegmentationBatch: The batch ready for the instance segmenter.
        """
        # Masks expanded on the device from the instance index maps, or the masks of the loader moved to it
        if "instance_index_maps" in batch:
            instance_masks = expand_instance_index_maps(
                index_maps=batch["instance_index_maps"].to(device=self.device),
                object_counts=[image_labels.shape[0] for image_labels in batch["labels"]])
        else:
            instance_masks = [masks.to(device=self.device) for masks in batch["instance_masks"]]
        prepared_batch = {**super().prepare_batch(batch=batch), "instance_masks": instance_masks}

        # Objects at the original resolution of the test images, only in the test batches of that evaluation
        if "original_instance_masks" in batch:
            prepared_batch["original_instance_masks"] = batch["original_instance_masks"]
            prepared_batch["original_bounding_boxes"] = batch["original_bounding_boxes"]
        return prepared_batch

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the training images, moving the instance masks with the geometric augmentations, the boxes
        becoming the tight boxes of the moved masks and the objects leaving the image being removed with their label
        and mask. The batch of the input viewer comes from the loader without prepare_batch: its instance index maps
        are expanded into the masks here, on the CPU, and left out of the augmented batch, which they no longer
        match.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The batch with its augmented images, boxes, labels and instance masks, and the
                'augmentation_reports' of its samples for the input viewer.
        """
        # Masks of the input viewer batch expanded from its maps, the prepared batches holding their masks already
        if "instance_index_maps" in batch:
            batch = {**{key: value for key, value in batch.items() if key != "instance_index_maps"},
                     "instance_masks": expand_instance_index_maps(
                         index_maps=batch["instance_index_maps"],
                         object_counts=[image_labels.shape[0] for image_labels in batch["labels"]])}
        images, bounding_boxes, labels, _, instance_masks, _, augmentation_reports = self.data_augmenter.augment_batch(
            images=batch["images"], bounding_boxes=batch["bounding_boxes"], labels=batch["labels"],
            instance_masks=batch["instance_masks"])
        return {**batch, "images": images, "bounding_boxes": bounding_boxes, "labels": labels,
                "instance_masks": instance_masks, "augmentation_reports": augmentation_reports}

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
        of every valid detection with every ground truth mask and the area of every detection mask, computed on the
        device of the model from the detection masks pasted into the image, for the mask average precision (the areas
        sizing the detections) and the matching of the confusion matrix. The masks themselves are not kept, since
        those of a whole test split would not fit in memory. With 'evaluation.original_resolution', the detection
        boxes are restored to the original image (clipped to it for the box average precision, whole for pasting the
        masks, which cover the box they were predicted in) and the masks are compared with the original ground truth
        masks; otherwise everything stays at the network resolution.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The outputs of the instance segmenter.
            image_index (int): The index of the image in the batch.

        Returns:
            Dict[str, torch.Tensor]: The detection 'boxes' [D, 4], 'scores' [D] and 'labels' [D] of the image, their
                'mask_overlaps' [D, G] with its objects and their 'mask_areas' [D].
        """
        detection = super().gather_image_detections(batch=batch, predictions=predictions, image_index=image_index)
        number_detections = int(predictions["number_detections"][image_index])
        network_boxes = predictions["final_detection_boxes"][image_index, :number_detections]

        # Ground truth masks of the original image, moved to the device one image at a time, with the detection boxes
        # restored to that image
        if self.original_resolution:
            ground_truth_masks = batch["original_instance_masks"][image_index].to(device=self.device)
            image_height, image_width = ground_truth_masks.shape[-2], ground_truth_masks.shape[-1]
            detection["boxes"] = self.restore_detection_boxes(boxes=network_boxes, image_height=image_height,
                                                              image_width=image_width, clip=True).cpu()
            paste_boxes = self.restore_detection_boxes(boxes=network_boxes, image_height=image_height,
                                                       image_width=image_width, clip=False)
        else:
            ground_truth_masks = batch["instance_masks"][image_index]
            paste_boxes = network_boxes
        mask_overlaps, mask_areas = self.compute_mask_overlaps(predictions=predictions, detection_boxes=paste_boxes,
                                                               ground_truth_masks=ground_truth_masks,
                                                               image_index=image_index)
        detection["mask_overlaps"], detection["mask_areas"] = mask_overlaps.cpu(), mask_areas.cpu()
        return detection

    def gather_image_ground_truth(self, batch: Dict[str, Any], image_index: int) -> Dict[str, torch.Tensor]:
        """
        Gives the objects of one test image like the detection trainer, with the number of pixels of every ground truth
        mask, which sizes the objects of the mask average precision. With 'evaluation.original_resolution', the boxes
        and the mask areas are those of the original annotations, in the order of the objects at the network
        resolution kept for the proposal recall and the error images.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            image_index (int): The index of the image in the batch.

        Returns:
            Dict[str, torch.Tensor]: The ground truth 'boxes' [G, 4], 'labels' [G] and 'mask_areas' [G] of the image.
        """
        ground_truth = super().gather_image_ground_truth(batch=batch, image_index=image_index)

        # Objects of the original annotations, or those of the network input
        if self.original_resolution:
            ground_truth["boxes"] = batch["original_bounding_boxes"][image_index].cpu()
            ground_truth_masks = batch["original_instance_masks"][image_index]
        else:
            ground_truth_masks = batch["instance_masks"][image_index]
        ground_truth["mask_areas"] = (ground_truth_masks > 0).flatten(start_dim=1).sum(dim=1).cpu()
        return ground_truth

    def restore_detection_boxes(self, boxes: torch.Tensor, image_height: int, image_width: int,
                                clip: bool) -> torch.Tensor:
        """
        Maps the detection boxes of the network input back to the original test image for the evaluation at the
        original resolution, undoing the resizing and the padding of the preprocessing (Data.image_settings) as a
        client of the exported model would.

        Args:
            boxes (torch.Tensor): The [D, 4] [x1, y1, x2, y2] detection boxes in network input pixels.
            image_height (int): The height of the original image.
            image_width (int): The width of the original image.
            clip (bool): Whether the boxes stop at the image border, for the box average precision, or are kept whole,
                for pasting the masks.

        Returns:
            torch.Tensor: The [D, 4] float32 boxes in original image pixels, on the device of the given boxes.
        """
        image_settings = self.model_configuration["Data"]["image_settings"]
        restored_boxes = restore_boxes_to_original_image(
            bounding_boxes=boxes.detach().to(dtype=torch.float32).cpu().numpy(), image_width=image_width,
            image_height=image_height, image_size=image_settings["size"], keep_ratio=image_settings["keep_ratio"],
            clip=clip)
        return torch.from_numpy(restored_boxes).to(device=boxes.device, dtype=torch.float32)

    def compute_mask_overlaps(self, predictions: Dict[str, Any], detection_boxes: torch.Tensor,
                              ground_truth_masks: torch.Tensor, image_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the mask intersection over union of the valid detections of an image with its ground truth masks and
        the area of every detection mask, the detection masks being pasted into their boxes at the size of the ground
        truth masks and thresholded at the mask threshold, a few at a time (fewer on larger images).

        Args:
            predictions (Dict[str, Any]): The outputs of the instance segmenter, with the masks of the detections.
            detection_boxes (torch.Tensor): The [D, 4] boxes the masks of the D valid detections are pasted into, in
                the pixels of the ground truth masks.
            ground_truth_masks (torch.Tensor): The [G, H, W] binary ground truth masks of the image.
            image_index (int): The index of the image in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The [D, G] mask intersection over union of the D valid detections, and
                the [D] number of pixels of their thresholded masks.
        """
        number_detections = int(predictions["number_detections"][image_index])
        detection_masks = predictions["final_detection_masks"][image_index, :number_detections]
        image_height, image_width = ground_truth_masks.shape[-2], ground_truth_masks.shape[-1]
        binary_ground_truth_masks = ground_truth_masks > 0
        overlaps = [torch.zeros(size=(0, ground_truth_masks.shape[0]), dtype=torch.float64,
                                device=ground_truth_masks.device)]
        mask_areas = [torch.zeros(size=(0,), dtype=torch.int64, device=ground_truth_masks.device)]

        # Masks compared at once, as many as a chunk of 1024 x 1024 masks takes, at least one
        chunk_size = max(1, MASK_OVERLAP_CHUNK_SIZE * MASK_OVERLAP_REFERENCE_PIXELS // (image_height * image_width))

        # Paste and compare a chunk of detections at a time, counting the pixels of their masks
        for chunk_start in range(0, number_detections, chunk_size):
            chunk_end = chunk_start + chunk_size
            pasted_masks = paste_instance_masks(masks=detection_masks[chunk_start:chunk_end],
                                                boxes=detection_boxes[chunk_start:chunk_end],
                                                image_height=image_height, image_width=image_width)
            binary_detection_masks = pasted_masks >= self.mask_threshold
            overlaps.append(compute_mask_intersection_over_union(
                predicted_masks=binary_detection_masks, ground_truth_masks=binary_ground_truth_masks,
                chunk_size=chunk_size))
            # Pixels counted in int32 (twice as fast as the int64 sum of booleans), kept in int64 for the evaluation
            mask_areas.append(binary_detection_masks.flatten(start_dim=1).sum(dim=1, dtype=torch.int32).to(
                dtype=torch.int64))
        return torch.cat(tensors=overlaps, dim=0), torch.cat(tensors=mask_areas)

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the mask and box mean average precisions of the test split with the three average precision methods
        and for every object size (by mask and by box area), the mask and box mean average recalls, the mask mAP@0.5 of
        the area method comparing the checkpoints, the score threshold of every class reaching the target precision
        with the masks and the proposal recall, and organises them for the report: the mask tables then the box tables
        (methods and object sizes always shown, then the mean average recalls and the proposal recall, classes with the
        per-class report, the thresholds in the mask table), the box scalars under the names of the object detection
        and the mask scalars under 'Mask' names. The box and mask average precisions and the thresholds of every class
        are saved together to class_metrics.csv in '<run>/[Checkpoint_]Evaluation/Iteration_<n>/', with the proposal
        recall (proposal_recall.csv), the class thresholds, the confusion matrix and the error images of the detection
        trainer (save_error_analysis). The report writes the rates as percentages, the headline mask mAP@0.5 included
        in the best checkpoint line of the evaluation of several checkpoints.

        Args:
            accumulator (Dict[str, Any]): The detections, with their mask overlaps and areas, and ground truths of the
                test split, with the confusion matrix, the error images and the proposal recall.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        # Mask average precisions with the class thresholds, then the box ones, at the resolution of the evaluation
        resolution_name = "original image" if self.original_resolution else "network input"
        mask_summary = summarise_average_precisions(
            detections=accumulator["detections"], ground_truths=accumulator["ground_truths"],
            number_classes=len(set(self.label_dictionary.values())), class_names=self.class_names, overlap_type="masks",
            title_prefix="Mask", name_prefix="Mask ", column_prefix="mask_", legend=MASK_TABLE_LEGEND,
            target_precision=self.target_precision, resolution_name=resolution_name)
        box_summary = summarise_average_precisions(
            detections=accumulator["detections"], ground_truths=accumulator["ground_truths"],
            number_classes=len(set(self.label_dictionary.values())), class_names=self.class_names, overlap_type="boxes",
            title_prefix="Box", legend=BOX_TABLE_LEGEND, resolution_name=resolution_name)
        proposal_recall = self.save_proposal_recall(accumulator=accumulator, iteration=iteration)

        # Box and mask average precisions of every class in one row per class, saved whatever the per-class report,
        # and the proposal recall
        class_rows = [{**box_row, **mask_row} for box_row, mask_row in zip(box_summary["class_rows"],
                                                                           mask_summary["class_rows"], strict=True)]
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        files = {"Class Metrics": write_class_metrics_file(class_rows=class_rows, evaluation_folder=evaluation_folder),
                 "Proposal Recall": proposal_recall["file"]}

        # Class thresholds, confusion matrix and error images of the detections of the deployed instance segmenter
        error_analysis = self.save_error_analysis(accumulator=accumulator, iteration=iteration,
                                                  class_score_thresholds=mask_summary["class_score_thresholds"])
        files.update(error_analysis["files"])

        # Tables of the methods and object sizes first, then the mean average recalls and the proposal recall, then the
        # per-class tables, each followed by its per-class recall table, the masks before the boxes
        tables = [mask_summary["tables"]["methods"], mask_summary["tables"]["sizes"], box_summary["tables"]["methods"],
                  box_summary["tables"]["sizes"], mask_summary["tables"]["recalls"], box_summary["tables"]["recalls"],
                  proposal_recall["table"], mask_summary["tables"]["classes"], mask_summary["tables"]["class_recalls"],
                  box_summary["tables"]["classes"], box_summary["tables"]["class_recalls"]]
        return {"evaluated_samples": f"{len(accumulator['detections'])} Test Images", "metrics": {},
                "notes": error_analysis["notes"], "tables": tables,
                "scalars": {**box_summary["scalars"], **mask_summary["scalars"], **proposal_recall["scalars"]},
                "class_scalars": {**box_summary["class_scalars"], **mask_summary["class_scalars"]},
                "images": error_analysis["images"], "files": files, "headline_metric": "Mask mAP@0.5/area",
                "percent_metrics": ["Mask mAP@0.5/area"]}

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
        Gives the image of a batch shown by the input viewer and the input samples of a training launch with the masks
        of its objects painted in the colour of their class, so that the user checks the masks and their augmentations
        along with the boxes. The validation and test batches come from the loader as they are: with
        'Data.instance_index_maps', the map of the image is expanded into its masks here.

        Args:
            batch (Dict[str, Any]): The training, validation or test batch, on the CPU.
            sample_index (int): The index of the image in the batch.

        Returns:
            torch.Tensor: The [3, H, W] image in [0, 255] with its masks.
        """
        image = convert_image_to_array(image=batch["images"][sample_index])

        # Masks of the image, expanded from its instance index map when the batch holds maps
        if "instance_index_maps" in batch:
            instance_masks = expand_instance_index_maps(
                index_maps=batch["instance_index_maps"][sample_index:sample_index + 1],
                object_counts=[batch["labels"][sample_index].shape[0]])[0]
        else:
            instance_masks = batch["instance_masks"][sample_index]
        painted_image = overlay_instance_masks(image=image, instance_masks=instance_masks.cpu().numpy(),
                                               labels=batch["labels"][sample_index].cpu().numpy(),
                                               class_colours=self.class_colours)
        return torch.from_numpy(painted_image).permute(2, 0, 1).float()
