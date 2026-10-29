"""
Trainer of the semantic segmentation task: the data loaders of the images and their masks (or of their tiles), the
U-Net segmenter with the balanced class weights of its cross entropy, the augmentation of the masks with the images,
the pixel accuracy and mean IoU tracked on every batch, and the evaluation of the checkpoints on the test split, the
test images being segmented tile by tile when the run trains on tiles and scored at the network or at their original
resolution (pixel confusion matrix, boundary counts, global and per-class metrics, with the per-class report the ROC
and precision-recall curves of the pixels of every class, sample grids of fixed test images and the predicted masks).
Registered for the semantic_segmentation task.
"""

import json
import cv2
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision_training.core.contracts import (IGNORED_LABEL_INDEX, SAMPLE_IDENTIFIERS_KEY, EvaluationResults,
                                            SegmentationBatch)
from vision_training.data.loaders import get_segmentation_dataloaders
from vision_training.data.preprocessing import restore_prediction_to_original_image
from vision_training.data.split_description import count_split_samples
from vision_training.evaluation.classification.confusion_matrix import (save_confusion_matrix_image,
                                                                        save_confusion_matrix_table)
from vision_training.evaluation.classification.score_curves import (compute_histogram_score_curves,
                                                                     compute_macro_curve_metrics,
                                                                     save_score_curve_images)
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.evaluation.segmentation.boundary_metrics import (BOUNDARY_COUNT_COLUMNS, compute_boundary_counts,
                                                                      compute_boundary_metrics)
from vision_training.evaluation.segmentation.segmentation_metrics import (SCORE_HISTOGRAM_BINS,
                                                                          compute_pixel_confusion_matrix,
                                                                          compute_pixel_score_histograms,
                                                                          compute_segmentation_metrics)
from vision_training.evaluation.segmentation.segmentation_report import (build_segmentation_class_rows,
                                                                         build_segmentation_class_scalars,
                                                                         build_segmentation_class_table)
from vision_training.losses.classification_losses import compute_balanced_class_weights
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.segmentation.sliding_window import segment_image_by_tiles
from vision_training.models.task_model import TaskModel
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.os_utilities import format_path_link, print_color
from vision_training.utilities.visualization import (build_class_colours, build_segmentation_sample_grid,
                                                     convert_image_to_array, overlay_segmentation_mask,
                                                     view_input_data)

# File of the run folder caching the pixels of every class in the training masks, read back when the run resumes
CLASS_PIXEL_COUNTS_FILE_NAME = "class_pixel_counts.json"

# Longest side of the panels of a sample image, the large test images segmented tile by tile being reduced to it
SAMPLE_PANEL_MAXIMUM_SIZE = 1024


class SegmentationTrainer(BaseTrainer):
    """
    Trainer of the semantic segmentation task: the U-Net segmenter trained on images and class-index masks (Native
    loader), the masks following every geometric augmentation of the images, with its cross entropy and Dice losses,
    pixel accuracy and mean IoU tracked on every batch, and every evaluated checkpoint scored on the labelled pixels of
    the test split, at the network resolution or at the original resolution of the images: pixel accuracy, mean IoU,
    frequency-weighted IoU, mean Dice, mean boundary IoU and boundary F1, the IoU, Dice, boundary IoU, boundary F1,
    precision and recall of every class, the pixel confusion matrix, the sample grids of a few fixed test images and,
    when asked, the predicted mask of every test image. With 'Data.tiling', the run trains on tiles of the scaled
    images and segments the whole test images tile by tile.
    """

    TASK_NAME = "semantic_segmentation"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the segmentation settings used by the hooks, then prepares the training run through the general trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, SegmentationDecoder,
                SegmentationLoss).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Losses of the model, the weighting of the classes in the cross entropy, the deep supervision heads, the
        # training on tiles, and the evaluation settings: saved predicted masks, resolution and boundary tolerance
        self.loss_configuration = model_configuration["SegmentationLoss"]
        self.deep_supervision_weights = model_configuration["SegmentationDecoder"]["deep_supervision_weights"]
        self.tiling_configuration = model_configuration["Data"]["tiling"]
        self.save_predicted_masks = experiment_configuration["evaluation"]["save_predicted_masks"]
        self.original_resolution = experiment_configuration["evaluation"]["original_resolution"]
        self.boundary_tolerance = experiment_configuration["evaluation"]["boundary_tolerance"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the segmentation loaders of the three splits, and takes the class names from the categories of the
        training manifest with a colour per class for the mask overlays and the sample grids.

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders.
        """
        loaders = get_segmentation_dataloaders(experiment_configuration=self.experiment_configuration,
                                               model_configuration=self.model_configuration,
                                               batch_size=self.batch_size)
        training_dataset = loaders[0].dataset
        self.class_names = [training_dataset.class_names[class_index]
                            for class_index in range(training_dataset.number_classes)]
        self.class_colours = build_class_colours(number_classes=training_dataset.number_classes)
        return loaders

    def list_class_names(self) -> Optional[List[str]]:
        """
        Gives the class names taken from the training manifest by build_dataloaders, the background first, for the
        record of the data of the run and the transfer of the class layers of the segmenter to a run adding classes
        after them ('SegmentationDecoder.pretrained.load_class_layers').

        Returns:
            Optional[List[str]]: The class names, in class index order.
        """
        return self.class_names

    def get_test_sample_classes(self) -> Dict[int, List[int]]:
        """
        Lists the test images without their classes, counting the pixels of every class of the test masks being too
        slow, so that the fixed test images drawn at every evaluation are evenly spread over the test split.

        Returns:
            Dict[int, List[int]]: An empty list for every identifier of the test images.
        """
        return {sample["identifier"]: [] for sample in self.test_dataloader.dataset.samples}

    def get_class_pixel_counts(self) -> List[int]:
        """
        Gives the number of pixels of every class in the training masks, from which the balanced class weights of the
        cross entropy are computed. Counting them reads every training mask, so the counts are saved in the run folder
        ('class_pixel_counts.json') and read back when the run resumes, as long as they were counted on the same
        training split (folder and number of images) with the same label mapping.

        Returns:
            List[int]: The number of training pixels of every class index.
        """
        training_dataset = self.train_dataloader.dataset
        counts_path = self.experiment_folder / CLASS_PIXEL_COUNTS_FILE_NAME
        count_settings = {"train_data_folder": str(self.experiment_configuration["train_data_folder"]),
                          "number_images": len(training_dataset), "label_mode": self.data_configuration["label_mode"],
                          "label_dictionary": {str(category): class_index
                                               for category, class_index in self.label_dictionary.items()}}

        # Counts of a previous training of the run on the same split and classes
        if counts_path.is_file():
            saved_counts = json.loads(s=counts_path.read_text())
            if saved_counts["settings"] == count_settings:
                print_color(color="blue", output=f"Class Pixel Counts Read From : {format_path_link(path=counts_path)}")
                return saved_counts["class_pixel_counts"]

        # Count every training mask, then keep the counts in the run folder when it exists (not in the input viewer)
        print_color(color="blue", output=f"Counting The Pixels Of Every Class In The {len(training_dataset)} Training "
                                         f"Masks...")
        class_pixel_counts = training_dataset.count_class_pixels()
        if self.experiment_folder.is_dir():
            counts_path.write_text(data=json.dumps(obj={"settings": count_settings,
                                                        "class_pixel_counts": class_pixel_counts}, indent=4))
        return class_pixel_counts

    def compute_class_weights(self) -> torch.Tensor:
        """
        Gives the weight of every class in the cross entropy for the current 'SegmentationLoss.class_weighting':
        inversely proportional to the number of pixels of the class in the training masks when "balanced", 1 for every
        class otherwise.

        Returns:
            torch.Tensor: The [K] float32 class weights.
        """
        if self.loss_configuration["class_weighting"] == "balanced":
            return compute_balanced_class_weights(class_counts=self.get_class_pixel_counts())
        return torch.ones(size=(len(self.class_names),), dtype=torch.float32)

    def build_model(self) -> TaskModel:
        """
        Builds the segmenter from the model registry, with the class weights of its cross entropy
        (compute_class_weights).

        Returns:
            TaskModel: The untrained model.
        """
        model = TASK_MODEL_CLASSES[self.TASK_NAME](configuration=self.model_configuration, device=self.device,
                                                   dtype=self.dtype)
        model.class_weights.copy_(self.compute_class_weights())
        return model

    def load_model(self, iteration: int) -> None:
        """
        Loads a checkpoint of the run, then sets the class weights of the current configuration again: they are saved
        in the checkpoint with the model, while the loss settings give no token to the run name and may change between
        two resumptions of a run (e.g. turning the balanced class weights on).

        Args:
            iteration (int): The exact iteration number identifying the checkpoint to load.
        """
        super().load_model(iteration=iteration)
        self.model.class_weights.copy_(self.compute_class_weights())

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the losses returned by the segmenter (the cross entropy, Dice, Lovász and boundary losses whose weight is
        not 0, and the loss of the deep supervision heads), the total loss, and the pixel accuracy and mean IoU tracked
        on every training and validation batch.

        Returns:
            Dict[str, str]: The display name of every tracked metric.
        """
        tracked_metrics = {"total_loss": "Total Loss"}
        for loss_key, weight_key, display_name in [("segmentation_loss", "cross_entropy_weight", "Cross Entropy Loss"),
                                                   ("dice_loss", "dice_weight", "Dice Loss"),
                                                   ("lovasz_loss", "lovasz_weight", "Lovasz Loss"),
                                                   ("boundary_loss", "boundary_weight", "Boundary Loss")]:
            if self.loss_configuration[weight_key] > 0:
                tracked_metrics[loss_key] = display_name
        if self.deep_supervision_weights:
            tracked_metrics["deep_supervision_loss"] = "Deep Supervision Loss"
        tracked_metrics.update({"pixel_accuracy": "Pixel Accuracy", "mean_iou": "Mean IoU"})
        return tracked_metrics

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the images and categories of a segmentation split from its labels.json manifest.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of images and their description with the number of categories.
        """
        number_images, number_categories = count_split_samples(split_data_folder=split_data_folder)
        return number_images, f"{number_images} images, {number_categories} categories"

    def prepare_batch(self, batch: Dict[str, Any]) -> SegmentationBatch:
        """
        Moves the images and masks of a segmentation batch to the device, the images in the precision of the model and
        the masks as int64 class indices.

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a segmentation loader.

        Returns:
            SegmentationBatch: The batch ready for the segmenter.
        """
        return {**batch, "images": batch["images"].to(device=self.device, dtype=self.dtype),
                "masks": batch["masks"].to(device=self.device, dtype=torch.int64)}

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the images of a training batch together with their masks, which follow every geometric
        augmentation with the nearest interpolation (the pixels uncovered by a rotation being ignored) and are left
        untouched by the photometric ones.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The batch with its augmented images and masks, and the 'augmentation_reports' of its
                samples for the input viewer.
        """
        augmented_images, _, _, augmented_masks, _, _, augmentation_reports = self.data_augmenter.augment_batch(
            images=batch["images"], masks=batch["masks"])
        return {**batch, "images": augmented_images, "masks": augmented_masks,
                "augmentation_reports": augmentation_reports}

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the segmenter with the masks of the batch, so that it returns its losses.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The segmenter outputs, with its losses.
        """
        return self.model(input_tensor=batch["images"], masks=batch["masks"])

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the segmenter without masks, as in deployment, on the whole test images: at once on the images resized
        to the network input, or tile by tile on the scaled images of a run trained on tiles, the probabilities of the
        overlapping tiles being averaged.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The [B, K, H, W] 'class_probabilities' of every pixel, with the 'class_logits' without
                tiling.
        """
        if not self.tiling_configuration["activated"]:
            return self.model(input_tensor=batch["images"])
        return {"class_probabilities": torch.stack(tensors=[
            segment_image_by_tiles(image=image, predict_tile_probabilities=self.predict_tile_probabilities,
                                   tile_size=self.input_image_size, overlap=self.tiling_configuration["overlap"],
                                   tiles_per_batch=self.batch_size) for image in batch["images"]])}

    def predict_tile_probabilities(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Gives the class probabilities of a batch of tiles of the test images, for the segmentation tile by tile of a
        run trained on tiles (segment_image_by_tiles).

        Args:
            tiles (torch.Tensor): The [N, 3, S, S] tiles, in the precision of the model.

        Returns:
            torch.Tensor: The [N, K, S, S] class probabilities of the tiles.
        """
        return self.model(input_tensor=tiles)["class_probabilities"]

    def compute_batch_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Computes the pixel accuracy and the mean IoU of a training or validation batch from the logits of its forward
        pass, without any extra forward pass, on its labelled pixels.

        Args:
            batch (Dict[str, Any]): The prepared batch.
            model_outputs (Dict[str, Any]): The segmenter outputs.

        Returns:
            Dict[str, torch.Tensor]: The 'pixel_accuracy' and 'mean_iou' of the batch.
        """
        with torch.no_grad():
            confusion_matrix = compute_pixel_confusion_matrix(class_maps=model_outputs["class_logits"].argmax(dim=1),
                                                              masks=batch["masks"],
                                                              number_classes=len(self.class_names))
            metrics = compute_segmentation_metrics(confusion_matrix=confusion_matrix)
        return {"pixel_accuracy": torch.tensor(data=metrics["pixel_accuracy"]),
                "mean_iou": torch.tensor(data=metrics["mean_iou"])}

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the pixel confusion matrix and the boundary counts filled on the device during the evaluation of the
        test split, with the per-class report the histograms of the pixel probabilities of the ROC and precision-recall
        curves, and the folder of the predicted masks of the checkpoint.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, which names its evaluation folder.

        Returns:
            Dict[str, Any]: The empty [K, K] 'confusion_matrix' and [K, 6] 'boundary_counts', with the per-class report
                the empty [K, 2, SCORE_HISTOGRAM_BINS] 'score_histograms', the 'number_images' evaluated and the
                'predicted_masks_folder'.
        """
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        number_classes = len(self.class_names)
        accumulator = {"confusion_matrix": torch.zeros(size=(number_classes, number_classes), dtype=torch.int64,
                                                       device=self.device),
                       "boundary_counts": torch.zeros(size=(number_classes, len(BOUNDARY_COUNT_COLUMNS)),
                                                      dtype=torch.int64, device=self.device),
                       "number_images": 0, "predicted_masks_folder": evaluation_folder / "Predicted_Masks"}

        # Histograms of the pixel probabilities of every class, only for the curves of the per-class report
        if self.per_class_report:
            accumulator["score_histograms"] = torch.zeros(size=(number_classes, 2, SCORE_HISTOGRAM_BINS),
                                                          dtype=torch.int64, device=self.device)
        return accumulator

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Adds the labelled pixels of the kept images of a test batch to the confusion matrix and to the boundary
        counts, with the per-class report their probabilities to the histograms of the curves, on the device, then,
        when asked, saves the predicted mask of every image. The predictions are compared with the masks of the batch
        (network input, or scaled image with tiling), or with 'original_resolution' with the original mask of every
        image, its probabilities being restored to the original image first.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, modified in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The segmenter outputs.
            sample_indices (List[int]): The indices of the images to keep.
        """
        test_dataset = self.test_dataloader.dataset
        number_classes = len(self.class_names)
        sample_identifiers = batch[SAMPLE_IDENTIFIERS_KEY][sample_indices].tolist()

        # Groups of predicted class maps and masks of the same size: the whole batch at the resolution of its masks, or
        # every image at its own original size, the letterbox padding being cut off unless the run trains on tiles
        if self.original_resolution:
            evaluated_groups = []
            for sample_index, sample_identifier in zip(sample_indices, sample_identifiers):
                original_mask = torch.from_numpy(test_dataset.read_class_mask(sample_identifier=sample_identifier))
                restored_probabilities = restore_prediction_to_original_image(
                    prediction=predictions["class_probabilities"][sample_index], original_height=original_mask.shape[0],
                    original_width=original_mask.shape[1],
                    keep_ratio=self.data_configuration["image_settings"]["keep_ratio"] and
                    not self.tiling_configuration["activated"])
                evaluated_groups.append((restored_probabilities.argmax(dim=0, keepdim=True),
                                         original_mask.unsqueeze(dim=0).to(device=self.device)))

                # Probabilities of the pixels of the image for the curves, before they are let go
                if self.per_class_report:
                    accumulator["score_histograms"] += compute_pixel_score_histograms(
                        class_probabilities=restored_probabilities.unsqueeze(dim=0),
                        masks=evaluated_groups[-1][1], number_bins=SCORE_HISTOGRAM_BINS)
        else:
            evaluated_groups = [(predictions["class_probabilities"][sample_indices].argmax(dim=1),
                                 batch["masks"][sample_indices])]

            # Probabilities of the pixels of the batch for the curves
            if self.per_class_report:
                accumulator["score_histograms"] += compute_pixel_score_histograms(
                    class_probabilities=predictions["class_probabilities"][sample_indices],
                    masks=batch["masks"][sample_indices], number_bins=SCORE_HISTOGRAM_BINS)

        # Pixel confusion matrix and boundary counts of the labelled pixels
        for class_maps, masks in evaluated_groups:
            accumulator["confusion_matrix"] += compute_pixel_confusion_matrix(class_maps=class_maps, masks=masks,
                                                                              number_classes=number_classes)
            accumulator["boundary_counts"] += compute_boundary_counts(class_maps=class_maps, masks=masks,
                                                                      number_classes=number_classes,
                                                                      tolerance=self.boundary_tolerance)
        accumulator["number_images"] += len(sample_indices)
        if not self.save_predicted_masks:
            return

        # Predicted class index of every pixel at the evaluated resolution, as a single-channel image
        accumulator["predicted_masks_folder"].mkdir(parents=True, exist_ok=True)
        evaluated_class_maps = [class_map for class_maps, _ in evaluated_groups
                                for class_map in class_maps.cpu().numpy()]
        for class_map, sample_identifier in zip(evaluated_class_maps, sample_identifiers):
            sample_name = test_dataset.get_sample_name(sample_identifier=sample_identifier)
            cv2.imwrite(filename=str(accumulator["predicted_masks_folder"] / f"{sample_name}.png"),
                        img=class_map.astype(np.uint8))

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws a fixed test image for the sample images of the evaluation: its input, its true and predicted masks and
        its error map side by side, captioned with its pixel accuracy over its labelled pixels, the panels of a large
        test image segmented tile by tile being reduced to SAMPLE_PANEL_MAXIMUM_SIZE pixels on their longest side.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The segmenter outputs.
            sample_index (int): The index of the image in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR grid of the sample.
        """
        image = convert_image_to_array(image=batch["images"][sample_index])
        mask = batch["masks"][sample_index].cpu().numpy()
        class_map = predictions["class_probabilities"][sample_index].argmax(dim=0).cpu().numpy()

        # Share of the labelled pixels given their class
        labelled_pixels = mask != IGNORED_LABEL_INDEX
        pixel_accuracy = float((class_map[labelled_pixels] == mask[labelled_pixels]).mean()) \
            if labelled_pixels.any() else 0.0

        # Panels of a large image reduced for display, the class indices keeping their values
        reduction = SAMPLE_PANEL_MAXIMUM_SIZE / max(image.shape[:2])
        if reduction < 1.0:
            panel_size = (max(1, int(round(image.shape[1] * reduction))),
                          max(1, int(round(image.shape[0] * reduction))))
            image = cv2.resize(src=image, dsize=panel_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(src=mask.astype(np.int32), dsize=panel_size, interpolation=cv2.INTER_NEAREST)
            class_map = cv2.resize(src=class_map.astype(np.int32), dsize=panel_size, interpolation=cv2.INTER_NEAREST)
        return build_segmentation_sample_grid(
            image=image, mask=mask, class_map=class_map, class_names=self.class_names,
            class_colours=self.class_colours,
            caption=f"Pixel Accuracy {pixel_accuracy:.4f}")

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the segmentation and boundary metrics of the labelled pixels of the test split and organises them for
        the report: the headline metrics, the evaluated pixels, their resolution, the boundary tolerance and the saved
        images as notes, the per-class table (shown with the per-class report, worst IoU first) and the per-class
        scalars. The metrics of every class and the pixel confusion matrix table are saved in the evaluation folder of
        the checkpoint, '<run>/Evaluation/Iteration_<n>/', whatever the per-class report; the confusion matrix image
        only with it, also logged to TensorBoard, with the shares of every true class in its cells, the pixel counts
        being too large to print. With the per-class report, the ROC and precision-recall curves of the pixels of every
        class, one against the rest, come from the histograms of their probabilities, add the macro ROC AUC and average
        precision to the headline metrics and the ROC AUC and average precision of every class to its metrics, and are
        saved and logged to TensorBoard.

        Args:
            accumulator (Dict[str, Any]): The pixel confusion matrix, the boundary counts, with the per-class report the
                histograms of the pixel probabilities, and the predicted masks folder of the test split.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        confusion_matrix = accumulator["confusion_matrix"].cpu()
        metrics = compute_segmentation_metrics(confusion_matrix=confusion_matrix)
        metrics.update(compute_boundary_metrics(boundary_counts=accumulator["boundary_counts"].cpu()))
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        evaluation_folder.mkdir(parents=True, exist_ok=True)
        headline_metrics = {"Mean IoU": metrics["mean_iou"], "Mean Dice": metrics["mean_dice"],
                            "Mean Boundary IoU": metrics["mean_boundary_iou"],
                            "Mean Boundary F1": metrics["mean_boundary_f1"],
                            "Pixel Accuracy": metrics["pixel_accuracy"],
                            "Frequency Weighted IoU": metrics["frequency_weighted_iou"]}

        # ROC and precision-recall curves of the pixels of every class with the per-class report, from the histograms
        # of their probabilities, their macro means following the headline metrics
        class_curves = None
        if self.per_class_report:
            class_curves = compute_histogram_score_curves(score_histograms=accumulator["score_histograms"].cpu())
            headline_metrics.update(compute_macro_curve_metrics(class_curves=class_curves))

        # Metrics of every class and pixel confusion matrix, saved whatever the per-class report
        class_rows = build_segmentation_class_rows(class_names=self.class_names, metrics=metrics,
                                                   class_curves=class_curves)
        files = {"Class Metrics": write_class_metrics_file(class_rows=class_rows, evaluation_folder=evaluation_folder),
                 "Confusion Matrix Table": save_confusion_matrix_table(confusion_matrix=confusion_matrix,
                                                                       class_names=self.class_names,
                                                                       output_folder=evaluation_folder)}

        # Confusion matrix image of the per-class report, saved and logged to TensorBoard
        images = {}
        if self.per_class_report:
            files["Confusion Matrix Image"], images["Confusion Matrix"] = save_confusion_matrix_image(
                confusion_matrix=confusion_matrix, class_names=self.class_names, output_folder=evaluation_folder,
                title=f"Pixel Confusion Matrix · Iteration {iteration} · Mean IoU {metrics['mean_iou']:.4f} · Pixel "
                      f"Accuracy {metrics['pixel_accuracy']:.4f}", show_counts=False)

            # ROC and precision-recall curves of the pixels, saved and logged to TensorBoard
            curve_files, curve_images = save_score_curve_images(
                class_curves=class_curves, class_names=self.class_names, output_folder=evaluation_folder,
                title_suffix=f" · Pixels · Iteration {iteration}")
            files.update(curve_files)
            images.update(curve_images)

        # Evaluated pixels and their resolution, the boundary tolerance, and the folder of the predicted masks
        if self.original_resolution:
            evaluated_resolution = "original resolution of the images"
        elif self.tiling_configuration["activated"]:
            evaluated_resolution = f"images scaled by {self.tiling_configuration['scale']:g}, segmented tile by tile"
        else:
            evaluated_resolution = "network resolution"
        notes = [f"Labelled Pixels : {int(confusion_matrix.sum())} at the {evaluated_resolution}, the padding and the "
                 f"unlabelled pixels being ignored",
                 f"Boundary Tolerance : {self.boundary_tolerance} pixels"]
        if self.save_predicted_masks:
            notes.append(f"Predicted Masks : {accumulator['number_images']} class index images")
            files["Predicted Masks"] = accumulator["predicted_masks_folder"]

        return {"evaluated_samples": f"{accumulator['number_images']} Test {self.SAMPLE_NAME}",
                "metrics": headline_metrics, "notes": notes,
                "tables": [build_segmentation_class_table(class_rows=class_rows)], "scalars": dict(headline_metrics),
                "class_scalars": build_segmentation_class_scalars(class_rows=class_rows), "images": images,
                "files": files, "headline_metric": "Mean IoU"}

    def describe_mask_classes(self, mask: torch.Tensor) -> str:
        """
        Names the classes present in a mask, background left out, for the captions of the input viewer.

        Args:
            mask (torch.Tensor): The [H, W] class index of every pixel.

        Returns:
            str: The class names of the mask, 'background only' without any other class.
        """
        class_indices = torch.unique(input=mask[mask > 0]).tolist()
        return ", ".join(self.class_names[class_index] for class_index in class_indices) or "background only"

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays an augmented training image next to a validation image, each with its mask painted over it and an
        information panel holding the classes of the mask and the augmentations of the image.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed images in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        painted_images, panel_lines = [], []
        for batch in [training_batch, validation_batch]:
            # Paint the mask over the image so that the classes can be checked against the pixels
            image = convert_image_to_array(image=batch["images"][sample_index])
            mask = batch["masks"][sample_index].detach().cpu()
            painted_image = overlay_segmentation_mask(image=image, class_map=mask.numpy(),
                                                      class_colours=self.class_colours)
            painted_images.append(torch.from_numpy(painted_image).permute(2, 0, 1))

            # The classes of the mask, followed by the augmentations drawn for the image
            panel_lines.append([f"Classes : {self.describe_mask_classes(mask=mask)}"] +
                               self.describe_input_augmentations(batch=batch, sample_index=sample_index))
        return view_input_data(images=painted_images, bounding_boxes=None, labels=None,
                               window_titles=["Training", "Validation"], display_input_size=self.input_image_size,
                               panel_lines=panel_lines)
