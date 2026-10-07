"""
Trainer of the semantic segmentation task: the data loaders of the images and their masks, the U-Net segmenter with
the balanced class weights of its cross entropy, the augmentation of the masks with the images, the pixel accuracy and
mean IoU tracked on every batch, and the evaluation of the checkpoints on the test split (pixel confusion matrix,
global and per-class metrics, sample grids of fixed test images and the predicted masks). Registered for the
semantic_segmentation task.
"""

import json
import cv2
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from vision_training.core.contracts import (IGNORED_LABEL_INDEX, SAMPLE_IDENTIFIERS_KEY, EvaluationResults,
                                            SegmentationBatch)
from vision_training.data.loaders import get_segmentation_dataloaders
from vision_training.data.split_description import count_split_samples
from vision_training.evaluation.classification.confusion_matrix import (save_confusion_matrix_image,
                                                                        save_confusion_matrix_table)
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.evaluation.segmentation.segmentation_metrics import (compute_pixel_confusion_matrix,
                                                                          compute_segmentation_metrics)
from vision_training.evaluation.segmentation.segmentation_report import (build_segmentation_class_rows,
                                                                         build_segmentation_class_scalars,
                                                                         build_segmentation_class_table)
from vision_training.losses.classification_losses import compute_balanced_class_weights
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.os_utilities import print_color
from vision_training.utilities.visualization import (build_class_colours, build_segmentation_sample_grid,
                                                     overlay_segmentation_mask, view_input_data)

# File of the run folder caching the pixels of every class in the training masks, read back when the run resumes
CLASS_PIXEL_COUNTS_FILE_NAME = "class_pixel_counts.json"


class SegmentationTrainer(BaseTrainer):
    """
    Trainer of the semantic segmentation task: the U-Net segmenter trained on images and class-index masks (Native
    loader), the masks following every geometric augmentation of the images, with its cross entropy and Dice losses,
    pixel accuracy and mean IoU tracked on every batch, and every evaluated checkpoint scored at the network resolution
    on the labelled pixels of the test split: pixel accuracy, mean IoU, frequency-weighted IoU and mean Dice, the IoU,
    Dice, precision and recall of every class, the pixel confusion matrix, the sample grids of a few fixed test images
    and, when asked, the predicted mask of every test image.
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
        # Losses of the model, the weighting of the classes in the cross entropy, and the evaluation settings: the
        # number of fixed test images drawn at every evaluation and whether the predicted masks are saved
        self.loss_configuration = model_configuration["SegmentationLoss"]
        self.number_sample_images = experiment_configuration["evaluation"]["number_sample_images"]
        self.save_predicted_masks = experiment_configuration["evaluation"]["save_predicted_masks"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the segmentation loaders of the three splits, takes the class names from the categories of the training
        manifest with a colour per class, and chooses the test images shown at every evaluation: spread evenly over the
        test images sorted by identifier, so that they stay the same between the evaluations and the resumptions of a
        run whatever the shuffling of the loaders.

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

        # Fixed test images of the sample grids, evenly spread over the sorted identifiers
        test_identifiers = sorted(sample["identifier"] for sample in loaders[2].dataset.samples)
        number_sample_images = min(self.number_sample_images, len(test_identifiers))
        chosen_positions = np.linspace(start=0, stop=len(test_identifiers) - 1, num=number_sample_images).round()
        self.sample_image_identifiers = {test_identifiers[int(position)] for position in chosen_positions}
        return loaders

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
                print_color(color="blue", output=f"Class Pixel Counts Read From : file://{counts_path.absolute()}")
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
                                                   dtype=self.dtype, verbose=self.diagnosis_modes["verbose"])
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
        Lists the losses returned by the segmenter (the cross entropy and the Dice loss whose weight is not 0), the
        total loss, and the pixel accuracy and mean IoU tracked on every training and validation batch.

        Returns:
            Dict[str, str]: The display name of every tracked metric.
        """
        tracked_metrics = {"total_loss": "Total Loss"}
        if self.loss_configuration["cross_entropy_weight"] > 0:
            tracked_metrics["segmentation_loss"] = "Cross Entropy Loss"
        if self.loss_configuration["dice_weight"] > 0:
            tracked_metrics["dice_loss"] = "Dice Loss"
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
            Dict[str, Any]: The batch with its augmented images and masks.
        """
        augmented_images, _, _, augmented_masks = self.data_augmenter.augment_batch(images=batch["images"],
                                                                                    masks=batch["masks"])
        return {**batch, "images": augmented_images, "masks": augmented_masks}

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
        Runs the segmenter without masks, as in deployment.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The class logits and probabilities of every pixel.
        """
        return self.model(input_tensor=batch["images"])

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
        Creates the pixel confusion matrix filled on the device during the evaluation of the test split, with the
        folders of the sample grids and of the predicted masks of the checkpoint.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, which names its evaluation folder.

        Returns:
            Dict[str, Any]: The empty [K, K] 'confusion_matrix', the 'number_images' evaluated, the 'iteration', the
                'samples_folder' and its empty 'sample_images' (RGB grids keyed by TensorBoard tag), and the
                'predicted_masks_folder'.
        """
        evaluation_folder = self.experiment_folder / "Evaluation" / f"Iteration_{iteration}"
        number_classes = len(self.class_names)
        return {"confusion_matrix": torch.zeros(size=(number_classes, number_classes), dtype=torch.int64,
                                                device=self.device),
                "number_images": 0, "iteration": iteration, "samples_folder": evaluation_folder / "Samples",
                "sample_images": {}, "predicted_masks_folder": evaluation_folder / "Predicted_Masks"}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Adds the labelled pixels of the kept images of a test batch to the confusion matrix, on the device, then, while
        the images are still in the batch, saves the sample grid of the fixed test images and, when asked, the
        predicted mask of every image.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, modified in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The segmenter outputs.
            sample_indices (List[int]): The indices of the images to keep.
        """
        class_maps = predictions["class_logits"].argmax(dim=1)[sample_indices]
        masks = batch["masks"][sample_indices]
        accumulator["confusion_matrix"] += compute_pixel_confusion_matrix(class_maps=class_maps, masks=masks,
                                                                          number_classes=len(self.class_names))
        accumulator["number_images"] += len(sample_indices)

        # Images of the batch written to the evaluation folder: the fixed sample images, and every predicted mask
        test_dataset = self.test_dataloader.dataset
        sample_identifiers = batch[SAMPLE_IDENTIFIERS_KEY][sample_indices].tolist()
        written_positions = [position for position, sample_identifier in enumerate(sample_identifiers)
                             if self.save_predicted_masks or sample_identifier in self.sample_image_identifiers]
        if not written_positions:
            return
        class_maps, masks = class_maps.cpu().numpy(), masks.cpu().numpy()
        for position in written_positions:
            sample_name = test_dataset.get_sample_name(sample_identifier=sample_identifiers[position])

            # Predicted class index of every pixel at the network resolution, as a single-channel image
            if self.save_predicted_masks:
                accumulator["predicted_masks_folder"].mkdir(parents=True, exist_ok=True)
                cv2.imwrite(filename=str(accumulator["predicted_masks_folder"] / f"{sample_name}.png"),
                            img=class_maps[position].astype(np.uint8))

            # Grid of the input, the true and predicted masks and the error map, captioned with its pixel accuracy
            if sample_identifiers[position] in self.sample_image_identifiers:
                self.save_sample_grid(accumulator=accumulator, sample_name=sample_name,
                                      image=batch["images"][sample_indices[position]], mask=masks[position],
                                      class_map=class_maps[position])

    def save_sample_grid(self, accumulator: Dict[str, Any], sample_name: str, image: torch.Tensor, mask: np.ndarray,
                         class_map: np.ndarray) -> None:
        """
        Saves the sample grid of a fixed test image in the 'Samples' folder of the evaluated checkpoint, and keeps it
        in RGB for TensorBoard, so that the predictions of the same images can be followed over the training.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, holding the 'samples_folder', the 'iteration' and
                the 'sample_images', extended in place.
            sample_name (str): The name of the test image.
            image (torch.Tensor): The [3, H, W] network input in BGR.
            mask (np.ndarray): The [H, W] true class index of every pixel, IGNORED_LABEL_INDEX on the ignored ones.
            class_map (np.ndarray): The [H, W] predicted class index of every pixel.
        """
        image_numpy = np.ascontiguousarray(image.detach().cpu().clamp(min=0.0, max=255.0).to(dtype=torch.uint8)
                                           .permute(1, 2, 0).numpy())
        labelled_pixels = mask != IGNORED_LABEL_INDEX
        pixel_accuracy = float((class_map[labelled_pixels] == mask[labelled_pixels]).mean()) \
            if labelled_pixels.any() else 0.0
        grid = build_segmentation_sample_grid(
            image=image_numpy, mask=mask, class_map=class_map, class_names=self.class_names,
            class_colours=self.class_colours,
            caption=f"Iteration {accumulator['iteration']} | {sample_name} | Pixel Accuracy {pixel_accuracy:.4f}")
        accumulator["samples_folder"].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filename=str(accumulator["samples_folder"] / f"{sample_name}.png"), img=grid)
        accumulator["sample_images"][f"Samples/{sample_name}"] = cv2.cvtColor(src=grid, code=cv2.COLOR_BGR2RGB)

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the segmentation metrics of the labelled pixels of the test split and organises them for the report:
        the headline metrics, the evaluated pixels and the saved images as notes, the per-class table (shown with the
        per-class report, worst IoU first) and the per-class scalars. The metrics of every class and the pixel
        confusion matrix table are saved in the evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/',
        whatever the per-class report; the confusion matrix image only with it, also logged to TensorBoard, with the
        shares of every true class in its cells, the pixel counts being too large to print. The sample grids are
        reported at every evaluation.

        Args:
            accumulator (Dict[str, Any]): The pixel confusion matrix and the saved images of the test split.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        confusion_matrix = accumulator["confusion_matrix"].cpu()
        metrics = compute_segmentation_metrics(confusion_matrix=confusion_matrix)
        evaluation_folder = self.experiment_folder / "Evaluation" / f"Iteration_{iteration}"
        evaluation_folder.mkdir(parents=True, exist_ok=True)
        headline_metrics = {"Mean IoU": metrics["mean_iou"], "Mean Dice": metrics["mean_dice"],
                            "Pixel Accuracy": metrics["pixel_accuracy"],
                            "Frequency Weighted IoU": metrics["frequency_weighted_iou"]}

        # Metrics of every class and pixel confusion matrix, saved whatever the per-class report
        class_rows = build_segmentation_class_rows(class_names=self.class_names, metrics=metrics)
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

        # Evaluated pixels, and the folders of the saved images
        notes = [f"Labelled Pixels : {int(confusion_matrix.sum())}, the padding and the unlabelled pixels being "
                 f"ignored"]
        if accumulator["sample_images"]:
            notes.append(f"Sample Images : {len(accumulator['sample_images'])} fixed test images")
            files["Sample Images"] = accumulator["samples_folder"]
        if self.save_predicted_masks:
            notes.append(f"Predicted Masks : {accumulator['number_images']} class index images")
            files["Predicted Masks"] = accumulator["predicted_masks_folder"]

        return {"evaluated_samples": f"{accumulator['number_images']} Test {self.SAMPLE_NAME}",
                "metrics": headline_metrics, "notes": notes,
                "tables": [build_segmentation_class_table(class_rows=class_rows)], "scalars": dict(headline_metrics),
                "class_scalars": build_segmentation_class_scalars(class_rows=class_rows), "images": images,
                "sample_images": accumulator["sample_images"], "files": files, "headline_metric": "Mean IoU"}

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
        Displays an augmented training image next to a validation image, each with its mask painted over it and
        captioned with the classes it holds.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed images in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        painted_images, captions = [], []
        for batch in [training_batch, validation_batch]:
            image = np.ascontiguousarray(batch["images"][sample_index].detach().cpu().clamp(min=0.0, max=255.0)
                                         .to(dtype=torch.uint8).permute(1, 2, 0).numpy())
            mask = batch["masks"][sample_index].detach().cpu()
            painted_image = overlay_segmentation_mask(image=image, class_map=mask.numpy(),
                                                      class_colours=self.class_colours)
            painted_images.append(torch.from_numpy(painted_image).permute(2, 0, 1))
            captions.append(f"Classes : {self.describe_mask_classes(mask=mask)}")
        return view_input_data(images=painted_images, bounding_boxes=None, labels=None,
                               window_titles=["Training", "Validation"], display_input_size=self.input_image_size,
                               captions=captions)
