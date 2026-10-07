"""
Trainer of the object detection task: the COCO-format loaders (DALI or Native), the two-stage detector with its four
losses, the geometric augmentations moving the bounding boxes, and the evaluation of the checkpoints on the test split
with the mean average precision and the average precision of every class, and the sample images of fixed test images
with their true and detected boxes. Registered for the object detection task in the task registry.
"""

import json
import numpy as np
import torch

from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from vision_training.core.contracts import DetectionBatch, EvaluationResults
from vision_training.data.label_mapping import get_class_names
from vision_training.data.loaders import get_detection_dataloaders
from vision_training.data.split_description import count_split_images_and_annotations
from vision_training.evaluation.detection.average_precision_report import (build_class_average_precision_rows,
                                                                           build_class_average_precision_table,
                                                                           build_mean_average_precision_table)
from vision_training.evaluation.detection.mean_average_precision import AVERAGE_PRECISION_METHODS, evaluate_detections
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.visualization import build_class_colours, build_detection_sample_grid, view_input_data


class DetectionTrainer(BaseTrainer):
    """
    Trainer of the object detection task: the two-stage detector trained on COCO-format datasets (DALI or Native
    loaders), with its four losses tracked, bounding boxes moved by the geometric augmentations, and every evaluated
    checkpoint scored with the mean average precision on the test split.
    """

    TASK_NAME = "object_detection"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the detection evaluation settings, then prepares the training run through the general trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, RegionProposal, Anchors,
                Detector).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Score threshold of the detections kept for the mAP, lower than the deployment one so that the
        # precision-recall curves keep their low-confidence end
        self.evaluation_score_threshold = experiment_configuration["evaluation"]["score_threshold"]

        # Score threshold of the detections drawn on the sample images, the deployment one, so that the images show
        # what the deployed detector returns
        self.sample_score_threshold = model_configuration["Detector"]["inference"]["score_threshold"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the COCO detection loaders (DALI or Native, depending on the configuration) of the three splits, names
        the classes after the categories of the test split for the per-class report, with a colour per class for the
        sample images, and keeps the file name of every test image, which names the sample images whatever the loader
        (the DALI loader has no dataset to ask).

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders.
        """
        test_labels_path = Path(self.experiment_configuration["test_data_folder"]) / "labels.json"
        with open(file=test_labels_path, mode="r") as labels_file:
            test_labels = json.load(fp=labels_file)
        self.class_names = get_class_names(categories=test_labels["categories"], label_dictionary=self.label_dictionary)
        self.class_colours = build_class_colours(number_classes=len(self.class_names))
        self.test_image_file_names = {image["id"]: image["file_name"] for image in test_labels["images"]}
        return get_detection_dataloaders(experiment_configuration=self.experiment_configuration,
                                         model_configuration=self.model_configuration, batch_size=self.batch_size)

    def build_model(self) -> TwoStageDetector:
        """
        Builds the two-stage detector from the model configuration.

        Returns:
            TwoStageDetector: The untrained detector.
        """
        return TwoStageDetector(configuration=self.model_configuration, device=self.device, dtype=self.dtype,
                                verbose=self.diagnosis_modes["verbose"])

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

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the images and annotations of a COCO split from its labels.json.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of images and their description with the number of annotations.
        """
        number_images, number_annotations = count_split_images_and_annotations(split_data_folder=split_data_folder)
        return number_images, f"{number_images} images, {number_annotations} annotations"

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
            Dict[str, Any]: The batch with its augmented images, boxes and labels.
        """
        images, bounding_boxes, labels, _ = self.data_augmenter.augment_batch(images=batch["images"],
                                                                              bounding_boxes=batch["bounding_boxes"],
                                                                              labels=batch["labels"])
        return {**batch, "images": images, "bounding_boxes": bounding_boxes, "labels": labels}

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
        Creates the lists of detections and ground truths filled during the evaluation of the test split.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, unused by the detection task.

        Returns:
            Dict[str, Any]: The empty 'detections' and 'ground_truths' lists.
        """
        return {"detections": [], "ground_truths": []}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the valid detections and the ground truth of every kept image of a test batch on the CPU, both expressed
        in network input pixels.

        Args:
            accumulator (Dict[str, Any]): The 'detections' and 'ground_truths' lists, extended in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The detector outputs.
            sample_indices (List[int]): The indices of the images to keep.
        """
        for image_index in sample_indices:
            number_detections = int(predictions["number_detections"][image_index])
            accumulator["detections"].append({
                "boxes": predictions["final_detection_boxes"][image_index, :number_detections].cpu(),
                "scores": predictions["final_detection_scores"][image_index, :number_detections].cpu(),
                "labels": predictions["final_detection_labels"][image_index, :number_detections].cpu()})
            accumulator["ground_truths"].append({"boxes": batch["bounding_boxes"][image_index].cpu(),
                                                 "labels": batch["labels"][image_index].cpu()})

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the mean average precision of the test split with the three average precision methods (the area
        method being the headline one, its mAP@0.5 comparing the checkpoints) and the average precisions of every class,
        and organises them for the report: the table of the methods, always shown, and the per-class table (shown with
        the per-class report, worst AP@0.5 first), with the same values as TensorBoard scalars (the AP@0.5 of every
        class only with the per-class report). The average precisions of every class are saved to class_metrics.csv
        in the evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/', whatever the per-class report.

        Args:
            accumulator (Dict[str, Any]): The detections and ground truths of the test split.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        results = evaluate_detections(detections=accumulator["detections"], ground_truths=accumulator["ground_truths"],
                                      number_classes=len(self.label_dictionary))

        # Average precisions of every class, saved whatever the per-class report
        evaluation_folder = self.experiment_folder / "Evaluation" / f"Iteration_{iteration}"
        class_rows = build_class_average_precision_rows(results=results, class_names=self.class_names)
        files = {"Class Metrics": write_class_metrics_file(class_rows=class_rows, evaluation_folder=evaluation_folder)}

        # The mean average precisions of every method as TensorBoard scalars, to follow them over the training, and the
        # AP@0.5 of every class for the per-class report
        scalars = {}
        for method in AVERAGE_PRECISION_METHODS:
            for metric_name in ["mAP@0.5", "mAP@0.75", "mAP@[0.5:0.95]"]:
                scalars[f"{metric_name}/{method}"] = results[method][metric_name]
        class_scalars = {f"AP@0.5/{self.class_names[class_index]}": average_precision
                         for class_index, average_precision in results["area"]["class_AP@0.5"].items()}

        return {"evaluated_samples": f"{len(accumulator['detections'])} Test Images", "metrics": {}, "notes": [],
                "tables": [build_mean_average_precision_table(results=results),
                           build_class_average_precision_table(class_rows=class_rows)],
                "scalars": scalars, "class_scalars": class_scalars, "images": {}, "files": files,
                "headline_metric": "mAP@0.5/area"}

    def get_test_sample_classes(self) -> Dict[int, List[int]]:
        """
        Lists the test images without their classes, so that the fixed test images drawn at every evaluation are
        evenly spread over the test split.

        Returns:
            Dict[int, List[int]]: An empty list for every COCO identifier of the test images.
        """
        return {image_identifier: [] for image_identifier in self.test_image_file_names}

    def get_test_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of a test image from the test labels.json, the DALI loader having no dataset: its file
        name without extension, its folders joined with underscores.

        Args:
            sample_identifier (int): The COCO identifier of the image, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the image, e.g. 'image_000012' for 'image_000012.png'.
        """
        return str(PurePosixPath(self.test_image_file_names[sample_identifier]).with_suffix("")).replace("/", "_")

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int,
                           sample_name: str, iteration: int) -> np.ndarray:
        """
        Draws a fixed test image for the sample images of the evaluation: its ground truth boxes next to its
        detections whose score reaches the deployment score threshold ('Detector.inference.score_threshold'), the
        evaluation keeping lower scores for the mAP, under a caption counting both.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The detector outputs, with the final detections of every image.
            sample_index (int): The index of the image in the batch.
            sample_name (str): The name of the image.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sample.
        """
        image = np.ascontiguousarray(batch["images"][sample_index].detach().cpu().clamp(min=0.0, max=255.0)
                                     .to(dtype=torch.uint8).permute(1, 2, 0).numpy())

        # Valid detections of the image, kept from the deployment score threshold
        number_detections = int(predictions["number_detections"][sample_index])
        scores = predictions["final_detection_scores"][sample_index, :number_detections].detach().float().cpu().numpy()
        kept_detections = scores >= self.sample_score_threshold
        predicted_boxes = predictions["final_detection_boxes"][sample_index, :number_detections].detach().float()
        predicted_labels = predictions["final_detection_labels"][sample_index, :number_detections].detach()
        ground_truth_boxes = batch["bounding_boxes"][sample_index].detach().float().cpu().numpy()

        return build_detection_sample_grid(
            image=image, ground_truth_boxes=ground_truth_boxes,
            ground_truth_labels=batch["labels"][sample_index].detach().cpu().numpy(),
            predicted_boxes=predicted_boxes.cpu().numpy()[kept_detections],
            predicted_labels=predicted_labels.cpu().numpy()[kept_detections], predicted_scores=scores[kept_detections],
            class_names=self.class_names, class_colours=self.class_colours,
            caption=f"Iteration {iteration} | {sample_name} | {len(ground_truth_boxes)} ground truth boxes | "
                    f"{int(kept_detections.sum())} detections with score >= {self.sample_score_threshold:.2f}")

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays an augmented training image next to a validation image, each with its ground truth boxes captioned
        with their class names.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed images in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        # Invert the label dictionary so that each box is captioned with its class name instead of its index
        label_names = {class_index: str(class_name) for class_name, class_index in self.label_dictionary.items()}
        return view_input_data(
            images=[training_batch["images"][sample_index], validation_batch["images"][sample_index]],
            bounding_boxes=[training_batch["bounding_boxes"][sample_index],
                            validation_batch["bounding_boxes"][sample_index]],
            labels=[training_batch["labels"][sample_index], validation_batch["labels"][sample_index]],
            window_titles=["Training", "Validation"], display_input_size=self.input_image_size,
            label_names=label_names)
