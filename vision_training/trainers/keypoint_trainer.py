"""
Trainer of the keypoint detection task, bottom-up (HigherHRNet): every sample is one whole image of a COCO-format split
with the boxes, classes and keypoints of all its objects, given by the keypoint loaders and augmented by the
DataAugmenter, every geometric type moving the keypoints without renaming them. The keypoint detector predicts on the
whole image one heatmap and one tag map per keypoint of every class, trained with the heatmap and tag losses on targets
built on the device, and its keypoint candidates are grouped into objects by their tags
(models/keypoints/keypoint_grouping.py). The checkpoints are evaluated on the test split with the average precisions on
the object keypoint similarity (OKS) of the objects found, and the share of matched objects, the PCK, the mean OKS and
the mean pixel error of every class and every keypoint, with sample images of fixed test images showing their true and
predicted objects. Registered for the keypoint detection task in the task registry.
"""

import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision_training.core.contracts import EvaluationResults, KeypointBatch
from vision_training.data.detection.coco_dataset import IGNORED_OBJECT_KEYS
from vision_training.data.loaders import get_keypoint_dataloaders
from vision_training.evaluation.detection.average_precision_report import summarise_average_precisions
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.evaluation.keypoints.object_keypoint_similarity import (build_keypoint_sigma_table,
                                                                             evaluate_image_keypoints,
                                                                             summarise_keypoint_accuracy)
from vision_training.models.keypoints.keypoint_detector import KeypointDetector, build_keypoint_channel_table
from vision_training.models.keypoints.keypoint_grouping import group_keypoint_candidates
from vision_training.trainers.coco_split_trainer import CocoSplitTrainer
from vision_training.utilities.visualization import (ViewerWindow, build_keypoint_sample_grid,
                                                     convert_image_to_array, draw_keypoint_objects)

# File of the accuracy of every keypoint of every class, in the evaluation folder of every checkpoint
KEYPOINT_METRICS_FILE_NAME = "keypoint_metrics.csv"

# Headline metric comparing the checkpoints, the mAP@0.5 on the object keypoint similarity with the area method
KEYPOINT_HEADLINE_METRIC = "OKS mAP@0.5/area"


class KeypointTrainer(CocoSplitTrainer):
    """
    Trainer of the keypoint detection task: the bottom-up keypoint detector trained on the whole images of COCO-format
    datasets with keypoints, its heatmap and tag losses tracked, and every evaluated checkpoint scored with the mean
    average precision on the object keypoint similarity of the objects it finds in the test images.
    """

    TASK_NAME = "keypoint_detection"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the keypoint definition of every class and the grouping settings of the keypoint candidates, builds the
        channel table of the grouping and the sigmas of the object keypoint similarity, then prepares the training run
        through the general trainer, which builds the tables again for the classes of a resumed run.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data with its injected
                'keypoint_categories', Backbone, HRNet, KeypointHead).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Keypoint tables of the classes of the mapping file, needed before the general trainer starts
        self.build_keypoint_tables(experiment_configuration=experiment_configuration,
                                   model_configuration=model_configuration)

        # Grouping of the candidates into objects: the evaluation keeps every object for the precision-recall curves,
        # the sample images only those reaching the object score threshold of the deployment
        inference_configuration = model_configuration["KeypointHead"]["inference"]
        self.keypoint_threshold = inference_configuration["keypoint_threshold"]
        self.tag_threshold = inference_configuration["tag_threshold"]
        self.object_score_threshold = inference_configuration["object_score_threshold"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_keypoint_tables(self, experiment_configuration: Dict[str, Any],
                              model_configuration: Dict[str, Any]) -> None:
        """
        Builds the keypoint tables of the evaluation and the sample images from the keypoint definitions of the
        classes: the definitions themselves (keypoint names and skeletons, which name the per-keypoint metrics and draw
        the objects), the heatmap channel of every keypoint of every class, which the grouping reads the candidates
        with, and the sigmas of the object keypoint similarity. Called before the general trainer starts, then again
        once a resumed run has restored its own classes, so that the grouping and the scores follow the classes the
        model and the loaders use.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section, giving the
                keypoint sigmas.
            model_configuration (Dict[str, Any]): The processed model sections, whose 'Data.keypoint_categories' give
                the keypoint definitions indexed by class.
        """
        # Keypoint definition of every class, injected by the configuration loader from the training labels.json, with
        # the heatmap channel of every keypoint of every class
        self.keypoint_categories = {int(class_index): definition for class_index, definition
                                    in model_configuration["Data"]["keypoint_categories"].items()}
        self.maximum_keypoints = max(len(definition["keypoints"]) for definition in self.keypoint_categories.values())
        channel_table, _ = build_keypoint_channel_table(keypoint_categories=self.keypoint_categories)
        self.channel_table = channel_table.numpy()

        # Tolerance of the object keypoint similarity of every keypoint of every class
        evaluation_configuration = experiment_configuration["evaluation"]
        self.sigma_table = build_keypoint_sigma_table(keypoint_categories=self.keypoint_categories,
                                                      default_sigma=evaluation_configuration["default_keypoint_sigma"],
                                                      keypoint_sigmas=evaluation_configuration["keypoint_sigmas"],
                                                      maximum_keypoints=self.maximum_keypoints)

    def restore_run_label_dictionary(self) -> None:
        """
        Keeps the classes of a resumed run through the general trainer, whose restore reads the keypoint definitions
        again for the restored classes, then builds the keypoint tables again from them: built from the mapping file
        before the general trainer started, they would otherwise read the heatmap channels and name the keypoints of
        other classes than the model and the loaders.
        """
        super().restore_run_label_dictionary()
        self.build_keypoint_tables(experiment_configuration=self.experiment_configuration,
                                   model_configuration=self.model_configuration)

    def build_dataloaders(self) -> Tuple[Iterable, Optional[Iterable], Iterable]:
        """
        Builds the keypoint loaders of the three splits, one whole image per sample, names the classes after the
        categories of the test split for the per-class report, with a colour per class for the sample images, and
        keeps the file name of every test image, which names the sample images.

        Returns:
            Tuple[Iterable, Optional[Iterable], Iterable]: The training, validation (None when it is not
                built) and test loaders.
        """
        # Class names, class colours and test image names from the test manifest, then the loaders of the three splits
        self.read_test_manifest()
        return get_keypoint_dataloaders(experiment_configuration=self.experiment_configuration,
                                        model_configuration=self.model_configuration, batch_size=self.batch_size,
                                        build_validation_loader=self.builds_validation_loader)

    def build_model(self) -> KeypointDetector:
        """
        Builds the bottom-up keypoint detector from the model configuration.

        Returns:
            KeypointDetector: The untrained keypoint detector.
        """
        return KeypointDetector(configuration=self.model_configuration, device=self.device, dtype=self.dtype)

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the total loss and the three losses of the keypoint detector, tracked on every training and validation
        batch.

        Returns:
            Dict[str, str]: The display name of every tracked loss.
        """
        return {"total_loss": "Total Loss", "heatmap_loss": "Heatmap Loss", "tag_pull_loss": "Tag Pull Loss",
                "tag_push_loss": "Tag Push Loss"}

    def prepare_batch(self, batch: Dict[str, Any]) -> KeypointBatch:
        """
        Moves the images (in the precision of the model) and the per-image boxes, classes and keypoints of a keypoint
        batch to the device, the targets of the losses being built from them by the model, with the ignored regions
        of the training and validation batches when the loaders give them ('KeypointHead.ignore_unlabelled_objects'),
        and the objects without labelled keypoint of the test batches, left on the CPU for the evaluation.

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a keypoint loader.

        Returns:
            KeypointBatch: The batch ready for the keypoint detector.
        """
        prepared_batch = {"images": batch["images"].to(device=self.device, dtype=self.dtype),
                          "bounding_boxes": [box_tensor.to(device=self.device, dtype=torch.float32)
                                             for box_tensor in batch["bounding_boxes"]],
                          "labels": [label_tensor.to(device=self.device, dtype=torch.int64)
                                     for label_tensor in batch["labels"]],
                          "keypoints": [keypoint_tensor.to(device=self.device, dtype=torch.float32)
                                        for keypoint_tensor in batch["keypoints"]],
                          "sample_identifiers": batch["sample_identifiers"]}

        # The binary regions of the objects without labelled keypoint, kept in uint8 like the instance masks
        if "ignored_regions" in batch:
            prepared_batch["ignored_regions"] = batch["ignored_regions"].to(device=self.device)

        # The boxes, classes and crowd flags of the same objects in the test batches, only read on the CPU
        for key in IGNORED_OBJECT_KEYS:
            if key in batch:
                prepared_batch[key] = batch[key]
        return prepared_batch

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the training images, every geometric type moving the boxes and the keypoints of their objects
        without renaming them, the keypoints leaving the image becoming unlabelled and an object being dropped with its
        box and label when less than MINIMUM_VISIBLE_AREA_FRACTION of its box and none of its labelled keypoints stay
        inside the image. The ignored regions, when the batch has them, move with the image as the binary mask of the
        augmenter, the uncovered border being kept in the loss.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The batch with its augmented images, boxes, labels, keypoints and ignored regions, and the
                'augmentation_reports' of its images for the input viewer.
        """
        ignored_regions = batch["ignored_regions"] if "ignored_regions" in batch else None
        images, bounding_boxes, labels, ignored_regions, _, keypoints, augmentation_reports = \
            self.data_augmenter.augment_batch(images=batch["images"], bounding_boxes=batch["bounding_boxes"],
                                              labels=batch["labels"], masks=ignored_regions,
                                              keypoints=batch["keypoints"])
        augmented_batch = {**batch, "images": images, "bounding_boxes": bounding_boxes, "labels": labels,
                           "keypoints": keypoints, "augmentation_reports": augmentation_reports}
        if ignored_regions is not None:
            augmented_batch["ignored_regions"] = ignored_regions
        return augmented_batch

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the keypoint detector with the keypoints and classes of the objects of every image, so that it builds its
        heatmap and tag targets and returns its three losses, the heatmap loss leaving out the ignored regions when the
        batch has them.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The detector outputs, with its 'heatmap_loss', 'tag_pull_loss' and 'tag_push_loss'.
        """
        return self.model(input_tensor=batch["images"], ground_truth_keypoints=batch["keypoints"],
                          ground_truth_labels=batch["labels"],
                          ignored_regions=batch["ignored_regions"] if "ignored_regions" in batch else None)

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the keypoint detector without ground truth, as in deployment, and groups the keypoint candidates of every
        image into objects (group_keypoint_candidates) with the grouping settings of 'KeypointHead.inference', every
        object being kept whatever its score so that the precision-recall curves keep their low-confidence end.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The detector outputs, with the 'grouped_objects' of every image ('labels', 'keypoints' in
                network input pixels, 'keypoint_scores' and 'scores', by decreasing score).
        """
        return self.group_candidates(predictions=self.model(input_tensor=batch["images"]))

    def group_candidates(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Groups the keypoint candidates of every image of a batch into objects on the CPU (group_keypoint_candidates),
        with the grouping settings of 'KeypointHead.inference' and every object kept whatever its score, for the test
        evaluation of the trained detector (predict_batch) and of its exported models, whose candidates are the same
        outputs (the test split parity of deployment/export_parity.py).

        Args:
            predictions (Dict[str, Any]): The outputs of the detector, with its [B, C, N, 2] 'candidate_positions',
                [B, C, N] 'candidate_scores' and [B, C, N] 'candidate_tags'.

        Returns:
            Dict[str, Any]: The outputs, with the 'grouped_objects' of every image ('labels', 'keypoints' in network
                input pixels, 'keypoint_scores' and 'scores', by decreasing score).
        """
        candidate_positions = predictions["candidate_positions"].detach().float().cpu().numpy()
        candidate_scores = predictions["candidate_scores"].detach().float().cpu().numpy()
        candidate_tags = predictions["candidate_tags"].detach().float().cpu().numpy()

        # Candidates of every image grouped into objects on the CPU
        grouped_objects = [group_keypoint_candidates(
            candidate_positions=candidate_positions[image_index], candidate_scores=candidate_scores[image_index],
            candidate_tags=candidate_tags[image_index], channel_table=self.channel_table,
            keypoint_threshold=self.keypoint_threshold, tag_threshold=self.tag_threshold, object_score_threshold=0.0)
            for image_index in range(candidate_positions.shape[0])]
        return {**predictions, "grouped_objects": grouped_objects}

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the list of the evaluated test images, filled batch by batch on the CPU.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint (unused).

        Returns:
            Dict[str, Any]: The empty 'images' list.
        """
        return {"images": []}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the ground truth objects of every kept image of a test batch on the CPU, with its objects without
        labelled keypoint, which the evaluation ignores, and the objects found in it, all in network input pixels, for
        evaluate_image_keypoints.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, its 'images' list extended in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The outputs of predict_batch.
            sample_indices (List[int]): The indices of the images to keep.
        """
        for image_index in sample_indices:
            ground_truth_keypoints = batch["keypoints"][image_index].detach().cpu()
            grouped_objects = predictions["grouped_objects"][image_index]
            image_height, image_width = batch["images"][image_index].shape[-2:]
            accumulator["images"].append({
                "labels": batch["labels"][image_index].detach().cpu(), "image_area": float(image_height * image_width),
                "bounding_boxes": batch["bounding_boxes"][image_index].detach().cpu(),
                "keypoints": ground_truth_keypoints[..., :2], "visibilities": ground_truth_keypoints[..., 2].long(),
                **{key: batch[key][image_index] for key in IGNORED_OBJECT_KEYS},
                "predicted_labels": torch.from_numpy(grouped_objects["labels"]),
                "predicted_keypoints": torch.from_numpy(grouped_objects["keypoints"]),
                "predicted_scores": torch.from_numpy(grouped_objects["scores"])})

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the average precisions of the test split on the object keypoint similarity of the objects found with
        the three average precision methods (the area method being the headline one, its mAP@0.5 comparing the
        checkpoints), of every object size and of every class, with the mean average recalls, and the keypoint accuracy
        of the ground truth objects matched to the objects found (matched objects, PCK, mean OKS and mean pixel error)
        of all the keypoints, of every class and of every keypoint of every class. The tables of the methods, object
        sizes and recalls are always shown, the per-class and per-keypoint tables with the per-class report. The class
        rows of both are saved to class_metrics.csv and the keypoint rows to keypoint_metrics.csv in the evaluation
        folder of the checkpoint, '<run>/[Checkpoint_]Evaluation/Iteration_<n>/'. The report writes the rates as
        percentages (the matched objects, PCK and mean OKS of the grid, the average precisions, recalls and keypoint
        accuracies of the tables, and the headline OKS mAP@0.5 in the best checkpoint line of the evaluation of several
        checkpoints), the mean pixel error keeping its 4 decimals.

        Args:
            accumulator (Dict[str, Any]): The evaluated test images.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.

        Raises:
            ValueError: If the classes of the average precisions and of the keypoint accuracy differ.
        """
        # Objects found compared with the ground truth objects of every image, then the average precisions on the
        # object keypoint similarity and the keypoint accuracy of the matched objects
        image_evaluation = evaluate_image_keypoints(images=accumulator["images"], sigma_table=self.sigma_table)
        average_precision_summary = summarise_average_precisions(
            detections=image_evaluation["detections"], ground_truths=image_evaluation["ground_truths"],
            number_classes=len(self.class_names), class_names=self.class_names, overlap_type="keypoints",
            title_prefix="OKS", name_prefix="OKS ", column_prefix="oks_", image_side=self.input_image_size)
        accuracy_summary = summarise_keypoint_accuracy(
            objects=image_evaluation["objects"], sigma_table=self.sigma_table,
            keypoint_categories=self.keypoint_categories, class_names=self.class_names)

        # Average precisions and keypoint accuracy of every class in one row per class, then the keypoint rows, saved
        # whatever the per-class report
        class_rows = []
        for precision_row, accuracy_row in zip(average_precision_summary["class_rows"], accuracy_summary["class_rows"],
                                               strict=True):
            if precision_row["class_index"] != accuracy_row["class_index"]:
                raise ValueError(f"The class {precision_row['class_index']} of the average precisions meets the class "
                                 f"{accuracy_row['class_index']} of the keypoint accuracy, both must list the same "
                                 f"classes in the same order.")
            class_rows.append({**precision_row, **accuracy_row})
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        files = {"Class Metrics": write_class_metrics_file(class_rows=class_rows, evaluation_folder=evaluation_folder),
                 "Keypoint Metrics": write_class_metrics_file(class_rows=accuracy_summary["keypoint_rows"],
                                                              evaluation_folder=evaluation_folder,
                                                              file_name=KEYPOINT_METRICS_FILE_NAME)}

        # Tables of the methods, object sizes and recalls first, then the per-class (with the per-class recall) and
        # per-keypoint tables
        tables = [average_precision_summary["tables"]["methods"], average_precision_summary["tables"]["sizes"],
                  average_precision_summary["tables"]["recalls"], average_precision_summary["tables"]["classes"],
                  average_precision_summary["tables"]["class_recalls"], accuracy_summary["tables"]["classes"],
                  accuracy_summary["tables"]["keypoints"]]
        return {"evaluated_samples": f"{len(accumulator['images'])} Test {self.SAMPLE_NAME}",
                "metrics": accuracy_summary["metrics"], "notes": [], "tables": tables,
                "scalars": {**average_precision_summary["scalars"], **accuracy_summary["scalars"]},
                "class_scalars": average_precision_summary["class_scalars"], "images": {}, "files": files,
                "headline_metric": KEYPOINT_HEADLINE_METRIC,
                "percent_metrics": accuracy_summary["percent_metrics"] + [KEYPOINT_HEADLINE_METRIC]}

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws a fixed test image for the sample images of the evaluation: its ground truth objects next to the objects
        found whose score reaches the deployment threshold ('KeypointHead.inference.object_score_threshold'), the
        evaluation keeping lower scores for the mAP, all with the skeleton of their class, under a caption counting
        both.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The outputs of predict_batch.
            sample_index (int): The index of the image in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sample.
        """
        ground_truth_keypoints = batch["keypoints"][sample_index].detach().cpu().numpy()
        ground_truth_labels = batch["labels"][sample_index].detach().cpu().numpy()

        # Objects found in the image, kept from the deployment object score threshold
        grouped_objects = predictions["grouped_objects"][sample_index]
        kept_objects = grouped_objects["scores"] >= self.object_score_threshold
        caption = (f"{len(ground_truth_labels)} ground truth objects | {int(kept_objects.sum())} predicted objects "
                   f"with score >= {self.object_score_threshold:.2f}")
        return build_keypoint_sample_grid(
            image=convert_image_to_array(image=batch["images"][sample_index]),
            ground_truth_keypoints=ground_truth_keypoints[..., :2],
            visibilities=ground_truth_keypoints[..., 2].astype(np.int64), ground_truth_labels=ground_truth_labels,
            predicted_keypoints=grouped_objects["keypoints"][kept_objects],
            predicted_labels=grouped_objects["labels"][kept_objects],
            predicted_scores=grouped_objects["scores"][kept_objects], keypoint_categories=self.keypoint_categories,
            class_names=self.class_names, class_colours=self.class_colours, caption=caption)

    def build_input_windows(self, batch: Dict[str, Any], sample_index: int, split_name: str) -> List[ViewerWindow]:
        """
        Draws one image for the input viewer and the input samples of a training launch: the image as the network
        receives it with the labelled keypoints of its objects and the skeleton of their class under their boxes
        captioned with their class names, next to an information panel holding its number of objects per class, its
        keypoints by visibility, the keypoint colours, its augmentations and the objects they removed.

        Args:
            batch (Dict[str, Any]): The batch holding the image, augmented for the training split, on the CPU.
            sample_index (int): The index of the image in the batch.
            split_name (str): "Training", "Validation" or "Test", the title of the window.

        Returns:
            List[ViewerWindow]: The window of the image.
        """
        # Class names of the test manifest, the label dictionary keys being the raw category identifiers of the COCO
        # labels, so that each box is captioned with its class name instead of its index
        label_names = self.class_names
        sample_labels = batch["labels"][sample_index]
        sample_keypoints = batch["keypoints"][sample_index]

        # Image with the labelled keypoints of its objects, the window drawing their boxes over it
        drawing = draw_keypoint_objects(image=convert_image_to_array(image=batch["images"][sample_index]),
                                        keypoints=sample_keypoints[..., :2].numpy(),
                                        visibilities=sample_keypoints[..., 2].long().numpy(),
                                        labels=sample_labels.numpy(), keypoint_categories=self.keypoint_categories,
                                        class_colours=self.class_colours)

        # Keypoints of the classes of the objects by visibility, the padding positions left out
        class_visibilities = torch.cat([sample_keypoints[object_index, :len(
            self.keypoint_categories[int(class_index)]["keypoints"]), 2]
            for object_index, class_index in enumerate(sample_labels.tolist())] + [torch.zeros(size=(0,))])

        # Objects of the image counted per class, in the order of the class indices, then its keypoints and the
        # keypoint colours
        label_list = sample_labels.tolist()
        class_counts = [f"{class_name} {label_list.count(class_index)}"
                        for class_index, class_name in label_names.items() if class_index in label_list]
        sample_lines = [f"Objects : {len(label_list)} ({', '.join(class_counts)})" if label_list else "Objects : none",
                        f"Keypoints : {int((class_visibilities == 2).sum())} visible, "
                        f"{int((class_visibilities == 1).sum())} occluded, "
                        f"{int((class_visibilities == 0).sum())} unlabelled",
                        "Colours : left blue, right orange, others yellow, rings occluded, skeleton in the class "
                        "colour"]

        # Then the augmentations of the image and the objects they removed
        augmentation_lines = self.describe_input_augmentations(batch=batch, sample_index=sample_index,
                                                               split_name=split_name, label_names=label_names)
        return [ViewerWindow(image=torch.from_numpy(drawing).permute(2, 0, 1).float(),
                             bounding_boxes=batch["bounding_boxes"][sample_index], labels=sample_labels,
                             label_names=label_names, title=split_name, target_lines=sample_lines,
                             augmentation_lines=augmentation_lines, display_input_size=self.input_image_size)]
