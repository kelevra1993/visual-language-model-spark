"""
Trainer of the keypoint detection task, top-down: every sample is one object of a COCO-format split, given by the
keypoint loaders with its whole resized image, box, class and keypoints, from which the trainer cuts a fixed-size crop
on the device (data/keypoints/crop_transform.py), randomly scaled, rotated, flipped and shifted for the training crops,
before the photometric augmentations of the DataAugmenter. The HRNet keypoint detector predicts one heatmap per keypoint
of every class and is trained on the channels of the class of every crop. The checkpoints are evaluated on the ground
truth boxes of the test split with the average precisions on the object keypoint similarity (OKS), the PCK, the mean
OKS and the mean pixel error of every class and every keypoint, with sample images of fixed test objects showing their
true and predicted keypoints. Registered for the keypoint detection task in the task registry.
"""

import json
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision_training.core.contracts import EvaluationResults, KeypointBatch
from vision_training.data.keypoints.crop_transform import (build_crop_affines, build_heatmap_targets,
                                                           compute_crop_regions, create_identity_augmentations,
                                                           describe_crop_augmentation, draw_crop_augmentations,
                                                           transform_points, warp_crops)
from vision_training.data.label_mapping import get_class_names
from vision_training.data.loaders import get_keypoint_dataloaders
from vision_training.data.split_description import count_split_images_and_annotations
from vision_training.evaluation.detection.average_precision_report import summarise_average_precisions
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.evaluation.keypoints.object_keypoint_similarity import (CORRECT_KEYPOINT_FRACTION,
                                                                             build_keypoint_sigma_table,
                                                                             group_image_objects,
                                                                             measure_keypoint_accuracy,
                                                                             summarise_keypoint_accuracy)
from vision_training.models.keypoints.keypoint_detector import HEATMAP_STRIDE, KeypointDetector
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.visualization import (build_keypoint_sample_grid, convert_image_to_array,
                                                     draw_keypoint_crop, view_input_data)

# File of the accuracy of every keypoint of every class, in the evaluation folder of every checkpoint
KEYPOINT_METRICS_FILE_NAME = "keypoint_metrics.csv"

# Values kept for every evaluated test object, concatenated over the test split by compute_evaluation_results
EVALUATED_OBJECT_KEYS = ["image_identifiers", "labels", "bounding_boxes", "keypoints", "visibilities",
                         "predicted_keypoints", "scores"]

# Headline metric comparing the checkpoints, the mAP@0.5 on the object keypoint similarity with the area method
KEYPOINT_HEADLINE_METRIC = "OKS mAP@0.5/area"


def build_keypoint_counts(keypoint_categories: Dict[int, Dict[str, Any]]) -> torch.Tensor:
    """
    Gathers the number of keypoints of every class index into the table the keypoint trainer indexes with the labels of
    a batch, which marks the valid keypoint positions of its predictions. The background and the classes without
    keypoints have none.

    Args:
        keypoint_categories (Dict[int, Dict[str, Any]]): The 'keypoints' names of every foreground class index
            ('Data.keypoint_categories').

    Returns:
        torch.Tensor: The [C] int64 keypoint counts, C being the largest class index plus one.
    """
    keypoint_counts = torch.zeros(size=(max(keypoint_categories) + 1,), dtype=torch.int64)

    # Number of keypoints of every foreground class written in its row
    for class_index, definition in keypoint_categories.items():
        keypoint_counts[class_index] = len(definition["keypoints"])
    return keypoint_counts


class KeypointTrainer(BaseTrainer):
    """
    Trainer of the keypoint detection task: the HRNet keypoint detector trained on the object crops of COCO-format
    datasets with keypoints, its heatmap loss tracked, and every evaluated checkpoint scored with the mean average
    precision on the object keypoint similarity of the test objects.
    """

    TASK_NAME = "keypoint_detection"

    # Every sample is one object, counted as such in the throughput and the evaluation notes
    SAMPLE_NAME = "Objects"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the crop settings, the crop augmentation and the keypoint definition of every class, builds the tables
        indexed by the labels of the batches and the sigmas of the object keypoint similarity, then prepares the
        training run through the general trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data with its injected
                'keypoint_categories', Backbone, HRNet, KeypointHead).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        data_configuration = model_configuration["Data"]

        # Size of the crops, enlargement of the boxes they cover and random geometric augmentation of the training crops
        self.crop_height = data_configuration["crop_settings"]["height"]
        self.crop_width = data_configuration["crop_settings"]["width"]
        self.box_enlargement = data_configuration["crop_settings"]["box_enlargement"]
        self.crop_augmentation_configuration = data_configuration["crop_augmentation"]
        self.heatmap_sigma = model_configuration["KeypointHead"]["heatmap_sigma"]

        # Keypoint definition of every class, injected by the configuration loader from the training labels.json, with
        # the keypoint counts indexed by the labels of the batches
        self.keypoint_categories = {int(class_index): definition
                                    for class_index, definition in data_configuration["keypoint_categories"].items()}
        self.maximum_keypoints = max(len(definition["keypoints"]) for definition in self.keypoint_categories.values())
        self.keypoint_counts = build_keypoint_counts(keypoint_categories=self.keypoint_categories)

        # Tolerance of the object keypoint similarity of every keypoint of every class
        evaluation_configuration = experiment_configuration["evaluation"]
        self.sigma_table = build_keypoint_sigma_table(keypoint_categories=self.keypoint_categories,
                                                      default_sigma=evaluation_configuration["default_keypoint_sigma"],
                                                      keypoint_sigmas=evaluation_configuration["keypoint_sigmas"],
                                                      maximum_keypoints=self.maximum_keypoints)
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the keypoint loaders of the three splits, one object per sample, and names the classes after the
        categories of the test split for the per-class report.

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders.
        """
        test_labels_path = Path(self.experiment_configuration["test_data_folder"]) / "labels.json"
        with open(file=test_labels_path, mode="r") as labels_file:
            test_labels = json.load(fp=labels_file)
        self.class_names = get_class_names(categories=test_labels["categories"], label_dictionary=self.label_dictionary)
        return get_keypoint_dataloaders(experiment_configuration=self.experiment_configuration,
                                        model_configuration=self.model_configuration, batch_size=self.batch_size)

    def list_class_names(self) -> Optional[List[str]]:
        """
        Gives the class names taken from the test manifest by build_dataloaders, the background first, for the record
        of the data of the run and the transfer of the components to a run adding classes.

        Returns:
            Optional[List[str]]: The class names, in class index order.
        """
        return [self.class_names[class_index] for class_index in range(len(self.class_names))]

    def build_model(self) -> KeypointDetector:
        """
        Builds the HRNet keypoint detector from the model configuration.

        Returns:
            KeypointDetector: The untrained keypoint detector.
        """
        return KeypointDetector(configuration=self.model_configuration, device=self.device, dtype=self.dtype,
                                verbose=self.diagnosis_modes["verbose"])

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the total loss and the heatmap loss of the keypoint detector, tracked on every training and validation
        batch.

        Returns:
            Dict[str, str]: The display name of every tracked loss.
        """
        return {"total_loss": "Total Loss", "keypoint_loss": "Keypoint Heatmap Loss"}

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the objects of a split, the samples of the keypoint loaders (every split leaving out the objects
        without labelled keypoint), with the images and annotations of its labels.json.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of objects and their description with the numbers of images and annotations.

        Raises:
            ValueError: If the folder is none of the split folders of the run.
        """
        number_images, number_annotations = count_split_images_and_annotations(split_data_folder=split_data_folder)

        # Loader of the split, the first split reading the folder
        for split_key, loader in [("train_data_folder", self.train_dataloader),
                                  ("validation_data_folder", self.validation_dataloader),
                                  ("test_data_folder", self.test_dataloader)]:
            if Path(self.experiment_configuration[split_key]) == Path(split_data_folder):
                number_objects = len(loader.dataset)
                return number_objects, (f"{number_objects} objects, from {number_images} images and "
                                        f"{number_annotations} annotations")
        raise ValueError(f"The folder {split_data_folder} is none of the split folders of the run.")

    def cut_object_crops(self, batch: Dict[str, Any],
                         augmentations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Cuts the crop of every object of a batch out of its resized image with its geometric augmentation, on the device
        of the images, and moves its keypoints into the crop, keeping their names and visibilities, turned into
        Gaussian heatmap targets with their weights. Used for the identity crops of every prepared batch, the augmented crops of
        the training batches and the crops shown by the input viewer.

        Args:
            batch (Dict[str, Any]): The batch, with its 'images', 'bounding_boxes', 'labels', 'keypoints' and
                'visibilities'.
            augmentations (Dict[str, torch.Tensor]): The 'scales', 'rotations', 'flips' and 'shifts' of the crops.

        Returns:
            Dict[str, torch.Tensor]: The 'crops' in the data type of the images, the 'crop_affines' and
                'crop_inverse_affines', the 'crop_keypoints' and 'crop_visibilities', the 'target_heatmaps' and the
                'heatmap_weights'.
        """
        centres, sizes = compute_crop_regions(bounding_boxes=batch["bounding_boxes"].to(dtype=torch.float32),
                                              crop_height=self.crop_height, crop_width=self.crop_width,
                                              box_enlargement=self.box_enlargement)
        crop_affines, crop_inverse_affines = build_crop_affines(centres=centres, sizes=sizes,
                                                                augmentations=augmentations,
                                                                crop_height=self.crop_height,
                                                                crop_width=self.crop_width)
        crops = warp_crops(images=batch["images"], crop_to_image=crop_inverse_affines, crop_height=self.crop_height,
                           crop_width=self.crop_width)

        # Keypoints moved into the crop, keeping their names and visibilities, and their heatmap targets
        crop_keypoints = transform_points(points=batch["keypoints"].to(dtype=torch.float32), affines=crop_affines)
        crop_visibilities = batch["visibilities"]
        target_heatmaps, heatmap_weights = build_heatmap_targets(
            crop_keypoints=crop_keypoints, visibilities=crop_visibilities, crop_height=self.crop_height,
            crop_width=self.crop_width, heatmap_stride=HEATMAP_STRIDE, heatmap_sigma=self.heatmap_sigma)
        return {"crops": crops, "crop_affines": crop_affines, "crop_inverse_affines": crop_inverse_affines,
                "crop_keypoints": crop_keypoints, "crop_visibilities": crop_visibilities,
                "target_heatmaps": target_heatmaps, "heatmap_weights": heatmap_weights}

    def prepare_batch(self, batch: Dict[str, Any]) -> KeypointBatch:
        """
        Moves the images (in the precision of the model), boxes, classes, keypoints and visibilities of a keypoint batch
        to the device and cuts the crop of every object without augmentation, the crops the validation and the test
        evaluation run on (augment_batch cuts new ones for the training batches).

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a keypoint loader.

        Returns:
            KeypointBatch: The batch ready for the keypoint detector.
        """
        prepared_batch = {"images": batch["images"].to(device=self.device, dtype=self.dtype),
                          "bounding_boxes": batch["bounding_boxes"].to(device=self.device, dtype=torch.float32),
                          "labels": batch["labels"].to(device=self.device, dtype=torch.int64),
                          "keypoints": batch["keypoints"].to(device=self.device, dtype=torch.float32),
                          "visibilities": batch["visibilities"].to(device=self.device, dtype=torch.int64),
                          "image_identifiers": batch["image_identifiers"],
                          "sample_identifiers": batch["sample_identifiers"]}
        identity_augmentations = create_identity_augmentations(number_samples=prepared_batch["labels"].shape[0])
        return {**prepared_batch, **self.cut_object_crops(batch=prepared_batch, augmentations=identity_augmentations)}

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the training crops: cuts them again with a random scale, rotation, horizontal flip and shift
        drawn from 'Data.crop_augmentation', their keypoints and targets following without being renamed, then applies
        the photometric augmentations of the DataAugmenter to the crops. The input viewer calls it on the loader batch on
        the CPU.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The batch with its augmented crops, keypoints and targets, and the drawn
                'crop_augmentations' and the 'augmentation_reports' of its crops for the input viewer.
        """
        crop_augmentations = draw_crop_augmentations(number_samples=batch["labels"].shape[0],
                                                     augmentation_configuration=self.crop_augmentation_configuration)
        augmented_batch = {**batch, **self.cut_object_crops(batch=batch, augmentations=crop_augmentations)}

        # Photometric augmentations of the crops, the geometric types of the DataAugmenter being off for this task
        crops, _, _, _, _, augmentation_reports = self.data_augmenter.augment_batch(images=augmented_batch["crops"])
        return {**augmented_batch, "crops": crops, "crop_augmentations": crop_augmentations,
                "augmentation_reports": augmentation_reports}

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the keypoint detector on the crops with their classes and heatmap targets, so that it returns its heatmap
        loss on the channels of the class of every crop.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The detector outputs, with its 'keypoint_loss'.
        """
        return self.model(input_tensor=batch["crops"], labels=batch["labels"], target_heatmaps=batch["target_heatmaps"],
                          heatmap_weights=batch["heatmap_weights"])

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the keypoints of the class of every test crop, as in deployment on the ground truth boxes, moves them
        back into the resized image where the object keypoint similarity is measured, and scores every object with the
        mean score of its keypoints, which ranks the objects in the average precision.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The detector outputs, with the [B, K, 2] 'image_keypoints' in resized image pixels and the
                [B] 'object_scores'.
        """
        predictions = self.model(input_tensor=batch["crops"], labels=batch["labels"])

        # Valid keypoint positions of the class of every crop, the padding positions left out of the object score
        labels = batch["labels"]
        valid_positions = (torch.arange(end=self.maximum_keypoints, device=labels.device)[None, :] <
                           self.keypoint_counts.to(device=labels.device)[labels][:, None])
        keypoint_scores = predictions["category_keypoint_scores"].float()
        object_scores = (keypoint_scores * valid_positions).sum(dim=1) / valid_positions.sum(dim=1).clamp(min=1)

        # Keypoints moved from the crop back into the resized image
        image_keypoints = transform_points(points=predictions["category_keypoints"].float(),
                                           affines=batch["crop_inverse_affines"])
        return {**predictions, "image_keypoints": image_keypoints, "object_scores": object_scores}

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the lists of the values of the evaluated test objects, filled batch by batch on the CPU.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint (unused).

        Returns:
            Dict[str, Any]: An empty list for every key of EVALUATED_OBJECT_KEYS.
        """
        return {object_key: [] for object_key in EVALUATED_OBJECT_KEYS}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the image, class, box and ground truth keypoints of every kept object of a test batch on the CPU, with its
        predicted keypoints and score, all in resized image pixels.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, its lists extended in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The outputs of predict_batch.
            sample_indices (List[int]): The indices of the objects to keep.
        """
        object_values = {"image_identifiers": torch.as_tensor(data=batch["image_identifiers"]),
                         "labels": batch["labels"], "bounding_boxes": batch["bounding_boxes"],
                         "keypoints": batch["keypoints"], "visibilities": batch["visibilities"],
                         "predicted_keypoints": predictions["image_keypoints"],
                         "scores": predictions["object_scores"]}
        for object_key in EVALUATED_OBJECT_KEYS:
            accumulator[object_key].append(object_values[object_key].detach().cpu()[sample_indices])

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the average precisions of the test split on the object keypoint similarity with the three average
        precision methods (the area method being the headline one, its mAP@0.5 comparing the checkpoints), of every
        object size and of every class, with the mean average recalls, and the keypoint accuracy (PCK, mean OKS and
        mean pixel error) of all the keypoints, of every class and of every keypoint of every class. The tables of the
        methods, object sizes and recalls are always shown, the per-class and per-keypoint tables with the per-class
        report. The class rows of both are saved to class_metrics.csv and the keypoint rows to keypoint_metrics.csv in
        the evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/'.

        Args:
            accumulator (Dict[str, Any]): The values of the evaluated test objects.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.

        Raises:
            ValueError: If the classes of the average precisions and of the keypoint accuracy differ.
        """
        objects = {object_key: torch.cat(tensors=values, dim=0) for object_key, values in accumulator.items()}

        # Average precisions on the object keypoint similarity, the objects of every image grouped
        detections, ground_truths = group_image_objects(objects=objects, sigma_table=self.sigma_table)
        average_precision_summary = summarise_average_precisions(
            detections=detections, ground_truths=ground_truths, number_classes=len(self.class_names),
            class_names=self.class_names, overlap_type="keypoints", title_prefix="OKS", name_prefix="OKS ",
            column_prefix="oks_")
        accuracy_summary = summarise_keypoint_accuracy(objects=objects, sigma_table=self.sigma_table,
                                                       keypoint_categories=self.keypoint_categories,
                                                       class_names=self.class_names)

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

        # Tables of the methods, object sizes and recalls first, then the per-class and per-keypoint tables
        tables = [average_precision_summary["tables"]["methods"], average_precision_summary["tables"]["sizes"],
                  average_precision_summary["tables"]["recalls"], average_precision_summary["tables"]["classes"],
                  accuracy_summary["tables"]["classes"], accuracy_summary["tables"]["keypoints"]]
        return {"evaluated_samples": f"{objects['labels'].shape[0]} Test Objects",
                "metrics": accuracy_summary["metrics"], "notes": [], "tables": tables,
                "scalars": {**average_precision_summary["scalars"], **accuracy_summary["scalars"]},
                "class_scalars": average_precision_summary["class_scalars"], "images": {}, "files": files,
                "headline_metric": KEYPOINT_HEADLINE_METRIC}

    def get_test_sample_classes(self) -> Dict[int, List[int]]:
        """
        Lists the class of every evaluated test object, so that the fixed test objects drawn at every evaluation are
        spread over the classes.

        Returns:
            Dict[int, List[int]]: The class index of every test object, keyed by its annotation identifier.
        """
        return {test_object["sample_identifier"]: [test_object["label"]]
                for test_object in self.test_dataloader.dataset.list_objects()}

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws a fixed test object for the sample images of the evaluation: its crop with its labelled keypoints next to
        the same crop with its predicted keypoints, both with the skeleton of its class, under a caption giving its
        class, its object keypoint similarity, its correct keypoints and its score.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The outputs of predict_batch.
            sample_index (int): The index of the object in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sample.
        """
        definition = self.keypoint_categories[int(batch["labels"][sample_index])]
        number_keypoints = len(definition["keypoints"])

        # Accuracy of the object alone, in resized image pixels as in the evaluation
        sample_object = {"labels": batch["labels"], "bounding_boxes": batch["bounding_boxes"],
                         "keypoints": batch["keypoints"], "visibilities": batch["visibilities"],
                         "predicted_keypoints": predictions["image_keypoints"]}
        sample_object = {object_key: values[sample_index:sample_index + 1].detach().cpu()
                         for object_key, values in sample_object.items()}
        accuracy = measure_keypoint_accuracy(objects=sample_object, sigma_table=self.sigma_table)
        caption = (f"{definition['name']} | OKS {float(accuracy['similarities'][0]):.3f} | "
                   f"{int(accuracy['correct'].sum())}/{int(accuracy['labelled'].sum())} keypoints within "
                   f"{CORRECT_KEYPOINT_FRACTION:g} of the box | "
                   f"score {float(predictions['object_scores'][sample_index]):.3f}")

        # True and predicted keypoints of the class, in crop pixels
        return build_keypoint_sample_grid(
            crop=convert_image_to_array(image=batch["crops"][sample_index]),
            ground_truth_keypoints=batch["crop_keypoints"][sample_index, :number_keypoints].detach().cpu().numpy(),
            visibilities=batch["crop_visibilities"][sample_index, :number_keypoints].detach().cpu().numpy(),
            predicted_keypoints=predictions["category_keypoints"][sample_index, :number_keypoints].detach().float()
            .cpu().numpy(), keypoint_names=definition["keypoints"], skeleton=definition["skeleton"], caption=caption)

    def get_input_crop(self, batch: Dict[str, Any], sample_index: int) -> Dict[str, torch.Tensor]:
        """
        Gives the crop of a sample shown by the input viewer with its keypoints: the augmented crop of a training batch,
        or the crop without augmentation cut for the sample alone from a validation batch, which the viewer receives as
        the loader gives it.

        Args:
            batch (Dict[str, Any]): The augmented training batch, or the validation batch, on the CPU.
            sample_index (int): The index of the sample in its batch.

        Returns:
            Dict[str, torch.Tensor]: The [3, h, w] 'crops', the [K, 2] 'crop_keypoints', the [K] 'crop_visibilities'
                and the [K] 'heatmap_weights' of the sample.
        """
        crop_keys = ["crops", "crop_keypoints", "crop_visibilities", "heatmap_weights"]
        if "crops" in batch:
            return {crop_key: batch[crop_key][sample_index] for crop_key in crop_keys}

        # Crop of the validation sample cut alone, without augmentation
        sample_batch = {object_key: batch[object_key][sample_index:sample_index + 1]
                        for object_key in ["images", "bounding_boxes", "labels", "keypoints", "visibilities"]}
        sample_batch["images"] = sample_batch["images"].to(dtype=torch.float32)
        sample_crop = self.cut_object_crops(batch=sample_batch, augmentations=create_identity_augmentations(
            number_samples=1))
        return {crop_key: sample_crop[crop_key][0] for crop_key in crop_keys}

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays an augmented training crop next to a validation crop, each with its labelled keypoints and the skeleton
        of its class, and an information panel holding its class, its keypoints by visibility, the keypoints trained
        inside the crop, the keypoint colours and, for the training crop, its crop augmentation and its photometric
        augmentations.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed crops in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        images, panel_lines = [], []
        for batch in [training_batch, validation_batch]:
            definition = self.keypoint_categories[int(batch["labels"][sample_index])]
            number_keypoints = len(definition["keypoints"])
            sample_crop = self.get_input_crop(batch=batch, sample_index=sample_index)
            visibilities = sample_crop["crop_visibilities"][:number_keypoints]

            # Crop with its labelled keypoints and skeleton, at the display size of the viewer
            drawing = draw_keypoint_crop(
                crop=convert_image_to_array(image=sample_crop["crops"]),
                keypoints=sample_crop["crop_keypoints"][:number_keypoints].numpy(),
                visibilities=visibilities.numpy(), keypoint_names=definition["keypoints"],
                skeleton=definition["skeleton"], display_size=self.input_image_size)
            images.append(torch.from_numpy(drawing).permute(2, 0, 1))

            # Target of the crop, then its augmentations, the crop augmentation first for a training crop
            sample_lines = [f"Class : {definition['name']}",
                            f"Keypoints : {int((visibilities == 2).sum())} visible, {int((visibilities == 1).sum())} "
                            f"occluded, {int((visibilities == 0).sum())} unlabelled, of {number_keypoints}",
                            f"Trained : {int(sample_crop['heatmap_weights'].sum())} labelled keypoints inside the crop",
                            "Colours : left blue, right orange, others yellow, rings occluded"]
            augmentation_lines = self.describe_input_augmentations(batch=batch, sample_index=sample_index)
            if "crop_augmentations" in batch:
                augmentation_lines = (augmentation_lines[:2] + describe_crop_augmentation(
                    augmentations=batch["crop_augmentations"], sample_index=sample_index) + augmentation_lines[2:])
            panel_lines.append(sample_lines + augmentation_lines)
        return view_input_data(images=images, bounding_boxes=None, labels=None,
                               window_titles=["Training", "Validation"], display_input_size=self.input_image_size,
                               panel_lines=panel_lines)
