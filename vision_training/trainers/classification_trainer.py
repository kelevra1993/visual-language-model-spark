"""
Trainer of the image classification task: the data loaders of the 'labels.json' manifests, the classifier, the
accuracy tracked on every batch, the view losses comparing every image with a second random view of it (Barlow Twins
and the contrastive loss, their projectors trained as auxiliary heads of the general trainer), and the evaluation of
the checkpoints on the test split (classification metrics, per-class metrics, confusion matrix, class thresholds, error
images and the predictions file). Registered for the classification task and extended by the sequence classification
trainer.
"""

import csv
import json
import cv2
import numpy as np
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision_training.core.contracts import (LOSS_KEY_SUFFIX, SAMPLE_IDENTIFIERS_KEY, ClassificationBatch,
                                            EvaluationResults)
from vision_training.data.loaders import get_classification_dataloaders
from vision_training.data.split_description import count_split_samples
from vision_training.data.view_augmentations import SECOND_VIEW_INDEX, ViewAugmenter, describe_view_transformations
from vision_training.evaluation.classification.classification_metrics import (compute_confusion_matrix,
                                                                              compute_metrics_from_confusion_matrix,
                                                                              compute_multi_label_metrics,
                                                                              compute_top_k_accuracy)
from vision_training.evaluation.classification.class_metrics_report import (build_class_metric_scalars,
                                                                             build_class_metrics_rows,
                                                                             build_class_metrics_table)
from vision_training.evaluation.classification.confusion_matrix import (save_confusion_matrix_image,
                                                                        save_confusion_matrix_table)
from vision_training.evaluation.classification.threshold_finder import find_class_thresholds
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.losses.classification_losses import compute_balanced_class_weights
from vision_training.models.classification.classification_head import build_global_pooling, compute_pooled_features
from vision_training.models.classification.view_losses import (VIEW_LOSS_NAMES, build_view_loss_projectors,
                                                               compute_view_losses, print_view_losses_summary)
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.visualization import (CORRECT_PREDICTION_COLOUR, WRONG_PREDICTION_COLOUR,
                                                     add_caption_bar, convert_image_to_array, view_input_data)

# Probability from which a class is predicted in multi-label mode
MULTI_LABEL_THRESHOLD = 0.5

# Metrics tracked on every training and validation batch for each view loss turned on, with their display names, the
# names of the self-supervised trainer
VIEW_LOSS_TRACKED_METRICS = {"barlow_twins": {"barlow_twins_loss": "Barlow Twins Loss",
                                              "cross_correlation_diagonal": "Diagonal Correlation",
                                              "cross_correlation_off_diagonal": "Off-Diagonal Correlation"},
                             "contrastive": {"contrastive_loss": "Contrastive Loss",
                                             "positive_pair_accuracy": "Positive Pair Accuracy"}}

# Key of the images of a training batch before its data augmentation, kept by augment_batch when a view loss is on so
# that the second view of every image is drawn independently of the augmentation of the image
VIEW_SOURCE_IMAGES_KEY = "view_source_images"


class ClassificationTrainer(BaseTrainer):
    """
    Trainer of the classification task: the image classifier trained on 'labels.json' manifests (Native loader), with
    its loss and its accuracy tracked on every batch, and every evaluated checkpoint scored on the test split with the
    accuracy, balanced accuracy, top-k accuracy, per-class precision, recall and F1, the confusion matrix and the
    per-class thresholds reaching the target precision. With 'ClassificationHead.view_losses' on, a second random view
    of every training and validation image goes through the classifier with it, and Barlow Twins and the contrastive
    loss of their representations, each through its projector kept as an auxiliary head, are added to the loss.

    The sequence classification trainer extends it, the classes, loss, batch metric and evaluation being the same
    once a sequence is classified as a whole.
    """

    TASK_NAME = "classification"

    # Name of the column of the evaluated samples of every class in the per-class table of the report ("Positives" in
    # multi-label mode, the samples of the classes present)
    CLASS_SUPPORT_COLUMN = "Samples"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the classification settings used by the hooks, then prepares the training run through the general
        trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, ClassificationHead).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # One class per sample (softmax) or several (sigmoid), the weighting of the classes in the loss, and the
        # evaluation settings: the k of the top-k accuracy, the precision targeted by the per-class thresholds,
        # whether the predictions of every test sample are saved, and the error images of the misclassified samples
        self.output_mode = model_configuration["ClassificationHead"]["output_mode"]
        self.class_weighting = model_configuration["ClassificationHead"]["class_weighting"]

        # Decorrelation loss of the hidden features of the head, a setting of the image classifier only: the sequence
        # tasks sharing this trainer have no 'ClassificationHead.decorrelation'
        self.decorrelation_activated = (self.TASK_NAME == "classification" and
                                        model_configuration["ClassificationHead"]["decorrelation"]["activated"])

        # View losses comparing every image with a second random view of it, a setting of the image classifier only
        # too: the second views are drawn on the device by the view augmenter, the final feature maps of both views are
        # pooled by the pooling of the view losses, and the projector of every loss turned on is built as an auxiliary
        # head once the classifier is built (build_auxiliary_heads)
        self.view_losses_configuration = None
        self.view_losses_activated = False
        self.view_augmenter = None
        self.view_pooling = None
        if self.TASK_NAME == "classification":
            self.view_losses_configuration = model_configuration["ClassificationHead"]["view_losses"]
            self.view_losses_activated = any(self.view_losses_configuration[loss_name]["activated"]
                                             for loss_name in VIEW_LOSS_NAMES)
        if self.view_losses_activated:
            self.view_augmenter = ViewAugmenter(
                view_augmentation_configuration=self.view_losses_configuration["view_augmentation"],
                image_size=model_configuration["Data"]["image_settings"]["size"])
            self.view_pooling = build_global_pooling(global_pooling=self.view_losses_configuration["global_pooling"])
        self.top_k = experiment_configuration["evaluation"]["top_k"]
        self.target_precision = experiment_configuration["evaluation"]["target_precision"]
        self.save_predictions = experiment_configuration["evaluation"]["save_predictions"]
        self.error_dumps_configuration = experiment_configuration["evaluation"]["error_dumps"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the classification loaders of the three splits, and takes the class names from the categories of the
        training manifest.

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders.
        """
        loaders = get_classification_dataloaders(experiment_configuration=self.experiment_configuration,
                                                 model_configuration=self.model_configuration,
                                                 batch_size=self.batch_size)
        training_dataset = loaders[0].dataset
        self.class_names = [training_dataset.class_names[class_index]
                            for class_index in range(training_dataset.number_classes)]
        return loaders

    def list_class_names(self) -> Optional[List[str]]:
        """
        Gives the class names taken from the training manifest by build_dataloaders, for the record of the data of the
        run and the transfer of the classification head to a run adding classes after them. Inherited by the sequence
        trainers, whose heads predict the classes of their manifests too.

        Returns:
            Optional[List[str]]: The class names, in class index order.
        """
        return self.class_names

    def build_model(self) -> TaskModel:
        """
        Builds the model of the task from the model registry, and weights its classes in the loss inversely to their
        frequency in the training split when 'ClassificationHead.class_weighting' is "balanced".

        Returns:
            TaskModel: The untrained model.
        """
        model = TASK_MODEL_CLASSES[self.TASK_NAME](configuration=self.model_configuration, device=self.device,
                                                   dtype=self.dtype, verbose=self.diagnosis_modes["verbose"])
        if self.class_weighting == "balanced":
            class_counts = self.train_dataloader.dataset.get_class_counts()
            model.class_weights.copy_(compute_balanced_class_weights(class_counts=class_counts))
        return model

    def build_auxiliary_heads(self) -> Dict[str, nn.Module]:
        """
        Builds the projector of every view loss turned on ('ClassificationHead.view_losses'), reading the final feature
        map of the backbone pooled by the pooling of the view losses, as the auxiliary heads of the general trainer:
        trained with the classifier but never exported, transferred nor averaged, and saved in the checkpoints with
        their own optimizers, so that a view loss can be turned on or off when a run resumes.

        Returns:
            Dict[str, nn.Module]: The projector of every view loss turned on, empty without view loss and for the
                sequence tasks.
        """
        if not self.view_losses_activated:
            return {}
        representation_features = compute_pooled_features(
            global_pooling=self.view_losses_configuration["global_pooling"], backbone=self.model.backbone)
        return build_view_loss_projectors(view_losses_configuration=self.view_losses_configuration,
                                          input_features=representation_features, device=self.device, dtype=self.dtype)

    def print_auxiliary_heads_summary(self) -> None:
        """
        Prints the view losses with their projectors after the summary of the classifier, when one of them is on.
        """
        if self.view_losses_activated:
            print_view_losses_summary(view_losses_configuration=self.view_losses_configuration,
                                      projectors=self.auxiliary_heads)

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the losses and the accuracy tracked on every training and validation batch, with the decorrelation loss
        and the mean absolute correlation of the decorrelated hidden features when 'ClassificationHead.decorrelation'
        is on, and the loss and batch metrics of every view loss turned on ('ClassificationHead.view_losses').

        Returns:
            Dict[str, str]: The display name of every tracked metric.
        """
        tracked_metrics = {"total_loss": "Total Loss", "classification_loss": "Classification Loss"}
        if self.decorrelation_activated:
            tracked_metrics.update({"decorrelation_loss": "Decorrelation Loss",
                                    "hidden_feature_correlation": "Hidden Feature Correlation"})
        for loss_name in self.list_activated_view_losses():
            tracked_metrics.update(VIEW_LOSS_TRACKED_METRICS[loss_name])
        tracked_metrics["accuracy"] = "Accuracy"
        return tracked_metrics

    def list_activated_view_losses(self) -> List[str]:
        """
        Lists the view losses turned on in 'ClassificationHead.view_losses', whose losses and batch metrics are tracked.

        Returns:
            List[str]: The names of the view losses turned on, in the order of VIEW_LOSS_NAMES, none for the sequence
                tasks.
        """
        if not self.view_losses_activated:
            return []
        return [loss_name for loss_name in VIEW_LOSS_NAMES if self.view_losses_configuration[loss_name]["activated"]]

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the samples and categories of a classification split from its labels.json manifest.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of samples and their description with the number of categories.
        """
        number_samples, number_categories = count_split_samples(split_data_folder=split_data_folder)
        return number_samples, f"{number_samples} images, {number_categories} categories"

    def prepare_batch(self, batch: Dict[str, Any]) -> ClassificationBatch:
        """
        Moves the images and labels of a classification batch to the device, the images in the precision of the model,
        the labels as int64 class indices (or multi-hot vectors in the precision of the model in multi-label mode).

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a classification loader.

        Returns:
            ClassificationBatch: The batch ready for the classifier.
        """
        labels_dtype = torch.int64 if self.output_mode == "multi_class" else self.dtype
        return {**batch, "images": batch["images"].to(device=self.device, dtype=self.dtype),
                "labels": batch["labels"].to(device=self.device, dtype=labels_dtype)}

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomly augments the images of a training batch like every task and, when a view loss of the image classifier
        is on, keeps the images before the augmentation under VIEW_SOURCE_IMAGES_KEY, the second views being drawn from
        them so that the two views of an image are drawn independently of each other. The sequence trainers inherit it
        through their own augmentation, without view losses.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The batch with its augmented images, their 'augmentation_reports' and, with a view loss on,
                the images before the augmentation.
        """
        augmented_batch = super().augment_batch(batch=batch)

        # Images before the augmentation, kept for the second views of the image classifier only
        if self.TASK_NAME == "classification" and self.view_losses_activated:
            augmented_batch[VIEW_SOURCE_IMAGES_KEY] = batch["images"]
        return augmented_batch

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the classifier with the labels of the batch, so that it returns its classification loss. With a view loss
        on, a second random view of every image is drawn from the image before the data augmentation (the image itself
        for a validation batch), both views go through the classifier, and the view losses of their final feature maps,
        pooled by the pooling of the view losses, are added to its outputs, summed with the classification loss into
        the total loss.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The classifier outputs, with its classification loss and the view losses turned on.
        """
        if not self.view_losses_activated:
            return self.model(input_tensor=batch["images"], labels=batch["labels"])

        # Second view of every image, drawn by the chain of the second view from the image before the data augmentation
        # of a training batch
        source_images = batch[VIEW_SOURCE_IMAGES_KEY] if VIEW_SOURCE_IMAGES_KEY in batch else batch["images"]
        second_views = self.view_augmenter.draw_view(images=source_images, view_index=SECOND_VIEW_INDEX)
        model_outputs = self.model(input_tensor=batch["images"], labels=batch["labels"],
                                   second_view_tensor=second_views)

        # View losses of the pooled final feature maps of both views, through the projectors kept as auxiliary heads
        model_outputs.update(compute_view_losses(first_feature_maps=model_outputs["final_feature_map"],
                                                 second_feature_maps=model_outputs["second_view_final_feature_map"],
                                                 labels=batch["labels"], global_pooling=self.view_pooling,
                                                 projectors=self.auxiliary_heads,
                                                 view_losses_configuration=self.view_losses_configuration))
        return model_outputs

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the classifier without labels, as in deployment.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The class logits and probabilities.
        """
        return self.model(input_tensor=batch["images"])

    def compute_batch_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Computes the accuracy of a training or validation batch from the logits of its forward pass, without any extra
        forward pass: the share of samples whose most probable class is right, or in multi-label mode the share of
        class decisions (probability above 0.5) that are right. With the decorrelation on, it also gives the mean
        absolute correlation of the decorrelated hidden features, returned by the classifier with its loss, and with a
        view loss on the batch metrics of its method, computed with its loss.

        Args:
            batch (Dict[str, Any]): The prepared batch.
            model_outputs (Dict[str, Any]): The classifier outputs.

        Returns:
            Dict[str, torch.Tensor]: The 'accuracy' of the batch, its 'hidden_feature_correlation' with the
                decorrelation on, and the batch metrics of the view losses turned on.
        """
        with torch.no_grad():
            if self.output_mode == "multi_class":
                correct_predictions = model_outputs["class_logits"].argmax(dim=-1) == batch["labels"]
            else:
                correct_predictions = ((model_outputs["class_probabilities"] >= MULTI_LABEL_THRESHOLD) ==
                                       (batch["labels"] >= 0.5))
            batch_metrics = {"accuracy": correct_predictions.to(dtype=torch.float32).mean()}

        # Correlation of the decorrelated hidden features, computed by the classifier with the decorrelation loss
        if self.decorrelation_activated:
            batch_metrics["hidden_feature_correlation"] = model_outputs["hidden_feature_correlation"]

        # Batch metrics of every view loss turned on, computed with its loss
        for loss_name in self.list_activated_view_losses():
            batch_metrics.update({metric_key: model_outputs[metric_key]
                                  for metric_key in VIEW_LOSS_TRACKED_METRICS[loss_name]
                                  if not metric_key.endswith(LOSS_KEY_SUFFIX)})
        return batch_metrics

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the lists of class probabilities, labels and sample identifiers filled during the evaluation of the
        test split, with the folder and the counts of the error images.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, which names its evaluation folder.

        Returns:
            Dict[str, Any]: The empty 'probabilities', 'labels' and 'sample_identifiers' lists, the 'errors_folder' of
                the error images and their empty 'error_counts' per class pair.
        """
        return {"probabilities": [], "labels": [], "sample_identifiers": [],
                "errors_folder": self.get_evaluation_folder(iteration=iteration) / "Errors",
                "error_counts": {}}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the class probabilities, the labels and the identifiers of the kept samples of a test batch on the CPU,
        the identifiers naming the samples in the predictions file, and saves the misclassified samples as images when
        the error images are on.

        Args:
            accumulator (Dict[str, Any]): The 'probabilities', 'labels' and 'sample_identifiers' lists, extended in
                place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The classifier outputs.
            sample_indices (List[int]): The indices of the samples to keep.
        """
        accumulator["probabilities"].append(predictions["class_probabilities"][sample_indices].detach().cpu().double())
        accumulator["labels"].append(batch["labels"][sample_indices].detach().cpu())
        accumulator["sample_identifiers"].append(batch[SAMPLE_IDENTIFIERS_KEY][sample_indices].detach().cpu())

        # Misclassified samples saved as images while their images are still in the batch
        if self.error_dumps_configuration["activated"]:
            self.dump_errors(accumulator=accumulator, batch=batch, predictions=predictions,
                             sample_indices=sample_indices)

    def reserve_error_image(self, accumulator: Dict[str, Any], true_class_index: int,
                            predicted_class_index: int) -> Optional[Path]:
        """
        Counts one more error image for a pair of true and predicted classes, and gives the folder it is written to,
        '<TRUE>-TO-<PREDICTED>', or None when the pair already holds the maximum number of images, so that the image
        is only built when it is saved.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, holding the 'errors_folder' and 'error_counts'.
            true_class_index (int): The index of the true class.
            predicted_class_index (int): The index of the predicted class.

        Returns:
            Optional[Path]: The folder of the class pair, created, or None when the pair is full.
        """
        return self.reserve_error_folder(
            accumulator=accumulator,
            folder_name=f"{self.class_names[true_class_index]}-TO-{self.class_names[predicted_class_index]}")

    def build_input_image(self, batch: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Builds the image of a test sample shown by the error images and the sample images of the evaluation: the
        network input of an image, as the model saw it. The sequence trainers override it to lay out the frames of a
        sequence or of a pair on a grid.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            sample_index (int): The index of the sample in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sample.
        """
        return convert_image_to_array(image=batch["images"][sample_index])

    def dump_errors(self, accumulator: Dict[str, Any], batch: Dict[str, Any], predictions: Dict[str, Any],
                    sample_indices: List[int]) -> None:
        """
        Saves every misclassified sample of a test batch as an image, to look at the errors of the checkpoint:
        'Errors/<TRUE>-TO-<PREDICTED>/<sample>_<score>.png' in its evaluation folder, the score being the probability
        of the predicted class and the image captioned with both classes. At most 'maximum_per_class_pair' images are
        saved per pair of classes, the first ones met. The sequence labelling trainer overrides it to save the wrong
        frames.

        Args:
            accumulator (Dict[str, Any]): The evaluation accumulator, holding the 'errors_folder' and 'error_counts'.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The classifier outputs.
            sample_indices (List[int]): The indices of the samples kept for the evaluation.
        """
        probabilities = predictions["class_probabilities"].detach()
        test_dataset = self.test_dataloader.dataset
        for sample_index in sample_indices:
            true_class_index = int(batch["labels"][sample_index])
            predicted_class_index = int(probabilities[sample_index].argmax())
            if true_class_index == predicted_class_index:
                continue

            # The folder of the class pair, unless it is full
            pair_folder = self.reserve_error_image(accumulator=accumulator, true_class_index=true_class_index,
                                                   predicted_class_index=predicted_class_index)
            if pair_folder is None:
                continue

            # The image captioned with both classes, named after the sample and the score of the predicted class
            score = float(probabilities[sample_index, predicted_class_index])
            caption = f"{self.class_names[true_class_index]} -> {self.class_names[predicted_class_index]} {score:.3f}"
            image = add_caption_bar(image=self.build_input_image(batch=batch, sample_index=sample_index),
                                    caption=caption)
            sample_identifier = int(batch[SAMPLE_IDENTIFIERS_KEY][sample_index])
            sample_name = test_dataset.get_sample_name(sample_identifier=sample_identifier)
            cv2.imwrite(filename=str(pair_folder / f"{sample_name}_{score:.3f}.png"), img=image)

    def get_test_sample_classes(self) -> Dict[int, List[int]]:
        """
        Lists the classes of every test sample (image, sequence or pair), so that the fixed test samples drawn at every
        evaluation are spread over the classes.

        Returns:
            Dict[int, List[int]]: The class indices of every test sample, keyed by sample identifier.
        """
        return self.test_dataloader.dataset.get_sample_class_indices()

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws a fixed test sample for the sample images of the evaluation: its input (the image, or the frame grid of a
        sequence or of a pair) captioned with its true class and its predicted class with its probability, or in
        multi-label mode its true classes and its predicted classes with their probabilities, the caption in green
        when the prediction is right and in red when it is wrong. The sequence labelling trainer overrides it to
        caption every frame.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The classifier outputs.
            sample_index (int): The index of the sample in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sample.
        """
        probabilities = predictions["class_probabilities"][sample_index].detach().float().cpu()
        label = batch["labels"][sample_index].detach().cpu()

        # True and predicted classes: the most probable class, or the class decisions at the multi-label threshold
        if self.output_mode == "multi_class":
            predicted_class_index = int(probabilities.argmax())
            correct_prediction = predicted_class_index == int(label)
            prediction_text = (f"{self.class_names[int(label)]} -> {self.class_names[predicted_class_index]} "
                               f"{float(probabilities[predicted_class_index]):.3f}")
        else:
            predicted_decisions = probabilities >= MULTI_LABEL_THRESHOLD
            correct_prediction = bool((predicted_decisions == (label >= 0.5)).all())
            predicted_classes_text = ", ".join(
                f"{self.class_names[class_index]} {float(probabilities[class_index]):.3f}"
                for class_index in torch.nonzero(input=predicted_decisions).flatten().tolist()) or "none"
            prediction_text = f"{self.get_label_caption(label=label)} -> {predicted_classes_text}"

        # Input of the sample captioned with its prediction, in green when right and in red when wrong
        return add_caption_bar(image=self.build_input_image(batch=batch, sample_index=sample_index),
                               caption=prediction_text,
                               caption_colour=CORRECT_PREDICTION_COLOUR if correct_prediction else
                               WRONG_PREDICTION_COLOUR)

    def describe_evaluated_samples(self, accumulator: Dict[str, Any]) -> str:
        """
        Describes the evaluated test samples in the header of the report. The sequence labelling trainer overrides
        it to count the sequences and their labelled frames.

        Args:
            accumulator (Dict[str, Any]): The accumulated class probabilities and labels of the test split.

        Returns:
            str: The number of evaluated samples, e.g. '400 Test Images'.
        """
        return f"{sum(len(labels) for labels in accumulator['labels'])} Test {self.SAMPLE_NAME}"

    def compute_extra_headline_metrics(self, accumulator: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes the metrics of the task reported after the classification metrics, none for the classification
        task. The sequence labelling trainer overrides it to add the sequence exact match.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split.

        Returns:
            Dict[str, float]: The extra metrics, keyed by display name.
        """
        return {}

    def describe_extra_notes(self, accumulator: Dict[str, Any]) -> List[str]:
        """
        Gives the notes of the task printed in the report before the note of the error images, none for the
        classification task. The sequence labelling trainer overrides it to count the sequences and segments of its
        segment metrics.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions and labels of the test split.

        Returns:
            List[str]: The extra notes.
        """
        return []

    def describe_prediction(self, probabilities: torch.Tensor, label: torch.Tensor) -> Dict[str, Any]:
        """
        Describes the prediction of one evaluated sample (or frame) in the predictions file: its true and predicted
        classes, whether the prediction is right, and the probability of every class.

        Args:
            probabilities (torch.Tensor): The [K] class probabilities of the sample.
            label (torch.Tensor): The class index of the sample, or its multi-hot vector in multi-label mode.

        Returns:
            Dict[str, Any]: The 'true_class', 'predicted_class' and 'correct' columns ('true_classes' and
                'predicted_classes' in multi-label mode, the classes separated by ';'), then one
                'probability_<class name>' column per class.
        """
        # One class per sample, or the class decisions at the multi-label threshold
        if self.output_mode == "multi_class":
            true_class_index, predicted_class_index = int(label), int(probabilities.argmax())
            prediction = {"true_class": self.class_names[true_class_index],
                          "predicted_class": self.class_names[predicted_class_index],
                          "correct": int(true_class_index == predicted_class_index)}
        else:
            true_decisions, predicted_decisions = label >= 0.5, probabilities >= MULTI_LABEL_THRESHOLD
            prediction = {"true_classes": ";".join(self.class_names[class_index] for class_index in
                                                   torch.nonzero(input=true_decisions).flatten().tolist()),
                          "predicted_classes": ";".join(self.class_names[class_index] for class_index in
                                                        torch.nonzero(input=predicted_decisions).flatten().tolist()),
                          "correct": int(bool((true_decisions == predicted_decisions).all()))}

        # Probability of every class, with six decimals
        for class_index, class_name in enumerate(self.class_names):
            prediction[f"probability_{class_name}"] = f"{float(probabilities[class_index]):.6f}"
        return prediction

    def build_prediction_rows(self, accumulator: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Builds the rows of the predictions file, one per evaluated test sample: the columns naming the sample, given by
        the test dataset, followed by its prediction. The sequence labelling trainer overrides it to write one row
        per labelled frame.

        Args:
            accumulator (Dict[str, Any]): The class probabilities, labels and sample identifiers of the test split.

        Returns:
            List[Dict[str, Any]]: The rows of the predictions file, in the order of the evaluation.
        """
        probabilities = torch.cat(tensors=accumulator["probabilities"], dim=0)
        labels = torch.cat(tensors=accumulator["labels"], dim=0)
        sample_identifiers = torch.cat(tensors=accumulator["sample_identifiers"], dim=0)
        test_dataset = self.test_dataloader.dataset

        # The name of every sample, then its prediction
        rows = []
        for row_index in range(len(labels)):
            row = dict(test_dataset.describe_sample(sample_identifier=int(sample_identifiers[row_index])))
            row.update(self.describe_prediction(probabilities=probabilities[row_index], label=labels[row_index]))
            rows.append(row)
        return rows

    def write_predictions_file(self, accumulator: Dict[str, Any], evaluation_folder: Path) -> Path:
        """
        Saves the predictions of every evaluated test sample to 'predictions.csv' in the evaluation folder of the
        checkpoint, so that the errors and the class thresholds can be studied again without running the model.

        Args:
            accumulator (Dict[str, Any]): The class probabilities, labels and sample identifiers of the test split.
            evaluation_folder (Path): The evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/'.

        Returns:
            Path: The written predictions file.
        """
        rows = self.build_prediction_rows(accumulator=accumulator)
        predictions_path = evaluation_folder / "predictions.csv"
        with open(file=predictions_path, mode="w", newline="") as predictions_file:
            writer = csv.DictWriter(f=predictions_file, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        return predictions_path

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the classification metrics of the test split and organises them for the report: the headline metrics
        (with the extra metrics of the task), the error images as a note, the per-class table (shown with the per-class
        report, worst F1 first) and the per-class scalars. The threshold of every class reaching the target precision,
        the metrics of every class, the confusion matrix table (multi-class mode) and, when asked, the predictions of
        every sample are saved in the evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/', whatever
        the per-class report; the confusion matrix image only with it, also logged to TensorBoard.

        Args:
            accumulator (Dict[str, Any]): The class probabilities and labels of the test split.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        probabilities = torch.cat(tensors=accumulator["probabilities"], dim=0)
        labels = torch.cat(tensors=accumulator["labels"], dim=0)
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        evaluation_folder.mkdir(parents=True, exist_ok=True)
        number_classes = len(self.class_names)
        files, images = {}, {}

        # Headline metrics: from the confusion matrix for a single class per sample, from the class decisions at 0.5
        # for several classes per sample
        if self.output_mode == "multi_class":
            confusion_matrix = compute_confusion_matrix(predicted_labels=probabilities.argmax(dim=1), labels=labels,
                                                        number_classes=number_classes)
            metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)
            headline_metrics = {"Accuracy": metrics["accuracy"], "Balanced Accuracy": metrics["balanced_accuracy"],
                                "Macro F1": metrics["macro_f1"],
                                f"Top-{self.top_k} Accuracy": compute_top_k_accuracy(probabilities=probabilities,
                                                                                     labels=labels,
                                                                                     top_k=self.top_k)}
            # One-hot positives for the class thresholds, and the confusion matrix saved as a table
            positive_masks = torch.nn.functional.one_hot(input=labels, num_classes=number_classes).to(dtype=torch.bool)
            files["Confusion Matrix Table"] = save_confusion_matrix_table(confusion_matrix=confusion_matrix,
                                                                          class_names=self.class_names,
                                                                          output_folder=evaluation_folder)
            support_column = self.CLASS_SUPPORT_COLUMN
        else:
            # Several classes per sample: the metrics of the class decisions at the multi-label threshold
            metrics = compute_multi_label_metrics(probabilities=probabilities, labels=labels,
                                                  threshold=MULTI_LABEL_THRESHOLD)
            headline_metrics = {"Subset Accuracy": metrics["subset_accuracy"], "Micro F1": metrics["micro_f1"],
                                "Macro F1": metrics["macro_f1"]}
            positive_masks = labels >= 0.5
            support_column = "Positives"

        # Extra metrics of the task, e.g. the sequence exact match of the sequence labelling task
        headline_metrics.update(self.compute_extra_headline_metrics(accumulator=accumulator))

        # Threshold of every class reaching the target precision, saved for the deployment of the checkpoint
        class_thresholds = find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                                 target_precision=self.target_precision)
        thresholds_path = evaluation_folder / "class_thresholds.json"
        thresholds_path.write_text(data=json.dumps(obj={
            "target_precision": self.target_precision,
            "classes": {class_name: None if class_threshold is None else
                        {"threshold": class_threshold[0], "recall": class_threshold[1]}
                        for class_name, class_threshold in zip(self.class_names, class_thresholds)}}, indent=4))
        files["Class Thresholds"] = thresholds_path

        # Metrics of every class, saved whatever the per-class report so that they stay available for many classes
        class_rows = build_class_metrics_rows(class_names=self.class_names, metrics=metrics,
                                              class_thresholds=class_thresholds)
        files["Class Metrics"] = write_class_metrics_file(class_rows=class_rows, evaluation_folder=evaluation_folder)

        # Confusion matrix image of the per-class report, saved and logged to TensorBoard
        if self.output_mode == "multi_class" and self.per_class_report:
            files["Confusion Matrix Image"], images["Confusion Matrix"] = save_confusion_matrix_image(
                confusion_matrix=confusion_matrix, class_names=self.class_names, output_folder=evaluation_folder,
                title=f"Confusion Matrix · Iteration {iteration} · Accuracy {metrics['accuracy']:.4f} · Balanced "
                      f"Accuracy {metrics['balanced_accuracy']:.4f}")

        # Predictions of every evaluated sample, when asked
        if self.save_predictions:
            files["Predictions"] = self.write_predictions_file(accumulator=accumulator,
                                                               evaluation_folder=evaluation_folder)

        # Notes of the task, then the number of error images and their folder when some were saved
        notes = self.describe_extra_notes(accumulator=accumulator)
        if self.error_dumps_configuration["activated"]:
            error_counts = accumulator["error_counts"]
            notes.append(f"Error Images : {sum(error_counts.values())} in {len(error_counts)} class pairs, at most "
                         f"{self.error_dumps_configuration['maximum_per_class_pair']} per pair")
            if error_counts:
                files["Error Images"] = accumulator["errors_folder"]

        return {"evaluated_samples": self.describe_evaluated_samples(accumulator=accumulator),
                "metrics": headline_metrics, "notes": notes,
                "tables": [build_class_metrics_table(class_rows=class_rows, target_precision=self.target_precision,
                                                     support_column=support_column)],
                "scalars": dict(headline_metrics),
                "class_scalars": build_class_metric_scalars(class_names=self.class_names, metrics=metrics),
                "images": images, "files": files,
                "headline_metric": "Accuracy" if self.output_mode == "multi_class" else "Subset Accuracy"}

    def get_label_caption(self, label: torch.Tensor) -> str:
        """
        Describes the label of one sample with its class names, for the captions of the input viewer.

        Args:
            label (torch.Tensor): The class index, or the multi-hot vector in multi-label mode, of the sample.

        Returns:
            str: The class names of the sample, 'none' for a multi-label sample without class.
        """
        if self.output_mode == "multi_class":
            return self.class_names[int(label)]
        class_indices = torch.nonzero(input=label >= 0.5).flatten().tolist()
        return ", ".join(self.class_names[class_index] for class_index in class_indices) or "none"

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays an augmented training image next to a validation image, each with an information panel holding its
        class and its augmentations. With a view loss on, the second view of the training image, drawn as during the
        training from the image before its augmentation, is shown in a third window with the view transformations drawn
        for it.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed images in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        # The class of every image, followed by the augmentations drawn for it
        panel_lines = [[f"Class : {self.get_label_caption(label=batch['labels'][sample_index])}"] +
                       self.describe_input_augmentations(batch=batch, sample_index=sample_index)
                       for batch in [training_batch, validation_batch]]
        images = [training_batch["images"][sample_index], validation_batch["images"][sample_index]]
        window_titles = ["Training", "Validation"]

        # Second view of the training image, described right after it is drawn, before the next view overwrites the
        # drawn parameters
        if self.view_losses_activated:
            second_view = self.view_augmenter.draw_view(
                images=training_batch[VIEW_SOURCE_IMAGES_KEY][sample_index:sample_index + 1],
                view_index=SECOND_VIEW_INDEX)
            view_description = describe_view_transformations(
                transformations=self.view_augmenter.view_transformations[SECOND_VIEW_INDEX], number_images=1)[0]
            images.append(second_view[0])
            window_titles.append("Second View")
            panel_lines.append([panel_lines[0][0], "", "View transformations"] +
                               ([f"  {line}" for line in view_description] or ["  none"]))
        return view_input_data(images=images, bounding_boxes=None, labels=None, window_titles=window_titles,
                               display_input_size=self.input_image_size, panel_lines=panel_lines)
