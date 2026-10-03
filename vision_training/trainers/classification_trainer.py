import csv
import json
import torch

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from vision_training.core.contracts import SAMPLE_IDENTIFIERS_KEY, ClassificationBatch, EvaluationResults
from vision_training.data.loaders import get_classification_dataloaders
from vision_training.data.split_description import count_split_samples
from vision_training.evaluation.classification.classification_metrics import (compute_confusion_matrix,
                                                                              compute_metrics_from_confusion_matrix,
                                                                              compute_multi_label_metrics,
                                                                              compute_top_k_accuracy)
from vision_training.evaluation.classification.confusion_matrix import save_confusion_matrix
from vision_training.evaluation.classification.threshold_finder import find_class_thresholds
from vision_training.losses.classification_losses import compute_balanced_class_weights
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.visualization import view_input_data

# Probability from which a class is predicted in multi-label mode
MULTI_LABEL_THRESHOLD = 0.5


class ClassificationTrainer(BaseTrainer):
    """
    Trainer of the classification task: the image classifier trained on 'labels.json' manifests (Native loader), with
    its loss and its accuracy tracked on every batch, and every evaluated checkpoint scored on the test split with the
    accuracy, balanced accuracy, top-k accuracy, per-class precision, recall and F1, the confusion matrix and the
    per-class thresholds reaching the target precision.

    The sequence classification trainer extends it, the classes, loss, batch metric and evaluation being the same
    once a sequence is classified as a whole.
    """

    TASK_NAME = "classification"

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
        # evaluation settings: the k of the top-k accuracy, the precision targeted by the per-class thresholds and
        # whether the predictions of every test sample are saved
        self.output_mode = model_configuration["ClassificationHead"]["output_mode"]
        self.class_weighting = model_configuration["ClassificationHead"]["class_weighting"]
        self.top_k = experiment_configuration["evaluation"]["top_k"]
        self.target_precision = experiment_configuration["evaluation"]["target_precision"]
        self.save_predictions = experiment_configuration["evaluation"]["save_predictions"]
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
                                                 batch_size=self.batch_size, number_of_workers=4)
        training_dataset = loaders[0].dataset
        self.class_names = [training_dataset.class_names[class_index]
                            for class_index in range(training_dataset.number_classes)]
        return loaders

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

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the losses and the accuracy tracked on every training and validation batch.

        Returns:
            Dict[str, str]: The display name of every tracked metric.
        """
        return {"total_loss": "Total Loss", "classification_loss": "Classification Loss", "accuracy": "Accuracy"}

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

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the classifier with the labels of the batch, so that it returns its classification loss.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The classifier outputs, with its classification loss.
        """
        return self.model(input_tensor=batch["images"], labels=batch["labels"])

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
        class decisions (probability above 0.5) that are right.

        Args:
            batch (Dict[str, Any]): The prepared batch.
            model_outputs (Dict[str, Any]): The classifier outputs.

        Returns:
            Dict[str, torch.Tensor]: The 'accuracy' of the batch.
        """
        with torch.no_grad():
            if self.output_mode == "multi_class":
                correct_predictions = model_outputs["class_logits"].argmax(dim=-1) == batch["labels"]
            else:
                correct_predictions = ((model_outputs["class_probabilities"] >= MULTI_LABEL_THRESHOLD) ==
                                       (batch["labels"] >= 0.5))
            return {"accuracy": correct_predictions.to(dtype=torch.float32).mean()}

    def create_evaluation_accumulator(self) -> Dict[str, Any]:
        """
        Creates the lists of class probabilities, labels and sample identifiers filled during the evaluation of the
        test split.

        Returns:
            Dict[str, Any]: The empty 'probabilities', 'labels' and 'sample_identifiers' lists.
        """
        return {"probabilities": [], "labels": [], "sample_identifiers": []}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the class probabilities, the labels and the identifiers of the kept samples of a test batch on the CPU,
        the identifiers naming the samples in the predictions file.

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

    def describe_evaluated_samples(self, accumulator: Dict[str, Any]) -> str:
        """
        Describes the evaluated test samples in the first line of the report. The sequence labelling trainer
        overrides it to count the windows and their labelled frames.

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
        Computes the classification metrics of the test split and formats them for the report: the headline metrics
        (with the extra metrics of the task), the precision, recall and F1 of every class, and the threshold of every
        class reaching the target precision with the recall it keeps. The confusion matrix (multi-class mode), the
        thresholds and, when asked, the predictions of every sample are saved in the evaluation folder of the
        checkpoint, '<run>/Evaluation/Iteration_<n>/'.

        Args:
            accumulator (Dict[str, Any]): The class probabilities and labels of the test split.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The lines, scalars and files of the evaluation.
        """
        probabilities = torch.cat(tensors=accumulator["probabilities"], dim=0)
        labels = torch.cat(tensors=accumulator["labels"], dim=0)
        evaluation_folder = self.experiment_folder / "Evaluation" / f"Iteration_{iteration}"
        number_classes = len(self.class_names)
        files = {}

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
            positive_masks = torch.nn.functional.one_hot(input=labels, num_classes=number_classes).to(dtype=torch.bool)
            files.update(save_confusion_matrix(confusion_matrix=confusion_matrix, class_names=self.class_names,
                                               output_folder=evaluation_folder))
        else:
            metrics = compute_multi_label_metrics(probabilities=probabilities, labels=labels,
                                                  threshold=MULTI_LABEL_THRESHOLD)
            headline_metrics = {"Subset Accuracy": metrics["subset_accuracy"], "Micro F1": metrics["micro_f1"],
                                "Macro F1": metrics["macro_f1"]}
            positive_masks = labels >= 0.5

        # Extra metrics of the task, e.g. the sequence exact match of the sequence labelling task
        headline_metrics.update(self.compute_extra_headline_metrics(accumulator=accumulator))

        # Threshold of every class reaching the target precision, saved for the deployment of the checkpoint
        class_thresholds = find_class_thresholds(probabilities=probabilities, positive_masks=positive_masks,
                                                 target_precision=self.target_precision)
        evaluation_folder.mkdir(parents=True, exist_ok=True)
        thresholds_path = evaluation_folder / "class_thresholds.json"
        thresholds_path.write_text(data=json.dumps(obj={
            "target_precision": self.target_precision,
            "classes": {class_name: None if class_threshold is None else
                        {"threshold": class_threshold[0], "recall": class_threshold[1]}
                        for class_name, class_threshold in zip(self.class_names, class_thresholds)}}, indent=4))
        files["Class Thresholds"] = thresholds_path

        # Predictions of every evaluated sample, when asked
        if self.save_predictions:
            files["Predictions"] = self.write_predictions_file(accumulator=accumulator,
                                                               evaluation_folder=evaluation_folder)

        # Headline line, then one line for the per-class metrics and one for the per-class thresholds
        lines = [f"Iteration {iteration} : {self.describe_evaluated_samples(accumulator=accumulator)}",
                 "  " + " | ".join(f"{metric_name} {metric_value:.4f}"
                                   for metric_name, metric_value in headline_metrics.items()),
                 "  Precision / Recall / F1 Per Class : " + " | ".join(
                     f"{class_name} {metrics['precision'][class_index]:.4f} / {metrics['recall'][class_index]:.4f} / "
                     f"{metrics['f1'][class_index]:.4f}" for class_index, class_name in enumerate(self.class_names)),
                 f"  Threshold / Recall At Precision {self.target_precision} Per Class : " + " | ".join(
                     f"{class_name} " + ("unreachable" if class_threshold is None else
                                         f"{class_threshold[0]:.4f} / {class_threshold[1]:.4f}")
                     for class_name, class_threshold in zip(self.class_names, class_thresholds))]

        # The same values as TensorBoard scalars, to follow them over the training
        scalars = {metric_name: metric_value for metric_name, metric_value in headline_metrics.items()}
        for class_index, class_name in enumerate(self.class_names):
            for metric_name in ["precision", "recall", "f1"]:
                scalars[f"{metric_name.capitalize()}/{class_name}"] = metrics[metric_name][class_index]

        return {"lines": lines, "scalars": scalars, "files": files}

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
        Displays an augmented training image next to a validation image, each captioned with its class.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU.
            sample_index (int): The index of the displayed images in their batches.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        return view_input_data(
            images=[training_batch["images"][sample_index], validation_batch["images"][sample_index]],
            bounding_boxes=None, labels=None, window_titles=["Training", "Validation"],
            display_input_size=self.input_image_size,
            captions=[f"Class : {self.get_label_caption(label=training_batch['labels'][sample_index])}",
                      f"Class : {self.get_label_caption(label=validation_batch['labels'][sample_index])}"])
