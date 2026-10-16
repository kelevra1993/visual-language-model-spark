"""
Trainer of the self-supervised task: the loaders of the 'labels.json' manifests keeping every image, the two random
views of every training and validation image made before the forward pass (parameters drawn on the CPU, images
transformed on the device), the loss and batch metrics of the method (SimCLR or Barlow Twins) tracked on every batch,
and the evaluation of the representations of the checkpoints on the test split (k-nearest neighbour classification of
the labelled test images, effective rank and standard deviation of the representations). Registered for the
"self_supervised" task; the backbone it trains then initialises the other tasks through 'Backbone.pretrained'.
"""

import numpy as np
import torch

from pathlib import Path
from torch.nn import functional as torch_functional
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision_training.core.contracts import (IGNORED_LABEL_INDEX, LOSS_KEY_SUFFIX, SAMPLE_IDENTIFIERS_KEY,
                                            EvaluationResults, SelfSupervisedBatch)
from vision_training.data.label_mapping import get_class_names
from vision_training.data.loaders import get_classification_dataloaders
from vision_training.data.split_description import count_split_samples
from vision_training.data.view_augmentations import ViewAugmenter
from vision_training.evaluation.classification.classification_metrics import (compute_confusion_matrix,
                                                                              compute_metrics_from_confusion_matrix)
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.evaluation.self_supervised.representation_metrics import (compute_effective_rank,
                                                                               compute_nearest_neighbour_predictions)
from vision_training.losses.self_supervised_losses import compute_embedding_standard_deviation
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.visualization import add_caption_bar, convert_image_to_array, view_input_data

# Metrics tracked on every training and validation batch for each self-supervised method, with their display names
METHOD_TRACKED_METRICS = {"simclr": {"total_loss": "Total Loss", "info_nce_loss": "InfoNCE Loss",
                                     "positive_pair_accuracy": "Positive Pair Accuracy",
                                     "embedding_standard_deviation": "Embedding Standard Deviation"},
                          "barlow_twins": {"total_loss": "Total Loss", "barlow_twins_loss": "Barlow Twins Loss",
                                           "cross_correlation_diagonal": "Diagonal Correlation",
                                           "cross_correlation_off_diagonal": "Off-Diagonal Correlation",
                                           "embedding_standard_deviation": "Embedding Standard Deviation"}}

# Name of the per-class table of the k-nearest neighbour classification and of its sort column, worst classes first
NEAREST_NEIGHBOUR_TABLE_TITLE = "Per Class k-NN Metrics"
NEAREST_NEIGHBOUR_SORT_COLUMN = "F1"

# Caption of an image without mapped category in the sample images and the input viewer
UNLABELLED_CAPTION = "unlabelled"

# Offset of the seed of the views of every sample image, added to its sample identifier, so that every evaluation and
# both the trained and the averaged weights compare the same two views of an image
SAMPLE_VIEWS_SEED = 20_000


class SelfSupervisedTrainer(BaseTrainer):
    """
    Trainer of the self-supervised task: the backbone, the global pooling and the projection head learn from two
    random views of every image (Data.view_augmentation), pulled together by the loss of 'ProjectionHead.method'.
    The views replace the data augmentation of the other tasks, so 'Data.data_augmentation' stays off and the views
    are drawn in the forward pass of the training and the validation batches alike, the validation loss needing views
    too. Every evaluated checkpoint is scored without training any classifier: its representations classify the
    labelled test images by their nearest neighbours among the other labelled test images (accuracy, balanced
    accuracy, macro F1 and per-class metrics), and their effective rank and standard deviation watch the collapse,
    the only metrics of a test split without labels.
    """

    TASK_NAME = "self_supervised"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the self-supervised settings used by the hooks and builds the view augmenter, then prepares the
        training run through the general trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, ProjectionHead).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Method of the run, giving its tracked metrics, and the number of neighbours voting in the evaluation
        self.method = model_configuration["ProjectionHead"]["method"]
        self.number_neighbours = experiment_configuration["evaluation"]["number_neighbours"]

        # Random views of the training and validation images, their parameters drawn on the CPU and the images
        # transformed on the device
        self.view_augmenter = ViewAugmenter(
            view_augmentation_configuration=model_configuration["Data"]["view_augmentation"],
            image_size=model_configuration["Data"]["image_settings"]["size"])
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds the loaders of the three splits from their classification manifests, every image being kept and the
        training and validation batches full, and names the classes of the k-nearest neighbour evaluation after the
        categories of the three manifests, so that a class absent from the test manifest keeps its category name.

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders.
        """
        loaders = get_classification_dataloaders(experiment_configuration=self.experiment_configuration,
                                                 model_configuration=self.model_configuration,
                                                 batch_size=self.batch_size, self_supervised=True)

        # Categories of the three manifests by identifier, the first split naming a category giving its name
        merged_categories = {}
        for loader in loaders:
            for category in loader.dataset.categories:
                merged_categories.setdefault(category["id"], category)
        class_names = get_class_names(categories=list(merged_categories.values()),
                                      label_dictionary=self.model_configuration["Data"]["label_dictionary"])
        self.class_names = [class_names[class_index] for class_index in range(len(class_names))]
        return loaders

    def list_class_names(self) -> Optional[List[str]]:
        """
        Gives the class names of the k-nearest neighbour evaluation for the record of the data of the run, None for a
        dataset without classes.

        Returns:
            Optional[List[str]]: The class names, in class index order, or None.
        """
        return self.class_names if self.class_names else None

    def build_model(self) -> TaskModel:
        """
        Builds the self-supervised model of the task from the model registry.

        Returns:
            TaskModel: The untrained model.
        """
        return TASK_MODEL_CLASSES[self.TASK_NAME](configuration=self.model_configuration, device=self.device,
                                                  dtype=self.dtype, verbose=self.diagnosis_modes["verbose"])

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Lists the loss and the batch metrics of the method of the run, tracked on every training and validation batch.

        Returns:
            Dict[str, str]: The display name of every tracked metric.
        """
        return METHOD_TRACKED_METRICS[self.method]

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the images and categories of a split from its labels.json manifest.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of images and their description with the number of categories.
        """
        number_samples, number_categories = count_split_samples(split_data_folder=split_data_folder)
        return number_samples, f"{number_samples} images, {number_categories} categories"

    def prepare_batch(self, batch: Dict[str, Any]) -> SelfSupervisedBatch:
        """
        Moves the images and labels of a batch to the device, the images in the precision of the model and the labels
        as int64 class indices.

        Args:
            batch (Dict[str, Any]): The batch dictionary yielded by a self-supervised loader.

        Returns:
            SelfSupervisedBatch: The batch ready for the views and the model.
        """
        return {**batch, "images": batch["images"].to(device=self.device, dtype=self.dtype),
                "labels": batch["labels"].to(device=self.device, dtype=torch.int64)}

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Leaves the training batch unchanged: the random views of the self-supervised task are drawn in forward_pass,
        for the training and the validation batches alike, and replace the data augmentation of the other tasks.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The same batch.
        """
        return batch

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draws the two random views of every image of a training or validation batch, then runs the model on both views
        so that it returns the loss of the method.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The model outputs, with the loss and the batch metrics of the method.
        """
        first_views, second_views = self.view_augmenter.create_views(images=batch["images"])
        return self.model(input_tensor=first_views, second_view_tensor=second_views)

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encodes the test images without views, as in deployment.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The representations and embeddings of the images.
        """
        return self.model(input_tensor=batch["images"])

    def compute_batch_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Gives the batch metrics of the method returned by the model with its loss (positive pair accuracy, diagonal and
        off-diagonal correlations, embedding standard deviation), without any extra computation.

        Args:
            batch (Dict[str, Any]): The prepared batch.
            model_outputs (Dict[str, Any]): The model outputs.

        Returns:
            Dict[str, torch.Tensor]: The tracked batch metrics of the method.
        """
        return {metric_key: model_outputs[metric_key] for metric_key in self.tracked_metrics_mapping
                if metric_key in model_outputs and not metric_key.endswith(LOSS_KEY_SUFFIX)}

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the lists of representations, labels and sample identifiers filled during the evaluation of the test
        split.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            Dict[str, Any]: The empty 'representations', 'labels' and 'sample_identifiers' lists.
        """
        return {"representations": [], "labels": [], "sample_identifiers": []}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the representations, the labels and the identifiers of the kept samples of a test batch on the CPU, the
        k-nearest neighbour evaluation comparing every test image with all the others once the split is encoded.

        Args:
            accumulator (Dict[str, Any]): The 'representations', 'labels' and 'sample_identifiers' lists, extended in
                place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The representations and embeddings of the test images.
            sample_indices (List[int]): The indices of the samples to keep.
        """
        accumulator["representations"].append(predictions["representations"][sample_indices].detach().float().cpu())
        accumulator["labels"].append(batch["labels"][sample_indices].detach().cpu())
        accumulator["sample_identifiers"].append(batch[SAMPLE_IDENTIFIERS_KEY][sample_indices].detach().cpu())

    def compute_nearest_neighbour_results(self, representations: torch.Tensor, labels: torch.Tensor,
                                          evaluation_folder: Path) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Classifies the labelled test images by their nearest neighbours among the other labelled test images and
        scores the classification like a classifier: accuracy, balanced accuracy, macro F1, and the precision, recall
        and F1 of every class, saved in class_metrics.csv and shown in the per-class table and scalars.

        Args:
            representations (torch.Tensor): The [N, F] representations of the labelled test images.
            labels (torch.Tensor): The [N] class indices of the labelled test images.
            evaluation_folder (Path): The evaluation folder of the checkpoint.

        Returns:
            Tuple[Dict[str, float], Dict[str, Any]]: The global k-NN metrics keyed by display name, and the per-class
                outputs: the 'table', the 'class_scalars' and the 'class_metrics_path'.
        """
        number_classes = len(self.class_names)
        predicted_labels = compute_nearest_neighbour_predictions(representations=representations, labels=labels,
                                                                 number_neighbours=self.number_neighbours,
                                                                 number_classes=number_classes)
        confusion_matrix = compute_confusion_matrix(predicted_labels=predicted_labels, labels=labels,
                                                    number_classes=number_classes)
        class_metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)
        global_metrics = {"k-NN Accuracy": class_metrics["accuracy"],
                          "k-NN Balanced Accuracy": class_metrics["balanced_accuracy"],
                          "k-NN Macro F1": class_metrics["macro_f1"]}

        # Metrics of every class, in the file, the table (worst F1 first) and the TensorBoard scalars
        class_rows, table_rows, class_scalars = [], [], {}
        for class_index, class_name in enumerate(self.class_names):
            number_class_samples = int(class_metrics["support"][class_index])
            class_values = {metric_name: class_metrics[metric_name][class_index]
                            for metric_name in ["precision", "recall", "f1"]}
            class_rows.append({"class_index": class_index, "class": class_name, "samples": number_class_samples,
                               **class_values})
            table_rows.append([class_name, number_class_samples] + [class_values[metric_name] if number_class_samples
                                                                    else None for metric_name in class_values])
            class_scalars.update({f"k-NN {metric_name.capitalize()}/{class_name}": value
                                  for metric_name, value in class_values.items()})
        table = {"title": NEAREST_NEIGHBOUR_TABLE_TITLE,
                 "columns": ["Class", "Samples", "Precision", "Recall", NEAREST_NEIGHBOUR_SORT_COLUMN],
                 "rows": table_rows, "sort_column": NEAREST_NEIGHBOUR_SORT_COLUMN, "per_class": True,
                 "legend": f"k-nearest neighbour classification ({self.number_neighbours} neighbours) of the labelled "
                           f"test images by the other ones. Worst F1 first, ties in class order. — : no labelled "
                           f"test image of the class."}
        class_metrics_path = write_class_metrics_file(class_rows=class_rows, evaluation_folder=evaluation_folder)
        return global_metrics, {"table": table, "class_scalars": class_scalars,
                                "class_metrics_path": class_metrics_path}

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Scores the representations of the test split and organises them for the report: the k-nearest neighbour
        metrics of the labelled test images when there are at least two of them, the headline metric being then the
        k-NN accuracy, and the effective rank and standard deviation of the representations of every test image, the
        effective rank being the headline metric of a test split without labels. The per-class k-NN metrics are saved
        in class_metrics.csv in the evaluation folder of the checkpoint, '<run>/Evaluation/Iteration_<n>/'.

        Args:
            accumulator (Dict[str, Any]): The representations and labels of the test split.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The results of the evaluation.
        """
        representations = torch.cat(tensors=accumulator["representations"], dim=0)
        labels = torch.cat(tensors=accumulator["labels"], dim=0)
        evaluation_folder = self.get_evaluation_folder(iteration=iteration)
        evaluation_folder.mkdir(parents=True, exist_ok=True)

        # Collapse metrics of the representations of every test image, labelled or not
        representation_metrics = {"Effective Rank": compute_effective_rank(representations=representations),
                                  "Representation Standard Deviation": float(compute_embedding_standard_deviation(
                                      embeddings=representations))}

        # k-nearest neighbour classification of the labelled test images, first in the report when it is computed
        labelled_mask = labels != IGNORED_LABEL_INDEX
        number_labelled_images = int(labelled_mask.sum())
        metrics, tables, class_scalars, files = {}, [], {}, {}
        headline_metric = "Effective Rank"
        notes = [f"Labelled Test Images : {number_labelled_images} of {len(labels)}"]
        if number_labelled_images >= 2 and self.class_names:
            metrics, class_outputs = self.compute_nearest_neighbour_results(
                representations=representations[labelled_mask], labels=labels[labelled_mask],
                evaluation_folder=evaluation_folder)
            tables.append(class_outputs["table"])
            class_scalars = class_outputs["class_scalars"]
            files["Class Metrics"] = class_outputs["class_metrics_path"]
            headline_metric = "k-NN Accuracy"
            notes.append(f"k-NN : {self.number_neighbours} neighbours among the other labelled test images")
        else:
            notes.append("k-NN : skipped, fewer than two labelled test images")
        metrics.update(representation_metrics)

        return {"evaluated_samples": f"{len(labels)} Test {self.SAMPLE_NAME}", "metrics": metrics, "notes": notes,
                "tables": tables, "scalars": dict(metrics), "class_scalars": class_scalars, "images": {},
                "files": files, "headline_metric": headline_metric}

    def get_test_sample_classes(self) -> Dict[int, List[int]]:
        """
        Lists the class of every test image, so that the fixed test images drawn at every evaluation are spread over
        the classes, the unlabelled images forming their own group.

        Returns:
            Dict[int, List[int]]: The class indices of every test image, keyed by sample identifier.
        """
        return self.test_dataloader.dataset.get_sample_class_indices()

    def get_label_caption(self, label: torch.Tensor) -> str:
        """
        Names the class of an image for the captions of the sample images and the input viewer.

        Args:
            label (torch.Tensor): The class index of the image, IGNORED_LABEL_INDEX without mapped category.

        Returns:
            str: The class name, or 'unlabelled'.
        """
        class_index = int(label)
        return UNLABELLED_CAPTION if class_index == IGNORED_LABEL_INDEX else self.class_names[class_index]

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws a fixed test image for the sample images of the evaluation: the image next to two random views of it,
        captioned with its class and the cosine similarity of the representations of both views, which a backbone
        invariant to the views brings close to 1. The views are drawn from a seed of the image, in a copy of the random
        state that leaves the one of the training unchanged, so that every evaluation and both the trained and the
        averaged weights compare the same two views.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The representations and embeddings of the test images.
            sample_index (int): The index of the image in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sample.
        """
        image = batch["images"][sample_index:sample_index + 1]
        with torch.random.fork_rng(devices=[] if self.device.type == "cpu" else [self.device]):
            torch.manual_seed(seed=SAMPLE_VIEWS_SEED + int(batch[SAMPLE_IDENTIFIERS_KEY][sample_index]))
            first_view, second_view = self.view_augmenter.create_views(images=image)

        # Similarity of the representations of both views, the model being in evaluation mode
        view_representations = self.model(input_tensor=torch.cat(tensors=[first_view, second_view],
                                                                 dim=0).to(dtype=self.dtype))["representations"]
        view_similarity = float(torch_functional.cosine_similarity(x1=view_representations[0:1],
                                                                   x2=view_representations[1:2], dim=1))

        # The image and its two views side by side, captioned with the class and the similarity
        sample_image = np.concatenate([convert_image_to_array(image=image[0]),
                                       convert_image_to_array(image=first_view[0]),
                                       convert_image_to_array(image=second_view[0])], axis=1)
        caption = (f"{self.get_label_caption(label=batch['labels'][sample_index])} | Views Similarity "
                   f"{view_similarity:.3f}")
        return add_caption_bar(image=sample_image, caption=caption)

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays the two random views of a training image side by side, exactly as the network receives them, each
        captioned with the class of the image. The validation images get views drawn the same way, so only the
        training image is shown.

        Args:
            training_batch (Dict[str, Any]): The training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU, unused.
            sample_index (int): The index of the displayed image in the training batch.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        first_view, second_view = self.view_augmenter.create_views(
            images=training_batch["images"][sample_index:sample_index + 1].to(dtype=torch.float32))
        caption = f"Class : {self.get_label_caption(label=training_batch['labels'][sample_index])}"
        return view_input_data(images=[first_view[0], second_view[0]], bounding_boxes=None, labels=None,
                               window_titles=["View 1", "View 2"], display_input_size=self.input_image_size,
                               captions=[caption, caption])
