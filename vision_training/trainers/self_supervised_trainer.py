"""
Trainer of the self-supervised task: the loaders of the 'labels.json' manifests keeping every image, the two random
views of every training and validation image drawn on the device before the forward pass by the data augmentation with
the self-supervised strengths (DataAugmenter.create_views of Data.data_augmentation), the loss and batch metrics of the
method (SimCLR or Barlow Twins) tracked on every batch, and the evaluation of the representations of the checkpoints on
the test split (k-nearest neighbour classification of the labelled test images by a fixed sample of labelled training
images, encoded with the evaluated weights, effective rank and standard deviation of the representations). Registered for the "self_supervised" task; the backbone it trains then initialises the other tasks
through 'Backbone.pretrained'.
"""

import numpy as np
import random
import torch

from collections import Counter
from pathlib import Path
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vision_training.core.contracts import (IGNORED_LABEL_INDEX, LOSS_KEY_SUFFIX, SAMPLE_IDENTIFIERS_KEY,
                                            EvaluationResults, SelfSupervisedBatch)
from vision_training.data.grouped_batch_sampler import GroupedBatchSampler
from vision_training.data.label_mapping import get_class_names
from vision_training.data.loaders import build_native_loader, get_classification_dataloaders
from vision_training.data.split_description import count_split_samples
from vision_training.evaluation.classification.class_metrics_report import (build_class_metric_scalars,
                                                                             build_class_metrics_rows,
                                                                             build_class_metrics_table)
from vision_training.evaluation.classification.classification_metrics import (compute_confusion_matrix,
                                                                              compute_metrics_from_confusion_matrix)
from vision_training.evaluation.evaluation_report import write_class_metrics_file
from vision_training.evaluation.recall_intervals import (CLOPPER_PEARSON_DESCRIPTION, build_recall_interval_table,
                                                         compute_clopper_pearson_intervals)
from vision_training.evaluation.self_supervised.representation_metrics import (compute_effective_rank,
                                                                               compute_nearest_neighbour_predictions,
                                                                               select_reference_samples)
from vision_training.losses.self_supervised_losses import compute_embedding_standard_deviation
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from vision_training.trainers.base_trainer import AVERAGED_WEIGHTS_LABEL, BaseTrainer
from vision_training.utilities.os_utilities import print_color
from vision_training.utilities.visualization import (add_caption_bar, build_batch_mosaic, convert_image_to_array,
                                                     show_viewer_windows, view_input_data)

# Metrics tracked on every training and validation batch for each self-supervised method, with their display names
METHOD_TRACKED_METRICS = {"simclr": {"total_loss": "Total Loss", "info_nce_loss": "InfoNCE Loss",
                                     "positive_pair_accuracy": "Positive Pair Accuracy",
                                     "embedding_standard_deviation": "Embedding Standard Deviation"},
                          "barlow_twins": {"total_loss": "Total Loss", "barlow_twins_loss": "Barlow Twins Loss",
                                           "cross_correlation_diagonal": "Diagonal Correlation",
                                           "cross_correlation_off_diagonal": "Off-Diagonal Correlation",
                                           "embedding_standard_deviation": "Embedding Standard Deviation"}}

# Names of the per-class table and of the per-class recall table of the k-nearest neighbour classification, worst
# classes first
NEAREST_NEIGHBOUR_TABLE_TITLE = "Per Class k-NN Metrics"
NEAREST_NEIGHBOUR_RECALL_TABLE_TITLE = "Per Class k-NN Recall"

# Caption of an image without mapped category in the sample images and the input viewer
UNLABELLED_CAPTION = "unlabelled"

# Title of the window of the batch mosaic of the input viewer
BATCH_WINDOW_TITLE = "Training Batch"

# Offset of the seed of the views of every sample image, added to its sample identifier, so that every evaluation and
# both the trained and the averaged weights compare the same two views of an image
SAMPLE_VIEWS_SEED = 20_000


class SelfSupervisedTrainer(BaseTrainer):
    """
    Trainer of the self-supervised task: the backbone, the global pooling and the projection head learn from two random
    views of every image, pulled together by the loss of 'ProjectionHead.method'. The views are drawn by the data
    augmentation ('Data.data_augmentation', with the stronger SELF_SUPERVISED_AUGMENTATION_STRENGTHS) in the forward
    pass of the training and the validation batches alike, the validation loss needing views too. Every evaluated
    checkpoint is scored without training any classifier: its representations classify the labelled test images by their
    nearest neighbours among the reference images, a fixed sample of labelled training images encoded with the same
    weights at every evaluation (accuracy, balanced accuracy, macro F1 and per-class metrics), and their effective rank
    and standard deviation watch the collapse, the only metrics of a test split without labels.
    """

    TASK_NAME = "self_supervised"

    # The training loader leaves out the last incomplete batch (see get_classification_dataloaders), the losses
    # comparing the samples of a full batch
    DROPS_LAST_TRAINING_BATCH = True

    # The data augmentation draws the two views of every image, with the self-supervised strengths
    SELF_SUPERVISED_AUGMENTATION = True

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: str | Path) -> None:
        """
        Reads the self-supervised settings used by the hooks, prepares the training run through the general trainer,
        which builds the data augmentation drawing the views, then chooses the reference images of the k-nearest
        neighbour evaluation among the training images (build_reference_dataloader).

        Args:
            experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
            model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, ProjectionHead).
            configuration_path (str | Path): The file path to the original YAML configuration used for this run.
        """
        # Method of the run, giving its tracked metrics, the number of neighbours voting in the evaluation and the
        # largest number of reference images of a class they are chosen among
        self.method = model_configuration["ProjectionHead"]["method"]
        self.number_neighbours = experiment_configuration["evaluation"]["number_neighbours"]
        self.number_reference_images_per_class = experiment_configuration["evaluation"][
            "number_reference_images_per_class"]
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)
        self.reference_dataloader, self.reference_notes = self.build_reference_dataloader()

    def build_reference_dataloader(self) -> Tuple[Optional[DataLoader], List[str]]:
        """
        Chooses the reference images of the k-nearest neighbour evaluation, the labelled training images that every
        labelled test image is compared with: at most 'evaluation.number_reference_images_per_class' images of every
        class, the same at every evaluation (select_reference_samples), read in their order without augmentation like
        the test images. Every class of the training split is kept, a class without labelled test image being a
        possible wrong prediction as for a classifier of every class, and a test class without labelled training image
        is announced, its test images never being classified correctly. Nothing is chosen without classes or without
        labelled test image, the k-NN being then skipped.

        Returns:
            Tuple[Optional[DataLoader], List[str]]: The loader of the reference images (None without k-NN), and the
                notes describing them in the report of every evaluation.

        Raises:
            ValueError: If the test split holds labelled images but the training split none, the k-NN having then no
                reference image to compare them with.
        """
        test_classes = {sample["class_indices"][0] for sample in self.test_dataloader.dataset.samples
                        if len(sample["class_indices"]) == 1}
        if not self.class_names or not test_classes:
            return None, []

        # Reference images of every class of the training split, none for a training split without labels
        training_dataset = self.train_dataloader.dataset
        reference_samples = select_reference_samples(
            samples=training_dataset.samples, number_images_per_class=self.number_reference_images_per_class)
        if not reference_samples:
            raise ValueError(f"The k-nearest neighbour evaluation compares the labelled test images with labelled "
                             f"training images, but the training split "
                             f"({self.experiment_configuration['train_data_folder']}) holds none: label some training "
                             f"images, or give the test split no labels for an evaluation by the effective rank only.")
        reference_classes = {sample["class_indices"][0] for sample in reference_samples}

        # Test classes the reference images cannot predict, and reference classes without labelled test image
        notes = [f"k-NN : {self.number_neighbours} neighbours among {len(reference_samples)} labelled training "
                 f"images of {len(reference_classes)} classes ({self.number_reference_images_per_class} per class at "
                 f"most)"]
        missing_test_classes = sorted(test_classes - reference_classes)
        if missing_test_classes:
            missing_class_names = ", ".join(self.class_names[class_index] for class_index in missing_test_classes[:10])
            warning = (f"k-NN : {len(missing_test_classes)} test classes without labelled training image, their test "
                       f"images never classified correctly ({missing_class_names}"
                       f"{', ...' if len(missing_test_classes) > 10 else ''})")
            print_color(color="yellow", output=warning)
            notes.append(warning)
        number_distractor_classes = len(reference_classes - test_classes)
        if number_distractor_classes:
            notes.append(f"k-NN : {number_distractor_classes} training classes without labelled test image, kept as "
                         f"possible predictions")

        # Loader of the reference images in their order, encoded like the test images at every evaluation
        reference_dataset = training_dataset.select_samples(samples=reference_samples)
        return build_native_loader(dataset=reference_dataset, batch_size=self.batch_size, shuffle=False), notes

    def list_dataloaders(self) -> List[Iterable]:
        """
        Lists the training, validation and test loaders and the loader of the reference images of the k-nearest
        neighbour evaluation, so that the workers of every one stop at the end of the training.

        Returns:
            List[Iterable]: The loaders of the trainer.
        """
        reference_loaders = [] if self.reference_dataloader is None else [self.reference_dataloader]
        return super().list_dataloaders() + reference_loaders

    def encode_reference_images(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the reference images of the k-nearest neighbour evaluation with the weights being evaluated (the
        trained ones, or the averaged ones within use_averaged_weights), through predict_batch and within the
        evaluation context like the test images, so that both are pooled the same way and without augmentation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The [M, F] representations of the reference images, on the CPU, and their
                [M] class indices.
        """
        # Representations and classes of every batch of reference images, encoded like a test batch
        representations, labels = [], []
        weights_description = f" ({AVERAGED_WEIGHTS_LABEL})" if self.evaluating_averaged_weights else ""
        with self.evaluation_context(), torch.no_grad():
            for reference_batch in tqdm(self.reference_dataloader, total=len(self.reference_dataloader),
                                        desc=f"k-NN Reference Images{weights_description}"):
                reference_batch = self.prepare_batch(batch=reference_batch)
                predictions = self.predict_batch(batch=reference_batch)
                representations.append(predictions["representations"].detach().float().cpu())
                labels.append(reference_batch["labels"].detach().cpu())
        return torch.cat(tensors=representations, dim=0), torch.cat(tensors=labels, dim=0)

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
                                                  dtype=self.dtype)

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
        Leaves the training batch unchanged: the two random views of every image are drawn in forward_pass, for the
        training and the validation batches alike.

        Args:
            batch (Dict[str, Any]): The training batch.

        Returns:
            Dict[str, Any]: The same batch.
        """
        return batch

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draws the two random views of every image of a training or validation batch with the data augmentation
        (DataAugmenter.create_views, in float32 even within the mixed precision autocast), then runs the model on both
        views so that it returns the loss of the method.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The model outputs, with the loss and the batch metrics of the method.
        """
        first_views, second_views, _, _ = self.data_augmenter.create_views(images=batch["images"])
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
        k-nearest neighbour evaluation comparing every labelled test image with the reference images once the split is
        encoded.

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
        Classifies the labelled test images by their nearest neighbours among the reference images, encoded with the
        evaluated weights (encode_reference_images), and scores the classification like a classifier: accuracy,
        balanced accuracy, macro F1, and the precision, recall (with its exact confidence interval over the labelled
        test images of the class) and F1 of every class, saved in class_metrics.csv and shown in the per-class table,
        the per-class recall table and the scalars.

        Args:
            representations (torch.Tensor): The [N, F] representations of the labelled test images.
            labels (torch.Tensor): The [N] class indices of the labelled test images.
            evaluation_folder (Path): The evaluation folder of the checkpoint.

        Returns:
            Tuple[Dict[str, float], Dict[str, Any]]: The global k-NN metrics keyed by display name, and the per-class
                outputs: the 'tables' (the per-class table, then the per-class recall table), the 'class_scalars' and
                the 'class_metrics_path'.
        """
        # Votes of the reference images, encoded with the weights being evaluated, for every labelled test image
        number_classes = len(self.class_names)
        reference_representations, reference_labels = self.encode_reference_images()
        predicted_labels = compute_nearest_neighbour_predictions(
            query_representations=representations, reference_representations=reference_representations,
            reference_labels=reference_labels, number_neighbours=self.number_neighbours, number_classes=number_classes)
        confusion_matrix = compute_confusion_matrix(predicted_labels=predicted_labels, labels=labels,
                                                    number_classes=number_classes)
        class_metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)
        global_metrics = {"k-NN Accuracy": class_metrics["accuracy"],
                          "k-NN Balanced Accuracy": class_metrics["balanced_accuracy"],
                          "k-NN Macro F1": class_metrics["macro_f1"]}

        # Metrics of every class without thresholds, the exact interval of its recall over its labelled test images, in
        # the file, the tables (worst F1, then worst recall first) and the TensorBoard scalars
        class_metrics["recall_lower"], class_metrics["recall_upper"] = compute_clopper_pearson_intervals(
            true_positives=np.array(class_metrics["true_positives"]), trials=np.array(class_metrics["support"]))
        class_rows = build_class_metrics_rows(class_names=self.class_names, metrics=class_metrics)
        table = build_class_metrics_table(class_rows=class_rows, target_precision=None, support_column="Samples",
                                          title=NEAREST_NEIGHBOUR_TABLE_TITLE,
                                          legend_introduction=f"k-nearest neighbour classification "
                                                              f"({self.number_neighbours} neighbours) of the labelled "
                                                              f"test images by labelled training images. ",
                                          support_description="labelled test image")
        recall_table = build_recall_interval_table(class_rows=class_rows, support_key="samples",
                                                   support_column="Samples", recall_columns=[("recall", "Recall")],
                                                   interval_description=CLOPPER_PEARSON_DESCRIPTION,
                                                   title=NEAREST_NEIGHBOUR_RECALL_TABLE_TITLE)
        class_scalars = build_class_metric_scalars(class_names=self.class_names, metrics=class_metrics,
                                                   tag_prefix="k-NN ")
        class_metrics_path = write_class_metrics_file(class_rows=class_rows, evaluation_folder=evaluation_folder)
        return global_metrics, {"tables": [table, recall_table], "class_scalars": class_scalars,
                                "class_metrics_path": class_metrics_path}

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Scores the representations of the test split and organises them for the report: the k-nearest neighbour
        metrics of the labelled test images when there is at least one, the headline metric being then the k-NN
        accuracy, and the effective rank and standard deviation of the representations of every test image, the
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
        if self.reference_dataloader is not None and number_labelled_images > 0:
            metrics, class_outputs = self.compute_nearest_neighbour_results(
                representations=representations[labelled_mask], labels=labels[labelled_mask],
                evaluation_folder=evaluation_folder)
            tables.extend(class_outputs["tables"])
            class_scalars = class_outputs["class_scalars"]
            files["Class Metrics"] = class_outputs["class_metrics_path"]
            headline_metric = "k-NN Accuracy"
            notes.extend(self.reference_notes)
        else:
            notes.append("k-NN : skipped, no labelled test image")
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
        averaged weights compare the same two views. The augmentation draws with Python's random module and the noise
        and the plasma maps with PyTorch, so both generators are seeded and both training states restored.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The representations and embeddings of the test images.
            sample_index (int): The index of the image in the batch.

        Returns:
            np.ndarray: The [H, W, 3] uint8 BGR image of the sample.
        """
        image = batch["images"][sample_index:sample_index + 1]
        views_seed = SAMPLE_VIEWS_SEED + int(batch[SAMPLE_IDENTIFIERS_KEY][sample_index])
        training_random_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[] if self.device.type == "cpu" else [self.device]):
                torch.manual_seed(seed=views_seed)
                random.seed(views_seed)
                first_view, second_view, _, _ = self.data_augmenter.create_views(images=image)
        finally:
            random.setstate(training_random_state)

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

    def show_input_batch(self, training_batch: Dict[str, Any], batch_index: int) -> bool:
        """
        Shows a training batch built by the GroupedBatchSampler in one window before the views of its images: the
        mosaic of its images in the order of the batch, cut into its blocks (build_batch_mosaic), under a header
        describing the batch (describe_input_batch). Nothing is shown for random batches, when 'Data.grouped_batches'
        is off or the training split has no groups.

        Args:
            training_batch (Dict[str, Any]): The training batch as the loader gives it, on the CPU.
            batch_index (int): The position of the batch in the pass of the viewer over the training split.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        # Random batches, without the grouped batch sampler, have no blocks to show
        batch_sampler = self.train_dataloader.batch_sampler
        if not isinstance(batch_sampler, GroupedBatchSampler):
            return False

        # Groups of every image of the batch, read from the dataset by the sample identifiers
        sample_identifiers = training_batch[SAMPLE_IDENTIFIERS_KEY]
        if isinstance(sample_identifiers, torch.Tensor):
            sample_identifiers = sample_identifiers.tolist()
        sample_records = self.train_dataloader.dataset.sample_records
        image_groups = [sample_records[sample_identifier]["groups"] for sample_identifier in sample_identifiers]

        # Mosaic of the blocks of the batch under the header describing it, shown alone in its window
        title, lines, block_captions, image_captions = self.describe_input_batch(
            image_groups=image_groups, batch_sampler=batch_sampler, batch_index=batch_index)
        mosaic = build_batch_mosaic(images=training_batch["images"], block_size=batch_sampler.images_per_block,
                                    block_captions=block_captions, image_captions=image_captions, title=title,
                                    lines=lines)
        return show_viewer_windows(window_images=[mosaic], window_titles=[BATCH_WINDOW_TITLE])

    def describe_input_batch(self, image_groups: List[Dict[str, str]], batch_sampler: GroupedBatchSampler,
                             batch_index: int) -> Tuple[str, List[str], List[str], Optional[List[str]]]:
        """
        Writes the texts of the batch mosaic of the input viewer: its title gives the position of the batch in the
        pass and whether it is grouped or a random batch of the leftover pool, and its lines the batches of a pass and
        the values of the keys of 'Data.grouped_batches' in the batch. A grouped batch captions every block with its
        block value; a random batch counts the images of every value, numbers its chunks of images_per_block images
        and captions every image with its values. The captions list every value found, so that a block mixing values
        would show it.

        Args:
            image_groups (List[Dict[str, str]]): The "groups" of every image of the batch, in its order.
            batch_sampler (GroupedBatchSampler): The sampler of the training split, giving whether the batch is grouped.
            batch_index (int): The position of the batch in the current pass of the sampler.

        Returns:
            Tuple[str, List[str], List[str], Optional[List[str]]]: The title, the lines of the header, the caption of
                every block, and the caption of every image (None for a grouped batch).
        """
        # Block and batch values of every image of the batch, and whether the sampler built it from blocks
        grouped_batches_configuration = self.data_configuration["grouped_batches"]
        block_key, batch_key = grouped_batches_configuration["block_key"], grouped_batches_configuration["batch_key"]
        block_size = batch_sampler.images_per_block
        block_values = [groups[block_key] for groups in image_groups]
        batch_values = None if batch_key is None else [groups[batch_key] for groups in image_groups]
        grouped = batch_sampler.grouped_batch_flags[batch_index]

        # Position and kind of the batch, then the batches of every pass
        title = (f"Training Batch {batch_index + 1} / {len(batch_sampler)} : "
                 f"{'Grouped' if grouped else 'Random (Leftover Pool)'}")
        lines = [f"Pass : {batch_sampler.number_grouped_batches} grouped and {batch_sampler.number_leftover_batches} "
                 f"random batches of {batch_sampler.batch_size} images", ""]
        block_starts = range(0, len(image_groups), block_size)

        # A grouped batch: its batch value, its blocks, and the block value of every block as its caption
        if grouped:
            if batch_values is not None:
                lines.append(f"Batch key '{batch_key}' : {', '.join(sorted(set(batch_values)))}")
            lines.append(f"Block key '{block_key}' : {len(block_starts)} blocks of {block_size} images")
            block_captions = [", ".join(sorted(set(block_values[start:start + block_size]))) for start in block_starts]
            return title, lines, block_captions, None

        # A random batch: the images of every value, most frequent first, and the values of every image
        for key_role, group_key, values in [("Batch", batch_key, batch_values), ("Block", block_key, block_values)]:
            if values is None:
                continue
            value_counts = sorted(Counter(values).items(), key=lambda value_count: (-value_count[1], value_count[0]))
            lines += [f"{key_role} key '{group_key}' : {len(value_counts)} values",
                      "  " + ", ".join(f"{value} x {count}" for value, count in value_counts)]
        block_captions = [f"Images {start + 1} to {min(start + block_size, len(image_groups))}"
                          for start in block_starts]
        image_captions = (block_values if batch_values is None else
                          [f"{batch_value} / {block_value}"
                           for batch_value, block_value in zip(batch_values, block_values)])
        return title, lines, block_captions, image_captions

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays the two random views of a training image side by side, exactly as the network receives them, each
        next to an information panel holding the class of the image and the augmentations drawn for the view with
        their values (DataAugmenter.describe_report). The validation images get views drawn the same way, so only the
        training image is shown.

        Args:
            training_batch (Dict[str, Any]): The training batch, on the CPU.
            validation_batch (Dict[str, Any]): The validation batch, on the CPU, unused.
            sample_index (int): The index of the displayed image in the training batch.

        Returns:
            bool: True if the user asked to stop viewing.
        """
        first_view, second_view, first_reports, second_reports = self.data_augmenter.create_views(
            images=training_batch["images"][sample_index:sample_index + 1].to(dtype=torch.float32))

        # The class of the image, then the augmentations of each view under their header
        class_line = f"Class : {self.get_label_caption(label=training_batch['labels'][sample_index])}"
        panel_lines = [[class_line, "", "View augmentations"] + self.data_augmenter.describe_report(report=reports[0])
                       for reports in [first_reports, second_reports]]
        return view_input_data(images=[first_view[0], second_view[0]], bounding_boxes=None, labels=None,
                               window_titles=["View 1", "View 2"], display_input_size=self.input_image_size,
                               panel_lines=panel_lines)
