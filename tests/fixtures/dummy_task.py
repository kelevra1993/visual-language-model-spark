"""
Dummy task (dataset, model and trainer) implementing the hooks of BaseTrainer of
vision_training/trainers/base_trainer.py with the smallest code, used by the trainer, checkpoint, optimization and
evaluation tests to check the general training loop without any real model or dataset.
"""

import numpy as np
import torch

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Iterable, List, Tuple

from vision_training.core.contracts import EvaluationResults
from vision_training.data.augmentations import AUGMENTATION_TYPES
from vision_training.models.task_model import TaskModel
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.visualization import add_caption_bar

# Side of the square random images of the dummy task
DUMMY_IMAGE_SIZE = 8


class DummyDataset(Dataset):
    """
    Tiny in-memory dataset of the dummy task used to test the general trainer on its own: seeded random images and
    binary labels, each sample carrying the sample identifier given for it, so that a test split can repeat samples
    in its last batch.
    """

    def __init__(self, sample_identifiers: List[int], seed: int) -> None:
        """
        Draws the images and labels of the samples.

        Args:
            sample_identifiers (List[int]): The identifier of every sample, repeated identifiers giving the same image.
            seed (int): The seed of the random images and labels.
        """
        generator = torch.Generator().manual_seed(seed)
        number_distinct_samples = max(sample_identifiers) + 1
        self.images = torch.rand(size=(number_distinct_samples, 3, DUMMY_IMAGE_SIZE, DUMMY_IMAGE_SIZE),
                                 generator=generator) * 255.0
        self.labels = torch.randint(low=0, high=2, size=(number_distinct_samples,), generator=generator)
        self.sample_identifiers = sample_identifiers

    def __len__(self) -> int:
        """
        Gives the number of samples, repeated ones included.

        Returns:
            int: The number of samples.
        """
        return len(self.sample_identifiers)

    def __getitem__(self, sample_index: int) -> Dict[str, Any]:
        """
        Returns one sample in the batch dictionary format of the framework.

        Args:
            sample_index (int): The index of the sample.

        Returns:
            Dict[str, Any]: The image, label and sample identifier of the sample.
        """
        sample_identifier = self.sample_identifiers[sample_index]
        return {"images": self.images[sample_identifier], "labels": self.labels[sample_identifier],
                "sample_identifiers": sample_identifier}


class DummyModel(TaskModel):
    """
    Tiny linear classifier of the dummy task, returning two losses so that the tests can check that the general
    trainer sums every '_loss' output.
    """

    def __init__(self) -> None:
        """
        Builds the linear layer mapping the flattened images to two class logits.
        """
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(in_features=3 * DUMMY_IMAGE_SIZE * DUMMY_IMAGE_SIZE, out_features=2)

    def forward(self, input_tensor: torch.Tensor, labels: torch.Tensor | None = None) -> Dict[str, Any]:
        """
        Classifies the images, and computes two losses when the labels are given.

        Args:
            input_tensor (torch.Tensor): The [B, 3, H, W] images in [0, 255].
            labels (torch.Tensor | None): The [B] labels, or None for predictions only.

        Returns:
            Dict[str, Any]: The 'class_logits', plus 'first_loss' and 'second_loss' (half of the first) with labels.
        """
        class_logits = self.linear(input_tensor.flatten(start_dim=1) / 255.0)
        outputs = {"class_logits": class_logits}
        if labels is not None:
            first_loss = nn.functional.cross_entropy(input=class_logits, target=labels)
            outputs.update({"first_loss": first_loss, "second_loss": 0.5 * first_loss})
        return outputs

    def print_summary(self) -> None:
        """
        Prints nothing: the dummy model has no architecture worth summarising.
        """

    def get_size_components(self) -> Dict[str, nn.Module]:
        """
        Lists the linear layer and the whole model for the size summary.

        Returns:
            Dict[str, nn.Module]: The components of the model.
        """
        return {"Linear": self.linear, "Total Model": self}


class DummyTrainer(BaseTrainer):
    """
    Trainer of the dummy task: the smallest implementation of the hooks of the general trainer, used to test its loop,
    checkpoints, resume, metric tracking and evaluation without any real model or dataset.
    """

    TASK_NAME = "dummy_task"

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: Path, datasets: Dict[str, Dataset]) -> None:
        """
        Keeps the datasets of the splits, then prepares the run through the general trainer.

        Args:
            experiment_configuration (Dict[str, Any]): The experiment settings, from build_dummy_configurations.
            model_configuration (Dict[str, Any]): The model settings, from build_dummy_configurations.
            configuration_path (Path): A configuration file, archived into the run folder.
            datasets (Dict[str, Dataset]): The 'train', 'validation' and 'test' datasets.
        """
        self.datasets = datasets
        super().__init__(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                         configuration_path=configuration_path)

    def get_run_name(self) -> str:
        """
        Gives a fixed run name, the dummy configuration having no backbone to describe.

        Returns:
            str: The run name.
        """
        return "dummy_run"

    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]:
        """
        Builds unshuffled loaders over the datasets of the three splits.

        Returns:
            Tuple[Iterable, Iterable, Iterable]: The training, validation and test loaders.
        """
        return tuple(DataLoader(dataset=self.datasets[split_name], batch_size=self.batch_size, shuffle=False)
                     for split_name in ["train", "validation", "test"])

    def build_model(self) -> DummyModel:
        """
        Builds the linear classifier.

        Returns:
            DummyModel: The untrained model.
        """
        return DummyModel()

    def get_tracked_metrics(self) -> Dict[str, str]:
        """
        Tracks the total loss, the two losses and the accuracy.

        Returns:
            Dict[str, str]: The display name of every tracked metric.
        """
        return {"total_loss": "Total Loss", "first_loss": "First Loss", "second_loss": "Second Loss",
                "accuracy": "Accuracy"}

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the samples of the split named by the folder.

        Args:
            split_data_folder (Path): The split folder of the configuration ('train', 'validation' or 'test').

        Returns:
            Tuple[int, str]: The number of samples and their description.
        """
        number_samples = len(self.datasets[Path(split_data_folder).name])
        return number_samples, f"{number_samples} samples"

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Moves the images and labels to the device of the model.

        Args:
            batch (Dict[str, Any]): The loader batch.

        Returns:
            Dict[str, Any]: The prepared batch.
        """
        return {"images": batch["images"].to(device=self.device, dtype=self.dtype),
                "labels": batch["labels"].to(device=self.device), "sample_identifiers": batch["sample_identifiers"]}

    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the model with the labels, so that it returns its losses.

        Args:
            batch (Dict[str, Any]): The prepared batch.

        Returns:
            Dict[str, Any]: The model outputs.
        """
        return self.model(input_tensor=batch["images"], labels=batch["labels"])

    def compute_batch_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Computes the accuracy of the batch from the logits of its forward pass.

        Args:
            batch (Dict[str, Any]): The prepared batch.
            model_outputs (Dict[str, Any]): The model outputs.

        Returns:
            Dict[str, torch.Tensor]: The accuracy of the batch.
        """
        predicted_labels = model_outputs["class_logits"].argmax(dim=1)
        return {"accuracy": (predicted_labels == batch["labels"]).float().mean()}

    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the model without labels.

        Args:
            batch (Dict[str, Any]): The prepared test batch.

        Returns:
            Dict[str, Any]: The model outputs.
        """
        return self.model(input_tensor=batch["images"])

    def create_evaluation_accumulator(self, iteration: int) -> Dict[str, Any]:
        """
        Creates the lists filled during the evaluation.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint, unused.

        Returns:
            Dict[str, Any]: The empty 'predicted_labels', 'labels' and 'sample_identifiers' lists.
        """
        return {"predicted_labels": [], "labels": [], "sample_identifiers": []}

    def accumulate_evaluation_batch(self, accumulator: Dict[str, Any], batch: Dict[str, Any],
                                    predictions: Dict[str, Any], sample_indices: List[int]) -> None:
        """
        Keeps the predicted label, label and identifier of every kept sample.

        Args:
            accumulator (Dict[str, Any]): The lists, extended in place.
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The model outputs.
            sample_indices (List[int]): The indices of the samples to keep.
        """
        for sample_index in sample_indices:
            accumulator["predicted_labels"].append(int(predictions["class_logits"][sample_index].argmax()))
            accumulator["labels"].append(int(batch["labels"][sample_index]))
            accumulator["sample_identifiers"].append(int(batch["sample_identifiers"][sample_index]))

    def compute_evaluation_results(self, accumulator: Dict[str, Any], iteration: int) -> EvaluationResults:
        """
        Computes the accuracy of the evaluated samples, and the recall of each of the two classes as the per-class
        table and scalars of the report.

        Args:
            accumulator (Dict[str, Any]): The accumulated predictions.
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            EvaluationResults: The number of samples, the accuracy and the per-class recalls.
        """
        number_samples = len(accumulator["labels"])
        accuracy = sum(predicted == label for predicted, label in zip(accumulator["predicted_labels"],
                                                                      accumulator["labels"])) / number_samples

        # Samples and recall of every class, None for a class without samples
        rows, class_scalars = [], {}
        for class_name, class_index in self.label_dictionary.items():
            class_predictions = [predicted for predicted, label in zip(accumulator["predicted_labels"],
                                                                       accumulator["labels"]) if label == class_index]
            recall = (sum(predicted == class_index for predicted in class_predictions) / len(class_predictions)
                      if class_predictions else None)
            rows.append([class_name, len(class_predictions), recall])
            if recall is not None:
                class_scalars[f"Recall/{class_name}"] = recall

        return {"evaluated_samples": f"{number_samples} Test Samples", "metrics": {"Accuracy": accuracy}, "notes": [],
                "tables": [{"title": "Per Class Recall", "columns": ["Class", "Samples", "Recall"], "rows": rows,
                            "sort_column": "Recall", "per_class": True, "legend": ""}],
                "scalars": {"Accuracy": accuracy}, "class_scalars": class_scalars, "images": {}, "files": {},
                "headline_metric": "Accuracy"}

    def get_test_sample_classes(self) -> Dict[int, List[int]]:
        """
        Gives the label of every distinct test sample, so that the fixed test samples are spread over the two classes.

        Returns:
            Dict[int, List[int]]: The label of every test sample, in a list, keyed by sample identifier.
        """
        test_dataset = self.datasets["test"]
        return {sample_identifier: [int(test_dataset.labels[sample_identifier])]
                for sample_identifier in set(test_dataset.sample_identifiers)}

    def get_test_sample_name(self, sample_identifier: int) -> str:
        """
        Names a test sample after its identifier, the dummy dataset having no file names.

        Args:
            sample_identifier (int): The identifier of the sample.

        Returns:
            str: The name 'sample_<identifier>'.
        """
        return f"sample_{sample_identifier}"

    def build_sample_image(self, batch: Dict[str, Any], predictions: Dict[str, Any], sample_index: int) -> np.ndarray:
        """
        Draws a test sample as its image under a caption giving its predicted label.

        Args:
            batch (Dict[str, Any]): The prepared test batch.
            predictions (Dict[str, Any]): The model outputs.
            sample_index (int): The index of the sample in the batch.

        Returns:
            np.ndarray: The [24 + H, W', 3] uint8 BGR image of the sample.
        """
        image = np.ascontiguousarray(batch["images"][sample_index].detach().cpu().to(dtype=torch.uint8)
                                     .permute(1, 2, 0).numpy())
        predicted_label = int(predictions["class_logits"][sample_index].argmax())
        return add_caption_bar(image=image, caption=f"Predicted label {predicted_label}")

    def show_input_samples(self, training_batch: Dict[str, Any], validation_batch: Dict[str, Any],
                           sample_index: int) -> bool:
        """
        Displays nothing, the dummy task has no viewer.

        Args:
            training_batch (Dict[str, Any]): The augmented training batch.
            validation_batch (Dict[str, Any]): The validation batch.
            sample_index (int): The index of the samples.

        Returns:
            bool: Always True, which stops the viewer.
        """
        return True


def build_dummy_configurations(experiment_folder: Path, number_iterations: int) -> Tuple[Dict[str, Any],
                                                                                        Dict[str, Any]]:
    """
    Builds the experiment and model settings read by the general trainer for the dummy task: batches of 2 samples,
    saves at iterations 2 (intermediate) and 4 (with test evaluation), a console summary and a validation batch every 2
    iterations, the per-class evaluation report on, no sample images, no MLflow tracking, no augmentation, no
    pretrained backbone, and plain Adam at a constant learning rate.

    Args:
        experiment_folder (Path): The experiments root folder of the run.
        number_iterations (int): The number of training iterations of the run.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations.
    """
    experiment_configuration = {"task": "dummy_task", "experiment_folder": experiment_folder,
                                "train_data_folder": Path("train"),
                                "validation_data_folder": Path("validation"), "test_data_folder": Path("test"),
                                "batch_size": 2, "information_dump": 2, "weight_saving_iterations": 4,
                                "intermediate_weight_saving_iterations": 2, "number_iterations": number_iterations,
                                "validation_iteration_ratio": 2, "learning_rate": 0.01,
                                "resume_training": True,
                                "evaluation": {"per_class_report": True, "number_sample_images": 0},
                                "tracking": {"activated": False, "tracking_uri": None, "experiment_name": None,
                                             "log_artifacts": True},
                                "diagnosis_modes": {"verbose": False, "compute_receptive_field": False,
                                                    "compute_model_size": False, "view_input_images": False}}
    augmentation_types = {name: {"activated": False, "probability": 0.0} for name in AUGMENTATION_TYPES}
    model_configuration = {"Data": {"label_dictionary": {"negative": 0, "positive": 1},
                                    "image_settings": {"size": DUMMY_IMAGE_SIZE},
                                    "data_augmentation": {"activated": False, "probability": 0.0,
                                                          "independent_frames": False,
                                                          "shuffle_frames": {"activated": False, "probability": 0.0},
                                                          "types": augmentation_types}},
                           "Backbone": {"pretrained": {"experiment_folder": None, "iteration": None, "freeze": False,
                                                       "freeze_normalization_parameters": False,
                                                       "learning_rate_multiplier": 1.0}},
                           "Optimization": {"optimizer": "adam", "weight_decay": 0.0,
                                            "learning_rate_schedule": "constant", "warmup_iterations": 0,
                                            "decay_iterations": 100000, "minimum_learning_rate_ratio": 0.01,
                                            "gradient_clipping_norm": None, "mixed_precision": False}}
    return experiment_configuration, model_configuration
