"""
Smoke tests of SelfSupervisedTrainer of vision_training/trainers/self_supervised_trainer.py on a synthetic dataset: a
Barlow Twins training saves, evaluates and resumes, with the tracked metrics of the method, the k-nearest neighbour
report of the labelled test images, its per-class metrics file and the sample images of two views checked, a dataset
mixing labelled and unlabelled images is trained on every image and evaluated on the labelled ones, the classes of
the evaluation are named after the categories of the three manifests (the validation one read without its loader in
a training without validation), the labelled test images are classified by the reference training images and not by
each other, the k-NN metrics above MAXIMUM_DENSE_CONFUSION_CLASSES classes come from the sparse confusion matrix and
equal those of the dense one, the reference images are the same for every trainer of a run and bounded per class, the
k-NN rates of the report and the positive pair accuracy of the SimCLR console summary are written as percentages, and
the input viewer of the grouped batches finds the groups of every image of a batch by its identifier.
"""

import csv
import json
import pytest
import re
import torch
import yaml

from collections import Counter
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.core.contracts import IGNORED_LABEL_INDEX, SAMPLE_IDENTIFIERS_KEY
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.grouped_batch_sampler import build_grouped_batch_sampler
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.evaluation.classification.classification_metrics import compute_confusion_matrix
from vision_training.evaluation.report_formatting import build_report_renderables, render_report_text
from vision_training.trainers import self_supervised_trainer
from vision_training.trainers.self_supervised_trainer import (METHOD_PERCENT_TRACKED_METRICS, METHOD_TRACKED_METRICS,
                                                              SelfSupervisedTrainer)
from vision_training.utilities.visualization import CAPTION_BAR_HEIGHT
from tests.data.classification.test_classification_dataset import write_split
from tests.fixtures.evaluation_checks import check_sample_images, read_results_blocks

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "self_supervised" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 64

# Metrics of the evaluation of a test split with labels, in the order of the report, the k-NN rates first
LABELLED_EVALUATION_METRICS = ["k-NN Accuracy", "k-NN Balanced Accuracy", "k-NN Macro F1", "Effective Rank",
                               "Representation Standard Deviation"]
NEAREST_NEIGHBOUR_RATES = LABELLED_EVALUATION_METRICS[:3]


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                              head_settings: Dict[str, Any], moving_average_activated: bool = False,
                              grouped_batches_activated: bool = False, run_name_suffix: Optional[str] = None) -> Path:
    """
    Writes a copy of the synthetic shapes self-supervised configuration adapted to a quick training smoke test: the
    generated dataset, 64 pixel images, batches of 4 images, a small projection head and a saving schedule that
    triggers an intermediate save, a save with test evaluation and a validation pass within a handful of iterations.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        head_settings (Dict[str, Any]): The 'ProjectionHead' settings replaced for the test.
        moving_average_activated (bool): Whether the run keeps a moving average of its weights, evaluated next to the
            trained weights. Defaults to False.
        grouped_batches_activated (bool): Whether the batches gather blocks of 2 images of the same camera, from the
            "groups" of the samples. Defaults to False.
        run_name_suffix (Optional[str]): The suffix ending the run name, None for none. Defaults to None.

    Returns:
        Path: The path of the written configuration file.
    """
    configuration = yaml.safe_load(stream=SYNTHETIC_CONFIGURATION_PATH.read_text())

    # Run in the temporary directory, on the generated splits, with a saving schedule fitting a few iterations
    experiment_configuration = configuration["ExperimentConfiguration"]
    experiment_configuration.update({"experiment_folder": str(temporary_directory / "experiments"),
                                     "train_data_folder": str(dataset_folder / "train"),
                                     "validation_data_folder": str(dataset_folder / "validation"),
                                     "test_data_folder": str(dataset_folder / "test"), "batch_size": 4,
                                     "information_dump": 2, "weight_saving_iterations": 4,
                                     "intermediate_weight_saving_iterations": 2, "number_iterations": number_iterations,
                                     "validation_iteration_ratio": 2,
                                     "run_name_suffix": run_name_suffix})

    # Small images, the label mapping of the generated dataset and a small head
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    configuration["ProjectionHead"].update({"fully_connected": [32, 16], **head_settings})
    configuration["Optimization"]["exponential_moving_average"]["activated"] = moving_average_activated

    # Grouped batches of two blocks of 2 images of one camera when asked for
    configuration["Data"]["grouped_batches"].update({"activated": grouped_batches_activated, "block_key": "camera",
                                                     "images_per_block": 2, "batch_key": None})

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> SelfSupervisedTrainer:
    """
    Loads a configuration file and builds the SelfSupervisedTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        SelfSupervisedTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return SelfSupervisedTrainer(experiment_configuration=experiment_configuration,
                                 model_configuration=model_configuration, configuration_path=configuration_path)


@pytest.mark.parametrize(argnames=("method", "run_name_ending"),
                         argvalues=[("barlow_twins", "_BT-OD0.005")])
def test_self_supervised_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                              method: str, run_name_ending: str) -> None:
    """
    Verifies the self-supervised training pipeline of each method on a generated synthetic dataset: 5 iterations save
    the intermediate, evaluated and final checkpoints with the loss and batch metrics of the method in TensorBoard,
    the evaluation of the test split is reported (Results.txt block with the k-NN metrics written as percentages,
    effective rank and representation standard deviation with 4 decimals, the k-NN accuracy as headline metric) with
    its per-class metrics file and the sample images of two views of every test image, and a second run with the same
    settings resumes from the last checkpoint with exactly its weights. No training step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
        method (str): The self-supervised method, 'ProjectionHead.method'.
        run_name_ending (str): The end of the run name of the method.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 6}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5,
        head_settings={"method": method}))
    assert trainer.run_name.startswith("SSL_") and trainer.run_name.endswith(run_name_ending)
    assert trainer.class_names == list(SHAPE_CATEGORIES)
    assert trainer.metric_tracker.percent_metric_keys == METHOD_PERCENT_TRACKED_METRICS[method]
    data_record = json.loads(s=(trainer.experiment_folder / "data.json").read_text())
    assert [class_record["class_name"] for class_record in data_record["classes"]] == list(SHAPE_CATEGORIES)
    with patch.object(target=trainer, attribute="report_evaluation", wraps=trainer.report_evaluation) as report_mock:
        trainer.run_training_loop()
    results = report_mock.call_args.kwargs["results"]

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]

    # Loss and batch metrics of the method tracked in training and validation
    for split_name in ["Train", "Validation"]:
        split_events = EventAccumulator(path=str(trainer.tensorboard_directory / split_name))
        split_events.Reload()
        assert set(METHOD_TRACKED_METRICS[method].values()) <= set(split_events.Tags()["scalars"]), split_name

    # Report of the evaluated checkpoint: the k-NN metrics of the 6 labelled test images first, then the collapse
    # metrics of the representations, all finite
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [(4, "6 Test Images")]
    assert list(results["metrics"]) == LABELLED_EVALUATION_METRICS and results["headline_metric"] == "k-NN Accuracy"
    assert all(torch.isfinite(input=torch.tensor(data=value)) for value in results["metrics"].values())
    assert 1.0 <= results["metrics"]["Effective Rank"] <= 6.0
    assert "Labelled Test Images : 6 of 6" in results["notes"]
    assert "Per Class k-NN Metrics" in results_blocks[0]["text"]
    assert len(results["class_scalars"]) == 3 * len(trainer.class_names)

    # The k-NN rates and the columns of both k-NN tables written as percentages, the collapse metrics with 4 decimals
    assert results["percent_metrics"] == NEAREST_NEIGHBOUR_RATES
    first_grid_line = results_blocks[0]["text"].splitlines()[0]
    assert f"{100.0 * results['metrics']['k-NN Accuracy']:.2f} %" in first_grid_line
    assert f"{results['metrics']['Effective Rank']:.4f}" in results_blocks[0]["text"]
    assert [table["percent_columns"] for table in results["tables"]] == [["Precision", "Recall", "F1"],
                                                                        ["Recall", "95 %\nLower", "95 %\nUpper"]]

    # Files of the evaluated checkpoint: the per-class k-NN metrics of every class, and the sample images
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir()) == ["Samples", "class_metrics.csv"]
    with open(file=evaluation_folder / "class_metrics.csv", mode="r", newline="") as class_metrics_file:
        class_rows = list(csv.DictReader(f=class_metrics_file))
    assert [row["class"] for row in class_rows] == trainer.class_names
    assert list(class_rows[0]) == ["class_index", "class", "samples", "precision", "recall", "recall_lower",
                                   "recall_upper", "f1"]
    assert sum(int(row["samples"]) for row in class_rows) == 6

    # Every test image is a sample image (8 asked for 6 images), next to its two views below the caption bar, a long
    # caption widening the image
    check_sample_images(evaluation_folder=evaluation_folder, results=results, number_sample_images=6,
                        sample_kind="images", correct_predictions=None)
    for sample_image in results["sample_images"].values():
        assert sample_image.shape[0] == CAPTION_BAR_HEIGHT + SMOKE_IMAGE_SIZE
        assert sample_image.shape[1] >= 3 * SMOKE_IMAGE_SIZE

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2,
        head_settings={"method": method}))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])
    resumed_trainer.run_training_loop()

    # The training loop catches errors and keeps going, so make sure that no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_dataset_mixing_labelled_and_unlabelled_images_is_trained_and_evaluated(tmp_path: Path,
                                                                                 capsys: pytest.CaptureFixture) -> None:
    """
    Verifies a self-supervised run on manifests mixing labelled images and images without 'category_id', like the
    hidden collections of the ficha pretraining layouts: every image is trained on, the record of the data counts the
    labelled images of every split only, and the k-nearest neighbour evaluation scores the labelled test images while
    the report covers every test image.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 6}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    # Remove the category of every other sample of every split, which leaves 4, 2 and 3 labelled images
    for split_name in ["train", "validation", "test"]:
        labels_path = dataset_folder / split_name / "labels.json"
        labels = json.loads(s=labels_path.read_text())
        labels["samples"] = [sample if sample_index % 2 == 0 else {"id": sample["id"], "file_name": sample["file_name"]}
                             for sample_index, sample in enumerate(labels["samples"])]
        labels_path.write_text(data=json.dumps(obj=labels))

    # Every training image kept, the record of the data counting the labelled images of every split only
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=4, head_settings={}))
    assert len(trainer.train_dataloader.dataset) == 8
    class_records = json.loads(s=(trainer.experiment_folder / "data.json").read_text())["classes"]
    assert {split_name: sum(class_record["images"][split_name] for class_record in class_records)
            for split_name in ["train", "validation", "test"]} == {"train": 4, "validation": 2, "test": 3}
    with patch.object(target=trainer, attribute="report_evaluation", wraps=trainer.report_evaluation) as report_mock:
        trainer.run_training_loop()
    results = report_mock.call_args.kwargs["results"]

    # k-NN metrics of the 3 labelled test images first, the report covering the 6 test images
    assert list(results["metrics"]) == LABELLED_EVALUATION_METRICS and results["headline_metric"] == "k-NN Accuracy"
    assert "Labelled Test Images : 3 of 6" in results["notes"]
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [(4, "6 Test Images")]
    with open(file=trainer.experiment_folder / "Evaluation" / "Iteration_4" / "class_metrics.csv", mode="r",
              newline="") as class_metrics_file:
        assert sum(int(row["samples"]) for row in csv.DictReader(f=class_metrics_file)) == 3

    # The training loop catches errors and keeps going, so make sure that no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


@pytest.mark.parametrize(argnames="builds_validation_loader", argvalues=[True, False])
def test_class_names_come_from_the_categories_of_every_split(tmp_path: Path, builds_validation_loader: bool) -> None:
    """
    Verifies that the classes of the k-nearest neighbour evaluation are named after the categories of the three
    manifests, so that a class named in the training or validation manifest only keeps its category name instead of
    its index, the validation manifest being read for its categories when its loader is not built (a training without
    validation), so that the class names never depend on the validation of the run.

    Args:
        tmp_path (Path): The pytest temporary directory.
        builds_validation_loader (bool): Whether the validation loader is built, as BaseTrainer decides it.
    """
    # Training manifest naming the first two classes, validation manifest the first and third, test manifest the
    # first one only
    split_categories = {"train": [{"id": 1, "name": "circle"}, {"id": 5, "name": "triangle"}],
                        "validation": [{"id": 1, "name": "circle"}, {"id": 3, "name": "square"}],
                        "test": [{"id": 1, "name": "circle"}]}
    for split_name, categories in split_categories.items():
        write_split(split_folder=tmp_path / split_name, categories=categories,
                    samples=[{"id": sample_identifier, "file_name": f"{sample_identifier}.png", "category_id": 1}
                             for sample_identifier in range(1, 3)])

    # Only the attributes read by build_dataloaders, without building the model of the trainer
    trainer = SimpleNamespace(
        experiment_configuration={f"{split_name}_data_folder": str(tmp_path / split_name)
                                  for split_name in split_categories},
        model_configuration={"Data": {"image_settings": {"size": 32, "keep_ratio": True},
                                      "label_dictionary": {1: 0, 5: 1, 3: 2},
                                      "grouped_batches": {"activated": False, "block_key": "camera",
                                                          "images_per_block": 2, "batch_key": None}}}, batch_size=2,
        builds_validation_loader=builds_validation_loader)
    loaders = SelfSupervisedTrainer.build_dataloaders(self=trainer)
    assert (loaders[1] is not None) == builds_validation_loader
    assert trainer.class_names == ["circle", "triangle", "square"]


def test_test_images_are_classified_by_the_reference_training_images(tmp_path: Path) -> None:
    """
    Verifies that the k-nearest neighbour evaluation classifies every labelled test image by the reference training
    images encoded with the evaluated weights, never by the other test images: with one test image per class, as in
    the held-out images of ImageNet-22k, two test images closer to each other than to the reference images of their
    classes are both classified correctly, where a vote among the test images would swap their classes.

    Args:
        tmp_path (Path): The pytest temporary directory, receiving class_metrics.csv.
    """
    # Two test images of classes 0 and 1, and one reference image of each class, encoded by the stub of the trainer
    reference_representations = torch.tensor(data=[[1.0, -0.5], [0.7, 0.7]])
    reference_labels = torch.tensor(data=[0, 1])
    trainer = SimpleNamespace(class_names=["circle", "square"], number_neighbours=1,
                              encode_reference_images=lambda: (reference_representations, reference_labels))
    global_metrics, _ = SelfSupervisedTrainer.compute_nearest_neighbour_results(
        self=trainer, representations=torch.tensor(data=[[1.0, 0.0], [0.95, 0.31]]), labels=torch.tensor(data=[0, 1]),
        evaluation_folder=tmp_path)
    assert global_metrics["k-NN Accuracy"] == 1.0


def test_nearest_neighbour_metrics_above_the_dense_class_limit_match_the_dense_matrix(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the k-nearest neighbour evaluation of a test split over more classes than
    MAXIMUM_DENSE_CONFUSION_CLASSES (the 21,841 classes of ImageNet-22k) asks for the sparse confusion matrix, and that
    its global metrics, per-class tables and scalars and class_metrics.csv are those of the dense matrix: 200 random
    test images of 9 of 12 classes voted by 48 reference images of all 12 classes, scored dense, then with the limit
    lowered to 5 classes.

    Args:
        tmp_path (Path): The pytest temporary directory, receiving both class_metrics.csv files.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture lowering the limit of the trainer module.
    """
    # Random representations, four reference images per class
    generator = torch.Generator().manual_seed(5)
    trainer = SimpleNamespace(class_names=[f"class_{class_index}" for class_index in range(12)], number_neighbours=3,
                              encode_reference_images=MagicMock(return_value=(
                                  torch.randn(size=(48, 8), generator=generator), torch.arange(end=48) % 12)))
    representations = torch.randn(size=(200, 8), generator=generator)
    labels = torch.randint(low=0, high=9, size=(200,), generator=generator)

    # Dense scoring, then sparse scoring with the limit lowered below the 12 classes
    with patch(target="vision_training.trainers.self_supervised_trainer.compute_confusion_matrix",
               wraps=compute_confusion_matrix) as confusion_matrix_mock:
        dense_metrics, dense_outputs = SelfSupervisedTrainer.compute_nearest_neighbour_results(
            self=trainer, representations=representations, labels=labels, evaluation_folder=tmp_path / "dense")
        monkeypatch.setattr(target=self_supervised_trainer, name="MAXIMUM_DENSE_CONFUSION_CLASSES", value=5)
        sparse_metrics, sparse_outputs = SelfSupervisedTrainer.compute_nearest_neighbour_results(
            self=trainer, representations=representations, labels=labels, evaluation_folder=tmp_path / "sparse")
    assert [call.kwargs["sparse"] for call in confusion_matrix_mock.call_args_list] == [False, True]

    # Same metrics, tables, scalars and class metrics file, to the last bit
    assert sparse_metrics == dense_metrics and 0.0 < dense_metrics["k-NN Accuracy"] < 1.0
    assert (sparse_outputs["tables"], sparse_outputs["class_scalars"]) == (dense_outputs["tables"],
                                                                         dense_outputs["class_scalars"])
    assert sparse_outputs["class_metrics_path"].read_bytes() == dense_outputs["class_metrics_path"].read_bytes()


@pytest.mark.parametrize(argnames=("test_labels", "expected_percent_metrics", "expected_grid_pattern",
                                   "expected_circle_rows"),
                         argvalues=[([0, 1], NEAREST_NEIGHBOUR_RATES,
                                     r"k-NN Accuracy +100\.00 % +k-NN Balanced Accuracy +100\.00 %",
                                     ["circle 1 100.00 % 100.00 % 100.00 %", "circle 1 100.00 % 2.50 % 100.00 %"]),
                                    ([IGNORED_LABEL_INDEX, IGNORED_LABEL_INDEX], [],
                                     r"Effective Rank +\d\.\d{4} +Representation Standard Deviation +\d\.\d{4}", [])])
def test_report_writes_the_nearest_neighbour_rates_as_percentages(tmp_path: Path, test_labels: List[int],
                                                                  expected_percent_metrics: List[str],
                                                                  expected_grid_pattern: str,
                                                                  expected_circle_rows: List[str]) -> None:
    """
    Verifies the report of the evaluation of the representations: with labelled test images, the k-NN accuracy,
    balanced accuracy and macro F1 and the columns of the per-class k-NN table and of the k-NN recall table are written
    as percentages ("100.00 %", the exact interval of a single image "2.50 %"), the effective rank and the standard
    deviation keeping 4 decimals; without labelled test image, the k-NN metrics are skipped, no value is a percentage
    and the report is still rendered, the report refusing a percent metric missing from the metrics.

    Args:
        tmp_path (Path): The pytest temporary directory, receiving class_metrics.csv.
        test_labels (List[int]): The class indices of the two test images, IGNORED_LABEL_INDEX without category.
        expected_percent_metrics (List[str]): The metrics expected as percentages.
        expected_grid_pattern (str): The pattern of the first line of the metrics grid.
        expected_circle_rows (List[str]): The rows of the class circle in the k-NN tables, spaces collapsed.
    """
    # Trainer reduced to the attributes of the evaluation, its two reference images voting for the test images
    trainer = SimpleNamespace(class_names=["circle", "square"], number_neighbours=1, SAMPLE_NAME="Images",
                              encode_reference_images=MagicMock(return_value=(torch.tensor(data=[[1.0, -0.5],
                                                                                                 [0.7, 0.7]]),
                                                                              torch.tensor(data=[0, 1]))),
                              reference_dataloader=MagicMock(), reference_notes=[],
                              get_evaluation_folder=MagicMock(return_value=tmp_path))
    trainer.compute_nearest_neighbour_results = MethodType(SelfSupervisedTrainer.compute_nearest_neighbour_results,
                                                           trainer)
    accumulator = {"representations": [torch.tensor(data=[[1.0, 0.0], [0.95, 0.31]])],
                   "labels": [torch.tensor(data=test_labels)], "sample_identifiers": [torch.tensor(data=[1, 2])]}
    results = SelfSupervisedTrainer.compute_evaluation_results(self=trainer, accumulator=accumulator, iteration=4)
    assert results["percent_metrics"] == expected_percent_metrics

    # Block of Results.txt: the grid, the rows of both k-NN tables, the collapse metrics with 4 decimals
    lines = render_report_text(renderables=build_report_renderables(results=results, iteration=4,
                                                                    timestamp="2026-11-15 12:00",
                                                                    per_class_report=True)).splitlines()
    text = "\n".join(lines)
    assert re.search(pattern=expected_grid_pattern, string=lines[1]) is not None
    assert [" ".join(line.split()) for line in lines if line.strip().startswith("circle")] == expected_circle_rows
    assert f"{results['metrics']['Effective Rank']:.4f}" in text and ("%" in text) == bool(expected_percent_metrics)


@pytest.mark.parametrize(argnames=("method", "expected_percent_keys"),
                         argvalues=[("simclr", ["positive_pair_accuracy"]), ("barlow_twins", [])])
def test_console_summary_writes_the_rates_of_the_method_as_percentages(method: str,
                                                                       expected_percent_keys: List[str]) -> None:
    """
    Verifies the tracked metrics written as percentages in the console summary of every method: the positive pair
    accuracy of SimCLR, none for Barlow Twins, whose correlations are not rates any more than the embedding standard
    deviation, every percent key being one of the tracked keys of the method, which the metric tracker requires.

    Args:
        method (str): The self-supervised method, 'ProjectionHead.method'.
        expected_percent_keys (List[str]): The tracked keys expected as percentages.
    """
    trainer = SimpleNamespace(method=method)
    percent_keys = SelfSupervisedTrainer.get_percent_tracked_metrics(self=trainer)
    assert percent_keys == expected_percent_keys
    assert set(percent_keys) <= set(SelfSupervisedTrainer.get_tracked_metrics(self=trainer))


def test_reference_images_are_fixed_and_bounded_per_class(tmp_path: Path) -> None:
    """
    Verifies the reference images of the k-nearest neighbour evaluation of a run: two trainers of the same
    configuration, whose datasets are shuffled differently, choose the same training images, at most
    'evaluation.number_reference_images_per_class' per class and every image of a smaller class, and the evaluation
    reports them in its notes.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 40, "validation": 4, "test": 6}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    # Two reference images per class at most
    configuration_path = write_smoke_configuration(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                   number_iterations=4, head_settings={})
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["ExperimentConfiguration"]["evaluation"]["number_reference_images_per_class"] = 2
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))

    # Reference images of two trainers, the same images in the same order
    reference_identifiers = []
    for _ in range(2):
        trainer = build_trainer(configuration_path=configuration_path)
        reference_samples = trainer.reference_dataloader.dataset.samples
        reference_identifiers.append([sample["identifier"] for sample in reference_samples])
        trainer.shutdown_loader_workers()
    assert reference_identifiers[0] == reference_identifiers[1]

    # At most two images of every class, every image of a class of fewer training images
    training_counts = Counter(sample["class_indices"][0] for sample in trainer.train_dataloader.dataset.samples)
    reference_counts = Counter(sample["class_indices"][0] for sample in reference_samples)
    assert reference_counts == {class_index: min(count, 2) for class_index, count in training_counts.items()}
    assert (f"k-NN : {trainer.number_neighbours} neighbours among {len(reference_samples)} labelled training images "
            f"of {len(reference_counts)} classes (2 per class at most)") in trainer.reference_notes


def test_input_viewer_finds_the_groups_of_the_batch_images(tmp_path: Path) -> None:
    """
    Verifies that the input viewer of the grouped batches finds every image of a training batch among the shuffled
    samples of the training split by its identifier, and describes the batch from the groups of its images, in the
    order of the batch, every image having groups of its own.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Eight images of two cameras, shuffled by the dataset, grouped in batches of two blocks of two images of a camera
    write_split(split_folder=tmp_path, categories=[{"id": 1, "name": "circle"}],
                samples=[{"id": sample_identifier, "file_name": f"{sample_identifier}.png", "category_id": 1,
                          "groups": {"camera": f"c{sample_identifier % 2}", "image": f"i{sample_identifier}"}}
                         for sample_identifier in range(1, 9)])
    dataset = ClassificationDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                    image_size=32, keep_ratio=True, label_dictionary={1: 0}, output_mode="multi_class",
                                    keep_unlabelled_samples=True)
    batch_sampler, _ = build_grouped_batch_sampler(
        samples=dataset.samples, grouped_batches_configuration={"activated": True, "block_key": "camera",
                                                                "images_per_block": 2, "batch_key": None},
        batch_size=4, split_name="train_data_folder")

    # The first batch of a pass as the loader gives it, its images named by their identifiers
    batch_indices = next(iter(batch_sampler))
    training_batch = {"images": torch.zeros(size=(4, 3, 32, 32), dtype=torch.uint8),
                      SAMPLE_IDENTIFIERS_KEY: torch.tensor(data=[dataset.samples[sample_index]["identifier"]
                                                                 for sample_index in batch_indices])}

    # The viewer of a trainer reduced to its training loader, without window
    trainer = SimpleNamespace(train_dataloader=SimpleNamespace(batch_sampler=batch_sampler, dataset=dataset),
                              describe_input_batch=MagicMock(return_value=("Training Batch", [], ["c0", "c1"], None)))
    with (patch(target="vision_training.trainers.self_supervised_trainer.build_batch_mosaic"),
          patch(target="vision_training.trainers.self_supervised_trainer.show_viewer_windows", return_value=False)):
        assert SelfSupervisedTrainer.show_input_batch(self=trainer, training_batch=training_batch,
                                                      batch_index=0) is False
    assert trainer.describe_input_batch.call_args.kwargs["image_groups"] == [
        dataset.samples[sample_index]["groups"] for sample_index in batch_indices]
