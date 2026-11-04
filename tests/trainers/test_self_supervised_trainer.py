"""
Smoke tests of SelfSupervisedTrainer of vision_training/trainers/self_supervised_trainer.py on a synthetic dataset: a
Barlow Twins training saves, evaluates and resumes, with the tracked metrics of the method, the k-nearest neighbour
report of the labelled test images, its per-class metrics file and the sample images of two views checked, a dataset
mixing labelled and unlabelled images is trained on every image and evaluated on the labelled ones, and the classes of
the evaluation are named after the categories of the three manifests.
"""

import csv
import json
import pytest
import torch
import yaml

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import patch

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.trainers.self_supervised_trainer import METHOD_TRACKED_METRICS, SelfSupervisedTrainer
from vision_training.utilities.visualization import CAPTION_BAR_HEIGHT
from tests.data.classification.test_classification_dataset import write_split
from tests.fixtures.evaluation_checks import check_sample_images, read_results_blocks

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "self_supervised" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 64

# Metrics of the evaluation of a test split with labels, in the order of the report
LABELLED_EVALUATION_METRICS = ["k-NN Accuracy", "k-NN Balanced Accuracy", "k-NN Macro F1", "Effective Rank",
                               "Representation Standard Deviation"]


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
    the evaluation of the test split is reported (Results.txt block with the k-NN metrics, effective rank and
    representation standard deviation, the k-NN accuracy as headline metric) with its per-class metrics file and the
    sample images of two views of every test image, and a second run with the same settings resumes from the last
    checkpoint with exactly its weights. No training step may fail silently.

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


def test_class_names_come_from_the_categories_of_every_split(tmp_path: Path) -> None:
    """
    Verifies that the classes of the k-nearest neighbour evaluation are named after the categories of the three
    manifests, so that a class named in the training manifest only keeps its category name instead of its index.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Training manifest naming both classes, validation and test manifests naming the first one only
    split_categories = {"train": [{"id": 1, "name": "circle"}, {"id": 5, "name": "triangle"}],
                        "validation": [{"id": 1, "name": "circle"}], "test": [{"id": 1, "name": "circle"}]}
    for split_name, categories in split_categories.items():
        write_split(split_folder=tmp_path / split_name, categories=categories,
                    samples=[{"id": sample_identifier, "file_name": f"{sample_identifier}.png", "category_id": 1}
                             for sample_identifier in range(1, 3)])

    # Only the attributes read by build_dataloaders, without building the model of the trainer
    trainer = SimpleNamespace(
        experiment_configuration={f"{split_name}_data_folder": str(tmp_path / split_name)
                                  for split_name in split_categories},
        model_configuration={"Data": {"image_settings": {"size": 32, "keep_ratio": True},
                                      "label_dictionary": {1: 0, 5: 1},
                                      "grouped_batches": {"activated": False, "block_key": "camera",
                                                          "images_per_block": 2, "batch_key": None}}}, batch_size=2)
    SelfSupervisedTrainer.build_dataloaders(self=trainer)
    assert trainer.class_names == ["circle", "triangle"]
