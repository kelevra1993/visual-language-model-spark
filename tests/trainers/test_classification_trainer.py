"""
Smoke tests of ClassificationTrainer of vision_training/trainers/classification_trainer.py on a synthetic dataset:
multi-class and multi-label training save, evaluate and resume, with the report, the per-class metrics, the
predictions file, the error images and the sample images checked, the per-class report, the predictions file and the
sample images can be turned off, a head reading the flattened feature map trains, and the decorrelation of the hidden
features turned on when a run resumes is tracked and summed into the total loss.
"""

import json
import pytest
import torch
import yaml

from pathlib import Path, PurePosixPath
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict
from unittest.mock import patch

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.evaluation.classification.threshold_finder import find_class_thresholds
from vision_training.trainers.classification_trainer import ClassificationTrainer
from tests.fixtures.evaluation_checks import (check_class_metrics_file, check_error_images, check_predictions_file,
                                              check_sample_images, read_predictions_file, read_results_blocks)

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "classification" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 64


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                              head_settings: Dict[str, Any], evaluation_settings: Dict[str, Any]) -> Path:
    """
    Writes a copy of the synthetic shapes classification configuration adapted to a quick training smoke test: the
    generated dataset, 64 pixel images, batches of 4 images and a saving schedule that triggers an intermediate save,
    a save with test evaluation and a validation pass within a handful of iterations.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        head_settings (Dict[str, Any]): The 'ClassificationHead' settings replaced for the test.
        evaluation_settings (Dict[str, Any]): The 'ExperimentConfiguration.evaluation' settings replaced for the test.

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
                                     "validation_iteration_ratio": 2})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": True, "view_input_images": False})
    experiment_configuration["evaluation"].update(evaluation_settings)

    # Small images, the generated label mapping, and the head settings of the test
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    configuration["ClassificationHead"].update(head_settings)

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> ClassificationTrainer:
    """
    Loads a configuration file and builds the ClassificationTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        ClassificationTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return ClassificationTrainer(experiment_configuration=experiment_configuration,
                                 model_configuration=model_configuration, configuration_path=configuration_path)


def test_classification_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the classification training pipeline on a generated synthetic dataset: the balanced class weights of the
    training split are set in the model, 5 iterations save the intermediate, evaluated and final checkpoints, the
    evaluation of the test split is reported (Results.txt block with its per-class table, per-class scalars and
    confusion matrix image) with its per-class metrics, confusion matrix and class thresholds files (the thresholds of
    two target precisions, found again from the probabilities of the predictions file, and the image of their
    thresholds), the record of the data lists the images of every split, and a second run with the same settings
    resumes from the last checkpoint with exactly its weights. No training step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 16, "validation": 8, "test": 6}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")
    head_settings = {"class_weighting": "balanced", "label_smoothing": 0.1}

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5, head_settings=head_settings,
        evaluation_settings={"error_dumps": {"activated": True, "maximum_per_class_pair": 1},
                             "target_precision": [0.99, 0.1]}))
    assert trainer.run_name.startswith("CLS_") and trainer.run_name.endswith("-LS0.1-CWB_MC35")
    assert trainer.class_names == list(SHAPE_CATEGORIES)
    # Balanced weights, a class without training image counting as one, as some of the 35 classes may be empty
    class_counts = torch.tensor(data=trainer.train_dataloader.dataset.get_class_counts(), dtype=torch.float32)
    clamped_class_counts = class_counts.clamp(min=1.0)
    expected_class_weights = clamped_class_counts.sum() / (len(SHAPE_CATEGORIES) * clamped_class_counts)
    assert torch.allclose(input=trainer.model.class_weights.cpu(), other=expected_class_weights)

    # Record of the data: the classes in class index order with their training images counted as the dataset counts
    # them, and the images of every split, in the order of their labels file
    data_record = json.loads(s=(trainer.experiment_folder / "data.json").read_text())
    assert [class_record["class_name"] for class_record in data_record["classes"]] == list(SHAPE_CATEGORIES)
    assert [class_record["images"]["train"] for class_record in data_record["classes"]] == \
        trainer.train_dataloader.dataset.get_class_counts()
    for split_name in ["train", "validation", "test"]:
        labels = json.loads(s=(dataset_folder / split_name / "labels.json").read_text())
        assert data_record["splits"][split_name]["images"] == [sample["file_name"] for sample in labels["samples"]]
    with patch.object(target=trainer, attribute="report_evaluation", wraps=trainer.report_evaluation) as report_mock:
        trainer.run_training_loop()
    results = report_mock.call_args.kwargs["results"]

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]

    # Report of the evaluated checkpoint: its block in Results.txt with the per-class table, and its per-class outputs
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [(4, "6 Test Images")]
    assert list(results["metrics"]) == ["Accuracy", "Balanced Accuracy", "Macro F1", "Top-5 Accuracy",
                                        "Macro ROC AUC", "Macro Average Precision"]
    assert all(0.0 <= results["metrics"][metric_name] <= 1.0
               for metric_name in ["Macro ROC AUC", "Macro Average Precision"])
    assert "Per Class Metrics" in results_blocks[0]["text"] and "ROC AUC" in results_blocks[0]["text"]
    assert [table["title"] for table in results["tables"]] == ["Per Class Metrics", "Per Class Thresholds"]
    assert "@ P0.1" not in "".join(results["tables"][0]["columns"])
    assert "Per Class Thresholds" in results_blocks[0]["text"] and "@ P0.1" in results_blocks[0]["text"]
    assert list(results["images"]) == ["Confusion Matrix", "ROC Curves", "Precision Recall Curves",
                                       "Precision Recall Thresholds"]
    assert all(image.ndim == 3 for image in results["images"].values())

    # Files of the evaluated checkpoint, the per-class scalars of the curves following the defined metrics of the file
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir() if path.name != "Errors") == [
        "Samples", "class_metrics.csv", "class_thresholds.json", "confusion_matrix.csv", "confusion_matrix.png",
        "precision_recall_curves.png", "precision_recall_thresholds.png", "predictions.csv", "roc_curves.png"]
    class_rows = check_class_metrics_file(evaluation_folder=evaluation_folder, class_names=trainer.class_names,
                                          number_samples=6, per_class_report=True, additional_target_precisions=[0.1])
    assert len(results["class_scalars"]) == 3 * len(trainer.class_names) + sum(
        row[metric_key] != "" for row in class_rows for metric_key in ["roc_auc", "average_precision"])
    rows = check_predictions_file(evaluation_folder=evaluation_folder, reported_accuracy=results["metrics"]["Accuracy"],
                                  expected_rows=6, class_names=trainer.class_names, identification_columns=["sample"])
    assert all(row["sample"].endswith(".png") for row in rows)
    check_error_images(evaluation_folder=evaluation_folder, rows=rows, notes=results["notes"],
                       maximum_per_class_pair=1)

    # Every test image is a sample image (8 asked for 6 images), captioned in the colour of its prediction
    check_sample_images(evaluation_folder=evaluation_folder, results=results, number_sample_images=6,
                        sample_kind="images",
                        correct_predictions={str(PurePosixPath(row["sample"]).with_suffix("")).replace("/", "_"):
                                             row["correct"] == "1" for row in rows})
    # Thresholds of the main target precision and of the additional one, equal to those found again from the float32
    # probabilities of the predictions file, written with enough digits for that: every class of the test split
    # reaches the precision 0.1 at its lowest score (at least 1 of the 6 images), so the comparison never holds only
    # missing thresholds
    class_thresholds = json.loads((evaluation_folder / "class_thresholds.json").read_text())
    assert list(class_thresholds["classes"]) == trainer.class_names and class_thresholds["target_precision"] == 0.99
    assert [target["target_precision"] for target in class_thresholds["additional_targets"]] == [0.1]
    file_probabilities = torch.tensor(data=[[float(row[f"probability_{class_name}"])
                                             for class_name in trainer.class_names] for row in rows],
                                      dtype=torch.float32)
    file_positives = torch.tensor(data=[[row["true_class"] == class_name for class_name in trainer.class_names]
                                        for row in rows])
    for target_thresholds in [class_thresholds] + class_thresholds["additional_targets"]:
        found_thresholds = find_class_thresholds(probabilities=file_probabilities, positive_masks=file_positives,
                                                 target_precision=target_thresholds["target_precision"])
        assert [None if class_threshold is None else {"threshold": class_threshold[0], "recall": class_threshold[1]}
                for class_threshold in found_thresholds] == list(target_thresholds["classes"].values())
    assert any(threshold is not None for threshold in class_thresholds["additional_targets"][0]["classes"].values())
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train", "Validation"]

    # Second run with the same settings and a total of 7 iterations: same run folder, resumed from iteration 5 with
    # exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=7, head_settings=head_settings,
        evaluation_settings={}))
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


def test_multi_label_classification_training(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the multi-label mode of the classification trainer on the synthetic multi-label dataset (5 to 10 shapes
    per image, every sample listing the distinct categories of its shapes in 'category_ids', the dataset pretraining
    the backbone of the sequence labelling model): the training runs, and the evaluation reports the multi-label
    metrics and the macro means of the curves, with the positives of every class in its per-class table, and saves
    the per-class metrics, the class thresholds and the ROC and precision-recall curves without confusion matrix, the
    predictions file listing the several true classes of every image.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    # Images of 128 pixels, resized to the smoke size by the loader, as up to 10 shapes do not always fit without
    # overlap in a 64 pixel image, and at most 7 shapes, as in the generator test
    dataset_folder = tmp_path / "synthetic_counted_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=2 * SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=7,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="multi_label_classification")

    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5,
        head_settings={"output_mode": "multi_label"}, evaluation_settings={}))
    assert trainer.run_name.endswith("_ML35")
    with patch.object(target=trainer, attribute="report_evaluation", wraps=trainer.report_evaluation) as report_mock:
        trainer.run_training_loop()
    results = report_mock.call_args.kwargs["results"]

    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [(4, "4 Test Images")]
    assert list(results["metrics"]) == ["Subset Accuracy", "Micro F1", "Macro F1", "Macro ROC AUC",
                                        "Macro Average Precision"]
    assert results["tables"][0]["columns"][1] == "Positives"
    assert list(results["images"]) == ["ROC Curves", "Precision Recall Curves", "Precision Recall Thresholds"]
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir()) == [
        "Samples", "class_metrics.csv", "class_thresholds.json", "precision_recall_curves.png",
        "precision_recall_thresholds.png", "predictions.csv", "roc_curves.png"]
    multi_label_rows = read_predictions_file(evaluation_folder=evaluation_folder)
    assert len(multi_label_rows) == 4
    assert list(multi_label_rows[0])[:4] == ["sample", "true_classes", "predicted_classes", "correct"]
    # The true classes of every image, separated by ';', are the distinct categories of its manifest sample
    test_labels = json.loads((dataset_folder / "test" / "labels.json").read_text())
    category_names = {category["id"]: category["name"] for category in test_labels["categories"]}
    expected_true_classes = {sample["file_name"]: [category_names[category_identifier]
                                                   for category_identifier in sample["category_ids"]]
                             for sample in test_labels["samples"]}
    assert all(len(true_classes) >= 1 for true_classes in expected_true_classes.values())
    assert any(len(true_classes) > 1 for true_classes in expected_true_classes.values())
    assert {PurePosixPath(row["sample"]).name: row["true_classes"].split(";") for row in multi_label_rows} == \
        expected_true_classes

    # Every test image is a sample image, green when all its class decisions are right
    check_sample_images(evaluation_folder=evaluation_folder, results=results, number_sample_images=4,
                        sample_kind="images",
                        correct_predictions={str(PurePosixPath(row["sample"]).with_suffix("")).replace("/", "_"):
                                             row["correct"] == "1" for row in multi_label_rows})

    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_predictions_file_per_class_report_and_sample_images_can_be_turned_off(tmp_path: Path) -> None:
    """
    Verifies that the evaluation writes no predictions file when 'evaluation.save_predictions' is False, for the large
    test sets whose file would be too big, that 'evaluation.per_class_report' False keeps the global metrics only:
    no per-class table in Results.txt, no per-class scalar, curve metric, confusion matrix nor curve image in
    TensorBoard and no confusion_matrix.png nor curve image, while class_metrics.csv (without the curve columns) and
    confusion_matrix.csv are still saved, and that
    'evaluation.number_sample_images' 0 draws no sample image.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=4, head_settings={},
        evaluation_settings={"save_predictions": False, "per_class_report": False, "number_sample_images": 0}))
    trainer.run_training_loop()

    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir()) == ["class_metrics.csv", "class_thresholds.json",
                                                                         "confusion_matrix.csv"]
    check_class_metrics_file(evaluation_folder=evaluation_folder, class_names=trainer.class_names, number_samples=4,
                             per_class_report=False)

    # The report keeps the global metrics only
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [block["iteration"] for block in results_blocks] == [4]
    assert "Accuracy" in results_blocks[0]["text"] and "Per Class Metrics" not in results_blocks[0]["text"]
    test_events = EventAccumulator(path=str(trainer.tensorboard_directory / "Test"))
    test_events.Reload()
    assert sorted(test_events.Tags()["scalars"]) == ["Accuracy", "Balanced Accuracy", "Macro F1", "Top-5 Accuracy"]
    assert test_events.Tags()["images"] == []


def test_intermediate_weight_saving_can_be_turned_off(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that 'intermediate_weight_saving_iterations' 0 saves no checkpoint between the evaluated ones: a run of 5
    iterations evaluated every 4 keeps only the evaluated checkpoint of iteration 4 and the last one of iteration 5,
    and its summary reports the intermediate saving as turned off.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    # The smoke configuration with its intermediate checkpoints of every 2 iterations turned off
    configuration_path = write_smoke_configuration(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                   number_iterations=5, head_settings={}, evaluation_settings={})
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["ExperimentConfiguration"]["intermediate_weight_saving_iterations"] = 0
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))

    trainer = build_trainer(configuration_path=configuration_path)
    trainer.run_training_loop()

    # No Iteration_2 checkpoint, only the evaluated and the last ones
    assert sorted(path.name for path in trainer.weights_directory.glob("Iteration_*")) == ["Iteration_4",
                                                                                         "Iteration_5"]
    assert "- Save Only Every         : Turned Off" in capsys.readouterr().out


def test_flattened_feature_map_training(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies a short classification training whose head reads the flattened final feature map: the run name carries
    HFLAT, the head reads every value of the 64 pixel feature map, and the test evaluation is written to Results.txt
    without any failed iteration.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=4,
        head_settings={"global_pooling": "flatten"}, evaluation_settings={}))
    assert "_HFLAT." in trainer.run_name

    # The head reads the channels, height and width of the final feature map of a 64 pixel image
    with torch.no_grad():
        final_feature_map, _ = trainer.model.backbone(input_tensor=torch.zeros(
            size=(1, 3, SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE), device=trainer.device, dtype=trainer.dtype))
    assert trainer.model.classification_head.input_features == final_feature_map[0].numel()
    trainer.run_training_loop()

    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [(4, "4 Test Images")]
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_decorrelation_turned_on_when_a_run_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the decorrelation of the hidden features in a training: a run trained without it resumes in the same run
    folder with it turned on (no run name token), the change being listed, and its training and validation batches
    track the decorrelation loss and the mean correlation of the hidden features, summed with the classification loss
    into the total loss, without any failed iteration. The run without it tracks neither.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 8, "test": 4}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    # Warm-up of the classification loss alone, then the same run resumed with the decorrelation of its hidden layer
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2, head_settings={},
        evaluation_settings={}))
    assert "decorrelation_loss" not in trainer.tracked_metrics_mapping
    trainer.run_training_loop()
    decorrelation_settings = {"decorrelation": {"activated": True, "weight": 0.01, "hidden_layer": 1}}
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=6,
        head_settings=decorrelation_settings, evaluation_settings={}))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder and resumed_trainer.start_iteration == 3
    assert ("ClassificationHead.decorrelation.activated", False, True) in \
        resumed_trainer.configuration_changes["changed_settings"]
    assert list(resumed_trainer.tracked_metrics_mapping.values()) == [
        "Total Loss", "Classification Loss", "Decorrelation Loss", "Hidden Feature Correlation", "Accuracy"]
    resumed_trainer.run_training_loop()

    # The resumed iterations write both metrics, the total loss being the sum of the two losses of every batch
    for split_name, expected_steps in [("Train", [3, 4, 5, 6]), ("Validation", [4, 6])]:
        events = EventAccumulator(path=str(resumed_trainer.tensorboard_directory / split_name))
        events.Reload()
        decorrelation_losses = {event.step: event.value for event in events.Scalars(tag="Decorrelation Loss")}
        classification_losses = {event.step: event.value for event in events.Scalars(tag="Classification Loss")}
        total_losses = {event.step: event.value for event in events.Scalars(tag="Total Loss")}
        correlations = [event.value for event in events.Scalars(tag="Hidden Feature Correlation")]
        assert sorted(decorrelation_losses) == expected_steps
        assert all(0.0 <= correlation <= 1.0 for correlation in correlations)
        for step, decorrelation_loss in decorrelation_losses.items():
            assert decorrelation_loss > 0.0
            assert total_losses[step] == pytest.approx(expected=classification_losses[step] + decorrelation_loss,
                                                       rel=1e-5)
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
