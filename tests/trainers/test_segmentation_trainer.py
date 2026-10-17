"""
Smoke tests of SegmentationTrainer of vision_training/trainers/segmentation_trainer.py on a synthetic dataset: a
training with balanced class weights saves, evaluates and resumes, with the report, the per-class and boundary
metrics, the pixel confusion matrix, the sample grids, the predicted masks and the cached class pixel counts checked, a
training with the Dice loss alone, binary labels and bilinear upsampling runs without the optional outputs, and a
training on tiles of scaled images with deep supervision and the Lovász and boundary losses evaluates the whole test
images tile by tile at their original resolution.
"""

import csv
import cv2
import numpy as np
import pytest
import torch
import yaml

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict
from unittest.mock import patch

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.losses.classification_losses import compute_balanced_class_weights
from vision_training.trainers.segmentation_trainer import CLASS_PIXEL_COUNTS_FILE_NAME, SegmentationTrainer
from tests.fixtures.evaluation_checks import read_results_blocks

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                "semantic_segmentation" / "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations, and number of test images
SMOKE_IMAGE_SIZE = 64
NUMBER_TEST_IMAGES = 6


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                              section_settings: Dict[str, Dict[str, Any]], evaluation_settings: Dict[str, Any]) -> Path:
    """
    Writes a copy of the synthetic shapes segmentation configuration adapted to a quick training smoke test: the
    generated dataset, 64 pixel images, batches of 4 images and a saving schedule that triggers an intermediate save,
    a save with test evaluation and a validation pass within a handful of iterations.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        section_settings (Dict[str, Dict[str, Any]]): The settings replaced in the model sections for the test, keyed
            by section name.
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
                                     "validation_iteration_ratio": 2, "resume_training": True})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": True, "compute_model_size": True,
                                                        "view_input_images": False})
    experiment_configuration["evaluation"].update(evaluation_settings)

    # Small images, two loader workers per split, the generated label mapping, and the section settings of the test
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["loading"]["number_workers"] = 2
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    for section_name, settings in section_settings.items():
        configuration[section_name].update(settings)

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> SegmentationTrainer:
    """
    Loads a configuration file and builds the SegmentationTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        SegmentationTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return SegmentationTrainer(experiment_configuration=experiment_configuration,
                               model_configuration=model_configuration, configuration_path=configuration_path)


def create_smoke_dataset(dataset_folder: Path, image_size: int = SMOKE_IMAGE_SIZE) -> None:
    """
    Generates the synthetic shapes segmentation dataset of the smoke tests: images holding up to 3 shapes with their
    class masks, 16 training, 8 validation and 6 test images, of the network input size or larger for the training on
    tiles.

    Args:
        dataset_folder (Path): The root folder of the generated dataset.
        image_size (int): The side of the generated images. Defaults to SMOKE_IMAGE_SIZE.
    """
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=image_size,
                             split_sizes={"train": 16, "validation": 8, "test": NUMBER_TEST_IMAGES}, maximum_objects=3,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="semantic_segmentation")


def test_segmentation_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the segmentation training pipeline on a generated synthetic dataset: the balanced class weights of the
    training pixels are set in the model and their counts cached in the run folder, 5 iterations save the
    intermediate, evaluated and final checkpoints, the evaluation of the test split is reported (Results.txt block with
    its per-class table, per-class scalars, confusion matrix image and sample grids in TensorBoard) with its per-class
    metrics, confusion matrix, sample grid and predicted mask files, and a second run with the same settings resumes
    from the last checkpoint with exactly its weights and the cached pixel counts, and a third run with other loss
    settings resumes in the same folder with the class weights of its configuration. No training step may fail
    silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_smoke_dataset(dataset_folder=dataset_folder)
    section_settings = {"SegmentationLoss": {"class_weighting": "balanced", "label_smoothing": 0.1}}
    evaluation_settings = {"number_sample_images": 2, "save_predicted_masks": True, "per_class_report": True}

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5,
        section_settings=section_settings, evaluation_settings=evaluation_settings))
    assert trainer.run_name.startswith("SEG_") and trainer.run_name.endswith("_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC35")
    assert trainer.class_names == ["background"] + list(SHAPE_CATEGORIES)
    assert list(trainer.tracked_metrics_mapping) == ["total_loss", "segmentation_loss", "dice_loss", "pixel_accuracy",
                                                     "mean_iou"]

    # Balanced weights of the training pixels, whose counts are cached in the run folder
    class_pixel_counts = trainer.train_dataloader.dataset.count_class_pixels()
    assert sum(class_pixel_counts) == 16 * SMOKE_IMAGE_SIZE * SMOKE_IMAGE_SIZE
    assert torch.allclose(input=trainer.model.class_weights.cpu(),
                          other=compute_balanced_class_weights(class_counts=class_pixel_counts))
    assert (trainer.experiment_folder / CLASS_PIXEL_COUNTS_FILE_NAME).is_file()
    with patch.object(target=trainer, attribute="report_evaluation", wraps=trainer.report_evaluation) as report_mock:
        trainer.run_training_loop()
    results = report_mock.call_args.kwargs["results"]

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]

    # Report of the evaluated checkpoint: its block in Results.txt with the per-class table, and its per-class outputs
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [
        (4, f"{NUMBER_TEST_IMAGES} Test Images")]
    assert list(results["metrics"]) == ["Mean IoU", "Mean Dice", "Mean Boundary IoU", "Mean Boundary F1",
                                        "Pixel Accuracy", "Frequency Weighted IoU"]
    assert all(0.0 <= metric_value <= 1.0 for metric_value in results["metrics"].values())
    assert results["headline_metric"] == "Mean IoU" and "Per Class Metrics" in results_blocks[0]["text"]
    assert list(results["images"]) == ["Confusion Matrix"]
    assert results["notes"][:2] == [f"Labelled Pixels : {NUMBER_TEST_IMAGES * SMOKE_IMAGE_SIZE * SMOKE_IMAGE_SIZE} at "
                                    f"the network resolution, the padding and the unlabelled pixels being ignored",
                                    "Boundary Tolerance : 2 pixels"]

    # Sample grids of the two fixed test images: four 64 pixel panels with their caption and legend bars, the caption
    # bar being wider than the panels when its text is
    assert len(results["sample_images"]) == 2
    for image_name, sample_image in results["sample_images"].items():
        assert image_name.startswith("Samples/test_")
        assert sample_image.shape[0] == 24 + SMOKE_IMAGE_SIZE + 24 and sample_image.shape[2] == 3
        assert sample_image.shape[1] >= 4 * SMOKE_IMAGE_SIZE + 3 * 4

    # Files of the evaluated checkpoint: per-class metrics of the 10 classes summing to the labelled pixels
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir()) == [
        "Predicted_Masks", "Samples", "class_metrics.csv", "confusion_matrix.csv", "confusion_matrix.png"]
    with open(file=evaluation_folder / "class_metrics.csv", mode="r", newline="") as class_metrics_file:
        class_rows = list(csv.DictReader(f=class_metrics_file))
    assert list(class_rows[0]) == ["class_index", "class", "pixels", "iou", "dice", "boundary_iou", "boundary_f1",
                                   "precision", "recall"]
    assert [row["class"] for row in class_rows] == trainer.class_names
    assert sum(int(row["pixels"]) for row in class_rows) == NUMBER_TEST_IMAGES * SMOKE_IMAGE_SIZE * SMOKE_IMAGE_SIZE
    assert all(row["iou"] == "" for row in class_rows if row["pixels"] == "0")
    assert len(results["class_scalars"]) == sum(value != "" for row in class_rows for key, value in row.items()
                                                if key not in ["class_index", "class", "pixels"])

    # Predicted mask of every test image, a single-channel image of class indices at the network resolution
    predicted_mask_paths = sorted((evaluation_folder / "Predicted_Masks").iterdir())
    assert len(predicted_mask_paths) == NUMBER_TEST_IMAGES
    predicted_mask = cv2.imread(filename=str(predicted_mask_paths[0]), flags=cv2.IMREAD_UNCHANGED)
    assert predicted_mask.shape == (SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE) and predicted_mask.dtype == np.uint8
    assert int(predicted_mask.max()) < len(trainer.class_names)
    assert len(list((evaluation_folder / "Samples").glob("*.png"))) == 2

    # TensorBoard of the test evaluations: the global and per-class scalars, the confusion matrix and the sample grids
    test_events = EventAccumulator(path=str(trainer.tensorboard_directory / "Test"))
    test_events.Reload()
    assert set(results["metrics"]) <= set(test_events.Tags()["scalars"])
    assert sorted(test_events.Tags()["images"]) == sorted(["Confusion Matrix"] + list(results["sample_images"]))

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights, the class
    # pixel counts being read back from the run folder
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2,
        section_settings=section_settings, evaluation_settings=evaluation_settings))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])
    resumed_trainer.run_training_loop()

    # Third run with other loss settings: the loss settings give no token, so the run resumes in the same folder from
    # iteration 7, its class weights following the new configuration instead of the balanced ones of the checkpoint
    changed_loss_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=1,
        section_settings={"SegmentationLoss": {"class_weighting": "none", "dice_weight": 0.0}},
        evaluation_settings=evaluation_settings))
    assert changed_loss_trainer.experiment_folder == trainer.experiment_folder
    assert changed_loss_trainer.start_iteration == 8
    assert torch.equal(input=changed_loss_trainer.model.class_weights.cpu(),
                       other=torch.ones(size=(len(trainer.class_names),)))
    assert "dice_loss" not in changed_loss_trainer.tracked_metrics_mapping
    changed_loss_trainer.run_training_loop()

    # The pixel counts were read back, and the training loop, which catches errors and keeps going, never failed
    console_output = capsys.readouterr().out
    assert "Class Pixel Counts Read From" in console_output
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_dice_only_binary_segmentation_training(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies a short training with the Dice loss alone, the binary labels (background and foreground) and the bilinear
    upsampling, without per-class report, sample grids nor predicted masks: the cross entropy is neither tracked nor
    named in the run, the evaluation reports the global metrics only and writes no optional file.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_smoke_dataset(dataset_folder=dataset_folder)
    section_settings = {"SegmentationLoss": {"cross_entropy_weight": 0.0},
                        "SegmentationDecoder": {"upsampling": "bilinear"}}
    configuration_path = write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=4,
        section_settings=section_settings,
        evaluation_settings={"number_sample_images": 0, "save_predicted_masks": False, "per_class_report": False})
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["Data"]["label_mapping"]["mode"] = "binary"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))

    trainer = build_trainer(configuration_path=configuration_path)
    assert trainer.run_name.endswith("_UBL-DBN-D2x64.2x64.2x32.2x32.1x16_BIN")
    assert trainer.class_names == ["background", "foreground"]
    assert "segmentation_loss" not in trainer.tracked_metrics_mapping
    trainer.run_training_loop()

    # Global metrics only, and no optional file in the evaluation folder
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [block["iteration"] for block in results_blocks] == [4]
    assert "Mean IoU" in results_blocks[0]["text"] and "Per Class Metrics" not in results_blocks[0]["text"]
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert sorted(path.name for path in evaluation_folder.iterdir()) == ["class_metrics.csv", "confusion_matrix.csv"]
    test_events = EventAccumulator(path=str(trainer.tensorboard_directory / "Test"))
    test_events.Reload()
    assert test_events.Tags()["images"] == []

    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_tiled_segmentation_training_evaluates_at_the_original_resolution(tmp_path: Path,
                                                                        capsys: pytest.CaptureFixture) -> None:
    """
    Verifies a short training on 64 pixel tiles of 96 pixel images scaled by 0.75, with deep supervision heads on two
    decoder levels and the Lovász and boundary losses next to the cross entropy and the Dice loss: the run is named
    after its tiling and deep supervision, trains on tiles, segments every whole test image alone tile by tile and
    scores it at its original resolution, the predicted masks being saved at that resolution, and the checkpoint
    keeps the deep supervision heads for the resumption.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    original_image_size = 96
    dataset_folder = tmp_path / "synthetic_shapes"
    create_smoke_dataset(dataset_folder=dataset_folder, image_size=original_image_size)
    section_settings = {"SegmentationLoss": {"lovasz_weight": 0.5, "boundary_weight": 0.5, "boundary_distance": 2},
                        "SegmentationDecoder": {"deep_supervision_weights": {"5": 0.5, "3": 0.25}},
                        "Data": {"tiling": {"activated": True, "scale": 0.75, "overlap": 16}}}
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=4,
        section_settings=section_settings,
        evaluation_settings={"number_sample_images": 1, "save_predicted_masks": True, "per_class_report": True,
                             "original_resolution": True, "boundary_tolerance": 1}))
    assert trainer.run_name.endswith("_TL0.75_UTC-DBN-D2x64.2x64.2x32.2x32.1x16-AUX5.3_MC35")
    assert list(trainer.tracked_metrics_mapping) == ["total_loss", "segmentation_loss", "dice_loss", "lovasz_loss",
                                                     "boundary_loss", "deep_supervision_loss", "pixel_accuracy",
                                                     "mean_iou"]

    # Training samples are tiles of the network size, the test images whole scaled images given one at a time
    training_sample = trainer.train_dataloader.dataset[0]
    assert tuple(training_sample["images"].shape[-2:]) == (SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE)
    test_batch = next(iter(trainer.test_dataloader))
    assert tuple(test_batch["images"].shape) == (1, 3, 72, 72)
    with patch.object(target=trainer, attribute="report_evaluation", wraps=trainer.report_evaluation) as report_mock:
        trainer.run_training_loop()
    results = report_mock.call_args.kwargs["results"]

    # Metrics of the original test pixels, with the boundary tolerance of the configuration
    assert results["evaluated_samples"] == f"{NUMBER_TEST_IMAGES} Test Images"
    assert results["notes"][:2] == [f"Labelled Pixels : {NUMBER_TEST_IMAGES * original_image_size ** 2} at the "
                                    f"original resolution of the images, the padding and the unlabelled pixels being "
                                    f"ignored", "Boundary Tolerance : 1 pixels"]
    assert all(0.0 <= metric_value <= 1.0 for metric_value in results["metrics"].values())

    # Predicted masks at the original size of the test images, and one sample grid of the scaled test image
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    predicted_mask_paths = sorted((evaluation_folder / "Predicted_Masks").iterdir())
    assert len(predicted_mask_paths) == NUMBER_TEST_IMAGES
    predicted_mask = cv2.imread(filename=str(predicted_mask_paths[0]), flags=cv2.IMREAD_UNCHANGED)
    assert predicted_mask.shape == (original_image_size, original_image_size)
    sample_image, = results["sample_images"].values()
    assert sample_image.shape[0] == 24 + 72 + 24

    # The deep supervision heads of both levels are saved with the checkpoint for the resumption
    saved_model_state = torch.load(f=trainer.weights_directory / "Iteration_4" / "model_000004.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    assert sorted({parameter_name.split(".")[1] for parameter_name in saved_model_state
                   if parameter_name.startswith("deep_supervision_heads.")}) == ["3", "5"]

    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
