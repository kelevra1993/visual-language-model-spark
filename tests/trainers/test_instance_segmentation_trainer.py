"""
Smoke test of InstanceSegmentationTrainer of vision_training/trainers/instance_segmentation_trainer.py on a synthetic
instance segmentation dataset: training with the geometric augmentations moving the masks saves checkpoints,
evaluates the box and mask average precisions with its sample images, and resumes.
"""

import csv
import pytest
import torch
import yaml

from pathlib import Path

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.trainers.instance_segmentation_trainer import InstanceSegmentationTrainer
from tests.fixtures.evaluation_checks import check_sample_images, read_results_blocks

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                "instance_segmentation" / "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 256

# Anchor scales of the smoke test: the configuration checker requires every scale to fit in the image
SMOKE_ANCHOR_SCALES = {"5": [16, 32, 64], "6": [32, 64, 128], "7": [64, 128, 256]}

# Geometric augmentations turned on, so that the instance masks are moved and their boxes recomputed
SMOKE_GEOMETRIC_AUGMENTATIONS = ["random_crop", "random_left_right_flip", "random_rotation"]


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int) -> Path:
    """
    Writes a copy of the synthetic shapes instance segmentation configuration adapted to a quick training smoke test:
    the generated dataset, 256 pixel images, batches of 2 images, every training image augmented with the geometric
    augmentations, small mask heads, and a saving schedule that triggers an intermediate save, a save with test
    evaluation and a validation pass within a handful of iterations.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.

    Returns:
        Path: The path of the written configuration file.
    """
    configuration = yaml.safe_load(stream=SYNTHETIC_CONFIGURATION_PATH.read_text())

    # Run in the temporary directory, on the generated splits, with a saving schedule fitting a few iterations
    experiment_configuration = configuration["ExperimentConfiguration"]
    experiment_configuration.update({"experiment_folder": str(temporary_directory / "experiments"),
                                     "train_data_folder": str(dataset_folder / "train"),
                                     "validation_data_folder": str(dataset_folder / "validation"),
                                     "test_data_folder": str(dataset_folder / "test"), "batch_size": 2,
                                     "information_dump": 2, "weight_saving_iterations": 4,
                                     "intermediate_weight_saving_iterations": 2, "number_iterations": number_iterations,
                                     "validation_iteration_ratio": 2, "resume_training": True})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": False,
                                                        "compute_model_size": False, "view_input_images": False})

    # Small images and mask heads, two loader workers, and the generated categories mapped to the model classes
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["loading"]["number_workers"] = 2
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    for level, scales in SMOKE_ANCHOR_SCALES.items():
        configuration["Anchors"]["scales_and_ratios"][level]["scales"] = scales
        configuration["MaskHead"]["head_configurations"][level] = [1, 16]
    configuration["MaskHead"]["roi_align_pool_size"] = 7

    # Every training image augmented by the geometric augmentations
    augmentation_configuration = configuration["Data"]["data_augmentation"]
    augmentation_configuration["probability"] = 1.0
    for augmentation_type in SMOKE_GEOMETRIC_AUGMENTATIONS:
        augmentation_configuration["types"][augmentation_type]["activated"] = True

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> InstanceSegmentationTrainer:
    """
    Loads a configuration file and builds the InstanceSegmentationTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        InstanceSegmentationTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return InstanceSegmentationTrainer(experiment_configuration=experiment_configuration,
                                       model_configuration=model_configuration, configuration_path=configuration_path)


def test_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole instance segmentation pipeline on a generated synthetic dataset: 5 iterations with the masks
    moved by the geometric augmentations save the intermediate, evaluated and final checkpoints and track the mask
    loss, the evaluation reports the mask then box tables of the methods and of the classes, the mask mAP@0.5 of the
    area method as headline metric, the box scalars under the object detection names and the mask scalars under
    'Mask' names, class_metrics.csv with the box and mask columns, and the sample images. A second run with the same
    settings resumes from the last checkpoint with exactly its weights and trains 2 more iterations, without any
    failed step.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=4,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="instance_segmentation")

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5))
    assert trainer.start_iteration == 1 and trainer.experiment_folder.name.startswith("INS_")
    assert "mask_loss" in trainer.get_tracked_metrics()
    trainer.run_training_loop()

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    results, averaged_weights_results = trainer.run_test_evaluation(iteration=5)
    assert averaged_weights_results is None
    assert results["headline_metric"] == "Mask mAP@0.5/area" and results["headline_metric"] in results["scalars"]
    assert {"mAP@0.5/area", "Mask mAP@[0.5:0.95]/101_point"} <= set(results["scalars"])

    # Report of the two evaluations, with the mask then box tables
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [block["iteration"] for block in results_blocks] == [4, 5]
    assert [table["title"] for table in results["tables"]] == [
        "Mask Mean Average Precision", "Box Mean Average Precision", "Mask Per Class Average Precision (Area)",
        "Box Per Class Average Precision (Area)"]
    assert "Mask Per Class Average Precision (Area)" in results_blocks[1]["text"]
    assert all(scalar_name.startswith(("AP@0.5/", "Mask AP@0.5/")) for scalar_name in results["class_scalars"])

    # Box and mask average precisions of every class, the ground truth objects summing to those of the test images
    test_dataset = trainer.test_dataloader.dataset
    number_objects = sum(test_dataset[sample_index]["instance_masks"].shape[0]
                         for sample_index in range(len(test_dataset)))
    with open(file=trainer.experiment_folder / "Evaluation" / "Iteration_5" / "class_metrics.csv", mode="r",
              newline="") as class_metrics_file:
        class_rows = list(csv.DictReader(f=class_metrics_file))
    assert list(class_rows[0]) == ["class_index", "class", "ground_truths", "average_precision_50",
                                   "average_precision_75", "average_precision_50_95", "mask_average_precision_50",
                                   "mask_average_precision_75", "mask_average_precision_50_95"]
    assert sum(int(class_row["ground_truths"]) for class_row in class_rows) == number_objects
    assert [class_row["class"] for class_row in class_rows] == list(SHAPE_CATEGORIES)

    # Every test image is a sample image, its ground truth panel next to its detection panel
    check_sample_images(evaluation_folder=trainer.experiment_folder / "Evaluation" / "Iteration_5", results=results,
                        number_sample_images=4, sample_kind="images", correct_predictions=None)

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])
    resumed_trainer.run_training_loop()
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_7"'

    # The training loop catches errors and keeps going, so make sure that no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
