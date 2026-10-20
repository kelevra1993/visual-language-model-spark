"""
Smoke test of KeypointTrainer of vision_training/trainers/keypoint_trainer.py on a generated synthetic keypoint
dataset, on the CPU: training on the whole images saves checkpoints, evaluates the objects found in the test images
with the object keypoint similarity (the average precision tables, the matched objects, PCK and mean OKS, the
per-class and per-keypoint tables, their CSV files and the sample images) and resumes from its last checkpoint. With
'KeypointHead.ignore_unlabelled_objects', the ignored regions of an object without labelled keypoint go from the
training loader through the augmentation to the heatmap loss.
"""

import csv
import json
import pytest
import torch
import yaml

from pathlib import Path
from typing import Dict, List

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.augmentations import AUGMENTATION_TYPES, DataAugmenter
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.evaluation.keypoints.object_keypoint_similarity import CORRECT_KEYPOINTS_NAME
from vision_training.trainers.keypoint_trainer import KeypointTrainer
from tests.fixtures.evaluation_checks import check_sample_images, read_results_blocks

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                "keypoint_detection" / "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations on the CPU, and a multiple of the stride 32
# of the lowest branch of the 3 stages of the synthetic high resolution network
SMOKE_IMAGE_SIZE = 128

# Number of sample images asked for, more than the generated test images so that every one of them is drawn
SMOKE_NUMBER_SAMPLE_IMAGES = 30


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int,
                              ignore_unlabelled_objects: bool = False) -> Path:
    """
    Writes a copy of the synthetic keypoint configuration adapted to a quick training smoke test: the generated
    dataset, 128 pixel images, batches of 2 images, a saving schedule that triggers an intermediate save, a save with
    test evaluation and a validation pass within a handful of iterations, a sample image of every test image, no
    moving average of the weights and none of the slow diagnoses.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the experiments.
        dataset_folder (Path): The root folder of the generated synthetic dataset.
        number_iterations (int): The number of training iterations of the run.
        ignore_unlabelled_objects (bool): Whether the heatmap loss leaves out the objects without labelled keypoint
            ('KeypointHead.ignore_unlabelled_objects'). Defaults to False.

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
    experiment_configuration["evaluation"]["number_sample_images"] = SMOKE_NUMBER_SAMPLE_IMAGES
    # Trained weights only, the evaluation of the moving average of the weights being tested by the base trainer tests
    configuration["Optimization"]["exponential_moving_average"]["activated"] = False

    # Read the small images with two loader workers per split, and map the generated categories to the model classes
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["loading"]["number_workers"] = 2
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")

    # Leave the objects without labelled keypoint out of the heatmap loss when asked
    configuration["KeypointHead"]["ignore_unlabelled_objects"] = ignore_unlabelled_objects

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> KeypointTrainer:
    """
    Loads a configuration file and builds the KeypointTrainer from it, exactly like main.py does, the loader
    injecting the keypoint definitions of the generated training labels.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        KeypointTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    return KeypointTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                           configuration_path=configuration_path)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Reads the rows of a metrics file written by the keypoint evaluation, keyed by the column names of its header.

    Args:
        csv_path (Path): The path of the CSV file.

    Returns:
        List[Dict[str, str]]: The rows of the file.
    """
    with open(file=csv_path, mode="r", newline="") as csv_file:
        return list(csv.DictReader(f=csv_file))


def test_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole keypoint training pipeline on a generated synthetic dataset: 5 iterations save the intermediate
    and evaluated checkpoints and the final one, evaluate the test images into Results.txt (the OKS average precision
    tables of the methods, object sizes and recalls, the per-class OKS average precisions, and the keypoint accuracy
    per class and per keypoint, with the matched objects, PCK and mean OKS of all the keypoints), class_metrics.csv
    (one row per class with its objects summing to the objects of the test images, its average precisions and its
    keypoint accuracy), keypoint_metrics.csv (one row per keypoint of every class) and a sample image of every test
    image, and write the three TensorBoard folders. A second run with the same settings resolves to the same run
    folder, resumes from the last checkpoint with exactly its weights, and trains 2 more iterations. No training step
    may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_keypoints"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 4, "validation": 2, "test": 2}, maximum_objects=2,
                             minimum_frames=1, maximum_frames=1, number_workers=1, task="keypoint_detection")

    # First run: saves at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=5))
    assert trainer.start_iteration == 1
    trainer.run_training_loop()

    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_5"'
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train", "Validation"]
    results, averaged_weights_results = trainer.run_test_evaluation(iteration=5)
    assert averaged_weights_results is None
    assert results["headline_metric"] == "OKS mAP@0.5/area" and results["headline_metric"] in results["scalars"]

    # Every test image is evaluated, in both evaluations of Results.txt, with the objects holding a labelled keypoint
    test_dataset = trainer.test_dataloader.dataset
    number_test_images = len(test_dataset)
    number_test_objects = sum(len(test_dataset[image_index]["labels"]) for image_index in range(number_test_images))
    assert number_test_images == 2 and number_test_objects > 0
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [
        (4, f"{number_test_images} Test Images"), (5, f"{number_test_images} Test Images")]

    # Tables of the OKS average precisions, then of the keypoint accuracy per class and per keypoint, all written in
    # Results.txt with the matched objects, the PCK and the mean OKS of all the keypoints
    table_titles = ["OKS Mean Average Precision", "OKS Mean Average Precision By Object Size (Area)",
                    "OKS Mean Average Recall", "OKS Per Class Average Precision (Area)", "Keypoint Accuracy Per Class",
                    "Keypoint Accuracy Per Keypoint"]
    assert [table["title"] for table in results["tables"]] == table_titles
    assert all(table_title in results_blocks[1]["text"] for table_title in table_titles)
    assert list(results["metrics"]) == ["Matched Objects", CORRECT_KEYPOINTS_NAME, "Mean OKS",
                                        "Mean Keypoint Error (pixels)"]
    assert all(0.0 <= results["metrics"][metric_name] <= 1.0
               for metric_name in ["Matched Objects", CORRECT_KEYPOINTS_NAME, "Mean OKS"])
    assert CORRECT_KEYPOINTS_NAME in results_blocks[1]["text"] and "Mean OKS" in results_blocks[1]["text"]
    assert {f"Keypoint Accuracy/{CORRECT_KEYPOINTS_NAME}", "Keypoint Accuracy/Mean OKS"} <= set(results["scalars"])
    assert all(scalar_name.startswith("OKS AP@0.5/") for scalar_name in results["class_scalars"])

    # One row per class in class_metrics.csv, its objects summing to the objects of the test images, with the OKS
    # average precisions and the keypoint accuracy of the class
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_5"
    class_rows = read_csv_rows(csv_path=evaluation_folder / "class_metrics.csv")
    assert {"class_index", "class", "objects", "matched_objects", "labelled_keypoints", "percentage_correct_keypoints",
            "mean_object_keypoint_similarity", "mean_error_pixels"} <= set(class_rows[0])
    assert any(column_name.startswith("oks_") for column_name in class_rows[0])
    assert [class_row["class"] for class_row in class_rows] == [trainer.class_names[class_index]
                                                                for class_index in sorted(trainer.class_names)
                                                                if class_index != 0]
    assert sum(int(class_row["objects"]) for class_row in class_rows) == number_test_objects

    # One row per keypoint of every class in keypoint_metrics.csv, in the order of the keypoint definitions
    keypoint_rows = read_csv_rows(csv_path=evaluation_folder / "keypoint_metrics.csv")
    assert list(keypoint_rows[0]) == ["class_index", "class", "keypoint", "labelled", "found",
                                      "percentage_correct_keypoints", "mean_error_pixels"]
    assert [(int(keypoint_row["class_index"]), keypoint_row["keypoint"]) for keypoint_row in keypoint_rows] == [
        (class_index, keypoint_name) for class_index, definition in sorted(trainer.keypoint_categories.items())
        for keypoint_name in definition["keypoints"]]
    assert results["files"]["Class Metrics"] == evaluation_folder / "class_metrics.csv"
    assert results["files"]["Keypoint Metrics"] == evaluation_folder / "keypoint_metrics.csv"

    # Every test image is a sample image named after its file: its true objects next to the objects found, between
    # the caption and legend bars, the caption bar at least as wide as the two panels
    check_sample_images(evaluation_folder=evaluation_folder, results=results, number_sample_images=number_test_images,
                        sample_kind="images", correct_predictions=None)
    assert sorted(results["sample_images"]) == sorted(f"Samples/{Path(file_name).stem}"
                                                      for file_name in trainer.test_image_file_names.values())
    assert all(sample_image.shape[0] == 24 + SMOKE_IMAGE_SIZE + 24 and sample_image.shape[1] >= 2 * SMOKE_IMAGE_SIZE + 4
               for sample_image in results["sample_images"].values())
    first_run_output = capsys.readouterr().out

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=2))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6
    saved_model_state = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                                   weights_only=True)["model_state"]
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=saved_model_state[parameter_name])

    # The resumed run trains iterations 6 (intermediate save) and 7 (end of the run)
    resumed_trainer.run_training_loop()
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5", "Iteration_6",
                                                                                 "Iteration_7"]
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_7"'

    # The training loop catches errors and keeps going, so make sure that no iteration of both runs failed
    console_output = first_run_output + capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_ignored_regions_go_from_the_loader_to_the_heatmap_loss(tmp_path: Path) -> None:
    """
    Verifies the path of the ignored regions through the trainer with 'KeypointHead.ignore_unlabelled_objects': a
    training object of a category with keypoints, none of them labelled, drawn by the polygon of its box, gives the
    [B, S, S] uint8 ignored regions of the training batch, the validation batches carrying regions too and the test
    batches, which compute no loss, none; the augmentation moves the regions as the binary mask of the augmenter, a
    left-right flip of every image flipping them; and the forward pass gives the heatmap loss of the model with the
    regions, below the loss without them.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    dataset_folder = tmp_path / "synthetic_keypoints"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 2, "validation": 2, "test": 2}, maximum_objects=2,
                             minimum_frames=1, maximum_frames=1, number_workers=1, task="keypoint_detection")

    # Every training and validation object is drawn by the polygon of its box, the generated shapes having no
    # segmentation, and the first training object becomes an object of a category with keypoints, none labelled,
    # drawn by the right triangle of the top left half of its box, which a left-right flip always changes
    for split_name in ["train", "validation"]:
        labels_path = dataset_folder / split_name / "labels.json"
        labels = json.loads(s=labels_path.read_text())
        for annotation in labels["annotations"]:
            left, top, width, height = annotation["bbox"]
            annotation["segmentation"] = [[left, top, left + width, top, left + width, top + height, left,
                                           top + height]]
        if split_name == "train":
            keypoint_category = next(category for category in labels["categories"] if len(category["keypoints"]) > 0)
            left, top, width, height = labels["annotations"][0]["bbox"]
            labels["annotations"][0].update({"category_id": keypoint_category["id"], "num_keypoints": 0,
                                             "keypoints": [0] * (3 * len(keypoint_category["keypoints"])),
                                             "segmentation": [[left, top, left + width, top, left, top + height]]})
        labels_path.write_text(data=json.dumps(obj=labels))
    trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=1,
        ignore_unlabelled_objects=True))

    # The training batch of the two images carries the binary regions, the validation batches too, the test ones none
    training_batch = next(iter(trainer.train_dataloader))
    ignored_regions = training_batch["ignored_regions"]
    assert ignored_regions.shape == (2, SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE) and ignored_regions.dtype == torch.uint8
    assert set(ignored_regions.unique().tolist()) == {0, 1}
    validation_regions = next(iter(trainer.validation_dataloader))["ignored_regions"]
    assert validation_regions.shape == (2, SMOKE_IMAGE_SIZE, SMOKE_IMAGE_SIZE)
    assert "ignored_regions" not in next(iter(trainer.test_dataloader))

    # The augmentation moves the regions as the binary mask of the augmenter, here a left-right flip of every image
    prepared_batch = trainer.prepare_batch(batch=training_batch)
    flipped_regions = prepared_batch["ignored_regions"].flip(dims=[2])
    assert not torch.equal(input=flipped_regions, other=prepared_batch["ignored_regions"])
    trainer.data_augmenter = DataAugmenter(augmentation_configuration={
        "activated": True, "probability": 1.0, "independent_frames": False,
        "shuffle_frames": {"activated": False, "probability": 0.0},
        "types": {name: {"activated": name == "random_left_right_flip", "probability": 1.0}
                  for name in AUGMENTATION_TYPES}})
    augmented_batch = trainer.augment_batch(batch=prepared_batch)
    assert torch.equal(input=augmented_batch["ignored_regions"], other=flipped_regions)

    # The forward pass gives the heatmap loss of the model with the regions, which leave out some of its terms
    with torch.no_grad():
        outputs = trainer.forward_pass(batch=prepared_batch)
        regions_outputs = trainer.model(input_tensor=prepared_batch["images"],
                                        ground_truth_keypoints=prepared_batch["keypoints"],
                                        ground_truth_labels=prepared_batch["labels"],
                                        ignored_regions=prepared_batch["ignored_regions"])
        every_cell_outputs = trainer.model(input_tensor=prepared_batch["images"],
                                           ground_truth_keypoints=prepared_batch["keypoints"],
                                           ground_truth_labels=prepared_batch["labels"])
    assert torch.allclose(input=outputs["heatmap_loss"], other=regions_outputs["heatmap_loss"])
    assert outputs["heatmap_loss"].item() < every_cell_outputs["heatmap_loss"].item()
