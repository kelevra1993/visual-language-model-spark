"""
Smoke test of DetectionTrainer of vision_training/trainers/detection_trainer.py on a synthetic dataset: training saves
checkpoints, evaluates with its sample images, its rates written as percentages in the report, and resumes. The
training target viewer ('view_training_targets') shows every training image with the targets of its losses and exits
without creating the run folder, skipped without a display, and the detections of the sample and error images are
captioned with their overlap with the object the matching gives them.
"""

import csv
import numpy as np
import pytest
import re
import torch
import yaml

from pathlib import Path
from unittest.mock import patch

from vision_training.data.synthetic.create_synthetic_dataset import SHAPE_CATEGORIES, create_synthetic_dataset
from vision_training.evaluation.classification.confusion_matrix import save_confusion_matrix_image
from vision_training.evaluation.detection.object_sizes import OBJECT_SIZE_NAMES
from vision_training.trainers.detection_trainer import DRAWN_NEGATIVE_TARGETS, DetectionTrainer
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.utilities.visualization import (GRID_GAP, LEGEND_BAR_HEIGHT, TRAINING_TARGET_VIEWER_NAME,
                                                     PanelTable, build_class_colours, build_detection_sample_grid)
from tests.fixtures.evaluation_checks import (check_detection_error_analysis, check_sample_images,
                                              read_results_blocks)

SYNTHETIC_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                                "configuration_synthetic_shapes.yaml")

# Image size of the smoke test, small enough for a few quick iterations
SMOKE_IMAGE_SIZE = 256

# Anchor scales of the smoke test: the configuration checker requires every scale to fit in the image
SMOKE_ANCHOR_SCALES = {"5": [16, 32, 64], "6": [32, 64, 128], "7": [64, 128, 256]}

# Largest number of error images saved per folder by the smoke test
SMOKE_MAXIMUM_ERROR_IMAGES = 2


def write_smoke_configuration(temporary_directory: Path, dataset_folder: Path, number_iterations: int) -> Path:
    """
    Writes a copy of the synthetic shapes configuration adapted to a quick training smoke test: the generated
    dataset, 256 pixel images, the Native loader, batches of 2 images, a saving schedule that triggers an
    intermediate save, a save with test evaluation and a validation pass within a handful of iterations, the error
    images of the evaluation, at most 2 per folder, and no moving average of the weights.

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
                                     "validation_iteration_ratio": 2})
    experiment_configuration["diagnosis_modes"].update({"compute_receptive_field": False, "view_input_images": False})
    # Trained weights only, the evaluation of the moving average of the weights being tested by the base trainer tests
    configuration["Optimization"]["exponential_moving_average"]["activated"] = False

    # Error images of the evaluation, a few per folder, with the deployment threshold lowered to the evaluation one so
    # that the barely trained detector gives false positives and confusions next to its missed objects
    experiment_configuration["evaluation"]["error_dumps"] = {"activated": True,
                                                             "maximum_per_class_pair": SMOKE_MAXIMUM_ERROR_IMAGES}
    evaluation_score_threshold = experiment_configuration["evaluation"]["score_threshold"]
    configuration["Detector"]["inference"]["score_threshold"] = evaluation_score_threshold

    # Read the small images, and map the generated categories to the model classes
    configuration["Data"]["image_settings"]["size"] = SMOKE_IMAGE_SIZE
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(dataset_folder / "label_mapping.json")
    for level, scales in SMOKE_ANCHOR_SCALES.items():
        configuration["DetectionLevels"][level]["anchors"]["scales"] = scales

    configuration_path = temporary_directory / f"configuration_{number_iterations}_iterations.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def build_trainer(configuration_path: Path) -> DetectionTrainer:
    """
    Loads a configuration file and builds the DetectionTrainer from it, exactly like main.py does.

    Args:
        configuration_path (Path): The configuration file of the run.

    Returns:
        DetectionTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    return DetectionTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                   configuration_path=configuration_path)


def test_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole training pipeline on a generated synthetic dataset: the first launch measures the anchor
    recall of the training and test objects for the five relative object sizes, prints one line per split and writes
    anchor_recall.csv and anchor_recall.png to the run folder, 5 iterations save the intermediate and evaluated
    checkpoints and the final one, evaluate the test split into Results.txt (the tables of the average precision
    methods, of the object sizes, of the mean average recall and of the proposal recall, and the per-class table with
    the class thresholds),
    class_metrics.csv (the ground truth boxes of every class summing to those of the test images, the average
    precisions of every object size), proposal_recall.csv, the confusion matrix, class thresholds and error images of
    the detections (check_detection_error_analysis) and the sample images of the test images, and write the three
    TensorBoard folders. A second run with the same settings resolves to the same run folder, keeps the anchor recall
    of the first launch, resumes from the last checkpoint with exactly its weights, and trains up to the total of 7
    iterations; an anchor recall measured at another threshold is measured again. No step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=4,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="object_detection")

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
    # Evaluation of the last checkpoint, its confusion matrix image drawn with the percentages of the report
    with patch(target="vision_training.trainers.detection_trainer.save_confusion_matrix_image", autospec=True,
               side_effect=save_confusion_matrix_image) as confusion_matrix_mock:
        results, averaged_weights_results = trainer.run_test_evaluation(iteration=5)
    assert confusion_matrix_mock.call_args.kwargs["in_percent"] is True
    assert averaged_weights_results is None
    assert results["headline_metric"] == "mAP@0.5/area" and results["headline_metric"] in results["scalars"]

    # Report of the two evaluations, with the table of the methods and the per-class table
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [
        (4, "4 Test Images"), (5, "4 Test Images")]
    assert [table["title"] for table in results["tables"]] == ["Mean Average Precision",
                                                              "Mean Average Precision By Object Size (Area)",
                                                              "Mean Average Recall", "Proposal Recall",
                                                              "Per Class Average Precision (Area)", "Per Class Recall"]
    assert "Per Class Average Precision (Area)" in results_blocks[1]["text"]
    assert "Proposal Recall" in results_blocks[1]["text"]
    assert all(scalar_name.startswith("AP@0.5/") for scalar_name in results["class_scalars"])
    assert {"Proposal Recall@0.5/all", "Proposal Recall@0.7/block_5", "Proposals Per Image/all"} <= set(
        results["scalars"])
    assert any(scalar_name.startswith("mAP@0.5 By Size/") for scalar_name in results["scalars"])
    assert {"mAR@0.5/all", "mAR@[0.5:0.95]/all"} <= set(results["scalars"])

    # The rates of every table written as percentages in Results.txt, the headline mAP@0.5 in the best checkpoint line
    assert results["percent_metrics"] == ["mAP@0.5/area"]
    assert all(table["percent_columns"] for table in results["tables"])
    area_row = next(line for line in results_blocks[1]["text"].splitlines() if line.strip().startswith("Area"))
    assert re.fullmatch(pattern=r"\s+Area(\s+\d+\.\d{2} %){4}", string=area_row)

    # The ground truth boxes of every class sum to the boxes of the test images, with the average precisions of every
    # object size before the thresholds
    test_dataset = trainer.test_dataloader.dataset
    number_boxes = sum(test_dataset[sample_index]["bounding_boxes"].shape[0]
                       for sample_index in range(len(test_dataset)))
    with open(file=trainer.experiment_folder / "Evaluation" / "Iteration_5" / "class_metrics.csv", mode="r",
              newline="") as class_metrics_file:
        class_rows = list(csv.DictReader(f=class_metrics_file))
    assert list(class_rows[0]) == ["class_index", "class", "ground_truths", "average_precision_50",
                                   "average_precision_75", "average_precision_95", "average_precision_50_95"] + [
        f"average_precision_{threshold}_{object_size}" for object_size in OBJECT_SIZE_NAMES
        for threshold in ["50", "50_95"]] + [f"recall_{threshold}{bound}" for threshold in ["50", "75"]
                                             for bound in ["", "_lower", "_upper"]] + [
        "threshold_at_target_precision", "recall_at_target_precision"]
    assert sum(int(class_row["ground_truths"]) for class_row in class_rows) == number_boxes
    assert [class_row["class"] for class_row in class_rows] == list(SHAPE_CATEGORIES)

    # Proposal recall of the test objects: every object counted once, by size, and the proposals of every level
    # summing to those of all the levels
    with open(file=trainer.experiment_folder / "Evaluation" / "Iteration_5" / "proposal_recall.csv", mode="r",
              newline="") as proposal_recall_file:
        proposal_rows = {row["group"]: row for row in csv.DictReader(f=proposal_recall_file)}
    assert list(proposal_rows) == ["all", *OBJECT_SIZE_NAMES, "block_5", "block_6", "block_7"]
    assert int(proposal_rows["all"]["objects"]) == number_boxes == sum(
        int(proposal_rows[object_size]["objects"]) for object_size in OBJECT_SIZE_NAMES)
    assert float(proposal_rows["all"]["proposals_per_image"]) == pytest.approx(expected=sum(
        float(proposal_rows[f"block_{level}"]["proposals_per_image"]) for level in [5, 6, 7]), abs=1e-5)
    assert results["files"]["Proposal Recall"].name == "proposal_recall.csv"

    # Anchor recall of the training and test objects, written to the run folder at the first launch, all the objects
    # then every size for each split, with the split folder and the threshold it was measured with
    train_dataset = trainer.train_dataloader.dataset
    number_training_boxes = sum(train_dataset[sample_index]["bounding_boxes"].shape[0]
                                for sample_index in range(len(train_dataset)))
    anchor_recall_path, anchor_figure_path = (trainer.experiment_folder / "anchor_recall.csv",
                                              trainer.experiment_folder / "anchor_recall.png")
    assert trainer.training_data_analysis_files == {"Anchor Recall": anchor_recall_path,
                                                    "Anchor Recall Figure": anchor_figure_path}
    assert anchor_figure_path.stat().st_size > 0
    with open(file=anchor_recall_path, mode="r", newline="") as anchor_recall_file:
        anchor_rows = list(csv.DictReader(f=anchor_recall_file))
    assert list(anchor_rows[0]) == ["split", "split_folder", "iou_threshold", "objects", "count", "recall_block_5",
                                    "recall_block_6", "recall_block_7", "recall_all_levels"]
    assert [(row["split"], row["objects"]) for row in anchor_rows] == [
        (split_name, group) for split_name in ["Training", "Test"] for group in ["all", *OBJECT_SIZE_NAMES]]
    assert {row["split_folder"] for row in anchor_rows[:6]} == {str(dataset_folder / "train")}
    foreground_threshold = trainer.model_configuration["RegionProposal"]["foreground_iou_threshold"]["min"]
    assert {float(row["iou_threshold"]) for row in anchor_rows} == {foreground_threshold}
    assert int(anchor_rows[0]["count"]) == number_training_boxes and int(anchor_rows[6]["count"]) == number_boxes
    assert sum(int(row["count"]) for row in anchor_rows[1:6]) == number_training_boxes
    assert float(anchor_rows[0]["recall_all_levels"]) >= max(float(anchor_rows[0][f"recall_block_{level}"])
                                                             for level in [5, 6, 7])

    # One console line per split instead of the tables, which only the figure draws
    first_run_output = capsys.readouterr().out
    assert "Anchor Recall Of The Training Objects" not in first_run_output
    assert re.search(pattern=rf"Anchor Recall Of The {number_training_boxes} Training Objects: \d+\.\d{{2}} % In All "
                             rf"Levels, \d+\.\d{{2}} % For The Smallest Objects \(", string=first_run_output)
    assert f"Anchor Recall Of The {number_boxes} Test Objects: " in first_run_output
    figure_modification_time = anchor_figure_path.stat().st_mtime_ns

    # Confusion matrix with its background row and column, class thresholds and error images of the detections
    check_detection_error_analysis(
        evaluation_folder=trainer.experiment_folder / "Evaluation" / "Iteration_5", results=results,
        class_names=[trainer.class_names[class_index] for class_index in range(len(trainer.class_names))],
        number_objects=number_boxes, maximum_per_class_pair=SMOKE_MAXIMUM_ERROR_IMAGES,
        target_precision=trainer.experiment_configuration["evaluation"]["target_precision"])

    # Every test image is a sample image (8 asked for 4 images), named after its file: its ground truth panel next to
    # its detection panel, between the caption and legend bars
    check_sample_images(evaluation_folder=trainer.experiment_folder / "Evaluation" / "Iteration_5", results=results,
                        number_sample_images=4, sample_kind="images", correct_predictions=None)
    assert sorted(results["sample_images"]) == sorted(f"Samples/{Path(file_name).stem}"
                                                      for file_name in trainer.test_image_file_names.values())
    assert all(sample_image.shape[0] == 24 + SMOKE_IMAGE_SIZE + 24 and sample_image.shape[1] >= 2 * SMOKE_IMAGE_SIZE + 4
               for sample_image in results["sample_images"].values())

    # Second run with the same settings: same run folder, resumed from iteration 5 with exactly its weights
    resumed_trainer = build_trainer(configuration_path=write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=7))
    assert resumed_trainer.experiment_folder == trainer.experiment_folder
    assert resumed_trainer.start_iteration == 6

    # The resumed launch keeps the anchor recall of the first launch instead of measuring it again, and has no new
    # analysis file to send to MLflow
    resumed_output = capsys.readouterr().out
    assert "Anchor recall measured at the first launch of the run" in resumed_output
    assert resumed_trainer.training_data_analysis_files == {}
    assert "Training Objects: " not in resumed_output
    assert anchor_figure_path.stat().st_mtime_ns == figure_modification_time
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

    # An anchor recall measured at another threshold is measured again at the next launch
    anchor_recall_path.write_text(data=anchor_recall_path.read_text().replace(
        f",{foreground_threshold:.6f},", ",0.123000,"))
    assert resumed_trainer.analyse_training_data() == {"Anchor Recall": anchor_recall_path,
                                                       "Anchor Recall Figure": anchor_figure_path}
    with open(file=anchor_recall_path, mode="r", newline="") as anchor_recall_file:
        assert {float(row["iou_threshold"]) for row in csv.DictReader(f=anchor_recall_file)} == {foreground_threshold}
    assert anchor_figure_path.stat().st_mtime_ns != figure_modification_time

    # The training loop catches errors and keeps going, so make sure that no iteration of both runs failed
    console_output = first_run_output + resumed_output + capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_training_target_viewer_shows_the_targets_of_every_training_image(tmp_path: Path,
                                                                          capsys: pytest.CaptureFixture,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies the 'view_training_targets' diagnosis mode on a generated synthetic dataset: without a display the viewer
    is skipped with its message, and with one every training image is shown in one window of the training target
    viewer, its image the three target panels above their legend, its panel counting the anchors of the model, the
    sampled regions of interest of the 'Detector' section (all of them, the barely trained detector giving enough
    proposals) with every object of the image copied among the positives, and a table of the targets of every object,
    the weights being the initial ones; the program then exits without training nor creating the run folder.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture removing the display variables.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 4, "validation": 2, "test": 2}, maximum_objects=3,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="object_detection")
    configuration_path = write_smoke_configuration(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                   number_iterations=5)
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["ExperimentConfiguration"]["diagnosis_modes"]["view_training_targets"] = True
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))

    # Without a display, the viewer is skipped with its message and the program exits
    monkeypatch.delenv(name="DISPLAY", raising=False)
    monkeypatch.delenv(name="WAYLAND_DISPLAY", raising=False)
    with pytest.raises(expected_exception=SystemExit):
        build_trainer(configuration_path=configuration_path)
    printed_output = capsys.readouterr().out
    assert "Skipping the training target viewer." in printed_output
    assert "Weights : initial (no checkpoint in the run folder)" in printed_output

    # With a display, one window per training image, SPACE pressed on every one
    with patch(target="vision_training.trainers.detection_trainer.view_input_data", return_value=False) as view_mock:
        with pytest.raises(expected_exception=SystemExit):
            build_trainer(configuration_path=configuration_path)
    assert not (tmp_path / "experiments").exists() or not any((tmp_path / "experiments").iterdir())
    assert view_mock.call_count == 4
    image_side = SMOKE_IMAGE_SIZE
    for call in view_mock.call_args_list:
        assert call.kwargs["viewer_name"] == TRAINING_TARGET_VIEWER_NAME
        (window,) = call.kwargs["windows"]
        assert window.title == "Training Targets" and window.bounding_boxes is None
        assert tuple(window.image.shape) == (3, image_side + LEGEND_BAR_HEIGHT, 3 * image_side + 2 * GRID_GAP)

        # Counts of the panel: the anchors of the model, every sampled region, every object copied among them
        lines = [line for line in window.target_lines if isinstance(line, str)]
        number_objects = int(next(line for line in lines if line.startswith("Objects :")).split(": ")[1])
        assert any(line.startswith("Anchors : ") and f"{DRAWN_NEGATIVE_TARGETS} of the sampled negatives" in line
                   for line in lines)
        total_samples = configuration["Detector"]["total_training_samples"]
        assert any(line.startswith(f"Sampled RoIs : {total_samples}, ") for line in lines)
        assert any(f"and {number_objects} ground truth copies (dashed)" in line for line in lines)
        (table,) = [line for line in window.target_lines if isinstance(line, PanelTable)]
        assert table.row_titles == ("Object", "Positive anchors", "Best proposal IoU", "Positive RoIs")
        assert len(table.columns) == number_objects
        assert all(int(column[3]) >= 1 for column in table.columns)
        assert any(line.startswith("Augmentations") for line in window.augmentation_lines)


def build_caption_test_trainer() -> DetectionTrainer:
    """
    Builds a DetectionTrainer holding only the attributes read by draw_test_image, without building the model: the
    deployment score threshold, the class names and colours of a circle and a square.

    Returns:
        DetectionTrainer: The partial trainer.
    """
    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.deployment_score_threshold = 0.5
    trainer.class_names = {0: "background", 1: "circle", 2: "square"}
    trainer.class_colours = build_class_colours(number_classes=3)
    return trainer


def test_sample_image_captions_give_the_overlap_with_the_matched_object() -> None:
    """
    Verifies the overlaps given to the captions of the detections of a sample image: the detections are matched as in
    the evaluation (the best scored circle finds the circle, its less overlapping duplicate finds nothing, the circle
    detected on the square finds the square as a confusion), each giving its box overlap with its object or, unmatched,
    with the object it overlaps the most, the detection under the deployment threshold being left out. An error image
    reads the matching of the evaluation instead of matching again.
    """
    batch = {"images": torch.zeros(size=(1, 3, 64, 64)),
             "bounding_boxes": [torch.tensor(data=[[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 60.0, 60.0]])],
             "labels": [torch.tensor(data=[1, 2])]}
    predictions = {"number_detections": torch.tensor(data=[4]),
                   "final_detection_scores": torch.tensor(data=[[0.9, 0.8, 0.7, 0.2]]),
                   "final_detection_boxes": torch.tensor(data=[[[10.0, 10.0, 30.0, 32.0], [12.0, 12.0, 30.0, 30.0],
                                                                [40.0, 40.0, 60.0, 60.0], [0.0, 0.0, 5.0, 5.0]]]),
                   "final_detection_labels": torch.tensor(data=[[1, 1, 1, 2]])}
    trainer = build_caption_test_trainer()
    with patch(target="vision_training.trainers.detection_trainer.build_detection_sample_grid", autospec=True,
               side_effect=build_detection_sample_grid) as grid_mock:
        trainer.build_sample_image(batch=batch, predictions=predictions, sample_index=0)
    grid_arguments = grid_mock.call_args.kwargs
    assert np.allclose(grid_arguments["detection_overlaps"]["boxes"], [400 / 440, 324 / 400, 1.0])
    assert grid_arguments["matched_detections"].tolist() == [True, False, True]

    # The matching of the evaluation, read as it is by an error image
    with patch(target="vision_training.trainers.detection_trainer.build_detection_sample_grid", autospec=True,
               side_effect=build_detection_sample_grid) as grid_mock:
        trainer.draw_test_image(batch=batch, predictions=predictions, sample_index=0, error_description="error",
                                highlighted_ground_truths=None, highlighted_detections=None,
                                matched_ground_truth_indices=torch.tensor(data=[-1, 0, 1]))
    grid_arguments = grid_mock.call_args.kwargs
    assert np.allclose(grid_arguments["detection_overlaps"]["boxes"], [400 / 440, 324 / 400, 1.0])
    assert grid_arguments["matched_detections"].tolist() == [False, True, True]


def test_confusion_matrix_keeps_the_detections_reaching_their_class_threshold() -> None:
    """
    Verifies that the confusion matrix of an evaluation counts the detections the exported detector returns, every
    detection kept when its score reaches the threshold of its class in class_thresholds.json, or the configured
    deployment threshold for a class that never reaches the target precision: a square at 0.3 kept by the threshold of
    its class (0.25), a circle at 0.4 dropped by the configured one (0.5), and a triangle at 0.6 dropped by the
    threshold of its class (0.7), although it reaches the configured one.
    """
    # Only the attributes read by the confusion matrix, without building the model
    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.class_names = {0: "background", 1: "square", 2: "circle", 3: "triangle"}
    trainer.deployment_score_threshold = 0.5

    # One image of three objects, each found by a detection of its class
    object_boxes = torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]])
    accumulator = {"detections": [{"boxes": object_boxes.clone(), "scores": torch.tensor(data=[0.3, 0.4, 0.6]),
                                   "labels": torch.tensor(data=[1, 2, 3])}],
                   "ground_truths": [{"boxes": object_boxes, "labels": torch.tensor(data=[1, 2, 3])}]}
    confusion_matrix = trainer.build_deployed_confusion_matrix(
        accumulator=accumulator, class_score_thresholds={1: (0.25, 0.9), 2: None, 3: (0.7, 0.8)})

    # The square found, the circle and the triangle missed
    assert confusion_matrix.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]

    # Without class thresholds, every class at the configured threshold: only the triangle found
    confusion_matrix = trainer.build_deployed_confusion_matrix(
        accumulator=accumulator, class_score_thresholds={1: None, 2: None, 3: None})
    assert confusion_matrix.tolist() == [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
