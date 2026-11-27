"""
Tests of the class score thresholds of the exported detectors (vision_training/deployment/detection_deployment.py):
the thresholds of the evaluation of the exported checkpoint found for the trained or the averaged weights
(find_class_thresholds_path), every class taking its threshold or the configured one, the file of another run refused,
the thresholds built into the inference detector and recorded next to the exported model.
"""

import json
import pytest
import torch

from pathlib import Path

from vision_training.deployment.detection_deployment import (CONFIGURED_THRESHOLD_SOURCE,
                                                             DEPLOYED_THRESHOLDS_FILE_NAME,
                                                             apply_class_score_thresholds,
                                                             build_class_score_thresholds,
                                                             write_deployed_thresholds_file)
from vision_training.deployment.frame_selection_deployment import THRESHOLDS_FILE_NAME, find_class_thresholds_path
from vision_training.models.detection.inference_detector import InferenceDetector
from tests.testing_utilities import build_test_model


def write_class_thresholds(evaluation_folder: Path, classes: dict) -> Path:
    """
    Writes a 'class_thresholds.json' as the evaluation of a detector writes it, with a target precision of 0.99.

    Args:
        evaluation_folder (Path): The iteration folder of the evaluation (or its 'EMA' subfolder).
        classes (dict): The threshold and recall of every class, or None, in class index order.

    Returns:
        Path: The written file.
    """
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    thresholds_path = evaluation_folder / THRESHOLDS_FILE_NAME
    thresholds_path.write_text(data=json.dumps(obj={"target_precision": 0.99, "classes": classes}))
    return thresholds_path


@pytest.mark.parametrize(argnames="evaluation_folder_name", argvalues=["Evaluation", "Checkpoint_Evaluation"])
def test_thresholds_of_the_exported_weights_are_found(tmp_path: Path, evaluation_folder_name: str) -> None:
    """
    Verifies that the thresholds of the exported checkpoint are found in the evaluation of the training or of the
    evaluation script, in the 'EMA' subfolder for the averaged weights only, none being found for the weights never
    evaluated, and that thresholds of the same weights in both folders are refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
        evaluation_folder_name (str): The evaluation folder holding the thresholds.
    """
    checkpoint_path = tmp_path / "Weights" / "Iteration_5" / "model_5.pt"
    averaged_path = write_class_thresholds(evaluation_folder=tmp_path / evaluation_folder_name / "Iteration_5" / "EMA",
                                           classes={"square": None})
    assert find_class_thresholds_path(experiment_folder=tmp_path, checkpoint_path=checkpoint_path,
                                      use_averaged_weights=True) == averaged_path
    assert find_class_thresholds_path(experiment_folder=tmp_path, checkpoint_path=checkpoint_path,
                                      use_averaged_weights=False) is None

    # The trained weights evaluated in both folders are refused
    for folder_name in ["Evaluation", "Checkpoint_Evaluation"]:
        write_class_thresholds(evaluation_folder=tmp_path / folder_name / "Iteration_5", classes={"square": None})
    with pytest.raises(expected_exception=ValueError, match="Two class_thresholds.json"):
        find_class_thresholds_path(experiment_folder=tmp_path, checkpoint_path=checkpoint_path,
                                   use_averaged_weights=False)


def test_every_class_takes_its_threshold_or_the_configured_one(tmp_path: Path) -> None:
    """
    Verifies the thresholds of the classes of an exported detector: a class reaching the target precision keeps its
    threshold, below or above the configured one, a class that never reaches it keeps the configured one, every class
    keeps the configured one without evaluation, and a file of another number of classes is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    thresholds_path = write_class_thresholds(evaluation_folder=tmp_path, classes={
        "square": {"threshold": 0.8, "recall": 0.7}, "circle": None, "triangle": {"threshold": 0.25, "recall": 0.9}})
    record = build_class_score_thresholds(thresholds_path=thresholds_path, configured_score_threshold=0.5,
                                          number_classes=4)
    assert [(class_record["class_index"], class_record["class_name"], class_record["score_threshold"],
             class_record["source"]) for class_record in record["classes"]] == [
        (1, "square", 0.8, THRESHOLDS_FILE_NAME), (2, "circle", 0.5, CONFIGURED_THRESHOLD_SOURCE),
        (3, "triangle", 0.25, THRESHOLDS_FILE_NAME)]
    assert (record["target_precision"], record["class_thresholds_file"]) == (0.99, str(thresholds_path))

    # Without evaluation, the configured threshold for every class
    record = build_class_score_thresholds(thresholds_path=None, configured_score_threshold=0.5, number_classes=3)
    assert [class_record["score_threshold"] for class_record in record["classes"]] == [0.5, 0.5]
    assert record["target_precision"] is None and record["class_thresholds_file"] is None

    # A file of another run
    with pytest.raises(expected_exception=ValueError, match="holds 3 classes, the detector has 4 foreground classes"):
        build_class_score_thresholds(thresholds_path=thresholds_path, configured_score_threshold=0.5, number_classes=5)


def test_thresholds_are_built_into_the_detector_and_recorded(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that apply_class_score_thresholds sets the thresholds of the evaluation of the exported checkpoint into
    the buffer of the inference detector, the classes without threshold at the configured one, prints how many
    classes take the threshold of the evaluation, and that the record written next to the exported model holds them.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    model = build_test_model(dtype=torch.float32)
    model.detection_score_threshold = 0.5
    detector = InferenceDetector(model=model).eval()
    number_foreground_classes = model.number_classes - 1
    classes = {f"class_{class_index}": None for class_index in range(1, model.number_classes)}
    classes["class_2"] = {"threshold": 0.75, "recall": 0.5}
    write_class_thresholds(evaluation_folder=tmp_path / "Evaluation" / "Iteration_5", classes=classes)

    record = apply_class_score_thresholds(inference_model=detector, experiment_folder=tmp_path,
                                          checkpoint_path=tmp_path / "Weights" / "Iteration_5" / "model_5.pt",
                                          use_averaged_weights=False)
    expected_thresholds = [0.5] * number_foreground_classes
    expected_thresholds[1] = 0.75
    assert detector.class_score_thresholds.tolist() == expected_thresholds
    assert f"1 of {number_foreground_classes} classes at their threshold of 99 % precision" in capsys.readouterr().out

    # The record next to the exported model
    record_path = write_deployed_thresholds_file(model_folder=tmp_path, record=record)
    assert record_path.name == DEPLOYED_THRESHOLDS_FILE_NAME
    assert [class_record["score_threshold"] for class_record in json.loads(s=record_path.read_text())[
        "classes"]] == expected_thresholds
