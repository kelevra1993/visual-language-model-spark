"""
Tests of find_saved_thresholds_path, check_saved_thresholds_location and read_saved_temperature of
vision_training/deployment/export_model.py, used by the export with use_saved_temperature: the thresholds saved from
the temperature explorer of the exported checkpoint are found in its evaluation folder in the evaluations folder of
scripts/evaluation/evaluate_checkpoint.py (its 'EMA' subfolder for the averaged weights), the export stopping when the
run folder holds them too, thresholds given by the export button of the page are accepted only from the evaluation of
the exported checkpoint and weights, and their temperature is read after checking them against the classes of the
exported model.
"""

import json
import pytest

from pathlib import Path
from typing import Any, Dict

from vision_training.deployment.export_model import (check_saved_thresholds_location, find_saved_thresholds_path,
                                                     read_saved_temperature)
from vision_training.evaluation.classification.temperature_explorer import SAVED_THRESHOLDS_FILE_NAME

# Exported run and checkpoint of the tests
RUN_NAME = "CLS_TEST_MC3"
EXPORTED_ITERATION = 6


def build_thresholds(logit_temperature: float, number_classes: int) -> Dict[str, Any]:
    """
    Builds thresholds in the format of class_thresholds.json, the file of the evaluation and the saved file sharing it.

    Args:
        logit_temperature (float): The temperature of the thresholds.
        number_classes (int): The number of classes of the thresholds.

    Returns:
        Dict[str, Any]: The thresholds.
    """
    return {"logit_temperature": logit_temperature, "target_precision": 0.99,
            "classes": {f"class_{class_index}": {"threshold": 0.8, "recall": 0.5}
                        for class_index in range(number_classes)}}


def write_evaluation(evaluation_folder: Path, saved_thresholds: Dict[str, Any], number_classes: int) -> Path:
    """
    Writes the class_thresholds.json of an evaluation, as the classification evaluation writes it, and the thresholds
    saved from its temperature explorer next to it.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint, created.
        saved_thresholds (Dict[str, Any]): The saved thresholds.
        number_classes (int): The number of classes of class_thresholds.json.

    Returns:
        Path: The saved thresholds file.
    """
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    (evaluation_folder / "class_thresholds.json").write_text(
        data=json.dumps(obj=build_thresholds(logit_temperature=1.0, number_classes=number_classes)))
    saved_path = evaluation_folder / SAVED_THRESHOLDS_FILE_NAME
    saved_path.write_text(data=json.dumps(obj=saved_thresholds))
    return saved_path


def get_checkpoint_path(experiment_folder: Path) -> Path:
    """
    Gives the checkpoint of the exported iteration of the test run, the export naming its evaluation folder after the
    folder of the checkpoint.

    Args:
        experiment_folder (Path): The exported run folder.

    Returns:
        Path: The checkpoint of the exported iteration.
    """
    return experiment_folder / "Weights" / f"Iteration_{EXPORTED_ITERATION}" / "model_000006.pt"


@pytest.mark.parametrize(argnames=("evaluation_subfolder", "use_averaged_weights"),
                         argvalues=[(f"evaluations/{RUN_NAME}/Evaluation/Iteration_6/EMA", True)])
def test_saved_temperature_is_read_from_the_evaluation_of_the_checkpoint(
        tmp_path: Path, evaluation_subfolder: str, use_averaged_weights: bool) -> None:
    """
    Verifies that the saved thresholds are found in the evaluation of the exported iteration and averaged weights in
    the evaluations folder, and that their temperature is read.
    """
    saved_path = write_evaluation(evaluation_folder=tmp_path / evaluation_subfolder,
                                  saved_thresholds=build_thresholds(logit_temperature=2.5, number_classes=3),
                                  number_classes=3)
    experiment_folder = tmp_path / RUN_NAME
    found_path = find_saved_thresholds_path(experiment_folder=experiment_folder,
                                            checkpoint_path=get_checkpoint_path(experiment_folder=experiment_folder),
                                            use_averaged_weights=use_averaged_weights)
    assert found_path == saved_path
    assert read_saved_temperature(saved_thresholds_path=found_path, number_classes=3) == 2.5


def test_thresholds_saved_in_both_folders_are_refused(tmp_path: Path) -> None:
    """
    Verifies that thresholds saved for the exported checkpoint in both the run folder and the evaluations folder stop
    the export, so that the deployed temperature is never picked silently between two saves.
    """
    for evaluation_subfolder in [f"{RUN_NAME}/Evaluation/Iteration_6",
                                 f"evaluations/{RUN_NAME}/Evaluation/Iteration_6"]:
        write_evaluation(evaluation_folder=tmp_path / evaluation_subfolder,
                         saved_thresholds=build_thresholds(logit_temperature=2.5, number_classes=3), number_classes=3)
    experiment_folder = tmp_path / RUN_NAME
    with pytest.raises(expected_exception=ValueError, match="Two saved_temperature_thresholds.json"):
        find_saved_thresholds_path(experiment_folder=experiment_folder,
                                   checkpoint_path=get_checkpoint_path(experiment_folder=experiment_folder),
                                   use_averaged_weights=False)


@pytest.mark.parametrize(argnames=("saved_thresholds", "evaluated_classes", "exported_classes", "expected_message"),
                         argvalues=[(build_thresholds(logit_temperature=2.5, number_classes=2), 2, 3,
                                     "the exported classifier predicts 3")])
def test_saved_thresholds_of_another_evaluation_or_model_are_refused(
        tmp_path: Path, saved_thresholds: Dict[str, Any], evaluated_classes: int, exported_classes: int,
        expected_message: str) -> None:
    """
    Verifies that saved thresholds holding another number of classes than the exported classifier stop the export, the
    error naming the mismatch.
    """
    saved_path = write_evaluation(evaluation_folder=tmp_path / RUN_NAME / "Evaluation" / "Iteration_6",
                                  saved_thresholds=saved_thresholds, number_classes=evaluated_classes)
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        read_saved_temperature(saved_thresholds_path=saved_path, number_classes=exported_classes)


@pytest.mark.parametrize(argnames=("saved_subfolder", "use_averaged_weights", "accepted"), argvalues=[
    (f"evaluations/{RUN_NAME}/Evaluation/Iteration_6/EMA", True, True),
    (f"{RUN_NAME}/Evaluation/Iteration_6", False, True),
    (f"{RUN_NAME}/Evaluation/Iteration_7", False, False),
    (f"evaluations/{RUN_NAME}/Evaluation/Iteration_6/EMA", False, False),
    (f"{RUN_NAME}/Evaluation/Iteration_6", True, False),
    ("evaluations/CLS_OTHER_MC3/Evaluation/Iteration_6", False, False)])
def test_given_saved_thresholds_must_be_those_of_the_exported_checkpoint(tmp_path: Path, saved_subfolder: str,
                                                                         use_averaged_weights: bool,
                                                                         accepted: bool) -> None:
    """
    Verifies that the thresholds given by the export button of the page are accepted only from the evaluation of the
    exported run, iteration and weights, in the run folder or the evaluations folder, and refused from the evaluation
    of another iteration, of the other weights or of another run.
    """
    experiment_folder = tmp_path / RUN_NAME
    saved_path = tmp_path / saved_subfolder / SAVED_THRESHOLDS_FILE_NAME
    location_settings = {"saved_thresholds_path": saved_path, "experiment_folder": experiment_folder,
                         "checkpoint_path": get_checkpoint_path(experiment_folder=experiment_folder),
                         "use_averaged_weights": use_averaged_weights}
    if accepted:
        check_saved_thresholds_location(**location_settings)
    else:
        with pytest.raises(expected_exception=ValueError, match="for the exported checkpoint"):
            check_saved_thresholds_location(**location_settings)
