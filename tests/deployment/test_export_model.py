"""
Tests of read_logit_temperature of vision_training/deployment/export_model.py: the logit temperature of an exported
classifier is read from the class_thresholds.json of the evaluation of the exported checkpoint, in the run folder or
in the evaluations folder of scripts/evaluation/evaluate_checkpoint.py (its 'EMA' subfolder for the averaged weights),
a file written before the temperature was recorded giving 1, and the thresholds of another run, iteration, kind of
weights or number of classes being refused with every mismatch named.
"""

import json
import pytest

from pathlib import Path
from typing import Optional

from vision_training.deployment.export_model import read_logit_temperature

# Exported run and checkpoint of the tests
RUN_NAME = "CLS_TEST_MC3"
EXPORTED_ITERATION = 6


def write_class_thresholds(evaluation_folder: Path, logit_temperature: Optional[float], number_classes: int) -> Path:
    """
    Writes a class_thresholds.json as the classification evaluation writes it, with or without its logit temperature.

    Args:
        evaluation_folder (Path): The evaluation folder of the checkpoint, created.
        logit_temperature (Optional[float]): The recorded temperature, None for a file written before it existed.
        number_classes (int): The number of classes of the thresholds.

    Returns:
        Path: The written file.
    """
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    thresholds = {"target_precision": 0.99,
                  "classes": {f"class_{class_index}": {"threshold": 0.8, "recall": 0.5}
                              for class_index in range(number_classes)}}
    if logit_temperature is not None:
        thresholds = {"logit_temperature": logit_temperature, **thresholds}
    thresholds_path = evaluation_folder / "class_thresholds.json"
    thresholds_path.write_text(data=json.dumps(obj=thresholds))
    return thresholds_path


@pytest.mark.parametrize(argnames=("evaluation_subfolder", "use_averaged_weights", "logit_temperature", "expected"),
                         argvalues=[(f"{RUN_NAME}/Evaluation/Iteration_6", False, 2.5, 2.5),
                                    (f"evaluations/{RUN_NAME}/Evaluation/Iteration_6/EMA", True, 3.0, 3.0),
                                    (f"{RUN_NAME}/Evaluation/Iteration_6", False, None, 1.0)])
def test_logit_temperature_is_read_from_the_evaluation_of_the_checkpoint(
        tmp_path: Path, evaluation_subfolder: str, use_averaged_weights: bool, logit_temperature: Optional[float],
        expected: float) -> None:
    """
    Verifies that the temperature is read from the evaluation of the exported checkpoint in the run folder or in the
    evaluations folder (averaged weights), and that a file written before the temperature was recorded gives 1.
    """
    thresholds_path = write_class_thresholds(evaluation_folder=tmp_path / evaluation_subfolder,
                                             logit_temperature=logit_temperature, number_classes=3)
    experiment_folder = tmp_path / RUN_NAME
    checkpoint_path = experiment_folder / "Weights" / f"Iteration_{EXPORTED_ITERATION}" / "model_000006.pt"
    assert read_logit_temperature(class_thresholds_path=thresholds_path, experiment_folder=experiment_folder,
                                  checkpoint_path=checkpoint_path, use_averaged_weights=use_averaged_weights,
                                  number_classes=3) == expected


@pytest.mark.parametrize(argnames=("evaluation_subfolder", "use_averaged_weights", "number_classes",
                                   "expected_message"), argvalues=[
    (f"{RUN_NAME}/Evaluation/Iteration_4", False, 3,
     "it evaluates Iteration_4, the exported checkpoint is Iteration_6"),
    (f"{RUN_NAME}/Evaluation/Iteration_6/EMA", False, 3, "it evaluates the averaged weights, the trained weights"),
    (f"{RUN_NAME}/Evaluation/Iteration_6", True, 3, "it evaluates the trained weights, the averaged weights"),
    ("CLS_OTHER_MC3/Evaluation/Iteration_6", False, 3, "it evaluates the run 'CLS_OTHER_MC3'"),
    (f"{RUN_NAME}/Evaluation/Iteration_6", False, 2, "holds the thresholds of 2 classes, the exported classifier "
                                                     "predicts 3"),
    (f"{RUN_NAME}/Iteration_6", False, 3, "it is not '<run>/Evaluation/Iteration_<n>/")])
def test_thresholds_of_another_checkpoint_are_refused(tmp_path: Path, evaluation_subfolder: str,
                                                      use_averaged_weights: bool, number_classes: int,
                                                      expected_message: str) -> None:
    """
    Verifies that the thresholds of another iteration, of the other kind of weights, of another run, of another number
    of classes or outside an evaluation folder stop the export, the error naming the mismatch.
    """
    thresholds_path = write_class_thresholds(evaluation_folder=tmp_path / evaluation_subfolder,
                                             logit_temperature=2.0, number_classes=number_classes)
    experiment_folder = tmp_path / RUN_NAME
    checkpoint_path = experiment_folder / "Weights" / f"Iteration_{EXPORTED_ITERATION}" / "model_000006.pt"
    with pytest.raises(expected_exception=ValueError, match="do not belong to the exported checkpoint") as error:
        read_logit_temperature(class_thresholds_path=thresholds_path, experiment_folder=experiment_folder,
                               checkpoint_path=checkpoint_path, use_averaged_weights=use_averaged_weights,
                               number_classes=3)
    assert expected_message in str(error.value)
