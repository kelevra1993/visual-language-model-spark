"""
Tests of scripts/evaluation/evaluate_checkpoint.py, the standalone evaluation of the checkpoints of a trained run: the
target precisions of the class thresholds replace the experiment's one (a list for the classification), and a list
given to a task keeping a single precision (the frame selection) is refused. The final non-maximum suppression of a
detection run is replaced by the evaluated one, whose evaluations get an output subfolder of their own when it differs
from the one of the run, invalid settings being refused and the tasks without detector skipped. The predictions file
of every evaluated sample is turned on or off for the tasks writing one, the others being skipped, and the error image
limit turns the error images on for the tasks saving them, a configuration without 'evaluation' section being refused.
"""

import pytest

from typing import Any, Dict, Optional, Tuple

from scripts.evaluation.evaluate_checkpoint import (apply_error_image_limit, apply_nms_criterion,
                                                    apply_save_predictions, apply_target_precisions)
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration

# Detection configuration of the run whose final NMS is replaced, set to the greedy NMS ("iou", exponent 1.0) of the
# runs trained before DIoU-NMS was kept
DETECTION_CONFIGURATION_PATH = "vision_training/configurations/object_detection/configuration_example.yaml"


def load_greedy_detection_configuration() -> Dict[str, Any]:
    """
    Loads the model configuration of the example detection run with the greedy final NMS ("iou", exponent 1.0), the
    one of the runs trained before stage 5b of the detection benchmark kept DIoU-NMS, so that the tests replace the
    NMS of such a run whatever the example configuration holds.

    Returns:
        Dict[str, Any]: The model configuration, its 'Detector' section with the greedy NMS.
    """
    _, model_configuration = load_experiment_configuration(configuration_path=DETECTION_CONFIGURATION_PATH)
    model_configuration["Detector"].update({"nms_criterion": "iou", "nms_distance_exponent": 1.0})
    return model_configuration


@pytest.mark.parametrize(argnames=("configuration_name", "target_precisions"),
                         argvalues=[("classification/configuration_synthetic_shapes.yaml", [0.99, 0.95, 0.90])])
def test_target_precisions_replace_the_experiment_one(configuration_name: str,
                                                      target_precisions: float | list[float]) -> None:
    """
    Verifies that the target precisions replace the 'evaluation.target_precision' of the experiment, a list for the
    classification, and that None leaves the experiment's one untouched.

    Args:
        configuration_name (str): The configuration of the evaluated run, below vision_training/configurations.
        target_precisions (float | list[float]): The target precisions replacing the experiment's one.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{configuration_name}")
    apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=None)
    assert experiment_configuration["evaluation"]["target_precision"] == 0.99

    apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=target_precisions)
    assert experiment_configuration["evaluation"]["target_precision"] == target_precisions


@pytest.mark.parametrize(argnames=("configuration_name", "target_precisions", "expected_message"), argvalues=[
    ("frame_selection/configuration_synthetic_sequences.yaml", [0.99, 0.95], "must be a single probability")])
def test_invalid_target_precisions_are_refused(configuration_name: str, target_precisions: list[float],
                                               expected_message: str) -> None:
    """
    Verifies that a list given to a task keeping one target precision stops the script before any evaluation, the
    configuration of the experiment being left as it was.

    Args:
        configuration_name (str): The configuration of the evaluated run, below vision_training/configurations.
        target_precisions (list[float]): The refused target precisions.
        expected_message (str): A part of the message of the refusal.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{configuration_name}")
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=target_precisions)
    assert experiment_configuration["evaluation"]["target_precision"] == 0.99


@pytest.mark.parametrize(argnames=("nms_criterion", "nms_distance_exponent", "expected_suffix", "expected_settings"),
                         argvalues=[(None, None, "", ("iou", 1.0)),
                                    ("distance_iou", None, "_nms_distance_iou_1.0", ("distance_iou", 1.0)),
                                    ("distance_iou", 0.6, "_nms_distance_iou_0.6", ("distance_iou", 0.6)),
                                    ("iou", None, "", ("iou", 1.0)),
                                    (None, 0.6, "", ("iou", 0.6))])
def test_nms_criterion_replaces_the_run_one(nms_criterion: Optional[str], nms_distance_exponent: Optional[float],
                                            expected_suffix: str, expected_settings: Tuple[str, float]) -> None:
    """
    Verifies that the criterion and the distance exponent of the evaluation replace those of the 'Detector' section of
    a run evaluated with the greedy NMS, None keeping the run's, and that the output subfolder gets a suffix naming the
    evaluated NMS only when it differs from the one of the run, an exponent changing nothing with the greedy NMS.

    Args:
        nms_criterion (Optional[str]): The evaluated criterion, None to keep the run's.
        nms_distance_exponent (Optional[float]): The evaluated exponent, None to keep the run's.
        expected_suffix (str): The expected suffix of the output subfolder.
        expected_settings (Tuple[str, float]): The criterion and exponent of the evaluation.
    """
    model_configuration = load_greedy_detection_configuration()
    assert apply_nms_criterion(model_configuration=model_configuration, task="object_detection",
                               nms_criterion=nms_criterion, nms_distance_exponent=nms_distance_exponent) == \
        expected_suffix
    assert (model_configuration["Detector"]["nms_criterion"],
            model_configuration["Detector"]["nms_distance_exponent"]) == expected_settings


def test_invalid_or_skipped_nms_criterion(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that an unknown criterion or a negative exponent stops the script before any evaluation, the 'Detector'
    section of the run being left as it was, that a run evaluated with DIoU-NMS gets a subfolder of its own for the
    greedy NMS, and that a task without detector is evaluated without the replacement, a warning telling it.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    model_configuration = load_greedy_detection_configuration()
    for nms_criterion, nms_distance_exponent, expected_message in [
            ("soft_nms", None, "'Detector.nms_criterion' must be one of"),
            (None, -1.0, "'Detector.nms_distance_exponent' must be a finite positive number")]:
        with pytest.raises(expected_exception=ValueError, match=expected_message):
            apply_nms_criterion(model_configuration=model_configuration, task="object_detection",
                                nms_criterion=nms_criterion, nms_distance_exponent=nms_distance_exponent)
        assert (model_configuration["Detector"]["nms_criterion"],
                model_configuration["Detector"]["nms_distance_exponent"]) == ("iou", 1.0)

    # A run evaluated with DIoU-NMS, then with the greedy NMS
    model_configuration["Detector"].update({"nms_criterion": "distance_iou", "nms_distance_exponent": 0.6})
    assert apply_nms_criterion(model_configuration=model_configuration, task="object_detection", nms_criterion="iou",
                               nms_distance_exponent=None) == "_nms_iou"

    # The classification has no final non-maximum suppression
    _, classification_configuration = load_experiment_configuration(
        configuration_path="vision_training/configurations/classification/configuration_synthetic_shapes.yaml")
    assert apply_nms_criterion(model_configuration=classification_configuration, task="classification",
                               nms_criterion="distance_iou", nms_distance_exponent=1.0) == ""
    assert "Detector" not in classification_configuration
    assert "nms_criterion and nms_distance_exponent are skipped" in capsys.readouterr().out


def test_error_image_limit_turns_the_error_images_on_or_is_skipped(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the error image limit turns on the 'evaluation.error_dumps' of a classification run with its limit,
    None leaving them untouched, that a task without error images (the self-supervised task) is evaluated without
    them, a warning telling it, and that a configuration without 'evaluation' section stops the script instead of
    being taken for a task without error images.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path="vision_training/configurations/classification/configuration_synthetic_shapes.yaml")
    error_dumps_before = dict(experiment_configuration["evaluation"]["error_dumps"])
    apply_error_image_limit(experiment_configuration=experiment_configuration,
                            maximum_error_images_per_class_pair=None)
    assert experiment_configuration["evaluation"]["error_dumps"] == error_dumps_before
    apply_error_image_limit(experiment_configuration=experiment_configuration, maximum_error_images_per_class_pair=7)
    assert experiment_configuration["evaluation"]["error_dumps"] == {**error_dumps_before, "activated": True,
                                                                     "maximum_per_class_pair": 7}

    # The self-supervised task saves no error image
    self_supervised_configuration, _ = load_experiment_configuration(
        configuration_path="vision_training/configurations/self_supervised/configuration_synthetic_shapes.yaml")
    apply_error_image_limit(experiment_configuration=self_supervised_configuration,
                            maximum_error_images_per_class_pair=7)
    assert "error_dumps" not in self_supervised_configuration["evaluation"]
    assert "maximum_error_images_per_class_pair is skipped" in capsys.readouterr().out

    # A configuration without its 'evaluation' section is refused, for every replaced evaluation setting
    del experiment_configuration["evaluation"]
    with pytest.raises(expected_exception=KeyError, match="evaluation"):
        apply_error_image_limit(experiment_configuration=experiment_configuration,
                                maximum_error_images_per_class_pair=7)
    with pytest.raises(expected_exception=KeyError, match="evaluation"):
        apply_save_predictions(experiment_configuration=experiment_configuration, save_predictions=True)
    with pytest.raises(expected_exception=KeyError, match="evaluation"):
        apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=0.9)
    assert capsys.readouterr().out == ""


def test_save_predictions_replaces_the_experiment_one_or_is_skipped(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that save_predictions replaces the 'evaluation.save_predictions' of a classification run, None leaving it
    untouched, and that a task without the setting (the detection) is evaluated without it, a warning telling it.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path="vision_training/configurations/classification/configuration_synthetic_shapes.yaml")
    apply_save_predictions(experiment_configuration=experiment_configuration, save_predictions=None)
    assert experiment_configuration["evaluation"]["save_predictions"] is True
    apply_save_predictions(experiment_configuration=experiment_configuration, save_predictions=False)
    assert experiment_configuration["evaluation"]["save_predictions"] is False

    # The detection writes no predictions file
    detection_configuration, _ = load_experiment_configuration(configuration_path=DETECTION_CONFIGURATION_PATH)
    apply_save_predictions(experiment_configuration=detection_configuration, save_predictions=True)
    assert "save_predictions" not in detection_configuration["evaluation"]
    assert "save_predictions is skipped" in capsys.readouterr().out
