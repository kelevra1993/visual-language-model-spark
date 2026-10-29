"""
Tests of scripts/evaluation/evaluate_checkpoint.py, the standalone evaluation of the checkpoints of a trained run: the
limit of error images per pair of classes turns the error images on for the tasks saving them (an
'evaluation.error_dumps' section, classification and detection), and is skipped with a warning for the tasks without
error images (semantic segmentation, keypoint detection, self-supervised learning) instead of stopping the script with
a KeyError (B56); the target precisions of the class thresholds replace the experiment's one (a list for the tasks
sharing the evaluation of the classification, a single precision for the frame selection and the detection), are
refused when invalid, and are skipped with a warning for the tasks without class thresholds; the logit temperature
must be a number above 0, and 1 for the tasks other than the classification.
"""

import pytest

from scripts.evaluation.evaluate_checkpoint import (apply_error_image_limit, apply_target_precisions,
                                                    check_logit_temperature)
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration


@pytest.mark.parametrize("task", ["classification", "object_detection"])
def test_error_image_limit_turns_on_the_error_images(task: str) -> None:
    """
    Verifies that the limit turns the error images of a task saving them on, with its own number per pair of classes,
    and that None leaves the settings of the experiment untouched.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{task}/configuration_synthetic_shapes.yaml")
    error_dumps_configuration = experiment_configuration["evaluation"]["error_dumps"]
    settings_before = dict(error_dumps_configuration)
    apply_error_image_limit(experiment_configuration=experiment_configuration, maximum_error_images_per_class_pair=None)
    assert error_dumps_configuration == settings_before

    apply_error_image_limit(experiment_configuration=experiment_configuration, maximum_error_images_per_class_pair=7)
    assert error_dumps_configuration["activated"] is True and error_dumps_configuration["maximum_per_class_pair"] == 7


@pytest.mark.parametrize("task", ["semantic_segmentation", "keypoint_detection", "self_supervised"])
def test_error_image_limit_is_skipped_without_error_images(task: str, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the limit is skipped with a warning naming the setting for the tasks whose configuration has no
    'evaluation.error_dumps' section, which they do not save, the configuration being left as it was.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{task}/configuration_synthetic_shapes.yaml")
    evaluation_before = dict(experiment_configuration["evaluation"])
    apply_error_image_limit(experiment_configuration=experiment_configuration, maximum_error_images_per_class_pair=7)
    assert experiment_configuration["evaluation"] == evaluation_before
    assert "maximum_error_images_per_class_pair" in capsys.readouterr().out


@pytest.mark.parametrize(argnames=("configuration_name", "target_precisions"),
                         argvalues=[("classification/configuration_synthetic_shapes.yaml", [0.99, 0.95, 0.90]),
                                    ("sequence_labelling/configuration_synthetic_sequences.yaml", [0.9, 0.8]),
                                    ("frame_selection/configuration_synthetic_sequences.yaml", 0.95),
                                    ("object_detection/configuration_synthetic_shapes.yaml", 0.9)])
def test_target_precisions_replace_the_experiment_one(configuration_name: str,
                                                      target_precisions: float | list[float]) -> None:
    """
    Verifies that the target precisions replace the 'evaluation.target_precision' of the experiment, a list for the
    tasks sharing the evaluation of the classification and a single precision for the others, and that None leaves
    the experiment's one untouched.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{configuration_name}")
    apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=None)
    assert experiment_configuration["evaluation"]["target_precision"] == 0.99

    apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=target_precisions)
    assert experiment_configuration["evaluation"]["target_precision"] == target_precisions


@pytest.mark.parametrize(argnames=("configuration_name", "target_precisions", "expected_message"), argvalues=[
    ("frame_selection/configuration_synthetic_sequences.yaml", [0.99, 0.95], "must be a single probability"),
    ("object_detection/configuration_synthetic_shapes.yaml", [0.99, 0.95], "must be a single probability"),
    ("classification/configuration_synthetic_shapes.yaml", [0.99, 1.5],
     r"'target_precisions\[1\]' must be a probability"),
    ("classification/configuration_synthetic_shapes.yaml", [0.99, 0.99], "must not repeat a probability")])
def test_invalid_target_precisions_are_refused(configuration_name: str, target_precisions: list[float],
                                               expected_message: str) -> None:
    """
    Verifies that a list given to a task keeping one target precision, a precision outside [0, 1] and a repeated
    precision stop the script before any evaluation, the configuration of the experiment being left as it was.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{configuration_name}")
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=target_precisions)
    assert experiment_configuration["evaluation"]["target_precision"] == 0.99


@pytest.mark.parametrize("task", ["semantic_segmentation", "keypoint_detection", "self_supervised"])
def test_target_precisions_are_skipped_without_class_thresholds(task: str, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the target precisions are skipped with a warning naming the setting for the tasks giving no class
    threshold (no 'evaluation.target_precision' setting), the configuration being left as it was.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{task}/configuration_synthetic_shapes.yaml")
    evaluation_before = dict(experiment_configuration["evaluation"])
    apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=[0.99, 0.95])
    assert experiment_configuration["evaluation"] == evaluation_before
    assert "target_precisions is skipped" in capsys.readouterr().out


@pytest.mark.parametrize(argnames=("task", "logit_temperature"),
                         argvalues=[("classification", 1.0), ("classification", 3), ("classification", 0.5),
                                    ("sequence_classification", 1.0), ("object_detection", 1)])
def test_logit_temperature_is_accepted(task: str, logit_temperature: float) -> None:
    """
    Verifies that a temperature above 0 is accepted for the classification, and the temperature 1 (the probabilities of
    the model) for every task.
    """
    check_logit_temperature(task=task, logit_temperature=logit_temperature)


@pytest.mark.parametrize(argnames=("task", "logit_temperature", "expected_message"), argvalues=[
    ("classification", 0.0, "must be a number above 0"), ("classification", -2.0, "must be a number above 0"),
    ("classification", "3", "must be a number above 0"), ("classification", True, "must be a number above 0"),
    ("classification", None, "must be a number above 0"),
    ("sequence_classification", 2.0, "must be 1 for the sequence_classification task"),
    ("frame_selection", 0.5, "must be 1 for the frame_selection task")])
def test_invalid_logit_temperature_is_refused(task: str, logit_temperature: float, expected_message: str) -> None:
    """
    Verifies that a temperature that is not a number above 0, or a temperature other than 1 for a task whose export
    has no logit temperature, stops the script before the trainer is built.
    """
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        check_logit_temperature(task=task, logit_temperature=logit_temperature)
