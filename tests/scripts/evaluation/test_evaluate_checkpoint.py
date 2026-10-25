"""
Tests of scripts/evaluation/evaluate_checkpoint.py, the standalone evaluation of the checkpoints of a trained run: the
limit of error images per pair of classes turns the error images on for the tasks saving them (an
'evaluation.error_dumps' section, classification and detection), and is skipped with a warning for the tasks without
error images (semantic segmentation, keypoint detection, self-supervised learning) instead of stopping the script with
a KeyError (B56).
"""

import pytest

from scripts.evaluation.evaluate_checkpoint import apply_error_image_limit
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
