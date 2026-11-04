"""
Tests of scripts/evaluation/evaluate_checkpoint.py, the standalone evaluation of the checkpoints of a trained run: the
target precisions of the class thresholds replace the experiment's one (a list for the classification), and a list
given to a task keeping a single precision (the frame selection) is refused.
"""

import pytest

from scripts.evaluation.evaluate_checkpoint import apply_target_precisions
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration


@pytest.mark.parametrize(argnames=("configuration_name", "target_precisions"),
                         argvalues=[("classification/configuration_synthetic_shapes.yaml", [0.99, 0.95, 0.90])])
def test_target_precisions_replace_the_experiment_one(configuration_name: str,
                                                      target_precisions: float | list[float]) -> None:
    """
    Verifies that the target precisions replace the 'evaluation.target_precision' of the experiment, a list for the
    classification, and that None leaves the experiment's one untouched.
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
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=f"vision_training/configurations/{configuration_name}")
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        apply_target_precisions(experiment_configuration=experiment_configuration, target_precisions=target_precisions)
    assert experiment_configuration["evaluation"]["target_precision"] == 0.99
