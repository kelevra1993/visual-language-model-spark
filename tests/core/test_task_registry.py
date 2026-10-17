"""
Tests of vision_training/core/task_registry.py: every trainable task is registered in every table of the framework
(trainer, model, configuration checks, run name, inference model and export inputs), the keypoint detection with its own
classes, whole-image input and run name code, and an unknown task has no trainer.
"""

import pytest

from vision_training.configuration_processing.configuration_checker import TASK_CONFIGURATION_CHECKS
from vision_training.configuration_processing.keypoint_checks import check_keypoint_configuration
from vision_training.configuration_processing.run_name import (TASK_CODES, TASK_RUN_NAME_SECTIONS,
                                                                build_keypoint_sections)
from vision_training.core.task_registry import TASK_TRAINER_CLASSES, get_trainer_class
from vision_training.deployment.export_specifications import (INFERENCE_MODEL_CLASSES, SEQUENCE_TASKS,
                                                              TASK_INPUT_NAMES, TASK_MODEL_NAMES)
from vision_training.models.keypoints.inference_keypoint_detector import InferenceKeypointDetector
from vision_training.models.keypoints.keypoint_detector import KeypointDetector
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.trainers.keypoint_trainer import KeypointTrainer


def test_every_task_is_registered_everywhere() -> None:
    """
    Verifies that every trainable task has its trainer, its model, its configuration checks, its run name sections
    and its run name code, its inference model, Triton model name and input names, and that every trainer trains the
    task it is registered for: a task registered in only some places would fail at training, loading or export time
    instead of at startup.
    """
    trainable_tasks = set(TASK_TRAINER_CLASSES)
    assert set(TASK_CONFIGURATION_CHECKS) == trainable_tasks
    assert set(TASK_MODEL_CLASSES) == trainable_tasks
    assert set(TASK_RUN_NAME_SECTIONS) == trainable_tasks
    assert trainable_tasks <= set(TASK_CODES)
    assert set(INFERENCE_MODEL_CLASSES) == set(TASK_MODEL_NAMES) == set(TASK_INPUT_NAMES) == trainable_tasks
    assert set(SEQUENCE_TASKS) <= trainable_tasks
    for task, trainer_class in TASK_TRAINER_CLASSES.items():
        assert trainer_class.TASK_NAME == task


def test_keypoint_detection_is_registered() -> None:
    """
    Verifies that the keypoint detection is trained by the keypoint trainer on the keypoint detector, checked by the
    keypoint checks, named with the KPT code and its high resolution network and head sections, and exported as the
    keypoint_detector Triton model of one 'images' input, the whole images, not as a sequence.
    """
    assert get_trainer_class(task="keypoint_detection") is KeypointTrainer
    assert TASK_MODEL_CLASSES["keypoint_detection"] is KeypointDetector
    assert TASK_CONFIGURATION_CHECKS["keypoint_detection"] is check_keypoint_configuration
    assert TASK_CODES["keypoint_detection"] == "KPT"
    assert TASK_RUN_NAME_SECTIONS["keypoint_detection"] is build_keypoint_sections
    assert INFERENCE_MODEL_CLASSES["keypoint_detection"] is InferenceKeypointDetector
    assert TASK_MODEL_NAMES["keypoint_detection"] == "keypoint_detector"
    assert TASK_INPUT_NAMES["keypoint_detection"] == ["images"]
    assert "keypoint_detection" not in SEQUENCE_TASKS


def test_unknown_task_has_no_trainer() -> None:
    """
    Verifies that asking for the trainer of a task that is not implemented raises an error listing the valid tasks.
    """
    with pytest.raises(expected_exception=KeyError, match="No trainer is registered"):
        get_trainer_class(task="segmentation")
