"""
Tests of vision_training/core/task_registry.py: every trainable task is registered in every table of the framework
(trainer, model, configuration checks, run name, inference model and export inputs), and an unknown task has no trainer.
"""

import pytest

from vision_training.configuration_processing.configuration_checker import TASK_CONFIGURATION_CHECKS
from vision_training.configuration_processing.run_name import TASK_CODES, TASK_RUN_NAME_SECTIONS
from vision_training.core.task_registry import TASK_TRAINER_CLASSES, get_trainer_class
from vision_training.deployment.export_specifications import (INFERENCE_MODEL_CLASSES, SEQUENCE_TASKS,
                                                              TASK_INPUT_NAMES, TASK_MODEL_NAMES)
from vision_training.models.model_registry import TASK_MODEL_CLASSES


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


def test_unknown_task_has_no_trainer() -> None:
    """
    Verifies that asking for the trainer of a task that is not implemented raises an error listing the valid tasks.
    """
    with pytest.raises(expected_exception=KeyError, match="No trainer is registered"):
        get_trainer_class(task="segmentation")
