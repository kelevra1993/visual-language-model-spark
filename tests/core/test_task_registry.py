import pytest

from vision_training.configuration.configuration_checker import TASK_CONFIGURATION_CHECKS
from vision_training.configuration.run_name import TASK_CODES, TASK_RUN_NAME_SECTIONS
from vision_training.core.task_registry import TASK_TRAINER_CLASSES, get_trainer_class
from vision_training.models.model_registry import TASK_MODEL_CLASSES


def test_every_task_is_registered_everywhere() -> None:
    """
    Verifies that every trainable task has its trainer, its model, its configuration checks, its run name sections
    and its run name code, and that every trainer trains the task it is registered for: a task registered in only
    some places would fail at training, loading or export time instead of at startup.
    """
    trainable_tasks = set(TASK_TRAINER_CLASSES)
    assert set(TASK_CONFIGURATION_CHECKS) == trainable_tasks
    assert set(TASK_MODEL_CLASSES) == trainable_tasks
    assert set(TASK_RUN_NAME_SECTIONS) == trainable_tasks
    assert trainable_tasks <= set(TASK_CODES)
    for task, trainer_class in TASK_TRAINER_CLASSES.items():
        assert trainer_class.TASK_NAME == task


def test_unknown_task_has_no_trainer() -> None:
    """
    Verifies that asking for the trainer of a task that is not implemented raises an error listing the valid tasks.
    """
    with pytest.raises(expected_exception=KeyError, match="No trainer is registered"):
        get_trainer_class(task="segmentation")
