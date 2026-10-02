from typing import Dict, Type

from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.trainers.classification_trainer import ClassificationTrainer
from vision_training.trainers.detection_trainer import DetectionTrainer

# Trainer class of every task, keyed by the value of 'ExperimentConfiguration.task'
TASK_TRAINER_CLASSES: Dict[str, Type[BaseTrainer]] = {"object_detection": DetectionTrainer,
                                                      "classification": ClassificationTrainer}


def get_trainer_class(task: str) -> Type[BaseTrainer]:
    """
    Gives the trainer of a task, so that the entry point and the scripts train or evaluate any configuration through
    the trainer matching its 'ExperimentConfiguration.task'.

    Args:
        task (str): The task of the configuration.

    Returns:
        Type[BaseTrainer]: The trainer class of the task.

    Raises:
        KeyError: If no trainer is registered for the task.
    """
    if task not in TASK_TRAINER_CLASSES:
        raise KeyError(f"No trainer is registered for the task {task!r}, expected one of {list(TASK_TRAINER_CLASSES)}.")
    return TASK_TRAINER_CLASSES[task]
