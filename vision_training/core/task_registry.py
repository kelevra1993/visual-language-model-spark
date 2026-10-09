"""
Registry of the trainer class of every task, keyed by 'ExperimentConfiguration.task'. Used by main.py,
scripts/evaluation/evaluate_checkpoint.py and the tests to build the trainer matching a configuration.
"""

from typing import Dict, Type

from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.trainers.classification_trainer import ClassificationTrainer
from vision_training.trainers.detection_trainer import DetectionTrainer
from vision_training.trainers.frame_selection_trainer import FrameSelectionTrainer
from vision_training.trainers.segmentation_trainer import SegmentationTrainer
from vision_training.trainers.sequence_classification_trainer import SequenceClassificationTrainer
from vision_training.trainers.sequence_comparison_trainer import SequenceComparisonTrainer
from vision_training.trainers.sequence_labelling_trainer import SequenceLabellingTrainer

# Trainer class of every task, keyed by the value of 'ExperimentConfiguration.task'
TASK_TRAINER_CLASSES: Dict[str, Type[BaseTrainer]] = {"object_detection": DetectionTrainer,
                                                      "classification": ClassificationTrainer,
                                                      "semantic_segmentation": SegmentationTrainer,
                                                      "sequence_classification": SequenceClassificationTrainer,
                                                      "sequence_labelling": SequenceLabellingTrainer,
                                                      "sequence_comparison": SequenceComparisonTrainer,
                                                      "frame_selection": FrameSelectionTrainer}


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
