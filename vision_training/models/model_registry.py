"""
Registry of the model class of every task, keyed by 'ExperimentConfiguration.task' (object detection, classification,
sequence classification, labelling and comparison). Read by the trainers and by model_loading.py to rebuild a run from
its saved configuration.
"""

from typing import Dict, Type

from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.models.sequence.sequence_classifier import SequenceClassifier
from vision_training.models.sequence.sequence_comparator import SequenceComparator
from vision_training.models.sequence.sequence_labeller import SequenceLabeller
from vision_training.models.task_model import TaskModel

# Model class of every task, keyed by the value of 'ExperimentConfiguration.task'. Every class is built with the same
# arguments (configuration, device, dtype), so that a trained run can be rebuilt from its saved configuration alone
TASK_MODEL_CLASSES: Dict[str, Type[TaskModel]] = {"object_detection": TwoStageDetector,
                                                  "classification": ImageClassifier,
                                                  "sequence_classification": SequenceClassifier,
                                                  "sequence_labelling": SequenceLabeller,
                                                  "sequence_comparison": SequenceComparator}
