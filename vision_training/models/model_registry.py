"""
Registry of the model class of every task, keyed by 'ExperimentConfiguration.task' (object detection, instance
segmentation, classification, semantic segmentation, sequence classification, labelling and comparison, frame
selection, self-supervised learning, keypoint detection). Read by the trainers and by model_loading.py to rebuild a run from its saved
configuration.
"""

import torch

from typing import Dict, Type

from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.models.instance_segmentation.instance_segmenter import InstanceSegmenter
from vision_training.models.keypoints.keypoint_detector import KeypointDetector
from vision_training.models.segmentation.semantic_segmenter import SemanticSegmenter
from vision_training.models.self_supervised.self_supervised_model import SelfSupervisedModel
from vision_training.models.sequence.frame_selector import FrameSelector
from vision_training.models.sequence.sequence_classifier import SequenceClassifier
from vision_training.models.sequence.sequence_comparator import SequenceComparator
from vision_training.models.sequence.sequence_labeller import SequenceLabeller
from vision_training.models.task_model import TaskModel

# Model class of every task, keyed by the value of 'ExperimentConfiguration.task'. Every class is built with the same
# arguments (configuration, device, dtype), so that a trained run can be rebuilt from its saved configuration alone
TASK_MODEL_CLASSES: Dict[str, Type[TaskModel]] = {"object_detection": TwoStageDetector,
                                                  "instance_segmentation": InstanceSegmenter,
                                                  "classification": ImageClassifier,
                                                  "semantic_segmentation": SemanticSegmenter,
                                                  "sequence_classification": SequenceClassifier,
                                                  "sequence_labelling": SequenceLabeller,
                                                  "sequence_comparison": SequenceComparator,
                                                  "frame_selection": FrameSelector,
                                                  "self_supervised": SelfSupervisedModel,
                                                  "keypoint_detection": KeypointDetector}

# Data type of the weights and inputs of every model, set here only: the trainings run in float32, mixed precision
# computing their forward passes in bfloat16 on top of it (Optimization.mixed_precision)
MODEL_DTYPE = torch.float32
