"""
Shared dictionary contracts of the training framework: the keys of the batches produced by the data loaders, of the
model outputs and of the evaluation results. The trainers, models and loaders of every task follow them, so that the
general trainer can sum the losses, evaluate every test sample once and report the evaluations without knowing the
task.

Batch dictionary, produced by the loaders and moved to the device by the 'prepare_batch' hook of the trainer:
- "images": float BGR pixels in [0, 255], [B, C, H, W] for the image tasks, [B, T, C, H, W] for the sequence tasks.
- "labels": the targets of the task (one [N_i] tensor per image for detection, [B] for classification, ...).
- "bounding_boxes": one [N_i, 4] tensor of [x1, y1, x2, y2] boxes per image, object detection only.
- "sequence_mask": [B, T] booleans, True for the real frames of padded sequences, sequence tasks only.
- "sample_identifiers": [B] identifiers of the samples, used to evaluate every test sample exactly once.

Model output dictionary:
- every key ending in "_loss" is a scalar loss, all of them being summed by the trainer into the total loss;
- the other keys are the predictions of the task, documented by its model.
"""
import torch

from pathlib import Path
from typing import Dict, List, TypedDict

# Key suffix of the losses returned by the models, all summed into the total loss by the trainer
LOSS_KEY_SUFFIX = "_loss"

# Batch key of the sample identifiers, used to evaluate every test sample exactly once
SAMPLE_IDENTIFIERS_KEY = "sample_identifiers"


class EvaluationResults(TypedDict):
    """
    Results of the evaluation of a checkpoint on the test split, computed by the task and reported by the general
    trainer: the lines are printed and appended to the Results.txt file of the run, the scalars are written to the
    Test TensorBoard writer, and the files are extra outputs saved by the task (confusion matrix, predictions...).
    """
    lines: List[str]
    scalars: Dict[str, float]
    files: Dict[str, Path]


class DetectionBatch(TypedDict):
    """
    Batch of the object detection task, on the device and in the precision of the model.
    """
    images: torch.Tensor
    bounding_boxes: List[torch.Tensor]
    labels: List[torch.Tensor]
    sample_identifiers: torch.Tensor
