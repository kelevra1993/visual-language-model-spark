"""
Shared dictionary contracts of the training framework: the keys of the batches produced by the data loaders, of the
model outputs and of the evaluation results. The trainers, models and loaders of every task follow them, so that the
general trainer can sum the losses, evaluate every test sample once and report the evaluations without knowing the
task.

Batch dictionary, produced by the loaders and moved to the device by the 'prepare_batch' hook of the trainer:
- "images": float BGR pixels in [0, 255], [B, C, H, W] for the image tasks, [B, T, C, H, W] for the sequence tasks,
  [B, 2T, C, H, W] for the sequence comparison task (the before member then the after member of every pair).
- "labels": the targets of the task (one [N_i] tensor per image for detection, [B] for classification, ...).
- "bounding_boxes": one [N_i, 4] tensor of [x1, y1, x2, y2] boxes per image, object detection, instance
  segmentation and keypoint detection only.
- "keypoints": one [N_i, K, 3] float tensor per image of the (x, y, visibility) keypoints of its objects in image
  pixels (visibility 0 unlabelled, 1 occluded, 2 visible), padded to the most keypoints of a class, in the order of
  its boxes, keypoint detection only.
- "instance_masks": one [N_i, H, W] uint8 tensor of binary masks per image, in the order of its boxes, instance
  segmentation only.
- "original_instance_masks" and "original_bounding_boxes": one [N_i, H0, W0] uint8 tensor of binary masks and one
  [N_i, 4] tensor of boxes per image at its original resolution, in the order of its boxes, test split of the instance
  segmentation evaluated at the original resolution only.
- "masks": [B, H, W] int64 class index of every pixel, semantic segmentation only (padding and unlabelled pixels at
  IGNORED_LABEL_INDEX).
- "sequence_mask": [B, T] booleans, True for the real frames of padded sequences, sequence tasks only ([B, 2T] for the
  pairs, the real frames coming first within each member).
- "sample_identifiers": [B] identifiers of the samples, used to evaluate every test sample exactly once.

Model output dictionary:
- every key ending in "_loss" is a scalar loss, all of them being summed by the trainer into the total loss;
- the other keys are the predictions of the task, documented by its model.
"""
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, List, NotRequired, Optional, TypedDict

# Key suffix of the losses returned by the models, all summed into the total loss by the trainer
LOSS_KEY_SUFFIX = "_loss"

# Batch key of the sample identifiers, used to evaluate every test sample exactly once
SAMPLE_IDENTIFIERS_KEY = "sample_identifiers"

# Label of the frames without class in the sequence labelling batches (padding frames, frames of unmapped
# categories) and of the pixels without class in the segmentation masks (padding, unlabelled pixels), ignored by the
# losses and metrics of these tasks
IGNORED_LABEL_INDEX = -100


class SequenceGrouping(TypedDict):
    """
    Grouping of the frames of a batch of images into sequences, given by the frame encoder of the sequence models to
    the sequence normalization layers of the backbone, which compute their statistics over the frames of each
    sequence. Either 'frames_per_sequence' is set (consecutive groups of the same number of frames, the frames of a
    [B, T] batch flattened in order, the path without masks of the deployment), or 'frame_sequence_indices' and
    'number_sequences' are (the group of every frame, for groups of any size, the path with masks of the training).
    """
    frames_per_sequence: Optional[int]
    frame_sequence_indices: Optional[torch.Tensor]
    number_sequences: Optional[int]


class EvaluationTable(TypedDict):
    """
    Table of the report of an evaluated checkpoint, rendered by evaluation/report_formatting.py in the console and in
    the Results.txt file of the run: its title, its column names (a line break makes a two-line header) and its rows of
    raw cells (text, integers, floats with 4 decimals, None written "—"), the per-class rows stored in class index
    order. The rows are shown by ascending value of the sort column, the worst classes first (None last, ties in
    stored order), or in stored order without one. A per-class table is only shown with the per-class report, and its
    legend is written below it.
    """
    title: str
    columns: List[str]
    rows: List[List[Any]]
    sort_column: Optional[str]
    per_class: bool
    legend: str


class EvaluationResults(TypedDict):
    """
    Results of the evaluation of a checkpoint on the test split, computed by the task and reported by the general
    trainer (evaluation/evaluation_report.py): the evaluated samples name the block of the report, the global metrics,
    notes and tables are printed and appended to the Results.txt file of the run, the scalars are written to the Test
    TensorBoard writer (and gathered in the evaluation summary of the checkpoints), and the files are extra outputs
    saved by the task (class metrics, confusion matrix, predictions...). The class scalars (one per class and metric)
    and the images (e.g. the confusion matrix) are per-class outputs, only reported with the per-class report, like
    the per-class tables. The sample images (the fixed test samples drawn with their targets and predictions by the
    build_sample_image hook of the task) are added by the general trainer, not by the task, and written to
    TensorBoard at every evaluation, whatever the per-class report. Every image is an [H, W, 3] uint8 RGB array. The
    headline metric is the name of the scalar comparing the checkpoints of a run, the higher the better.
    """
    evaluated_samples: str
    metrics: Dict[str, float]
    notes: List[str]
    tables: List[EvaluationTable]
    scalars: Dict[str, float]
    class_scalars: Dict[str, float]
    images: Dict[str, np.ndarray]
    sample_images: Dict[str, np.ndarray]
    files: Dict[str, Path]
    headline_metric: str


class ClassificationBatch(TypedDict):
    """
    Batch of the classification task, on the device and in the precision of the model: one int64 class index per
    image ([B]), or one float multi-hot vector per image ([B, K]) in multi-label mode.
    """
    images: torch.Tensor
    labels: torch.Tensor
    sample_identifiers: torch.Tensor


class SelfSupervisedBatch(TypedDict):
    """
    Batch of the self-supervised task, on the device and in the precision of the model: [B, 3, H, W] images, from
    which the trainer draws the two random views of every image, and one int64 class index per image ([B]), used by
    the k-nearest neighbour evaluation only, IGNORED_LABEL_INDEX for the unlabelled images.
    """
    images: torch.Tensor
    labels: torch.Tensor
    sample_identifiers: torch.Tensor


class SequenceBatch(TypedDict):
    """
    Batch of the sequence tasks, on the device and in the precision of the model: [B, T, C, H, W] frames padded with
    black frames, the [B, T] sequence mask (True for the real frames, which come first), and one int64 class index
    per sequence ([B], many to one) or per frame ([B, T] with -100 on the frames without class, many to many). For the
    sequence comparison task, the frames and the mask hold the before member followed by the after member of every
    pair ([B, 2T, C, H, W] and [B, 2T], the real frames first within each member), with one class index per pair.
    """
    images: torch.Tensor
    sequence_mask: torch.Tensor
    labels: torch.Tensor
    sample_identifiers: torch.Tensor


class SegmentationBatch(TypedDict):
    """
    Batch of the semantic segmentation task, on the device and in the precision of the model: [B, 3, H, W] images and
    the [B, H, W] int64 class index of every pixel, IGNORED_LABEL_INDEX on the padding and the unlabelled pixels.
    """
    images: torch.Tensor
    masks: torch.Tensor
    sample_identifiers: torch.Tensor


class DetectionBatch(TypedDict):
    """
    Batch of the object detection task, on the device and in the precision of the model.
    """
    images: torch.Tensor
    bounding_boxes: List[torch.Tensor]
    labels: List[torch.Tensor]
    sample_identifiers: torch.Tensor


class InstanceSegmentationBatch(DetectionBatch):
    """
    Batch of the instance segmentation task, on the device: the detection batch with the binary mask of every object,
    one [N_i, H, W] uint8 tensor per image in the order of its boxes and labels, and, for the test split evaluated at
    the original resolution only, the masks ([N_i, H0, W0] uint8) and boxes ([N_i, 4]) of the same objects at the
    original resolution of every image.
    """
    instance_masks: List[torch.Tensor]
    original_instance_masks: NotRequired[List[torch.Tensor]]
    original_bounding_boxes: NotRequired[List[torch.Tensor]]


class KeypointBatch(DetectionBatch):
    """
    Batch of the keypoint detection task, on the device: the detection batch of the whole images with the keypoints of
    every object, one [N_i, K, 3] float tensor of (x, y, visibility) per image in the order of its boxes and labels, in
    continuous image pixels (the first pixel centre at 0.5), padded to the most keypoints of a class. With
    'KeypointHead.ignore_unlabelled_objects', the training and validation batches hold the [B, H, W] uint8
    'ignored_regions', 1 on the pixels of the objects left out for having no labelled keypoint, which the heatmap loss
    leaves out. A training batch also holds the DataAugmenter 'augmentation_reports' of its images, for the input
    viewer.
    """
    keypoints: List[torch.Tensor]
    ignored_regions: NotRequired[torch.Tensor]
    augmentation_reports: NotRequired[List[Dict[str, Any]]]
