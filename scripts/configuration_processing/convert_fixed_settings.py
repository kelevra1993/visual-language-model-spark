"""
When run, removes from configuration files written before the fixed settings the settings whose value is now fixed in
the code (FIXED_SETTINGS, refused by check_removed_settings of
vision_training/configuration_processing/common_checks.py: the settings of the first conversion, then the batch
normalization momentums of the task heads, the strict fallback assignment, the localisation loss beta and the box loss
of the region proposal network of the detection tasks, then the gradient clipping norm, then the anchor centres of the
detection tasks, then the 'mask_scoring' section of the mask heads of the instance segmentation, off), and adds the
sections and the settings a task now requires with the values reproducing the runs written before them
(ADDED_SECTIONS: the 'Neck' of the detection tasks, without neck; ADDED_SETTINGS: the 'box_loss' of
their 'Detector' section, the smooth L1, the 'nms_criterion' and 'nms_distance_exponent' of their 'Detector' section,
the greedy NMS, the 'proposal_selection' of their 'RegionProposal' section, over all the levels at once, the
'instance_index_maps' of the 'Data' section of the instance segmentation, one mask per object, and the finer inputs and
fusion convolutions of its 'MaskHead' section, none; the 'benchmark_speed' diagnosis mode of every task, off; the
'shared' setting of the 'MaskHead' section of the instance segmentation, one mask head per level; the 'class_layer'
flag of the 'pretrained' block of the 'ClassificationHead' section of the
classification-like tasks and the 'class_layers' flag of the 'pretrained' block of the 'Detector' section of the
detection tasks, of the 'MaskHead' section of the instance segmentation and of the 'KeypointHead' section of the
keypoint detection, True for heads transferred from another run, which loaded their class layers, False otherwise;
the 'pretrained' block of the 'Neck' section of the detection tasks, a neck starting from random weights), and
renames the settings of RENAMED_SETTINGS ('SegmentationDecoder.pretrained.load_class_layers', now 'class_layers'),
so that the training_configuration.yaml of the saved runs, reloaded to resume a run, by
scripts/evaluation/evaluate_checkpoint.py and by the export, and the launch files of vision_training/main.py load
again.
The files are edited as text: their comments stay, the comment lines directly above a removed setting go with it, the
sections left empty are removed, a renamed setting keeps its value and the comments of its section name it under its
new name, and the settings still held by some tasks are kept in their files (the sequence settings of the sequence
tasks, the number of candidate frames of the frame selection). Every converted file is first copied to a backup, and a
conversion changing anything else than the removed, added and renamed settings and the added sections stops the
script. A file holding another value than the fixed one is listed with the values its runs change
to, except for the settings of RECORD_ONLY_SETTINGS, whose other values the code no longer trains (the CIoU box loss of
the region proposal network, a gradient clipping norm, the anchors centred on the cell centres, scored masks): such a
file is reported and left as it is, its runs staying as records. A file whose backbone would change (inception modules
starting at another block) is refused, the code no longer building its layers. Run as a script with the run roots and
the configuration folders searched given as arguments (parse_conversion_arguments), at least one folder being needed,
the files only listed unless --write_files is given, and its help printed when no argument is given.
"""

import argparse
import re
import shutil
import sys
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from vision_training.configuration_processing.common_checks import FIXED_MOMENTUM_COMPONENTS, REMOVED_SETTINGS
from vision_training.configuration_processing.configuration_checker import TASK_CONFIGURATION_CHECKS
from vision_training.configuration_processing.configuration_keys import CLASSIFICATION_LIKE_TASKS
from vision_training.data.loaders import NUMBER_LOADER_WORKERS, PREFETCH_BATCHES
from vision_training.evaluation.classification.classification_metrics import TOP_K_ACCURACY_CLASSES
from vision_training.models.backbones.backbone import INCEPTION_STARTING_BLOCK
from vision_training.models.components.convolution_block import BATCH_NORMALIZATION_MOMENTUM
from vision_training.tracking.experiment_tracker import ARTIFACTS_FOLDER_NAME
from vision_training.utilities.os_utilities import format_path_link, print_color

# Removed settings whose value is fixed in the code, the code taking the value of FIXED_VALUES instead of the one of the
# file. The other removed settings of REMOVED_SETTINGS (the views of the self-supervised task, the view losses) moved
# to other settings, which a deletion would lose, or were renamed (RENAMED_SETTINGS)
FIXED_SETTINGS = ["ExperimentConfiguration.evaluation.top_k", "ExperimentConfiguration.tracking.log_artifacts",
                  "ExperimentConfiguration.resume_training",
                  "ExperimentConfiguration.diagnosis_modes.compute_model_size",
                  "ExperimentConfiguration.diagnosis_modes.verbose", "Optimization.mixed_precision", "Data.loading",
                  "Backbone.modules.inception_starting_block", *FIXED_MOMENTUM_COMPONENTS,
                  "Data.data_augmentation.independent_frames", "Data.data_augmentation.shuffle_frames",
                  "RegionProposal.strict_fallback_assignment", "RegionProposal.localisation_loss_beta",
                  "Detector.localisation_loss_beta", "RegionProposal.box_loss", "Optimization.gradient_clipping_norm",
                  "RegionProposal.anchor_centres", "MaskHead.mask_scoring"]

# Value the code always uses for every fixed setting, by the path of the setting in a file: a file holding another
# value changes the behaviour of its runs once converted. The localisation loss beta is compared as the files write it,
# the fraction '1/9' of LOCALISATION_LOSS_BETA (vision_training/losses/detection_losses.py) read by YAML as a text
FIXED_VALUES = {"ExperimentConfiguration.evaluation.top_k": TOP_K_ACCURACY_CLASSES,
                "ExperimentConfiguration.tracking.log_artifacts": True, "ExperimentConfiguration.resume_training": True,
                "ExperimentConfiguration.diagnosis_modes.compute_model_size": True,
                "ExperimentConfiguration.diagnosis_modes.verbose": False, "Optimization.mixed_precision": True,
                "Data.loading.number_workers": NUMBER_LOADER_WORKERS, "Data.loading.prefetch_batches": PREFETCH_BATCHES,
                "Backbone.modules.inception_starting_block": INCEPTION_STARTING_BLOCK,
                **{key_path: BATCH_NORMALIZATION_MOMENTUM for key_path in FIXED_MOMENTUM_COMPONENTS},
                "Data.data_augmentation.independent_frames": False,
                "Data.data_augmentation.shuffle_frames.activated": False,
                "RegionProposal.strict_fallback_assignment": False, "RegionProposal.localisation_loss_beta": "1/9",
                "Detector.localisation_loss_beta": "1/9", "RegionProposal.box_loss": "smooth_l1",
                "Optimization.gradient_clipping_norm": None, "RegionProposal.anchor_centres": "cell_corners",
                "MaskHead.mask_scoring.activated": False}

# Fixed settings whose other values trained otherwise than the code now can, with the reason: a file holding another
# value than the one of FIXED_VALUES is not converted, its runs staying as records (their numbers kept in the results
# files) instead of loading as runs of the fixed value
RECORD_ONLY_SETTINGS = {"RegionProposal.box_loss": "the CIoU box loss of the region proposal network was removed, the "
                                                   "region proposal network always using the smooth L1",
                        "Optimization.gradient_clipping_norm": "the gradient clipping was removed, the gradients "
                                                               "never being clipped",
                        "RegionProposal.anchor_centres": "the anchors centred on the cell centres were removed, the "
                                                         "anchors always lying on the cell corners",
                        "MaskHead.mask_scoring.activated": "the mask scoring of Mask Scoring R-CNN was removed, the "
                                                           "masks always being ranked by their class scores"}

# Lines of the 'pretrained' block added to the 'Neck' section of the detection runs written before the neck transfer,
# whose neck started from random weights
NECK_PRETRAINED_LINES = [
    '# Neck of another DET or INS run initialising this one: same type (not "none"), output_channels, normalization,',
    '# levels and level block channels; load it from the run of the heads loaded behind it, which read its outputs.',
    'pretrained:',
    '  # Run folder of the loaded neck, null for random weights; not reloaded when the run resumes.',
    '  experiment_folder: null',
    '  # Checkpoint iteration, null for the last one.',
    '  iteration: null',
    '  # Freezes the loaded neck (no gradient, frozen batch statistics). Requires experiment_folder.',
    '  freeze: False',
    '  # Learning rate factor of the loaded neck, e.g. 0.1 to train it 10x slower.',
    '  learning_rate_multiplier: 1.0']

# Text of the 'Neck' section added to the detection runs written before the neck, whose heads read the blocks, with
# the 'pretrained' block of a neck starting from random weights
NECK_SECTION_TEXT = """# Neck between the level blocks and the heads, every level keeping the size of its block.
Neck:
  # "none" (heads read the blocks), "feature_pyramid" (top-down path from the coarse levels) or "path_aggregation"
  # (the feature pyramid, then a bottom-up path from the fine levels).
  type: "none"
  # Channels of every level after the neck, read by the heads (unused with "none").
  output_channels: 32
  normalization:
    # "batch", "layer" (over channels) or "none"
    feature_map_normalization: "batch"
""" + "".join(f"  {line}\n" for line in NECK_PRETRAINED_LINES)

# Sections a task now requires, added to the files of its runs written before them: the tasks holding the section, the
# top-level section it is written before, and its text, whose values reproduce those runs
ADDED_SECTIONS = {"Neck": {"tasks": ["object_detection", "instance_segmentation"], "before": "RegionProposal",
                           "text": NECK_SECTION_TEXT}}

# Lines of the 'box_loss' setting added to the 'Detector' section of the detection runs written before the CIoU loss,
# which trained with the smooth L1 loss: the comment lines, then the setting
DETECTOR_BOX_LOSS_LINES = [
    '# Box loss of the positive proposals: "smooth_l1" on the offsets, or "complete_iou" (1 - CIoU of the',
    '# decoded boxes, https://arxiv.org/abs/2005.03572).', 'box_loss: "smooth_l1"']

# Lines of the final NMS settings added to the 'Detector' section of the detection runs written before DIoU-NMS, which
# were evaluated with the greedy NMS: the comment lines, then the setting
DETECTOR_NMS_CRITERION_LINES = [
    '# Criterion of the final NMS: "iou", or "distance_iou" (the IoU minus the normalized distance of the centres,',
    '# https://arxiv.org/abs/1911.08287), compared with nms_iou_threshold.', 'nms_criterion: "iou"']
DETECTOR_NMS_DISTANCE_EXPONENT_LINES = [
    '# Exponent of the distance term of "distance_iou": 1.0 as in the paper, 0.6 in YOLOv4 (more boxes kept).',
    'nms_distance_exponent: 1.0']

# Lines of the proposal selection added to the 'RegionProposal' section of the detection runs written before the
# selection level by level, which selected their proposals over all the levels at once: the comment lines, then the
# setting
REGION_PROPOSAL_SELECTION_LINES = [
    '# "joint" (one top-k and NMS over all the levels) or "per_level" (top-k and NMS of every level, then the best',
    '# over all the levels, as torchvision, empty proposals dropped), which makes pre_nms_proposals a count per level.',
    'proposal_selection: "joint"']

# Lines of the instance index maps added to the 'Data' section of the instance segmentation runs written before them,
# which loaded one mask per object: the comment lines, then the setting
DATA_INSTANCE_INDEX_MAPS_LINES = [
    '# One uint8 map numbering the objects of an image instead of one mask per object (faster loader, expanded on the',
    '# GPU); needs run length encodings whose masks never overlap (synthetic datasets), False for COCO (polygons).',
    'instance_index_maps: False']

# Lines of the finer inputs of the mask heads added to the 'MaskHead' section of the instance segmentation runs written
# before them, whose mask heads read the map of their level alone: the comment lines, then the setting
MASK_HEAD_SKIP_BLOCKS_AT_POOL_SIZE_LINES = [
    '# Backbone blocks (before their pooling, 0 the normalized input image) pooled at the pool size of every level and',
    '# concatenated to its pooled features (U-Net skips), e.g. [4] (stride 8); [] for none.',
    'skip_blocks_at_pool_size: []']
MASK_HEAD_IMAGE_BRANCH_CONVOLUTIONS_LINES = [
    '# Image branch: [layers, channels] of plain 3x3 convolutions on the box crop of the normalized input image at M,',
    '# shared by the levels and concatenated after the doublings, e.g. [5, 32]; [0, 0] for none.',
    'image_branch_convolutions: [0, 0]']
MASK_HEAD_SKIP_BLOCKS_AT_MASK_RESOLUTION_LINES = [
    '# Backbone blocks (before their pooling) pooled at M and concatenated after the doublings, e.g. [3] (stride 4);',
    '# [] for none.', 'skip_blocks_at_mask_resolution: []']
MASK_HEAD_FUSION_CONVOLUTIONS_LINES = [
    '# [layers, channels] of plain 3x3 convolutions at M before the predictor, mixing the concatenated features, e.g.',
    '# [1, 64]; [0, 0] for none, the predictor following the doublings as in Mask R-CNN.',
    'fusion_convolutions: [0, 0]']

# Lines of the sharing of the mask heads added to the 'MaskHead' section of the instance segmentation runs written
# before it, whose mask heads are one per level: the comment lines, then the setting
MASK_HEAD_SHARED_LINES = [
    '# True: one mask head read by every level (Mask R-CNN), trained on all the positives; needs the same mask_head',
    '# on every level and equal level channels (a neck). False: one mask head per level.',
    'shared: False']

# Comment lines of the class layer flag added to the 'pretrained' block of the 'ClassificationHead' section of the runs
# of the classification-like tasks written before it, then the setting: False for the heads trained from random
# weights, True for the heads transferred from another run, which loaded their class layer
CLASSIFICATION_HEAD_CLASS_LAYER_COMMENT_LINES = [
    '# True also loads the class layer: source classes (its data.json) first in order or by name, others random; needs',
    '# experiment_folder. False loads the hidden layers only, the class layer random for any classes (HRCL token).']
CLASSIFICATION_HEAD_CLASS_LAYER_LINES = CLASSIFICATION_HEAD_CLASS_LAYER_COMMENT_LINES + ['class_layer: False']
TRANSFERRED_CLASSIFICATION_HEAD_CLASS_LAYER_LINES = CLASSIFICATION_HEAD_CLASS_LAYER_COMMENT_LINES + ['class_layer: True']

# Comment lines of the class layer flag added to the 'pretrained' block of the 'Detector' section of the detection
# tasks, of the 'MaskHead' section of the instance segmentation and of the 'KeypointHead' section of the keypoint
# detection, written before it, by section; the setting follows, False for the heads trained from random weights, True
# for the heads transferred from another run, which loaded their class layers
HEAD_CLASS_LAYERS_COMMENT_LINES = {
    "Detector": [
        ('# True also loads the class layers (classifier, box regressor): source classes (its data.json) first in '
         'order or by'),
        '# name, others random; needs experiment_folder. False leaves them random for any classes (DRCL token).'],
    "MaskHead": [
        '# True also loads the class layer (mask predictor): source classes (its data.json) first in order or by name,',
        '# others random; needs experiment_folder. False leaves it random for any classes (MRCL token).'],
    "KeypointHead": [
        ('# True also loads the heatmap and tag convolutions: same classes and keypoints in the same order (its '
         'data.json);'),
        ('# needs experiment_folder. False leaves them random for any keypoints, as their HH branch inputs (KHRCL '
         'token).')]}

# Tasks of the sections of HEAD_CLASS_LAYERS_COMMENT_LINES
HEAD_CLASS_LAYERS_TASKS = {"Detector": ["object_detection", "instance_segmentation"],
                           "MaskHead": ["instance_segmentation"], "KeypointHead": ["keypoint_detection"]}

# Lines of the speed benchmark diagnosis mode added to the 'diagnosis_modes' of the runs of every task written before
# it, off: the comment lines, then the setting
DIAGNOSIS_BENCHMARK_SPEED_LINES = [
    '# Times the training step and the inference (PyTorch, TorchScript, ONNX Runtime, TensorRT) on one training batch,',
    '# at the configured and largest batch sizes that fit, then exits without training; no run folder is created.',
    'benchmark_speed: False']

# Settings a task now requires inside sections its files hold, added to the files of its runs written before them: the
# tasks holding the setting, the setting it is written after (at its indentation), and its lines, whose value
# reproduces those runs, or the lines of 'lines_when_set' for a file setting its 'key_path' to another value than null
# (see select_added_lines). They are added in this order, so that a setting can follow one added before it
ADDED_SETTINGS = {"Detector.box_loss": {"tasks": ["object_detection", "instance_segmentation"],
                                        "after": "Detector.total_training_samples", "lines": DETECTOR_BOX_LOSS_LINES},
                  "Detector.nms_criterion": {"tasks": ["object_detection", "instance_segmentation"],
                                             "after": "Detector.nms_iou_threshold",
                                             "lines": DETECTOR_NMS_CRITERION_LINES},
                  "Detector.nms_distance_exponent": {"tasks": ["object_detection", "instance_segmentation"],
                                                     "after": "Detector.nms_criterion",
                                                     "lines": DETECTOR_NMS_DISTANCE_EXPONENT_LINES},
                  "RegionProposal.proposal_selection": {"tasks": ["object_detection", "instance_segmentation"],
                                                        "after": "RegionProposal.normalization",
                                                        "lines": REGION_PROPOSAL_SELECTION_LINES},
                  "Data.instance_index_maps": {"tasks": ["instance_segmentation"], "after": "Data.image_settings",
                                               "lines": DATA_INSTANCE_INDEX_MAPS_LINES},
                  "MaskHead.skip_blocks_at_pool_size": {"tasks": ["instance_segmentation"],
                                                        "after": "MaskHead.upsampling",
                                                        "lines": MASK_HEAD_SKIP_BLOCKS_AT_POOL_SIZE_LINES},
                  "MaskHead.image_branch_convolutions": {"tasks": ["instance_segmentation"],
                                                         "after": "MaskHead.skip_blocks_at_pool_size",
                                                         "lines": MASK_HEAD_IMAGE_BRANCH_CONVOLUTIONS_LINES},
                  "MaskHead.skip_blocks_at_mask_resolution": {"tasks": ["instance_segmentation"],
                                                              "after": "MaskHead.image_branch_convolutions",
                                                              "lines": MASK_HEAD_SKIP_BLOCKS_AT_MASK_RESOLUTION_LINES},
                  "MaskHead.fusion_convolutions": {"tasks": ["instance_segmentation"],
                                                   "after": "MaskHead.skip_blocks_at_mask_resolution",
                                                   "lines": MASK_HEAD_FUSION_CONVOLUTIONS_LINES},
                  "ExperimentConfiguration.diagnosis_modes.benchmark_speed": {
                      "tasks": list(TASK_CONFIGURATION_CHECKS),
                      "after": "ExperimentConfiguration.diagnosis_modes.view_input_images",
                      "lines": DIAGNOSIS_BENCHMARK_SPEED_LINES},
                  "MaskHead.shared": {"tasks": ["instance_segmentation"], "after": "MaskHead.fusion_convolutions",
                                      "lines": MASK_HEAD_SHARED_LINES},
                  "ClassificationHead.pretrained.class_layer": {
                      "tasks": list(CLASSIFICATION_LIKE_TASKS), "after": "ClassificationHead.pretrained.freeze",
                      "lines": CLASSIFICATION_HEAD_CLASS_LAYER_LINES,
                      "lines_when_set": {"key_path": "ClassificationHead.pretrained.experiment_folder",
                                         "lines": TRANSFERRED_CLASSIFICATION_HEAD_CLASS_LAYER_LINES}},
                  **{f"{section_name}.pretrained.class_layers": {
                      "tasks": HEAD_CLASS_LAYERS_TASKS[section_name], "after": f"{section_name}.pretrained.freeze",
                      "lines": comment_lines + ["class_layers: False"],
                      "lines_when_set": {"key_path": f"{section_name}.pretrained.experiment_folder",
                                         "lines": comment_lines + ["class_layers: True"]}}
                     for section_name, comment_lines in HEAD_CLASS_LAYERS_COMMENT_LINES.items()},
                  "Neck.pretrained": {"tasks": ["object_detection", "instance_segmentation"],
                                      "after": "Neck.normalization", "lines": NECK_PRETRAINED_LINES}}

# Settings renamed in the files of some tasks, by their path before the renaming: the tasks holding the setting and its
# new name, the setting keeping its value and its place, and the mentions of the old name in the comments of its
# top-level section taking the new name
RENAMED_SETTINGS = {"SegmentationDecoder.pretrained.load_class_layers": {"tasks": ["semantic_segmentation"],
                                                                         "new_name": "class_layers"}}

# Name of the configuration saved in every run folder
RUN_CONFIGURATION_NAME = "training_configuration.yaml"

# Folder of a configuration folder keeping the backups of its launch files, mirroring their subfolders
PREVIOUS_CONFIGURATIONS_FOLDER_NAME = "previous_configurations"

# Key line of a YAML file: its indentation, its key and what follows the colon
KEY_LINE_PATTERN = re.compile(pattern=r"^(\s*)([A-Za-z0-9_]+):(.*)$")

# Suffix of the backups of the latest conversion unless the command line gives another one: '_before_fixed_settings'
# (2026-10-29), '_before_fixed_head_settings' (2026-11-05), '_before_neck' for the 'Neck' section added on 2026-11-07,
# '_before_box_loss' for the 'box_loss' settings added on 2026-11-08, '_before_nms_criterion' for the final NMS settings
# added on 2026-11-08, '_before_region_proposal_box_loss' for the 'RegionProposal.box_loss' removed on 2026-11-10,
# '_before_gradient_clipping_norm' for the 'Optimization.gradient_clipping_norm' removed on 2026-11-11,
# '_before_proposal_selection' for the 'RegionProposal.proposal_selection' added on 2026-11-11,
# '_before_anchor_centres' for the 'RegionProposal.anchor_centres' added on 2026-11-12,
# '_before_anchor_centres_removal' for its removal on 2026-11-13, '_before_instance_index_maps' for the
# 'Data.instance_index_maps' added on 2026-11-13, '_before_mask_fine_inputs' for the finer inputs and fusion
# convolutions of the 'MaskHead' section added on 2026-11-15, '_before_benchmark_speed' for the
# 'diagnosis_modes.benchmark_speed' added on 2026-11-16, '_before_shared_mask_head' for the 'MaskHead.shared' added on
# 2026-11-20, '_before_mask_scoring' for the 'MaskHead.mask_scoring' added on 2026-11-21, '_before_class_layer' for the
# 'ClassificationHead.pretrained.class_layer' added on 2026-11-24, then '_before_class_layers' for the
# 'pretrained.class_layers' of the detection, mask and keypoint heads added and the
# 'SegmentationDecoder.pretrained.load_class_layers' renamed on 2026-11-24, then '_before_mask_scoring_removal' for the
# removal of the 'MaskHead.mask_scoring' on 2026-11-27, then '_before_neck_pretrained' for the 'Neck.pretrained' block
# added on 2026-11-27
DEFAULT_BACKUP_SUFFIX = "_before_neck_pretrained"


def get_indentation(line: str) -> int:
    """
    Counts the leading spaces of a line of a configuration file, which give its depth in the YAML sections.

    Args:
        line (str): The line.

    Returns:
        int: The number of leading spaces.
    """
    return len(line) - len(line.lstrip(" "))


def is_comment_or_blank(line: str) -> bool:
    """
    Tells whether a line of a configuration file holds no setting, a comment or an empty line.

    Args:
        line (str): The line.

    Returns:
        bool: True for a comment or blank line.
    """
    return not line.strip() or line.strip().startswith("#")


def find_key_paths(lines: List[str]) -> List[Optional[str]]:
    """
    Gives the dotted path of the key of every line of a configuration file (e.g. 'Data.loading.number_workers'), so
    that the lines of the fixed settings are found whatever their indentation and their neighbours.

    Args:
        lines (List[str]): The lines of the file.

    Returns:
        List[Optional[str]]: The key path of every line, None for the comments, blank lines and list items.
    """
    open_keys = []
    key_paths = []
    for line in lines:
        # Lines without a key of their own
        key_match = KEY_LINE_PATTERN.match(line)
        if is_comment_or_blank(line=line) or key_match is None:
            key_paths.append(None)
            continue

        # Close the sections at the same or a deeper indentation, then open the key of the line
        indentation = len(key_match.group(1))
        while open_keys and open_keys[-1][0] >= indentation:
            open_keys.pop()
        open_keys.append((indentation, key_match.group(2)))
        key_paths.append(".".join(key for _, key in open_keys))
    return key_paths


def find_block_end(lines: List[str], start_index: int) -> int:
    """
    Finds the end of the block of the key of a line, its own line and the deeper lines below it (its settings and the
    comments among them), the blank lines after its last setting staying in the file.

    Args:
        lines (List[str]): The lines of the file.
        start_index (int): The index of the line of the key.

    Returns:
        int: The index after the last line of the block.
    """
    indentation = get_indentation(line=lines[start_index])
    last_content_index = start_index
    for line_index in range(start_index + 1, len(lines)):
        # Blank lines are skipped, the first line at the indentation of the key or above closing the block
        if not lines[line_index].strip():
            continue
        if get_indentation(line=lines[line_index]) <= indentation:
            break
        last_content_index = line_index
    return last_content_index + 1


def mark_block_with_comments(lines: List[str], start_index: int, end_index: int, removed_indices: Set[int]) -> None:
    """
    Marks the lines of a block for removal with the comment lines directly above it at its indentation, which
    describe it.

    Args:
        lines (List[str]): The lines of the file.
        start_index (int): The index of the line of the key.
        end_index (int): The index after the last line of the block.
        removed_indices (Set[int]): The indices of the lines to remove, completed in place.
    """
    removed_indices.update(range(start_index, end_index))

    # Comment lines of the block, directly above its key and at its indentation
    indentation = get_indentation(line=lines[start_index])
    line_index = start_index - 1
    while line_index >= 0 and lines[line_index].strip().startswith("#") and \
            get_indentation(line=lines[line_index]) == indentation:
        removed_indices.add(line_index)
        line_index -= 1


def remove_fixed_settings(configuration_text: str, task: str) -> str:
    """
    Removes the fixed settings from the text of a configuration file, keeping those still held by its task, then the
    sections left without any setting, each with the comment lines directly above it.

    Args:
        configuration_text (str): The text of the configuration file.
        task (str): The 'ExperimentConfiguration.task' of the file.

    Returns:
        str: The text without the fixed settings, identical when the file holds none.
    """
    lines = configuration_text.split("\n")
    removed_indices: Set[int] = set()

    # Blocks of the fixed settings the task no longer holds
    for line_index, key_path in enumerate(find_key_paths(lines=lines)):
        if key_path in FIXED_SETTINGS and task not in REMOVED_SETTINGS[key_path][1]:
            mark_block_with_comments(lines=lines, start_index=line_index, removed_indices=removed_indices,
                                     end_index=find_block_end(lines=lines, start_index=line_index))

    # Sections left without any setting, until none is left
    section_removed = True
    while section_removed:
        section_removed = False
        for line_index, line in enumerate(lines):
            key_match = KEY_LINE_PATTERN.match(line)
            if line_index in removed_indices or key_match is None or is_comment_or_blank(line=line) or \
                    key_match.group(3).strip():
                continue
            end_index = find_block_end(lines=lines, start_index=line_index)
            setting_indices = [index for index in range(line_index + 1, end_index)
                               if not is_comment_or_blank(line=lines[index])]
            if setting_indices and all(index in removed_indices for index in setting_indices):
                mark_block_with_comments(lines=lines, start_index=line_index, end_index=end_index,
                                         removed_indices=removed_indices)
                section_removed = True

    # Kept lines, the blank line following a removed block dropped when the line before the block is blank too, so
    # that a block removed between two blank lines leaves a single one
    kept_lines: List[str] = []
    previous_kept_index = -1
    for line_index, line in enumerate(lines):
        if line_index in removed_indices:
            continue
        if not line.strip() and kept_lines and not kept_lines[-1].strip() and previous_kept_index != line_index - 1:
            continue
        kept_lines.append(line)
        previous_kept_index = line_index
    return "\n".join(kept_lines)


def rename_settings(configuration_text: str, task: str) -> str:
    """
    Renames in the text of a configuration file the settings of RENAMED_SETTINGS its task holds, keeping their value,
    their indentation and their place, and gives the new name to the mentions of the old name in the comment lines of
    their top-level section (e.g. 'Its class convolution and supervision heads need load_class_layers.'), so that the
    runs written before the renaming load again.

    Args:
        configuration_text (str): The text of the configuration file.
        task (str): The 'ExperimentConfiguration.task' of the file.

    Returns:
        str: The text with the renamed settings, identical when the file holds none.
    """
    lines = configuration_text.split("\n")
    for key_path, renamed_setting in RENAMED_SETTINGS.items():
        key_paths = find_key_paths(lines=lines)
        if task not in renamed_setting["tasks"] or key_path not in key_paths:
            continue

        # Key of the setting, its indentation and value kept
        setting_index = key_paths.index(key_path)
        old_name, new_name = key_path.rsplit(".", 1)[1], renamed_setting["new_name"]
        key_match = KEY_LINE_PATTERN.match(lines[setting_index])
        lines[setting_index] = f"{key_match.group(1)}{new_name}:{key_match.group(3)}"

        # Mentions of the old name in the comments of the top-level section holding the setting
        section_index = key_paths.index(key_path.split(".", 1)[0])
        for line_index in range(section_index, find_block_end(lines=lines, start_index=section_index)):
            if lines[line_index].strip().startswith("#"):
                lines[line_index] = re.sub(pattern=rf"\b{old_name}\b", repl=new_name, string=lines[line_index])
    return "\n".join(lines)


def add_required_sections(configuration_text: str, task: str, configuration_path: Path) -> str:
    """
    Adds to the text of a configuration file the sections of ADDED_SECTIONS its task requires and it does not hold,
    each written before its top-level section and the comment lines directly above it, followed by a blank line, so
    that the runs written before the section load again with the values that reproduce them.

    Args:
        configuration_text (str): The text of the configuration file.
        task (str): The 'ExperimentConfiguration.task' of the file.
        configuration_path (Path): The file, named in the error.

    Returns:
        str: The text with the added sections, identical when the file holds them or its task requires none.

    Raises:
        ValueError: If the file lacks the top-level section an added section is written before.
    """
    lines = configuration_text.split("\n")
    for section_name, added_section in ADDED_SECTIONS.items():
        # Sections of other tasks, and sections the file already holds
        top_level_keys = [key_match.group(2) for key_match in map(KEY_LINE_PATTERN.match, lines)
                          if key_match is not None and not key_match.group(1)]
        if task not in added_section["tasks"] or section_name in top_level_keys:
            continue
        if added_section["before"] not in top_level_keys:
            raise ValueError(f"{format_path_link(path=configuration_path)} has no '{added_section['before']}' section "
                             f"to write the '{section_name}' section before.")

        # Line of the following section, moved above the comment lines describing it
        insertion_index = next(line_index for line_index, line in enumerate(lines)
                               if line.startswith(f"{added_section['before']}:"))
        while insertion_index > 0 and lines[insertion_index - 1].startswith("#"):
            insertion_index -= 1
        lines[insertion_index:insertion_index] = added_section["text"].split("\n")
    return "\n".join(lines)


def holds_parent_section(key_paths: List[str], key_path: str) -> bool:
    """
    Tells whether a configuration file holds the section of a setting of ADDED_SETTINGS (e.g.
    'ExperimentConfiguration.diagnosis_modes' for 'ExperimentConfiguration.diagnosis_modes.benchmark_speed'), so that
    the setting is only added to the files holding its section, the partial drafts of settings being left as they are.

    Args:
        key_paths (List[str]): The dotted paths of the keys of the file, its sections and settings.
        key_path (str): The dotted path of the added setting.

    Returns:
        bool: True when a key of the file is the section of the setting or lies inside it.
    """
    section_path = key_path.rsplit(".", 1)[0]
    return any(path == section_path or path.startswith(f"{section_path}.") for path in key_paths)


def select_added_lines(added_setting: Dict[str, Any], settings: Dict[str, Any]) -> List[str]:
    """
    Gives the lines of a setting of ADDED_SETTINGS for one configuration file, so that a setting whose value
    reproducing the runs written before it depends on another setting of the file is added with the right value: its
    'lines', or the 'lines' of its 'lines_when_set' when the file sets the setting named there to another value than
    null, e.g. 'class_layer: True' for a classification head transferred from another run, which loaded its class
    layer, and 'class_layer: False' for a head trained from random weights.

    Args:
        added_setting (Dict[str, Any]): A value of ADDED_SETTINGS.
        settings (Dict[str, Any]): The loaded configuration file, before the setting is added.

    Returns:
        List[str]: The comment lines and the setting to add.
    """
    if "lines_when_set" not in added_setting:
        return added_setting["lines"]

    # Setting deciding the value, set when the file holds it with another value than null
    deciding_path = added_setting["lines_when_set"]["key_path"]
    flattened_settings = flatten_settings(settings=settings)
    if deciding_path in flattened_settings and flattened_settings[deciding_path] is not None:
        return added_setting["lines_when_set"]["lines"]
    return added_setting["lines"]


def add_required_settings(configuration_text: str, task: str, configuration_path: Path) -> str:
    """
    Adds to the text of a configuration file the settings of ADDED_SETTINGS its task requires and it does not hold,
    each written with its comment lines right after the block of the setting it follows and at its indentation, so
    that the runs written before the setting load again with the value that reproduces them. A file without the
    section of a setting (a partial draft of settings, such as a file of defaults without 'diagnosis_modes') is left
    without it.

    Args:
        configuration_text (str): The text of the configuration file.
        task (str): The 'ExperimentConfiguration.task' of the file.
        configuration_path (Path): The file, named in the error.

    Returns:
        str: The text with the added settings, identical when the file holds them or its task requires none.

    Raises:
        ValueError: If the file holds the section of an added setting but lacks the setting it is written after.
    """
    lines = configuration_text.split("\n")
    settings = yaml.safe_load(configuration_text)
    for key_path, added_setting in ADDED_SETTINGS.items():
        # Settings of other tasks, settings the file already holds, and settings of a section the file does not hold
        key_paths = find_key_paths(lines=lines)
        if task not in added_setting["tasks"] or key_path in key_paths or \
                not holds_parent_section(key_paths=[path for path in key_paths if path is not None],
                                         key_path=key_path):
            continue
        if added_setting["after"] not in key_paths:
            raise ValueError(f"{format_path_link(path=configuration_path)} has no '{added_setting['after']}' setting "
                             f"to write the '{key_path}' setting after.")

        # Lines written after the block of the preceding setting, at its indentation, with the value of the file
        after_index = key_paths.index(added_setting["after"])
        insertion_index = find_block_end(lines=lines, start_index=after_index)
        indentation = " " * get_indentation(line=lines[after_index])
        lines[insertion_index:insertion_index] = [
            indentation + line for line in select_added_lines(added_setting=added_setting, settings=settings)]
    return "\n".join(lines)


def flatten_settings(settings: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Gives every setting of a loaded configuration under its dotted path, so that a converted file can be compared
    with the original setting by setting.

    Args:
        settings (Dict[str, Any]): The loaded configuration, or one of its sections.
        prefix (str): The path of the section. Defaults to "" for the whole configuration.

    Returns:
        Dict[str, Any]: The value of every setting, an empty section counting as one setting.
    """
    flattened_settings = {}
    for key, value in settings.items():
        key_path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            flattened_settings.update(flatten_settings(settings=value, prefix=key_path))
        else:
            flattened_settings[key_path] = value
    return flattened_settings


def check_conversion(original_text: str, converted_text: str, task: str, configuration_path: Path) -> List[str]:
    """
    Checks that a conversion only removed the fixed settings the task no longer holds, renamed the settings of
    RENAMED_SETTINGS with their values and added the sections of ADDED_SECTIONS and the settings of ADDED_SETTINGS the
    task requires and the file lacked, with their values, every other setting keeping its value so that a mistake of
    the text edition never changes a run, and that the backbone
    of the file keeps its layers. Lists the removed settings whose value differs from the one the code now uses, which
    change the runs of the file once converted (e.g. a launch file with 'resume_training: false' resuming its run).

    Args:
        original_text (str): The text of the file before the conversion.
        converted_text (str): The text after the conversion.
        task (str): The 'ExperimentConfiguration.task' of the file.
        configuration_path (Path): The file, named in the error.

    Returns:
        List[str]: Every removed setting holding another value than the fixed one, as '<path> <value> -> <fixed>'.

    Raises:
        ValueError: If the converted file loses or changes another setting, keeps a removed or renamed one or lacks an
            added one, or if the inception modules of its backbone start at another block than the fixed one.
    """
    removed_paths = [key_path for key_path in FIXED_SETTINGS if task not in REMOVED_SETTINGS[key_path][1]]
    original_settings = flatten_settings(settings=yaml.safe_load(original_text))
    converted_settings = flatten_settings(settings=yaml.safe_load(converted_text))

    # Inception modules starting at another block give other layers, which the code no longer builds, so that the
    # checkpoints of the run would no longer load
    inception_path, starting_block_path = "Backbone.modules.inception", "Backbone.modules.inception_starting_block"
    if inception_path in original_settings and original_settings[inception_path] is True and \
            starting_block_path in original_settings and \
            original_settings[starting_block_path] != INCEPTION_STARTING_BLOCK:
        raise ValueError(f"The backbone of {format_path_link(path=configuration_path)} starts its inception modules at "
                         f"block {original_settings[starting_block_path]}, the code always starting them at block "
                         f"{INCEPTION_STARTING_BLOCK}: its runs cannot be rebuilt, the file is not converted.")

    # The original settings without the removed ones, the sections left empty disappearing
    expected_settings = {key_path: value for key_path, value in original_settings.items()
                         if not any(key_path == removed_path or key_path.startswith(f"{removed_path}.")
                                    for removed_path in removed_paths)}
    expected_settings = {key_path: value for key_path, value in expected_settings.items()
                         if value != {} or key_path in converted_settings}

    # Renamed settings, under their new name with their value
    for key_path, renamed_setting in RENAMED_SETTINGS.items():
        if task in renamed_setting["tasks"] and key_path in expected_settings:
            new_path = f"{key_path.rsplit('.', 1)[0]}.{renamed_setting['new_name']}"
            expected_settings[new_path] = expected_settings.pop(key_path)

    # Settings of the sections the task requires and the original file lacked, with the values of ADDED_SECTIONS
    for section_name, added_section in ADDED_SECTIONS.items():
        original_section_paths = [key_path for key_path in original_settings
                                  if key_path == section_name or key_path.startswith(f"{section_name}.")]
        if task in added_section["tasks"] and not original_section_paths:
            expected_settings.update(flatten_settings(settings=yaml.safe_load(added_section["text"])))

    # Settings the task requires inside its sections and the original file lacked, with the values of ADDED_SETTINGS
    # chosen for the original file, a file without the section of a setting (a partial draft) staying without it. An
    # added block of settings ('Neck.pretrained') is held by the file when it holds any setting inside it, and its
    # values are compared setting by setting
    for key_path, added_setting in ADDED_SETTINGS.items():
        original_holds_setting = key_path in original_settings or \
            any(path.startswith(f"{key_path}.") for path in original_settings)
        if task in added_setting["tasks"] and not original_holds_setting and \
                holds_parent_section(key_paths=list(expected_settings), key_path=key_path):
            added_lines = select_added_lines(added_setting=added_setting, settings=yaml.safe_load(original_text))
            added_value = yaml.safe_load(stream="\n".join(added_lines))[key_path.split(".")[-1]]
            if isinstance(added_value, dict) and added_value:
                expected_settings.update(flatten_settings(settings=added_value, prefix=key_path))
            else:
                expected_settings[key_path] = added_value
    if converted_settings != expected_settings:
        # Settings lost or added by the conversion, and settings whose value changed
        changed_paths = sorted(set(expected_settings) ^ set(converted_settings) |
                               {key_path for key_path in set(expected_settings) & set(converted_settings)
                                if expected_settings[key_path] != converted_settings[key_path]})
        raise ValueError(f"The conversion of {format_path_link(path=configuration_path)} changes other settings than "
                         f"the fixed ones: {', '.join(changed_paths)}.")

    # Removed settings holding another value than the one the code now uses
    return [f"{key_path} {original_settings[key_path]!r} -> {fixed_value!r}"
            for key_path, fixed_value in FIXED_VALUES.items()
            if key_path in original_settings and key_path not in converted_settings and
            original_settings[key_path] != fixed_value]


def get_backup_path(configuration_path: Path, configuration_folder: Optional[Path], backup_suffix: str) -> Path:
    """
    Gives the backup of a converted file: next to it for the configuration of a run folder, as the backups of the
    earlier conversions, and in the 'previous_configurations' folder of its configuration folder, at the same place,
    for a launch file, so that the backups are not taken for launch files.

    Args:
        configuration_path (Path): The converted file.
        configuration_folder (Optional[Path]): The configuration folder holding the launch file, None for the
            configuration of a run folder.
        backup_suffix (str): The suffix added to the name of the file, one per conversion.

    Returns:
        Path: The backup file.
    """
    backup_name = f"{configuration_path.stem}{backup_suffix}{configuration_path.suffix}"
    if configuration_folder is None:
        return configuration_path.with_name(backup_name)
    return (configuration_folder / PREVIOUS_CONFIGURATIONS_FOLDER_NAME /
            configuration_path.parent.relative_to(configuration_folder) / backup_name)


def list_configuration_files(run_roots: List[Path], configuration_folders: List[Path]) -> List[Dict[str, Any]]:
    """
    Lists the files to convert: the training_configuration.yaml of every run folder under the run roots, without the
    copies MLflow keeps in its artifacts folder, and the launch files of the configuration folders, without their
    'previous_configurations' backups, each with the configuration folder deciding where its backup goes.

    Args:
        run_roots (List[Path]): The folders searched for run folders, e.g. the 'experiment_folder' of the launches.
        configuration_folders (List[Path]): The folders of launch files, searched with their subfolders.

    Returns:
        List[Dict[str, Any]]: Every file as {"path": Path, "configuration_folder": Optional[Path]}.
    """
    configuration_files = []

    # Configurations saved in the run folders
    for run_root in run_roots:
        configuration_files += [{"path": path, "configuration_folder": None}
                                for path in sorted(run_root.rglob(RUN_CONFIGURATION_NAME))
                                if ARTIFACTS_FOLDER_NAME not in path.parts]

    # Launch files, without their backups
    for configuration_folder in configuration_folders:
        configuration_files += [
            {"path": path, "configuration_folder": configuration_folder}
            for path in sorted(configuration_folder.rglob("*.yaml"))
            if PREVIOUS_CONFIGURATIONS_FOLDER_NAME not in path.relative_to(configuration_folder).parts]
    return configuration_files


def convert_fixed_settings(run_roots: List[Path], configuration_folders: List[Path], write_files: bool,
                           backup_suffix: str) -> List[Path]:
    """
    Converts the configuration files of the saved runs and of the launch folders to the fixed settings, the added
    sections and settings and the renamed settings, each checked by check_conversion and copied to its backup before
    being written, or only lists them without writing anything. A file holding a setting of RECORD_ONLY_SETTINGS with
    another value than the fixed one is reported and left as it is, whatever write_files.

    Args:
        run_roots (List[Path]): The folders searched for the training_configuration.yaml of run folders.
        configuration_folders (List[Path]): The folders of launch files.
        write_files (bool): True to write the converted files and their backups, False to only list them.
        backup_suffix (str): The suffix of the backups, different for every conversion so that the backups of an
            earlier one are never overwritten.

    Returns:
        List[Path]: The files holding fixed settings or settings to rename, or lacking an added section or setting,
            converted when write_files is True.

    Raises:
        FileExistsError: If the backup of a file to convert already exists, which would be overwritten.
    """
    converted_paths = []
    for configuration_file in list_configuration_files(run_roots=run_roots,
                                                       configuration_folders=configuration_folders):
        # Only the configurations of a task, not the other YAML files of the folders
        configuration_path = configuration_file["path"]
        original_text = configuration_path.read_text(encoding="utf-8")
        settings = yaml.safe_load(original_text)
        if not isinstance(settings, dict) or "ExperimentConfiguration" not in settings:
            continue

        # Files whose runs trained with a value the code no longer trains, reported and left as they are
        path_link = format_path_link(path=configuration_path, label=str(configuration_path))
        flattened_settings = flatten_settings(settings=settings)
        record_only_values = [f"{key_path} {flattened_settings[key_path]!r} ({reason})"
                              for key_path, reason in RECORD_ONLY_SETTINGS.items()
                              if key_path in flattened_settings and
                              flattened_settings[key_path] != FIXED_VALUES[key_path]]
        if record_only_values:
            print_color(color="red", output=f"Not converted, its runs staying as records: {path_link}: "
                                            f"{', '.join(record_only_values)}")
            continue

        # Conversion of the text, checked setting by setting, the files without fixed settings nor missing sections
        # or settings left as they are
        task = settings["ExperimentConfiguration"]["task"]
        text_without_fixed_settings = remove_fixed_settings(configuration_text=original_text, task=task)
        renamed_text = rename_settings(configuration_text=text_without_fixed_settings, task=task)
        text_with_sections = add_required_sections(configuration_text=renamed_text, task=task,
                                                   configuration_path=configuration_path)
        converted_text = add_required_settings(configuration_text=text_with_sections, task=task,
                                               configuration_path=configuration_path)
        if converted_text == original_text:
            continue
        changed_values = check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                          configuration_path=configuration_path)
        converted_paths.append(configuration_path)
        removed_lines = len(original_text.split("\n")) - len(text_without_fixed_settings.split("\n"))
        added_lines = len(converted_text.split("\n")) - len(text_without_fixed_settings.split("\n"))
        renamed_settings = [key_path for key_path, renamed_setting in RENAMED_SETTINGS.items()
                            if task in renamed_setting["tasks"] and key_path in flattened_settings]
        line_counts = f"{removed_lines} lines removed, {added_lines} added, {len(renamed_settings)} settings renamed"

        # The values the runs of the file change to once converted, listed before anything is written
        if changed_values:
            print_color(color="yellow", output=f"Values changed by the conversion of {path_link}: "
                                               f"{', '.join(changed_values)}")
        if not write_files:
            print_color(color="blue", output=f"To convert ({line_counts}): {path_link}")
            continue

        # Backup of the original, never overwritten, then the converted file
        backup_path = get_backup_path(configuration_path=configuration_path, backup_suffix=backup_suffix,
                                      configuration_folder=configuration_file["configuration_folder"])
        if backup_path.exists():
            raise FileExistsError(f"The backup {format_path_link(path=backup_path)} of "
                                  f"{format_path_link(path=configuration_path)} already exists.")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src=configuration_path, dst=backup_path)
        configuration_path.write_text(data=converted_text, encoding="utf-8")
        print_color(color="green", output=f"Converted ({line_counts}): {path_link}, backup "
                                          f"{format_path_link(path=backup_path)}")
    return converted_paths


def parse_conversion_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of a conversion from the command line of the script, so that the saved runs and the launch
    files of any machine are converted without editing the file: the run roots and the configuration folders have no
    default, the folders of every machine being its own, but at least one folder is needed; the files are only listed
    unless --write_files is given, and the backups take DEFAULT_BACKUP_SUFFIX unless another suffix is given. Launched
    without any argument, the script prints the help of every argument and stops. A '~' left by the shell at the start
    of a path ('--run_roots=~/...') is expanded to the home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The run_roots, configuration_folders, write_files and backup_suffix of
            convert_fixed_settings.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when no folder is
            given or a given folder does not exist.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m scripts.configuration_processing.convert_fixed_settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Converts the configuration files written before a setting was fixed, added or renamed (the\n"
                    "training_configuration.yaml of saved runs and launch files), backing up every converted file.",
        epilog="Example, listing the files to convert in a folder of runs and a launch folder (add --write_files to\n"
               "convert them):\n"
               "  uv run python -m scripts.configuration_processing.convert_fixed_settings \\\n"
               "    --run_roots ~/Data/<runs> --configuration_folders vision_training/configurations/object_detection")
    # The folders searched, at least one of them
    parser.add_argument("--run_roots", type=Path, nargs="+", default=[], metavar="PATH",
                        help="Folders searched for run folders, whose training_configuration.yaml is converted, the "
                             "copies of the MLflow artifacts folder being left as logged (default: none).")
    parser.add_argument("--configuration_folders", type=Path, nargs="+", default=[], metavar="PATH",
                        help="Folders of launch files, converted with their subfolders, their backups going to "
                             f"{PREVIOUS_CONFIGURATIONS_FOLDER_NAME} (default: none).")
    # Whether the files are written, and the name of their backups
    parser.add_argument("--write_files", action="store_true",
                        help="Convert the files after copying each to its backup (default: only list the files to "
                             "convert).")
    parser.add_argument("--backup_suffix", default=DEFAULT_BACKUP_SUFFIX,
                        help="Suffix of the backups, one per conversion so that the backups of an earlier one are "
                             "never overwritten (default: %(default)s).")

    # Without any argument, the script describes its arguments instead of searching no folder
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    parsed_arguments.run_roots = [run_root.expanduser() for run_root in parsed_arguments.run_roots]
    parsed_arguments.configuration_folders = [configuration_folder.expanduser()
                                              for configuration_folder in parsed_arguments.configuration_folders]

    # A conversion without any folder or with a mistyped one would list no file instead of failing
    searched_folders = parsed_arguments.run_roots + parsed_arguments.configuration_folders
    if not searched_folders:
        parser.error(message="give at least one folder with --run_roots or --configuration_folders")
    missing_folders = [str(folder) for folder in searched_folders if not folder.is_dir()]
    if missing_folders:
        parser.error(message=f"no folder {', '.join(missing_folders)}")
    return parsed_arguments


if __name__ == "__main__":
    # Run roots, configuration folders, writing and backup suffix read from the command line, the help printed when
    # none is given
    conversion_arguments = parse_conversion_arguments(arguments=sys.argv[1:])

    # List the files to convert, or convert them with their backups
    files_to_convert = convert_fixed_settings(run_roots=conversion_arguments.run_roots,
                                              configuration_folders=conversion_arguments.configuration_folders,
                                              write_files=conversion_arguments.write_files,
                                              backup_suffix=conversion_arguments.backup_suffix)
    print_color(color="green", output=f"{len(files_to_convert)} Files "
                                      f"{'Converted' if conversion_arguments.write_files else 'To Convert'}")
