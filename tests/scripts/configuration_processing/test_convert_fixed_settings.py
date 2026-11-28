"""
Tests of scripts/configuration_processing/convert_fixed_settings.py, the conversion of the configuration files written
before the fixed settings and the added sections and settings: the values changed by a conversion, the neck section
added to the detection files, the box loss added to their detector section, the box loss of their region proposal
section removed (a CIoU one leaving the file as it is), the turned off gradient clipping norm removed (a clipping file
left as it is), the final NMS settings added to their detector section, the proposal selection added to their region
proposal section, the anchors centred on the cell corners removed (a file centring them on the cell centres left as
it is), the instance index maps (one mask per object) added to the data section of the instance segmentation files,
the finer inputs of the mask heads (none) added to their mask heads section, the region proposal and detection heads
shared by the levels (one head per level) added to their sections, the speed benchmark diagnosis mode (off)
added to the diagnosis modes of every task (a partial draft without them left as it is), the training target viewer
(off) added to the diagnosis modes of the detection tasks only, the class layer flag added to
the classification heads and to the detection, mask and keypoint heads (True for heads transferred from another run,
False otherwise), the class layer flag of the segmentation decoder renamed, and the refusals of a conversion changing
another setting or the backbone. The script prints its help when launched without arguments, reads every setting of
convert_fixed_settings from its command line, only listing the files and taking DEFAULT_BACKUP_SUFFIX unless told
otherwise, refuses a command line without any folder or with a missing one, and converts the folders its arguments
name.
"""

import inspect
import pytest
import runpy
import sys

from pathlib import Path
from typing import List

from scripts.configuration_processing import convert_fixed_settings as conversion_script
from scripts.configuration_processing.convert_fixed_settings import (DEFAULT_BACKUP_SUFFIX, DETECTOR_SHARED_LINES,
                                                                      NECK_PRETRAINED_LINES, NECK_SECTION_TEXT,
                                                                      REGION_PROPOSAL_SHARED_LINES,
                                                                      add_required_sections, add_required_settings,
                                                                      check_conversion, convert_fixed_settings,
                                                                      parse_conversion_arguments,
                                                                      remove_fixed_settings, rename_settings)
from vision_training.data.loaders import NUMBER_LOADER_WORKERS
from vision_training.evaluation.classification.classification_metrics import TOP_K_ACCURACY_CLASSES

# Every argument of the conversion script, which its help must describe
CONVERSION_SCRIPT_ARGUMENTS = ["--run_roots", "--configuration_folders", "--write_files", "--backup_suffix"]

# Configuration of a task given by its placeholder, written before the fixed settings, with their comments
CONFIGURATION_TEXT = """ExperimentConfiguration:
  task: {task}
  # Resumes the run from its last checkpoint
  resume_training: True
  evaluation:
    # Classes counted by the top-k accuracy
    top_k: 2
    target_precision: 0.99
Data:
  # Loader settings
  loading:
    # Worker processes of every loader
    number_workers: 8
    prefetch_batches: 2
  data_augmentation:
    activated: True
    independent_frames: False
    shuffle_frames:
      activated: False
      probability: 0.0
Backbone:
  modules:
    inception: False

    # Block from which inception modules replace the plain convolutions
    inception_starting_block: 3

    squeeze_and_excite: True
"""

# The configuration of an image task once converted, the block removed between two blank lines leaving a single one
CONVERTED_CLASSIFICATION_TEXT = """ExperimentConfiguration:
  task: classification
  evaluation:
    target_precision: 0.99
Data:
  data_augmentation:
    activated: True
Backbone:
  modules:
    inception: False

    squeeze_and_excite: True
"""


def test_conversion_lists_the_changed_values_and_refuses_other_changes(tmp_path: Path) -> None:
    """
    Verifies the check of a conversion: the text without the fixed settings is accepted with the list of the values
    its runs change to, a converted text whose other settings changed is refused, and so is a backbone whose inception
    modules start at another block than the fixed one, whose layers the code no longer builds.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    original_text = CONFIGURATION_TEXT.format(task="classification")
    changed_values = check_conversion(original_text=original_text, converted_text=CONVERTED_CLASSIFICATION_TEXT,
                                      task="classification", configuration_path=tmp_path / "configuration.yaml")
    assert changed_values == [f"ExperimentConfiguration.evaluation.top_k 2 -> {TOP_K_ACCURACY_CLASSES}",
                              f"Data.loading.number_workers 8 -> {NUMBER_LOADER_WORKERS}"]
    with pytest.raises(expected_exception=ValueError, match="ExperimentConfiguration.evaluation.target_precision"):
        check_conversion(original_text=original_text, task="classification",
                         converted_text=CONVERTED_CLASSIFICATION_TEXT.replace("0.99", "0.95"),
                         configuration_path=tmp_path / "configuration.yaml")

    # Inception modules from block 2, refused, while a starting block without inception changes no layer
    inception_text = original_text.replace("inception: False", "inception: True").replace(
        "inception_starting_block: 3", "inception_starting_block: 2")
    with pytest.raises(expected_exception=ValueError, match="starts its inception modules at block 2"):
        check_conversion(original_text=inception_text, task="classification",
                         converted_text=remove_fixed_settings(configuration_text=inception_text, task="classification"),
                         configuration_path=tmp_path / "configuration.yaml")
    unused_block_text = original_text.replace("inception_starting_block: 3", "inception_starting_block: 2")
    assert "Backbone.modules.inception_starting_block 2 -> 3" in check_conversion(
        original_text=unused_block_text, task="classification", configuration_path=tmp_path / "configuration.yaml",
        converted_text=remove_fixed_settings(configuration_text=unused_block_text, task="classification"))


# Detection sections written before the head settings were fixed, with a strict fallback assignment turned on
DETECTION_CONFIGURATION_TEXT = """ExperimentConfiguration:
  task: object_detection
Data:
  image_settings:
    size: 512
RegionProposal:
  normalization:
    feature_map_normalization: "batch"
    # Momentum of the batch normalization running statistics
    batch_normalization_momentum: 0.1
  # Assigns the best-overlapping anchors as foreground even below the minimum threshold.
  strict_fallback_assignment: True
  number_training_positives: 128
  total_training_samples: 256
  # Smooth L1 beta of the box regression loss.
  localisation_loss_beta: 1/9
Detector:
  normalization:
    feature_map_normalization: "batch"
    batch_normalization_momentum: 0.1
  nms_iou_threshold: 0.5
  total_training_samples: 128
  localisation_loss_beta: 1/9
"""

# The detection sections once converted
CONVERTED_DETECTION_TEXT = """ExperimentConfiguration:
  task: object_detection
Data:
  image_settings:
    size: 512
RegionProposal:
  normalization:
    feature_map_normalization: "batch"
  number_training_positives: 128
  total_training_samples: 256
Detector:
  normalization:
    feature_map_normalization: "batch"
  nms_iou_threshold: 0.5
  total_training_samples: 128
"""


def test_conversion_removes_the_fixed_head_settings(tmp_path: Path) -> None:
    """
    Verifies that the conversion removes the batch normalization momentums of the detection heads, the strict fallback
    assignment and the localisation loss betas with their comments, the check of the whole conversion listing the
    strict assignment turned on as a value its runs change to, the momentums of 0.1 and the betas of 1/9 being those
    of the code.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    converted_text = remove_fixed_settings(configuration_text=DETECTION_CONFIGURATION_TEXT, task="object_detection")
    assert converted_text == CONVERTED_DETECTION_TEXT
    converted_text = add_required_sections(configuration_text=converted_text, task="object_detection",
                                           configuration_path=tmp_path / "configuration.yaml")
    converted_text = add_required_settings(configuration_text=converted_text, task="object_detection",
                                           configuration_path=tmp_path / "configuration.yaml")
    changed_values = check_conversion(original_text=DETECTION_CONFIGURATION_TEXT, converted_text=converted_text,
                                      task="object_detection", configuration_path=tmp_path / "configuration.yaml")
    assert changed_values == ["RegionProposal.strict_fallback_assignment True -> False"]


# The converted detection sections with the neck section added before the region proposal, and its comment
CONVERTED_DETECTION_TEXT_WITH_NECK = CONVERTED_DETECTION_TEXT.replace(
    "RegionProposal:\n", f"{NECK_SECTION_TEXT}\n# Region proposal heads\nRegionProposal:\n")


def test_conversion_adds_the_neck_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the 'Neck' section without neck before the region proposal section and its
    comment in the detection files written before the neck, and only there: a classification file and a detection file
    holding a neck are left as they are, a detection file without region proposal section is refused, and the check
    refuses a converted file lacking the section or holding other values than those reproducing its runs.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    original_text = CONVERTED_DETECTION_TEXT.replace("RegionProposal:\n", "# Region proposal heads\nRegionProposal:\n")
    converted_text = add_required_sections(configuration_text=original_text, task="object_detection",
                                           configuration_path=configuration_path)
    assert converted_text == CONVERTED_DETECTION_TEXT_WITH_NECK
    converted_text = add_required_settings(configuration_text=converted_text, task="object_detection",
                                           configuration_path=configuration_path)
    assert check_conversion(original_text=original_text, converted_text=converted_text, task="object_detection",
                            configuration_path=configuration_path) == []

    # Files of another task or already holding the section are left as they are
    classification_text = CONVERTED_CLASSIFICATION_TEXT
    assert add_required_sections(configuration_text=classification_text, task="classification",
                                 configuration_path=configuration_path) == classification_text
    assert add_required_sections(configuration_text=converted_text, task="object_detection",
                                 configuration_path=configuration_path) == converted_text

    # A detection file without region proposal section, and conversions without the section or with another neck
    with pytest.raises(expected_exception=ValueError, match="has no 'RegionProposal' section"):
        add_required_sections(configuration_text="ExperimentConfiguration:\n  task: object_detection\n",
                              task="object_detection", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError, match="Neck.type"):
        check_conversion(original_text=original_text, task="object_detection", configuration_path=configuration_path,
                         converted_text=add_required_settings(configuration_text=original_text, task="object_detection",
                                                              configuration_path=configuration_path))
    with pytest.raises(expected_exception=ValueError, match="Neck.type"):
        check_conversion(original_text=original_text, task="object_detection", configuration_path=configuration_path,
                         converted_text=converted_text.replace('type: "none"', 'type: "feature_pyramid"'))


# Lines the conversion of 2026-11-08 wrote after the sampling budget of the region proposal section, removed since the
# region proposal network always uses the smooth L1, and the lines it writes after that of the detector section
REGION_PROPOSAL_BOX_LOSS_TEXT = (
    '  # Box loss of the positive anchors: "smooth_l1" on the offsets, or "complete_iou" (1 - CIoU of the\n'
    '  # decoded boxes, https://arxiv.org/abs/2005.03572).\n'
    '  box_loss: "smooth_l1"\n')
DETECTOR_BOX_LOSS_TEXT = REGION_PROPOSAL_BOX_LOSS_TEXT.replace("positive anchors", "positive proposals")

# The converted detection sections with the neck, once the box loss is added after the sampling budget of the
# detector, before the final NMS settings
CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS = CONVERTED_DETECTION_TEXT_WITH_NECK.replace(
    "  total_training_samples: 128\n", f"  total_training_samples: 128\n{DETECTOR_BOX_LOSS_TEXT}")

# Lines the conversion writes after the IoU threshold of the final NMS of the detector section
DETECTOR_NMS_TEXT = (
    '  # Criterion of the final NMS: "iou", or "distance_iou" (the IoU minus the normalized distance of the centres,\n'
    '  # https://arxiv.org/abs/1911.08287), compared with nms_iou_threshold.\n'
    '  nms_criterion: "iou"\n'
    '  # Exponent of the distance term of "distance_iou": 1.0 as in the paper, 0.6 in YOLOv4 (more boxes kept).\n'
    '  nms_distance_exponent: 1.0\n')

# The converted detection sections with the neck and the final NMS settings, without and with the box loss, the
# latter holding every added setting
CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS = CONVERTED_DETECTION_TEXT_WITH_NECK.replace(
    "  nms_iou_threshold: 0.5\n", f"  nms_iou_threshold: 0.5\n{DETECTOR_NMS_TEXT}")
CONVERTED_DETECTION_TEXT_WITH_NMS = CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS.replace(
    "  nms_iou_threshold: 0.5\n", f"  nms_iou_threshold: 0.5\n{DETECTOR_NMS_TEXT}")

# Lines the conversion writes after the normalization of the region proposal section, the selection of the proposals
# over all the levels at once, and the region proposal section the text of every file starts with
REGION_PROPOSAL_SELECTION_TEXT = (
    '  # "joint" (one top-k and NMS over all the levels) or "per_level" (top-k and NMS of every level, then the best\n'
    '  # over all the levels, as torchvision, empty proposals dropped), which makes pre_nms_proposals a count per '
    'level.\n'
    '  proposal_selection: "joint"\n')
REGION_PROPOSAL_START_TEXT = 'RegionProposal:\n  normalization:\n    feature_map_normalization: "batch"\n'


def add_proposal_selection_text(configuration_text: str) -> str:
    """
    Writes the lines of the proposal selection after the normalization of the region proposal section of a detection
    text, as the conversion does, for the texts of the files converted after 2026-11-11.

    Args:
        configuration_text (str): The detection text, holding the region proposal section of the texts above.

    Returns:
        str: The text with the proposal selection.
    """
    return configuration_text.replace(REGION_PROPOSAL_START_TEXT,
                                      f"{REGION_PROPOSAL_START_TEXT}{REGION_PROPOSAL_SELECTION_TEXT}")


# Lines the conversion writes after the proposal selection of the region proposal section and after the box loss of
# the detector section in the files written before the sharing of these heads, whose heads are one per level
REGION_PROPOSAL_SHARED_TEXT = "".join(f"  {line}\n" for line in REGION_PROPOSAL_SHARED_LINES)
DETECTOR_SHARED_TEXT = "".join(f"  {line}\n" for line in DETECTOR_SHARED_LINES)


def add_shared_heads_text(configuration_text: str) -> str:
    """
    Writes the lines of the sharing of the region proposal heads after the proposal selection of a detection text, and
    those of the sharing of the detection heads after its box loss, as the conversion does, for the texts of the files
    converted after 2026-11-28.

    Args:
        configuration_text (str): The detection text, holding the proposal selection and the box loss of the texts
            above.

    Returns:
        str: The text with the sharing of both heads.
    """
    return configuration_text.replace(REGION_PROPOSAL_SELECTION_TEXT,
                                      f"{REGION_PROPOSAL_SELECTION_TEXT}{REGION_PROPOSAL_SHARED_TEXT}").replace(
        DETECTOR_BOX_LOSS_TEXT, f"{DETECTOR_BOX_LOSS_TEXT}{DETECTOR_SHARED_TEXT}")


# The converted detection sections holding every added setting: the neck, the box loss, the final NMS settings, the
# proposal selection and the sharing of the region proposal and detection heads
CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS = add_shared_heads_text(configuration_text=add_proposal_selection_text(
    configuration_text=CONVERTED_DETECTION_TEXT_WITH_NMS))

# Data section the detection texts start with, and the lines the conversion writes after its image settings in the
# instance segmentation files, which loaded one mask per object
DATA_START_TEXT = "Data:\n  image_settings:\n    size: 512\n"
INSTANCE_INDEX_MAPS_TEXT = (
    '  # One uint8 map numbering the objects of an image instead of one mask per object (faster loader, expanded on '
    'the\n'
    '  # GPU); needs run length encodings whose masks never overlap (synthetic datasets), False for COCO (polygons).\n'
    '  instance_index_maps: False\n')

# Mask heads section ending the instance segmentation texts, and the lines the conversion writes after its upsampling
# in the files written before the finer inputs of the mask heads, whose heads read the map of their level alone
MASK_HEAD_UPSAMPLING_TEXT = '  upsampling: "transposed"\n'
MASK_HEAD_TEXT = (f'MaskHead:\n  mask_resolution: 28\n{MASK_HEAD_UPSAMPLING_TEXT}  normalization:\n'
                  f'    feature_map_normalization: "batch"\n')
MASK_HEAD_FINE_INPUTS_TEXT = (
    '  # Backbone blocks (before their pooling, 0 the normalized input image) pooled at the pool size of every level '
    'and\n'
    '  # concatenated to its pooled features (U-Net skips), e.g. [4] (stride 8); [] for none.\n'
    '  skip_blocks_at_pool_size: []\n'
    '  # Image branch: [layers, channels] of plain 3x3 convolutions on the box crop of the normalized input image at '
    'M,\n'
    '  # shared by the levels and concatenated after the doublings, e.g. [5, 32]; [0, 0] for none.\n'
    '  image_branch_convolutions: [0, 0]\n'
    '  # Backbone blocks (before their pooling) pooled at M and concatenated after the doublings, e.g. [3] (stride '
    '4);\n'
    '  # [] for none.\n'
    '  skip_blocks_at_mask_resolution: []\n'
    '  # [layers, channels] of plain 3x3 convolutions at M before the predictor, mixing the concatenated features, '
    'e.g.\n'
    '  # [1, 64]; [0, 0] for none, the predictor following the doublings as in Mask R-CNN.\n'
    '  fusion_convolutions: [0, 0]\n')

# Lines the conversion writes after the fusion convolutions of the mask heads section in the files written before the
# sharing of the mask head, whose mask heads are one per level
MASK_HEAD_SHARED_TEXT = (
    '  # True: one mask head read by every level (Mask R-CNN), trained on all the positives; needs the same mask_head\n'
    '  # on every level and equal level channels (a neck). False: one mask head per level.\n'
    '  shared: False\n')


def write_original_task_text(configuration_text: str, task: str) -> str:
    """
    Writes a detection text for a task as the files written before the instance index maps hold it: its task name,
    and for the instance segmentation its mask heads section, without their finer inputs.

    Args:
        configuration_text (str): The detection text.
        task (str): "object_detection" or "instance_segmentation".

    Returns:
        str: The text of the task.
    """
    task_text = configuration_text.replace("task: object_detection", f"task: {task}")
    if task == "instance_segmentation":
        task_text += MASK_HEAD_TEXT
    return task_text


def write_task_text(configuration_text: str, task: str) -> str:
    """
    Writes a detection text for a task: its task name, and for the instance segmentation the instance index maps after
    the image settings and its mask heads section with their finer inputs after the upsampling, then their sharing
    after the fusion convolutions, as the conversion writes them, for the instance segmentation texts converted after
    2026-11-20.

    Args:
        configuration_text (str): The detection text, holding the data section of the texts above.
        task (str): "object_detection" or "instance_segmentation".

    Returns:
        str: The text of the task.
    """
    task_text = configuration_text.replace("task: object_detection", f"task: {task}")
    if task == "instance_segmentation":
        task_text = task_text.replace(DATA_START_TEXT, f"{DATA_START_TEXT}{INSTANCE_INDEX_MAPS_TEXT}")
        task_text += MASK_HEAD_TEXT.replace(
            MASK_HEAD_UPSAMPLING_TEXT, f"{MASK_HEAD_UPSAMPLING_TEXT}{MASK_HEAD_FINE_INPUTS_TEXT}{MASK_HEAD_SHARED_TEXT}")
    return task_text


# The detection sections of the files converted between 2026-11-08 and 2026-11-10, holding the box loss of the region
# proposal section as well
DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(
    "  total_training_samples: 256\n", f"  total_training_samples: 256\n{REGION_PROPOSAL_BOX_LOSS_TEXT}")


def test_conversion_adds_the_box_loss_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the smooth L1 box loss of the runs written before the CIoU loss, with its
    comment lines, right after the sampling budget of the detector section of the detection and instance segmentation
    files holding the final NMS settings, and only there, the region proposal section getting none: a classification
    file and a file holding the box loss are left as they are, a file without sampling budget is refused, and the check
    refuses a converted file lacking the box loss or holding the CIoU loss, which would change its runs.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        original_text = write_original_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS,
                                                 task=task)
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                                 task=task)
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # Files of another task or already holding the box loss are left as they are
    assert add_required_settings(configuration_text=CONVERTED_CLASSIFICATION_TEXT, task="classification",
                                 configuration_path=configuration_path) == CONVERTED_CLASSIFICATION_TEXT
    assert add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                 task="object_detection", configuration_path=configuration_path) == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS

    # A detector section without sampling budget, and conversions without the box loss or with the CIoU loss
    with pytest.raises(expected_exception=ValueError, match="has no 'Detector.total_training_samples' setting"):
        add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS.replace(
            "  total_training_samples: 128\n", ""), task="object_detection", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: Detector\.box_loss\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS, task="object_detection",
                         converted_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(
                             DETECTOR_BOX_LOSS_TEXT, ""), configuration_path=configuration_path)
    complete_iou_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(
        DETECTOR_BOX_LOSS_TEXT, DETECTOR_BOX_LOSS_TEXT.replace('box_loss: "smooth_l1"', 'box_loss: "complete_iou"'))
    with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: Detector\.box_loss\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NECK_AND_NMS, converted_text=complete_iou_text,
                         task="object_detection", configuration_path=configuration_path)


def test_conversion_removes_the_box_loss_of_the_region_proposal_section(tmp_path: Path,
                                                                        capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the conversion removes the smooth L1 box loss of the region proposal section, with its comment lines,
    from the detection and instance segmentation files written between the CIoU loss and its removal from the region
    proposal network, the detector keeping its box loss, and that convert_fixed_settings converts such a launch file
    with its backup while it reports and leaves untouched a file whose region proposal section holds the CIoU loss, its
    runs staying as records, without stopping on it.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        original_text = write_task_text(configuration_text=DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS, task=task)
        converted_text = remove_fixed_settings(configuration_text=original_text, task=task)
        assert converted_text == write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                                 task=task)
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # A launch folder holding a file of each region proposal box loss, the CIoU one listed first
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    complete_iou_text = DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS.replace(
        REGION_PROPOSAL_BOX_LOSS_TEXT, REGION_PROPOSAL_BOX_LOSS_TEXT.replace('"smooth_l1"', '"complete_iou"'))
    (configuration_folder / "run-1.yaml").write_text(data=complete_iou_text, encoding="utf-8")
    (configuration_folder / "run-2.yaml").write_text(data=DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS,
                                                     encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_region_proposal_box_loss")

    # The smooth L1 file converted with its backup, the CIoU file reported and left as it was
    assert converted_paths == [configuration_folder / "run-2.yaml"]
    assert (configuration_folder / "run-2.yaml").read_text(encoding="utf-8") == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    assert (configuration_folder / "previous_configurations" / "run-2_before_region_proposal_box_loss.yaml").read_text(
        encoding="utf-8") == DETECTION_TEXT_WITH_REGION_PROPOSAL_BOX_LOSS
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == complete_iou_text
    backup_folder = configuration_folder / "previous_configurations"
    assert not (backup_folder / "run-1_before_region_proposal_box_loss.yaml").exists()
    printed_output = capsys.readouterr().out
    assert "Not converted, its runs staying as records" in printed_output
    assert "RegionProposal.box_loss 'complete_iou'" in printed_output


# Configuration of a task given by its placeholder, written before the removal of the gradient clipping, its turned
# off norm and comment line between the cosine minimum and the moving average of the weights
CLIPPING_CONFIGURATION_TEXT = """ExperimentConfiguration:
  task: {task}
Optimization:
  # Final cosine learning rate, as a fraction of the learning rate (cosine only).
  minimum_learning_rate_ratio: 0.01
  # Maximum global gradient norm, clipped before every step, null to disable.
  gradient_clipping_norm: null
  # Exponential moving average of the weights.
  exponential_moving_average:
    activated: False
    decay: 0.9998
"""

# The same configuration once converted, the moving average directly after the cosine minimum
CONVERTED_CLIPPING_CONFIGURATION_TEXT = """ExperimentConfiguration:
  task: {task}
Optimization:
  # Final cosine learning rate, as a fraction of the learning rate (cosine only).
  minimum_learning_rate_ratio: 0.01
  # Exponential moving average of the weights.
  exponential_moving_average:
    activated: False
    decay: 0.9998
"""


def test_conversion_removes_the_gradient_clipping_norm(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the conversion removes the turned off gradient clipping norm, with its comment line, from the files of
    the tasks written before its removal, and that convert_fixed_settings converts such a launch file with its backup
    while it reports and leaves untouched a file clipping its gradients, its runs staying as records, without stopping
    on it.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["classification", "sequence_labelling", "self_supervised"]:
        original_text = CLIPPING_CONFIGURATION_TEXT.format(task=task)
        converted_text = remove_fixed_settings(configuration_text=original_text, task=task)
        assert converted_text == CONVERTED_CLIPPING_CONFIGURATION_TEXT.format(task=task)
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # A launch folder holding a file clipping its gradients, listed first, and a file without clipping
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    original_text = CLIPPING_CONFIGURATION_TEXT.format(task="classification")
    clipping_text = original_text.replace("gradient_clipping_norm: null", "gradient_clipping_norm: 1.0")
    (configuration_folder / "run-1.yaml").write_text(data=clipping_text, encoding="utf-8")
    (configuration_folder / "run-2.yaml").write_text(data=original_text, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_gradient_clipping_norm")

    # The file without clipping converted with its backup, the clipping file reported and left as it was
    backup_folder = configuration_folder / "previous_configurations"
    assert converted_paths == [configuration_folder / "run-2.yaml"]
    assert (configuration_folder / "run-2.yaml").read_text(encoding="utf-8") == \
        CONVERTED_CLIPPING_CONFIGURATION_TEXT.format(task="classification")
    assert (backup_folder / "run-2_before_gradient_clipping_norm.yaml").read_text(encoding="utf-8") == original_text
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == clipping_text
    assert not (backup_folder / "run-1_before_gradient_clipping_norm.yaml").exists()
    printed_output = capsys.readouterr().out
    assert "Not converted, its runs staying as records" in printed_output
    assert "Optimization.gradient_clipping_norm 1.0" in printed_output


def test_conversion_adds_the_final_nms_settings_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the greedy final NMS of the runs written before DIoU-NMS, the criterion "iou"
    and the exponent 1.0 with their comment lines, right after the IoU threshold of the final NMS of the detector
    section of the detection and instance segmentation files converted for the box loss, and only there: a
    classification file and a file holding the settings are left as they are, a detector section without IoU threshold
    is refused, and the check refuses a converted file lacking the settings or holding DIoU-NMS, which would change the
    evaluations of its runs.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        original_text = write_original_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS, task=task)
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                                 task=task)
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # Files of another task or already holding the settings are left as they are
    assert add_required_settings(configuration_text=CONVERTED_CLASSIFICATION_TEXT, task="classification",
                                 configuration_path=configuration_path) == CONVERTED_CLASSIFICATION_TEXT
    instance_segmentation_text = write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                                 task="instance_segmentation")
    assert add_required_settings(configuration_text=instance_segmentation_text, task="instance_segmentation",
                                 configuration_path=configuration_path) == instance_segmentation_text

    # A detector section without IoU threshold, and conversions without the settings or with DIoU-NMS
    with pytest.raises(expected_exception=ValueError, match="has no 'Detector.nms_iou_threshold' setting"):
        add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS.replace(
            "  nms_iou_threshold: 0.5\n", ""), task="object_detection", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: Detector\.nms_criterion, Detector\.nms_distance_exponent\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS, task="object_detection",
                         converted_text=add_shared_heads_text(configuration_text=add_proposal_selection_text(
                             configuration_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS)),
                         configuration_path=configuration_path)
    distance_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace('nms_criterion: "iou"',
                                                                         'nms_criterion: "distance_iou"')
    with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: Detector\.nms_criterion\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSS, converted_text=distance_text,
                         task="object_detection", configuration_path=configuration_path)


def test_conversion_adds_the_proposal_selection_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the selection of the proposals over all the levels at once of the runs written
    before the selection level by level, "joint" with its comment lines, right after the normalization of the region
    proposal section of the detection and instance segmentation files, and only there: a classification file and a
    file holding the selection are left as they are, a region proposal section without normalization is refused, the
    check refuses a converted file lacking the selection or selecting level by level, which would change its runs, and
    convert_fixed_settings converts a launch file with its backup.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        original_text = write_original_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_NMS, task=task)
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                                 task=task)
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # Files of another task or already holding the selection are left as they are
    assert add_required_settings(configuration_text=CONVERTED_CLASSIFICATION_TEXT, task="classification",
                                 configuration_path=configuration_path) == CONVERTED_CLASSIFICATION_TEXT
    per_level_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace('proposal_selection: "joint"',
                                                                          'proposal_selection: "per_level"')
    assert add_required_settings(configuration_text=per_level_text, task="object_detection",
                                 configuration_path=configuration_path) == per_level_text

    # A region proposal section without normalization, and conversions without the selection or level by level
    with pytest.raises(expected_exception=ValueError, match="has no 'RegionProposal.normalization' setting"):
        add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_NMS.replace(
            REGION_PROPOSAL_START_TEXT, "RegionProposal:\n"), task="object_detection",
            configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: RegionProposal\.proposal_selection\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NMS, task="object_detection",
                         converted_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(
                             REGION_PROPOSAL_SELECTION_TEXT, ""), configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: RegionProposal\.proposal_selection\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NMS, converted_text=per_level_text,
                         task="object_detection", configuration_path=configuration_path)

    # A launch file converted with its backup, the original kept in it
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    (configuration_folder / "run-1.yaml").write_text(data=CONVERTED_DETECTION_TEXT_WITH_NMS, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_proposal_selection")
    assert converted_paths == [configuration_folder / "run-1.yaml"]
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    assert (configuration_folder / "previous_configurations" / "run-1_before_proposal_selection.yaml").read_text(
        encoding="utf-8") == CONVERTED_DETECTION_TEXT_WITH_NMS


# Lines the conversion of 2026-11-12 wrote after the proposal selection of the region proposal section, removed since
# the anchors are always centred on the cell corners
REGION_PROPOSAL_ANCHOR_CENTRES_TEXT = (
    '  # "cell_corners" (anchor centres at column x stride, as torchvision) or "cell_centres" ((column + 0.5) x '
    'stride,\n'
    '  # where RoIAlign and the convolutions place the cells, the offset 0.5 of Detectron2).\n'
    '  anchor_centres: "cell_corners"\n')

# The detection sections of the files converted between 2026-11-12 and 2026-11-13, holding the anchor centres after
# the proposal selection
DETECTION_TEXT_WITH_ANCHOR_CENTRES = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(
    REGION_PROPOSAL_SELECTION_TEXT, f"{REGION_PROPOSAL_SELECTION_TEXT}{REGION_PROPOSAL_ANCHOR_CENTRES_TEXT}")


def test_conversion_removes_the_anchor_centres(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the conversion removes the anchors centred on the cell corners, "cell_corners" with its comment lines,
    from the region proposal section of the detection and instance segmentation files written between the anchors
    centred on the cell centres and their removal, whichever the proposal selection, and that convert_fixed_settings
    converts such a launch file with its backup while it reports and leaves untouched a file centring its anchors on
    the cell centres, its runs staying as records, without stopping on it.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        for proposal_selection in ["joint", "per_level"]:
            original_text = write_task_text(configuration_text=DETECTION_TEXT_WITH_ANCHOR_CENTRES,
                                            task=task).replace('proposal_selection: "joint"',
                                                               f'proposal_selection: "{proposal_selection}"')
            converted_text = remove_fixed_settings(configuration_text=original_text, task=task)
            assert converted_text == write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                                     task=task).replace('proposal_selection: "joint"',
                                                                        f'proposal_selection: "{proposal_selection}"')
            assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                    configuration_path=configuration_path) == []

    # A launch folder holding a file of each anchor centring, the cell centres one listed first
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    cell_centres_text = DETECTION_TEXT_WITH_ANCHOR_CENTRES.replace('anchor_centres: "cell_corners"',
                                                                   'anchor_centres: "cell_centres"')
    (configuration_folder / "run-1.yaml").write_text(data=cell_centres_text, encoding="utf-8")
    (configuration_folder / "run-2.yaml").write_text(data=DETECTION_TEXT_WITH_ANCHOR_CENTRES, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_anchor_centres_removal")

    # The cell corners file converted with its backup, the cell centres file reported and left as it was
    backup_folder = configuration_folder / "previous_configurations"
    assert converted_paths == [configuration_folder / "run-2.yaml"]
    assert (configuration_folder / "run-2.yaml").read_text(encoding="utf-8") == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    assert (backup_folder / "run-2_before_anchor_centres_removal.yaml").read_text(encoding="utf-8") == \
        DETECTION_TEXT_WITH_ANCHOR_CENTRES
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == cell_centres_text
    assert not (backup_folder / "run-1_before_anchor_centres_removal.yaml").exists()
    printed_output = capsys.readouterr().out
    assert "Not converted, its runs staying as records" in printed_output
    assert "RegionProposal.anchor_centres 'cell_centres'" in printed_output


def test_conversion_adds_the_instance_index_maps_to_the_instance_segmentation_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the one mask per object of the runs written before the instance index maps,
    'instance_index_maps: False' with its comment lines, right after the image settings of the data section of the
    instance segmentation files, and only there: an object detection file and an instance segmentation file holding
    the maps are left as they are, a data section without image settings is refused, the check refuses a converted
    file lacking the setting or turning the maps on, and convert_fixed_settings converts a launch file with its backup.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    expected_text = write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                    task="instance_segmentation")
    original_text = expected_text.replace(INSTANCE_INDEX_MAPS_TEXT, "")
    converted_text = add_required_settings(configuration_text=original_text, task="instance_segmentation",
                                           configuration_path=configuration_path)
    assert converted_text == expected_text and INSTANCE_INDEX_MAPS_TEXT in converted_text
    assert check_conversion(original_text=original_text, converted_text=converted_text, task="instance_segmentation",
                            configuration_path=configuration_path) == []

    # An object detection file, and an instance segmentation file holding the maps turned on, are left as they are
    assert add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                 task="object_detection", configuration_path=configuration_path) == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    maps_text = expected_text.replace("instance_index_maps: False", "instance_index_maps: True")
    assert add_required_settings(configuration_text=maps_text, task="instance_segmentation",
                                 configuration_path=configuration_path) == maps_text

    # A data section without image settings, and conversions without the setting or turning the maps on
    with pytest.raises(expected_exception=ValueError, match="has no 'Data.image_settings' setting"):
        add_required_settings(configuration_text=original_text.replace(DATA_START_TEXT, "Data:\n  other: 1\n"),
                              task="instance_segmentation", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: Data\.instance_index_maps\.$"):
        check_conversion(original_text=original_text, converted_text=original_text, task="instance_segmentation",
                         configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: Data\.instance_index_maps\.$"):
        check_conversion(original_text=original_text, converted_text=maps_text, task="instance_segmentation",
                         configuration_path=configuration_path)

    # A launch file converted with its backup, the original kept in it
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    (configuration_folder / "run-1.yaml").write_text(data=original_text, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_instance_index_maps")
    assert converted_paths == [configuration_folder / "run-1.yaml"]
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == expected_text
    assert (configuration_folder / "previous_configurations" / "run-1_before_instance_index_maps.yaml").read_text(
        encoding="utf-8") == original_text


def test_conversion_adds_the_mask_head_fine_inputs_to_the_instance_segmentation_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the finer inputs of the mask heads turned off, no backbone skip, no image branch
    and no fusion convolution with their comment lines, right after the upsampling of the mask heads section of the
    instance segmentation files written before them, in the order of their data flow, and only there: an object
    detection file and an instance segmentation file holding an image branch are left as they are, a mask heads section
    without upsampling is refused, the check refuses a converted file lacking the settings or turning one on, and
    convert_fixed_settings converts a launch file with its backup.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    expected_text = write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                    task="instance_segmentation")
    original_text = expected_text.replace(MASK_HEAD_FINE_INPUTS_TEXT, "")
    converted_text = add_required_settings(configuration_text=original_text, task="instance_segmentation",
                                           configuration_path=configuration_path)
    assert converted_text == expected_text and MASK_HEAD_FINE_INPUTS_TEXT in converted_text
    assert check_conversion(original_text=original_text, converted_text=converted_text, task="instance_segmentation",
                            configuration_path=configuration_path) == []

    # An object detection file, and an instance segmentation file holding an image branch, are left as they are
    assert add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                 task="object_detection", configuration_path=configuration_path) == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    image_branch_text = expected_text.replace("image_branch_convolutions: [0, 0]", "image_branch_convolutions: [5, 32]")
    assert add_required_settings(configuration_text=image_branch_text, task="instance_segmentation",
                                 configuration_path=configuration_path) == image_branch_text

    # A mask heads section without upsampling, and conversions without the settings or turning the image branch on
    with pytest.raises(expected_exception=ValueError, match="has no 'MaskHead.upsampling' setting"):
        add_required_settings(configuration_text=original_text.replace(MASK_HEAD_UPSAMPLING_TEXT, ""),
                              task="instance_segmentation", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: MaskHead\.fusion_convolutions, "
                             r"MaskHead\.image_branch_convolutions, MaskHead\.skip_blocks_at_mask_resolution, "
                             r"MaskHead\.skip_blocks_at_pool_size\.$"):
        check_conversion(original_text=original_text, converted_text=original_text, task="instance_segmentation",
                         configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: MaskHead\.image_branch_convolutions\.$"):
        check_conversion(original_text=original_text, converted_text=image_branch_text, task="instance_segmentation",
                         configuration_path=configuration_path)

    # A launch file converted with its backup, the original kept in it
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    (configuration_folder / "run-1.yaml").write_text(data=original_text, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_mask_fine_inputs")
    assert converted_paths == [configuration_folder / "run-1.yaml"]
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == expected_text
    assert (configuration_folder / "previous_configurations" / "run-1_before_mask_fine_inputs.yaml").read_text(
        encoding="utf-8") == original_text


def test_conversion_adds_the_mask_head_sharing_to_the_instance_segmentation_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the one mask head per level of the runs written before the sharing of the mask
    head, 'shared: False' with its comment lines, right after the fusion convolutions of the mask heads section of the
    instance segmentation files, and only there: an object detection file and an instance segmentation file sharing
    its head are left as they are, the check refuses a converted file lacking the setting or sharing the head, and
    convert_fixed_settings converts a launch file with its backup.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    expected_text = write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                    task="instance_segmentation")
    original_text = expected_text.replace(MASK_HEAD_SHARED_TEXT, "")
    converted_text = add_required_settings(configuration_text=original_text, task="instance_segmentation",
                                           configuration_path=configuration_path)
    assert converted_text == expected_text and "  fusion_convolutions: [0, 0]\n" + MASK_HEAD_SHARED_TEXT in \
        converted_text
    assert check_conversion(original_text=original_text, converted_text=converted_text, task="instance_segmentation",
                            configuration_path=configuration_path) == []

    # An object detection file, and an instance segmentation file sharing its head, are left as they are
    assert add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                 task="object_detection", configuration_path=configuration_path) == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    shared_text = expected_text.replace(MASK_HEAD_SHARED_TEXT, MASK_HEAD_SHARED_TEXT.replace("shared: False",
                                                                                         "shared: True"))
    assert add_required_settings(configuration_text=shared_text, task="instance_segmentation",
                                 configuration_path=configuration_path) == shared_text

    # Conversions without the setting or sharing the head are refused
    for refused_text in [original_text, shared_text]:
        with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: MaskHead\.shared\.$"):
            check_conversion(original_text=original_text, converted_text=refused_text, task="instance_segmentation",
                             configuration_path=configuration_path)

    # A launch file converted with its backup, the original kept in it
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    (configuration_folder / "run-1.yaml").write_text(data=original_text, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_shared_mask_head")
    assert converted_paths == [configuration_folder / "run-1.yaml"]
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == expected_text
    assert (configuration_folder / "previous_configurations" / "run-1_before_shared_mask_head.yaml").read_text(
        encoding="utf-8") == original_text


def test_conversion_adds_the_sharing_of_the_region_proposal_and_detection_heads(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the one head per level of the runs written before the sharing of the region
    proposal and detection heads, 'shared: False' with its comment lines, right after the proposal selection of the
    region proposal section and right after the box loss of the detector section of the detection and instance
    segmentation files, and only there: a classification file and a file sharing its heads are left as they are, the
    check refuses a converted file lacking a setting or sharing a head, and convert_fixed_settings converts a launch
    file with a backup of the latest suffix.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        expected_text = write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS, task=task)
        original_text = expected_text.replace(REGION_PROPOSAL_SHARED_TEXT, "").replace(DETECTOR_SHARED_TEXT, "")
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == expected_text, task
        assert REGION_PROPOSAL_SELECTION_TEXT + REGION_PROPOSAL_SHARED_TEXT in converted_text
        assert DETECTOR_BOX_LOSS_TEXT + DETECTOR_SHARED_TEXT in converted_text
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # A classification file, and a detection file sharing its heads, are left as they are
    assert add_required_settings(configuration_text=CONVERTED_CLASSIFICATION_TEXT, task="classification",
                                 configuration_path=configuration_path) == CONVERTED_CLASSIFICATION_TEXT
    shared_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace("shared: False", "shared: True")
    assert add_required_settings(configuration_text=shared_text, task="object_detection",
                                 configuration_path=configuration_path) == shared_text

    # Conversions without a setting or sharing a head are refused
    original_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(REGION_PROPOSAL_SHARED_TEXT, "").replace(
        DETECTOR_SHARED_TEXT, "")
    detector_sharing_missing_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(DETECTOR_SHARED_TEXT, "")
    refused_texts = {r"Detector\.shared": detector_sharing_missing_text,
                     r"Detector\.shared, RegionProposal\.shared": shared_text}
    for refused_settings, refused_text in refused_texts.items():
        with pytest.raises(expected_exception=ValueError, match=rf"than the fixed ones: {refused_settings}\.$"):
            check_conversion(original_text=original_text, converted_text=refused_text, task="object_detection",
                             configuration_path=configuration_path)

    # A launch file converted with its backup, the original kept in it
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    (configuration_folder / "run-1.yaml").write_text(data=original_text, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix=DEFAULT_BACKUP_SUFFIX)
    assert converted_paths == [configuration_folder / "run-1.yaml"]
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    assert (configuration_folder / "previous_configurations" / "run-1_before_shared_detection_heads.yaml").read_text(
        encoding="utf-8") == original_text


# Lines the conversion of 2026-11-21 wrote after the sharing of the mask heads, removed with the mask scoring on
# 2026-11-27: the comment lines, then the section, off, and its settings
MASK_HEAD_MASK_SCORING_TEXT = (
    '  # Mask Scoring R-CNN (arxiv.org/abs/1903.00241): a mask IoU head per mask head predicts the IoU of every mask, '
    'its\n'
    '  # mask score being the class score times it (mask AP ranking, mask_scores export output).\n'
    '  mask_scoring:\n'
    '    # True to score the masks, with mask_iou_loss in Optimization.loss_weights (1.0 in the paper); needs M a\n'
    '    # multiple of the pool sizes.\n'
    '    activated: False\n'
    '    # [layers, channels] of its 3x3 convolutions on the box features and the pooled mask, the last of stride 2;\n'
    '    # [4, 256] in the paper.\n'
    '    convolutions: [4, 64]\n'
    '    # Sizes of its hidden fully connected layers, then one IoU per class; [1024, 1024] in the paper.\n'
    '    fully_connected: [256, 256]\n')


def test_conversion_removes_the_mask_scoring(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the conversion removes the mask scoring section, off, with its settings and comment lines, from the
    mask heads section of the instance segmentation files written between its addition and its removal, and that
    convert_fixed_settings converts such a launch file with its backup while it reports and leaves untouched a file
    scoring its masks, its runs staying as records, without stopping on it.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = tmp_path / "configuration.yaml"
    expected_text = write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                    task="instance_segmentation")
    original_text = expected_text.replace(MASK_HEAD_SHARED_TEXT,
                                          f"{MASK_HEAD_SHARED_TEXT}{MASK_HEAD_MASK_SCORING_TEXT}")
    assert original_text != expected_text and original_text.count("    activated: False\n") == 1
    converted_text = remove_fixed_settings(configuration_text=original_text, task="instance_segmentation")
    assert converted_text == expected_text
    assert check_conversion(original_text=original_text, converted_text=converted_text, task="instance_segmentation",
                            configuration_path=configuration_path) == []

    # A launch folder holding a file scoring its masks, listed first, and a file of unscored masks
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    scored_text = original_text.replace("    activated: False\n", "    activated: True\n")
    (configuration_folder / "run-1.yaml").write_text(data=scored_text, encoding="utf-8")
    (configuration_folder / "run-2.yaml").write_text(data=original_text, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_mask_scoring_removal")

    # The unscored file converted with its backup, the scored file reported and left as it was
    backup_folder = configuration_folder / "previous_configurations"
    assert converted_paths == [configuration_folder / "run-2.yaml"]
    assert (configuration_folder / "run-2.yaml").read_text(encoding="utf-8") == expected_text
    assert (backup_folder / "run-2_before_mask_scoring_removal.yaml").read_text(encoding="utf-8") == original_text
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == scored_text
    assert not (backup_folder / "run-1_before_mask_scoring_removal.yaml").exists()
    printed_output = capsys.readouterr().out
    assert "Not converted, its runs staying as records" in printed_output
    assert "MaskHead.mask_scoring.activated True" in printed_output


# Classification run written before the speed benchmark, its diagnosis modes indented deeper than their section as in
# the configurations of the repository, followed by a blank line and the next section
DIAGNOSIS_MODES_TEXT = """ExperimentConfiguration:
  task: classification
  diagnosis_modes:
      # Computes the receptive field
      compute_receptive_field: True
      # Shows the network input
      view_input_images: False

Optimization:
  optimizer: "adam"
"""
BENCHMARK_SPEED_TEXT = (
    "      # Times the training step and the inference (PyTorch, TorchScript, ONNX Runtime, TensorRT) on one training"
    " batch,\n      # at the configured and largest batch sizes that fit, then exits without training; no run folder is"
    " created.\n      benchmark_speed: False\n")
CONVERTED_DIAGNOSIS_MODES_TEXT = DIAGNOSIS_MODES_TEXT.replace("view_input_images: False\n",
                                                              f"view_input_images: False\n{BENCHMARK_SPEED_TEXT}")


def test_conversion_adds_the_speed_benchmark_to_the_diagnosis_modes(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the speed benchmark diagnosis mode turned off, with its comment lines, right
    after 'view_input_images' at its indentation, for every task, and only in the files holding 'diagnosis_modes': a
    partial draft without the section is left as it is, diagnosis modes without 'view_input_images' are refused, the
    check refuses a converted file lacking the setting or turning it on, and convert_fixed_settings converts a launch
    file with its backup.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"

    # The setting written after the viewer, before the blank line closing the section, for an image and a sequence task
    for task in ["classification", "sequence_comparison"]:
        original_text = DIAGNOSIS_MODES_TEXT.replace("task: classification", f"task: {task}")
        expected_text = CONVERTED_DIAGNOSIS_MODES_TEXT.replace("task: classification", f"task: {task}")
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == expected_text
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # A partial draft without diagnosis modes is left as it is, and the check expects nothing to be added to it
    draft_text = 'ExperimentConfiguration:\n  task: classification\nOptimization:\n  optimizer: "adam"\n'
    assert add_required_settings(configuration_text=draft_text, task="classification",
                                 configuration_path=configuration_path) == draft_text
    assert check_conversion(original_text=draft_text, converted_text=draft_text, task="classification",
                            configuration_path=configuration_path) == []

    # Diagnosis modes without the viewer, and conversions without the setting or turning the benchmark on
    with pytest.raises(expected_exception=ValueError,
                       match="has no 'ExperimentConfiguration.diagnosis_modes.view_input_images' setting"):
        add_required_settings(configuration_text=DIAGNOSIS_MODES_TEXT.replace("      view_input_images: False\n", ""),
                              task="classification", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: ExperimentConfiguration\.diagnosis_modes\.benchmark_speed\.$"):
        check_conversion(original_text=DIAGNOSIS_MODES_TEXT, converted_text=DIAGNOSIS_MODES_TEXT,
                         task="classification", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: ExperimentConfiguration\.diagnosis_modes\.benchmark_speed\.$"):
        check_conversion(original_text=DIAGNOSIS_MODES_TEXT, task="classification",
                         converted_text=CONVERTED_DIAGNOSIS_MODES_TEXT.replace("benchmark_speed: False",
                                                                               "benchmark_speed: True"),
                         configuration_path=configuration_path)

    # A launch file converted with its backup, the original kept in it
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    (configuration_folder / "run-1.yaml").write_text(data=DIAGNOSIS_MODES_TEXT, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix="_before_benchmark_speed")
    assert converted_paths == [configuration_folder / "run-1.yaml"]
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == CONVERTED_DIAGNOSIS_MODES_TEXT
    assert (configuration_folder / "previous_configurations" / "run-1_before_benchmark_speed.yaml").read_text(
        encoding="utf-8") == DIAGNOSIS_MODES_TEXT


# Lines of the training target viewer the conversion writes after the input viewer of the detection runs written
# before it, at the indentation of the diagnosis modes of the repository
VIEW_TRAINING_TARGETS_TEXT = (
    "      # Draws the anchors, proposals and sampled RoIs the losses use on every training image (with the mask targets"
    " in\n      # instance segmentation) with the last checkpoint if any. SPACE: next, ENTER: exit, no training nor run"
    " folder.\n      view_training_targets: False\n")


def test_conversion_adds_the_training_target_viewer_to_the_detection_diagnosis_modes(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the training target viewer diagnosis mode turned off, with its comment lines,
    right after 'view_input_images' and before the speed benchmark, for the object detection and the instance
    segmentation only, the classification file getting the speed benchmark alone, and that the check refuses a
    converted detection file lacking the setting or turning it on.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"

    # The viewer written after the input viewer, before the speed benchmark added in the same conversion, in files
    # already holding the neck section of the detection tasks
    for task in ["object_detection", "instance_segmentation"]:
        original_text = DIAGNOSIS_MODES_TEXT.replace("task: classification", f"task: {task}") + NECK_SECTION_TEXT
        expected_text = original_text.replace("view_input_images: False\n", "view_input_images: False\n" +
                                              VIEW_TRAINING_TARGETS_TEXT + BENCHMARK_SPEED_TEXT)
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == expected_text
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

        # Conversions without the setting, or turning the viewer on, are refused
        for wrong_text in [expected_text.replace(VIEW_TRAINING_TARGETS_TEXT, ""),
                           expected_text.replace("view_training_targets: False", "view_training_targets: True")]:
            with pytest.raises(expected_exception=ValueError, match=r"ExperimentConfiguration\.diagnosis_modes\."
                                                                    r"view_training_targets\.$"):
                check_conversion(original_text=original_text, converted_text=wrong_text, task=task,
                                 configuration_path=configuration_path)

    # The other tasks never get it
    assert "view_training_targets" not in add_required_settings(
        configuration_text=DIAGNOSIS_MODES_TEXT, task="classification", configuration_path=configuration_path)


# Classification head of a task given by its placeholder, written before the class layer flag, trained from random
# weights (experiment folder null) or transferred from another run, and the lines of the flag the conversion adds
CLASSIFICATION_HEAD_TEXT = """ExperimentConfiguration:
  task: {task}
ClassificationHead:
  fully_connected: [128]
  pretrained:
    experiment_folder: {experiment_folder}
    iteration: null
    freeze: False
    learning_rate_multiplier: 1.0
"""
CLASS_LAYER_TEXT = (
    "    # True also loads the class layer: source classes (its data.json) first in order or by name, others random;"
    " needs\n    # experiment_folder. False loads the hidden layers only, the class layer random for any classes (HRCL"
    " token).\n    class_layer: {class_layer}\n")


def write_class_layer_text(configuration_text: str, class_layer: str) -> str:
    """
    Writes the class layer flag into a classification head text as the conversion does, right after 'freeze'.

    Args:
        configuration_text (str): A text of CLASSIFICATION_HEAD_TEXT.
        class_layer (str): "True" or "False".

    Returns:
        str: The text holding the flag.
    """
    return configuration_text.replace("    freeze: False\n",
                                      "    freeze: False\n" + CLASS_LAYER_TEXT.format(class_layer=class_layer))


def test_conversion_adds_the_class_layer_flag_to_the_classification_heads(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the class layer flag of the classification head, with its comment lines, right
    after 'freeze' in its 'pretrained' block, for the classification-like tasks: False for a head trained from random
    weights, True for a head transferred from another run, which loaded its class layer, so that the run names and the
    parameter groups of the saved runs stay; the check refuses a transferred head written False, the files of the
    other tasks are left as they are, and convert_fixed_settings converts a run configuration with its backup.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"

    # The flag of every head, False without source run and True with one
    for task in ["classification", "sequence_labelling", "frame_selection"]:
        for experiment_folder, class_layer in [("null", "False"), ('"/runs/source"', "True")]:
            original_text = CLASSIFICATION_HEAD_TEXT.format(task=task, experiment_folder=experiment_folder)
            converted_text = add_required_settings(configuration_text=original_text, task=task,
                                                   configuration_path=configuration_path)
            assert converted_text == write_class_layer_text(configuration_text=original_text, class_layer=class_layer)
            assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                    configuration_path=configuration_path) == []

    # A transferred head written False would change its runs, and a task without classification head gets nothing
    transferred_text = CLASSIFICATION_HEAD_TEXT.format(task="classification", experiment_folder='"/runs/source"')
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: ClassificationHead\.pretrained\.class_layer\.$"):
        check_conversion(original_text=transferred_text, task="classification", configuration_path=configuration_path,
                         converted_text=write_class_layer_text(configuration_text=transferred_text, class_layer="False"))
    detection_text = CLASSIFICATION_HEAD_TEXT.format(task="object_detection", experiment_folder="null")
    assert add_required_settings(configuration_text=detection_text, task="object_detection",
                                 configuration_path=configuration_path) == detection_text

    # A run configuration converted with its backup next to it
    run_folder = tmp_path / "runs" / "CLS_run"
    run_folder.mkdir(parents=True)
    (run_folder / "training_configuration.yaml").write_text(data=transferred_text, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[tmp_path / "runs"], configuration_folders=[], write_files=True,
                                             backup_suffix="_before_class_layer")
    assert converted_paths == [run_folder / "training_configuration.yaml"]
    assert (run_folder / "training_configuration.yaml").read_text(encoding="utf-8") == write_class_layer_text(
        configuration_text=transferred_text, class_layer="True")
    assert (run_folder / "training_configuration_before_class_layer.yaml").read_text(
        encoding="utf-8") == transferred_text


# Transferable heads of the detection tasks and of the keypoint detection written before their class layer flag, with
# the settings and the neck (with its transfer, last) added before it, the detection and keypoint heads trained from
# random weights or transferred (placeholder), the mask heads transferred, then the texts of the flags the conversion
# adds after 'freeze'
HEADS_TEXT = """ExperimentConfiguration:
  task: {task}
Detector:
  total_training_samples: 256
  box_loss: "smooth_l1"
  shared: False
  nms_iou_threshold: 0.5
  nms_criterion: "iou"
  nms_distance_exponent: 1.0
  pretrained:
    experiment_folder: {experiment_folder}
    iteration: null
    freeze: False
    learning_rate_multiplier: 1.0
MaskHead:
  upsampling: "transposed"
  skip_blocks_at_pool_size: []
  image_branch_convolutions: [0, 0]
  skip_blocks_at_mask_resolution: []
  fusion_convolutions: [0, 0]
  shared: False
  pretrained:
    experiment_folder: "/runs/source"
    iteration: null
    freeze: True
    learning_rate_multiplier: 1.0
KeypointHead:
  pretrained:
    experiment_folder: {experiment_folder}
    iteration: null
    freeze: False
    learning_rate_multiplier: 1.0
Neck:
  type: "none"
  pretrained:
    experiment_folder: null
    iteration: null
    freeze: False
    learning_rate_multiplier: 1.0
"""
DETECTOR_CLASS_LAYERS_TEXT = (
    "    # True also loads the class layers (classifier, box regressor): source classes (its data.json) first in order"
    " or by\n    # name, others random; needs experiment_folder. False leaves them random for any classes (DRCL"
    " token).\n    class_layers: {class_layers}\n")
MASK_HEAD_CLASS_LAYERS_TEXT = (
    "    # True also loads the class layer (mask predictor): source classes (its data.json) first in order or by "
    "name,\n"
    "    # others random; needs experiment_folder. False leaves it random for any classes (MRCL token).\n"
    "    class_layers: True\n")
KEYPOINT_HEAD_CLASS_LAYERS_TEXT = (
    "    # True also loads the heatmap and tag convolutions: same classes and keypoints in the same order (its"
    " data.json);\n    # needs experiment_folder. False leaves them random for any keypoints, as their HH branch inputs"
    " (KHRCL token).\n    class_layers: {class_layers}\n")


def test_conversion_adds_the_class_layers_flag_to_the_detection_mask_and_keypoint_heads(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the class layer flag of the detection heads (object detection and instance
    segmentation), of the mask heads (instance segmentation) and of the keypoint head (keypoint detection), with its
    comment lines, right after 'freeze' in their 'pretrained' block: False for heads trained from random weights, True
    for heads transferred from another run, which loaded their class layers, so that the run names and the parameter
    groups of the saved runs stay; the heads of the other tasks are left as they are, and the check refuses transferred
    heads written False.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for experiment_folder, class_layers in [("null", "False"), ('"/runs/source"', "True")]:
        # Flags of the sections of every task, the mask heads being transferred in every text
        expected_insertions = {"object_detection": [("Detector", DETECTOR_CLASS_LAYERS_TEXT)],
                               "instance_segmentation": [("Detector", DETECTOR_CLASS_LAYERS_TEXT),
                                                         ("MaskHead", MASK_HEAD_CLASS_LAYERS_TEXT)],
                               "keypoint_detection": [("KeypointHead", KEYPOINT_HEAD_CLASS_LAYERS_TEXT)]}
        for task, insertions in expected_insertions.items():
            original_text = HEADS_TEXT.format(task=task, experiment_folder=experiment_folder)
            expected_text = original_text
            for section_name, flag_text in insertions:
                freeze_line = "    freeze: True\n" if section_name == "MaskHead" else "    freeze: False\n"
                section_start = expected_text.index(f"{section_name}:\n")
                freeze_index = expected_text.index(freeze_line, section_start) + len(freeze_line)
                expected_text = (expected_text[:freeze_index] + flag_text.format(class_layers=class_layers) +
                                 expected_text[freeze_index:])
            converted_text = add_required_settings(configuration_text=original_text, task=task,
                                                   configuration_path=configuration_path)
            assert converted_text == expected_text, task
            assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                    configuration_path=configuration_path) == []

    # Heads of another task left as they are, and transferred detection heads written False refused
    classification_text = HEADS_TEXT.format(task="classification", experiment_folder="null")
    assert add_required_settings(configuration_text=classification_text, task="classification",
                                 configuration_path=configuration_path) == classification_text
    transferred_text = HEADS_TEXT.format(task="object_detection", experiment_folder='"/runs/source"')
    written_false_text = transferred_text.replace(
        "    freeze: False\n", "    freeze: False\n" + DETECTOR_CLASS_LAYERS_TEXT.format(class_layers="False"), 1)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: Detector\.pretrained\.class_layers\.$"):
        check_conversion(original_text=transferred_text, converted_text=written_false_text, task="object_detection",
                         configuration_path=configuration_path)


# Lines the conversion writes after the normalization of the neck section, its 'pretrained' block without source run,
# the normalization of the neck section the detection texts hold, and the detection sections of the files converted
# between the neck and its transfer, holding every other added setting
NECK_PRETRAINED_TEXT = "".join(f"  {line}\n" for line in NECK_PRETRAINED_LINES)
NECK_NORMALIZATION_TEXT = ('  normalization:\n    # "batch", "layer" (over channels) or "none"\n'
                           '    feature_map_normalization: "batch"\n')
DETECTION_TEXT_WITHOUT_NECK_TRANSFER = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(NECK_PRETRAINED_TEXT, "")


def test_conversion_adds_the_neck_transfer_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the 'pretrained' block of the neck without source run, with its comment lines,
    right after the normalization of the neck section of the detection and instance segmentation files written between
    the neck and its transfer, whose necks started from random weights, and only there: a classification file and a
    file holding the block, a loaded neck included, are left as they are, a neck section without normalization is
    refused, the check refuses a converted file lacking the block or loading a neck, which would change its runs, and
    convert_fixed_settings converts a launch file with a backup of the latest suffix.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    assert CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.count(NECK_PRETRAINED_TEXT) == 1
    assert DETECTION_TEXT_WITHOUT_NECK_TRANSFER.count(NECK_NORMALIZATION_TEXT) == 1
    for task in ["object_detection", "instance_segmentation"]:
        original_text = write_task_text(configuration_text=DETECTION_TEXT_WITHOUT_NECK_TRANSFER, task=task)
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == write_task_text(configuration_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS,
                                                 task=task)
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # Files of another task or already holding the block, a neck loaded from another run included, left as they are
    assert add_required_settings(configuration_text=CONVERTED_CLASSIFICATION_TEXT, task="classification",
                                 configuration_path=configuration_path) == CONVERTED_CLASSIFICATION_TEXT
    loaded_neck_text = CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(
        'type: "none"', 'type: "feature_pyramid"').replace("    experiment_folder: null\n",
                                                          '    experiment_folder: "/runs/source"\n')
    assert add_required_settings(configuration_text=loaded_neck_text, task="object_detection",
                                 configuration_path=configuration_path) == loaded_neck_text
    assert check_conversion(original_text=loaded_neck_text, converted_text=loaded_neck_text, task="object_detection",
                            configuration_path=configuration_path) == []

    # A neck section without normalization, and conversions without the block or loading a neck
    with pytest.raises(expected_exception=ValueError, match="has no 'Neck.normalization' setting"):
        add_required_settings(configuration_text=DETECTION_TEXT_WITHOUT_NECK_TRANSFER.replace(
            NECK_NORMALIZATION_TEXT, ""), task="object_detection", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: Neck\.pretrained\.experiment_folder, Neck\.pretrained\.freeze, "
                             r"Neck\.pretrained\.iteration, Neck\.pretrained\.learning_rate_multiplier\.$"):
        check_conversion(original_text=DETECTION_TEXT_WITHOUT_NECK_TRANSFER,
                         converted_text=DETECTION_TEXT_WITHOUT_NECK_TRANSFER, task="object_detection",
                         configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError,
                       match=r"than the fixed ones: Neck\.pretrained\.experiment_folder\.$"):
        check_conversion(original_text=DETECTION_TEXT_WITHOUT_NECK_TRANSFER,
                         converted_text=CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS.replace(
                             "    experiment_folder: null\n", '    experiment_folder: "/runs/source"\n'),
                         task="object_detection", configuration_path=configuration_path)

    # A launch file converted with its backup, the original kept in it
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    (configuration_folder / "run-1.yaml").write_text(data=DETECTION_TEXT_WITHOUT_NECK_TRANSFER, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[], configuration_folders=[configuration_folder],
                                             write_files=True, backup_suffix=DEFAULT_BACKUP_SUFFIX)
    # The default names the latest conversion, the 'RegionProposal.shared' and 'Detector.shared' added after the neck
    # block
    assert DEFAULT_BACKUP_SUFFIX == "_before_shared_detection_heads"
    assert converted_paths == [configuration_folder / "run-1.yaml"]
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == \
        CONVERTED_DETECTION_TEXT_WITH_ADDED_SETTINGS
    assert (configuration_folder / "previous_configurations" / f"run-1{DEFAULT_BACKUP_SUFFIX}.yaml").read_text(
        encoding="utf-8") == DETECTION_TEXT_WITHOUT_NECK_TRANSFER


# Segmentation decoder written before the renaming of its class layer flag, its section comment naming it
SEGMENTATION_DECODER_TEXT = """ExperimentConfiguration:
  task: {task}
  # Keeps load_class_layers of the decoder as written.
  run_name_suffix: ""
SegmentationDecoder:
  # Decoder of another run; its class convolution and supervision heads need load_class_layers.
  pretrained:
    experiment_folder: "/runs/source"
    freeze: False
    # Also loads the segmentation head and deep supervision heads.
    load_class_layers: True # loaded
    learning_rate_multiplier: 1.0
"""


def test_conversion_renames_the_class_layers_flag_of_the_segmentation_decoder(tmp_path: Path) -> None:
    """
    Verifies that the conversion renames 'SegmentationDecoder.pretrained.load_class_layers' to 'class_layers' in the
    files of the semantic segmentation, keeping its value, its place and its comment, and names it under its new name
    in the comments of the 'SegmentationDecoder' section only; the check refuses a file still holding the former name,
    a file of another task is left as it is, and convert_fixed_settings converts a run configuration with its backup.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    original_text = SEGMENTATION_DECODER_TEXT.format(task="semantic_segmentation")
    expected_text = original_text.replace(
        "    load_class_layers: True # loaded", "    class_layers: True # loaded").replace(
        "supervision heads need load_class_layers.", "supervision heads need class_layers.")
    converted_text = rename_settings(configuration_text=original_text, task="semantic_segmentation")
    assert converted_text == expected_text
    assert "  # Keeps load_class_layers of the decoder as written." in converted_text
    assert check_conversion(original_text=original_text, converted_text=converted_text, task="semantic_segmentation",
                            configuration_path=configuration_path) == []

    # The former name kept refused, and another task left as it is
    with pytest.raises(expected_exception=ValueError, match="SegmentationDecoder.pretrained.class_layers"):
        check_conversion(original_text=original_text, converted_text=original_text, task="semantic_segmentation",
                         configuration_path=configuration_path)
    other_text = SEGMENTATION_DECODER_TEXT.format(task="classification")
    assert rename_settings(configuration_text=other_text, task="classification") == other_text

    # A run configuration converted with its backup next to it
    run_folder = tmp_path / "runs" / "SEG_run"
    run_folder.mkdir(parents=True)
    (run_folder / "training_configuration.yaml").write_text(data=original_text, encoding="utf-8")
    converted_paths = convert_fixed_settings(run_roots=[tmp_path / "runs"], configuration_folders=[], write_files=True,
                                             backup_suffix="_before_class_layers")
    assert converted_paths == [run_folder / "training_configuration.yaml"]
    assert (run_folder / "training_configuration.yaml").read_text(encoding="utf-8") == expected_text
    assert (run_folder / "training_configuration_before_class_layers.yaml").read_text(
        encoding="utf-8") == original_text


def test_conversion_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the conversion script launched without any argument prints the help of every one of its arguments
    and stops without searching any folder, the folders of every machine being its own.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_conversion_arguments(arguments=[])
    assert exit_information.value.code == 0
    help_text = capsys.readouterr().out
    assert [argument for argument in CONVERSION_SCRIPT_ARGUMENTS if argument not in help_text] == []


def test_conversion_arguments_only_list_by_default(tmp_path: Path) -> None:
    """
    Verifies that the conversion script given a single folder searches no other folder, only lists the files to
    convert and names the backups with DEFAULT_BACKUP_SUFFIX, and that its arguments are the parameters of
    convert_fixed_settings, so that the main block passes them all.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for a launch folder.
    """
    parsed_arguments = parse_conversion_arguments(arguments=["--configuration_folders", str(tmp_path)])
    assert vars(parsed_arguments) == {"run_roots": [], "configuration_folders": [tmp_path], "write_files": False,
                                      "backup_suffix": DEFAULT_BACKUP_SUFFIX}
    assert set(vars(parsed_arguments)) == set(inspect.signature(convert_fixed_settings).parameters)
    assert DEFAULT_BACKUP_SUFFIX.startswith("_before_")


def test_conversion_arguments_reach_every_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that every setting of convert_fixed_settings can be given on the command line, several folders of each
    kind, paths starting with a '~' the shell leaves after '=' being expanded to the home folder.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    for folder_name in ["runs", "other_runs", "launch_files"]:
        (tmp_path / folder_name).mkdir()
    parsed_arguments = parse_conversion_arguments(arguments=[
        "--run_roots", "~/runs", str(tmp_path / "other_runs"), "--configuration_folders=~/launch_files",
        "--write_files", "--backup_suffix", "_before_test"])
    assert vars(parsed_arguments) == {"run_roots": [tmp_path / "runs", tmp_path / "other_runs"],
                                      "configuration_folders": [tmp_path / "launch_files"], "write_files": True,
                                      "backup_suffix": "_before_test"}


@pytest.mark.parametrize(argnames="arguments", argvalues=[
    ["--write_files"],
    ["--backup_suffix", "_before_test"],
    ["--run_roots"],
    ["--configuration_folders", "missing_launch_files"]])
def test_conversion_script_refuses_a_command_line_without_an_existing_folder(arguments: List[str],
                                                                            capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the conversion script stops with its usage and the error, before any file is read, when no folder is
    given or a given folder does not exist, which would otherwise list no file.

    Args:
        arguments (List[str]): The command line arguments.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_conversion_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_conversion_script_converts_the_folders_of_its_arguments(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the main block of the conversion script passes every argument to convert_fixed_settings: a launch
    file holding fixed settings is only listed without --write_files, then converted with --write_files, its backup
    named with the given suffix.

    Args:
        tmp_path (Path): The pytest temporary directory, holding the launch folder.
        monkeypatch (pytest.MonkeyPatch): Sets the command line of the script.
    """
    configuration_folder = tmp_path / "launch_files"
    configuration_folder.mkdir()
    original_text = CONFIGURATION_TEXT.format(task="classification")
    (configuration_folder / "run-1.yaml").write_text(data=original_text, encoding="utf-8")

    # Listed only, then converted with its backup, as the script run as a module would do
    for extra_arguments in [[], ["--write_files"]]:
        monkeypatch.setattr(target=sys, name="argv", value=[
            conversion_script.__file__, "--configuration_folders", str(configuration_folder), "--backup_suffix",
            "_before_test", *extra_arguments])
        runpy.run_path(path_name=conversion_script.__file__, run_name="__main__")
        if not extra_arguments:
            assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == original_text
            assert not (configuration_folder / "previous_configurations").exists()
    assert (configuration_folder / "run-1.yaml").read_text(encoding="utf-8") == CONVERTED_CLASSIFICATION_TEXT
    assert (configuration_folder / "previous_configurations" / "run-1_before_test.yaml").read_text(
        encoding="utf-8") == original_text
