"""
Tests of scripts/configuration_processing/convert_fixed_settings.py, the conversion of the configuration files written
before the fixed settings and the added sections and settings: the values changed by a conversion, the neck section
added to the detection files and the box losses added to their region proposal and detector sections, and the
refusals of a conversion changing another setting or the backbone.
"""

import pytest

from pathlib import Path

from scripts.configuration_processing.convert_fixed_settings import (NECK_SECTION_TEXT, add_required_sections,
                                                                      add_required_settings, check_conversion,
                                                                      remove_fixed_settings)
from vision_training.data.loaders import NUMBER_LOADER_WORKERS
from vision_training.evaluation.classification.classification_metrics import TOP_K_ACCURACY_CLASSES

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
  total_training_samples: 128
  localisation_loss_beta: 1/9
"""

# The detection sections once converted
CONVERTED_DETECTION_TEXT = """ExperimentConfiguration:
  task: object_detection
RegionProposal:
  normalization:
    feature_map_normalization: "batch"
  number_training_positives: 128
  total_training_samples: 256
Detector:
  normalization:
    feature_map_normalization: "batch"
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


# Lines the conversion writes after the sampling budget of the region proposal and of the detector sections
REGION_PROPOSAL_BOX_LOSS_TEXT = (
    '  # Box loss of the positive anchors: "smooth_l1" on the offsets, or "complete_iou" (1 - CIoU of the\n'
    '  # decoded boxes, https://arxiv.org/abs/2005.03572).\n'
    '  box_loss: "smooth_l1"\n')
DETECTOR_BOX_LOSS_TEXT = REGION_PROPOSAL_BOX_LOSS_TEXT.replace("positive anchors", "positive proposals")

# The converted detection sections with the neck, once the box losses are added after their sampling budgets
CONVERTED_DETECTION_TEXT_WITH_BOX_LOSSES = CONVERTED_DETECTION_TEXT_WITH_NECK.replace(
    "  total_training_samples: 256\n", f"  total_training_samples: 256\n{REGION_PROPOSAL_BOX_LOSS_TEXT}").replace(
    "  total_training_samples: 128\n", f"  total_training_samples: 128\n{DETECTOR_BOX_LOSS_TEXT}")


def test_conversion_adds_the_box_losses_to_the_detection_files(tmp_path: Path) -> None:
    """
    Verifies that the conversion writes the smooth L1 box loss of the runs written before the CIoU loss, with its
    comment lines, right after the sampling budget of the region proposal and of the detector sections of the detection
    and instance segmentation files, and only there: a classification file and a file holding the box losses are left
    as they are, a file without sampling budget is refused, and the check refuses a converted file lacking a box loss
    or holding the CIoU loss, which would change its runs.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"
    for task in ["object_detection", "instance_segmentation"]:
        original_text = CONVERTED_DETECTION_TEXT_WITH_NECK.replace("task: object_detection", f"task: {task}")
        converted_text = add_required_settings(configuration_text=original_text, task=task,
                                               configuration_path=configuration_path)
        assert converted_text == CONVERTED_DETECTION_TEXT_WITH_BOX_LOSSES.replace("task: object_detection",
                                                                                  f"task: {task}")
        assert check_conversion(original_text=original_text, converted_text=converted_text, task=task,
                                configuration_path=configuration_path) == []

    # Files of another task or already holding the box losses are left as they are
    assert add_required_settings(configuration_text=CONVERTED_CLASSIFICATION_TEXT, task="classification",
                                 configuration_path=configuration_path) == CONVERTED_CLASSIFICATION_TEXT
    assert add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_BOX_LOSSES, task="object_detection",
                                 configuration_path=configuration_path) == CONVERTED_DETECTION_TEXT_WITH_BOX_LOSSES

    # A detector section without sampling budget, and conversions without a box loss or with the CIoU loss
    with pytest.raises(expected_exception=ValueError, match="has no 'Detector.total_training_samples' setting"):
        add_required_settings(configuration_text=CONVERTED_DETECTION_TEXT_WITH_NECK.replace(
            "  total_training_samples: 128\n", ""), task="object_detection", configuration_path=configuration_path)
    with pytest.raises(expected_exception=ValueError, match="Detector.box_loss, RegionProposal.box_loss"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NECK, task="object_detection",
                         converted_text=CONVERTED_DETECTION_TEXT_WITH_NECK, configuration_path=configuration_path)
    complete_iou_text = CONVERTED_DETECTION_TEXT_WITH_BOX_LOSSES.replace(
        DETECTOR_BOX_LOSS_TEXT, DETECTOR_BOX_LOSS_TEXT.replace('box_loss: "smooth_l1"', 'box_loss: "complete_iou"'))
    with pytest.raises(expected_exception=ValueError, match=r"than the fixed ones: Detector\.box_loss\.$"):
        check_conversion(original_text=CONVERTED_DETECTION_TEXT_WITH_NECK, converted_text=complete_iou_text,
                         task="object_detection", configuration_path=configuration_path)
