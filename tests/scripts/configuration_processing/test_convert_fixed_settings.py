"""
Tests of scripts/configuration_processing/convert_fixed_settings.py, the conversion of the configuration files written
before the fixed settings: the values changed by a conversion, and the refusals of a conversion changing another
setting or the backbone.
"""

import pytest

from pathlib import Path

from scripts.configuration_processing.convert_fixed_settings import check_conversion, remove_fixed_settings
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
  # Smooth L1 beta of the box regression loss.
  localisation_loss_beta: 1/9
Detector:
  normalization:
    feature_map_normalization: "batch"
    batch_normalization_momentum: 0.1
  localisation_loss_beta: 1/9
"""

# The detection sections once converted
CONVERTED_DETECTION_TEXT = """ExperimentConfiguration:
  task: object_detection
RegionProposal:
  normalization:
    feature_map_normalization: "batch"
  number_training_positives: 128
Detector:
  normalization:
    feature_map_normalization: "batch"
"""


def test_conversion_removes_the_fixed_head_settings(tmp_path: Path) -> None:
    """
    Verifies that the conversion removes the batch normalization momentums of the detection heads, the strict fallback
    assignment and the localisation loss betas with their comments, listing the strict assignment turned on as a value
    its runs change to, the momentums of 0.1 and the betas of 1/9 being those of the code.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    converted_text = remove_fixed_settings(configuration_text=DETECTION_CONFIGURATION_TEXT, task="object_detection")
    assert converted_text == CONVERTED_DETECTION_TEXT
    changed_values = check_conversion(original_text=DETECTION_CONFIGURATION_TEXT, converted_text=converted_text,
                                      task="object_detection", configuration_path=tmp_path / "configuration.yaml")
    assert changed_values == ["RegionProposal.strict_fallback_assignment True -> False"]
