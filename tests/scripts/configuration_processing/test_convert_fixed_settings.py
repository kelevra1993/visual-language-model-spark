"""
Tests of scripts/configuration_processing/convert_fixed_settings.py, the conversion of the configuration files written
before the fixed settings: the fixed settings removed with their comments and the sections left empty, the settings
kept by the tasks still holding them, the files of the run folders and launch folders converted with their backups
(the MLflow copies, the backups and the other YAML files left alone), the values changed by a conversion, and the
refusals of a conversion changing another setting or the backbone and of an existing backup.
"""

import pytest

from pathlib import Path

from scripts.configuration_processing.convert_fixed_settings import (FIXED_SETTINGS, check_conversion,
                                                                     convert_fixed_settings, remove_fixed_settings)
from vision_training.configuration_processing.common_checks import REMOVED_SETTINGS
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


def test_fixed_settings_are_removed_with_their_comments() -> None:
    """
    Verifies the conversion of the configuration of an image task: every fixed setting goes with the comment lines
    directly above it, the loading section left empty goes with its comment, a setting removed between two blank lines
    leaves a single one, and the other settings stay. Every fixed setting is one of the removed settings refused by the
    configuration checker.
    """
    assert all(key_path in REMOVED_SETTINGS for key_path in FIXED_SETTINGS)
    converted_text = remove_fixed_settings(configuration_text=CONFIGURATION_TEXT.format(task="classification"),
                                           task="classification")
    assert converted_text == CONVERTED_CLASSIFICATION_TEXT
    assert remove_fixed_settings(configuration_text=converted_text, task="classification") == converted_text


def test_tasks_holding_a_setting_keep_it() -> None:
    """
    Verifies that the frame selection keeps the number of candidate frames of its deployment and the sequence settings,
    while losing the settings fixed for every task.
    """
    converted_text = remove_fixed_settings(configuration_text=CONFIGURATION_TEXT.format(task="frame_selection"),
                                           task="frame_selection")
    assert "top_k: 2" in converted_text and "# Classes counted by the top-k accuracy" in converted_text
    assert "independent_frames: False" in converted_text and "shuffle_frames:" in converted_text
    assert "resume_training" not in converted_text and "loading" not in converted_text
    assert "inception_starting_block" not in converted_text


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


def test_files_are_converted_with_their_backups(tmp_path: Path) -> None:
    """
    Verifies the conversion of the files: the configuration of a run folder with its backup next to it, a launch file
    with its backup in the 'previous_configurations' folder of its configuration folder, the MLflow copy, the earlier
    backups and the other YAML files left alone, nothing written without write_files, nothing left to convert after
    the conversion, and an existing backup refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    original_text = CONFIGURATION_TEXT.format(task="classification")
    run_configuration = tmp_path / "experiments" / "run" / "training_configuration.yaml"
    mlflow_copy = tmp_path / "experiments" / "mlartifacts" / "1" / "artifacts" / "training_configuration.yaml"
    launch_file = tmp_path / "launch_files" / "group" / "launch.yaml"
    earlier_backup = tmp_path / "launch_files" / "previous_configurations" / "group" / "launch_before_change.yaml"
    other_file = tmp_path / "launch_files" / "labels.yaml"
    for path, text in [(run_configuration, original_text), (mlflow_copy, original_text), (launch_file, original_text),
                       (earlier_backup, original_text), (other_file, "labels: [1, 2]\n")]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data=text, encoding="utf-8")
    folders = {"run_roots": [tmp_path / "experiments"], "configuration_folders": [tmp_path / "launch_files"]}

    # Listed without writing anything, then converted with their backups
    assert convert_fixed_settings(**folders, write_files=False) == [run_configuration, launch_file]
    assert run_configuration.read_text(encoding="utf-8") == original_text
    assert convert_fixed_settings(**folders, write_files=True) == [run_configuration, launch_file]
    run_backup = run_configuration.with_name("training_configuration_before_fixed_settings.yaml")
    launch_backup = earlier_backup.with_name("launch_before_fixed_settings.yaml")
    for converted_path, backup_path in [(run_configuration, run_backup), (launch_file, launch_backup)]:
        assert converted_path.read_text(encoding="utf-8") == CONVERTED_CLASSIFICATION_TEXT
        assert backup_path.read_text(encoding="utf-8") == original_text
    assert mlflow_copy.read_text(encoding="utf-8") == original_text
    assert earlier_backup.read_text(encoding="utf-8") == original_text
    assert convert_fixed_settings(**folders, write_files=True) == []

    # A file written again with the fixed settings is not converted over its existing backup
    run_configuration.write_text(data=original_text, encoding="utf-8")
    with pytest.raises(expected_exception=FileExistsError, match="already exists"):
        convert_fixed_settings(**folders, write_files=True)
