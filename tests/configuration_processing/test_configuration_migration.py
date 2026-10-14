"""
Tests of vision_training/configuration_processing/configuration_migration.py: every migration step (task, pretrained
settings, normalization flag, evaluation outputs, per-class report, sample images, MLflow tracking, sequence
augmentation, normalization scope, optimization section, loading settings, batch padding, positional embedding,
segmentation settings, batch normalization momentums, classification head normalization, pre-activation flag of the
residual shortcuts, weight initialisation and moving average of the weights) is added once to an old configuration,
and migrate_run_folder renames old run folders after their settings.
"""

import json
import pytest
import re
import yaml

from pathlib import Path

from vision_training.configuration_processing.configuration_migration import (
    BOOLEAN_HEAD_NORMALIZATION_COMMENT_LINES, EXPONENTIAL_MOVING_AVERAGE_LINES, HEAD_NORMALIZATION_COMMENT_LINES,
    HEAD_NORMALIZATION_LINE_START, MIXED_PRECISION_LINE_START, RESIDUAL_NETS_V2_LINES, TRACKING_SECTION_LINES,
    WEIGHT_INITIALIZATION_LINES, add_batch_normalization_momentums_to_configuration,
    add_batch_padding_to_configuration, add_evaluation_outputs_to_configuration,
    add_head_normalization_to_configuration,
    add_freeze_normalization_parameters_to_configuration, add_loading_settings_to_configuration,
    add_optimization_section_to_configuration, add_per_class_report_to_configuration,
    add_positional_embedding_to_configuration, add_pretrained_settings_to_configuration,
    add_sample_images_to_configuration, add_segmentation_settings_to_configuration,
    add_sequence_augmentation_to_configuration, add_sequence_normalization_scope_to_configuration,
    add_residual_nets_v2_to_configuration, add_task_to_configuration, add_tracking_to_configuration,
    add_weight_settings_to_configuration, migrate_run_folder, remove_dali_settings_from_configuration)

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")
SEQUENCE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                               "sequence_classification" / "configuration_synthetic_sequences.yaml")
COMPARISON_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                 "sequence_comparison" / "configuration_synthetic_pairs.yaml")
SEGMENTATION_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                   "semantic_segmentation" / "configuration_synthetic_shapes.yaml")
CLASSIFICATION_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                     "classification" / "configuration_example.yaml")
INSTANCE_SEGMENTATION_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                            "instance_segmentation" / "configuration_example.yaml")

# Pretrained backbone settings written by the migration of the backbone transfer, before the normalization flag existed
OLD_PRETRAINED_BACKBONE_BLOCK = ("  # Backbone of another training run initialising this run, null to start from "
                                 "random weights.\n"
                                 "  pretrained:\n"
                                 "    experiment_folder: null\n"
                                 "    iteration: null\n"
                                 "    freeze: False\n"
                                 "    learning_rate_multiplier: 1.0\n")

# Pretrained settings of a run trained from random weights
RANDOM_PRETRAINED_SETTINGS = {"experiment_folder": None, "iteration": None, "freeze": False,
                              "learning_rate_multiplier": 1.0}
RANDOM_PRETRAINED_BACKBONE_SETTINGS = {"experiment_folder": None, "iteration": None, "freeze": False,
                                       "freeze_normalization_parameters": False, "learning_rate_multiplier": 1.0}


def write_old_run_folder(temporary_directory: Path) -> Path:
    """
    Writes a run folder as trained before the task-aware configuration, the component transfers and the
    optimization settings: an old-style name (without task code, with the augmentation, batch size and learning rate
    tokens no longer written), and a saved configuration without task, pretrained backbone, region proposal and
    detection head settings, per-class report switch, number of sample images, tracking settings, optimization
    section nor batch normalization momentums, and with the block of the removed DALI loader (a copy of the example
    configuration, with binary labels so that no mapping file is needed).

    Args:
        temporary_directory (Path): The pytest temporary directory, used as the experiments root folder.

    Returns:
        Path: The run folder.
    """
    configuration = yaml.safe_load(stream=EXAMPLE_CONFIGURATION_PATH.read_text())
    del configuration["ExperimentConfiguration"]["task"]
    del configuration["ExperimentConfiguration"]["evaluation"]["per_class_report"]
    del configuration["ExperimentConfiguration"]["evaluation"]["number_sample_images"]
    del configuration["ExperimentConfiguration"]["tracking"]
    del configuration["Backbone"]["pretrained"]
    del configuration["RegionProposal"]["pretrained"]
    del configuration["Detector"]["pretrained"]
    del configuration["Optimization"]
    del configuration["Data"]["loading"]
    for section_name in ["Backbone", "RegionProposal", "Detector"]:
        del configuration[section_name]["normalization"]["batch_normalization_momentum"]
    configuration["Data"]["DALI"] = {"activated": True}
    configuration["Data"]["label_mapping"] = {"mode": "binary", "mapping_file_path": None}

    run_folder = temporary_directory / "IN-KAR-1024_old_run_name_AUG-BS40-LR2e-4"
    run_folder.mkdir()
    (run_folder / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=configuration))
    return run_folder


def test_task_is_added_once(tmp_path: Path) -> None:
    """
    Verifies that the task is added to a configuration written before the task-aware configuration, and that a
    configuration which already has a task is left untouched.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_old_run_folder(temporary_directory=tmp_path) / "training_configuration.yaml"

    assert add_task_to_configuration(configuration_path=configuration_path, task="object_detection") is True
    upgraded_configuration = yaml.safe_load(stream=configuration_path.read_text())
    assert upgraded_configuration["ExperimentConfiguration"]["task"] == "object_detection"
    assert add_task_to_configuration(configuration_path=configuration_path, task="object_detection") is False


def test_run_folder_is_renamed_after_its_settings(tmp_path: Path) -> None:
    """
    Verifies that migrating an old run upgrades its configuration (its DALI block removed), leaves its folder in
    place unless renaming is asked, renames it to the name the current code builds (starting with the task code,
    without the augmentation, batch size and learning rate tokens), that migrating it again changes nothing, and that
    another old run differing only by its learning rate is refused instead of being merged into the renamed folder.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    run_folder = write_old_run_folder(temporary_directory=tmp_path)

    assert migrate_run_folder(run_folder=run_folder, task="object_detection", rename_folder=False) == run_folder
    assert run_folder.exists()
    assert "DALI" not in yaml.safe_load(stream=(run_folder / "training_configuration.yaml").read_text())["Data"]

    renamed_folder = migrate_run_folder(run_folder=run_folder, task="object_detection", rename_folder=True)
    assert renamed_folder.name.startswith("DET_") and renamed_folder.name.endswith("_BIN")
    assert not any(token in renamed_folder.name for token in ["AUG", "BS40", "LR2e-4"])
    assert renamed_folder.exists() and not run_folder.exists()

    assert migrate_run_folder(run_folder=renamed_folder, task="object_detection", rename_folder=True) == renamed_folder

    # Another old run with the same settings except its learning rate now has the same name, so it is not renamed
    other_run_folder = write_old_run_folder(temporary_directory=tmp_path)
    other_configuration_path = other_run_folder / "training_configuration.yaml"
    other_configuration = yaml.safe_load(stream=other_configuration_path.read_text())
    other_configuration["ExperimentConfiguration"]["learning_rate"] = 1e-3
    other_configuration_path.write_text(data=yaml.safe_dump(data=other_configuration))
    with pytest.raises(expected_exception=FileExistsError):
        migrate_run_folder(run_folder=other_run_folder, task="object_detection", rename_folder=True)
    assert other_run_folder.exists()


def test_pretrained_settings_are_added_once(tmp_path: Path) -> None:
    """
    Verifies that the pretrained backbone settings of a run trained from random weights, normalization flag included,
    are added to a configuration written before the backbone transfer, keeping its other backbone settings, that a
    configuration which already has them is left untouched, that the region proposal and detection heads of a
    detection configuration get the pretrained settings of heads trained from random weights the same way, and that a
    detection configuration, which has no temporal module nor classification head, is not changed by their
    migrations.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_old_run_folder(temporary_directory=tmp_path) / "training_configuration.yaml"
    backbone_before = yaml.safe_load(stream=configuration_path.read_text())["Backbone"]

    assert add_pretrained_settings_to_configuration(configuration_path=configuration_path,
                                                    section_name="Backbone") is True
    backbone_after = yaml.safe_load(stream=configuration_path.read_text())["Backbone"]
    assert backbone_after["pretrained"] == RANDOM_PRETRAINED_BACKBONE_SETTINGS
    assert {key: value for key, value in backbone_after.items() if key != "pretrained"} == backbone_before
    assert add_pretrained_settings_to_configuration(configuration_path=configuration_path,
                                                    section_name="Backbone") is False
    assert add_freeze_normalization_parameters_to_configuration(configuration_path=configuration_path) is False

    # Region proposal and detection heads of a detection run written before their transfer
    for section_name in ["RegionProposal", "Detector"]:
        section_before = yaml.safe_load(stream=configuration_path.read_text())[section_name]
        assert add_pretrained_settings_to_configuration(configuration_path=configuration_path,
                                                        section_name=section_name) is True
        section_after = yaml.safe_load(stream=configuration_path.read_text())[section_name]
        assert section_after["pretrained"] == RANDOM_PRETRAINED_SETTINGS
        assert {key: value for key, value in section_after.items() if key != "pretrained"} == section_before
        assert add_pretrained_settings_to_configuration(configuration_path=configuration_path,
                                                        section_name=section_name) is False

    # Sections the detection configuration does not have
    text_before = configuration_path.read_text()
    for section_name in ["TemporalModel", "ClassificationHead"]:
        assert add_pretrained_settings_to_configuration(configuration_path=configuration_path,
                                                        section_name=section_name) is False
    assert configuration_path.read_text() == text_before


def write_configuration_without_normalization_flag(configuration_path: Path, backbone_block_first: bool) -> None:
    """
    Writes the example detection configuration as it was before the normalization flag of the backbone existed: its
    pretrained settings closing the backbone section (the shipped configurations), or opening it (the settings
    inserted by the migration of the backbone transfer).

    Args:
        configuration_path (Path): The written configuration file.
        backbone_block_first (bool): True for the pretrained settings right below the backbone header.
    """
    lines = EXAMPLE_CONFIGURATION_PATH.read_text().splitlines(keepends=True)

    # Lines of the normalization flag and its comment, right after the freeze flag of the backbone
    flag_index = next(line_index for line_index, line in enumerate(lines)
                      if line.startswith("    freeze_normalization_parameters:"))
    freeze_index = next(line_index for line_index in range(flag_index, 0, -1)
                        if lines[line_index].startswith("    freeze:"))
    lines = lines[:freeze_index + 1] + lines[flag_index + 1:]

    # Pretrained settings moved right below the backbone header, as inserted by the older migration
    if backbone_block_first:
        pretrained_index = next(line_index for line_index in range(freeze_index, 0, -1)
                                if lines[line_index] == "  pretrained:\n")
        block_start = next(line_index for line_index in range(pretrained_index, 0, -1)
                           if not lines[line_index - 1].startswith("  #"))
        block_end = next(line_index for line_index in range(pretrained_index + 1, len(lines))
                         if not lines[line_index].startswith("    "))
        lines = lines[:block_start] + lines[block_end:]
        header_index = lines.index("Backbone:\n")
        lines = lines[:header_index + 1] + [OLD_PRETRAINED_BACKBONE_BLOCK] + lines[header_index + 1:]
    configuration_path.write_text(data="".join(lines))


def test_normalization_flag_is_added_after_the_freeze_flag(tmp_path: Path) -> None:
    """
    Verifies that the normalization flag of the backbone is added, off, right after the freeze flag of its pretrained
    settings, whether they close the backbone section or open it, without changing any other setting, and only once.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    for backbone_block_first in [False, True]:
        configuration_path = tmp_path / f"configuration_{backbone_block_first}.yaml"
        write_configuration_without_normalization_flag(configuration_path=configuration_path,
                                                       backbone_block_first=backbone_block_first)
        configuration_before = yaml.safe_load(stream=configuration_path.read_text())
        assert "freeze_normalization_parameters" not in configuration_before["Backbone"]["pretrained"]

        assert add_freeze_normalization_parameters_to_configuration(configuration_path=configuration_path) is True
        configuration_after = yaml.safe_load(stream=configuration_path.read_text())
        assert list(configuration_after["Backbone"]["pretrained"].items()) == list(
            RANDOM_PRETRAINED_BACKBONE_SETTINGS.items())
        configuration_after["Backbone"]["pretrained"].pop("freeze_normalization_parameters")
        assert configuration_after == configuration_before
        assert add_freeze_normalization_parameters_to_configuration(configuration_path=configuration_path) is False


def test_old_sequence_run_is_fully_migrated(tmp_path: Path) -> None:
    """
    Verifies the migration of a sequence classification run trained before the task-aware configuration, the
    transfers and the optimization settings: its configuration gets its task, the pretrained settings of its
    backbone, temporal module and classification head, and its optimization section, then passes every check, and
    its folder is renamed after its settings; migrating it again changes nothing.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration = yaml.safe_load(stream=SEQUENCE_CONFIGURATION_PATH.read_text())
    del configuration["ExperimentConfiguration"]["task"]
    for key_name in ["save_predictions", "error_dumps", "per_class_report", "number_sample_images"]:
        del configuration["ExperimentConfiguration"]["evaluation"][key_name]
    for key_name in ["independent_frames", "shuffle_frames"]:
        del configuration["Data"]["data_augmentation"][key_name]
    for section_name in ["Backbone", "TemporalModel", "ClassificationHead"]:
        del configuration[section_name]["pretrained"]
    del configuration["Optimization"]
    del configuration["Data"]["loading"]
    del configuration["Data"]["sequence_settings"]["batch_padding"]
    del configuration["TemporalModel"]["positional_embedding"]
    del configuration["Backbone"]["modules"]["residual_nets_v2"]
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.write_text(data=json.dumps(obj={"1": 0, "2": 1, "3": 2, "4": 3}))
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(mapping_path)
    run_folder = tmp_path / "IN-KAR-128_old_sequence_run"
    run_folder.mkdir()
    (run_folder / "training_configuration.yaml").write_text(data=yaml.safe_dump(data=configuration))

    renamed_folder = migrate_run_folder(run_folder=run_folder, task="sequence_classification", rename_folder=True)
    assert renamed_folder.name.startswith("M2O_") and "PT" not in renamed_folder.name.split("_")
    migrated_configuration = yaml.safe_load(stream=(renamed_folder / "training_configuration.yaml").read_text())
    assert migrated_configuration["Backbone"]["pretrained"] == RANDOM_PRETRAINED_BACKBONE_SETTINGS
    for section_name in ["TemporalModel", "ClassificationHead"]:
        assert migrated_configuration[section_name]["pretrained"] == RANDOM_PRETRAINED_SETTINGS
    assert migrated_configuration["ExperimentConfiguration"]["evaluation"]["save_predictions"] is False
    assert migrated_configuration["ExperimentConfiguration"]["evaluation"]["error_dumps"] == {
        "activated": False, "maximum_per_class_pair": 20}
    assert migrated_configuration["ExperimentConfiguration"]["evaluation"]["per_class_report"] is True
    assert migrated_configuration["ExperimentConfiguration"]["evaluation"]["number_sample_images"] == 0
    assert migrated_configuration["Data"]["data_augmentation"]["independent_frames"] is False
    assert migrated_configuration["Data"]["loading"] == {"number_workers": 4, "prefetch_batches": 2}
    assert migrated_configuration["Data"]["sequence_settings"]["batch_padding"] == "sequence_length"
    assert migrated_configuration["TemporalModel"]["positional_embedding"] is True
    assert migrated_configuration["Backbone"]["modules"]["residual_nets_v2"] is False
    assert migrated_configuration["Optimization"]["weight_initialization"] == "default"
    assert migrated_configuration["Optimization"]["exponential_moving_average"] == {"activated": False,
                                                                                    "decay": 0.9998}
    assert migrate_run_folder(run_folder=renamed_folder, task="sequence_classification",
                              rename_folder=True) == renamed_folder


def test_evaluation_outputs_are_added_once(tmp_path: Path) -> None:
    """
    Verifies that the predictions and error images settings of the classification-like tasks are added, off as in
    the runs written before them, after the target precision, that the error images setting alone is added to a
    configuration which already has the predictions setting, that a configuration which has both is left untouched,
    and that a detection configuration, whose evaluation has no target precision, is not changed.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration = yaml.safe_load(stream=SEQUENCE_CONFIGURATION_PATH.read_text())
    for key_name in ["save_predictions", "error_dumps"]:
        del configuration["ExperimentConfiguration"]["evaluation"][key_name]
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    assert add_evaluation_outputs_to_configuration(configuration_path=configuration_path) is True
    evaluation_settings = yaml.safe_load(stream=configuration_path.read_text())["ExperimentConfiguration"]["evaluation"]
    assert list(evaluation_settings.items())[-2:] == [("save_predictions", False),
                                                      ("error_dumps", {"activated": False,
                                                                       "maximum_per_class_pair": 20})]
    assert add_evaluation_outputs_to_configuration(configuration_path=configuration_path) is False

    # A configuration with the predictions setting only gets the error images setting
    configuration["ExperimentConfiguration"]["evaluation"]["save_predictions"] = True
    configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))
    assert add_evaluation_outputs_to_configuration(configuration_path=configuration_path) is True
    evaluation_settings = yaml.safe_load(stream=configuration_path.read_text())["ExperimentConfiguration"]["evaluation"]
    assert evaluation_settings["save_predictions"] is True and evaluation_settings["error_dumps"]["activated"] is False

    # The detection configuration keeps its evaluation settings
    detection_text = EXAMPLE_CONFIGURATION_PATH.read_text()
    detection_path = tmp_path / "detection_configuration.yaml"
    detection_path.write_text(data=detection_text)
    assert add_evaluation_outputs_to_configuration(configuration_path=detection_path) is False
    assert detection_path.read_text() == detection_text


@pytest.mark.parametrize(argnames="configuration_path", argvalues=[SEQUENCE_CONFIGURATION_PATH,
                                                                   EXAMPLE_CONFIGURATION_PATH])
def test_per_class_report_is_added_once(tmp_path: Path, configuration_path: Path) -> None:
    """
    Verifies that the per-class report switch is added, on as in the runs written before it, as the first evaluation
    setting of a classification-like run and of a detection run, that the other evaluation settings are kept, that a
    configuration which already has it is left untouched, and that a configuration without evaluation settings is not
    changed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_path (Path): The shipped configuration copied as an old run.
    """
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    evaluation_settings = configuration["ExperimentConfiguration"]["evaluation"]
    del evaluation_settings["per_class_report"]
    old_configuration_path = tmp_path / "configuration.yaml"
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    assert add_per_class_report_to_configuration(configuration_path=old_configuration_path) is True
    migrated_configuration = yaml.safe_load(stream=old_configuration_path.read_text())
    migrated_settings = migrated_configuration["ExperimentConfiguration"]["evaluation"]
    assert list(migrated_settings.items()) == [("per_class_report", True)] + list(evaluation_settings.items())
    assert add_per_class_report_to_configuration(configuration_path=old_configuration_path) is False

    # A configuration without evaluation settings is left as it is
    del configuration["ExperimentConfiguration"]["evaluation"]
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))
    unchanged_text = old_configuration_path.read_text()
    assert add_per_class_report_to_configuration(configuration_path=old_configuration_path) is False
    assert old_configuration_path.read_text() == unchanged_text


def test_sample_images_are_added_once(tmp_path: Path) -> None:
    """
    Verifies that the number of sample images is added, at 0 as in the runs written before it was shared by every task,
    right after the per-class report switch, the other evaluation settings being kept, only once; that a segmentation
    run, which already has it, and a configuration without evaluation settings are left untouched; and that evaluation
    settings without the per-class report switch are refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration = yaml.safe_load(stream=SEQUENCE_CONFIGURATION_PATH.read_text())
    evaluation_settings = configuration["ExperimentConfiguration"]["evaluation"]
    del evaluation_settings["number_sample_images"]
    old_configuration_path = tmp_path / "configuration.yaml"
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    # Added right after the per-class report switch, once
    assert add_sample_images_to_configuration(configuration_path=old_configuration_path) is True
    migrated_configuration = yaml.safe_load(stream=old_configuration_path.read_text())
    migrated_settings = migrated_configuration["ExperimentConfiguration"]["evaluation"]
    expected_settings = list(evaluation_settings.items())
    expected_settings.insert(list(evaluation_settings).index("per_class_report") + 1, ("number_sample_images", 0))
    assert list(migrated_settings.items()) == expected_settings
    assert add_sample_images_to_configuration(configuration_path=old_configuration_path) is False

    # A segmentation run already has it
    segmentation_path = tmp_path / "segmentation_configuration.yaml"
    segmentation_path.write_text(data=SEGMENTATION_CONFIGURATION_PATH.read_text())
    assert add_sample_images_to_configuration(configuration_path=segmentation_path) is False
    assert segmentation_path.read_text() == SEGMENTATION_CONFIGURATION_PATH.read_text()

    # Evaluation settings without the per-class report switch must be migrated by its step first
    del evaluation_settings["per_class_report"]
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))
    with pytest.raises(expected_exception=ValueError):
        add_sample_images_to_configuration(configuration_path=old_configuration_path)

    # A configuration without evaluation settings is left as it is
    del configuration["ExperimentConfiguration"]["evaluation"]
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))
    unchanged_text = old_configuration_path.read_text()
    assert add_sample_images_to_configuration(configuration_path=old_configuration_path) is False
    assert old_configuration_path.read_text() == unchanged_text


def test_tracking_is_added_once(tmp_path: Path) -> None:
    """
    Verifies that the MLflow tracking section is added, off, right after the evaluation settings of the experiment
    section: a commented configuration gets back exactly its current text, a configuration written by yaml gets the
    section before its next key with its other settings kept, evaluation settings ending the experiment section keep
    the blank line before the next section after the inserted one, the section is added only once, and an experiment
    section without evaluation settings is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "configuration.yaml"

    # A commented configuration without the section gets back the text of the current configuration
    current_text = CLASSIFICATION_CONFIGURATION_PATH.read_text()
    configuration_path.write_text(data=current_text.replace(TRACKING_SECTION_LINES, ""))
    assert add_tracking_to_configuration(configuration_path=configuration_path) is True
    assert configuration_path.read_text() == current_text
    assert add_tracking_to_configuration(configuration_path=configuration_path) is False
    assert configuration_path.read_text() == current_text

    # A configuration written by yaml, its keys sorted, the evaluation settings being followed by experiment_folder
    configuration = yaml.safe_load(stream=current_text)
    tracking_settings = configuration["ExperimentConfiguration"].pop("tracking")
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    assert add_tracking_to_configuration(configuration_path=configuration_path) is True
    migrated_configuration = yaml.safe_load(stream=configuration_path.read_text())
    migrated_tracking_settings = migrated_configuration["ExperimentConfiguration"].pop("tracking")
    assert migrated_tracking_settings == {**tracking_settings, "activated": False}
    assert migrated_configuration == configuration

    # Evaluation settings ending the experiment section, before a blank line and the next section
    configuration_path.write_text(data="ExperimentConfiguration:\n  evaluation:\n    per_class_report: True\n\n"
                                       "Optimization:\n  optimizer: \"adam\"\n")
    assert add_tracking_to_configuration(configuration_path=configuration_path) is True
    assert configuration_path.read_text() == ("ExperimentConfiguration:\n  evaluation:\n    per_class_report: True\n" +
                                              TRACKING_SECTION_LINES + "\nOptimization:\n  optimizer: \"adam\"\n")

    # An experiment section without evaluation settings must be migrated by the per-class report step first
    configuration_path.write_text(data="ExperimentConfiguration:\n  task: \"classification\"\n")
    with pytest.raises(expected_exception=ValueError):
        add_tracking_to_configuration(configuration_path=configuration_path)


def test_sequence_augmentation_is_added_once(tmp_path: Path) -> None:
    """
    Verifies that the sequence augmentations are added, off as in the runs written before them, right before the
    augmentation types, and that a configuration which already has them is left untouched.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration = yaml.safe_load(stream=EXAMPLE_CONFIGURATION_PATH.read_text())
    for key_name in ["independent_frames", "shuffle_frames"]:
        del configuration["Data"]["data_augmentation"][key_name]
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    assert add_sequence_augmentation_to_configuration(configuration_path=configuration_path) is True
    augmentation_settings = yaml.safe_load(stream=configuration_path.read_text())["Data"]["data_augmentation"]
    assert list(augmentation_settings)[2:5] == ["independent_frames", "shuffle_frames", "types"]
    assert augmentation_settings["independent_frames"] is False
    assert augmentation_settings["shuffle_frames"] == {"activated": False, "probability": 0.5}
    assert add_sequence_augmentation_to_configuration(configuration_path=configuration_path) is False


def test_sequence_normalization_scope_is_added_once(tmp_path: Path) -> None:
    """
    Verifies that the scope of the sequence normalization of the comparison task is added, every member normalised by
    its own frames as in the runs written before it, that a configuration which already has it is left untouched,
    and that a configuration without fusion section is not changed.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration = yaml.safe_load(stream=COMPARISON_CONFIGURATION_PATH.read_text())
    del configuration["Fusion"]["sequence_normalization_scope"]
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    assert add_sequence_normalization_scope_to_configuration(configuration_path=configuration_path) is True
    fusion_settings = yaml.safe_load(stream=configuration_path.read_text())["Fusion"]
    assert fusion_settings == {**configuration["Fusion"], "sequence_normalization_scope": "member"}
    assert add_sequence_normalization_scope_to_configuration(configuration_path=configuration_path) is False

    # A configuration of another task keeps its settings
    sequence_text = SEQUENCE_CONFIGURATION_PATH.read_text()
    sequence_path = tmp_path / "sequence_configuration.yaml"
    sequence_path.write_text(data=sequence_text)
    assert add_sequence_normalization_scope_to_configuration(configuration_path=sequence_path) is False
    assert sequence_path.read_text() == sequence_text


def test_optimization_section_is_added_once(tmp_path: Path) -> None:
    """
    Verifies that the optimization section of a run trained with plain Adam at a constant learning rate is appended
    to a configuration written before it, and that a configuration which already has it is left untouched.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_old_run_folder(temporary_directory=tmp_path) / "training_configuration.yaml"

    assert add_optimization_section_to_configuration(configuration_path=configuration_path) is True
    assert yaml.safe_load(stream=configuration_path.read_text())["Optimization"] == {
        "optimizer": "adam", "weight_decay": 0.0, "learning_rate_schedule": "constant", "warmup_iterations": 0,
        "decay_iterations": 100000, "minimum_learning_rate_ratio": 0.01, "gradient_clipping_norm": None,
        "mixed_precision": False}
    assert add_optimization_section_to_configuration(configuration_path=configuration_path) is False


def test_loading_settings_are_added_once(tmp_path: Path) -> None:
    """
    Verifies that the loading settings of the runs written before them (4 worker processes) are inserted below the
    header of the data section, and that a configuration which already has them is left untouched.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_old_run_folder(temporary_directory=tmp_path) / "training_configuration.yaml"

    assert add_loading_settings_to_configuration(configuration_path=configuration_path) is True
    assert yaml.safe_load(stream=configuration_path.read_text())["Data"]["loading"] == {"number_workers": 4,
                                                                                        "prefetch_batches": 2}
    assert add_loading_settings_to_configuration(configuration_path=configuration_path) is False


def test_dali_settings_are_removed_once(tmp_path: Path) -> None:
    """
    Verifies that the block of the removed DALI loader is cut out of the data section with its comments, every other
    line of the file being kept, and that a configuration without it is left untouched.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = tmp_path / "training_configuration.yaml"
    other_lines = ("Data:\n  # Settings of the loaders.\n  loading:\n    number_workers: 4\n    prefetch_batches: 2\n"
                   "  image_settings:\n    size: 512\n")
    configuration_path.write_text(data=other_lines.replace("  image_settings:\n", "  DALI:\n    # GPU loader.\n"
                                                           "    activated: False\n  image_settings:\n"))

    assert remove_dali_settings_from_configuration(configuration_path=configuration_path) is True
    assert configuration_path.read_text() == other_lines
    assert remove_dali_settings_from_configuration(configuration_path=configuration_path) is False


@pytest.mark.parametrize(argnames="configuration_path", argvalues=[SEQUENCE_CONFIGURATION_PATH,
                                                                    COMPARISON_CONFIGURATION_PATH])
def test_batch_padding_is_added_once(tmp_path: Path, configuration_path: Path) -> None:
    """
    Verifies that the padding of the batches of a sequence run written before it is added right after the handling of
    the short sequences, every window padded to the sequence length as in the runs of that time, that a configuration
    which already has it is left untouched, and that a configuration without sequence settings is not changed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_path (Path): The shipped sequence configuration the old configuration is made from.
    """
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    del configuration["Data"]["sequence_settings"]["batch_padding"]
    old_configuration_path = tmp_path / "configuration.yaml"
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    assert add_batch_padding_to_configuration(configuration_path=old_configuration_path) is True
    sequence_settings = yaml.safe_load(stream=old_configuration_path.read_text())["Data"]["sequence_settings"]
    assert sequence_settings == {**configuration["Data"]["sequence_settings"], "batch_padding": "sequence_length"}
    assert list(sequence_settings)[-2:] == ["short_sequences", "batch_padding"]
    assert add_batch_padding_to_configuration(configuration_path=old_configuration_path) is False

    # A configuration of an image task keeps its settings
    image_text = EXAMPLE_CONFIGURATION_PATH.read_text()
    image_path = tmp_path / "image_configuration.yaml"
    image_path.write_text(data=image_text)
    assert add_batch_padding_to_configuration(configuration_path=image_path) is False
    assert image_path.read_text() == image_text


@pytest.mark.parametrize(argnames="configuration_path", argvalues=[SEQUENCE_CONFIGURATION_PATH,
                                                                    COMPARISON_CONFIGURATION_PATH])
def test_positional_embedding_is_added_once(tmp_path: Path, configuration_path: Path) -> None:
    """
    Verifies that the positional embedding flag of a sequence run written before it is added to its temporal module,
    on as in the transformers of that time, that a configuration which already has it is left untouched, and that a
    configuration without temporal module is not changed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_path (Path): The shipped sequence configuration the old configuration is made from.
    """
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    del configuration["TemporalModel"]["positional_embedding"]
    old_configuration_path = tmp_path / "configuration.yaml"
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    assert add_positional_embedding_to_configuration(configuration_path=old_configuration_path) is True
    temporal_configuration = yaml.safe_load(stream=old_configuration_path.read_text())["TemporalModel"]
    assert temporal_configuration == {**configuration["TemporalModel"], "positional_embedding": True}
    assert add_positional_embedding_to_configuration(configuration_path=old_configuration_path) is False

    # A configuration of an image task keeps its settings
    image_text = EXAMPLE_CONFIGURATION_PATH.read_text()
    image_path = tmp_path / "image_configuration.yaml"
    image_path.write_text(data=image_text)
    assert add_positional_embedding_to_configuration(configuration_path=image_path) is False
    assert image_path.read_text() == image_text


def test_segmentation_settings_are_added_once(tmp_path: Path) -> None:
    """
    Verifies that the settings of a segmentation run written before the tiling, the deep supervision, the Lovász and
    boundary losses and the boundary metrics are inserted in their sections, off as in the runs of that time, that a
    configuration which already has them is left untouched, and that a configuration of another task is not changed.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration = yaml.safe_load(stream=SEGMENTATION_CONFIGURATION_PATH.read_text())
    del configuration["Data"]["tiling"]
    del configuration["SegmentationDecoder"]["deep_supervision_weights"]
    for key_name in ["lovasz_weight", "boundary_weight", "boundary_distance"]:
        del configuration["SegmentationLoss"][key_name]
    for key_name in ["original_resolution", "boundary_tolerance"]:
        del configuration["ExperimentConfiguration"]["evaluation"][key_name]
    old_configuration_path = tmp_path / "configuration.yaml"
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    assert add_segmentation_settings_to_configuration(configuration_path=old_configuration_path) is True
    migrated_configuration = yaml.safe_load(stream=old_configuration_path.read_text())
    assert migrated_configuration["Data"]["tiling"] == {"activated": False, "scale": 1.0, "overlap": 128}
    assert migrated_configuration["SegmentationDecoder"]["deep_supervision_weights"] == {}
    assert {key_name: migrated_configuration["SegmentationLoss"][key_name]
            for key_name in ["lovasz_weight", "boundary_weight", "boundary_distance"]} == {
        "lovasz_weight": 0.0, "boundary_weight": 0.0, "boundary_distance": 3}
    evaluation_configuration = migrated_configuration["ExperimentConfiguration"]["evaluation"]
    assert evaluation_configuration["original_resolution"] is False and evaluation_configuration[
        "boundary_tolerance"] == 2
    assert add_segmentation_settings_to_configuration(configuration_path=old_configuration_path) is False

    # A configuration of another task keeps its settings
    classification_text = CLASSIFICATION_CONFIGURATION_PATH.read_text()
    classification_path = tmp_path / "classification_configuration.yaml"
    classification_path.write_text(data=classification_text)
    assert add_segmentation_settings_to_configuration(configuration_path=classification_path) is False
    assert classification_path.read_text() == classification_text


# Sections holding a feature map normalization, with the keys from the section down to its settings
NORMALIZATION_SETTINGS_PATHS = {"Backbone": ["Backbone", "normalization"],
                                "RegionProposal": ["RegionProposal", "normalization"],
                                "Detector": ["Detector", "normalization"], "MaskHead": ["MaskHead", "normalization"],
                                "SegmentationDecoder": ["SegmentationDecoder"]}


@pytest.mark.parametrize(argnames="configuration_path", argvalues=[INSTANCE_SEGMENTATION_CONFIGURATION_PATH,
                                                                    SEGMENTATION_CONFIGURATION_PATH])
def test_batch_normalization_momentums_are_added_once(tmp_path: Path, configuration_path: Path) -> None:
    """
    Verifies that the batch normalization momentum of every normalized section of a run written before it (backbone,
    region proposal, detection and mask heads of INS, decoder of SEG) is added right after its feature map
    normalization, at the PyTorch default of 0.1 used by the runs of that time, and that a configuration which already
    has them is left untouched.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_path (Path): The shipped configuration the old configuration is made from.
    """
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    original_settings = {}
    for section_name, settings_path in NORMALIZATION_SETTINGS_PATHS.items():
        if section_name not in configuration:
            continue
        normalization_settings = configuration
        for key_name in settings_path:
            normalization_settings = normalization_settings[key_name]
        original_settings[section_name] = dict(normalization_settings)
        del normalization_settings["batch_normalization_momentum"]
    old_configuration_path = tmp_path / "configuration.yaml"
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    # Every normalized section gets its momentum right after its feature map normalization
    assert add_batch_normalization_momentums_to_configuration(configuration_path=old_configuration_path) is True
    migrated_configuration = yaml.safe_load(stream=old_configuration_path.read_text())
    for section_name, settings in original_settings.items():
        migrated_settings = migrated_configuration
        for key_name in NORMALIZATION_SETTINGS_PATHS[section_name]:
            migrated_settings = migrated_settings[key_name]
        assert migrated_settings == {**settings, "batch_normalization_momentum": 0.1}
        setting_names = list(migrated_settings)
        assert setting_names[setting_names.index("feature_map_normalization") + 1] == "batch_normalization_momentum"
    assert len(original_settings) >= 2
    assert add_batch_normalization_momentums_to_configuration(configuration_path=old_configuration_path) is False


def test_head_normalization_is_added_once(tmp_path: Path) -> None:
    """
    Verifies that the normalization of the hidden layers of the classification head of a run written before it is
    added right after its hidden layers, "none" with the PyTorch default momentum as in the runs of that time, that a
    configuration which already has it is left untouched, and that a configuration without classification head is not
    changed.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration = yaml.safe_load(stream=CLASSIFICATION_CONFIGURATION_PATH.read_text())
    original_head_configuration = dict(configuration["ClassificationHead"])
    del configuration["ClassificationHead"]["fully_connected_normalization"]
    del configuration["ClassificationHead"]["batch_normalization_momentum"]
    old_configuration_path = tmp_path / "configuration.yaml"
    old_configuration_path.write_text(data=yaml.safe_dump(data=configuration, sort_keys=False))

    # Both settings follow the hidden layers, the second migration changing nothing
    assert add_head_normalization_to_configuration(configuration_path=old_configuration_path) is True
    head_configuration = yaml.safe_load(stream=old_configuration_path.read_text())["ClassificationHead"]
    assert head_configuration == {**original_head_configuration, "fully_connected_normalization": "none",
                                  "batch_normalization_momentum": 0.1}
    setting_names = list(head_configuration)
    fully_connected_index = setting_names.index("fully_connected")
    assert setting_names[fully_connected_index + 1:fully_connected_index + 3] == ["fully_connected_normalization",
                                                                                  "batch_normalization_momentum"]
    assert add_head_normalization_to_configuration(configuration_path=old_configuration_path) is False

    # A configuration without classification head keeps its settings
    detection_text = EXAMPLE_CONFIGURATION_PATH.read_text()
    detection_path = tmp_path / "detection_configuration.yaml"
    detection_path.write_text(data=detection_text)
    assert add_head_normalization_to_configuration(configuration_path=detection_path) is False
    assert detection_path.read_text() == detection_text


@pytest.mark.parametrize(argnames=("boolean_normalization", "expected_normalization"),
                         argvalues=[(True, "batch"), (False, "none")])
def test_boolean_head_normalization_is_converted(tmp_path: Path, boolean_normalization: bool,
                                                expected_normalization: str) -> None:
    """
    Verifies that the True or False normalization of the hidden layers of a run written before the layer
    normalization becomes "batch" or "none": under the current comment in place of the comment of that time, every
    other line of the file being kept, and in a configuration written by yaml.safe_dump, without comments; a second
    migration changes nothing.

    Args:
        tmp_path (Path): The pytest temporary directory.
        boolean_normalization (bool): The normalization written before the layer normalization.
        expected_normalization (str): The normalization it becomes.
    """
    # Current file with the setting and comment of a run written before the layer normalization, whatever the
    # normalization the current file sets
    current_text = CLASSIFICATION_CONFIGURATION_PATH.read_text()
    current_normalization = yaml.safe_load(stream=current_text)["ClassificationHead"]["fully_connected_normalization"]
    current_setting = "".join(HEAD_NORMALIZATION_COMMENT_LINES) + \
        f"{HEAD_NORMALIZATION_LINE_START} \"{current_normalization}\"\n"
    old_setting = "".join(BOOLEAN_HEAD_NORMALIZATION_COMMENT_LINES) + \
        f"{HEAD_NORMALIZATION_LINE_START} {boolean_normalization}\n"
    assert current_text.count(current_setting) == 1
    old_configuration_path = tmp_path / "configuration.yaml"
    old_configuration_path.write_text(data=current_text.replace(current_setting, old_setting))

    # Only the setting and its comment change, the second migration changing nothing
    assert add_head_normalization_to_configuration(configuration_path=old_configuration_path) is True
    expected_setting = "".join(HEAD_NORMALIZATION_COMMENT_LINES) + \
        f"{HEAD_NORMALIZATION_LINE_START} \"{expected_normalization}\"\n"
    assert old_configuration_path.read_text() == current_text.replace(current_setting, expected_setting)
    assert add_head_normalization_to_configuration(configuration_path=old_configuration_path) is False

    # A configuration written by yaml.safe_dump, its keys in alphabetical order
    configuration = yaml.safe_load(stream=current_text)
    configuration["ClassificationHead"]["fully_connected_normalization"] = boolean_normalization
    dumped_configuration_path = tmp_path / "dumped_configuration.yaml"
    dumped_configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    assert add_head_normalization_to_configuration(configuration_path=dumped_configuration_path) is True
    migrated_configuration = yaml.safe_load(stream=dumped_configuration_path.read_text())
    assert migrated_configuration == {**configuration, "ClassificationHead": {
        **configuration["ClassificationHead"], "fully_connected_normalization": expected_normalization}}


def test_residual_nets_v2_is_added_once(tmp_path: Path) -> None:
    """
    Verifies that the pre-activation flag of the residual shortcuts is inserted, off as in every run written before
    it, right after the residual shortcuts flag of the backbone modules, every other line of the file being kept, and
    that a configuration which already has it is left untouched.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Current file without the flag and its comment
    current_text = CLASSIFICATION_CONFIGURATION_PATH.read_text()
    old_text = re.sub(pattern=r"\n    # Residual Nets v2 .*?    residual_nets_v2: \w+\n", repl="", string=current_text,
                      count=1, flags=re.DOTALL)
    assert "residual_nets_v2" not in old_text
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(data=old_text)

    # Only the flag is inserted, the second migration changing nothing
    assert add_residual_nets_v2_to_configuration(configuration_path=configuration_path) is True
    migrated_text = configuration_path.read_text()
    assert migrated_text.replace(RESIDUAL_NETS_V2_LINES, "", 1) == old_text
    module_names = list(yaml.safe_load(stream=migrated_text)["Backbone"]["modules"])
    assert module_names[module_names.index("residual_nets") + 1] == "residual_nets_v2"
    assert yaml.safe_load(stream=migrated_text)["Backbone"]["modules"]["residual_nets_v2"] is False
    assert add_residual_nets_v2_to_configuration(configuration_path=configuration_path) is False


def test_weight_settings_are_added_once(tmp_path: Path) -> None:
    """
    Verifies that the weight initialisation and the moving average of the weights are inserted, "default" and off as
    in every run written before them, right after the mixed precision flag of the optimization section, every other
    line of the file being kept, that a configuration missing only one of them gets that one, and that a
    configuration which has both is left untouched.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Current file without both settings and their comments
    current_text = CLASSIFICATION_CONFIGURATION_PATH.read_text()
    old_text = re.sub(pattern=r"  # Starting weights of the model.*?    decay: [^\n]*\n", repl="", string=current_text,
                      count=1, flags=re.DOTALL)
    assert "weight_initialization" not in old_text and "exponential_moving_average" not in old_text
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(data=old_text)

    # Both settings are inserted after the mixed precision flag, the second migration changing nothing
    assert add_weight_settings_to_configuration(configuration_path=configuration_path) is True
    migrated_text = configuration_path.read_text()
    assert migrated_text.replace(WEIGHT_INITIALIZATION_LINES + EXPONENTIAL_MOVING_AVERAGE_LINES, "", 1) == old_text
    optimization_configuration = yaml.safe_load(stream=migrated_text)["Optimization"]
    setting_names = list(optimization_configuration)
    mixed_precision_index = setting_names.index("mixed_precision")
    assert setting_names[mixed_precision_index + 1:mixed_precision_index + 3] == ["weight_initialization",
                                                                                  "exponential_moving_average"]
    assert optimization_configuration["weight_initialization"] == "default"
    assert optimization_configuration["exponential_moving_average"] == {"activated": False, "decay": 0.9998}
    assert add_weight_settings_to_configuration(configuration_path=configuration_path) is False

    # A configuration with the weight initialisation but without the moving average only gets the moving average
    assert old_text.count(MIXED_PRECISION_LINE_START) == 1
    configuration_path.write_text(data=old_text.replace(MIXED_PRECISION_LINE_START,
                                                        WEIGHT_INITIALIZATION_LINES + MIXED_PRECISION_LINE_START))
    assert add_weight_settings_to_configuration(configuration_path=configuration_path) is True
    assert configuration_path.read_text().count("weight_initialization:") == 1
    assert yaml.safe_load(stream=configuration_path.read_text())["Optimization"]["exponential_moving_average"] == {
        "activated": False, "decay": 0.9998}
