import json
import yaml

from pathlib import Path

from vision_training.configuration.configuration_migration import (
    add_evaluation_outputs_to_configuration, add_freeze_normalization_parameters_to_configuration,
    add_optimization_section_to_configuration, add_pretrained_settings_to_configuration, add_task_to_configuration,
    migrate_run_folder)

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")
SEQUENCE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                               "sequence_classification" / "configuration_synthetic_sequences.yaml")

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
    Writes a run folder as trained before the task-aware configuration, the backbone transfer and the optimization
    settings: an old-style name, and a saved configuration without task, pretrained backbone settings nor
    optimization section (a copy of the example configuration, with binary labels so that no mapping file is
    needed).

    Args:
        temporary_directory (Path): The pytest temporary directory, used as the experiments root folder.

    Returns:
        Path: The run folder.
    """
    configuration = yaml.safe_load(stream=EXAMPLE_CONFIGURATION_PATH.read_text())
    del configuration["ExperimentConfiguration"]["task"]
    del configuration["Backbone"]["pretrained"]
    del configuration["Optimization"]
    configuration["Data"]["label_mapping"] = {"mode": "binary", "mapping_file_path": None}

    run_folder = temporary_directory / "IN-KAR-1024_old_run_name"
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
    Verifies that migrating an old run upgrades its configuration, leaves its folder in place unless renaming is
    asked, renames it to the name the current code builds (starting with the task code), and that migrating it again
    changes nothing.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    run_folder = write_old_run_folder(temporary_directory=tmp_path)

    assert migrate_run_folder(run_folder=run_folder, task="object_detection", rename_folder=False) == run_folder
    assert run_folder.exists()

    renamed_folder = migrate_run_folder(run_folder=run_folder, task="object_detection", rename_folder=True)
    assert renamed_folder.name.startswith("DET_")
    assert renamed_folder.exists() and not run_folder.exists()

    assert migrate_run_folder(run_folder=renamed_folder, task="object_detection", rename_folder=True) == renamed_folder


def test_pretrained_settings_are_added_once(tmp_path: Path) -> None:
    """
    Verifies that the pretrained backbone settings of a run trained from random weights, normalization flag included,
    are added to a configuration written before the backbone transfer, keeping its other backbone settings, that a
    configuration which already has them is left untouched, and that a detection configuration, which has no
    temporal module nor classification head, is not changed by their migrations.

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
    for key_name in ["save_predictions", "error_dumps"]:
        del configuration["ExperimentConfiguration"]["evaluation"][key_name]
    for section_name in ["Backbone", "TemporalModel", "ClassificationHead"]:
        del configuration[section_name]["pretrained"]
    del configuration["Optimization"]
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
