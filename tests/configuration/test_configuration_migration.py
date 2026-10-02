import yaml

from pathlib import Path

from vision_training.configuration.configuration_migration import add_task_to_configuration, migrate_run_folder

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")


def write_old_run_folder(temporary_directory: Path) -> Path:
    """
    Writes a run folder as trained before the task-aware configuration: an old-style name, and a saved configuration
    without task (a copy of the example configuration, with binary labels so that no mapping file is needed).

    Args:
        temporary_directory (Path): The pytest temporary directory, used as the experiments root folder.

    Returns:
        Path: The run folder.
    """
    configuration = yaml.safe_load(stream=EXAMPLE_CONFIGURATION_PATH.read_text())
    del configuration["ExperimentConfiguration"]["task"]
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
