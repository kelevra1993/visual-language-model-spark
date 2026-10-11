"""
Tests of the command line of the training entry point vision_training/main.py: the configuration file of a launch is
found from its task folder (--task) and its file name (--name), and a missing task folder or file stops the launch
before any training.
"""

import sys
import pytest

from typing import List
from vision_training.main import CONFIGURATIONS_FOLDER, parse_configuration_path


def test_configuration_path_is_built_from_the_task_folder_and_the_file_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that --task and --name select the configuration file of the task folder.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
    """
    monkeypatch.setattr(target=sys, name="argv",
                        value=["main.py", "--task=classification", "--name=configuration_example.yaml"])

    assert parse_configuration_path() == CONFIGURATIONS_FOLDER / "classification" / "configuration_example.yaml"


@pytest.mark.parametrize("arguments", [["--task=classification", "--name=missing_configuration.yaml"],
                                       ["--task=missing_task", "--name=configuration_example.yaml"],
                                       ["--name=configuration_example.yaml"]])
def test_missing_task_folder_or_file_stops_the_launch(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
                                                      arguments: List[str]) -> None:
    """
    Verifies that a configuration file absent from the task folder, a task folder absent from
    vision_training/configurations/ or a missing argument stops the launch with an argument error, the error of a
    missing file listing the configuration files of the task folder.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the error message.
        arguments (List[str]): The command line arguments of the launch.
    """
    monkeypatch.setattr(target=sys, name="argv", value=["main.py"] + arguments)

    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_configuration_path()
    assert exit_information.value.code == 2
    if "--name=missing_configuration.yaml" in arguments:
        assert "configuration_example.yaml" in capsys.readouterr().err
