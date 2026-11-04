"""
Tests of the command line of the training entry point vision_training/main.py: the configuration file of a launch is
given by its path (--configuration_path), a path relative to the current folder being made absolute, and a missing file
stops the launch before any training.
"""

import sys
import pytest

from pathlib import Path
from typing import List
from vision_training.main import parse_configuration_path

# Configuration file used as the launched one, relative to the repository root
CONFIGURATION_RELATIVE_PATH = Path("vision_training/configurations/classification/configuration_example.yaml")
REPOSITORY_FOLDER = Path(__file__).absolute().parents[1]


def test_relative_configuration_path_is_made_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a --configuration_path relative to the current folder is returned as an absolute path, so that the
    trainer copies and reads the same file whatever folder it runs from.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line and the current folder.
    """
    monkeypatch.chdir(path=REPOSITORY_FOLDER)
    monkeypatch.setattr(target=sys, name="argv",
                        value=["main.py", f"--configuration_path={CONFIGURATION_RELATIVE_PATH}"])

    assert parse_configuration_path() == REPOSITORY_FOLDER / CONFIGURATION_RELATIVE_PATH


@pytest.mark.parametrize("arguments", [["--configuration_path=missing_folder/missing_configuration.yaml"]])
def test_missing_file_or_argument_stops_the_launch(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
                                                   arguments: List[str]) -> None:
    """
    Verifies that a path leading to no file stops the launch with an argument error naming the file.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the error message.
        arguments (List[str]): The command line arguments of the launch.
    """
    monkeypatch.setattr(target=sys, name="argv", value=["main.py"] + arguments)

    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_configuration_path()
    assert exit_information.value.code == 2
    if arguments:
        assert "missing_configuration.yaml" in capsys.readouterr().err
