"""
Tests of the command line of the training entry point vision_training/main.py: the configuration file of a launch is
given by its path (--configuration_path), a path relative to the current folder being made absolute, a missing file
stops the launch before any training, and the configuration and the trainer are built with every manifest parsed once
(read_manifests_once), the shared manifests being freed before the training starts.
"""

import sys
import pytest

from pathlib import Path
from typing import Any, Dict, List
from vision_training import main as main_module
from vision_training.data import manifest_cache
from vision_training.main import parse_configuration_path

# Configuration file used as the launched one, relative to the repository root
CONFIGURATION_RELATIVE_PATH = Path("vision_training/configurations/classification/configuration_example.yaml")
REPOSITORY_FOLDER = Path(__file__).absolute().parents[1]


class LaunchRecordingTrainer:
    """
    Stands for the trainer of a task in the test of the entry point, recording whether the manifests of the launch were
    shared while it was built and while it trained.
    """

    # Whether the manifests were shared at every construction and at every training, in order
    shared_while_built: List[bool] = []
    shared_while_training: List[bool] = []

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: Path) -> None:
        """
        Records whether the manifests are shared while the trainer is built.

        Args:
            experiment_configuration (Dict[str, Any]): The experiment configuration, unused.
            model_configuration (Dict[str, Any]): The model configuration, unused.
            configuration_path (Path): The launched configuration file, unused.
        """
        LaunchRecordingTrainer.shared_while_built.append(manifest_cache.shared_manifests is not None)

    def run_training_loop(self) -> None:
        """
        Records whether the manifests are still shared when the training starts.
        """
        LaunchRecordingTrainer.shared_while_training.append(manifest_cache.shared_manifests is not None)


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


def test_launch_shares_the_manifests_until_the_trainer_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the entry point loads the configuration and builds the trainer while the manifests are parsed once
    and shared (read_manifests_once), and that they are freed before the training starts, so that the loader workers
    forked by the first batch never inherit them.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line, the configuration loader and
            the trainer of the task.
    """
    configuration_states = []
    LaunchRecordingTrainer.shared_while_built, LaunchRecordingTrainer.shared_while_training = [], []
    monkeypatch.setattr(target=sys, name="argv",
                        value=["main.py", f"--configuration_path={REPOSITORY_FOLDER / CONFIGURATION_RELATIVE_PATH}"])
    monkeypatch.setattr(target=main_module, name="enable_expandable_cuda_segments", value=lambda: None)
    monkeypatch.setattr(target=main_module, name="load_experiment_configuration", value=lambda configuration_path: (
        configuration_states.append(manifest_cache.shared_manifests is not None) or ({"task": "classification"}, {})))
    monkeypatch.setattr(target=main_module, name="get_trainer_class", value=lambda task: LaunchRecordingTrainer)

    main_module.main()

    assert configuration_states == [True] and LaunchRecordingTrainer.shared_while_built == [True]
    assert LaunchRecordingTrainer.shared_while_training == [False]
