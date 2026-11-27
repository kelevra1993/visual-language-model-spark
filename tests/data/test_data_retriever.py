"""
Tests of vision_training/data/data_retriever.py, the FiftyOne Zoo download of COCO 2017 and Open Images V7, without
network and without FiftyOne: the script prints its help when launched without arguments, reads every setting of
retrieve_dataset from its command line, DEFAULT_DATASET_DIRECTORY and DEFAULT_RETRIEVER_SETTINGS for those not given,
refuses a missing dataset name or an unknown value, and runs its parser where FiftyOne is not installed (the Docker
test stage), FiftyOne being imported only by retrieve_dataset, which hands its settings to the zoo.
"""

import inspect
import subprocess
import sys
import types

import pytest

from pathlib import Path
from typing import Any, Dict, List

from vision_training.data.data_retriever import (DEFAULT_DATASET_DIRECTORY, DEFAULT_RETRIEVER_SETTINGS,
                                                 parse_retriever_arguments, retrieve_dataset)

# Repository root, the folder the scripts run from
REPOSITORY_FOLDER = Path(__file__).absolute().parents[2]

# Every argument of the retriever script, which its help must describe
RETRIEVER_SCRIPT_ARGUMENTS = ["--dataset_name", "--dataset_directory", "--maximum_samples", "--splits",
                              "--label_types"]

# Dataset of the parser tests, which download nothing
SCRIPT_DATASET_NAME = "coco-2017"

# Python run in a fresh process where FiftyOne cannot be imported, as in the Docker image, printing the dataset name
# the parser of the script reads
PARSER_WITHOUT_FIFTYONE_CODE = ("import sys; sys.modules['fiftyone'] = None; "
                                "from vision_training.data.data_retriever import parse_retriever_arguments; "
                                f"print(parse_retriever_arguments(arguments=['--dataset_name', "
                                f"'{SCRIPT_DATASET_NAME}']).dataset_name)")


class ZooDownload:
    """
    Stands in for fiftyone.zoo.load_zoo_dataset: keeps the keyword arguments of every call and returns a marker
    instead of downloading anything.
    """

    def __init__(self) -> None:
        """
        Starts without any call.
        """
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **keyword_arguments: Any) -> str:
        """
        Records the call.

        Args:
            **keyword_arguments (Any): The arguments of the zoo download.

        Returns:
            str: The marker standing for the FiftyOne dataset.
        """
        self.calls.append(keyword_arguments)
        return "zoo dataset"


def test_retriever_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the retriever script launched without any argument prints the help of every one of its arguments
    and stops without downloading anything.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_retriever_arguments(arguments=[])
    assert exit_information.value.code == 0
    help_text = capsys.readouterr().out
    assert [argument for argument in RETRIEVER_SCRIPT_ARGUMENTS if argument not in help_text] == []


def test_retriever_arguments_default_to_the_retriever_settings() -> None:
    """
    Verifies that the retriever script given only the dataset name downloads into the 'datasets' folder of the
    repository with DEFAULT_RETRIEVER_SETTINGS (1,000 images of every split), and that the parsed arguments are
    exactly the parameters of retrieve_dataset, so that the main block passes them all.
    """
    parsed_arguments = parse_retriever_arguments(arguments=["--dataset_name", SCRIPT_DATASET_NAME])
    assert vars(parsed_arguments) == {"dataset_name": SCRIPT_DATASET_NAME,
                                      "dataset_directory": DEFAULT_DATASET_DIRECTORY, **DEFAULT_RETRIEVER_SETTINGS}
    assert DEFAULT_DATASET_DIRECTORY == REPOSITORY_FOLDER / "datasets"
    assert set(vars(parsed_arguments)) == set(inspect.signature(retrieve_dataset).parameters)


def test_retriever_arguments_reach_every_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that every setting of retrieve_dataset can be given on the command line of the retriever script: 'None'
    as maximum number of images for the full dataset, one split and one label type, and a folder starting with a '~'
    the shell leaves after '=' expanded to the home folder.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    parsed_arguments = parse_retriever_arguments(arguments=[
        "--dataset_name", "open-images-v7", "--dataset_directory=~/datasets", "--maximum_samples", "None", "--splits",
        "validation", "--label_types", "detections"])
    assert vars(parsed_arguments) == {"dataset_name": "open-images-v7", "dataset_directory": tmp_path / "datasets",
                                      "maximum_samples": None, "splits": ["validation"],
                                      "label_types": ["detections"]}
    assert parse_retriever_arguments(arguments=["--dataset_name", SCRIPT_DATASET_NAME, "--maximum_samples",
                                                "20"]).maximum_samples == 20


@pytest.mark.parametrize(argnames=("arguments", "error_text"), argvalues=[
    (["--maximum_samples", "10"], "the following arguments are required: --dataset_name"),
    (["--dataset_name", "coco-2014"], "invalid choice: 'coco-2014'"),
    (["--dataset_name", SCRIPT_DATASET_NAME, "--maximum_samples", "many"], "expected None or a positive integer"),
    (["--dataset_name", SCRIPT_DATASET_NAME, "--maximum_samples", "0"], "expected None or a positive integer"),
    (["--dataset_name", SCRIPT_DATASET_NAME, "--splits", "val"], "invalid choice: 'val'")])
def test_retriever_script_refuses_a_missing_dataset_or_unknown_values(arguments: List[str], error_text: str,
                                                                     capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the retriever script stops with its usage and the error, before anything is downloaded, when the
    dataset name is missing or a dataset, a number of images or a split is not one the retriever accepts.

    Args:
        arguments (List[str]): The command line arguments.
        error_text (str): A part of the expected error message.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_retriever_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert error_text in capsys.readouterr().err


def test_retriever_parser_runs_without_fiftyone() -> None:
    """
    Verifies that the module and its parser load in a process where FiftyOne cannot be imported, as in the Docker
    image that leaves the datasets group out, so that FiftyOne is only needed to download a dataset.
    """
    completed_process = subprocess.run(args=[sys.executable, "-c", PARSER_WITHOUT_FIFTYONE_CODE],
                                       cwd=REPOSITORY_FOLDER, capture_output=True, text=True)
    assert completed_process.returncode == 0, completed_process.stderr
    assert completed_process.stdout.strip() == SCRIPT_DATASET_NAME


def test_retrieve_dataset_hands_its_settings_to_the_zoo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that retrieve_dataset imports FiftyOne when it runs, points the zoo folder of FiftyOne to the dataset
    folder and asks the zoo for the dataset with its splits, label types and number of images, the splits and label
    types defaulting to those of DEFAULT_RETRIEVER_SETTINGS, with stand-in modules instead of FiftyOne.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the dataset folder.
        monkeypatch (pytest.MonkeyPatch): Puts the stand-in FiftyOne modules in place of the real ones.
    """
    zoo_download = ZooDownload()
    fiftyone_module = types.ModuleType(name="fiftyone")
    fiftyone_module.config = types.SimpleNamespace(dataset_zoo_dir=None)
    zoo_module = types.ModuleType(name="fiftyone.zoo")
    zoo_module.load_zoo_dataset = zoo_download
    monkeypatch.setitem(sys.modules, "fiftyone", fiftyone_module)
    monkeypatch.setitem(sys.modules, "fiftyone.zoo", zoo_module)

    dataset = retrieve_dataset(dataset_name=SCRIPT_DATASET_NAME, dataset_directory=str(tmp_path), maximum_samples=5)
    assert dataset == "zoo dataset"
    assert fiftyone_module.config.dataset_zoo_dir == str(tmp_path)
    assert len(zoo_download.calls) == 1
    number_workers = zoo_download.calls[0].pop("num_workers")
    assert isinstance(number_workers, int) and number_workers >= 0
    assert zoo_download.calls[0] == {"name_or_url": SCRIPT_DATASET_NAME, "splits": DEFAULT_RETRIEVER_SETTINGS["splits"],
                                     "label_types": DEFAULT_RETRIEVER_SETTINGS["label_types"], "max_samples": 5,
                                     "persistent": True}
