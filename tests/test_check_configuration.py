"""
Tests of the command line check of configuration files, vision_training/check_configuration.py: a valid file exits
with 0 and prints the run name it would train under, an invalid file exits with 1 and prints its problems, every file
of a call is reported, a file the loader cannot read (missing, unparsable, without its label mapping file) included,
and --check_paths requires the split folders with their labels.json and data folder (masks folder for the semantic
segmentation), the label mapping file and the pretrained runs with their checkpoint, never the experiment folder. The
loader of the trainings keeps printing the same red list and exiting with 1, and the module runs with python -m.
"""

import json
import shutil
import subprocess
import sys
import pytest
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional

from vision_training.check_configuration import main
from vision_training.configuration_processing.common_checks import PRETRAINED_COMPONENT_NAMES
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.configuration_processing.run_name import build_run_name
from vision_training.data.label_mapping import write_run_label_dictionary
from vision_training.models.model_transfer import TRANSFERABLE_COMPONENTS
from vision_training.trainers.checkpointing import get_checkpoint_file, update_checkpoint_registry

# Repository root, the folder a pipeline runs the command from
REPOSITORY_FOLDER = Path(__file__).absolute().parents[1]

# Configurations copied by the tests: the object detection example, whose binary labels need no mapping file, and the
# semantic segmentation example, whose splits also hold the class masks
DETECTION_CONFIGURATION_PATH = (REPOSITORY_FOLDER / "vision_training" / "configurations" / "object_detection" /
                                "configuration_example.yaml")
SEGMENTATION_CONFIGURATION_PATH = (REPOSITORY_FOLDER / "vision_training" / "configurations" / "semantic_segmentation" /
                                   "configuration_example.yaml")

# Image size of the invalid copies, not a multiple of 2 to the power of the 6 poolings of the detection backbone, and
# the start of the problem it gives
INVALID_IMAGE_SIZE_SETTINGS = {"Data.image_settings.size": 1000}
IMAGE_SIZE_PROBLEM = "'Data.image_settings.size' (1000) must be a multiple of 64"

# Iteration of the checkpoint of the source runs written by the tests
SOURCE_ITERATION = 2


def write_configuration(source_path: Path, output_path: Path, settings: Dict[str, Any]) -> Path:
    """
    Writes a copy of a shipped configuration with some settings replaced, so that each test hands the command line the
    file of one situation of a pipeline.

    Args:
        source_path (Path): The shipped configuration file copied.
        output_path (Path): The written file, its folder created if missing.
        settings (Dict[str, Any]): The new value of every replaced setting, keyed by its dotted key path.

    Returns:
        Path: The written file.
    """
    configuration = yaml.safe_load(stream=source_path.read_text())

    # Walk down to the section holding every replaced setting
    for key_path, value in settings.items():
        *parent_keys, setting_key = key_path.split(".")
        section = configuration
        for parent_key in parent_keys:
            section = section[parent_key]
        section[setting_key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data=yaml.safe_dump(data=configuration))
    return output_path


def create_split_folders(dataset_folder: Path, folder_names: List[str]) -> Dict[str, str]:
    """
    Creates the training, validation and test folders a training reads, each holding a 'labels.json' and the given
    folders, and gives the settings pointing a configuration at them.

    Args:
        dataset_folder (Path): The folder receiving the three split folders.
        folder_names (List[str]): The folders of every split, e.g. ["data"] or ["data", "masks"].

    Returns:
        Dict[str, str]: The folder of every split, keyed by the dotted key path of its setting.
    """
    split_settings = {}
    for split_name in ["train", "validation", "test"]:
        split_folder = dataset_folder / split_name
        split_folder.mkdir(parents=True)
        (split_folder / "labels.json").write_text(data=json.dumps(obj={"images": [], "annotations": [],
                                                                       "categories": []}))
        for folder_name in folder_names:
            (split_folder / folder_name).mkdir()
        split_settings[f"ExperimentConfiguration.{split_name}_data_folder"] = str(split_folder)
    return split_settings


def write_source_run(source_folder: Path, with_configuration: bool, with_checkpoint: bool) -> None:
    """
    Writes the run folder of a pretrained component: its 'training_configuration.yaml', and an empty placeholder of the
    weight file of iteration SOURCE_ITERATION named by the registry of its 'Weights' folder, the check reading only
    whether the files exist.

    Args:
        source_folder (Path): The run folder written.
        with_configuration (bool): Whether the run folder holds its 'training_configuration.yaml'.
        with_checkpoint (bool): Whether the run folder holds a checkpoint, its 'Weights' folder staying empty otherwise.
    """
    weights_directory = source_folder / "Weights"
    weights_directory.mkdir(parents=True)
    if with_configuration:
        (source_folder / "training_configuration.yaml").write_text(data=DETECTION_CONFIGURATION_PATH.read_text())
    if with_checkpoint:
        checkpoint_file = get_checkpoint_file(weights_directory=weights_directory, iteration=SOURCE_ITERATION)
        checkpoint_file.parent.mkdir()
        checkpoint_file.touch()
        update_checkpoint_registry(weights_directory=weights_directory, iteration=SOURCE_ITERATION)


def run_check(monkeypatch: pytest.MonkeyPatch, arguments: List[str]) -> int:
    """
    Runs the command line check with the given arguments, as `python -m vision_training.check_configuration` does.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        arguments (List[str]): The command line arguments, e.g. ["--configuration_path=...", "--check_paths"].

    Returns:
        int: The exit code of the check.
    """
    monkeypatch.setattr(target=sys, name="argv", value=["check_configuration.py"] + arguments)
    return main()


def test_valid_file_prints_its_run_name(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a valid file exits with 0 and prints, in green, the name of the run folder a training of it would get.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=DETECTION_CONFIGURATION_PATH)
    run_name = build_run_name(experiment_configuration=experiment_configuration,
                              model_configuration=model_configuration)
    capsys.readouterr()

    assert run_check(monkeypatch=monkeypatch, arguments=[f"--configuration_path={DETECTION_CONFIGURATION_PATH}"]) == 0
    printed_output = capsys.readouterr().out
    assert "Valid Configuration File : " in printed_output and f"Run Name : {run_name}" in printed_output
    assert "Invalid Configuration File" not in printed_output


def test_invalid_file_prints_its_problems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that an invalid file exits with 1 and prints its problems under the title of the trainings, without a run
    name.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_configuration(source_path=DETECTION_CONFIGURATION_PATH,
                                             output_path=tmp_path / "invalid.yaml",
                                             settings=INVALID_IMAGE_SIZE_SETTINGS)

    assert run_check(monkeypatch=monkeypatch, arguments=[f"--configuration_path={configuration_path}"]) == 1
    printed_output = capsys.readouterr().out
    assert "Invalid Configuration File : " in printed_output and IMAGE_SIZE_PROBLEM in printed_output
    assert "Run Name : " not in printed_output


def test_every_file_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a call checks every file it is given, whatever comes before: a file refused by the checks, an
    unparsable file, a file whose label mapping file is missing, a missing file, then a valid file, which is still
    checked and named, the call exiting with 1.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    invalid_path = write_configuration(source_path=DETECTION_CONFIGURATION_PATH, output_path=tmp_path / "invalid.yaml",
                                       settings=INVALID_IMAGE_SIZE_SETTINGS)
    unparsable_path = tmp_path / "unparsable.yaml"
    unparsable_path.write_text(data="ExperimentConfiguration: [unclosed\n")
    unmapped_path = write_configuration(source_path=DETECTION_CONFIGURATION_PATH,
                                        output_path=tmp_path / "unmapped.yaml",
                                        settings={"Data.label_mapping.mode": "multi_class",
                                                  "Data.label_mapping.mapping_file_path": str(tmp_path / "none.json")})
    checked_paths = [invalid_path, unparsable_path, unmapped_path, tmp_path / "missing.yaml",
                     DETECTION_CONFIGURATION_PATH]

    assert run_check(monkeypatch=monkeypatch,
                     arguments=[f"--configuration_path={checked_path}" for checked_path in checked_paths]) == 1

    # One block per invalid file, each naming its problem on one line, then the run name of the valid file
    printed_output = capsys.readouterr().out
    assert printed_output.count("Invalid Configuration File : ") == 4
    for expected_problem in [IMAGE_SIZE_PROBLEM, "  - ValueError: Error parsing YAML file at",
                             "FileNotFoundError: Label mapping file not found at",
                             "FileNotFoundError: Configuration file not found at"]:
        assert expected_problem in printed_output
    assert "\n  - while parsing a flow sequence" not in printed_output
    assert printed_output.count("Valid Configuration File : ") == 1
    assert printed_output.rindex("Run Name : ") > printed_output.rindex("Invalid Configuration File : ")


def test_training_loader_prints_the_same_problems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the loader of the trainings still stops an invalid file with the exit code 1, after printing exactly
    the block the command line prints for it.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_configuration(source_path=DETECTION_CONFIGURATION_PATH,
                                             output_path=tmp_path / "invalid.yaml",
                                             settings=INVALID_IMAGE_SIZE_SETTINGS)
    assert run_check(monkeypatch=monkeypatch, arguments=[f"--configuration_path={configuration_path}"]) == 1
    check_output = capsys.readouterr().out

    with pytest.raises(expected_exception=SystemExit) as exit_information:
        load_experiment_configuration(configuration_path=configuration_path)
    assert exit_information.value.code == 1
    assert capsys.readouterr().out == check_output


@pytest.mark.parametrize(argnames=("removed_path", "expected_problem"), argvalues=[
    (None, None),
    ("validation", "'ExperimentConfiguration.validation_data_folder' is not a folder : "),
    ("train/labels.json", "'ExperimentConfiguration.train_data_folder' holds no 'labels.json' : "),
    ("test/data", "'ExperimentConfiguration.test_data_folder' holds no 'data' folder : ")])
def test_check_paths_requires_the_split_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture, removed_path: Optional[str],
                                                expected_problem: Optional[str]) -> None:
    """
    Verifies that --check_paths requires every split folder with its labels.json and data folder, which the loading
    alone never reads, and accepts an experiment folder that does not exist yet, the training creating it.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        removed_path (Optional[str]): The split folder or file removed, relative to the dataset folder, None for none.
        expected_problem (Optional[str]): The start of the problem printed, None when the file is valid.
    """
    split_settings = create_split_folders(dataset_folder=tmp_path / "dataset", folder_names=["data"])
    configuration_path = write_configuration(
        source_path=DETECTION_CONFIGURATION_PATH, output_path=tmp_path / "configuration.yaml",
        settings={**split_settings, "ExperimentConfiguration.experiment_folder": str(tmp_path / "missing_experiments")})
    # Remove one split folder, labels file or data folder
    removed_entry = tmp_path / "dataset" / removed_path if removed_path is not None else None
    if removed_entry is not None and removed_entry.is_dir():
        shutil.rmtree(path=removed_entry)
    elif removed_entry is not None:
        removed_entry.unlink()

    # The loading alone reads none of these paths
    assert run_check(monkeypatch=monkeypatch, arguments=[f"--configuration_path={configuration_path}"]) == 0
    capsys.readouterr()

    exit_code = run_check(monkeypatch=monkeypatch, arguments=[f"--configuration_path={configuration_path}",
                                                              "--check_paths"])
    printed_output = capsys.readouterr().out
    if expected_problem is None:
        assert exit_code == 0 and "Run Name : " in printed_output
    else:
        assert exit_code == 1 and expected_problem in printed_output and "Run Name : " not in printed_output


def test_check_paths_requires_the_segmentation_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that --check_paths also requires the masks folder of every split of the semantic segmentation, whose
    loaders read the class masks there.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.write_text(data=json.dumps(obj={"1": 1, "2": 2, "3": 3, "4": 4}))
    split_settings = create_split_folders(dataset_folder=tmp_path / "dataset", folder_names=["data"])
    configuration_path = write_configuration(
        source_path=SEGMENTATION_CONFIGURATION_PATH, output_path=tmp_path / "configuration.yaml",
        settings={**split_settings, "Data.label_mapping.mapping_file_path": str(mapping_path)})
    arguments = [f"--configuration_path={configuration_path}", "--check_paths"]

    # Every split without its masks folder is listed
    assert run_check(monkeypatch=monkeypatch, arguments=arguments) == 1
    printed_output = capsys.readouterr().out
    for split_key in ["train_data_folder", "validation_data_folder", "test_data_folder"]:
        assert f"'ExperimentConfiguration.{split_key}' holds no 'masks' folder : " in printed_output

    # The masks folders make the file valid
    for split_name in ["train", "validation", "test"]:
        (tmp_path / "dataset" / split_name / "masks").mkdir()
    assert run_check(monkeypatch=monkeypatch, arguments=arguments) == 0


def test_check_paths_requires_the_label_mapping_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that --check_paths requires the mapping file of the 'multi_class' mode even when the loading takes the
    classes saved in a run folder (the 'label_dictionary.json' next to the configuration), a new training building its
    classes from the file.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    run_folder = tmp_path / "run"
    split_settings = create_split_folders(dataset_folder=tmp_path / "dataset", folder_names=["data"])
    configuration_path = write_configuration(
        source_path=DETECTION_CONFIGURATION_PATH, output_path=run_folder / "training_configuration.yaml",
        settings={**split_settings, "Data.label_mapping.mode": "multi_class",
                  "Data.label_mapping.mapping_file_path": str(tmp_path / "moved_label_mapping.json")})
    write_run_label_dictionary(label_dictionary={"background": 0, 1: 1}, run_folder=run_folder)

    # The classes of the run folder load the file, the missing mapping file being required by --check_paths only
    assert run_check(monkeypatch=monkeypatch, arguments=[f"--configuration_path={configuration_path}"]) == 0
    capsys.readouterr()
    assert run_check(monkeypatch=monkeypatch, arguments=[f"--configuration_path={configuration_path}",
                                                         "--check_paths"]) == 1
    assert "'Data.label_mapping.mapping_file_path' is not a file : " in capsys.readouterr().out


@pytest.mark.parametrize(argnames=("section_name", "source_state", "iteration", "expected_problem"), argvalues=[
    ("Backbone", "complete", SOURCE_ITERATION, None),
    ("Backbone", "complete", None, None),
    ("Detector", "complete", SOURCE_ITERATION + 1, f"'Detector.pretrained.iteration' ({SOURCE_ITERATION + 1}): No "
                                                   f"checkpoint found : "),
    ("Backbone", "without_checkpoint", None, "'Backbone.pretrained.iteration' (null, the last checkpoint): No "
                                             "checkpoint found in the run : "),
    ("RegionProposal", "without_configuration", SOURCE_ITERATION, "'RegionProposal.pretrained.experiment_folder' "
                                                                  "holds no 'training_configuration.yaml' : "),
    ("Backbone", "missing", SOURCE_ITERATION, "'Backbone.pretrained.experiment_folder' is not a folder : ")])
def test_check_paths_requires_the_pretrained_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture, section_name: str, source_state: str,
                                                  iteration: Optional[int], expected_problem: Optional[str]) -> None:
    """
    Verifies that --check_paths requires the source run of every pretrained section, with its configuration and the
    checkpoint of its iteration, the last one of the registry when it is null, while the loading alone never reads it.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the command line.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        section_name (str): The section whose component is transferred.
        source_state (str): The source run folder written: "complete", "without_checkpoint", "without_configuration",
            or "missing" for none.
        iteration (Optional[int]): The checkpoint iteration requested, None for the last one.
        expected_problem (Optional[str]): The start of the problem printed, None when the file is valid.
    """
    source_folder = tmp_path / "source_run"
    if source_state != "missing":
        write_source_run(source_folder=source_folder, with_configuration=source_state != "without_configuration",
                         with_checkpoint=source_state != "without_checkpoint")
    split_settings = create_split_folders(dataset_folder=tmp_path / "dataset", folder_names=["data"])
    configuration_path = write_configuration(
        source_path=DETECTION_CONFIGURATION_PATH, output_path=tmp_path / "configuration.yaml",
        settings={**split_settings, f"{section_name}.pretrained.experiment_folder": str(source_folder),
                  f"{section_name}.pretrained.iteration": iteration})

    # The loading alone never reads the source run
    assert run_check(monkeypatch=monkeypatch, arguments=[f"--configuration_path={configuration_path}"]) == 0
    capsys.readouterr()

    exit_code = run_check(monkeypatch=monkeypatch, arguments=[f"--configuration_path={configuration_path}",
                                                              "--check_paths"])
    printed_output = capsys.readouterr().out
    if expected_problem is None:
        assert exit_code == 0 and "Run Name : " in printed_output
    else:
        assert exit_code == 1 and expected_problem in printed_output


def test_checked_pretrained_sections_are_those_the_trainer_loads() -> None:
    """
    Verifies that the sections whose source runs --check_paths requires are the sections the trainer loads from other
    runs, so that a component made transferable is checked too.
    """
    assert set(PRETRAINED_COMPONENT_NAMES) == set(TRANSFERABLE_COMPONENTS)


def test_command_line_exit_code(tmp_path: Path) -> None:
    """
    Verifies the command a pipeline runs from the repository root, python -m vision_training.check_configuration, on a
    valid and an invalid file: both are reported and the process exits with 1, the exit codes of main being checked by
    the other tests.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    invalid_path = write_configuration(source_path=DETECTION_CONFIGURATION_PATH, output_path=tmp_path / "invalid.yaml",
                                       settings=INVALID_IMAGE_SIZE_SETTINGS)
    command = [sys.executable, "-m", "vision_training.check_configuration",
               f"--configuration_path={DETECTION_CONFIGURATION_PATH}", f"--configuration_path={invalid_path}"]

    completed_process = subprocess.run(args=command, cwd=REPOSITORY_FOLDER, capture_output=True, text=True)
    assert completed_process.returncode == 1
    assert "Run Name : " in completed_process.stdout and IMAGE_SIZE_PROBLEM in completed_process.stdout
