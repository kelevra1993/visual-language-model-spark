import copy
import pytest
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, List

from vision_training.configuration.configuration_loader import load_experiment_configuration

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")


def write_modified_configuration(temporary_directory: Path,
                                 modify_function: Callable[[Dict[str, Any]], None]) -> Path:
    """
    Writes a copy of the example configuration, altered by the given function, so that each test can check how
    load_experiment_configuration reacts to one specific inconsistency.

    Args:
        temporary_directory (Path): The pytest temporary directory in which the configuration file is written.
        modify_function (Callable[[Dict[str, Any]], None]): Function altering the raw configuration in place.

    Returns:
        Path: The path of the written configuration file.
    """
    with open(EXAMPLE_CONFIGURATION_PATH, "r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    modified_configuration = copy.deepcopy(configuration)
    modify_function(configuration=modified_configuration)

    configuration_path = temporary_directory / "configuration.yaml"
    with open(configuration_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(modified_configuration, file)

    return configuration_path


def set_image_size(configuration: Dict[str, Any]) -> None:
    """
    Sets an image size that is not a power of two.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Data"]["image_settings"]["size"] = 1000


def set_overlapping_thresholds(configuration: Dict[str, Any]) -> None:
    """
    Makes the background IoU range overlap the foreground IoU range.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["RegionProposal"]["background_iou_threshold"]["max"] = 0.6


def remove_anchor_level(configuration: Dict[str, Any]) -> None:
    """
    Removes one detection level from the anchors only.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    del configuration["Anchors"]["scales_and_ratios"]["7"]


def set_string_flag(configuration: Dict[str, Any]) -> None:
    """
    Writes a module flag as a quoted string instead of a boolean.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Backbone"]["modules"]["inception"] = "False"


def set_detector_samples(configuration: Dict[str, Any]) -> None:
    """
    Asks the detector for more training samples than the proposals kept after NMS.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Detector"]["total_training_samples"] = 1000


def set_intermediate_weight_saving(configuration: Dict[str, Any]) -> None:
    """
    Sets intermediate checkpoints that are not more frequent than the evaluated ones. Both values are set
    explicitly so that the expected message does not depend on the saving values of the example configuration.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["weight_saving_iterations"] = 10000
    configuration["ExperimentConfiguration"]["intermediate_weight_saving_iterations"] = 10000


def set_evaluation_score_threshold(configuration: Dict[str, Any]) -> None:
    """
    Sets an evaluation score threshold outside of [0, 1].

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"]["score_threshold"] = 1.5


def set_unknown_task(configuration: Dict[str, Any]) -> None:
    """
    Sets a task that no trainer implements.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["task"] = "segmentation"


def remove_task_key(configuration: Dict[str, Any]) -> None:
    """
    Removes the task of the configuration, as in the configurations written before the task-aware configuration.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    del configuration["ExperimentConfiguration"]["task"]


def set_inception_channels(configuration: Dict[str, Any]) -> None:
    """
    Enables inception on a Backbone block whose output channels are not a multiple of 8.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Backbone"]["modules"]["inception"] = True
    configuration["Backbone"]["convolutions"]["4"] = [4, 20]


def set_multiple_errors(configuration: Dict[str, Any]) -> None:
    """
    Introduces two unrelated inconsistencies at once.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Data"]["image_settings"]["size"] = 1000
    configuration["Detector"]["number_training_positives"] = 500


def remove_module_key(configuration: Dict[str, Any]) -> None:
    """
    Removes a required module flag from the Detector section.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    del configuration["Detector"]["modules"]["residual_nets"]


def test_example_configuration_is_consistent() -> None:
    """
    Verifies that the example configuration shipped with the repository passes every consistency check.
    """
    load_experiment_configuration(configuration_path=EXAMPLE_CONFIGURATION_PATH)


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_image_size, ["'Data.image_settings.size' must be a power of two"]),
    (set_overlapping_thresholds, ["'RegionProposal.background_iou_threshold.max' (0.6) must be <="]),
    (remove_anchor_level, ["'Anchors.scales_and_ratios' levels ['5', '6'] do not match"]),
    (set_string_flag, ["'Backbone.modules.inception' must be True or False"]),
    (set_detector_samples, ["'Detector.total_training_samples' (1000) must be <="]),
    (set_inception_channels, ["'Backbone.convolutions.4' has 20 output channels"]),
    (set_intermediate_weight_saving, ["'ExperimentConfiguration.intermediate_weight_saving_iterations' (10000) "
                                      "must be smaller than"]),
    (set_unknown_task, ["'ExperimentConfiguration.task' must be one of ['object_detection'], got 'segmentation'."]),
    (set_evaluation_score_threshold, ["'ExperimentConfiguration.evaluation.score_threshold' must be a probability "
                                      "within [0, 1], got 1.5."]),
    (set_multiple_errors, ["'Data.image_settings.size' must be a power of two",
                           "'Detector.number_training_positives' (500) must be <="])])
def test_inconsistent_configuration_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                modify_function: Callable[[Dict[str, Any]], None],
                                                expected_messages: List[str]) -> None:
    """
    Verifies that each inconsistency stops the loading with a short message naming the faulty key, and that
    several inconsistencies are all printed at once.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_modified_configuration(temporary_directory=tmp_path, modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    # Every expected problem must be printed, without any traceback
    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


def test_missing_configuration_key_is_reported_with_its_path(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a key missing from the configuration is reported with its full dotted path.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_modified_configuration(temporary_directory=tmp_path, modify_function=remove_module_key)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    assert "Missing configuration key 'Detector.modules.residual_nets'." in capsys.readouterr().out


def test_missing_task_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a configuration without task, written before the task-aware configuration, is rejected with the
    path of the missing key (scripts/configuration/migrate_configuration.py adds it to the runs of that time).

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_modified_configuration(temporary_directory=tmp_path, modify_function=remove_task_key)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    assert "Missing configuration key 'ExperimentConfiguration.task'." in capsys.readouterr().out
