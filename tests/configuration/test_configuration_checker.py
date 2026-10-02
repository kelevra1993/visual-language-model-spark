import copy
import json
import pytest
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, List

from vision_training.configuration.configuration_loader import load_experiment_configuration

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")

# Configurations of the classification task, checked with a label mapping file written by the tests
CLASSIFICATION_CONFIGURATIONS_FOLDER = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                        "classification")


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
    (set_unknown_task, ["'ExperimentConfiguration.task' must be one of ['object_detection', 'classification'",
                        "got 'segmentation'."]),
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


def write_classification_configuration(temporary_directory: Path, configuration_name: str,
                                       modify_function: Callable[[Dict[str, Any]], None] | None) -> Path:
    """
    Writes a copy of a classification configuration whose label mapping file (giving the classes 0 to 3) is written
    in the temporary directory, the mapping files of the shipped configurations only existing once their datasets
    are created, then altered by the given function.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the mapping file.
        configuration_name (str): The file name of the classification configuration.
        modify_function (Callable[[Dict[str, Any]], None] | None): Function altering the raw configuration in place,
            or None to keep it.

    Returns:
        Path: The path of the written configuration file.
    """
    configuration = yaml.safe_load(stream=(CLASSIFICATION_CONFIGURATIONS_FOLDER / configuration_name).read_text())
    mapping_path = temporary_directory / "label_mapping.json"
    mapping_path.write_text(data=json.dumps(obj={"1": 0, "2": 1, "3": 2, "4": 3}))
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(mapping_path)
    if modify_function is not None:
        modify_function(configuration=configuration)

    configuration_path = temporary_directory / "classification_configuration.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def activate_dali(configuration: Dict[str, Any]) -> None:
    """
    Activates the DALI loaders, which only read the object detection format.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["Data"]["DALI"]["activated"] = True


def use_binary_labels(configuration: Dict[str, Any]) -> None:
    """
    Uses the binary label mapping, which only makes sense next to the background class of the object detector.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["Data"]["label_mapping"]["mode"] = "binary"


def add_detection_block(configuration: Dict[str, Any]) -> None:
    """
    Lists a detection block, although the classification head uses the final feature map.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["Backbone"]["detection_convolution_block_indices"] = [3]


def set_invalid_head(configuration: Dict[str, Any]) -> None:
    """
    Sets an unknown pooling, a hidden layer without unit, a dropout of 1, and unknown output mode and class weighting.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"].update({"global_pooling": "median", "fully_connected": [0], "dropout_rate": 1.0,
                                                "output_mode": "multi", "class_weighting": "inverse"})


def set_invalid_classification_evaluation(configuration: Dict[str, Any]) -> None:
    """
    Sets a top-k accuracy of zero classes and a target precision above 1.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"].update({"top_k": 0, "target_precision": 1.5})


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml",
                                                                   "configuration_synthetic_shapes.yaml"])
def test_classification_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the classification configurations shipped with the repository pass every consistency check, and
    that their label dictionary has no background class.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the classification configuration.
    """
    configuration_path = write_classification_configuration(temporary_directory=tmp_path,
                                                            configuration_name=configuration_name,
                                                            modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    assert experiment_configuration["task"] == "classification"
    assert model_configuration["Data"]["label_dictionary"] == {1: 0, 2: 1, 3: 2, 4: 3}


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (activate_dali, ["'Data.DALI.activated' must be False"]),
    (use_binary_labels, ["'Data.label_mapping.mode' must be 'multi_class' for the classification tasks"]),
    (add_detection_block, ["'Backbone.detection_convolution_block_indices' must be [] for the classification tasks"]),
    (set_invalid_head, ["'ClassificationHead.global_pooling' must be one of ['average', 'max'], got 'median'.",
                        "'ClassificationHead.fully_connected' must be a list of positive integers, got [0].",
                        "'ClassificationHead.dropout_rate' must be a number within [0, 1), got 1.0.",
                        "'ClassificationHead.output_mode' must be one of ['multi_class', 'multi_label']",
                        "'ClassificationHead.class_weighting' must be one of ['none', 'balanced']"]),
    (set_invalid_classification_evaluation, ["'ExperimentConfiguration.evaluation.top_k' must be an integer >= 1",
                                             "'ExperimentConfiguration.evaluation.target_precision' must be a "
                                             "probability within [0, 1], got 1.5."])])
def test_inconsistent_classification_configuration_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                               modify_function: Callable[[Dict[str, Any]], None],
                                                               expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the classification sections stops the loading with a short message naming
    the faulty key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_classification_configuration(temporary_directory=tmp_path,
                                                            configuration_name="configuration_synthetic_shapes.yaml",
                                                            modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


def test_classification_classes_must_be_contiguous(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a label mapping file whose class indices have a gap is rejected, the classifiers predicting the
    classes 0 to K-1.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_classification_configuration(temporary_directory=tmp_path,
                                                            configuration_name="configuration_synthetic_shapes.yaml",
                                                            modify_function=None)
    (tmp_path / "label_mapping.json").write_text(data=json.dumps(obj={"1": 0, "2": 2}))

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    assert "indexed 0 to K-1 without gaps, got [0, 2]." in capsys.readouterr().out
