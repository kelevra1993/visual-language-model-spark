"""
Tests of the configuration checks of vision_training/configuration_processing/configuration_checker.py, run through
load_experiment_configuration: the configurations shipped for every task are accepted, while inconsistent settings
(detection, classification, sequence, comparison, segmentation, normalization, pretrained transfers, optimization) and
missing keys are rejected with errors naming their path.
"""

import copy
import json
import pytest
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, List

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")

# Folder of the configurations of every task, checked with a label mapping file written by the tests for the tasks
# whose mapping file only exists once their dataset is created
CONFIGURATIONS_FOLDER = Path(__file__).parents[2] / "vision_training" / "configurations"

# Label mapping file written by the tests: the categories 1 to 4 give the classes 0 to 3 of the classification and
# sequence tasks, or the classes 1 to 4 of the semantic segmentation, next to its background class 0
CLASSIFICATION_LABEL_MAPPING = {"1": 0, "2": 1, "3": 2, "4": 3}
SEGMENTATION_LABEL_MAPPING = {"1": 1, "2": 2, "3": 3, "4": 4}


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


def set_invalid_detection_error_analysis(configuration: Dict[str, Any]) -> None:
    """
    Sets a target precision of the class thresholds outside of [0, 1], the error images switch as a quoted string and
    zero error images per folder.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"].update({
        "target_precision": -0.1, "error_dumps": {"activated": "yes", "maximum_per_class_pair": 0}})


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


def use_sequence_normalization(configuration: Dict[str, Any]) -> None:
    """
    Uses the sequence normalization in the backbone, refused for the detector, which has no sequences.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["Backbone"]["normalization"]["feature_map_normalization"] = "sequence"


def augment_detection_frames(configuration: Dict[str, Any]) -> None:
    """
    Turns the sequence augmentations on in the detection configuration, with a shuffling probability above 1.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Data"]["data_augmentation"]["independent_frames"] = True
    configuration["Data"]["data_augmentation"]["shuffle_frames"] = {"activated": True, "probability": 1.5}


def set_invalid_loading(configuration: Dict[str, Any]) -> None:
    """
    Sets the detection loaders without worker, which loads in the main process, and without prefetched batch.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Data"]["loading"] = {"number_workers": 0, "prefetch_batches": 0}


def write_per_class_report_as_text(configuration: Dict[str, Any]) -> None:
    """
    Writes the switch of the per-class evaluation report as a quoted string instead of a boolean.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"]["per_class_report"] = "yes"


def set_negative_number_sample_images(configuration: Dict[str, Any]) -> None:
    """
    Sets a negative number of sample images, a setting shared by every task.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"]["number_sample_images"] = -1


def set_invalid_tracking(configuration: Dict[str, Any]) -> None:
    """
    Writes every MLflow tracking setting with an invalid value: the switches as text, an empty server address and an
    experiment name that is not a string.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["tracking"].update({"activated": "yes", "log_artifacts": "no",
                                                                 "tracking_uri": " ", "experiment_name": 12})


def set_invalid_batch_normalization_momentums(configuration: Dict[str, Any]) -> None:
    """
    Sets a batch normalization momentum of 0 in the backbone, above 1 in the region proposal heads and written as text
    in the detection heads.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["Backbone"]["normalization"]["batch_normalization_momentum"] = 0
    configuration["RegionProposal"]["normalization"]["batch_normalization_momentum"] = 1.5
    configuration["Detector"]["normalization"]["batch_normalization_momentum"] = "fast"


def set_pre_activation_without_residual_shortcuts(configuration: Dict[str, Any]) -> None:
    """
    Turns the pre-activation placement of the residual shortcuts on with the residual shortcuts off.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["Backbone"]["modules"].update({"residual_nets": False, "residual_nets_v2": True})


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_invalid_loading, ["'Data.loading.prefetch_batches' must be an integer >= 1, got 0."]),
    (set_pre_activation_without_residual_shortcuts, ["'Backbone.modules.residual_nets_v2' must be False when "
                                                     "'Backbone.modules.residual_nets' is False"]),
    (set_invalid_batch_normalization_momentums, ["'Backbone.normalization.batch_normalization_momentum' must be a "
                                                 "number within (0, 1], got 0.",
                                                 "'RegionProposal.normalization.batch_normalization_momentum' must be "
                                                 "a number within (0, 1], got 1.5.",
                                                 "'Detector.normalization.batch_normalization_momentum' must be a "
                                                 "number within (0, 1], got 'fast'."]),
    (use_sequence_normalization, ["'Backbone.normalization.feature_map_normalization' cannot be 'sequence' for "
                                     "this task, whose samples have no frames."]),
    (augment_detection_frames, ["'Data.data_augmentation.shuffle_frames.activated' must be False for this task, "
                                "whose samples have no frames.",
                                "'Data.data_augmentation.independent_frames' must be False for this task",
                                "'Data.data_augmentation.shuffle_frames.probability' must be a probability within "
                                "[0, 1], got 1.5."]),
    (set_image_size, ["'Data.image_settings.size' must be a power of two"]),
    (set_overlapping_thresholds, ["'RegionProposal.background_iou_threshold.max' (0.6) must be <="]),
    (remove_anchor_level, ["'Anchors.scales_and_ratios' levels ['5', '6'] do not match"]),
    (set_string_flag, ["'Backbone.modules.inception' must be True or False"]),
    (set_detector_samples, ["'Detector.total_training_samples' (1000) must be <="]),
    (set_inception_channels, ["'Backbone.convolutions.4' has 20 output channels"]),
    (set_intermediate_weight_saving, ["'ExperimentConfiguration.intermediate_weight_saving_iterations' (10000) "
                                      "must be smaller than"]),
    (set_unknown_task, ["'ExperimentConfiguration.task' must be one of ['object_detection', 'instance_segmentation', "
                        "'classification', 'semantic_segmentation', 'sequence_classification', 'sequence_labelling', "
                        "'sequence_comparison', 'frame_selection'], got 'segmentation'."]),
    (set_evaluation_score_threshold, ["'ExperimentConfiguration.evaluation.score_threshold' must be a probability "
                                      "within [0, 1], got 1.5."]),
    (set_invalid_detection_error_analysis, ["'ExperimentConfiguration.evaluation.target_precision' must be a "
                                            "probability within [0, 1], got -0.1.",
                                            "'ExperimentConfiguration.evaluation.error_dumps.activated' must be True "
                                            "or False, got 'yes'.",
                                            "'ExperimentConfiguration.evaluation.error_dumps.maximum_per_class_pair' "
                                            "must be an integer >= 1, got 0."]),
    (set_multiple_errors, ["'Data.image_settings.size' must be a power of two",
                           "'Detector.number_training_positives' (500) must be <="]),
    (write_per_class_report_as_text, ["'ExperimentConfiguration.evaluation.per_class_report' must be True or False, "
                                      "got 'yes'."]),
    (set_negative_number_sample_images, ["'ExperimentConfiguration.evaluation.number_sample_images' must be an "
                                         "integer >= 0, got -1."]),
    (set_invalid_tracking, ["'ExperimentConfiguration.tracking.activated' must be True or False, got 'yes'.",
                            "'ExperimentConfiguration.tracking.log_artifacts' must be True or False, got 'no'.",
                            "'ExperimentConfiguration.tracking.tracking_uri' must be null or a non-empty string, "
                            "got ' '.",
                            "'ExperimentConfiguration.tracking.experiment_name' must be null or a non-empty string, "
                            "got 12."])])
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
    path of the missing key (scripts/configuration_processing/migrate_configuration.py adds it to the runs of that
    time).

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_modified_configuration(temporary_directory=tmp_path, modify_function=remove_task_key)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    assert "Missing configuration key 'ExperimentConfiguration.task'." in capsys.readouterr().out


def write_task_configuration(temporary_directory: Path, task_folder: str, configuration_name: str,
                             modify_function: Callable[[Dict[str, Any]], None] | None) -> Path:
    """
    Writes a copy of a configuration of a classification, sequence, semantic or instance segmentation task whose label
    mapping file (giving the classes 0 to 3, or 1 to 4 next to the background of the segmentations) is written in the
    temporary directory, the mapping files of the shipped configurations only existing once their datasets are
    created, then altered by the given function.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and the mapping file.
        task_folder (str): The folder of the configurations of the task, e.g. "classification", also selecting its
            label mapping.
        configuration_name (str): The file name of the configuration.
        modify_function (Callable[[Dict[str, Any]], None] | None): Function altering the raw configuration in place,
            or None to keep it.

    Returns:
        Path: The path of the written configuration file.
    """
    configuration = yaml.safe_load(stream=(CONFIGURATIONS_FOLDER / task_folder / configuration_name).read_text())
    mapping_path = temporary_directory / "label_mapping.json"
    label_mapping = SEGMENTATION_LABEL_MAPPING if task_folder in ["semantic_segmentation", "instance_segmentation"] \
        else CLASSIFICATION_LABEL_MAPPING
    mapping_path.write_text(data=json.dumps(obj=label_mapping))
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(mapping_path)
    if modify_function is not None:
        modify_function(configuration=configuration)

    configuration_path = temporary_directory / f"{task_folder}_configuration.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def set_negative_number_workers(configuration: Dict[str, Any]) -> None:
    """
    Sets a negative number of loader workers.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["Data"]["loading"]["number_workers"] = -1


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


def normalize_head_without_hidden_layers(configuration: Dict[str, Any]) -> None:
    """
    Turns the batch normalization of the hidden layers on in a head without hidden layers, with a momentum written as
    a boolean.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"].update({"fully_connected": [], "fully_connected_normalization": "batch",
                                                "batch_normalization_momentum": True})


def use_sequence_normalization_in_head(configuration: Dict[str, Any]) -> None:
    """
    Normalizes the hidden layers with the sequence normalization, a normalization of the backbone feature maps only.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"]["fully_connected_normalization"] = "sequence"


def write_head_normalization_as_boolean(configuration: Dict[str, Any]) -> None:
    """
    Writes the normalization of the hidden layers as True, as the configurations written before the layer
    normalization.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"]["fully_connected_normalization"] = True


def set_invalid_classification_evaluation(configuration: Dict[str, Any]) -> None:
    """
    Sets a top-k accuracy of zero classes, a target precision above 1 and the predictions setting as a quoted string.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"].update({"top_k": 0, "target_precision": 1.5,
                                                                   "save_predictions": "yes"})


def dump_multi_label_errors(configuration: Dict[str, Any]) -> None:
    """
    Turns the error images on in multi-label mode, with a maximum of zero images per class pair.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"]["output_mode"] = "multi_label"
    configuration["ExperimentConfiguration"]["evaluation"]["error_dumps"] = {"activated": True,
                                                                           "maximum_per_class_pair": 0}


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
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="classification",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    assert experiment_configuration["task"] == "classification"
    assert model_configuration["Data"]["label_dictionary"] == {1: 0, 2: 1, 3: 2, 4: 3}


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_negative_number_workers, ["'Data.loading.number_workers' must be an integer >= 0, got -1."]),
    (use_binary_labels, ["'Data.label_mapping.mode' must be 'multi_class' for the classification tasks"]),
    (add_detection_block, ["'Backbone.detection_convolution_block_indices' must be [] for the classification tasks"]),
    (set_invalid_head, ["'ClassificationHead.global_pooling' must be one of ['average', 'max', 'flatten'], got "
                        "'median'.",
                        "'ClassificationHead.fully_connected' must be a list of positive integers, got [0].",
                        "'ClassificationHead.dropout_rate' must be a number within [0, 1), got 1.0.",
                        "'ClassificationHead.output_mode' must be one of ['multi_class', 'multi_label']",
                        "'ClassificationHead.class_weighting' must be one of ['none', 'balanced']"]),
    (normalize_head_without_hidden_layers, ["'ClassificationHead.fully_connected_normalization' must be 'none' when "
                                            "'ClassificationHead.fully_connected' is []: the head has no hidden layer "
                                            "to normalize.",
                                            "'ClassificationHead.batch_normalization_momentum' must be a number within "
                                            "(0, 1], got True."]),
    (use_sequence_normalization_in_head, ["'ClassificationHead.fully_connected_normalization' must be one of "
                                          "['batch', 'layer', 'none'], got 'sequence'."]),
    (write_head_normalization_as_boolean, ["'ClassificationHead.fully_connected_normalization' must be one of "
                                           "['batch', 'layer', 'none'], got True, written before the layer "
                                           "normalization: it is now 'batch'."]),
    (set_invalid_classification_evaluation, ["'ExperimentConfiguration.evaluation.top_k' must be an integer >= 1",
                                             "'ExperimentConfiguration.evaluation.target_precision' must be a "
                                             "probability within [0, 1], got 1.5.",
                                             "'ExperimentConfiguration.evaluation.save_predictions' must be True or "
                                             "False, got 'yes'."]),
    (dump_multi_label_errors, ["'ExperimentConfiguration.evaluation.error_dumps.maximum_per_class_pair' must be an "
                               "integer >= 1, got 0.",
                               "'ExperimentConfiguration.evaluation.error_dumps.activated' must be False with "
                               "'ClassificationHead.output_mode' 'multi_label'"]),
    (write_per_class_report_as_text, ["'ExperimentConfiguration.evaluation.per_class_report' must be True or False, "
                                      "got 'yes'."]),
    (set_negative_number_sample_images, ["'ExperimentConfiguration.evaluation.number_sample_images' must be an "
                                         "integer >= 0, got -1."])])
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
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="classification",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


def flatten_final_feature_map(configuration: Dict[str, Any]) -> None:
    """
    Gives the classification head all the values of the final feature map instead of their average.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["ClassificationHead"]["global_pooling"] = "flatten"


def test_flattened_feature_map_is_accepted_for_classification(tmp_path: Path) -> None:
    """
    Verifies that the classification task accepts a head reading the flattened final feature map.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="classification",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=flatten_final_feature_map)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["ClassificationHead"]["global_pooling"] == "flatten"


@pytest.mark.parametrize(argnames=("task_folder", "configuration_name"),
                         argvalues=[("sequence_classification", "configuration_synthetic_sequences.yaml"),
                                    ("sequence_labelling", "configuration_synthetic_sequences.yaml"),
                                    ("sequence_comparison", "configuration_synthetic_pairs.yaml"),
                                    ("frame_selection", "configuration_synthetic_sequences.yaml")])
def test_flattened_feature_map_is_rejected_for_the_sequence_tasks(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                                  task_folder: str, configuration_name: str) -> None:
    """
    Verifies that the sequence tasks refuse to flatten the final feature map of their frames, their temporal module
    and fusion reading frame vectors as wide as the last backbone block.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        task_folder (str): The folder of the configurations of the sequence task.
        configuration_name (str): The file name of the synthetic configuration of the task.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                  configuration_name=configuration_name,
                                                  modify_function=flatten_final_feature_map)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    assert "'ClassificationHead.global_pooling' cannot be 'flatten' for the sequence tasks" in printed_output
    assert "Traceback" not in printed_output


def test_classification_classes_must_be_contiguous(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a label mapping file whose class indices have a gap is rejected, the classifiers predicting the
    classes 0 to K-1.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="classification",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=None)
    (tmp_path / "label_mapping.json").write_text(data=json.dumps(obj={"1": 0, "2": 2}))

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    assert "indexed 0 to K-1 without gaps, got [0, 2]." in capsys.readouterr().out


def set_causal_temporal_model(configuration: Dict[str, Any]) -> None:
    """
    Makes the bidirectional temporal module of the synthetic sequences configuration causal.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    configuration["TemporalModel"]["causal"] = True


def set_invalid_transformer_and_settings(configuration: Dict[str, Any]) -> None:
    """
    Uses a transformer whose model size is not a multiple of its heads, empty windows, an unknown handling of the
    short sequences, an unknown padding of the batches and several classes per sequence.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    configuration["TemporalModel"].update({"type": "transformer", "hidden_size": 100, "number_heads": 8})
    configuration["Data"]["sequence_settings"].update({"sequence_length": 0, "short_sequences": "repeat",
                                                       "batch_padding": "shortest"})
    configuration["ClassificationHead"]["output_mode"] = "multi_label"


def set_even_temporal_kernel(configuration: Dict[str, Any]) -> None:
    """
    Uses a temporal convolution with kernels of an even number of frames.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    configuration["TemporalModel"].update({"type": "temporal_convolution", "kernel_size": 4})


def set_unknown_temporal_type(configuration: Dict[str, Any]) -> None:
    """
    Uses a temporal module type that does not exist.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    configuration["TemporalModel"]["type"] = "rnn"


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml",
                                                                   "configuration_synthetic_sequences.yaml"])
def test_sequence_classification_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the sequence classification configurations shipped with the repository pass every consistency
    check.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the sequence classification configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_classification",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    assert experiment_configuration["task"] == "sequence_classification"
    assert model_configuration["Data"]["label_dictionary"] == {1: 0, 2: 1, 3: 2, 4: 3}


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_causal_temporal_model, ["'TemporalModel.causal' must be False for this task, which summarises whole "
                                 "sequences",
                                 "'TemporalModel.bidirectional' must be False when 'TemporalModel.causal' is True"]),
    (set_invalid_transformer_and_settings, ["'TemporalModel.hidden_size' (100) must be a multiple of "
                                            "'TemporalModel.number_heads' (8) for the transformer.",
                                            "'Data.sequence_settings.sequence_length' must be an integer >= 1",
                                            "'Data.sequence_settings.short_sequences' must be one of ['pad', 'skip']",
                                            "'Data.sequence_settings.batch_padding' must be one of "
                                            "['sequence_length', 'longest'], got 'shortest'.",
                                            "'ClassificationHead.output_mode' must be 'multi_class' for the sequence "
                                            "tasks"]),
    (set_even_temporal_kernel, ["'TemporalModel.kernel_size' must be odd for the temporal convolution, got 4."]),
    (set_unknown_temporal_type, ["'TemporalModel.type' must be one of ['mean_pooling', 'max_pooling', "
                                 "'attention_pooling', 'gru', 'lstm', 'transformer', 'temporal_convolution'] for "
                                 "this task, got 'rnn'."])])
def test_inconsistent_sequence_classification_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the sequence sections stops the loading with a short message naming the
    faulty key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_classification",
                                                  configuration_name="configuration_synthetic_sequences.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


def use_pooling_for_labelling(configuration: Dict[str, Any]) -> None:
    """
    Uses an order-blind pooling, which summarises a whole sequence, for the sequence labelling task.

    Args:
        configuration (Dict[str, Any]): The raw sequence labelling configuration, modified in place.
    """
    configuration["TemporalModel"]["type"] = "mean_pooling"


def set_causal_bidirectional_recurrent_model(configuration: Dict[str, Any]) -> None:
    """
    Makes the recurrent module of the synthetic sequence labelling configuration both causal and bidirectional.

    Args:
        configuration (Dict[str, Any]): The raw sequence labelling configuration, modified in place.
    """
    configuration["TemporalModel"].update({"causal": True, "bidirectional": True})


def set_textual_positional_embedding(configuration: Dict[str, Any]) -> None:
    """
    Writes the positional embedding flag of the synthetic sequence labelling configuration as a text.

    Args:
        configuration (Dict[str, Any]): The raw sequence labelling configuration, modified in place.
    """
    configuration["TemporalModel"]["positional_embedding"] = "no"


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml",
                                                                   "configuration_synthetic_sequences.yaml"])
def test_sequence_labelling_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the sequence labelling configurations shipped with the repository pass every consistency check,
    the example one with a causal transformer.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the sequence labelling configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_labelling",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, _ = load_experiment_configuration(configuration_path=configuration_path)
    assert experiment_configuration["task"] == "sequence_labelling"


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (use_pooling_for_labelling, ["'TemporalModel.type' must be one of ['gru', 'lstm', 'transformer', "
                                 "'temporal_convolution'] for this task, got 'mean_pooling'."]),
    (set_causal_bidirectional_recurrent_model, ["'TemporalModel.bidirectional' must be False when "
                                                "'TemporalModel.causal' is True"]),
    (set_textual_positional_embedding, ["'TemporalModel.positional_embedding' must be True or False, got 'no'."])])
def test_inconsistent_sequence_labelling_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that the temporal modules unable to keep one vector per frame, causal modules reading the sequence
    backwards and a positional embedding flag that is not a boolean are rejected for the sequence labelling task,
    without rejecting the causal flag itself.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_labelling",
                                                  configuration_name="configuration_synthetic_sequences.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "'TemporalModel.causal' must be False" not in printed_output


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml",
                                                                   "configuration_synthetic_sequences.yaml"])
def test_frame_selection_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the frame selection configurations shipped with the repository pass every consistency check,
    without the window stride nor the short sequences that the frame selection does not read, every sequence being
    one window.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the frame selection configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="frame_selection",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    assert experiment_configuration["task"] == "frame_selection"
    assert sorted(model_configuration["Data"]["sequence_settings"]) == ["batch_padding", "sequence_length"]


def set_invalid_selection(configuration: Dict[str, Any]) -> None:
    """
    Gives every setting of the 'Selection' section an invalid value.

    Args:
        configuration (Dict[str, Any]): The raw frame selection configuration, modified in place.
    """
    configuration["Selection"] = {"unselected_category": " ", "decoding": "beam", "frames_per_class": 0,
                                  "selection_loss_weight": -1.0}


def use_pooling_for_selection(configuration: Dict[str, Any]) -> None:
    """
    Uses an order-blind pooling, which summarises a whole sequence into one vector, for the frame selection.

    Args:
        configuration (Dict[str, Any]): The raw frame selection configuration, modified in place.
    """
    configuration["TemporalModel"]["type"] = "max_pooling"


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_invalid_selection, ["'Selection.unselected_category' must be the non-empty name of a manifest category, "
                             "got ' '.", "'Selection.decoding' must be one of ['hungarian', 'greedy'], got 'beam'.",
                             "'Selection.frames_per_class' must be", "'Selection.selection_loss_weight' must be a "
                             "number >= 0, got -1.0."]),
    (use_pooling_for_selection, ["'TemporalModel.type' must be one of ['gru', 'lstm', 'transformer', "
                                 "'temporal_convolution'] for this task, got 'max_pooling'."])])
def test_inconsistent_frame_selection_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that an invalid 'Selection' section, and a temporal module unable to keep one vector per frame, stop the
    loading of a frame selection configuration with a short message naming the faulty key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="frame_selection",
                                                  configuration_name="configuration_synthetic_sequences.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


def set_unknown_fusion(configuration: Dict[str, Any]) -> None:
    """
    Uses a fusion of the members that does not exist, with a dropout of 1.

    Args:
        configuration (Dict[str, Any]): The raw sequence comparison configuration, modified in place.
    """
    configuration["Fusion"].update({"type": "sum", "dropout_rate": 1.0})


def set_cross_attention_heads_not_dividing_channels(configuration: Dict[str, Any]) -> None:
    """
    Splits the 64 channels of the last backbone block between 3 cross attention heads.

    Args:
        configuration (Dict[str, Any]): The raw sequence comparison configuration, modified in place.
    """
    configuration["Fusion"].update({"type": "cross_attention", "number_heads": 3})


def set_causal_comparison(configuration: Dict[str, Any]) -> None:
    """
    Summarises the members of a pair with a causal transformer.

    Args:
        configuration (Dict[str, Any]): The raw sequence comparison configuration, modified in place.
    """
    configuration["TemporalModel"].update({"type": "transformer", "causal": True, "bidirectional": False})


def set_unknown_normalization_scope(configuration: Dict[str, Any]) -> None:
    """
    Uses sequence normalization with a scope of the statistics that does not exist.

    Args:
        configuration (Dict[str, Any]): The raw sequence comparison configuration, modified in place.
    """
    configuration["Backbone"]["normalization"]["feature_map_normalization"] = "sequence"
    configuration["Fusion"]["sequence_normalization_scope"] = "batch"


def test_sequence_normalization_is_accepted_for_the_sequence_tasks(tmp_path: Path) -> None:
    """
    Verifies that the sequence normalization of the backbone passes the checks of the sequence tasks, with both scopes
    of the comparison task.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    for task_folder, configuration_name in [("sequence_classification", "configuration_synthetic_sequences.yaml"),
                                            ("sequence_comparison", "configuration_synthetic_pairs.yaml")]:
        configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                      configuration_name=configuration_name,
                                                      modify_function=use_sequence_normalization)
        _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
        assert model_configuration["Backbone"]["normalization"]["feature_map_normalization"] == "sequence"


def remove_window_stride(configuration: Dict[str, Any]) -> None:
    """
    Removes the window stride from the sequence settings.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    del configuration["Data"]["sequence_settings"]["window_stride"]


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml",
                                                                   "configuration_synthetic_pairs.yaml"])
def test_sequence_comparison_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the sequence comparison configurations shipped with the repository pass every consistency check,
    without the window stride that the comparison task does not use.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the sequence comparison configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_comparison",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    assert experiment_configuration["task"] == "sequence_comparison"
    assert "window_stride" not in model_configuration["Data"]["sequence_settings"]


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_unknown_fusion, ["'Fusion.type' must be one of ['concatenation', 'difference', 'cross_attention'], got "
                          "'sum'.", "'Fusion.dropout_rate' must be a number within [0, 1), got 1.0."]),
    (set_cross_attention_heads_not_dividing_channels, ["The channels of the last backbone block (64) must be a "
                                                       "multiple of 'Fusion.number_heads' (3) for the cross "
                                                       "attention."]),
    (set_causal_comparison, ["'TemporalModel.causal' must be False for this task, which summarises whole "
                             "sequences"]),
    (set_unknown_normalization_scope, ["'Fusion.sequence_normalization_scope' must be one of ['member', 'pair'], "
                                       "got 'batch'."])])
def test_inconsistent_sequence_comparison_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the fusion of the members, and a causal temporal module summarising them,
    stops the loading of a sequence comparison configuration with a short message naming the faulty key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_comparison",
                                                  configuration_name="configuration_synthetic_pairs.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


def test_sequence_classification_still_requires_the_window_stride(tmp_path: Path,
                                                                  capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the sequence classification task, which cuts its sequences into windows, still requires the window
    stride that the sequence comparison task does not read.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_classification",
                                                  configuration_name="configuration_synthetic_sequences.yaml",
                                                  modify_function=remove_window_stride)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    assert "Missing configuration key 'Data.sequence_settings.window_stride'." in capsys.readouterr().out


def freeze_random_backbone(configuration: Dict[str, Any]) -> None:
    """
    Freezes the backbone without loading one from another run, and multiplies its learning rate by zero.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Backbone"]["pretrained"].update({"freeze": True, "learning_rate_multiplier": 0})


def set_invalid_pretrained_run(configuration: Dict[str, Any]) -> None:
    """
    Gives the pretrained backbone a number instead of a run folder path, and a checkpoint iteration of zero.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Backbone"]["pretrained"].update({"experiment_folder": 5, "iteration": 0})


def freeze_normalization_of_random_backbone(configuration: Dict[str, Any]) -> None:
    """
    Keeps the normalization parameters of a backbone trained from random weights fixed.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Backbone"]["pretrained"]["freeze_normalization_parameters"] = True


def freeze_normalization_of_frozen_backbone(configuration: Dict[str, Any]) -> None:
    """
    Keeps the normalization parameters of a frozen transferred backbone fixed.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Backbone"]["pretrained"].update({"experiment_folder": "/runs/source", "freeze": True,
                                                    "freeze_normalization_parameters": True})


def freeze_normalization_without_normalization_layers(configuration: Dict[str, Any]) -> None:
    """
    Keeps the normalization parameters of a transferred backbone without normalization layers fixed, and writes the
    flag of another transferred backbone as a string.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Backbone"]["pretrained"].update({"experiment_folder": "/runs/source",
                                                    "freeze_normalization_parameters": True})
    configuration["Backbone"]["normalization"]["feature_map_normalization"] = "none"


def write_normalization_flag_as_text(configuration: Dict[str, Any]) -> None:
    """
    Writes the normalization flag of the backbone as a quoted string instead of a boolean.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Backbone"]["pretrained"]["freeze_normalization_parameters"] = "False"


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (freeze_random_backbone, ["'Backbone.pretrained.freeze' must be False when 'Backbone.pretrained.experiment_folder' "
                              "is null: only a transferred backbone can be frozen.",
                              "'Backbone.pretrained.learning_rate_multiplier' must be > 0, got 0."]),
    (set_invalid_pretrained_run, ["'Backbone.pretrained.experiment_folder' must be null or the path of a run "
                                  "folder, got 5.",
                                  "'Backbone.pretrained.iteration' must be an integer >= 1, got 0."]),
    (freeze_normalization_of_random_backbone, ["'Backbone.pretrained.freeze_normalization_parameters' must be False "
                                               "when 'Backbone.pretrained.experiment_folder' is null"]),
    (freeze_normalization_of_frozen_backbone, ["'Backbone.pretrained.freeze_normalization_parameters' must be False "
                                               "when 'Backbone.pretrained.freeze' is True"]),
    (freeze_normalization_without_normalization_layers, [
        "'Backbone.pretrained.freeze_normalization_parameters' must be False when "
        "'Backbone.normalization.feature_map_normalization' is 'none': the backbone has no normalization layer."]),
    (write_normalization_flag_as_text, ["'Backbone.pretrained.freeze_normalization_parameters' must be True or False, "
                                        "got 'False'."])])
def test_inconsistent_pretrained_backbone_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                      modify_function: Callable[[Dict[str, Any]], None],
                                                      expected_messages: List[str]) -> None:
    """
    Verifies that inconsistent pretrained backbone settings stop the loading with a short message naming the faulty
    key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_modified_configuration(temporary_directory=tmp_path, modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output


def transfer_all_sequence_components(configuration: Dict[str, Any]) -> None:
    """
    Initialises the backbone, the temporal module and the classification head of a sequence classifier with those of
    another run: the backbone with its normalization parameters fixed, the temporal module frozen and the head ten
    times slower.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    configuration["Backbone"]["pretrained"].update({"experiment_folder": "/runs/source",
                                                    "freeze_normalization_parameters": True})
    configuration["TemporalModel"]["pretrained"].update({"experiment_folder": "/runs/source", "iteration": 3,
                                                         "freeze": True})
    configuration["ClassificationHead"]["pretrained"].update({"experiment_folder": "/runs/source",
                                                              "learning_rate_multiplier": 0.1})


def test_transferred_sequence_components_are_accepted(tmp_path: Path) -> None:
    """
    Verifies that consistent pretrained settings of every component of a sequence classifier pass the checks, the
    source run being only read when the training starts.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_classification",
                                                  configuration_name="configuration_synthetic_sequences.yaml",
                                                  modify_function=transfer_all_sequence_components)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["TemporalModel"]["pretrained"]["freeze"] is True


def transfer_mean_pooling(configuration: Dict[str, Any]) -> None:
    """
    Initialises a mean pooling temporal module, which has no weights, with the one of another run.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    configuration["TemporalModel"]["type"] = "mean_pooling"
    configuration["TemporalModel"]["pretrained"]["experiment_folder"] = "/runs/source"


def set_invalid_temporal_and_head_transfers(configuration: Dict[str, Any]) -> None:
    """
    Freezes the temporal module without loading one from another run and multiplies its learning rate by zero, and
    gives the classification head a number instead of a run folder path and a checkpoint iteration of zero.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    configuration["TemporalModel"]["pretrained"].update({"freeze": True, "learning_rate_multiplier": 0})
    configuration["ClassificationHead"]["pretrained"].update({"experiment_folder": 5, "iteration": 0})


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (transfer_mean_pooling, ["'TemporalModel.pretrained.experiment_folder' must be null for the 'mean_pooling' "
                             "temporal module, which has no weights to transfer."]),
    (set_invalid_temporal_and_head_transfers, [
        "'TemporalModel.pretrained.freeze' must be False when 'TemporalModel.pretrained.experiment_folder' is null: "
        "only a transferred temporal module can be frozen.",
        "'TemporalModel.pretrained.learning_rate_multiplier' must be > 0, got 0.",
        "'ClassificationHead.pretrained.experiment_folder' must be null or the path of a run folder, got 5.",
        "'ClassificationHead.pretrained.iteration' must be an integer >= 1, got 0."])])
def test_inconsistent_temporal_and_head_transfers_are_rejected(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                               modify_function: Callable[[Dict[str, Any]], None],
                                                               expected_messages: List[str]) -> None:
    """
    Verifies that inconsistent pretrained settings of the temporal module or of the classification head stop the
    loading with a short message naming the faulty key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_classification",
                                                  configuration_name="configuration_synthetic_sequences.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


def set_invalid_optimization(configuration: Dict[str, Any]) -> None:
    """
    Sets an unknown optimizer and schedule, a negative weight decay, an empty decay, a minimum learning rate above
    the learning rate, a zero clipping norm, a mixed precision flag that is not a boolean, and a moving average
    switch that is not a boolean with a decay of 1, which would never move.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Optimization"].update({"optimizer": "rmsprop", "learning_rate_schedule": "step",
                                          "weight_decay": -0.1, "decay_iterations": 0,
                                          "minimum_learning_rate_ratio": 1.5, "gradient_clipping_norm": 0,
                                          "mixed_precision": "yes",
                                          "exponential_moving_average": {"activated": "yes", "decay": 1.0}})


def test_inconsistent_optimization_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that every inconsistency of the 'Optimization' section is reported with the faulty key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_modified_configuration(temporary_directory=tmp_path,
                                                      modify_function=set_invalid_optimization)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in ["'Optimization.optimizer' must be one of ['adam', 'adamw', 'sgd'], got 'rmsprop'.",
                             "'Optimization.learning_rate_schedule' must be one of ['constant', 'cosine'], "
                             "got 'step'.",
                             "'Optimization.weight_decay' must be >= 0, got -0.1.",
                             "'Optimization.decay_iterations' must be an integer >= 1, got 0.",
                             "'Optimization.minimum_learning_rate_ratio' must be a probability within [0, 1], got 1.5.",
                             "'Optimization.gradient_clipping_norm' must be null or > 0, got 0.",
                             "'Optimization.mixed_precision' must be True or False, got 'yes'.",
                             "'Optimization.exponential_moving_average.activated' must be True or False, got 'yes'.",
                             "'Optimization.exponential_moving_average.decay' must be strictly between 0 and 1, "
                             "got 1.0."]:
        assert expected_message in printed_output


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml",
                                                                   "configuration_synthetic_shapes.yaml"])
def test_segmentation_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the semantic segmentation configurations shipped with the repository pass every consistency check,
    and that their label dictionary holds the background class 0 next to the mapped categories.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the semantic segmentation configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="semantic_segmentation",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    assert experiment_configuration["task"] == "semantic_segmentation"
    assert model_configuration["Data"]["label_dictionary"] == {1: 1, 2: 2, 3: 3, 4: 4, "background": 0}


def use_last_max_pooling_with_its_decoder_level(configuration: Dict[str, Any]) -> None:
    """
    Pools the output of the last backbone block, which then becomes one more skip connection, with its decoder level,
    and uses the binary label mapping.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["Backbone"]["last_max_pooling"] = True
    configuration["SegmentationDecoder"]["convolutions"]["6"] = [2, 128]
    configuration["Data"]["label_mapping"]["mode"] = "binary"


def test_segmentation_last_max_pooling_and_binary_labels_are_accepted(tmp_path: Path) -> None:
    """
    Verifies that the semantic segmentation accepts a pooled last backbone block when the decoder has the level of
    its skip connection, and the binary label mapping giving the background and foreground classes.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="semantic_segmentation",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=use_last_max_pooling_with_its_decoder_level)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["Data"]["label_dictionary"] == {"background": 0, "foreground": 1}


def remove_decoder_level(configuration: Dict[str, Any]) -> None:
    """
    Removes the decoder level of the skip connection of the first backbone block.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    del configuration["SegmentationDecoder"]["convolutions"]["1"]


def use_last_max_pooling_without_its_decoder_level(configuration: Dict[str, Any]) -> None:
    """
    Pools the output of the last backbone block without adding the decoder level of its skip connection.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["Backbone"]["last_max_pooling"] = True


def set_invalid_decoder(configuration: Dict[str, Any]) -> None:
    """
    Sets an unknown upsampling and normalization, a batch normalization momentum of 2, a decoder level without layer,
    a module flag written as a string and a dropout of 1.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["SegmentationDecoder"].update({"upsampling": "nearest", "feature_map_normalization": "group",
                                                 "batch_normalization_momentum": 2, "dropout_rate": 1.0})
    configuration["SegmentationDecoder"]["convolutions"]["3"] = [0, 32]
    configuration["SegmentationDecoder"]["modules"]["squeeze_and_excite"] = "yes"


def freeze_random_decoder(configuration: Dict[str, Any]) -> None:
    """
    Freezes the decoder and loads its class layers without loading one from another run, and multiplies its learning
    rate by zero.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["SegmentationDecoder"]["pretrained"].update({"freeze": True, "learning_rate_multiplier": 0,
                                                               "load_class_layers": True})


def write_class_layers_flag_as_text(configuration: Dict[str, Any]) -> None:
    """
    Writes the class layers flag of a transferred decoder as a quoted string instead of a boolean.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["SegmentationDecoder"]["pretrained"].update({"experiment_folder": "/runs/source",
                                                               "load_class_layers": "True"})


def remove_both_losses(configuration: Dict[str, Any]) -> None:
    """
    Gives every loss of the regions a weight of 0 with a boundary loss alone, and sets a negative focusing parameter,
    a label smoothing of 1, an unknown class weighting, a boundary weight given as a boolean and a negative boundary
    distance.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["SegmentationLoss"].update({"cross_entropy_weight": 0, "dice_weight": 0.0, "lovasz_weight": 0.0,
                                              "boundary_weight": True, "boundary_distance": -1, "focal_gamma": -1.0,
                                              "label_smoothing": 1.0, "class_weighting": "inverse"})


def set_invalid_deep_supervision(configuration: Dict[str, Any]) -> None:
    """
    Puts a deep supervision head on the finest decoder level, whose output is the final segmentation, and gives an
    intermediate level a weight of 0.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["SegmentationDecoder"]["deep_supervision_weights"] = {"1": 0.5, "3": 0}


def set_invalid_tiling(configuration: Dict[str, Any]) -> None:
    """
    Turns the training on tiles on with a scale of 0 and an overlap as large as a tile.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["Data"]["tiling"] = {"activated": True, "scale": 0, "overlap": 512}


def set_invalid_segmentation_evaluation(configuration: Dict[str, Any]) -> None:
    """
    Sets a negative number of sample images, the predicted masks setting as a quoted string, the original resolution
    setting as a number and a boundary tolerance of 0.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"].update({"number_sample_images": -1,
                                                                   "save_predicted_masks": "yes",
                                                                   "original_resolution": 1, "boundary_tolerance": 0})


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (add_detection_block, ["'Backbone.detection_convolution_block_indices' must be [] for the semantic segmentation "
                           "task, its decoder reads every block, got [3]."]),
    (use_sequence_normalization, ["'Backbone.normalization.feature_map_normalization' cannot be 'sequence' for "
                                  "this task, whose samples have no frames."]),
    (remove_decoder_level, ["'SegmentationDecoder.convolutions' keys must be the backbone blocks of its skip "
                            "connections ['5', '4', '3', '2', '1']", "missing ['1'], unexpected []."]),
    (use_last_max_pooling_without_its_decoder_level, ["'SegmentationDecoder.convolutions' keys must be the backbone "
                                                      "blocks of its skip connections ['6', '5', '4', '3', '2', '1']",
                                                      "missing ['6'], unexpected []."]),
    (set_invalid_decoder, ["'SegmentationDecoder.upsampling' must be one of ['transposed', 'bilinear'], got "
                           "'nearest'.",
                           "'SegmentationDecoder.convolutions.3' must be [number_layers, output_channels] with "
                           "positive integers, got [0, 32].",
                           "'SegmentationDecoder.modules.squeeze_and_excite' must be True or False, got 'yes'.",
                           "'SegmentationDecoder.feature_map_normalization' must be one of ['batch', 'layer', "
                           "'none'], got 'group'.",
                           "'SegmentationDecoder.batch_normalization_momentum' must be a number within (0, 1], got "
                           "2.",
                           "'SegmentationDecoder.dropout_rate' must be a number within [0, 1), got 1.0."]),
    (freeze_random_decoder, ["'SegmentationDecoder.pretrained.freeze' must be False when "
                             "'SegmentationDecoder.pretrained.experiment_folder' is null: only a transferred "
                             "segmentation decoder can be frozen.",
                             "'SegmentationDecoder.pretrained.learning_rate_multiplier' must be > 0, got 0.",
                             "'SegmentationDecoder.pretrained.load_class_layers' must be False when "
                             "'SegmentationDecoder.pretrained.experiment_folder' is null: the class layers are loaded "
                             "from the source run of the decoder."]),
    (write_class_layers_flag_as_text, ["'SegmentationDecoder.pretrained.load_class_layers' must be True or False, got "
                                       "'True'."]),
    (remove_both_losses, ["'SegmentationLoss.cross_entropy_weight', 'SegmentationLoss.dice_weight' and "
                          "'SegmentationLoss.lovasz_weight' cannot all be 0, the regions would have no loss.",
                          "'SegmentationLoss.boundary_weight' must be a number >= 0, got True.",
                          "'SegmentationLoss.boundary_distance' must be an integer >= 0, got -1.",
                          "'SegmentationLoss.focal_gamma' must be a number >= 0, got -1.0.",
                          "'SegmentationLoss.label_smoothing' must be a number within [0, 1), got 1.0.",
                          "'SegmentationLoss.class_weighting' must be one of ['none', 'balanced'], got 'inverse'."]),
    (set_invalid_segmentation_evaluation, ["'ExperimentConfiguration.evaluation.number_sample_images' must be an "
                                           "integer >= 0, got -1.",
                                           "'ExperimentConfiguration.evaluation.save_predicted_masks' must be True or "
                                           "False, got 'yes'.",
                                           "'ExperimentConfiguration.evaluation.original_resolution' must be True or "
                                           "False, got 1.",
                                           "'ExperimentConfiguration.evaluation.boundary_tolerance' must be an integer "
                                           ">= 1, got 0."]),
    (set_invalid_deep_supervision, ["'SegmentationDecoder.deep_supervision_weights' keys must be intermediate decoder "
                                    "levels ['5', '4', '3', '2'] (the finest level gives the final segmentation), got "
                                    "['1'].",
                                    "'SegmentationDecoder.deep_supervision_weights.3' must be a number > 0, got 0."]),
    (set_invalid_tiling, ["'Data.tiling.scale' must be a number > 0, got 0.",
                          "'Data.tiling.overlap' must be smaller than the tile size 'Data.image_settings.size' (512), "
                          "got 512."]),
    (write_per_class_report_as_text, ["'ExperimentConfiguration.evaluation.per_class_report' must be True or False, "
                                      "got 'yes'."])])
def test_inconsistent_segmentation_configuration_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                             modify_function: Callable[[Dict[str, Any]], None],
                                                             expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the semantic segmentation sections stops the loading with a short message
    naming the faulty key, the decoder levels being checked against the backbone blocks.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="semantic_segmentation",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


@pytest.mark.parametrize(argnames=("label_mapping", "expected_message"), argvalues=[
    ({"1": 1, "2": 3}, "indexed 0 (background) to K-1 without gaps, got [0, 1, 3]."),
    ({"1": 1, "300": 2}, "The categories of the label mapping file must be mask values between 1 and 254 (0 is the "
                         "background, 255 the unlabelled pixels), got [300]."),
    ({"0": 1, "1": 0}, "The categories of the label mapping file must be mask values between 1 and 254"),
    ({"0": 0, "1": 1, "2": 2}, None)])
def test_segmentation_label_mapping_gives_mask_values(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                      label_mapping: Dict[str, int],
                                                      expected_message: str | None) -> None:
    """
    Verifies that the label mapping file of the semantic segmentation maps mask values (1 to 254, the value 0 being
    allowed for the background class only) to classes indexed from the background 0 without gaps.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        label_mapping (Dict[str, int]): The content of the label mapping file.
        expected_message (str | None): The message fragment printed for the mapping, None when it is accepted.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="semantic_segmentation",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=None)
    (tmp_path / "label_mapping.json").write_text(data=json.dumps(obj=label_mapping))

    # An accepted mapping keeps its background category, a rejected one is reported without traceback
    if expected_message is None:
        _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
        assert model_configuration["Data"]["label_dictionary"] == {0: 0, 1: 1, 2: 2}
        return
    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)
    printed_output = capsys.readouterr().out
    assert expected_message in printed_output
    assert "Traceback" not in printed_output


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml",
                                                                   "configuration_synthetic_shapes.yaml"])
def test_instance_segmentation_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the instance segmentation configurations shipped with the repository pass every consistency check,
    those of the object detection sections and of the mask heads.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the instance segmentation configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="instance_segmentation",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    assert experiment_configuration["task"] == "instance_segmentation"
    assert model_configuration["Data"]["label_dictionary"]["background"] == 0
    assert model_configuration["MaskHead"]["mask_resolution"] == 28
    assert model_configuration["MaskHead"]["head_configurations"]["5"]["roi_align_pool_size"] == 14


def transfer_mask_heads_and_class_layers(configuration: Dict[str, Any]) -> None:
    """
    Initialises the frozen mask heads of an instance segmenter with those of another run, or the decoder of a semantic
    segmenter and its class layers, depending on the sections of the configuration.

    Args:
        configuration (Dict[str, Any]): The raw instance or semantic segmentation configuration, modified in place.
    """
    if "MaskHead" in configuration:
        configuration["MaskHead"]["pretrained"].update({"experiment_folder": "/runs/source", "freeze": True})
    else:
        configuration["SegmentationDecoder"]["pretrained"].update({"experiment_folder": "/runs/source",
                                                                   "load_class_layers": True})


@pytest.mark.parametrize(argnames="task_folder", argvalues=["instance_segmentation", "semantic_segmentation"])
def test_transferred_mask_heads_and_class_layers_are_accepted(tmp_path: Path, task_folder: str) -> None:
    """
    Verifies that consistent pretrained settings of the mask heads of an instance segmenter, and of the decoder of a
    semantic segmenter loading its class layers, pass the checks, the source run being only read when the training
    starts.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task_folder (str): The folder of the configurations of the task.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=transfer_mask_heads_and_class_layers)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    if task_folder == "instance_segmentation":
        assert model_configuration["MaskHead"]["pretrained"]["freeze"] is True
    else:
        assert model_configuration["SegmentationDecoder"]["pretrained"]["load_class_layers"] is True


def set_mismatched_mask_head_levels(configuration: Dict[str, Any]) -> None:
    """
    Gives the mask heads a level that the backbone does not feed to the detection heads.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["MaskHead"]["head_configurations"]["8"] = configuration["MaskHead"]["head_configurations"].pop("7")


def set_invalid_mask_head(configuration: Dict[str, Any]) -> None:
    """
    Sets an invalid mask resolution, upsampling, pool size, convolutions, normalization, batch normalization momentum,
    module flag and mask threshold, and inception modules with a channel count that is not a multiple of 8.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    mask_head_configuration = configuration["MaskHead"]
    mask_head_configuration["mask_resolution"] = 0
    mask_head_configuration["upsampling"] = "nearest"
    mask_head_configuration["head_configurations"]["5"] = {"roi_align_pool_size": 0, "convolutions": [4]}
    mask_head_configuration["head_configurations"]["6"]["convolutions"] = [4, 60]
    mask_head_configuration["normalization"]["feature_map_normalization"] = "group"
    mask_head_configuration["normalization"]["batch_normalization_momentum"] = -0.1
    mask_head_configuration["modules"].update({"inception": True, "residual_nets": "yes"})
    mask_head_configuration["inference"]["mask_threshold"] = 1.5


def set_pool_size_above_the_mask_resolution(configuration: Dict[str, Any]) -> None:
    """
    Gives level 7 a pool size larger than the masks, which the mask head could only shrink.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["MaskHead"]["head_configurations"]["7"]["roi_align_pool_size"] = 32


def freeze_random_mask_heads(configuration: Dict[str, Any]) -> None:
    """
    Freezes the mask heads without loading them from another run, and multiplies their learning rate by zero.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["MaskHead"]["pretrained"].update({"freeze": True, "learning_rate_multiplier": 0})


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (freeze_random_mask_heads, ["'MaskHead.pretrained.freeze' must be False when "
                                "'MaskHead.pretrained.experiment_folder' is null: only a transferred mask heads can "
                                "be frozen.",
                                "'MaskHead.pretrained.learning_rate_multiplier' must be > 0, got 0."]),
    (set_pool_size_above_the_mask_resolution, ["'MaskHead.head_configurations.7.roi_align_pool_size' (32) must not "
                                               "exceed 'MaskHead.mask_resolution' (28), the pooled features being "
                                               "only enlarged to the masks."]),
    (set_mismatched_mask_head_levels, ["'MaskHead.head_configurations' levels ['5', '6', '8'] do not match "
                                       "'Backbone.detection_convolution_block_indices' ['5', '6', '7']."]),
    (set_invalid_mask_head, ["'MaskHead.mask_resolution' must be an integer >= 1, got 0.",
                             "'MaskHead.upsampling' must be one of ['transposed', 'bilinear'], got 'nearest'.",
                             "'MaskHead.head_configurations.5.roi_align_pool_size' must be an integer >= 1, got 0.",
                             "'MaskHead.head_configurations.5.convolutions' must be [number_layers, output_channels] "
                             "with positive integers, got [4].",
                             "'MaskHead.head_configurations.6.convolutions' has 60 output channels, inception modules "
                             "require a multiple of 8.",
                             "'MaskHead.normalization.feature_map_normalization' must be one of",
                             "'MaskHead.normalization.batch_normalization_momentum' must be a number within (0, 1], "
                             "got -0.1.",
                             "'MaskHead.modules.residual_nets' must be True or False, got 'yes'.",
                             "'MaskHead.inference.mask_threshold' must be a probability within [0, 1], got 1.5."])])
def test_inconsistent_instance_segmentation_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the mask head section stops the loading with a short message naming the
    faulty key, its levels being checked against the detection blocks like the detection heads.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="instance_segmentation",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output
