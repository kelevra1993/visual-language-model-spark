"""
Tests of the configuration checks of vision_training/configuration_processing/configuration_checker.py, run through
load_experiment_configuration: the configurations shipped for every task are accepted, while inconsistent settings
(detection, classification, sequence, comparison, segmentation, self-supervised, keypoint detection, normalization,
pretrained transfers with the initialisation of their new classes, optimization) and missing keys are rejected with
errors naming their path, the run name suffix being accepted null or made of letters, digits, '-' and '.' only, and the
intermediate weight saving smaller than the evaluated weight saving. The augmentation types moving the image only are
refused for the tasks with boxes, masks or keypoints, the jigsaw refuses an image size that is not a multiple of its
grid size, and a setting removed from the repository is refused alone, before the other checks. The keypoint
definitions injected by the loader from the training labels are checked too, and so is the label dictionary saved with
a run, taken by the configuration of its folder.
"""

import copy
import json
import pytest
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from vision_training.configuration_processing.common_checks import REMOVED_SETTINGS, check_removed_settings
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.configuration_processing.detection_checks import check_detection_levels
from vision_training.configuration_processing.instance_segmentation_checks import (compute_level_stride,
                                                                                 compute_skip_stride)
from vision_training.data.augmentations import SELF_SUPERVISED_AUGMENTATION_STRENGTHS, SUPERVISED_AUGMENTATION_STRENGTHS
from vision_training.data.label_mapping import write_run_label_dictionary
from vision_training.data.synthetic.create_synthetic_dataset import (KEYPOINT_CATEGORIES, build_keypoint_label_mapping,
                                                                     build_keypoint_labels)

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")

# Folder of the configurations of every task, checked with a label mapping file written by the tests for the tasks
# whose mapping file only exists once their dataset is created
CONFIGURATIONS_FOLDER = Path(__file__).parents[2] / "vision_training" / "configurations"

# Label mapping file written by the tests: the categories 1 to 4 give the classes 0 to 3 of the classification and
# sequence tasks, or the classes 1 to 4 of the semantic segmentation, next to its background class 0
CLASSIFICATION_LABEL_MAPPING = {"1": 0, "2": 1, "3": 2, "4": 3}
SEGMENTATION_LABEL_MAPPING = {"1": 1, "2": 2, "3": 3, "4": 4}

# COCO person category written in the training labels of the keypoint example configuration: its 17 keypoint names and
# its 1-based skeleton, as in the COCO keypoint annotation files
COCO_PERSON_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                         "right_knee", "left_ankle", "right_ankle"]
COCO_PERSON_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


def write_modified_configuration(temporary_directory: Path, modify_function: Callable[[Dict[str, Any]], None]) -> Path:
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
    Sets an image size that is not a multiple of 64, the 6 pooling layers of the detection backbone halving it.

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


def set_invalid_neck_settings(configuration: Dict[str, Any]) -> None:
    """
    Gives the neck an unknown type, no output channel and an unknown normalization.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Neck"] = {"type": "bidirectional_pyramid", "output_channels": 0,
                             "normalization": {"feature_map_normalization": "group"}}


def set_unknown_box_loss(configuration: Dict[str, Any]) -> None:
    """
    Gives the detector section a box loss the code does not compute.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Detector"]["box_loss"] = "iou"


def set_invalid_final_nms_settings(configuration: Dict[str, Any]) -> None:
    """
    Gives the final non-maximum suppression of the detector a criterion the code does not compute and a distance
    exponent of zero.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Detector"]["nms_criterion"] = "soft_nms"
    configuration["Detector"]["nms_distance_exponent"] = 0


def set_unknown_proposal_selection(configuration: Dict[str, Any]) -> None:
    """
    Gives the region proposal filter a selection the code does not run, per image instead of over all the levels or
    level by level.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["RegionProposal"]["proposal_selection"] = "per_image"


def set_infinite_nms_distance_exponent(configuration: Dict[str, Any]) -> None:
    """
    Turns DIoU-NMS on with an infinite exponent of its distance term, which would reduce it to the greedy NMS.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Detector"]["nms_criterion"] = "distance_iou"
    configuration["Detector"]["nms_distance_exponent"] = float("inf")


def set_invalid_detection_level_settings(configuration: Dict[str, Any]) -> None:
    """
    Gives the fifth detection level an anchor scale of zero, a negative aspect ratio, region proposal convolutions
    without channels, a pool size of zero and no fully connected layer, and the sixth level detection convolutions of
    60 channels with inception modules in the detection heads.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    level_configuration = configuration["DetectionLevels"]["5"]
    level_configuration["anchors"] = {"scales": [0, 64], "aspect_ratios": [-1.0]}
    level_configuration["region_proposal"]["convolutions"] = [2]
    level_configuration["detector"].update({"roi_align_pool_size": 0, "fully_connected": []})
    configuration["Detector"]["modules"]["inception"] = True
    configuration["DetectionLevels"]["6"]["detector"]["convolutions"] = [2, 60]


def remove_task_key(configuration: Dict[str, Any]) -> None:
    """
    Removes the task of the configuration, as in the configurations written before the task-aware configuration.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    del configuration["ExperimentConfiguration"]["task"]


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


def lower_background_minimum(configuration: Dict[str, Any]) -> None:
    """
    Starts the background IoU range of the region proposals at 0.0, every anchor overlapping no object being sampled
    as background too.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["RegionProposal"]["background_iou_threshold"]["min"] = 0.0


def test_background_minimum_loads_without_warning(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the background IoU range of the shipped example, starting at 0.1 (kept by stage 2 of the detection
    benchmark, only the hard negatives being sampled), loads without error or warning, and so does a range starting at
    0.0.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    # The shipped configuration ignores the anchors overlapping no object, silently
    _, model_configuration = load_experiment_configuration(configuration_path=EXAMPLE_CONFIGURATION_PATH)
    assert model_configuration["RegionProposal"]["background_iou_threshold"]["min"] == 0.1
    assert "background_iou_threshold.min" not in capsys.readouterr().out

    # A minimum of 0.0, sampling every anchor overlapping no object as background, is accepted silently as well
    configuration_path = write_modified_configuration(temporary_directory=tmp_path,
                                                      modify_function=lower_background_minimum)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["RegionProposal"]["background_iou_threshold"]["min"] == 0.0
    assert "background_iou_threshold.min" not in capsys.readouterr().out


def set_invalid_benchmark_speed(configuration: Dict[str, Any]) -> None:
    """
    Writes the switch of the speed benchmark diagnosis mode as text instead of a boolean.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["diagnosis_modes"]["benchmark_speed"] = "yes"


def remove_benchmark_speed(configuration: Dict[str, Any]) -> None:
    """
    Removes the switch of the speed benchmark diagnosis mode, as in a configuration written before it.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    del configuration["ExperimentConfiguration"]["diagnosis_modes"]["benchmark_speed"]


def set_invalid_loss_weights(configuration: Dict[str, Any]) -> None:
    """
    Sets a negative weight of the classification loss of the region proposals and writes the weight of the box loss of
    the detection heads as text.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["Optimization"]["loss_weights"].update({"region_proposal_classification_loss": -1.0,
                                                          "detection_regression_loss": "high"})


def activate_image_only_augmentation_types(configuration: Dict[str, Any]) -> None:
    """
    Turns on the perspective and the jigsaw of the data augmentation, which move the image only.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    for augmentation_type in ["random_perspective", "random_jigsaw"]:
        configuration["Data"]["data_augmentation"]["types"][augmentation_type]["activated"] = True


# Refusals of the perspective and the jigsaw by the tasks whose boxes, masks or keypoints cannot follow them
IMAGE_ONLY_TYPE_REFUSALS = [f"'Data.data_augmentation.types.{augmentation_type}.activated' must be False for this "
                            f"task: {augmentation_type} moves the image only, and its bounding boxes, masks or "
                            f"keypoints cannot follow it." for augmentation_type in ["random_perspective",
                                                                                     "random_jigsaw"]]


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_image_size, ["'Data.image_settings.size' (1000) must be a multiple of 64, 2 to the power of the 6 Backbone "
                      "pooling layers"]),
    (set_overlapping_thresholds, ["'RegionProposal.background_iou_threshold.max' (0.6) must be <="]),
    (set_invalid_neck_settings, ["'Neck.type' must be one of ['none', 'feature_pyramid', 'path_aggregation'], got "
                                 "'bidirectional_pyramid'.", "'Neck.output_channels'",
                                 "'Neck.normalization.feature_map_normalization' must be one of ['batch', 'layer', "
                                 "'none'], got 'group'."]),
    (set_unknown_box_loss, ["'Detector.box_loss' must be one of ['smooth_l1', 'complete_iou'], got 'iou'."]),
    (set_invalid_final_nms_settings, ["'Detector.nms_criterion' must be one of ['iou', 'distance_iou'], got "
                                      "'soft_nms'.",
                                      "'Detector.nms_distance_exponent' must be a finite positive number, got 0."]),
    (set_infinite_nms_distance_exponent, ["'Detector.nms_distance_exponent' must be a finite positive number, got "
                                          "inf."]),
    (set_unknown_proposal_selection, ["'RegionProposal.proposal_selection' must be one of ['joint', 'per_level'], "
                                      "got 'per_image'."]),
    (set_invalid_detection_level_settings, ["'DetectionLevels.5.anchors.scales' must be integers within [1, 1024], "
                                            "got [0, 64].",
                                            "'DetectionLevels.5.anchors.aspect_ratios' must be positive numbers, "
                                            "got [-1.0].",
                                            "'DetectionLevels.5.region_proposal.convolutions' must be "
                                            "[number_layers, output_channels] with positive integers, got [2].",
                                            "'DetectionLevels.5.detector.roi_align_pool_size' must be an integer "
                                            ">= 1, got 0.",
                                            "'DetectionLevels.5.detector.fully_connected' must be a non-empty list "
                                            "of positive integers, got [].",
                                            "'DetectionLevels.6.detector.convolutions' has 60 output channels, "
                                            "inception modules require a multiple of 8."]),
    (set_multiple_errors, ["'Data.image_settings.size' (1000) must be a multiple of 64",
                           "'Detector.number_training_positives' (500) must be <="]),
    (set_invalid_loss_weights, ["'Optimization.loss_weights.region_proposal_classification_loss' must be a number "
                                ">= 0, got -1.0.",
                                "'Optimization.loss_weights.detection_regression_loss' must be a number >= 0, got "
                                "'high'."]),
    (activate_image_only_augmentation_types, IMAGE_ONLY_TYPE_REFUSALS),
    (set_invalid_benchmark_speed, ["'ExperimentConfiguration.diagnosis_modes.benchmark_speed' must be True or False, "
                                   "got 'yes'."]),
    (remove_benchmark_speed, ["Missing configuration key 'ExperimentConfiguration.diagnosis_modes.benchmark_speed'."])])
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


@pytest.mark.parametrize(argnames=("run_name_suffix", "accepted"), argvalues=[(None, True), ("runs/v1", False)])
def test_run_name_suffix_is_checked(tmp_path: Path, capsys: pytest.CaptureFixture, run_name_suffix: Any,
                                    accepted: bool) -> None:
    """
    Verifies that the run name suffix is null or a non-empty string of letters, digits, '-' and '.', any other value
    being refused with its key, so that the suffix always stays one section of the run name ('_' separating the
    sections) and inside the experiment folder.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        run_name_suffix (Any): The run name suffix written in the configuration.
        accepted (bool): Whether the configuration must be accepted.
    """
    configuration = yaml.safe_load(stream=EXAMPLE_CONFIGURATION_PATH.read_text())
    configuration["ExperimentConfiguration"]["run_name_suffix"] = run_name_suffix
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(yaml.safe_dump(configuration))

    # An accepted suffix is loaded as written, a refused one stops the loading with the message of its key
    if accepted:
        experiment_configuration, _ = load_experiment_configuration(configuration_path=configuration_path)
        assert experiment_configuration["run_name_suffix"] == run_name_suffix
    else:
        with pytest.raises(expected_exception=SystemExit):
            load_experiment_configuration(configuration_path=configuration_path)
        assert (f"'ExperimentConfiguration.run_name_suffix' must be null or a non-empty string of letters, digits, "
                f"'-' and '.', got {run_name_suffix!r}.") in capsys.readouterr().out


@pytest.mark.parametrize(argnames=("intermediate_weight_saving_iterations", "expected_message"),
                         argvalues=[(5000, "'ExperimentConfiguration.intermediate_weight_saving_iterations' (5000) "
                                           "must be smaller than 'ExperimentConfiguration.weight_saving_iterations' "
                                           "(5000).")])
def test_intermediate_weight_saving_iterations_is_checked(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                          intermediate_weight_saving_iterations: int,
                                                          expected_message: Optional[str]) -> None:
    """
    Verifies that an intermediate weight saving not smaller than weight_saving_iterations (5000 here), so not more
    frequent than the evaluated checkpoints, is refused with its key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        intermediate_weight_saving_iterations (int): The value written in the configuration.
        expected_message (Optional[str]): The refusal message, None when the value must be accepted.
    """
    configuration = yaml.safe_load(stream=EXAMPLE_CONFIGURATION_PATH.read_text())
    configuration["ExperimentConfiguration"]["weight_saving_iterations"] = 5000
    configuration["ExperimentConfiguration"]["intermediate_weight_saving_iterations"] = \
        intermediate_weight_saving_iterations
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(yaml.safe_dump(configuration))

    # An accepted value is loaded as written, a refused one stops the loading with the message of its key
    if expected_message is None:
        experiment_configuration, _ = load_experiment_configuration(configuration_path=configuration_path)
        assert experiment_configuration["intermediate_weight_saving_iterations"] == \
            intermediate_weight_saving_iterations
    else:
        with pytest.raises(expected_exception=SystemExit):
            load_experiment_configuration(configuration_path=configuration_path)
        assert expected_message in capsys.readouterr().out


@pytest.mark.parametrize(argnames=("evaluation_score_threshold", "inference_score_threshold", "expected_message"),
                         argvalues=[(0.3, 0.3, None),
                                    (0.5, 0.3, "'ExperimentConfiguration.evaluation.score_threshold' (0.5) must not "
                                               "exceed 'Detector.inference.score_threshold' (0.3)")])
def test_evaluation_score_threshold_above_the_deployment_one_is_refused(
        tmp_path: Path, capsys: pytest.CaptureFixture, evaluation_score_threshold: float,
        inference_score_threshold: float, expected_message: Optional[str]) -> None:
    """
    Verifies that the object detection refuses an evaluation score threshold above the deployment one, with both keys:
    the confusion matrix, error images and sample images keep the detections from the deployment threshold among those
    of the evaluation, so the detections between both thresholds, returned by the deployed detector, would be counted
    as missed objects; an evaluation threshold equal to the deployment one is accepted.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        evaluation_score_threshold (float): The 'ExperimentConfiguration.evaluation.score_threshold' written.
        inference_score_threshold (float): The 'Detector.inference.score_threshold' written.
        expected_message (Optional[str]): The start of the refusal message, None when the values must be accepted.
    """
    configuration = yaml.safe_load(stream=EXAMPLE_CONFIGURATION_PATH.read_text())
    configuration["ExperimentConfiguration"]["evaluation"]["score_threshold"] = evaluation_score_threshold
    configuration["Detector"]["inference"]["score_threshold"] = inference_score_threshold
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(yaml.safe_dump(configuration))

    # Accepted thresholds are loaded as written, refused ones stop the loading with the message naming both keys
    if expected_message is None:
        experiment_configuration, model_configuration = load_experiment_configuration(
            configuration_path=configuration_path)
        assert experiment_configuration["evaluation"]["score_threshold"] == evaluation_score_threshold
        assert model_configuration["Detector"]["inference"]["score_threshold"] == inference_score_threshold
    else:
        with pytest.raises(expected_exception=SystemExit):
            load_experiment_configuration(configuration_path=configuration_path)
        assert expected_message in capsys.readouterr().out


def empty_detection_levels(configuration: Dict[str, Any]) -> None:
    """
    Removes every detection level.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["DetectionLevels"] = {}


def reverse_detection_levels(configuration: Dict[str, Any]) -> None:
    """
    Lists the detection levels from the coarsest to the finest one.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["DetectionLevels"] = dict(reversed(list(configuration["DetectionLevels"].items())))


@pytest.mark.parametrize(argnames=("modify_function", "expected_errors"), argvalues=[
    (empty_detection_levels, ["'DetectionLevels' must map at least one backbone block number to the settings of its "
                              "level, got {}."]),
    (reverse_detection_levels, ["'DetectionLevels' keys must be in increasing order, got ['7', '6', '5']."])])
def test_inconsistent_detection_levels_are_rejected(modify_function: Callable[[Dict[str, Any]], None],
                                                    expected_errors: List[str]) -> None:
    """
    Verifies the keys of the detection levels on the example configuration itself, since writing it back to a YAML
    file would sort its keys: at least one level must be given, the levels being in increasing order.

    Args:
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_errors (List[str]): The exact errors of the detection level checks.
    """
    configuration = yaml.safe_load(stream=EXAMPLE_CONFIGURATION_PATH.read_text())
    modify_function(configuration=configuration)
    assert check_detection_levels(model_configuration=configuration) == expected_errors


def test_missing_task_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a configuration without task, written before the task-aware configuration, is rejected with the
    path of the missing key.

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


def use_binary_labels(configuration: Dict[str, Any]) -> None:
    """
    Uses the binary label mapping, which only makes sense next to the background class of the object detector.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["Data"]["label_mapping"]["mode"] = "binary"


def set_invalid_head(configuration: Dict[str, Any]) -> None:
    """
    Sets an unknown pooling, a hidden layer without unit, a dropout of 1, and unknown output mode and class weighting.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"].update({"global_pooling": "median", "fully_connected": [0], "dropout_rate": 1.0,
                                                "output_mode": "multi", "class_weighting": "inverse"})


def list_invalid_target_precisions(configuration: Dict[str, Any]) -> None:
    """
    Gives a list of target precisions holding a precision above 1.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"]["target_precision"] = [0.99, 1.5]


def set_invalid_decorrelation(configuration: Dict[str, Any]) -> None:
    """
    Writes the decorrelation flag as a quoted string, with a weight of 0 and a hidden layer 0.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"]["decorrelation"] = {"activated": "yes", "weight": 0, "hidden_layer": 0}


def normalize_the_head_of_single_image_batches(configuration: Dict[str, Any]) -> None:
    """
    Normalizes the hidden layer of the head with the batch normalization, with batches of a single image.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["batch_size"] = 1
    configuration["ClassificationHead"].update({"fully_connected": [64], "fully_connected_normalization": "batch"})


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml"])
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
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert experiment_configuration["task"] == "classification"
    assert model_configuration["Data"]["label_dictionary"] == {1: 0, 2: 1, 3: 2, 4: 3}


@pytest.mark.parametrize(argnames=("jigsaw_grid_size", "image_size", "jigsaw_activated", "refused_by_jigsaw", "loads"),
                         argvalues=[(3, 256, True, True, False)])
def test_jigsaw_image_size_is_checked(tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch,
                                      jigsaw_grid_size: int, image_size: int, jigsaw_activated: bool,
                                      refused_by_jigsaw: bool, loads: bool) -> None:
    """
    Verifies that an activated random_jigsaw refuses a side of the network input that is not a multiple of its grid
    size, which would stop the training at its first batch, a grid of 3x3 pieces in both strength sets standing for a
    change of the hard-coded grid size.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture setting the grid size of both strength sets.
        jigsaw_grid_size (int): The grid size of random_jigsaw in both strength sets.
        image_size (int): The side of the network input, 'Data.image_settings.size'.
        jigsaw_activated (bool): Whether random_jigsaw is activated.
        refused_by_jigsaw (bool): Whether the jigsaw check must refuse the image size.
        loads (bool): Whether the configuration must load, an image of 1 pixel being refused by the size checks too.
    """
    monkeypatch.setitem(SUPERVISED_AUGMENTATION_STRENGTHS, "jigsaw_grid_size", jigsaw_grid_size)
    monkeypatch.setitem(SELF_SUPERVISED_AUGMENTATION_STRENGTHS, "jigsaw_grid_size", jigsaw_grid_size)
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="self_supervised",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=None)
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["Data"]["image_settings"]["size"] = image_size
    configuration["Data"]["data_augmentation"]["types"]["random_jigsaw"]["activated"] = jigsaw_activated
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))

    # The configuration loads or stops, the message of the jigsaw being printed only for its refusal
    if loads:
        load_experiment_configuration(configuration_path=configuration_path)
    else:
        with pytest.raises(expected_exception=SystemExit):
            load_experiment_configuration(configuration_path=configuration_path)
    jigsaw_message = (f"'Data.image_settings.size' must be a multiple of {jigsaw_grid_size} when "
                      f"'Data.data_augmentation.types.random_jigsaw.activated' is True, the jigsaw cutting the image "
                      f"into {jigsaw_grid_size}x{jigsaw_grid_size} pieces of whole pixels, got {image_size}.")
    assert (jigsaw_message in capsys.readouterr().out) == refused_by_jigsaw


# Message of the image size check for the synthetic classification backbone of 6 blocks, 5 pooling layers without
# the last max pooling and 6 with it
POOLING_SIZE_MESSAGE = ("'Data.image_settings.size' ({image_size}) must be a multiple of {required_multiple}, 2 to the "
                        "power of the {number_pooling_layers} Backbone pooling layers, so that every feature map "
                        "entering a pooling is even, e.g. {nearest_sizes}.")


@pytest.mark.parametrize(argnames=("image_size", "last_max_pooling", "expected_messages"), argvalues=[
    (224, True, [POOLING_SIZE_MESSAGE.format(image_size=224, required_multiple=64, number_pooling_layers=6,
                                             nearest_sizes="192 or 256")])])
def test_image_size_must_keep_every_pooled_feature_map_even(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                            image_size: int, last_max_pooling: bool,
                                                            expected_messages: List[str]) -> None:
    """
    Verifies that the image size is refused, with the nearest valid sizes, when a feature map entering a max pooling of
    the Backbone is odd: the last max pooling adds a halving, which refuses the odd 7x7 map that 224 gives after the 5
    other poolings.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        image_size (int): The side of the network input, 'Data.image_settings.size'.
        last_max_pooling (bool): Whether the last block of the backbone of 6 blocks ends with a max pooling.
        expected_messages (List[str]): The messages of the size checks, empty when the configuration must load.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="classification",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=None)
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["Data"]["image_settings"]["size"] = image_size
    configuration["Backbone"]["last_max_pooling"] = last_max_pooling
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))

    # An accepted size loads, a refused one stops the loading with every message of the size checks
    if not expected_messages:
        load_experiment_configuration(configuration_path=configuration_path)
        return
    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)
    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output


def keep_the_fixed_settings(configuration: Dict[str, Any]) -> None:
    """
    Keeps every setting removed from the configuration files for its fixed value (top-k, artifacts, resumption,
    diagnosis modes, mixed precision, gradient clipping norm, loaders, inception start, batch normalization
    momentums), next to an unknown pooling of the classification head.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    experiment_configuration = configuration["ExperimentConfiguration"]
    experiment_configuration["evaluation"]["top_k"] = 5
    experiment_configuration["tracking"]["log_artifacts"] = True
    experiment_configuration["resume_training"] = True
    experiment_configuration["diagnosis_modes"].update({"verbose": False, "compute_model_size": True})
    configuration["Optimization"]["mixed_precision"] = True
    configuration["Optimization"]["gradient_clipping_norm"] = None
    configuration["Data"]["loading"] = {"number_workers": 4, "prefetch_batches": 2}
    configuration["Backbone"]["modules"]["inception_starting_block"] = 3
    configuration["Backbone"]["normalization"]["batch_normalization_momentum"] = 0.1
    configuration["ClassificationHead"]["batch_normalization_momentum"] = 0.1
    configuration["ClassificationHead"]["global_pooling"] = "median"


def keep_the_fixed_head_settings(configuration: Dict[str, Any]) -> None:
    """
    Keeps the settings of the task heads removed from the configuration files for their fixed value, the batch
    normalization momentum of every head the configuration holds, and for the detection tasks the strict fallback
    assignment, the localisation loss betas, the box loss of the region proposal network and the anchors centred on
    the cell corners, next to an unknown optimizer.

    Args:
        configuration (Dict[str, Any]): The raw configuration of a task with heads, modified in place.
    """
    # Momentums written in a 'normalization' section or in the section of the head itself
    for section_name in ["RegionProposal", "Detector", "MaskHead", "HRNet"]:
        if section_name in configuration:
            configuration[section_name]["normalization"]["batch_normalization_momentum"] = 0.1
    for section_name in ["SegmentationDecoder", "ProjectionHead"]:
        if section_name in configuration:
            configuration[section_name]["batch_normalization_momentum"] = 0.1

    # Settings of the anchor assignment and of the box losses of the detection tasks
    if "RegionProposal" in configuration:
        configuration["RegionProposal"]["strict_fallback_assignment"] = False
        configuration["RegionProposal"]["localisation_loss_beta"] = "1/9"
        configuration["Detector"]["localisation_loss_beta"] = "1/9"
        configuration["RegionProposal"]["box_loss"] = "smooth_l1"
        configuration["RegionProposal"]["anchor_centres"] = "cell_corners"
    configuration["Optimization"]["optimizer"] = "lion"


@pytest.mark.parametrize(argnames=("task_folder", "modify_function", "removed_settings", "hidden_message"), argvalues=[
    ("classification", keep_the_fixed_settings,
     ["ExperimentConfiguration.evaluation.top_k", "ExperimentConfiguration.tracking.log_artifacts",
      "ExperimentConfiguration.resume_training", "ExperimentConfiguration.diagnosis_modes.compute_model_size",
      "ExperimentConfiguration.diagnosis_modes.verbose", "Optimization.mixed_precision",
      "Optimization.gradient_clipping_norm", "Data.loading", "Backbone.modules.inception_starting_block",
      "Backbone.normalization.batch_normalization_momentum",
      "ClassificationHead.batch_normalization_momentum"], "'ClassificationHead.global_pooling' must be one of"),
    ("instance_segmentation", keep_the_fixed_head_settings,
     ["RegionProposal.normalization.batch_normalization_momentum",
      "Detector.normalization.batch_normalization_momentum", "MaskHead.normalization.batch_normalization_momentum",
      "RegionProposal.strict_fallback_assignment", "RegionProposal.localisation_loss_beta",
      "Detector.localisation_loss_beta", "RegionProposal.box_loss", "RegionProposal.anchor_centres"],
     "'Optimization.optimizer' must be one of"),
    ("semantic_segmentation", keep_the_fixed_head_settings, ["SegmentationDecoder.batch_normalization_momentum"],
     "'Optimization.optimizer' must be one of"),
    ("self_supervised", keep_the_fixed_head_settings, ["ProjectionHead.batch_normalization_momentum"],
     "'Optimization.optimizer' must be one of")])
def test_removed_settings_are_refused_alone(tmp_path: Path, capsys: pytest.CaptureFixture, task_folder: str,
                                            modify_function: Callable[[Dict[str, Any]], None],
                                            removed_settings: List[str], hidden_message: str) -> None:
    """
    Verifies that a configuration still holding settings removed from the repository (here the settings fixed in the
    code) is refused with a message naming every removed setting, what replaced it and that it must be deleted, and
    alone: the other checks do not run, so that the reason of the refusal is never hidden behind the keys the removed
    setting leaves missing.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        task_folder (str): The folder of the configurations of the task.
        modify_function (Callable[[Dict[str, Any]], None]): Function keeping the removed settings.
        removed_settings (List[str]): The key paths of the removed settings.
        hidden_message (str): A message of another check, which must not be printed.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for removed_setting in removed_settings:
        replacement, _ = REMOVED_SETTINGS[removed_setting]
        assert (f"'{removed_setting}' was removed: {replacement}. Delete it from the configuration file."
                in printed_output)
    assert hidden_message not in printed_output and "Traceback" not in printed_output


def test_high_resolution_network_momentum_is_refused() -> None:
    """
    Verifies that the batch normalization momentum of the high resolution network of the keypoint detection, removed
    from the configuration files, is refused with what replaced it, checked on the sections directly since the
    synthetic keypoint configuration reads its labels before the checks.
    """
    errors = check_removed_settings(experiment_configuration={"task": "keypoint_detection"},
                                    model_configuration={"HRNet": {"normalization": {
                                        "feature_map_normalization": "batch", "batch_normalization_momentum": 0.1}}})
    replacement, _ = REMOVED_SETTINGS["HRNet.normalization.batch_normalization_momentum"]
    assert errors == [f"'HRNet.normalization.batch_normalization_momentum' was removed: {replacement}. Delete it from "
                      f"the configuration file."]


@pytest.mark.parametrize(argnames=("task_folder", "configuration_name"), argvalues=[
    ("sequence_classification", "configuration_synthetic_sequences.yaml")])
def test_sequence_tasks_keep_their_settings(tmp_path: Path, task_folder: str, configuration_name: str) -> None:
    """
    Verifies that the settings removed from the other tasks but still held by the sequence tasks, the sequence
    augmentation settings, load in their configurations, the number of candidate frames 'evaluation.top_k' being held
    by the frame selection only.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task_folder (str): The folder of the configurations of the task.
        configuration_name (str): The synthetic configuration of the task.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                  configuration_name=configuration_name, modify_function=None)

    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)

    augmentation_configuration = model_configuration["Data"]["data_augmentation"]
    assert "independent_frames" in augmentation_configuration and "shuffle_frames" in augmentation_configuration
    assert ("top_k" in experiment_configuration["evaluation"]) == (task_folder == "frame_selection")


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (use_binary_labels, ["'Data.label_mapping.mode' must be 'multi_class' for the classification tasks"]),
    (set_invalid_head, ["'ClassificationHead.global_pooling' must be one of ['average', 'max', 'flatten'], got "
                        "'median'.",
                        "'ClassificationHead.fully_connected' must be a list of positive integers, got [0].",
                        "'ClassificationHead.dropout_rate' must be a number within [0, 1), got 1.0.",
                        "'ClassificationHead.output_mode' must be one of ['multi_class', 'multi_label']",
                        "'ClassificationHead.class_weighting' must be one of ['none', 'balanced']"]),
    (list_invalid_target_precisions, ["'ExperimentConfiguration.evaluation.target_precision[1]' must be a "
                                      "probability within [0, 1], got 1.5."]),
    (set_invalid_decorrelation, ["'ClassificationHead.decorrelation.activated' must be True or False, got 'yes'.",
                                 "'ClassificationHead.decorrelation.weight' must be a number > 0, got 0.",
                                 "'ClassificationHead.decorrelation.hidden_layer' must be an integer >= 1, got 0."]),
    (normalize_the_head_of_single_image_batches, ["'ExperimentConfiguration.batch_size' must be at least 2 with "
                                                  "'ClassificationHead.fully_connected_normalization' 'batch': the "
                                                  "batch normalization of the hidden layers needs the statistics of "
                                                  "several samples."])])
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


def set_invalid_transformer_and_settings(configuration: Dict[str, Any]) -> None:
    """
    Uses a transformer whose model size is not a multiple of its heads, empty sequences and several classes per
    sequence.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    configuration["TemporalModel"].update({"type": "transformer", "hidden_size": 100, "number_heads": 8})
    configuration["Data"]["sequence_settings"]["maximum_sequence_length"] = 0
    configuration["ClassificationHead"]["output_mode"] = "multi_label"


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml"])
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
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert experiment_configuration["task"] == "sequence_classification"
    assert model_configuration["Data"]["label_dictionary"] == {1: 0, 2: 1, 3: 2, 4: 3}


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_invalid_transformer_and_settings, ["'TemporalModel.hidden_size' (100) must be a multiple of "
                                            "'TemporalModel.number_heads' (8) for the transformer.",
                                            "'Data.sequence_settings.maximum_sequence_length' must be an integer >= "
                                            "1",
                                            "'ClassificationHead.output_mode' must be 'multi_class' for the sequence "
                                            "tasks"])])
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


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml"])
def test_sequence_labelling_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the sequence labelling configurations shipped with the repository pass every consistency check,
    the example one with a bidirectional LSTM.

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
                                 "'temporal_convolution'] for this task, got 'mean_pooling'."])])
def test_inconsistent_sequence_labelling_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that the temporal modules unable to keep one vector per frame are rejected for the sequence labelling
    task, without rejecting the causal flag itself.

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


def use_causal_transformer(configuration: Dict[str, Any]) -> None:
    """
    Turns the temporal module into a causal transformer reading the frames forwards only, the online setting of the
    sequence labelling task in which every frame only sees the earlier frames, which no shipped sequence labelling
    configuration holds since the example one uses a bidirectional LSTM.

    Args:
        configuration (Dict[str, Any]): The raw sequence labelling configuration, modified in place.
    """
    configuration["TemporalModel"].update({"type": "transformer", "causal": True, "bidirectional": False})


def test_causal_transformer_is_accepted_for_sequence_labelling(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a sequence labelling (M2M) configuration with a causal, unidirectional transformer passes the loader
    and every consistency check, the many to many task being the only one allowed causal modules, and that the same
    configuration made bidirectional is refused, a causal module never reading the sequence backwards.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    # The causal transformer built from the example configuration is loaded without any consistency error
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_labelling",
                                                  configuration_name="configuration_example.yaml",
                                                  modify_function=use_causal_transformer)
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert experiment_configuration["task"] == "sequence_labelling"
    assert {key: model_configuration["TemporalModel"][key] for key in ["type", "causal", "bidirectional"]} == \
        {"type": "transformer", "causal": True, "bidirectional": False}
    assert "'TemporalModel." not in capsys.readouterr().out

    # The same configuration reading the sequence both ways is refused with the message of the causal flag
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["TemporalModel"]["bidirectional"] = True
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)
    assert ("'TemporalModel.bidirectional' must be False when 'TemporalModel.causal' is True"
            in capsys.readouterr().out)


@pytest.mark.parametrize(argnames=("task_folder", "temporal_settings", "refused"), argvalues=[
    ("sequence_labelling", {"type": "transformer", "causal": True, "bidirectional": False}, True),
    ("sequence_labelling", {"type": "gru", "causal": False, "bidirectional": False}, True),
    ("frame_selection", {"type": "temporal_convolution", "causal": True, "bidirectional": False}, True),
    ("frame_selection", {"type": "lstm", "causal": False, "bidirectional": False}, True),
    ("sequence_labelling", {"type": "lstm", "causal": False, "bidirectional": True}, False),
    ("frame_selection", {"type": "transformer", "causal": False, "bidirectional": False}, False),
    ("sequence_classification", {"type": "gru", "causal": False, "bidirectional": False}, False)])
def test_causal_temporal_module_refuses_the_sequence_normalization(
        tmp_path: Path, capsys: pytest.CaptureFixture, task_folder: str, temporal_settings: Dict[str, Any],
        refused: bool) -> None:
    """
    Verifies that the sequence labelling and frame selection refuse a causal temporal module (the causal flag, or a GRU
    or LSTM reading the frames forwards only) with the sequence normalization of the backbone, whose statistics come
    from every frame of the sequence, so that a model meant for live streams never sees the frames after a frame,
    while the modules seeing the whole sequence, and the sequence classification summarising it, keep this
    normalization.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        task_folder (str): The folder of the configurations of the task.
        temporal_settings (Dict[str, Any]): The type and direction flags given to the temporal module.
        refused (bool): Whether the configuration must be refused.
    """
    # The example configuration of the task with the temporal module of the case and the sequence normalization
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                  configuration_name="configuration_example.yaml",
                                                  modify_function=None)
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["TemporalModel"].update(temporal_settings)
    configuration["Backbone"]["normalization"]["feature_map_normalization"] = "sequence"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    expected_message = ("'Backbone.normalization.feature_map_normalization' cannot be 'sequence' with a causal "
                        "temporal module")

    # A causal module stops the loading with the message of the normalization, the others load without it
    if refused:
        with pytest.raises(expected_exception=SystemExit):
            load_experiment_configuration(configuration_path=configuration_path)
        assert expected_message in capsys.readouterr().out
    else:
        _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
        assert model_configuration["Backbone"]["normalization"]["feature_map_normalization"] == "sequence"
        assert expected_message not in capsys.readouterr().out


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml"])
def test_frame_selection_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the frame selection configurations shipped with the repository pass every consistency check, their
    sequence settings holding the maximum sequence length only.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the frame selection configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="frame_selection",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert experiment_configuration["task"] == "frame_selection"
    assert sorted(model_configuration["Data"]["sequence_settings"]) == ["maximum_sequence_length"]


def set_invalid_selection(configuration: Dict[str, Any]) -> None:
    """
    Gives every setting of the 'Selection' section an invalid value.

    Args:
        configuration (Dict[str, Any]): The raw frame selection configuration, modified in place.
    """
    configuration["Selection"] = {"unselected_category": " ", "decoding": "beam", "frames_per_class": 0,
                                  "selection_loss_weight": -1.0}


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_invalid_selection, ["'Selection.unselected_category' must be the non-empty name of a manifest category, "
                             "got ' '.", "'Selection.decoding' must be one of ['hungarian', 'greedy'], got 'beam'.",
                             "'Selection.frames_per_class' must be", "'Selection.selection_loss_weight' must be a "
                             "number >= 0, got -1.0."])])
def test_inconsistent_frame_selection_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that an invalid 'Selection' section stops the loading of a frame selection configuration with a short
    message naming every faulty key.

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


def remove_maximum_sequence_length(configuration: Dict[str, Any]) -> None:
    """
    Removes the maximum sequence length from the sequence settings.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    del configuration["Data"]["sequence_settings"]["maximum_sequence_length"]


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml"])
def test_sequence_comparison_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the sequence comparison configurations shipped with the repository pass every consistency check,
    their sequence settings holding the maximum sequence length only.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the sequence comparison configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="sequence_comparison",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert experiment_configuration["task"] == "sequence_comparison"
    assert sorted(model_configuration["Data"]["sequence_settings"]) == ["maximum_sequence_length"]


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_unknown_fusion, ["'Fusion.type' must be one of ['concatenation', 'difference', 'cross_attention'], got "
                          "'sum'.", "'Fusion.dropout_rate' must be a number within [0, 1), got 1.0."])])
def test_inconsistent_sequence_comparison_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that an inconsistent fusion of the members stops the loading of a sequence comparison configuration with
    a short message naming the faulty key.

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


@pytest.mark.parametrize(argnames=("task_folder", "configuration_name"),
                         argvalues=[("sequence_classification", "configuration_synthetic_sequences.yaml")])
def test_sequence_tasks_require_the_maximum_sequence_length(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                            task_folder: str, configuration_name: str) -> None:
    """
    Verifies that a sequence task requires the maximum sequence length, which leaves the longer sequences out of
    the splits and sizes the positional embedding of the transformer.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        task_folder (str): The folder of the configurations of the sequence task.
        configuration_name (str): The file name of the synthetic configuration of the task.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                  configuration_name=configuration_name,
                                                  modify_function=remove_maximum_sequence_length)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    assert ("Missing configuration key 'Data.sequence_settings.maximum_sequence_length'." in
            capsys.readouterr().out)


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


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (freeze_random_backbone, ["'Backbone.pretrained.freeze' must be False when 'Backbone.pretrained.experiment_folder' "
                              "is null: only a transferred backbone can be frozen.",
                              "'Backbone.pretrained.learning_rate_multiplier' must be > 0, got 0."]),
    (set_invalid_pretrained_run, ["'Backbone.pretrained.experiment_folder' must be null or the path of a run "
                                  "folder, got 5.",
                                  "'Backbone.pretrained.iteration' must be an integer >= 1, got 0."])])
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
    the learning rate, and a moving average switch that is not a boolean with a decay of 1, which would never move.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Optimization"].update({"optimizer": "rmsprop", "learning_rate_schedule": "step",
                                          "weight_decay": -0.1, "decay_iterations": 0,
                                          "minimum_learning_rate_ratio": 1.5,
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
                             "'Optimization.exponential_moving_average.activated' must be True or False, got 'yes'.",
                             "'Optimization.exponential_moving_average.decay' must be strictly between 0 and 1, "
                             "got 1.0."]:
        assert expected_message in printed_output


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml"])
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
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert experiment_configuration["task"] == "semantic_segmentation"
    assert model_configuration["Data"]["label_dictionary"] == {1: 1, 2: 2, 3: 3, 4: 4, "background": 0}


def remove_decoder_level(configuration: Dict[str, Any]) -> None:
    """
    Removes the decoder level of the skip connection of the first backbone block.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    del configuration["SegmentationDecoder"]["convolutions"]["1"]


def set_invalid_decoder(configuration: Dict[str, Any]) -> None:
    """
    Sets an unknown upsampling and normalization, a decoder level without layer, a module flag written as a string and
    a dropout of 1.

    Args:
        configuration (Dict[str, Any]): The raw segmentation configuration, modified in place.
    """
    configuration["SegmentationDecoder"].update({"upsampling": "nearest", "feature_map_normalization": "group",
                                                 "dropout_rate": 1.0})
    configuration["SegmentationDecoder"]["convolutions"]["3"] = [0, 32]
    configuration["SegmentationDecoder"]["modules"]["squeeze_and_excite"] = "yes"


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


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (remove_decoder_level, ["'SegmentationDecoder.convolutions' keys must be the backbone blocks of its skip "
                            "connections ['5', '4', '3', '2', '1']", "missing ['1'], unexpected []."]),
    (set_invalid_decoder, ["'SegmentationDecoder.upsampling' must be one of ['transposed', 'bilinear'], got "
                           "'nearest'.",
                           "'SegmentationDecoder.convolutions.3' must be [number_layers, output_channels] with "
                           "positive integers, got [0, 32].",
                           "'SegmentationDecoder.modules.squeeze_and_excite' must be True or False, got 'yes'.",
                           "'SegmentationDecoder.feature_map_normalization' must be one of ['batch', 'layer', "
                           "'none'], got 'group'.",
                           "'SegmentationDecoder.dropout_rate' must be a number within [0, 1), got 1.0."]),
    (remove_both_losses, ["'SegmentationLoss.cross_entropy_weight', 'SegmentationLoss.dice_weight' and "
                          "'SegmentationLoss.lovasz_weight' cannot all be 0, the regions would have no loss.",
                          "'SegmentationLoss.boundary_weight' must be a number >= 0, got True.",
                          "'SegmentationLoss.boundary_distance' must be an integer >= 0, got -1.",
                          "'SegmentationLoss.focal_gamma' must be a number >= 0, got -1.0.",
                          "'SegmentationLoss.label_smoothing' must be a number within [0, 1), got 1.0.",
                          "'SegmentationLoss.class_weighting' must be one of ['none', 'balanced'], got 'inverse'."])])
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
    ({"1": 1, "2": 3}, "indexed 0 (background) to K-1 without gaps, got [0, 1, 3].")])
def test_segmentation_label_mapping_gives_mask_values(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                      label_mapping: Dict[str, int],
                                                      expected_message: str | None) -> None:
    """
    Verifies that the label mapping file of the semantic segmentation maps its mask values to classes indexed from the
    background 0 without gaps.

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

    # An accepted mapping keeps its background category next to the background class, a rejected one is reported
    # without traceback
    if expected_message is None:
        _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
        assert model_configuration["Data"]["label_dictionary"] == {0: 0, 1: 1, 2: 2, "background": 0}
        return
    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)
    printed_output = capsys.readouterr().out
    assert expected_message in printed_output
    assert "Traceback" not in printed_output


@pytest.mark.parametrize(argnames=("label_mapping", "expected_error", "expected_warning"), argvalues=[
    ({"1": 1, "3": 3}, "indexed 0 (background) to K-1 without gaps, got [0, 1, 3].", None)])
def test_detection_label_mapping_indexes_its_classes_from_the_background(
        tmp_path: Path, capsys: pytest.CaptureFixture, label_mapping: Dict[str, int], expected_error: str | None,
        expected_warning: str | None) -> None:
    """
    Verifies that the label mapping of the object detection, whose classes are indexed from the background 0, refuses
    classes with gaps (original COCO identifiers kept).

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        label_mapping (Dict[str, int]): The content of the label mapping file.
        expected_error (str | None): The error printed for the mapping, None when it is accepted.
        expected_warning (str | None): The warning printed for an accepted mapping, None for none.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="object_detection",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=None)
    (tmp_path / "label_mapping.json").write_text(data=json.dumps(obj=label_mapping))

    # An accepted mapping gives the background class 0 and counts every class once
    if expected_error is None:
        _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
        label_dictionary = model_configuration["Data"]["label_dictionary"]
        assert label_dictionary["background"] == 0 and len(set(label_dictionary.values())) == 3
        printed_output = capsys.readouterr().out
        assert (expected_warning in printed_output) if expected_warning is not None else (
            "mapped to the background class 0" not in printed_output)
        return
    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)
    printed_output = capsys.readouterr().out
    assert expected_error in printed_output
    assert "Traceback" not in printed_output


def test_run_folder_configuration_takes_its_saved_label_dictionary(tmp_path: Path) -> None:
    """
    Verifies that the configuration of a run folder (evaluation, export, model loading) takes the label dictionary
    saved with the run ('label_dictionary.json') rather than the mapping file edited since, the classes of its
    checkpoints keeping their meaning, and that a configuration without saved dictionary takes the mapping file.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="object_detection",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=None)
    (tmp_path / "label_mapping.json").write_text(data=json.dumps(obj={"1": 1, "2": 2}))
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["Data"]["label_dictionary"] == {"background": 0, 1: 1, 2: 2}

    # The run saved its classes before the mapping file swapped them: the saved classes are kept
    write_run_label_dictionary(label_dictionary={"background": 0, 1: 2, 2: 1}, run_folder=tmp_path)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["Data"]["label_dictionary"] == {"background": 0, 1: 2, 2: 1}


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml"])
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
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert experiment_configuration["task"] == "instance_segmentation"
    assert model_configuration["Data"]["label_dictionary"]["background"] == 0
    assert model_configuration["MaskHead"]["mask_resolution"] == 28
    assert model_configuration["DetectionLevels"]["5"]["mask_head"]["roi_align_pool_size"] == 14
    assert model_configuration["Data"]["instance_index_maps"] is False


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


@pytest.mark.parametrize(argnames="task_folder", argvalues=["instance_segmentation"])
def test_transferred_mask_heads_and_class_layers_are_accepted(tmp_path: Path, task_folder: str) -> None:
    """
    Verifies that consistent pretrained settings of the mask heads of an instance segmenter pass the checks, the source
    run being only read when the training starts.

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


def remove_level_mask_head(configuration: Dict[str, Any]) -> None:
    """
    Removes the mask head of the seventh detection level, whose boxes would then have no mask.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    del configuration["DetectionLevels"]["7"]["mask_head"]


def set_invalid_mask_head(configuration: Dict[str, Any]) -> None:
    """
    Sets an invalid mask resolution, upsampling, pool size, convolutions, normalization, module flag and mask
    threshold, and inception modules with a channel count that is not a multiple of 8.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    mask_head_configuration = configuration["MaskHead"]
    mask_head_configuration["mask_resolution"] = 0
    mask_head_configuration["upsampling"] = "nearest"
    configuration["DetectionLevels"]["5"]["mask_head"] = {"roi_align_pool_size": 0, "convolutions": [4]}
    configuration["DetectionLevels"]["6"]["mask_head"]["convolutions"] = [4, 60]
    mask_head_configuration["normalization"]["feature_map_normalization"] = "group"
    mask_head_configuration["modules"].update({"inception": True, "residual_nets": "yes"})
    mask_head_configuration["inference"]["mask_threshold"] = 1.5


def set_invalid_instance_index_maps(configuration: Dict[str, Any]) -> None:
    """
    Sets the instance index maps of the loaders to a string instead of a boolean.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["Data"]["instance_index_maps"] = "yes"


def remove_instance_index_maps(configuration: Dict[str, Any]) -> None:
    """
    Removes the instance index maps of the loaders, as in the files written before the setting.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    del configuration["Data"]["instance_index_maps"]


def use_every_mask_head_fine_input(configuration: Dict[str, Any]) -> None:
    """
    Turns on every finer input of the mask heads, the backbone skips including the normalized input image (block 0)
    and block 5 before its pooling (stride 16), the coarsest skip finer than the finest detection level 5 (stride 32).

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["MaskHead"].update({"skip_blocks_at_pool_size": [4, 5], "image_branch_convolutions": [5, 32],
                                      "skip_blocks_at_mask_resolution": [0, 3], "fusion_convolutions": [1, 64]})


def test_mask_head_fine_inputs_are_accepted(tmp_path: Path) -> None:
    """
    Verifies that the finer inputs of the mask heads and their fusion convolutions pass the checks when consistent.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="instance_segmentation",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=use_every_mask_head_fine_input)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["MaskHead"]["image_branch_convolutions"] == [5, 32]
    assert model_configuration["MaskHead"]["skip_blocks_at_mask_resolution"] == [0, 3]


def set_invalid_mask_head_fine_inputs(configuration: Dict[str, Any]) -> None:
    """
    Sets fusion convolutions with layers but no channel, an image branch without channel count, a skip block listed
    twice at the pool size and a skip block beyond the 7 blocks of the backbone at the mask resolution.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["MaskHead"].update({"fusion_convolutions": [1, 0], "image_branch_convolutions": [2],
                                      "skip_blocks_at_pool_size": [4, 4], "skip_blocks_at_mask_resolution": [8]})


def use_skip_blocks_as_coarse_as_the_levels(configuration: Dict[str, Any]) -> None:
    """
    Reads block 6 before its pooling (stride 32, as coarse as the finest detection level 5) at the pool size and the
    last block of the backbone, block 7 (stride 64), at the mask resolution, neither finer than the levels.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["MaskHead"].update({"skip_blocks_at_pool_size": [4, 6], "skip_blocks_at_mask_resolution": [7]})


def use_skip_blocks_coarser_at_the_mask_resolution(configuration: Dict[str, Any]) -> None:
    """
    Reads block 4 (stride 8) at the mask resolution while block 3 (stride 4) is read at the pool size, the finer map
    joining the coarser stage, with block 3 read at both sizes.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["MaskHead"].update({"skip_blocks_at_pool_size": [3], "skip_blocks_at_mask_resolution": [0, 3, 4]})


def transfer_mask_heads_with_an_image_branch(configuration: Dict[str, Any]) -> None:
    """
    Initialises the mask heads with those of another run while turning the image branch on, which the transfer does
    not load.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["MaskHead"]["pretrained"]["experiment_folder"] = "/runs/source"
    configuration["MaskHead"]["image_branch_convolutions"] = [5, 32]


def remove_mask_head_fusion_convolutions(configuration: Dict[str, Any]) -> None:
    """
    Removes the fusion convolutions of the mask heads, as in the files written before the finer inputs.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    del configuration["MaskHead"]["fusion_convolutions"]


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (remove_level_mask_head, ["'DetectionLevels.7' must hold exactly ['anchors', 'region_proposal', 'detector', "
                              "'mask_head'], got ['anchors', 'detector', 'region_proposal']."]),
    (set_invalid_instance_index_maps, ["'Data.instance_index_maps' must be True or False, got 'yes'."]),
    (remove_instance_index_maps, ["Missing configuration key 'Data.instance_index_maps'."]),
    (set_invalid_mask_head_fine_inputs, [
        "'MaskHead.fusion_convolutions' must be [number_layers, output_channels] with positive integers, or [0, 0] "
        "when unused, got [1, 0].",
        "'MaskHead.image_branch_convolutions' must be [number_layers, output_channels] with positive integers, or "
        "[0, 0] when unused, got [2].",
        "'MaskHead.skip_blocks_at_pool_size' must be a list of distinct backbone blocks between 0 (the normalized "
        "input image) and 7, empty when unused, got [4, 4].",
        "'MaskHead.skip_blocks_at_mask_resolution' must be a list of distinct backbone blocks between 0 (the "
        "normalized input image) and 7, empty when unused, got [8]."]),
    (use_skip_blocks_as_coarse_as_the_levels, [
        "'MaskHead.skip_blocks_at_pool_size' must only hold blocks finer than the finest detection level 5 (stride "
        "32), a block before its pooling being at stride 2 ** (block - 1), got [6].",
        "'MaskHead.skip_blocks_at_mask_resolution' must only hold blocks finer than the finest detection level 5 "
        "(stride 32), a block before its pooling being at stride 2 ** (block - 1), got [7]."]),
    (use_skip_blocks_coarser_at_the_mask_resolution, [
        "'MaskHead.skip_blocks_at_pool_size' and 'MaskHead.skip_blocks_at_mask_resolution' must not share a block, "
        "the same features being then read twice, got [3].",
        "'MaskHead.skip_blocks_at_mask_resolution' must only hold blocks finer than every block of "
        "'MaskHead.skip_blocks_at_pool_size' [3] (finest stride 4), got [4]."]),
    (transfer_mask_heads_with_an_image_branch, [
        "'MaskHead.image_branch_convolutions' must be [0, 0] when 'MaskHead.pretrained.experiment_folder' is set: the "
        "image branch is not transferred with the mask heads, got [5, 32]."]),
    (remove_mask_head_fusion_convolutions, ["Missing configuration key 'MaskHead.fusion_convolutions'."]),
    (set_invalid_mask_head, ["'MaskHead.mask_resolution' must be an integer >= 1, got 0.",
                             "'MaskHead.upsampling' must be one of ['transposed', 'bilinear'], got 'nearest'.",
                             "'DetectionLevels.5.mask_head.roi_align_pool_size' must be an integer >= 1, got 0.",
                             "'DetectionLevels.5.mask_head.convolutions' must be [number_layers, output_channels] "
                             "with positive integers, got [4].",
                             "'DetectionLevels.6.mask_head.convolutions' has 60 output channels, inception "
                             "modules require a multiple of 8.",
                             "'MaskHead.normalization.feature_map_normalization' must be one of",
                             "'MaskHead.modules.residual_nets' must be True or False, got 'yes'.",
                             "'MaskHead.inference.mask_threshold' must be a probability within [0, 1], got 1.5."])])
def test_inconsistent_instance_segmentation_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the mask head section and of the instance index maps stops the loading with
    a short message naming the faulty key, the mask head of every detection level included.

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


def test_skip_and_level_strides() -> None:
    """
    Verifies the strides the skip rules of the mask heads compare: the normalized input image and block 1 before its
    pooling at stride 1, block b before its pooling at 2 ** (b - 1), a detection level at 2 ** level after its pooling,
    and the last block of the backbone at 2 ** (level - 1) unless 'Backbone.last_max_pooling' pools it.
    """
    assert [compute_skip_stride(block=block) for block in [0, 1, 3, 4, 7]] == [1, 1, 4, 8, 64]
    assert compute_level_stride(level=5, number_blocks=7, last_max_pooling=False) == 32
    assert compute_level_stride(level=7, number_blocks=7, last_max_pooling=False) == 64
    assert compute_level_stride(level=7, number_blocks=7, last_max_pooling=True) == 128


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml"])
def test_self_supervised_configurations_are_consistent(tmp_path: Path, configuration_name: str) -> None:
    """
    Verifies that the self-supervised configurations shipped with the repository pass every consistency check, with
    the classes of the label mapping file for the k-nearest neighbour evaluation.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the self-supervised configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="self_supervised",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert experiment_configuration["task"] == "self_supervised"
    assert model_configuration["Data"]["label_dictionary"] == {1: 0, 2: 1, 3: 2, 4: 3}


def set_invalid_projection_head(configuration: Dict[str, Any]) -> None:
    """
    Sets an unknown pooling and method, an empty list of layers, a temperature of 0 and an off-diagonal weight given
    as a boolean in the projection head.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["ProjectionHead"].update({"global_pooling": "median", "fully_connected": [], "method": "byol",
                                            "simclr": {"temperature": 0},
                                            "barlow_twins": {"off_diagonal_weight": True}})


def set_invalid_grouped_batches(configuration: Dict[str, Any]) -> None:
    """
    Turns the grouped batches on with a block key that is not a string, an empty batch key and blocks of 3 images,
    which do not divide the batch of 128 images.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["Data"]["grouped_batches"].update({"activated": True, "block_key": 3, "batch_key": "",
                                                     "images_per_block": 3})


def set_invalid_reference_images(configuration: Dict[str, Any]) -> None:
    """
    Asks for no reference training image per class in the k-nearest neighbour evaluation, which would leave the
    labelled test images without neighbours.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"]["number_reference_images_per_class"] = 0


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_invalid_projection_head, ["'ProjectionHead.global_pooling' must be one of ['average', 'max', 'flatten'], "
                                   "got 'median'.",
                                   "'ProjectionHead.fully_connected' must be a non-empty list of positive integers, "
                                   "the last one being the embedding size, got [].",
                                   "'ProjectionHead.method' must be one of ['simclr', 'barlow_twins'], got 'byol'.",
                                   "'ProjectionHead.simclr.temperature' must be a number > 0, got 0.",
                                   "'ProjectionHead.barlow_twins.off_diagonal_weight' must be a number > 0, got "
                                   "True."]),
    (set_invalid_grouped_batches, ["'Data.grouped_batches.block_key' must be a non-empty string, got 3.",
                                   "'Data.grouped_batches.batch_key' must be null or a non-empty string, got ''.",
                                   "'Data.grouped_batches.images_per_block' (3) must divide "
                                   "'ExperimentConfiguration.batch_size' (128) when the grouped batches are on: a "
                                   "batch holds a whole number of blocks."]),
    (set_invalid_reference_images, ["'ExperimentConfiguration.evaluation.number_reference_images_per_class' must be "
                                    "an integer >= 1, got 0."])])
def test_inconsistent_self_supervised_configuration_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                                modify_function: Callable[[Dict[str, Any]], None],
                                                                expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the self-supervised sections (projection head, grouped batches, reference
    images of the k-NN evaluation) stops the loading with a short message naming the faulty key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="self_supervised",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


def write_keypoint_configuration(temporary_directory: Path, configuration_name: str,
                                 modify_function: Callable[[Dict[str, Any]], None] | None,
                                 label_mapping: Dict[str, int] | None = None) -> Path:
    """
    Writes a copy of a keypoint detection configuration with the two files the configuration loader reads before the
    checks, both written in the temporary directory since the shipped ones only exist once the datasets are created:
    the 'labels.json' of the training split, whose categories give the keypoint definitions injected into
    'Data.keypoint_categories' (the 37 synthetic categories, or the COCO person of the example configuration), and the
    label mapping file (the mapping written by the synthetic dataset, which leaves out the hollow distractors, or the
    person to class 1, unless a mapping is given), then alters it by the given function.

    Args:
        temporary_directory (Path): The pytest temporary directory receiving the configuration and its files.
        configuration_name (str): The file name of the keypoint detection configuration.
        modify_function (Callable[[Dict[str, Any]], None] | None): Function altering the raw configuration in place,
            or None to keep it.
        label_mapping (Dict[str, int] | None): The class of every raw category identifier. Defaults to None, in which
            case the synthetic categories with keypoints get the classes 1 to 33 and the COCO person the class 1.

    Returns:
        Path: The path of the written configuration file.
    """
    configuration = yaml.safe_load(stream=(CONFIGURATIONS_FOLDER / "keypoint_detection" /
                                           configuration_name).read_text())

    # Categories of the training split: the synthetic shapes, or the COCO person with its 1-based skeleton
    if configuration_name == "configuration_synthetic_shapes.yaml":
        categories = build_keypoint_labels(image_records=[], image_size=512, split_name="train")[0]["categories"]
    else:
        categories = [{"id": 1, "name": "person", "supercategory": "person", "keypoints": COCO_PERSON_KEYPOINTS,
                       "skeleton": COCO_PERSON_SKELETON}]
    train_folder = temporary_directory / "train"
    train_folder.mkdir(exist_ok=True)
    (train_folder / "labels.json").write_text(data=json.dumps(obj={"images": [], "annotations": [],
                                                                   "categories": categories}))
    configuration["ExperimentConfiguration"]["train_data_folder"] = str(train_folder)

    # Label mapping of the categories: the one of the synthetic dataset, or the person to its own class by default
    if label_mapping is None and configuration_name == "configuration_synthetic_shapes.yaml":
        label_mapping = build_keypoint_label_mapping()
    elif label_mapping is None:
        label_mapping = {str(category["id"]): category["id"] for category in categories}
    mapping_path = temporary_directory / "label_mapping.json"
    mapping_path.write_text(data=json.dumps(obj=label_mapping))
    configuration["Data"]["label_mapping"]["mapping_file_path"] = str(mapping_path)
    if modify_function is not None:
        modify_function(configuration=configuration)

    configuration_path = temporary_directory / "keypoint_detection_configuration.yaml"
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    return configuration_path


def test_keypoint_configurations_are_consistent_with_their_injected_categories(tmp_path: Path) -> None:
    """
    Verifies that both keypoint detection configurations shipped with the repository pass every consistency check, and
    that the configuration loader injects the keypoint definition of every class from the training labels: the
    keypoint names and the skeleton turned 0-based, without any other field (the keypoints keeping their names whatever
    the geometric augmentation).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Example configuration: the COCO person
    example_path = write_keypoint_configuration(temporary_directory=tmp_path,
                                                configuration_name="configuration_example.yaml", modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=example_path)
    assert experiment_configuration["task"] == "keypoint_detection"
    assert model_configuration["Data"]["label_dictionary"] == {"background": 0, 1: 1}
    person = model_configuration["Data"]["keypoint_categories"][1]
    assert person == {"name": "person", "keypoints": COCO_PERSON_KEYPOINTS,
                      "skeleton": [[start - 1, end - 1] for start, end in COCO_PERSON_SKELETON]}

    # Synthetic configuration: the 33 classes with keypoints
    synthetic_path = write_keypoint_configuration(
        temporary_directory=tmp_path, configuration_name="configuration_synthetic_shapes.yaml", modify_function=None)
    _, model_configuration = load_experiment_configuration(configuration_path=synthetic_path)
    keypoint_categories = model_configuration["Data"]["keypoint_categories"]
    label_dictionary = model_configuration["Data"]["label_dictionary"]
    assert sorted(keypoint_categories) == list(range(1, 34))
    assert all(KEYPOINT_CATEGORIES[name] not in label_dictionary
               for name in ["circle", "ellipse", "annulus", "hollow_ellipse"])
    triangle = keypoint_categories[label_dictionary[KEYPOINT_CATEGORIES["triangle"]]]
    assert triangle == {"name": "triangle", "keypoints": ["top", "bottom_left", "bottom_right"],
                        "skeleton": [[0, 2], [2, 1], [1, 0]]}
    stick_figure = keypoint_categories[label_dictionary[KEYPOINT_CATEGORIES["stick_figure"]]]
    assert stick_figure["skeleton"][0] == [0, 1]


def set_image_size_not_matching_the_branches(configuration: Dict[str, Any]) -> None:
    """
    Sets an image size that is not a multiple of the stride of the lowest resolution branch of the high resolution
    network, whose upsampled feature maps would then not add up cell by cell with the higher resolution ones.

    Args:
        configuration (Dict[str, Any]): The raw keypoint detection configuration, modified in place.
    """
    configuration["Data"]["image_settings"]["size"] = 500


def set_unknown_keypoint_sigmas(configuration: Dict[str, Any]) -> None:
    """
    Sets a zero default keypoint sigma, and keypoint sigmas naming an unknown keypoint of a learnt category and an
    unknown category.

    Args:
        configuration (Dict[str, Any]): The raw keypoint detection configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"]["default_keypoint_sigma"] = 0
    configuration["ExperimentConfiguration"]["evaluation"]["keypoint_sigmas"] = {"stick_figure": {"left_wing": 0.05},
                                                                                 "dragon": {"tail": 0.1}}


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_image_size_not_matching_the_branches, ["'Data.image_settings.size' (500) must be a multiple of 32, the "
                                                "stride of the lowest resolution branch of the 3 stages of "
                                                "'HRNet.modules_per_stage'."]),
    (set_unknown_keypoint_sigmas, ["'ExperimentConfiguration.evaluation.default_keypoint_sigma' must be a positive "
                                   "number, got 0.",
                                   "'ExperimentConfiguration.evaluation.keypoint_sigmas.dragon' names a category that "
                                   "is not learnt from the training split",
                                   "'ExperimentConfiguration.evaluation.keypoint_sigmas.stick_figure' names unknown "
                                   "keypoints ['left_wing'], the keypoints of the category being ['head', 'neck', "
                                   "'left_elbow', 'right_elbow', 'left_hand', 'right_hand', 'pelvis', 'left_knee', "
                                   "'right_knee', 'left_foot', 'right_foot']."])])
def test_inconsistent_keypoint_configuration_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                         modify_function: Callable[[Dict[str, Any]], None],
                                                         expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the keypoint detection sections (image size against the branches of the high
    resolution network, keypoint sigmas of the evaluation) stops the loading with a short message naming the faulty
    key.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        modify_function (Callable[[Dict[str, Any]], None]): Function introducing the inconsistency.
        expected_messages (List[str]): Message fragments that must all appear in the printed output.
    """
    configuration_path = write_keypoint_configuration(temporary_directory=tmp_path,
                                                      configuration_name="configuration_synthetic_shapes.yaml",
                                                      modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    for expected_message in expected_messages:
        assert expected_message in printed_output
    assert "Traceback" not in printed_output


def give_merged_categories_different_sigmas(configuration: Dict[str, Any]) -> None:
    """
    Gives two different sigmas to the top left corner of the square and of the rectangle, merged into one class by the
    label mapping of the test.

    Args:
        configuration (Dict[str, Any]): The raw keypoint detection configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"]["keypoint_sigmas"] = {"square": {"top_left": 0.03},
                                                                                 "rectangle": {"top_left": 0.05}}


def test_merged_keypoint_categories_cannot_give_different_sigmas(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the categories merged into one class, which share its heatmap channels and its sigmas, cannot give
    different sigmas to one keypoint: the square and the rectangle, both made of the same four corners, are learnt as
    the class 'square+rectangle', so their conflicting top left sigmas stop the loading.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration_path = write_keypoint_configuration(
        temporary_directory=tmp_path, configuration_name="configuration_synthetic_shapes.yaml",
        modify_function=give_merged_categories_different_sigmas,
        label_mapping={str(KEYPOINT_CATEGORIES["square"]): 1, str(KEYPOINT_CATEGORIES["rectangle"]): 1})

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    assert ("The categories merged into the class 'square+rectangle' give different sigmas [0.03, 0.05] to the "
            "keypoint 'top_left' in 'evaluation.keypoint_sigmas'.") in printed_output
    assert "Traceback" not in printed_output
