"""
Tests of the configuration checks of vision_training/configuration_processing/configuration_checker.py, run through
load_experiment_configuration: the configurations shipped for every task are accepted, while inconsistent settings
(detection, classification, sequence, comparison, segmentation, self-supervised, keypoint detection, normalization,
pretrained transfers with the initialisation of their new classes, optimization) and missing keys are rejected with
errors naming their path, the run name suffix being accepted null or made of letters, digits, '-' and '.' only, and the
intermediate weight saving 0 (none) or a positive integer smaller than the evaluated weight saving. The
augmentation types moving the image only are refused for the tasks with boxes, masks or keypoints and accepted for the
others, the jigsaw refuses an image size that is not a multiple of its grid size, and a section removed from the
repository is refused alone, before the other checks. The keypoint definitions injected by the loader from the training
labels are checked too, and so is the label dictionary saved with a run, taken by the configuration of its folder.
"""

import copy
import json
import pytest
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from vision_training.configuration_processing.common_checks import REMOVED_SECTIONS
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.configuration_processing.detection_checks import check_detection_levels
from vision_training.data.augmentations import SELF_SUPERVISED_AUGMENTATION_STRENGTHS, SUPERVISED_AUGMENTATION_STRENGTHS
from vision_training.data.label_mapping import write_run_label_dictionary
from vision_training.data.synthetic.create_synthetic_dataset import (KEYPOINT_CATEGORIES, build_keypoint_label_mapping,
                                                                     build_keypoint_labels)
from vision_training.utilities.os_utilities import color_text

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


def remove_level_anchors(configuration: Dict[str, Any]) -> None:
    """
    Removes the anchors of the seventh detection level, whose region proposal head then has no outputs.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    del configuration["DetectionLevels"]["7"]["anchors"]


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


def set_single_input_channel(configuration: Dict[str, Any]) -> None:
    """
    Builds the backbone on single-channel images, while the loaders give BGR images.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["Backbone"]["input_channels"] = 1


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


def raise_background_minimum(configuration: Dict[str, Any]) -> None:
    """
    Starts the background IoU range of the region proposals above 0.0, keeping only the hard negatives.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["RegionProposal"]["background_iou_threshold"]["min"] = 0.1


def test_background_minimum_above_zero_only_warns(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a background IoU range starting above 0.0 loads with a yellow warning instead of an error, the hard
    negatives being a choice, and that the shipped value 0.0 prints no warning.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    # The shipped configuration samples every anchor overlapping no object as background, silently
    _, model_configuration = load_experiment_configuration(configuration_path=EXAMPLE_CONFIGURATION_PATH)
    assert model_configuration["RegionProposal"]["background_iou_threshold"]["min"] == 0.0
    assert "background_iou_threshold.min" not in capsys.readouterr().out

    # A minimum above 0.0 is accepted, and announced in yellow
    configuration_path = write_modified_configuration(temporary_directory=tmp_path,
                                                      modify_function=raise_background_minimum)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["RegionProposal"]["background_iou_threshold"]["min"] == 0.1
    printed_output = capsys.readouterr().out
    expected_warning = ("Warning: 'RegionProposal.background_iou_threshold.min' is 0.1, above 0.0: the anchors whose "
                        "best IoU is below it are ignored, only the hard negatives being sampled as background.")
    assert color_text(color="yellow", text=expected_warning) in printed_output


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


def add_mask_loss_weight(configuration: Dict[str, Any]) -> None:
    """
    Gives a weight to the mask loss, which only the instance segmenter has, in an object detection configuration.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["Optimization"]["loss_weights"]["mask_loss"] = 1.0


def set_invalid_loss_weights(configuration: Dict[str, Any]) -> None:
    """
    Sets a negative weight of the classification loss of the region proposals and writes the weight of the box loss of
    the detection heads as text.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["Optimization"]["loss_weights"].update({"region_proposal_classification_loss": -1.0,
                                                          "detection_regression_loss": "high"})


def zero_every_loss_weight(configuration: Dict[str, Any]) -> None:
    """
    Sets the weight of every loss of the detector to 0.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    loss_weights = configuration["Optimization"]["loss_weights"]
    loss_weights.update({loss_name: 0 for loss_name in loss_weights})


def set_empty_black_out_grid(configuration: Dict[str, Any]) -> None:
    """
    Sets a grid of 0 rows and columns for the random_black_out type of the data augmentation.

    Args:
        configuration (Dict[str, Any]): The raw configuration, modified in place.
    """
    configuration["Data"]["data_augmentation"]["types"]["random_black_out"]["grid_size"] = 0


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
    (set_invalid_loading, ["'Data.loading.prefetch_batches' must be an integer >= 1, got 0."]),
    (set_empty_black_out_grid, ["'Data.data_augmentation.types.random_black_out.grid_size' must be an integer within "
                                "[1, 'Data.image_settings.size' (1024)], got 0: a grid finer than the image would "
                                "hide cells holding no pixel."]),
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
    (remove_level_anchors, ["'DetectionLevels.7' must hold exactly ['anchors', 'region_proposal', 'detector'], "
                            "got ['detector', 'region_proposal']."]),
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
    (set_single_input_channel, ["'Backbone.input_channels' must be 3, the loaders giving 3-channel BGR images and "
                                "the exports taking 3-channel inputs, got 1."]),
    (set_string_flag, ["'Backbone.modules.inception' must be True or False"]),
    (set_detector_samples, ["'Detector.total_training_samples' (1000) must be <="]),
    (set_inception_channels, ["'Backbone.convolutions.4' has 20 output channels"]),
    (set_intermediate_weight_saving, ["'ExperimentConfiguration.intermediate_weight_saving_iterations' (10000) "
                                      "must be smaller than"]),
    (set_unknown_task, ["'ExperimentConfiguration.task' must be one of ['object_detection', 'instance_segmentation', "
                        "'classification', 'semantic_segmentation', 'sequence_classification', 'sequence_labelling', "
                        "'sequence_comparison', 'frame_selection', 'self_supervised', 'keypoint_detection'], got "
                        "'segmentation'."]),
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
                            "got 12."]),
    (add_mask_loss_weight, ["'Optimization.loss_weights' must give the weight of exactly the losses "
                            "['region_proposal_classification_loss', 'region_proposal_localisation_loss', "
                            "'detection_classification_loss', 'detection_regression_loss'], got"]),
    (set_invalid_loss_weights, ["'Optimization.loss_weights.region_proposal_classification_loss' must be a number "
                                ">= 0, got -1.0.",
                                "'Optimization.loss_weights.detection_regression_loss' must be a number >= 0, got "
                                "'high'."]),
    (zero_every_loss_weight, ["'Optimization.loss_weights' cannot all be 0, the model would have no loss."]),
    (activate_image_only_augmentation_types, IMAGE_ONLY_TYPE_REFUSALS)])
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


@pytest.mark.parametrize(argnames=("run_name_suffix", "accepted"),
                         argvalues=[(None, True), ("v1", True), ("batch-384.v2", True), ("", False),
                                    ("v1_batch-384", False), ("runs/v1", False), ("v 1", False), (12, False),
                                    (True, False)])
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
                         argvalues=[(0, None), (2500, None),
                                    (-1, "'ExperimentConfiguration.intermediate_weight_saving_iterations' must be an "
                                         "integer >= 0, got -1."),
                                    (5000, "'ExperimentConfiguration.intermediate_weight_saving_iterations' (5000) "
                                           "must be smaller than 'ExperimentConfiguration.weight_saving_iterations' "
                                           "(5000).")])
def test_intermediate_weight_saving_iterations_is_checked(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                          intermediate_weight_saving_iterations: int,
                                                          expected_message: Optional[str]) -> None:
    """
    Verifies that the intermediate weight saving is 0, which saves no checkpoint between the evaluated ones, or a
    positive integer smaller than weight_saving_iterations (5000 in the example configuration), a negative value or one
    not more frequent than the evaluated checkpoints being refused with its key.

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


def test_configuration_without_run_name_suffix_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a configuration written before the run name suffix, without the key, is refused with its path
    instead of being named as if the suffix were null.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    configuration = yaml.safe_load(stream=EXAMPLE_CONFIGURATION_PATH.read_text())
    del configuration["ExperimentConfiguration"]["run_name_suffix"]
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(yaml.safe_dump(configuration))

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)
    assert "Missing configuration key 'ExperimentConfiguration.run_name_suffix'." in capsys.readouterr().out


def empty_detection_levels(configuration: Dict[str, Any]) -> None:
    """
    Removes every detection level.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["DetectionLevels"] = {}


def unquote_detection_level(configuration: Dict[str, Any]) -> None:
    """
    Writes the key of the seventh detection level as a number instead of a quoted block number.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["DetectionLevels"][7] = configuration["DetectionLevels"].pop("7")


def add_level_beyond_the_backbone(configuration: Dict[str, Any]) -> None:
    """
    Adds an eighth detection level, while the backbone of the example has seven blocks.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["DetectionLevels"]["8"] = copy.deepcopy(configuration["DetectionLevels"]["7"])


def reverse_detection_levels(configuration: Dict[str, Any]) -> None:
    """
    Lists the detection levels from the coarsest to the finest one.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["DetectionLevels"] = dict(reversed(list(configuration["DetectionLevels"].items())))


def add_mask_head_to_detection_level(configuration: Dict[str, Any]) -> None:
    """
    Gives the fifth detection level of the object detection a mask head, which only the instance segmentation has.

    Args:
        configuration (Dict[str, Any]): The raw example configuration, modified in place.
    """
    configuration["DetectionLevels"]["5"]["mask_head"] = {"roi_align_pool_size": 14, "convolutions": [4, 64]}


@pytest.mark.parametrize(argnames=("modify_function", "expected_errors"), argvalues=[
    (empty_detection_levels, ["'DetectionLevels' must map at least one backbone block number to the settings of its "
                              "level, got {}."]),
    (unquote_detection_level, ["'DetectionLevels' keys must be quoted backbone block numbers between \"1\" and "
                               "\"7\", got ['5', '6', 7]."]),
    (add_level_beyond_the_backbone, ["'DetectionLevels' keys must be quoted backbone block numbers between \"1\" "
                                     "and \"7\", got ['5', '6', '7', '8']."]),
    (reverse_detection_levels, ["'DetectionLevels' keys must be in increasing order, got ['7', '6', '5']."]),
    (add_mask_head_to_detection_level, ["'DetectionLevels.5' must hold exactly ['anchors', 'region_proposal', "
                                        "'detector'], got ['anchors', 'region_proposal', 'detector', 'mask_head']."])])
def test_inconsistent_detection_levels_are_rejected(modify_function: Callable[[Dict[str, Any]], None],
                                                    expected_errors: List[str]) -> None:
    """
    Verifies the keys and components of the detection levels on the example configuration itself, since writing it
    back to a YAML file would sort its keys and turn its unquoted keys back into numbers of their own: the levels must
    be the quoted numbers of existing backbone blocks, in increasing order, each holding the anchors and the heads of
    the task only.

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


def set_invalid_decorrelation(configuration: Dict[str, Any]) -> None:
    """
    Writes the decorrelation flag as a quoted string, with a weight of 0 and a hidden layer 0.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"]["decorrelation"] = {"activated": "yes", "weight": 0, "hidden_layer": 0}


def decorrelate_a_missing_hidden_layer(configuration: Dict[str, Any]) -> None:
    """
    Turns the decorrelation on for a second hidden layer in a head of a single hidden layer.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"]["fully_connected"] = [128]
    configuration["ClassificationHead"]["decorrelation"] = {"activated": True, "weight": 0.01, "hidden_layer": 2}


def normalize_the_head_of_single_image_batches(configuration: Dict[str, Any]) -> None:
    """
    Normalizes the hidden layer of the head with the batch normalization, with batches of a single image.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["batch_size"] = 1
    configuration["ClassificationHead"].update({"fully_connected": [64], "fully_connected_normalization": "batch"})


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


def decorrelate_the_last_hidden_layer(configuration: Dict[str, Any]) -> None:
    """
    Turns the decorrelation on for the last of two hidden layers.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"]["fully_connected"] = [128, 64]
    configuration["ClassificationHead"]["decorrelation"] = {"activated": True, "weight": 0.01, "hidden_layer": 2}


def test_decorrelation_of_a_hidden_layer_is_accepted(tmp_path: Path) -> None:
    """
    Verifies that a classification configuration decorrelating the last hidden layer of its head passes the checks.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="classification",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=decorrelate_the_last_hidden_layer)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["ClassificationHead"]["decorrelation"] == {"activated": True, "weight": 0.01,
                                                                          "hidden_layer": 2}


@pytest.mark.parametrize(argnames=("task_folder", "configuration_name"), argvalues=[
    ("classification", "configuration_synthetic_shapes.yaml"),
    ("sequence_classification", "configuration_synthetic_sequences.yaml"),
    ("self_supervised", "configuration_synthetic_shapes.yaml")])
def test_image_only_augmentation_types_are_accepted_without_targets(tmp_path: Path, task_folder: str,
                                                                    configuration_name: str) -> None:
    """
    Verifies that the tasks whose samples hold no box, mask or keypoint, here the image and sequence classifications
    and the self-supervised task, accept the perspective and the jigsaw, the augmentation types moving the image only.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task_folder (str): The folder of the configurations of the task.
        configuration_name (str): The file name of the configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                  configuration_name=configuration_name,
                                                  modify_function=activate_image_only_augmentation_types)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    augmentation_types = model_configuration["Data"]["data_augmentation"]["types"]
    assert augmentation_types["random_perspective"]["activated"] and augmentation_types["random_jigsaw"]["activated"]


@pytest.mark.parametrize(argnames=("jigsaw_grid_size", "image_size", "jigsaw_activated", "refused_by_jigsaw", "loads"),
                         argvalues=[(2, 1, True, True, False), (2, 1, False, False, False), (2, 256, True, False, True),
                                    (3, 256, True, True, False), (3, 256, False, False, True)])
def test_jigsaw_image_size_is_checked(tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch,
                                      jigsaw_grid_size: int, image_size: int, jigsaw_activated: bool,
                                      refused_by_jigsaw: bool, loads: bool) -> None:
    """
    Verifies that an activated random_jigsaw refuses a side of the network input that is not a multiple of its grid
    size, which would stop the training at its first batch, and that the jigsaw turned off leaves the size to the
    other checks: the image of 1 pixel passes the power of two check but not the 2x2 grid, and a grid of 3x3 pieces in
    both strength sets stands for a change of the hard-coded grid size.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture setting the grid size of both strength sets.
        jigsaw_grid_size (int): The grid size of random_jigsaw in both strength sets.
        image_size (int): The side of the network input, 'Data.image_settings.size'.
        jigsaw_activated (bool): Whether random_jigsaw is activated.
        refused_by_jigsaw (bool): Whether the jigsaw check must refuse the image size.
        loads (bool): Whether the configuration must load, an image of 1 pixel being refused by the backbone checks too.
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


def keep_the_removed_view_losses(configuration: Dict[str, Any]) -> None:
    """
    Keeps the section of the view losses removed from the image classifier, next to an unknown pooling of its head.

    Args:
        configuration (Dict[str, Any]): The raw classification configuration, modified in place.
    """
    configuration["ClassificationHead"]["view_losses"] = {"global_pooling": "average"}
    configuration["ClassificationHead"]["global_pooling"] = "median"


def keep_the_removed_view_augmentation(configuration: Dict[str, Any]) -> None:
    """
    Keeps the section of the views of the self-supervised task, removed for the types of the data augmentation, next
    to an unknown method of the projection head.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["Data"]["view_augmentation"] = {"rotation": {"probability": 0.5, "maximum_degrees": 180.0}}
    configuration["ProjectionHead"]["method"] = "byol"


@pytest.mark.parametrize(argnames=("task_folder", "modify_function", "removed_section", "hidden_message"), argvalues=[
    ("classification", keep_the_removed_view_losses, "ClassificationHead.view_losses",
     "'ClassificationHead.global_pooling' must be one of"),
    ("self_supervised", keep_the_removed_view_augmentation, "Data.view_augmentation",
     "'ProjectionHead.method' must be one of")])
def test_removed_sections_are_refused_alone(tmp_path: Path, capsys: pytest.CaptureFixture, task_folder: str,
                                            modify_function: Callable[[Dict[str, Any]], None], removed_section: str,
                                            hidden_message: str) -> None:
    """
    Verifies that a configuration still holding a section removed from the repository (the view losses of the image
    classifier, the views of the self-supervised task now drawn by the data augmentation) is refused with a message
    naming the section, what replaced it and that it must be deleted, and alone: the other checks do not run, so that
    the reason of the refusal is never hidden behind the keys the removed section leaves missing.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
        task_folder (str): The folder of the configurations of the task.
        modify_function (Callable[[Dict[str, Any]], None]): Function keeping the removed section.
        removed_section (str): The key path of the removed section.
        hidden_message (str): A message of another check, which must not be printed.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=modify_function)

    with pytest.raises(expected_exception=SystemExit):
        load_experiment_configuration(configuration_path=configuration_path)

    printed_output = capsys.readouterr().out
    assert (f"'{removed_section}' was removed: {REMOVED_SECTIONS[removed_section]}. Delete it from the configuration "
            f"file.") in printed_output
    assert hidden_message not in printed_output and "Traceback" not in printed_output


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_negative_number_workers, ["'Data.loading.number_workers' must be an integer >= 0, got -1."]),
    (use_binary_labels, ["'Data.label_mapping.mode' must be 'multi_class' for the classification tasks"]),
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
                                         "integer >= 0, got -1."]),
    (set_invalid_decorrelation, ["'ClassificationHead.decorrelation.activated' must be True or False, got 'yes'.",
                                 "'ClassificationHead.decorrelation.weight' must be a number > 0, got 0.",
                                 "'ClassificationHead.decorrelation.hidden_layer' must be an integer >= 1, got 0."]),
    (decorrelate_a_missing_hidden_layer, ["'ClassificationHead.decorrelation.hidden_layer' must be at most the number "
                                          "of hidden layers of 'ClassificationHead.fully_connected' (1) when "
                                          "'ClassificationHead.decorrelation.activated' is True, got 2."]),
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


@pytest.mark.parametrize(argnames=("task_folder", "configuration_name"),
                         argvalues=[("sequence_classification", "configuration_synthetic_sequences.yaml"),
                                    ("sequence_labelling", "configuration_synthetic_sequences.yaml"),
                                    ("sequence_comparison", "configuration_synthetic_pairs.yaml"),
                                    ("frame_selection", "configuration_synthetic_sequences.yaml")])
def test_flattened_feature_map_is_accepted_for_the_sequence_tasks(tmp_path: Path, task_folder: str,
                                                                  configuration_name: str) -> None:
    """
    Verifies that the sequence tasks accept to flatten the final feature map of every frame, their temporal module
    and fusion then reading frame vectors of one value per position and channel of the last backbone block.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task_folder (str): The folder of the configurations of the sequence task.
        configuration_name (str): The file name of the synthetic configuration of the task.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder=task_folder,
                                                  configuration_name=configuration_name,
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


def set_causal_temporal_model(configuration: Dict[str, Any]) -> None:
    """
    Makes the bidirectional temporal module of the synthetic sequences configuration causal.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    configuration["TemporalModel"]["causal"] = True


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
                                            "'Data.sequence_settings.maximum_sequence_length' must be an integer >= "
                                            "1",
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
    Verifies that the frame selection configurations shipped with the repository pass every consistency check, their
    sequence settings holding the maximum sequence length only.

    Args:
        tmp_path (Path): The pytest temporary directory.
        configuration_name (str): The file name of the frame selection configuration.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="frame_selection",
                                                  configuration_name=configuration_name, modify_function=None)
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
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


def remove_maximum_sequence_length(configuration: Dict[str, Any]) -> None:
    """
    Removes the maximum sequence length from the sequence settings.

    Args:
        configuration (Dict[str, Any]): The raw sequence configuration, modified in place.
    """
    del configuration["Data"]["sequence_settings"]["maximum_sequence_length"]


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml",
                                                                   "configuration_synthetic_pairs.yaml"])
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
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    assert experiment_configuration["task"] == "sequence_comparison"
    assert sorted(model_configuration["Data"]["sequence_settings"]) == ["maximum_sequence_length"]


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


@pytest.mark.parametrize(argnames=("task_folder", "configuration_name"),
                         argvalues=[("sequence_classification", "configuration_synthetic_sequences.yaml"),
                                    ("sequence_labelling", "configuration_synthetic_sequences.yaml"),
                                    ("sequence_comparison", "configuration_synthetic_pairs.yaml"),
                                    ("frame_selection", "configuration_synthetic_sequences.yaml")])
def test_sequence_tasks_require_the_maximum_sequence_length(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                            task_folder: str, configuration_name: str) -> None:
    """
    Verifies that every sequence task requires the maximum sequence length, which leaves the longer sequences out of
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
                                      "got 'yes'."]),
    (activate_image_only_augmentation_types, IMAGE_ONLY_TYPE_REFUSALS)])
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
    ({"1": 1, "2": 1, "3": 2}, None, None),
    ({"1": 1, "3": 3}, "indexed 0 (background) to K-1 without gaps, got [0, 1, 3].", None),
    ({"1": 0, "2": 1, "3": 2}, None, "Warning: the categories [1] of the label mapping are mapped to the background "
                                     "class 0: their objects are left out of the training."),
    ({"background": 1, "1": 0, "2": 2}, "The 'background' of the label mapping must be the class 0, got 1.", None)])
def test_detection_label_mapping_indexes_its_classes_from_the_background(
        tmp_path: Path, capsys: pytest.CaptureFixture, label_mapping: Dict[str, int], expected_error: str | None,
        expected_warning: str | None) -> None:
    """
    Verifies the label mapping of the object detection: merged categories give one class each, counted once, classes
    with gaps (original COCO identifiers kept) and a background key on another class than 0 are refused, and a
    category mapped to 0, left out of the training, is accepted with a warning.

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
    assert model_configuration["DetectionLevels"]["5"]["mask_head"]["roi_align_pool_size"] == 14


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


def remove_level_mask_head(configuration: Dict[str, Any]) -> None:
    """
    Removes the mask head of the seventh detection level, whose boxes would then have no mask.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    del configuration["DetectionLevels"]["7"]["mask_head"]


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
    configuration["DetectionLevels"]["5"]["mask_head"] = {"roi_align_pool_size": 0, "convolutions": [4]}
    configuration["DetectionLevels"]["6"]["mask_head"]["convolutions"] = [4, 60]
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
    configuration["DetectionLevels"]["7"]["mask_head"]["roi_align_pool_size"] = 32


def write_original_resolution_as_number(configuration: Dict[str, Any]) -> None:
    """
    Writes the evaluation at the original resolution of the instance segmentation as a number instead of a boolean.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"]["original_resolution"] = 1


def freeze_random_mask_heads(configuration: Dict[str, Any]) -> None:
    """
    Freezes the mask heads without loading them from another run, and multiplies their learning rate by zero.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    configuration["MaskHead"]["pretrained"].update({"freeze": True, "learning_rate_multiplier": 0})


def remove_mask_loss_weight(configuration: Dict[str, Any]) -> None:
    """
    Leaves the mask loss of the instance segmenter without weight.

    Args:
        configuration (Dict[str, Any]): The raw instance segmentation configuration, modified in place.
    """
    del configuration["Optimization"]["loss_weights"]["mask_loss"]


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (freeze_random_mask_heads, ["'MaskHead.pretrained.freeze' must be False when "
                                "'MaskHead.pretrained.experiment_folder' is null: only a transferred mask heads can "
                                "be frozen.",
                                "'MaskHead.pretrained.learning_rate_multiplier' must be > 0, got 0."]),
    (set_pool_size_above_the_mask_resolution, ["'DetectionLevels.7.mask_head.roi_align_pool_size' (32) must not "
                                               "exceed 'MaskHead.mask_resolution' (28), the pooled features being "
                                               "only enlarged to the masks."]),
    (remove_level_mask_head, ["'DetectionLevels.7' must hold exactly ['anchors', 'region_proposal', 'detector', "
                              "'mask_head'], got ['anchors', 'detector', 'region_proposal']."]),
    (set_invalid_mask_head, ["'MaskHead.mask_resolution' must be an integer >= 1, got 0.",
                             "'MaskHead.upsampling' must be one of ['transposed', 'bilinear'], got 'nearest'.",
                             "'DetectionLevels.5.mask_head.roi_align_pool_size' must be an integer >= 1, got 0.",
                             "'DetectionLevels.5.mask_head.convolutions' must be [number_layers, output_channels] "
                             "with positive integers, got [4].",
                             "'DetectionLevels.6.mask_head.convolutions' has 60 output channels, inception "
                             "modules require a multiple of 8.",
                             "'MaskHead.normalization.feature_map_normalization' must be one of",
                             "'MaskHead.normalization.batch_normalization_momentum' must be a number within (0, 1], "
                             "got -0.1.",
                             "'MaskHead.modules.residual_nets' must be True or False, got 'yes'.",
                             "'MaskHead.inference.mask_threshold' must be a probability within [0, 1], got 1.5."]),
    (write_original_resolution_as_number, ["'ExperimentConfiguration.evaluation.original_resolution' must be True or "
                                           "False, got 1."]),
    (remove_mask_loss_weight, ["'Optimization.loss_weights' must give the weight of exactly the losses "
                               "['region_proposal_classification_loss', 'region_proposal_localisation_loss', "
                               "'detection_classification_loss', 'detection_regression_loss', 'mask_loss'], got"]),
    (activate_image_only_augmentation_types, IMAGE_ONLY_TYPE_REFUSALS)])
def test_inconsistent_instance_segmentation_configuration_is_rejected(
        tmp_path: Path, capsys: pytest.CaptureFixture, modify_function: Callable[[Dict[str, Any]], None],
        expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the mask head section and of the evaluation at the original resolution stops
    the loading with a short message naming the faulty key, the mask head of every detection level included.

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


@pytest.mark.parametrize(argnames="configuration_name", argvalues=["configuration_example.yaml",
                                                                   "configuration_synthetic_shapes.yaml"])
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
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    assert experiment_configuration["task"] == "self_supervised"
    assert model_configuration["Data"]["label_dictionary"] == {1: 0, 2: 1, 3: 2, 4: 3}


def keep_the_whole_images_in_the_views(configuration: Dict[str, Any]) -> None:
    """
    Turns off the random resized crop of the views, random_crop, its probability below 1 being then left aside.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["Data"]["data_augmentation"]["types"]["random_crop"] = {"activated": False, "probability": 0.5}


def test_views_without_crop_are_accepted(tmp_path: Path) -> None:
    """
    Verifies that the self-supervised task accepts views drawn without the random resized crop, keeping the whole
    images, the probability of a crop turned off being left aside.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_task_configuration(temporary_directory=tmp_path, task_folder="self_supervised",
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=keep_the_whole_images_in_the_views)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    assert model_configuration["Data"]["data_augmentation"]["types"]["random_crop"] == {"activated": False,
                                                                                        "probability": 0.5}


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


def normalize_projection_without_hidden_layers(configuration: Dict[str, Any]) -> None:
    """
    Normalizes the hidden layers of a projection head holding the embedding layer only.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["ProjectionHead"].update({"fully_connected": [128], "fully_connected_normalization": "batch"})


def deactivate_the_augmentation_of_the_views(configuration: Dict[str, Any]) -> None:
    """
    Turns off the data augmentation, whose types draw the two views of the self-supervised task.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["Data"]["data_augmentation"]["activated"] = False


def crop_half_of_the_views(configuration: Dict[str, Any]) -> None:
    """
    Draws the random resized crop of the views, random_crop, for half of them only.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["Data"]["data_augmentation"]["types"]["random_crop"]["probability"] = 0.5


def set_invalid_grouped_batches(configuration: Dict[str, Any]) -> None:
    """
    Turns the grouped batches on with a block key that is not a string, an empty batch key and blocks of 3 images,
    which do not divide the batch of 128 images.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["Data"]["grouped_batches"].update({"activated": True, "block_key": 3, "batch_key": "",
                                                     "images_per_block": 3})


def set_single_image_grouped_blocks(configuration: Dict[str, Any]) -> None:
    """
    Sets a quoted flag for the grouped batches, read as a string, and blocks of a single image.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["Data"]["grouped_batches"].update({"activated": "False", "images_per_block": 1})


def set_invalid_self_supervised_evaluation(configuration: Dict[str, Any]) -> None:
    """
    Sets no neighbour for the k-nearest neighbour evaluation and batches of a single image, whose views the losses
    cannot compare with those of other images.

    Args:
        configuration (Dict[str, Any]): The raw self-supervised configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["evaluation"]["number_neighbours"] = 0
    configuration["ExperimentConfiguration"]["batch_size"] = 1


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_invalid_projection_head, ["'ProjectionHead.global_pooling' must be one of ['average', 'max', 'flatten'], "
                                   "got 'median'.",
                                   "'ProjectionHead.fully_connected' must be a non-empty list of positive integers, "
                                   "the last one being the embedding size, got [].",
                                   "'ProjectionHead.method' must be one of ['simclr', 'barlow_twins'], got 'byol'.",
                                   "'ProjectionHead.simclr.temperature' must be a number > 0, got 0.",
                                   "'ProjectionHead.barlow_twins.off_diagonal_weight' must be a number > 0, got "
                                   "True."]),
    (normalize_projection_without_hidden_layers, ["'ProjectionHead.fully_connected_normalization' must be 'none' when "
                                                  "'ProjectionHead.fully_connected' holds the embedding layer only: "
                                                  "the head has no hidden layer to normalize."]),
    (deactivate_the_augmentation_of_the_views, ["'Data.data_augmentation.activated' must be True for the "
                                                "self-supervised task: its types draw the two views of every image, "
                                                "which would otherwise both be the image itself."]),
    (crop_half_of_the_views, ["'Data.data_augmentation.types.random_crop.probability' must be 1.0 for the "
                              "self-supervised task when 'Data.data_augmentation.types.random_crop.activated' is True, "
                              "got 0.5: the random resized crop is drawn for every view, or turned off to keep the "
                              "whole images."]),
    (set_invalid_grouped_batches, ["'Data.grouped_batches.block_key' must be a non-empty string, got 3.",
                                   "'Data.grouped_batches.batch_key' must be null or a non-empty string, got ''.",
                                   "'Data.grouped_batches.images_per_block' (3) must divide "
                                   "'ExperimentConfiguration.batch_size' (128) when the grouped batches are on: a "
                                   "batch holds a whole number of blocks."]),
    (set_single_image_grouped_blocks, ["'Data.grouped_batches.activated' must be True or False, got 'False'.",
                                       "'Data.grouped_batches.images_per_block' must be an integer >= 2, got 1."]),
    (set_invalid_self_supervised_evaluation, ["'ExperimentConfiguration.evaluation.number_neighbours' must be an "
                                              "integer >= 1, got 0.",
                                              "'ExperimentConfiguration.batch_size' must be at least 2 for the "
                                              "self-supervised task: the losses compare the views of the images of a "
                                              "batch with each other."])])
def test_inconsistent_self_supervised_configuration_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                                modify_function: Callable[[Dict[str, Any]], None],
                                                                expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the self-supervised sections (projection head, data augmentation drawing the
    views, grouped batches, evaluation and batch size) stops the loading with a short message naming the faulty key.

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
    synthetic_path = write_keypoint_configuration(temporary_directory=tmp_path,
                                                  configuration_name="configuration_synthetic_shapes.yaml",
                                                  modify_function=None)
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


def set_invalid_keypoint_head(configuration: Dict[str, Any]) -> None:
    """
    Sets a higher resolution branch flag and an ignored regions flag written as text, a negative hidden width, a
    negative tag pull weight, no keypoint candidate, a keypoint threshold of 1, a zero tag threshold and an object
    score threshold above 1.

    Args:
        configuration (Dict[str, Any]): The raw keypoint detection configuration, modified in place.
    """
    configuration["KeypointHead"].update({"higher_resolution": "yes", "ignore_unlabelled_objects": "no",
                                          "hidden_width": -1, "tag_pull_weight": -0.1})
    configuration["KeypointHead"]["inference"].update({"maximum_candidates": 0, "keypoint_threshold": 1.0,
                                                       "tag_threshold": 0, "object_score_threshold": 1.5})


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


def end_stem_at_an_eighth(configuration: Dict[str, Any]) -> None:
    """
    Adds the last max pooling to the 3 stem blocks, so that the stem ends at 1/8 of the image instead of 1/4.

    Args:
        configuration (Dict[str, Any]): The raw keypoint detection configuration, modified in place.
    """
    configuration["Backbone"]["last_max_pooling"] = True


def set_invalid_high_resolution_network(configuration: Dict[str, Any]) -> None:
    """
    Sets a zero base width, a stage without exchange module, a module flag written as text, an unknown normalization
    and a zero heatmap sigma.

    Args:
        configuration (Dict[str, Any]): The raw keypoint detection configuration, modified in place.
    """
    configuration["HRNet"].update({"base_width": 0, "modules_per_stage": [1, 0]})
    configuration["HRNet"]["modules"]["residual_nets"] = "yes"
    configuration["HRNet"]["normalization"]["feature_map_normalization"] = "group"
    configuration["KeypointHead"]["heatmap_sigma"] = 0


def use_binary_keypoint_labels(configuration: Dict[str, Any]) -> None:
    """
    Uses the binary label mapping, which would merge categories with different keypoints into one class.

    Args:
        configuration (Dict[str, Any]): The raw keypoint detection configuration, modified in place.
    """
    configuration["Data"]["label_mapping"]["mode"] = "binary"


def remove_training_labels(configuration: Dict[str, Any]) -> None:
    """
    Points the training split to a folder without 'labels.json', so that no keypoint definition can be injected.

    Args:
        configuration (Dict[str, Any]): The raw keypoint detection configuration, modified in place.
    """
    configuration["ExperimentConfiguration"]["train_data_folder"] += "_missing"


@pytest.mark.parametrize(argnames=("modify_function", "expected_messages"), argvalues=[
    (set_image_size_not_matching_the_branches, ["'Data.image_settings.size' (500) must be a multiple of 32, the "
                                                "stride of the lowest resolution branch of the 3 stages of "
                                                "'HRNet.modules_per_stage'."]),
    (set_invalid_keypoint_head, ["'KeypointHead.higher_resolution' must be True or False, got 'yes'.",
                                 "'KeypointHead.ignore_unlabelled_objects' must be True or False, got 'no'.",
                                 "'KeypointHead.hidden_width' must be an integer >= 0, got -1.",
                                 "'KeypointHead.tag_pull_weight' must be a number >= 0, got -0.1.",
                                 "'KeypointHead.inference.maximum_candidates' must be an integer >= 1, got 0.",
                                 "'KeypointHead.inference.keypoint_threshold' must be a number within ]0, 1[, got "
                                 "1.0.",
                                 "'KeypointHead.inference.tag_threshold' must be a positive number, got 0.",
                                 "'KeypointHead.inference.object_score_threshold' must be a number within [0, 1], "
                                 "got 1.5."]),
    (set_unknown_keypoint_sigmas, ["'ExperimentConfiguration.evaluation.default_keypoint_sigma' must be a positive "
                                   "number, got 0.",
                                   "'ExperimentConfiguration.evaluation.keypoint_sigmas.dragon' names a category that "
                                   "is not learnt from the training split",
                                   "'ExperimentConfiguration.evaluation.keypoint_sigmas.stick_figure' names unknown "
                                   "keypoints ['left_wing'], the keypoints of the category being ['head', 'neck', "
                                   "'left_elbow', 'right_elbow', 'left_hand', 'right_hand', 'pelvis', 'left_knee', "
                                   "'right_knee', 'left_foot', 'right_foot']."]),
    (end_stem_at_an_eighth, ["The Backbone stem of the keypoint detector must end at 1/4 of the image, got 1/8 from "
                             "its 3 blocks with 'Backbone.last_max_pooling' True: use 3 blocks without the last "
                             "max pooling, or 2 blocks with it."]),
    (set_invalid_high_resolution_network, ["'HRNet.base_width' must be an integer >= 1, got 0.",
                                           "'HRNet.modules_per_stage' must be a non-empty list of positive integers, "
                                           "the exchange modules of every stage, got [1, 0].",
                                           "'HRNet.modules.residual_nets' must be True or False, got 'yes'.",
                                           "'HRNet.normalization.feature_map_normalization' must be one of ['batch', "
                                           "'layer', 'none'], got 'group'.",
                                           "'KeypointHead.heatmap_sigma' must be a positive number of heatmap cells, "
                                           "got 0."]),
    (use_binary_keypoint_labels, ["'Data.label_mapping.mode' must be 'multi_class' for the keypoint detection, every "
                                  "class having its own keypoints, got 'binary'."]),
    (remove_training_labels, ["The keypoint definitions of the classes could not be read: the 'labels.json' of "
                              "'ExperimentConfiguration.train_data_folder' must exist"]),
    (activate_image_only_augmentation_types, IMAGE_ONLY_TYPE_REFUSALS)])
def test_inconsistent_keypoint_configuration_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                         modify_function: Callable[[Dict[str, Any]], None],
                                                         expected_messages: List[str]) -> None:
    """
    Verifies that every inconsistency of the keypoint detection sections (image size against the branches of the high
    resolution network, keypoint head and its grouping settings, keypoint sigmas of the evaluation, stem, high
    resolution network and heatmap sigma, label mode and training labels) stops the loading with a short message
    naming the faulty key.

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


def activate_flips_and_quarter_turns(configuration: Dict[str, Any]) -> None:
    """
    Turns on both flips and the 90 degree rotations of the DataAugmenter, which move the keypoints without renaming
    them, a choice left to the user (the tip of a flame keeps its name when flipped).

    Args:
        configuration (Dict[str, Any]): The raw keypoint detection configuration, modified in place.
    """
    for augmentation_name in ["random_left_right_flip", "random_up_down_flip", "random_90_rotation"]:
        configuration["Data"]["data_augmentation"]["types"][augmentation_name]["activated"] = True


def test_keypoint_geometric_augmentations_are_accepted(tmp_path: Path) -> None:
    """
    Verifies that the keypoint detection accepts every geometric augmentation, flips and 90 degree rotations included,
    the keypoints being moved with the image and keeping their names, so that no augmentation is forbidden.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration_path = write_keypoint_configuration(temporary_directory=tmp_path,
                                                      configuration_name="configuration_synthetic_shapes.yaml",
                                                      modify_function=activate_flips_and_quarter_turns)
    _, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    augmentation_types = model_configuration["Data"]["data_augmentation"]["types"]
    assert all(augmentation_types[augmentation_name]["activated"]
               for augmentation_name in ["random_left_right_flip", "random_up_down_flip", "random_90_rotation"])


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
