"""
Tests of build_run_name of vision_training/configuration_processing/run_name.py on the reference configurations of every
task: the reference run names are fixed, changed settings (backbone modules, detection levels, classes, fusions, tiling,
mask heads, transfers, sequence normalization, optimizer, weight decay) change the run name with their token, the run
name suffix ends the name, and the training settings that can change between two resumptions (augmentation, batch
size, learning rate, schedule, moving average of the weights, dropout rates, grouped batches) keep it.
"""

import copy
import pytest
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from vision_training.configuration_processing.run_name import build_run_name
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")

# Run name of the reference settings set by get_reference_configurations
REFERENCE_RUN_NAME = ("DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512.598295_"
                      "RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_BIN")

CLASSIFICATION_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                     "classification" / "configuration_synthetic_shapes.yaml")

# Run name of the classification reference settings set by get_classification_reference_configurations
CLASSIFICATION_REFERENCE_RUN_NAME = "CLS_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC4"

# Pretrained settings of a temporal module trained from random weights, and of a classification head, whose class
# layer flag is False without source run
NO_TRANSFER_SETTINGS = {"experiment_folder": None, "iteration": None, "freeze": False, "learning_rate_multiplier": 1.0}
NO_HEAD_TRANSFER_SETTINGS = {**NO_TRANSFER_SETTINGS, "class_layer": False}


def get_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads the example configuration and sets every setting used by the run name explicitly, so that the expected
    names do not change when the example configuration is edited.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=EXAMPLE_CONFIGURATION_PATH)

    # Task of the run, giving the first section of the name
    experiment_configuration["task"] = "object_detection"

    # No run name suffix, which would end the name
    experiment_configuration["run_name_suffix"] = None

    # Input normalization, resizing mode and size
    model_configuration["Backbone"]["normalization"].update({"input_normalization": True,
                                                             "per_sample_normalization": False,
                                                             "feature_map_normalization": "batch"})
    model_configuration["Data"]["image_settings"].update({"size": 1024, "keep_ratio": True})

    # Backbone modules and blocks
    model_configuration["Backbone"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False})
    model_configuration["Backbone"]["convolutions"] = {"1": [1, 16], "2": [2, 32], "3": [2, 32], "4": [4, 64],
                                                       "5": [4, 64], "6": [6, 128], "7": [6, 128]}

    # Detection levels 5 to 7, whose anchors, region proposal heads and detector heads are identical on every level
    # except for the anchor scales
    model_configuration["DetectionLevels"] = {
        level: {"anchors": {"scales": scales, "aspect_ratios": [0.5, 1.0, 2.0]},
                "region_proposal": {"convolutions": [2, 64]},
                "detector": {"roi_align_pool_size": 7, "convolutions": [2, 64], "fully_connected": [256, 128]}}
        for level, scales in [("5", [32, 64, 128]), ("6", [64, 128, 256]), ("7", [128, 256, 512])]}
    model_configuration["Neck"] = {"type": "none", "output_channels": 32,
                                   "normalization": {"feature_map_normalization": "batch"},
                                   "pretrained": dict(NO_TRANSFER_SETTINGS)}
    model_configuration["RegionProposal"]["normalization"]["feature_map_normalization"] = "batch"
    # Proposals selected over all the levels at once, which gives no token
    model_configuration["RegionProposal"]["proposal_selection"] = "joint"
    model_configuration["Detector"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False})
    model_configuration["Detector"]["normalization"]["feature_map_normalization"] = "batch"

    # Smooth L1 box loss of the detection heads, which gives no token
    model_configuration["Detector"]["box_loss"] = "smooth_l1"

    # Classes, augmentation, batch size and learning rate
    model_configuration["Data"]["label_mode"] = "binary"
    model_configuration["Data"]["label_dictionary"] = {"background": 0, "foreground": 1}
    model_configuration["Data"]["data_augmentation"]["activated"] = True
    experiment_configuration["batch_size"] = 40
    experiment_configuration["learning_rate"] = 2.0e-4
    return experiment_configuration, model_configuration


def enable_inception(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Turns on inception in the backbone, from its third block, and in the detector.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["modules"]["inception"] = True
    model_configuration["Detector"]["modules"]["inception"] = True


def change_detection_levels(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Detects on blocks 6 and 7 only.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    del model_configuration["DetectionLevels"]["5"]


def add_feature_pyramid(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Adds a feature pyramid neck of 32 channels with batch normalization between the blocks and the heads.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Neck"]["type"] = "feature_pyramid"


def add_path_aggregation(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Adds a path aggregation neck of 64 channels with layer normalization between the blocks and the heads.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Neck"] = {"type": "path_aggregation", "output_channels": 64,
                                   "normalization": {"feature_map_normalization": "layer"},
                                   "pretrained": dict(NO_TRANSFER_SETTINGS)}


def use_complete_iou_detector_box_loss(experiment_configuration: Dict[str, Any],
                                       model_configuration: Dict[str, Any]) -> None:
    """
    Trains the box regression of the detection heads with the CIoU loss, the region proposal network always using the
    smooth L1, as in the Faster R-CNN experiments of the CIoU paper.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Detector"]["box_loss"] = "complete_iou"


def select_proposals_per_level(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Selects the region proposals level by level, a top-k and a suppression for every level, as torchvision and
    Detectron2.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["RegionProposal"]["proposal_selection"] = "per_level"


def change_training_settings(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Changes every training setting that can change between two resumptions of a run: turns off data augmentation,
    and trains with another batch size, learning rate and schedule and a moving average of the weights.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["data_augmentation"]["activated"] = False
    experiment_configuration["batch_size"] = 16
    experiment_configuration["learning_rate"] = 1.5e-3
    model_configuration["Optimization"].update({"learning_rate_schedule": "cosine", "warmup_iterations": 500,
                                                "decay_iterations": 20000,
                                                "exponential_moving_average": {"activated": True, "decay": 0.999}})


def test_reference_run_name() -> None:
    """
    Verifies the run name of the reference settings, that the same settings always give the same name, which lets a
    run resume in its own folder, and that the training settings that can change between two resumptions (augmentation,
    batch size, learning rate and schedule, moving average of the weights) keep that name.
    """
    experiment_configuration, model_configuration = get_reference_configurations()
    run_name = build_run_name(experiment_configuration=experiment_configuration,
                              model_configuration=model_configuration)
    assert run_name == REFERENCE_RUN_NAME
    assert build_run_name(experiment_configuration=copy.deepcopy(experiment_configuration),
                          model_configuration=copy.deepcopy(model_configuration)) == run_name

    # Changed training settings resume in the same folder
    change_training_settings(experiment_configuration=experiment_configuration, model_configuration=model_configuration)
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == run_name


@pytest.mark.parametrize(argnames=("nms_criterion", "nms_distance_exponent"),
                         argvalues=[("iou", 1.0), ("distance_iou", 1.0), ("distance_iou", 0.6)])
def test_final_nms_settings_keep_the_run_name(nms_criterion: str, nms_distance_exponent: float) -> None:
    """
    Verifies that the greedy NMS, DIoU-NMS and the exponent of its distance term give no token of the run name, the
    final non-maximum suppression running in evaluation and inference only, without changing any trained weight, so
    that a run keeps its folder whatever its final NMS.

    Args:
        nms_criterion (str): The 'Detector.nms_criterion' of the final NMS.
        nms_distance_exponent (float): The 'Detector.nms_distance_exponent' of DIoU-NMS.
    """
    experiment_configuration, model_configuration = get_reference_configurations()
    model_configuration["Detector"].update({"nms_criterion": nms_criterion,
                                            "nms_distance_exponent": nms_distance_exponent})
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == REFERENCE_RUN_NAME


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (enable_inception, REFERENCE_RUN_NAME.replace("_SE-RES-BN-", "_SE-RES-INC3-BN-").replace(
        "DSE-DRES-DBN", "DSE-DRES-DINC-DBN")),
    (change_detection_levels, REFERENCE_RUN_NAME.replace("_L567_A9.32-512.598295_", "_L67_A9.64-512.169502_")),
    (add_feature_pyramid, REFERENCE_RUN_NAME.replace("_L567_", "_L567-NFPN32BN_")),
    (add_path_aggregation, REFERENCE_RUN_NAME.replace("_L567_", "_L567-NPAN64LN_")),
    (use_complete_iou_detector_box_loss, REFERENCE_RUN_NAME.replace("-D7.2x64.256.128_", "-D7.2x64.256.128-DCIOU_")),
    (select_proposals_per_level, REFERENCE_RUN_NAME.replace("_RBN-R2x64_", "_RBN-R2x64-RPL_"))])
def test_changed_settings_change_run_name(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None],
                                          expected_run_name: str) -> None:
    """
    Verifies that changing a setting changes its token of the run name, so that changed settings never train in the
    folder of other settings.

    Args:
        modify_function (Callable[[Dict[str, Any], Dict[str, Any]], None]): Function changing the settings.
        expected_run_name (str): The expected run name of the changed settings.
    """
    experiment_configuration, model_configuration = get_reference_configurations()
    modify_function(experiment_configuration=experiment_configuration, model_configuration=model_configuration)
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == expected_run_name


def get_classification_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads the synthetic shapes classification configuration and sets every setting used by the run name explicitly,
    so that the expected names do not change when the configuration is edited. The label dictionary is set directly,
    the label mapping file of the configuration only existing once the synthetic dataset is created.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    model_configuration = yaml.safe_load(stream=CLASSIFICATION_CONFIGURATION_PATH.read_text())
    experiment_configuration = model_configuration.pop("ExperimentConfiguration")

    # Task, input and backbone of the run, without run name suffix
    experiment_configuration["task"] = "classification"
    experiment_configuration["run_name_suffix"] = None
    model_configuration["Data"]["image_settings"].update({"size": 256, "keep_ratio": True})
    model_configuration["Backbone"]["normalization"].update({"input_normalization": True,
                                                             "per_sample_normalization": False,
                                                             "feature_map_normalization": "batch"})
    model_configuration["Backbone"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False})
    model_configuration["Backbone"]["convolutions"] = {"1": [1, 16], "2": [2, 32], "3": [2, 32], "4": [2, 64],
                                                       "5": [2, 64], "6": [2, 128]}

    # Classification head and classes
    model_configuration["ClassificationHead"] = {"global_pooling": "average", "fully_connected": [128],
                                                 "fully_connected_normalization": "none",
                                                 "batch_normalization_momentum": 0.1,
                                                 "dropout_rate": 0.2, "output_mode": "multi_class",
                                                 "label_smoothing": 0.0, "class_weighting": "none",
                                                 "pretrained": dict(NO_HEAD_TRANSFER_SETTINGS)}
    model_configuration["Data"]["label_dictionary"] = {1: 0, 2: 1, 3: 2, 4: 3}

    # Augmentation, batch size and learning rate
    model_configuration["Data"]["data_augmentation"]["activated"] = True
    experiment_configuration["batch_size"] = 64
    experiment_configuration["learning_rate"] = 1.0e-3
    return experiment_configuration, model_configuration


def use_multi_label_outputs(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Predicts any number of classes per image, among 5 classes.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["ClassificationHead"]["output_mode"] = "multi_label"
    model_configuration["Data"]["label_dictionary"] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (None, CLASSIFICATION_REFERENCE_RUN_NAME),
    (use_multi_label_outputs, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_MC4", "_ML5"))])
def test_classification_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                                  expected_run_name: str) -> None:
    """
    Verifies the run names of the classification task: the CLS task code, the shared input and backbone sections, and
    the classification head and classes sections, the multi-label outputs changing the classes token.

    Args:
        modify_function (Callable[[Dict[str, Any], Dict[str, Any]], None] | None): Function changing the settings, or
            None for the reference settings.
        expected_run_name (str): The expected run name of the settings.
    """
    experiment_configuration, model_configuration = get_classification_reference_configurations()
    if modify_function is not None:
        modify_function(experiment_configuration=experiment_configuration, model_configuration=model_configuration)
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == expected_run_name


SEQUENCE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                               "sequence_classification" / "configuration_synthetic_sequences.yaml")

# Run name of the sequence reference settings set by get_sequence_reference_configurations
SEQUENCE_REFERENCE_RUN_NAME = "M2O_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128-BI_HAVG.64_MC4"


def get_sequence_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads the synthetic sequences configuration and sets every setting used by the run name explicitly, so that the
    expected names do not change when the configuration is edited. The label dictionary is set directly, the label
    mapping file of the configuration only existing once the synthetic dataset is created.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    model_configuration = yaml.safe_load(stream=SEQUENCE_CONFIGURATION_PATH.read_text())
    experiment_configuration = model_configuration.pop("ExperimentConfiguration")

    # Task, input and backbone of the run, without run name suffix
    experiment_configuration["task"] = "sequence_classification"
    experiment_configuration["run_name_suffix"] = None
    model_configuration["Data"]["image_settings"].update({"size": 128, "keep_ratio": True})
    model_configuration["Backbone"]["normalization"].update({"input_normalization": True,
                                                             "per_sample_normalization": False,
                                                             "feature_map_normalization": "batch"})
    model_configuration["Backbone"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False})
    model_configuration["Backbone"]["convolutions"] = {"1": [1, 16], "2": [2, 32], "3": [2, 32], "4": [2, 64],
                                                       "5": [2, 64]}

    # Maximum sequence length, temporal module, classification head and classes
    model_configuration["Data"]["sequence_settings"] = {"maximum_sequence_length": 8}
    model_configuration["TemporalModel"] = {"type": "gru", "hidden_size": 128, "number_layers": 1, "number_heads": 4,
                                            "kernel_size": 3, "dropout_rate": 0.0, "bidirectional": True,
                                            "causal": False, "positional_embedding": True,
                                            "pretrained": dict(NO_TRANSFER_SETTINGS)}
    model_configuration["ClassificationHead"] = {"global_pooling": "average", "fully_connected": [64],
                                                 "fully_connected_normalization": "none",
                                                 "batch_normalization_momentum": 0.1,
                                                 "dropout_rate": 0.2, "output_mode": "multi_class",
                                                 "label_smoothing": 0.0, "class_weighting": "none",
                                                 "pretrained": dict(NO_HEAD_TRANSFER_SETTINGS)}
    model_configuration["Data"]["label_dictionary"] = {1: 0, 2: 1, 3: 2, 4: 3}

    # Augmentation, batch size and learning rate
    model_configuration["Data"]["data_augmentation"]["activated"] = True
    experiment_configuration["batch_size"] = 16
    experiment_configuration["learning_rate"] = 1.0e-3
    return experiment_configuration, model_configuration


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (None, SEQUENCE_REFERENCE_RUN_NAME)])
def test_sequence_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                            expected_run_name: str) -> None:
    """
    Verifies the run names of the sequence classification task: the M2O task code, the sequence section, the temporal
    module section with the settings its type uses, then the classification head and classes sections.

    Args:
        modify_function (Callable[[Dict[str, Any], Dict[str, Any]], None] | None): Function changing the settings, or
            None for the reference settings.
        expected_run_name (str): The expected run name of the settings.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    if modify_function is not None:
        modify_function(experiment_configuration=experiment_configuration, model_configuration=model_configuration)
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == expected_run_name


def test_sequence_labelling_run_name() -> None:
    """
    Verifies the run name of the sequence labelling task: the M2M task code followed by the same sections as the
    sequence classification task, here with a causal temporal convolution.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    experiment_configuration["task"] = "sequence_labelling"
    model_configuration["TemporalModel"].update({"type": "temporal_convolution", "number_layers": 3, "kernel_size": 3,
                                                 "bidirectional": False, "causal": True})
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SEQUENCE_REFERENCE_RUN_NAME.replace(
        "M2O_", "M2M_").replace("_GRU1x128-BI_", "_TCN3x128.3-CAU_")


@pytest.mark.parametrize(argnames=("frames_per_class", "selection_loss_weight", "decoding", "expected_section"),
                         argvalues=[(1, 1.0, "hungarian", "_FPC1-SLW1")])
def test_frame_selection_run_names(frames_per_class: int, selection_loss_weight: float, decoding: str,
                                   expected_section: str) -> None:
    """
    Verifies the run names of the frame selection task: the SEL task code, the sections of the sequence tasks, then
    the selection section with the frames per class and the weight of the selection loss.

    Args:
        frames_per_class (int): The number of frames chosen for every wanted class.
        selection_loss_weight (float): The weight of the selection loss.
        decoding (str): The decoding of the selection.
        expected_section (str): The expected selection section of the run name, with its separator.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    experiment_configuration["task"] = "frame_selection"
    model_configuration["Selection"] = {"unselected_category": "unselected", "decoding": decoding,
                                        "frames_per_class": frames_per_class,
                                        "selection_loss_weight": selection_loss_weight}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SEQUENCE_REFERENCE_RUN_NAME.replace(
        "M2O_", "SEL_") + expected_section


# Run name of the sequence comparison reference settings set by get_sequence_comparison_reference_configurations
SEQUENCE_COMPARISON_REFERENCE_RUN_NAME = "CMP_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_MEAN_FDIF_HAVG.64_MC2"


def get_sequence_comparison_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds the sequence comparison reference settings from the sequence reference settings: the comparison task,
    a mean pooling, the difference fusion and 2 classes.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    experiment_configuration["task"] = "sequence_comparison"
    model_configuration["TemporalModel"]["type"] = "mean_pooling"
    model_configuration["Fusion"] = {"type": "difference", "number_heads": 4, "dropout_rate": 0.0,
                                     "sequence_normalization_scope": "member"}
    model_configuration["Data"]["label_dictionary"] = {1: 0, 2: 1}
    return experiment_configuration, model_configuration


def use_cross_attention(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Fuses the members with the cross attention of 4 heads, without dropout.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Fusion"]["type"] = "cross_attention"


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (use_cross_attention, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("_FDIF_", "_FXA4_"))])
def test_sequence_comparison_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                                       expected_run_name: str) -> None:
    """
    Verifies the run names of the sequence comparison task: the CMP task code, the member length section, the
    temporal module section, the fusion section, then the classification head and classes sections.

    Args:
        modify_function (Callable[[Dict[str, Any], Dict[str, Any]], None] | None): Function changing the settings, or
            None for the reference settings.
        expected_run_name (str): The expected run name of the settings.
    """
    experiment_configuration, model_configuration = get_sequence_comparison_reference_configurations()
    if modify_function is not None:
        modify_function(experiment_configuration=experiment_configuration, model_configuration=model_configuration)
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == expected_run_name


@pytest.mark.parametrize(argnames=("freeze", "learning_rate_multiplier", "freeze_normalization", "expected_tokens"),
                         argvalues=[(True, 1.0, False, "-PT-FRZ_")])
def test_pretrained_backbone_run_names(freeze: bool, learning_rate_multiplier: float, freeze_normalization: bool,
                                       expected_tokens: str) -> None:
    """
    Verifies the tokens of a transferred backbone at the end of the backbone section: PT, then FRZ when it is frozen.

    Args:
        freeze (bool): Whether the transferred backbone is frozen.
        learning_rate_multiplier (float): The learning rate multiplier of the transferred backbone.
        freeze_normalization (bool): Whether the normalization parameters of the backbone are kept fixed.
        expected_tokens (str): The expected end of the backbone section.
    """
    experiment_configuration, model_configuration = get_classification_reference_configurations()
    model_configuration["Backbone"]["pretrained"] = {"experiment_folder": "/runs/source", "iteration": None,
                                                     "freeze": freeze,
                                                     "freeze_normalization_parameters": freeze_normalization,
                                                     "learning_rate_multiplier": learning_rate_multiplier}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == CLASSIFICATION_REFERENCE_RUN_NAME.replace(
        "2x64.2x128_", f"2x64.2x128{expected_tokens}")


@pytest.mark.parametrize(argnames=("section_name", "freeze", "learning_rate_multiplier", "expected_section"),
                         argvalues=[("TemporalModel", True, 1.0, "_GRU1x128-BI-TPT-TFRZ_")])
def test_pretrained_temporal_module_and_head_run_names(section_name: str, freeze: bool,
                                                       learning_rate_multiplier: float, expected_section: str) -> None:
    """
    Verifies the tokens of a transferred temporal module at the end of its section: TPT, then TFRZ when it is frozen.
    The rest of the name is unchanged.

    Args:
        section_name (str): The section of the transferred component.
        freeze (bool): Whether the transferred component is frozen.
        learning_rate_multiplier (float): The learning rate multiplier of the transferred component.
        expected_section (str): The expected section of the component in the run name.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    model_configuration[section_name]["pretrained"] = {"experiment_folder": "/runs/source", "iteration": None,
                                                       "freeze": freeze,
                                                       "learning_rate_multiplier": learning_rate_multiplier}
    original_section = "_GRU1x128-BI_" if section_name == "TemporalModel" else "_HAVG.64_"
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SEQUENCE_REFERENCE_RUN_NAME.replace(
        original_section, expected_section)


@pytest.mark.parametrize(argnames=("class_layer", "freeze", "expected_section"),
                         argvalues=[(True, False, "_HAVG.64-HPT_"), (False, False, "_HAVG.64-HPT-HRCL_"),
                                    (False, True, "_HAVG.64-HPT-HRCL-HFRZ_")])
def test_pretrained_classification_head_run_names(class_layer: bool, freeze: bool, expected_section: str) -> None:
    """
    Verifies the tokens of a transferred classification head at the end of its section: HPT, then HRCL when it is
    loaded without its class layer, which starts random, then HFRZ when it is frozen. A head loaded with its class
    layer keeps the name of the runs written before 'class_layer' existed, which loaded it. The rest of the name is
    unchanged.

    Args:
        class_layer (bool): Whether the class layer is loaded with the head.
        freeze (bool): Whether the transferred head is frozen.
        expected_section (str): The expected section of the head in the run name.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    model_configuration["ClassificationHead"]["pretrained"] = {**NO_HEAD_TRANSFER_SETTINGS,
                                                               "experiment_folder": "/runs/source",
                                                               "freeze": freeze, "class_layer": class_layer}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SEQUENCE_REFERENCE_RUN_NAME.replace(
        "_HAVG.64_", expected_section)


@pytest.mark.parametrize(argnames=("head_dropout_rate", "temporal_dropout_rate", "fusion_dropout_rate"),
                         argvalues=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.1),
                                    (0.5, 0.3, 0.1)])
def test_dropout_rates_keep_run_name(head_dropout_rate: float, temporal_dropout_rate: float,
                                     fusion_dropout_rate: float) -> None:
    """
    Verifies that the dropout rates of the classification head, of the temporal module and of the cross attention
    fusion give no token, so that a run can be resumed in its folder with changed rates.

    Args:
        head_dropout_rate (float): The dropout of the classification head.
        temporal_dropout_rate (float): The dropout between the layers of the GRU.
        fusion_dropout_rate (float): The dropout of the cross attention weights.
    """
    experiment_configuration, model_configuration = get_sequence_comparison_reference_configurations()
    model_configuration["TemporalModel"].update({"type": "gru", "number_layers": 2, "bidirectional": False})
    model_configuration["Fusion"]["type"] = "cross_attention"
    reference_run_name = build_run_name(experiment_configuration=experiment_configuration,
                                        model_configuration=model_configuration)

    # Change every dropout rate, the run name staying the same
    model_configuration["ClassificationHead"]["dropout_rate"] = head_dropout_rate
    model_configuration["TemporalModel"]["dropout_rate"] = temporal_dropout_rate
    model_configuration["Fusion"]["dropout_rate"] = fusion_dropout_rate
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == reference_run_name
    assert reference_run_name == SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("_MEAN_FDIF_", "_GRU2x128_FXA4_")


@pytest.mark.parametrize(argnames=("feature_map_normalization", "normalization_scope", "expected_sections"),
                         argvalues=[("sequence", "member", ("-SN-B1x16", "_FDIF_"))])
def test_sequence_normalization_run_name_tokens(feature_map_normalization: str, normalization_scope: str,
                                                expected_sections: Tuple[str, str]) -> None:
    """
    Verifies the tokens of the sequence normalization of a comparison model: SN instead of BN in the backbone section,
    the fusion section keeping no token when every member has its own statistics.

    Args:
        feature_map_normalization (str): The feature map normalization of the backbone.
        normalization_scope (str): The scope of the statistics of the sequence normalization.
        expected_sections (Tuple[str, str]): The expected normalization of the backbone section and fusion section.
    """
    experiment_configuration, model_configuration = get_sequence_comparison_reference_configurations()
    model_configuration["Backbone"]["normalization"]["feature_map_normalization"] = feature_map_normalization
    model_configuration["Fusion"]["sequence_normalization_scope"] = normalization_scope
    expected_run_name = SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("-BN-B1x16", expected_sections[0]).replace(
        "_FDIF_", expected_sections[1])
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == expected_run_name


def test_optimization_run_name_tokens() -> None:
    """
    Verifies the optimizer section at the end of the run name, written only for an optimizer other than Adam or a
    weight decay, the schedule giving no token.
    """
    experiment_configuration, model_configuration = get_classification_reference_configurations()
    model_configuration["Optimization"] = {"optimizer": "adamw", "weight_decay": 1e-4,
                                           "learning_rate_schedule": "cosine", "warmup_iterations": 500,
                                           "decay_iterations": 20000, "minimum_learning_rate_ratio": 0.01}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == CLASSIFICATION_REFERENCE_RUN_NAME.replace(
        "_MC4", "_MC4_ADAMW-WD1e-4")


@pytest.mark.parametrize(argnames=("run_name_suffix", "expected_ending"), argvalues=[("v1", "_MC4_ADAMW-WD1e-4_v1")])
def test_run_name_suffix_ends_the_run_name(run_name_suffix: str | None, expected_ending: str) -> None:
    """
    Verifies that the run name suffix is the last section of the run name, after the optimizer section, so that a run
    of the same settings given a suffix trains in a folder of its own instead of resuming the run without suffix.

    Args:
        run_name_suffix (str | None): The run name suffix of the configuration.
        expected_ending (str): The expected end of the run name, from its classes section.
    """
    experiment_configuration, model_configuration = get_classification_reference_configurations()
    model_configuration["Optimization"].update({"optimizer": "adamw", "weight_decay": 1e-4})
    experiment_configuration["run_name_suffix"] = run_name_suffix
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == CLASSIFICATION_REFERENCE_RUN_NAME.replace(
        "_MC4", expected_ending)


SEGMENTATION_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                   "semantic_segmentation" / "configuration_synthetic_shapes.yaml")

# Run name of the semantic segmentation reference settings set by get_segmentation_reference_configurations
SEGMENTATION_REFERENCE_RUN_NAME = ("SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_"
                                   "UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC4")


def get_segmentation_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads the synthetic shapes semantic segmentation configuration and sets every setting used by the run name
    explicitly, so that the expected names do not change when the configuration is edited. The label dictionary is
    set directly, the label mapping file of the configuration only existing once the synthetic dataset is created.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    model_configuration = yaml.safe_load(stream=SEGMENTATION_CONFIGURATION_PATH.read_text())
    experiment_configuration = model_configuration.pop("ExperimentConfiguration")

    # Task, input and backbone of the run, without run name suffix
    experiment_configuration["task"] = "semantic_segmentation"
    experiment_configuration["run_name_suffix"] = None
    model_configuration["Data"]["image_settings"].update({"size": 512, "keep_ratio": True})
    model_configuration["Data"]["tiling"] = {"activated": False, "scale": 1.0, "overlap": 128}
    model_configuration["Backbone"]["normalization"].update({"input_normalization": True,
                                                             "per_sample_normalization": False,
                                                             "feature_map_normalization": "batch"})
    model_configuration["Backbone"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False})
    model_configuration["Backbone"]["convolutions"] = {"1": [1, 16], "2": [2, 32], "3": [2, 32], "4": [2, 64],
                                                       "5": [2, 64], "6": [2, 128]}

    # Decoder, losses and classes, the background class 0 being added to the four mapped categories
    model_configuration["SegmentationDecoder"] = {"upsampling": "transposed",
                                                  "convolutions": {"5": [2, 64], "4": [2, 64], "3": [2, 32],
                                                                   "2": [2, 32], "1": [1, 16]},
                                                  "modules": {"squeeze_and_excite": False, "residual_nets": False},
                                                  "feature_map_normalization": "batch", "dropout_rate": 0.0,
                                                  "deep_supervision_weights": {},
                                                  "pretrained": {**NO_TRANSFER_SETTINGS, "class_layers": False}}
    model_configuration["SegmentationLoss"] = {"cross_entropy_weight": 1.0, "focal_gamma": 0.0,
                                               "label_smoothing": 0.0, "class_weighting": "none", "dice_weight": 1.0,
                                               "lovasz_weight": 0.0, "boundary_weight": 0.0, "boundary_distance": 3}
    model_configuration["Data"]["label_mode"] = "multi_class"
    model_configuration["Data"]["label_dictionary"] = {1: 1, 2: 2, 3: 3, 4: 4, "background": 0}
    return experiment_configuration, model_configuration


def train_on_half_scale_tiles(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Trains on tiles of the images scaled by one half, with evaluation tiles overlapping by 64 pixels.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["tiling"] = {"activated": True, "scale": 0.5, "overlap": 64}


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (None, SEGMENTATION_REFERENCE_RUN_NAME),
    (train_on_half_scale_tiles, SEGMENTATION_REFERENCE_RUN_NAME.replace("_UTC-", "_TL0.5_UTC-"))])
def test_segmentation_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                                expected_run_name: str) -> None:
    """
    Verifies the run names of the semantic segmentation task: the SEG task code, the shared input and backbone
    sections, the tiling section with the scale of the tiles, the decoder section with its levels from the deepest
    one, and the classes section counting the foreground classes.

    Args:
        modify_function (Callable[[Dict[str, Any], Dict[str, Any]], None] | None): Function changing the settings, or
            None for the reference settings.
        expected_run_name (str): The expected run name of the settings.
    """
    experiment_configuration, model_configuration = get_segmentation_reference_configurations()
    if modify_function is not None:
        modify_function(experiment_configuration=experiment_configuration, model_configuration=model_configuration)
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == expected_run_name


@pytest.mark.parametrize(argnames=("freeze", "class_layers", "learning_rate_multiplier", "expected_tokens"),
                         argvalues=[(True, True, 1.0, "-DPT-DCL-DFRZ_"), (False, False, 0.5, "-DPT_")])
def test_pretrained_segmentation_decoder_run_names(freeze: bool, class_layers: bool,
                                                   learning_rate_multiplier: float, expected_tokens: str) -> None:
    """
    Verifies the tokens of a transferred segmentation decoder at the end of the decoder section: DPT, then DCL when
    its class layers are loaded too, then DFRZ when it is frozen, and that the dropout rate of the segmentation head
    gives no token.

    Args:
        freeze (bool): Whether the transferred decoder is frozen.
        class_layers (bool): Whether the segmentation head and deep supervision heads are loaded with the decoder.
        learning_rate_multiplier (float): The learning rate multiplier of the transferred decoder.
        expected_tokens (str): The expected end of the decoder section.
    """
    experiment_configuration, model_configuration = get_segmentation_reference_configurations()
    model_configuration["SegmentationDecoder"].update({"dropout_rate": 0.3,
                                                       "pretrained": {"experiment_folder": "/runs/source",
                                                                      "iteration": None, "freeze": freeze,
                                                                      "class_layers": class_layers,
                                                                      "learning_rate_multiplier":
                                                                          learning_rate_multiplier}})
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SEGMENTATION_REFERENCE_RUN_NAME.replace(
        "2x32.1x16_", f"2x32.1x16{expected_tokens}")


def get_instance_segmentation_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds the reference settings of the instance segmentation: the object detection reference settings with the task
    of the instance segmentation and its mask heads set explicitly.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    experiment_configuration, model_configuration = get_reference_configurations()
    experiment_configuration["task"] = "instance_segmentation"
    for level_configuration in model_configuration["DetectionLevels"].values():
        level_configuration["mask_head"] = {"roi_align_pool_size": 14, "convolutions": [4, 64]}
    model_configuration["MaskHead"] = {"shared": False, "mask_resolution": 28, "upsampling": "transposed",
                                       "skip_blocks_at_pool_size": [], "image_branch_convolutions": [0, 0],
                                       "skip_blocks_at_mask_resolution": [], "fusion_convolutions": [0, 0],
                                       "normalization": {"feature_map_normalization": "batch"},
                                       "modules": {"squeeze_and_excite": False, "residual_nets": True,
                                                   "inception": False},
                                       "inference": {"mask_threshold": 0.5}, "pretrained": dict(NO_TRANSFER_SETTINGS)}
    return experiment_configuration, model_configuration


def change_mask_heads(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Turns on every module of the mask heads, uses layer normalization, bilinear upsamplings, larger masks and a
    different pool size and head on level 7.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    mask_head_configuration = model_configuration["MaskHead"]
    mask_head_configuration["modules"].update({"squeeze_and_excite": True, "inception": True})
    mask_head_configuration["normalization"]["feature_map_normalization"] = "layer"
    mask_head_configuration["upsampling"] = "bilinear"
    mask_head_configuration["mask_resolution"] = 56
    model_configuration["DetectionLevels"]["7"]["mask_head"] = {"roi_align_pool_size": 7, "convolutions": [2, 128]}


def use_every_mask_head_fine_input(experiment_configuration: Dict[str, Any],
                                   model_configuration: Dict[str, Any]) -> None:
    """
    Turns on every finer input of the mask heads, listed out of the order of the tokens: the fusion convolutions, the
    backbone skips of blocks 4 and 5 at the pool size, the image branch and the skips of the normalized input image
    and block 3 at the mask resolution.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["MaskHead"].update({"fusion_convolutions": [1, 64], "skip_blocks_at_pool_size": [4, 5],
                                            "image_branch_convolutions": [5, 32],
                                            "skip_blocks_at_mask_resolution": [0, 3]})


def use_mask_head_fusion_convolutions(experiment_configuration: Dict[str, Any],
                                      model_configuration: Dict[str, Any]) -> None:
    """
    Adds two fusion convolutions of 32 channels to the mask heads, without finer input.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["MaskHead"]["fusion_convolutions"] = [2, 32]


def share_the_mask_head_with_the_image_branch(experiment_configuration: Dict[str, Any],
                                              model_configuration: Dict[str, Any]) -> None:
    """
    Shares one mask head between the levels, with the image branch and fusion convolutions of design A of stage 3b.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["MaskHead"].update({"shared": True, "image_branch_convolutions": [5, 32],
                                            "fusion_convolutions": [1, 64]})


# Run name of the instance segmentation reference settings: the detection sections with the mask heads before the
# classes
INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME = REFERENCE_RUN_NAME.replace("DET_", "INS_").replace(
    "_BIN", "_MRES-MBN-MUTC-M14.4x64-MR28_BIN")


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (change_mask_heads, INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME.replace(
        "_MRES-MBN-MUTC-M14.4x64-MR28_", "_MSE-MRES-MINC-MLN-MUBL-M14.4x64+14.4x64+7.2x128-MR56_")),
    (use_every_mask_head_fine_input, INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME.replace(
        "-MR28_", "-MR28-MSP4.5-MIB5x32-MSM0.3-MF1x64_")),
    (use_mask_head_fusion_convolutions, INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME.replace("-MR28_", "-MR28-MF2x32_")),
    (share_the_mask_head_with_the_image_branch, INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME.replace(
        "-M14.4x64-MR28_", "-M14.4x64-MSH-MR28-MIB5x32-MF1x64_"))])
def test_instance_segmentation_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                                         expected_run_name: str) -> None:
    """
    Verifies the run names of the instance segmentation: the INS code, the detection sections, and the mask heads
    section (modules, normalization, upsampling, pool size and convolutions of every level, MSH for a mask head shared
    by the levels, mask resolution, then the finer inputs and fusion convolutions in the order of their data flow when
    used) before the classes.

    Args:
        modify_function (Callable[[Dict[str, Any], Dict[str, Any]], None] | None): Function changing the reference
            settings, None to keep them.
        expected_run_name (str): The expected run name.
    """
    experiment_configuration, model_configuration = get_instance_segmentation_reference_configurations()
    if modify_function is not None:
        modify_function(experiment_configuration=experiment_configuration, model_configuration=model_configuration)
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == expected_run_name


@pytest.mark.parametrize(argnames=("region_proposal_freeze", "detector_freeze", "detector_class_layers",
                                   "expected_sections"), argvalues=[
    (True, True, True, "_RBN-R2x64-RPT-RFRZ_DSE-DRES-DBN-D7.2x64.256.128-DPT-DFRZ_MRES"),
    (False, False, False, "_RBN-R2x64-RPT_DSE-DRES-DBN-D7.2x64.256.128-DPT-DRCL_MRES"),
    (None, True, False, "_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128-DPT-DRCL-DFRZ_MRES")])
def test_pretrained_detection_head_run_names(region_proposal_freeze: bool | None, detector_freeze: bool | None,
                                             detector_class_layers: bool, expected_sections: str) -> None:
    """
    Verifies the tokens of transferred region proposal and detection heads at the end of their sections, before the
    mask heads of an instance segmentation run: RPT and DPT, then DRCL when the detection heads are loaded without
    their class layers, then RFRZ and DFRZ when they are frozen, the learning rate multiplier giving no token. Heads
    loaded with their class layers keep the name of the runs written before 'class_layers' existed, which loaded them.

    Args:
        region_proposal_freeze (bool | None): Whether the transferred region proposal heads are frozen, None when they
            are not transferred.
        detector_freeze (bool | None): Whether the transferred detection heads are frozen, None when they are not
            transferred.
        detector_class_layers (bool): Whether the class layers are loaded with the detection heads.
        expected_sections (str): The expected region proposal and detector sections of the run name.
    """
    experiment_configuration, model_configuration = get_instance_segmentation_reference_configurations()
    for section_name, freeze in [("RegionProposal", region_proposal_freeze), ("Detector", detector_freeze)]:
        if freeze is not None:
            model_configuration[section_name]["pretrained"] = {"experiment_folder": "/runs/source", "iteration": None,
                                                               "freeze": freeze, "learning_rate_multiplier": 0.1}
    model_configuration["Detector"]["pretrained"]["class_layers"] = detector_class_layers
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME.replace(
        "_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MRES", expected_sections)


@pytest.mark.parametrize(argnames=("neck_type", "freeze", "expected_levels_section"), argvalues=[
    ("feature_pyramid", False, "_L567-NFPN32BN-NPT_"), ("feature_pyramid", True, "_L567-NFPN32BN-NPT-NFRZ_"),
    ("path_aggregation", True, "_L567-NPAN32BN-NPT-NFRZ_")])
def test_pretrained_neck_run_names(neck_type: str, freeze: bool, expected_levels_section: str) -> None:
    """
    Verifies the tokens of a transferred neck at the end of the levels section, after the token of the neck: NPT, then
    NFRZ when it is frozen, the learning rate multiplier giving no token, so that the runs whose neck starts from random
    weights keep their names.

    Args:
        neck_type (str): The 'Neck.type' of the run.
        freeze (bool): Whether the transferred neck is frozen.
        expected_levels_section (str): The expected levels section of the run name.
    """
    experiment_configuration, model_configuration = get_reference_configurations()
    model_configuration["Neck"]["type"] = neck_type
    model_configuration["Neck"]["pretrained"] = {"experiment_folder": "/runs/source", "iteration": None,
                                                 "freeze": freeze, "learning_rate_multiplier": 0.1}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == REFERENCE_RUN_NAME.replace(
        "_L567_", expected_levels_section)


@pytest.mark.parametrize(argnames=("freeze", "class_layers", "expected_tokens"),
                         argvalues=[(True, True, "-MR28-MPT-MFRZ_"), (False, False, "-MR28-MPT-MRCL_")])
def test_pretrained_mask_head_run_names(freeze: bool, class_layers: bool, expected_tokens: str) -> None:
    """
    Verifies the tokens of transferred mask heads at the end of the mask heads section of an instance segmentation
    run: MPT, then MRCL when they are loaded without their class layers, then MFRZ when they are frozen, the learning
    rate multiplier giving no token.

    Args:
        freeze (bool): Whether the transferred mask heads are frozen.
        class_layers (bool): Whether the class layers are loaded with the mask heads.
        expected_tokens (str): The expected end of the mask heads section.
    """
    experiment_configuration, model_configuration = get_instance_segmentation_reference_configurations()
    model_configuration["MaskHead"]["pretrained"] = {"experiment_folder": "/runs/source", "iteration": None,
                                                     "freeze": freeze, "class_layers": class_layers,
                                                     "learning_rate_multiplier": 0.1}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME.replace(
        "-MR28_", expected_tokens)


SELF_SUPERVISED_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                      "self_supervised" / "configuration_synthetic_shapes.yaml")

# Run name of the self-supervised reference settings set by get_self_supervised_reference_configurations
SELF_SUPERVISED_REFERENCE_RUN_NAME = ("SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_"
                                      "SIMCLR-T0.2")


def get_self_supervised_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads the synthetic shapes self-supervised configuration and sets every setting used by the run name explicitly,
    so that the expected names do not change when the configuration is edited. The label dictionary is set directly,
    the label mapping file of the configuration only existing once the synthetic dataset is created.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    model_configuration = yaml.safe_load(stream=SELF_SUPERVISED_CONFIGURATION_PATH.read_text())
    experiment_configuration = model_configuration.pop("ExperimentConfiguration")

    # No run name suffix, which would end the name
    experiment_configuration["run_name_suffix"] = None

    # Input and backbone of the run
    model_configuration["Data"]["image_settings"].update({"size": 256, "keep_ratio": True})
    model_configuration["Backbone"]["normalization"].update({"input_normalization": True,
                                                             "per_sample_normalization": False,
                                                             "feature_map_normalization": "batch"})
    model_configuration["Backbone"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "residual_nets_v2": False, "inception": False})
    model_configuration["Backbone"]["convolutions"] = {"1": [1, 16], "2": [2, 32], "3": [2, 32], "4": [2, 64],
                                                       "5": [2, 64], "6": [2, 128]}

    # Projection head, method, optimizer and the classes of the evaluation
    model_configuration["ProjectionHead"].update({"global_pooling": "average", "fully_connected": [512, 128],
                                                  "fully_connected_normalization": "batch", "method": "simclr",
                                                  "simclr": {"temperature": 0.2},
                                                  "barlow_twins": {"off_diagonal_weight": 0.005}})
    model_configuration["Optimization"].update({"optimizer": "adam", "weight_decay": 0.0})
    model_configuration["Data"]["label_mode"] = "multi_class"
    model_configuration["Data"]["label_dictionary"] = {1: 0, 2: 1, 3: 2, 4: 3}
    return experiment_configuration, model_configuration


@pytest.mark.parametrize(argnames=("head_settings", "label_dictionary", "expected_sections"), argvalues=[
    ({}, {1: 0, 2: 1, 3: 2, 4: 3}, "_PAVG.512.128-PBN_SIMCLR-T0.2")])
def test_self_supervised_run_names(head_settings: Dict[str, Any], label_dictionary: Dict[Any, int],
                                   expected_sections: str) -> None:
    """
    Verifies the sections of a self-supervised run: the projection head (pooling, layers and the normalization of its
    hidden layers), then the method with the setting changing its loss, the data augmentation drawing the views and
    the grouped batches giving no token.

    Args:
        head_settings (Dict[str, Any]): The 'ProjectionHead' settings replaced for the test.
        label_dictionary (Dict[Any, int]): The label dictionary of the run, empty for a dataset without labels.
        expected_sections (str): The expected sections after the backbone.
    """
    experiment_configuration, model_configuration = get_self_supervised_reference_configurations()
    model_configuration["ProjectionHead"].update(head_settings)
    model_configuration["Data"]["label_dictionary"] = label_dictionary
    model_configuration["Data"]["data_augmentation"]["types"]["random_rotation"]["probability"] = 0.0
    model_configuration["Data"]["data_augmentation"]["types"]["random_solarize"]["probability"] = 0.5
    model_configuration["Data"]["grouped_batches"].update({"activated": True, "batch_key": "site"})
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SELF_SUPERVISED_REFERENCE_RUN_NAME.replace(
        "_PAVG.512.128-PBN_SIMCLR-T0.2", expected_sections)


KEYPOINT_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                               "keypoint_detection" / "configuration_example.yaml")

# Run name of the keypoint example configuration: the COCO person of 17 keypoints in whole images of 512 pixels, with
# the high resolution network of base width 32, its 1, 4 and 3 modules per stage of 4 blocks per branch, and the
# higher resolution branch of the keypoint head
KEYPOINT_REFERENCE_RUN_NAME = "KPT_IN-KAR-512_SE-RES-BN-B1x64.2x64.2x64_HRSE-HRRES-HRBN-HR32.1.4.3x4_K17-HH_MC1"

# Keypoint names of the COCO person, the only class of the keypoint example configuration
COCO_PERSON_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                         "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                         "right_knee", "left_ankle", "right_ankle"]


def get_keypoint_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Reads the keypoint detection example configuration as shipped, whose run name is the one documented for it, and
    sets the label dictionary and the keypoint categories directly, both being written by the configuration loader
    from the label mapping file and the training labels, which only exist once the COCO dataset is downloaded.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    model_configuration = yaml.safe_load(stream=KEYPOINT_CONFIGURATION_PATH.read_text())
    experiment_configuration = model_configuration.pop("ExperimentConfiguration")

    # The COCO person as the only foreground class, next to the background
    model_configuration["Data"]["label_mode"] = "multi_class"
    model_configuration["Data"]["label_dictionary"] = {"background": 0, 1: 1}
    model_configuration["Data"]["keypoint_categories"] = {1: {"name": "person", "keypoints": COCO_PERSON_KEYPOINTS,
                                                              "skeleton": []}}
    return experiment_configuration, model_configuration


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (None, KEYPOINT_REFERENCE_RUN_NAME)])
def test_keypoint_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                            expected_run_name: str) -> None:
    """
    Verifies the run names of the keypoint detection: the KPT task code, then after the backbone the high resolution
    network (modules, normalization, widths, stages and blocks), the heatmap channels of the keypoint head with HH for
    its higher resolution branch, and the foreground classes.

    Args:
        modify_function (Callable[[Dict[str, Any], Dict[str, Any]], None] | None): Function changing the reference
            settings, or None for the reference itself.
        expected_run_name (str): The expected run name.
    """
    experiment_configuration, model_configuration = get_keypoint_reference_configurations()
    if modify_function is not None:
        modify_function(experiment_configuration=experiment_configuration, model_configuration=model_configuration)
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == expected_run_name


@pytest.mark.parametrize(argnames=("freeze", "class_layers", "expected_section"),
                         argvalues=[(False, True, "_K17-HH-KHPT_"), (True, False, "_K17-HH-KHPT-KHRCL-KHFRZ_")])
def test_pretrained_keypoint_head_run_names(freeze: bool, class_layers: bool, expected_section: str) -> None:
    """
    Verifies the tokens of a transferred keypoint head at the end of its section: KHPT, then KHRCL when it is loaded
    without its keypoint channel layers, which start random, then KHFRZ when it is frozen. A head loaded with them
    keeps the name of the runs written before 'class_layers' existed, which loaded them.

    Args:
        freeze (bool): Whether the transferred keypoint head is frozen.
        class_layers (bool): Whether the keypoint channel layers are loaded with the head.
        expected_section (str): The expected section of the keypoint head in the run name.
    """
    experiment_configuration, model_configuration = get_keypoint_reference_configurations()
    model_configuration["KeypointHead"]["pretrained"].update({"experiment_folder": "/runs/source", "freeze": freeze,
                                                              "class_layers": class_layers})
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == KEYPOINT_REFERENCE_RUN_NAME.replace(
        "_K17-HH_", expected_section)
