"""
Tests of build_run_name of vision_training/configuration/run_name.py on the reference configurations of every task: the
reference run names are fixed, and every changed setting (architecture, inputs, heads, temporal modules, fusions,
transfers, augmentations, normalization, optimization) changes the run name with its token.
"""

import copy
import pytest
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from vision_training.configuration.run_name import build_run_name
from vision_training.configuration.configuration_loader import load_experiment_configuration

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")

# Run name of the reference settings set by get_reference_configurations
REFERENCE_RUN_NAME = ("DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_"
                      "DSE-DRES-DBN-D7.2x64.256.128_BIN_AUG-BS40-LR2e-4")

CLASSIFICATION_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                     "classification" / "configuration_synthetic_shapes.yaml")

# Run name of the classification reference settings set by get_classification_reference_configurations
CLASSIFICATION_REFERENCE_RUN_NAME = ("CLS_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128-DO0.2_MC4_"
                                     "AUG-BS64-LR1e-3")

# Pretrained settings of a temporal module or classification head trained from random weights
NO_TRANSFER_SETTINGS = {"experiment_folder": None, "iteration": None, "freeze": False, "learning_rate_multiplier": 1.0}


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

    # Input normalization, resizing mode and size
    model_configuration["Backbone"]["normalization"].update({"input_normalization": True,
                                                             "per_sample_normalization": False,
                                                             "feature_map_normalization": "batch"})
    model_configuration["Data"]["image_settings"].update({"size": 1024, "keep_ratio": True})

    # Backbone modules, blocks and detection levels
    model_configuration["Backbone"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False, "inception_starting_block": 3})
    model_configuration["Backbone"]["convolutions"] = {"1": [1, 16], "2": [2, 32], "3": [2, 32], "4": [4, 64],
                                                       "5": [4, 64], "6": [6, 128], "7": [6, 128]}
    model_configuration["Backbone"]["detection_convolution_block_indices"] = [5, 6, 7]

    # Anchors, region proposal heads and detector heads, identical on every level
    model_configuration["Anchors"]["scales_and_ratios"] = {
        level: {"scales": scales, "aspect_ratios": [0.5, 1.0, 2.0]}
        for level, scales in [("5", [32, 64, 128]), ("6", [64, 128, 256]), ("7", [128, 256, 512])]}
    model_configuration["RegionProposal"]["normalization"]["feature_map_normalization"] = "batch"
    model_configuration["RegionProposal"]["head_configurations"] = {level: [2, 64] for level in ["5", "6", "7"]}
    model_configuration["Detector"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False})
    model_configuration["Detector"]["normalization"]["feature_map_normalization"] = "batch"
    model_configuration["Detector"]["head_configurations"] = {
        level: {"roi_align_pool_size": 7, "convolutions": [2, 64], "fully_connected": [256, 128]}
        for level in ["5", "6", "7"]}

    # Classes, augmentation, batch size and learning rate
    model_configuration["Data"]["label_mode"] = "binary"
    model_configuration["Data"]["label_dictionary"] = {"background": 0, "foreground": 1}
    model_configuration["Data"]["data_augmentation"]["activated"] = True
    experiment_configuration["batch_size"] = 40
    experiment_configuration["learning_rate"] = 2.0e-4
    return experiment_configuration, model_configuration


def disable_squeeze_and_excite(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Turns off squeeze-and-excite in the backbone.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["modules"]["squeeze_and_excite"] = False


def enable_inception(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Turns on inception in the backbone, from its third block, and in the detector.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["modules"]["inception"] = True
    model_configuration["Detector"]["modules"]["inception"] = True


def change_input_settings(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Replaces the input normalization by the per-sample one and resizes without keeping the aspect ratio.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["normalization"].update({"input_normalization": False,
                                                             "per_sample_normalization": True})
    model_configuration["Data"]["image_settings"]["keep_ratio"] = False


def change_feature_map_normalizations(experiment_configuration: Dict[str, Any],
                                      model_configuration: Dict[str, Any]) -> None:
    """
    Uses layer normalization in the backbone and the detector, and no normalization in the region proposal heads.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["normalization"]["feature_map_normalization"] = "layer"
    model_configuration["RegionProposal"]["normalization"]["feature_map_normalization"] = "none"
    model_configuration["Detector"]["normalization"]["feature_map_normalization"] = "layer"


def widen_last_block(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Doubles the filters of the last backbone block.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["convolutions"]["7"] = [6, 256]


def change_detection_levels(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Detects on blocks 6 and 7 only.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["detection_convolution_block_indices"] = [6, 7]


def differ_heads_per_level(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Gives level 5 four anchor scales and a larger detector head, and level 7 a deeper and wider region proposal
    head, so that the per-level settings are listed level by level.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Anchors"]["scales_and_ratios"]["5"]["scales"] = [16, 32, 64, 128]
    model_configuration["RegionProposal"]["head_configurations"]["7"] = [3, 128]
    model_configuration["Detector"]["head_configurations"]["5"] = {"roi_align_pool_size": 14, "convolutions": [2, 64],
                                                                   "fully_connected": [512]}


def use_multi_class_labels(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Uses a multi-class label mapping of 4 foreground classes.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["label_mode"] = "multi_class"
    model_configuration["Data"]["label_dictionary"] = {1: 1, 2: 2, 3: 3, 4: 4, "background": 0}


def change_training_settings(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Turns off data augmentation, and trains with another batch size and learning rate.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["data_augmentation"]["activated"] = False
    experiment_configuration["batch_size"] = 16
    experiment_configuration["learning_rate"] = 1.5e-3


def test_reference_run_name() -> None:
    """
    Verifies the run name of the reference settings, and that the same settings always give the same name, which
    lets a run resume in its own folder.
    """
    experiment_configuration, model_configuration = get_reference_configurations()
    run_name = build_run_name(experiment_configuration=experiment_configuration,
                              model_configuration=model_configuration)
    assert run_name == REFERENCE_RUN_NAME
    assert build_run_name(experiment_configuration=copy.deepcopy(experiment_configuration),
                          model_configuration=copy.deepcopy(model_configuration)) == run_name


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (disable_squeeze_and_excite, REFERENCE_RUN_NAME.replace("_SE-RES-BN-", "_RES-BN-")),
    (enable_inception, REFERENCE_RUN_NAME.replace("_SE-RES-BN-", "_SE-RES-INC3-BN-").replace(
        "DSE-DRES-DBN", "DSE-DRES-DINC-DBN")),
    (change_input_settings, REFERENCE_RUN_NAME.replace("_IN-KAR-1024_", "_PSN-1024_")),
    (change_feature_map_normalizations, REFERENCE_RUN_NAME.replace("-BN-B", "-LN-B").replace(
        "RBN", "RNN").replace("DBN", "DLN")),
    (widen_last_block, REFERENCE_RUN_NAME.replace("6x128.6x128", "6x128.6x256")),
    (change_detection_levels, REFERENCE_RUN_NAME.replace("_L567_A9.32-512_", "_L67_A9.64-512_")),
    (differ_heads_per_level, REFERENCE_RUN_NAME.replace("A9.32-512", "A12+9+9.16-512").replace(
        "R2x64", "R2x64+2x64+3x128").replace("D7.2x64.256.128", "D14.2x64.512+7.2x64.256.128+7.2x64.256.128")),
    (use_multi_class_labels, REFERENCE_RUN_NAME.replace("_BIN_", "_MC4_")),
    (change_training_settings, REFERENCE_RUN_NAME.replace("_AUG-BS40-LR2e-4", "_BS16-LR1.5e-3"))])
def test_changed_settings_change_run_name(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None],
                                          expected_run_name: str) -> None:
    """
    Verifies that changing a setting changes its token of the run name, a flag token being left out when its setting
    is off and per-level settings being listed level by level when they differ, so that changed settings never train
    in the folder of other settings.

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

    # Task, input and backbone of the run
    experiment_configuration["task"] = "classification"
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
                                                 "dropout_rate": 0.2, "output_mode": "multi_class",
                                                 "label_smoothing": 0.0, "class_weighting": "none",
                                                 "pretrained": dict(NO_TRANSFER_SETTINGS)}
    model_configuration["Data"]["label_dictionary"] = {1: 0, 2: 1, 3: 2, 4: 3}

    # Augmentation, batch size and learning rate
    model_configuration["Data"]["data_augmentation"]["activated"] = True
    experiment_configuration["batch_size"] = 64
    experiment_configuration["learning_rate"] = 1.0e-3
    return experiment_configuration, model_configuration


def use_max_pooling_and_two_hidden_layers(experiment_configuration: Dict[str, Any],
                                          model_configuration: Dict[str, Any]) -> None:
    """
    Pools the final feature map with its maximum, and adds a second hidden layer to the head.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["ClassificationHead"].update({"global_pooling": "max", "fully_connected": [256, 64]})


def remove_hidden_layers_and_dropout(experiment_configuration: Dict[str, Any],
                                     model_configuration: Dict[str, Any]) -> None:
    """
    Classifies the pooled features directly, without hidden layer nor dropout.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["ClassificationHead"].update({"fully_connected": [], "dropout_rate": 0.0})


def smooth_labels_and_balance_classes(experiment_configuration: Dict[str, Any],
                                      model_configuration: Dict[str, Any]) -> None:
    """
    Smooths the labels of the loss and balances its classes.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["ClassificationHead"].update({"label_smoothing": 0.1, "class_weighting": "balanced"})


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
    (use_max_pooling_and_two_hidden_layers, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_HAVG.128-", "_HMAX.256.64-")),
    (remove_hidden_layers_and_dropout, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_HAVG.128-DO0.2_", "_HAVG-DO0_")),
    (smooth_labels_and_balance_classes, CLASSIFICATION_REFERENCE_RUN_NAME.replace("-DO0.2_", "-DO0.2-LS0.1-CWB_")),
    (use_multi_label_outputs, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_MC4_", "_ML5_"))])
def test_classification_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                                  expected_run_name: str) -> None:
    """
    Verifies the run names of the classification task: the CLS task code, the shared input and backbone sections, the
    classification head and classes sections, and that changing a head setting changes its token.

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
SEQUENCE_REFERENCE_RUN_NAME = ("M2O_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128-BI_HAVG.64-DO0.2_MC4_"
                               "AUG-BS16-LR1e-3")


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

    # Task, input and backbone of the run
    experiment_configuration["task"] = "sequence_classification"
    model_configuration["Data"]["image_settings"].update({"size": 128, "keep_ratio": True})
    model_configuration["Backbone"]["normalization"].update({"input_normalization": True,
                                                             "per_sample_normalization": False,
                                                             "feature_map_normalization": "batch"})
    model_configuration["Backbone"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False})
    model_configuration["Backbone"]["convolutions"] = {"1": [1, 16], "2": [2, 32], "3": [2, 32], "4": [2, 64],
                                                       "5": [2, 64]}

    # Windows, temporal module, classification head and classes
    model_configuration["Data"]["sequence_settings"] = {"sequence_length": 8, "frame_stride": 1, "window_stride": 8,
                                                        "short_sequences": "pad"}
    model_configuration["TemporalModel"] = {"type": "gru", "hidden_size": 128, "number_layers": 1, "number_heads": 4,
                                            "kernel_size": 3, "dropout_rate": 0.0, "bidirectional": True,
                                            "causal": False, "pretrained": dict(NO_TRANSFER_SETTINGS)}
    model_configuration["ClassificationHead"] = {"global_pooling": "average", "fully_connected": [64],
                                                 "dropout_rate": 0.2, "output_mode": "multi_class",
                                                 "label_smoothing": 0.0, "class_weighting": "none",
                                                 "pretrained": dict(NO_TRANSFER_SETTINGS)}
    model_configuration["Data"]["label_dictionary"] = {1: 0, 2: 1, 3: 2, 4: 3}

    # Augmentation, batch size and learning rate
    model_configuration["Data"]["data_augmentation"]["activated"] = True
    experiment_configuration["batch_size"] = 16
    experiment_configuration["learning_rate"] = 1.0e-3
    return experiment_configuration, model_configuration


def skip_frames(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Cuts windows of 16 frames, keeping one frame every two.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["sequence_settings"].update({"sequence_length": 16, "frame_stride": 2})


def use_deep_unidirectional_lstm(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Uses a unidirectional LSTM of 3 layers of 64 units, with a dropout between its layers.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["TemporalModel"].update({"type": "lstm", "hidden_size": 64, "number_layers": 3,
                                                 "dropout_rate": 0.1, "bidirectional": False})


def use_causal_transformer(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Uses a causal transformer of 2 layers with 8 heads.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["TemporalModel"].update({"type": "transformer", "number_layers": 2, "number_heads": 8,
                                                 "bidirectional": False, "causal": True})


def use_temporal_convolution(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Uses a temporal convolution of 4 layers with kernels of 5 frames.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["TemporalModel"].update({"type": "temporal_convolution", "number_layers": 4,
                                                 "kernel_size": 5})


def use_attention_pooling(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Uses the attention pooling, whose only setting is the hidden size of its scoring network (its dropout is unused).

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["TemporalModel"].update({"type": "attention_pooling", "dropout_rate": 0.3})


def use_mean_pooling(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Uses the mean pooling, which has no setting.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["TemporalModel"]["type"] = "mean_pooling"


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (None, SEQUENCE_REFERENCE_RUN_NAME),
    (skip_frames, SEQUENCE_REFERENCE_RUN_NAME.replace("_T8_", "_T16S2_")),
    (use_deep_unidirectional_lstm, SEQUENCE_REFERENCE_RUN_NAME.replace("_GRU1x128-BI_", "_LSTM3x64-TDO0.1_")),
    (use_causal_transformer, SEQUENCE_REFERENCE_RUN_NAME.replace("_GRU1x128-BI_", "_TRF2x128.8-CAU_")),
    (use_temporal_convolution, SEQUENCE_REFERENCE_RUN_NAME.replace("_GRU1x128-BI_", "_TCN4x128.5_")),
    (use_attention_pooling, SEQUENCE_REFERENCE_RUN_NAME.replace("_GRU1x128-BI_", "_ATT128_")),
    (use_mean_pooling, SEQUENCE_REFERENCE_RUN_NAME.replace("_GRU1x128-BI_", "_MEAN_"))])
def test_sequence_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                            expected_run_name: str) -> None:
    """
    Verifies the run names of the sequence classification task: the M2O task code, the window section, the temporal
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


# Run name of the sequence comparison reference settings set by get_sequence_comparison_reference_configurations
SEQUENCE_COMPARISON_REFERENCE_RUN_NAME = ("CMP_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_MEAN_FDIF_HAVG.64-"
                                          "DO0.2_MC2_AUG-BS16-LR1e-3")


def get_sequence_comparison_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds the sequence comparison reference settings from the sequence reference settings: the comparison task,
    member windows without window stride, a mean pooling, the difference fusion and 2 classes.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    experiment_configuration["task"] = "sequence_comparison"
    del model_configuration["Data"]["sequence_settings"]["window_stride"]
    model_configuration["TemporalModel"]["type"] = "mean_pooling"
    model_configuration["Fusion"] = {"type": "difference", "number_heads": 4, "dropout_rate": 0.0,
                                     "sequence_normalization_scope": "member"}
    model_configuration["Data"]["label_dictionary"] = {1: 0, 2: 1}
    return experiment_configuration, model_configuration


def use_concatenation(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Concatenates the member vectors without their difference.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Fusion"]["type"] = "concatenation"


def use_cross_attention(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Fuses the members with the cross attention of 4 heads, without dropout.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Fusion"]["type"] = "cross_attention"


def use_cross_attention_with_dropout(experiment_configuration: Dict[str, Any],
                                     model_configuration: Dict[str, Any]) -> None:
    """
    Fuses the members with the cross attention of 4 heads, with a dropout of its attention weights.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Fusion"].update({"type": "cross_attention", "dropout_rate": 0.1})


def skip_member_frames(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Keeps one frame every two of every member.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["sequence_settings"]["frame_stride"] = 2


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (None, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME),
    (use_concatenation, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("_FDIF_", "_FCAT_")),
    (use_cross_attention, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("_FDIF_", "_FXA4_")),
    (use_cross_attention_with_dropout, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("_FDIF_", "_FXA4-FDO0.1_")),
    (skip_member_frames, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("_T8_", "_T8S2_"))])
def test_sequence_comparison_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                                       expected_run_name: str) -> None:
    """
    Verifies the run names of the sequence comparison task: the CMP task code, the member window section, the
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
                         argvalues=[(False, 1.0, False, "-PT_"), (True, 1.0, False, "-PT-FRZ_"),
                                    (False, 0.1, False, "-PT-BLR0.1_"), (True, 0.1, False, "-PT-FRZ_"),
                                    (False, 1.0, True, "-PT-FBN_"), (False, 0.1, True, "-PT-BLR0.1-FBN_")])
def test_pretrained_backbone_run_names(freeze: bool, learning_rate_multiplier: float, freeze_normalization: bool,
                                       expected_tokens: str) -> None:
    """
    Verifies the tokens of a transferred backbone at the end of the backbone section: PT, then FRZ when it is frozen
    or BLR<multiplier> when it learns at another speed than the heads (a frozen backbone does not learn at all), then
    FBN when the scale and shift of its normalization layers are kept fixed.

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
                         argvalues=[("TemporalModel", False, 1.0, "_GRU1x128-BI-TPT_"),
                                    ("TemporalModel", True, 1.0, "_GRU1x128-BI-TPT-TFRZ_"),
                                    ("TemporalModel", False, 0.5, "_GRU1x128-BI-TPT-TLR0.5_"),
                                    ("ClassificationHead", False, 1.0, "_HAVG.64-DO0.2-HPT_"),
                                    ("ClassificationHead", True, 0.1, "_HAVG.64-DO0.2-HPT-HFRZ_"),
                                    ("ClassificationHead", False, 0.1, "_HAVG.64-DO0.2-HPT-HLR0.1_")])
def test_pretrained_temporal_module_and_head_run_names(section_name: str, freeze: bool,
                                                       learning_rate_multiplier: float, expected_section: str) -> None:
    """
    Verifies the tokens of a transferred temporal module or classification head at the end of its section: TPT or
    HPT, then TFRZ or HFRZ when it is frozen, or TLR<multiplier> or HLR<multiplier> when it learns at its own speed.
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
    original_section = "_GRU1x128-BI_" if section_name == "TemporalModel" else "_HAVG.64-DO0.2_"
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SEQUENCE_REFERENCE_RUN_NAME.replace(
        original_section, expected_section)


@pytest.mark.parametrize(argnames=("augmentation_activated", "shuffle_frames", "independent_frames",
                                   "expected_tokens"), argvalues=[(True, True, False, "_AUG-SHUF-BS16-"),
                                                                  (True, False, True, "_AUG-IND-BS16-"),
                                                                  (True, True, True, "_AUG-SHUF-IND-BS16-"),
                                                                  (False, True, True, "_BS16-")])
def test_sequence_augmentation_run_name_tokens(augmentation_activated: bool, shuffle_frames: bool,
                                               independent_frames: bool, expected_tokens: str) -> None:
    """
    Verifies the tokens of the sequence augmentations after AUG in the training section: SHUF when the frames of the
    windows are shuffled, IND when the photometric augmentations are drawn frame by frame, neither when the
    augmentation is off.

    Args:
        augmentation_activated (bool): Whether the data augmentation is on.
        shuffle_frames (bool): Whether the frames of the windows are shuffled.
        independent_frames (bool): Whether the photometric augmentations are drawn frame by frame.
        expected_tokens (str): The expected start of the training section.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    model_configuration["Data"]["data_augmentation"].update({
        "activated": augmentation_activated, "independent_frames": independent_frames,
        "shuffle_frames": {"activated": shuffle_frames, "probability": 0.5}})
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SEQUENCE_REFERENCE_RUN_NAME.replace(
        "_AUG-BS16-", expected_tokens)


@pytest.mark.parametrize(argnames=("feature_map_normalization", "normalization_scope", "expected_sections"),
                         argvalues=[("sequence", "member", ("-SN-B1x16", "_FDIF_")),
                                    ("sequence", "pair", ("-SN-B1x16", "_FDIF-SNP_")),
                                    ("batch", "pair", ("-BN-B1x16", "_FDIF_"))])
def test_sequence_normalization_run_name_tokens(feature_map_normalization: str, normalization_scope: str,
                                                expected_sections: Tuple[str, str]) -> None:
    """
    Verifies the tokens of the sequence normalization of a comparison model: SN instead of BN in the backbone section,
    and SNP in the fusion section when both members of a pair share the statistics, the scope being ignored with
    batch normalization.

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
    Verifies the optimization tokens at the end of the training section, written only for the settings that differ
    from plain Adam at a constant learning rate.
    """
    experiment_configuration, model_configuration = get_classification_reference_configurations()
    model_configuration["Optimization"] = {"optimizer": "adamw", "weight_decay": 1e-4,
                                           "learning_rate_schedule": "cosine", "warmup_iterations": 500,
                                           "decay_iterations": 20000, "minimum_learning_rate_ratio": 0.01,
                                           "gradient_clipping_norm": 1.0, "mixed_precision": True}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == CLASSIFICATION_REFERENCE_RUN_NAME.replace(
        "_AUG-BS64-LR1e-3", "_AUG-BS64-LR1e-3-ADAMW-WD1e-4-WU500-COS20000-GC1-BF16")
