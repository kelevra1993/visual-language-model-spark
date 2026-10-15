"""
Tests of build_run_name of vision_training/configuration_processing/run_name.py on the reference configurations of every
task: the reference run names are fixed, every changed setting (architecture, inputs, heads, temporal modules, fusions,
segmentation decoders and losses, transfers, normalization, optimizer, weight decay) changes the run name with its
token, and the training settings that can change between two resumptions (augmentation, batch size, learning rate,
schedule, clipping, mixed precision, moving average of the weights, dropout rates, segmentation
losses) keep it.
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
REFERENCE_RUN_NAME = ("DET_IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_"
                      "DSE-DRES-DBN-D7.2x64.256.128_BIN")

CLASSIFICATION_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" /
                                     "classification" / "configuration_synthetic_shapes.yaml")

# Run name of the classification reference settings set by get_classification_reference_configurations
CLASSIFICATION_REFERENCE_RUN_NAME = "CLS_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC4"

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

    # Backbone modules and blocks
    model_configuration["Backbone"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False, "inception_starting_block": 3})
    model_configuration["Backbone"]["convolutions"] = {"1": [1, 16], "2": [2, 32], "3": [2, 32], "4": [4, 64],
                                                       "5": [4, 64], "6": [6, 128], "7": [6, 128]}

    # Detection levels 5 to 7, whose anchors, region proposal heads and detector heads are identical on every level
    # except for the anchor scales
    model_configuration["DetectionLevels"] = {
        level: {"anchors": {"scales": scales, "aspect_ratios": [0.5, 1.0, 2.0]},
                "region_proposal": {"convolutions": [2, 64]},
                "detector": {"roi_align_pool_size": 7, "convolutions": [2, 64], "fully_connected": [256, 128]}}
        for level, scales in [("5", [32, 64, 128]), ("6", [64, 128, 256]), ("7", [128, 256, 512])]}
    model_configuration["RegionProposal"]["normalization"]["feature_map_normalization"] = "batch"
    model_configuration["Detector"]["modules"].update({"squeeze_and_excite": True, "residual_nets": True,
                                                       "inception": False})
    model_configuration["Detector"]["normalization"]["feature_map_normalization"] = "batch"

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


def use_pre_activation_shortcuts(experiment_configuration: Dict[str, Any],
                                 model_configuration: Dict[str, Any]) -> None:
    """
    Places the residual shortcuts of the backbone before the activations (residual_nets_v2).

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["modules"]["residual_nets_v2"] = True


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
    del model_configuration["DetectionLevels"]["5"]


def differ_heads_per_level(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Gives level 5 four anchor scales and a larger detector head, and level 7 a deeper and wider region proposal
    head, so that the per-level settings are listed level by level.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    detection_levels = model_configuration["DetectionLevels"]
    detection_levels["5"]["anchors"]["scales"] = [16, 32, 64, 128]
    detection_levels["7"]["region_proposal"]["convolutions"] = [3, 128]
    detection_levels["5"]["detector"] = {"roi_align_pool_size": 14, "convolutions": [2, 64], "fully_connected": [512]}


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
    Changes every training setting that can change between two resumptions of a run: turns off data augmentation,
    and trains with another batch size, learning rate and schedule, gradient clipping, mixed precision and a moving
    average of the weights.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["data_augmentation"]["activated"] = False
    experiment_configuration["batch_size"] = 16
    experiment_configuration["learning_rate"] = 1.5e-3
    model_configuration["Optimization"].update({"learning_rate_schedule": "cosine", "warmup_iterations": 500,
                                                "decay_iterations": 20000, "gradient_clipping_norm": 1.0,
                                                "mixed_precision": True,
                                                "exponential_moving_average": {"activated": True, "decay": 0.999}})


def test_reference_run_name() -> None:
    """
    Verifies the run name of the reference settings, that the same settings always give the same name, which lets a
    run resume in its own folder, and that the training settings that can change between two resumptions (augmentation,
    batch size, learning rate and schedule, gradient clipping, mixed precision, moving average of the weights) keep
    that name.
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


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (disable_squeeze_and_excite, REFERENCE_RUN_NAME.replace("_SE-RES-BN-", "_RES-BN-")),
    (use_pre_activation_shortcuts, REFERENCE_RUN_NAME.replace("_SE-RES-BN-", "_SE-RES2-BN-")),
    (enable_inception, REFERENCE_RUN_NAME.replace("_SE-RES-BN-", "_SE-RES-INC3-BN-").replace(
        "DSE-DRES-DBN", "DSE-DRES-DINC-DBN")),
    (change_input_settings, REFERENCE_RUN_NAME.replace("_IN-KAR-1024_", "_PSN-1024_")),
    (change_feature_map_normalizations, REFERENCE_RUN_NAME.replace("-BN-B", "-LN-B").replace(
        "RBN", "RNN").replace("DBN", "DLN")),
    (widen_last_block, REFERENCE_RUN_NAME.replace("6x128.6x128", "6x128.6x256")),
    (change_detection_levels, REFERENCE_RUN_NAME.replace("_L567_A9.32-512_", "_L67_A9.64-512_")),
    (differ_heads_per_level, REFERENCE_RUN_NAME.replace("A9.32-512", "A12+9+9.16-512").replace(
        "R2x64", "R2x64+2x64+3x128").replace("D7.2x64.256.128", "D14.2x64.512+7.2x64.256.128+7.2x64.256.128")),
    (use_multi_class_labels, REFERENCE_RUN_NAME.replace("_BIN", "_MC4"))])
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
                                                 "fully_connected_normalization": "none",
                                                 "batch_normalization_momentum": 0.1,
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


def flatten_final_feature_map(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Gives the head all the values of the final feature map instead of their average.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["ClassificationHead"]["global_pooling"] = "flatten"


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


def normalize_hidden_layers_and_balance_classes(experiment_configuration: Dict[str, Any],
                                                model_configuration: Dict[str, Any]) -> None:
    """
    Normalizes the outputs of the hidden layer of the head with batch normalization, and balances the classes of the
    loss, whose token follows.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["ClassificationHead"].update({"fully_connected_normalization": "batch",
                                                      "class_weighting": "balanced"})


def layer_normalize_hidden_layers(experiment_configuration: Dict[str, Any],
                                  model_configuration: Dict[str, Any]) -> None:
    """
    Normalizes the outputs of the hidden layer of the head with layer normalization.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["ClassificationHead"]["fully_connected_normalization"] = "layer"


def change_batch_normalization_momentums(experiment_configuration: Dict[str, Any],
                                         model_configuration: Dict[str, Any]) -> None:
    """
    Changes the batch normalization momentum of the backbone and of the normalized hidden layers of the head, which
    changes no weight.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["normalization"]["batch_normalization_momentum"] = 0.01
    model_configuration["ClassificationHead"].update({"fully_connected_normalization": "batch",
                                                      "batch_normalization_momentum": 0.01})


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
    (use_max_pooling_and_two_hidden_layers, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_HAVG.128_", "_HMAX.256.64_")),
    (flatten_final_feature_map, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_HAVG.128_", "_HFLAT.128_")),
    (remove_hidden_layers_and_dropout, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_HAVG.128_", "_HAVG_")),
    (smooth_labels_and_balance_classes, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_HAVG.128_",
                                                                                  "_HAVG.128-LS0.1-CWB_")),
    (normalize_hidden_layers_and_balance_classes, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_HAVG.128_",
                                                                                            "_HAVG.128-HBN-CWB_")),
    (layer_normalize_hidden_layers, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_HAVG.128_", "_HAVG.128-HLN_")),
    (change_batch_normalization_momentums, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_HAVG.128_",
                                                                                     "_HAVG.128-HBN_")),
    (use_multi_label_outputs, CLASSIFICATION_REFERENCE_RUN_NAME.replace("_MC4", "_ML5"))])
def test_classification_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                                  expected_run_name: str) -> None:
    """
    Verifies the run names of the classification task: the CLS task code, the shared input and backbone sections, the
    classification head and classes sections, that changing a head setting changes its token, and that the batch
    normalization momentums give none.

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
    model_configuration["Data"]["sequence_settings"] = {"sequence_length": 8, "window_stride": 8,
                                                        "short_sequences": "pad"}
    model_configuration["TemporalModel"] = {"type": "gru", "hidden_size": 128, "number_layers": 1, "number_heads": 4,
                                            "kernel_size": 3, "dropout_rate": 0.0, "bidirectional": True,
                                            "causal": False, "positional_embedding": True,
                                            "pretrained": dict(NO_TRANSFER_SETTINGS)}
    model_configuration["ClassificationHead"] = {"global_pooling": "average", "fully_connected": [64],
                                                 "fully_connected_normalization": "none",
                                                 "batch_normalization_momentum": 0.1,
                                                 "dropout_rate": 0.2, "output_mode": "multi_class",
                                                 "label_smoothing": 0.0, "class_weighting": "none",
                                                 "pretrained": dict(NO_TRANSFER_SETTINGS)}
    model_configuration["Data"]["label_dictionary"] = {1: 0, 2: 1, 3: 2, 4: 3}

    # Augmentation, batch size and learning rate
    model_configuration["Data"]["data_augmentation"]["activated"] = True
    experiment_configuration["batch_size"] = 16
    experiment_configuration["learning_rate"] = 1.0e-3
    return experiment_configuration, model_configuration


def use_longer_windows(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Cuts windows of 16 frames.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["sequence_settings"]["sequence_length"] = 16


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


def use_order_blind_transformer(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Uses a transformer of 2 layers with 8 heads without positional embedding, treating a window as a set of frames.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["TemporalModel"].update({"type": "transformer", "number_layers": 2, "number_heads": 8,
                                                 "bidirectional": False, "positional_embedding": False})


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
    (use_longer_windows, SEQUENCE_REFERENCE_RUN_NAME.replace("_T8_", "_T16_")),
    (use_deep_unidirectional_lstm, SEQUENCE_REFERENCE_RUN_NAME.replace("_GRU1x128-BI_", "_LSTM3x64_")),
    (use_causal_transformer, SEQUENCE_REFERENCE_RUN_NAME.replace("_GRU1x128-BI_", "_TRF2x128.8-CAU_")),
    (use_order_blind_transformer, SEQUENCE_REFERENCE_RUN_NAME.replace("_GRU1x128-BI_", "_TRF2x128.8-NPE_")),
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


@pytest.mark.parametrize(argnames=("frames_per_class", "selection_loss_weight", "decoding", "expected_section"),
                         argvalues=[(1, 1.0, "hungarian", "_FPC1-SLW1"), (2, 0.5, "hungarian", "_FPC2-SLW0.5"),
                                    (1, 0.0, "hungarian", "_FPC1"), (1, 1.0, "greedy", "_FPC1-SLW1")])
def test_frame_selection_run_names(frames_per_class: int, selection_loss_weight: float, decoding: str,
                                   expected_section: str) -> None:
    """
    Verifies the run names of the frame selection task: the SEL task code, the sections of the sequence tasks, then
    the selection section with the frames per class and the weight of the selection loss when it is used, the
    decoding giving no token since both decodings are scored at every evaluation.

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


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (None, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME),
    (use_concatenation, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("_FDIF_", "_FCAT_")),
    (use_cross_attention, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("_FDIF_", "_FXA4_")),
    (use_cross_attention_with_dropout, SEQUENCE_COMPARISON_REFERENCE_RUN_NAME.replace("_FDIF_", "_FXA4_"))])
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
                                    (False, 0.1, False, "-PT_"), (True, 0.1, False, "-PT-FRZ_"),
                                    (False, 1.0, True, "-PT-FBN_"), (False, 0.1, True, "-PT-FBN_")])
def test_pretrained_backbone_run_names(freeze: bool, learning_rate_multiplier: float, freeze_normalization: bool,
                                       expected_tokens: str) -> None:
    """
    Verifies the tokens of a transferred backbone at the end of the backbone section: PT, then FRZ when it is frozen,
    then FBN when the scale and shift of its normalization layers are kept fixed. The learning rate multiplier gives
    no token, so that it can change between two resumptions of a run.

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


@pytest.mark.parametrize(argnames=("section_name", "freeze", "learning_rate_multiplier", "new_class_initialisation",
                                   "expected_section"),
                         argvalues=[("TemporalModel", False, 1.0, None, "_GRU1x128-BI-TPT_"),
                                    ("TemporalModel", True, 1.0, None, "_GRU1x128-BI-TPT-TFRZ_"),
                                    ("TemporalModel", False, 0.5, None, "_GRU1x128-BI-TPT_"),
                                    ("ClassificationHead", False, 1.0, "source", "_HAVG.64-HPT_"),
                                    ("ClassificationHead", True, 0.1, "source", "_HAVG.64-HPT-HFRZ_"),
                                    ("ClassificationHead", False, 0.1, "source", "_HAVG.64-HPT_"),
                                    ("ClassificationHead", False, 1.0, "random", "_HAVG.64-HPT-HRND_"),
                                    ("ClassificationHead", True, 1.0, "random", "_HAVG.64-HPT-HRND-HFRZ_")])
def test_pretrained_temporal_module_and_head_run_names(section_name: str, freeze: bool, learning_rate_multiplier: float,
                                                       new_class_initialisation: str | None,
                                                       expected_section: str) -> None:
    """
    Verifies the tokens of a transferred temporal module or classification head at the end of its section: TPT or
    HPT, then HRND when the classes added after those of the source run start random, then TFRZ or HFRZ when it is
    frozen, the learning rate multiplier giving no token. The rest of the name is unchanged.

    Args:
        section_name (str): The section of the transferred component.
        freeze (bool): Whether the transferred component is frozen.
        learning_rate_multiplier (float): The learning rate multiplier of the transferred component.
        new_class_initialisation (str | None): The initialisation of the new classes of the classification head, None
            for the temporal module, which has no class layer.
        expected_section (str): The expected section of the component in the run name.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    model_configuration[section_name]["pretrained"] = {"experiment_folder": "/runs/source", "iteration": None,
                                                       "freeze": freeze,
                                                       "learning_rate_multiplier": learning_rate_multiplier}
    if new_class_initialisation is not None:
        model_configuration[section_name]["pretrained"]["new_class_initialisation"] = new_class_initialisation
    original_section = "_GRU1x128-BI_" if section_name == "TemporalModel" else "_HAVG.64_"
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SEQUENCE_REFERENCE_RUN_NAME.replace(
        original_section, expected_section)


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


@pytest.mark.parametrize(argnames=("augmentation_activated", "shuffle_frames", "independent_frames"),
                         argvalues=[(True, True, False), (True, False, True), (True, True, True), (False, True, True)])
def test_sequence_augmentation_keeps_run_name(augmentation_activated: bool, shuffle_frames: bool,
                                              independent_frames: bool) -> None:
    """
    Verifies that the sequence augmentations (frames of the windows shuffled, photometric augmentations drawn frame by
    frame) give no token, so that a sequence run can be resumed in its folder with other augmentation settings.

    Args:
        augmentation_activated (bool): Whether the data augmentation is on.
        shuffle_frames (bool): Whether the frames of the windows are shuffled.
        independent_frames (bool): Whether the photometric augmentations are drawn frame by frame.
    """
    experiment_configuration, model_configuration = get_sequence_reference_configurations()
    model_configuration["Data"]["data_augmentation"].update({
        "activated": augmentation_activated, "independent_frames": independent_frames,
        "shuffle_frames": {"activated": shuffle_frames, "probability": 0.5}})
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == SEQUENCE_REFERENCE_RUN_NAME


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
    Verifies the optimizer section at the end of the run name, written only for an optimizer other than Adam or a
    weight decay, the schedule, gradient clipping and mixed precision giving no token.
    """
    experiment_configuration, model_configuration = get_classification_reference_configurations()
    model_configuration["Optimization"] = {"optimizer": "adamw", "weight_decay": 1e-4,
                                           "learning_rate_schedule": "cosine", "warmup_iterations": 500,
                                           "decay_iterations": 20000, "minimum_learning_rate_ratio": 0.01,
                                           "gradient_clipping_norm": 1.0, "mixed_precision": True}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == CLASSIFICATION_REFERENCE_RUN_NAME.replace(
        "_MC4", "_MC4_ADAMW-WD1e-4")


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

    # Task, input and backbone of the run
    experiment_configuration["task"] = "semantic_segmentation"
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
                                                  "feature_map_normalization": "batch",
                                                  "batch_normalization_momentum": 0.1, "dropout_rate": 0.0,
                                                  "deep_supervision_weights": {},
                                                  "pretrained": {**NO_TRANSFER_SETTINGS, "load_class_layers": False}}
    model_configuration["SegmentationLoss"] = {"cross_entropy_weight": 1.0, "focal_gamma": 0.0,
                                               "label_smoothing": 0.0, "class_weighting": "none", "dice_weight": 1.0,
                                               "lovasz_weight": 0.0, "boundary_weight": 0.0, "boundary_distance": 3}
    model_configuration["Data"]["label_mode"] = "multi_class"
    model_configuration["Data"]["label_dictionary"] = {1: 1, 2: 2, 3: 3, 4: 4, "background": 0}
    return experiment_configuration, model_configuration


def use_bilinear_upsampling_and_decoder_modules(experiment_configuration: Dict[str, Any],
                                                model_configuration: Dict[str, Any]) -> None:
    """
    Upsamples the decoder levels bilinearly, with Squeeze-and-Excitation, residual shortcuts and layer normalization
    in the decoder blocks.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["SegmentationDecoder"].update({"upsampling": "bilinear", "feature_map_normalization": "layer",
                                                       "modules": {"squeeze_and_excite": True,
                                                                   "residual_nets": True}})


def add_last_block_decoder_level(experiment_configuration: Dict[str, Any],
                                 model_configuration: Dict[str, Any]) -> None:
    """
    Pools the last backbone block, whose skip connection gets a decoder level, written first as the deepest one.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Backbone"]["last_max_pooling"] = True
    model_configuration["SegmentationDecoder"]["convolutions"]["6"] = [3, 128]


def use_cross_entropy_only(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Trains with the plain cross entropy only.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["SegmentationLoss"]["dice_weight"] = 0.0


def use_dice_only(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Trains with the Dice loss only, with settings of the unused cross entropy.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["SegmentationLoss"].update({"cross_entropy_weight": 0.0, "label_smoothing": 0.1,
                                                    "class_weighting": "balanced", "focal_gamma": 2.0})


def use_weighted_focal_cross_entropy(experiment_configuration: Dict[str, Any],
                                     model_configuration: Dict[str, Any]) -> None:
    """
    Weighs the smoothed, class-balanced focal cross entropy twice, and the Dice loss half.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["SegmentationLoss"].update({"cross_entropy_weight": 2.0, "label_smoothing": 0.1,
                                                    "class_weighting": "balanced", "focal_gamma": 2.0,
                                                    "dice_weight": 0.5})


def use_lovasz_and_boundary_losses(experiment_configuration: Dict[str, Any],
                                   model_configuration: Dict[str, Any]) -> None:
    """
    Adds the Lovasz-Softmax and boundary losses, and evaluates at the original resolution with a wider boundary
    tolerance.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["SegmentationLoss"].update({"lovasz_weight": 0.5, "boundary_weight": 1.0,
                                                    "boundary_distance": 5})
    experiment_configuration["evaluation"].update({"original_resolution": True, "boundary_tolerance": 4})


def add_deep_supervision(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Puts deep supervision heads on the decoder levels of the blocks 3 and 5.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["SegmentationDecoder"]["deep_supervision_weights"] = {"3": 0.3, "5": 0.1}


def train_on_half_scale_tiles(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Trains on tiles of the images scaled by one half, with evaluation tiles overlapping by 64 pixels.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["tiling"] = {"activated": True, "scale": 0.5, "overlap": 64}


def use_binary_segmentation_labels(experiment_configuration: Dict[str, Any],
                                   model_configuration: Dict[str, Any]) -> None:
    """
    Segments the background and the foreground only.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["Data"]["label_mode"] = "binary"
    model_configuration["Data"]["label_dictionary"] = {"background": 0, "foreground": 1}


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (None, SEGMENTATION_REFERENCE_RUN_NAME),
    (use_bilinear_upsampling_and_decoder_modules, SEGMENTATION_REFERENCE_RUN_NAME.replace("_UTC-DBN-",
                                                                                          "_UBL-DSE-DRES-DLN-")),
    (add_last_block_decoder_level, SEGMENTATION_REFERENCE_RUN_NAME.replace("-D2x64.2x64.", "-D3x128.2x64.2x64.")),
    (use_cross_entropy_only, SEGMENTATION_REFERENCE_RUN_NAME),
    (use_dice_only, SEGMENTATION_REFERENCE_RUN_NAME),
    (use_weighted_focal_cross_entropy, SEGMENTATION_REFERENCE_RUN_NAME),
    (use_lovasz_and_boundary_losses, SEGMENTATION_REFERENCE_RUN_NAME),
    (add_deep_supervision, SEGMENTATION_REFERENCE_RUN_NAME.replace("1x16_", "1x16-AUX5.3_")),
    (train_on_half_scale_tiles, SEGMENTATION_REFERENCE_RUN_NAME.replace("_UTC-", "_TL0.5_UTC-")),
    (use_binary_segmentation_labels, SEGMENTATION_REFERENCE_RUN_NAME.replace("_MC4", "_BIN"))])
def test_segmentation_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                                expected_run_name: str) -> None:
    """
    Verifies the run names of the semantic segmentation task: the SEG task code, the shared input and backbone
    sections, the tiling section with the scale of the tiles, the decoder section with its levels from the deepest
    one and the levels of its deep supervision heads, and the classes section counting the foreground classes, the
    loss and evaluation settings giving no token so that they can change between two resumptions of a run.

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


@pytest.mark.parametrize(argnames=("freeze", "load_class_layers", "learning_rate_multiplier",
                                   "new_class_initialisation", "expected_tokens"),
                         argvalues=[(False, False, 1.0, "source", "-DPT_"), (True, False, 1.0, "source", "-DPT-DFRZ_"),
                                    (False, False, 0.1, "source", "-DPT_"), (False, True, 1.0, "source", "-DPT-DCL_"),
                                    (True, True, 1.0, "source", "-DPT-DCL-DFRZ_"),
                                    (False, False, 1.0, "random", "-DPT_"),
                                    (False, True, 1.0, "random", "-DPT-DCL-DRND_"),
                                    (True, True, 1.0, "random", "-DPT-DCL-DRND-DFRZ_")])
def test_pretrained_segmentation_decoder_run_names(freeze: bool, load_class_layers: bool,
                                                   learning_rate_multiplier: float, new_class_initialisation: str,
                                                   expected_tokens: str) -> None:
    """
    Verifies the tokens of a transferred segmentation decoder at the end of the decoder section: DPT, then DCL when
    its class layers are loaded too, then DRND when these class layers start their new classes random, then DFRZ when
    it is frozen, and that the learning rate multiplier of the decoder, the dropout rate of the segmentation head and
    the initialisation of the new classes of a decoder that does not load its class layers give no token.

    Args:
        freeze (bool): Whether the transferred decoder is frozen.
        load_class_layers (bool): Whether the segmentation head and deep supervision heads are loaded with the decoder.
        learning_rate_multiplier (float): The learning rate multiplier of the transferred decoder.
        new_class_initialisation (str): The initialisation of the classes added after those of the source run.
        expected_tokens (str): The expected end of the decoder section.
    """
    experiment_configuration, model_configuration = get_segmentation_reference_configurations()
    model_configuration["SegmentationDecoder"].update({"dropout_rate": 0.3,
                                                       "pretrained": {"experiment_folder": "/runs/source",
                                                                      "iteration": None, "freeze": freeze,
                                                                      "load_class_layers": load_class_layers,
                                                                      "learning_rate_multiplier":
                                                                          learning_rate_multiplier,
                                                                      "new_class_initialisation":
                                                                          new_class_initialisation}})
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
    model_configuration["MaskHead"] = {"mask_resolution": 28, "upsampling": "transposed",
                                       "normalization": {"feature_map_normalization": "batch",
                                                         "batch_normalization_momentum": 0.1},
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


def change_mask_threshold(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Changes the mask threshold of the inference, which gives no token.

    Args:
        experiment_configuration (Dict[str, Any]): The experiment configuration, modified in place.
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    model_configuration["MaskHead"]["inference"]["mask_threshold"] = 0.3


# Run name of the instance segmentation reference settings: the detection sections with the mask heads before the
# classes
INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME = REFERENCE_RUN_NAME.replace("DET_", "INS_").replace(
    "_BIN", "_MRES-MBN-MUTC-M14.4x64-MR28_BIN")


@pytest.mark.parametrize(argnames=("modify_function", "expected_run_name"), argvalues=[
    (None, INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME),
    (change_mask_heads, INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME.replace(
        "_MRES-MBN-MUTC-M14.4x64-MR28_", "_MSE-MRES-MINC-MLN-MUBL-M14.4x64+14.4x64+7.2x128-MR56_")),
    (change_mask_threshold, INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME)])
def test_instance_segmentation_run_names(modify_function: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
                                         expected_run_name: str) -> None:
    """
    Verifies the run names of the instance segmentation: the INS code, the detection sections, and the mask heads
    section (modules, normalization, upsampling, pool size and convolutions of every level, mask resolution) before
    the classes, the mask threshold giving no token.

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


@pytest.mark.parametrize(argnames=("region_proposal_freeze", "detector_freeze", "new_class_initialisation",
                                   "expected_sections"), argvalues=[
    (None, False, "source", "_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128-DPT_MRES"),
    (False, None, "source", "_RBN-R2x64-RPT_DSE-DRES-DBN-D7.2x64.256.128_MRES"),
    (True, True, "source", "_RBN-R2x64-RPT-RFRZ_DSE-DRES-DBN-D7.2x64.256.128-DPT-DFRZ_MRES"),
    (None, False, "random", "_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128-DPT-DRND_MRES"),
    (True, True, "random", "_RBN-R2x64-RPT-RFRZ_DSE-DRES-DBN-D7.2x64.256.128-DPT-DRND-DFRZ_MRES")])
def test_pretrained_detection_head_run_names(region_proposal_freeze: bool | None, detector_freeze: bool | None,
                                             new_class_initialisation: str, expected_sections: str) -> None:
    """
    Verifies the tokens of transferred region proposal and detection heads at the end of their sections, before the
    mask heads of an instance segmentation run: RPT and DPT, then DRND when the classes the detection heads add after
    those of the source run start random (the region proposal heads have no class layer), then RFRZ and DFRZ when
    they are frozen, the learning rate multiplier giving no token.

    Args:
        region_proposal_freeze (bool | None): Whether the transferred region proposal heads are frozen, None when they
            are not transferred.
        detector_freeze (bool | None): Whether the transferred detection heads are frozen, None when they are not
            transferred.
        new_class_initialisation (str): The initialisation of the classes added after those of the source run.
        expected_sections (str): The expected region proposal and detector sections of the run name.
    """
    experiment_configuration, model_configuration = get_instance_segmentation_reference_configurations()
    for section_name, freeze in [("RegionProposal", region_proposal_freeze), ("Detector", detector_freeze)]:
        if freeze is not None:
            model_configuration[section_name]["pretrained"] = {"experiment_folder": "/runs/source", "iteration": None,
                                                               "freeze": freeze, "learning_rate_multiplier": 0.1,
                                                               "new_class_initialisation": new_class_initialisation}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME.replace(
        "_RBN-R2x64_DSE-DRES-DBN-D7.2x64.256.128_MRES", expected_sections)


@pytest.mark.parametrize(argnames=("freeze", "new_class_initialisation", "expected_tokens"),
                         argvalues=[(False, "source", "-MR28-MPT_"), (True, "source", "-MR28-MPT-MFRZ_"),
                                    (False, "random", "-MR28-MPT-MRND_"), (True, "random", "-MR28-MPT-MRND-MFRZ_")])
def test_pretrained_mask_head_run_names(freeze: bool, new_class_initialisation: str, expected_tokens: str) -> None:
    """
    Verifies the tokens of transferred mask heads at the end of the mask heads section of an instance segmentation
    run: MPT, then MRND when the classes added after those of the source run start random, then MFRZ when they are
    frozen, the learning rate multiplier giving no token.

    Args:
        freeze (bool): Whether the transferred mask heads are frozen.
        new_class_initialisation (str): The initialisation of the classes added after those of the source run.
        expected_tokens (str): The expected end of the mask heads section.
    """
    experiment_configuration, model_configuration = get_instance_segmentation_reference_configurations()
    model_configuration["MaskHead"]["pretrained"] = {"experiment_folder": "/runs/source", "iteration": None,
                                                     "freeze": freeze, "learning_rate_multiplier": 0.1,
                                                     "new_class_initialisation": new_class_initialisation}
    assert build_run_name(experiment_configuration=experiment_configuration,
                          model_configuration=model_configuration) == INSTANCE_SEGMENTATION_REFERENCE_RUN_NAME.replace(
        "-MR28_", expected_tokens)
