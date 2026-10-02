import copy
import pytest

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from vision_training.utilities.os_utilities import build_run_name, load_experiment_configuration

EXAMPLE_CONFIGURATION_PATH = (Path(__file__).parents[2] / "vision_training" / "configurations" / "object_detection" /
                              "configuration_example.yaml")

# Run name of the reference settings set by get_reference_configurations
REFERENCE_RUN_NAME = ("IN-KAR-1024_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.6x128.6x128_L567_A9.32-512_RBN-R2x64_"
                      "DSE-DRES-DBN-D7.2x64.256.128_BIN_AUG-BS40-LR2e-4")


def get_reference_configurations() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads the example configuration and sets every setting used by the run name explicitly, so that the expected
    names do not change when the example configuration is edited.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The experiment and model configurations of the reference settings.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=EXAMPLE_CONFIGURATION_PATH)

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
    (change_input_settings, REFERENCE_RUN_NAME.replace("IN-KAR-1024_", "PSN-1024_")),
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
