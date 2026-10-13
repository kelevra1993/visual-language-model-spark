"""
Checks of the configuration sections specific to the instance segmentation task: every check of the object detection
task, whose two-stage detector the instance segmenter extends, and the 'MaskHead' section of its mask heads. Run by
configuration_checker.py for the instance_segmentation task.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                           check_batch_normalization_momentum,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value)
from vision_training.configuration_processing.detection_checks import check_detection_configuration


def check_mask_head_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'MaskHead' section built into one MaskHead per detection level: the RoIAlign pool size shared by
    every level, the [number_layers, output_channels] of every level (a multiple of 8 channels with inception
    modules), the normalization and the momentum of its batch statistics, the module flags and the mask threshold of
    the inference. Its levels are compared with the detection blocks by check_detection_levels.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'MaskHead' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_positive_integers(configuration=model_configuration, key_paths=["MaskHead.roi_align_pool_size"],
                                     allow_zero=False)
    inception = get_configuration_value(configuration=model_configuration, key_path="MaskHead.modules.inception")

    # Each head is described by [number of layers, number of output channels]
    head_configurations = get_configuration_value(configuration=model_configuration,
                                                  key_path="MaskHead.head_configurations")
    for level, head_definition in head_configurations.items():
        if not isinstance(head_definition, list) or len(head_definition) != 2 or \
                not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1
                        for value in head_definition):
            errors.append(f"'MaskHead.head_configurations.{level}' must be [number_layers, output_channels] with "
                          f"positive integers, got {head_definition!r}.")
        elif inception is True and head_definition[1] % 8 != 0:
            errors.append(f"'MaskHead.head_configurations.{level}' has {head_definition[1]} output channels, "
                          f"inception modules require a multiple of 8.")

    # Feature map normalization strategy, momentum of its batch statistics and module flags of the mask head
    # convolution blocks
    feature_map_normalization = get_configuration_value(configuration=model_configuration,
                                                        key_path="MaskHead.normalization.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'MaskHead.normalization.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")
    errors += check_batch_normalization_momentum(configuration=model_configuration,
                                                 key_path="MaskHead.normalization.batch_normalization_momentum")
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["MaskHead.modules.squeeze_and_excite", "MaskHead.modules.residual_nets",
                                             "MaskHead.modules.inception"])

    # Probability from which a pixel belongs to its object
    errors += check_probability_value(
        value=get_configuration_value(configuration=model_configuration, key_path="MaskHead.inference.mask_threshold"),
        key_path="MaskHead.inference.mask_threshold")
    return errors


def check_instance_segmentation_configuration(experiment_configuration: Dict[str, Any],
                                              model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the instance segmentation task: those of the object detection task, the mask head
    levels being compared with the detection blocks there, then the mask head section.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_detection_configuration(experiment_configuration=experiment_configuration,
                                           model_configuration=model_configuration)
    errors += check_mask_head_configuration(model_configuration=model_configuration)
    return errors
