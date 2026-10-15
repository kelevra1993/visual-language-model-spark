"""
Checks of the configuration sections specific to the instance segmentation task: every check of the object detection
task, whose two-stage detector the instance segmenter extends, and the 'MaskHead' section of its mask heads. Run by
configuration_checker.py for the instance_segmentation task.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                           VALID_UPSAMPLINGS,
                                                                           check_batch_normalization_momentum,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value)
from vision_training.configuration_processing.detection_checks import check_detection_configuration


def check_mask_head_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'MaskHead' section built into one MaskHead per detection level: the mask resolution and the
    upsampling shared by every level, the RoIAlign pool size of every level (at most the mask resolution, the pooled
    features being only enlarged) and its [number_layers, output_channels] convolutions (a multiple of 8 channels with
    inception modules), the normalization and the momentum of its batch statistics, the module flags and the mask
    threshold of the inference. Its levels are compared with the detection blocks by check_detection_levels.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'MaskHead' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Side of the masks of every level and doubling of the pooled features
    mask_resolution_errors = check_positive_integers(configuration=model_configuration,
                                                     key_paths=["MaskHead.mask_resolution"], allow_zero=False)
    errors = list(mask_resolution_errors)
    mask_resolution = get_configuration_value(configuration=model_configuration, key_path="MaskHead.mask_resolution")
    upsampling = get_configuration_value(configuration=model_configuration, key_path="MaskHead.upsampling")
    if upsampling not in VALID_UPSAMPLINGS:
        errors.append(f"'MaskHead.upsampling' must be one of {VALID_UPSAMPLINGS}, got {upsampling!r}.")
    inception = get_configuration_value(configuration=model_configuration, key_path="MaskHead.modules.inception")

    # Each head defines its RoIAlign pool size and its [number of layers, number of output channels] convolutions
    head_configurations = get_configuration_value(configuration=model_configuration,
                                                  key_path="MaskHead.head_configurations")
    for level in head_configurations:
        key_prefix = f"MaskHead.head_configurations.{level}"
        pool_size_errors = check_positive_integers(configuration=model_configuration,
                                                   key_paths=[f"{key_prefix}.roi_align_pool_size"], allow_zero=False)
        errors += pool_size_errors
        pool_size = get_configuration_value(configuration=model_configuration,
                                            key_path=f"{key_prefix}.roi_align_pool_size")
        if not pool_size_errors and not mask_resolution_errors and pool_size > mask_resolution:
            errors.append(f"'{key_prefix}.roi_align_pool_size' ({pool_size}) must not exceed "
                          f"'MaskHead.mask_resolution' ({mask_resolution}), the pooled features being only enlarged "
                          f"to the masks.")

        convolutions = get_configuration_value(configuration=model_configuration, key_path=f"{key_prefix}.convolutions")
        if not isinstance(convolutions, list) or len(convolutions) != 2 or \
                not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1
                        for value in convolutions):
            errors.append(f"'{key_prefix}.convolutions' must be [number_layers, output_channels] with positive "
                          f"integers, got {convolutions!r}.")
        elif inception is True and convolutions[1] % 8 != 0:
            errors.append(f"'{key_prefix}.convolutions' has {convolutions[1]} output channels, inception modules "
                          f"require a multiple of 8.")

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
