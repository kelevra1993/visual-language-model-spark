"""
Checks of the configuration sections specific to the instance segmentation task: every check of the object detection
task, whose two-stage detector the instance segmenter extends, its mask heads (the 'MaskHead' section and the mask
head of every detection level) and the evaluation at the original resolution. Run by configuration_checker.py for the
instance_segmentation task.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                           VALID_UPSAMPLINGS,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value)
from vision_training.configuration_processing.common_checks import check_pretrained_configuration
from vision_training.configuration_processing.detection_checks import (MASK_HEAD_LEVEL_COMPONENT,
                                                                       check_convolution_layers,
                                                                       check_detection_configuration,
                                                                       list_checked_detection_levels)


def check_mask_head_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the mask heads, one MaskHead per detection level: the mask resolution and the upsampling shared by every
    level in the 'MaskHead' section, the RoIAlign pool size of every level in 'DetectionLevels.<level>.mask_head' (at
    most the mask resolution, the pooled features being only enlarged) and its [number_layers, output_channels]
    convolutions (a multiple of 8 channels with inception modules), the normalization and the momentum of its batch
    statistics, the module flags, the mask threshold of the inference and the 'pretrained' settings initialising the
    mask heads with those of another run. The presence of the mask head of every level is checked by
    check_detection_levels.

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

    # The mask head of every level defines its RoIAlign pool size and its [number of layers, number of output
    # channels] convolutions
    for level_key in list_checked_detection_levels(model_configuration=model_configuration,
                                                   component_names=[MASK_HEAD_LEVEL_COMPONENT]):
        key_prefix = f"DetectionLevels.{level_key}.{MASK_HEAD_LEVEL_COMPONENT}"
        pool_size_errors = check_positive_integers(configuration=model_configuration,
                                                   key_paths=[f"{key_prefix}.roi_align_pool_size"], allow_zero=False)
        errors += pool_size_errors
        pool_size = get_configuration_value(configuration=model_configuration,
                                            key_path=f"{key_prefix}.roi_align_pool_size")
        if not pool_size_errors and not mask_resolution_errors and pool_size > mask_resolution:
            errors.append(f"'{key_prefix}.roi_align_pool_size' ({pool_size}) must not exceed "
                          f"'MaskHead.mask_resolution' ({mask_resolution}), the pooled features being only enlarged "
                          f"to the masks.")
        errors += check_convolution_layers(
            convolutions=get_configuration_value(configuration=model_configuration,
                                                 key_path=f"{key_prefix}.convolutions"),
            key_path=f"{key_prefix}.convolutions", inception=inception)

    # Feature map normalization strategy and module flags of the mask head convolution blocks
    feature_map_normalization = get_configuration_value(configuration=model_configuration,
                                                        key_path="MaskHead.normalization.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'MaskHead.normalization.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["MaskHead.modules.squeeze_and_excite", "MaskHead.modules.residual_nets",
                                             "MaskHead.modules.inception"])

    # Probability from which a pixel belongs to its object
    errors += check_probability_value(
        value=get_configuration_value(configuration=model_configuration, key_path="MaskHead.inference.mask_threshold"),
        key_path="MaskHead.inference.mask_threshold")

    # Mask heads initialised by those of another run
    errors += check_pretrained_configuration(model_configuration=model_configuration, section_name="MaskHead")
    return errors


def check_instance_segmentation_configuration(experiment_configuration: Dict[str, Any],
                                              model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the instance segmentation task: those of the object detection task, every detection
    level having to hold a mask head there, then the mask heads and the evaluation at the original resolution of the
    test images.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_detection_configuration(experiment_configuration=experiment_configuration,
                                           model_configuration=model_configuration)
    errors += check_mask_head_configuration(model_configuration=model_configuration)

    # Scoring of the boxes and masks at the original resolution of the test images, nested under the section name so
    # that the reported key path matches its location in the YAML file
    errors += check_boolean_flags(configuration={"ExperimentConfiguration": experiment_configuration},
                                  key_paths=["ExperimentConfiguration.evaluation.original_resolution"])
    return errors
