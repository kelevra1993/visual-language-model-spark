import math

from typing import Any, Dict, List

from vision_training.configuration.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                check_boolean_flags,
                                                                check_intersection_over_union_value,
                                                                check_positive_integers,
                                                                check_probability_value,
                                                                get_configuration_value)


def check_detection_levels(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies that the same detection levels are configured everywhere. The TwoStageDetector builds one set of
    anchors, one Region Proposal head and one Detection head per detection block, so the three per-level dictionaries
    must use exactly the block numbers listed in 'Backbone.detection_convolution_block_indices'.

    Args:
        model_configuration (Dict[str, Any]): The model configuration.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    detection_block_indices = get_configuration_value(configuration=model_configuration,
                                                      key_path="Backbone.detection_convolution_block_indices")
    expected_levels = sorted(str(index) for index in detection_block_indices)

    # Compare the levels of every per-level dictionary with the Backbone detection blocks
    for key_path in ["Anchors.scales_and_ratios", "RegionProposal.head_configurations",
                     "Detector.head_configurations"]:
        levels = sorted(get_configuration_value(configuration=model_configuration, key_path=key_path).keys())
        if levels != expected_levels:
            errors.append(f"'{key_path}' levels {levels} do not match "
                          f"'Backbone.detection_convolution_block_indices' {expected_levels}.")

    return errors


def check_anchors_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the anchor scales and aspect ratios of every detection level. Scales are anchor sizes in pixels, so
    they must be positive integers that fit in the input image; aspect ratios (height / width) must be positive.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Anchors' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    image_size = get_configuration_value(configuration=model_configuration, key_path="Data.image_settings.size")
    scales_and_ratios = get_configuration_value(configuration=model_configuration,
                                                key_path="Anchors.scales_and_ratios")

    # Bound the scales by the image size, unless the image size itself is invalid (reported by the Data check)
    maximum_scale = image_size if isinstance(image_size, int) and not isinstance(image_size, bool) else math.inf

    # Check the scales and aspect ratios of each level
    for level in scales_and_ratios:
        scales = get_configuration_value(configuration=model_configuration,
                                         key_path=f"Anchors.scales_and_ratios.{level}.scales")
        aspect_ratios = get_configuration_value(configuration=model_configuration,
                                                key_path=f"Anchors.scales_and_ratios.{level}.aspect_ratios")

        if not scales or not all(isinstance(scale, int) and not isinstance(scale, bool) and 0 < scale <= maximum_scale
                                 for scale in scales):
            errors.append(f"'Anchors.scales_and_ratios.{level}.scales' must be integers within [1, {maximum_scale}], "
                          f"got {scales!r}.")
        if not aspect_ratios or not all(isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio > 0
                                        for ratio in aspect_ratios):
            errors.append(f"'Anchors.scales_and_ratios.{level}.aspect_ratios' must be positive numbers, "
                          f"got {aspect_ratios!r}.")

    return errors


def check_region_proposal_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'RegionProposal' section: head definitions, IoU thresholds used to label anchors, proposal counts
    kept around non-maximum suppression, and the anchor sampling budget of the proposal losses.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'RegionProposal' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Each head is described by [number of layers, number of output channels]
    head_configurations = get_configuration_value(configuration=model_configuration,
                                                  key_path="RegionProposal.head_configurations")
    for level, head_definition in head_configurations.items():
        if not isinstance(head_definition, list) or len(head_definition) != 2 or \
                not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1
                        for value in head_definition):
            errors.append(f"'RegionProposal.head_configurations.{level}' must be [number_layers, output_channels] "
                          f"with positive integers, got {head_definition!r}.")

    # Feature map normalization strategy of the region proposal convolution blocks
    feature_map_normalization = get_configuration_value(
        configuration=model_configuration, key_path="RegionProposal.normalization.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'RegionProposal.normalization.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")

    # Every IoU threshold must lie within [0, 1]
    threshold_key_paths = ["RegionProposal.foreground_iou_threshold.min", "RegionProposal.foreground_iou_threshold.max",
                           "RegionProposal.background_iou_threshold.min", "RegionProposal.background_iou_threshold.max",
                           "RegionProposal.nms_iou_threshold"]
    threshold_errors = []
    for key_path in threshold_key_paths:
        threshold_errors += check_intersection_over_union_value(
            value=get_configuration_value(configuration=model_configuration, key_path=key_path), key_path=key_path)
    errors += threshold_errors

    # The ranges must be ordered and the background range must end before the foreground range starts
    if not threshold_errors:
        foreground_threshold = get_configuration_value(configuration=model_configuration,
                                                       key_path="RegionProposal.foreground_iou_threshold")
        background_threshold = get_configuration_value(configuration=model_configuration,
                                                       key_path="RegionProposal.background_iou_threshold")
        if foreground_threshold["min"] > foreground_threshold["max"]:
            errors.append("'RegionProposal.foreground_iou_threshold' min must be <= max.")
        if background_threshold["min"] > background_threshold["max"]:
            errors.append("'RegionProposal.background_iou_threshold' min must be <= max.")
        if background_threshold["max"] > foreground_threshold["min"]:
            errors.append(f"'RegionProposal.background_iou_threshold.max' ({background_threshold['max']}) must be <= "
                          f"'RegionProposal.foreground_iou_threshold.min' ({foreground_threshold['min']}), otherwise "
                          f"anchors can be labelled both background and foreground.")

    # Proposal counts kept before and after non-maximum suppression
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["RegionProposal.strict_fallback_assignment"])
    for phase in ["training", "inference"]:
        count_errors = check_positive_integers(configuration=model_configuration,
                                               key_paths=[f"RegionProposal.{phase}.pre_nms_proposals",
                                                          f"RegionProposal.{phase}.post_nms_proposals"],
                                               allow_zero=False)
        errors += count_errors
        pre_nms_proposals = get_configuration_value(configuration=model_configuration,
                                                    key_path=f"RegionProposal.{phase}.pre_nms_proposals")
        post_nms_proposals = get_configuration_value(configuration=model_configuration,
                                                     key_path=f"RegionProposal.{phase}.post_nms_proposals")
        if not count_errors and post_nms_proposals > pre_nms_proposals:
            errors.append(f"'RegionProposal.{phase}.post_nms_proposals' ({post_nms_proposals}) must be <= "
                          f"'RegionProposal.{phase}.pre_nms_proposals' ({pre_nms_proposals}).")

    # Anchor sampling budget and regression loss of the proposal stage
    errors += check_sampling_configuration(model_configuration=model_configuration, section_key="RegionProposal")

    return errors


def check_detector_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Detector' section: head definitions, normalization, modules, IoU thresholds, and that the
    proposal sampling budget fits within the proposals kept by the Region Proposal stage during training.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Detector' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    inception = get_configuration_value(configuration=model_configuration, key_path="Detector.modules.inception")

    # Each head defines its RoIAlign pool size, its convolution block and its fully connected layers
    head_configurations = get_configuration_value(configuration=model_configuration,
                                                  key_path="Detector.head_configurations")
    for level in head_configurations:
        key_prefix = f"Detector.head_configurations.{level}"
        errors += check_positive_integers(configuration=model_configuration,
                                          key_paths=[f"{key_prefix}.roi_align_pool_size"], allow_zero=False)

        convolutions = get_configuration_value(configuration=model_configuration, key_path=f"{key_prefix}.convolutions")
        if not isinstance(convolutions, list) or len(convolutions) != 2 or \
                not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1 for value in convolutions):
            errors.append(f"'{key_prefix}.convolutions' must be [number_layers, output_channels] with positive "
                          f"integers, got {convolutions!r}.")
        elif inception is True and convolutions[1] % 8 != 0:
            errors.append(f"'{key_prefix}.convolutions' has {convolutions[1]} output channels, inception modules "
                          f"require a multiple of 8.")

        fully_connected = get_configuration_value(configuration=model_configuration,
                                                  key_path=f"{key_prefix}.fully_connected")
        if not isinstance(fully_connected, list) or not fully_connected or \
                not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1
                        for value in fully_connected):
            errors.append(f"'{key_prefix}.fully_connected' must be a non-empty list of positive integers, "
                          f"got {fully_connected!r}.")

    # Feature map normalization strategy and module flags of the detection head convolution blocks
    feature_map_normalization = get_configuration_value(configuration=model_configuration,
                                                        key_path="Detector.normalization.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'Detector.normalization.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["Detector.modules.squeeze_and_excite", "Detector.modules.residual_nets",
                                             "Detector.modules.inception"])

    # IoU thresholds used to label proposals and to filter the final detections
    for key_path in ["Detector.foreground_iou_threshold", "Detector.nms_iou_threshold"]:
        errors += check_intersection_over_union_value(
            value=get_configuration_value(configuration=model_configuration, key_path=key_path), key_path=key_path)

    # Final detection post-processing: score threshold and fixed number of detections per image
    errors += check_probability_value(
        value=get_configuration_value(configuration=model_configuration, key_path="Detector.inference.score_threshold"),
        key_path="Detector.inference.score_threshold")
    maximum_detection_errors = check_positive_integers(configuration=model_configuration,
                                                       key_paths=["Detector.inference.maximum_detections"],
                                                       allow_zero=False)
    errors += maximum_detection_errors

    # The detections are selected among every (proposal, foreground class) pair kept at inference
    maximum_detections = get_configuration_value(configuration=model_configuration,
                                                 key_path="Detector.inference.maximum_detections")
    inference_proposals = get_configuration_value(configuration=model_configuration,
                                                  key_path="RegionProposal.inference.post_nms_proposals")
    number_foreground_classes = len(get_configuration_value(configuration=model_configuration,
                                                            key_path="Data.label_dictionary")) - 1
    if not maximum_detection_errors and isinstance(inference_proposals, int) and \
            maximum_detections > inference_proposals * number_foreground_classes:
        errors.append(f"'Detector.inference.maximum_detections' ({maximum_detections}) must be <= "
                      f"'RegionProposal.inference.post_nms_proposals' ({inference_proposals}) x the "
                      f"{number_foreground_classes} foreground classes.")

    # Proposal sampling budget and regression loss of the detection stage
    sampling_errors = check_sampling_configuration(model_configuration=model_configuration, section_key="Detector")
    errors += sampling_errors

    # The detector samples its training targets among the proposals kept after non-maximum suppression
    total_training_samples = get_configuration_value(configuration=model_configuration,
                                                     key_path="Detector.total_training_samples")
    post_nms_proposals = get_configuration_value(configuration=model_configuration,
                                                 key_path="RegionProposal.training.post_nms_proposals")
    if not sampling_errors and isinstance(post_nms_proposals, int) and total_training_samples > post_nms_proposals:
        errors.append(f"'Detector.total_training_samples' ({total_training_samples}) must be <= "
                      f"'RegionProposal.training.post_nms_proposals' ({post_nms_proposals}), since the detector "
                      f"samples its targets among these proposals.")

    return errors


def check_sampling_configuration(model_configuration: Dict[str, Any], section_key: str) -> List[str]:
    """
    Verifies the training target sampling budget and the Smooth L1 beta shared by the Region Proposal and Detector
    losses: positives must fit in the total sample budget, and beta must be a positive number.

    Args:
        model_configuration (Dict[str, Any]): The model configuration.
        section_key (str): The section to verify, either 'RegionProposal' or 'Detector'.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_positive_integers(configuration=model_configuration,
                                     key_paths=[f"{section_key}.number_training_positives",
                                                f"{section_key}.total_training_samples"], allow_zero=False)

    # The positive samples are part of the total sampling budget
    if not errors:
        number_training_positives = get_configuration_value(configuration=model_configuration,
                                                            key_path=f"{section_key}.number_training_positives")
        total_training_samples = get_configuration_value(configuration=model_configuration,
                                                         key_path=f"{section_key}.total_training_samples")
        if number_training_positives > total_training_samples:
            errors.append(f"'{section_key}.number_training_positives' ({number_training_positives}) must be <= "
                          f"'{section_key}.total_training_samples' ({total_training_samples}).")

    # The Smooth L1 transition point must be a finite positive number (fractions are converted at loading time)
    localisation_loss_beta = get_configuration_value(configuration=model_configuration,
                                                     key_path=f"{section_key}.localisation_loss_beta")
    if isinstance(localisation_loss_beta, bool) or not isinstance(localisation_loss_beta, (int, float)) or \
            not math.isfinite(localisation_loss_beta) or localisation_loss_beta <= 0:
        errors.append(f"'{section_key}.localisation_loss_beta' must be a positive number, "
                      f"got {localisation_loss_beta!r}.")

    return errors
