"""
Checks of the configuration sections specific to the object detection task: the detection levels (anchors, region
proposal and detection heads of every level), the neck, the region proposal network, the detector heads, the proposal
and anchor sampling, the box losses, the final non-maximum suppression, the weights of the losses and the evaluation
settings. Run by configuration_checker.py for the object_detection task, and by instance_segmentation_checks.py for
the instance segmentation task, which adds its mask heads to the detector.
"""

import math

from typing import Any, Dict, List

from vision_training.configuration_processing.common_checks import (check_error_dumps_configuration,
                                                                    check_image_only_augmentation_types,
                                                                    check_image_task_sequence_settings,
                                                                    check_loss_weights_configuration,
                                                                    check_pretrained_configuration)
from vision_training.configuration_processing.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                           check_boolean_flags,
                                                                           check_intersection_over_union_value,
                                                                           check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value, is_number)
from vision_training.losses.detection_losses import BOX_LOSSES
from vision_training.models.detection.feature_neck import NECK_TYPES
from vision_training.models.detection.postprocessing import NMS_CRITERIA
from vision_training.utilities.os_utilities import print_color


# Settings of every detection level of the object detection, the instance segmentation adding the mask head of the
# level ('mask_head', checked by instance_segmentation_checks.py)
DETECTION_LEVEL_COMPONENTS = ["anchors", "region_proposal", "detector"]
MASK_HEAD_LEVEL_COMPONENT = "mask_head"

# Loss outputs of the two-stage detector, summed into the total loss with the weights of 'Optimization.loss_weights',
# the instance segmentation adding the loss of its mask heads
DETECTION_LOSS_NAMES = ["region_proposal_classification_loss", "region_proposal_localisation_loss",
                        "detection_classification_loss", "detection_regression_loss"]
MASK_LOSS_NAME = "mask_loss"


def check_convolution_layers(convolutions: Any, key_path: str, inception: Any) -> List[str]:
    """
    Verifies the [number_layers, output_channels] convolution block of a head of a detection level (region proposal,
    detection or mask head), shared by the object detection and instance segmentation checks: two positive integers,
    the output channels being a multiple of 8 when the heads use inception modules.

    Args:
        convolutions (Any): The configured convolutions of the head.
        key_path (str): The location of the convolutions in the YAML file, named in the errors.
        inception (Any): The 'modules.inception' flag of the heads, False for the region proposal heads, which have
            no modules.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    if not isinstance(convolutions, list) or len(convolutions) != 2 or \
            not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1 for value in convolutions):
        return [f"'{key_path}' must be [number_layers, output_channels] with positive integers, got {convolutions!r}."]
    if inception is True and convolutions[1] % 8 != 0:
        return [f"'{key_path}' has {convolutions[1]} output channels, inception modules require a multiple of 8."]
    return []


def list_checked_detection_levels(model_configuration: Dict[str, Any], component_names: List[str]) -> List[str]:
    """
    Lists the levels of the 'DetectionLevels' section whose settings can be read by their key path, the quoted block
    numbers holding a dictionary with the components to check, so that the checks of the settings of every level (here
    and in instance_segmentation_checks.py) skip the levels already reported by check_detection_levels.

    Args:
        model_configuration (Dict[str, Any]): The model configuration.
        component_names (List[str]): The components of a level whose settings are checked, e.g. ['mask_head'].

    Returns:
        List[str]: The keys of the levels whose settings are checked.
    """
    detection_levels = get_configuration_value(configuration=model_configuration, key_path="DetectionLevels")
    if not isinstance(detection_levels, dict):
        return []
    return [level_key for level_key, level_configuration in detection_levels.items()
            if isinstance(level_key, str) and isinstance(level_configuration, dict) and
            all(component_name in level_configuration for component_name in component_names)]


def check_detection_levels(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'DetectionLevels' section, which lists the backbone blocks feeding the detection heads with the
    settings of every level: its anchors, its region proposal head and its detection head, and its mask head for the
    instance segmentation (whose settings are checked by instance_segmentation_checks.py). The TwoStageDetector builds
    one set of anchors and one head of every kind per level, so the levels must be quoted numbers of existing backbone
    blocks, from the finest to the coarsest one, each holding exactly the settings of the heads of the task. The anchor
    scales are anchor sizes in pixels, positive integers fitting in the input image, and the aspect ratios (height /
    width) positive numbers.

    Args:
        model_configuration (Dict[str, Any]): The model configuration.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    detection_levels = get_configuration_value(configuration=model_configuration, key_path="DetectionLevels")
    if not isinstance(detection_levels, dict) or not detection_levels:
        return [f"'DetectionLevels' must map at least one backbone block number to the settings of its level, got "
                f"{detection_levels!r}."]
    errors = []

    # Levels are quoted numbers of the backbone blocks, listed from the finest to the coarsest one
    backbone_convolutions = get_configuration_value(configuration=model_configuration,
                                                    key_path="Backbone.convolutions")
    number_blocks = len(backbone_convolutions) if isinstance(backbone_convolutions, dict) else 0
    valid_level_keys = [str(block_number) for block_number in range(1, number_blocks + 1)]
    level_keys = list(detection_levels.keys())
    if any(level_key not in valid_level_keys for level_key in level_keys):
        errors.append(f"'DetectionLevels' keys must be quoted backbone block numbers between \"1\" and "
                      f"\"{number_blocks}\", got {level_keys!r}.")
    elif level_keys != sorted(level_keys, key=int):
        errors.append(f"'DetectionLevels' keys must be in increasing order, got {level_keys!r}.")

    # Every level holds the settings of the heads of the task, the mask head for the instance segmentation only
    expected_components = DETECTION_LEVEL_COMPONENTS + (
        [MASK_HEAD_LEVEL_COMPONENT] if "MaskHead" in model_configuration else [])
    for level_key, level_configuration in detection_levels.items():
        level_components = list(level_configuration) if isinstance(level_configuration, dict) else level_configuration
        if not isinstance(level_configuration, dict) or sorted(level_configuration) != sorted(expected_components):
            errors.append(f"'DetectionLevels.{level_key}' must hold exactly {expected_components}, got "
                          f"{level_components!r}.")

    # Bound the scales by the image size, unless the image size itself is invalid (reported by the Data check)
    image_size = get_configuration_value(configuration=model_configuration, key_path="Data.image_settings.size")
    maximum_scale = image_size if isinstance(image_size, int) and not isinstance(image_size, bool) else math.inf
    detector_inception = get_configuration_value(configuration=model_configuration,
                                                 key_path="Detector.modules.inception")

    # Settings of the anchors and of the heads of every level
    for level_key in list_checked_detection_levels(model_configuration=model_configuration,
                                                   component_names=DETECTION_LEVEL_COMPONENTS):
        key_prefix = f"DetectionLevels.{level_key}"

        # Anchor sizes in pixels and height / width ratios
        scales = get_configuration_value(configuration=model_configuration, key_path=f"{key_prefix}.anchors.scales")
        aspect_ratios = get_configuration_value(configuration=model_configuration,
                                                key_path=f"{key_prefix}.anchors.aspect_ratios")
        if not isinstance(scales, list) or not scales or \
                not all(isinstance(scale, int) and not isinstance(scale, bool) and 0 < scale <= maximum_scale
                        for scale in scales):
            errors.append(f"'{key_prefix}.anchors.scales' must be integers within [1, {maximum_scale}], "
                          f"got {scales!r}.")
        if not isinstance(aspect_ratios, list) or not aspect_ratios or \
                not all(isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio > 0
                        for ratio in aspect_ratios):
            errors.append(f"'{key_prefix}.anchors.aspect_ratios' must be positive numbers, got {aspect_ratios!r}.")

        # Region proposal head, described by the [number of layers, number of output channels] of its convolutions
        errors += check_convolution_layers(
            convolutions=get_configuration_value(configuration=model_configuration,
                                                 key_path=f"{key_prefix}.region_proposal.convolutions"),
            key_path=f"{key_prefix}.region_proposal.convolutions", inception=False)

        # Detection head: its RoIAlign pool size, its convolution block and its fully connected layers
        errors += check_positive_integers(configuration=model_configuration,
                                          key_paths=[f"{key_prefix}.detector.roi_align_pool_size"], allow_zero=False)
        errors += check_convolution_layers(
            convolutions=get_configuration_value(configuration=model_configuration,
                                                 key_path=f"{key_prefix}.detector.convolutions"),
            key_path=f"{key_prefix}.detector.convolutions", inception=detector_inception)
        fully_connected = get_configuration_value(configuration=model_configuration,
                                                  key_path=f"{key_prefix}.detector.fully_connected")
        if not isinstance(fully_connected, list) or not fully_connected or \
                not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1
                        for value in fully_connected):
            errors.append(f"'{key_prefix}.detector.fully_connected' must be a non-empty list of positive integers, "
                          f"got {fully_connected!r}.")

    return errors


def check_neck_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Neck' section, the layers between the backbone blocks of the detection levels and the region
    proposal, detection and mask heads: its type, the channels of every level after the neck and their normalization,
    all read even without neck so that switching the type is the only change of a run with a neck. The sizes of the
    levels, which must be whole multiples of each other, are checked by FeatureNeck once the backbone is built.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Neck' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    neck_type = get_configuration_value(configuration=model_configuration, key_path="Neck.type")
    if neck_type not in NECK_TYPES:
        errors.append(f"'Neck.type' must be one of {NECK_TYPES}, got {neck_type!r}.")
    errors += check_positive_integers(configuration=model_configuration, key_paths=["Neck.output_channels"],
                                      allow_zero=False)

    # Normalization of every convolution of the neck
    feature_map_normalization = get_configuration_value(configuration=model_configuration,
                                                        key_path="Neck.normalization.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'Neck.normalization.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")
    return errors


def check_region_proposal_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'RegionProposal' section, the settings shared by the region proposal heads of every level (their
    convolutions being checked with the levels): the normalization, the IoU thresholds used to label anchors, the
    proposal counts kept around non-maximum suppression, the anchor sampling budget of the proposal losses, their box
    loss, and the 'pretrained' settings initialising the heads with those of another run. A background range starting
    above 0.0 is allowed but printed as a yellow warning, since the anchors overlapping no object are then never
    sampled as background, only the hard negatives touching an object.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'RegionProposal' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

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

        # A background range starting above 0.0 ignores the anchors overlapping no object, the region proposal
        # network then only learning from the hard negatives: allowed, but announced
        if background_threshold["min"] > 0.0:
            print_color(color="yellow",
                        output=f"Warning: 'RegionProposal.background_iou_threshold.min' is "
                               f"{background_threshold['min']}, above 0.0: the anchors whose best IoU is below it "
                               f"are ignored, only the hard negatives being sampled as background.")

    # Proposal counts kept before and after non-maximum suppression
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

    # Anchor sampling budget and localisation loss of the proposal stage
    errors += check_sampling_configuration(model_configuration=model_configuration, section_key="RegionProposal")
    errors += check_box_loss_configuration(model_configuration=model_configuration, section_key="RegionProposal")

    # Region proposal heads possibly initialised by those of another run
    errors += check_pretrained_configuration(model_configuration=model_configuration, section_name="RegionProposal")

    return errors


def check_detector_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Detector' section, the settings shared by the detection heads of every level (their pool size,
    convolutions and fully connected layers being checked with the levels): the normalization, the modules, the IoU
    thresholds, the criterion of the final non-maximum suppression, that the proposal sampling budget fits within the
    proposals kept by the Region Proposal stage during training, the box loss, and the 'pretrained' settings
    initialising the heads with those of another run.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Detector' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

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

    # Criterion and distance exponent of the final per-class non-maximum suppression
    errors += check_nms_criterion_configuration(model_configuration=model_configuration)

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
    number_foreground_classes = len(set(get_configuration_value(configuration=model_configuration,
                                                                key_path="Data.label_dictionary").values())) - 1
    if not maximum_detection_errors and isinstance(inference_proposals, int) and \
            maximum_detections > inference_proposals * number_foreground_classes:
        errors.append(f"'Detector.inference.maximum_detections' ({maximum_detections}) must be <= "
                      f"'RegionProposal.inference.post_nms_proposals' ({inference_proposals}) x the "
                      f"{number_foreground_classes} foreground classes.")

    # Proposal sampling budget of the detection stage
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

    # Localisation loss of the detection stage
    errors += check_box_loss_configuration(model_configuration=model_configuration, section_key="Detector")

    # Detection heads possibly initialised by those of another run
    errors += check_pretrained_configuration(model_configuration=model_configuration, section_name="Detector")

    return errors


def check_sampling_configuration(model_configuration: Dict[str, Any], section_key: str) -> List[str]:
    """
    Verifies the training target sampling budget of the Region Proposal and Detector losses: positives must fit in
    the total sample budget.

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

    return errors


def check_box_loss_configuration(model_configuration: Dict[str, Any], section_key: str) -> List[str]:
    """
    Verifies the localisation loss of the Region Proposal or Detector losses, the smooth L1 on the regression offsets
    or the CIoU loss on the decoded boxes, which detection_losses.py computes on the sampled positives.

    Args:
        model_configuration (Dict[str, Any]): The model configuration.
        section_key (str): The section to verify, either 'RegionProposal' or 'Detector'.

    Returns:
        List[str]: The description of the inconsistency found, empty when the loss is known.
    """
    box_loss = get_configuration_value(configuration=model_configuration, key_path=f"{section_key}.box_loss")
    if box_loss not in BOX_LOSSES:
        return [f"'{section_key}.box_loss' must be one of {BOX_LOSSES}, got {box_loss!r}."]
    return []


def check_nms_criterion_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the criterion of the final per-class non-maximum suppression of the detector, the IoU of the greedy NMS or
    the Distance-IoU of DIoU-NMS (models/detection/postprocessing.py), and the exponent of its distance term, a finite
    positive number, checked whatever the criterion since every detection file holds it. Also run by
    scripts/evaluation/evaluate_checkpoint.py on the NMS it evaluates in place of the one of the run.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Detector' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    nms_criterion = get_configuration_value(configuration=model_configuration, key_path="Detector.nms_criterion")
    if nms_criterion not in NMS_CRITERIA:
        errors.append(f"'Detector.nms_criterion' must be one of {NMS_CRITERIA}, got {nms_criterion!r}.")

    # Exponent of the normalized centre distance, 1.0 in the paper and 0.6 in YOLOv4
    nms_distance_exponent = get_configuration_value(configuration=model_configuration,
                                                    key_path="Detector.nms_distance_exponent")
    if not is_number(value=nms_distance_exponent) or not math.isfinite(nms_distance_exponent) or \
            nms_distance_exponent <= 0:
        errors.append(f"'Detector.nms_distance_exponent' must be a finite positive number, got "
                      f"{nms_distance_exponent!r}.")
    return errors


def check_detection_evaluation_configuration(experiment_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'ExperimentConfiguration.evaluation' settings of the object detection and instance segmentation
    tasks: the score threshold of the detections kept for the mean average precision and the precision targeted by
    the score threshold of every class must be probabilities, and the error images need a switch and a positive
    number of images per folder.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}
    errors = []
    for key_path in ["ExperimentConfiguration.evaluation.score_threshold",
                     "ExperimentConfiguration.evaluation.target_precision"]:
        errors += check_probability_value(value=get_configuration_value(configuration=configuration, key_path=key_path),
                                          key_path=key_path)
    errors += check_error_dumps_configuration(experiment_configuration=experiment_configuration)
    return errors


def check_anchor_coverage_configuration(experiment_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'ExperimentConfiguration.diagnosis_modes.compute_anchor_coverage' diagnosis mode of the object
    detection and instance segmentation tasks, which prints the anchor recall of the training, validation and test
    splits then exits: its switch must be a boolean and its threshold an intersection over union within [0, 1].

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}
    key_prefix = "ExperimentConfiguration.diagnosis_modes.compute_anchor_coverage"
    errors = check_boolean_flags(configuration=configuration, key_paths=[f"{key_prefix}.activated"])
    errors += check_intersection_over_union_value(
        value=get_configuration_value(configuration=configuration, key_path=f"{key_prefix}.iou_threshold"),
        key_path=f"{key_prefix}.iou_threshold")
    return errors


def check_detection_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the label mapping of the object detection and the instance segmentation: in multi-class mode, the classes
    of the label dictionary are at least two, indexed 0 (the background) to K-1 without gaps, so that the model gets one
    output per class and every target index exists (original COCO identifiers kept as class indices would leave gaps),
    and its "background" key is the class 0. A category mapped to 0 is allowed, its objects being left out of the
    training, but announced in yellow, since a mapping numbered from 0 (as written for the classifiers) silently drops
    its first category this way.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    label_mode = get_configuration_value(configuration=model_configuration, key_path="Data.label_mapping.mode")
    if label_mode != "multi_class":
        return errors

    # Classes indexed from the background 0 without gaps, the background key naming the class 0
    label_dictionary = get_configuration_value(configuration=model_configuration, key_path="Data.label_dictionary")
    class_indices = sorted(set(label_dictionary.values()))
    if len(class_indices) < 2 or class_indices != list(range(len(class_indices))):
        errors.append(f"The classes of the label mapping must be at least two, indexed 0 (background) to K-1 without "
                      f"gaps, got {class_indices}.")
    if "background" in label_dictionary and label_dictionary["background"] != 0:
        errors.append(f"The 'background' of the label mapping must be the class 0, got "
                      f"{label_dictionary['background']!r}.")

    # Categories mapped to the background: their objects are not learnt, a way to leave a category out, announced
    background_categories = [category for category, class_index in label_dictionary.items()
                             if category != "background" and class_index == 0]
    if background_categories:
        print_color(color="yellow", output=f"Warning: the categories {background_categories} of the label mapping are "
                                           f"mapped to the background class 0: their objects are left out of the "
                                           f"training.")
    return errors


def check_detection_configuration(experiment_configuration: Dict[str, Any],
                                  model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the object detection task: its evaluation settings, its label mapping, its data
    settings without the sequence settings nor the augmentation types moving the image only, its detection levels
    (anchors and heads of every level), neck, region proposal and detector sections, and the weights of its losses,
    with the mask loss for the instance segmentation, which runs these checks too.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_detection_evaluation_configuration(experiment_configuration=experiment_configuration)
    errors += check_anchor_coverage_configuration(experiment_configuration=experiment_configuration)
    errors += check_detection_data_configuration(model_configuration=model_configuration)
    errors += check_image_task_sequence_settings(model_configuration=model_configuration)
    errors += check_image_only_augmentation_types(model_configuration=model_configuration)
    errors += check_detection_levels(model_configuration=model_configuration)
    errors += check_neck_configuration(model_configuration=model_configuration)
    errors += check_region_proposal_configuration(model_configuration=model_configuration)
    errors += check_detector_configuration(model_configuration=model_configuration)

    # Weight of every loss of the task, the instance segmentation (whose configuration has a MaskHead) adding its mask
    # loss
    loss_names = DETECTION_LOSS_NAMES + ([MASK_LOSS_NAME] if "MaskHead" in model_configuration else [])
    errors += check_loss_weights_configuration(model_configuration=model_configuration, loss_names=loss_names)
    return errors
