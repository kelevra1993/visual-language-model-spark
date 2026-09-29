import math

from typing import Any, Dict, List

# Feature map normalization strategies supported by the ConvolutionBlock
VALID_FEATURE_MAP_NORMALIZATIONS = ["batch", "layer", "none"]

# Label mapping modes handled by the configuration loader
VALID_LABEL_MODES = ["binary", "multi_class"]


class ConfigurationError(Exception):
    """
    Raised when the configuration file is incomplete or inconsistent. It carries the plain list of problems so that
    'load_experiment_configuration' can display them as short messages instead of a full traceback.
    """

    def __init__(self, errors: List[str]) -> None:
        """
        Stores the configuration problems found by the checker.

        Args:
            errors (List[str]): One short description per configuration problem.
        """
        super(ConfigurationError, self).__init__("\n".join(errors))
        self.errors = errors


def get_configuration_value(configuration: Dict[str, Any], key_path: str) -> Any:
    """
    Retrieves a nested configuration value from a dotted key path (e.g. 'Backbone.modules.inception').

    The configuration checker relies on this lookup so that a key missing from the YAML file is reported with its
    full location in the file, rather than as a bare KeyError raised deep inside the model construction.

    Args:
        configuration (Dict[str, Any]): The configuration dictionary to search.
        key_path (str): The dot-separated path of keys leading to the value.

    Returns:
        Any: The value stored at the given key path.

    Raises:
        ConfigurationError: If any key along the path is missing from the configuration.
    """
    current_value = configuration

    # Walk down the nested dictionaries one key at a time
    for key in key_path.split("."):
        if not isinstance(current_value, dict) or key not in current_value:
            raise ConfigurationError(errors=[f"Missing configuration key '{key_path}'."])
        current_value = current_value[key]

    return current_value


def check_boolean_flags(configuration: Dict[str, Any], key_paths: List[str]) -> List[str]:
    """
    Verifies that on/off flags are real booleans. A quoted "False" in the YAML is a non-empty string, which Python
    evaluates as True, so it would silently enable a feature of the pipeline.

    Args:
        configuration (Dict[str, Any]): The configuration dictionary containing the flags.
        key_paths (List[str]): The dotted key paths of the flags to verify.

    Returns:
        List[str]: The description of every flag that is not a boolean.
    """
    errors = []

    # Check the type of every listed flag
    for key_path in key_paths:
        value = get_configuration_value(configuration=configuration, key_path=key_path)
        if not isinstance(value, bool):
            errors.append(f"'{key_path}' must be True or False, got {value!r}.")

    return errors


def check_positive_integers(configuration: Dict[str, Any], key_paths: List[str], allow_zero: bool) -> List[str]:
    """
    Verifies that counts used by the pipeline (iterations, batch sizes, proposal numbers...) are integers and
    strictly positive, or non-negative when zero is meaningful.

    Args:
        configuration (Dict[str, Any]): The configuration dictionary containing the values.
        key_paths (List[str]): The dotted key paths of the values to verify.
        allow_zero (bool): Whether zero is an accepted value.

    Returns:
        List[str]: The description of every invalid value.
    """
    errors = []
    minimum_value = 0 if allow_zero else 1

    # Booleans are integers in Python, so they are rejected explicitly
    for key_path in key_paths:
        value = get_configuration_value(configuration=configuration, key_path=key_path)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum_value:
            errors.append(f"'{key_path}' must be an integer >= {minimum_value}, got {value!r}.")

    return errors


def check_intersection_over_union_value(value: Any, key_path: str) -> List[str]:
    """
    Verifies that an intersection over union threshold is a number within [0, 1], the only range an IoU can take
    during anchor assignment, proposal assignment and non-maximum suppression.

    Args:
        value (Any): The threshold value read from the configuration.
        key_path (str): The dotted key path of the value, used in the error message.

    Returns:
        List[str]: A single error description if the value is invalid, otherwise an empty list.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        return [f"'{key_path}' must be a number within [0, 1], got {value!r}."]
    return []


def check_experiment_configuration(experiment_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the training loop settings of the 'ExperimentConfiguration' section consumed by the Trainer.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}

    # Counts driving the training loop, logging and checkpointing
    errors += check_positive_integers(configuration=configuration,
                                      key_paths=["ExperimentConfiguration.batch_size",
                                                 "ExperimentConfiguration.information_dump",
                                                 "ExperimentConfiguration.weight_saving_iterations",
                                                 "ExperimentConfiguration.number_iterations"], allow_zero=False)
    errors += check_positive_integers(configuration=configuration,
                                      key_paths=["ExperimentConfiguration.validation_iteration_ratio"], allow_zero=True)

    # The optimizer requires a strictly positive learning rate
    learning_rate = get_configuration_value(configuration=configuration, key_path="ExperimentConfiguration.learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        errors.append(f"'ExperimentConfiguration.learning_rate' must be > 0, got {learning_rate!r}.")

    # Flags toggling training resumption and the diagnosis modes
    errors += check_boolean_flags(configuration=configuration,
                                  key_paths=["ExperimentConfiguration.resume_training",
                                             "ExperimentConfiguration.diagnosis_modes.verbose",
                                             "ExperimentConfiguration.diagnosis_modes.compute_receptive_field",
                                             "ExperimentConfiguration.diagnosis_modes.compute_model_size",
                                             "ExperimentConfiguration.diagnosis_modes.view_input_images"])

    return errors


def check_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Data' section: loader flags, label mapping mode and the input image size. The image size must be
    a power of two so that every max pooling layer of the Backbone halves the feature maps exactly, which keeps
    the anchor strides and the feature map sizes consistent across the detection levels.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Data' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Loader and augmentation flags
    augmentation_types = get_configuration_value(configuration=model_configuration,
                                                 key_path="Data.data_augmentation.types")
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["Data.DALI.activated", "Data.image_settings.keep_ratio",
                                             "Data.data_augmentation.activated"] +
                                            [f"Data.data_augmentation.types.{name}" for name in augmentation_types])

    # Only the binary and multi-class label mappings are handled by the configuration loader
    label_mode = get_configuration_value(configuration=model_configuration, key_path="Data.label_mapping.mode")
    if label_mode not in VALID_LABEL_MODES:
        errors.append(f"'Data.label_mapping.mode' must be one of {VALID_LABEL_MODES}, got {label_mode!r}.")

    # The image size must be a power of two to be halved exactly by every pooling layer
    image_size = get_configuration_value(configuration=model_configuration, key_path="Data.image_settings.size")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 1 or \
            image_size & (image_size - 1) != 0:
        errors.append(f"'Data.image_settings.size' must be a power of two, got {image_size!r}.")

    return errors


def check_backbone_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Backbone' section: the convolution blocks, their normalization and modules, the detection block
    indices, and that the image survives every pooling layer with at least one pixel.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Backbone' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    convolutions = get_configuration_value(configuration=model_configuration, key_path="Backbone.convolutions")
    number_blocks = len(convolutions)

    # Blocks are built in order, so their keys must be the consecutive block numbers "1" to "N"
    expected_block_keys = [str(block_number) for block_number in range(1, number_blocks + 1)]
    if list(convolutions.keys()) != expected_block_keys:
        errors.append(f"'Backbone.convolutions' keys must be {expected_block_keys} in that order, "
                      f"got {list(convolutions.keys())}.")

    # Each block is described by [number of layers, number of output channels]
    malformed_blocks = False
    for block_key, block_definition in convolutions.items():
        if not isinstance(block_definition, list) or len(block_definition) != 2 or \
                not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1
                        for value in block_definition):
            malformed_blocks = True
            errors.append(f"'Backbone.convolutions.{block_key}' must be [number_layers, output_channels] with "
                          f"positive integers, got {block_definition!r}.")

    errors += check_positive_integers(configuration=model_configuration, key_paths=["Backbone.input_channels"],
                                      allow_zero=False)
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["Backbone.last_max_pooling", "Backbone.normalization.input_normalization",
                                             "Backbone.normalization.per_sample_normalization",
                                             "Backbone.modules.squeeze_and_excite", "Backbone.modules.residual_nets",
                                             "Backbone.modules.inception"])

    # Feature map normalization strategy of the convolution blocks
    feature_map_normalization = get_configuration_value(configuration=model_configuration,
                                                        key_path="Backbone.normalization.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'Backbone.normalization.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")

    # Detection blocks must exist in the Backbone and be listed from the finest to the coarsest level
    detection_block_indices = get_configuration_value(configuration=model_configuration,
                                                      key_path="Backbone.detection_convolution_block_indices")
    if not detection_block_indices or any(index not in range(1, number_blocks + 1) for index in detection_block_indices):
        errors.append(f"'Backbone.detection_convolution_block_indices' must be block numbers between 1 and "
                      f"{number_blocks}, got {detection_block_indices!r}.")
    elif detection_block_indices != sorted(set(detection_block_indices)):
        errors.append(f"'Backbone.detection_convolution_block_indices' must be unique and in increasing order, "
                      f"got {detection_block_indices!r}.")

    # Every block except the last one is followed by a max pooling layer, the last one only if requested
    image_size = get_configuration_value(configuration=model_configuration, key_path="Data.image_settings.size")
    last_max_pooling = get_configuration_value(configuration=model_configuration, key_path="Backbone.last_max_pooling")
    number_pooling_layers = number_blocks - 1 + int(bool(last_max_pooling))
    if isinstance(image_size, int) and not isinstance(image_size, bool) and image_size < 2 ** number_pooling_layers:
        errors.append(f"'Data.image_settings.size' ({image_size}) is too small for the {number_pooling_layers} "
                      f"Backbone pooling layers, it must be at least {2 ** number_pooling_layers}.")

    # Inception modules start at a valid block and need output channels that are multiples of 8
    inception = get_configuration_value(configuration=model_configuration, key_path="Backbone.modules.inception")
    if inception is True and not malformed_blocks and list(convolutions.keys()) == expected_block_keys:
        inception_starting_block = get_configuration_value(configuration=model_configuration,
                                                           key_path="Backbone.modules.inception_starting_block")
        if inception_starting_block not in range(1, number_blocks + 1):
            errors.append(f"'Backbone.modules.inception_starting_block' must be between 1 and {number_blocks}, "
                          f"got {inception_starting_block!r}.")
        else:
            for block_number in range(inception_starting_block, number_blocks + 1):
                output_channels = convolutions[str(block_number)][1]
                if output_channels % 8 != 0:
                    errors.append(f"'Backbone.convolutions.{block_number}' has {output_channels} output channels, "
                                  f"inception modules require a multiple of 8.")

    return errors


def check_detection_levels(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies that the same detection levels are configured everywhere. The Model builds one set of anchors, one
    Region Proposal head and one Detection head per detection block, so the three per-level dictionaries must use
    exactly the block numbers listed in 'Backbone.detection_convolution_block_indices'.

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


def check_configuration(experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any]) -> None:
    """
    Verifies that a loaded configuration file is complete and consistent before any model or data loader is built.
    It is run by 'load_experiment_configuration', so an inconsistent file is rejected at startup with the full list
    of problems instead of failing later inside model construction or, worse, training silently with wrong values.

    Args:
        experiment_configuration (Dict[str, Any]): The processed 'ExperimentConfiguration' section.
        model_configuration (Dict[str, Any]): The processed model sections (Data, Backbone, RegionProposal, Anchors,
            Detector).

    Raises:
        ConfigurationError: If a required configuration key is missing, or if one or more configuration values are
            inconsistent, listing all of them.
    """
    errors = []

    # Run every section check and collect all the problems so they can be fixed in a single pass
    errors += check_experiment_configuration(experiment_configuration=experiment_configuration)
    errors += check_data_configuration(model_configuration=model_configuration)
    errors += check_backbone_configuration(model_configuration=model_configuration)
    errors += check_detection_levels(model_configuration=model_configuration)
    errors += check_anchors_configuration(model_configuration=model_configuration)
    errors += check_region_proposal_configuration(model_configuration=model_configuration)
    errors += check_detector_configuration(model_configuration=model_configuration)

    if errors:
        raise ConfigurationError(errors=errors)
