"""
Checks of the configuration sections specific to the semantic segmentation task: the data settings (Native loader,
label mapping with the background class 0, training on tiles), the decoder levels against the backbone blocks with
their deep supervision weights, the losses and the evaluation settings. Run by configuration_checker.py for the
semantic_segmentation task.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.classification_checks import VALID_CLASS_WEIGHTINGS
from vision_training.configuration_processing.common_checks import (check_image_only_augmentation_types,
                                                                    check_image_task_sequence_settings,
                                                                    check_pretrained_configuration)
from vision_training.configuration_processing.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                           VALID_UPSAMPLINGS,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           get_configuration_value, is_number)
from vision_training.data.segmentation.segmentation_dataset import IGNORED_MASK_VALUE
from vision_training.models.segmentation.segmentation_decoder import compute_skip_block_numbers


def check_segmentation_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the data settings of the semantic segmentation task: a label mapping whose classes are
    indexed 0 to K-1 with the background at 0, whose categories are mask values (1 to 254, 0 being the background and
    255 the unlabelled pixels), and the training on tiles (a positive scale of the images and an overlap of the
    evaluation tiles smaller than a tile).

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Multi-class mapping: classes indexed from the background 0 without gaps, the categories being values of the
    # mask files, 0 being allowed for the background only (the binary mapping always gives the background 0 and the
    # foreground 1, an unknown mode being reported by the common checks)
    label_mode = get_configuration_value(configuration=model_configuration, key_path="Data.label_mapping.mode")
    if label_mode == "multi_class":
        label_dictionary = get_configuration_value(configuration=model_configuration, key_path="Data.label_dictionary")
        class_indices = sorted(set(label_dictionary.values()))
        if len(class_indices) < 2 or class_indices != list(range(len(class_indices))):
            errors.append(f"The classes of the label mapping must be at least two, indexed 0 (background) to K-1 "
                          f"without gaps, got {class_indices}.")
        invalid_categories = [category for category, class_index in label_dictionary.items()
                              if category != "background" and not (category == 0 and class_index == 0) and
                              (not isinstance(category, int) or not 0 < category < IGNORED_MASK_VALUE)]
        if invalid_categories:
            errors.append(f"The categories of the label mapping file must be mask values between 1 and "
                          f"{IGNORED_MASK_VALUE - 1} (0 is the background, {IGNORED_MASK_VALUE} the unlabelled "
                          f"pixels), got {invalid_categories}.")

    # Training on tiles of the scaled images, and overlap of the tiles of the evaluation, smaller than a tile so that
    # the tiles advance, the tile size being only checked against it when the tiling is on
    tiling_errors = check_boolean_flags(configuration=model_configuration, key_paths=["Data.tiling.activated"])
    errors += tiling_errors
    scale = get_configuration_value(configuration=model_configuration, key_path="Data.tiling.scale")
    if not is_number(value=scale) or scale <= 0:
        errors.append(f"'Data.tiling.scale' must be a number > 0, got {scale!r}.")
    overlap_errors = check_positive_integers(configuration=model_configuration, key_paths=["Data.tiling.overlap"],
                                             allow_zero=True)
    errors += overlap_errors
    overlap = get_configuration_value(configuration=model_configuration, key_path="Data.tiling.overlap")
    image_size = get_configuration_value(configuration=model_configuration, key_path="Data.image_settings.size")
    tiling_activated = not tiling_errors and get_configuration_value(configuration=model_configuration,
                                                                     key_path="Data.tiling.activated")
    if tiling_activated and not overlap_errors and is_number(value=image_size) and overlap >= image_size:
        errors.append(f"'Data.tiling.overlap' must be smaller than the tile size 'Data.image_settings.size' "
                      f"({image_size}), got {overlap}.")
    return errors


def check_segmentation_decoder_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'SegmentationDecoder' section: the upsampling, one convolution block per skip connection of the
    backbone (every block before the last one, and the last one too with Backbone.last_max_pooling), the modules, the
    normalization, the dropout of the segmentation head, the deep supervision weights (positive weights of decoder
    levels other than the finest one, whose output is the final segmentation), and the 'pretrained' settings
    initialising the decoder with the one of another segmentation run.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    upsampling = get_configuration_value(configuration=model_configuration, key_path="SegmentationDecoder.upsampling")
    if upsampling not in VALID_UPSAMPLINGS:
        errors.append(f"'SegmentationDecoder.upsampling' must be one of {VALID_UPSAMPLINGS}, got {upsampling!r}.")

    # One level per skip connection of the backbone, each described by [number of layers, output channels]
    convolutions = get_configuration_value(configuration=model_configuration,
                                           key_path="SegmentationDecoder.convolutions")
    backbone_convolutions = get_configuration_value(configuration=model_configuration, key_path="Backbone.convolutions")
    last_max_pooling = get_configuration_value(configuration=model_configuration, key_path="Backbone.last_max_pooling")
    expected_levels = [str(block_number) for block_number in compute_skip_block_numbers(
        number_blocks=len(backbone_convolutions), last_max_pooling=bool(last_max_pooling))]
    missing_levels = [level_key for level_key in expected_levels if level_key not in convolutions]
    unexpected_levels = [level_key for level_key in convolutions if level_key not in expected_levels]
    if missing_levels or unexpected_levels:
        errors.append(f"'SegmentationDecoder.convolutions' keys must be the backbone blocks of its skip connections "
                      f"{expected_levels} (every block before the last one, and the last one with "
                      f"'Backbone.last_max_pooling'), missing {missing_levels}, unexpected {unexpected_levels}.")
    for level_key, level_definition in convolutions.items():
        if not isinstance(level_definition, list) or len(level_definition) != 2 or \
                not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1
                        for value in level_definition):
            errors.append(f"'SegmentationDecoder.convolutions.{level_key}' must be [number_layers, output_channels] "
                          f"with positive integers, got {level_definition!r}.")

    # Modules and normalization of the decoder blocks, dropout of the segmentation head
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["SegmentationDecoder.modules.squeeze_and_excite",
                                             "SegmentationDecoder.modules.residual_nets"])
    feature_map_normalization = get_configuration_value(
        configuration=model_configuration, key_path="SegmentationDecoder.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'SegmentationDecoder.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")
    dropout_rate = get_configuration_value(configuration=model_configuration,
                                           key_path="SegmentationDecoder.dropout_rate")
    if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float)) or not 0.0 <= dropout_rate < 1.0:
        errors.append(f"'SegmentationDecoder.dropout_rate' must be a number within [0, 1), got {dropout_rate!r}.")

    # Deep supervision heads on the intermediate levels, each with a positive loss weight
    deep_supervision_weights = get_configuration_value(configuration=model_configuration,
                                                       key_path="SegmentationDecoder.deep_supervision_weights")
    intermediate_levels = expected_levels[:-1]
    if not isinstance(deep_supervision_weights, dict):
        errors.append(f"'SegmentationDecoder.deep_supervision_weights' must be a mapping of decoder levels to loss "
                      f"weights, {{}} for no deep supervision, got {deep_supervision_weights!r}.")
    else:
        unexpected_supervised_levels = [level_key for level_key in deep_supervision_weights
                                        if level_key not in intermediate_levels]
        if unexpected_supervised_levels:
            errors.append(f"'SegmentationDecoder.deep_supervision_weights' keys must be intermediate decoder levels "
                          f"{intermediate_levels} (the finest level gives the final segmentation), got "
                          f"{unexpected_supervised_levels}.")
        for level_key, level_weight in deep_supervision_weights.items():
            if not is_number(value=level_weight) or level_weight <= 0:
                errors.append(f"'SegmentationDecoder.deep_supervision_weights.{level_key}' must be a number > 0, "
                              f"got {level_weight!r}.")

    # Decoder possibly initialised by the one of another run
    errors += check_pretrained_configuration(model_configuration=model_configuration,
                                             section_name="SegmentationDecoder")
    return errors


def check_segmentation_loss_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'SegmentationLoss' section: the non-negative weights of the cross entropy, Dice, Lovász and boundary
    losses, the losses of the regions (cross entropy, Dice, Lovász) not all 0, the focusing parameter of the focal
    factor, the label smoothing and the class weighting of the cross entropy, and the distance of the boundary loss.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Weights of the losses and focusing parameter, non-negative numbers
    for key_name in ["cross_entropy_weight", "dice_weight", "lovasz_weight", "boundary_weight", "focal_gamma"]:
        key_path = f"SegmentationLoss.{key_name}"
        value = get_configuration_value(configuration=model_configuration, key_path=key_path)
        if not is_number(value=value) or value < 0:
            errors.append(f"'{key_path}' must be a number >= 0, got {value!r}.")

    # At least one loss of the regions, the boundary loss only scoring the pixels near the edges
    region_loss_weights = [get_configuration_value(configuration=model_configuration,
                                                   key_path=f"SegmentationLoss.{key_name}")
                           for key_name in ["cross_entropy_weight", "dice_weight", "lovasz_weight"]]
    if all(region_loss_weight == 0 for region_loss_weight in region_loss_weights):
        errors.append("'SegmentationLoss.cross_entropy_weight', 'SegmentationLoss.dice_weight' and "
                      "'SegmentationLoss.lovasz_weight' cannot all be 0, the regions would have no loss.")
    errors += check_positive_integers(configuration=model_configuration,
                                      key_paths=["SegmentationLoss.boundary_distance"], allow_zero=True)

    # Label smoothing strictly below 1 so that some signal always remains, and class weighting of the cross entropy
    label_smoothing = get_configuration_value(configuration=model_configuration,
                                              key_path="SegmentationLoss.label_smoothing")
    if isinstance(label_smoothing, bool) or not isinstance(label_smoothing, (int, float)) or \
            not 0.0 <= label_smoothing < 1.0:
        errors.append(f"'SegmentationLoss.label_smoothing' must be a number within [0, 1), got {label_smoothing!r}.")
    class_weighting = get_configuration_value(configuration=model_configuration,
                                              key_path="SegmentationLoss.class_weighting")
    if class_weighting not in VALID_CLASS_WEIGHTINGS:
        errors.append(f"'SegmentationLoss.class_weighting' must be one of {VALID_CLASS_WEIGHTINGS}, "
                      f"got {class_weighting!r}.")
    return errors


def check_segmentation_evaluation_configuration(experiment_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'ExperimentConfiguration.evaluation' settings of the semantic segmentation task: whether the predicted
    mask of every test image is saved, whether the metrics are computed at the original resolution of the images, and
    the tolerance of the boundary metrics in pixels. The number of sample images, shared by every task, is checked
    with the experiment settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}
    errors = check_boolean_flags(configuration=configuration,
                                 key_paths=["ExperimentConfiguration.evaluation.save_predicted_masks",
                                            "ExperimentConfiguration.evaluation.original_resolution"])
    errors += check_positive_integers(configuration=configuration,
                                      key_paths=["ExperimentConfiguration.evaluation.boundary_tolerance"],
                                      allow_zero=False)
    return errors


def check_segmentation_configuration(experiment_configuration: Dict[str, Any],
                                     model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the semantic segmentation task: its data settings, without the sequence settings nor
    the augmentation types moving the image only, its decoder, its losses and its evaluation settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_segmentation_data_configuration(model_configuration=model_configuration)
    errors += check_image_task_sequence_settings(model_configuration=model_configuration)
    errors += check_image_only_augmentation_types(model_configuration=model_configuration)
    errors += check_segmentation_decoder_configuration(model_configuration=model_configuration)
    errors += check_segmentation_loss_configuration(model_configuration=model_configuration)
    errors += check_segmentation_evaluation_configuration(experiment_configuration=experiment_configuration)
    return errors
