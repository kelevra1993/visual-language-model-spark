"""
Checks of the configuration sections specific to the semantic segmentation task: the data settings (Native loader,
label mapping with the background class 0), the decoder levels against the backbone blocks, the losses and the
evaluation settings. Run by configuration_checker.py for the semantic_segmentation task.
"""

from typing import Any, Dict, List

from vision_training.configuration.classification_checks import VALID_CLASS_WEIGHTINGS
from vision_training.configuration.common_checks import (check_image_task_sequence_settings,
                                                         check_pretrained_configuration)
from vision_training.configuration.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                check_boolean_flags, get_configuration_value)
from vision_training.data.segmentation.segmentation_dataset import IGNORED_MASK_VALUE
from vision_training.models.segmentation.segmentation_decoder import compute_skip_block_numbers

# Upsampling of the decoder levels: a learned 2x2 transposed convolution, or a bilinear upsampling followed by a 1x1
# convolution
VALID_UPSAMPLINGS = ["transposed", "bilinear"]


def check_segmentation_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the data settings of the semantic segmentation task: the Native loader, a label mapping whose classes are
    indexed 0 to K-1 with the background at 0, whose categories are mask values (1 to 254, 0 being the background and
    255 the unlabelled pixels), and no detection block, the decoder reading every block of the backbone.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # The DALI pipelines only read the COCO detection format
    if get_configuration_value(configuration=model_configuration, key_path="Data.DALI.activated") is True:
        errors.append("'Data.DALI.activated' must be False: the DALI loaders are only available for the object "
                      "detection task, the semantic segmentation uses the Native loader.")

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

    # The decoder reads the output of every block, so no block is selected for detection heads
    detection_block_indices = get_configuration_value(configuration=model_configuration,
                                                      key_path="Backbone.detection_convolution_block_indices")
    if detection_block_indices != []:
        errors.append(f"'Backbone.detection_convolution_block_indices' must be [] for the semantic segmentation "
                      f"task, its decoder reads every block, got {detection_block_indices!r}.")
    return errors


def check_segmentation_decoder_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'SegmentationDecoder' section: the upsampling, one convolution block per skip connection of the
    backbone (every block before the last one, and the last one too with Backbone.last_max_pooling), the modules, the
    normalization, the dropout of the segmentation head, and the 'pretrained' settings initialising the decoder with
    the one of another segmentation run.

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

    errors += check_pretrained_configuration(model_configuration=model_configuration,
                                             section_name="SegmentationDecoder")
    return errors


def check_segmentation_loss_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'SegmentationLoss' section: the non-negative weights of the cross entropy and of the Dice loss, not
    both 0, the focusing parameter of the focal factor, the label smoothing and the class weighting of the cross
    entropy.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Weights of the two losses and focusing parameter, non-negative numbers
    for key_name in ["cross_entropy_weight", "dice_weight", "focal_gamma"]:
        key_path = f"SegmentationLoss.{key_name}"
        value = get_configuration_value(configuration=model_configuration, key_path=key_path)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            errors.append(f"'{key_path}' must be a number >= 0, got {value!r}.")
    cross_entropy_weight = get_configuration_value(configuration=model_configuration,
                                                   key_path="SegmentationLoss.cross_entropy_weight")
    dice_weight = get_configuration_value(configuration=model_configuration, key_path="SegmentationLoss.dice_weight")
    if cross_entropy_weight == 0 and dice_weight == 0:
        errors.append("'SegmentationLoss.cross_entropy_weight' and 'SegmentationLoss.dice_weight' cannot both be 0, "
                      "the model would have no loss.")

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
    mask of every test image is saved. The number of sample images, shared by every task, is checked with the
    experiment settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}
    return check_boolean_flags(configuration=configuration,
                               key_paths=["ExperimentConfiguration.evaluation.save_predicted_masks"])


def check_segmentation_configuration(experiment_configuration: Dict[str, Any],
                                     model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the semantic segmentation task: its data settings, without the sequence settings,
    its decoder, its losses and its evaluation settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_segmentation_data_configuration(model_configuration=model_configuration)
    errors += check_image_task_sequence_settings(model_configuration=model_configuration)
    errors += check_segmentation_decoder_configuration(model_configuration=model_configuration)
    errors += check_segmentation_loss_configuration(model_configuration=model_configuration)
    errors += check_segmentation_evaluation_configuration(experiment_configuration=experiment_configuration)
    return errors
