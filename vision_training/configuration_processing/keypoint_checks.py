"""
Checks of the configuration sections specific to the keypoint detection task: the crop size and the crop augmentation
of the 'Data' section with the keypoint definitions read from the training split, the Backbone stem ending at a quarter
of the crop, the 'HRNet' and 'KeypointHead' sections, and the object keypoint similarity settings of the evaluation.
Run by configuration_checker.py for the keypoint_detection task.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.common_checks import (check_image_task_sequence_settings,
                                                                    check_pretrained_configuration)
from vision_training.configuration_processing.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                           check_batch_normalization_momentum,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value, is_number)
from vision_training.data.augmentations import GEOMETRIC_AUGMENTATION_TYPES
from vision_training.evaluation.keypoints.object_keypoint_similarity import build_keypoint_sigma_table
from vision_training.models.keypoints.hrnet import BRANCH_SCALE_FACTOR
from vision_training.models.keypoints.keypoint_detector import HEATMAP_STRIDE


# Largest rotation of the crop augmentation in degrees, a half turn either way covering every orientation
MAXIMUM_CROP_ROTATION_DEGREES = 180.0

# Largest shift of the crop centre, as a fraction of the sides of the crop region, beyond which the object can leave
# the crop
MAXIMUM_CROP_SHIFT_FRACTION = 0.5


def list_category_names(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the names of the categories of the training split whose keypoints are learnt, the categories merged into one
    class being joined with '+' in their injected definition, so that the flip pairs and the sigmas given by category
    name can be verified.

    Args:
        model_configuration (Dict[str, Any]): The model configuration, with the 'Data.keypoint_categories' injected by
            the configuration loader when the training split could be read.

    Returns:
        List[str]: The category names, empty when the keypoint definitions could not be read.
    """
    if "keypoint_categories" not in model_configuration["Data"]:
        return []
    return [name for definition in model_configuration["Data"]["keypoint_categories"].values()
            for name in definition["name"].split("+")]


def check_crop_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies 'Data.crop_settings', the crops every object is cut to before the keypoint detector: their height and
    width, divisible by the stride of the lowest resolution branch of the high resolution network so that the
    upsampled branches add up cell by cell, and the enlargement of the box keeping some context around the object.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Data' and 'HRNet' sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_positive_integers(configuration=model_configuration,
                                     key_paths=["Data.crop_settings.height", "Data.crop_settings.width"],
                                     allow_zero=False)

    # The lowest resolution branch halves the quarter resolution feature map once per stage
    modules_per_stage = get_configuration_value(configuration=model_configuration, key_path="HRNet.modules_per_stage")
    if not errors and isinstance(modules_per_stage, list) and modules_per_stage:
        lowest_branch_stride = HEATMAP_STRIDE * BRANCH_SCALE_FACTOR ** len(modules_per_stage)
        for side_name in ["height", "width"]:
            side = get_configuration_value(configuration=model_configuration,
                                           key_path=f"Data.crop_settings.{side_name}")
            if side % lowest_branch_stride != 0:
                errors.append(f"'Data.crop_settings.{side_name}' ({side}) must be a multiple of "
                              f"{lowest_branch_stride}, the stride of the lowest resolution branch of the "
                              f"{len(modules_per_stage)} stages of 'HRNet.modules_per_stage'.")

    # The crop region covers at least the box
    box_enlargement = get_configuration_value(configuration=model_configuration,
                                              key_path="Data.crop_settings.box_enlargement")
    if not is_number(value=box_enlargement) or box_enlargement < 1.0:
        errors.append(f"'Data.crop_settings.box_enlargement' must be a number >= 1, got {box_enlargement!r}.")
    return errors


def check_crop_augmentation_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies 'Data.crop_augmentation', the geometric augmentation drawn in the warp of every training crop (the
    geometric types of 'Data.data_augmentation' would move the crops without their keypoints, so they must be off):
    the range of the scale of the region, the largest rotation and its probability, the flip probability, the largest
    shift of the region centre and the flip pairs given by category name, replacing the pairs deduced from the
    left and right keypoint names.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Data' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # The photometric types only run on the crops, after the warp
    for augmentation_type in GEOMETRIC_AUGMENTATION_TYPES:
        key_path = f"Data.data_augmentation.types.{augmentation_type}.activated"
        if get_configuration_value(configuration=model_configuration, key_path=key_path) is True:
            errors.append(f"'{key_path}' must be False for the keypoint detection: the geometry of the crops is drawn "
                          f"by 'Data.crop_augmentation', which moves their keypoints with them.")

    # Range of the scale of the crop region, above 1 the object looking smaller
    scale_range = get_configuration_value(configuration=model_configuration,
                                          key_path="Data.crop_augmentation.scale_range")
    if not isinstance(scale_range, list) or len(scale_range) != 2 or \
            not all(is_number(value=scale) and scale > 0 for scale in scale_range) or scale_range[0] > scale_range[1]:
        errors.append(f"'Data.crop_augmentation.scale_range' must be [minimum, maximum] with positive numbers and "
                      f"minimum <= maximum, got {scale_range!r}.")

    # Largest rotation, its probability and the flip probability
    maximum_rotation = get_configuration_value(configuration=model_configuration,
                                               key_path="Data.crop_augmentation.maximum_rotation_degrees")
    if not is_number(value=maximum_rotation) or not 0.0 <= maximum_rotation <= MAXIMUM_CROP_ROTATION_DEGREES:
        errors.append(f"'Data.crop_augmentation.maximum_rotation_degrees' must be a number within "
                      f"[0, {MAXIMUM_CROP_ROTATION_DEGREES:g}], got {maximum_rotation!r}.")
    for key_path in ["Data.crop_augmentation.rotation_probability", "Data.crop_augmentation.flip_probability"]:
        errors += check_probability_value(value=get_configuration_value(configuration=model_configuration,
                                                                        key_path=key_path), key_path=key_path)

    # Largest shift of the region centre, as a fraction of its sides
    maximum_shift = get_configuration_value(configuration=model_configuration,
                                            key_path="Data.crop_augmentation.maximum_shift_fraction")
    if not is_number(value=maximum_shift) or not 0.0 <= maximum_shift <= MAXIMUM_CROP_SHIFT_FRACTION:
        errors.append(f"'Data.crop_augmentation.maximum_shift_fraction' must be a number within "
                      f"[0, {MAXIMUM_CROP_SHIFT_FRACTION:g}], got {maximum_shift!r}.")

    # Flip pairs given by hand: pairs of keypoint names of a category of the training split
    flip_pairs = get_configuration_value(configuration=model_configuration,
                                         key_path="Data.crop_augmentation.flip_pairs")
    if not isinstance(flip_pairs, dict):
        return errors + [f"'Data.crop_augmentation.flip_pairs' must map category names to lists of "
                         f"[left keypoint, right keypoint] pairs ({{}} for none), got {flip_pairs!r}."]
    category_names = list_category_names(model_configuration=model_configuration)
    for category_name, category_pairs in flip_pairs.items():
        if category_names and category_name not in category_names:
            errors.append(f"'Data.crop_augmentation.flip_pairs' names the category {category_name!r}, which is not "
                          f"a learnt category of the training split ({category_names}).")
        if not isinstance(category_pairs, list) or \
                not all(isinstance(flip_pair, list) and len(flip_pair) == 2 and
                        all(isinstance(name, str) for name in flip_pair) for flip_pair in category_pairs):
            errors.append(f"'Data.crop_augmentation.flip_pairs.{category_name}' must be a list of [left keypoint, "
                          f"right keypoint] name pairs, got {category_pairs!r}.")
    return errors


def check_keypoint_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Data' settings of the keypoint detection that are not about the crops: the multi-class label
    mapping, each class having its own keypoints, and the keypoint definitions injected by the configuration loader
    from the 'categories' of the training split, one for every foreground class.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Data' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    label_mode = get_configuration_value(configuration=model_configuration, key_path="Data.label_mapping.mode")
    if label_mode != "multi_class":
        errors.append(f"'Data.label_mapping.mode' must be 'multi_class' for the keypoint detection, every class having "
                      f"its own keypoints, got {label_mode!r}.")
        return errors

    # The definitions are read from the 'labels.json' of the training split by the configuration loader
    data_configuration = model_configuration["Data"]
    if "keypoint_categories" not in data_configuration:
        errors.append("The keypoint definitions of the classes could not be read: the 'labels.json' of "
                      "'ExperimentConfiguration.train_data_folder' must exist, with the 'keypoints' names of every "
                      "category, and 'Data.crop_augmentation.flip_pairs' must be a mapping.")
        return errors

    # Every foreground class of the label dictionary has its keypoint definition
    if "label_dictionary" in data_configuration:
        foreground_classes = sorted(set(data_configuration["label_dictionary"].values()) - {0})
        missing_classes = [class_index for class_index in foreground_classes
                           if class_index not in data_configuration["keypoint_categories"]]
        if missing_classes:
            errors.append(f"The classes {missing_classes} of 'Data.label_mapping' have no category with keypoints in "
                          f"the 'labels.json' of the training split.")
    return errors


def check_keypoint_stem_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies that the Backbone, used as the stem of the keypoint detector, ends at a quarter of the crop resolution,
    the resolution of the highest resolution branch and of the heatmaps: its blocks are followed by max pooling layers
    except the last one, unless 'Backbone.last_max_pooling' adds it, so that 2 pooling layers are needed.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Backbone' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    convolutions = get_configuration_value(configuration=model_configuration, key_path="Backbone.convolutions")
    last_max_pooling = get_configuration_value(configuration=model_configuration, key_path="Backbone.last_max_pooling")
    if not isinstance(convolutions, dict) or not isinstance(last_max_pooling, bool):
        return []
    number_pooling_layers = len(convolutions) - 1 + int(last_max_pooling)
    if BRANCH_SCALE_FACTOR ** number_pooling_layers != HEATMAP_STRIDE:
        return [f"The Backbone stem of the keypoint detector must end at 1/{HEATMAP_STRIDE} of the crop, got 1/"
                f"{BRANCH_SCALE_FACTOR ** number_pooling_layers} from its {len(convolutions)} blocks with "
                f"'Backbone.last_max_pooling' {last_max_pooling}: use 3 blocks without the last max pooling, or 2 "
                f"blocks with it."]
    return []


def check_high_resolution_network_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'HRNet' section, the body of the keypoint detector on the stem: the width of its highest resolution
    branch, the number of exchange modules of every stage (one branch added per stage), the residual pairs of every
    branch, the modules and normalization of its convolutions, and the 'pretrained' settings initialising it with the
    one of another run; and the 'KeypointHead' section, the standard deviation of the target Gaussians in heatmap cells
    and its 'pretrained' settings.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'HRNet' and 'KeypointHead'
            sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_positive_integers(configuration=model_configuration,
                                     key_paths=["HRNet.base_width", "HRNet.blocks_per_branch"], allow_zero=False)

    # One stage per added branch, each with at least one exchange module
    modules_per_stage = get_configuration_value(configuration=model_configuration, key_path="HRNet.modules_per_stage")
    if not isinstance(modules_per_stage, list) or not modules_per_stage or \
            not all(isinstance(value, int) and not isinstance(value, bool) and value >= 1
                    for value in modules_per_stage):
        errors.append(f"'HRNet.modules_per_stage' must be a non-empty list of positive integers, the exchange modules "
                      f"of every stage, got {modules_per_stage!r}.")

    # Modules and normalization of the branch convolutions
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["HRNet.modules.squeeze_and_excite", "HRNet.modules.residual_nets"])
    feature_map_normalization = get_configuration_value(configuration=model_configuration,
                                                        key_path="HRNet.normalization.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'HRNet.normalization.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")
    errors += check_batch_normalization_momentum(configuration=model_configuration,
                                                 key_path="HRNet.normalization.batch_normalization_momentum")
    errors += check_pretrained_configuration(model_configuration=model_configuration, section_name="HRNet")

    # Width of the target Gaussians and transfer of the heatmap head
    heatmap_sigma = get_configuration_value(configuration=model_configuration, key_path="KeypointHead.heatmap_sigma")
    if not is_number(value=heatmap_sigma) or heatmap_sigma <= 0:
        errors.append(f"'KeypointHead.heatmap_sigma' must be a positive number of heatmap cells, "
                      f"got {heatmap_sigma!r}.")
    errors += check_pretrained_configuration(model_configuration=model_configuration, section_name="KeypointHead")
    return errors


def check_keypoint_evaluation_configuration(experiment_configuration: Dict[str, Any],
                                            model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the object keypoint similarity settings of 'ExperimentConfiguration.evaluation': the default standard
    deviation of the keypoints, as a fraction of the square root of the box area, and the per-keypoint values given by
    category and keypoint name, the keypoints left out keeping the default, the categories merged into one class having
    to agree on the values of their keypoints.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The model configuration, with the keypoint definitions naming the
            keypoints.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}
    errors = []
    default_sigma = get_configuration_value(configuration=configuration,
                                            key_path="ExperimentConfiguration.evaluation.default_keypoint_sigma")
    if not is_number(value=default_sigma) or default_sigma <= 0:
        errors.append(f"'ExperimentConfiguration.evaluation.default_keypoint_sigma' must be a positive number, "
                      f"got {default_sigma!r}.")

    # Per-keypoint sigmas by category name, then keypoint name
    keypoint_sigmas = get_configuration_value(configuration=configuration,
                                              key_path="ExperimentConfiguration.evaluation.keypoint_sigmas")
    if not isinstance(keypoint_sigmas, dict):
        return errors + [f"'ExperimentConfiguration.evaluation.keypoint_sigmas' must map category names to "
                         f"{{keypoint name: sigma}} mappings ({{}} for none), got {keypoint_sigmas!r}."]
    keypoint_names = {}
    if "keypoint_categories" in model_configuration["Data"]:
        for definition in model_configuration["Data"]["keypoint_categories"].values():
            for category_name in definition["name"].split("+"):
                keypoint_names[category_name] = definition["keypoints"]
    for category_name, category_sigmas in keypoint_sigmas.items():
        key_path = f"ExperimentConfiguration.evaluation.keypoint_sigmas.{category_name}"
        if keypoint_names and category_name not in keypoint_names:
            errors.append(f"'{key_path}' names a category that is not learnt from the training split "
                          f"({sorted(keypoint_names)}).")
            continue
        if not isinstance(category_sigmas, dict) or \
                not all(is_number(value=sigma) and sigma > 0 for sigma in category_sigmas.values()):
            errors.append(f"'{key_path}' must map keypoint names to positive numbers, got {category_sigmas!r}.")
            continue
        unknown_names = [name for name in category_sigmas
                         if category_name in keypoint_names and name not in keypoint_names[category_name]]
        if unknown_names:
            errors.append(f"'{key_path}' names unknown keypoints {unknown_names}, the keypoints of the category being "
                          f"{keypoint_names[category_name]}.")

    # The categories merged into one class share its sigmas, so they cannot give different values to a keypoint
    if not errors and keypoint_names:
        keypoint_categories = model_configuration["Data"]["keypoint_categories"]
        try:
            build_keypoint_sigma_table(keypoint_categories=keypoint_categories, default_sigma=default_sigma,
                                       keypoint_sigmas=keypoint_sigmas,
                                       maximum_keypoints=max(len(definition["keypoints"])
                                                             for definition in keypoint_categories.values()))
        except ValueError as error:
            errors.append(str(error))
    return errors


def check_keypoint_configuration(experiment_configuration: Dict[str, Any],
                                 model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the keypoint detection task: its data settings without the sequence settings, its
    crops and crop augmentation, the stem resolution of its Backbone, its high resolution network and keypoint head, and
    its evaluation settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_image_task_sequence_settings(model_configuration=model_configuration)
    errors += check_keypoint_data_configuration(model_configuration=model_configuration)
    errors += check_crop_configuration(model_configuration=model_configuration)
    errors += check_crop_augmentation_configuration(model_configuration=model_configuration)
    errors += check_keypoint_stem_configuration(model_configuration=model_configuration)
    errors += check_high_resolution_network_configuration(model_configuration=model_configuration)
    errors += check_keypoint_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                      model_configuration=model_configuration)
    return errors
