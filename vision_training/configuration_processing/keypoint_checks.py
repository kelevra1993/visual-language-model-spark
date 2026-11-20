"""
Checks of the configuration sections specific to the keypoint detection task: the keypoint definitions of the 'Data'
section read from the training split, the Backbone stem ending at a quarter of the image, the 'HRNet' section with the
image size its lowest resolution branch divides, the 'KeypointHead' section (higher resolution branch, hidden width,
ignored regions, tag loss weights and grouping settings), and the object keypoint similarity settings of the
evaluation. Run by configuration_checker.py for the keypoint_detection task. The geometric augmentations of
'Data.data_augmentation' are allowed, the augmenter moving the keypoints with the image, but the perspective and the
jigsaw, which move the image only (IMAGE_ONLY_AUGMENTATION_TYPES).
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.common_checks import (check_image_only_augmentation_types,
                                                                    check_image_task_sequence_settings,
                                                                    check_pretrained_configuration)
from vision_training.configuration_processing.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           get_configuration_value, is_number)
from vision_training.evaluation.keypoints.object_keypoint_similarity import build_keypoint_sigma_table
from vision_training.models.keypoints.hrnet import BRANCH_SCALE_FACTOR
from vision_training.models.keypoints.keypoint_detector import HEATMAP_STRIDE


def check_keypoint_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Data' settings of the keypoint detection: the multi-class label mapping, each class having its own
    keypoints, with at least two classes indexed 0 (the background) to K-1 without gaps, so that the class names of the
    evaluation and of the reports cover every class the targets carry (a gap would leave a class unnamed and outside the
    per-class metrics), and the keypoint definitions injected by the configuration loader from the 'categories' of the
    training split, one for every foreground class.

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

    # Classes indexed from the background 0 without gaps, as the class names of the run are built for 0 to K-1
    data_configuration = model_configuration["Data"]
    if "label_dictionary" in data_configuration:
        class_indices = sorted(set(data_configuration["label_dictionary"].values()))
        if len(class_indices) < 2 or class_indices != list(range(len(class_indices))):
            errors.append(f"The classes of the label mapping must be at least two, indexed 0 (background) to K-1 "
                          f"without gaps, got {class_indices}.")

    # The definitions are read from the 'labels.json' of the training split by the configuration loader
    if "keypoint_categories" not in data_configuration:
        errors.append("The keypoint definitions of the classes could not be read: the 'labels.json' of "
                      "'ExperimentConfiguration.train_data_folder' must exist, with the 'keypoints' names of every "
                      "category.")
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
    Verifies that the Backbone, used as the stem of the keypoint detector, ends at a quarter of the image resolution,
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
        return [f"The Backbone stem of the keypoint detector must end at 1/{HEATMAP_STRIDE} of the image, got 1/"
                f"{BRANCH_SCALE_FACTOR ** number_pooling_layers} from its {len(convolutions)} blocks with "
                f"'Backbone.last_max_pooling' {last_max_pooling}: use 3 blocks without the last max pooling, or 2 "
                f"blocks with it."]
    return []


def check_high_resolution_network_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'HRNet' section, the body of the keypoint detector on the stem: the width of its highest resolution
    branch, the number of exchange modules of every stage (one branch added per stage), which sets the stride of its
    lowest resolution branch that the image size must be a multiple of so that the upsampled branches add up cell by
    cell, the residual pairs of every branch, the modules and normalization of its convolutions, and the 'pretrained'
    settings initialising it with the one of another run.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'HRNet' and 'Data' sections.

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
    else:
        # The lowest resolution branch halves the quarter resolution feature map once per stage
        lowest_branch_stride = HEATMAP_STRIDE * BRANCH_SCALE_FACTOR ** len(modules_per_stage)
        image_size = get_configuration_value(configuration=model_configuration, key_path="Data.image_settings.size")
        if isinstance(image_size, int) and not isinstance(image_size, bool) and \
                image_size % lowest_branch_stride != 0:
            errors.append(f"'Data.image_settings.size' ({image_size}) must be a multiple of {lowest_branch_stride}, "
                          f"the stride of the lowest resolution branch of the {len(modules_per_stage)} stages of "
                          f"'HRNet.modules_per_stage'.")

    # Modules and normalization of the branch convolutions
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["HRNet.modules.squeeze_and_excite", "HRNet.modules.residual_nets"])
    feature_map_normalization = get_configuration_value(configuration=model_configuration,
                                                        key_path="HRNet.normalization.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'HRNet.normalization.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")
    errors += check_pretrained_configuration(model_configuration=model_configuration, section_name="HRNet")
    return errors


def check_keypoint_head_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'KeypointHead' section of the bottom-up keypoint detector: the higher resolution branch flag, the
    width of the hidden convolution of the heads, the standard deviation of the target Gaussians in heatmap cells, the
    flag leaving the objects without labelled keypoint out of the heatmap loss, the weights of the tag pull and push
    losses, the grouping settings of 'KeypointHead.inference' used by the evaluation, the sample images and the
    deployment, and the 'pretrained' settings initialising the head with the one of another run.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'KeypointHead' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_boolean_flags(configuration=model_configuration,
                                 key_paths=["KeypointHead.higher_resolution", "KeypointHead.ignore_unlabelled_objects"])

    # Width of the hidden convolution of the heads, 0 for the 1x1 predictions read on the features
    errors += check_positive_integers(configuration=model_configuration, key_paths=["KeypointHead.hidden_width"],
                                      allow_zero=True)

    # Width of the target Gaussians
    heatmap_sigma = get_configuration_value(configuration=model_configuration, key_path="KeypointHead.heatmap_sigma")
    if not is_number(value=heatmap_sigma) or heatmap_sigma <= 0:
        errors.append(f"'KeypointHead.heatmap_sigma' must be a positive number of heatmap cells, "
                      f"got {heatmap_sigma!r}.")

    # Weights of the tag losses, 0 leaving the tags untrained
    for key_path in ["KeypointHead.tag_pull_weight", "KeypointHead.tag_push_weight"]:
        tag_weight = get_configuration_value(configuration=model_configuration, key_path=key_path)
        if not is_number(value=tag_weight) or tag_weight < 0:
            errors.append(f"'{key_path}' must be a number >= 0, got {tag_weight!r}.")

    # Candidates of every channel and thresholds of the grouping
    errors += check_positive_integers(configuration=model_configuration,
                                      key_paths=["KeypointHead.inference.maximum_candidates"], allow_zero=False)
    keypoint_threshold = get_configuration_value(configuration=model_configuration,
                                                 key_path="KeypointHead.inference.keypoint_threshold")
    if not is_number(value=keypoint_threshold) or not 0.0 < keypoint_threshold < 1.0:
        errors.append(f"'KeypointHead.inference.keypoint_threshold' must be a number within ]0, 1[, "
                      f"got {keypoint_threshold!r}.")
    tag_threshold = get_configuration_value(configuration=model_configuration,
                                            key_path="KeypointHead.inference.tag_threshold")
    if not is_number(value=tag_threshold) or tag_threshold <= 0:
        errors.append(f"'KeypointHead.inference.tag_threshold' must be a positive number, got {tag_threshold!r}.")
    object_score_threshold = get_configuration_value(configuration=model_configuration,
                                                     key_path="KeypointHead.inference.object_score_threshold")
    if not is_number(value=object_score_threshold) or not 0.0 <= object_score_threshold <= 1.0:
        errors.append(f"'KeypointHead.inference.object_score_threshold' must be a number within [0, 1], "
                      f"got {object_score_threshold!r}.")

    # Transfer of the head from another run
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
    # The keypoint names of every category, a merged class "a+b" giving its keypoints to each of its categories
    keypoint_names = {}
    if "keypoint_categories" in model_configuration["Data"]:
        for definition in model_configuration["Data"]["keypoint_categories"].values():
            for category_name in definition["name"].split("+"):
                keypoint_names[category_name] = definition["keypoints"]
    # Every category given must be learnt, with positive sigmas for keypoints it has
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
    Runs every check specific to the keypoint detection task: its data settings without the sequence settings nor the
    augmentation types moving the image only, the stem resolution of its Backbone, its high resolution network with the
    image size, its keypoint head, and its evaluation settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_image_task_sequence_settings(model_configuration=model_configuration)
    errors += check_image_only_augmentation_types(model_configuration=model_configuration)
    errors += check_keypoint_data_configuration(model_configuration=model_configuration)
    errors += check_keypoint_stem_configuration(model_configuration=model_configuration)
    errors += check_high_resolution_network_configuration(model_configuration=model_configuration)
    errors += check_keypoint_head_configuration(model_configuration=model_configuration)
    errors += check_keypoint_evaluation_configuration(experiment_configuration=experiment_configuration,
                                                      model_configuration=model_configuration)
    return errors
