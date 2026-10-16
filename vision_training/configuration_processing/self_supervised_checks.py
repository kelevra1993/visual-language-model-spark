"""
Checks of the configuration sections specific to the self-supervised task: the data settings (label mapping of the
k-nearest neighbour evaluation, the views of 'Data.view_augmentation' replacing the data augmentation), the projection
head with its method (SimCLR or Barlow Twins) and the evaluation settings. Run by configuration_checker.py for the
self-supervised task.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.classification_checks import (VALID_GLOBAL_POOLINGS,
                                                                            VALID_HEAD_NORMALIZATIONS)
from vision_training.configuration_processing.common_checks import check_image_task_sequence_settings
from vision_training.configuration_processing.configuration_values import (check_batch_normalization_momentum,
                                                                           check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value, is_number)
from vision_training.losses.self_supervised_losses import SELF_SUPERVISED_METHODS

# Transformations of the views drawn with a probability, under 'Data.view_augmentation.<name>.probability'
VIEW_TRANSFORMATIONS_WITH_PROBABILITY = ["left_right_flip", "up_down_flip", "rotation", "colour_jitter", "grayscale",
                                         "gaussian_blur"]

# Largest hue shift of the colour jitter, half a turn of the hue circle
MAXIMUM_HUE_SHIFT = 0.5

# Largest rotation angle of the views, in degrees
MAXIMUM_ROTATION_DEGREES = 180.0


def check_number_range(value: Any, key_path: str, minimum_value: float, maximum_value: float) -> List[str]:
    """
    Verifies that a [minimum, maximum] range of the view augmentation (crop scale, aspect ratio, blur sigma) is a list
    of two numbers in increasing order, strictly above the minimum value and at most the maximum value.

    Args:
        value (Any): The range read from the configuration.
        key_path (str): The dotted key path of the range, used in the error message.
        minimum_value (float): The value both bounds must be strictly above.
        maximum_value (float): The value both bounds must not exceed.

    Returns:
        List[str]: A single error description if the range is invalid, otherwise an empty list.
    """
    if not isinstance(value, list) or len(value) != 2 or not all(is_number(value=bound) for bound in value) or \
            not minimum_value < value[0] <= value[1] <= maximum_value:
        return [f"'{key_path}' must be a list of two numbers [low, high] with {minimum_value:g} < low <= high <= "
                f"{maximum_value:g}, got {value!r}."]
    return []


def check_self_supervised_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the data settings of the self-supervised task: a multi-class label mapping whose class indices are 0 to
    K-1 (the classes of the k-nearest neighbour evaluation, possibly none for a dataset without labels), and the
    data augmentation off, the views of 'Data.view_augmentation' replacing it.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # The evaluation classes are the mapped classes, indexed from 0 without gaps
    label_mode = get_configuration_value(configuration=model_configuration, key_path="Data.label_mapping.mode")
    if label_mode != "multi_class":
        errors.append(f"'Data.label_mapping.mode' must be 'multi_class' for the self-supervised task (the mapping file "
                      f"gives the class of every category for the k-nearest neighbour evaluation), got {label_mode!r}.")
    else:
        class_indices = sorted(set(get_configuration_value(configuration=model_configuration,
                                                           key_path="Data.label_dictionary").values()))
        if class_indices != list(range(len(class_indices))):
            errors.append(f"The classes of the label mapping file must be indexed 0 to K-1 without gaps, "
                          f"got {class_indices}.")

    # The views replace the data augmentation of the other tasks
    if get_configuration_value(configuration=model_configuration, key_path="Data.data_augmentation.activated") is True:
        errors.append("'Data.data_augmentation.activated' must be False for the self-supervised task: the random views "
                      "of 'Data.view_augmentation' replace the data augmentation.")
    return errors


def check_view_augmentation_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Data.view_augmentation' section drawing the two views of every image: the area and aspect ratio
    ranges of the random resized crop, the probability of every other transformation, the largest rotation angle, the
    strengths of the colour jitter and the kernel size and sigma range of the Gaussian blur, so that kornia receives
    valid parameters.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    key_prefix = "Data.view_augmentation"

    # Area of the crop as a fraction of the image, and aspect ratio of the crop
    errors += check_number_range(value=get_configuration_value(configuration=model_configuration,
                                                               key_path=f"{key_prefix}.random_resized_crop.scale"),
                                 key_path=f"{key_prefix}.random_resized_crop.scale", minimum_value=0.0,
                                 maximum_value=1.0)
    errors += check_number_range(value=get_configuration_value(
                                     configuration=model_configuration,
                                     key_path=f"{key_prefix}.random_resized_crop.aspect_ratio"),
                                 key_path=f"{key_prefix}.random_resized_crop.aspect_ratio", minimum_value=0.0,
                                 maximum_value=float("inf"))

    # Probability of every transformation applied to a part of the views
    for transformation_name in VIEW_TRANSFORMATIONS_WITH_PROBABILITY:
        key_path = f"{key_prefix}.{transformation_name}.probability"
        errors += check_probability_value(value=get_configuration_value(configuration=model_configuration,
                                                                        key_path=key_path), key_path=key_path)

    # Largest rotation angle, in either direction
    maximum_degrees = get_configuration_value(configuration=model_configuration,
                                              key_path=f"{key_prefix}.rotation.maximum_degrees")
    if not is_number(value=maximum_degrees) or not 0.0 <= maximum_degrees <= MAXIMUM_ROTATION_DEGREES:
        errors.append(f"'{key_prefix}.rotation.maximum_degrees' must be a number within [0, "
                      f"{MAXIMUM_ROTATION_DEGREES:g}], got {maximum_degrees!r}.")

    # Strengths of the colour jitter, the hue shift being at most half a turn
    for strength_name in ["brightness", "contrast", "saturation"]:
        key_path = f"{key_prefix}.colour_jitter.{strength_name}"
        strength = get_configuration_value(configuration=model_configuration, key_path=key_path)
        if not is_number(value=strength) or strength < 0.0:
            errors.append(f"'{key_path}' must be a number >= 0, got {strength!r}.")
    hue = get_configuration_value(configuration=model_configuration, key_path=f"{key_prefix}.colour_jitter.hue")
    if not is_number(value=hue) or not 0.0 <= hue <= MAXIMUM_HUE_SHIFT:
        errors.append(f"'{key_prefix}.colour_jitter.hue' must be a number within [0, {MAXIMUM_HUE_SHIFT:g}], "
                      f"got {hue!r}.")

    # Odd kernel of the Gaussian blur, centred on its pixel, and its range of sigmas
    kernel_size = get_configuration_value(configuration=model_configuration,
                                          key_path=f"{key_prefix}.gaussian_blur.kernel_size")
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        errors.append(f"'{key_prefix}.gaussian_blur.kernel_size' must be an odd integer >= 3, got {kernel_size!r}.")
    errors += check_number_range(value=get_configuration_value(configuration=model_configuration,
                                                               key_path=f"{key_prefix}.gaussian_blur.sigma"),
                                 key_path=f"{key_prefix}.gaussian_blur.sigma", minimum_value=0.0,
                                 maximum_value=float("inf"))
    return errors


def check_projection_head_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'ProjectionHead' section: the global pooling, the layers (at least the embedding layer), the
    normalization of the hidden layers and its batch momentum, the self-supervised method and the settings of both
    methods, every one of them being required whatever the method so that a run can switch method by one key.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    global_pooling = get_configuration_value(configuration=model_configuration,
                                             key_path="ProjectionHead.global_pooling")
    if global_pooling not in VALID_GLOBAL_POOLINGS:
        errors.append(f"'ProjectionHead.global_pooling' must be one of {VALID_GLOBAL_POOLINGS}, "
                      f"got {global_pooling!r}.")

    # Layers of the head, the last one being the embedding, each with a positive number of units
    fully_connected = get_configuration_value(configuration=model_configuration,
                                              key_path="ProjectionHead.fully_connected")
    if not isinstance(fully_connected, list) or not fully_connected or not all(
            isinstance(units, int) and not isinstance(units, bool) and units >= 1 for units in fully_connected):
        errors.append(f"'ProjectionHead.fully_connected' must be a non-empty list of positive integers, the last one "
                      f"being the embedding size, got {fully_connected!r}.")

    # Normalization of the hidden layers, which needs at least one of them
    fully_connected_normalization = get_configuration_value(
        configuration=model_configuration, key_path="ProjectionHead.fully_connected_normalization")
    if fully_connected_normalization not in VALID_HEAD_NORMALIZATIONS:
        errors.append(f"'ProjectionHead.fully_connected_normalization' must be one of {VALID_HEAD_NORMALIZATIONS}, "
                      f"got {fully_connected_normalization!r}.")
    elif fully_connected_normalization != "none" and isinstance(fully_connected, list) and len(fully_connected) == 1:
        errors.append("'ProjectionHead.fully_connected_normalization' must be 'none' when "
                      "'ProjectionHead.fully_connected' holds the embedding layer only: the head has no hidden layer "
                      "to normalize.")
    errors += check_batch_normalization_momentum(configuration=model_configuration,
                                                 key_path="ProjectionHead.batch_normalization_momentum")

    # Method of the run, and the strictly positive setting of each method
    method = get_configuration_value(configuration=model_configuration, key_path="ProjectionHead.method")
    if method not in SELF_SUPERVISED_METHODS:
        errors.append(f"'ProjectionHead.method' must be one of {SELF_SUPERVISED_METHODS}, got {method!r}.")
    for key_path in ["ProjectionHead.simclr.temperature", "ProjectionHead.barlow_twins.off_diagonal_weight"]:
        value = get_configuration_value(configuration=model_configuration, key_path=key_path)
        if not is_number(value=value) or value <= 0.0:
            errors.append(f"'{key_path}' must be a number > 0, got {value!r}.")
    return errors


def check_self_supervised_evaluation_configuration(experiment_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the settings of the self-supervised task in 'ExperimentConfiguration': the number of neighbours voting
    in the k-nearest neighbour evaluation, and a batch of at least two images, the losses comparing the views of the
    images of a batch with each other.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}
    errors = check_positive_integers(configuration=configuration,
                                     key_paths=["ExperimentConfiguration.evaluation.number_neighbours"],
                                     allow_zero=False)
    batch_size = get_configuration_value(configuration=configuration, key_path="ExperimentConfiguration.batch_size")
    if isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size == 1:
        errors.append("'ExperimentConfiguration.batch_size' must be at least 2 for the self-supervised task: the "
                      "losses compare the views of the images of a batch with each other.")
    return errors


def check_self_supervised_configuration(experiment_configuration: Dict[str, Any],
                                        model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the self-supervised task: its data settings without the sequence settings, its view
    augmentation, its projection head and its evaluation settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_self_supervised_data_configuration(model_configuration=model_configuration)
    errors += check_image_task_sequence_settings(model_configuration=model_configuration)
    errors += check_view_augmentation_configuration(model_configuration=model_configuration)
    errors += check_projection_head_configuration(model_configuration=model_configuration)
    errors += check_self_supervised_evaluation_configuration(experiment_configuration=experiment_configuration)
    return errors
