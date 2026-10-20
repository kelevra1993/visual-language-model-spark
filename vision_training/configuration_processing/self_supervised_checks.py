"""
Checks of the configuration sections specific to the self-supervised task: the data settings (label mapping of the
k-nearest neighbour evaluation, the views of 'Data.view_augmentation' replacing the data augmentation, checked by
check_view_augmentation_configuration of common_checks.py, shared with the view losses of the image classifier), the
grouped batches of 'Data.grouped_batches', the projection head with its method (SimCLR or Barlow Twins) and the
evaluation settings. Run by configuration_checker.py for the self-supervised task.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.classification_checks import (VALID_GLOBAL_POOLINGS,
                                                                            VALID_HEAD_NORMALIZATIONS)
from vision_training.configuration_processing.common_checks import (check_image_task_sequence_settings,
                                                                    check_view_augmentation_configuration)
from vision_training.configuration_processing.configuration_values import (check_batch_normalization_momentum,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           get_configuration_value, is_number)
from vision_training.losses.self_supervised_losses import SELF_SUPERVISED_METHODS


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


def check_grouped_batches_configuration(experiment_configuration: Dict[str, Any],
                                        model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Data.grouped_batches' section of the self-supervised task, whose batches gather blocks of images
    sharing an attribute of their "groups" (vision_training/data/grouped_batch_sampler.py): the activated flag, the
    block key (a string), the batch key (null or a string) and the number of images of a block, an integer of at least
    2 that divides the batch size when the section is on, so that a batch holds a whole number of blocks while a batch
    size changed with the section off needs no change of the section.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration, giving the batch size.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_boolean_flags(configuration=model_configuration, key_paths=["Data.grouped_batches.activated"])
    activated = get_configuration_value(configuration=model_configuration, key_path="Data.grouped_batches.activated")

    # Attribute shared by the images of a block, and the one shared by every image of a batch when given
    block_key = get_configuration_value(configuration=model_configuration, key_path="Data.grouped_batches.block_key")
    if not isinstance(block_key, str) or not block_key:
        errors.append(f"'Data.grouped_batches.block_key' must be a non-empty string, got {block_key!r}.")
    batch_key = get_configuration_value(configuration=model_configuration, key_path="Data.grouped_batches.batch_key")
    if batch_key is not None and (not isinstance(batch_key, str) or not batch_key):
        errors.append(f"'Data.grouped_batches.batch_key' must be null or a non-empty string, got {batch_key!r}.")

    # Images of a block, a whole number of blocks filling a batch when the section is on, the batch size being checked
    # by the common checks
    images_per_block = get_configuration_value(configuration=model_configuration,
                                               key_path="Data.grouped_batches.images_per_block")
    batch_size = experiment_configuration["batch_size"]
    batch_size_valid = isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size >= 1
    if isinstance(images_per_block, bool) or not isinstance(images_per_block, int) or images_per_block < 2:
        errors.append(f"'Data.grouped_batches.images_per_block' must be an integer >= 2, got {images_per_block!r}.")
    elif activated is True and batch_size_valid and batch_size % images_per_block != 0:
        errors.append(f"'Data.grouped_batches.images_per_block' ({images_per_block}) must divide "
                      f"'ExperimentConfiguration.batch_size' ({batch_size}) when the grouped batches are on: a batch "
                      f"holds a whole number of blocks.")
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
    augmentation, its grouped batches, its projection head and its evaluation settings.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    errors += check_self_supervised_data_configuration(model_configuration=model_configuration)
    errors += check_image_task_sequence_settings(model_configuration=model_configuration)
    errors += check_view_augmentation_configuration(model_configuration=model_configuration,
                                                    key_prefix="Data.view_augmentation")
    errors += check_grouped_batches_configuration(experiment_configuration=experiment_configuration,
                                                  model_configuration=model_configuration)
    errors += check_projection_head_configuration(model_configuration=model_configuration)
    errors += check_self_supervised_evaluation_configuration(experiment_configuration=experiment_configuration)
    return errors
