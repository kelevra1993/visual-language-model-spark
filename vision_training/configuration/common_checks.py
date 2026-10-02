from typing import Any, Dict, List

from vision_training.configuration.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                check_boolean_flags,
                                                                check_positive_integers,
                                                                check_probability_value,
                                                                get_configuration_value)
from vision_training.data.augmentations import AUGMENTATION_TYPES


# Label mapping modes handled by the configuration loader
VALID_LABEL_MODES = ["binary", "multi_class"]


def check_experiment_configuration(experiment_configuration: Dict[str, Any], valid_tasks: List[str]) -> List[str]:
    """
    Verifies the task and the training loop settings of the 'ExperimentConfiguration' section consumed by the
    trainers.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        valid_tasks (List[str]): The tasks that can be trained, i.e. those with registered configuration checks.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}

    # Task trained by the configuration, selecting the trainer, the model and the task-specific checks
    task = get_configuration_value(configuration=configuration, key_path="ExperimentConfiguration.task")
    if task not in valid_tasks:
        errors.append(f"'ExperimentConfiguration.task' must be one of {valid_tasks}, got {task!r}.")

    # Counts driving the training loop, logging and checkpointing
    errors += check_positive_integers(configuration=configuration,
                                      key_paths=["ExperimentConfiguration.batch_size",
                                                 "ExperimentConfiguration.information_dump",
                                                 "ExperimentConfiguration.weight_saving_iterations",
                                                 "ExperimentConfiguration.number_iterations"], allow_zero=False)

    # Intermediate checkpoints (saved without test evaluation) must be more frequent than the evaluated ones
    intermediate_saving_errors = check_positive_integers(
        configuration=configuration, key_paths=["ExperimentConfiguration.intermediate_weight_saving_iterations"],
        allow_zero=False)
    errors += intermediate_saving_errors
    weight_saving_iterations = get_configuration_value(configuration=configuration,
                                                       key_path="ExperimentConfiguration.weight_saving_iterations")
    intermediate_weight_saving_iterations = get_configuration_value(
        configuration=configuration, key_path="ExperimentConfiguration.intermediate_weight_saving_iterations")
    if not intermediate_saving_errors and isinstance(weight_saving_iterations, int) and \
            intermediate_weight_saving_iterations >= weight_saving_iterations:
        errors.append(f"'ExperimentConfiguration.intermediate_weight_saving_iterations' "
                      f"({intermediate_weight_saving_iterations}) must be smaller than "
                      f"'ExperimentConfiguration.weight_saving_iterations' ({weight_saving_iterations}).")
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

    # Loader flags and global augmentation settings
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["Data.DALI.activated", "Data.image_settings.keep_ratio",
                                             "Data.data_augmentation.activated"])
    errors += check_probability_value(
        value=get_configuration_value(configuration=model_configuration, key_path="Data.data_augmentation.probability"),
        key_path="Data.data_augmentation.probability")

    # Every supported augmentation type must be configured, and no unknown type may be listed
    augmentation_types = get_configuration_value(configuration=model_configuration,
                                                 key_path="Data.data_augmentation.types")
    unknown_types = sorted(set(augmentation_types) - set(AUGMENTATION_TYPES))
    if unknown_types:
        errors.append(f"'Data.data_augmentation.types' contains unknown augmentation types {unknown_types}, "
                      f"supported types are {AUGMENTATION_TYPES}.")

    # Each augmentation type has its own activation flag and probability
    for augmentation_type in AUGMENTATION_TYPES:
        key_prefix = f"Data.data_augmentation.types.{augmentation_type}"
        errors += check_boolean_flags(configuration=model_configuration, key_paths=[f"{key_prefix}.activated"])
        errors += check_probability_value(
            value=get_configuration_value(configuration=model_configuration, key_path=f"{key_prefix}.probability"),
            key_path=f"{key_prefix}.probability")

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

    # Blocks feeding task heads (detection levels) must exist in the Backbone and be listed from the finest to the
    # coarsest level; the tasks whose heads use the final feature map list none
    detection_block_indices = get_configuration_value(configuration=model_configuration,
                                                      key_path="Backbone.detection_convolution_block_indices")
    if any(index not in range(1, number_blocks + 1) for index in detection_block_indices):
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

    errors += check_pretrained_backbone_configuration(model_configuration=model_configuration)
    return errors


def check_pretrained_backbone_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Backbone.pretrained' settings, which initialise the backbone with the one of another run: the source
    run folder or null, the checkpoint iteration or null for the last one, the freeze flag, only meaningful with a
    source run, and the learning rate multiplier of the transferred backbone. The source run itself is only read when
    a new training starts, so that a trained run can still be evaluated and exported once its source run is gone.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Backbone' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    experiment_folder = get_configuration_value(configuration=model_configuration,
                                                key_path="Backbone.pretrained.experiment_folder")
    iteration = get_configuration_value(configuration=model_configuration, key_path="Backbone.pretrained.iteration")

    # The source run is given by the path of its folder
    if experiment_folder is not None and not isinstance(experiment_folder, str):
        errors.append(f"'Backbone.pretrained.experiment_folder' must be null or the path of a run folder, "
                      f"got {experiment_folder!r}.")
    if iteration is not None:
        errors += check_positive_integers(configuration=model_configuration,
                                          key_paths=["Backbone.pretrained.iteration"], allow_zero=False)

    # A random backbone is never frozen
    freeze_errors = check_boolean_flags(configuration=model_configuration, key_paths=["Backbone.pretrained.freeze"])
    errors += freeze_errors
    if not freeze_errors and experiment_folder is None and get_configuration_value(
            configuration=model_configuration, key_path="Backbone.pretrained.freeze"):
        errors.append("'Backbone.pretrained.freeze' must be False when 'Backbone.pretrained.experiment_folder' is "
                      "null: only a transferred backbone can be frozen.")

    learning_rate_multiplier = get_configuration_value(configuration=model_configuration,
                                                       key_path="Backbone.pretrained.learning_rate_multiplier")
    if isinstance(learning_rate_multiplier, bool) or not isinstance(learning_rate_multiplier, (int, float)) or \
            learning_rate_multiplier <= 0:
        errors.append(f"'Backbone.pretrained.learning_rate_multiplier' must be > 0, got {learning_rate_multiplier!r}.")
    return errors
