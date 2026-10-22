"""
Checks of the configuration sections shared by every task: ExperimentConfiguration, Data, Backbone, the 'pretrained'
settings of the transferable components and Optimization, after the refusal of the sections removed from the
configuration files. Run first by configuration_checker.py, before the checks of the task sections, and reused by the
task checks (e.g. refusing the sequence settings for the image tasks, the geometric augmentation types moving the image
only for the tasks with boxes, masks or keypoints, the error images settings of the classification-like and detection
tasks, or the loss weights of the detection tasks).
"""

import re

from typing import Any, Dict, List

from vision_training.configuration_processing.configuration_values import (BACKBONE_FEATURE_MAP_NORMALIZATIONS,
                                                                           check_batch_normalization_momentum,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value, is_number)
from vision_training.data.augmentations import AUGMENTATION_TYPES, IMAGE_ONLY_AUGMENTATION_TYPES


# Label mapping modes handled by the configuration loader
VALID_LABEL_MODES = ["binary", "multi_class"]

# Channels of the images given by the loaders of every task (BGR) and taken by the exported models
IMAGE_CHANNELS = 3

# Characters of the suffix of the run names: letters, digits, '-' and '.', so that the suffix stays one section of the
# name ('_' separating the sections) and inside the experiment folder ('/' separating the folders)
RUN_NAME_SUFFIX_PATTERN = re.compile(pattern=r"[A-Za-z0-9.-]+")

# Optimizers and learning rate schedules of the 'Optimization' section
VALID_OPTIMIZERS = ["adam", "adamw", "sgd"]
VALID_LEARNING_RATE_SCHEDULES = ["constant", "cosine"]

# Sections removed from the configuration files, refused with what replaced them so that the settings of a file written
# before are never silently ignored: the random views of the self-supervised task, now drawn by the types of
# 'Data.data_augmentation', and the view losses of the image classifier
REMOVED_SECTIONS = {"Data.view_augmentation": "the self-supervised task draws its two views with the types of "
                                              "'Data.data_augmentation', whose strengths are hard-coded "
                                              "(SELF_SUPERVISED_AUGMENTATION_STRENGTHS of "
                                              "vision_training/data/augmentations.py), so its probabilities go to "
                                              "those types",
                    "ClassificationHead.view_losses": "the image classifier no longer has view losses (Barlow Twins "
                                                      "and the contrastive loss on a second view of every image)"}

# Name of the component of every section whose 'pretrained' settings initialise it with the one of another run, used
# in the messages
PRETRAINED_COMPONENT_NAMES = {"Backbone": "backbone", "TemporalModel": "temporal module",
                              "ClassificationHead": "classification head",
                              "SegmentationDecoder": "segmentation decoder", "RegionProposal": "region proposal heads",
                              "Detector": "detection heads", "MaskHead": "mask heads",
                              "HRNet": "high resolution network", "KeypointHead": "keypoint head"}


def check_removed_sections(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Refuses the sections of REMOVED_SECTIONS, removed from the configuration files, naming what replaced each of them.
    Run by configuration_checker.py before every other check, since a file written before the removal also lacks the
    keys added with it, and the first missing key stops the other checks without saying why.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every removed section found.
    """
    errors = []
    for key_path, replacement in REMOVED_SECTIONS.items():
        # Walk down to the section holding the removed key, a missing section holding nothing
        *parent_keys, removed_key = key_path.split(".")
        section = model_configuration
        for parent_key in parent_keys:
            section = section.get(parent_key) if isinstance(section, dict) else None
        if isinstance(section, dict) and removed_key in section:
            errors.append(f"'{key_path}' was removed: {replacement}. Delete it from the configuration file.")
    return errors


def check_experiment_configuration(experiment_configuration: Dict[str, Any], valid_tasks: List[str]) -> List[str]:
    """
    Verifies the task, the run name suffix, the training loop settings, the per-class report switch, the number of
    sample images and the MLflow tracking settings of the 'ExperimentConfiguration' section consumed by the trainers.

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

    # Suffix appended as the last section of the run name (build_run_name), null for none
    run_name_suffix = get_configuration_value(configuration=configuration,
                                              key_path="ExperimentConfiguration.run_name_suffix")
    if run_name_suffix is not None and (not isinstance(run_name_suffix, str) or
                                        RUN_NAME_SUFFIX_PATTERN.fullmatch(string=run_name_suffix) is None):
        errors.append(f"'ExperimentConfiguration.run_name_suffix' must be null or a non-empty string of letters, "
                      f"digits, '-' and '.', got {run_name_suffix!r}.")

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

    # Fixed test samples drawn with their targets and predictions at every evaluation, 0 for none
    errors += check_positive_integers(configuration=configuration,
                                      key_paths=["ExperimentConfiguration.evaluation.number_sample_images"],
                                      allow_zero=True)

    # The optimizer requires a strictly positive learning rate
    learning_rate = get_configuration_value(configuration=configuration,
                                            key_path="ExperimentConfiguration.learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        errors.append(f"'ExperimentConfiguration.learning_rate' must be > 0, got {learning_rate!r}.")

    # Flags toggling training resumption, the per-class part of the evaluation reports, the diagnosis modes and the
    # MLflow tracking with its artifacts
    errors += check_boolean_flags(configuration=configuration,
                                  key_paths=["ExperimentConfiguration.resume_training",
                                             "ExperimentConfiguration.allow_data_change_on_resume",
                                             "ExperimentConfiguration.evaluation.per_class_report",
                                             "ExperimentConfiguration.diagnosis_modes.verbose",
                                             "ExperimentConfiguration.diagnosis_modes.compute_receptive_field",
                                             "ExperimentConfiguration.diagnosis_modes.compute_model_size",
                                             "ExperimentConfiguration.diagnosis_modes.view_input_images",
                                             "ExperimentConfiguration.tracking.activated",
                                             "ExperimentConfiguration.tracking.log_artifacts"])

    # Address of the MLflow server and name of the MLflow experiment, null for the local database and the task name
    for key_path in ["ExperimentConfiguration.tracking.tracking_uri",
                     "ExperimentConfiguration.tracking.experiment_name"]:
        tracking_value = get_configuration_value(configuration=configuration, key_path=key_path)
        if tracking_value is not None and (not isinstance(tracking_value, str) or not tracking_value.strip()):
            errors.append(f"'{key_path}' must be null or a non-empty string, got {tracking_value!r}.")

    return errors


def check_data_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Data' section: loading settings, augmentation types with the grid of random_black_out, label
    mapping mode and the input image size. The image size must be a power of two so that every max pooling layer of the
    Backbone halves the feature maps exactly, which keeps the anchor strides and the feature map sizes consistent
    across the detection levels.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Data' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Resizing flag and global augmentation settings
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["Data.image_settings.keep_ratio", "Data.data_augmentation.activated"])
    errors += check_probability_value(
        value=get_configuration_value(configuration=model_configuration, key_path="Data.data_augmentation.probability"),
        key_path="Data.data_augmentation.probability")

    # Worker processes of the loaders (0 loads in the main process) and batches prepared in advance by each of them
    errors += check_positive_integers(configuration=model_configuration, key_paths=["Data.loading.number_workers"],
                                      allow_zero=True)
    errors += check_positive_integers(configuration=model_configuration, key_paths=["Data.loading.prefetch_batches"],
                                      allow_zero=False)

    # Sequence augmentations: the photometric types drawn frame by frame, and the shuffling of the frames of a sequence
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["Data.data_augmentation.independent_frames",
                                             "Data.data_augmentation.shuffle_frames.activated"])
    errors += check_probability_value(
        value=get_configuration_value(configuration=model_configuration,
                                      key_path="Data.data_augmentation.shuffle_frames.probability"),
        key_path="Data.data_augmentation.shuffle_frames.probability")

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

    # Rows and columns of the grid of random_black_out, within the side of the network input
    grid_size_key_path = "Data.data_augmentation.types.random_black_out.grid_size"
    errors += check_black_out_grid_size(grid_size=get_configuration_value(configuration=model_configuration,
                                                                          key_path=grid_size_key_path),
                                        key_path=grid_size_key_path, image_size=image_size)

    return errors


def check_image_task_sequence_settings(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Refuses the settings of the sequences for the image tasks (classification, object detection), whose samples have
    no frames: the frame shuffling and the per-frame augmentation draws, and the sequence normalization of the
    backbone.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Data' and 'Backbone' sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    for key_path in ["Data.data_augmentation.shuffle_frames.activated", "Data.data_augmentation.independent_frames"]:
        if get_configuration_value(configuration=model_configuration, key_path=key_path) is True:
            errors.append(f"'{key_path}' must be False for this task, whose samples have no frames.")
    if get_configuration_value(configuration=model_configuration,
                               key_path="Backbone.normalization.feature_map_normalization") == "sequence":
        errors.append("'Backbone.normalization.feature_map_normalization' cannot be 'sequence' for this task, whose "
                      "samples have no frames.")
    return errors


def check_image_only_augmentation_types(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Refuses the geometric augmentation types moving the image only (IMAGE_ONLY_AUGMENTATION_TYPES, the perspective and
    the jigsaw) for the tasks whose targets move with the image, the object detection, the instance segmentation, the
    semantic segmentation and the keypoint detection: DataAugmenter cannot move their boxes, masks or keypoints with
    these types and stops at the first training batch given to one of them.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Data' section.

    Returns:
        List[str]: The description of every activated type moving the image only.
    """
    errors = []
    for augmentation_type in IMAGE_ONLY_AUGMENTATION_TYPES:
        key_path = f"Data.data_augmentation.types.{augmentation_type}.activated"
        if get_configuration_value(configuration=model_configuration, key_path=key_path) is True:
            errors.append(f"'{key_path}' must be False for this task: {augmentation_type} moves the image only, and "
                          f"its bounding boxes, masks or keypoints cannot follow it.")
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

    # The loaders of every task give 3-channel BGR images, and the exports take 3-channel inputs
    input_channels = get_configuration_value(configuration=model_configuration, key_path="Backbone.input_channels")
    if isinstance(input_channels, bool) or input_channels != IMAGE_CHANNELS:
        errors.append(f"'Backbone.input_channels' must be {IMAGE_CHANNELS}, the loaders giving 3-channel BGR images "
                      f"and the exports taking 3-channel inputs, got {input_channels!r}.")
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["Backbone.last_max_pooling", "Backbone.normalization.input_normalization",
                                             "Backbone.normalization.per_sample_normalization",
                                             "Backbone.modules.squeeze_and_excite", "Backbone.modules.residual_nets",
                                             "Backbone.modules.residual_nets_v2", "Backbone.modules.inception"])

    # The pre-activation placement is a placement of the residual shortcuts
    residual_nets = get_configuration_value(configuration=model_configuration,
                                            key_path="Backbone.modules.residual_nets")
    residual_nets_v2 = get_configuration_value(configuration=model_configuration,
                                               key_path="Backbone.modules.residual_nets_v2")
    if residual_nets_v2 is True and residual_nets is False:
        errors.append("'Backbone.modules.residual_nets_v2' must be False when 'Backbone.modules.residual_nets' is "
                      "False: it places the residual shortcuts before the activations.")

    # Feature map normalization strategy of the convolution blocks, and the momentum of its batch statistics
    feature_map_normalization = get_configuration_value(configuration=model_configuration,
                                                        key_path="Backbone.normalization.feature_map_normalization")
    if feature_map_normalization not in BACKBONE_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'Backbone.normalization.feature_map_normalization' must be one of "
                      f"{BACKBONE_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")
    errors += check_batch_normalization_momentum(configuration=model_configuration,
                                                 key_path="Backbone.normalization.batch_normalization_momentum")

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

    errors += check_pretrained_configuration(model_configuration=model_configuration, section_name="Backbone")
    return errors


def check_pretrained_configuration(model_configuration: Dict[str, Any], section_name: str) -> List[str]:
    """
    Verifies the 'pretrained' settings of a section (Backbone, TemporalModel, ClassificationHead, SegmentationDecoder,
    RegionProposal, Detector, MaskHead, HRNet or KeypointHead), which initialise its component with the trained one of
    another run: the source run folder or null, the checkpoint iteration or null for the last one, the freeze flag,
    only meaningful with a source run, and the learning rate multiplier of the transferred component. For the backbone,
    the 'freeze_normalization_parameters' flag keeps the scale and shift of its normalization layers fixed, which needs
    a source run that is not frozen anyway and a backbone with normalization layers. For the segmentation decoder, the
    'load_class_layers' flag also loads the segmentation head and the deep supervision heads, which needs a source run.
    The source run itself is only read when a new training starts, so that a trained run can still be evaluated and
    exported once its source run is gone.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the section.
        section_name (str): The section holding the 'pretrained' settings, a key of PRETRAINED_COMPONENT_NAMES.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    component_name = PRETRAINED_COMPONENT_NAMES[section_name]
    key_prefix = f"{section_name}.pretrained"
    experiment_folder = get_configuration_value(configuration=model_configuration,
                                                key_path=f"{key_prefix}.experiment_folder")
    iteration = get_configuration_value(configuration=model_configuration, key_path=f"{key_prefix}.iteration")

    # The source run is given by the path of its folder
    if experiment_folder is not None and not isinstance(experiment_folder, str):
        errors.append(f"'{key_prefix}.experiment_folder' must be null or the path of a run folder, "
                      f"got {experiment_folder!r}.")
    if iteration is not None:
        errors += check_positive_integers(configuration=model_configuration, key_paths=[f"{key_prefix}.iteration"],
                                          allow_zero=False)

    # A random component is never frozen
    freeze_errors = check_boolean_flags(configuration=model_configuration, key_paths=[f"{key_prefix}.freeze"])
    errors += freeze_errors
    freeze = get_configuration_value(configuration=model_configuration, key_path=f"{key_prefix}.freeze")
    if not freeze_errors and experiment_folder is None and freeze:
        errors.append(f"'{key_prefix}.freeze' must be False when '{key_prefix}.experiment_folder' is null: only a "
                      f"transferred {component_name} can be frozen.")

    # Factor of the learning rate of the component, strictly positive
    learning_rate_multiplier =get_configuration_value(configuration=model_configuration,
                                                       key_path=f"{key_prefix}.learning_rate_multiplier")
    if isinstance(learning_rate_multiplier, bool) or not isinstance(learning_rate_multiplier, (int, float)) or \
            learning_rate_multiplier <= 0:
        errors.append(f"'{key_prefix}.learning_rate_multiplier' must be > 0, got {learning_rate_multiplier!r}.")

    # The class layers of the segmentation decoder are only loaded from a source run
    if section_name == "SegmentationDecoder":
        class_layers_errors = check_boolean_flags(configuration=model_configuration,
                                                  key_paths=[f"{key_prefix}.load_class_layers"])
        errors += class_layers_errors
        load_class_layers = get_configuration_value(configuration=model_configuration,
                                                    key_path=f"{key_prefix}.load_class_layers")
        if not class_layers_errors and experiment_folder is None and load_class_layers:
            errors.append(f"'{key_prefix}.load_class_layers' must be False when '{key_prefix}.experiment_folder' is "
                          f"null: the class layers are loaded from the source run of the decoder.")
    if section_name != "Backbone":
        return errors

    # The scale and shift of the normalization layers are kept fixed in a transferred backbone that still learns
    normalization_errors = check_boolean_flags(configuration=model_configuration,
                                               key_paths=["Backbone.pretrained.freeze_normalization_parameters"])
    errors += normalization_errors

    # Settings the fixed normalization depends on: the flag itself and the normalization of the backbone
    freeze_normalization = get_configuration_value(configuration=model_configuration,
                                                   key_path="Backbone.pretrained.freeze_normalization_parameters")
    feature_map_normalization = get_configuration_value(
        configuration=model_configuration, key_path="Backbone.normalization.feature_map_normalization")

    # Only a transferred backbone that still learns and has normalization layers can keep them fixed
    if not normalization_errors and freeze_normalization:
        if experiment_folder is None:
            errors.append("'Backbone.pretrained.freeze_normalization_parameters' must be False when "
                          "'Backbone.pretrained.experiment_folder' is null: only the normalization of a transferred "
                          "backbone can be kept fixed.")
        elif freeze is True:
            errors.append("'Backbone.pretrained.freeze_normalization_parameters' must be False when "
                          "'Backbone.pretrained.freeze' is True: a frozen backbone already keeps every parameter "
                          "fixed.")
        if feature_map_normalization == "none":
            errors.append("'Backbone.pretrained.freeze_normalization_parameters' must be False when "
                          "'Backbone.normalization.feature_map_normalization' is 'none': the backbone has no "
                          "normalization layer.")
    return errors


def check_error_dumps_configuration(experiment_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'ExperimentConfiguration.evaluation.error_dumps' settings of the tasks saving their errors as images
    (the classification-like and detection tasks): a boolean switch, and a positive number of images per folder.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Nest the section under its name so that every reported key path matches its location in the YAML file
    configuration = {"ExperimentConfiguration": experiment_configuration}
    errors = check_boolean_flags(configuration=configuration,
                                 key_paths=["ExperimentConfiguration.evaluation.error_dumps.activated"])
    errors += check_positive_integers(configuration=configuration, allow_zero=False,
                                      key_paths=["ExperimentConfiguration.evaluation.error_dumps."
                                                 "maximum_per_class_pair"])
    return errors


def check_optimization_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the 'Optimization' section shared by every task: the optimizer, its weight decay, the learning rate
    schedule (warm-up, cosine decay and its minimum), the gradient clipping, the mixed precision flag, which runs
    the forward passes in bfloat16 on top of the float32 weights of every model, the starting weights of the model and
    the moving average of its weights, whose decay must be strictly between 0 and 1. The loss weights of the object
    detection and instance segmentation are checked with their task (check_loss_weights_configuration).

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Optimization' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []
    optimizer = get_configuration_value(configuration=model_configuration, key_path="Optimization.optimizer")
    if optimizer not in VALID_OPTIMIZERS:
        errors.append(f"'Optimization.optimizer' must be one of {VALID_OPTIMIZERS}, got {optimizer!r}.")
    learning_rate_schedule = get_configuration_value(configuration=model_configuration,
                                                     key_path="Optimization.learning_rate_schedule")
    if learning_rate_schedule not in VALID_LEARNING_RATE_SCHEDULES:
        errors.append(f"'Optimization.learning_rate_schedule' must be one of {VALID_LEARNING_RATE_SCHEDULES}, "
                      f"got {learning_rate_schedule!r}.")

    # Weight decay and schedule lengths
    weight_decay = get_configuration_value(configuration=model_configuration, key_path="Optimization.weight_decay")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        errors.append(f"'Optimization.weight_decay' must be >= 0, got {weight_decay!r}.")
    errors += check_positive_integers(configuration=model_configuration, key_paths=["Optimization.warmup_iterations"],
                                      allow_zero=True)
    errors += check_positive_integers(configuration=model_configuration, key_paths=["Optimization.decay_iterations"],
                                      allow_zero=False)
    errors += check_probability_value(
        value=get_configuration_value(configuration=model_configuration,
                                      key_path="Optimization.minimum_learning_rate_ratio"),
        key_path="Optimization.minimum_learning_rate_ratio")

    # Gradient clipping, disabled with null
    gradient_clipping_norm = get_configuration_value(configuration=model_configuration,
                                                     key_path="Optimization.gradient_clipping_norm")
    if gradient_clipping_norm is not None and (isinstance(gradient_clipping_norm, bool) or
                                               not isinstance(gradient_clipping_norm, (int, float)) or
                                               gradient_clipping_norm <= 0):
        errors.append(f"'Optimization.gradient_clipping_norm' must be null or > 0, got {gradient_clipping_norm!r}.")

    # Mixed precision computes in bfloat16 from the float32 weights of every model
    errors += check_boolean_flags(configuration=model_configuration, key_paths=["Optimization.mixed_precision"])

    # Moving average of the weights, its decay giving the fraction of the average kept at every step
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["Optimization.exponential_moving_average.activated"])
    decay = get_configuration_value(configuration=model_configuration,
                                    key_path="Optimization.exponential_moving_average.decay")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 < decay < 1:
        errors.append(f"'Optimization.exponential_moving_average.decay' must be strictly between 0 and 1, "
                      f"got {decay!r}.")
    return errors


def check_loss_weights_configuration(model_configuration: Dict[str, Any], loss_names: List[str]) -> List[str]:
    """
    Verifies 'Optimization.loss_weights', the weight of every loss of the model in the total loss of the training,
    read by the trainers of the tasks weighting their losses (object detection and instance segmentation): exactly one
    weight per loss of the model, keyed by the name of its output, every weight a number >= 0 (0 leaving the layers
    only that loss trains unchanged), and not all 0.

    Args:
        model_configuration (Dict[str, Any]): The processed model sections.
        loss_names (List[str]): The names of the loss outputs of the model of the task.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    loss_weights = get_configuration_value(configuration=model_configuration, key_path="Optimization.loss_weights")
    if not isinstance(loss_weights, dict) or sorted(loss_weights) != sorted(loss_names):
        return [f"'Optimization.loss_weights' must give the weight of exactly the losses {loss_names}, "
                f"got {loss_weights!r}."]

    # Non-negative weights, at least one of them above 0 so that the model has a loss
    errors = [f"'Optimization.loss_weights.{loss_name}' must be a number >= 0, got {weight!r}."
              for loss_name, weight in loss_weights.items() if not is_number(value=weight) or weight < 0]
    if not errors and all(weight == 0 for weight in loss_weights.values()):
        errors.append("'Optimization.loss_weights' cannot all be 0, the model would have no loss.")
    return errors


def check_black_out_grid_size(grid_size: Any, key_path: str, image_size: Any) -> List[str]:
    """
    Verifies the number of rows and columns of the grid of the random_black_out type of the data augmentation of every
    task ('Data.data_augmentation.types.random_black_out.grid_size'): an integer of at least 1, and at most the side of
    the network input, so that every cell holds at least one pixel.

    Args:
        grid_size (Any): The grid size read from the configuration.
        key_path (str): The dotted key path of the grid size, used in the error message.
        image_size (Any): The side of the square network input, 'Data.image_settings.size', the upper bound being left
            unchecked when it is not a positive integer (reported by check_data_configuration).

    Returns:
        List[str]: A single error description if the grid size is invalid, otherwise an empty list.
    """
    image_size_valid = not isinstance(image_size, bool) and isinstance(image_size, int) and image_size >= 1
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1 or \
            (image_size_valid and grid_size > image_size):
        return [f"'{key_path}' must be an integer within [1, 'Data.image_settings.size' ({image_size!r})], got "
                f"{grid_size!r}: a grid finer than the image would hide cells holding no pixel."]
    return []
