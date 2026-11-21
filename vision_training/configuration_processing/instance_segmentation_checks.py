"""
Checks of the configuration sections specific to the instance segmentation task: every check of the object detection
task, whose two-stage detector the instance segmenter extends, its mask heads (the 'MaskHead' section, with the finer
inputs of the heads, the head shared by the levels and the mask IoU heads scoring the masks, and the mask head of every
detection level), the instance index maps of the loaders and the evaluation at the original resolution.
Run by configuration_checker.py for the instance_segmentation task.
"""

from typing import Any, Dict, List

from vision_training.configuration_processing.configuration_values import (VALID_FEATURE_MAP_NORMALIZATIONS,
                                                                           VALID_UPSAMPLINGS,
                                                                           check_boolean_flags, check_positive_integers,
                                                                           check_probability_value,
                                                                           get_configuration_value)
from vision_training.configuration_processing.common_checks import check_pretrained_configuration
from vision_training.configuration_processing.detection_checks import (MASK_HEAD_LEVEL_COMPONENT,
                                                                       MASK_IOU_LOSS_NAME,
                                                                       check_convolution_layers,
                                                                       check_detection_configuration,
                                                                       list_checked_detection_levels)


def check_mask_head_configuration(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the mask heads, one MaskHead per detection level: the mask resolution and the upsampling shared by every
    level in the 'MaskHead' section, the RoIAlign pool size of every level in 'DetectionLevels.<level>.mask_head' (at
    most the mask resolution, the pooled features being only enlarged) and its [number_layers, output_channels]
    convolutions (a multiple of 8 channels with inception modules), the feature map normalization, the module flags,
    the mask threshold of the inference, the 'pretrained' settings initialising the mask heads with those of another
    run, their finer inputs (check_mask_head_fine_inputs), the head shared by the levels (check_shared_mask_head) and
    the mask IoU heads scoring the masks (check_mask_scoring).
    The presence of the mask head of every level is checked by check_detection_levels.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'MaskHead' section.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    # Side of the masks of every level and doubling of the pooled features
    mask_resolution_errors = check_positive_integers(configuration=model_configuration,
                                                     key_paths=["MaskHead.mask_resolution"], allow_zero=False)
    errors = list(mask_resolution_errors)
    mask_resolution = get_configuration_value(configuration=model_configuration, key_path="MaskHead.mask_resolution")
    upsampling = get_configuration_value(configuration=model_configuration, key_path="MaskHead.upsampling")
    if upsampling not in VALID_UPSAMPLINGS:
        errors.append(f"'MaskHead.upsampling' must be one of {VALID_UPSAMPLINGS}, got {upsampling!r}.")
    inception = get_configuration_value(configuration=model_configuration, key_path="MaskHead.modules.inception")

    # The mask head of every level defines its RoIAlign pool size and its [number of layers, number of output
    # channels] convolutions
    for level_key in list_checked_detection_levels(model_configuration=model_configuration,
                                                   component_names=[MASK_HEAD_LEVEL_COMPONENT]):
        key_prefix = f"DetectionLevels.{level_key}.{MASK_HEAD_LEVEL_COMPONENT}"
        pool_size_errors = check_positive_integers(configuration=model_configuration,
                                                   key_paths=[f"{key_prefix}.roi_align_pool_size"], allow_zero=False)
        errors += pool_size_errors
        pool_size = get_configuration_value(configuration=model_configuration,
                                            key_path=f"{key_prefix}.roi_align_pool_size")
        if not pool_size_errors and not mask_resolution_errors and pool_size > mask_resolution:
            errors.append(f"'{key_prefix}.roi_align_pool_size' ({pool_size}) must not exceed "
                          f"'MaskHead.mask_resolution' ({mask_resolution}), the pooled features being only enlarged "
                          f"to the masks.")
        errors += check_convolution_layers(
            convolutions=get_configuration_value(configuration=model_configuration,
                                                 key_path=f"{key_prefix}.convolutions"),
            key_path=f"{key_prefix}.convolutions", inception=inception)

    # Feature map normalization strategy and module flags of the mask head convolution blocks
    feature_map_normalization = get_configuration_value(configuration=model_configuration,
                                                        key_path="MaskHead.normalization.feature_map_normalization")
    if feature_map_normalization not in VALID_FEATURE_MAP_NORMALIZATIONS:
        errors.append(f"'MaskHead.normalization.feature_map_normalization' must be one of "
                      f"{VALID_FEATURE_MAP_NORMALIZATIONS}, got {feature_map_normalization!r}.")
    errors += check_boolean_flags(configuration=model_configuration,
                                  key_paths=["MaskHead.modules.squeeze_and_excite", "MaskHead.modules.residual_nets",
                                             "MaskHead.modules.inception"])

    # Probability from which a pixel belongs to its object
    errors += check_probability_value(
        value=get_configuration_value(configuration=model_configuration, key_path="MaskHead.inference.mask_threshold"),
        key_path="MaskHead.inference.mask_threshold")

    # Mask heads initialised by those of another run
    errors += check_pretrained_configuration(model_configuration=model_configuration, section_name="MaskHead")

    # Finer inputs of the mask heads than the map of their level, and the fusion convolutions before their predictor
    errors += check_mask_head_fine_inputs(model_configuration=model_configuration)

    # One mask head read by every level instead of one head per level
    errors += check_shared_mask_head(model_configuration=model_configuration)

    # Mask IoU heads scoring the masks
    errors += check_mask_scoring(model_configuration=model_configuration)
    return errors


def is_positive_integer(value: Any) -> bool:
    """
    Tells whether a configuration value read by check_mask_scoring is a positive integer, booleans excluded, so that
    the values in a wrong form, reported by their own checks, are left out of its divisibility check.

    Args:
        value (Any): The configuration value.

    Returns:
        bool: True for an integer >= 1 that is not a boolean.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def check_mask_scoring(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies 'MaskHead.mask_scoring', the mask IoU head of Mask Scoring R-CNN (stage 8 of the instance segmentation
    benchmark): its 'activated' flag, its [number_layers, output_channels] 3x3 'convolutions' and its non-empty list of
    'fully_connected' sizes, checked even when it is off so that switching it on needs no other change. When it is on,
    the mask resolution must be a multiple of the pool size of every level, the masks being max-pooled to the box
    features of their head, and the mask IoU loss must have a positive weight, the head training on it alone. Mask
    resolutions and pool sizes in a wrong form are reported by check_mask_head_configuration, the loss weights by
    check_loss_weights_configuration, so they are left out here.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'MaskHead', 'DetectionLevels' and
            'Optimization' sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_boolean_flags(configuration=model_configuration, key_paths=["MaskHead.mask_scoring.activated"])
    errors += check_convolution_layers(
        convolutions=get_configuration_value(configuration=model_configuration,
                                             key_path="MaskHead.mask_scoring.convolutions"),
        key_path="MaskHead.mask_scoring.convolutions", inception=False)
    fully_connected = get_configuration_value(configuration=model_configuration,
                                              key_path="MaskHead.mask_scoring.fully_connected")
    if not isinstance(fully_connected, list) or not fully_connected or \
            not all(is_positive_integer(value=value) for value in fully_connected):
        errors.append(f"'MaskHead.mask_scoring.fully_connected' must be a non-empty list of positive integers, got "
                      f"{fully_connected!r}.")
    activated = get_configuration_value(configuration=model_configuration, key_path="MaskHead.mask_scoring.activated")
    if errors or activated is not True:
        return errors

    # Masks max-pooled from the mask resolution to the pool size of every level
    mask_resolution = get_configuration_value(configuration=model_configuration, key_path="MaskHead.mask_resolution")
    if is_positive_integer(value=mask_resolution):
        for level_key in list_checked_detection_levels(model_configuration=model_configuration,
                                                       component_names=[MASK_HEAD_LEVEL_COMPONENT]):
            key_path = f"DetectionLevels.{level_key}.{MASK_HEAD_LEVEL_COMPONENT}.roi_align_pool_size"
            pool_size = get_configuration_value(configuration=model_configuration, key_path=key_path)
            if is_positive_integer(value=pool_size) and mask_resolution % pool_size != 0:
                errors.append(f"'MaskHead.mask_scoring.activated' needs 'MaskHead.mask_resolution' "
                              f"({mask_resolution}) to be a multiple of '{key_path}' ({pool_size}), the masks being "
                              f"max-pooled to the box features of the mask IoU head.")

    # The mask IoU head trains on its loss alone
    loss_weights = get_configuration_value(configuration=model_configuration, key_path="Optimization.loss_weights")
    if isinstance(loss_weights, dict) and MASK_IOU_LOSS_NAME in loss_weights and loss_weights[MASK_IOU_LOSS_NAME] == 0:
        errors.append(f"'MaskHead.mask_scoring.activated' needs a positive "
                      f"'Optimization.loss_weights.{MASK_IOU_LOSS_NAME}', the mask IoU head training on it alone.")
    return errors


def check_shared_mask_head(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies 'MaskHead.shared', True for a single mask head read by every level as in Mask R-CNN (stage 5 of the
    instance segmentation benchmark), False for one head per level: the shared head needs the same
    'DetectionLevels.<level>.mask_head' on every level (one pool size and convolution block), and the same channel
    count on every level, which a neck gives ('Neck.output_channels') and which the backbone blocks of the levels
    ('Backbone.convolutions') must give without one. Levels without mask head and blocks in a wrong form are reported
    by their own checks, so they are left out here.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'MaskHead', 'DetectionLevels',
            'Neck' and 'Backbone' sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_boolean_flags(configuration=model_configuration, key_paths=["MaskHead.shared"])
    shared = get_configuration_value(configuration=model_configuration, key_path="MaskHead.shared")
    detection_levels = get_configuration_value(configuration=model_configuration, key_path="DetectionLevels")
    if errors or not shared or not isinstance(detection_levels, dict):
        return errors

    # One pool size and convolution block for every level
    level_mask_heads = {level_key: level_configuration[MASK_HEAD_LEVEL_COMPONENT]
                        for level_key, level_configuration in detection_levels.items()
                        if isinstance(level_configuration, dict) and MASK_HEAD_LEVEL_COMPONENT in level_configuration}
    if any(level_mask_head != next(iter(level_mask_heads.values())) for level_mask_head in level_mask_heads.values()):
        errors.append(f"'MaskHead.shared' needs the same '{MASK_HEAD_LEVEL_COMPONENT}' on every detection level, one "
                      f"head reading them all, got {level_mask_heads!r}.")

    # Without a neck, the levels keep the channels of their backbone blocks, which must then be equal
    neck_type = get_configuration_value(configuration=model_configuration, key_path="Neck.type")
    block_convolutions = get_configuration_value(configuration=model_configuration, key_path="Backbone.convolutions")
    if neck_type == "none" and isinstance(block_convolutions, dict):
        level_channels = {level_key: block_convolutions[level_key][1] for level_key in detection_levels
                          if level_key in block_convolutions and isinstance(block_convolutions[level_key], list)
                          and len(block_convolutions[level_key]) == 2}
        if len(set(level_channels.values())) > 1:
            errors.append(f"'MaskHead.shared' needs the same channel count on every detection level, the backbone "
                          f"blocks read without a neck having {level_channels!r} channels: set 'Neck.type' or give "
                          f"the blocks of the levels the same channels.")
    return errors


def compute_skip_stride(block: int) -> int:
    """
    Gives the stride of a backbone skip of the mask heads, read by RoIAlign at its spatial scale: the normalized input
    image (block 0) is at stride 1, and block b before its pooling at 2 ** (b - 1), each block before it pooling once
    (only the last block of the backbone may skip its pooling, and no block follows it).

    Args:
        block (int): The skip block, 0 for the normalized input image.

    Returns:
        int: The stride of the skip in pixels of the input image.
    """
    return 1 if block == 0 else 2 ** (block - 1)


def compute_level_stride(level: int, number_blocks: int, last_max_pooling: bool) -> int:
    """
    Gives the stride of a detection level, the output of its backbone block after its pooling, which the last block
    of the backbone skips unless 'Backbone.last_max_pooling' adds it, so that the skips of the mask heads are compared
    with the levels the heads already read.

    Args:
        level (int): The backbone block of the detection level.
        number_blocks (int): The number of blocks of the backbone.
        last_max_pooling (bool): Whether the last block of the backbone pools its output.

    Returns:
        int: The stride of the level in pixels of the input image.
    """
    return 2 ** level if level < number_blocks or last_max_pooling else 2 ** (level - 1)


def check_skip_block_strides(model_configuration: Dict[str, Any], valid_skip_blocks: Dict[str, List[int]],
                             number_blocks: int) -> List[str]:
    """
    Verifies that the backbone skips of the mask heads bring finer features than the maps the heads already read
    (stage 3b of the instance segmentation benchmark): every skip below the stride of the finest detection level,
    every skip at the mask resolution finer than every skip at the pool size, as U-Net joins its finest encoder maps
    at its finest decoder stage, and no block read at both sizes, reported on its own. Skips listed in a wrong form,
    detection levels that are not block numbers and a 'Backbone.last_max_pooling' that is not a boolean are reported
    by their own checks, so the rules are then left out.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'Backbone', 'DetectionLevels' and
            'MaskHead' sections.
        valid_skip_blocks (Dict[str, List[int]]): The skip lists that passed their own check, keyed by their name in
            'MaskHead' ('skip_blocks_at_pool_size', 'skip_blocks_at_mask_resolution').
        number_blocks (int): The number of blocks of the backbone.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Stride of the finest detection level, from the quoted block numbers of 'DetectionLevels'
    detection_levels = get_configuration_value(configuration=model_configuration, key_path="DetectionLevels")
    last_max_pooling = get_configuration_value(configuration=model_configuration, key_path="Backbone.last_max_pooling")
    levels = [int(level_key) for level_key in detection_levels if isinstance(level_key, str) and level_key.isdigit()] \
        if isinstance(detection_levels, dict) else []
    if levels and isinstance(last_max_pooling, bool):
        finest_level = min(levels)
        finest_level_stride = compute_level_stride(level=finest_level, number_blocks=number_blocks,
                                                   last_max_pooling=last_max_pooling)

        # Every skip finer than the finest level, which the mask heads already read
        for key_name, skip_blocks in valid_skip_blocks.items():
            coarse_blocks = [block for block in skip_blocks if compute_skip_stride(block=block) >= finest_level_stride]
            if coarse_blocks:
                errors.append(f"'MaskHead.{key_name}' must only hold blocks finer than the finest detection level "
                              f"{finest_level} (stride {finest_level_stride}), a block before its pooling being at "
                              f"stride 2 ** (block - 1), got {coarse_blocks!r}.")

    # Skips at the mask resolution finer than the skips at the pool size
    pool_size_blocks = valid_skip_blocks["skip_blocks_at_pool_size"] \
        if "skip_blocks_at_pool_size" in valid_skip_blocks else []
    mask_resolution_blocks = valid_skip_blocks["skip_blocks_at_mask_resolution"] \
        if "skip_blocks_at_mask_resolution" in valid_skip_blocks else []
    if pool_size_blocks and mask_resolution_blocks:
        # A block read at both sizes, reported on its own before the order of the other blocks
        shared_blocks = sorted(set(pool_size_blocks) & set(mask_resolution_blocks))
        if shared_blocks:
            errors.append(f"'MaskHead.skip_blocks_at_pool_size' and 'MaskHead.skip_blocks_at_mask_resolution' must not "
                          f"share a block, the same features being then read twice, got {shared_blocks!r}.")
        finest_pool_size_stride = min(compute_skip_stride(block=block) for block in pool_size_blocks)
        coarse_blocks = [block for block in mask_resolution_blocks if block not in shared_blocks and
                         compute_skip_stride(block=block) >= finest_pool_size_stride]
        if coarse_blocks:
            errors.append(f"'MaskHead.skip_blocks_at_mask_resolution' must only hold blocks finer than every block of "
                          f"'MaskHead.skip_blocks_at_pool_size' {pool_size_blocks!r} (finest stride "
                          f"{finest_pool_size_stride}), got {coarse_blocks!r}.")
    return errors


def check_mask_head_fine_inputs(model_configuration: Dict[str, Any]) -> List[str]:
    """
    Verifies the finer inputs of the mask heads (stage 3b of the instance segmentation benchmark), all off by default:
    the [number_layers, output_channels] of the fusion convolutions before the predictor
    ('MaskHead.fusion_convolutions') and of the image branch on the crop of the input image
    ('MaskHead.image_branch_convolutions'), [0, 0] when unused, and the backbone blocks whose outputs before pooling
    are pooled at the pool size or at the mask resolution
    ('MaskHead.skip_blocks_at_pool_size', 'MaskHead.skip_blocks_at_mask_resolution'), distinct block numbers of
    'Backbone.convolutions' or 0 for the normalized input image. A skip must bring finer features than the levels: its
    stride must be below that of the finest detection level, and every skip at the mask resolution must be finer than
    every skip at the pool size, as U-Net joins its finest encoder maps at its finest decoder stage, and no block may
    be read at both sizes. The image branch is not part of the transferred mask heads, so it is refused with mask heads
    initialised by another run, whose heads were trained on another branch.

    Args:
        model_configuration (Dict[str, Any]): The model configuration containing the 'MaskHead' and 'Backbone'
            sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = []

    # Plain convolutions given as [number_layers, output_channels], both zero when unused or both positive
    valid_convolutions = {}
    for key_name in ["fusion_convolutions", "image_branch_convolutions"]:
        key_path = f"MaskHead.{key_name}"
        convolutions = get_configuration_value(configuration=model_configuration, key_path=key_path)
        if isinstance(convolutions, list) and len(convolutions) == 2 and \
                all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in convolutions) \
                and (convolutions[0] == 0) == (convolutions[1] == 0):
            valid_convolutions[key_name] = convolutions
        else:
            errors.append(f"'{key_path}' must be [number_layers, output_channels] with positive integers, or [0, 0] "
                          f"when unused, got {convolutions!r}.")

    # Backbone blocks read before their pooling, 0 standing for the normalized input image
    number_blocks = len(get_configuration_value(configuration=model_configuration, key_path="Backbone.convolutions"))
    valid_skip_blocks = {}
    for key_name in ["skip_blocks_at_pool_size", "skip_blocks_at_mask_resolution"]:
        key_path = f"MaskHead.{key_name}"
        skip_blocks = get_configuration_value(configuration=model_configuration, key_path=key_path)
        valid_blocks = isinstance(skip_blocks, list) and \
            all(isinstance(block, int) and not isinstance(block, bool) and 0 <= block <= number_blocks
                for block in skip_blocks)
        if not valid_blocks or len(set(skip_blocks)) != len(skip_blocks):
            errors.append(f"'{key_path}' must be a list of distinct backbone blocks between 0 (the normalized input "
                          f"image) and {number_blocks}, empty when unused, got {skip_blocks!r}.")
        else:
            valid_skip_blocks[key_name] = skip_blocks
    errors += check_skip_block_strides(model_configuration=model_configuration, valid_skip_blocks=valid_skip_blocks,
                                       number_blocks=number_blocks)

    # The image branch is not transferred with the mask heads, which would read an untrained branch
    experiment_folder = get_configuration_value(configuration=model_configuration,
                                                key_path="MaskHead.pretrained.experiment_folder")
    if experiment_folder is not None and "image_branch_convolutions" in valid_convolutions and \
            valid_convolutions["image_branch_convolutions"][0] > 0:
        errors.append(f"'MaskHead.image_branch_convolutions' must be [0, 0] when "
                      f"'MaskHead.pretrained.experiment_folder' is set: the image branch is not transferred with the "
                      f"mask heads, got {valid_convolutions['image_branch_convolutions']!r}.")
    return errors


def check_instance_segmentation_configuration(experiment_configuration: Dict[str, Any],
                                              model_configuration: Dict[str, Any]) -> List[str]:
    """
    Runs every check specific to the instance segmentation task: those of the object detection task, every detection
    level having to hold a mask head there, then the mask heads, the instance index maps of the loaders
    ('Data.instance_index_maps') and the evaluation at the original resolution of the test images.

    Args:
        experiment_configuration (Dict[str, Any]): The processed experiment configuration.
        model_configuration (Dict[str, Any]): The processed model sections.

    Returns:
        List[str]: The description of every inconsistency found.
    """
    errors = check_detection_configuration(experiment_configuration=experiment_configuration,
                                           model_configuration=model_configuration)
    errors += check_mask_head_configuration(model_configuration=model_configuration)

    # Masks of every image loaded as one instance index map, for the run length encodings that never overlap
    errors += check_boolean_flags(configuration=model_configuration, key_paths=["Data.instance_index_maps"])

    # Scoring of the boxes and masks at the original resolution of the test images, nested under the section name so
    # that the reported key path matches its location in the YAML file
    errors += check_boolean_flags(configuration={"ExperimentConfiguration": experiment_configuration},
                                  key_paths=["ExperimentConfiguration.evaluation.original_resolution"])
    return errors
