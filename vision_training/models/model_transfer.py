"""
Transfer of trained components between runs: the architecture checks of the backbone, temporal module, classification
head, segmentation decoder, region proposal heads, detection heads, mask heads, high resolution network, keypoint head
and neck of the detection tasks (whose heads only move between runs of the same neck type and channels, with a warning
when the neck of the new run is not the one they were trained behind), the loading of a pretrained component from the
checkpoint of another run (its averaged weights when it kept a moving average and the new run keeps one), the first
blocks of a deeper backbone initialising a shorter one (the stem of the keypoint detector from an ImageNet backbone) and
a shorter backbone initialising the first blocks of a deeper one (its next blocks staying random, and only its loaded
blocks frozen or fine-tuned at the multiplier of the transfer), the class layers of a run adding classes after those of
its source run or holding them in another order (the source classes keeping their trained rows, at the index of the same
class name, the new classes the random rows of the new run), the classification, detection, mask and keypoint heads
loaded without their class layers for a run of other classes (their class layers keeping the random initialisation of
the new run and learning at its learning rate), the class layers of the semantic segmentation loaded with its decoder,
and the freezing of components, normalization parameters and batch statistics.
Used by the base trainer and by scripts/evaluation/evaluate_checkpoint.py.
"""

import re
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Optional, Tuple

from vision_training.configuration_processing.common_checks import (CLASS_LAYER_OPTION_SECTIONS,
                                                                   PRETRAINED_CLASS_LAYER_KEYS)
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.data.data_record import (CLASS_NAMES_RECORD_KEY, CLASSES_RECORD_KEY, DATA_RECORD_FILE_NAME,
                                              KEYPOINTS_RECORD_KEY)
from vision_training.models.backbones.backbone import INCEPTION_STARTING_BLOCK
from vision_training.trainers.checkpointing import EXPONENTIAL_MOVING_AVERAGE_STATE_KEY, get_checkpoint_path
from vision_training.trainers.weight_averaging import AVERAGED_MODEL_STATE_KEY
from vision_training.utilities.os_utilities import format_path_link, print_color, read_json

# Configuration sections whose component can be initialised by the trained component of another run, with the
# attribute holding the component in every task model, which is also the prefix of its weights in the state
# dictionaries. Their order is the order of the optimizer parameter groups of the components learning at their own
# speed, the heads of the detection tasks, then the mask heads of the instance segmentation and the high resolution
# network and keypoint head (heatmap and tag convolution with the higher resolution branch) of the keypoint detection,
# then the neck of the detection tasks, coming last so that the runs written before them keep their groups
TRANSFERABLE_COMPONENTS = {"Backbone": "backbone", "TemporalModel": "temporal_module",
                           "ClassificationHead": "classification_head", "SegmentationDecoder": "segmentation_decoder",
                           "RegionProposal": "region_proposer_dictionary", "Detector": "detector_heads_dictionary",
                           "MaskHead": "mask_heads_dictionary", "HRNet": "high_resolution_network",
                           "KeypointHead": "keypoint_head", "Neck": "neck"}

# Names of the components whose attribute does not read as a name in the messages of the transfer
COMPONENT_NAMES = {"RegionProposal": "region proposal heads", "Detector": "detection heads",
                   "MaskHead": "mask heads"}

# Differing keypoint channels, or source classes the new run does not have, named in the refusal of a transfer, the
# others being only counted
MAXIMUM_LISTED_DIFFERENCES = 5

# Attributes of the semantic segmenter holding its class layers, outside the decoder so that a decoder serves runs of
# other classes, loaded with the decoder when 'SegmentationDecoder.pretrained.class_layers' is True: the 1x1
# convolution of the segmentation head and the 1x1 convolutions of the deep supervision heads
SEGMENTATION_CLASS_LAYER_ATTRIBUTES = ["segmentation_head", "deep_supervision_heads"]

# Backbone settings defining its layers and their parameters, which must be identical in the source run and in the
# run receiving its backbone. The detection levels only select the blocks read by the heads, and the convolutions do
# not depend on the image size, so both may differ
ARCHITECTURE_BACKBONE_KEYS = ["input_channels", "convolutions", "modules", "normalization", "last_max_pooling"]

# Name of the tensors of a backbone block in the backbone state, e.g. 'convolution_blocks.0.layers.0.weight', the
# group giving the index of the block from 0
BACKBONE_BLOCK_TENSOR_PATTERN = r"convolution_blocks\.(\d+)\."

# Checkpoint metadata of the trainers giving the number of backbone blocks loaded from a shorter source backbone, None
# when the whole backbone (or none of it) comes from another run, read back by a resumed run whose source run may be
# gone, so that it freezes and fine-tunes the same blocks
LOADED_BACKBONE_BLOCKS_METADATA_KEY = "pretrained_backbone_blocks"

# Backbone module settings added after the first runs, with the value of the runs written before them: a source run
# without the pre-activation placement of the shortcuts used the original one
BACKBONE_MODULE_DEFAULTS = {"residual_nets_v2": False}

# Neck section of the detection runs written before the neck existed, whose heads read the backbone blocks
NECK_DEFAULTS = {"Neck": {"type": "none"}}

# Backbone module settings removed from the configuration files, with the value the code always uses: a source run
# written before their removal still holds them, and has the same layers when it used that value
FIXED_BACKBONE_MODULE_SETTINGS = {"inception_starting_block": INCEPTION_STARTING_BLOCK}

# Feature map normalizations holding the same tensors (scale, shift and running statistics), so that a backbone moves
# between batch normalization and sequence normalization through the transfer
BATCH_STATISTICS_NORMALIZATIONS = ["batch", "sequence"]

# Normalization settings that change how the batch statistics follow the training batches but no tensor, so that a
# component moves between runs of different momentums, a source run written before the setting existed included, and
# a component from a run written before its momentum was fixed (BATCH_NORMALIZATION_MOMENTUM)
NON_WEIGHT_NORMALIZATION_KEYS = ["batch_normalization_momentum"]

# Temporal module settings defining its weights, for every type that has weights. The dropout only acts in training
# and the causal flag only changes which frames attend to each other, so neither changes the weights. The number of
# heads of the transformer does not change the shape of its weights, but it splits them differently between the heads,
# and its positional embedding is a weight of its own
TEMPORAL_WEIGHT_KEYS = {"attention_pooling": ["hidden_size"], "gru": ["hidden_size", "number_layers", "bidirectional"],
                        "lstm": ["hidden_size", "number_layers", "bidirectional"],
                        "transformer": ["hidden_size", "number_layers", "number_heads", "positional_embedding"],
                        "temporal_convolution": ["hidden_size", "number_layers", "kernel_size"]}

# Temporal modules whose frames may only see the frames before them
CAUSAL_TEMPORAL_TYPES = ["transformer", "temporal_convolution"]

# Segmentation decoder settings defining its weights. The dropout only acts in the segmentation head, which is not
# part of the decoder, so that a decoder serves runs of other classes
SEGMENTATION_DECODER_WEIGHT_KEYS = ["upsampling", "convolutions", "modules", "feature_map_normalization"]

# High resolution network settings defining its weights, the normalization compared without its batch normalization
# momentum. The width of the stem feature map it reads is compared on the weights themselves
HIGH_RESOLUTION_NETWORK_WEIGHT_KEYS = ["base_width", "modules_per_stage", "blocks_per_branch", "modules",
                                       "normalization"]

# High resolution network settings defining the weights of the keypoint head: the width of the features read by its
# heatmap and tag convolution, and the residual pairs, modules and normalization of the refinement of its higher
# resolution branch
HIGHER_RESOLUTION_BRANCH_WEIGHT_KEYS = ["base_width", "blocks_per_branch", "modules", "normalization"]

# Component of every detection level ('DetectionLevels.<level>.<component>') holding the layers of the region
# proposal, detection or mask head of that level, compared level by level. The RoIAlign pool size of the detection
# heads gives the size of their first fully connected layer
DETECTION_LEVEL_COMPONENTS = {"RegionProposal": "region_proposal", "Detector": "detector", "MaskHead": "mask_head"}

# Settings of the region proposal, detection and mask heads shared by every level and defining their weights. The
# number of anchors per position of the region proposal heads, giving the size of their output layers, is compared
# apart, the anchor scales and ratios themselves not being weights. The mask resolution and the upsampling of the
# mask heads give their upsampling layers, their backbone skips at the pool size the input of their convolution block,
# their fine inputs (image branch, backbone skips at the mask resolution) and fusion convolutions the layers before
# their predictor, and the sharing of every kind of head by the levels the number of heads, so that heads trained per
# level are refused by a run sharing one head and the reverse
DETECTION_HEAD_WEIGHT_KEYS = {"RegionProposal": ["shared", "normalization"],
                              "Detector": ["shared", "normalization", "modules"],
                              "MaskHead": ["shared", "mask_resolution", "upsampling", "normalization", "modules",
                                           "skip_blocks_at_pool_size", "image_branch_convolutions",
                                           "skip_blocks_at_mask_resolution", "fusion_convolutions"]}

# Settings of the neck defining its weights besides its type, which is compared first: the channels of its outputs and
# the normalization of its convolutions (without the batch normalization momentum, which changes no tensor)
NECK_WEIGHT_KEYS = ["output_channels", "normalization"]

# Settings of the heads added after the first runs, with the value of the runs written before them: the mask heads of
# a source run written before their finer inputs read the map of their level alone, and the heads of every kind
# written before their sharing are one head per level
DETECTION_HEAD_DEFAULTS = {"RegionProposal": {"shared": False}, "Detector": {"shared": False},
                           "MaskHead": {"shared": False, "skip_blocks_at_pool_size": [],
                                        "image_branch_convolutions": [0, 0], "skip_blocks_at_mask_resolution": [],
                                        "fusion_convolutions": [0, 0]}}

# Normalization layers whose scale and shift can be kept fixed, and batch normalization layers whose statistics a
# frozen component keeps
NORMALIZATION_LAYER_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)
BATCH_NORMALIZATION_LAYER_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

# Class layers of the components, one row (or output channel) per class or 4 rows per class for the box regressor, as
# [module name pattern, rows per class], the pattern matching the name of the module in the component state (in the
# state of the segmentation head and deep supervision heads for the decoder). When a run adds classes after those of
# its source run or holds them in another order, the rows of the source classes are copied to the index of their class
# in the new run and those of the new classes keep the random initialisation the layer of the new run was built with
CLASS_LAYERS = {"ClassificationHead": [[r"classifier", 1]],
                "Detector": [[r"\w+\.classifier", 1], [r"\w+\.bounding_box_regressor", 4]],
                "MaskHead": [[r"\w+\.mask_predictor", 1]],
                "SegmentationDecoder": [[r"segmentation_head\.\d+", 1], [r"deep_supervision_heads\.\w+", 1]]}

# Layers of the keypoint head whose output channels follow the keypoints of the classes, left out with its class
# layers: the last convolution of the quarter resolution head (heatmaps then tags, after the hidden convolution of a
# hidden width) and of the heatmap head of the higher resolution branch. They are not CLASS_LAYERS, whose rows move to
# the classes of the new run, a keypoint head loaded with them needing the same keypoint channels
KEYPOINT_CHANNEL_LAYERS = [r"prediction_convolution\.\d+", r"higher_resolution_branch\.heatmap_head\.\d+"]

# Transposed convolution of the higher resolution branch, reading the features of the network then the predictions of
# the quarter resolution head: its first input rows, as many as its output channels (the feature width), are loaded
# without the class layers, its rows of the predictions keeping the random initialisation of the new run
KEYPOINT_PREDICTION_INPUT_LAYER = r"higher_resolution_branch\.transposed_convolution"


def get_component_name(section_name: str) -> str:
    """
    Names the component of a transferable configuration section in the messages of the transfer, e.g. "temporal
    module" for the 'TemporalModel' section and "detection heads" for the 'Detector' section.

    Args:
        section_name (str): A key of TRANSFERABLE_COMPONENTS.

    Returns:
        str: The name of the component, in lower case.
    """
    if section_name in COMPONENT_NAMES:
        return COMPONENT_NAMES[section_name]
    return TRANSFERABLE_COMPONENTS[section_name].replace("_", " ")


def get_weight_normalization(normalization_configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the normalization settings of a backbone or of detection heads as far as their weights are concerned: batch
    and sequence normalization holding the same tensors, both are named "batch", so that a backbone moves between them
    through the transfer, and the batch normalization momentum, which changes no tensor, is left out.

    Args:
        normalization_configuration (Dict[str, Any]): The 'Backbone.normalization' settings, or the 'normalization'
            settings of the 'RegionProposal', 'Detector' or 'MaskHead' section.

    Returns:
        Dict[str, Any]: A copy of the settings without the momentum, sequence normalization written as batch
            normalization.
    """
    feature_map_normalization = normalization_configuration["feature_map_normalization"]
    if feature_map_normalization in BATCH_STATISTICS_NORMALIZATIONS:
        feature_map_normalization = "batch"
    weight_settings = {key_name: value for key_name, value in normalization_configuration.items()
                       if key_name not in NON_WEIGHT_NORMALIZATION_KEYS}
    return {**weight_settings, "feature_map_normalization": feature_map_normalization}


def count_shared_backbone_blocks(source_backbone_configuration: Dict[str, Any],
                                 target_backbone_configuration: Dict[str, Any]) -> Optional[int]:
    """
    Tells whether one backbone of a transfer is made of the first blocks of the other, the same layers with fewer
    blocks, and how many blocks they share: a shorter backbone takes the first blocks of a deeper source backbone (the
    stem of the keypoint detector taken from a backbone trained on ImageNet, for instance), and a deeper backbone starts
    with the blocks of a shorter source backbone, its blocks after them keeping their random initialisation. The blocks
    are compared on their key, layers and channels in order; the residual shortcuts and inception modules of a block
    follow its position, so that the shared blocks hold the same tensors. The pooling after the last block has no
    weights, so it may differ.

    Args:
        source_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the source run.
        target_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the run receiving the backbone.

    Returns:
        Optional[int]: The number of blocks of the shorter backbone when they are the first blocks of the other, None
            when both backbones have as many blocks or when their first blocks differ.
    """
    source_blocks = list(source_backbone_configuration["convolutions"].items())
    target_blocks = list(target_backbone_configuration["convolutions"].items())
    number_shared_blocks = min(len(source_blocks), len(target_blocks))
    if len(source_blocks) == len(target_blocks) or \
            source_blocks[:number_shared_blocks] != target_blocks[:number_shared_blocks]:
        return None
    return number_shared_blocks


def compare_backbone_architectures(source_backbone_configuration: Dict[str, Any],
                                   target_backbone_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the architecture-defining settings of two backbones, so that a backbone is only
    transferred between runs whose backbones have exactly the same layers, or when one backbone is made of the first
    blocks of the other, shorter or deeper (see count_shared_backbone_blocks). Batch and sequence normalization count
    as the same layers, holding the same tensors, and the batch normalization momentum may differ. A source run
    written before a module setting of BACKBONE_MODULE_DEFAULTS counts as having its default value, and one written
    before the removal of a module setting of FIXED_BACKBONE_MODULE_SETTINGS has the same layers only when it used the
    fixed value.

    Args:
        source_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the source run.
        target_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the run receiving the backbone.

    Returns:
        List[str]: The description of every difference, empty when the architectures match.
    """
    # The blocks of two backbones of which one is made of the first blocks of the other are compared by
    # count_shared_backbone_blocks, the pooling after the last block having no weights
    compared_keys = ARCHITECTURE_BACKBONE_KEYS
    if count_shared_backbone_blocks(source_backbone_configuration=source_backbone_configuration,
                                    target_backbone_configuration=target_backbone_configuration) is not None:
        compared_keys = [key_name for key_name in ARCHITECTURE_BACKBONE_KEYS
                         if key_name not in ["convolutions", "last_max_pooling"]]
    differences = []
    for key_name in compared_keys:
        source_value = source_backbone_configuration[key_name] if key_name in source_backbone_configuration else None
        target_value = target_backbone_configuration[key_name]

        # Normalization layers compared by their tensors, batch and sequence normalization being the same, the modules
        # of a source run written before a module setting completed with its default value, and the fixed module
        # settings, held only by the source runs written before their removal, compared with their fixed value
        compared_source_value, compared_target_value = source_value, target_value
        if key_name == "modules" and source_value is not None:
            compared_source_value = {**BACKBONE_MODULE_DEFAULTS, **FIXED_BACKBONE_MODULE_SETTINGS, **source_value}
            compared_target_value = {**target_value, **FIXED_BACKBONE_MODULE_SETTINGS}
        if key_name == "normalization" and source_value is not None:
            compared_source_value = get_weight_normalization(normalization_configuration=source_value)
            compared_target_value = get_weight_normalization(normalization_configuration=target_value)
        if compared_source_value != compared_target_value:
            differences.append(f"'Backbone.{key_name}' is {source_value!r} in the source run and {target_value!r} in "
                               f"this run.")
    return differences


def compare_flattened_feature_maps(source_configuration: Dict[str, Any],
                                   target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings giving the size of the vector made from the final feature map of the
    backbone, read by the classification head of the image classifier and by the temporal module of the sequence
    models: a flattened map gives one value per position and channel, a pooled one a value per channel only, and the
    positions of a flattened map follow the image size.

    Args:
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the component.

    Returns:
        List[str]: The description of every difference, empty when both runs give vectors of the same size.
    """
    differences = []
    source_pooling = source_configuration["ClassificationHead"]["global_pooling"]
    target_pooling = target_configuration["ClassificationHead"]["global_pooling"]
    if "flatten" in [source_pooling, target_pooling] and source_pooling != target_pooling:
        differences.append(f"'ClassificationHead.global_pooling' is {source_pooling!r} in the source run and "
                           f"{target_pooling!r} in this run: a flattened feature map and a pooled one give input "
                           f"vectors of different sizes.")
    elif source_pooling == "flatten":
        source_image_size = source_configuration["Data"]["image_settings"]["size"]
        target_image_size = target_configuration["Data"]["image_settings"]["size"]
        if source_image_size != target_image_size:
            differences.append(f"'Data.image_settings.size' is {source_image_size!r} in the source run and "
                               f"{target_image_size!r} in this run: the flattened feature map, the input vector, "
                               f"has a size following the image size.")
    return differences


def compare_temporal_modules(source_configuration: Dict[str, Any], target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the weights of the temporal modules of two runs, so that a
    temporal module is only transferred to a module of the same type and sizes: the type, the settings of that type
    (TEMPORAL_WEIGHT_KEYS), the channels of the last backbone block and, for a flattened feature map, the image size,
    which give the size of the frame vectors the module reads, and for a transformer with a positional embedding the
    maximum sequence length, which gives the size of the embedding.

    Args:
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the temporal module.

    Returns:
        List[str]: The description of every difference, empty when the weights of the modules match.
    """
    source_temporal_configuration = source_configuration["TemporalModel"]
    target_temporal_configuration = target_configuration["TemporalModel"]

    # Modules of different types have nothing in common
    if source_temporal_configuration["type"] != target_temporal_configuration["type"]:
        return [f"'TemporalModel.type' is {source_temporal_configuration['type']!r} in the source run and "
                f"{target_temporal_configuration['type']!r} in this run."]

    # Settings defining the weights of the type, a source run written before one of them having to be given it
    differences = []
    differing_keys = []
    temporal_type = target_temporal_configuration["type"]
    for key_name in TEMPORAL_WEIGHT_KEYS[temporal_type]:
        if key_name not in source_temporal_configuration:
            differing_keys.append(key_name)
            differences.append(f"'TemporalModel.{key_name}' is missing from the source run, written before the "
                               f"setting existed: add it with the value the run was trained with to its "
                               f"training_configuration.yaml.")
        elif source_temporal_configuration[key_name] != target_temporal_configuration[key_name]:
            differing_keys.append(key_name)
            differences.append(f"'TemporalModel.{key_name}' is {source_temporal_configuration[key_name]!r} in the "
                               f"source run and {target_temporal_configuration[key_name]!r} in this run.")

    # Size of the frame vectors read by the module, the channels of the last backbone block, times the positions of the
    # map when it is flattened
    source_frame_features = list(source_configuration["Backbone"]["convolutions"].values())[-1][1]
    target_frame_features = list(target_configuration["Backbone"]["convolutions"].values())[-1][1]
    if source_frame_features != target_frame_features:
        differences.append(f"The last backbone block has {source_frame_features} channels in the source run and "
                           f"{target_frame_features} in this run, the size of the frame vectors of the temporal "
                           f"module.")
    differences += compare_flattened_feature_maps(source_configuration=source_configuration,
                                                  target_configuration=target_configuration)

    # Positional embedding of both transformers, one vector per possible frame of a sequence, absent from an
    # order-blind one, a source run written before the setting existed having to be given it
    if temporal_type == "transformer" and target_temporal_configuration["positional_embedding"] and \
            "positional_embedding" not in differing_keys:
        source_sequence_settings = source_configuration["Data"]["sequence_settings"]
        target_sequence_length = target_configuration["Data"]["sequence_settings"]["maximum_sequence_length"]
        if "maximum_sequence_length" not in source_sequence_settings:
            differences.append("'Data.sequence_settings.maximum_sequence_length' is missing from the source run, "
                               "written before the setting existed: add it with the size of its positional embedding "
                               "to its training_configuration.yaml.")
        elif source_sequence_settings["maximum_sequence_length"] != target_sequence_length:
            differences.append(f"'Data.sequence_settings.maximum_sequence_length' is "
                               f"{source_sequence_settings['maximum_sequence_length']} in the source run and "
                               f"{target_sequence_length} in this run, the size of the positional embedding of the "
                               f"transformer.")
    return differences


def compare_classification_heads(source_configuration: Dict[str, Any],
                                 target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the layers of the classification heads of two runs: the
    hidden fully connected layers, their normalization and, for a flattened feature map, the settings giving the
    size of its input (the flattening itself, and the image size). The dropout rate and the batch normalization
    momentum, held by the source runs written before it was fixed, are not compared: the heads always hold their
    dropout layers, so the rate does not change the names of their weights, and the momentum changes no tensor.
    The other sizes of the input vectors (e.g. the channels of the last backbone block) are compared on the weights
    themselves, and the classes by load_pretrained_component, the run receiving the head being allowed to add classes
    after those of the source run or to hold them at other indices under the same names, or to have unrelated classes
    when the head is loaded without its class layer ('ClassificationHead.pretrained.class_layer' False, the default).

    Args:
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the classification head.

    Returns:
        List[str]: The description of every difference, empty when the layers of the heads match.
    """
    source_head_configuration = source_configuration["ClassificationHead"]
    target_head_configuration = target_configuration["ClassificationHead"]
    differences = []
    if source_head_configuration["fully_connected"] != target_head_configuration["fully_connected"]:
        differences.append(f"'ClassificationHead.fully_connected' is {source_head_configuration['fully_connected']!r} "
                           f"in the source run and {target_head_configuration['fully_connected']!r} in this run.")

    # Normalization layers of the hidden layers, a source run written before the setting existed, or with True or
    # False before the layer normalization existed, having to be given its current value
    if "fully_connected_normalization" not in source_head_configuration:
        differences.append("'ClassificationHead.fully_connected_normalization' is missing from the source run, "
                           "written before the setting existed: add it with the value the run was trained with to its "
                           "training_configuration.yaml.")
    elif isinstance(source_head_configuration["fully_connected_normalization"], bool):
        differences.append(f"'ClassificationHead.fully_connected_normalization' is "
                           f"{source_head_configuration['fully_connected_normalization']!r} in the source run, written "
                           f"before the layer normalization existed: write it as \"batch\" (True) or \"none\" (False) "
                           f"in its training_configuration.yaml.")
    elif source_head_configuration["fully_connected_normalization"] != \
            target_head_configuration["fully_connected_normalization"]:
        differences.append(f"'ClassificationHead.fully_connected_normalization' is "
                           f"{source_head_configuration['fully_connected_normalization']!r} in the source run and "
                           f"{target_head_configuration['fully_connected_normalization']!r} in this run.")

    # A flattened feature map gives one input per position and channel, a pooled one an input per channel only
    differences += compare_flattened_feature_maps(source_configuration=source_configuration,
                                                  target_configuration=target_configuration)
    return differences


def compare_segmentation_decoders(source_configuration: Dict[str, Any],
                                  target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the weights of the segmentation decoders of two runs: the
    upsampling, the convolutions of its levels (and therefore its levels), its modules and its normalization, and the
    channels of the backbone blocks it reads, through its skip connections and its deepest level. The segmentation
    head is not part of the decoder, so that the runs may segment other classes; it is loaded with the decoder by
    load_segmentation_class_layers when 'SegmentationDecoder.pretrained.class_layers' is True.

    Args:
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the decoder.

    Returns:
        List[str]: The description of every difference, empty when the weights of the decoders match.
    """
    source_decoder_configuration = source_configuration["SegmentationDecoder"]
    target_decoder_configuration = target_configuration["SegmentationDecoder"]
    differences = []
    for key_name in SEGMENTATION_DECODER_WEIGHT_KEYS:
        if source_decoder_configuration[key_name] != target_decoder_configuration[key_name]:
            differences.append(f"'SegmentationDecoder.{key_name}' is {source_decoder_configuration[key_name]!r} in "
                               f"the source run and {target_decoder_configuration[key_name]!r} in this run.")

    # Channels of the backbone blocks read by the decoder levels, in block order
    block_channels = []
    for configuration in [source_configuration, target_configuration]:
        backbone_convolutions = configuration["Backbone"]["convolutions"]
        block_channels.append([backbone_convolutions[block_key][1]
                               for block_key in sorted(backbone_convolutions, key=int)])
    if block_channels[0] != block_channels[1]:
        differences.append(f"The backbone blocks have {block_channels[0]} output channels in the source run and "
                           f"{block_channels[1]} in this run, the channels of the skip connections and of the "
                           f"bottleneck read by the decoder.")
    return differences


def compare_detection_heads(section_name: str, source_configuration: Dict[str, Any],
                            target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the weights of the region proposal heads, of the detection
    heads or of the mask heads of two runs of the detection tasks (DET or INS, INS only for the mask heads): their
    detection levels (the keys of 'DetectionLevels'), the neck whose outputs they read (the same 'Neck.type' in both
    runs, a source run written before the neck counting as none, and with a neck the same 'Neck.output_channels', the
    channels of the levels they read; the normalization of the neck changes no weight of the heads), the head of every
    level (DETECTION_LEVEL_COMPONENTS), the settings of DETECTION_HEAD_WEIGHT_KEYS (the normalization without its
    batch normalization momentum, which changes no tensor, and a source run written before a setting holding the value
    of DETECTION_HEAD_DEFAULTS) and, for the region proposal heads, the number of anchors per position of every level.
    The channels of the backbone blocks they read without neck are compared on the weights themselves, and the
    classes of the detection and mask heads (classifier, per-class box regressor, per-class mask predictor) by
    load_pretrained_component, the run receiving the heads being allowed to add classes after those of the source run
    or to hold them at other indices under the same names, or to have unrelated classes when the heads are loaded
    without their class layers ('pretrained.class_layers' False).

    Args:
        section_name (str): "RegionProposal", "Detector" or "MaskHead".
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the heads.

    Returns:
        List[str]: The description of every difference, empty when the weights of the heads match.
    """
    # Heads of other levels have nothing in common, one head being built per level
    source_levels = sorted(source_configuration["DetectionLevels"], key=int)
    target_levels = sorted(target_configuration["DetectionLevels"], key=int)
    if source_levels != target_levels:
        return [f"The detection levels are {source_levels!r} in the source run and {target_levels!r} in this run, "
                f"the levels of the {get_component_name(section_name=section_name)}."]

    # Heads read the levels after the neck: heads trained on the blocks would read the outputs of a neck and the
    # reverse, so they only move between runs of the same neck type, whose outputs have the same channels. Heads
    # trained behind a neck read its features best behind the same trained neck ('Neck.pretrained'), which
    # warn_heads_behind_another_neck reports when it comes from elsewhere
    differences = []
    component_name = get_component_name(section_name=section_name)
    source_neck_configuration = {**NECK_DEFAULTS, **source_configuration}["Neck"]
    target_neck_configuration = target_configuration["Neck"]
    if source_neck_configuration["type"] != target_neck_configuration["type"]:
        differences.append(f"'Neck.type' is {source_neck_configuration['type']!r} in the source run and "
                           f"{target_neck_configuration['type']!r} in this run: the {component_name} read the levels "
                           f"after the neck, the backbone blocks without neck.")
    elif target_neck_configuration["type"] != "none" and \
            source_neck_configuration["output_channels"] != target_neck_configuration["output_channels"]:
        differences.append(f"'Neck.output_channels' is {source_neck_configuration['output_channels']!r} in the source "
                           f"run and {target_neck_configuration['output_channels']!r} in this run, the channels of "
                           f"the levels read by the {component_name}.")

    # Layers of the head of every level
    level_component = DETECTION_LEVEL_COMPONENTS[section_name]
    for level_key in target_levels:
        source_level_value = source_configuration["DetectionLevels"][level_key][level_component]
        target_level_value = target_configuration["DetectionLevels"][level_key][level_component]
        if source_level_value != target_level_value:
            differences.append(f"'DetectionLevels.{level_key}.{level_component}' is {source_level_value!r} in the "
                               f"source run and {target_level_value!r} in this run.")

    # Settings of the layers of the heads shared by every level, a source run written before a setting holding its
    # former value (DETECTION_HEAD_DEFAULTS)
    source_section_configuration = {**DETECTION_HEAD_DEFAULTS[section_name], **source_configuration[section_name]}
    target_section_configuration = target_configuration[section_name]
    for key_name in DETECTION_HEAD_WEIGHT_KEYS[section_name]:
        compared_source_value = source_section_configuration[key_name]
        compared_target_value = target_section_configuration[key_name]
        if key_name == "normalization":
            compared_source_value = get_weight_normalization(normalization_configuration=compared_source_value)
            compared_target_value = get_weight_normalization(normalization_configuration=compared_target_value)
        if compared_source_value != compared_target_value:
            differences.append(f"'{section_name}.{key_name}' is {source_section_configuration[key_name]!r} in the "
                               f"source run and {target_section_configuration[key_name]!r} in this run.")

    # Anchors per position of every level (scales x ratios), the number of outputs of the region proposal heads
    if section_name == "RegionProposal":
        anchor_counts = []
        for configuration in [source_configuration, target_configuration]:
            detection_levels = configuration["DetectionLevels"]
            anchor_counts.append({level_key: len(detection_levels[level_key]["anchors"]["scales"]) *
                                  len(detection_levels[level_key]["anchors"]["aspect_ratios"])
                                  for level_key in target_levels})
        if anchor_counts[0] != anchor_counts[1]:
            differences.append(f"The levels have {anchor_counts[0]} anchors per position in the source run and "
                               f"{anchor_counts[1]} in this run, the outputs of the region proposal heads.")
    return differences


def compare_necks(source_configuration: Dict[str, Any], target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the weights of the necks of two runs of the detection tasks
    (DET or INS), the layers between the backbone blocks of the detection levels and the region proposal, detection and
    mask heads: the type of the neck, then its output channels and the normalization of its convolutions
    (NECK_WEIGHT_KEYS, without the batch normalization momentum, which changes no tensor), the detection levels, one
    lateral and one output convolution being built per level, and the output channels of the backbone blocks of the
    levels, the input channels of the laterals. A run without neck ('none') has no layer to load, which the
    configuration check refuses before the transfer.

    Args:
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the neck.

    Returns:
        List[str]: The description of every difference, empty when the weights of the necks match.
    """
    # Necks of other types hold other layers, nothing else being comparable
    source_neck_configuration = source_configuration["Neck"]
    target_neck_configuration = target_configuration["Neck"]
    if source_neck_configuration["type"] != target_neck_configuration["type"]:
        return [f"'Neck.type' is {source_neck_configuration['type']!r} in the source run and "
                f"{target_neck_configuration['type']!r} in this run."]

    # Output channels and normalization of every convolution of the neck
    differences = []
    for key_name in NECK_WEIGHT_KEYS:
        compared_source_value = source_neck_configuration[key_name]
        compared_target_value = target_neck_configuration[key_name]
        if key_name == "normalization":
            compared_source_value = get_weight_normalization(normalization_configuration=compared_source_value)
            compared_target_value = get_weight_normalization(normalization_configuration=compared_target_value)
        if compared_source_value != compared_target_value:
            differences.append(f"'Neck.{key_name}' is {source_neck_configuration[key_name]!r} in the source run and "
                               f"{target_neck_configuration[key_name]!r} in this run.")

    # One lateral and one output convolution per level, whose blocks give the input channels of the laterals
    source_levels = sorted(source_configuration["DetectionLevels"], key=int)
    target_levels = sorted(target_configuration["DetectionLevels"], key=int)
    if source_levels != target_levels:
        differences.append(f"The detection levels are {source_levels!r} in the source run and {target_levels!r} in "
                           f"this run, one lateral and one output convolution of the neck per level.")
        return differences
    block_channels = [{level_key: configuration["Backbone"]["convolutions"][level_key][1]
                       for level_key in target_levels}
                      for configuration in [source_configuration, target_configuration]]
    if block_channels[0] != block_channels[1]:
        differences.append(f"The backbone blocks of the levels have {block_channels[0]} output channels in the source "
                           f"run and {block_channels[1]} in this run, the input channels of the laterals of the neck.")
    return differences


def compare_high_resolution_networks(source_configuration: Dict[str, Any], target_configuration: Dict[str, Any],
                                     key_names: List[str]) -> List[str]:
    """
    Lists the differences between the 'HRNet' settings of two keypoint detection runs that define the weights of a
    component: for the high resolution network itself (HIGH_RESOLUTION_NETWORK_WEIGHT_KEYS), the width of its highest
    resolution branch, its stages, the convolutions of its branches, their modules and their normalization (without
    the batch normalization momentum, which changes no tensor), the width of the stem feature map it reads being
    compared on the weights themselves; for the keypoint head, the subset its layers follow
    (HIGHER_RESOLUTION_BRANCH_WEIGHT_KEYS).

    Args:
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the component.
        key_names (List[str]): The 'HRNet' settings compared.

    Returns:
        List[str]: The description of every difference, empty when the weights of the components match.
    """
    differences = []
    for key_name in key_names:
        source_value = source_configuration["HRNet"][key_name]
        target_value = target_configuration["HRNet"][key_name]
        compared_source_value, compared_target_value = source_value, target_value
        if key_name == "normalization":
            compared_source_value = get_weight_normalization(normalization_configuration=source_value)
            compared_target_value = get_weight_normalization(normalization_configuration=target_value)
        if compared_source_value != compared_target_value:
            differences.append(f"'HRNet.{key_name}' is {source_value!r} in the source run and {target_value!r} in "
                               f"this run.")
    return differences


def compare_keypoint_heads(source_configuration: Dict[str, Any], target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the weights of the keypoint heads of two keypoint detection
    runs: the higher resolution branch, on in both runs or in neither, the width of the hidden convolution of the
    heads, the width of the high resolution network read by the heads, the normalization of the hidden convolution
    and, with the higher resolution branch, the residual pairs, modules and normalization of its refinement. A run of
    the top-down keypoint detector, saved before the bottom-up one, has no tag channels and cannot initialise the head.
    The heatmap and tag channels follow the keypoints of the classes, which load_pretrained_component checks on the
    channel names of the runs (compare_keypoint_channels) and on the weights themselves when the head is loaded with
    its keypoint channel layers ('KeypointHead.pretrained.class_layers' True), and leaves out otherwise.

    Args:
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the head.

    Returns:
        List[str]: The description of every difference, empty when the weights of the heads match.
    """
    # The top-down keypoint detector predicted the heatmaps of object crops, without tags nor higher resolution branch
    if "higher_resolution" not in source_configuration["KeypointHead"]:
        return ["The source run is a top-down keypoint detector, whose keypoint head has no tag channels."]
    source_higher_resolution = source_configuration["KeypointHead"]["higher_resolution"]
    target_higher_resolution = target_configuration["KeypointHead"]["higher_resolution"]
    if source_higher_resolution != target_higher_resolution:
        return [f"'KeypointHead.higher_resolution' is {source_higher_resolution!r} in the source run and "
                f"{target_higher_resolution!r} in this run."]

    # Width of the hidden convolution of both heads, 0 for the 1x1 predictions read on the features
    source_hidden_width = source_configuration["KeypointHead"]["hidden_width"]
    target_hidden_width = target_configuration["KeypointHead"]["hidden_width"]
    if source_hidden_width != target_hidden_width:
        return [f"'KeypointHead.hidden_width' is {source_hidden_width!r} in the source run and "
                f"{target_hidden_width!r} in this run."]

    # Settings of the high resolution network the layers of the head follow, its normalization being that of the
    # hidden convolution
    key_names = list(HIGHER_RESOLUTION_BRANCH_WEIGHT_KEYS) if target_higher_resolution else ["base_width"]
    if target_hidden_width > 0 and "normalization" not in key_names:
        key_names.append("normalization")
    return compare_high_resolution_networks(source_configuration=source_configuration,
                                            target_configuration=target_configuration, key_names=key_names)


def compare_component_configurations(section_name: str, source_configuration: Dict[str, Any],
                                     target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings of the component of a section in the source run and in the run
    receiving it, read from the configurations before any weight is loaded so that the user gets the settings to
    change rather than the names of mismatching tensors.

    Args:
        section_name (str): A key of TRANSFERABLE_COMPONENTS.
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the component.

    Returns:
        List[str]: The description of every difference, empty when the components match.
    """
    # The backbone, the temporal module and the segmentation decoder, each compared on the settings shaping its weights
    if section_name == "Backbone":
        return compare_backbone_architectures(source_backbone_configuration=source_configuration["Backbone"],
                                              target_backbone_configuration=target_configuration["Backbone"])
    if section_name == "TemporalModel":
        return compare_temporal_modules(source_configuration=source_configuration,
                                        target_configuration=target_configuration)
    if section_name == "SegmentationDecoder":
        return compare_segmentation_decoders(source_configuration=source_configuration,
                                             target_configuration=target_configuration)
    # The neck and the heads of the detectors, the high resolution network and the keypoint head
    if section_name == "Neck":
        return compare_necks(source_configuration=source_configuration, target_configuration=target_configuration)
    if section_name in DETECTION_HEAD_WEIGHT_KEYS:
        return compare_detection_heads(section_name=section_name, source_configuration=source_configuration,
                                       target_configuration=target_configuration)
    if section_name == "HRNet":
        return compare_high_resolution_networks(source_configuration=source_configuration,
                                                target_configuration=target_configuration,
                                                key_names=HIGH_RESOLUTION_NETWORK_WEIGHT_KEYS)
    if section_name == "KeypointHead":
        return compare_keypoint_heads(source_configuration=source_configuration,
                                      target_configuration=target_configuration)
    return compare_classification_heads(source_configuration=source_configuration,
                                        target_configuration=target_configuration)


def compare_state_dictionaries(source_state: Dict[str, torch.Tensor],
                               target_state: Dict[str, torch.Tensor]) -> List[str]:
    """
    Lists the tensors that prevent the weights of a component of the source run from being loaded strictly into the
    component of the new run: the tensors missing from the source run, the tensors the new run does not have, and the
    tensors of different shapes, e.g. the first layer of a sequence classifier head loaded into a comparison head,
    which reads two or three member vectors.

    Args:
        source_state (Dict[str, torch.Tensor]): The tensors of the component in the source checkpoint, without prefix.
        target_state (Dict[str, torch.Tensor]): The state dictionary of the component of the new run.

    Returns:
        List[str]: The description of every mismatch, empty when the source tensors fit the component exactly.
    """
    differences = []
    missing_names = [tensor_name for tensor_name in target_state if tensor_name not in source_state]
    unexpected_names = [tensor_name for tensor_name in source_state if tensor_name not in target_state]
    if missing_names:
        differences.append(f"Tensors missing from the source run: {missing_names}.")
    if unexpected_names:
        differences.append(f"Tensors of the source run unknown to this run: {unexpected_names}.")

    # Tensors present in both, compared shape by shape
    for tensor_name, target_tensor in target_state.items():
        if tensor_name in source_state and source_state[tensor_name].shape != target_tensor.shape:
            differences.append(f"'{tensor_name}' has the shape {list(source_state[tensor_name].shape)} in the source "
                               f"run and {list(target_tensor.shape)} in this run.")
    return differences


def read_source_class_names(source_folder: Path) -> Optional[List[str]]:
    """
    Reads the class names of a source run, in class index order, from the record of its data ('data.json' of its run
    folder), so that a run adding classes after those of its source run is checked to keep them, first and in the
    same order or under the same names (see compare_class_names). The names are read from the 'classes' of the
    record, or from the 'class_names' of the records written before the classes were counted; the runs written before
    the class names were recorded have none.

    Args:
        source_folder (Path): The run folder of the source run.

    Returns:
        Optional[List[str]]: The class names of the source run, None when its data record or its class names are
            missing.
    """
    data_record_path = source_folder / DATA_RECORD_FILE_NAME
    if not data_record_path.is_file():
        return None
    data_record = read_json(path=str(data_record_path))

    # Records with counted classes, then records with the class names only, then records without classes
    if CLASSES_RECORD_KEY in data_record:
        return None if data_record[CLASSES_RECORD_KEY] is None else [
            class_record["class_name"] for class_record in data_record[CLASSES_RECORD_KEY]]
    if CLASS_NAMES_RECORD_KEY in data_record:
        return data_record[CLASS_NAMES_RECORD_KEY]
    return None


def read_source_keypoint_channels(source_folder: Path) -> Optional[List[str]]:
    """
    Reads the heatmap and tag channels of the keypoint head of a source run from the record of its data ('data.json'),
    the 'class.keypoint' name of every channel, the keypoints of every class in their order and the classes in index
    order, as KeypointDetector assigns them (build_keypoint_channel_table), so that the transfer of the head checks
    that every channel predicts the same keypoint in the new run.

    Args:
        source_folder (Path): The run folder of the source run.

    Returns:
        Optional[List[str]]: The name of every channel, None when the source run has no data record, no class records
            or no keypoint names in them (runs written before the keypoints were recorded).
    """
    data_record_path = source_folder / DATA_RECORD_FILE_NAME
    if not data_record_path.is_file():
        return None
    data_record = read_json(path=str(data_record_path))
    if CLASSES_RECORD_KEY not in data_record or data_record[CLASSES_RECORD_KEY] is None:
        return None

    # Channels of the classes defining keypoints, the record listing the classes in class index order
    class_records = data_record[CLASSES_RECORD_KEY]
    if not any(KEYPOINTS_RECORD_KEY in class_record for class_record in class_records):
        return None
    return [f"{class_record['class_name']}.{keypoint_name}" for class_record in class_records
            if KEYPOINTS_RECORD_KEY in class_record for keypoint_name in class_record[KEYPOINTS_RECORD_KEY]]


def list_keypoint_channels(keypoint_categories: Dict[Any, Dict[str, Any]], class_names: List[str]) -> List[str]:
    """
    Names the heatmap and tag channels of the keypoint head of the new run like read_source_keypoint_channels names
    those of the source run: 'class.keypoint', the keypoints of every class in their order and the classes in index
    order, the order of build_keypoint_channel_table, with the class names of the record of the data.

    Args:
        keypoint_categories (Dict[Any, Dict[str, Any]]): The 'keypoints' names of every class index
            ('Data.keypoint_categories' of the new run).
        class_names (List[str]): The class names of the new run, in class index order.

    Returns:
        List[str]: The name of every channel.
    """
    return [f"{class_names[int(class_index)]}.{keypoint_name}"
            for class_index in sorted(keypoint_categories, key=int)
            for keypoint_name in keypoint_categories[class_index]["keypoints"]]


def compare_keypoint_channels(source_channels: List[str], target_channels: List[str]) -> List[str]:
    """
    Lists the channels of a source keypoint head that would predict another keypoint in the new run: the same number
    of keypoints in another order, other keypoints, or the same total split differently between the classes give
    tensors of the same shapes, copied as they are, channel 0 of the source run being read as channel 0 of the new run.

    Args:
        source_channels (List[str]): The 'class.keypoint' name of every channel of the source run.
        target_channels (List[str]): The 'class.keypoint' name of every channel of the new run.

    Returns:
        List[str]: The description of every difference (the first MAXIMUM_LISTED_DIFFERENCES differing channels and
            their count), empty when every channel predicts the same keypoint.
    """
    if len(source_channels) != len(target_channels):
        return [f"The Source Run Predicts {len(source_channels)} Keypoint Channels And This Run "
                f"{len(target_channels)}."]

    # Channels predicting another keypoint, the first ones named and the others counted
    differing_channels = [f"Channel {channel_index} Predicts {source_channel!r} In The Source Run And "
                          f"{target_channel!r} In This Run."
                          for channel_index, (source_channel, target_channel)
                          in enumerate(zip(source_channels, target_channels)) if source_channel != target_channel]
    if len(differing_channels) > MAXIMUM_LISTED_DIFFERENCES:
        return (differing_channels[:MAXIMUM_LISTED_DIFFERENCES] +
                [f"({len(differing_channels)} Channels Differ.)"])
    return differing_channels


def find_class_layer(section_name: str, module_name: str) -> Optional[int]:
    """
    Tells whether a module of a component is one of its class layers (see CLASS_LAYERS), whose rows follow the classes
    of the run and move to the classes of a run adding classes after those of its source run or reordering them.

    Args:
        section_name (str): A key of CLASS_LAYERS.
        module_name (str): The name of the module in the component state, e.g. 'classifier' or '5.mask_predictor'.

    Returns:
        Optional[int]: The number of rows of every class, None for a module that is not a class layer.
    """
    for module_pattern, rows_per_class in CLASS_LAYERS[section_name]:
        if re.fullmatch(pattern=module_pattern, string=module_name):
            return rows_per_class
    return None


def count_state_classes(state: Dict[str, torch.Tensor], section_name: str) -> Optional[int]:
    """
    Counts the classes predicted by a component from the rows of its first class layer, for the source component read
    from a checkpoint and for the component of the new run, every class layer of a component having the same classes.

    Args:
        state (Dict[str, torch.Tensor]): The tensors of the component, without prefix.
        section_name (str): A key of CLASS_LAYERS.

    Returns:
        Optional[int]: The number of classes, None for a component without class layer (e.g. the decoder itself).
    """
    for tensor_name, tensor_value in state.items():
        if not tensor_name.endswith(".weight"):
            continue
        rows_per_class = find_class_layer(section_name=section_name, module_name=tensor_name[:-len(".weight")])
        if rows_per_class is not None:
            return tensor_value.shape[0] // rows_per_class
    return None


def compare_class_names(source_class_names: Optional[List[str]], target_class_names: Optional[List[str]],
                        number_source_classes: int, number_target_classes: int) -> Tuple[List[int], List[str]]:
    """
    Finds where the classes of a source run sit in a new run, so that the class layers of the source run initialise
    those of the new run, and lists what prevents it: the new run keeps every class of the source run, either first
    and in the same order (background first for the detection and segmentation tasks) with its new classes after them,
    or, when both runs have class names, at any index under the same name, e.g. the 1,000 ImageNet-1k classes of a
    source run scattered among the 21,842 classes sorted by WordNet identifier of a merged ImageNet run. The names of a
    source run written before they were recorded being unknown, its classes are taken as the first classes of the new
    run.

    Args:
        source_class_names (Optional[List[str]]): The class names of the source run, None when it has none.
        target_class_names (Optional[List[str]]): The class names of the new run, None for a task without them.
        number_source_classes (int): The number of classes of the class layers of the source checkpoint.
        number_target_classes (int): The number of classes of the class layers of the new run.

    Returns:
        Tuple[List[int], List[str]]: The index in the new run of every source class, in source class order, and the
            description of every difference (the first MAXIMUM_LISTED_DIFFERENCES missing classes and their count),
            empty when every source class is found.
    """
    first_class_indices = list(range(number_source_classes))
    if number_target_classes < number_source_classes:
        return first_class_indices, [f"The source run predicts {number_source_classes} classes and this run "
                                     f"{number_target_classes}: a run keeps every class of its source run."]
    if source_class_names is None or target_class_names is None:
        return first_class_indices, []

    # The record of the data of the source run follows its last start, which may have changed its classes
    if len(source_class_names) != number_source_classes:
        return first_class_indices, [f"The data record of the source run names {len(source_class_names)} classes and "
                                     f"its checkpoint predicts {number_source_classes} classes."]

    # Source classes first and in the same order: every source class keeps its index
    if target_class_names[:number_source_classes] == source_class_names:
        return first_class_indices, []

    # Otherwise every source class is found by its name, which must name a single class in each run
    target_class_index_lists = {}
    for class_index, class_name in enumerate(target_class_names):
        target_class_index_lists.setdefault(class_name, []).append(class_index)
    source_class_counts = {}
    for class_name in source_class_names:
        source_class_counts[class_name] = source_class_counts.get(class_name, 0) + 1
    repeated_classes = [class_name for class_name, class_count in source_class_counts.items()
                        if class_count > 1 or len(target_class_index_lists.get(class_name, [])) > 1]
    if repeated_classes:
        return first_class_indices, [f"The class names {repeated_classes[:MAXIMUM_LISTED_DIFFERENCES]} name several "
                                     f"classes of the source run or of this run, so that the classes of the source "
                                     f"run cannot be found by their names."]
    missing_classes = [class_name for class_name in source_class_names if class_name not in target_class_index_lists]
    if missing_classes:
        counted_classes = f" ({len(missing_classes)} classes missing)" if len(missing_classes) > \
            MAXIMUM_LISTED_DIFFERENCES else ""
        return first_class_indices, [f"The classes {missing_classes[:MAXIMUM_LISTED_DIFFERENCES]}{counted_classes} of "
                                     f"the source run are not classes of this run: a run keeps every class of its "
                                     f"source run, first and in the same order or under the same name."]
    return [target_class_index_lists[class_name][0] for class_name in source_class_names], []


def extend_class_layer(weight: torch.Tensor, bias: torch.Tensor, target_weight: torch.Tensor,
                       target_bias: torch.Tensor, target_class_indices: List[int],
                       rows_per_class: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Moves a class layer of a source run to the classes of a new run that keeps its classes, after them or under the
    same names at other indices (see compare_class_names): the rows (output channels of a convolution) of every source
    class are copied to the rows of its class in the new run, and the rows of the other classes keep the random
    initialisation the layer of the new run received when it was built, as in a run trained from scratch. The rows of a
    class follow each other (4 for the box regressor), so class c owns the rows c x rows per class to (c + 1) x rows per
    class - 1 in both layers.

    Args:
        weight (torch.Tensor): The [source classes x rows per class, ...] weight of the source layer.
        bias (torch.Tensor): The [source classes x rows per class] bias of the source layer.
        target_weight (torch.Tensor): The [target classes x rows per class, ...] randomly initialised weight of the
            layer of the new run.
        target_bias (torch.Tensor): The [target classes x rows per class] randomly initialised bias of that layer.
        target_class_indices (List[int]): The index in the new run of every source class, in source class order.
        rows_per_class (int): The number of rows of every class (see CLASS_LAYERS).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [target classes x rows per class, ...] weight and the bias of the layer
            of the new run, on the device and in the precision of the source layer.
    """
    # Random initialisation of the new run kept for the other classes
    extended_weight = target_weight.detach().clone().to(device=weight.device, dtype=weight.dtype)
    extended_bias = target_bias.detach().clone().to(device=bias.device, dtype=bias.dtype)

    # Rows of every source class in the new layer, in the order of the source rows
    class_indices = torch.tensor(data=target_class_indices, dtype=torch.long, device=weight.device)
    row_offsets = torch.arange(end=rows_per_class, dtype=torch.long, device=weight.device)
    target_rows = (class_indices[:, None] * rows_per_class + row_offsets[None, :]).reshape(-1)
    extended_weight[target_rows] = weight
    extended_bias[target_rows] = bias
    return extended_weight, extended_bias


def extend_class_layers(component_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor],
                        section_name: str, target_class_indices: List[int]) -> Dict[str, torch.Tensor]:
    """
    Moves every class layer of a source component to the classes of the new run (see extend_class_layer) when its
    layer in the new run reads inputs of the same size and holds the rows of every target class index, the other
    tensors being kept as they are, so that compare_state_dictionaries still reports every other mismatch. A layer
    whose source classes keep their indices in a run of as many classes is kept as it is.

    Args:
        component_state (Dict[str, torch.Tensor]): The tensors of the source component, without prefix.
        target_state (Dict[str, torch.Tensor]): The state dictionary of the component of the new run, holding the
            random initialisation of its layers.
        section_name (str): A key of CLASS_LAYERS.
        target_class_indices (List[int]): The index in the new run of every source class, in source class order (see
            compare_class_names).

    Returns:
        Dict[str, torch.Tensor]: The tensors of the source component with its class layers moved to the classes of
            the new run.
    """
    extended_state = dict(component_state)
    keeps_class_indices = target_class_indices == list(range(len(target_class_indices)))
    for tensor_name, source_weight in component_state.items():
        if not tensor_name.endswith(".weight"):
            continue
        module_name = tensor_name[:-len(".weight")]
        bias_name = f"{module_name}.bias"
        rows_per_class = find_class_layer(section_name=section_name, module_name=module_name)
        if rows_per_class is None or tensor_name not in target_state or bias_name not in component_state:
            continue

        # Only a layer reading inputs of the same size, holding the rows of the source classes and of their target
        # indices, moves; a layer of as many classes in the same order stays
        target_weight = target_state[tensor_name]
        if target_weight.shape[1:] != source_weight.shape[1:] or \
                source_weight.shape[0] != len(target_class_indices) * rows_per_class or \
                target_weight.shape[0] < (max(target_class_indices, default=-1) + 1) * rows_per_class or \
                (keeps_class_indices and target_weight.shape[0] == source_weight.shape[0]):
            continue
        extended_state[tensor_name], extended_state[bias_name] = extend_class_layer(
            weight=source_weight, bias=component_state[bias_name], target_weight=target_weight,
            target_bias=target_state[bias_name], target_class_indices=target_class_indices,
            rows_per_class=rows_per_class)
    return extended_state


def prepare_class_layers(component_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor],
                         section_name: str, source_folder: Path,
                         class_names: Optional[List[str]]) -> Dict[str, torch.Tensor]:
    """
    Checks the classes of a source component against those of the new run (see compare_class_names) and moves its
    class layers to the classes of the new run (see extend_class_layers) when the new run adds classes after them or
    holds them at other indices under the same names, the other classes keeping the random initialisation of the new
    run, for the components with class layers: the classification head, the detection heads, the mask heads and the
    class layers of the semantic segmentation. A source run without class names prints that its classes are taken as
    the first classes of the new run.

    Args:
        component_state (Dict[str, torch.Tensor]): The tensors of the source component, without prefix.
        target_state (Dict[str, torch.Tensor]): The state dictionary of the component of the new run.
        section_name (str): A key of CLASS_LAYERS.
        source_folder (Path): The run folder of the source run.
        class_names (Optional[List[str]]): The class names of the new run, in class index order, None for a task
            without them.

    Returns:
        Dict[str, torch.Tensor]: The tensors of the source component, its class layers moved to the classes of the new
            run.

    Raises:
        ValueError: If a class of the source run is neither among the first classes of the new run, in the same order,
            nor found under its name in the new run.
    """
    number_source_classes = count_state_classes(state=component_state, section_name=section_name)
    number_target_classes = count_state_classes(state=target_state, section_name=section_name)
    if number_source_classes is None or number_target_classes is None:
        return component_state

    # Every source class must be found in the new run, at its index or under its name
    component_name = get_component_name(section_name=section_name)
    source_class_names = read_source_class_names(source_folder=source_folder)
    target_class_indices, class_differences = compare_class_names(
        source_class_names=source_class_names, target_class_names=class_names,
        number_source_classes=number_source_classes, number_target_classes=number_target_classes)
    if class_differences:
        raise ValueError(f"The {component_name.title()} Of The Source Run {format_path_link(path=source_folder)} "
                         f"Cannot Initialise This Run, Their Classes Differ: " + " ".join(class_differences))
    keeps_class_indices = target_class_indices == list(range(number_source_classes))
    if number_target_classes == number_source_classes and keeps_class_indices:
        return component_state

    # New classes appended after those of the source run, or source classes moved to the index of their name
    number_other_classes = number_target_classes - number_source_classes
    if source_class_names is None:
        print_color(color="yellow", output=f"The Source Run Has No Class Names In Its {DATA_RECORD_FILE_NAME} (Written "
                                           f"Before They Were Recorded): Its {number_source_classes} Classes Are Taken "
                                           f"As The First {number_source_classes} Classes Of This Run.")
    if keeps_class_indices:
        print_color(color="green", output=f"{component_name.title()} : {number_source_classes} Classes Of The Source "
                                          f"Run Kept, {number_other_classes} New Classes Added After Them, Randomly "
                                          f"Initialised.")
    else:
        print_color(color="green", output=f"{component_name.title()} : {number_source_classes} Classes Of The Source "
                                          f"Run Moved To The Index Of Their Name In This Run, {number_other_classes} "
                                          f"Other Classes Randomly Initialised.")
    return extend_class_layers(component_state=component_state, target_state=target_state, section_name=section_name,
                               target_class_indices=target_class_indices)


def transfers_class_layers(section_name: str, pretrained_configuration: Dict[str, Any]) -> bool:
    """
    Tells whether the transfer of a component loads its class layers, which decides whether the classes of the source
    run are checked against those of the new run (prepare_class_layers, or the keypoint channels for the keypoint head)
    or left out (keep_target_class_layers), and whether the class layers belong to the transferred part that the trainer
    freezes and fine-tunes at the learning rate multiplier of the transfer (select_transferred_part): the flag of
    PRETRAINED_CLASS_LAYER_KEYS ('class_layer' of the classification head, 'class_layers' of the detection, mask and
    keypoint heads) for the sections of CLASS_LAYER_OPTION_SECTIONS, always True for the other sections, whose
    components hold no class layer (the class layers of the segmentation decoder lying outside it).

    Args:
        section_name (str): The configuration section of the component, a key of TRANSFERABLE_COMPONENTS.
        pretrained_configuration (Dict[str, Any]): The 'pretrained' settings of the section.

    Returns:
        bool: True when the class layers of the source component are loaded with it.
    """
    if section_name not in CLASS_LAYER_OPTION_SECTIONS:
        return True
    return pretrained_configuration[PRETRAINED_CLASS_LAYER_KEYS[section_name]]


def find_left_out_layer(section_name: str, module_name: str) -> bool:
    """
    Tells whether a module of a component is one of the layers its transfer leaves out with the class layers
    ('pretrained.class_layer' or 'class_layers' False): a class layer of CLASS_LAYERS, or a layer of the keypoint head
    whose output channels follow the keypoints (KEYPOINT_CHANNEL_LAYERS). The transposed convolution of the higher
    resolution branch, only partly left out, is not one of them (see KEYPOINT_PREDICTION_INPUT_LAYER).

    Args:
        section_name (str): A section of CLASS_LAYER_OPTION_SECTIONS.
        module_name (str): The name of the module in the component state, e.g. '5.classifier' or
            'prediction_convolution.1'.

    Returns:
        bool: True for a layer left out with the class layers.
    """
    if section_name == "KeypointHead":
        return any(re.fullmatch(pattern=module_pattern, string=module_name) is not None
                   for module_pattern in KEYPOINT_CHANNEL_LAYERS)
    return find_class_layer(section_name=section_name, module_name=module_name) is not None


def keep_target_class_layers(component_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor],
                             section_name: str, source_folder: Path) -> Dict[str, torch.Tensor]:
    """
    Leaves the class layers of a source component out of its transfer ('pretrained.class_layer' or 'class_layers'
    False), so that a head trained on other classes (e.g. the 1,000 ImageNet-1k classes, or the shapes of a detection
    run) still gives its other layers to a run of unrelated classes: the tensors of every class layer (see
    find_left_out_layer) take the random initialisation the layer of the new run was built with, whatever the classes
    or keypoint channels of both runs, which are not compared, and the other tensors are kept as they are, so that
    compare_state_dictionaries still reports their mismatches. The transposed convolution of the higher resolution
    branch of the keypoint head keeps its input rows of the features from the source run and its rows of the
    predictions from the new run (KEYPOINT_PREDICTION_INPUT_LAYER).

    Args:
        component_state (Dict[str, torch.Tensor]): The tensors of the source component, without prefix.
        target_state (Dict[str, torch.Tensor]): The state dictionary of the component of the new run, holding the
            random initialisation of its layers.
        section_name (str): A section of CLASS_LAYER_OPTION_SECTIONS.
        source_folder (Path): The run folder of the source run, named in the messages.

    Returns:
        Dict[str, torch.Tensor]: The tensors of the source component, its class layers replaced by those of the new
            run.

    Raises:
        ValueError: If the source component holds no tensor besides its class layers, so that nothing would be loaded.
    """
    # Tensors of the source component outside its class layers
    component_name = get_component_name(section_name=section_name)
    kept_state = {tensor_name: tensor_value for tensor_name, tensor_value in component_state.items()
                  if not find_left_out_layer(section_name=section_name, module_name=tensor_name.rsplit(".", 1)[0])}
    if not kept_state:
        raise ValueError(f"The {component_name.title()} Of The Source Run {format_path_link(path=source_folder)} Has "
                         f"No Layer Besides Its Class Layers: Loaded Without Them, It Would Initialise Nothing.")

    # Class layers of the new run, as it built them
    class_layer_names = {tensor_name.rsplit(".", 1)[0] for tensor_name in target_state
                         if find_left_out_layer(section_name=section_name, module_name=tensor_name.rsplit(".", 1)[0])}
    kept_state.update({tensor_name: tensor_value.detach().clone() for tensor_name, tensor_value in target_state.items()
                       if tensor_name.rsplit(".", 1)[0] in class_layer_names})

    # Input rows of the features of the transposed convolution of the higher resolution branch loaded, its rows of the
    # predictions as the new run built them, when both runs read features of the same width
    for tensor_name, target_value in target_state.items():
        if section_name != "KeypointHead" or tensor_name not in component_state or \
                not re.fullmatch(pattern=f"{KEYPOINT_PREDICTION_INPUT_LAYER}\\.weight", string=tensor_name):
            continue
        source_value = component_state[tensor_name]
        feature_width = target_value.shape[1]
        if source_value.shape[1:] == target_value.shape[1:] and source_value.shape[0] >= feature_width:
            partial_value = target_value.detach().clone().to(device=source_value.device, dtype=source_value.dtype)
            partial_value[:feature_width] = source_value[:feature_width]
            kept_state[tensor_name] = partial_value

    # Tell that the class layers start random, for the classes (keypoint channels) of both runs
    possessive = "Their" if component_name.endswith("heads") else "Its"
    if section_name == "KeypointHead":
        number_source_outputs = count_keypoint_channels(state=component_state)
        number_target_outputs = count_keypoint_channels(state=target_state)
        layer_words, output_words = f"{possessive} Keypoint Channel Layers, Which Keep", "Keypoint Channels"
    else:
        number_source_outputs = count_state_classes(state=component_state, section_name=section_name)
        number_target_outputs = count_state_classes(state=target_state, section_name=section_name)
        layer_words = f"{possessive} Class Layer, Which Keeps" if len(class_layer_names) == 1 else \
            f"{possessive} Class Layers, Which Keep"
        output_words = "Classes"
    print_color(color="green", output=f"{component_name.title()} : Loaded Without {layer_words} The Random "
                                      f"Initialisation Of This Run For {possessive} {number_target_outputs} "
                                      f"{output_words} (The Source Run Predicts {number_source_outputs} "
                                      f"{output_words}).")
    return kept_state


def count_keypoint_channels(state: Dict[str, torch.Tensor]) -> Optional[int]:
    """
    Counts the keypoint channels of a keypoint head from the output channels of its last quarter resolution
    convolution, which predicts a heatmap then a tag per channel, for the message of a keypoint head loaded without its
    keypoint channel layers (keep_target_class_layers).

    Args:
        state (Dict[str, torch.Tensor]): The tensors of the keypoint head, without prefix.

    Returns:
        Optional[int]: The number of keypoint channels, None when the state holds no prediction convolution.
    """
    for tensor_name, tensor_value in state.items():
        if tensor_name.endswith(".weight") and re.fullmatch(pattern=KEYPOINT_CHANNEL_LAYERS[0],
                                                            string=tensor_name[:-len(".weight")]):
            return tensor_value.shape[0] // 2
    return None


def read_checkpoint_model_state(checkpoint_path: Path, device: torch.device,
                                use_averaged_weights: bool) -> Tuple[Dict[str, torch.Tensor], bool]:
    """
    Reads the weights of a model from a checkpoint of a source run, for the transfer of its components: the moving
    average of its weights when the checkpoint holds one and the new run keeps one too
    ('Optimization.exponential_moving_average.activated'), otherwise its trained weights. The averaged model is a copy
    of the model, so both states name their tensors the same way.

    Args:
        checkpoint_path (Path): The checkpoint file of the source run.
        device (torch.device): The device on which the tensors are loaded.
        use_averaged_weights (bool): Whether the new run keeps a moving average of its weights.

    Returns:
        Tuple[Dict[str, torch.Tensor], bool]: The model state, and whether it holds the averaged weights.
    """
    checkpoint = torch.load(f=checkpoint_path, map_location=device, weights_only=True)
    if use_averaged_weights and EXPONENTIAL_MOVING_AVERAGE_STATE_KEY in checkpoint:
        return checkpoint[EXPONENTIAL_MOVING_AVERAGE_STATE_KEY][AVERAGED_MODEL_STATE_KEY], True
    return checkpoint["model_state"], False


def describe_backbone_input(configuration: Dict[str, Any]) -> str:
    """
    Describes the images the backbone of a run sees, for the message of the transfer of a backbone between runs of
    other input sizes: the network input images, or the object crops of the keypoint detection runs saved before the
    bottom-up keypoint detector (their 'Data.crop_settings'), whose backbones still serve as stems.

    Args:
        configuration (Dict[str, Any]): The configuration of the run, whose 'Data' section gives its input size.

    Returns:
        str: The input size, e.g. '256 Pixel Images' or '256x192 Pixel Crops'.
    """
    if "crop_settings" in configuration["Data"]:
        crop_settings = configuration["Data"]["crop_settings"]
        return f"{crop_settings['height']}x{crop_settings['width']} Pixel Crops"
    return f"{configuration['Data']['image_settings']['size']} Pixel Images"


def find_backbone_block_index(tensor_name: str) -> Optional[int]:
    """
    Gives the block of a backbone tensor from its name in the backbone state, e.g. 0 for
    'convolution_blocks.0.layers.0.weight', so that the tensors of the blocks shared by two backbones of different
    depths are told apart from those of the blocks only one of them has.

    Args:
        tensor_name (str): The name of the tensor in the backbone state, without prefix.

    Returns:
        Optional[int]: The index of its block from 0, None for a tensor outside the convolution blocks.
    """
    block_match = re.match(pattern=BACKBONE_BLOCK_TENSOR_PATTERN, string=tensor_name)
    return None if block_match is None else int(block_match.group(1))


def describe_block_range(first_block: int, last_block: int) -> str:
    """
    Names backbone blocks numbered from 1, as the configuration numbers them, in the console messages of a backbone
    deeper than its source backbone, whose blocks after the loaded ones start random.

    Args:
        first_block (int): The first block of the range.
        last_block (int): The last block of the range, at least first_block.

    Returns:
        str: The blocks, e.g. 'Blocks 6 To 7' or 'Block 7'.
    """
    if first_block == last_block:
        return f"Block {first_block}"
    return f"Blocks {first_block} To {last_block}"


def select_backbone_prefix_state(component_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor],
                                 number_shared_blocks: int) -> Dict[str, torch.Tensor]:
    """
    Gives the tensors loaded into a backbone that shares its first blocks with the source backbone (see
    count_shared_backbone_blocks): the tensors of the shared blocks and those outside the blocks come from the source,
    the blocks of a deeper source backbone after them are left out (the stem of the keypoint detector, for instance),
    and the blocks of a deeper receiving backbone after them keep the random initialisation the new run was built
    with, so that the state still fits the receiving backbone exactly.

    Args:
        component_state (Dict[str, torch.Tensor]): The tensors of the source backbone, without prefix.
        target_state (Dict[str, torch.Tensor]): The state dictionary of the backbone of the new run, holding the
            random initialisation of its blocks.
        number_shared_blocks (int): The number of blocks of the shorter backbone, the first blocks of both.

    Returns:
        Dict[str, torch.Tensor]: The tensors of the backbone of the new run.
    """
    # Source tensors of the shared blocks, and those outside the blocks
    prefix_state = {}
    for tensor_name, tensor_value in component_state.items():
        block_index = find_backbone_block_index(tensor_name=tensor_name)
        if block_index is None or block_index < number_shared_blocks:
            prefix_state[tensor_name] = tensor_value

    # Blocks of a deeper receiving backbone after the shared ones, kept as the new run initialised them
    for tensor_name, tensor_value in target_state.items():
        block_index = find_backbone_block_index(tensor_name=tensor_name)
        if block_index is not None and block_index >= number_shared_blocks:
            prefix_state[tensor_name] = tensor_value
    return prefix_state


def count_loaded_backbone_blocks(model_configuration: Dict[str, Any]) -> Optional[int]:
    """
    Counts the blocks of the backbone of a new run loaded from a shorter source backbone, read from the configuration
    saved in the source run folder, so that the trainer freezes, keeps the batch statistics and fine-tunes at the
    learning rate multiplier only these blocks, the blocks after them starting random and learning at the learning rate
    of the run as the other new layers (see select_transferred_part). Called once load_pretrained_component has loaded
    the backbone, which checked both architectures.

    Args:
        model_configuration (Dict[str, Any]): The model configuration of the new run, whose 'Backbone.pretrained'
            names the source run.

    Returns:
        Optional[int]: The number of loaded blocks when the backbone is deeper than its source backbone, None when it
            is not transferred or loads every block (a source backbone of the same depth or deeper).
    """
    source_folder = model_configuration["Backbone"]["pretrained"]["experiment_folder"]
    if source_folder is None:
        return None
    source_configuration = load_configuration(configuration_path=Path(source_folder) / "training_configuration.yaml")
    number_shared_blocks = count_shared_backbone_blocks(
        source_backbone_configuration=source_configuration["Backbone"],
        target_backbone_configuration=model_configuration["Backbone"])

    # Only a receiving backbone with more blocks than the shared ones has random blocks
    if number_shared_blocks is None or number_shared_blocks == len(model_configuration["Backbone"]["convolutions"]):
        return None
    return number_shared_blocks


def select_transferred_part(component: nn.Module, section_name: str, number_loaded_blocks: Optional[int],
                            with_class_layers: bool) -> nn.Module:
    """
    Gives the part of a transferred component that holds the weights of its source run, which the trainer freezes
    ('pretrained.freeze'), keeps the batch statistics of, fixes the normalization parameters of
    ('freeze_normalization_parameters') and fine-tunes at the learning rate multiplier of the transfer: the whole
    component, the first blocks of a backbone deeper than its source backbone (see count_loaded_backbone_blocks),
    whose blocks after them start random and learn at the learning rate of the run as the other new layers, or the
    layers of a component loaded without its class layers ('pretrained.class_layer' or 'class_layers' False, see
    transfers_class_layers), whose class layers start random and learn at the learning rate of the run too, as does the
    transposed convolution of the higher resolution branch of a keypoint head, whose input rows of the predictions
    start random (KEYPOINT_PREDICTION_INPUT_LAYER).

    Args:
        component (nn.Module): The transferred component of the model.
        section_name (str): The configuration section of the component, a key of TRANSFERABLE_COMPONENTS.
        number_loaded_blocks (Optional[int]): The number of blocks of a backbone loaded from a shorter source backbone,
            None for a component loaded whole.
        with_class_layers (bool): Whether the class layers of the component were loaded with it (see
            transfers_class_layers), always True for a component without class layers.

    Returns:
        nn.Module: The component itself, or a module list holding its loaded blocks (the input normalization of the
            backbone holds no tensor) or its modules holding tensors outside its class layers, whose parameters and
            modules are those of the model.
    """
    if number_loaded_blocks is not None:
        return nn.ModuleList(modules=list(component.convolution_blocks)[:number_loaded_blocks])
    if with_class_layers or section_name not in CLASS_LAYER_OPTION_SECTIONS:
        return component

    # Every module holding its own parameters or buffers (layers, normalizations) but the class layers and the partly
    # random transposed convolution of the keypoint head, so that the part holds every loaded tensor once
    return nn.ModuleList(modules=[
        module for module_name, module in component.named_modules()
        if module_name and not find_left_out_layer(section_name=section_name, module_name=module_name) and
        not (section_name == "KeypointHead" and re.fullmatch(pattern=KEYPOINT_PREDICTION_INPUT_LAYER,
                                                             string=module_name) is not None) and
        (next(module.parameters(recurse=False), None) is not None or
         next(module.buffers(recurse=False), None) is not None)])


def warn_heads_behind_another_neck(section_name: str, model_configuration: Dict[str, Any], source_folder: Path,
                                   checkpoint_path: Path) -> None:
    """
    Warns, when the region proposal, detection or mask heads of a run with a neck are loaded from another run, that the
    neck of this run is not the one they were trained behind: a neck starting from random weights, the neck of another
    run or the neck of another checkpoint of their run gives them other features than those they learned to read. The
    heads still load, their weights fitting any neck of the same type and channels (compare_detection_heads), so that
    the neck can also be trained again on purpose; 'Neck.pretrained' naming the checkpoint of the heads gives no
    warning.

    Args:
        section_name (str): "RegionProposal", "Detector" or "MaskHead".
        model_configuration (Dict[str, Any]): The model configuration of the run receiving the heads, with the
            'Neck.pretrained' settings.
        source_folder (Path): The run folder the heads are loaded from.
        checkpoint_path (Path): The checkpoint file the heads are loaded from.
    """
    # Without neck the heads read the backbone blocks, whatever their source
    if model_configuration["Neck"]["type"] == "none":
        return

    # Where the neck of this run comes from, compared with the checkpoint of the heads
    neck_pretrained_configuration = model_configuration["Neck"]["pretrained"]
    neck_folder = neck_pretrained_configuration["experiment_folder"]
    if neck_folder is None:
        neck_description = "Starts From Random Weights"
    elif Path(neck_folder).resolve() != source_folder.resolve():
        neck_description = f"Comes From Another Run ({format_path_link(path=neck_folder)})"
    else:
        neck_checkpoint_path = get_checkpoint_path(experiment_folder=Path(neck_folder),
                                                   iteration=neck_pretrained_configuration["iteration"])
        if neck_checkpoint_path.resolve() == checkpoint_path.resolve():
            return
        neck_description = f"Comes From Another Checkpoint Of Their Run ({format_path_link(path=neck_checkpoint_path)})"
    print_color(color="yellow", output=f"The Pretrained {get_component_name(section_name=section_name).title()} Were "
                                       f"Trained Behind The Neck Of Their Source Run, But The Neck Of This Run "
                                       f"{neck_description}: Set 'Neck.pretrained' To The Same Run And Iteration So "
                                       f"That They Read The Features They Were Trained On.")


def load_pretrained_component(component: nn.Module, section_name: str, model_configuration: Dict[str, Any],
                              device: torch.device, class_names: Optional[List[str]] = None) -> Path:
    """
    Initialises a component of a new run (backbone, temporal module, classification head, segmentation decoder,
    region proposal heads, detection heads, mask heads, high resolution network, keypoint head or neck) with the
    trained component of another run, as set by
    '<section>.pretrained': every task model stores the shared components under the same attributes, so a backbone
    trained for classification can initialise a detector or a segmenter, the temporal module of a sequence classifier
    a sequence labeller or comparator, the neck and heads of an object detector an instance segmenter, and so on. Heads
    trained behind a neck are reported when the neck of the new run does not come from their checkpoint (see
    warn_heads_behind_another_neck). The settings
    defining the component must be identical in both runs (see compare_component_configurations), then its tensors
    must fit exactly (see compare_state_dictionaries). A backbone may come from a run of another input image size,
    the convolutions not depending on it, which is reported since the features were learned at another scale, a
    shorter backbone from the first blocks of a deeper one and a deeper backbone from a shorter one, whose blocks after
    the loaded ones keep their random initialisation (see count_shared_backbone_blocks), which are reported too, a
    temporal module from a causal or non-causal run alike, and region proposal heads from a run of other anchor scales
    or ratios with the same number of anchors per position, which are reported too.
    The component is read from the averaged weights of the checkpoint when it holds a moving average and the new run
    keeps one, from its trained weights otherwise (see read_checkpoint_model_state). The new run may add classes after
    those of the source run or hold them at other indices under the same names: the class layers of the classification
    head, detection heads and mask heads keep the rows of the source classes, moved to the index of their name, the rows
    of the other classes keeping their random initialisation (see prepare_class_layers). A classification, detection,
    mask or keypoint head loaded without its class layers ('pretrained.class_layer' or 'class_layers' False, the
    default) serves a run of any classes, its class layers keeping the random initialisation of the new run (see
    keep_target_class_layers).
    The keypoint head loaded with its keypoint channel layers needs the keypoint channels of its source run, read from
    the record of its data, every heatmap and tag channel having to predict the same keypoint of the same class in both
    runs.

    Args:
        component (nn.Module): The component of the new run, loaded in place.
        section_name (str): The configuration section of the component, a key of TRANSFERABLE_COMPONENTS.
        model_configuration (Dict[str, Any]): The model configuration of the new run, whose section holds the
            'pretrained' settings (the source 'experiment_folder' and checkpoint 'iteration', None for the last one).
        device (torch.device): The device on which the checkpoint tensors are loaded.
        class_names (Optional[List[str]]): The class names of the new run, in class index order, compared with those
            of the source run for the components with class layers, None for a task without them.

    Returns:
        Path: The checkpoint file the component was loaded from.

    Raises:
        ValueError: If the source run has no such component, if its settings differ, listing every difference, if one
            of its classes is not a class of the new run (when its class layers are loaded), if it holds nothing but
            its class layers when they are left out, if the channels of its keypoint head are unknown or predict other
            keypoints than those of the new run, or if its tensors do not fit the component, listing every mismatching
            tensor.
    """
    component_name = get_component_name(section_name=section_name)
    pretrained_configuration = model_configuration[section_name]["pretrained"]
    source_folder = Path(pretrained_configuration["experiment_folder"])

    # Only the configuration of the source run is read, not checked, so that a run of any task can serve
    source_configuration = load_configuration(configuration_path=source_folder / "training_configuration.yaml")
    if section_name not in source_configuration:
        raise ValueError(f"The Source Run {format_path_link(path=source_folder)} Has No '{section_name}' Section: "
                         f"Its Task Has No {component_name.title()} To Initialise This Run.")
    differences = compare_component_configurations(section_name=section_name, source_configuration=source_configuration,
                                                   target_configuration=model_configuration)
    if differences:
        raise ValueError(f"The {component_name.title()} Of The Source Run {format_path_link(path=source_folder)} "
                         f"Cannot Initialise This Run, Their Architectures Differ: " + " ".join(differences))

    # Settings that change what the component learned without changing its weights, reported only
    if section_name == "Backbone":
        source_input = describe_backbone_input(configuration=source_configuration)
        target_input = describe_backbone_input(configuration=model_configuration)
        if source_input != target_input:
            print_color(color="yellow", output=f"The Pretrained Backbone Was Trained On {source_input}, This Run Uses "
                                               f"{target_input}.")
        # The feature map normalization changes the statistics the backbone weights were trained with
        source_normalization = source_configuration["Backbone"]["normalization"]["feature_map_normalization"]
        target_normalization = model_configuration["Backbone"]["normalization"]["feature_map_normalization"]
        if source_normalization != target_normalization:
            print_color(color="yellow", output=f"The Pretrained Backbone Was Trained With {source_normalization!r} "
                                               f"Normalization, This Run Uses {target_normalization!r} Normalization.")
    # A temporal module loaded into a run of the other causality, and region proposal heads of other anchors
    if section_name == "TemporalModel" and model_configuration["TemporalModel"]["type"] in CAUSAL_TEMPORAL_TYPES:
        source_causal = source_configuration["TemporalModel"]["causal"]
        target_causal = model_configuration["TemporalModel"]["causal"]
        if source_causal != target_causal:
            print_color(color="yellow", output=f"The Pretrained Temporal Module Was Trained With 'causal' "
                                               f"{source_causal}, This Run Uses 'causal' {target_causal}.")
    if section_name == "RegionProposal":
        source_anchors = {level_key: level_configuration["anchors"]
                          for level_key, level_configuration in source_configuration["DetectionLevels"].items()}
        target_anchors = {level_key: level_configuration["anchors"]
                          for level_key, level_configuration in model_configuration["DetectionLevels"].items()}
        if source_anchors != target_anchors:
            print_color(color="yellow", output=f"The Pretrained Region Proposal Heads Were Trained With The Anchors "
                                               f"{source_anchors}, This Run Uses The Anchors {target_anchors}.")

    # The heatmap and tag channels of the keypoint head follow the keypoints of the classes, which must be the same
    # keypoints of the same classes in the same order, refused when the source run does not record them, unless the
    # head is loaded without its keypoint channel layers
    with_class_layers = transfers_class_layers(section_name=section_name,
                                               pretrained_configuration=pretrained_configuration)
    if section_name == "KeypointHead" and with_class_layers:
        source_channels = read_source_keypoint_channels(source_folder=source_folder)
        if source_channels is None:
            raise ValueError(f"The Keypoint Head Of The Source Run {format_path_link(path=source_folder)} Cannot "
                             f"Initialise This Run: Its {DATA_RECORD_FILE_NAME} Records No Keypoint Names Of Its "
                             f"Classes (Runs Started Before 2026-11-20), So Its Heatmap And Tag Channels Cannot Be "
                             f"Matched To Those Of This Run.")
        if class_names is None:
            raise ValueError("The Keypoint Head Cannot Be Initialised Without The Class Names Of This Run, Which Name "
                             "Its Heatmap And Tag Channels.")
        target_channels = list_keypoint_channels(keypoint_categories=model_configuration["Data"]["keypoint_categories"],
                                                 class_names=class_names)
        channel_differences = compare_keypoint_channels(source_channels=source_channels,
                                                        target_channels=target_channels)
        if channel_differences:
            raise ValueError(f"The Keypoint Head Of The Source Run {format_path_link(path=source_folder)} Cannot "
                             f"Initialise This Run, Its Heatmap And Tag Channels Predicting Other Keypoints: " +
                             " ".join(channel_differences))

    # Heads trained behind a neck read other features when the neck of this run is not the one of their checkpoint
    checkpoint_path = get_checkpoint_path(experiment_folder=source_folder,
                                          iteration=pretrained_configuration["iteration"])
    if section_name in DETECTION_LEVEL_COMPONENTS:
        warn_heads_behind_another_neck(section_name=section_name, model_configuration=model_configuration,
                                       source_folder=source_folder, checkpoint_path=checkpoint_path)

    # Keep the tensors of the component in the source model, its averaged weights when both runs keep a moving
    # average, without their prefix
    model_state, averaged_weights = read_checkpoint_model_state(
        checkpoint_path=checkpoint_path, device=device,
        use_averaged_weights=model_configuration["Optimization"]["exponential_moving_average"]["activated"])
    state_prefix = f"{TRANSFERABLE_COMPONENTS[section_name]}."
    component_state = {tensor_name[len(state_prefix):]: tensor_value
                       for tensor_name, tensor_value in model_state.items() if tensor_name.startswith(state_prefix)}

    # A backbone sharing its first blocks with the source backbone loads them only: a shorter backbone leaves out the
    # deeper source blocks, a deeper one keeps its blocks after them as the new run initialised them
    target_state = component.state_dict()
    number_shared_blocks = None
    if section_name == "Backbone":
        number_shared_blocks = count_shared_backbone_blocks(
            source_backbone_configuration=source_configuration["Backbone"],
            target_backbone_configuration=model_configuration["Backbone"])
    if number_shared_blocks is not None:
        component_state = select_backbone_prefix_state(component_state=component_state, target_state=target_state,
                                                       number_shared_blocks=number_shared_blocks)

        # Tell which blocks come from the pretrained backbone and which keep the random weights of this run
        number_source_blocks = len(source_configuration["Backbone"]["convolutions"])
        number_target_blocks = len(model_configuration["Backbone"]["convolutions"])
        if number_target_blocks < number_source_blocks:
            transfer_message = (f"The Backbone Of This Run Takes The First {number_target_blocks} Of The "
                                f"{number_source_blocks} Blocks Of The Pretrained Backbone.")
        else:
            random_blocks = describe_block_range(first_block=number_shared_blocks + 1, last_block=number_target_blocks)
            transfer_message = (f"The Backbone Of This Run Starts With The {number_source_blocks} Blocks Of The "
                                f"Pretrained Backbone, Its {random_blocks} Keeping Random Weights And Learning At The "
                                f"Learning Rate Of The Run.")
        print_color(color="yellow", output=transfer_message)

    # Class layers left out of the transfer, keeping the random initialisation of this run whatever its classes, or
    # checked against the classes of this run and moved when it adds classes or reorders the source ones
    if section_name in CLASS_LAYER_OPTION_SECTIONS and not with_class_layers:
        component_state = keep_target_class_layers(component_state=component_state, target_state=target_state,
                                                   section_name=section_name, source_folder=source_folder)
    elif section_name in CLASS_LAYERS:
        component_state = prepare_class_layers(component_state=component_state, target_state=target_state,
                                               section_name=section_name, source_folder=source_folder,
                                               class_names=class_names)

    # Every tensor of the component must then fit exactly
    state_differences = compare_state_dictionaries(source_state=component_state, target_state=target_state)
    if state_differences:
        raise ValueError(f"The {component_name.title()} Tensors Of The Checkpoint "
                         f"{format_path_link(path=checkpoint_path)} Do Not Fit This Run: " +
                         " ".join(state_differences))
    component.load_state_dict(state_dict=component_state, strict=True)

    weights_name = "Averaged Weights" if averaged_weights else "Trained Weights"
    print_color(color="green", add_separators=True,
                output=f"{component_name.title()} Initialised From The {weights_name} Of : "
                       f"{format_path_link(path=checkpoint_path)}")
    return checkpoint_path


def load_segmentation_class_layers(segmenter: nn.Module, model_configuration: Dict[str, Any], device: torch.device,
                                   class_names: Optional[List[str]]) -> Path:
    """
    Initialises the class layers of a semantic segmenter, outside its decoder (SEGMENTATION_CLASS_LAYER_ATTRIBUTES: the
    1x1 convolution of the segmentation head and those of the deep supervision heads), with those of the run its decoder
    comes from, when 'SegmentationDecoder.pretrained.class_layers' is True. Called after load_pretrained_component
    has loaded the decoder, whose checks cover the channels these layers read. The same weights are read as for the
    decoder (averaged or trained), every class of the source run must be a class of the new run, first and in the
    same order or under the same name, the new run adding its other classes (see prepare_class_layers), and the deep
    supervision heads of the levels both runs supervise
    are loaded, those of the levels only the new run supervises keeping their random weights.

    Args:
        segmenter (nn.Module): The semantic segmenter of the new run, whose class layers are loaded in place.
        model_configuration (Dict[str, Any]): The model configuration of the new run, with the
            'SegmentationDecoder.pretrained' settings of the source run.
        device (torch.device): The device on which the checkpoint tensors are loaded.
        class_names (Optional[List[str]]): The class names of the new run, in class index order, background first.

    Returns:
        Path: The checkpoint file the class layers were loaded from.

    Raises:
        ValueError: If a class of the source run is not a class of the new run, or if the tensors of the class layers
            do not fit, listing every mismatching tensor.
    """
    pretrained_configuration = model_configuration["SegmentationDecoder"]["pretrained"]
    source_folder = Path(pretrained_configuration["experiment_folder"])
    checkpoint_path = get_checkpoint_path(experiment_folder=source_folder,
                                          iteration=pretrained_configuration["iteration"])
    model_state, averaged_weights = read_checkpoint_model_state(
        checkpoint_path=checkpoint_path, device=device,
        use_averaged_weights=model_configuration["Optimization"]["exponential_moving_average"]["activated"])

    # Tensors of the class layers in both runs, named from the segmenter
    source_state = {tensor_name: tensor_value for tensor_name, tensor_value in model_state.items()
                    if tensor_name.split(".")[0] in SEGMENTATION_CLASS_LAYER_ATTRIBUTES}
    target_state = {tensor_name: tensor_value for tensor_name, tensor_value in segmenter.state_dict().items()
                    if tensor_name.split(".")[0] in SEGMENTATION_CLASS_LAYER_ATTRIBUTES}

    # Deep supervision heads of the levels supervised by both runs only, the others being listed
    source_levels = {tensor_name.split(".")[1] for tensor_name in source_state
                     if tensor_name.startswith("deep_supervision_heads.")}
    target_levels = {tensor_name.split(".")[1] for tensor_name in target_state
                     if tensor_name.startswith("deep_supervision_heads.")}
    random_levels = sorted(target_levels - source_levels, key=int)
    if random_levels:
        print_color(color="yellow", output=f"The Source Run Has No Deep Supervision Head On The Levels "
                                           f"{random_levels}, Which Keep Their Random Weights.")
    shared_state_names = [tensor_name for tensor_name in target_state
                          if not tensor_name.startswith("deep_supervision_heads.") or
                          tensor_name.split(".")[1] in source_levels]
    source_state = {tensor_name: tensor_value for tensor_name, tensor_value in source_state.items()
                    if tensor_name in shared_state_names}
    target_state = {tensor_name: target_state[tensor_name] for tensor_name in shared_state_names}

    # Classes checked and moved as for the heads, then every tensor must fit exactly
    source_state = prepare_class_layers(component_state=source_state, target_state=target_state,
                                        section_name="SegmentationDecoder", source_folder=source_folder,
                                        class_names=class_names)
    state_differences = compare_state_dictionaries(source_state=source_state, target_state=target_state)
    if state_differences:
        raise ValueError(f"The Segmentation Class Layer Tensors Of The Checkpoint "
                         f"{format_path_link(path=checkpoint_path)} Do Not Fit This Run: " +
                         " ".join(state_differences))

    # Only the class layers are loaded, the rest of the segmenter keeping its weights
    segmenter.load_state_dict(state_dict=source_state, strict=False)
    weights_name = "Averaged Weights" if averaged_weights else "Trained Weights"
    print_color(color="green", add_separators=True,
                output=f"Segmentation Class Layers Initialised From The {weights_name} Of : "
                       f"{format_path_link(path=checkpoint_path)}")
    return checkpoint_path


def freeze_component(component: nn.Module) -> None:
    """
    Keeps a transferred component fixed during the training: its parameters stop requiring gradients, so that only
    the other components learn. The trainer also keeps its batch normalization layers in evaluation mode (see
    freeze_batch_normalization_statistics), so that their statistics stay those of the source run. The trainer gives
    it the transferred part of the component (see select_transferred_part): the loaded blocks of a backbone deeper
    than its source backbone, its next blocks learning.

    Args:
        component (nn.Module): The component (or its loaded blocks) to freeze, modified in place.
    """
    for parameter in component.parameters():
        parameter.requires_grad = False


def freeze_normalization_parameters(component: nn.Module) -> None:
    """
    Keeps the scale and shift of the batch, layer and group normalization layers of a transferred component fixed,
    while its other layers learn: the features keep the calibration of the source run while the convolutions adapt
    to the new task. The batch normalization statistics are not parameters, so they keep following the new data. Only
    the loaded blocks of a backbone deeper than its source backbone keep them (see select_transferred_part).

    Args:
        component (nn.Module): The component (or its loaded blocks) whose normalization parameters are frozen,
            modified in place.
    """
    for module in component.modules():
        if isinstance(module, NORMALIZATION_LAYER_TYPES):
            for parameter in module.parameters():
                parameter.requires_grad = False


def freeze_batch_normalization_statistics(component: nn.Module) -> None:
    """
    Puts the batch normalization layers of a frozen component in evaluation mode, after the trainer has put the whole
    model in training mode, so that their running statistics stay those of the source run. The other layers stay in
    training mode: the gradients of the trainable components before it still flow through a frozen component, and
    the cuDNN backward of the recurrent layers only runs in training mode. The blocks of a backbone deeper than its
    source backbone after the loaded ones are not frozen, so their statistics follow the new data.

    Args:
        component (nn.Module): The frozen component (or the loaded blocks of the backbone), modified in place.
    """
    for module in component.modules():
        if isinstance(module, BATCH_NORMALIZATION_LAYER_TYPES):
            module.eval()


def disable_pretrained_settings(model_configuration: Dict[str, Any]) -> None:
    """
    Turns off the transfer of every component of a configuration, for the tools that load the trained weights of a
    run from its own checkpoint (e.g. the evaluation of a checkpoint): the components and the segmentation class layers
    the run started from are then neither loaded nor frozen, and the source runs need not exist anymore. The class
    layer flags (PRETRAINED_CLASS_LAYER_KEYS) go back to False, the configuration check refusing True without a source
    run.

    Args:
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    for section_name in TRANSFERABLE_COMPONENTS:
        if section_name in model_configuration:
            model_configuration[section_name]["pretrained"].update({"experiment_folder": None, "freeze": False})
    model_configuration["Backbone"]["pretrained"]["freeze_normalization_parameters"] = False
    for section_name, key_name in PRETRAINED_CLASS_LAYER_KEYS.items():
        if section_name in model_configuration:
            model_configuration[section_name]["pretrained"][key_name] = False
