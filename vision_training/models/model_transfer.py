"""
Transfer of trained components between runs: the architecture checks of the backbone, temporal module,
classification head, segmentation decoder, region proposal heads, detection heads, mask heads, high resolution network
and keypoint head, the loading of a pretrained component from the checkpoint of another run (its averaged weights when
it kept a moving average and the new run keeps one), the first blocks of a deeper backbone initialising a shorter one
(the stem of the keypoint detector from an ImageNet backbone), the class layers of a run adding classes after those
of its source run (the source classes keeping their trained rows, the new classes the random rows of the new run), the
class layers of the semantic segmentation loaded with its decoder, and the freezing of components, normalization
parameters and batch statistics. Used by the base trainer and by scripts/evaluation/evaluate_checkpoint.py.
"""

import re
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Optional, Tuple

from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.data.data_record import CLASS_NAMES_RECORD_KEY, DATA_RECORD_FILE_NAME
from vision_training.trainers.checkpointing import EXPONENTIAL_MOVING_AVERAGE_STATE_KEY, get_checkpoint_path
from vision_training.trainers.weight_averaging import AVERAGED_MODEL_STATE_KEY
from vision_training.utilities.os_utilities import format_path_link, print_color, read_json

# Configuration sections whose component can be initialised by the trained component of another run, with the
# attribute holding the component in every task model, which is also the prefix of its weights in the state
# dictionaries. Their order is the order of the optimizer parameter groups of the components learning at their own
# speed, the heads of the detection tasks, then the mask heads of the instance segmentation and the high resolution
# network and heatmap head of the keypoint detection, coming last so that the runs written before them keep their groups
TRANSFERABLE_COMPONENTS = {"Backbone": "backbone", "TemporalModel": "temporal_module",
                           "ClassificationHead": "classification_head", "SegmentationDecoder": "segmentation_decoder",
                           "RegionProposal": "region_proposer_dictionary", "Detector": "detector_heads_dictionary",
                           "MaskHead": "mask_heads_dictionary", "HRNet": "high_resolution_network",
                           "KeypointHead": "keypoint_head"}

# Names of the components whose attribute does not read as a name in the messages of the transfer
COMPONENT_NAMES = {"RegionProposal": "region proposal heads", "Detector": "detection heads",
                   "MaskHead": "mask heads"}

# Attributes of the semantic segmenter holding its class layers, outside the decoder so that a decoder serves runs of
# other classes, loaded with the decoder when 'SegmentationDecoder.pretrained.load_class_layers' is True: the 1x1
# convolution of the segmentation head and the 1x1 convolutions of the deep supervision heads
SEGMENTATION_CLASS_LAYER_ATTRIBUTES = ["segmentation_head", "deep_supervision_heads"]

# Backbone settings defining its layers and their parameters, which must be identical in the source run and in the
# run receiving its backbone. The detection levels only select the blocks read by the heads, and the convolutions do
# not depend on the image size, so both may differ
ARCHITECTURE_BACKBONE_KEYS = ["input_channels", "convolutions", "modules", "normalization", "last_max_pooling"]

# Backbone module settings added after the first runs, with the value of the runs written before them: a source run
# without the pre-activation placement of the shortcuts used the original one
BACKBONE_MODULE_DEFAULTS = {"residual_nets_v2": False}

# Feature map normalizations holding the same tensors (scale, shift and running statistics), so that a backbone moves
# between batch normalization and sequence normalization through the transfer
BATCH_STATISTICS_NORMALIZATIONS = ["batch", "sequence"]

# Normalization settings that change how the batch statistics follow the training batches but no tensor, so that a
# component moves between runs of different momentums, a source run written before the setting existed included
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

# Component of every detection level ('DetectionLevels.<level>.<component>') holding the layers of the region
# proposal, detection or mask head of that level, compared level by level. The RoIAlign pool size of the detection
# heads gives the size of their first fully connected layer
DETECTION_LEVEL_COMPONENTS = {"RegionProposal": "region_proposal", "Detector": "detector", "MaskHead": "mask_head"}

# Settings of the region proposal, detection and mask heads shared by every level and defining their weights. The
# number of anchors per position of the region proposal heads, giving the size of their output layers, is compared
# apart, the anchor scales and ratios themselves not being weights. The mask resolution and the upsampling of the
# mask heads give their upsampling layers
DETECTION_HEAD_WEIGHT_KEYS = {"RegionProposal": ["normalization"], "Detector": ["normalization", "modules"],
                              "MaskHead": ["mask_resolution", "upsampling", "normalization", "modules"]}

# Normalization layers whose scale and shift can be kept fixed, and batch normalization layers whose statistics a
# frozen component keeps
NORMALIZATION_LAYER_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)
BATCH_NORMALIZATION_LAYER_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

# Class layers of the components, one row (or output channel) per class or 4 rows per class for the box regressor, as
# [module name pattern, rows per class], the pattern matching the name of the module in the component state (in the
# state of the segmentation head and deep supervision heads for the decoder). When a run adds classes after those of
# its source run, the rows of the source classes are copied and those of the new classes keep the random
# initialisation the layer of the new run was built with
CLASS_LAYERS = {"ClassificationHead": [[r"classifier", 1]],
                "Detector": [[r"\w+\.classifier", 1], [r"\w+\.bounding_box_regressor", 4]],
                "MaskHead": [[r"\w+\.mask_predictor", 1]],
                "SegmentationDecoder": [[r"segmentation_head\.\d+", 1], [r"deep_supervision_heads\.\w+", 1]]}


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


def is_backbone_prefix(source_backbone_configuration: Dict[str, Any],
                       target_backbone_configuration: Dict[str, Any]) -> bool:
    """
    Tells whether the backbone of a run is made of the first blocks of the backbone of a source run, the same layers
    with fewer blocks, so that the first blocks of the source backbone initialise it: the stem of the keypoint detector
    taken from a backbone trained on ImageNet, for instance. The pooling after the last block has no weights, so it may
    differ.

    Args:
        source_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the source run.
        target_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the run receiving the backbone.

    Returns:
        bool: True when the target blocks are the first blocks of the source, in the same order and fewer of them.
    """
    source_blocks = list(source_backbone_configuration["convolutions"].items())
    target_blocks = list(target_backbone_configuration["convolutions"].items())
    return len(target_blocks) < len(source_blocks) and source_blocks[:len(target_blocks)] == target_blocks


def compare_backbone_architectures(source_backbone_configuration: Dict[str, Any],
                                   target_backbone_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the architecture-defining settings of two backbones, so that a backbone is only
    transferred between runs whose backbones have exactly the same layers, or whose backbone is made of the first
    blocks of the source backbone (see is_backbone_prefix). Batch and sequence normalization count as the same layers,
    holding the same tensors, and the batch normalization momentum may differ. A source run written before a module
    setting of BACKBONE_MODULE_DEFAULTS counts as having its default value.

    Args:
        source_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the source run.
        target_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the run receiving the backbone.

    Returns:
        List[str]: The description of every difference, empty when the architectures match.
    """
    # The blocks of a shorter backbone made of the first source blocks are compared by is_backbone_prefix, the pooling
    # after its last block having no weights
    compared_keys = ARCHITECTURE_BACKBONE_KEYS
    if is_backbone_prefix(source_backbone_configuration=source_backbone_configuration,
                          target_backbone_configuration=target_backbone_configuration):
        compared_keys = [key_name for key_name in ARCHITECTURE_BACKBONE_KEYS
                         if key_name not in ["convolutions", "last_max_pooling"]]
    differences = []
    for key_name in compared_keys:
        source_value = source_backbone_configuration[key_name] if key_name in source_backbone_configuration else None
        target_value = target_backbone_configuration[key_name]

        # Normalization layers compared by their tensors, batch and sequence normalization being the same, and the
        # modules of a source run written before a module setting completed with its default value
        compared_source_value, compared_target_value = source_value, target_value
        if key_name == "modules" and source_value is not None:
            compared_source_value = {**BACKBONE_MODULE_DEFAULTS, **source_value}
        if key_name == "normalization" and source_value is not None:
            compared_source_value = get_weight_normalization(normalization_configuration=source_value)
            compared_target_value = get_weight_normalization(normalization_configuration=target_value)
        if compared_source_value != compared_target_value:
            differences.append(f"'Backbone.{key_name}' is {source_value!r} in the source run and {target_value!r} in "
                               f"this run.")
    return differences


def compare_temporal_modules(source_configuration: Dict[str, Any], target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the weights of the temporal modules of two runs, so that a
    temporal module is only transferred to a module of the same type and sizes: the type, the settings of that type
    (TEMPORAL_WEIGHT_KEYS), the channels of the last backbone block, which give the size of the frame vectors the
    module reads, and for a transformer with a positional embedding the sequence length, which gives the size of the
    embedding.

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

    # Size of the frame vectors read by the module, the channels of the last backbone block
    source_frame_features = list(source_configuration["Backbone"]["convolutions"].values())[-1][1]
    target_frame_features = list(target_configuration["Backbone"]["convolutions"].values())[-1][1]
    if source_frame_features != target_frame_features:
        differences.append(f"The last backbone block has {source_frame_features} channels in the source run and "
                           f"{target_frame_features} in this run, the size of the frame vectors of the temporal "
                           f"module.")

    # Positional embedding of both transformers, one vector per frame of a window, absent from an order-blind one
    if temporal_type == "transformer" and target_temporal_configuration["positional_embedding"] and \
            "positional_embedding" not in differing_keys:
        source_sequence_length = source_configuration["Data"]["sequence_settings"]["sequence_length"]
        target_sequence_length = target_configuration["Data"]["sequence_settings"]["sequence_length"]
        if source_sequence_length != target_sequence_length:
            differences.append(f"'Data.sequence_settings.sequence_length' is {source_sequence_length} in the source "
                               f"run and {target_sequence_length} in this run, the size of the positional embedding "
                               f"of the transformer.")
    return differences


def compare_classification_heads(source_configuration: Dict[str, Any],
                                 target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the layers of the classification heads of two runs: the
    hidden fully connected layers, their normalization and, for a flattened feature map, the settings giving the
    size of its input (the flattening itself, and the image size). The dropout rate and the batch normalization
    momentum are not compared: the heads always hold their dropout layers, so the rate does not change the names of
    their weights, and the momentum changes no tensor.
    The other sizes of the input vectors (e.g. the channels of the last backbone block) are compared on the weights
    themselves, and the classes by load_pretrained_component, the run receiving the head being allowed to add classes
    after those of the source run.

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

    # A flattened feature map gives one input per position and channel, a pooled one an input per channel only, and
    # the positions of a flattened map follow the image size
    source_pooling = source_head_configuration["global_pooling"]
    target_pooling = target_head_configuration["global_pooling"]
    if "flatten" in [source_pooling, target_pooling] and source_pooling != target_pooling:
        differences.append(f"'ClassificationHead.global_pooling' is {source_pooling!r} in the source run and "
                           f"{target_pooling!r} in this run: a flattened feature map and a pooled one give input "
                           f"vectors of different sizes.")
    elif source_pooling == "flatten":
        source_image_size = source_configuration["Data"]["image_settings"]["size"]
        target_image_size = target_configuration["Data"]["image_settings"]["size"]
        if source_image_size != target_image_size:
            differences.append(f"'Data.image_settings.size' is {source_image_size!r} in the source run and "
                               f"{target_image_size!r} in this run: the flattened feature map, the input of the head, "
                               f"has a size following the image size.")
    return differences


def compare_segmentation_decoders(source_configuration: Dict[str, Any],
                                  target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the weights of the segmentation decoders of two runs: the
    upsampling, the convolutions of its levels (and therefore its levels), its modules and its normalization, and the
    channels of the backbone blocks it reads, through its skip connections and its deepest level. The segmentation
    head is not part of the decoder, so that the runs may segment other classes; it is loaded with the decoder by
    load_segmentation_class_layers when 'SegmentationDecoder.pretrained.load_class_layers' is True.

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
    detection levels (the keys of 'DetectionLevels'), the head of every level (DETECTION_LEVEL_COMPONENTS), the
    settings of DETECTION_HEAD_WEIGHT_KEYS (the normalization without its batch normalization momentum, which
    changes no tensor) and, for the region proposal heads, the number of anchors per position of every level.
    The channels of the backbone blocks they read are compared on the weights themselves, and the classes of the
    detection and mask heads (classifier, per-class box regressor, per-class mask predictor) by
    load_pretrained_component, the run receiving the heads being allowed to add classes after those of the source run.

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

    # Layers of the head of every level
    differences = []
    level_component = DETECTION_LEVEL_COMPONENTS[section_name]
    for level_key in target_levels:
        source_level_value = source_configuration["DetectionLevels"][level_key][level_component]
        target_level_value = target_configuration["DetectionLevels"][level_key][level_component]
        if source_level_value != target_level_value:
            differences.append(f"'DetectionLevels.{level_key}.{level_component}' is {source_level_value!r} in the "
                               f"source run and {target_level_value!r} in this run.")

    # Settings of the layers of the heads shared by every level
    source_section_configuration = source_configuration[section_name]
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


def compare_high_resolution_networks(source_configuration: Dict[str, Any],
                                     target_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the settings defining the weights of the high resolution networks of two keypoint
    detection runs: the width of its highest resolution branch, its stages, the convolutions of its branches, their
    modules and their normalization (without the batch normalization momentum, which changes no tensor). The width of
    the stem feature map it reads is compared on the weights themselves.

    Args:
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the network.

    Returns:
        List[str]: The description of every difference, empty when the weights of the networks match.
    """
    differences = []
    for key_name in HIGH_RESOLUTION_NETWORK_WEIGHT_KEYS:
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
    Lists the differences between the settings defining the weights of the heatmap heads of two keypoint detection
    runs: the width of the high resolution network they read. Their heatmap channels follow the keypoints of the
    classes, which load_pretrained_component checks on the class names of the runs and on the weights themselves.

    Args:
        source_configuration (Dict[str, Any]): The whole configuration of the source run, as saved in its folder.
        target_configuration (Dict[str, Any]): The model configuration of the run receiving the head.

    Returns:
        List[str]: The description of every difference, empty when the weights of the heads match.
    """
    source_width = source_configuration["HRNet"]["base_width"]
    target_width = target_configuration["HRNet"]["base_width"]
    if source_width != target_width:
        return [f"'HRNet.base_width' is {source_width!r} in the source run and {target_width!r} in this run, the "
                f"channels read by the keypoint head."]
    return []


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
    if section_name == "Backbone":
        return compare_backbone_architectures(source_backbone_configuration=source_configuration["Backbone"],
                                              target_backbone_configuration=target_configuration["Backbone"])
    if section_name == "TemporalModel":
        return compare_temporal_modules(source_configuration=source_configuration,
                                        target_configuration=target_configuration)
    if section_name == "SegmentationDecoder":
        return compare_segmentation_decoders(source_configuration=source_configuration,
                                             target_configuration=target_configuration)
    if section_name in DETECTION_HEAD_WEIGHT_KEYS:
        return compare_detection_heads(section_name=section_name, source_configuration=source_configuration,
                                       target_configuration=target_configuration)
    if section_name == "HRNet":
        return compare_high_resolution_networks(source_configuration=source_configuration,
                                                target_configuration=target_configuration)
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
    folder), so that a run adding classes after those of its source run is checked to keep them first and in the same
    order. The runs written before the class names were recorded have none.

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
    if CLASS_NAMES_RECORD_KEY not in data_record:
        return None
    return data_record[CLASS_NAMES_RECORD_KEY]


def find_class_layer(section_name: str, module_name: str) -> Optional[int]:
    """
    Tells whether a module of a component is one of its class layers (see CLASS_LAYERS), whose rows follow the classes
    of the run and grow when a run adds classes after those of its source run.

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
                        number_source_classes: int, number_target_classes: int) -> List[str]:
    """
    Lists what prevents the class layers of a source run from initialising those of a new run: the new run may only
    keep the classes of the source run, first and in the same order (background first for the detection and
    segmentation tasks), and add classes after them. The class names are compared when both runs have them, the
    names of a source run written before they were recorded being unknown.

    Args:
        source_class_names (Optional[List[str]]): The class names of the source run, None when it has none.
        target_class_names (Optional[List[str]]): The class names of the new run, None for a task without them.
        number_source_classes (int): The number of classes of the class layers of the source checkpoint.
        number_target_classes (int): The number of classes of the class layers of the new run.

    Returns:
        List[str]: The description of every difference, empty when the source classes are the first classes of the new
            run.
    """
    if number_target_classes < number_source_classes:
        return [f"The source run predicts {number_source_classes} classes and this run {number_target_classes}: a "
                f"run can only add classes after those of its source run."]
    if source_class_names is None or target_class_names is None:
        return []

    # The record of the data of the source run follows its last start, which may have changed its classes
    if len(source_class_names) != number_source_classes:
        return [f"The data record of the source run names {len(source_class_names)} classes and its checkpoint "
                f"predicts {number_source_classes} classes."]

    # Every source class keeps its index in the new run
    renamed_classes = [f"{class_index} is {source_name!r} in the source run and {target_name!r} in this run"
                       for class_index, (source_name, target_name) in
                       enumerate(zip(source_class_names, target_class_names)) if source_name != target_name]
    if renamed_classes:
        return ["The classes of the source run must be the first classes of this run, in the same order: class " +
                ", class ".join(renamed_classes) + "."]
    return []


def extend_class_layer(weight: torch.Tensor, bias: torch.Tensor, target_weight: torch.Tensor,
                       target_bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Grows a class layer of a source run to the classes of a new run that adds classes after them: the rows (output
    channels of a convolution) of the source classes are copied, and the rows of the new classes keep the random
    initialisation the layer of the new run received when it was built, as in a run trained from scratch. The rows of
    a class follow each other (4 for the box regressor), so the rows of the source classes are the first rows of the
    layer of the new run.

    Args:
        weight (torch.Tensor): The [source classes x rows per class, ...] weight of the source layer.
        bias (torch.Tensor): The [source classes x rows per class] bias of the source layer.
        target_weight (torch.Tensor): The [target classes x rows per class, ...] randomly initialised weight of the
            layer of the new run, which has more classes than the source layer.
        target_bias (torch.Tensor): The [target classes x rows per class] randomly initialised bias of that layer.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [target classes x rows per class, ...] weight and the bias of the layer
            of the new run, on the device and in the precision of the source layer.
    """
    # Random initialisation of the new run kept for the new classes, its first rows replaced by the source classes
    extended_weight = target_weight.detach().clone().to(device=weight.device, dtype=weight.dtype)
    extended_bias = target_bias.detach().clone().to(device=bias.device, dtype=bias.dtype)
    extended_weight[:weight.shape[0]] = weight
    extended_bias[:bias.shape[0]] = bias
    return extended_weight, extended_bias


def extend_class_layers(component_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor],
                        section_name: str) -> Dict[str, torch.Tensor]:
    """
    Grows every class layer of a source component whose layer in the new run has more classes and reads inputs of the
    same size (see extend_class_layer), the other tensors being kept as they are, so that compare_state_dictionaries
    still reports every other mismatch.

    Args:
        component_state (Dict[str, torch.Tensor]): The tensors of the source component, without prefix.
        target_state (Dict[str, torch.Tensor]): The state dictionary of the component of the new run, holding the
            random initialisation of its layers.
        section_name (str): A key of CLASS_LAYERS.

    Returns:
        Dict[str, torch.Tensor]: The tensors of the source component with its class layers grown.
    """
    extended_state = dict(component_state)
    for tensor_name, source_weight in component_state.items():
        if not tensor_name.endswith(".weight"):
            continue
        module_name = tensor_name[:-len(".weight")]
        bias_name = f"{module_name}.bias"
        if find_class_layer(section_name=section_name, module_name=module_name) is None or \
                tensor_name not in target_state or bias_name not in component_state:
            continue

        # Only a layer reading inputs of the same size with more classes in the new run grows
        target_weight = target_state[tensor_name]
        if target_weight.shape[1:] != source_weight.shape[1:] or target_weight.shape[0] <= source_weight.shape[0]:
            continue
        extended_state[tensor_name], extended_state[bias_name] = extend_class_layer(
            weight=source_weight, bias=component_state[bias_name], target_weight=target_weight,
            target_bias=target_state[bias_name])
    return extended_state


def prepare_class_layers(component_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor],
                         section_name: str, source_folder: Path,
                         class_names: Optional[List[str]]) -> Dict[str, torch.Tensor]:
    """
    Checks the classes of a source component against those of the new run (see compare_class_names) and grows its
    class layers when the new run adds classes after them (see extend_class_layers), the new classes keeping the random
    initialisation of the new run, for the components with class layers: the classification head, the detection
    heads, the mask heads and the class layers of the semantic segmentation. A source run without class names prints
    that its classes are taken as the first classes of the new run.

    Args:
        component_state (Dict[str, torch.Tensor]): The tensors of the source component, without prefix.
        target_state (Dict[str, torch.Tensor]): The state dictionary of the component of the new run.
        section_name (str): A key of CLASS_LAYERS.
        source_folder (Path): The run folder of the source run.
        class_names (Optional[List[str]]): The class names of the new run, in class index order, None for a task
            without them.

    Returns:
        Dict[str, torch.Tensor]: The tensors of the source component, its class layers grown to the classes of the new
            run.

    Raises:
        ValueError: If the classes of the source run are not the first classes of the new run.
    """
    number_source_classes = count_state_classes(state=component_state, section_name=section_name)
    number_target_classes = count_state_classes(state=target_state, section_name=section_name)
    if number_source_classes is None or number_target_classes is None:
        return component_state

    # The source classes must keep their indices in the new run
    component_name = get_component_name(section_name=section_name)
    source_class_names = read_source_class_names(source_folder=source_folder)
    class_differences = compare_class_names(source_class_names=source_class_names, target_class_names=class_names,
                                            number_source_classes=number_source_classes,
                                            number_target_classes=number_target_classes)
    if class_differences:
        raise ValueError(f"The {component_name.title()} Of The Source Run {format_path_link(path=source_folder)} "
                         f"Cannot Initialise This Run, Their Classes Differ: " + " ".join(class_differences))
    if number_target_classes == number_source_classes:
        return component_state

    # New classes appended after those of the source run
    if source_class_names is None:
        print_color(color="yellow", output=f"The Source Run Has No Class Names In Its {DATA_RECORD_FILE_NAME} (Written "
                                           f"Before They Were Recorded): Its {number_source_classes} Classes Are Taken "
                                           f"As The First {number_source_classes} Classes Of This Run.")
    print_color(color="green", output=f"{component_name.title()} : {number_source_classes} Classes Of The Source Run "
                                      f"Kept, {number_target_classes - number_source_classes} New Classes Added After "
                                      "Them, Randomly Initialised.")
    return extend_class_layers(component_state=component_state, target_state=target_state, section_name=section_name)


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
    other input sizes: the object crops of the keypoint detection, the network input images of the other tasks.

    Args:
        configuration (Dict[str, Any]): The configuration of the run, whose 'Data' section gives its input size.

    Returns:
        str: The input size, e.g. '256 Pixel Images' or '256x192 Pixel Crops'.
    """
    if "crop_settings" in configuration["Data"]:
        crop_settings = configuration["Data"]["crop_settings"]
        return f"{crop_settings['height']}x{crop_settings['width']} Pixel Crops"
    return f"{configuration['Data']['image_settings']['size']} Pixel Images"


def select_backbone_prefix_state(component_state: Dict[str, torch.Tensor],
                                 number_target_blocks: int) -> Dict[str, torch.Tensor]:
    """
    Keeps the tensors of the first blocks of a source backbone, and its input normalization, for a backbone made of
    its first blocks (see is_backbone_prefix), the stem of the keypoint detector for instance.

    Args:
        component_state (Dict[str, torch.Tensor]): The tensors of the source backbone, without prefix.
        number_target_blocks (int): The number of blocks of the backbone receiving the tensors.

    Returns:
        Dict[str, torch.Tensor]: The tensors of the blocks below number_target_blocks and those outside the blocks.
    """
    prefix_state = {}
    for tensor_name, tensor_value in component_state.items():
        block_match = re.match(pattern=r"convolution_blocks\.(\d+)\.", string=tensor_name)
        if block_match is None or int(block_match.group(1)) < number_target_blocks:
            prefix_state[tensor_name] = tensor_value
    return prefix_state


def load_pretrained_component(component: nn.Module, section_name: str, model_configuration: Dict[str, Any],
                              device: torch.device, class_names: Optional[List[str]] = None) -> Path:
    """
    Initialises a component of a new run (backbone, temporal module, classification head, segmentation decoder,
    region proposal heads, detection heads, mask heads, high resolution network or keypoint head) with the trained
    component of another run, as set by
    '<section>.pretrained': every task model stores the shared components under the same attributes, so a backbone
    trained for classification can initialise a detector or a segmenter, the temporal module of a sequence classifier
    a sequence labeller or comparator, the heads of an object detector an instance segmenter, and so on. The settings
    defining the component must be identical in both runs (see compare_component_configurations), then its tensors
    must fit exactly (see compare_state_dictionaries). A backbone may come from a run of another input image size,
    the convolutions not depending on it, which is reported since the features were learned at another scale, and a
    shorter backbone from the first blocks of a deeper one (see is_backbone_prefix), which is reported too, a
    temporal module from a causal or non-causal run alike, and region proposal heads from a run of other anchor scales
    or ratios with the same number of anchors per position, which are reported too.
    The component is read from the averaged weights of the checkpoint when it holds a moving average and the new run
    keeps one, from its trained weights otherwise (see read_checkpoint_model_state). The new run may add classes after
    those of the source run: the class layers of the classification head, detection heads and mask heads keep the rows
    of the source classes, the rows of the new ones keeping their random initialisation (see prepare_class_layers).
    The keypoint head needs the classes of its source run, its heatmap channels following their keypoints.

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
        ValueError: If the source run has no such component, if its settings differ, listing every difference, if its
            classes are not the first classes of the new run (the classes of the new run for the keypoint head), or if
            its tensors do not fit the component, listing every
            mismatching tensor.
    """
    component_name = get_component_name(section_name=section_name)
    pretrained_configuration = model_configuration[section_name]["pretrained"]
    source_folder = Path(pretrained_configuration["experiment_folder"])

    # Only the configuration of the source run is read, not checked, so that a run of any task can serve
    source_configuration = load_configuration(configuration_path=source_folder / "training_configuration.yaml")
    if section_name not in source_configuration:
        raise ValueError(f"The Source Run {format_path_link(path=source_folder)} Has No '{section_name}' Section: "
                         f"Its Task Has No {component_name.title()} To Initialise This Run.")
    differences = compare_component_configurations(section_name=section_name,
                                                   source_configuration=source_configuration,
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
        source_normalization = source_configuration["Backbone"]["normalization"]["feature_map_normalization"]
        target_normalization = model_configuration["Backbone"]["normalization"]["feature_map_normalization"]
        if source_normalization != target_normalization:
            print_color(color="yellow", output=f"The Pretrained Backbone Was Trained With {source_normalization!r} "
                                               f"Normalization, This Run Uses {target_normalization!r} Normalization.")
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

    # The heatmap channels of the keypoint head follow the keypoints of the classes, which must be the same
    if section_name == "KeypointHead":
        source_class_names = read_source_class_names(source_folder=source_folder)
        if source_class_names is not None and class_names is not None and source_class_names != class_names:
            raise ValueError(f"The Keypoint Head Of The Source Run {format_path_link(path=source_folder)} Cannot "
                             f"Initialise This Run, Its Heatmap Channels Following The Keypoints Of Other Classes: "
                             f"{source_class_names} In The Source Run And {class_names} In This Run.")

    # Keep the tensors of the component in the source model, its averaged weights when both runs keep a moving
    # average, without their prefix
    checkpoint_path = get_checkpoint_path(experiment_folder=source_folder,
                                          iteration=pretrained_configuration["iteration"])
    model_state, averaged_weights = read_checkpoint_model_state(
        checkpoint_path=checkpoint_path, device=device,
        use_averaged_weights=model_configuration["Optimization"]["exponential_moving_average"]["activated"])
    state_prefix = f"{TRANSFERABLE_COMPONENTS[section_name]}."
    component_state = {tensor_name[len(state_prefix):]: tensor_value
                       for tensor_name, tensor_value in model_state.items() if tensor_name.startswith(state_prefix)}

    # A shorter backbone keeps the tensors of the first blocks of the source backbone only
    if section_name == "Backbone" and is_backbone_prefix(
            source_backbone_configuration=source_configuration["Backbone"],
            target_backbone_configuration=model_configuration["Backbone"]):
        number_target_blocks = len(model_configuration["Backbone"]["convolutions"])
        component_state = select_backbone_prefix_state(component_state=component_state,
                                                       number_target_blocks=number_target_blocks)
        print_color(color="yellow", output=f"The Backbone Of This Run Takes The First {number_target_blocks} Of The "
                                           f"{len(source_configuration['Backbone']['convolutions'])} Blocks Of The "
                                           f"Pretrained Backbone.")

    # Class layers checked against the classes of this run, and grown when it adds classes after the source ones
    target_state = component.state_dict()
    if section_name in CLASS_LAYERS:
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
    comes from, when 'SegmentationDecoder.pretrained.load_class_layers' is True. Called after load_pretrained_component
    has loaded the decoder, whose checks cover the channels these layers read. The same weights are read as for the
    decoder (averaged or trained), the classes of the source run must be the first classes of the new run, which may
    add classes after them (see prepare_class_layers), and the deep supervision heads of the levels both runs supervise
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
        ValueError: If the classes of the source run are not the first classes of the new run, or if the tensors of
            the class layers do not fit, listing every mismatching tensor.
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

    # Classes checked and grown as for the heads, then every tensor must fit exactly
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
    freeze_batch_normalization_statistics), so that their statistics stay those of the source run.

    Args:
        component (nn.Module): The component to freeze, modified in place.
    """
    for parameter in component.parameters():
        parameter.requires_grad = False


def freeze_normalization_parameters(component: nn.Module) -> None:
    """
    Keeps the scale and shift of the batch, layer and group normalization layers of a transferred component fixed,
    while its other layers learn: the features keep the calibration of the source run while the convolutions adapt
    to the new task. The batch normalization statistics are not parameters, so they keep following the new data.

    Args:
        component (nn.Module): The component whose normalization parameters are frozen, modified in place.
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
    the cuDNN backward of the recurrent layers only runs in training mode.

    Args:
        component (nn.Module): The frozen component, modified in place.
    """
    for module in component.modules():
        if isinstance(module, BATCH_NORMALIZATION_LAYER_TYPES):
            module.eval()


def disable_pretrained_settings(model_configuration: Dict[str, Any]) -> None:
    """
    Turns off the transfer of every component of a configuration, for the tools that load the trained weights of a
    run from its own checkpoint (e.g. the evaluation of a checkpoint): the components and the segmentation class layers
    the run started from are then neither loaded nor frozen, and the source runs need not exist anymore.

    Args:
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    for section_name in TRANSFERABLE_COMPONENTS:
        if section_name in model_configuration:
            model_configuration[section_name]["pretrained"].update({"experiment_folder": None, "freeze": False})
    model_configuration["Backbone"]["pretrained"]["freeze_normalization_parameters"] = False
    if "SegmentationDecoder" in model_configuration:
        model_configuration["SegmentationDecoder"]["pretrained"]["load_class_layers"] = False
