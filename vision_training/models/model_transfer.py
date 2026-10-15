"""
Transfer of trained components between runs: the architecture checks of the backbone, temporal module,
classification head, segmentation decoder, region proposal heads, detection heads and mask heads, the loading of a
pretrained component from the checkpoint of another run (its averaged weights when it kept a moving average and the new
run keeps one), the class layers of a run adding classes after those of its source run (the source classes keeping
their trained rows, the new classes starting quiet), the class layers of the semantic segmentation loaded with its
decoder, and the freezing of components, normalization parameters and batch statistics. Used by the base trainer and by
scripts/evaluation/evaluate_checkpoint.py.
"""

import re
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Optional, Tuple

from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.data.data_record import CLASS_NAMES_RECORD_KEY, DATA_RECORD_FILE_NAME
from vision_training.models.instance_segmentation.mask_head import LEGACY_UPSAMPLING_PARAMETER_NAMES
from vision_training.trainers.checkpointing import EXPONENTIAL_MOVING_AVERAGE_STATE_KEY, get_checkpoint_path
from vision_training.trainers.weight_averaging import AVERAGED_MODEL_STATE_KEY
from vision_training.utilities.os_utilities import format_path_link, print_color, read_json

# Configuration sections whose component can be initialised by the trained component of another run, with the
# attribute holding the component in every task model, which is also the prefix of its weights in the state
# dictionaries. Their order is the order of the optimizer parameter groups of the components learning at their own
# speed, the heads of the detection tasks, then the mask heads of the instance segmentation, coming last so that the
# runs written before them keep their groups
TRANSFERABLE_COMPONENTS = {"Backbone": "backbone", "TemporalModel": "temporal_module",
                           "ClassificationHead": "classification_head", "SegmentationDecoder": "segmentation_decoder",
                           "RegionProposal": "region_proposer_dictionary", "Detector": "detector_heads_dictionary",
                           "MaskHead": "mask_heads_dictionary"}

# Names of the components whose attribute does not read as a name in the messages of the transfer
COMPONENT_NAMES = {"RegionProposal": "region proposal heads", "Detector": "detection heads",
                   "MaskHead": "mask heads"}

# Attributes of the semantic segmenter holding its class layers, outside the decoder so that a decoder serves runs of
# other classes, loaded with the decoder when 'SegmentationDecoder.pretrained.load_class_layers' is True: the 1x1
# convolution of the segmentation head and the 1x1 convolutions of the deep supervision heads
SEGMENTATION_CLASS_LAYER_ATTRIBUTES = ["segmentation_head", "deep_supervision_heads"]

# Backbone settings defining its layers and their parameters, which must be identical in the source run and in the
# run receiving its backbone. The detection block indices only select outputs, and the convolutions do not depend
# on the image size, so both may differ
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

# Settings of the region proposal and detection heads defining their weights, compared level by level. The RoIAlign
# pool size of the detection heads gives the size of their first fully connected layer. The number of anchors per
# position of the region proposal heads, giving the size of their output layers, is compared apart, the anchor scales
# and ratios themselves not being weights. The mask resolution and the upsampling of the mask heads give their
# upsampling layers, and their pool size and convolutions are in their head configurations
DETECTION_HEAD_WEIGHT_KEYS = {"RegionProposal": ["head_configurations", "normalization"],
                              "Detector": ["head_configurations", "normalization", "modules"],
                              "MaskHead": ["mask_resolution", "upsampling", "head_configurations", "normalization",
                                           "modules"]}

# Normalization layers whose scale and shift can be kept fixed, and batch normalization layers whose statistics a
# frozen component keeps
NORMALIZATION_LAYER_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)
BATCH_NORMALIZATION_LAYER_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

# Bias of the new classes of a run adding classes after those of its source run, below the mean bias of the source
# classes, so that the new classes start quiet: on an image the source model has no opinion on, every new class weighs
# about e^-3, 5 % of an average source class, in the softmax, so that the predictions of the source classes are kept at
# the first iteration. The training raises the bias of a new class as soon as its images come
NEW_CLASS_BIAS_OFFSET = 3.0

# Sections whose class index 0 is the background class, kept first in both runs and left out of the means initialising
# the new classes: most proposals and pixels are background, so its row stands apart from the other classes
SECTIONS_WITH_BACKGROUND_CLASS = ["Detector", "MaskHead", "SegmentationDecoder"]

# Class layers of the components, one row (or output channel) per class or 4 rows per class for the box regressor, as
# [module name pattern, initialisation of the new classes, rows per class], the pattern matching the name of the module
# in the component state (in the state of the segmentation head and deep supervision heads for the decoder). The new
# classes are initialised with:
# - "quiet": zero weights and a bias NEW_CLASS_BIAS_OFFSET below the mean bias of the source classes, for the logits
#   competing with the source classes in a softmax (or starting at a low probability in the sigmoid of the multi-label
#   classification)
# - "zero": zero weights and biases, for the box regression of the new classes, keeping their proposal boxes unchanged
# - "average": the mean of the rows of the source classes, for the mask predictor, whose classes do not compete (every
#   object only trains the mask of its own class), a new class starting from a generic object mask it then specialises
CLASS_LAYERS = {"ClassificationHead": [[r"classifier", "quiet", 1]],
                "Detector": [[r"\w+\.classifier", "quiet", 1], [r"\w+\.bounding_box_regressor", "zero", 4]],
                "MaskHead": [[r"\w+\.mask_predictor", "average", 1]],
                "SegmentationDecoder": [[r"segmentation_head\.\d+", "quiet", 1],
                                        [r"deep_supervision_heads\.\w+", "quiet", 1]]}


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


def compare_backbone_architectures(source_backbone_configuration: Dict[str, Any],
                                   target_backbone_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the architecture-defining settings of two backbones, so that a backbone is only
    transferred between runs whose backbones have exactly the same layers. Batch and sequence normalization count as
    the same layers, holding the same tensors, and the batch normalization momentum may differ. A source run written
    before a module setting of BACKBONE_MODULE_DEFAULTS counts as having its default value.

    Args:
        source_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the source run.
        target_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the run receiving the backbone.

    Returns:
        List[str]: The description of every difference, empty when the architectures match.
    """
    differences = []
    for key_name in ARCHITECTURE_BACKBONE_KEYS:
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

    # Settings defining the weights of the type, a source run written before one of them having to be migrated
    differences = []
    differing_keys = []
    temporal_type = target_temporal_configuration["type"]
    for key_name in TEMPORAL_WEIGHT_KEYS[temporal_type]:
        if key_name not in source_temporal_configuration:
            differing_keys.append(key_name)
            differences.append(f"'TemporalModel.{key_name}' is missing from the source run, written before the "
                               f"setting existed: migrate it with "
                               f"scripts/configuration_processing/migrate_configuration.py.")
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
    # False before the layer normalization existed, having to be migrated
    if "fully_connected_normalization" not in source_head_configuration:
        differences.append("'ClassificationHead.fully_connected_normalization' is missing from the source run, "
                           "written before the setting existed: migrate it with "
                           "scripts/configuration_processing/migrate_configuration.py.")
    elif isinstance(source_head_configuration["fully_connected_normalization"], bool):
        differences.append(f"'ClassificationHead.fully_connected_normalization' is "
                           f"{source_head_configuration['fully_connected_normalization']!r} in the source run, written "
                           f"before the layer normalization existed: migrate it with "
                           f"scripts/configuration_processing/migrate_configuration.py.")
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
    detection levels, then the settings of DETECTION_HEAD_WEIGHT_KEYS (the normalization without its batch
    normalization momentum, which changes no tensor) and, for the region proposal heads, the number of anchors per
    position of every level.
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
    source_levels = source_configuration["Backbone"]["detection_convolution_block_indices"]
    target_levels = target_configuration["Backbone"]["detection_convolution_block_indices"]
    if source_levels != target_levels:
        return [f"'Backbone.detection_convolution_block_indices' is {source_levels!r} in the source run and "
                f"{target_levels!r} in this run, the levels of the {get_component_name(section_name=section_name)}."]

    # Settings of the layers of the heads
    differences = []
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
            scales_and_ratios = configuration["Anchors"]["scales_and_ratios"]
            anchor_counts.append({str(level): len(scales_and_ratios[str(level)]["scales"]) *
                                  len(scales_and_ratios[str(level)]["aspect_ratios"]) for level in target_levels})
        if anchor_counts[0] != anchor_counts[1]:
            differences.append(f"The levels have {anchor_counts[0]} anchors per position in the source run and "
                               f"{anchor_counts[1]} in this run, the outputs of the region proposal heads.")
    return differences


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


def find_class_layer(section_name: str, module_name: str) -> Optional[Tuple[str, int]]:
    """
    Tells whether a module of a component is one of its class layers (see CLASS_LAYERS), whose rows follow the classes
    of the run and grow when a run adds classes after those of its source run.

    Args:
        section_name (str): A key of CLASS_LAYERS.
        module_name (str): The name of the module in the component state, e.g. 'classifier' or '5.mask_predictor'.

    Returns:
        Optional[Tuple[str, int]]: The initialisation of the new classes and the number of rows of every class, None
            for a module that is not a class layer.
    """
    for module_pattern, initialisation, rows_per_class in CLASS_LAYERS[section_name]:
        if re.fullmatch(pattern=module_pattern, string=module_name):
            return initialisation, rows_per_class
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
        class_layer = find_class_layer(section_name=section_name, module_name=tensor_name[:-len(".weight")])
        if class_layer is not None:
            return tensor_value.shape[0] // class_layer[1]
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


def extend_class_layer(weight: torch.Tensor, bias: torch.Tensor, number_target_classes: int, rows_per_class: int,
                       initialisation: str, has_background_class: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Grows a class layer of a source run to the classes of a new run that adds classes after them: the rows (output
    channels of a convolution) of the source classes are kept, and the rows of the new classes are appended, set by the
    initialisation of the layer (see CLASS_LAYERS) from the source classes other than the background.

    Args:
        weight (torch.Tensor): The [source classes x rows per class, ...] weight of the source layer.
        bias (torch.Tensor): The [source classes x rows per class] bias of the source layer.
        number_target_classes (int): The number of classes of the new run, more than those of the source layer.
        rows_per_class (int): The number of rows of every class (4 for the box regressor, 1 otherwise).
        initialisation (str): "quiet", "zero" or "average".
        has_background_class (bool): Whether class 0 is the background, left out of the means of the new classes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [target classes x rows per class, ...] weight and the bias of the layer
            of the new run.
    """
    # Rows grouped by class, [classes, rows per class, ...]
    number_source_classes = weight.shape[0] // rows_per_class
    number_new_classes = number_target_classes - number_source_classes
    weight_rows = weight.reshape(number_source_classes, rows_per_class, *weight.shape[1:])
    bias_rows = bias.reshape(number_source_classes, rows_per_class)

    # Source classes the new ones are initialised from: the classes other than the background, in float32 for the means
    first_reference_class = 1 if has_background_class and number_source_classes > 1 else 0
    reference_weight_rows = weight_rows[first_reference_class:].float()
    reference_bias_rows = bias_rows[first_reference_class:].float()

    # Mean of the source classes, or zero weights with a zero bias or a bias below the mean source bias
    if initialisation == "average":
        new_weight_rows = reference_weight_rows.mean(dim=0, keepdim=True).expand(number_new_classes,
                                                                                 *weight_rows.shape[1:])
        new_bias_rows = reference_bias_rows.mean(dim=0, keepdim=True).expand(number_new_classes, rows_per_class)
    else:
        new_weight_rows = torch.zeros(size=(number_new_classes, *weight_rows.shape[1:]), device=weight.device)
        new_bias = float(reference_bias_rows.mean()) - NEW_CLASS_BIAS_OFFSET if initialisation == "quiet" else 0.0
        new_bias_rows = torch.full(size=(number_new_classes, rows_per_class), fill_value=new_bias, device=bias.device)

    # Source rows first, then the new classes, in the precision of the source layer
    extended_weight = torch.cat(tensors=[weight_rows, new_weight_rows.to(dtype=weight.dtype)], dim=0)
    extended_bias = torch.cat(tensors=[bias_rows, new_bias_rows.to(dtype=bias.dtype)], dim=0)
    return extended_weight.reshape(number_target_classes * rows_per_class, *weight.shape[1:]), extended_bias.reshape(-1)


def extend_class_layers(component_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor],
                        section_name: str) -> Dict[str, torch.Tensor]:
    """
    Grows every class layer of a source component whose layer in the new run has more classes and reads inputs of the
    same size (see extend_class_layer), the other tensors being kept as they are, so that compare_state_dictionaries
    still reports every other mismatch.

    Args:
        component_state (Dict[str, torch.Tensor]): The tensors of the source component, without prefix.
        target_state (Dict[str, torch.Tensor]): The state dictionary of the component of the new run.
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
        class_layer = find_class_layer(section_name=section_name, module_name=module_name)
        if class_layer is None or tensor_name not in target_state or bias_name not in component_state:
            continue

        # Only a layer reading inputs of the same size with more classes in the new run grows
        target_weight = target_state[tensor_name]
        if target_weight.shape[1:] != source_weight.shape[1:] or target_weight.shape[0] <= source_weight.shape[0]:
            continue
        initialisation, rows_per_class = class_layer
        extended_state[tensor_name], extended_state[bias_name] = extend_class_layer(
            weight=source_weight, bias=component_state[bias_name],
            number_target_classes=target_weight.shape[0] // rows_per_class, rows_per_class=rows_per_class,
            initialisation=initialisation, has_background_class=section_name in SECTIONS_WITH_BACKGROUND_CLASS)
    return extended_state


def prepare_class_layers(component_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor],
                         section_name: str, source_folder: Path,
                         class_names: Optional[List[str]]) -> Dict[str, torch.Tensor]:
    """
    Checks the classes of a source component against those of the new run (see compare_class_names) and grows its
    class layers when the new run adds classes after them (see extend_class_layers), for the components with class
    layers: the classification head, the detection heads, the mask heads and the class layers of the semantic
    segmentation. A source run without class names prints that its classes are taken as the first classes of the new
    run.

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
                                      f"Them.")
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


def rename_legacy_mask_head_tensors(component_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Renames the tensors of the mask heads of a source run written before the mask geometry could be configured, whose
    single transposed convolution is '<level>.upsampling', to the first upsampling layer of the current heads, as
    MaskHead._load_from_state_dict does when such a run resumes, so that their tensors are compared under the names of
    the current heads.

    Args:
        component_state (Dict[str, torch.Tensor]): The tensors of the source mask heads, without prefix.

    Returns:
        Dict[str, torch.Tensor]: The same tensors, the legacy ones renamed.
    """
    renamed_state = {}
    for tensor_name, tensor_value in component_state.items():
        level_name, _, parameter_path = tensor_name.partition(".")
        if parameter_path in [f"upsampling.{parameter_name}" for parameter_name in LEGACY_UPSAMPLING_PARAMETER_NAMES]:
            tensor_name = f"{level_name}.upsampling_layers.0.{parameter_path.split('.')[-1]}"
        renamed_state[tensor_name] = tensor_value
    return renamed_state


def load_pretrained_component(component: nn.Module, section_name: str, model_configuration: Dict[str, Any],
                              device: torch.device, class_names: Optional[List[str]] = None) -> Path:
    """
    Initialises a component of a new run (backbone, temporal module, classification head, segmentation decoder,
    region proposal heads, detection heads or mask heads) with the trained component of another run, as set by
    '<section>.pretrained': every task model stores the shared components under the same attributes, so a backbone
    trained for classification can initialise a detector or a segmenter, the temporal module of a sequence classifier
    a sequence labeller or comparator, the heads of an object detector an instance segmenter, and so on. The settings
    defining the component must be identical in both runs (see compare_component_configurations), then its tensors
    must fit exactly (see compare_state_dictionaries). A backbone may come from a run of another input image size,
    the convolutions not depending on it, which is reported since the features were learned at another scale, a
    temporal module from a causal or non-causal run alike, and region proposal heads from a run of other anchor scales
    or ratios with the same number of anchors per position, which are reported too.
    The component is read from the averaged weights of the checkpoint when it holds a moving average and the new run
    keeps one, from its trained weights otherwise (see read_checkpoint_model_state). The new run may add classes after
    those of the source run: the class layers of the classification head, detection heads and mask heads keep the rows
    of the source classes and get rows for the new ones (see prepare_class_layers).

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
            classes are not the first classes of the new run, or if its tensors do not fit the component, listing every
            mismatching tensor.
    """
    component_name = get_component_name(section_name=section_name)
    pretrained_configuration = model_configuration[section_name]["pretrained"]
    source_folder = Path(pretrained_configuration["experiment_folder"])

    # Only the configuration of the source run is read, not checked, so that a run of any task and any age can serve
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
        source_image_size = source_configuration["Data"]["image_settings"]["size"]
        target_image_size = model_configuration["Data"]["image_settings"]["size"]
        if source_image_size != target_image_size:
            print_color(color="yellow", output=f"The Pretrained Backbone Was Trained On {source_image_size} Pixel "
                                               f"Images, This Run Uses {target_image_size} Pixel Images.")
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
        source_anchors = source_configuration["Anchors"]["scales_and_ratios"]
        target_anchors = model_configuration["Anchors"]["scales_and_ratios"]
        if source_anchors != target_anchors:
            print_color(color="yellow", output=f"The Pretrained Region Proposal Heads Were Trained With The Anchors "
                                               f"{source_anchors}, This Run Uses The Anchors {target_anchors}.")

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
    if section_name == "MaskHead":
        component_state = rename_legacy_mask_head_tensors(component_state=component_state)

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
