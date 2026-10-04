"""
Transfer of trained components between runs: the architecture checks of the backbone, temporal module and
classification head, the loading of a pretrained component from the checkpoint of another run, and the freezing of
components, normalization parameters and batch statistics. Used by the base trainer and by
scripts/evaluation/evaluate_checkpoint.py.
"""

import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List

from vision_training.configuration.configuration_loader import load_configuration
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.utilities.os_utilities import print_color

# Configuration sections whose component can be initialised by the trained component of another run, with the
# attribute holding the component in every task model, which is also the prefix of its weights in the state
# dictionaries. Their order is the order of the optimizer parameter groups of the components learning at their own
# speed
TRANSFERABLE_COMPONENTS = {"Backbone": "backbone", "TemporalModel": "temporal_module",
                           "ClassificationHead": "classification_head"}

# Backbone settings defining its layers and their parameters, which must be identical in the source run and in the
# run receiving its backbone. The detection block indices only select outputs, and the convolutions do not depend
# on the image size, so both may differ
ARCHITECTURE_BACKBONE_KEYS = ["input_channels", "convolutions", "modules", "normalization", "last_max_pooling"]

# Feature map normalizations holding the same tensors (scale, shift and running statistics), so that a backbone moves
# between batch normalization and sequence normalization through the transfer
BATCH_STATISTICS_NORMALIZATIONS = ["batch", "sequence"]

# Temporal module settings defining its weights, for every type that has weights. The dropout only acts in training
# and the causal flag only changes which frames attend to each other, so neither changes the weights. The number of
# heads of the transformer does not change the shape of its weights, but it splits them differently between the heads
TEMPORAL_WEIGHT_KEYS = {"attention_pooling": ["hidden_size"], "gru": ["hidden_size", "number_layers", "bidirectional"],
                        "lstm": ["hidden_size", "number_layers", "bidirectional"],
                        "transformer": ["hidden_size", "number_layers", "number_heads"],
                        "temporal_convolution": ["hidden_size", "number_layers", "kernel_size"]}

# Temporal modules whose frames may only see the frames before them
CAUSAL_TEMPORAL_TYPES = ["transformer", "temporal_convolution"]

# Normalization layers whose scale and shift can be kept fixed, and batch normalization layers whose statistics a
# frozen component keeps
NORMALIZATION_LAYER_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)
BATCH_NORMALIZATION_LAYER_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def get_component_name(section_name: str) -> str:
    """
    Names the component of a transferable configuration section in the messages of the transfer, e.g. "temporal
    module" for the 'TemporalModel' section.

    Args:
        section_name (str): A key of TRANSFERABLE_COMPONENTS.

    Returns:
        str: The name of the component, in lower case.
    """
    return TRANSFERABLE_COMPONENTS[section_name].replace("_", " ")


def get_weight_normalization(normalization_configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the normalization settings of a backbone as far as its weights are concerned: batch and sequence
    normalization holding the same tensors, both are named "batch", so that a backbone moves between them through the
    transfer.

    Args:
        normalization_configuration (Dict[str, Any]): The 'Backbone.normalization' settings.

    Returns:
        Dict[str, Any]: A copy of the settings, sequence normalization written as batch normalization.
    """
    feature_map_normalization = normalization_configuration["feature_map_normalization"]
    if feature_map_normalization in BATCH_STATISTICS_NORMALIZATIONS:
        feature_map_normalization = "batch"
    return {**normalization_configuration, "feature_map_normalization": feature_map_normalization}


def compare_backbone_architectures(source_backbone_configuration: Dict[str, Any],
                                   target_backbone_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the architecture-defining settings of two backbones, so that a backbone is only
    transferred between runs whose backbones have exactly the same layers. Batch and sequence normalization count as
    the same layers, holding the same tensors.

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

        # Normalization layers compared by their tensors, batch and sequence normalization being the same
        compared_source_value, compared_target_value = source_value, target_value
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
    module reads, and for the transformer the sequence length, which gives the size of its positional embedding.

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

    # Settings defining the weights of the type
    differences = []
    temporal_type = target_temporal_configuration["type"]
    for key_name in TEMPORAL_WEIGHT_KEYS[temporal_type]:
        if source_temporal_configuration[key_name] != target_temporal_configuration[key_name]:
            differences.append(f"'TemporalModel.{key_name}' is {source_temporal_configuration[key_name]!r} in the "
                               f"source run and {target_temporal_configuration[key_name]!r} in this run.")

    # Size of the frame vectors read by the module, the channels of the last backbone block
    source_frame_features = list(source_configuration["Backbone"]["convolutions"].values())[-1][1]
    target_frame_features = list(target_configuration["Backbone"]["convolutions"].values())[-1][1]
    if source_frame_features != target_frame_features:
        differences.append(f"The last backbone block has {source_frame_features} channels in the source run and "
                           f"{target_frame_features} in this run, the size of the frame vectors of the temporal "
                           f"module.")

    # Positional embedding of the transformer, one vector per frame of a window
    if temporal_type == "transformer":
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
    hidden fully connected layers, whether the head uses a dropout, which shifts the numbering of its layers, and for
    a flattened feature map the settings giving the size of its input (the flattening itself, and the image size).
    The other sizes of the input vectors (e.g. the channels of the last backbone block) and the number of classes are
    compared on the weights themselves.

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

    # A dropout layer before every fully connected layer shifts the numbering of the layers of the head
    source_uses_dropout = source_head_configuration["dropout_rate"] > 0
    target_uses_dropout = target_head_configuration["dropout_rate"] > 0
    if source_head_configuration["fully_connected"] and source_uses_dropout != target_uses_dropout:
        differences.append(f"'ClassificationHead.dropout_rate' is {source_head_configuration['dropout_rate']!r} in the "
                           f"source run and {target_head_configuration['dropout_rate']!r} in this run: a head with "
                           f"dropout numbers its layers differently from a head without.")
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


def load_pretrained_component(component: nn.Module, section_name: str, model_configuration: Dict[str, Any],
                              device: torch.device) -> Path:
    """
    Initialises a component of a new run (backbone, temporal module or classification head) with the trained
    component of another run, as set by '<section>.pretrained': every task model stores the shared components under
    the same attributes, so a backbone trained for classification can initialise a detector, the temporal module of a
    sequence classifier a sequence labeller or comparator, and so on. The settings defining the component must be
    identical in both runs (see compare_component_configurations), then its tensors must fit exactly (see
    compare_state_dictionaries). A backbone may come from a run of another input image size, the convolutions not
    depending on it, which is reported since the features were learned at another scale, and a temporal module from a
    causal or non-causal run alike, which is reported too.

    Args:
        component (nn.Module): The component of the new run, loaded in place.
        section_name (str): The configuration section of the component, a key of TRANSFERABLE_COMPONENTS.
        model_configuration (Dict[str, Any]): The model configuration of the new run, whose section holds the
            'pretrained' settings (the source 'experiment_folder' and checkpoint 'iteration', None for the last one).
        device (torch.device): The device on which the checkpoint tensors are loaded.

    Returns:
        Path: The checkpoint file the component was loaded from.

    Raises:
        ValueError: If the source run has no such component, if its settings differ, listing every difference, or if
            its tensors do not fit the component, listing every mismatching tensor.
    """
    component_name = get_component_name(section_name=section_name)
    pretrained_configuration = model_configuration[section_name]["pretrained"]
    source_folder = Path(pretrained_configuration["experiment_folder"])

    # Only the configuration of the source run is read, not checked, so that a run of any task and any age can serve
    source_configuration = load_configuration(configuration_path=source_folder / "training_configuration.yaml")
    if section_name not in source_configuration:
        raise ValueError(f"The Source Run file://{source_folder.absolute()} Has No '{section_name}' Section: Its Task "
                         f"Has No {component_name.title()} To Initialise This Run.")
    differences = compare_component_configurations(section_name=section_name,
                                                   source_configuration=source_configuration,
                                                   target_configuration=model_configuration)
    if differences:
        raise ValueError(f"The {component_name.title()} Of The Source Run file://{source_folder.absolute()} Cannot "
                         f"Initialise This Run, Their Architectures Differ: " + " ".join(differences))

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

    # Keep the tensors of the component in the source model, without their prefix, and require every one of them
    checkpoint_path = get_checkpoint_path(experiment_folder=source_folder,
                                          iteration=pretrained_configuration["iteration"])
    model_state = torch.load(f=checkpoint_path, map_location=device, weights_only=True)["model_state"]
    state_prefix = f"{TRANSFERABLE_COMPONENTS[section_name]}."
    component_state = {tensor_name[len(state_prefix):]: tensor_value
                       for tensor_name, tensor_value in model_state.items() if tensor_name.startswith(state_prefix)}
    state_differences = compare_state_dictionaries(source_state=component_state, target_state=component.state_dict())
    if state_differences:
        raise ValueError(f"The {component_name.title()} Tensors Of The Checkpoint file://{checkpoint_path.absolute()} "
                         f"Do Not Fit This Run: " + " ".join(state_differences))
    component.load_state_dict(state_dict=component_state, strict=True)

    print_color(color="green", add_separators=True,
                output=f"{component_name.title()} Initialised From : file://{checkpoint_path.absolute()}")
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
    run from its own checkpoint (e.g. the evaluation of a checkpoint): the components the run started from are then
    neither loaded nor frozen, and the source runs need not exist anymore.

    Args:
        model_configuration (Dict[str, Any]): The model configuration, modified in place.
    """
    for section_name in TRANSFERABLE_COMPONENTS:
        if section_name in model_configuration:
            model_configuration[section_name]["pretrained"].update({"experiment_folder": None, "freeze": False})
    model_configuration["Backbone"]["pretrained"]["freeze_normalization_parameters"] = False
