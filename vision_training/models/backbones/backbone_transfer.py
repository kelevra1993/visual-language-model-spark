import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.configuration.configuration_loader import load_configuration
from vision_training.models.backbones.backbone import Backbone
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.utilities.os_utilities import print_color

# Backbone settings defining its layers and their parameters, which must be identical in the source run and in the
# run receiving its backbone. The detection block indices only select outputs, and the convolutions do not depend
# on the image size, so both may differ
ARCHITECTURE_BACKBONE_KEYS = ["input_channels", "convolutions", "modules", "normalization", "last_max_pooling"]

# Prefix of the backbone parameters in the state dictionary of every task model
BACKBONE_STATE_PREFIX = "backbone."


def compare_backbone_architectures(source_backbone_configuration: Dict[str, Any],
                                   target_backbone_configuration: Dict[str, Any]) -> List[str]:
    """
    Lists the differences between the architecture-defining settings of two backbones, so that a backbone is only
    transferred between runs whose backbones have exactly the same layers.

    Args:
        source_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the source run.
        target_backbone_configuration (Dict[str, Any]): The 'Backbone' section of the run receiving the backbone.

    Returns:
        List[str]: The description of every difference, empty when the architectures match.
    """
    differences = []
    for key_name in ARCHITECTURE_BACKBONE_KEYS:
        source_value = source_backbone_configuration[key_name] if key_name in source_backbone_configuration else None
        if source_value != target_backbone_configuration[key_name]:
            differences.append(f"'Backbone.{key_name}' is {source_value!r} in the source run and "
                               f"{target_backbone_configuration[key_name]!r} in this run.")
    return differences


def load_pretrained_backbone(backbone: Backbone, backbone_configuration: Dict[str, Any], input_image_size: int,
                             device: torch.device) -> Path:
    """
    Initialises the backbone of a new run with the trained backbone of another run, of any task, as set by
    'Backbone.pretrained': every task model stores the shared Backbone class under 'backbone', so a backbone trained
    for classification can initialise a detector, a detection backbone a classifier, and so on. The source backbone
    must have exactly the same architecture; a different input image size is allowed, the convolutions not depending
    on it, but is reported since the features were learned at another scale.

    Args:
        backbone (Backbone): The backbone of the new run, loaded in place.
        backbone_configuration (Dict[str, Any]): The 'Backbone' section of the new run, with its 'pretrained' settings
            (the source 'experiment_folder' and checkpoint 'iteration', None for the last one).
        input_image_size (int): The input image size of the new run.
        device (torch.device): The device on which the checkpoint tensors are loaded.

    Returns:
        Path: The checkpoint file the backbone was loaded from.

    Raises:
        ValueError: If the backbone of the source run has another architecture, listing every difference.
    """
    pretrained_configuration = backbone_configuration["pretrained"]
    source_folder = Path(pretrained_configuration["experiment_folder"])

    # Only the backbone section of the source configuration is read, so that a run of any task and any age can serve
    source_configuration = load_configuration(configuration_path=source_folder / "training_configuration.yaml")
    differences = compare_backbone_architectures(source_backbone_configuration=source_configuration["Backbone"],
                                                 target_backbone_configuration=backbone_configuration)
    if differences:
        raise ValueError(f"The backbone of the source run file://{source_folder.absolute()} cannot initialise this "
                         f"run, their architectures differ: " + " ".join(differences))
    source_image_size = source_configuration["Data"]["image_settings"]["size"]
    if source_image_size != input_image_size:
        print_color(color="yellow", output=f"The pretrained backbone was trained on {source_image_size} pixel images, "
                                           f"this run uses {input_image_size} pixel images.")

    # Keep the backbone parameters of the source model, without their prefix, and require every one of them
    checkpoint_path = get_checkpoint_path(experiment_folder=source_folder,
                                          iteration=pretrained_configuration["iteration"])
    model_state = torch.load(f=checkpoint_path, map_location=device, weights_only=True)["model_state"]
    backbone_state = {parameter_name[len(BACKBONE_STATE_PREFIX):]: parameter_value
                      for parameter_name, parameter_value in model_state.items()
                      if parameter_name.startswith(BACKBONE_STATE_PREFIX)}
    backbone.load_state_dict(state_dict=backbone_state, strict=True)

    print_color(color="green", output=f"Backbone Initialised From : file://{checkpoint_path.absolute()}",
                add_separators=True)
    return checkpoint_path


def freeze_backbone(backbone: Backbone) -> None:
    """
    Keeps a transferred backbone fixed during the training: its parameters stop requiring gradients, so that only the
    heads of the task learn. The trainer also keeps it in evaluation mode, so that its batch normalization statistics
    stay those of the source run.

    Args:
        backbone (Backbone): The backbone to freeze, modified in place.
    """
    for parameter in backbone.parameters():
        parameter.requires_grad = False
