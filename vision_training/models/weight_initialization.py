"""
Starting weights of the task models, set by 'Optimization.weight_initialization': "default" keeps the PyTorch
initialisation of every layer, "he" draws the layers followed by a ReLU from the He normal distribution ("Delving Deep
into Rectifiers", He et al., 2015, https://arxiv.org/abs/1502.01852), designed so that the activations of a ReLU network
keep their scale from the first layer to the last. Applied by the base trainer to the model it builds, before the
components transferred from other runs are loaded and before a resume loads its checkpoint, which both replace it.
"""

from torch import nn

from vision_training.models.components.convolution_block import ConvolutionBlock
from vision_training.models.components.fully_connected_block import FullyConnectedBlock
from vision_training.models.components.squeeze_and_excitation import SqueezeAndExcitation

# Starting weights of 'Optimization.weight_initialization'
WEIGHT_INITIALIZATIONS = ["default", "he"]


def initialise_he_layer(layer: nn.Module) -> None:
    """
    Draws the weights of a convolution or linear layer followed by a ReLU from the He normal distribution, of standard
    deviation sqrt(2 / fan_in): a ReLU halves the second moment of its input, and the factor 2 compensates for it, so
    that the layer keeps the scale of the activations. Its bias starts at 0.

    Args:
        layer (nn.Module): The nn.Conv2d or nn.Linear layer, initialised in place.
    """
    nn.init.kaiming_normal_(tensor=layer.weight, mode="fan_in", nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(tensor=layer.bias)


def initialise_weights(model: nn.Module, weight_initialization: str) -> int:
    """
    Sets the starting weights of a task model built by a trainer. With "he", the layers followed by a ReLU get the He
    normal initialisation (see initialise_he_layer): every convolution of the convolution blocks (backbone, region
    proposal and detection heads, mask heads, segmentation decoder), the inception convolutions included, every linear
    layer of the fully connected blocks of the heads, and the reduce layer of Squeeze-and-Excitation. The layers not
    followed by a ReLU keep the PyTorch initialisation, as the final layer of the torchvision ResNets: the output layers
    (class logits, box regressors, mask and segmentation predictors), the expand layer of Squeeze-and-Excitation
    (followed by a sigmoid), the transposed convolutions and the temporal modules. The normalization layers keep their
    scale of 1 and offset of 0. "default" leaves the model unchanged.

    Args:
        model (nn.Module): The task model, initialised in place.
        weight_initialization (str): "default" or "he", one of WEIGHT_INITIALIZATIONS.

    Returns:
        int: The number of layers given the He initialisation, 0 for "default".

    Raises:
        ValueError: If the initialisation is not one of WEIGHT_INITIALIZATIONS.
    """
    if weight_initialization not in WEIGHT_INITIALIZATIONS:
        raise ValueError(f"Unknown weight initialisation {weight_initialization!r}, expected one of "
                         f"{WEIGHT_INITIALIZATIONS}.")
    if weight_initialization == "default":
        return 0

    # Layers followed by a ReLU in the building blocks of the task models
    he_layers = []
    for module in model.modules():
        # Every convolution of a convolution block is followed by a normalization and a ReLU (the next layer or the
        # end of the block with the pre-activation placement), the inception convolutions included
        if isinstance(module, ConvolutionBlock):
            he_layers += [layer for layer in module.block.modules() if isinstance(layer, nn.Conv2d)]
        # Every linear layer of the hidden layers of a head is followed by a ReLU, after its optional normalization
        elif isinstance(module, FullyConnectedBlock):
            he_layers += [layer for layer in module if isinstance(layer, nn.Linear)]
        # The reduce layer of the channel recalibration is followed by a ReLU, its expand layer by a sigmoid
        elif isinstance(module, SqueezeAndExcitation):
            he_layers.append(module.first_adaptive_recalibrator)

    # Draw the weights of every selected layer
    for layer in he_layers:
        initialise_he_layer(layer=layer)
    return len(he_layers)
