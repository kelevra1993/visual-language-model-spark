# Definition of the convolution block
import torch
from torch import nn
from typing import Union, Tuple, Optional, List, Dict, Any

from architecture_modules.inception import InceptionConvolution
from architecture_modules.squeeze_and_excitation import SqueezeAndExcitation


class ChannelLayerNormalizer(nn.Module):
    """
    Applies Layer Normalization over the channel dimension for 4D image tensors.
    
    This class is required because PyTorch's native LayerNorm expects the 
    normalized dimension to be the last dimension. This wrapper permutes 
    the tensor so normalization is applied across the channel dimension correctly.
    """

    def __init__(self, channels: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the ChannelLayerNormalizer.
        
        Args:
            channels (int): The number of channels in the input tensor.
            device (torch.device): The device on which to allocate the parameters.
            dtype (torch.dtype): The desired data type of returned parameters.
        """
        super().__init__()
        self.normalizer = nn.LayerNorm(normalized_shape=channels, device=device, dtype=dtype)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for channel-wise layer normalization.
        
        Args:
            input_tensor (torch.Tensor): The input image tensor of shape (Batch, Channels, Height, Width).
            
        Returns:
            torch.Tensor: The normalized tensor of shape (Batch, Channels, Height, Width).
        """
        # Permute from [Batch, Channels, Height, Width] to [Batch, Height, Width, Channels]
        input_tensor = input_tensor.permute(0, 2, 3, 1)
        input_tensor = self.normalizer(input_tensor)

        # Permute back to [Batch, Channels, Height, Width]
        return input_tensor.permute(0, 3, 1, 2)


class ConvolutionBlock(nn.Module):
    """
    Constructs a block of convolutional layers for feature extraction in the object detection pipeline.

    This block allows for a configurable number of convolutional layers, optionally followed by
    normalization, an activation function, dropout, and a final pooling layer to reduce spatial dimensions.
    It is shared by the Backbone, the Region Proposal heads and the Detection heads, and can optionally
    integrate the following optional architecture modules:
        - Squeeze-and-Excitation channel recalibration after every convolution layer.
        - Residual shortcuts added every two convolution layers.
        - Inception modules replacing the plain convolution of every layer.

    The layers are stored in `self.block` in the exact order of the historical nn.Sequential implementation
    (convolution, normalization, activation, dropout per layer, then pooling), so checkpoints trained with the
    plain configuration keep identical state_dict keys.

    Args:
        input_channels (int): The number of channels in the input feature map.
        output_channels (int): The number of channels produced by the convolution.
        bias (bool): Whether to include a learnable bias term in the convolutional layers.
        number_layers (int): The total number of convolutional layers to stack within this block.
        kernel_size (int): The size of the square convolving kernel.
        stride (int): The stride of the convolution.
        padding (int): The padding added to both sides of the input.
        feature_map_normalization (str): Strategy for normalization ("batch", "layer", or "none").
        activation (bool): Whether to apply a ReLU activation function after each convolution.
        dropout_rate (float): The probability of an element to be zeroed in the dropout layer.
        add_pooling (bool): Whether to apply a max pooling layer at the very end of the block.
        device (torch.device): The device on which to allocate the parameters.
        dtype (torch.dtype): The desired data type of returned parameters.
        squeeze_and_excite (bool): Whether to recalibrate channels with Squeeze-and-Excitation after every layer.
        residual (bool): Whether to add residual shortcuts every two convolution layers.
        inception (bool): Whether to replace every plain convolution with an Inception module.
    """

    def __init__(self, input_channels: int, output_channels: int, bias: bool, number_layers: int,
                 kernel_size: int, stride: int, padding: int, feature_map_normalization: str, activation: bool,
                 dropout_rate: float, add_pooling: bool, device: torch.device, dtype: torch.dtype,
                 squeeze_and_excite: bool, residual: bool, inception: bool) -> None:

        super(ConvolutionBlock, self).__init__()

        # Store the structural parameters for summaries and receptive field computation
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.number_layers = number_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.add_pooling = add_pooling
        self.squeeze_and_excite = squeeze_and_excite
        self.residual = residual
        self.inception = inception

        # Validate the requested normalization strategy
        normalization_strategy = str(feature_map_normalization).lower().strip()
        valid_strategies = ["batch", "layer", "none", "null"]
        if normalization_strategy not in valid_strategies:
            raise ValueError(f"Invalid feature_map_normalization: {feature_map_normalization}."
                             f" Must be one of {valid_strategies}.")

        # Residual additions and inception concatenations require the spatial size to be preserved by every layer
        if (residual or inception) and stride != 1:
            raise ValueError(f"Residual shortcuts and inception modules require a stride of 1, but got {stride}.")

        # When the first layer changes the channel count, the block input cannot be added to a layer output,
        # so the shortcut pairs are shifted by one layer (parity 1) and start from the first layer's output
        self.residual_parity = 1 if input_channels != output_channels else 0

        # Initialize a list to hold the sequence of layers for the block
        layers = []

        # Per layer, keep the indices of its convolution/normalization/activation modules and of its dropout module
        self.layer_module_indices = []
        self.layer_dropout_indices = []

        # Squeeze-and-Excitation modules are kept outside self.block so the plain configuration keys are unchanged
        self.squeeze_and_excitation_layers = nn.ModuleList()

        # Determine the number of input channels for the current layer being constructed
        current_input_channels = input_channels

        # Iteratively build the specified number of convolutional layers
        for layer_index in range(number_layers):
            layer_start_index = len(layers)

            # Append the feature extraction layer: an Inception module or a plain convolution
            if inception:
                layers.append(InceptionConvolution(input_channels=current_input_channels,
                                                   output_channels=output_channels, bias=bias,
                                                   device=device, dtype=dtype))
            else:
                layers.append(nn.Conv2d(in_channels=current_input_channels, out_channels=output_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
                                        device=device, dtype=dtype))

            # Conditionally apply the selected normalization strategy
            if normalization_strategy == "batch":
                layers.append(nn.BatchNorm2d(num_features=output_channels, device=device, dtype=dtype))
            elif normalization_strategy == "layer":
                layers.append(ChannelLayerNormalizer(channels=output_channels, device=device, dtype=dtype))

            # Conditionally apply the activation function to introduce non-linearity to the network
            if activation:
                layers.append(nn.ReLU(inplace=True))

            self.layer_module_indices.append(list(range(layer_start_index, len(layers))))

            # Apply dropout if a rate greater than zero is specified to prevent model overfitting
            if dropout_rate > 0.0:
                self.layer_dropout_indices.append(len(layers))
                layers.append(nn.Dropout2d(p=dropout_rate))
            else:
                self.layer_dropout_indices.append(None)

            # Create the channel recalibration module associated with this layer
            if squeeze_and_excite:
                self.squeeze_and_excitation_layers.append(SqueezeAndExcitation(channels=output_channels,
                                                                               device=device, dtype=dtype))

            # Update the input channels for any subsequent convolutional layers within this block
            current_input_channels = output_channels

        # Conditionally apply max pooling at the end of the block to reduce the spatial dimensions of the feature map
        self.pooling_index = None
        if add_pooling:
            self.pooling_index = len(layers)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Group all constructed layers into a module list, iterated layer by layer in the forward pass
        self.block = nn.ModuleList(modules=layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the sequence of convolutional layers to extract features.

        Each layer applies convolution, normalization and activation, then the optional residual addition,
        Squeeze-and-Excitation recalibration and dropout. The optional pooling layer is applied last.

        Args:
            input_tensor (torch.Tensor): The input feature map to be processed by the network of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: The resulting feature map after passing through the convolutional block of shape [Batch, Out_Channels, Out_Height, Out_Width].
        """
        current_tensor = input_tensor
        shortcut_tensor = None

        # Iterate through each convolution layer of the block
        for layer_index, module_indices in enumerate(self.layer_module_indices):
            # Store the layer input as the shortcut that will be added two layers later
            if self.residual and layer_index % 2 == self.residual_parity:
                shortcut_tensor = current_tensor

            # Apply the convolution, normalization and activation modules of this layer
            for module_index in module_indices:
                current_tensor = self.block[module_index](current_tensor)

            # Add the residual shortcut after the activation, closing a pair of layers
            if self.residual and layer_index % 2 == (1 - self.residual_parity) and layer_index != 0:
                current_tensor = current_tensor + shortcut_tensor

            # Recalibrate the channels of the layer output
            if self.squeeze_and_excite:
                current_tensor = self.squeeze_and_excitation_layers[layer_index](input_tensor=current_tensor)

            # Apply the dropout of this layer, if any
            if self.layer_dropout_indices[layer_index] is not None:
                current_tensor = self.block[self.layer_dropout_indices[layer_index]](current_tensor)

        # Reduce the spatial dimensions with the final pooling layer, if any
        if self.pooling_index is not None:
            current_tensor = self.block[self.pooling_index](current_tensor)

        return current_tensor

    def get_receptive_field_layers(self) -> List[Dict[str, Any]]:
        """
        Lists the spatial operations of the block used by the Backbone to compute its receptive field.

        Inception layers are described by their largest spatial path (1x1 then 5x5, equivalent to a 5x5 kernel).
        Squeeze-and-Excitation and residual shortcuts do not change the convolutional receptive field.

        Returns:
            List[Dict[str, Any]]: One dictionary per spatial operation with keys 'name', 'kernel_size',
                                  'stride' and 'padding'.
        """
        receptive_field_layers = []

        # Describe every convolution layer of the block
        for layer_index in range(self.number_layers):
            if self.inception:
                receptive_field_layers.append({"name": f"Inception {layer_index + 1}", "kernel_size": 5,
                                               "stride": 1, "padding": 2})
            else:
                receptive_field_layers.append({"name": f"Convolution {layer_index + 1}",
                                               "kernel_size": self.kernel_size, "stride": self.stride,
                                               "padding": self.padding})

        # Describe the final pooling layer
        if self.add_pooling:
            receptive_field_layers.append({"name": "Max Pooling", "kernel_size": 2, "stride": 2, "padding": 0})

        return receptive_field_layers
