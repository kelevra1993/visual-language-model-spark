# Definition of the convolution block
import torch
from torch import nn
from typing import Union, Tuple, Optional


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
    batch normalization, layer normalization, an activation function, dropout, and a final pooling layer 
    to reduce spatial dimensions.
    
    Args:
        input_channels (int): The number of channels in the input feature map.
        output_channels (int): The number of channels produced by the convolution.
        bias (bool): Whether to include a learnable bias term in the convolutional layers.
        number_layers (int): The total number of convolutional layers to stack within this block.
        kernel_size (Union[int, Tuple[int, int]]): The size of the convolving kernel.
        stride (Union[int, Tuple[int, int]]): The stride of the convolution.
        padding (Union[int, Tuple[int, int]]): The padding added to both sides of the input.
        batch_normalization (bool): Whether to apply batch normalization after each convolution.
        layer_normalization (bool): Whether to apply layer normalization after each convolution.
        activation (bool): Whether to apply a ReLU activation function after each convolution.
        dropout_rate (float): The probability of an element to be zeroed in the dropout layer.
        add_pooling (bool): Whether to apply a max pooling layer at the very end of the block.
        device (torch.device): The device on which to allocate the parameters.
        dtype (torch.dtype): The desired data type of returned parameters.
    """

    def __init__(self, input_channels: int, output_channels: int, bias: bool, number_layers: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 batch_normalization: bool, layer_normalization: bool, activation: bool,
                 dropout_rate: float, add_pooling: bool,
                 device: torch.device, dtype: torch.dtype) -> None:

        super(ConvolutionBlock, self).__init__()

        # Initialize a list to hold the sequence of layers for the block
        layers = []

        # Determine the number of input channels for the current layer being constructed
        current_input_channels = input_channels

        # Iteratively build the specified number of convolutional layers
        for layer_index in range(number_layers):
            # Append the convolutional layer to extract spatial features from the input
            layers.append(nn.Conv2d(in_channels=current_input_channels, out_channels=output_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
                                    device=device, dtype=dtype))

            # Conditionally apply batch normalization to stabilize training dynamics
            if batch_normalization:
                layers.append(nn.BatchNorm2d(num_features=output_channels, device=device, dtype=dtype))

            # Conditionally apply layer normalization using the custom permuting normalizer
            if layer_normalization:
                layers.append(ChannelLayerNormalizer(channels=output_channels, device=device, dtype=dtype))

            # Conditionally apply the activation function to introduce non-linearity to the network
            if activation:
                layers.append(nn.ReLU(inplace=True))

            # Apply dropout if a rate greater than zero is specified to prevent model overfitting
            if dropout_rate > 0.0:
                layers.append(nn.Dropout2d(p=dropout_rate))

            # Update the input channels for any subsequent convolutional layers within this block
            current_input_channels = output_channels

        # Conditionally apply max pooling at the end of the block to reduce the spatial dimensions of the feature map
        if add_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Group all constructed layers into a sequential module for execution
        self.block = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the sequence of convolutional layers to extract features.
        
        Args:
            input_tensor (torch.Tensor): The input feature map to be processed by the network.
            
        Returns:
            torch.Tensor: The resulting feature map after passing through the convolutional block.
        """

        # Process the input tensor through the sequential block and return the output feature map
        return self.block(input_tensor)
