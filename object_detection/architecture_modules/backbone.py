import torch
from torch import nn
from typing import Dict, Any, List

from architecture_modules.convolution_block import ConvolutionBlock


class Backbone(nn.Module):
    """
    Constructs the backbone feature extractor consisting of multiple stacked ConvolutionBlocks.
    
    This module builds a sequence of convolution blocks based on the configuration dictionary.
    It extracts spatial features from the input images, iteratively reducing spatial dimensions
    via pooling and increasing feature channels for downstream detection tasks.
    """

    def __init__(self, input_channels: int, convolutions: Dict[str, List[int]], modules: Dict[str, bool],
                 last_max_pooling: bool, normalization: Dict[str, bool],
                 device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the Backbone with the specified convolution blocks and parameters.
        
        Args:
            input_channels (int): The number of initial input channels for the images.
            convolutions (Dict[str, List[int]]): Dictionary mapping block indices to [number_layers, output_channels].
            modules (Dict[str, bool]): Dictionary specifying which advanced modules to use (e.g., residual).
            last_max_pooling (bool): Whether to apply max pooling at the very end of the last convolution block.
            normalization (Dict[str, bool]): Dictionary containing normalization settings like batch_normalization.
            device (torch.device): The device on which to allocate the parameters.
            dtype (torch.dtype): The desired data type of returned parameters.
        """
        super(Backbone, self).__init__()

        # Initialize the sequential container for the backbone convolution blocks
        self.blocks = nn.ModuleList()

        # Set the current input channels to the initial provided argument
        current_input_channels = input_channels

        # Extract batch and layer normalization settings from the normalization configuration dictionary
        batch_normalization = normalization.get("batch_normalization", True)
        layer_normalization = normalization.get("layer_normalization", False)

        # Count total blocks to identify the last block during iteration
        number_of_blocks = len(convolutions)

        # Iterate through the convolution blocks configuration to build the feature extraction path
        # In current Python implementations, dictionary items maintain their insertion order
        for index, (convolution_block_index, convolution_block_information) in enumerate(convolutions.items()):
            number_layers, output_channels = convolution_block_information

            # Determine whether to add pooling based on if this is the final block in the backbone
            is_last_block = (index == number_of_blocks - 1)
            add_pooling = last_max_pooling if is_last_block else True

            # Instantiate the convolution block with the explicitly provided parameters
            block = ConvolutionBlock(input_channels=current_input_channels, output_channels=output_channels, bias=True, 
                                     number_layers=number_layers, kernel_size=3, stride=1, padding=1, 
                                     batch_normalization=batch_normalization, layer_normalization=layer_normalization,
                                     activation=True, dropout_rate=0.0, 
                                     add_pooling=add_pooling, device=device, dtype=dtype)

            # Append the constructed convolution block to the sequential module list
            self.blocks.append(block)

            # Update the input channels for the subsequent convolution block
            current_input_channels = output_channels

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the sequence of convolution blocks.
        
        Args:
            input_tensor (torch.Tensor): The input image tensor to be processed.
            
        Returns:
            torch.Tensor: The resulting feature map after passing through the backbone.
        """
        # Initialize the current tensor to the input tensor before passing through the blocks
        current_tensor = input_tensor

        # Iterate through each convolution block and sequentially process the feature map
        for block in self.blocks:
            current_tensor = block(input_tensor=current_tensor)

        return current_tensor
