import torch
from torch import nn
from typing import Dict, Any, List, Tuple
from rich.console import Console
from rich.panel import Panel

from architecture_modules.convolution_block import ConvolutionBlock
from utilities.os_utilities import print_yellow, print_blue, print_green


class Backbone(nn.Module):
    """
    Constructs the backbone feature extractor consisting of multiple stacked ConvolutionBlocks.
    
    This module builds a sequence of convolution blocks based on the configuration dictionary.
    It extracts spatial features from the input images, iteratively reducing spatial dimensions
    via pooling and increasing feature channels for downstream detection tasks.
    """

    def __init__(self, input_channels: int, convolutions: Dict[str, List[int]], modules: Dict[str, bool],
                 last_max_pooling: bool, normalization: Dict[str, str], enhancer_convolution_indices: List[int],
                 device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the Backbone with the specified convolution blocks and parameters.
        
        Args:
            input_channels (int): The number of initial input channels for the images.
            convolutions (Dict[str, List[int]]): Dictionary mapping block indices to [number_layers, output_channels].
            modules (Dict[str, bool]): Dictionary specifying which advanced modules to use (e.g., residual).
            last_max_pooling (bool): Whether to apply max pooling at the very end of the last convolution block.
            normalization (Dict[str, str]): Dictionary containing normalization settings like feature_map_normalization.
            device (torch.device): The device on which to allocate the parameters.
            dtype (torch.dtype): The desired data type of returned parameters.
        """
        super(Backbone, self).__init__()

        # Initialize the sequential container for the backbone convolution blocks
        self.convolution_blocks = nn.ModuleList()

        # Set the current input channels to the initial provided argument
        current_input_channels = input_channels

        # Extract and securely store the chosen feature map normalization strategy for downstream architectural transparency
        self.feature_map_normalization = normalization.get("feature_map_normalization", "none")

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
            convolution_block = ConvolutionBlock(input_channels=current_input_channels, output_channels=output_channels,
                                                 bias=True, number_layers=number_layers,
                                                 kernel_size=3, stride=1, padding=1,
                                                 feature_map_normalization=self.feature_map_normalization,
                                                 activation=True, dropout_rate=0.0,
                                                 add_pooling=add_pooling, device=device, dtype=dtype)

            # Append the constructed convolution block to the sequential module list
            self.convolution_blocks.append(convolution_block)

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

        # Creation of output tensor dictionary

        # Iterate through each convolution block and sequentially process the feature map
        for block_index, block in enumerate(self.convolution_blocks, start=1):
            print(block_index)
            current_tensor = block(input_tensor=current_tensor)

        return current_tensor

    def print_summary(self, expected_image_size: Tuple[int, int]) -> None:
        """
        Prints a structural summary of the Backbone architecture given a specific input image size.
        
        This function creates a dummy tensor and passes it through the backbone blocks to trace 
        and display the spatial and channel dimensions at each significant step of the feature 
        extraction pipeline. This summary is crucial for visually verifying that the spatial resolutions 
        and channel depths accurately align with the required downstream region proposal and detection modules.
        
        Args:
            expected_image_size (Tuple[int, int]): The expected height and width of the input image.
        """
        # Initialize the rich console for formatted output, specifying a wide width to comfortably accommodate the panel
        console = Console(width=60)

        # Determine input channels from the very first convolutional layer within the first backbone block
        first_convolutional_layer = self.convolution_blocks[0].block[0]
        input_channels = first_convolutional_layer.in_channels

        # Create a dummy tensor on the appropriate device to mathematically trace the dimension transformations
        device_type = next(self.parameters()).device
        dummy_tensor = torch.zeros(size=(1, input_channels, expected_image_size[0], expected_image_size[1]),
                                   device=device_type)

        # Print the overarching title and the initial tensor shape to begin the structured summary
        print_yellow(output="Backbone Architecture Summary", add_separators=True)
        print_blue(output=f"Input image shape: {list(dummy_tensor.shape)}", add_separators=True)

        # Initialize the textual representation of the architecture,
        # including the applied normalization strategy prominently
        architecture_text = (f"[bold]Normalization Strategy:[/bold]"
                             f" {str(object=self.feature_map_normalization).capitalize()}\n\n")

        # Iterate through each block to sequentially simulate the forward pass and thoroughly record spatial mutations
        for index, block in enumerate(self.convolution_blocks):
            # Append the precise structural dimension information of the current block to the visual text summary
            architecture_text += f"[bold] - Convolution Block {index + 1}[/bold]\n"
            architecture_text += f"   - Input Shape  : {list(dummy_tensor.shape)}\n"

            # Pass the dummy tensor through the block to compute the progressively
            # downsampled spatial and channel dimensions
            dummy_tensor = block(input_tensor=dummy_tensor)
            architecture_text += f"   - Output Shape : {list(dummy_tensor.shape)}\n\n"

        # Construct the rich panel wrapping the comprehensive textual summary for an elegant terminal visualization
        architecture_panel = Panel(renderable=architecture_text.strip(), title="Backbone Architecture",
                                   border_style="blue")

        # Display the cleanly constructed panel directly to the user's terminal console
        console.print(architecture_panel)

        # Output the final resulting feature map shape, representing what is securely passed to the downstream detector
        print_green(output=f"Final Backbone output shape: {list(dummy_tensor.shape)}", add_separators=True)
