import torch
from torch import nn
from typing import Dict, Any, List, Tuple
from rich.console import Console
from rich.panel import Panel

from architecture_modules.convolution_block import ConvolutionBlock
from utilities.os_utilities import print_yellow, print_blue, print_green


class Backbone(nn.Module):
    """
    Constructs the foundational feature extractor backbone consisting of multiple stacked ConvolutionBlocks.
    
    In the context of the visual language model pipeline, this module is responsible for ingesting the raw input 
    images and iteratively downsampling their spatial dimensions while expanding their channel capacity. The resulting 
    high-level semantic feature maps, as well as intermediate multi-scale representations (detection blocks), are 
    subsequently passed to downstream modules like the Region Proposal Network and the Feature Pyramid Network 
    to localize and identify objects.
    """

    def __init__(self, input_channels: int, convolutions: Dict[str, List[int]], modules: Dict[str, bool],
                 last_max_pooling: bool, normalization: Dict[str, str], detection_convolution_block_indices: List[int],
                 input_image_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Initializes the Backbone with the specified convolution blocks and structural parameters.
        
        Args:
            input_channels (int): The number of initial input channels for the images.
            convolutions (Dict[str, List[int]]): Dictionary mapping block indices to [number_layers, output_channels].
            modules (Dict[str, bool]): Dictionary specifying which advanced modules to use (e.g., residual).
            last_max_pooling (bool): Whether to apply max pooling at the very end of the last convolution block.
            normalization (Dict[str, str]): Dictionary containing normalization settings like feature_map_normalization.
            detection_convolution_block_indices (List[int]): A list of block indices (1-indexed) whose intermediate feature maps 
                are explicitly captured and returned for downstream multi-scale feature enhancement.
            input_image_size (int): The spatial dimension (height and width) of the square input image, used to dynamically compute downstream tensor sizes.
            device (torch.device): The device on which to allocate the parameters.
            dtype (torch.dtype): The desired data type of returned parameters.
        """
        super(Backbone, self).__init__()

        # Store device and data type for dynamic dummy tensor generation
        self.device = device
        self.dtype = dtype

        # Get input image size
        self.input_image_size = input_image_size

        # Initialize the sequential container for the backbone convolution blocks
        self.convolution_blocks = nn.ModuleList()

        # Set detection convolution indices
        self.detection_convolution_block_indices = detection_convolution_block_indices

        # Set the current input channels to the initial provided argument
        current_input_channels = input_channels

        # Extract chosen feature map normalization strategy
        self.feature_map_normalization = normalization.get("feature_map_normalization")

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

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Processes the input tensor through the sequence of convolution blocks.
        
        Args:
            input_tensor (torch.Tensor): The input image tensor to be processed.
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing the final resulting feature map 
            after passing through the backbone, and a dictionary containing the intermediate feature maps 
            from the designated detection blocks.
        """
        # Initialize the current tensor to the input tensor before passing through the blocks
        current_tensor = input_tensor

        # Creation of output tensor dictionary
        output_tensor_dictionary = {}

        # Iterate through each convolution block and sequentially process the feature map
        for block_index, block in enumerate(self.convolution_blocks, start=1):
            current_tensor = block(input_tensor=current_tensor)

            # Check if the current block is designated as an detection and save its output
            if block_index in self.detection_convolution_block_indices:
                output_tensor_dictionary[str(block_index)] = current_tensor

        return current_tensor, output_tensor_dictionary

    def print_summary(self) -> None:
        """
        Prints a structural summary of the Backbone architecture using the initialized input image size.
        
        This function creates a dummy tensor and passes it through the backbone blocks to trace 
        and display the spatial and channel dimensions at each significant step of the feature 
        extraction pipeline. This summary is crucial for visually verifying that the spatial resolutions 
        and channel depths accurately align with the required downstream region proposal and detection modules.
        """
        # Initialize the rich console for formatted output, specifying a wide width to comfortably accommodate the panel
        console = Console(width=60)

        # Determine input channels from the very first convolutional layer within the first backbone block
        first_convolutional_layer = self.convolution_blocks[0].block[0]
        input_channels = first_convolutional_layer.in_channels

        # Create a dummy tensor on the appropriate device to mathematically trace the dimension transformations
        dummy_tensor = torch.zeros(size=(1, input_channels, self.input_image_size, self.input_image_size),
                                   device=self.device)

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

    def compute_detection_input_information(self) -> Dict[str, Dict[str, int]]:
        """
        Executes a dry-run forward pass with a dummy tensor to securely determine the 
        spatial feature map sizes (assuming square maps) and channel depths for each 
        designated detection block using the initialized input image size.
        
        This dynamically resolves the exact structural dimensions required by the Region Proposal 
        Network (and Anchors) during its initialization phase, avoiding error-prone 
        hardcoded size calculations that could break if pooling logic or strides change.
        
        Returns:
            Dict[str, Dict[str, int]]: A dictionary mapping the stringified detection block index to 
            a dictionary containing 'feature_map_size' and 'number_channels'.
        """
        # Determine the input channels expected by the backbone from its very first layer
        first_convolutional_layer = self.convolution_blocks[0].block[0]
        input_channels = first_convolutional_layer.in_channels

        # Instantiate a dummy input tensor of the specified shape using self.device
        dummy_tensor = torch.zeros(size=(1, input_channels, self.input_image_size, self.input_image_size),
                                   device=self.device)

        # Dictionary to store the dynamically computed structural information for each detection block
        detection_information = {}

        # Sequentially pass the dummy tensor through all convolution blocks
        for block_index, block in enumerate(self.convolution_blocks, start=1):
            dummy_tensor = block(input_tensor=dummy_tensor)

            # If this block is designated as an detection, extract its spatial dimension and channel depth
            if block_index in self.detection_convolution_block_indices:
                detection_information[str(object=block_index)] = {"feature_map_size": dummy_tensor.shape[2],
                                                                 "number_channels": dummy_tensor.shape[1]}

        return detection_information
