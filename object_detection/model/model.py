import torch
from torch import nn
from typing import Dict, Any, Tuple
from architecture_modules.backbone import Backbone
from architecture_modules.region_proposer import RegionProposal


class Model(nn.Module):
    """
    Constructs the overall object detection model by combining the backbone,
    region proposal, enhancer, and detector modules based on the configuration.
    """

    def __init__(self, configuration: Dict[str, Any], device: torch.device = None, dtype: torch.dtype = None) -> None:
        """
        Initializes the Model components.
        
        Args:
            configuration (Dict[str, Any]): The complete configuration dictionary for the experiment.
            device (torch.device): The device for parameter allocation.
            dtype (torch.dtype): The data type for parameter allocation.
        """
        super(Model, self).__init__()

        # Assign device and dtype globally for the model
        self.device = device
        self.dtype = dtype

        # Get configuration for each component of our model
        data_configuration = configuration.get('Data')
        backbone_configuration = configuration.get('Backbone')
        anchors_configuration = configuration.get('Anchors')

        # Get input image size
        self.input_image_size = data_configuration.get("image_settings").get("size")

        # Initialize the backbone feature extractor using the dedicated builder method
        self.backbone = self._build_backbone(backbone_configuration=backbone_configuration)

        # Initialize the region proposal dictionary (since we are using a Feature Pyramid Network Approach)
        self.region_proposer_dictionary = self._build_region_proposer_dictionary(
            backbone_configuration=backbone_configuration,
            anchors_configuration=anchors_configuration)

    def _build_backbone(self, backbone_configuration: Dict[str, Any]) -> Backbone:
        """
        Instantiates the Backbone module based on the provided configuration blueprint.
        
        This builder method neatly encapsulates the construction of the visual feature 
        extractor, passing along the global structural parameters (like image size, 
        device, and precision type).
        
        Args:
            backbone_configuration (Dict[str, Any]): Configuration dictating the backbone topology and hyperparameters.
            
        Returns:
            Backbone: The fully initialized and configured feature extractor network.
        """
        backbone = Backbone(input_channels=backbone_configuration.get("input_channels"),
                            convolutions=backbone_configuration.get("convolutions"),
                            modules=backbone_configuration.get("modules"),
                            last_max_pooling=backbone_configuration.get("last_max_pooling"),
                            normalization=backbone_configuration.get("normalization"),
                            enhancer_convolution_indices=backbone_configuration.get("enhancer_convolution_indices"),
                            input_image_size=self.input_image_size,
                            device=self.device,
                            dtype=self.dtype)

        return backbone

    def _build_region_proposer_dictionary(self, backbone_configuration: Dict[str, Any],
                                          anchors_configuration: Dict[str, Any]) -> nn.ModuleDict:
        """
        Dynamically constructs the Region Proposal components tailored for each enhancer feature map scale.
        
        This builder method queries the Backbone for the exact spatial dimensions and channel depths 
        of the intermediate feature maps. It then parses the specified anchor configurations to securely 
        instantiate a distinct RegionProposal module for each scale level in the Feature Pyramid Network.
        
        Args:
            backbone_configuration (Dict[str, Any]): Configuration dictating backbone topology.
            anchors_configuration (Dict[str, Any]): Configuration governing anchor scales and aspect ratios
                                                    per feature level.
            
        Returns:
            nn.ModuleDict: A PyTorch ModuleDict mapping the stringified enhancer block indices to
                           their initialized RegionProposal modules.
        """
        region_proposer_dictionary = nn.ModuleDict()

        # First get the enhancer feature information, because they are used for the region proposals
        enhancer_input_information = self.backbone.compute_enhancer_input_information()

        # Iterate over the designated enhancer blocks and instantiate a RegionProposal for each multi-scale level
        for enhancer_index in backbone_configuration.get("enhancer_convolution_indices"):
            index_string = str(object=enhancer_index)

            # Retrieve the specific anchor scales and ratios for this scale level
            anchor_configuration = anchors_configuration.get("scales_and_ratios").get(index_string)
            scales = anchor_configuration.get("scales")
            aspect_ratios = anchor_configuration.get("aspect_ratios")

            # Dynamically fetch the accurate structural size and channel depth generated by the Backbone
            feature_map_size = enhancer_input_information[index_string]["feature_map_size"]
            input_channels = enhancer_input_information[index_string]["number_channels"]

            # Initialize and store the RegionProposal module
            region_proposer_dictionary[index_string] = RegionProposal(input_channels=input_channels,
                                                                      scales=scales,
                                                                      aspect_ratios=aspect_ratios,
                                                                      input_image_size=self.input_image_size,
                                                                      feature_map_size=feature_map_size,
                                                                      dtype=self.dtype,
                                                                      device=self.device)

        return region_proposer_dictionary

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes the forward pass of the Model.
        
        Args:
            input_tensor (torch.Tensor): The raw input image tensor.
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple containing the final backbone output 
            and a dictionary mapping enhancer block indices to their intermediate feature map tensors.
        """
        # Pass the raw image through the backbone to extract the multiscale feature maps
        final_backbone_tensor, backbone_output_tensor_dictionary = self.backbone(input_tensor=input_tensor)
        
        return final_backbone_tensor, backbone_output_tensor_dictionary

    def print_summary(self) -> None:
        """
        Prints a structural summary of the overarching Model architecture.
        
        This method delegates to the underlying Backbone to print its layer-by-layer 
        spatial dimension summary, ensuring the structural integrity of the visual pipeline.
        """
        self.backbone.print_summary()
