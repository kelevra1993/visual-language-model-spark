import torch
from torch import nn
from typing import Dict, Any, Tuple
from architecture_modules.backbone import Backbone


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
        data_configuration = configuration.get('Data', {})
        backbone_configuration = configuration.get('Backbone', {})

        # Set up the backbone feature extractor using the specified configurations
        self.backbone = Backbone(
            input_channels=backbone_configuration.get("input_channels"),
            convolutions=backbone_configuration.get("convolutions"),
            modules=backbone_configuration.get("modules"),
            last_max_pooling=backbone_configuration.get("last_max_pooling"),
            normalization=backbone_configuration.get("normalization"),
            device=self.device,
            dtype=self.dtype
        )

    def print_summary(self, expected_image_size: Tuple[int, int]) -> None:
        """
        todo to be documented
        Args:
            expected_image_size:

        Returns:

        """
        self.backbone.print_summary(expected_image_size=expected_image_size)
