import torch

from torch import nn
from architecture_modules.backbone import Backbone

class Model(nn.Module):

    def __init__(self, configuration: dict, device: torch.device = None, dtype: torch.dtype = None):
        """
        Initializes the model using a configuration dictionary.
        # todo we will later describe the model with Backbone, RPN, Enhancer...e.t.c

        Args:
            configuration (dict): A dictionary parsed from a YAML configuration file.
            device (torch.device, optional): Device on which the model should run.
            dtype (torch.dtype, optional): Data type of the model.
        """
        super().__init__()

        self.device = device
        self.dtype = dtype

        # Get configuration for each component of our model
        backbone_configuration = configuration.get('Backbone', {})


        # # Set up the backbone
        # self.backbone = Backbone(
        #     # todo to be filled
        #     device=self.device,
        #     dtype=self.dtype)