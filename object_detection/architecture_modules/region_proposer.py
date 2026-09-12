import torch

from torch import nn
from typing import List, Union, Dict, Tuple

from architecture_modules.anchors import Anchors
from utilities.tensor_utilities import print_tensor_shape
from utilities.model.model_utilities import clamp_boxes_to_image_boundaries, apply_regression_predictions


class RegionProposal(nn.Module):
    def __init__(self, input_channels: int, scales: List[float], aspect_ratios: List[float],
                 input_image_size: int, feature_map_size: int,
                 dtype: torch.dtype, device: torch.device) -> None:
        """"""
        super(RegionProposal, self).__init__()

        self.dtype = dtype
        self.device = device

        # Get image and feature map sizes of interest
        self.input_image_size = input_image_size
        self.feature_map_size = feature_map_size

        # Get particular anchors for this region proposal
        self.region_proposal_anchor_object = Anchors(scales=scales,
                                                     aspect_ratios=aspect_ratios,
                                                     input_image_size=input_image_size,
                                                     feature_map_size=feature_map_size,
                                                     dtype=self.dtype,
                                                     device=self.device)

        self.number_anchors_per_location = self.region_proposal_anchor_object.number_anchors_per_location

        # 3x3 Convolution Layer : Todo later complexify this step (using ConvBlock)
        # TODO : Note that there was no batch normalisation or even a layer normalisation that was applied here.
        self.region_proposal_convolution = nn.Conv2d(in_channels=input_channels,
                                                     out_channels=input_channels,
                                                     kernel_size=3, stride=1, padding=1)

        # 1x1 Convolution Layer for classification : Todo later complexify this step
        self.region_proposal_classifier = nn.Conv2d(in_channels=input_channels,
                                                    out_channels=self.number_anchors_per_location,
                                                    kernel_size=1, stride=1)

        # 1x1 Convolution Layer for regression : Todo later complexify this step
        self.region_proposal_bounding_box_regressor = nn.Conv2d(in_channels=input_channels,
                                                                out_channels=self.number_anchors_per_location * 4,
                                                                kernel_size=1, stride=1)

    def forward(self, input_tensor, target_tensor):
        """#todo add documentation to this function"""
        # Get feature map dimensions
        batch_dimension, _, feature_map_height, feature_map_width = input_tensor.shape

        output_tensor = nn.ReLU()(self.region_proposal_convolution(input_tensor))

        proposal_scores = self.region_proposal_classifier(output_tensor)
        proposal_boxes_transformations = self.region_proposal_bounding_box_regressor(output_tensor)

        # Reshape proposal score  from :
        #  - [batch, number_anchors_per_location, feature_map_height, feature_map_width]
        # to [batch, feature_map_height, feature_map_width, number_anchors_per_location]
        proposal_scores = proposal_scores.permute(0, 2, 3, 1)
        # Then reshape to [batch, feature_map_height * feature_map_width * number_anchors_per_location]
        proposal_scores = proposal_scores.reshape(
            batch_dimension, feature_map_height * feature_map_width * self.number_anchors_per_location)

        # Reshape proposal box transformations from
        #  -   [batch, 4*number_anchors_per_location, feature_map_height, feature_map_width]
        # to   [batch, number_anchors_per_location, 4, feature_map_height, feature_map_width]
        # then [batch, feature_map_height * feature_map_width, number_anchors_per_location, 4]
        proposal_boxes_transformations = proposal_boxes_transformations.reshape(batch_dimension,
                                                                                self.number_anchors_per_location, 4,
                                                                                feature_map_height, feature_map_width)
        proposal_boxes_transformations = proposal_boxes_transformations.permute(0, 3, 4, 1, 2)
        proposal_boxes_transformations = proposal_boxes_transformations.reshape(
            batch_dimension, feature_map_height * feature_map_width * self.number_anchors_per_location, 4)

        return proposal_scores, proposal_boxes_transformations


