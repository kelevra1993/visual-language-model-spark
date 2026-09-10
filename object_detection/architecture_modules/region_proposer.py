import torch

from torch import nn
from typing import List, Union, Dict

from architecture_modules.anchors import Anchors


class RegionProposal(nn.Module):
    def __init__(self, input_channels: int, scales: List[float], aspect_ratios: List[float],
                 input_image_size: int, feature_map_size: int,
                 foreground_iou_threshold, background_iou_threholds, nms_iou_threshold,
                 dtype: torch.dtype, device: torch.device) -> None:
        """"""
        super(RegionProposal, self).__init__()

        self.dtype = dtype
        self.device = device

        # Get image and feature map sizes of interest
        self.input_image_size = input_image_size
        self.feature_map_size = feature_map_size

        # Get particular anchors for this region proposal
        self.region_proposal_anchor_object = Anchors(
            # todo to be filled up
        )
        self.number_anchors = self.region_proposal_anchor_object.number_anchors

        # 3x3 Convolution Layer : Todo later complexify this step (using ConvBlock)
        self.region_proposal_convolution = nn.Conv2d(in_channels=input_channels,
                                                     out_channels=input_channels,
                                                     kernel_size=3, stride=1, padding=1)

        # 1x1 Convolution Layer for classification : Todo later complexify this step
        self.region_proposal_classifier = nn.Conv2d(in_channels=input_channels,
                                                    out_channels=self.number_anchors,
                                                    kernel_size=1, stride=1)

        # 1x1 Convolution Layer for regression : Todo later complexify this step
        self.region_proposal_bounding_box_regressor = nn.Conv2d(in_channels=input_channels,
                                                                out_channels=self.number_anchors * 4,
                                                                kernel_size=1, stride=1)

    def forward(self, input_tensor, target_tensor):
        pass