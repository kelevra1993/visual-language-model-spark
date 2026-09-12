import torch

from torch import nn
from torchvision.ops import nms
from typing import List, Union, Dict, Tuple

from architecture_modules.anchors import Anchors
from utilities.tensor_utilities import print_tensor_shape
from utilities.model.model_utilities import clamp_boxes_to_image_boundaries, apply_regression_predictions


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
            scales=scales,
            aspect_ratios=aspect_ratios,
            input_image_size=input_image_size,
            feature_map_size=feature_map_size,
            dtype=self.dtype,
            device=self.device)

        self.number_anchors_per_location = self.region_proposal_anchor_object.number_anchors_per_location

        # 3x3 Convolution Layer : Todo later complexify this step (using ConvBlock)
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

    # todo be extremely careful with the shapes that are being put in here.
    def filter_proposals(self, proposal_boxes: torch.Tensor,
                         proposal_scores: torch.Tensor,
                         input_image_size: int,
                         pre_nms_filter_proposals: int,
                         nms_iou_threshold: float,
                         post_nms_filter_proposals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        todo the documentation might need changes.
        Filters region proposal bounding boxes using objectness scores and Non-Maximum Suppression (NMS).

        This function processes the raw proposals generated by the Region Proposal Network. It applies
        a sigmoid activation to the scores, selects the top-k highest scoring proposals (Pre-NMS),
        clamps their coordinates to the image boundaries, applies NMS to remove overlapping duplicates,
        and finally selects the top-k proposals from the remaining set (Post-NMS).

        Args:
            proposal_boxes (torch.Tensor): A tensor containing the proposed bounding boxes.
            proposal_scores (torch.Tensor): A tensor containing the raw objectness logits for each box.
            input_image_size (int): The spatial dimension (height and width) of the square input image.
            pre_nms_filter_proposals (int): The maximum number of proposals to keep before applying NMS.
            nms_iou_threshold (float): The Intersection over Union (IoU) threshold for NMS.
            post_nms_filter_proposals (int): The maximum number of proposals to keep after applying NMS.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - proposal_boxes (torch.Tensor): The filtered bounding boxes.
                - proposal_scores (torch.Tensor): The objectness scores for the filtered boxes.
        """
        # Just putting all scores in one line
        proposal_scores = proposal_scores.reshape(-1)
        proposal_scores = torch.sigmoid(input=proposal_scores)
        _, top_proposal_scores = proposal_scores.topk(k=pre_nms_filter_proposals)

        proposal_scores = proposal_scores[top_proposal_scores]
        proposal_boxes = proposal_boxes[:, top_proposal_scores, :]

        # Clamp boxes to image boundary
        proposal_boxes = clamp_boxes_to_image_boundaries(boxes=proposal_boxes, input_image_size=input_image_size)

        # Apply NMS based on objectness : Remember here we kept the best proposal
        # Note that they do not all have values above 0.5
        proposal_post_nms_indices = nms(boxes=proposal_boxes,
                                        scores=proposal_scores,
                                        iou_threshold=nms_iou_threshold)

        # Post NMS top k filtering
        proposal_boxes = proposal_boxes[proposal_post_nms_indices][:post_nms_filter_proposals]
        proposal_scores = proposal_scores[proposal_post_nms_indices][:post_nms_filter_proposals]

        return proposal_boxes, proposal_scores
