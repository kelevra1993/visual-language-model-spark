import torch

from torch import nn
from typing import List, Union, Dict, Tuple, Any, Literal
from torchvision.ops import nms

from architecture_modules.anchors import Anchors
from utilities.tensor_utilities import print_tensor_shape
from utilities.model.model_utilities import apply_regression_predictions, clamp_boxes_to_image_boundaries


class RegionProposal(nn.Module):
    def __init__(self, input_channels: int, scales: List[float], aspect_ratios: List[float],
                 input_image_size: int, feature_map_size: int,
                 dtype: torch.dtype, device: torch.device) -> None:
        """
        Initializes the Region Proposal module for a specific feature map scale.
        
        This module constructs convolutional layers to generate objectness scores and bounding box 
        regression targets from the backbone's feature maps. It also generates the anchor boxes 
        necessary to convert relative regressions into absolute coordinates.
        
        Args:
            input_channels (int): The number of channels output by the corresponding backbone layer.
            scales (List[float]): A list of scaling factors for the anchor boxes at this specific feature map scale.
            aspect_ratios (List[float]): A list of aspect ratios (width/height) for the anchor boxes.
            input_image_size (int): The absolute size (height and width) of the original input image.
            feature_map_size (int): The spatial resolution (height and width) of the input feature map.
            dtype (torch.dtype): The tensor data type.
            device (torch.device): The computational device.
        """
        super(RegionProposal, self).__init__()

        self.dtype = dtype
        self.device = device

        # Get scales and aspect ratios
        self.scales = scales
        self.aspect_ratios = aspect_ratios

        # Get image and feature map sizes of interest
        self.input_image_size = input_image_size
        self.feature_map_size = feature_map_size

        # Get particular anchors for this region proposal
        self.region_proposal_anchor_object = Anchors(scales=self.scales,
                                                     aspect_ratios=self.aspect_ratios,
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

    def forward(self, input_tensor):
        """
        Processes a batched feature map to generate region proposal predictions.
        
        This executes the primary convolutions to extract objectness scores and bounding box regressions,
        reshapes the output tensors to align with the generated anchors, and applies the regression
        transformations to the base anchors to yield proposed absolute bounding boxes.
        
        Args:
            input_tensor (torch.Tensor): A batched feature map tensor of shape [B, C, H, W].
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - proposal_scores (torch.Tensor): The objectness logits for each anchor, shape [B, H*W*A].
                - proposal_boxes_transformations (torch.Tensor): The regressed [dx, dy, dw, dh] offsets, shape [B, H*W*A, 4].
                - proposal_boxes (torch.Tensor): The absolute [x1, y1, x2, y2] bounding boxes, shape [B, H*W*A, 4].
        """
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

        proposal_boxes = apply_regression_predictions(
            regression_predictions=proposal_boxes_transformations.detach().unsqueeze(dim=-2),
            boxes=self.region_proposal_anchor_object.anchors).squeeze(dim=-2)

        return proposal_scores, proposal_boxes_transformations, proposal_boxes


class RegionProposalFilter(nn.Module):
    def __init__(self, configuration: Dict[str, Any], mode: Literal["training", "inference"]) -> None:
        """
        Initializes the region proposal filtering module with configurations specific to the execution phase.
        This module filters down the raw bounding box predictions generated by the region proposal network. 
        It loads specific thresholds (like NMS limits) from the configuration dictionary, picking different
        cutoff sizes depending on whether the pipeline is currently in 'training' or 'inference' mode.

        Args:
            configuration (Dict[str, Any]): The 'RegionProposal' dictionary from the model configuration file.
            mode (Literal["training", "inference"]): The current execution mode,
                                                     must be either "training" or "inference".
        """
        super(RegionProposalFilter, self).__init__()
        self.mode = mode

        # Load mode-specific proposal constraints
        mode_configuration = configuration[self.mode]
        self.pre_nms_filter_proposals = mode_configuration["pre_nms_proposals"]
        self.post_nms_filter_proposals = mode_configuration["post_nms_proposals"]

        # Load global NMS threshold
        self.nms_iou_threshold = configuration["nms_iou_threshold"]

    def forward(self, proposal_boxes: torch.Tensor,
                proposal_scores: torch.Tensor,
                input_image_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters region proposal bounding boxes using objectness scores and Non-Maximum Suppression (NMS).

        This function processes the batched raw proposals generated by the Region Proposal Network. It iterates
        through each image in the batch, applies a sigmoid activation to the scores, selects the top-k highest 
        scoring proposals (Pre-NMS), clamps their coordinates to the image boundaries, applies NMS to remove 
        overlapping duplicates, and finally selects the top-k proposals from the remaining set (Post-NMS).

        Args:
            proposal_boxes (torch.Tensor): A tensor containing the proposed bounding boxes of shape [B, N, 4].
            proposal_scores (torch.Tensor): A tensor containing the raw objectness logits of shape [B, N].
            input_image_size (int): The spatial dimension (height and width) of the square input image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - filtered_proposal_boxes (torch.Tensor): The filtered bounding boxes of shape [B, Post_NMS, 4].
                - filtered_proposal_scores (torch.Tensor): The objectness scores for the filtered boxes of shape [B, Post_NMS].
        """
        batch_size = proposal_scores.shape[0]

        filtered_boxes_list = []
        filtered_scores_list = []

        for batch_index in range(batch_size):
            scores = proposal_scores[batch_index]
            boxes = proposal_boxes[batch_index]

            scores = torch.sigmoid(input=scores)

            # Clamp the topk limit to avoid out-of-bounds errors if total anchors < pre_nms threshold
            maximum_pre_nms_proposals = min(self.pre_nms_filter_proposals, scores.shape[0])
            _, top_proposal_indices = scores.topk(k=maximum_pre_nms_proposals)

            scores = scores[top_proposal_indices]
            boxes = boxes[top_proposal_indices]

            # Clamp boxes to image boundary
            boxes = clamp_boxes_to_image_boundaries(boxes=boxes, input_image_size=input_image_size)

            # todo later on we should also remove boxes of small area.

            # Apply NMS based on objectness
            proposal_post_nms_indices = nms(boxes=boxes,
                                            scores=scores,
                                            iou_threshold=self.nms_iou_threshold)

            # Post NMS top k filtering
            boxes = boxes[proposal_post_nms_indices][:self.post_nms_filter_proposals]
            scores = scores[proposal_post_nms_indices][:self.post_nms_filter_proposals]

            # Pad boxes and scores if necessary
            boxes, scores = self._pad_proposals_to_target_size(boxes=boxes, scores=scores)

            filtered_boxes_list.append(boxes)
            filtered_scores_list.append(scores)

        # Re-stack into a single batched tensor
        return torch.stack(tensors=filtered_boxes_list, dim=0), torch.stack(tensors=filtered_scores_list, dim=0)

    def _pad_proposals_to_target_size(self, boxes: torch.Tensor,
                                      scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads the filtered bounding boxes and scores up to post_nms_filter_proposals size by duplicating 
        valid proposals to ensure uniform tensor shapes across the batch.
        
        Args:
            boxes (torch.Tensor): The remaining bounding boxes after NMS.
            scores (torch.Tensor): The objectness scores for those boxes.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The padded boxes and scores.
        """
        number_of_proposals = boxes.shape[0]
        if number_of_proposals < self.post_nms_filter_proposals and number_of_proposals > 0:
            # Calculate how many extra proposals we need to reach the uniform batched size
            missing_proposals_count = self.post_nms_filter_proposals - number_of_proposals

            # Sample random indices from the valid proposals to duplicate them
            random_indices = torch.randint(low=0, high=number_of_proposals, size=(missing_proposals_count,),
                                           device=boxes.device, dtype=torch.long)

            extra_boxes = boxes[random_indices]
            extra_scores = scores[random_indices]

            # Add the extra boxes and scores
            boxes = torch.cat(tensors=[boxes, extra_boxes], dim=0)
            scores = torch.cat(tensors=[scores, extra_scores], dim=0)

        elif number_of_proposals == 0:
            # Extreme edge case: NMS returned zero proposals
            dummy_boxes = torch.zeros(size=(self.post_nms_filter_proposals, 4), device=boxes.device, dtype=boxes.dtype)
            dummy_scores = torch.zeros(size=(self.post_nms_filter_proposals,), device=scores.device, dtype=scores.dtype)
            boxes = dummy_boxes
            scores = dummy_scores

        return boxes, scores
