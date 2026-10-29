"""
First stage of the object detector: RegionProposal predicts the objectness scores and box offsets of the anchors of
one feature map scale, and RegionProposalFilter keeps the best proposals with non-maximum suppression and pads them
to a fixed number, with different limits in training and inference. Both are built by the TwoStageDetector. The
selection of the best proposals of every image (select_best_proposals) handles the whole batch at once and is shared
with the deployed InferenceDetector, so that training and deployment filter the proposals identically.
"""

import torch

from torch import nn
from typing import List, Dict, Tuple, Any
from torchvision.ops import batched_nms, nms

from vision_training.models.detection.anchors import Anchors
from vision_training.models.components.convolution_block import ConvolutionBlock
from vision_training.models.detection.box_operations import (apply_regression_predictions,
                                                             clamp_boxes_to_image_boundaries)


def suppress_overlapping_proposals(boxes: torch.Tensor, scores: torch.Tensor, nms_iou_threshold: float,
                                   suppress_per_image: bool) -> torch.Tensor:
    """
    Applies the non-maximum suppression to the best proposals of every image of a batch for select_best_proposals,
    each image on its own. The eager training and evaluation run one torchvision nms call per image on views of the
    batch, the fastest form on the GPU (batched_nms loops over the images itself for large batches, with more
    overhead, and its single call over the whole batch is slower still), while the deployed detector runs a single
    batched_nms call, the only form that the TorchScript and ONNX exports keep for any batch size. Both keep the same
    proposals.

    Args:
        boxes (torch.Tensor): The candidate boxes of every image, shape [B, K, 4].
        scores (torch.Tensor): The objectness scores of the candidates, shape [B, K].
        nms_iou_threshold (float): The overlap above which the less confident of two proposals is suppressed.
        suppress_per_image (bool): True for one nms call per image (eager), False for one batched_nms call (export).

    Returns:
        torch.Tensor: The [B, K] boolean mask of the candidates kept by the suppression.
    """
    kept_mask = torch.zeros_like(input=scores, dtype=torch.bool)

    # One suppression per image on views of the batch, the kept candidates marked without waiting for the GPU
    if suppress_per_image:
        for image_index in range(boxes.shape[0]):
            kept_indices = nms(boxes=boxes[image_index], scores=scores[image_index], iou_threshold=nms_iou_threshold)
            kept_mask[image_index, kept_indices] = True
        return kept_mask

    # A single suppression call over the whole batch, every image its own group
    batch_size, number_candidates = scores.shape
    batch_indices = torch.arange(end=batch_size, device=scores.device).unsqueeze(dim=1)
    kept_indices = batched_nms(boxes=boxes.reshape(-1, 4), scores=scores.reshape(-1),
                               idxs=batch_indices.expand(-1, number_candidates).reshape(-1),
                               iou_threshold=nms_iou_threshold)
    kept_mask.view(-1)[kept_indices] = True
    return kept_mask


def select_best_proposals(proposal_scores: torch.Tensor, proposal_boxes: torch.Tensor, proposal_levels: torch.Tensor,
                          pre_nms_proposals: int, post_nms_proposals: int, nms_iou_threshold: float,
                          input_image_size: int,
                          suppress_per_image: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keeps the best proposals of every image of a batch, for the whole batch at once and with a fixed output size: the
    proposals the detection heads classify, selected the same way by the RegionProposalFilter of the training and
    evaluation and by the InferenceDetector of the deployment. The sigmoid of the objectness, in the precision of the
    boxes even under mixed precision, ranks the proposals, the pre_nms_proposals best ones of every image are clamped
    to the image, the non-maximum suppression of every image removes the overlapping duplicates
    (suppress_overlapping_proposals), and the post_nms_proposals best survivors of every image are returned, best
    first, the suppressed proposals only filling the slots left (marked invalid).

    Args:
        proposal_scores (torch.Tensor): The objectness logits of all anchors, shape [B, N].
        proposal_boxes (torch.Tensor): The decoded proposal boxes, shape [B, N, 4].
        proposal_levels (torch.Tensor): The detection level of every proposal, shape [B, N].
        pre_nms_proposals (int): The number of best proposals of every image kept before the suppression, at most N.
        post_nms_proposals (int): The number of proposals returned for every image, clamped to pre_nms_proposals.
        nms_iou_threshold (float): The overlap above which the less confident of two proposals is suppressed.
        input_image_size (int): The spatial dimension (height and width) of the square input image.
        suppress_per_image (bool): True for one suppression call per image (eager training and evaluation), False for
            a single call over the batch (deployed detector, exported).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The kept boxes [B, P, 4], their objectness
            scores [B, P] (-1 for the padding slots), their levels [B, P] and a boolean mask [B, P] which is False for
            the padding slots when fewer than P proposals survive.
    """
    # Keep the best pre-NMS proposals of every image and clip them to the image, the probabilities computed in the
    # precision of the boxes, decoded from the anchors (float32 under mixed precision), since the bfloat16 logits give
    # the confident anchors only a few distinct probabilities, and the suppression needs boxes and scores of one type
    top_scores, top_indices = torch.topk(input=torch.sigmoid(input=proposal_scores.to(dtype=proposal_boxes.dtype)),
                                         k=pre_nms_proposals, dim=1)
    top_boxes = clamp_boxes_to_image_boundaries(
        boxes=torch.gather(input=proposal_boxes, dim=1, index=top_indices.unsqueeze(dim=-1).expand(-1, -1, 4)),
        input_image_size=input_image_size)
    top_levels = torch.gather(input=proposal_levels, dim=1, index=top_indices)

    # Suppress the overlapping proposals of every image
    kept_mask = suppress_overlapping_proposals(boxes=top_boxes, scores=top_scores, nms_iou_threshold=nms_iou_threshold,
                                               suppress_per_image=suppress_per_image)

    # Select the best surviving proposals of every image, suppressed ones only fill the remaining slots, at most as
    # many as the proposals kept before the suppression (fewer anchors than post_nms_proposals, e.g. coarse levels)
    candidate_scores = torch.where(condition=kept_mask, input=top_scores,
                                   other=torch.full_like(input=top_scores, fill_value=-1.0))
    selected_scores, selected_indices = torch.topk(input=candidate_scores,
                                                   k=min(post_nms_proposals, pre_nms_proposals), dim=1)

    selected_boxes = torch.gather(input=top_boxes, dim=1, index=selected_indices.unsqueeze(dim=-1).expand(-1, -1, 4))
    selected_levels = torch.gather(input=top_levels, dim=1, index=selected_indices)
    return selected_boxes, selected_scores, selected_levels, selected_scores >= 0.0


class RegionProposal(nn.Module):
    def __init__(self, convolutions: List[int], input_channels: int, scales: List[float],
                 aspect_ratios: List[float], input_image_size: int, feature_map_size: int,
                 feature_map_normalization: str, batch_normalization_momentum: float, dtype: torch.dtype,
                 device: torch.device) -> None:
        """
        Initializes the Region Proposal module for a specific feature map scale.
        
        This module constructs convolutional layers to generate objectness scores and bounding box 
        regression targets from the backbone's feature maps. It also generates the anchor boxes 
        necessary to convert relative regressions into absolute coordinates.
        
        Args:
            convolutions (List[int]): The [number_layers, output_channels] of the convolution block of this level.
            input_channels (int): The number of channels output by the corresponding backbone layer.
            scales (List[float]): A list of scaling factors for the anchor boxes at this specific feature map scale.
            aspect_ratios (List[float]): A list of aspect ratios (width/height) for the anchor boxes.
            input_image_size (int): The absolute size (height and width) of the original input image.
            feature_map_size (int): The spatial resolution (height and width) of the input feature map.
            feature_map_normalization (str): The normalization strategy to use in the convolution block (e.g. 'batch',
                'none').
            batch_normalization_momentum (float): The momentum of the running statistics of its batch normalization.
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
        self.feature_map_normalization = feature_map_normalization

        # Get particular anchors for this region proposal
        self.region_proposal_anchor_object = Anchors(scales=self.scales,
                                                     aspect_ratios=self.aspect_ratios,
                                                     input_image_size=input_image_size,
                                                     feature_map_size=feature_map_size,
                                                     dtype=self.dtype,
                                                     device=self.device)

        self.number_anchors_per_location = self.region_proposal_anchor_object.number_anchors_per_location

        # Region Proposal enhancement convolution block
        self.convolution_number_layers = convolutions[0]
        convolution_output_channels = convolutions[1]

        # 3x3 Convolution Layers
        # Convolutional feature extraction block, kept plain since the Region Proposal has no modules configuration
        self.region_proposal_convolutional_block = ConvolutionBlock(
            input_channels=input_channels, output_channels=convolution_output_channels,
            bias=True, number_layers=self.convolution_number_layers,
            kernel_size=3, stride=1, padding=1,
            feature_map_normalization=self.feature_map_normalization,
            batch_normalization_momentum=batch_normalization_momentum,
            activation=True, dropout_rate=0.0, add_pooling=False, device=self.device, dtype=self.dtype,
            squeeze_and_excite=False, residual=False, pre_activation=False, inception=False)

        self.region_proposal_last_convolution = nn.Conv2d(in_channels=convolution_output_channels,
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

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                - proposal_boxes_transformations (torch.Tensor): The regressed [dx, dy, dw, dh] offsets, shape [B,
                  H*W*A, 4].
                - proposal_boxes (torch.Tensor): The absolute [x1, y1, x2, y2] bounding boxes, shape [B, H*W*A, 4].
        """
        # Get feature map dimensions
        batch_dimension, _, feature_map_height, feature_map_width = input_tensor.shape

        output_tensor = self.region_proposal_convolutional_block(input_tensor)
        output_tensor = nn.ReLU()(self.region_proposal_last_convolution(output_tensor))

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
    def __init__(self, configuration: Dict[str, Any]) -> None:
        """
        Initializes the region proposal filtering module with configurations specific to the execution phase.
        This module filters down the raw bounding box predictions generated by the region proposal network. 
        It loads specific thresholds (like NMS limits) from the configuration dictionary, utilizing different
        cutoff sizes during the 'forward' pass depending on whether the pipeline is currently in training 
        or inference mode (accessed via self.training).

        Args:
            configuration (Dict[str, Any]): The 'RegionProposal' dictionary from the model configuration file.
        """
        super(RegionProposalFilter, self).__init__()

        # Load both training and inference proposal constraints to be switched dynamically during forward pass
        self.training_pre_nms_proposals = configuration["training"]["pre_nms_proposals"]
        self.training_post_nms_proposals = configuration["training"]["post_nms_proposals"]

        self.inference_pre_nms_proposals = configuration["inference"]["pre_nms_proposals"]
        self.inference_post_nms_proposals = configuration["inference"]["post_nms_proposals"]

        # Load global NMS threshold
        self.nms_iou_threshold = configuration["nms_iou_threshold"]

    def forward(self, proposal_boxes: torch.Tensor, proposal_scores: torch.Tensor, proposal_origins: torch.Tensor,
                input_image_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Filters region proposal bounding boxes using objectness scores and Non-Maximum Suppression (NMS).

        This function processes the batched raw proposals generated by the Region Proposal Network, for all the
        images at once (select_best_proposals, shared with the deployed InferenceDetector): it applies a sigmoid
        activation to the scores, selects the top-k highest scoring proposals of every image (Pre-NMS), clamps their
        coordinates to the image boundaries, applies NMS to remove overlapping duplicates, and finally selects the
        top-k proposals from the remaining set (Post-NMS), the slots of an image left empty being filled with copies
        of its random survivors. The mask of the survivors tells them from these copies, for the proposal recall of
        the evaluation, which counts the proposals of every level.

        Args:
            proposal_boxes (torch.Tensor): A tensor containing the proposed bounding boxes of shape [B, N, 4].
            proposal_scores (torch.Tensor): A tensor containing the raw objectness logits of shape [B, N].
            proposal_origins (torch.Tensor): A tensor tracking the FPN level origin of shape [B, N].
            input_image_size (int): The spatial dimension (height and width) of the square input image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - filtered_proposal_boxes (torch.Tensor): The filtered bounding boxes of shape [B, Post_NMS, 4].
                - filtered_proposal_scores (torch.Tensor): The objectness scores for the filtered boxes of shape [B,
                  Post_NMS].
                - filtered_proposal_origins (torch.Tensor): The origin indices of shape [B, Post_NMS].
                - valid_proposal_mask (torch.Tensor): The [B, Post_NMS] boolean mask of the survivors, False for the
                  padding copies.
                Post_NMS is the post-NMS count of the mode, capped by its pre-NMS count and by the number of anchors.
        """
        # Proposal counts of the current mode, before and after the suppression
        pre_nms_filter_proposals = (self.training_pre_nms_proposals if self.training else
                                    self.inference_pre_nms_proposals)
        post_nms_filter_proposals = (self.training_post_nms_proposals if self.training else
                                     self.inference_post_nms_proposals)

        # Best proposals of every image after the suppression, the top-k limit clamped to the number of anchors
        boxes, scores, origins, valid_mask = select_best_proposals(
            proposal_scores=proposal_scores, proposal_boxes=proposal_boxes, proposal_levels=proposal_origins,
            pre_nms_proposals=min(pre_nms_filter_proposals, proposal_scores.shape[1]),
            post_nms_proposals=post_nms_filter_proposals, nms_iou_threshold=self.nms_iou_threshold,
            input_image_size=input_image_size, suppress_per_image=True)

        # Slots left empty by the suppression filled with copies of random survivors of the same image, the survivors
        # keeping their slots
        boxes, scores, origins = self._pad_proposals_to_target_size(boxes=boxes, scores=scores, origins=origins,
                                                                    valid_mask=valid_mask)
        return boxes, scores, origins, valid_mask

    def _pad_proposals_to_target_size(self, boxes: torch.Tensor, scores: torch.Tensor, origins: torch.Tensor,
                                      valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fills the slots of every image left without proposal after the suppression with copies of random valid
        proposals of the same image, for all the images at once, so that every image gives the target assignment and
        the detection heads the same number of real boxes.

        Args:
            boxes (torch.Tensor): The selected bounding boxes of shape [B, P, 4], the valid ones first in every image.
            scores (torch.Tensor): The objectness scores for those boxes of shape [B, P].
            origins (torch.Tensor): The block origin indices for those boxes of shape [B, P].
            valid_mask (torch.Tensor): The boolean mask of the valid proposals of shape [B, P], at least one per image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The padded boxes of shape [B, P, 4], scores of shape
                [B, P], and origins of shape [B, P].
        """
        # Random valid slot of every image, uniform over its number of valid proposals (the first slots)
        number_valid = valid_mask.sum(dim=1, keepdim=True)
        random_valid_slots = (torch.rand(size=valid_mask.shape, device=valid_mask.device) * number_valid).long()
        random_valid_slots = torch.minimum(input=random_valid_slots, other=number_valid - 1)

        # Every valid slot keeps its own proposal, every padding slot takes the one of its random valid slot
        slots = torch.arange(end=valid_mask.shape[1], device=valid_mask.device).expand_as(valid_mask)
        source_slots = torch.where(condition=valid_mask, input=slots, other=random_valid_slots)
        boxes = torch.gather(input=boxes, dim=1, index=source_slots.unsqueeze(dim=-1).expand(-1, -1, 4))
        scores = torch.gather(input=scores, dim=1, index=source_slots)
        origins = torch.gather(input=origins, dim=1, index=source_slots)
        return boxes, scores, origins
