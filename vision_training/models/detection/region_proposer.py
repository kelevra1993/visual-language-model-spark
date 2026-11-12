"""
First stage of the object detector: RegionProposal predicts the objectness scores and box offsets of the anchors of
one feature map scale, and RegionProposalFilter keeps the best proposals with non-maximum suppression and pads them
to a fixed number, with different limits in training and inference. Both are built by the TwoStageDetector. The
selection of the best proposals of every image (select_best_proposals), over all the levels at once or level by level
('RegionProposal.proposal_selection'), handles the whole batch at once and is shared with the deployed
InferenceDetector, so that training and deployment filter the proposals identically.
"""

import torch

from torch import nn
from typing import List, Dict, Tuple, Any
from torchvision.ops import nms

from vision_training.models.detection.anchors import Anchors
from vision_training.models.detection.postprocessing import suppress_groups_apart
from vision_training.models.components.convolution_block import ConvolutionBlock
from vision_training.models.detection.box_operations import (apply_regression_predictions,
                                                             clamp_boxes_to_image_boundaries)

# Selections of the proposals of every image ('RegionProposal.proposal_selection'): one top-k and one non-maximum
# suppression over the anchors of all the levels ("joint"), or a top-k and a suppression for every level, then the best
# survivors over all the levels ("per_level", the selection of torchvision and Detectron2)
PROPOSAL_SELECTIONS = ["joint", "per_level"]

# Smallest width and height, in input pixels, of a proposal kept by "per_level" after its clamping to the image, the
# min_size of the region proposal network of torchvision: a proposal regressed out of the image becomes an empty box
MINIMUM_PROPOSAL_SIDE = 1e-3


def suppress_overlapping_proposals(boxes: torch.Tensor, scores: torch.Tensor, nms_iou_threshold: float,
                                   suppress_per_image: bool) -> torch.Tensor:
    """
    Applies the non-maximum suppression to the best proposals of every image of a batch for select_best_proposals,
    each image on its own. The eager training and evaluation run one torchvision nms call per image on views of the
    batch, the fastest form on the GPU (batched_nms loops over the images itself for large batches, with more
    overhead, and its single call over the whole batch is slower still), while the deployed detector runs a single
    nms call with the images moved apart (suppress_groups_apart), the only form that the TorchScript and ONNX exports
    keep for any batch size. Both keep the same proposals.

    Args:
        boxes (torch.Tensor): The candidate boxes of every image, shape [B, K, 4].
        scores (torch.Tensor): The objectness scores of the candidates, shape [B, K].
        nms_iou_threshold (float): The overlap above which the less confident of two proposals is suppressed.
        suppress_per_image (bool): True for one nms call per image (eager), False for one call over the batch
            (export).

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
    kept_indices = suppress_groups_apart(boxes=boxes.reshape(-1, 4), scores=scores.reshape(-1),
                                         group_indices=batch_indices.expand(-1, number_candidates).reshape(-1),
                                         iou_threshold=nms_iou_threshold)
    kept_mask.view(-1)[kept_indices] = True
    return kept_mask


def find_nonempty_proposals(boxes: torch.Tensor, proposal_selection: str) -> torch.Tensor:
    """
    Marks the proposals select_best_proposals keeps after their clamping to the image: with "per_level", those whose
    width and height reach MINIMUM_PROPOSAL_SIDE, a proposal regressed out of the image becoming an empty box that
    torchvision removes before its suppression (remove_small_boxes); with "joint", every proposal, so that the
    selection over all the levels keeps the proposals of the runs trained before the setting.

    Args:
        boxes (torch.Tensor): The clamped [x_min, y_min, x_max, y_max] proposals, shape [B, K, 4].
        proposal_selection (str): One of PROPOSAL_SELECTIONS.

    Returns:
        torch.Tensor: The [B, K] boolean mask of the proposals kept.
    """
    if proposal_selection == "joint":
        return torch.ones_like(input=boxes[..., 0], dtype=torch.bool)
    return ((boxes[..., 2] - boxes[..., 0] >= MINIMUM_PROPOSAL_SIDE) &
            (boxes[..., 3] - boxes[..., 1] >= MINIMUM_PROPOSAL_SIDE))


def select_best_proposals(proposal_scores: torch.Tensor, proposal_boxes: torch.Tensor, proposal_levels: torch.Tensor,
                          level_sizes: List[int], pre_nms_proposals: int, post_nms_proposals: int,
                          nms_iou_threshold: float, input_image_size: int, suppress_per_image: bool,
                          proposal_selection: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keeps the best proposals of every image of a batch, for the whole batch at once and with a fixed output size: the
    proposals the detection heads classify, selected the same way by the RegionProposalFilter of the training and
    evaluation and by the InferenceDetector of the deployment. The anchors are ranked and suppressed by groups
    ('RegionProposal.proposal_selection'): all the anchors of an image at once ("joint"), or the anchors of every
    level on their own ("per_level", as torchvision and Detectron2). In every group, the sigmoid of the objectness, in
    the precision of the boxes even under mixed precision, ranks the proposals, the pre_nms_proposals best ones of
    every image are clamped to the image, and the non-maximum suppression of every image removes the overlapping
    duplicates (suppress_overlapping_proposals), "per_level" also leaving out the proposals clamped to an empty box.
    The post_nms_proposals best survivors of every image over all the groups are returned, best first, the suppressed
    proposals only filling the slots left (marked invalid).

    Args:
        proposal_scores (torch.Tensor): The objectness logits of all anchors, shape [B, N], the levels concatenated in
            the order of level_sizes.
        proposal_boxes (torch.Tensor): The decoded proposal boxes, shape [B, N, 4].
        proposal_levels (torch.Tensor): The detection level of every proposal, shape [B, N].
        level_sizes (List[int]): The number of anchors of every level, in the order of the concatenation, summing to
            N, plain numbers so that the exports never read them from a tensor shape.
        pre_nms_proposals (int): The number of best proposals of every image kept before the suppression, over all the
            anchors ("joint") or for every level ("per_level"), at most the anchors of the group.
        post_nms_proposals (int): The number of proposals returned for every image, at most the proposals kept before
            the suppression.
        nms_iou_threshold (float): The overlap above which the less confident of two proposals is suppressed.
        input_image_size (int): The spatial dimension (height and width) of the square input image.
        suppress_per_image (bool): True for one suppression call per image (eager training and evaluation), False for
            a single call over the batch (deployed detector, exported), for every group.
        proposal_selection (str): One of PROPOSAL_SELECTIONS.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The kept boxes [B, P, 4], their objectness
            scores [B, P] (-1 for the padding slots), their levels [B, P] and a boolean mask [B, P] which is False for
            the padding slots when fewer than P proposals survive.

    Raises:
        ValueError: If the selection is not one of PROPOSAL_SELECTIONS.
    """
    # Groups of anchors ranked and suppressed on their own, a fixed number of fixed slices for the exports
    if proposal_selection not in PROPOSAL_SELECTIONS:
        raise ValueError(f"Unknown proposal selection {proposal_selection!r}, expected one of {PROPOSAL_SELECTIONS}.")
    group_sizes = [sum(level_sizes)] if proposal_selection == "joint" else level_sizes
    group_ends = [sum(group_sizes[:group_index + 1]) for group_index in range(len(group_sizes))]

    group_scores, group_boxes, group_levels, group_kept_masks = [], [], [], []
    for group_start, group_end in zip([0] + group_ends[:-1], group_ends):
        # Keep the best pre-NMS proposals of every image in the group and clip them to the image, the probabilities in
        # the precision of the boxes (float32 under mixed precision), since the bfloat16 logits give the confident
        # anchors only a few distinct probabilities, and the suppression needs boxes and scores of one type
        top_scores, top_indices = torch.topk(input=torch.sigmoid(input=proposal_scores[:, group_start:group_end].to(
            dtype=proposal_boxes.dtype)), k=min(pre_nms_proposals, group_end - group_start), dim=1)
        top_indices = top_indices + group_start
        top_boxes = clamp_boxes_to_image_boundaries(
            boxes=torch.gather(input=proposal_boxes, dim=1, index=top_indices.unsqueeze(dim=-1).expand(-1, -1, 4)),
            input_image_size=input_image_size)
        top_levels = torch.gather(input=proposal_levels, dim=1, index=top_indices)

        # Suppress the overlaps of every image in the group, the empty boxes of "per_level" scored -1 to suppress none
        nonempty_mask = find_nonempty_proposals(boxes=top_boxes, proposal_selection=proposal_selection)
        kept_mask = suppress_overlapping_proposals(
            boxes=top_boxes, scores=torch.where(condition=nonempty_mask, input=top_scores, other=-1.0),
            nms_iou_threshold=nms_iou_threshold, suppress_per_image=suppress_per_image)
        group_scores.append(top_scores)
        group_boxes.append(top_boxes)
        group_levels.append(top_levels)
        group_kept_masks.append(kept_mask & nonempty_mask)

    # Candidates of all the groups, at most pre_nms_proposals per group, counted from plain numbers
    top_scores, top_boxes = torch.cat(tensors=group_scores, dim=1), torch.cat(tensors=group_boxes, dim=1)
    top_levels, kept_mask = torch.cat(tensors=group_levels, dim=1), torch.cat(tensors=group_kept_masks, dim=1)
    number_candidates = sum(min(pre_nms_proposals, group_size) for group_size in group_sizes)

    # Select the best surviving proposals of every image over all the groups, suppressed ones only filling the
    # remaining slots, at most as many as the proposals kept before the suppression (fewer anchors than
    # post_nms_proposals, e.g. coarse levels)
    candidate_scores = torch.where(condition=kept_mask, input=top_scores,
                                   other=torch.full_like(input=top_scores, fill_value=-1.0))
    selected_scores, selected_indices = torch.topk(input=candidate_scores,
                                                   k=min(post_nms_proposals, number_candidates), dim=1)

    selected_boxes = torch.gather(input=top_boxes, dim=1, index=selected_indices.unsqueeze(dim=-1).expand(-1, -1, 4))
    selected_levels = torch.gather(input=top_levels, dim=1, index=selected_indices)
    return selected_boxes, selected_scores, selected_levels, selected_scores >= 0.0


class RegionProposal(nn.Module):
    def __init__(self, convolutions: List[int], input_channels: int, scales: List[float],
                 aspect_ratios: List[float], input_image_size: int, feature_map_size: int, anchor_centres: str,
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
            anchor_centres (str): The 'RegionProposal.anchor_centres', the anchors centred on the cell corners or on
                the cell centres (ANCHOR_CENTRE_OFFSETS of anchors.py).
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

        # Get particular anchors for this region proposal, centred on the cell corners or on the cell centres
        self.region_proposal_anchor_object = Anchors(scales=self.scales, aspect_ratios=self.aspect_ratios,
                                                     input_image_size=input_image_size,
                                                     feature_map_size=feature_map_size, anchor_centres=anchor_centres,
                                                     dtype=self.dtype, device=self.device)

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

        # 1x1 Convolution Layer for classification, the depth of the head being set by its convolutions
        self.region_proposal_classifier = nn.Conv2d(in_channels=input_channels,
                                                    out_channels=self.number_anchors_per_location,
                                                    kernel_size=1, stride=1)

        # 1x1 Convolution Layer for regression, the depth of the head being set by its convolutions
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
    def __init__(self, configuration: Dict[str, Any], level_sizes: List[int]) -> None:
        """
        Initializes the region proposal filtering module with configurations specific to the execution phase.
        This module filters down the raw bounding box predictions generated by the region proposal network.
        It loads specific thresholds (like NMS limits) from the configuration dictionary, utilizing different
        cutoff sizes during the 'forward' pass depending on whether the pipeline is currently in training
        or inference mode (accessed via self.training), and the selection over all the levels or level by level.

        Args:
            configuration (Dict[str, Any]): The 'RegionProposal' dictionary from the model configuration file.
            level_sizes (List[int]): The number of anchors of every level, in the order the TwoStageDetector
                concatenates the proposals of the levels (aggregate_proposals_and_anchors).
        """
        super(RegionProposalFilter, self).__init__()

        # Load both training and inference proposal constraints to be switched dynamically during forward pass
        self.training_pre_nms_proposals = configuration["training"]["pre_nms_proposals"]
        self.training_post_nms_proposals = configuration["training"]["post_nms_proposals"]

        self.inference_pre_nms_proposals = configuration["inference"]["pre_nms_proposals"]
        self.inference_post_nms_proposals = configuration["inference"]["post_nms_proposals"]

        # Load global NMS threshold
        self.nms_iou_threshold = configuration["nms_iou_threshold"]

        # Selection over all the levels at once or level by level, and the anchors of every level it groups
        self.proposal_selection = configuration["proposal_selection"]
        self.level_sizes = list(level_sizes)

    def forward(self, proposal_boxes: torch.Tensor, proposal_scores: torch.Tensor, proposal_origins: torch.Tensor,
                input_image_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Filters region proposal bounding boxes using objectness scores and Non-Maximum Suppression (NMS).

        This function processes the batched raw proposals generated by the Region Proposal Network, for all the
        images at once (select_best_proposals, shared with the deployed InferenceDetector): it applies a sigmoid
        activation to the scores, selects the top-k highest scoring proposals of every image (Pre-NMS), over all the
        levels or for every level ('proposal_selection'), clamps their coordinates to the image boundaries, applies
        NMS to remove overlapping duplicates, and finally selects the top-k proposals from the remaining set
        (Post-NMS), the slots of an image left empty being filled with copies of its random survivors. The mask of the
        survivors tells them from these copies, for the proposal recall of the evaluation, which counts the proposals
        of every level.

        Args:
            proposal_boxes (torch.Tensor): A tensor containing the proposed bounding boxes of shape [B, N, 4].
            proposal_scores (torch.Tensor): A tensor containing the raw objectness logits of shape [B, N].
            proposal_origins (torch.Tensor): A tensor tracking the detection level origin of shape [B, N].
            input_image_size (int): The spatial dimension (height and width) of the square input image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - filtered_proposal_boxes (torch.Tensor): The filtered bounding boxes of shape [B, Post_NMS, 4].
                - filtered_proposal_scores (torch.Tensor): The objectness scores for the filtered boxes of shape [B,
                  Post_NMS].
                - filtered_proposal_origins (torch.Tensor): The origin indices of shape [B, Post_NMS].
                - valid_proposal_mask (torch.Tensor): The [B, Post_NMS] boolean mask of the survivors, False for the
                  padding copies.
                Post_NMS is the post-NMS count of the mode, capped by the proposals kept before the suppression.

        Raises:
            ValueError: If the number of proposals differs from the anchors of the levels of the filter.
        """
        # The proposals must be those of the levels the filter groups, in their order
        if proposal_scores.shape[1] != sum(self.level_sizes):
            raise ValueError(f"The proposal filter received {proposal_scores.shape[1]} proposals per image, expected "
                             f"the {sum(self.level_sizes)} anchors of its levels {self.level_sizes}.")

        # Proposal counts of the current mode, before and after the suppression
        pre_nms_filter_proposals = (self.training_pre_nms_proposals if self.training else
                                    self.inference_pre_nms_proposals)
        post_nms_filter_proposals = (self.training_post_nms_proposals if self.training else
                                     self.inference_post_nms_proposals)

        # Best proposals of every image after the suppression, the top-k limit clamped to the anchors of every group
        boxes, scores, origins, valid_mask = select_best_proposals(
            proposal_scores=proposal_scores, proposal_boxes=proposal_boxes, proposal_levels=proposal_origins,
            level_sizes=self.level_sizes, pre_nms_proposals=pre_nms_filter_proposals,
            post_nms_proposals=post_nms_filter_proposals, nms_iou_threshold=self.nms_iou_threshold,
            input_image_size=input_image_size, suppress_per_image=True, proposal_selection=self.proposal_selection)

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
