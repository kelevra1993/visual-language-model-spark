import torch

from pathlib import Path
from torch import nn
from torchvision.ops import batched_nms
from typing import Optional, Tuple

from model.model import Model
from utilities.model.model_utilities import clamp_boxes_to_image_boundaries, postprocess_detections
from utilities.os_utilities import load_experiment_configuration


class InferenceDetector(nn.Module):
    """
    Inference-only version of the trained detector, used for deployment (TorchScript and ONNX exports served by
    Triton) and for local inference.

    It reuses the submodules of the trained Model (no weight is duplicated) but replaces the data-dependent parts of
    its forward pass by batched, fixed-shape operations: proposals are filtered for the whole batch at once, padding
    proposals are masked instead of randomly duplicated, and no ground truth, target or loss code is involved. It
    returns a fixed number of detections per image.
    """

    def __init__(self, model: Model) -> None:
        """
        Wraps the submodules and inference settings of a trained Model.

        Args:
            model (Model): The trained detector, whose submodules and configuration are reused.
        """
        super(InferenceDetector, self).__init__()

        self.backbone = model.backbone
        self.region_proposer_dictionary = model.region_proposer_dictionary
        self.detector_heads_dictionary = model.detector_heads_dictionary

        # Detection levels, ordered like the backbone outputs
        self.level_names = [str(index) for index in sorted(int(name) for name in model.detector_heads_dictionary)]

        # Region proposal filtering and final post-processing settings of the inference mode
        self.input_image_size = model.input_image_size
        # Number of anchors of all levels, stored as a plain number so that tracing never reads it from a tensor shape
        self.number_anchors = model.number_anchors
        self.pre_nms_proposals = model.region_proposal_filter.inference_pre_nms_proposals
        self.post_nms_proposals = model.region_proposal_filter.inference_post_nms_proposals
        self.proposal_nms_iou_threshold = model.region_proposal_filter.nms_iou_threshold
        self.score_threshold = model.detection_score_threshold
        self.detection_nms_iou_threshold = model.detection_nms_iou_threshold
        self.maximum_detections = model.maximum_detections

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detects the objects of a batch of preprocessed images.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255] of shape [B, 3, H, W], resized and padded to the network
                input size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The detection boxes [B, K, 4] in network
                input pixels, their scores [B, K], their int64 labels [B, K] (0 for padding) and the int64 number of
                valid detections per image [B].
        """
        _, feature_maps = self.backbone(input_tensor=images)

        # Run every region proposal head and concatenate the proposals of all levels, remembering their level
        proposal_scores, proposal_boxes, proposal_levels = [], [], []
        for level_index, level_name in enumerate(self.level_names):
            level_scores, _, level_boxes = self.region_proposer_dictionary[level_name](
                input_tensor=feature_maps[level_name])
            proposal_scores.append(level_scores)
            proposal_boxes.append(level_boxes)
            proposal_levels.append(torch.full_like(input=level_scores, fill_value=level_index, dtype=torch.int64))

        filtered_boxes, filtered_levels, valid_proposal_mask = self.filter_proposals(
            proposal_scores=torch.cat(tensors=proposal_scores, dim=1),
            proposal_boxes=torch.cat(tensors=proposal_boxes, dim=1),
            proposal_levels=torch.cat(tensors=proposal_levels, dim=1))

        classification_scores, detection_boxes = self.detect(feature_maps=feature_maps, proposal_boxes=filtered_boxes,
                                                             proposal_levels=filtered_levels)

        detections = postprocess_detections(classification_scores=classification_scores,
                                            detection_boxes=detection_boxes, valid_proposal_mask=valid_proposal_mask,
                                            score_threshold=self.score_threshold,
                                            nms_iou_threshold=self.detection_nms_iou_threshold,
                                            maximum_detections=self.maximum_detections,
                                            input_image_size=self.input_image_size)

        return (detections["final_detection_boxes"], detections["final_detection_scores"],
                detections["final_detection_labels"], detections["number_detections"])

    def filter_proposals(self, proposal_scores: torch.Tensor, proposal_boxes: torch.Tensor,
                         proposal_levels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Keeps, for every image, the best 'post_nms_proposals' proposals after non-maximum suppression, like the
        training-time region proposal filter but for the whole batch at once and with a fixed output size.

        Args:
            proposal_scores (torch.Tensor): The objectness logits of all anchors, shape [B, N].
            proposal_boxes (torch.Tensor): The decoded proposal boxes, shape [B, N, 4].
            proposal_levels (torch.Tensor): The detection level index of every proposal, shape [B, N].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The kept boxes [B, P, 4], their level indices [B, P] and a
                boolean mask [B, P] which is False for padding slots when fewer than P proposals survive.
        """
        batch_size = proposal_scores.shape[0]

        # Keep the best pre-NMS proposals of every image and clip them to the image
        top_scores, top_indices = torch.topk(input=torch.sigmoid(input=proposal_scores),
                                             k=min(self.pre_nms_proposals, self.number_anchors), dim=1)
        top_boxes = clamp_boxes_to_image_boundaries(
            boxes=torch.gather(input=proposal_boxes, dim=1, index=top_indices.unsqueeze(dim=-1).expand(-1, -1, 4)),
            input_image_size=self.input_image_size)
        top_levels = torch.gather(input=proposal_levels, dim=1, index=top_indices)

        # Suppress overlapping proposals of the same image with a single NMS call over the whole batch
        number_candidates = top_scores.shape[1]
        batch_indices = torch.arange(end=batch_size, device=proposal_scores.device).unsqueeze(dim=1)
        kept_indices = batched_nms(boxes=top_boxes.reshape(-1, 4), scores=top_scores.reshape(-1),
                                   idxs=batch_indices.expand(-1, number_candidates).reshape(-1),
                                   iou_threshold=self.proposal_nms_iou_threshold)
        kept_mask = torch.zeros_like(input=top_scores.reshape(-1), dtype=torch.bool)
        kept_mask[kept_indices] = True

        # Select the best surviving proposals of every image, suppressed ones only fill the remaining slots
        candidate_scores = torch.where(kept_mask.reshape(batch_size, -1), top_scores,
                                       torch.full_like(input=top_scores, fill_value=-1.0))
        selected_scores, selected_indices = torch.topk(input=candidate_scores, k=self.post_nms_proposals, dim=1)

        selected_boxes = torch.gather(input=top_boxes, dim=1,
                                      index=selected_indices.unsqueeze(dim=-1).expand(-1, -1, 4))
        selected_levels = torch.gather(input=top_levels, dim=1, index=selected_indices)
        return selected_boxes, selected_levels, selected_scores >= 0.0

    def detect(self, feature_maps: dict, proposal_boxes: torch.Tensor, proposal_levels: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """
        Classifies and refines every proposal with the detection head of its own level.

        Every head processes all the proposals and each proposal keeps the output of its own level's head. Selecting
        the proposals of each level instead would give data-dependent tensor sizes, which the ONNX export cannot
        handle; the results are identical, at the cost of running the (small) detection heads on every proposal.

        Args:
            feature_maps (dict): The backbone feature maps, keyed by level name.
            proposal_boxes (torch.Tensor): The filtered proposals, shape [B, P, 4].
            proposal_levels (torch.Tensor): The level index of every proposal, shape [B, P].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The class logits [B, P, C] and per-class boxes [B, P, C, 4].
        """
        batch_size, number_proposals, _ = proposal_boxes.shape

        # Prepend the image index of every proposal, as expected by RoIAlign
        batch_indices = torch.arange(end=batch_size, device=proposal_boxes.device, dtype=proposal_boxes.dtype)
        batch_indices = batch_indices.unsqueeze(dim=1).expand(-1, number_proposals).reshape(-1, 1)
        region_boxes = torch.cat(tensors=[batch_indices, proposal_boxes.reshape(-1, 4)], dim=1)
        flat_levels = proposal_levels.reshape(-1)

        classification_scores: Optional[torch.Tensor] = None
        detection_boxes: Optional[torch.Tensor] = None

        # Run every level's head on all the proposals and keep, for each proposal, the output of its own level
        for level_index, level_name in enumerate(self.level_names):
            level_scores, _, level_boxes = self.detector_heads_dictionary[level_name](
                proposal_boxes=region_boxes, input_tensor=feature_maps[level_name])

            if classification_scores is None or detection_boxes is None:
                classification_scores, detection_boxes = level_scores, level_boxes
            else:
                level_mask = (flat_levels == level_index).unsqueeze(dim=1)
                classification_scores = torch.where(level_mask, level_scores, classification_scores)
                detection_boxes = torch.where(level_mask.unsqueeze(dim=2), level_boxes, detection_boxes)

        number_classes = classification_scores.shape[-1]
        return (classification_scores.reshape(batch_size, number_proposals, number_classes),
                detection_boxes.reshape(batch_size, number_proposals, number_classes, 4))


def get_checkpoint_path(experiment_folder: Path, iteration: Optional[int]) -> Path:
    """
    Finds the weight file of a trained experiment, as saved by the Trainer in '<experiment>/Weights'.

    Args:
        experiment_folder (Path): The experiment folder holding 'Weights' and 'training_configuration.yaml'.
        iteration (Optional[int]): The iteration to load, or None for the last one recorded in the 'full-checkpoint'
            registry.

    Returns:
        Path: The path of the checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint exists for the requested iteration.
    """
    weights_directory = experiment_folder / "Weights"

    # The first line of the registry names the last saved iteration, e.g. 'model_checkpoint_path: "Iteration_2000"'
    if iteration is None:
        registry_path = weights_directory / "full-checkpoint"
        if not registry_path.exists():
            raise FileNotFoundError(f"No checkpoint registry found : file://{registry_path.absolute()}")
        iteration = int(registry_path.read_text().splitlines()[0].split('"')[1].split("_")[-1])

    checkpoint_path = weights_directory / f"Iteration_{iteration}" / f"model_{iteration:06}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found : file://{checkpoint_path.absolute()}")
    return checkpoint_path


def load_trained_model(configuration_path: Path, checkpoint_path: Path, device: torch.device) -> Model:
    """
    Rebuilds a trained detector from its saved training configuration and checkpoint, in evaluation mode, as the
    starting point of every deployment and inference script.

    Args:
        configuration_path (Path): The 'training_configuration.yaml' saved in the experiment folder.
        checkpoint_path (Path): The checkpoint file written by the Trainer.
        device (torch.device): The device on which to load the model.

    Returns:
        Model: The trained detector in evaluation mode.
    """
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)

    model = Model(configuration=model_configuration, device=device, dtype=experiment_configuration["dtype"])
    checkpoint = torch.load(f=checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=checkpoint["model_state"])

    return model.to(device=device).eval()
