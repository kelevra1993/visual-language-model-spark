"""
Final postprocessing of the object detector: class probabilities, background removal, score threshold, per-class
batched non-maximum suppression and a fixed number of detections per image, vectorised so that it exports. Shared by
the TwoStageDetector at inference and by the InferenceDetector of the deployment.
"""

import torch

from typing import Dict
from torchvision.ops import batched_nms

from vision_training.models.detection.box_operations import clamp_boxes_to_image_boundaries


def postprocess_detections(classification_scores: torch.Tensor, detection_boxes: torch.Tensor,
                           valid_proposal_mask: torch.Tensor, score_threshold: float, nms_iou_threshold: float,
                           maximum_detections: int, input_image_size: int,
                           suppress_candidates_only: bool) -> Dict[str, torch.Tensor]:
    """
    Turns the raw detection head outputs into the final detections returned at inference and by the deployed
    models: class probabilities, removal of the background class, a score threshold, per-class non-maximum
    suppression, and a fixed number of detections per image.

    Every step is vectorised over the whole batch (a single batched NMS call grouped by image and class, then a fixed
    top-k). The eager training and evaluation give the NMS the pairs reaching the score threshold only, while the
    deployed detector gives it every pair, without data-dependent Python control flow nor data-dependent sizes, so
    that the same code can be exported to TorchScript and ONNX. Both keep the same detections. The outputs have a
    fixed size; padded entries have a score of 0, a label of 0 and an empty box.

    Args:
        classification_scores (torch.Tensor): The class logits of every proposal, shape [B, P, C] (C includes the
            background class 0).
        detection_boxes (torch.Tensor): The per-class decoded [x1, y1, x2, y2] boxes, shape [B, P, C, 4].
        valid_proposal_mask (torch.Tensor): Boolean mask of shape [B, P], False for padding proposals that must never
            produce a detection (used by the exportable inference path, which pads with placeholder proposals).
        score_threshold (float): The minimum class probability of a returned detection.
        nms_iou_threshold (float): The IoU above which a lower scoring box of the same class is suppressed.
        maximum_detections (int): The fixed number K of detections returned per image, padded when the image has
            fewer (proposal, foreground class) pairs, as in the validation, which classifies the sampled proposals only.
        input_image_size (int): The size of the square network input, used to clip the boxes to the image.
        suppress_candidates_only (bool): True for the NMS on the pairs reaching the score threshold (eager), False for
            the NMS on every pair (export).

    Returns:
        Dict[str, torch.Tensor]: The final detections:
            - "final_detection_boxes" (torch.Tensor): Shape [B, K, 4].
            - "final_detection_scores" (torch.Tensor): Shape [B, K], sorted in decreasing order.
            - "final_detection_labels" (torch.Tensor): Shape [B, K], int64 class indices (0 for padding).
            - "number_detections" (torch.Tensor): Shape [B], int64 number of valid detections per image.
            - "final_detection_proposal_indices" (torch.Tensor): Shape [B, K], int64 index of the proposal of every
              detection among the P proposals of its image (0 for padding), read by the instance segmenter to run
              the mask head of the level of that proposal.
    """
    batch_size, number_proposals, number_classes = classification_scores.shape

    # Class probabilities without the background class, zero for padding proposals, flattened over proposals and
    # foreground classes
    probabilities = torch.softmax(input=classification_scores, dim=-1)[..., 1:]
    probabilities = (probabilities * valid_proposal_mask.unsqueeze(dim=-1)).reshape(batch_size, -1)
    boxes = clamp_boxes_to_image_boundaries(boxes=detection_boxes[..., 1:, :],
                                            input_image_size=input_image_size).reshape(batch_size, -1, 4)
    labels = torch.arange(start=1, end=number_classes, device=classification_scores.device)
    labels = labels.repeat(number_proposals).unsqueeze(dim=0).expand(batch_size, -1)

    # Candidates that can become detections: the pairs of valid proposals reaching the score threshold
    batch_indices = torch.arange(end=batch_size, device=classification_scores.device).unsqueeze(dim=1).expand_as(labels)
    group_indices = batch_indices * number_classes + labels
    valid_candidate_mask = valid_proposal_mask.unsqueeze(dim=-1).expand(-1, -1, number_classes - 1)
    candidate_mask = (probabilities >= score_threshold) & valid_candidate_mask.reshape(batch_size, -1)

    # Suppress overlapping boxes of the same class and image with a single NMS call over the whole batch. The eager
    # path only gives it the candidates: a box below the threshold never suppresses a higher scoring one, so the
    # detections are the same, without the per-group NMS loop torchvision runs on the hundreds of thousands of pairs
    # of a whole batch. The exports give it every pair, the dynamo ONNX export refusing a data-dependent number of
    # boxes
    if suppress_candidates_only:
        candidate_indices = torch.nonzero(input=candidate_mask.reshape(-1)).flatten()
        kept_indices = candidate_indices[batched_nms(boxes=boxes.reshape(-1, 4)[candidate_indices],
                                                     scores=probabilities.reshape(-1)[candidate_indices],
                                                     idxs=group_indices.reshape(-1)[candidate_indices],
                                                     iou_threshold=nms_iou_threshold)]
    else:
        kept_indices = batched_nms(boxes=boxes.reshape(-1, 4), scores=probabilities.reshape(-1),
                                   idxs=group_indices.reshape(-1), iou_threshold=nms_iou_threshold)

    # Keep the NMS survivors among the candidates, the other pairs get a score below any valid one
    kept_mask = torch.zeros_like(input=probabilities.reshape(-1), dtype=torch.bool)
    kept_mask[kept_indices] = True
    kept_mask = kept_mask.reshape(batch_size, -1) & candidate_mask
    candidate_scores = torch.where(kept_mask, probabilities, torch.full_like(input=probabilities, fill_value=-1.0))

    # Append maximum_detections invalid candidates (empty box, background label), so that an image with fewer
    # candidates than detections (few proposals and classes) still fills its fixed number of detections, without
    # any Python condition on the tensor sizes that the exports would freeze
    candidate_scores = torch.nn.functional.pad(input=candidate_scores, pad=(0, maximum_detections), value=-1.0)
    boxes = torch.nn.functional.pad(input=boxes, pad=(0, 0, 0, maximum_detections))
    labels = torch.nn.functional.pad(input=labels, pad=(0, maximum_detections))

    # Select the fixed number of best detections per image and zero out the padded entries
    top_scores, top_indices = torch.topk(input=candidate_scores, k=maximum_detections, dim=1)
    valid_mask = top_scores >= 0.0
    final_boxes = torch.gather(input=boxes, dim=1, index=top_indices.unsqueeze(dim=-1).expand(-1, -1, 4))

    final_labels = torch.gather(input=labels, dim=1, index=top_indices)

    # Proposal of every detection, the candidates being flattened proposal by proposal over the foreground classes
    final_proposal_indices = torch.div(input=top_indices, other=number_classes - 1, rounding_mode="floor")
    return {"final_detection_boxes": torch.where(valid_mask.unsqueeze(dim=-1), final_boxes,
                                                 torch.zeros_like(input=final_boxes)),
            "final_detection_scores": torch.where(valid_mask, top_scores, torch.zeros_like(input=top_scores)),
            "final_detection_labels": torch.where(valid_mask, final_labels, torch.zeros_like(input=final_labels)),
            "number_detections": valid_mask.sum(dim=1),
            "final_detection_proposal_indices": torch.where(valid_mask, final_proposal_indices,
                                                            torch.zeros_like(input=final_proposal_indices))}
