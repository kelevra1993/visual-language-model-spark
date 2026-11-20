"""
Losses of the sequence labelling (M2M) and frame selection (SEL) tasks: the per-frame classification loss with label
smoothing, ignoring the padding frames (IGNORED_LABEL_INDEX), and the selection loss making the frames of a sequence
compete for every wanted class. Called by models/sequence/sequence_labeller.py and models/sequence/frame_selector.py
during training.
"""

import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX


def compute_sequence_labelling_loss(class_logits: torch.Tensor, labels: torch.Tensor, label_smoothing: float,
                                    class_weights: torch.Tensor) -> torch.Tensor:
    """
    Computes the loss of the sequence labelling task: the cross entropy of the softmax of every frame, averaged over
    the labelled frames only, the padding frames and the frames of unmapped categories (label -100) being ignored.
    It supports label smoothing and per-class weights like the classification loss. The frame losses are summed and
    divided by the class weights of the labelled frames, clamped above 0 like the segmentation losses, which gives
    the weighted mean of the cross entropy, and 0 instead of NaN for a batch without any labelled frame, whose NaN
    would otherwise reach the tracked loss of the interval.

    Args:
        class_logits (torch.Tensor): The [B, T, K] class logits of every frame.
        labels (torch.Tensor): The [B, T] int64 class index of every frame, -100 for the frames without class.
        label_smoothing (float): The label smoothing, 0 for none.
        class_weights (torch.Tensor): The [K] weight of every class in the loss.

    Returns:
        torch.Tensor: The scalar loss, 0 for a batch without labelled frame.
    """
    flat_labels = labels.reshape(-1)

    # Weighted and smoothed cross entropy of every frame, 0 on the frames without class
    frame_losses = torch.nn.functional.cross_entropy(input=class_logits.reshape(-1, class_logits.shape[-1]),
                                                     target=flat_labels, weight=class_weights,
                                                     ignore_index=IGNORED_LABEL_INDEX, label_smoothing=label_smoothing,
                                                     reduction="none")

    # Weight of every labelled frame, the class weight of its class, normalising the loss like the weighted mean
    labelled_frames = flat_labels != IGNORED_LABEL_INDEX
    frame_weights = class_weights[flat_labels.clamp(min=0)] * labelled_frames
    return frame_losses.sum() / frame_weights.sum().clamp(min=torch.finfo(torch.float32).tiny)


def compute_frame_selection_loss(class_logits: torch.Tensor, labels: torch.Tensor,
                                 wanted_class_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the selection loss of the frame selection task, which answers "which frames of this sequence are the
    best for this class" rather than "which class is this frame": for every wanted class, the softmax of its logit
    over the frames of the sequence is trained towards its true frames (1 / N on each of its N true frames). The
    frames of a sequence then compete for every class, as the decoding of the selection makes them do. The padding
    frames (label -100) take no part, and a class without true frame in a sequence is left out.

    Args:
        class_logits (torch.Tensor): The [B, T, K] class logits of every frame.
        labels (torch.Tensor): The [B, T] int64 class index of every frame, -100 for the padding frames.
        wanted_class_mask (torch.Tensor): The [K] weights of the classes, 1 for the wanted classes and 0 for the
            unselected class.

    Returns:
        torch.Tensor: The scalar loss, averaged over the wanted classes of every sequence.
    """
    real_frames = labels != IGNORED_LABEL_INDEX
    logits = class_logits.float()

    # Distribution of every class over the real frames of its sequence, the padding frames getting no probability
    masked_logits = logits.masked_fill(mask=~real_frames[:, :, None], value=torch.finfo(logits.dtype).min)
    frame_log_probabilities = torch.log_softmax(input=masked_logits, dim=1).masked_fill(
        mask=~real_frames[:, :, None], value=0.0)

    # Target of every class: an equal share on each of its true frames
    true_frames = torch.nn.functional.one_hot(input=labels.clamp(min=0), num_classes=logits.shape[-1]).to(
        dtype=logits.dtype) * real_frames[:, :, None]
    number_true_frames = true_frames.sum(dim=1)
    target_distributions = true_frames / number_true_frames.clamp(min=1.0)[:, None, :]

    # Cross entropy of every class of every sequence, averaged over the wanted classes holding true frames
    class_losses = -(target_distributions * frame_log_probabilities).sum(dim=1)
    counted_classes = (number_true_frames > 0).to(dtype=logits.dtype) * wanted_class_mask[None, :].to(
        dtype=logits.dtype)
    return (class_losses * counted_classes).sum() / counted_classes.sum().clamp(min=1.0)
