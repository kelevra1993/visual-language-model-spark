"""
Tests of compute_sequence_labelling_loss of vision_training/losses/sequence_losses.py, which only counts the labelled
frames.
"""

import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.losses.sequence_losses import compute_sequence_labelling_loss


def test_sequence_labelling_loss_only_counts_the_labelled_frames() -> None:
    """
    Verifies that the sequence labelling loss is the weighted, smoothed cross entropy of the labelled frames only:
    the logits of the padding frames and of the frames of unmapped categories (label -100) change nothing.
    """
    class_logits = torch.randn(size=(2, 3, 4), generator=torch.Generator().manual_seed(0))
    labels = torch.tensor(data=[[0, 3, IGNORED_LABEL_INDEX], [2, IGNORED_LABEL_INDEX, IGNORED_LABEL_INDEX]])
    class_weights = torch.tensor(data=[1.0, 2.0, 0.5, 1.5])

    loss = compute_sequence_labelling_loss(class_logits=class_logits, labels=labels, label_smoothing=0.1,
                                           class_weights=class_weights)
    labelled_frames = labels != IGNORED_LABEL_INDEX
    expected_loss = torch.nn.functional.cross_entropy(input=class_logits[labelled_frames],
                                                      target=labels[labelled_frames], weight=class_weights,
                                                      label_smoothing=0.1)
    assert torch.allclose(input=loss, other=expected_loss)

    # Any logits on the ignored frames give the same loss
    changed_logits = class_logits.clone()
    changed_logits[~labelled_frames] = 100.0
    assert torch.allclose(input=compute_sequence_labelling_loss(class_logits=changed_logits, labels=labels,
                                                                label_smoothing=0.1, class_weights=class_weights),
                          other=loss)
