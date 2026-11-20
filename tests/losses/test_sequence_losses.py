"""
Tests of vision_training/losses/sequence_losses.py: compute_sequence_labelling_loss, which only counts the labelled
frames and gives 0 for a batch without any, and compute_frame_selection_loss, the softmax of every wanted class over
the frames of its sequence.
"""

import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.losses.sequence_losses import compute_frame_selection_loss, compute_sequence_labelling_loss


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


def test_sequence_labelling_loss_of_a_batch_without_labelled_frame_is_zero() -> None:
    """
    Verifies that a batch whose frames are all padding or of unmapped categories gives a loss of 0 with finite
    gradients, instead of the NaN of the mean over no frame, which would reach the tracked loss of the interval, and
    that a class weight below 1 on the single labelled frame of a batch still gives the plain cross entropy of that
    frame, the normaliser being the class weight of the labelled frames.
    """
    class_logits = torch.randn(size=(2, 3, 4), generator=torch.Generator().manual_seed(0), requires_grad=True)
    class_weights = torch.tensor(data=[1.0, 2.0, 0.5, 1.5])

    # No labelled frame: a loss of 0 and gradients of 0
    loss = compute_sequence_labelling_loss(class_logits=class_logits,
                                           labels=torch.full(size=(2, 3), fill_value=IGNORED_LABEL_INDEX),
                                           label_smoothing=0.1, class_weights=class_weights)
    loss.backward()
    assert loss.item() == 0.0 and bool(torch.isfinite(class_logits.grad).all())

    # A single labelled frame of the class weighted 0.5
    labels = torch.full(size=(2, 3), fill_value=IGNORED_LABEL_INDEX)
    labels[1, 2] = 2
    loss = compute_sequence_labelling_loss(class_logits=class_logits, labels=labels, label_smoothing=0.0,
                                           class_weights=class_weights)
    expected_loss = torch.nn.functional.cross_entropy(input=class_logits[1, 2:], target=labels[1, 2:])
    assert loss.item() == pytest.approx(expected=expected_loss.item(), rel=1e-6)


def test_frame_selection_loss_makes_the_frames_compete_for_every_wanted_class() -> None:
    """
    Verifies the selection loss on two sequences: for every wanted class, the cross entropy of the softmax of its
    logit over the real frames of the sequence against its true frames (half on each of two true frames), averaged
    over the wanted classes of every sequence, the unselected class and the padding frames taking no part.
    """
    class_logits = torch.randn(size=(2, 4, 3), generator=torch.Generator().manual_seed(0))
    # Classes 0 and 1 are wanted, class 2 is unselected; the first sequence holds class 0 twice, the second sequence
    # ends with a padding frame
    labels = torch.tensor(data=[[0, 1, 0, 2], [2, 1, 0, IGNORED_LABEL_INDEX]])
    wanted_class_mask = torch.tensor(data=[1.0, 1.0, 0.0])
    loss = compute_frame_selection_loss(class_logits=class_logits, labels=labels, wanted_class_mask=wanted_class_mask)

    # Expected loss, class by class, over the real frames of every sequence
    first_class_zero = -0.5 * (torch.log_softmax(input=class_logits[0, :, 0], dim=0)[[0, 2]]).sum()
    first_class_one = -torch.log_softmax(input=class_logits[0, :, 1], dim=0)[1]
    second_class_zero = -torch.log_softmax(input=class_logits[1, :3, 0], dim=0)[2]
    second_class_one = -torch.log_softmax(input=class_logits[1, :3, 1], dim=0)[1]
    expected_loss = (first_class_zero + first_class_one + second_class_zero + second_class_one) / 4
    assert loss.item() == pytest.approx(expected_loss.item(), rel=1e-5)

    # Any logits on the padding frame, or for the unselected class, give the same loss
    changed_logits = class_logits.clone()
    changed_logits[1, 3] = 100.0
    changed_logits[:, :, 2] = -50.0
    assert compute_frame_selection_loss(class_logits=changed_logits, labels=labels,
                                        wanted_class_mask=wanted_class_mask).item() == pytest.approx(loss.item(),
                                                                                                     rel=1e-5)
