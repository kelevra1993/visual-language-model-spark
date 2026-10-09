"""
Tests of FrameSelector of vision_training/models/sequence/frame_selector.py: the per-frame outputs and the two losses,
the selection loss weighted by 'Selection.selection_loss_weight', and the unselected class kept in the wanted class
mask saved with the weights.
"""

import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.losses.sequence_losses import compute_frame_selection_loss
from vision_training.models.sequence.frame_selector import FrameSelector
from tests.fixtures.configuration_fixtures import get_sequence_model_configuration


@pytest.mark.parametrize(argnames="selection_loss_weight", argvalues=[0.0, 2.5])
def test_frame_selector_outputs_and_losses(selection_loss_weight: float) -> None:
    """
    Verifies the outputs of the frame selector: [B, T, K] logits and probabilities, the per-frame classification loss,
    and the selection loss over the wanted classes multiplied by its weight, 0 when the weight is 0.

    Args:
        selection_loss_weight (float): The weight of the selection loss.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_sequence_model_configuration(temporal_type="gru")
    configuration["Selection"] = {"unselected_category": "unselected", "decoding": "hungarian",
                                  "frames_per_class": 1, "selection_loss_weight": selection_loss_weight}
    model = FrameSelector(configuration=configuration, device=device, dtype=dtype).eval()
    model.wanted_class_mask[2] = 0.0
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(1)) * 255.0
    sequence_mask = torch.tensor(data=[[True, True, True, True], [True, True, True, False]])
    labels = torch.tensor(data=[[0, 2, 1, 2], [1, 0, 2, IGNORED_LABEL_INDEX]])

    with torch.no_grad():
        outputs = model(input_tensor=frames, sequence_mask=sequence_mask, labels=labels)
    assert outputs["class_probabilities"].shape == (2, 4, 3)
    assert "classification_loss" in outputs
    expected_selection_loss = selection_loss_weight * compute_frame_selection_loss(
        class_logits=outputs["class_logits"], labels=labels, wanted_class_mask=model.wanted_class_mask)
    assert outputs["selection_loss"].item() == pytest.approx(expected_selection_loss.item(), abs=1e-6)


def test_unselected_class_is_saved_with_the_weights() -> None:
    """
    Verifies that the unselected class is unknown until the trainer marks it in the wanted class mask, and that the
    mask is part of the saved weights, so that a reloaded frame selector knows its unselected class.
    """
    configuration, device, dtype = get_sequence_model_configuration(temporal_type="gru")
    configuration["Selection"] = {"unselected_category": "unselected", "decoding": "greedy", "frames_per_class": 1,
                                  "selection_loss_weight": 1.0}
    model = FrameSelector(configuration=configuration, device=device, dtype=dtype)
    with pytest.raises(expected_exception=ValueError, match="exactly one unselected class"):
        model.get_unselected_class_index()

    model.wanted_class_mask[1] = 0.0
    reloaded_model = FrameSelector(configuration=configuration, device=device, dtype=dtype)
    reloaded_model.load_state_dict(state_dict=model.state_dict())
    assert reloaded_model.get_unselected_class_index() == 1
