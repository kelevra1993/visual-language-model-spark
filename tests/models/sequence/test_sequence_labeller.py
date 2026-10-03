"""
Tests of SequenceLabeller of vision_training/models/sequence/sequence_labeller.py: one prediction per frame, and no
look-ahead with a causal temporal module.
"""

import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.models.sequence.sequence_labeller import SequenceLabeller
from tests.fixtures.configuration_fixtures import get_sequence_model_configuration


@pytest.mark.parametrize(argnames="temporal_type", argvalues=["gru", "lstm", "transformer", "temporal_convolution"])
def test_sequence_labeller_outputs_one_prediction_per_frame(temporal_type: str) -> None:
    """
    Verifies the outputs of the sequence labeller with every temporal module keeping the time axis: [B, T, K] logits
    and softmax probabilities, and a loss computed over the labelled frames only, unchanged by the pixels of the
    padding frames.

    Args:
        temporal_type (str): The type of the temporal module.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_sequence_model_configuration(temporal_type=temporal_type)
    model = SequenceLabeller(configuration=configuration, device=device, dtype=dtype).eval()
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(1)) * 255.0
    sequence_mask = torch.tensor(data=[[True, True, True, True], [True, True, False, False]])
    labels = torch.tensor(data=[[0, 1, 2, 1], [2, 0, IGNORED_LABEL_INDEX, IGNORED_LABEL_INDEX]])

    with torch.no_grad():
        outputs = model(input_tensor=frames, sequence_mask=sequence_mask, labels=labels)
        # White padding frames instead of noise ones, whose pooled features would clearly differ if they were used
        noisy_frames = frames.clone()
        noisy_frames[1, 2:] = 255.0
        noisy_outputs = model(input_tensor=noisy_frames, sequence_mask=sequence_mask, labels=labels)

    assert outputs["class_logits"].shape == outputs["class_probabilities"].shape == (2, 4, 3)
    assert torch.allclose(input=outputs["class_probabilities"].sum(dim=-1), other=torch.ones(size=(2, 4)))
    assert torch.allclose(input=outputs["classification_loss"], other=noisy_outputs["classification_loss"])
    assert torch.allclose(input=outputs["class_logits"][sequence_mask],
                          other=noisy_outputs["class_logits"][sequence_mask], atol=1e-5)


@pytest.mark.parametrize(argnames=("temporal_type", "causal"), argvalues=[("gru", False), ("transformer", True),
                                                                          ("temporal_convolution", True)])
def test_causal_sequence_labeller_does_not_look_ahead(temporal_type: str, causal: bool) -> None:
    """
    Verifies that a causal sequence labeller (unidirectional GRU, causal transformer or temporal convolution) predicts
    every frame from that frame and the frames before it only, as required for online use: changing the frames after
    a frame leaves the predictions of that frame and of the frames before it unchanged.

    Args:
        temporal_type (str): The type of the temporal module.
        causal (bool): The causal flag of the transformer and temporal convolution.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_sequence_model_configuration(temporal_type=temporal_type)
    configuration["TemporalModel"]["causal"] = causal
    model = SequenceLabeller(configuration=configuration, device=device, dtype=dtype).eval()
    frames = torch.rand(size=(1, 4, 3, 64, 64), generator=torch.Generator().manual_seed(2)) * 255.0
    sequence_mask = torch.ones(size=(1, 4), dtype=torch.bool)

    # Black future frames, whose pooled features clearly differ from those of the noise frames
    changed_frames = frames.clone()
    changed_frames[:, 2:] = 0.0
    with torch.no_grad():
        outputs = model(input_tensor=frames, sequence_mask=sequence_mask)["class_logits"]
        changed_outputs = model(input_tensor=changed_frames, sequence_mask=sequence_mask)["class_logits"]
    assert torch.allclose(input=outputs[:, :2], other=changed_outputs[:, :2], atol=1e-6)
    assert not torch.allclose(input=outputs[:, 2:], other=changed_outputs[:, 2:], atol=1e-6)
