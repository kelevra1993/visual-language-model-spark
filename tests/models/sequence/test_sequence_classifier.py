"""
Tests of SequenceClassifier of vision_training/models/sequence/sequence_classifier.py: outputs with padding frames.
"""

import pytest
import torch

from vision_training.models.sequence.sequence_classifier import SequenceClassifier
from tests.fixtures.configuration_fixtures import get_sequence_model_configuration


@pytest.mark.parametrize(argnames="temporal_type", argvalues=["transformer"])
def test_sequence_classifier_outputs_and_padding_frames(temporal_type: str) -> None:
    """
    Verifies the outputs of the sequence classifier with the transformer temporal module: [B, K] logits and softmax
    probabilities, the classification loss returned only with labels, and padding frames that do not change the
    prediction of their sequence whatever their pixels.

    Args:
        temporal_type (str): The type of the temporal module.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_sequence_model_configuration(temporal_type=temporal_type)
    model = SequenceClassifier(configuration=configuration, device=device, dtype=dtype).eval()
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(1)) * 255.0
    sequence_mask = torch.tensor(data=[[True, True, True, True], [True, True, False, False]])

    with torch.no_grad():
        outputs = model(input_tensor=frames, sequence_mask=sequence_mask)
        # White padding frames instead of noise ones, whose pooled features would clearly differ if they were used
        noisy_frames = frames.clone()
        noisy_frames[1, 2:] = 255.0
        noisy_outputs = model(input_tensor=noisy_frames, sequence_mask=sequence_mask)
        loss_outputs = model(input_tensor=frames, sequence_mask=sequence_mask, labels=torch.tensor(data=[0, 2]))

    assert outputs["class_logits"].shape == outputs["class_probabilities"].shape == (2, 3)
    assert torch.allclose(input=outputs["class_probabilities"].sum(dim=1), other=torch.ones(size=(2,)))
    assert "classification_loss" not in outputs and loss_outputs["classification_loss"].ndim == 0
    assert torch.allclose(input=outputs["class_logits"], other=noisy_outputs["class_logits"], atol=1e-5)
