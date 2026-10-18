"""
Tests of SequenceClassifier of vision_training/models/sequence/sequence_classifier.py and of encode_frames: outputs with
padding frames, frames encoded without the padding frames and sequence normalization ignoring them.
"""

import pytest
import torch

from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.models.sequence.sequence_classifier import SequenceClassifier
from tests.fixtures.configuration_fixtures import get_sequence_model_configuration


@pytest.mark.parametrize(argnames="temporal_type", argvalues=["mean_pooling", "attention_pooling", "gru", "lstm",
                                                              "transformer", "temporal_convolution"])
def test_sequence_classifier_outputs_and_padding_frames(temporal_type: str) -> None:
    """
    Verifies the outputs of the sequence classifier with every kind of temporal module: [B, K] logits and softmax
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


def test_frames_are_encoded_without_the_padding_frames() -> None:
    """
    Verifies that the frame encoder runs the backbone on the real frames only, placing their vectors at their
    positions and zeros at the padding positions: with the batch normalization in training mode, padding frames made
    of NaN would spread NaN to every frame vector through the batch statistics if they reached the backbone.
    """
    configuration, device, dtype = get_sequence_model_configuration(temporal_type="gru")
    model = SequenceClassifier(configuration=configuration, device=device, dtype=dtype).train()
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(2)) * 255.0
    sequence_mask = torch.tensor(data=[[True, True, True, False], [True, False, False, False]])
    frames[~sequence_mask] = float("nan")

    with torch.no_grad():
        frame_features = encode_frames(backbone=model.backbone, global_pooling=model.global_pooling,
                                       input_tensor=frames, sequence_mask=sequence_mask)

    assert frame_features.shape == (2, 4, 32)
    assert not torch.isnan(input=frame_features).any()
    assert torch.equal(input=frame_features[~sequence_mask], other=torch.zeros(size=(4, 32)))
    assert frame_features[sequence_mask].abs().sum(dim=1).gt(0).all()


def test_sequence_normalization_ignores_the_padding_frames() -> None:
    """
    Verifies that a sequence classifier whose backbone uses sequence normalization normalises every sequence with its
    real frames only: the masked path of a sequence padded with black frames gives the prediction of the path without
    masks on its real frames, and changing a frame of another sequence of the batch changes nothing.
    """
    torch.manual_seed(seed=0)
    configuration, device, _ = get_sequence_model_configuration(temporal_type="gru")
    configuration["Backbone"]["normalization"]["feature_map_normalization"] = "sequence"
    model = SequenceClassifier(configuration=configuration, device=device, dtype=torch.float64).eval()
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 255.0
    frames[0, 3] = 0.0
    sequence_mask = torch.tensor(data=[[True, True, True, False], [True, True, True, True]])

    with torch.no_grad():
        masked_probabilities = model(input_tensor=frames, sequence_mask=sequence_mask)["class_probabilities"]
        real_frame_probabilities = model(input_tensor=frames[:1, :3])["class_probabilities"]
        changed_frames = frames.clone()
        changed_frames[1, 0] = 255.0 - changed_frames[1, 0]
        changed_probabilities = model(input_tensor=changed_frames, sequence_mask=sequence_mask)["class_probabilities"]

    assert torch.allclose(input=masked_probabilities[0], other=real_frame_probabilities[0], atol=1e-9)
    assert torch.allclose(input=changed_probabilities[0], other=masked_probabilities[0], atol=1e-12)
    assert not torch.allclose(input=changed_probabilities[1], other=masked_probabilities[1])
