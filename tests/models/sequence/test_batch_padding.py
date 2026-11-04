"""
Tests of the padding of the sequence batches to their longest sequence (collate_padded_sequences): the sequence
classifier of vision_training/models/sequence/ gives the same outputs and losses for a batch padded with extra frames
and for the same batch padded to its longest sequence only, the padding frames being masked everywhere, so that the
output of a sequence does not depend on the other sequences of its batch.
"""

import pytest
import torch

from vision_training.models.sequence.sequence_classifier import SequenceClassifier
from tests.fixtures.configuration_fixtures import get_sequence_model_configuration

# Real frames of the two sequences of the tested batches, padded to the 4 frames of the fixture maximum sequence length,
# the longest sequence holding 3 of them
SEQUENCE_MASK = torch.tensor(data=[[True, True, True, False], [True, True, False, False]])
LONGEST_SEQUENCE = 3


@pytest.mark.parametrize(argnames="temporal_type", argvalues=["transformer"])
def test_sequence_classifier_outputs_do_not_depend_on_the_batch_padding(temporal_type: str) -> None:
    """
    Verifies that the sequence classifier gives the same logits and loss for a batch of sequences padded to the
    maximum sequence length and for the same batch cut to its longest sequence, with the transformer temporal module.
    Computed in float64, so that only the order of the floating point operations differs.

    Args:
        temporal_type (str): The type of the temporal module.
    """
    torch.manual_seed(seed=0)
    configuration, device, _ = get_sequence_model_configuration(temporal_type=temporal_type)
    model = SequenceClassifier(configuration=configuration, device=device, dtype=torch.float64).eval()
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 255.0
    labels = torch.tensor(data=[0, 2])

    # Same sequences padded to the maximum sequence length, then cut to the longest sequence of the batch
    with torch.no_grad():
        padded_outputs = model(input_tensor=frames, sequence_mask=SEQUENCE_MASK, labels=labels)
        cut_outputs = model(input_tensor=frames[:, :LONGEST_SEQUENCE],
                            sequence_mask=SEQUENCE_MASK[:, :LONGEST_SEQUENCE], labels=labels)

    assert torch.allclose(input=cut_outputs["class_logits"], other=padded_outputs["class_logits"], atol=1e-10)
    assert torch.allclose(input=cut_outputs["classification_loss"], other=padded_outputs["classification_loss"],
                          atol=1e-10)
