"""
Tests of the padding of the sequence batches to their longest window ('Data.sequence_settings.batch_padding'): the
sequence classifier, the sequence labeller and the sequence comparator of vision_training/models/sequence/ give the
same outputs and losses for a batch padded to the sequence length and for the same batch cut to its longest window, the
padding frames being masked everywhere, so that the setting only changes the speed and the memory of the training.
"""

import pytest
import torch

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.models.sequence.sequence_classifier import SequenceClassifier
from vision_training.models.sequence.sequence_comparator import SequenceComparator
from vision_training.models.sequence.sequence_labeller import SequenceLabeller
from vision_training.models.sequence.temporal_modules import ORDERED_TEMPORAL_TYPES, POOLING_TEMPORAL_TYPES
from tests.fixtures.configuration_fixtures import (get_sequence_comparison_model_configuration,
                                                   get_sequence_model_configuration)

# Fusions of the members covered by the comparison test
TESTED_FUSION_TYPES = ["concatenation", "difference", "cross_attention"]

# Real frames of the two windows of the tested batches, padded to the 4 frames of the fixture sequence length, the
# longest window holding 3 of them
WINDOW_MASK = torch.tensor(data=[[True, True, True, False], [True, True, False, False]])
LONGEST_WINDOW = 3


@pytest.mark.parametrize(argnames="temporal_type", argvalues=POOLING_TEMPORAL_TYPES + ORDERED_TEMPORAL_TYPES)
def test_sequence_classifier_outputs_do_not_depend_on_the_batch_padding(temporal_type: str) -> None:
    """
    Verifies that the sequence classifier gives the same logits and loss for a batch of windows padded to the sequence
    length and for the same batch cut to its longest window, with every temporal module. Computed in float64, so that
    only the order of the floating point operations differs.

    Args:
        temporal_type (str): The type of the temporal module.
    """
    torch.manual_seed(seed=0)
    configuration, device, _ = get_sequence_model_configuration(temporal_type=temporal_type)
    model = SequenceClassifier(configuration=configuration, device=device, dtype=torch.float64).eval()
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 255.0
    labels = torch.tensor(data=[0, 2])

    # Same windows padded to the sequence length, then cut to the longest window of the batch
    with torch.no_grad():
        padded_outputs = model(input_tensor=frames, sequence_mask=WINDOW_MASK, labels=labels)
        cut_outputs = model(input_tensor=frames[:, :LONGEST_WINDOW], sequence_mask=WINDOW_MASK[:, :LONGEST_WINDOW],
                            labels=labels)

    assert torch.allclose(input=cut_outputs["class_logits"], other=padded_outputs["class_logits"], atol=1e-10)
    assert torch.allclose(input=cut_outputs["classification_loss"], other=padded_outputs["classification_loss"],
                          atol=1e-10)


@pytest.mark.parametrize(argnames="temporal_type", argvalues=ORDERED_TEMPORAL_TYPES)
def test_sequence_labeller_outputs_do_not_depend_on_the_batch_padding(temporal_type: str) -> None:
    """
    Verifies that the sequence labeller gives the same logits on the real frames and the same loss for a batch of
    windows padded to the sequence length and for the same batch cut to its longest window, with its frame labels cut
    the same way, with every temporal module keeping the time axis. Computed in float64.

    Args:
        temporal_type (str): The type of the temporal module.
    """
    torch.manual_seed(seed=0)
    configuration, device, _ = get_sequence_model_configuration(temporal_type=temporal_type)
    model = SequenceLabeller(configuration=configuration, device=device, dtype=torch.float64).eval()
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 255.0
    labels = torch.tensor(data=[[0, 1, 2, IGNORED_LABEL_INDEX], [2, 0, IGNORED_LABEL_INDEX, IGNORED_LABEL_INDEX]])

    # Same windows and frame labels padded to the sequence length, then cut to the longest window of the batch
    with torch.no_grad():
        padded_outputs = model(input_tensor=frames, sequence_mask=WINDOW_MASK, labels=labels)
        cut_outputs = model(input_tensor=frames[:, :LONGEST_WINDOW], sequence_mask=WINDOW_MASK[:, :LONGEST_WINDOW],
                            labels=labels[:, :LONGEST_WINDOW])

    assert cut_outputs["class_logits"].shape == (2, LONGEST_WINDOW, 3)
    assert torch.allclose(input=cut_outputs["class_logits"][WINDOW_MASK[:, :LONGEST_WINDOW]],
                          other=padded_outputs["class_logits"][WINDOW_MASK], atol=1e-10)
    assert torch.allclose(input=cut_outputs["classification_loss"], other=padded_outputs["classification_loss"],
                          atol=1e-10)


@pytest.mark.parametrize(argnames="fusion_type", argvalues=TESTED_FUSION_TYPES)
@pytest.mark.parametrize(argnames="temporal_type", argvalues=POOLING_TEMPORAL_TYPES + ORDERED_TEMPORAL_TYPES)
def test_sequence_comparator_outputs_do_not_depend_on_the_batch_padding(temporal_type: str, fusion_type: str) -> None:
    """
    Verifies that the sequence comparator gives the same logits and loss for a batch of pairs whose members are padded
    to the sequence length and for the same batch whose members are both cut to the longest member of the batch, with
    every fusion and every temporal module. Computed in float64.

    Args:
        temporal_type (str): The type of the temporal module.
        fusion_type (str): The type of the fusion of the members.
    """
    torch.manual_seed(seed=0)
    configuration, device, _ = get_sequence_comparison_model_configuration(temporal_type=temporal_type,
                                                                           fusion_type=fusion_type)
    model = SequenceComparator(configuration=configuration, device=device, dtype=torch.float64).eval()
    random_generator = torch.Generator().manual_seed(2)
    before_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator, dtype=torch.float64) * 255.0
    after_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator, dtype=torch.float64) * 255.0
    before_mask = WINDOW_MASK
    after_mask = torch.tensor(data=[[True, False, False, False], [True, True, True, False]])
    labels = torch.tensor(data=[0, 2])

    # Same pairs with members padded to the sequence length, then both members cut to the longest member of the batch
    with torch.no_grad():
        padded_outputs = model(before_images=before_images, after_images=after_images, before_mask=before_mask,
                               after_mask=after_mask, labels=labels)
        cut_outputs = model(before_images=before_images[:, :LONGEST_WINDOW],
                            after_images=after_images[:, :LONGEST_WINDOW], before_mask=before_mask[:, :LONGEST_WINDOW],
                            after_mask=after_mask[:, :LONGEST_WINDOW], labels=labels)

    assert torch.allclose(input=cut_outputs["class_logits"], other=padded_outputs["class_logits"], atol=1e-10)
    assert torch.allclose(input=cut_outputs["classification_loss"], other=padded_outputs["classification_loss"],
                          atol=1e-10)
