"""
Tests of SequenceBatchNormalization of vision_training/models/components/sequence_normalization.py: a frame depends on
its sequence only and the running statistics are kept but not used.
"""

import torch

from torch import nn

from vision_training.models.components.sequence_normalization import SequenceBatchNormalization

# Grouping of the frames of the tests: 2 sequences of 3 frames, given as equal groups
EQUAL_GROUPING = {"frames_per_sequence": 3, "frame_sequence_indices": None, "number_sequences": None}


def build_layer() -> SequenceBatchNormalization:
    """
    Builds a sequence normalization layer of 4 channels in float64 with a non-trivial scale and shift.

    Returns:
        SequenceBatchNormalization: The layer.
    """
    layer = SequenceBatchNormalization(num_features=4, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(data=[1.0, 2.0, 0.5, -1.0], dtype=torch.float64))
        layer.bias.copy_(torch.tensor(data=[0.0, 1.0, -1.0, 0.5], dtype=torch.float64))
    return layer


def test_a_frame_depends_on_its_sequence_only() -> None:
    """
    Verifies that changing a frame of the first sequence changes the outputs of every frame of that sequence and of
    no frame of the second sequence, also for sequences of different lengths given frame by frame.
    """
    features = torch.randn(size=(5, 4, 3, 3), generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    grouping = {"frames_per_sequence": None, "frame_sequence_indices": torch.tensor(data=[0, 0, 1, 1, 1]),
                "number_sequences": 2}
    layer = build_layer()
    changed_features = features.clone()
    changed_features[0] += 3.0

    outputs = layer(input_tensor=features, sequence_grouping=grouping)
    changed_outputs = layer(input_tensor=changed_features, sequence_grouping=grouping)
    assert not torch.allclose(input=outputs[1], other=changed_outputs[1])
    assert torch.allclose(input=outputs[2:], other=changed_outputs[2:], atol=1e-12)


def test_running_statistics_are_kept_but_not_used() -> None:
    """
    Verifies that the running statistics follow the training batches exactly like batch normalization, while the
    outputs are the same in training and evaluation modes, the statistics of the current sequence being always used,
    and that the layer holds the tensors of batch normalization.
    """
    features = torch.randn(size=(6, 4, 5, 5), generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 3.0
    layer = build_layer()
    batch_normalization = nn.BatchNorm2d(num_features=4, dtype=torch.float64)
    assert list(layer.state_dict()) == list(batch_normalization.state_dict())

    # Two training steps on the same batches
    for _ in range(2):
        training_outputs = layer(input_tensor=features, sequence_grouping=EQUAL_GROUPING)
        batch_normalization(features)
    assert torch.allclose(input=layer.running_mean, other=batch_normalization.running_mean, atol=1e-12)
    assert torch.allclose(input=layer.running_var, other=batch_normalization.running_var, atol=1e-12)
    assert int(layer.num_batches_tracked) == 2

    # Evaluation gives the same outputs and leaves the running statistics untouched
    running_mean_before = layer.running_mean.clone()
    layer.eval()
    evaluation_outputs = layer(input_tensor=features, sequence_grouping=EQUAL_GROUPING)
    assert torch.allclose(input=evaluation_outputs, other=training_outputs, atol=1e-12)
    assert torch.equal(input=layer.running_mean, other=running_mean_before)

    # The tensors of a batch normalization layer load into the sequence normalization layer
    layer.load_state_dict(state_dict=batch_normalization.state_dict(), strict=True)
