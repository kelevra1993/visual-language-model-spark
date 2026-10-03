"""
Tests of SequenceBatchNormalization of vision_training/models/components/sequence_normalization.py: the statistics are
those of each sequence, a frame depends on its sequence only and the running statistics are kept but not used.
"""

import torch

from torch import nn

from vision_training.models.components.sequence_normalization import SequenceBatchNormalization

# Grouping of the frames of the tests: 2 sequences of 3 frames, given as equal groups or as a group per frame
EQUAL_GROUPING = {"frames_per_sequence": 3, "frame_sequence_indices": None, "number_sequences": None}
INDEX_GROUPING = {"frames_per_sequence": None, "frame_sequence_indices": torch.tensor(data=[0, 0, 0, 1, 1, 1]),
                  "number_sequences": 2}


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


def test_statistics_are_those_of_every_sequence() -> None:
    """
    Verifies that every frame is normalised with the mean and the biased variance of its channel over the frames of
    its sequence and their positions, followed by the scale and shift, that the reshaped and the segment paths agree,
    and that without grouping the whole batch is one sequence.
    """
    features = torch.randn(size=(6, 4, 5, 5), generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    layer = build_layer()

    # Manual normalization of each sequence of 3 frames
    expected_outputs = []
    for sequence_features in features.split(split_size=3, dim=0):
        mean = sequence_features.mean(dim=(0, 2, 3), keepdim=True)
        variance = sequence_features.var(dim=(0, 2, 3), keepdim=True, correction=0)
        normalized = (sequence_features - mean) / torch.sqrt(variance + layer.eps)
        expected_outputs.append(normalized * layer.weight[None, :, None, None] + layer.bias[None, :, None, None])

    equal_outputs = layer(input_tensor=features, sequence_grouping=EQUAL_GROUPING)
    index_outputs = layer(input_tensor=features, sequence_grouping=INDEX_GROUPING)
    assert torch.allclose(input=equal_outputs, other=torch.cat(tensors=expected_outputs, dim=0), atol=1e-12)
    assert torch.allclose(input=index_outputs, other=equal_outputs, atol=1e-12)

    # Without grouping, the statistics are those of the whole batch, as in the training mode of batch normalization
    batch_outputs = layer(input_tensor=features, sequence_grouping=None)
    reference_outputs = nn.functional.batch_norm(input=features, running_mean=None, running_var=None,
                                                 weight=layer.weight, bias=layer.bias, training=True, eps=layer.eps)
    assert torch.allclose(input=batch_outputs, other=reference_outputs, atol=1e-12)


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
