"""
Tests of the hidden fully connected layers of vision_training/models/components/fully_connected_block.py and of their
batch or layer normalization in the classification head: the layers of a batch normalized block, the batch
normalization of vectors of any leading shape, the padding vectors left out of the batch statistics, and the features
of a hidden layer kept before its ReLU for the decorrelation loss.
"""

import pytest
import torch

from torch import nn

from vision_training.models.classification.classification_head import ClassificationHead
from vision_training.models.components.fully_connected_block import (FeatureBatchNormalization,
                                                                     build_fully_connected_block)

DEVICE = torch.device(device="cpu")
DTYPE = torch.float32


@pytest.mark.parametrize(argnames=("normalization", "expected_layer_types"), argvalues=[
    ("batch", [nn.Dropout, nn.Linear, FeatureBatchNormalization, nn.ReLU, nn.Dropout, nn.Linear,
               FeatureBatchNormalization, nn.ReLU])])
def test_fully_connected_layers(normalization: str, expected_layer_types: list) -> None:
    """
    Verifies the layers of the hidden fully connected block: dropout, linear and ReLU per layer, with a batch
    normalization at the configured momentum between every linear layer and its ReLU.

    Args:
        normalization (str): The normalization of the hidden layers, "none", "batch" or "layer".
        expected_layer_types (list): The expected type of every layer of the block, in order.
    """
    fully_connected_block = build_fully_connected_block(input_dimension=12, dimensions=[8, 4], dropout_rate=0.2,
                                                        add_dropout_layers=True, normalization=normalization,
                                                        batch_normalization_momentum=0.01, device=DEVICE, dtype=DTYPE)

    assert [type(layer) for layer in fully_connected_block] == expected_layer_types
    assert all(layer.momentum == 0.01 for layer in fully_connected_block
               if isinstance(layer, FeatureBatchNormalization))
    assert [layer.normalized_shape for layer in fully_connected_block
            if isinstance(layer, nn.LayerNorm)] == ([(8,), (4,)] if normalization == "layer" else [])

    # Without normalization, the weights keep the names of the block written before the normalization existed
    if normalization == "none":
        assert list(fully_connected_block.state_dict().keys()) == ["1.weight", "1.bias", "4.weight", "4.bias"]


def test_feature_batch_normalization_of_frame_vectors() -> None:
    """
    Verifies that the batch normalization of feature vectors normalizes every feature over all the vectors of the
    batch, whatever their leading shape, exactly as nn.BatchNorm1d does on the flattened vectors, and that its running
    statistics follow the training batches at its momentum.
    """
    torch.manual_seed(seed=0)
    frame_vectors = torch.randn(size=(3, 5, 4)) * 2.0 + 1.0
    feature_normalization = FeatureBatchNormalization(num_features=4, momentum=0.01, device=DEVICE, dtype=DTYPE)
    reference_normalization = nn.BatchNorm1d(num_features=4, momentum=0.01, device=DEVICE, dtype=DTYPE)

    normalized_vectors = feature_normalization(input_tensor=frame_vectors)
    reference_vectors = reference_normalization(input=frame_vectors.reshape(15, 4)).reshape(3, 5, 4)

    assert normalized_vectors.shape == (3, 5, 4)
    assert torch.allclose(input=normalized_vectors, other=reference_vectors, atol=1e-6)
    expected_running_mean = 0.01 * frame_vectors.reshape(15, 4).mean(dim=0)
    assert torch.allclose(input=feature_normalization.running_mean, other=expected_running_mean, atol=1e-6)


def test_padding_vectors_take_no_part_in_the_head_statistics() -> None:
    """
    Verifies that a classification head with batch normalized hidden layers classifies the real vectors of the sequences
    of the sequence labeller with the statistics of the real vectors only, whatever the values of the padding vectors,
    and gives the padding vectors zero logits; without normalization or with the layer normalization, the mask changes
    nothing.
    """
    torch.manual_seed(seed=0)
    head = ClassificationHead(input_features=6, fully_connected=[8], fully_connected_normalization="batch",
                              batch_normalization_momentum=0.1, dropout_rate=0.0, number_classes=3, device=DEVICE,
                              dtype=DTYPE)
    frame_vectors = torch.randn(size=(2, 4, 6))
    frame_mask = torch.tensor(data=[[True, True, True, False], [True, True, False, False]])

    # Padding vectors of huge values, which would dominate the batch statistics if they were included
    padded_vectors = frame_vectors.clone()
    padded_vectors[~frame_mask] = 1000.0
    class_logits = head(features=padded_vectors, feature_mask=frame_mask)
    real_logits, _ = head.compute_logits(features=frame_vectors[frame_mask], kept_hidden_layer=None)

    assert class_logits.shape == (2, 4, 3)
    assert torch.allclose(input=class_logits[frame_mask], other=real_logits, atol=1e-6)
    assert torch.equal(input=class_logits[~frame_mask], other=torch.zeros(size=(3, 3)))

    # Without batch statistics, every vector goes through the head, padding vectors included
    for normalization in ["none", "layer"]:
        unmasked_head = ClassificationHead(input_features=6, fully_connected=[8],
                                           fully_connected_normalization=normalization,
                                           batch_normalization_momentum=0.1, dropout_rate=0.0, number_classes=3,
                                           device=DEVICE, dtype=DTYPE)
        assert torch.equal(input=unmasked_head(features=padded_vectors, feature_mask=frame_mask),
                           other=unmasked_head(features=padded_vectors))


@pytest.mark.parametrize(argnames="normalization", argvalues=["batch"])
def test_head_keeps_the_features_of_a_hidden_layer_before_its_relu(normalization: str) -> None:
    """
    Verifies that the classification head keeps the features of the asked hidden layer as its ReLU receives them,
    after the normalization of the layer, for the decorrelation loss of the image classifier: they equal the output of
    the layers before that ReLU, keep their negative values although the in-place ReLU then runs, and the logits are
    those of the head keeping no features.

    Args:
        normalization (str): The normalization of the hidden layers, "none", "batch" or "layer".
    """
    torch.manual_seed(seed=0)
    head = ClassificationHead(input_features=6, fully_connected=[8, 5], fully_connected_normalization=normalization,
                              batch_normalization_momentum=0.1, dropout_rate=0.0, number_classes=3, device=DEVICE,
                              dtype=DTYPE)
    head.eval()
    feature_vectors = torch.randn(size=(7, 6))
    reference_logits, no_kept_features = head.compute_logits(features=feature_vectors, kept_hidden_layer=None)
    assert no_kept_features is None

    # Every hidden layer ends with its ReLU: the kept features are the output of the layers before it
    relu_indices = [index for index, layer in enumerate(head.fully_connected_block) if isinstance(layer, nn.ReLU)]
    for hidden_layer_number, relu_index in enumerate(relu_indices, start=1):
        class_logits, kept_features = head.compute_logits(features=feature_vectors,
                                                          kept_hidden_layer=hidden_layer_number)
        expected_features = head.fully_connected_block[:relu_index](input=feature_vectors)
        assert kept_features.shape == (7, [8, 5][hidden_layer_number - 1])
        assert torch.allclose(input=kept_features, other=expected_features, atol=1e-6)
        assert (kept_features < 0).any()
        assert torch.allclose(input=class_logits, other=reference_logits, atol=1e-6)


def test_unknown_normalization_is_refused() -> None:
    """
    Verifies that a misspelled normalization of the hidden layers reaching the builder without the configuration
    checks stops with the list of the valid normalizations, instead of building the layers without normalization.
    """
    with pytest.raises(expected_exception=ValueError, match="must be one of .* got 'batchnorm'"):
        build_fully_connected_block(input_dimension=12, dimensions=[8], dropout_rate=0.0, add_dropout_layers=True,
                                    normalization="batchnorm", batch_normalization_momentum=0.1, device=DEVICE,
                                    dtype=DTYPE)
