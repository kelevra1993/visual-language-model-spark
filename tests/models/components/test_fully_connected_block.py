"""
Tests of the hidden fully connected layers of vision_training/models/components/fully_connected_block.py and of their
batch normalization in the classification head: the layers and weight names with and without normalization, the
normalization of vectors of any leading shape, the padding vectors left out of the batch statistics, and the
momentum of the batch normalization layers of the backbone and of the head built from the configuration.
"""

import pytest
import torch

from torch import nn

from tests.fixtures.configuration_fixtures import get_classification_model_configuration
from vision_training.models.classification.classification_head import ClassificationHead
from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.components.fully_connected_block import (FeatureBatchNormalization,
                                                                     build_fully_connected_block)
from vision_training.models.components.sequence_normalization import SequenceBatchNormalization

DEVICE = torch.device(device="cpu")
DTYPE = torch.float32


@pytest.mark.parametrize(argnames=("batch_normalization", "expected_layer_types"), argvalues=[
    (False, [nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU]),
    (True, [nn.Dropout, nn.Linear, FeatureBatchNormalization, nn.ReLU, nn.Dropout, nn.Linear,
            FeatureBatchNormalization, nn.ReLU])])
def test_fully_connected_layers(batch_normalization: bool, expected_layer_types: list) -> None:
    """
    Verifies the layers of the hidden fully connected block: dropout, linear and ReLU per layer without normalization,
    so that the classification heads trained before the normalization keep the names of their weights, and a batch
    normalization at the configured momentum between every linear layer and its ReLU with it.

    Args:
        batch_normalization (bool): Whether the hidden layers are normalized.
        expected_layer_types (list): The expected type of every layer of the block, in order.
    """
    fully_connected_block = build_fully_connected_block(input_dimension=12, dimensions=[8, 4], dropout_rate=0.2,
                                                        add_dropout_layers=True,
                                                        batch_normalization=batch_normalization,
                                                        batch_normalization_momentum=0.01, device=DEVICE, dtype=DTYPE)

    assert [type(layer) for layer in fully_connected_block] == expected_layer_types
    assert all(layer.momentum == 0.01 for layer in fully_connected_block
               if isinstance(layer, FeatureBatchNormalization))

    # Without normalization, the weights keep the names of the block written before the normalization existed
    if not batch_normalization:
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
    Verifies that a classification head with normalized hidden layers classifies the real vectors of the windows of
    the sequence labeller with the statistics of the real vectors only, whatever the values of the padding vectors,
    and gives the padding vectors zero logits; without normalization, the mask changes nothing.
    """
    torch.manual_seed(seed=0)
    head = ClassificationHead(input_features=6, fully_connected=[8], fully_connected_normalization=True,
                              batch_normalization_momentum=0.1, dropout_rate=0.0, number_classes=3, device=DEVICE,
                              dtype=DTYPE)
    frame_vectors = torch.randn(size=(2, 4, 6))
    frame_mask = torch.tensor(data=[[True, True, True, False], [True, True, False, False]])

    # Padding vectors of huge values, which would dominate the batch statistics if they were included
    padded_vectors = frame_vectors.clone()
    padded_vectors[~frame_mask] = 1000.0
    class_logits = head(features=padded_vectors, feature_mask=frame_mask)
    real_logits = head.compute_logits(features=frame_vectors[frame_mask])

    assert class_logits.shape == (2, 4, 3)
    assert torch.allclose(input=class_logits[frame_mask], other=real_logits, atol=1e-6)
    assert torch.equal(input=class_logits[~frame_mask], other=torch.zeros(size=(3, 3)))

    # Without normalization, every vector goes through the head, padding vectors included
    plain_head = ClassificationHead(input_features=6, fully_connected=[8], fully_connected_normalization=False,
                                    batch_normalization_momentum=0.1, dropout_rate=0.0, number_classes=3,
                                    device=DEVICE, dtype=DTYPE)
    assert torch.equal(input=plain_head(features=padded_vectors, feature_mask=frame_mask),
                       other=plain_head(features=padded_vectors))


@pytest.mark.parametrize(argnames="feature_map_normalization", argvalues=["batch", "sequence"])
def test_batch_normalization_momentums_reach_every_layer(feature_map_normalization: str) -> None:
    """
    Verifies that the image classifier built from the configuration gives every batch (or sequence) normalization
    layer of its backbone the momentum of 'Backbone.normalization' and every normalization layer of its head the
    momentum of 'ClassificationHead', two different values.

    Args:
        feature_map_normalization (str): The feature map normalization of the backbone.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    configuration["Backbone"]["normalization"].update({"feature_map_normalization": feature_map_normalization,
                                                       "batch_normalization_momentum": 0.01})
    configuration["ClassificationHead"].update({"fully_connected_normalization": True,
                                                "batch_normalization_momentum": 0.05})
    classifier = ImageClassifier(configuration=configuration, verbose=False, device=device, dtype=dtype)

    backbone_layer_type = nn.BatchNorm2d if feature_map_normalization == "batch" else SequenceBatchNormalization
    backbone_momentums = [module.momentum for module in classifier.backbone.modules()
                          if isinstance(module, backbone_layer_type)]
    head_momentums = [module.momentum for module in classifier.classification_head.modules()
                      if isinstance(module, FeatureBatchNormalization)]
    assert backbone_momentums and set(backbone_momentums) == {0.01}
    assert head_momentums == [0.05]
