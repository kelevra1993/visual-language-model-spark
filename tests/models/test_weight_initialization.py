"""
Tests of the starting weights of vision_training/models/weight_initialization.py, applied by the trainers to the task
models: "default" leaves the PyTorch initialisation, "he" draws the layers followed by a ReLU from the He normal
distribution and leaves the other layers unchanged.
"""

import math
import pytest
import torch

from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.components.convolution_block import ConvolutionBlock
from vision_training.models.weight_initialization import initialise_weights
from tests.fixtures.configuration_fixtures import get_classification_model_configuration


def build_classifier() -> ImageClassifier:
    """
    Builds the small image classifier of the classification model tests, with Squeeze-and-Excitation, residual
    shortcuts and one hidden fully connected layer, so that every kind of He layer and of unchanged layer is present.

    Returns:
        ImageClassifier: The classifier with the PyTorch initialisation.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    return ImageClassifier(configuration=configuration, device=device, dtype=dtype)


def test_default_initialisation_leaves_the_model_unchanged() -> None:
    """
    Verifies that "default" keeps every weight of the PyTorch initialisation, so that the runs written before the
    setting keep their behaviour.
    """
    classifier = build_classifier()
    initial_state = {name: tensor.clone() for name, tensor in classifier.state_dict().items()}

    assert initialise_weights(model=classifier, weight_initialization="default") == 0
    for name, tensor in classifier.state_dict().items():
        assert torch.equal(input=tensor, other=initial_state[name]), name


def test_he_initialisation_draws_the_layers_followed_by_a_relu() -> None:
    """
    Verifies that "he" gives the He initialisation to the convolutions of the backbone blocks, the hidden fully
    connected layer of the head and the reduce layers of Squeeze-and-Excitation, with zero biases, and leaves the class
    logits layer, the expand layers of Squeeze-and-Excitation and the normalization layers unchanged.
    """
    classifier = build_classifier()
    initial_state = {name: tensor.clone() for name, tensor in classifier.state_dict().items()}
    number_convolutions = sum(isinstance(module, torch.nn.Conv2d) for module in classifier.backbone.modules())
    number_reduce_layers = sum(name.endswith("first_adaptive_recalibrator.weight")
                               for name in classifier.state_dict())

    # Convolutions, reduce layers and the single hidden layer of the head
    assert initialise_weights(model=classifier, weight_initialization="he") == (number_convolutions +
                                                                                 number_reduce_layers + 1)

    changed_names = {name for name, tensor in classifier.state_dict().items()
                     if not torch.equal(input=tensor, other=initial_state[name])}
    assert all(name.startswith(("backbone.", "classification_head.fully_connected_block."))
               for name in changed_names)
    assert not any("second_adaptive_recalibrator" in name for name in changed_names)
    assert not any(name.startswith("classification_head.classifier.") for name in changed_names)

    # Zero biases of the He layers, the normalization layers keeping their scale of 1
    for module in classifier.backbone.modules():
        if isinstance(module, torch.nn.Conv2d):
            assert torch.count_nonzero(input=module.bias) == 0
        if isinstance(module, torch.nn.BatchNorm2d):
            assert torch.equal(input=module.weight, other=torch.ones_like(input=module.weight))


def test_he_initialisation_standard_deviation() -> None:
    """
    Verifies the scale of the He normal initialisation on a wide convolution block, whose 36,864 weights per layer give
    a precise estimate: a standard deviation of sqrt(2 / fan_in), 3 x 3 x 64 inputs per output, about 2.45 times the
    one of the PyTorch initialisation, sqrt(1 / (3 x fan_in)).
    """
    torch.manual_seed(seed=0)
    convolution_block = ConvolutionBlock(input_channels=64, output_channels=64, bias=True, number_layers=2,
                                         kernel_size=3, stride=1, padding=1, feature_map_normalization="batch",
                                         batch_normalization_momentum=0.1, activation=True, dropout_rate=0.0,
                                         add_pooling=False, device=torch.device(device="cpu"), dtype=torch.float32,
                                         squeeze_and_excite=False, residual=True, pre_activation=True,
                                         inception=False)
    fan_in = 3 * 3 * 64
    default_standard_deviation = convolution_block.block[2].weight.std().item()
    assert default_standard_deviation == pytest.approx(expected=math.sqrt(1 / (3 * fan_in)), rel=0.03)

    assert initialise_weights(model=convolution_block, weight_initialization="he") == 2
    for module in convolution_block.block:
        if isinstance(module, torch.nn.Conv2d):
            assert module.weight.std().item() == pytest.approx(expected=math.sqrt(2 / fan_in), rel=0.03)


def test_unknown_initialisation_is_refused() -> None:
    """
    Verifies that an initialisation other than "default" and "he" is refused.
    """
    with pytest.raises(expected_exception=ValueError):
        initialise_weights(model=build_classifier(), weight_initialization="xavier")
