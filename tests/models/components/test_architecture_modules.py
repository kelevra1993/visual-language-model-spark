"""
Tests of the building blocks of vision_training/models/components and of the backbone: squeeze and excitation, inception
convolution, input normalization strategies, convolution block keys, residual shortcuts, backbone shapes and receptive
field.
"""

import pytest
import torch

from vision_training.models.backbones.backbone import Backbone
from vision_training.models.components.convolution_block import ConvolutionBlock
from vision_training.models.components.inception import InceptionConvolution
from vision_training.models.components.input_normalization import InputNormalizer
from vision_training.models.components.squeeze_and_excitation import SqueezeAndExcitation

DEVICE = torch.device(device="cpu")
DTYPE = torch.float32

# Complete Backbone modules configuration with every module disabled, as every key must be explicitly set
PLAIN_MODULES = {"squeeze_and_excite": False, "residual_nets": False, "inception": False, "inception_starting_block": 3}


def test_squeeze_and_excitation_rescales_channels() -> None:
    """
    Verifies that Squeeze-and-Excitation keeps the feature map shape and only rescales each channel by a
    factor in (0, 1), which is the contract relied upon by the ConvolutionBlock layers of the backbone.
    """
    torch.manual_seed(seed=0)
    squeeze_and_excitation_module = SqueezeAndExcitation(channels=16, device=DEVICE, dtype=DTYPE)
    input_tensor = torch.rand(size=(2, 16, 8, 8)) + 0.1

    output_tensor = squeeze_and_excitation_module(input_tensor=input_tensor)

    # The recalibration keeps the shape and applies one scale per channel, identical at every spatial position
    assert output_tensor.shape == input_tensor.shape
    channel_scales = output_tensor / input_tensor
    assert torch.allclose(input=channel_scales, other=channel_scales[:, :, :1, :1].expand_as(channel_scales), atol=1e-6)
    assert torch.all(channel_scales > 0.0) and torch.all(channel_scales < 1.0)


def test_inception_output_channels_and_validation() -> None:
    """
    Verifies that the Inception module preserves the spatial size, outputs the requested number of channels,
    and rejects channel counts that cannot be split across its four branches.
    """
    inception_module = InceptionConvolution(input_channels=3, output_channels=32, bias=True, device=DEVICE, dtype=DTYPE)
    output_tensor = inception_module(input_tensor=torch.rand(size=(2, 3, 16, 16)))
    assert list(output_tensor.shape) == [2, 32, 16, 16]

    # Output channels must be multiples of 8
    with pytest.raises(expected_exception=ValueError):
        InceptionConvolution(input_channels=3, output_channels=12, bias=True, device=DEVICE, dtype=DTYPE)


def test_input_normalization_strategies() -> None:
    """
    Verifies the two input normalization strategies applied at the entrance of the Backbone:
    range normalization to [-0.5, 0.5] and per-sample standardization to zero mean and unit variance.
    """
    torch.manual_seed(seed=0)
    input_tensor = torch.randint(low=0, high=256, size=(3, 3, 32, 32)).to(dtype=DTYPE)

    # Range normalization maps 0 to -0.5 and 255 to 0.5
    range_normalizer = InputNormalizer(input_normalization=True, per_sample_normalization=False)
    assert torch.allclose(input=range_normalizer(input_tensor=input_tensor), other=input_tensor / 255.0 - 0.5)

    # Per-sample standardization gives every image a zero mean and a unit standard deviation
    per_sample_normalizer = InputNormalizer(input_normalization=True, per_sample_normalization=True)
    standardized_tensor = per_sample_normalizer(input_tensor=input_tensor)
    assert torch.allclose(input=standardized_tensor.mean(dim=(1, 2, 3)), other=torch.zeros(size=(3,)), atol=1e-5)
    assert torch.allclose(input=standardized_tensor.std(dim=(1, 2, 3), unbiased=False), other=torch.ones(size=(3,)),
                          atol=1e-4)

    # A uniform image is lower-bounded by 1/sqrt(N) instead of dividing by a zero deviation
    uniform_tensor = torch.full(size=(1, 3, 4, 4), fill_value=7.0)
    assert torch.all(torch.isfinite(per_sample_normalizer(input_tensor=uniform_tensor)))

    # Disabled normalization leaves the images untouched
    identity_normalizer = InputNormalizer(input_normalization=False, per_sample_normalization=False)
    assert torch.equal(input=identity_normalizer(input_tensor=input_tensor), other=input_tensor)


def test_plain_convolution_block_state_dictionary_keys() -> None:
    """
    Verifies that the plain ConvolutionBlock keeps the state_dict keys of the historical nn.Sequential
    implementation, so that checkpoints trained before the architecture modules existed can still be loaded.
    """
    convolution_block = ConvolutionBlock(input_channels=3, output_channels=16, bias=True, number_layers=2,
                                         kernel_size=3, stride=1, padding=1, feature_map_normalization="batch",
                                         batch_normalization_momentum=0.1, activation=True, dropout_rate=0.0,
                                         add_pooling=True,
                                         device=DEVICE, dtype=DTYPE, squeeze_and_excite=False, residual=False,
                                         inception=False)

    # Layout per layer: convolution (0 / 3), batch normalization (1 / 4), activation (2 / 5), then pooling (6)
    expected_keys = ["block.0.weight", "block.0.bias",
                     "block.1.weight", "block.1.bias", "block.1.running_mean", "block.1.running_var",
                     "block.1.num_batches_tracked",
                     "block.3.weight", "block.3.bias",
                     "block.4.weight", "block.4.bias", "block.4.running_mean", "block.4.running_var",
                     "block.4.num_batches_tracked"]
    assert list(convolution_block.state_dict().keys()) == expected_keys


@pytest.mark.parametrize(argnames="squeeze_and_excite", argvalues=[False, True])
def test_residual_shortcut_placement(squeeze_and_excite: bool) -> None:
    """
    Verifies the residual shortcut pairing: with equal input and output channels the block
    input is added to the output of the second layer, before its activation. Zeroing every convolution makes both
    branch layers output zero, so the block output must be the ReLU of the block input: the input itself where it is
    positive, which an addition after the activation would leave negative. With Squeeze-and-Excitation, whose zeroed
    gates are 0.5, the recalibration comes before the addition and rescales the zero branch only, so the shortcut
    must not be halved.

    Args:
        squeeze_and_excite (bool): Whether the layers recalibrate their channels with Squeeze-and-Excitation.
    """
    convolution_block = ConvolutionBlock(input_channels=8, output_channels=8, bias=True, number_layers=2,
                                         kernel_size=3, stride=1, padding=1, feature_map_normalization="none",
                                         batch_normalization_momentum=0.1, activation=True, dropout_rate=0.0,
                                         add_pooling=False,
                                         device=DEVICE, dtype=DTYPE, squeeze_and_excite=squeeze_and_excite,
                                         residual=True, inception=False)

    # Zero every parameter so that each convolution outputs 0, the second layer then giving ReLU(0 + input)
    with torch.no_grad():
        for parameter in convolution_block.parameters():
            parameter.zero_()

    input_tensor = torch.randn(size=(2, 8, 6, 6))
    assert torch.equal(input=convolution_block(input_tensor=input_tensor), other=torch.relu(input=input_tensor))

    # Residual shortcuts require the spatial size to be preserved
    with pytest.raises(expected_exception=ValueError):
        ConvolutionBlock(input_channels=8, output_channels=8, bias=True, number_layers=2, kernel_size=3, stride=2,
                         padding=1, feature_map_normalization="none", batch_normalization_momentum=0.1,
                         activation=True, dropout_rate=0.0,
                         add_pooling=False, device=DEVICE, dtype=DTYPE, squeeze_and_excite=False, residual=True,
                         inception=False)


@pytest.mark.parametrize(argnames="modules", argvalues=[{**PLAIN_MODULES, "squeeze_and_excite": True},
                                                        {**PLAIN_MODULES, "residual_nets": True},
                                                        {**PLAIN_MODULES, "inception": True}])
def test_backbone_modules_forward_shapes(modules: dict) -> None:
    """
    Verifies that each architecture module keeps the Backbone feature map shapes identical to the plain backbone,
    so the downstream Region Proposal and Detection heads are unaffected by the module choice.

    Args:
        modules (dict): The Backbone modules configuration to test.
    """
    backbone_arguments = {"input_channels": 3, "convolutions": {"1": [1, 16], "2": [2, 32], "3": [3, 32], "4": [2, 64]},
                          "last_max_pooling": False,
                          "normalization": {"feature_map_normalization": "batch", "batch_normalization_momentum": 0.1,
                                            "input_normalization": True, "per_sample_normalization": True},
                          "detection_convolution_block_indices": [3, 4], "input_image_size": 64,
                          "device": DEVICE, "dtype": DTYPE}

    plain_backbone = Backbone(modules=PLAIN_MODULES, **backbone_arguments)
    module_backbone = Backbone(modules=modules, **backbone_arguments)

    input_tensor = torch.randint(low=0, high=256, size=(2, 3, 64, 64)).to(dtype=DTYPE)
    plain_output, plain_dictionary = plain_backbone(input_tensor=input_tensor)
    module_output, module_dictionary = module_backbone(input_tensor=input_tensor)

    assert module_output.shape == plain_output.shape
    for block_index, feature_map in plain_dictionary.items():
        assert module_dictionary[block_index].shape == feature_map.shape


def test_backbone_receptive_field() -> None:
    """
    Verifies the receptive field recursion on the backbone of the example configuration (1024 pixel input,
    pooling after every block but the last), whose detection blocks 5, 6 and 7 are expected to observe
    250, 666 and 1434 input pixels respectively (hand computed).
    """
    convolutions = {"1": [1, 16], "2": [2, 32], "3": [2, 32], "4": [4, 32], "5": [4, 64], "6": [6, 64], "7": [6, 64]}
    backbone = Backbone(input_channels=3, convolutions=convolutions, modules=PLAIN_MODULES, last_max_pooling=False,
                        normalization={"feature_map_normalization": "batch", "batch_normalization_momentum": 0.1,
                                       "input_normalization": False, "per_sample_normalization": False},
                        detection_convolution_block_indices=[5, 6, 7], input_image_size=1024,
                        device=DEVICE, dtype=DTYPE)

    receptive_field_information = backbone.compute_receptive_field_information()
    detection_receptive_fields = {layer["block_number"]: layer["receptive_field"]
                                  for layer in receptive_field_information if layer["is_detection_block_output"]}

    assert detection_receptive_fields == {5: 250, 6: 666, 7: 1434}
