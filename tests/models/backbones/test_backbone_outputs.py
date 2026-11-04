"""
Tests of the intermediate outputs of the backbone of vision_training/models/backbones/backbone.py: the output of every
block before its pooling, read by the segmentation decoder, and the outputs of the detection blocks, unchanged.
"""

import pytest
import torch

from tests.fixtures.configuration_fixtures import get_classification_model_configuration
from vision_training.models.backbones.backbone import build_backbone


@pytest.mark.parametrize(argnames=("last_max_pooling", "expected_final_size"), argvalues=[(True, 4)])
def test_pre_pooling_outputs_of_every_block(last_max_pooling: bool, expected_final_size: int) -> None:
    """
    Verifies the output of every block before its pooling: block b at 1 / 2**(b-1) of the 64 pixel input resolution
    with the channels of the block, the pooled output of a block being its pre-pooling output pooled, and the final
    feature map pooled once more with the last max pooling.

    Args:
        last_max_pooling (bool): Whether the last block ends with a max pooling.
        expected_final_size (int): The expected height and width of the final feature map.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    configuration["Backbone"]["last_max_pooling"] = last_max_pooling
    backbone = build_backbone(backbone_configuration=configuration["Backbone"], input_image_size=64, device=device,
                              dtype=dtype).eval()
    images = torch.rand(size=(2, 3, 64, 64), device=device, dtype=dtype) * 255

    with torch.no_grad():
        final_tensor, pre_pooling_outputs = backbone(input_tensor=images, return_pre_pooling_outputs=True)

    # Block b keeps the resolution of its input, the channels being those of its configuration
    assert list(pre_pooling_outputs) == ["1", "2", "3", "4"]
    for block_number, block_channels in [(1, 8), (2, 16), (3, 16), (4, 32)]:
        block_size = 64 // 2 ** (block_number - 1)
        assert pre_pooling_outputs[str(block_number)].shape == (2, block_channels, block_size, block_size)
    assert final_tensor.shape == (2, 32, expected_final_size, expected_final_size)

    # The input of block b + 1 is the pooled pre-pooling output of block b
    pooled_first_block = torch.nn.functional.max_pool2d(input=pre_pooling_outputs["1"], kernel_size=2, stride=2)
    second_block_input = backbone.convolution_blocks[0](input_tensor=backbone.input_normalizer(input_tensor=images))
    assert torch.equal(input=pooled_first_block, other=second_block_input)


def test_default_outputs_are_the_detection_blocks() -> None:
    """
    Verifies that by default the backbone keeps the outputs of its detection blocks after their pooling, as before the
    pre-pooling outputs existed, and the same final feature map in both modes.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    backbone = build_backbone(backbone_configuration=configuration["Backbone"], input_image_size=64, device=device,
                              dtype=dtype, detection_convolution_block_indices=[2, 3]).eval()
    images = torch.rand(size=(2, 3, 64, 64), device=device, dtype=dtype) * 255

    with torch.no_grad():
        final_tensor, detection_outputs = backbone(input_tensor=images)
        pre_pooling_final_tensor, _ = backbone(input_tensor=images, return_pre_pooling_outputs=True)

    assert list(detection_outputs) == ["2", "3"]
    assert detection_outputs["2"].shape == (2, 16, 16, 16) and detection_outputs["3"].shape == (2, 16, 8, 8)
    assert torch.equal(input=final_tensor, other=pre_pooling_final_tensor)
