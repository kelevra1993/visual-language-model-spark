"""
Tests of the intermediate outputs of the backbone of vision_training/models/backbones/backbone.py: the output of every
block before its pooling, read by the segmentation decoder, the outputs of the detection blocks, unchanged, and the
skip outputs read by the mask heads of the instance segmentation.
"""

import pytest
import torch

from tests.fixtures.configuration_fixtures import get_classification_model_configuration
from vision_training.models.backbones.backbone import build_backbone, split_skip_outputs


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


def test_skip_outputs_of_the_mask_heads() -> None:
    """
    Verifies the skip outputs read by the mask heads of the instance segmentation: 'skip_0' is the normalized input
    image and 'skip_<b>' the output of block b before its pooling, returned beside the unchanged detection block outputs
    and separated from them by split_skip_outputs, the mode of the segmentation decoder returning none;
    compute_skip_output_information gives their sizes and channels without moving the batch statistics, and skip blocks
    beyond the last block are refused.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    backbone = build_backbone(backbone_configuration=configuration["Backbone"], input_image_size=64, device=device,
                              dtype=dtype, detection_convolution_block_indices=[2, 3], skip_block_indices=[3, 0, 4])
    images = torch.rand(size=(2, 3, 64, 64), device=device, dtype=dtype) * 255

    # The sizes and channels come from the poolings alone, the running statistics of training mode staying as they are
    backbone.train()
    initial_state = {key: tensor.clone() for key, tensor in backbone.state_dict().items()}
    skip_information = backbone.compute_skip_output_information()
    assert all(torch.equal(input=initial_state[key], other=tensor) for key, tensor in backbone.state_dict().items())
    assert skip_information == {"skip_0": {"feature_map_size": 64, "number_channels": 3},
                                "skip_3": {"feature_map_size": 16, "number_channels": 16},
                                "skip_4": {"feature_map_size": 8, "number_channels": 32}}

    backbone.eval()
    with torch.no_grad():
        final_tensor, outputs = backbone(input_tensor=images)
        pre_pooling_final_tensor, pre_pooling_outputs = backbone(input_tensor=images, return_pre_pooling_outputs=True)
    level_outputs, skip_outputs = split_skip_outputs(output_tensor_dictionary=outputs)

    # Detection outputs unchanged, the skip outputs under their own keys, none in the mode of the segmentation decoder
    assert list(level_outputs) == ["2", "3"] and sorted(skip_outputs) == ["skip_0", "skip_3", "skip_4"]
    assert list(pre_pooling_outputs) == ["1", "2", "3", "4"]
    assert torch.equal(input=final_tensor, other=pre_pooling_final_tensor)

    # The normalized input image and the outputs of the blocks before their pooling, of the announced sizes
    assert torch.equal(input=skip_outputs["skip_0"], other=backbone.input_normalizer(input_tensor=images))
    assert torch.equal(input=skip_outputs["skip_3"], other=pre_pooling_outputs["3"])
    assert torch.equal(input=skip_outputs["skip_4"], other=pre_pooling_outputs["4"])
    for skip_key, skip_tensor in skip_outputs.items():
        feature_map_size = skip_information[skip_key]["feature_map_size"]
        assert skip_tensor.shape == (2, skip_information[skip_key]["number_channels"], feature_map_size,
                                     feature_map_size)

    # A skip block beyond the 4 blocks of the backbone is refused
    with pytest.raises(expected_exception=ValueError, match="must lie between 0"):
        build_backbone(backbone_configuration=configuration["Backbone"], input_image_size=64, device=device,
                       dtype=dtype, skip_block_indices=[5])


def test_detection_input_information_leaves_the_backbone_untouched() -> None:
    """
    Verifies that the shape probe of the detection blocks, run by the detector at construction, leaves every weight and
    batch normalization buffer as it was and keeps the training mode, and that it gives the sizes of a backbone whose
    last block receives a 1 x 1 map, which a probe of one image in training mode refuses ("Expected more than 1 value
    per channel") although training on batches of 2 works.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    backbone = build_backbone(backbone_configuration=configuration["Backbone"], input_image_size=8, device=device,
                              dtype=dtype, detection_convolution_block_indices=[3, 4]).train()
    initial_state = {key: tensor.clone() for key, tensor in backbone.state_dict().items()}

    # Blocks 3 and 4 receive 2 x 2 and 1 x 1 maps from the 8 pixel input, their poolings giving 1 x 1 outputs
    detection_information = backbone.compute_detection_input_information()
    assert detection_information == {"3": {"feature_map_size": 1, "number_channels": 16},
                                     "4": {"feature_map_size": 1, "number_channels": 32}}

    # Running statistics and batch counters unchanged, the training mode of the backbone and its blocks restored
    assert all(torch.equal(input=initial_state[key], other=tensor) for key, tensor in backbone.state_dict().items())
    assert backbone.training and all(module.training for module in backbone.modules())
