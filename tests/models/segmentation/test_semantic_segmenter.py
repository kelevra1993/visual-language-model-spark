"""
Tests of SemanticSegmenter of vision_training/models/segmentation/semantic_segmenter.py and of its decoder: the logits
at the input resolution for every decoder option and backbone pooling, the losses only with masks and only for the
weighted losses, the probabilities in evaluation mode, the ignored pixels without gradient, the weight names
independent of the dropout rate, and the architecture summary.
"""

import pytest
import torch

from tests.fixtures.configuration_fixtures import get_segmentation_model_configuration
from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.models.segmentation.segmentation_decoder import compute_skip_block_numbers
from vision_training.models.segmentation.semantic_segmenter import SemanticSegmenter


@pytest.mark.parametrize(argnames=("upsampling", "feature_map_normalization", "squeeze_and_excite", "residual_nets"),
                         argvalues=[("transposed", "batch", False, False), ("bilinear", "layer", True, True),
                                    ("transposed", "none", True, False)])
def test_logits_at_the_input_resolution(upsampling: str, feature_map_normalization: str, squeeze_and_excite: bool,
                                        residual_nets: bool) -> None:
    """
    Verifies that every decoder option gives the logits of every class at every pixel of the input, the losses being
    returned only with masks and the probabilities only in evaluation mode.

    Args:
        upsampling (str): The upsampling of the decoder levels.
        feature_map_normalization (str): The normalization of the decoder blocks.
        squeeze_and_excite (bool): Whether the decoder blocks use Squeeze-and-Excitation.
        residual_nets (bool): Whether the decoder blocks use residual shortcuts.
    """
    configuration, device, dtype = get_segmentation_model_configuration(upsampling=upsampling)
    configuration["SegmentationDecoder"].update({"feature_map_normalization": feature_map_normalization,
                                                 "modules": {"squeeze_and_excite": squeeze_and_excite,
                                                             "residual_nets": residual_nets}})
    model = SemanticSegmenter(configuration=configuration, device=device, dtype=dtype)
    images = torch.rand(size=(2, 3, 64, 64)) * 255.0
    masks = torch.randint(low=0, high=3, size=(2, 64, 64))

    # Training mode with masks: logits and both losses, no probabilities
    outputs = model(input_tensor=images, masks=masks)
    assert outputs["class_logits"].shape == (2, 3, 64, 64)
    assert sorted(outputs) == ["class_logits", "dice_loss", "segmentation_loss"]

    # Evaluation mode without masks: logits and probabilities summing to 1 at every pixel
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor=images)
    assert sorted(outputs) == ["class_logits", "class_probabilities"]
    assert torch.allclose(input=outputs["class_probabilities"].sum(dim=1), other=torch.ones(size=(2, 64, 64)))


def test_last_max_pooling_adds_a_decoder_level() -> None:
    """
    Verifies that with a max pooling after the last backbone block, its output before pooling becomes one more skip
    connection, the decoder still ending at the input resolution.
    """
    assert compute_skip_block_numbers(number_blocks=4, last_max_pooling=False) == [3, 2, 1]
    assert compute_skip_block_numbers(number_blocks=4, last_max_pooling=True) == [4, 3, 2, 1]

    configuration, device, dtype = get_segmentation_model_configuration(upsampling="transposed")
    configuration["Backbone"]["last_max_pooling"] = True
    configuration["SegmentationDecoder"]["convolutions"] = {"4": [1, 32], "3": [1, 16], "2": [1, 16], "1": [1, 8]}
    model = SemanticSegmenter(configuration=configuration, device=device, dtype=dtype)

    assert list(model.segmentation_decoder.convolution_blocks) == ["4", "3", "2", "1"]
    assert model(input_tensor=torch.rand(size=(1, 3, 64, 64)) * 255.0)["class_logits"].shape == (1, 3, 64, 64)


def test_losses_follow_their_weights_and_ignore_pixels() -> None:
    """
    Verifies that a loss of weight 0 is neither computed nor returned, that the returned loss is multiplied by its
    weight, and that the ignored pixels of the masks give no gradient to the logits.
    """
    configuration, device, dtype = get_segmentation_model_configuration(upsampling="transposed")
    configuration["SegmentationLoss"].update({"cross_entropy_weight": 2.0, "dice_weight": 0.0})
    model = SemanticSegmenter(configuration=configuration, device=device, dtype=dtype).eval()
    images = torch.rand(size=(1, 3, 64, 64)) * 255.0
    masks = torch.randint(low=0, high=3, size=(1, 64, 64))
    masks[:, :32] = IGNORED_LABEL_INDEX

    outputs = model(input_tensor=images, masks=masks)
    assert "dice_loss" not in outputs
    expected_loss = 2.0 * torch.nn.functional.cross_entropy(input=outputs["class_logits"], target=masks,
                                                            ignore_index=IGNORED_LABEL_INDEX)
    assert torch.allclose(input=outputs["segmentation_loss"], other=expected_loss)

    # The gradient of the loss with respect to the logits is 0 on the ignored half of the image
    class_logits = outputs["class_logits"]
    logits_gradient, = torch.autograd.grad(outputs=outputs["segmentation_loss"], inputs=class_logits)
    assert torch.all(logits_gradient[:, :, :32] == 0) and bool(torch.any(logits_gradient[:, :, 32:] != 0))


def test_weight_names_do_not_depend_on_the_dropout_rate() -> None:
    """
    Verifies that the dropout layer of the segmentation head is kept at a rate of 0, so that a run can change its
    dropout rate between two resumptions without renaming its weights.
    """
    configuration, device, dtype = get_segmentation_model_configuration(upsampling="bilinear")
    model_without_dropout = SemanticSegmenter(configuration=configuration, device=device, dtype=dtype)
    configuration["SegmentationDecoder"]["dropout_rate"] = 0.3
    model_with_dropout = SemanticSegmenter(configuration=configuration, device=device, dtype=dtype)

    assert list(model_without_dropout.state_dict()) == list(model_with_dropout.state_dict())


def test_summary_and_size_components(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the architecture summary lists every decoder level with its skip connection and the logits at the
    input resolution, and that the size summary covers the backbone, the decoder and the head.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    configuration, device, dtype = get_segmentation_model_configuration(upsampling="transposed")
    model = SemanticSegmenter(configuration=configuration, device=device, dtype=dtype)

    model.print_summary()
    console_output = capsys.readouterr().out
    for expected_fragment in ["Level Of Block 3", "Level Of Block 1", "64x64x8", "Class Logits      : 64x64x3"]:
        assert expected_fragment in console_output
    assert list(model.get_size_components()) == ["Backbone", "Segmentation Decoder", "Segmentation Head",
                                                 "Total Model"]
