"""
Tests of SemanticSegmenter of vision_training/models/segmentation/semantic_segmenter.py and of its decoder: the logits
at the input resolution, the losses only with masks and only for the weighted losses, the probabilities in evaluation
mode, the ignored pixels without gradient, and the deep supervision heads of the intermediate decoder levels with the
Lovász and boundary losses.
"""

import pytest
import torch

from tests.fixtures.configuration_fixtures import get_segmentation_model_configuration
from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.losses.segmentation_losses import compute_boundary_loss, compute_lovasz_softmax_loss
from vision_training.models.segmentation.semantic_segmenter import SemanticSegmenter
from vision_training.utilities.mask_morphology import compute_edge_distance_weights


@pytest.mark.parametrize(argnames=("upsampling", "feature_map_normalization", "squeeze_and_excite", "residual_nets"),
                         argvalues=[("bilinear", "layer", True, True)])
def test_logits_at_the_input_resolution(upsampling: str, feature_map_normalization: str, squeeze_and_excite: bool,
                                        residual_nets: bool) -> None:
    """
    Verifies that the decoder gives the logits of every class at every pixel of the input, the losses being
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


def test_deep_supervision_heads_add_the_losses_of_the_intermediate_levels() -> None:
    """
    Verifies that the deep supervision heads give the decoder levels of their keys the same losses as the final
    logits, the Lovász and boundary losses included, upsampled to the input size and weighted by level, that the
    gradient of that loss reaches the heads, and that the heads are left out without masks, with the decoder output
    unchanged by the return of the level outputs.
    """
    configuration, device, dtype = get_segmentation_model_configuration(upsampling="bilinear")
    configuration["SegmentationDecoder"]["deep_supervision_weights"] = {"3": 0.5, "2": 0.25}
    configuration["SegmentationLoss"].update({"lovasz_weight": 2.0, "boundary_weight": 3.0, "boundary_distance": 2})
    model = SemanticSegmenter(configuration=configuration, device=device, dtype=dtype).eval()
    images = torch.rand(size=(2, 3, 64, 64)) * 255.0
    masks = torch.randint(low=0, high=3, size=(2, 64, 64))

    outputs = model(input_tensor=images, masks=masks)
    assert sorted(outputs) == ["boundary_loss", "class_logits", "class_probabilities", "deep_supervision_loss",
                               "dice_loss", "lovasz_loss", "segmentation_loss"]

    # The Lovász and boundary losses of the final logits are multiplied by their weights
    edge_weights = compute_edge_distance_weights(masks=masks, maximum_distance=2)
    assert torch.allclose(input=outputs["lovasz_loss"], other=2.0 * compute_lovasz_softmax_loss(
        class_logits=outputs["class_logits"], masks=masks))
    assert torch.allclose(input=outputs["boundary_loss"], other=3.0 * compute_boundary_loss(
        class_logits=outputs["class_logits"], masks=masks, edge_weights=edge_weights))

    # The decoder levels of blocks 3 and 2 are at a quarter and half of the input size, their losses recomputed by hand
    final_feature_map, skip_tensors = model.backbone(input_tensor=images, return_pre_pooling_outputs=True)
    decoded_feature_map, level_outputs = model.segmentation_decoder(
        bottleneck_tensor=final_feature_map, skip_tensors=skip_tensors, return_level_outputs=True)
    assert torch.equal(input=decoded_feature_map, other=model.segmentation_decoder(
        bottleneck_tensor=final_feature_map, skip_tensors=skip_tensors))
    assert {level_key: tuple(level_output.shape[-2:]) for level_key, level_output in level_outputs.items()} == {
        "3": (16, 16), "2": (32, 32), "1": (64, 64)}
    expected_loss = 0.0
    for level_key, level_weight in [("3", 0.5), ("2", 0.25)]:
        level_logits = torch.nn.functional.interpolate(
            input=model.deep_supervision_heads[level_key](level_outputs[level_key]), size=(64, 64), mode="bilinear",
            align_corners=False)
        level_losses = model.compute_mask_losses(class_logits=level_logits, masks=masks, edge_weights=edge_weights)
        assert sorted(level_losses) == ["boundary_loss", "dice_loss", "lovasz_loss", "segmentation_loss"]
        expected_loss = expected_loss + level_weight * sum(level_losses.values())
    assert torch.allclose(input=outputs["deep_supervision_loss"], other=expected_loss)

    # The deep supervision loss trains the heads, which take no part in the outputs without masks
    head_gradient, = torch.autograd.grad(outputs=outputs["deep_supervision_loss"],
                                         inputs=model.deep_supervision_heads["3"].weight)
    assert bool(torch.any(head_gradient != 0))
    with torch.no_grad():
        assert sorted(model(input_tensor=images)) == ["class_logits", "class_probabilities"]
