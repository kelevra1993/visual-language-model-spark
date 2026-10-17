"""
Tests of vision_training/models/keypoints/keypoint_detector.py, the model of the keypoint detection task: the heatmap
channel table (consecutive channels per class in increasing class order, -1 for the padding positions and the classes
without keypoints, the 'class.keypoint' names, an empty definition refused), the decoding of the heatmaps (a Gaussian
target of the crop transform decodes back within a quarter of a cell of its keypoint, a maximum on the border is not
shifted, the scores are the maxima), and the forward pass of the KeypointDetector on the keypoint fixture (the outputs
and their shapes, the selection of the channels of the class of every crop, and the heatmap loss given only with the
targets, which sends no gradient to the channels of the other classes nor to the untrained keypoints).
"""

import pytest
import torch

from functools import partial
from torch import nn
from typing import List, Tuple

from vision_training.data.keypoints.crop_transform import build_heatmap_targets
from vision_training.losses.keypoint_losses import compute_keypoint_heatmap_loss
from vision_training.models.keypoints.keypoint_detector import (HEATMAP_STRIDE, KeypointDetector,
                                                                build_keypoint_channel_table, decode_heatmaps)
from tests.fixtures.configuration_fixtures import get_keypoint_model_configuration

# Crop size of the keypoint fixture and the size of its heatmaps
CROP_HEIGHT, CROP_WIDTH = 64, 48
HEATMAP_HEIGHT, HEATMAP_WIDTH = CROP_HEIGHT // HEATMAP_STRIDE, CROP_WIDTH // HEATMAP_STRIDE


def retain_heatmap_gradient(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor,
                            retained_heatmaps: List[torch.Tensor]) -> None:
    """
    Forward hook of the heatmap head keeping the gradient of the heatmaps it predicts, so that the test of the loss
    sees which channels receive a gradient from it.

    Args:
        module (nn.Module): The heatmap head.
        inputs (Tuple[torch.Tensor, ...]): The inputs of the head.
        output (torch.Tensor): The [B, C, h, w] predicted heatmaps.
        retained_heatmaps (List[torch.Tensor]): The list receiving the heatmaps.
    """
    output.retain_grad()
    retained_heatmaps.append(output)


def test_channel_table_gives_consecutive_channels_per_class() -> None:
    """
    Verifies that the channels of every class follow each other in the order of its keypoints, the classes in
    increasing order whatever the order of the definition, that the padding positions, the background and a class
    without definition read -1, that every channel is named 'class.keypoint', and that an empty definition is refused.
    """
    keypoint_categories = {3: {"name": "rectangle", "keypoints": ["top_left", "top_right", "bottom_right"]},
                           1: {"name": "circle", "keypoints": ["centre"]}}
    channel_table, channel_names = build_keypoint_channel_table(keypoint_categories=keypoint_categories)
    assert channel_table.dtype == torch.int64
    assert channel_table.tolist() == [[-1, -1, -1], [0, -1, -1], [-1, -1, -1], [1, 2, 3]]
    assert channel_names == ["circle.centre", "rectangle.top_left", "rectangle.top_right", "rectangle.bottom_right"]
    with pytest.raises(ValueError, match="No class defines keypoints"):
        build_keypoint_channel_table(keypoint_categories={})


def test_heatmap_targets_decode_back_to_their_keypoints() -> None:
    """
    Verifies that the Gaussian targets of random keypoints inside the crop decode back within a quarter of a cell of
    their keypoints, exactly for a keypoint at a cell centre and for a keypoint a quarter of a cell right of it, that
    the scores are the maxima of the heatmaps, and that a maximum on the border of the heatmap is not shifted.
    """
    generator = torch.Generator().manual_seed(0)
    random_keypoints = torch.rand(size=(2, 30, 2), generator=generator) * torch.tensor(data=[CROP_WIDTH - 8.0,
                                                                                            CROP_HEIGHT - 8.0]) + 4.0
    exact_keypoints = torch.tensor(data=[[[18.0, 26.0], [19.0, 26.0]]]).expand(2, 2, 2)
    crop_keypoints = torch.cat(tensors=[random_keypoints, exact_keypoints], dim=1)
    heatmaps, _ = build_heatmap_targets(crop_keypoints=crop_keypoints, visibilities=torch.full(size=(2, 32),
                                                                                               fill_value=2),
                                        crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH, heatmap_stride=HEATMAP_STRIDE,
                                        heatmap_sigma=2.0)
    keypoints, keypoint_scores = decode_heatmaps(heatmaps=heatmaps, heatmap_stride=HEATMAP_STRIDE)
    assert keypoints.shape == (2, 32, 2) and keypoints.dtype == keypoint_scores.dtype == torch.float32
    assert float((keypoints - crop_keypoints).abs().max()) <= HEATMAP_STRIDE / 4.0 + 1e-5
    assert torch.equal(input=keypoints[:, 30:], other=exact_keypoints)
    assert torch.equal(input=keypoint_scores, other=heatmaps.flatten(start_dim=2).max(dim=2).values)

    # A maximum in the top left cell has no neighbour above nor on its left and stays at the cell centre
    border_heatmaps = torch.zeros(size=(1, 1, HEATMAP_HEIGHT, HEATMAP_WIDTH))
    border_heatmaps[0, 0, 0, 0], border_heatmaps[0, 0, 0, 1], border_heatmaps[0, 0, 1, 0] = 0.9, 0.5, 0.4
    border_keypoints, border_scores = decode_heatmaps(heatmaps=border_heatmaps, heatmap_stride=HEATMAP_STRIDE)
    assert border_keypoints.tolist() == [[[2.0, 2.0]]] and border_scores.tolist() == [[pytest.approx(0.9)]]


def test_detector_selects_the_channels_of_the_class_of_every_crop() -> None:
    """
    Verifies the outputs of the keypoint detector of the fixture (7 channels: the 3 of the triangle, then the 4 of the
    rectangle): without labels only the heatmaps, keypoints and scores of every channel, with the labels the keypoints
    and scores of the channels of the class of every crop, the padding position of the triangle being zero, and the
    loss only with the targets.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_keypoint_model_configuration()
    model = KeypointDetector(configuration=configuration, device=device, dtype=dtype).eval()
    assert model.channel_table.tolist() == [[-1, -1, -1, -1], [0, 1, 2, -1], [3, 4, 5, 6]]
    crops = torch.rand(size=(2, 3, CROP_HEIGHT, CROP_WIDTH)) * 255.0

    # Every channel of every crop is decoded
    with torch.no_grad():
        outputs = model(input_tensor=crops)
    assert set(outputs) == {"heatmaps", "keypoints", "keypoint_scores"}
    assert outputs["heatmaps"].shape == (2, 7, HEATMAP_HEIGHT, HEATMAP_WIDTH)
    assert outputs["keypoints"].shape == (2, 7, 2) and outputs["keypoint_scores"].shape == (2, 7)

    # The triangle crop reads channels 0 to 2 with a zero padding, the rectangle crop channels 3 to 6
    with torch.no_grad():
        labelled_outputs = model(input_tensor=crops, labels=torch.tensor(data=[1, 2]))
    assert "keypoint_loss" not in labelled_outputs
    assert torch.equal(input=labelled_outputs["category_keypoints"][0, :3], other=outputs["keypoints"][0, :3])
    assert torch.equal(input=labelled_outputs["category_keypoints"][1], other=outputs["keypoints"][1, 3:])
    assert torch.equal(input=labelled_outputs["category_keypoint_scores"][1], other=outputs["keypoint_scores"][1, 3:])
    assert labelled_outputs["category_keypoints"][0, 3].tolist() == [0.0, 0.0]
    assert float(labelled_outputs["category_keypoint_scores"][0, 3]) == 0.0


def test_heatmap_loss_trains_only_the_channels_of_the_class() -> None:
    """
    Verifies that the heatmap loss of a triangle crop and a rectangle crop is the loss of the channels of their
    classes, and that its gradient reaches the trained keypoints only: neither the channels of the other class, nor the
    unlabelled keypoint of the rectangle, nor the padding position of the triangle, given a weight of 1 on purpose.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_keypoint_model_configuration()
    model = KeypointDetector(configuration=configuration, device=device, dtype=dtype).train()
    crops = torch.rand(size=(2, 3, CROP_HEIGHT, CROP_WIDTH)) * 255.0
    labels = torch.tensor(data=[1, 2])

    # Targets of 4 keypoint positions, the rectangle keypoint 2 untrained and the triangle padding weighted 1
    target_keypoints = torch.tensor(data=[[[24.0, 10.0], [10.0, 50.0], [38.0, 50.0], [0.0, 0.0]],
                                          [[8.0, 20.0], [40.0, 20.0], [40.0, 44.0], [8.0, 44.0]]])
    target_heatmaps, _ = build_heatmap_targets(crop_keypoints=target_keypoints, visibilities=torch.full(size=(2, 4),
                                                                                                        fill_value=2),
                                               crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH,
                                               heatmap_stride=HEATMAP_STRIDE, heatmap_sigma=2.0)
    heatmap_weights = torch.tensor(data=[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])

    # The heatmaps of the head keep their gradient to see which channels the loss reaches
    retained_heatmaps = []
    hook_handle = model.keypoint_head.register_forward_hook(
        hook=partial(retain_heatmap_gradient, retained_heatmaps=retained_heatmaps))
    outputs = model(input_tensor=crops, labels=labels, target_heatmaps=target_heatmaps, heatmap_weights=heatmap_weights)
    hook_handle.remove()
    outputs["keypoint_loss"].backward()
    heatmaps = retained_heatmaps[0]

    # The loss is the one of the channels of the class of every crop, the triangle padding left out
    expected_loss = compute_keypoint_heatmap_loss(
        predicted_heatmaps=torch.stack(tensors=[heatmaps[0, [0, 1, 2, 0]], heatmaps[1, 3:]]),
        target_heatmaps=target_heatmaps, heatmap_weights=torch.tensor(data=[[1.0, 1.0, 1.0, 0.0],
                                                                            [1.0, 1.0, 0.0, 1.0]]))
    assert torch.allclose(input=outputs["keypoint_loss"], other=expected_loss)

    # Gradient on the trained channels only
    channel_gradients = heatmaps.grad.abs().flatten(start_dim=2).sum(dim=2)
    assert bool((channel_gradients[0, :3] > 0).all()) and float(channel_gradients[0, 3:].sum()) == 0.0
    assert float(channel_gradients[1, :3].sum()) == 0.0 and float(channel_gradients[1, 5]) == 0.0
    assert bool((channel_gradients[1, [3, 4, 6]] > 0).all())
