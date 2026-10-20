"""
Tests of vision_training/models/keypoints/keypoint_detector.py, the bottom-up model of the keypoint detection task: the
heatmap channel table (consecutive channels per class in increasing class order, -1 for the padding positions and the
classes without keypoints, the 'class.keypoint' names, an empty definition refused), the extraction of the keypoint
candidates (the Gaussian targets of build_keypoint_targets come back at the vertex of the parabola through their
log-heatmap, within a sixth of a cell of their keypoints, with the tag of their cell, a round Gaussian peak is found
exactly wherever it lies in its cell, a maximum on the border is not moved, the candidates are sorted and padded with
zero scores, and the higher resolution heatmaps are averaged at half the image resolution), and the forward pass of
the KeypointDetector on the keypoint fixture with and without the higher resolution branch and the 3x3 hidden
convolution of the prediction heads (the outputs and their shapes, the first heatmaps at the prior probability and the
first tags near 0, the heatmaps the sigmoid of the logits, the losses given only with the ground truth and equal to the
losses of the logits and of the targets, the heatmap loss training every heatmap channel and the tag losses the tags of
the cells of the labelled keypoints only, and the heatmap loss leaving out the cells of the ignored regions but their
keypoint cells), and its training under the bfloat16 autocast of the mixed precision (float32 losses close to the
float32 ones, finite float32 gradients, float32 candidates).
"""

import pytest
import torch

from functools import partial
from torch import nn
from typing import List, Tuple

from vision_training.data.keypoints.heatmap_targets import build_keypoint_targets
from vision_training.losses.keypoint_losses import compute_heatmap_loss, compute_tag_losses
from vision_training.models.keypoints.keypoint_detector import (HEATMAP_STRIDE, KeypointDetector,
                                                                build_keypoint_channel_table,
                                                                extract_keypoint_candidates)
from tests.fixtures.configuration_fixtures import get_keypoint_model_configuration

# Image size of the keypoint fixture and the size of its heatmaps
IMAGE_SIZE = 64
HEATMAP_SIZE = IMAGE_SIZE // HEATMAP_STRIDE

# Keypoints of the fixture image: two rectangles (class 2) and a triangle (class 1) whose right keypoint is unlabelled
GROUND_TRUTH_KEYPOINTS = [torch.tensor(data=[[[6.0, 6.0, 2.0], [26.0, 6.0, 2.0], [26.0, 26.0, 2.0], [6.0, 26.0, 2.0]],
                                             [[38.0, 38.0, 2.0], [58.0, 38.0, 2.0], [58.0, 58.0, 2.0],
                                              [38.0, 58.0, 2.0]],
                                             [[46.0, 6.0, 2.0], [38.0, 22.0, 2.0], [54.0, 22.0, 0.0],
                                              [0.0, 0.0, 0.0]]]),
                          torch.zeros(size=(0, 4, 3))]
GROUND_TRUTH_LABELS = [torch.tensor(data=[2, 2, 1]), torch.zeros(size=(0,), dtype=torch.int64)]


def keep_predictions(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor,
                     retained_predictions: List[torch.Tensor]) -> None:
    """
    Forward hook of the prediction heads keeping their predictions, so that the test of the losses sees which heatmap
    and tag cells receive a gradient from them and computes the expected heatmap loss from the logits.

    Args:
        module (nn.Module): The heatmap and tag head or the higher resolution branch.
        inputs (Tuple[torch.Tensor, ...]): The inputs of the module.
        output (torch.Tensor): The [B, 2C, h, w] predicted heatmap logits then tags, or the [B, C, 2h, 2w] higher
            resolution heatmap logits.
        retained_predictions (List[torch.Tensor]): The list receiving the predictions.
    """
    retained_predictions.append(output)


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


def test_candidates_come_back_to_their_keypoints() -> None:
    """
    Verifies that the Gaussian targets of random keypoints, one per channel, give a first candidate at the vertex of
    the parabola through the log-heatmap of its cell and of its neighbours: the cell of a keypoint being set to 1, a
    keypoint t_x and t_y cells from the centre of its cell comes back t_x / (1 + t_x² + t_y²) cells from it along x
    (and likewise along y), within a sixth of a cell of the keypoint and exactly for a keypoint at a cell centre. Its
    score is the maximum of its heatmap and its tag the tag of its cell, and the other candidates follow by decreasing
    score, padded with zero scores when the heatmap has fewer local maxima.
    """
    generator = torch.Generator().manual_seed(0)
    random_keypoints = torch.rand(size=(30, 2), generator=generator) * (IMAGE_SIZE - 8.0) + 4.0
    exact_keypoints = torch.tensor(data=[[18.0, 26.0], [19.0, 26.0]])
    keypoints = torch.cat(tensors=[random_keypoints, exact_keypoints])
    channel_table = torch.arange(end=32)[None]
    labelled_keypoints = torch.cat(tensors=[keypoints, torch.full(size=(32, 1), fill_value=2.0)], dim=1)
    targets = build_keypoint_targets(keypoints=[labelled_keypoints[None]],
                                     labels=[torch.tensor(data=[0])], channel_table=channel_table, number_channels=32,
                                     image_height=IMAGE_SIZE, image_width=IMAGE_SIZE, heatmap_strides=[HEATMAP_STRIDE],
                                     tag_stride=HEATMAP_STRIDE, heatmap_sigma=2.0)
    heatmaps = targets["heatmaps"][0]
    tags = torch.randn(size=heatmaps.shape, generator=generator)
    candidates = extract_keypoint_candidates(heatmaps=heatmaps, tags=tags, higher_resolution_heatmaps=None,
                                             maximum_candidates=3)
    assert candidates["candidate_positions"].shape == (1, 32, 3, 2)
    assert candidates["candidate_scores"].shape == (1, 32, 3)
    first_positions = candidates["candidate_positions"][0, :, 0]

    # Offset of every keypoint from the centre of its cell in cells, and the vertex of the parabola of its target
    cells = torch.div(input=keypoints, other=HEATMAP_STRIDE, rounding_mode="floor")
    keypoint_offsets = keypoints / HEATMAP_STRIDE - 0.5 - cells
    squared_offset_norms = (keypoint_offsets ** 2).sum(dim=1, keepdim=True)
    expected_positions = (cells + 0.5 + keypoint_offsets / (1.0 + squared_offset_norms)) * HEATMAP_STRIDE
    assert torch.allclose(input=first_positions, other=expected_positions, atol=1e-4)
    assert float((first_positions - keypoints).abs().max()) <= HEATMAP_STRIDE / 6.0 + 1e-5
    assert torch.equal(input=first_positions[30], other=exact_keypoints[0])
    assert torch.equal(input=candidates["candidate_scores"][0, :, 0],
                       other=heatmaps[0].flatten(start_dim=1).max(dim=1).values)

    # The tag of the first candidate is the tag of its cell, and a single Gaussian leaves zero scores behind it
    cells = cells.to(dtype=torch.int64)
    assert torch.equal(input=candidates["candidate_tags"][0, :, 0],
                       other=tags[0, torch.arange(end=32), cells[:, 1], cells[:, 0]])
    assert bool((candidates["candidate_scores"][0, :, 1:] == 0.0).all())

    # A maximum in the top left cell has no neighbour above nor on its left and stays at the cell centre
    border_heatmaps = torch.zeros(size=(1, 1, HEATMAP_SIZE, HEATMAP_SIZE))
    border_heatmaps[0, 0, 0, 0], border_heatmaps[0, 0, 0, 1], border_heatmaps[0, 0, 1, 0] = 0.9, 0.5, 0.4
    border_heatmaps[0, 0, 8, 8] = 0.7
    border_candidates = extract_keypoint_candidates(heatmaps=border_heatmaps,
                                                    tags=torch.zeros_like(input=border_heatmaps),
                                                    higher_resolution_heatmaps=None, maximum_candidates=2)
    assert border_candidates["candidate_positions"][0, 0].tolist() == [[2.0, 2.0], [34.0, 34.0]]
    assert border_candidates["candidate_scores"][0, 0].tolist() == [pytest.approx(0.9), pytest.approx(0.7)]


def test_candidates_reach_a_round_gaussian_peak_exactly() -> None:
    """
    Verifies that the candidate of a round Gaussian heatmap, whose logarithm is a paraboloid, is its peak wherever the
    peak lies within its cell: the vertex of the parabola through the log-heatmap along each axis is exact for it, and
    two equal cells next to each other put the candidate on their common border.
    """
    generator = torch.Generator().manual_seed(1)
    peak_centres = torch.rand(size=(8, 2), generator=generator) * (HEATMAP_SIZE - 4.0) + 2.0
    cell_indices = torch.arange(end=HEATMAP_SIZE, dtype=torch.float32)
    squared_distances = ((cell_indices[None, None, :] - peak_centres[:, 0, None, None]) ** 2 +
                         (cell_indices[None, :, None] - peak_centres[:, 1, None, None]) ** 2)
    gaussian_heatmaps = torch.exp(input=-squared_distances / (2.0 * 2.0 ** 2))[None]
    candidates = extract_keypoint_candidates(heatmaps=gaussian_heatmaps, tags=torch.zeros_like(input=gaussian_heatmaps),
                                             higher_resolution_heatmaps=None, maximum_candidates=1)
    assert torch.allclose(input=candidates["candidate_positions"][0, :, 0],
                          other=(peak_centres + 0.5) * HEATMAP_STRIDE, atol=1e-3)

    # A plateau of two equal cells gives the border between them, whichever of the two the candidate is
    plateau_heatmaps = torch.zeros(size=(1, 1, HEATMAP_SIZE, HEATMAP_SIZE))
    plateau_heatmaps[0, 0, 5, 6:8] = 0.8
    plateau_heatmaps[0, 0, 4:7:2, 6:8] = 0.3
    plateau_candidates = extract_keypoint_candidates(heatmaps=plateau_heatmaps,
                                                     tags=torch.zeros_like(input=plateau_heatmaps),
                                                     higher_resolution_heatmaps=None, maximum_candidates=1)
    assert plateau_candidates["candidate_positions"][0, 0, 0].tolist() == [7.0 * HEATMAP_STRIDE,
                                                                           5.5 * HEATMAP_STRIDE]


def test_higher_resolution_heatmaps_are_averaged_at_half_the_resolution() -> None:
    """
    Verifies that with the higher resolution heatmaps, the quarter resolution heatmaps and tags are upsampled to half
    the image resolution: zero quarter resolution heatmaps and the Gaussian of a keypoint at the centre of a cell at
    stride 2 give a candidate at the keypoint with half the score, a constant tag map gives its constant, and the
    number of candidates is capped by the number of cells.
    """
    keypoint = torch.tensor(data=[[[21.0, 37.0, 2.0]]])
    targets = build_keypoint_targets(keypoints=[keypoint], labels=[torch.tensor(data=[1])],
                                     channel_table=torch.tensor(data=[[-1], [0]]), number_channels=1,
                                     image_height=IMAGE_SIZE, image_width=IMAGE_SIZE, heatmap_strides=[2],
                                     tag_stride=HEATMAP_STRIDE, heatmap_sigma=2.0)
    candidates = extract_keypoint_candidates(heatmaps=torch.zeros(size=(1, 1, HEATMAP_SIZE, HEATMAP_SIZE)),
                                             tags=torch.full(size=(1, 1, HEATMAP_SIZE, HEATMAP_SIZE), fill_value=0.3),
                                             higher_resolution_heatmaps=targets["heatmaps"][0],
                                             maximum_candidates=10000)
    number_cells = (2 * HEATMAP_SIZE) ** 2
    assert candidates["candidate_positions"].shape == (1, 1, number_cells, 2)
    assert torch.allclose(input=candidates["candidate_positions"][0, 0, 0], other=keypoint[0, 0, :2], atol=1e-4)
    assert float(candidates["candidate_scores"][0, 0, 0]) == pytest.approx(float(targets["heatmaps"][0].max()) / 2.0)
    assert torch.allclose(input=candidates["candidate_tags"], other=torch.tensor(data=0.3))


@pytest.mark.parametrize("higher_resolution, hidden_width", [(False, 0), (True, 0), (False, 16), (True, 16)])
def test_detector_gives_candidates_without_ground_truth(higher_resolution: bool, hidden_width: int) -> None:
    """
    Verifies the inference outputs of the keypoint detector of the fixture (7 channels: the 3 of the triangle, then the
    4 of the rectangle): the heatmaps and tags at a quarter of the image, the higher resolution heatmaps at half of it
    with the branch, and the 5 candidates of every channel inside the image, without any loss. With a hidden width,
    every prediction head starts with a 3x3 convolution of that width before its 1x1 convolution. The first heatmaps
    are the prior probability of 0.01 at every cell and the first tags near 0.

    Args:
        higher_resolution (bool): Whether the detector has the higher resolution branch.
        hidden_width (int): The channels of the 3x3 convolution of the prediction heads, 0 for none.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_keypoint_model_configuration(higher_resolution=higher_resolution,
                                                                    hidden_width=hidden_width)
    model = KeypointDetector(configuration=configuration, device=device, dtype=dtype).eval()
    assert model.channel_table.tolist() == [[-1, -1, -1, -1], [0, 1, 2, -1], [3, 4, 5, 6]]

    # Every prediction head ends with a 1x1 convolution, after a 3x3 convolution of the hidden width if any
    prediction_heads = [model.keypoint_head["prediction_convolution"]]
    if higher_resolution:
        prediction_heads.append(model.keypoint_head["higher_resolution_branch"].heatmap_head)
    for prediction_head in prediction_heads:
        assert len(prediction_head) == (2 if hidden_width > 0 else 1)
        assert prediction_head[-1].kernel_size == (1, 1)
        if hidden_width > 0:
            hidden_convolution = prediction_head[0].block[0]
            assert hidden_convolution.kernel_size == (3, 3) and hidden_convolution.out_channels == hidden_width
            assert prediction_head[-1].in_channels == hidden_width
    # The inference outputs of two random images: heatmaps and tags at a quarter of the image, without any loss
    with torch.no_grad():
        outputs = model(input_tensor=torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0)
    expected_keys = {"heatmaps", "tags", "candidate_positions", "candidate_scores", "candidate_tags"}
    assert set(outputs) == expected_keys | ({"higher_resolution_heatmaps"} if higher_resolution else set())
    assert outputs["heatmaps"].shape == outputs["tags"].shape == (2, 7, HEATMAP_SIZE, HEATMAP_SIZE)
    if higher_resolution:
        assert outputs["higher_resolution_heatmaps"].shape == (2, 7, 2 * HEATMAP_SIZE, 2 * HEATMAP_SIZE)
    # The first heatmaps hold the prior probability of 0.01 and the first tags stay near 0
    assert torch.allclose(input=outputs["heatmaps"], other=torch.full_like(input=outputs["heatmaps"], fill_value=0.01),
                          atol=1e-3)
    assert float(outputs["tags"].abs().max()) < 0.1
    if higher_resolution:
        assert torch.allclose(input=outputs["higher_resolution_heatmaps"], atol=1e-3,
                              other=torch.full_like(input=outputs["higher_resolution_heatmaps"], fill_value=0.01))
    # The 5 candidates of every channel lie inside the image
    assert outputs["candidate_positions"].shape == (2, 7, 5, 2)
    assert outputs["candidate_scores"].shape == outputs["candidate_tags"].shape == (2, 7, 5)
    assert bool(((outputs["candidate_positions"] > 0.0) & (outputs["candidate_positions"] < IMAGE_SIZE)).all())


@pytest.mark.parametrize("higher_resolution, hidden_width", [(False, 0), (True, 0), (True, 16)])
def test_losses_train_every_heatmap_and_the_tags_of_the_keypoints(higher_resolution: bool, hidden_width: int) -> None:
    """
    Verifies that with the ground truth of an image of two rectangles and a triangle next to an image without object,
    the heatmaps are the sigmoid of the predicted logits, the heatmap loss is the sum of the heatmap losses of these
    logits and of the targets at every stride and the tag losses are the weighted pull and push losses of the targets,
    that every heatmap channel of both images receives a gradient from the heatmap loss (the empty ones are trained
    towards zero), and that the tags receive one from the tag losses only at the cells of the labelled keypoints (the
    higher resolution branch reads the tags too, so the heatmap loss reaches them).

    Args:
        higher_resolution (bool): Whether the detector has the higher resolution branch.
        hidden_width (int): The channels of the 3x3 convolution of the prediction heads, 0 for none.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_keypoint_model_configuration(higher_resolution=higher_resolution,
                                                                    hidden_width=hidden_width)
    model = KeypointDetector(configuration=configuration, device=device, dtype=dtype).train()
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0

    # The predictions of the heads are kept to see which cells the losses reach, the quarter resolution logits and
    # tags first, then the higher resolution logits
    retained_predictions = []
    prediction_modules = [model.keypoint_head["prediction_convolution"]]
    if higher_resolution:
        prediction_modules.append(model.keypoint_head["higher_resolution_branch"])
    hook_handles = [prediction_module.register_forward_hook(
        hook=partial(keep_predictions, retained_predictions=retained_predictions))
        for prediction_module in prediction_modules]
    outputs = model(input_tensor=images, ground_truth_keypoints=GROUND_TRUTH_KEYPOINTS,
                    ground_truth_labels=GROUND_TRUTH_LABELS)
    for hook_handle in hook_handles:
        hook_handle.remove()
    assert {"heatmap_loss", "tag_pull_loss", "tag_push_loss"} <= set(outputs)

    # The heatmaps are the sigmoid of the logits
    heatmap_logits = [retained_predictions[0][:, :7]] + retained_predictions[1:]
    assert torch.allclose(input=outputs["heatmaps"], other=torch.sigmoid(heatmap_logits[0]))
    if higher_resolution:
        assert torch.allclose(input=outputs["higher_resolution_heatmaps"], other=torch.sigmoid(heatmap_logits[1]))

    # The losses are those of the targets, at the strides 4 and 2 with the branch
    heatmap_strides = [HEATMAP_STRIDE, HEATMAP_STRIDE // 2] if higher_resolution else [HEATMAP_STRIDE]
    targets = build_keypoint_targets(keypoints=GROUND_TRUTH_KEYPOINTS, labels=GROUND_TRUTH_LABELS,
                                     channel_table=model.channel_table, number_channels=7, image_height=IMAGE_SIZE,
                                     image_width=IMAGE_SIZE, heatmap_strides=heatmap_strides,
                                     tag_stride=HEATMAP_STRIDE, heatmap_sigma=2.0)
    expected_heatmap_loss = sum(compute_heatmap_loss(predicted_logits=logits, target_heatmaps=target)
                                for logits, target in zip(heatmap_logits, targets["heatmaps"], strict=True))
    pull_loss, push_loss = compute_tag_losses(tag_maps=outputs["tags"], tag_indices=targets["tag_indices"],
                                              tag_valid=targets["tag_valid"], object_labels=targets["object_labels"])
    assert torch.allclose(input=outputs["heatmap_loss"], other=expected_heatmap_loss)
    assert torch.allclose(input=outputs["tag_pull_loss"], other=0.001 * pull_loss)
    assert torch.allclose(input=outputs["tag_push_loss"], other=0.001 * push_loss)
    assert push_loss.item() > 0.0

    # Every heatmap channel of both images is trained by the heatmap loss
    heatmap_gradients = torch.autograd.grad(outputs=outputs["heatmap_loss"], inputs=retained_predictions[0],
                                            retain_graph=True)[0]
    assert bool((heatmap_gradients[:, :7].abs().flatten(start_dim=2).sum(dim=2) > 0.0).all())

    # The tag losses reach the tags only, at the cells of the 10 labelled keypoints
    tag_gradients = torch.autograd.grad(outputs=outputs["tag_pull_loss"] + outputs["tag_push_loss"],
                                        inputs=retained_predictions[0])[0]
    assert not tag_gradients[:, :7].any()
    tag_gradient_cells = (tag_gradients[:, 7:].reshape(2, -1) != 0.0).nonzero()
    trained_cells = targets["tag_indices"][0][targets["tag_valid"][0]]
    assert len(trained_cells) == 10 and not tag_gradient_cells[:, 0].any()
    assert set(tag_gradient_cells[:, 1].tolist()) == set(trained_cells.tolist())


@pytest.mark.parametrize("higher_resolution", [False, True])
def test_heatmap_loss_leaves_out_the_ignored_regions(higher_resolution: bool) -> None:
    """
    Verifies that with ignored regions, the quarter of the first image holding the second rectangle and a corner of the
    empty second image, the heatmap loss of the detector is the sum over the strides of the heatmap losses of the
    logits without the ignored cells of the targets, below the loss of every cell; and that at the last stride, the
    ignored cells receive no gradient from it but the 4 keypoint cells of the rectangle, which stay trained like every
    cell outside the regions.

    Args:
        higher_resolution (bool): Whether the detector has the higher resolution branch.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_keypoint_model_configuration(higher_resolution=higher_resolution,
                                                                    hidden_width=0)
    model = KeypointDetector(configuration=configuration, device=device, dtype=dtype).train()
    ignored_regions = torch.zeros(size=(2, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    ignored_regions[0, 32:, 32:] = 1
    ignored_regions[1, :16, :16] = 1

    # The predictions of the heads are kept, the logits of the last stride feeding only the heatmap loss of that stride
    retained_predictions = []
    prediction_modules = [model.keypoint_head["prediction_convolution"]]
    if higher_resolution:
        prediction_modules.append(model.keypoint_head["higher_resolution_branch"])
    hook_handles = [prediction_module.register_forward_hook(
        hook=partial(keep_predictions, retained_predictions=retained_predictions))
        for prediction_module in prediction_modules]
    outputs = model(input_tensor=torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0,
                    ground_truth_keypoints=GROUND_TRUTH_KEYPOINTS, ground_truth_labels=GROUND_TRUTH_LABELS,
                    ignored_regions=ignored_regions)
    for hook_handle in hook_handles:
        hook_handle.remove()

    # The heatmap loss is that of the logits without the ignored cells of the targets, at the strides 4 and 2 with the
    # branch, and below the loss of every cell
    heatmap_logits = [retained_predictions[0][:, :7]] + retained_predictions[1:]
    heatmap_strides = [HEATMAP_STRIDE, HEATMAP_STRIDE // 2] if higher_resolution else [HEATMAP_STRIDE]
    targets = build_keypoint_targets(keypoints=GROUND_TRUTH_KEYPOINTS, labels=GROUND_TRUTH_LABELS,
                                     channel_table=model.channel_table, number_channels=7, image_height=IMAGE_SIZE,
                                     image_width=IMAGE_SIZE, heatmap_strides=heatmap_strides,
                                     tag_stride=HEATMAP_STRIDE, heatmap_sigma=2.0, ignored_regions=ignored_regions)
    expected_heatmap_loss = sum(compute_heatmap_loss(predicted_logits=logits, target_heatmaps=target,
                                                     ignored_cells=stride_ignored_cells)
                                for logits, target, stride_ignored_cells
                                in zip(heatmap_logits, targets["heatmaps"], targets["ignored_cells"], strict=True))
    every_cell_heatmap_loss = sum(compute_heatmap_loss(predicted_logits=logits, target_heatmaps=target)
                                  for logits, target in zip(heatmap_logits, targets["heatmaps"], strict=True))
    assert torch.allclose(input=outputs["heatmap_loss"], other=expected_heatmap_loss)
    assert outputs["heatmap_loss"].item() < every_cell_heatmap_loss.item()

    # At the last stride, the ignored cells receive no gradient but the keypoint cells of the rectangle, which stay
    # trained like every cell outside the regions
    heatmap_gradients = torch.autograd.grad(outputs=outputs["heatmap_loss"], inputs=retained_predictions[-1])[0][:, :7]
    ignored_cells = targets["ignored_cells"][-1].expand_as(other=heatmap_gradients)
    keypoint_cells = targets["heatmaps"][-1] == 1.0
    trained_cells = heatmap_gradients != 0.0
    assert int((ignored_cells & keypoint_cells).sum()) == 4
    assert not trained_cells[ignored_cells & ~keypoint_cells].any()
    assert bool(trained_cells[~ignored_cells | keypoint_cells].all())


@pytest.mark.parametrize("higher_resolution", [False, True])
def test_detector_trains_under_the_bfloat16_autocast(higher_resolution: bool) -> None:
    """
    Verifies the keypoint detector under the bfloat16 autocast of 'Optimization.mixed_precision', on the CPU, against
    float32 with the same weights and images: its predicted heatmap logits and tags come out in bfloat16, and its three
    losses are finite and those of these predictions computed in float32, the heatmap and push losses within 1 % of
    the float32 ones (the pull loss of the untrained tags, about 1e-9, being only their rounding); the backward pass of
    their sum, outside the autocast like the trainer, gives every parameter a finite float32 gradient, pointing like
    the float32 one; and the keypoint candidates of the inference are float32.

    Args:
        higher_resolution (bool): Whether the detector has the higher resolution branch.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_keypoint_model_configuration(higher_resolution=higher_resolution,
                                                                    hidden_width=16)
    model = KeypointDetector(configuration=configuration, device=device, dtype=dtype).train()
    images = torch.rand(size=(2, 3, IMAGE_SIZE, IMAGE_SIZE)) * 255.0
    loss_names = ["heatmap_loss", "tag_pull_loss", "tag_push_loss"]

    # The predictions of the heads are kept, those of the last pass staying in the list
    retained_predictions = []
    prediction_modules = [model.keypoint_head["prediction_convolution"]]
    if higher_resolution:
        prediction_modules.append(model.keypoint_head["higher_resolution_branch"])
    hook_handles = [prediction_module.register_forward_hook(
        hook=partial(keep_predictions, retained_predictions=retained_predictions))
        for prediction_module in prediction_modules]

    # The losses and gradients of the same weights and images in float32, then under the autocast of the trainer
    outputs, gradients = {}, {}
    for precision in ["float32", "bfloat16"]:
        model.zero_grad()
        retained_predictions.clear()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=precision == "bfloat16"):
            outputs[precision] = model(input_tensor=images, ground_truth_keypoints=GROUND_TRUTH_KEYPOINTS,
                                       ground_truth_labels=GROUND_TRUTH_LABELS)
        sum(outputs[precision][loss_name] for loss_name in loss_names).backward()
        gradients[precision] = [parameter.grad for parameter in model.parameters()]
    for hook_handle in hook_handles:
        hook_handle.remove()

    # The predictions come out in bfloat16, and the losses are finite and those of these predictions in float32
    assert all(prediction.dtype == torch.bfloat16 for prediction in retained_predictions)
    heatmap_strides = [HEATMAP_STRIDE, HEATMAP_STRIDE // 2] if higher_resolution else [HEATMAP_STRIDE]
    targets = build_keypoint_targets(keypoints=GROUND_TRUTH_KEYPOINTS, labels=GROUND_TRUTH_LABELS,
                                     channel_table=model.channel_table, number_channels=7, image_height=IMAGE_SIZE,
                                     image_width=IMAGE_SIZE, heatmap_strides=heatmap_strides,
                                     tag_stride=HEATMAP_STRIDE, heatmap_sigma=2.0)
    heatmap_logits = [retained_predictions[0][:, :7]] + retained_predictions[1:]
    pull_loss, push_loss = compute_tag_losses(tag_maps=retained_predictions[0][:, 7:].to(dtype=torch.float32),
                                              tag_indices=targets["tag_indices"], tag_valid=targets["tag_valid"],
                                              object_labels=targets["object_labels"])
    expected_losses = {"heatmap_loss": sum(compute_heatmap_loss(predicted_logits=logits.to(dtype=torch.float32),
                                                                target_heatmaps=target)
                                           for logits, target in zip(heatmap_logits, targets["heatmaps"], strict=True)),
                       "tag_pull_loss": 0.001 * pull_loss, "tag_push_loss": 0.001 * push_loss}
    for loss_name in loss_names:
        loss_value = outputs["bfloat16"][loss_name]
        assert loss_value.dtype == torch.float32 and bool(torch.isfinite(loss_value))
        assert torch.allclose(input=loss_value, other=expected_losses[loss_name], rtol=1e-5, atol=0.0)

    # The heatmap and push losses are close to the float32 ones
    for loss_name in ["heatmap_loss", "tag_push_loss"]:
        assert torch.allclose(input=outputs["bfloat16"][loss_name], other=outputs["float32"][loss_name], rtol=0.01,
                              atol=0.0)

    # Every parameter receives a finite float32 gradient, pointing like the float32 one
    assert all(gradient is not None and gradient.dtype == torch.float32 and bool(torch.isfinite(gradient).all())
               for gradient in gradients["bfloat16"])
    gradient_similarity = nn.functional.cosine_similarity(
        x1=torch.cat(tensors=[gradient.flatten() for gradient in gradients["bfloat16"]]),
        x2=torch.cat(tensors=[gradient.flatten() for gradient in gradients["float32"]]), dim=0)
    assert gradient_similarity.item() > 0.99

    # The keypoint candidates of the inference are float32
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        inference_outputs = model.eval()(input_tensor=images)
    assert all(inference_outputs[output_name].dtype == torch.float32
               for output_name in ["candidate_positions", "candidate_scores", "candidate_tags"])
