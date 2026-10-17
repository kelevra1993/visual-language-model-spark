"""
Tests of vision_training/data/keypoints/heatmap_targets.py, the training targets of the bottom-up keypoint detector:
draw_gaussian_heatmaps (a Gaussian of peak 1 at the cell of its keypoint, the values of the neighbouring cells, the
truncation at 3 sigma, the cell containing a keypoint away from its centre set to 1 with the neighbours of the
Gaussian centred on the keypoint, a keypoint on the border of two cells peaking in the cell of its tag, and the maximum
kept where two Gaussians of one heatmap overlap) and build_keypoint_targets
(the channel of every keypoint given by the class of its object, the untrained keypoints left out: unlabelled, outside
the image or on the padding positions of a class with fewer keypoints, the heatmaps at every stride, the tag cell of
every keypoint, the objects padded to the most objects of an image, and a batch without object).
"""

import math
import torch

from vision_training.data.keypoints.heatmap_targets import build_keypoint_targets, draw_gaussian_heatmaps

# Channels of the classes of the keypoint fixture: the 3 keypoints of the triangle (class 1), then the 4 of the
# rectangle (class 2), the background and the padding positions reading -1
CHANNEL_TABLE = torch.tensor(data=[[-1, -1, -1, -1], [0, 1, 2, -1], [3, 4, 5, 6]])
NUMBER_CHANNELS = 7


def compute_tag_cell(channel: int, keypoint: torch.Tensor, tag_stride: int, tag_size: int) -> int:
    """
    Gives the flat index of the cell of a channel containing a keypoint in the [C x tag_size x tag_size] tag map of its
    image, the index read by the tag losses for that keypoint.

    Args:
        channel (int): The heatmap channel of the keypoint.
        keypoint (torch.Tensor): The (x, y) keypoint in image pixels, possibly followed by its visibility.
        tag_stride (int): The image pixels per tag map cell.
        tag_size (int): The number of rows and columns of the tag map.

    Returns:
        int: The flat index of the cell.
    """
    return (channel * tag_size + int(keypoint[1] // tag_stride)) * tag_size + int(keypoint[0] // tag_stride)


def test_gaussian_peaks_at_the_cell_of_its_keypoint() -> None:
    """
    Verifies that a keypoint at the centre of a cell gives 1 at that cell and exp(-1 / (2 sigma ** 2)) at its four
    neighbours, nothing beyond ceil(3 sigma) cells, that a second keypoint of the same heatmap one cell away keeps both
    peaks at 1 (maximum, not sum), and that a keypoint of another heatmap leaves the first one unchanged.
    """
    # Keypoint at the centre of cell (row 5, column 6) of a stride 4 heatmap, (6 + 0.5) x 4 = 26, (5 + 0.5) x 4 = 22
    heatmaps = draw_gaussian_heatmaps(points=torch.tensor(data=[[26.0, 22.0]]), map_indices=torch.tensor(data=[0]),
                                      number_maps=2, heatmap_height=16, heatmap_width=16, heatmap_stride=4,
                                      heatmap_sigma=1.0)
    assert heatmaps.shape == (2, 16, 16) and heatmaps.dtype == torch.float32
    assert heatmaps[0, 5, 6] == 1.0 and float(heatmaps[0].max()) == 1.0 and not heatmaps[1].any()
    neighbour_value = math.exp(-0.5)
    for row, column in [(4, 6), (6, 6), (5, 5), (5, 7)]:
        assert math.isclose(float(heatmaps[0, row, column]), neighbour_value, rel_tol=1e-6)

    # Nothing beyond 3 cells of the peak
    rows, columns = torch.nonzero(input=heatmaps[0], as_tuple=True)
    assert int((rows - 5).abs().max()) == 3 and int((columns - 6).abs().max()) == 3

    # A keypoint away from the centre of its cell, 1.5 pixels right of it, sets its cell to 1 and its neighbours to the
    # Gaussian centred on it, higher on its right; on the border of two cells, it peaks in the cell of its tag, the
    # right one
    shifted = draw_gaussian_heatmaps(points=torch.tensor(data=[[27.5, 22.0], [28.0, 42.0]]),
                                     map_indices=torch.tensor(data=[0, 0]), number_maps=1, heatmap_height=16,
                                     heatmap_width=16, heatmap_stride=4, heatmap_sigma=1.0)
    assert shifted[0, 5, 6] == 1.0 and shifted[0, 10, 7] == 1.0 and float(shifted[0, 10, 6]) < 1.0
    assert math.isclose(float(shifted[0, 5, 7]), math.exp(-0.625 ** 2 / 2.0), rel_tol=1e-6)
    assert math.isclose(float(shifted[0, 5, 5]), math.exp(-1.375 ** 2 / 2.0), rel_tol=1e-6)

    # Two keypoints of one heatmap one cell apart keep their peaks at 1, the other heatmap holding the third keypoint
    overlapping = draw_gaussian_heatmaps(points=torch.tensor(data=[[26.0, 22.0], [30.0, 22.0], [26.0, 22.0]]),
                                         map_indices=torch.tensor(data=[0, 0, 1]), number_maps=2, heatmap_height=16,
                                         heatmap_width=16, heatmap_stride=4, heatmap_sigma=1.0)
    assert overlapping[0, 5, 6] == 1.0 and overlapping[0, 5, 7] == 1.0 and float(overlapping.max()) == 1.0
    assert torch.equal(input=overlapping[1], other=heatmaps[0])


def test_targets_follow_the_channels_of_the_class_of_every_object() -> None:
    """
    Verifies the targets of a batch of two 64 x 64 images, the first with a triangle and a rectangle, the second with
    a rectangle whose keypoint 2 is unlabelled and keypoint 3 outside the image: the trained keypoints peak in the
    channel of their class at both strides, the padding position of the triangle (labelled on purpose) and the two
    untrained rectangle keypoints are left out, the tag index of every keypoint is the cell of its channel containing
    it, and the objects are padded to two per image with the class 0.
    """
    triangle = torch.tensor(data=[[10.0, 6.0, 2.0], [6.0, 18.0, 2.0], [14.0, 18.0, 1.0], [30.0, 30.0, 2.0]])
    rectangle = torch.tensor(data=[[34.0, 34.0, 2.0], [58.0, 34.0, 2.0], [58.0, 58.0, 2.0], [34.0, 58.0, 2.0]])
    second_rectangle = torch.tensor(data=[[2.0, 42.0, 2.0], [26.0, 42.0, 2.0], [26.0, 62.0, 0.0], [-2.0, 62.0, 2.0]])
    targets = build_keypoint_targets(keypoints=[torch.stack(tensors=[triangle, rectangle]), second_rectangle[None]],
                                     labels=[torch.tensor(data=[1, 2]), torch.tensor(data=[2])],
                                     channel_table=CHANNEL_TABLE, number_channels=NUMBER_CHANNELS, image_height=64,
                                     image_width=64, heatmap_strides=[4, 2], tag_stride=4, heatmap_sigma=1.0)
    assert [tuple(heatmaps.shape) for heatmaps in targets["heatmaps"]] == [(2, 7, 16, 16), (2, 7, 32, 32)]
    assert targets["object_labels"].tolist() == [[1, 2], [2, 0]]
    assert targets["tag_valid"].tolist() == [[[True, True, True, False], [True, True, True, True]],
                                             [[True, True, False, False], [False, False, False, False]]]

    # Every trained keypoint peaks at 1 in the channel of its class at the cell containing it at every stride
    expected_peaks = {(0, 0): triangle[0], (0, 1): triangle[1], (0, 2): triangle[2], (0, 3): rectangle[0],
                      (0, 6): rectangle[3], (1, 3): second_rectangle[0], (1, 4): second_rectangle[1]}
    for (image_index, channel), keypoint in expected_peaks.items():
        for heatmaps, stride in zip(targets["heatmaps"], [4, 2], strict=True):
            column, row = int(keypoint[0] // stride), int(keypoint[1] // stride)
            assert float(heatmaps[image_index, channel].max()) == float(heatmaps[image_index, channel, row, column])
            assert float(heatmaps[image_index, channel, row, column]) == 1.0

    # The untrained keypoints of the second rectangle give no Gaussian in their channels
    assert not targets["heatmaps"][0][1, [0, 1, 2, 5, 6]].any() and not targets["heatmaps"][1][1, 5:].any()

    # Tag cell of every keypoint in the [7 x 16 x 16] tag map of its image, the untrained ones at 0
    assert targets["tag_indices"][0, 0, :3].tolist() == [
        compute_tag_cell(channel=channel, keypoint=keypoint, tag_stride=4, tag_size=16)
        for channel, keypoint in enumerate(triangle[:3])]
    assert targets["tag_indices"][0, 1].tolist() == [
        compute_tag_cell(channel=3 + index, keypoint=keypoint, tag_stride=4, tag_size=16)
        for index, keypoint in enumerate(rectangle)]
    assert targets["tag_indices"][0, 0, 3] == 0 and targets["tag_indices"][1, 0, 2:].tolist() == [0, 0]


def test_batch_without_objects_gives_empty_targets() -> None:
    """
    Verifies that images without any object give zero heatmaps, one padding object of class 0 per image and no trained
    keypoint, so that the losses see empty images as negatives instead of failing.
    """
    targets = build_keypoint_targets(keypoints=[torch.zeros(size=(0, 4, 3)), torch.zeros(size=(0, 4, 3))],
                                     labels=[torch.zeros(size=(0,), dtype=torch.int64)] * 2,
                                     channel_table=CHANNEL_TABLE, number_channels=NUMBER_CHANNELS, image_height=32,
                                     image_width=32, heatmap_strides=[4], tag_stride=4, heatmap_sigma=2.0)
    assert targets["heatmaps"][0].shape == (2, 7, 8, 8) and not targets["heatmaps"][0].any()
    assert targets["object_labels"].tolist() == [[0], [0]] and not targets["tag_valid"].any()
