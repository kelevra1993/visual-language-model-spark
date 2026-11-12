"""
Tests of the DIoU-NMS of vision_training/models/detection/postprocessing.py, the final per-class non-maximum suppression
of 'Detector.nms_criterion' "distance_iou" evaluated at stage 5b of the detection benchmark: batched_distance_nms gives
the boxes of a greedy loop written from the authors' code (cython diounms and darknet diounms_sort), those of the
batched_nms of torchvision when no centre distance separates the boxes, keeps two overlapping boxes whose centres are
apart where the greedy NMS keeps one, never lets two groups suppress each other, handles empty inputs, single boxes and
equal scores, and keeps the same boxes whatever its chunks and the precision of the boxes.
"""

import pytest
import torch

from typing import Any, Dict, List, Tuple
from torchvision.ops import batched_nms

from vision_training.models.detection import postprocessing
from vision_training.models.detection.box_operations import get_intersection_over_union
from vision_training.models.detection.postprocessing import batched_distance_nms, suppress_groups_apart


def run_greedy_distance_nms(boxes: torch.Tensor, scores: torch.Tensor, group_indices: torch.Tensor,
                            iou_threshold: float, distance_exponent: float) -> List[int]:
    """
    Runs the greedy DIoU-NMS of the authors' Faster R-CNN code (lib/utils/cython_nms.pyx, diounms) and of darknet
    (src/box.c, diounms_sort), box after box, as the reference of batched_distance_nms: in every group, the boxes taken
    by decreasing score, every kept box suppressing the lower boxes of the group whose IoU minus the normalized centre
    distance to the power of the exponent exceeds the threshold, with the continuous coordinates, the strict suppression
    and the 1e-7 on the squared diagonal chosen for batched_distance_nms.

    Args:
        boxes (torch.Tensor): The [N, 4] boxes [x_min, y_min, x_max, y_max].
        scores (torch.Tensor): Their [N] scores.
        group_indices (torch.Tensor): Their [N] groups.
        iou_threshold (float): The threshold of the criterion.
        distance_exponent (float): The exponent of the normalized centre distance.

    Returns:
        List[int]: The indices of the kept boxes, by decreasing score, equal scores by increasing index.
    """
    box_list, score_list, group_list = boxes.tolist(), scores.tolist(), group_indices.tolist()
    score_order = sorted(range(len(box_list)), key=lambda index: (-score_list[index], index))
    kept_indices = []
    for group in sorted(set(group_list)):
        members = [index for index in score_order if group_list[index] == group]
        suppressed = set()
        for position, kept_index in enumerate(members):
            if kept_index in suppressed:
                continue
            kept_indices.append(kept_index)
            kept_box = box_list[kept_index]

            # Every lower box of the group still kept, compared with the kept box
            for other_index in members[position + 1:]:
                other_box = box_list[other_index]
                intersection = (max(0.0, min(kept_box[2], other_box[2]) - max(kept_box[0], other_box[0])) *
                                max(0.0, min(kept_box[3], other_box[3]) - max(kept_box[1], other_box[1])))
                union = ((kept_box[2] - kept_box[0]) * (kept_box[3] - kept_box[1]) +
                         (other_box[2] - other_box[0]) * (other_box[3] - other_box[1]) - intersection)
                enclosing_diagonal = ((max(kept_box[2], other_box[2]) - min(kept_box[0], other_box[0])) ** 2 +
                                      (max(kept_box[3], other_box[3]) - min(kept_box[1], other_box[1])) ** 2 + 1e-7)
                centre_distance = (((kept_box[0] + kept_box[2]) - (other_box[0] + other_box[2])) ** 2 +
                                   ((kept_box[1] + kept_box[3]) - (other_box[1] + other_box[3])) ** 2) / 4
                criterion = intersection / union - (centre_distance / enclosing_diagonal) ** distance_exponent
                if criterion > iou_threshold:
                    suppressed.add(other_index)
    return sorted(kept_indices, key=lambda index: (-score_list[index], index))


def draw_clustered_boxes(seed: int, number_boxes: int, number_groups: int) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                    torch.Tensor]:
    """
    Draws boxes gathered around five centres, so that many of them overlap above the thresholds, with distinct random
    scores and groups, in float64 so that the comparisons do not depend on rounding.

    Args:
        seed (int): The seed of the generator.
        number_boxes (int): The number of boxes.
        number_groups (int): The number of groups the boxes are spread over.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The [N, 4] boxes, their [N] scores and their [N] groups.
    """
    generator = torch.Generator().manual_seed(seed)
    centres = torch.rand(size=(5, 2), generator=generator, dtype=torch.float64) * 100
    centre_indices = torch.randint(low=0, high=5, size=(number_boxes,), generator=generator)
    box_centres = centres[centre_indices] + torch.randn(size=(number_boxes, 2), generator=generator,
                                                        dtype=torch.float64) * 4
    half_sizes = 5 + torch.rand(size=(number_boxes, 2), generator=generator, dtype=torch.float64) * 15
    boxes = torch.cat(tensors=[box_centres - half_sizes, box_centres + half_sizes], dim=1)
    scores = torch.rand(size=(number_boxes,), generator=generator, dtype=torch.float64)
    group_indices = torch.randint(low=0, high=number_groups, size=(number_boxes,), generator=generator)
    return boxes, scores, group_indices


def test_distance_nms_equals_the_greedy_loop_of_the_authors() -> None:
    """
    Verifies that the Cluster-NMS of batched_distance_nms keeps exactly the boxes of the greedy loop of the authors'
    code, by decreasing score, on random clustered draws of 10 to 69 boxes in 1 to 7 groups, with the exponents 1.0
    and 0.6 and the thresholds 0.45, 0.5 and 0.7.
    """
    for seed in range(40):
        boxes, scores, group_indices = draw_clustered_boxes(seed=seed, number_boxes=10 + (seed * 7) % 60,
                                                            number_groups=1 + seed % 7)
        for distance_exponent in [1.0, 0.6]:
            for iou_threshold in [0.45, 0.5, 0.7]:
                kept_indices = batched_distance_nms(boxes=boxes, scores=scores, group_indices=group_indices,
                                                    iou_threshold=iou_threshold, distance_exponent=distance_exponent)
                reference_indices = run_greedy_distance_nms(boxes=boxes, scores=scores, group_indices=group_indices,
                                                            iou_threshold=iou_threshold,
                                                            distance_exponent=distance_exponent)
                assert kept_indices.dtype == torch.int64
                assert kept_indices.tolist() == reference_indices, \
                    f"seed {seed}, exponent {distance_exponent}, threshold {iou_threshold}"


def test_distance_nms_without_centre_distance_equals_batched_nms() -> None:
    """
    Verifies that boxes sharing the centre of their group, whose distance term is 0, keep exactly the boxes of the
    batched_nms of torchvision, which suppresses above the threshold strictly too, at several thresholds, and that two
    such boxes overlapping by exactly the threshold are both kept by either.
    """
    for seed in range(10):
        generator = torch.Generator().manual_seed(seed)
        group_indices = torch.randint(low=0, high=4, size=(40,), generator=generator)
        group_centres = torch.rand(size=(4, 2), generator=generator, dtype=torch.float64) * 100
        half_sizes = 5 + torch.rand(size=(40, 2), generator=generator, dtype=torch.float64) * 10
        boxes = torch.cat(tensors=[group_centres[group_indices] - half_sizes,
                                   group_centres[group_indices] + half_sizes], dim=1)
        scores = torch.rand(size=(40,), generator=generator, dtype=torch.float64)
        for iou_threshold in [0.3, 0.5, 0.7]:
            kept_indices = batched_distance_nms(boxes=boxes, scores=scores, group_indices=group_indices,
                                                iou_threshold=iou_threshold, distance_exponent=1.0)
            reference_indices = batched_nms(boxes=boxes, scores=scores, idxs=group_indices,
                                            iou_threshold=iou_threshold)
            assert torch.equal(input=kept_indices, other=reference_indices), f"seed {seed}, threshold {iou_threshold}"

    # Boxes sharing their centre with an IoU of exactly 0.5 (2 / 4), kept by the strict suppression of both
    boxes = torch.tensor(data=[[0.0, 0.0, 2.0, 2.0], [0.0, 0.5, 2.0, 1.5]])
    scores, group_indices = torch.tensor(data=[0.9, 0.8]), torch.zeros(size=(2,), dtype=torch.int64)
    assert float(get_intersection_over_union(boxes_1=boxes[:1], boxes_2=boxes[1:])) == 0.5
    assert batched_nms(boxes=boxes, scores=scores, idxs=group_indices, iou_threshold=0.5).tolist() == [0, 1]
    assert batched_distance_nms(boxes=boxes, scores=scores, group_indices=group_indices, iou_threshold=0.5,
                                distance_exponent=1.0).tolist() == [0, 1]


@pytest.mark.parametrize(argnames=("second_box", "expected_kept_counts"), argvalues=[
    # A wider box sharing the right edge: IoU 0.526, distance term 0.044, kept apart with both exponents
    ([-9.0, 0.0, 10.0, 10.0], {"iou": 1, 1.0: 2, 0.6: 2}),
    # An IoU of 0.6 with the distance term 0.029, kept apart with the exponent 0.6 only (0.571 and 0.480)
    ([-20.0 / 3.0, 0.0, 10.0, 10.0], {"iou": 1, 1.0: 1, 0.6: 2}),
    # A larger box sharing the centre, without distance term: suppressed by every criterion
    ([-1.0, -1.0, 11.0, 11.0], {"iou": 1, 1.0: 1, 0.6: 1})])
def test_distance_nms_keeps_overlapping_boxes_whose_centres_are_apart(second_box: List[float],
                                                                      expected_kept_counts: Dict[Any, int]) -> None:
    """
    Verifies the case of the paper on a box of 10 pixels and a lower scoring box of its class overlapping it by more
    than the threshold 0.5: the greedy NMS keeps one box, DIoU-NMS keeps both when the IoU minus the distance term to
    the power of the exponent falls below the threshold, more often with the exponent 0.6 than with 1.0, and never
    when the centres coincide.

    Args:
        second_box (List[float]): The lower scoring box.
        expected_kept_counts (Dict[Any, int]): The number of kept boxes of the greedy NMS ("iou") and of DIoU-NMS with
            each exponent.
    """
    boxes = torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], second_box], dtype=torch.float64)
    scores = torch.tensor(data=[0.9, 0.8], dtype=torch.float64)
    group_indices = torch.zeros(size=(2,), dtype=torch.int64)
    assert float(get_intersection_over_union(boxes_1=boxes[:1], boxes_2=boxes[1:])) > 0.5

    assert len(batched_nms(boxes=boxes, scores=scores, idxs=group_indices, iou_threshold=0.5)) == \
        expected_kept_counts["iou"]
    for distance_exponent in [1.0, 0.6]:
        kept_indices = batched_distance_nms(boxes=boxes, scores=scores, group_indices=group_indices,
                                            iou_threshold=0.5, distance_exponent=distance_exponent)
        assert kept_indices.tolist() == [0, 1][:expected_kept_counts[distance_exponent]]


def test_distance_nms_groups_empty_inputs_and_equal_scores() -> None:
    """
    Verifies that boxes of different groups never suppress each other, even when equal, that an empty input keeps
    nothing and a single box is kept, that equal scores keep the first box of the input, and that the kept boxes come
    by decreasing score across the groups, like those of batched_nms.
    """
    box = [0.0, 0.0, 10.0, 10.0]
    empty_indices = batched_distance_nms(boxes=torch.zeros(size=(0, 4)), scores=torch.zeros(size=(0,)),
                                         group_indices=torch.zeros(size=(0,), dtype=torch.int64), iou_threshold=0.5,
                                         distance_exponent=1.0)
    assert empty_indices.shape == (0,) and empty_indices.dtype == torch.int64
    assert batched_distance_nms(boxes=torch.tensor(data=[box]), scores=torch.tensor(data=[0.3]),
                                group_indices=torch.tensor(data=[4]), iou_threshold=0.5,
                                distance_exponent=1.0).tolist() == [0]

    # Equal boxes: one per group is kept, the best of its group, or the first one for equal scores
    boxes = torch.tensor(data=[box] * 5)
    kept_indices = batched_distance_nms(boxes=boxes, scores=torch.tensor(data=[0.5, 0.5, 0.7, 0.6, 0.6]),
                                        group_indices=torch.tensor(data=[0, 0, 0, 1, 1]), iou_threshold=0.5,
                                        distance_exponent=1.0)
    assert kept_indices.tolist() == [2, 3]

    # Boxes apart, kept by decreasing score whatever their groups
    far_boxes = torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0], [100.0, 0.0, 110.0, 10.0]])
    assert batched_distance_nms(boxes=far_boxes, scores=torch.tensor(data=[0.2, 0.9, 0.5]),
                                group_indices=torch.tensor(data=[1, 0, 1]), iou_threshold=0.5,
                                distance_exponent=1.0).tolist() == [1, 2, 0]


def test_distance_nms_chunks_and_reduced_precision_keep_the_same_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that splitting the groups into chunks of one group, or of a few, keeps the boxes of a single chunk, and
    that boxes of the bfloat16 mixed precision are compared in float32, keeping the boxes of their float32 values.

    Args:
        monkeypatch (pytest.MonkeyPatch): Lowers the pair budget of the chunks of batched_distance_nms.
    """
    draws = [draw_clustered_boxes(seed=seed, number_boxes=60, number_groups=6) for seed in range(8)]
    single_chunk_indices = [batched_distance_nms(boxes=boxes, scores=scores, group_indices=group_indices,
                                                 iou_threshold=0.5, distance_exponent=1.0)
                            for boxes, scores, group_indices in draws]

    # Chunks of one group, then of the groups fitting 300 pairs
    for maximum_pairs in [1, 300]:
        monkeypatch.setattr(target=postprocessing, name="MAXIMUM_DISTANCE_NMS_PAIRS", value=maximum_pairs)
        for (boxes, scores, group_indices), reference_indices in zip(draws, single_chunk_indices, strict=True):
            assert torch.equal(input=batched_distance_nms(boxes=boxes, scores=scores, group_indices=group_indices,
                                                          iou_threshold=0.5, distance_exponent=1.0),
                               other=reference_indices)
    monkeypatch.undo()

    # Boxes in bfloat16 keep the boxes of their values in float32
    for boxes, scores, group_indices in draws:
        reduced_boxes = boxes.to(dtype=torch.bfloat16)
        for distance_exponent in [1.0, 0.6]:
            assert torch.equal(
                input=batched_distance_nms(boxes=reduced_boxes, scores=scores.float(), group_indices=group_indices,
                                           iou_threshold=0.5, distance_exponent=distance_exponent),
                other=batched_distance_nms(boxes=reduced_boxes.float(), scores=scores.float(),
                                           group_indices=group_indices, iou_threshold=0.5,
                                           distance_exponent=distance_exponent))


@pytest.mark.parametrize(argnames="number_boxes", argvalues=[200, 3000])
def test_groups_apart_keep_the_boxes_of_batched_nms(number_boxes: int) -> None:
    """
    Verifies that the suppression of the exports, every group moved beyond the largest coordinate then one nms call
    (suppress_groups_apart), keeps the same boxes in the same order as batched_nms of torchvision, below and above the
    4000 coordinates from which batched_nms loops over the groups on the CPU, on overlapping boxes of 7 groups, and
    that boxes of different groups never suppress each other.

    Args:
        number_boxes (int): The number of boxes, 800 or 12,000 coordinates.
    """
    generator = torch.Generator().manual_seed(number_boxes)
    corners = torch.rand(size=(number_boxes, 2), generator=generator) * 200.0
    sides = torch.rand(size=(number_boxes, 2), generator=generator) * 60.0 + 4.0
    boxes = torch.cat(tensors=[corners, corners + sides], dim=1)
    scores = torch.rand(size=(number_boxes,), generator=generator)
    group_indices = torch.randint(low=0, high=7, size=(number_boxes,), generator=generator)
    kept_indices = suppress_groups_apart(boxes=boxes, scores=scores, group_indices=group_indices, iou_threshold=0.5)
    assert torch.equal(input=kept_indices, other=batched_nms(boxes=boxes, scores=scores, idxs=group_indices,
                                                             iou_threshold=0.5))

    # Two identical boxes of different groups are both kept, of the same group only the higher scoring one
    same_boxes = torch.tensor(data=[[10.0, 10.0, 50.0, 50.0], [10.0, 10.0, 50.0, 50.0]])
    same_scores = torch.tensor(data=[0.9, 0.8])
    assert suppress_groups_apart(boxes=same_boxes, scores=same_scores, group_indices=torch.tensor(data=[0, 1]),
                                 iou_threshold=0.5).tolist() == [0, 1]
    assert suppress_groups_apart(boxes=same_boxes, scores=same_scores, group_indices=torch.tensor(data=[3, 3]),
                                 iou_threshold=0.5).tolist() == [0]
