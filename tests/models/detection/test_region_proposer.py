"""
Tests of vision_training/models/detection/region_proposer.py: the batched proposal filter, which keeps the best
proposals of every image after the suppression and pads the images left short with copies of their own survivors,
the per-image suppression of the training keeping the same proposals as the single call of the deployment, and the
selection level by level ('RegionProposal.proposal_selection' "per_level"), which ranks and suppresses the proposals
of every level on its own and leaves out the proposals clamped to an empty box.
"""

import pytest
import torch

from vision_training.models.detection.region_proposer import (RegionProposalFilter, select_best_proposals,
                                                               suppress_overlapping_proposals)


def test_proposal_filter_keeps_the_best_survivors_of_every_image() -> None:
    """
    Verifies the filtered proposals of a batch of two images: the duplicates of a better proposal are suppressed, the
    survivors of every image come in decreasing score order with their score and level, a proposal outside the
    pre-suppression limit is never kept, and the image left with one survivor fills its other slots with copies of it,
    the mask of the survivors leaving these copies out.
    """
    proposal_filter = RegionProposalFilter(configuration={
        "training": {"pre_nms_proposals": 4, "post_nms_proposals": 3},
        "inference": {"pre_nms_proposals": 4, "post_nms_proposals": 3}, "nms_iou_threshold": 0.5,
        "proposal_selection": "joint"}, level_sizes=[5]).train()
    boxes = torch.tensor(data=[[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 9.0], [20.0, 20.0, 30.0, 30.0],
                                [40.0, 40.0, 50.0, 50.0], [60.0, 60.0, 70.0, 70.0]],
                               [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 9.5], [0.0, 0.0, 9.5, 10.0],
                                [1.0, 0.0, 10.0, 10.0], [60.0, 60.0, 70.0, 70.0]]])
    logits = torch.tensor(data=[[3.0, 2.0, 1.0, -5.0, -6.0], [3.0, 2.5, 2.2, 2.1, -6.0]])
    origins = torch.tensor(data=[[5, 6, 7, 5, 6], [7, 6, 5, 7, 6]])

    filtered_boxes, filtered_scores, filtered_origins, valid_proposal_mask = proposal_filter(
        proposal_boxes=boxes, proposal_scores=logits, proposal_origins=origins, input_image_size=100)
    assert filtered_boxes[0].tolist() == [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]]
    assert torch.equal(input=filtered_scores[0], other=torch.sigmoid(input=logits[0, [0, 2, 3]]))
    assert filtered_origins.tolist() == [[5, 7, 5], [7, 7, 7]]
    assert filtered_boxes[1].tolist() == [[0.0, 0.0, 10.0, 10.0]] * 3
    assert valid_proposal_mask.tolist() == [[True, True, True], [True, False, False]]


def test_per_image_and_batched_suppressions_keep_the_same_proposals() -> None:
    """
    Verifies that the suppression of the training, one call per image, and that of the deployed detector, a single
    call over the batch, keep exactly the same proposals of every image.
    """
    generator = torch.Generator().manual_seed(0)
    corners = torch.rand(size=(4, 300, 2), generator=generator) * 400
    boxes = torch.cat(tensors=[corners, corners + 8 + torch.rand(size=(4, 300, 2), generator=generator) * 100], dim=2)
    scores = torch.rand(size=(4, 300), generator=generator)
    per_image_mask = suppress_overlapping_proposals(boxes=boxes, scores=scores, nms_iou_threshold=0.5,
                                                    suppress_per_image=True)
    batched_mask = suppress_overlapping_proposals(boxes=boxes, scores=scores, nms_iou_threshold=0.5,
                                                  suppress_per_image=False)
    assert torch.equal(input=per_image_mask, other=batched_mask)
    assert 0 < int(per_image_mask.sum()) < per_image_mask.numel()


def test_proposals_are_ranked_in_float32_under_mixed_precision() -> None:
    """
    Verifies that the proposals are ranked on float32 probabilities when the region proposal heads give bfloat16
    logits, as under the mixed precision of the training: the confident logits, whose bfloat16 sigmoids collapse to a
    few values, keep their order and distinct scores, the best proposal coming first.
    """
    # Disjoint boxes with distinct confident logits in bfloat16, the best ones last
    number_proposals, number_selected = 64, 16
    corners = torch.arange(end=number_proposals, dtype=torch.float32) * 20.0
    boxes = torch.stack(tensors=[corners, corners, corners + 10.0, corners + 10.0], dim=1).unsqueeze(dim=0)
    logits = torch.linspace(start=4.0, end=9.0, steps=number_proposals).to(dtype=torch.bfloat16).unsqueeze(dim=0)
    levels = torch.arange(end=number_proposals).unsqueeze(dim=0)

    # Selection under the bfloat16 autocast of the training, the levels giving the index of every selected proposal
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, selected_scores, selected_levels, valid_mask = select_best_proposals(
            proposal_scores=logits, proposal_boxes=boxes, proposal_levels=levels, level_sizes=[number_proposals],
            pre_nms_proposals=number_selected, post_nms_proposals=number_selected, nms_iou_threshold=0.5,
            input_image_size=2000, suppress_per_image=True, proposal_selection="joint")
    assert selected_scores.dtype == torch.float32 and bool(valid_mask.all())
    assert selected_levels[0].tolist() == list(range(number_proposals - 1, number_proposals - number_selected - 1, -1))
    assert len(set(selected_scores[0].tolist())) == number_selected


def select_two_level_proposals(proposal_selection: str, pre_nms_proposals: int, post_nms_proposals: int) -> tuple:
    """
    Selects the proposals of one image holding two levels for the selection tests: a level of five disjoint boxes
    scored 10 to 6, whose first box is duplicated (score 9.5), then a level of three disjoint boxes scored 3 to 1, whose
    best box is the duplicate of the best box of the first level, so that the first level takes every slot jointly and
    a suppression across the levels removes the best box of the second.

    Args:
        proposal_selection (str): "joint" or "per_level".
        pre_nms_proposals (int): The proposals kept before the suppression, of all the anchors or of every level.
        post_nms_proposals (int): The proposals kept after it.

    Returns:
        tuple: The selected boxes, scores, levels and valid mask of the image.
    """
    first_level_boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 9.8], [20.0, 0.0, 30.0, 10.0],
                         [40.0, 0.0, 50.0, 10.0], [60.0, 0.0, 70.0, 10.0], [80.0, 0.0, 90.0, 10.0]]
    second_level_boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 40.0, 20.0, 60.0], [40.0, 40.0, 60.0, 60.0]]
    boxes = torch.tensor(data=[first_level_boxes + second_level_boxes])
    logits = torch.tensor(data=[[10.0, 9.5, 9.0, 8.0, 7.0, 6.0, 3.0, 2.0, 1.0]])
    levels = torch.tensor(data=[[4] * 6 + [5] * 3])
    selected_boxes, selected_scores, selected_levels, valid_mask = select_best_proposals(
        proposal_scores=logits, proposal_boxes=boxes, proposal_levels=levels, level_sizes=[6, 3],
        pre_nms_proposals=pre_nms_proposals, post_nms_proposals=post_nms_proposals, nms_iou_threshold=0.7,
        input_image_size=100, suppress_per_image=True, proposal_selection=proposal_selection)
    return selected_boxes[0], selected_scores[0], selected_levels[0], valid_mask[0]


def test_per_level_selection_keeps_the_best_proposals_of_every_level() -> None:
    """
    Verifies the selection level by level against the selection over all the levels on an image where the first level
    is the most confident: jointly, the four best proposals all come from the first level, its duplicate is suppressed
    and the second level gets no slot; level by level, the four best of every level reach their own suppression, the
    duplicate of the first level is suppressed within it, the best box of the second level, duplicating a box of the
    first, is kept, and the post-NMS count is filled with the best survivors of both levels.
    """
    # Jointly: the four best anchors are all of the first level, its duplicate suppressed, the second level left out
    joint_boxes, _, joint_levels, joint_valid_mask = select_two_level_proposals(
        proposal_selection="joint", pre_nms_proposals=4, post_nms_proposals=4)
    assert joint_levels[joint_valid_mask].tolist() == [4, 4, 4]
    assert joint_valid_mask.tolist() == [True, True, True, False]

    # Level by level: the survivors of both levels, the second level keeping the box of the first level's best box
    per_level_boxes, per_level_scores, per_level_levels, per_level_valid_mask = select_two_level_proposals(
        proposal_selection="per_level", pre_nms_proposals=4, post_nms_proposals=6)
    assert per_level_valid_mask.tolist() == [True] * 6
    assert per_level_levels.tolist() == [4, 4, 4, 5, 5, 5]
    assert per_level_boxes.tolist() == [[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0], [40.0, 0.0, 50.0, 10.0],
                                        [0.0, 0.0, 10.0, 10.0], [0.0, 40.0, 20.0, 60.0], [40.0, 40.0, 60.0, 60.0]]
    assert torch.equal(input=per_level_scores, other=torch.sigmoid(input=torch.tensor(data=[10.0, 9.0, 8.0, 3.0, 2.0,
                                                                                               1.0])))

    # The post-NMS count is taken over all the levels, by score, and limited by the proposals kept before the
    # suppression (4 of the first level and the 3 anchors of the second)
    _, _, best_levels, best_valid_mask = select_two_level_proposals(proposal_selection="per_level",
                                                                    pre_nms_proposals=4, post_nms_proposals=4)
    assert best_levels.tolist() == [4, 4, 4, 5] and bool(best_valid_mask.all())
    _, all_scores, _, all_valid_mask = select_two_level_proposals(proposal_selection="per_level",
                                                                  pre_nms_proposals=4, post_nms_proposals=20)
    assert all_scores.shape[0] == 7 and all_valid_mask.tolist() == [True] * 6 + [False]


def test_per_level_selection_leaves_out_the_empty_proposals() -> None:
    """
    Verifies that the selection level by level leaves out the proposals clamped to a box narrower than 1e-3 pixel,
    which the selection over all the levels keeps, an isolated one as well as one overlapping another proposal, and
    that such a proposal suppresses no other one even when it is the most confident: a box of 1.1e-3 pixel overlapping
    it by an IoU of about 0.73, above the threshold, is kept, as when the empty proposals are removed before the
    suppression.
    """
    # The best proposal regressed past the right border, clamped to a width of 8e-4, a proposal left of the image,
    # clamped to a width of 0, then a box 1.1e-3 wide containing the first one, then a box far from all of them
    boxes = torch.tensor(data=[[[99.9992, 0.0, 100.5, 10.0], [-5.0, 50.0, 0.0, 60.0], [99.9989, 0.0, 100.0, 10.0],
                                [0.0, 0.0, 10.0, 10.0]]], dtype=torch.float64)
    logits = torch.tensor(data=[[5.0, 4.5, 4.0, 3.0]], dtype=torch.float64)
    levels = torch.tensor(data=[[4, 4, 4, 4]])
    expected_selections = {"joint": ([[99.9992, 0.0, 100.0, 10.0], [0.0, 50.0, 0.0, 60.0], [0.0, 0.0, 10.0, 10.0]],
                                     [True, True, True, False]),
                           "per_level": ([[99.9989, 0.0, 100.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
                                         [True, True, False, False])}
    for proposal_selection, (expected_boxes, expected_valid_mask) in expected_selections.items():
        selected_boxes, _, _, valid_mask = select_best_proposals(
            proposal_scores=logits, proposal_boxes=boxes, proposal_levels=levels, level_sizes=[4],
            pre_nms_proposals=4, post_nms_proposals=4, nms_iou_threshold=0.7, input_image_size=100,
            suppress_per_image=True, proposal_selection=proposal_selection)
        assert valid_mask[0].tolist() == expected_valid_mask
        assert selected_boxes[0, :len(expected_boxes)].tolist() == expected_boxes


def test_per_level_selection_keeps_the_same_proposals_in_the_eager_and_export_forms() -> None:
    """
    Verifies on random batches of three levels that the selection level by level keeps exactly the same proposals with
    the suppression of the training, one call per image and level, and with that of the deployed detector, one call
    over the batch for every level, and that some proposals of every level are suppressed and kept.
    """
    generator = torch.Generator().manual_seed(0)
    level_sizes = [400, 100, 25]
    corners = torch.rand(size=(3, sum(level_sizes), 2), generator=generator) * 400
    sides = 8 + torch.rand(size=(3, sum(level_sizes), 2), generator=generator) * 100
    boxes = torch.cat(tensors=[corners, corners + sides], dim=2)
    logits = torch.randn(size=(3, sum(level_sizes)), generator=generator)
    levels = torch.cat(tensors=[torch.full(size=(3, level_size), fill_value=level_index)
                                for level_index, level_size in enumerate(level_sizes)], dim=1)
    selections = [select_best_proposals(proposal_scores=logits, proposal_boxes=boxes, proposal_levels=levels,
                                        level_sizes=level_sizes, pre_nms_proposals=60, post_nms_proposals=80,
                                        nms_iou_threshold=0.5, input_image_size=400,
                                        suppress_per_image=suppress_per_image, proposal_selection="per_level")
                  for suppress_per_image in [True, False]]
    for eager_output, export_output in zip(*selections):
        assert torch.equal(input=eager_output, other=export_output)

    # Every level keeps proposals, fewer than its candidates (25 at most for the smallest level)
    selected_levels, valid_mask = selections[0][2], selections[0][3]
    for level_index in range(len(level_sizes)):
        assert 0 < int(((selected_levels == level_index) & valid_mask).sum()) < 3 * min(60, level_sizes[level_index])


def test_unknown_selection_and_proposals_of_other_levels_are_refused() -> None:
    """
    Verifies that the selection refuses a value outside PROPOSAL_SELECTIONS, and that the proposal filter refuses
    proposals whose number differs from the anchors of its levels, which would group the anchors into wrong levels.
    """
    boxes = torch.tensor(data=[[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]])
    logits = torch.tensor(data=[[1.0, 2.0]])
    levels = torch.tensor(data=[[4, 5]])
    with pytest.raises(expected_exception=ValueError, match="Unknown proposal selection 'per_image'"):
        select_best_proposals(proposal_scores=logits, proposal_boxes=boxes, proposal_levels=levels, level_sizes=[1, 1],
                              pre_nms_proposals=1, post_nms_proposals=1, nms_iou_threshold=0.5, input_image_size=100,
                              suppress_per_image=True, proposal_selection="per_image")
    proposal_filter = RegionProposalFilter(configuration={
        "training": {"pre_nms_proposals": 1, "post_nms_proposals": 1},
        "inference": {"pre_nms_proposals": 1, "post_nms_proposals": 1}, "nms_iou_threshold": 0.5,
        "proposal_selection": "per_level"}, level_sizes=[1, 2]).eval()
    with pytest.raises(expected_exception=ValueError, match=r"received 2 proposals per image, expected the 3 anchors"):
        proposal_filter(proposal_boxes=boxes, proposal_scores=logits, proposal_origins=levels, input_image_size=100)
