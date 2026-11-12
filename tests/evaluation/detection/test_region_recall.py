"""
Tests of vision_training/evaluation/detection/region_recall.py, the anchor recall measured at the launch of the
object detection and instance segmentation runs (and on every split by the anchor coverage diagnosis mode) and the
proposal recall of every evaluated checkpoint: the boxes of an instance segmentation split read from the segmentation,
the best anchor overlaps of every level, the rows and table of anchor_recall.csv, and the proposal recall of the valid
proposals only, with its rows, scalars and table.
"""

import json
import pytest
import torch

from pathlib import Path

from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.evaluation.detection.region_recall import (
    accumulate_proposal_recall, compute_best_anchor_overlaps, create_proposal_recall_accumulator, gather_object_boxes,
    summarise_anchor_recall, summarise_proposal_recall)
from vision_training.models.detection.anchors import Anchors
from vision_training.models.detection.box_operations import get_intersection_over_union


def test_best_anchor_overlaps_of_every_level() -> None:
    """
    Verifies that every object gets the overlap of its best anchor in every level, equal to the highest overlap with
    every anchor of the level, the anchors being in float32 like those of the model (MODEL_DTYPE).
    """
    object_boxes = torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 36.0, 36.0], [50.0, 2.0, 63.0, 30.0]])
    level_anchor_generators = {
        3: Anchors(scales=[8.0, 16.0], aspect_ratios=[0.5, 1.0, 2.0], input_image_size=64, feature_map_size=8,
                   anchor_centres="cell_corners", dtype=torch.float32, device=torch.device("cpu")),
        4: Anchors(scales=[32.0], aspect_ratios=[1.0], input_image_size=64, feature_map_size=2,
                   anchor_centres="cell_corners", dtype=torch.float32, device=torch.device("cpu"))}
    best_anchor_overlaps = compute_best_anchor_overlaps(object_boxes=object_boxes,
                                                        level_anchor_generators=level_anchor_generators)
    assert sorted(best_anchor_overlaps) == [3, 4]
    for level, anchor_generator in level_anchor_generators.items():
        expected_overlaps = get_intersection_over_union(boxes_1=object_boxes, boxes_2=anchor_generator.anchors.float())
        assert best_anchor_overlaps[level].dtype == torch.float32
        assert torch.allclose(input=best_anchor_overlaps[level], other=expected_overlaps.max(dim=1).values, atol=1e-6)

    # Without objects, every level gives an empty overlap tensor
    empty_overlaps = compute_best_anchor_overlaps(object_boxes=torch.zeros(size=(0, 4)),
                                                  level_anchor_generators=level_anchor_generators)
    assert all(level_overlaps.shape == (0,) for level_overlaps in empty_overlaps.values())


def test_anchor_recall_rows_and_table() -> None:
    """
    Verifies the rows of anchor_recall.csv and the console table: the recall of every level alone, and of all the
    levels together where an object is covered by its best anchor of any level, for all the objects and every size.
    """
    anchor_recall = summarise_anchor_recall(
        best_anchor_overlaps={4: torch.tensor(data=[0.1, 0.75]), 3: torch.tensor(data=[0.8, 0.2])},
        object_areas=torch.tensor(data=[100.0, 10000.0]), intersection_over_union_threshold=0.7,
        split_name="Validation", threshold_setting="ExperimentConfiguration.diagnosis_modes.compute_anchor_coverage")
    assert anchor_recall["rows"] == [
        {"objects": "all", "count": 2, "recall_block_3": 0.5, "recall_block_4": 0.5, "recall_all_levels": 1.0},
        {"objects": "small", "count": 1, "recall_block_3": 1.0, "recall_block_4": 0.0, "recall_all_levels": 1.0},
        {"objects": "medium", "count": 0, "recall_block_3": None, "recall_block_4": None, "recall_all_levels": None},
        {"objects": "large", "count": 1, "recall_block_3": 0.0, "recall_block_4": 1.0, "recall_all_levels": 1.0}]

    # The table shows the same values under the displayed names, the levels in block order
    table = anchor_recall["table"]
    assert table["title"] == "Anchor Recall Of The Validation Objects"
    assert table["columns"] == ["Objects", "Count", "Block 3", "Block 4", "All Levels"]
    assert table["rows"][0] == ["All Objects", 2, 0.5, 0.5, 1.0]
    assert [row[0] for row in table["rows"][1:]] == ["Small (< 32²)", "Medium (32² to 96²)", "Large (>= 96²)"]
    assert "0.7 (ExperimentConfiguration.diagnosis_modes.compute_anchor_coverage)" in table["legend"]
    assert not table["per_class"]


def test_proposal_recall_of_the_valid_proposals() -> None:
    """
    Verifies the proposal recall of three test images: the padding copy of a proposal never covers an object, an
    image without any valid proposal leaves its object uncovered, an image without objects only counts its proposals,
    and the summary gives the recall at 0.5 and 0.7 of all the objects, of every size and of the proposals of every
    level, the mean valid proposals per image, the scalars of the defined values and the table of the report.
    """
    proposal_accumulator = create_proposal_recall_accumulator(levels=[4, 3])

    # A small object found exactly by a block 3 proposal, whose padding copy is labelled block 4, and a small object
    # overlapped by 0.625 by a block 4 proposal
    accumulate_proposal_recall(
        proposal_accumulator=proposal_accumulator,
        proposal_boxes=torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]),
        proposal_levels=torch.tensor(data=[3, 4, 4]), valid_proposal_mask=torch.tensor(data=[True, False, True]),
        ground_truth_boxes=torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 66.0, 60.0]]))
    assert torch.allclose(proposal_accumulator["best_overlaps"][0],
                          torch.tensor(data=[[1.0, 1.0, 0.0], [0.625, 0.0, 0.625]]))

    # A large object without any valid proposal, then an image without objects and one valid proposal
    accumulate_proposal_recall(
        proposal_accumulator=proposal_accumulator, proposal_boxes=torch.tensor(data=[[0.0, 0.0, 10.0, 10.0]]),
        proposal_levels=torch.tensor(data=[3]), valid_proposal_mask=torch.tensor(data=[False]),
        ground_truth_boxes=torch.tensor(data=[[0.0, 0.0, 100.0, 100.0]]))
    accumulate_proposal_recall(
        proposal_accumulator=proposal_accumulator, proposal_boxes=torch.tensor(data=[[0.0, 0.0, 10.0, 10.0]]),
        proposal_levels=torch.tensor(data=[3]), valid_proposal_mask=torch.tensor(data=[True]),
        ground_truth_boxes=torch.zeros(size=(0, 4)))
    assert proposal_accumulator["level_proposals"] == {3: 2, 4: 1} and proposal_accumulator["images"] == 3

    # Rows of proposal_recall.csv: all the objects, every size, then the proposals of every level alone
    proposal_recall = summarise_proposal_recall(proposal_accumulator=proposal_accumulator)
    assert proposal_recall["rows"] == [
        {"group": "all", "objects": 3, "recall_50": pytest.approx(expected=2 / 3),
         "recall_70": pytest.approx(expected=1 / 3), "proposals_per_image": pytest.approx(expected=1.0)},
        {"group": "small", "objects": 2, "recall_50": 1.0, "recall_70": 0.5, "proposals_per_image": None},
        {"group": "medium", "objects": 0, "recall_50": None, "recall_70": None, "proposals_per_image": None},
        {"group": "large", "objects": 1, "recall_50": 0.0, "recall_70": 0.0, "proposals_per_image": None},
        {"group": "block_3", "objects": 3, "recall_50": pytest.approx(expected=1 / 3),
         "recall_70": pytest.approx(expected=1 / 3), "proposals_per_image": pytest.approx(expected=2 / 3)},
        {"group": "block_4", "objects": 3, "recall_50": pytest.approx(expected=1 / 3), "recall_70": 0.0,
         "proposals_per_image": pytest.approx(expected=1 / 3)}]

    # Scalars of the defined values only, and the table of the report with the displayed names
    scalars = proposal_recall["scalars"]
    assert scalars["Proposal Recall@0.5/all"] == pytest.approx(expected=2 / 3)
    assert scalars["Proposals Per Image/block_4"] == pytest.approx(expected=1 / 3)
    assert "Proposal Recall@0.5/medium" not in scalars and "Proposals Per Image/small" not in scalars
    table = proposal_recall["table"]
    assert table["title"] == "Proposal Recall" and not table["per_class"]
    assert [row[0] for row in table["rows"]] == ["All Objects", "Small (< 32²)", "Medium (32² to 96²)",
                                                 "Large (>= 96²)", "Block 3 Proposals", "Block 4 Proposals"]


def test_instance_segmentation_object_boxes_come_from_the_segmentation(tmp_path: Path) -> None:
    """
    Verifies that the boxes of the training objects of an instance segmentation split, whose annotations have no
    'bbox', are read from their segmentation, without any image file being read.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = tmp_path / "labels.json"
    labels_file.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "square.png", "width": 80, "height": 80}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                         "segmentation": [[10.0, 20.0, 30.0, 20.0, 30.0, 50.0, 10.0, 50.0]]},
                        {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 0,
                         "segmentation": {"size": [80, 80], "counts": [80 * 40 + 5, 10, 80 * 40 - 15]}}],
        "categories": [{"id": 1, "name": "first"}]}))
    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "missing"), labels_file=str(labels_file),
                                   image_size=80, keep_ratio=True, label_mode="multi_class", label_dictionary={1: 1},
                                   load_instance_masks=True, shuffle=False)
    object_boxes = gather_object_boxes(dataset=dataset)
    assert torch.equal(object_boxes, torch.tensor(data=[[10.0, 20.0, 30.0, 50.0], [40.0, 5.0, 41.0, 15.0]]))
