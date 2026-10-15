"""
Tests of vision_training/evaluation/detection/region_recall.py, the anchor recall measured at the launch of the
object detection and instance segmentation runs and the proposal recall of every evaluated checkpoint: the training
boxes read from the annotations (from the segmentation for the instance segmentation) and placed like the images,
the best anchor overlaps whatever the chunk size, the recall of all the objects and of every size, the rows and table
of anchor_recall.csv, and the proposal recall of the valid proposals only, with its rows, scalars and table.
"""

import json
import pytest
import torch

from pathlib import Path

from vision_training.data.detection.coco_dataset import CocoDetectionDataset
from vision_training.evaluation.detection.region_recall import (
    accumulate_proposal_recall, compute_best_anchor_overlaps, compute_group_recalls,
    create_proposal_recall_accumulator, gather_training_object_boxes, summarise_anchor_recall,
    summarise_proposal_recall)


def test_training_object_boxes_follow_the_preprocessing(tmp_path: Path) -> None:
    """
    Verifies that the boxes of the training objects are read from the annotations in image order, scaled and shifted
    into network input pixels like the letterboxed images, the crowd annotation, the category mapped to the background
    and the image without objects left out, without any image file being read.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    labels_file = tmp_path / "labels.json"
    labels_file.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "wide.png", "width": 40, "height": 20},
                   {"id": 2, "file_name": "square.png", "width": 80, "height": 80},
                   {"id": 3, "file_name": "empty.png", "width": 80, "height": 80}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 4, 4], "iscrowd": 0},
                        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 40, 20], "iscrowd": 1},
                        {"id": 3, "image_id": 1, "category_id": 3, "bbox": [10, 10, 5, 5], "iscrowd": 0},
                        {"id": 4, "image_id": 2, "category_id": 2, "bbox": [10, 10, 20, 30], "iscrowd": 0}],
        "categories": [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}, {"id": 3, "name": "ignored"}]}))
    dataset = CocoDetectionDataset(data_directory=str(tmp_path / "missing"), labels_file=str(labels_file),
                                   image_size=80, keep_ratio=True, label_mode="multi_class",
                                   label_dictionary={1: 1, 2: 2}, shuffle=False)

    # The 40 x 20 image is scaled twice with 20 rows of padding above it, the 80 x 80 one is kept as it is
    object_boxes = gather_training_object_boxes(dataset=dataset)
    assert object_boxes.dtype == torch.float32
    assert torch.equal(object_boxes, torch.tensor(data=[[4.0, 24.0, 12.0, 32.0], [10.0, 10.0, 30.0, 40.0]]))

    # A split without any learnt object gives an empty [0, 4] tensor
    dataset.images = [dataset.images[2]]
    assert gather_training_object_boxes(dataset=dataset).shape == (0, 4)


@pytest.mark.parametrize(argnames="chunk_size", argvalues=[1, 2048])
def test_best_anchor_overlaps_of_every_level(chunk_size: int) -> None:
    """
    Verifies that every object gets the overlap of its best anchor in every level, the half precision anchors being
    compared in float32, whatever the number of objects compared at once.

    Args:
        chunk_size (int): The number of objects compared with the anchors at once.
    """
    object_boxes = torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    level_anchors = {3: torch.tensor(data=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 40.0, 40.0]], dtype=torch.float16),
                     4: torch.tensor(data=[[0.0, 0.0, 20.0, 20.0]])}
    best_anchor_overlaps = compute_best_anchor_overlaps(object_boxes=object_boxes, level_anchors=level_anchors,
                                                        chunk_size=chunk_size)
    assert sorted(best_anchor_overlaps) == [3, 4]
    assert torch.allclose(best_anchor_overlaps[3], torch.tensor(data=[1.0, 0.25]))
    assert torch.allclose(best_anchor_overlaps[4], torch.tensor(data=[0.25, 0.0]))

    # Without objects, every level gives an empty overlap tensor
    empty_overlaps = compute_best_anchor_overlaps(object_boxes=torch.zeros(size=(0, 4)), level_anchors=level_anchors,
                                                  chunk_size=chunk_size)
    assert all(level_overlaps.shape == (0,) for level_overlaps in empty_overlaps.values())


def test_group_recalls_of_every_size() -> None:
    """
    Verifies the recall of all the objects and of every size at a threshold, the 32² area belonging to the medium
    objects and the recall of a size without objects being undefined.
    """
    group_recalls = compute_group_recalls(best_overlaps=torch.tensor(data=[0.8, 0.2, 0.9, 0.7]),
                                          object_areas=torch.tensor(data=[100.0, 2000.0, 500.0, 1024.0]),
                                          intersection_over_union_threshold=0.7)
    assert group_recalls["all"] == {"objects": 4, "recall": pytest.approx(expected=0.75)}
    assert group_recalls["small"] == {"objects": 2, "recall": pytest.approx(expected=1.0)}
    assert group_recalls["medium"] == {"objects": 2, "recall": pytest.approx(expected=0.5)}
    assert group_recalls["large"] == {"objects": 0, "recall": None}


def test_anchor_recall_rows_and_table() -> None:
    """
    Verifies the rows of anchor_recall.csv and the console table: the recall of every level alone, and of all the
    levels together where an object is covered by its best anchor of any level, for all the objects and every size.
    """
    anchor_recall = summarise_anchor_recall(
        best_anchor_overlaps={4: torch.tensor(data=[0.1, 0.75]), 3: torch.tensor(data=[0.8, 0.2])},
        object_areas=torch.tensor(data=[100.0, 10000.0]), foreground_threshold=0.7)
    assert anchor_recall["rows"] == [
        {"objects": "all", "count": 2, "recall_block_3": 0.5, "recall_block_4": 0.5, "recall_all_levels": 1.0},
        {"objects": "small", "count": 1, "recall_block_3": 1.0, "recall_block_4": 0.0, "recall_all_levels": 1.0},
        {"objects": "medium", "count": 0, "recall_block_3": None, "recall_block_4": None, "recall_all_levels": None},
        {"objects": "large", "count": 1, "recall_block_3": 0.0, "recall_block_4": 1.0, "recall_all_levels": 1.0}]

    # The table shows the same values under the displayed names, the levels in block order
    table = anchor_recall["table"]
    assert table["title"] == "Anchor Recall Of The Training Objects"
    assert table["columns"] == ["Objects", "Count", "Block 3", "Block 4", "All Levels"]
    assert table["rows"][0] == ["All Objects", 2, 0.5, 0.5, 1.0]
    assert [row[0] for row in table["rows"][1:]] == ["Small (< 32²)", "Medium (32² to 96²)", "Large (>= 96²)"]
    assert "0.7" in table["legend"] and not table["per_class"]


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


def test_proposal_recall_without_any_image() -> None:
    """
    Verifies that the proposal recall of an evaluation without any test image has no recall and no proposal, instead
    of failing on the empty accumulator.
    """
    proposal_recall = summarise_proposal_recall(
        proposal_accumulator=create_proposal_recall_accumulator(levels=[3]))
    assert all(row["objects"] == 0 and row["recall_50"] is None for row in proposal_recall["rows"])
    assert proposal_recall["scalars"] == {"Proposals Per Image/all": 0.0, "Proposals Per Image/block_3": 0.0}


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
    object_boxes = gather_training_object_boxes(dataset=dataset)
    assert torch.equal(object_boxes, torch.tensor(data=[[10.0, 20.0, 30.0, 50.0], [40.0, 5.0, 41.0, 15.0]]))
