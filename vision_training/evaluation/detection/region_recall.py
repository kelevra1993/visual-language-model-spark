"""
Recall of the anchors and of the proposals of the two-stage detector of the object detection and instance segmentation
tasks, the share of the objects the detection heads can find at all, split by object size (object_sizes.py). The
anchor recall is measured once per launch on the objects of the training split, and on the training, validation and
test splits by the 'compute_anchor_coverage' diagnosis mode, read from the annotations without loading any image: an
object whose best anchor overlaps it by less than the foreground threshold of the region proposal network only gets
that single best anchor as foreground. The proposal recall is measured at every evaluated checkpoint on the test split,
from the proposals the region proposal network gives the detection heads with its inference settings: an object no
proposal overlaps cannot be detected whatever the detection heads learn. Used by the detection trainer
(analyse_training_data, diagnose_data_splits and the evaluation of the checkpoints), whose tables, scalars and files
(anchor_recall.csv in the run folder, proposal_recall.csv in the evaluation folder) are built here. The tables write the
recalls as percentages ("94.03 %"), the counts of objects and proposals keeping their values, and the files and scalars
keep the recalls between 0 and 1.
"""

import numpy as np
import torch

from tqdm import tqdm
from typing import Any, Dict, List, Optional

from vision_training.core.contracts import EvaluationTable
from vision_training.data.detection.coco_dataset import CocoDetectionDataset, select_image_objects
from vision_training.data.preprocessing import preprocess_boxes
from vision_training.evaluation.detection.object_sizes import OBJECT_SIZE_NAMES, compute_box_areas, is_in_object_size
from vision_training.models.detection.anchors import Anchors
from vision_training.models.detection.box_operations import get_intersection_over_union

# Intersection over union thresholds of the proposal recall: the one of the headline mAP@0.5, and a stricter one
# telling whether the proposals are close enough for the detection heads to refine them into precise boxes
PROPOSAL_RECALL_THRESHOLDS = [0.5, 0.7]

# Key and displayed name of the group of all the objects, before the object sizes
ALL_OBJECTS_GROUP = "all"
ALL_OBJECTS_NAME = "All Objects"


def gather_object_boxes(dataset: CocoDetectionDataset, progress_description: Optional[str] = None) -> torch.Tensor:
    """
    Gathers the boxes of every object of a split in network input pixels, the objects the network learns
    (select_image_objects, without the background) placed like the dataset places them, for the anchor recall of the
    training split at launch and of every split in the 'compute_anchor_coverage' diagnosis mode. Read from the
    annotations and the image sizes of the labels file, without loading any image; the augmentations, random, are left
    out.

    Args:
        dataset (CocoDetectionDataset): The dataset of the split.
        progress_description (Optional[str]): The text of a progress bar over the images of the split, None for no
            progress bar. Defaults to None.

    Returns:
        torch.Tensor: The [N, 4] float32 [x1, y1, x2, y2] boxes of the N objects of the split.
    """
    object_boxes = []
    for image_index in tqdm(iterable=range(len(dataset)), desc=progress_description, unit="image",
                            disable=progress_description is None):
        # Objects of the image the network learns, in original pixels, the instance segmentation boxes coming from
        # the segmentation of every object like in the dataset
        image_information, image_annotations = dataset.get_image_record(image_index=image_index)
        bounding_boxes, _, _ = select_image_objects(annotations=image_annotations, label_mode=dataset.label_mode,
                                                    label_dictionary=dataset.label_dictionary,
                                                    boxes_from_segmentation=dataset.load_instance_masks)
        if not bounding_boxes:
            continue

        # Scaled and shifted like the image, into network input pixels
        object_boxes += preprocess_boxes(bounding_boxes=bounding_boxes, image_width=image_information["width"],
                                         image_height=image_information["height"], image_size=dataset.image_size,
                                         keep_ratio=dataset.keep_ratio)
    return torch.tensor(data=np.array(object=object_boxes, dtype=np.float32).reshape(-1, 4))


def compute_best_anchor_overlaps(object_boxes: torch.Tensor,
                                 level_anchor_generators: Dict[int, Anchors]) -> Dict[int, torch.Tensor]:
    """
    Finds, for every object of a split, the intersection over union of its best anchor in every level, for the anchor
    recall: the region proposal network makes an anchor foreground when it overlaps an object by at least its
    foreground threshold. Every level gives the overlap of the grid anchor nearest every object of each anchor shape
    (Anchors.compute_best_overlaps), exactly the best of all its anchors, on the CPU.

    Args:
        object_boxes (torch.Tensor): The [N, 4] boxes of the objects, in network input pixels.
        level_anchor_generators (Dict[int, Anchors]): The anchor generator of every level, keyed by block number
            (TwoStageDetector.get_level_anchor_generators).

    Returns:
        Dict[int, torch.Tensor]: The [N] float32 best overlap of every object in every level, on the CPU.
    """
    return {level: anchor_generator.compute_best_overlaps(boxes=object_boxes)
            for level, anchor_generator in level_anchor_generators.items()}


def compute_group_recalls(best_overlaps: torch.Tensor, object_areas: torch.Tensor,
                          intersection_over_union_threshold: float) -> Dict[str, Dict[str, Any]]:
    """
    Computes the share of the objects whose best anchor or proposal overlaps them by at least a threshold, for all the
    objects and for every object size, the recall of a group without objects being undefined.

    Args:
        best_overlaps (torch.Tensor): The [N] best overlap of every object.
        object_areas (torch.Tensor): The [N] box area of every object, in squared network input pixels.
        intersection_over_union_threshold (float): The minimum overlap for an object to be covered.

    Returns:
        Dict[str, Dict[str, Any]]: The 'objects' count and the 'recall' (None without objects) of the group of all the
            objects ('all') and of every object size.
    """
    covered = best_overlaps >= intersection_over_union_threshold
    group_masks = {ALL_OBJECTS_GROUP: torch.ones_like(input=covered)}
    group_masks.update({object_size: is_in_object_size(areas=object_areas, object_size=object_size)
                        for object_size in OBJECT_SIZE_NAMES})
    return {group: {"objects": int(group_mask.sum()),
                    "recall": float(covered[group_mask].float().mean()) if bool(group_mask.any()) else None}
            for group, group_mask in group_masks.items()}


def summarise_anchor_recall(best_anchor_overlaps: Dict[int, torch.Tensor], object_areas: torch.Tensor,
                            intersection_over_union_threshold: float, split_name: str,
                            threshold_setting: str) -> Dict[str, Any]:
    """
    Organises the anchor recall of a split, the training split at the launch of a detection run or every split in the
    'compute_anchor_coverage' diagnosis mode: the share of the objects with an anchor overlapping them by at least the
    threshold in every level and in all the levels together (the region proposal network assigns the objects to the
    anchors of all the levels at once), for all the objects and every object size, as the rows of anchor_recall.csv and
    as the table printed in the console, its recalls written as percentages.

    Args:
        best_anchor_overlaps (Dict[int, torch.Tensor]): The [N] best anchor overlap of every object in every level
            (compute_best_anchor_overlaps).
        object_areas (torch.Tensor): The [N] box area of every object, in squared network input pixels.
        intersection_over_union_threshold (float): The minimum overlap for an object to be covered.
        split_name (str): The name of the split in the title of the table, e.g. 'Training'.
        threshold_setting (str): The configuration setting giving the threshold, named in the legend, e.g.
            'RegionProposal.foreground_iou_threshold.min'.

    Returns:
        Dict[str, Any]: The 'rows' of anchor_recall.csv (one per group of objects: 'objects', 'count', then the
            'recall_block_<n>' of every level and the 'recall_all_levels') and the 'table' of the console.
    """
    levels = sorted(best_anchor_overlaps)
    all_level_overlaps = torch.stack(tensors=[best_anchor_overlaps[level] for level in levels]).max(dim=0).values

    # Recall of every group of objects in every level and in all of them
    level_recalls = {f"block_{level}": compute_group_recalls(
        best_overlaps=best_anchor_overlaps[level], object_areas=object_areas,
        intersection_over_union_threshold=intersection_over_union_threshold) for level in levels}
    level_recalls["all_levels"] = compute_group_recalls(
        best_overlaps=all_level_overlaps, object_areas=object_areas,
        intersection_over_union_threshold=intersection_over_union_threshold)

    # One row per group of objects, in the file and in the table
    rows, table_rows = [], []
    for group, group_name in [(ALL_OBJECTS_GROUP, ALL_OBJECTS_NAME)] + list(OBJECT_SIZE_NAMES.items()):
        rows.append({"objects": group, "count": level_recalls["all_levels"][group]["objects"]})
        rows[-1].update({f"recall_{level_key}": recalls[group]["recall"]
                         for level_key, recalls in level_recalls.items()})
        table_rows.append([group_name, rows[-1]["count"]] + [recalls[group]["recall"]
                                                            for recalls in level_recalls.values()])

    # The recall of every level and of all of them written as percentages, the objects of every group as a count
    columns = ["Objects", "Count"] + [f"Block {level}" for level in levels] + ["All Levels"]
    table: EvaluationTable = {
        "title": f"Anchor Recall Of The {split_name} Objects", "columns": columns, "rows": table_rows,
        "sort_column": None, "per_class": False, "percent_columns": columns[2:],
        "legend": f"Share of the objects whose best anchor overlaps them by at least "
                  f"{intersection_over_union_threshold:g} ({threshold_setting}), sizes by box area in network input "
                  f"pixels. An object under RegionProposal.foreground_iou_threshold.min in every level only gets its "
                  f"single best anchor as foreground. — : no object of the size."}
    return {"rows": rows, "table": table}


def create_proposal_recall_accumulator(levels: List[int]) -> Dict[str, Any]:
    """
    Creates the accumulator of the proposal recall of an evaluated checkpoint, filled image by image during the
    evaluation of the test split.

    Args:
        levels (List[int]): The block numbers of the detection levels.

    Returns:
        Dict[str, Any]: The empty 'best_overlaps' (per image, [G, 1 + L]: all the proposals, then those of every
            level) and 'object_areas' lists, the 'levels', the zero 'level_proposals' count of every level and the
            zero count of 'images'.
    """
    return {"best_overlaps": [], "object_areas": [], "levels": sorted(levels),
            "level_proposals": {level: 0 for level in sorted(levels)}, "images": 0}


def accumulate_proposal_recall(proposal_accumulator: Dict[str, Any], proposal_boxes: torch.Tensor,
                               proposal_levels: torch.Tensor, valid_proposal_mask: torch.Tensor,
                               ground_truth_boxes: torch.Tensor) -> None:
    """
    Adds one test image to the proposal recall: the best overlap of every object of the image with the valid
    proposals the region proposal network gives the detection heads (the padding copies left out), all of them and
    those of every level, computed on the device of the proposals, and the number of valid proposals of every level.

    Args:
        proposal_accumulator (Dict[str, Any]): The accumulator (create_proposal_recall_accumulator), filled in place.
        proposal_boxes (torch.Tensor): The [P, 4] filtered proposals of the image, in network input pixels.
        proposal_levels (torch.Tensor): The [P] block number of every proposal.
        valid_proposal_mask (torch.Tensor): The [P] mask of the valid proposals.
        ground_truth_boxes (torch.Tensor): The [G, 4] boxes of the objects of the image.
    """
    proposal_boxes, proposal_levels = proposal_boxes[valid_proposal_mask], proposal_levels[valid_proposal_mask]
    overlaps = get_intersection_over_union(boxes_1=ground_truth_boxes.to(device=proposal_boxes.device,
                                                                         dtype=torch.float32),
                                           boxes_2=proposal_boxes.to(dtype=torch.float32))

    # Best overlap of every object with all the proposals, then with those of every level (0 without any)
    level_masks = [torch.ones_like(input=proposal_levels, dtype=torch.bool)] + [
        proposal_levels == level for level in proposal_accumulator["levels"]]
    best_overlaps = torch.stack(tensors=[overlaps.masked_fill(mask=~level_mask, value=0.0).max(dim=1).values
                                         if overlaps.shape[1] > 0 else overlaps.new_zeros(size=(overlaps.shape[0],))
                                         for level_mask in level_masks], dim=1)
    proposal_accumulator["best_overlaps"].append(best_overlaps.cpu())
    proposal_accumulator["object_areas"].append(compute_box_areas(boxes=ground_truth_boxes.float()).cpu())

    # Valid proposals of every level, and the image counted for their mean
    for level, level_mask in zip(proposal_accumulator["levels"], level_masks[1:]):
        proposal_accumulator["level_proposals"][level] += int(level_mask.sum())
    proposal_accumulator["images"] += 1


def summarise_proposal_recall(proposal_accumulator: Dict[str, Any]) -> Dict[str, Any]:
    """
    Organises the proposal recall of an evaluated checkpoint for its report: the share of the test objects overlapped
    by a proposal by at least 0.5 and 0.7, for all the objects, every object size and the proposals of every level
    alone, with the mean number of proposals per image, as the table of the report (its recalls written as
    percentages), the TensorBoard scalars and the rows of proposal_recall.csv.

    Args:
        proposal_accumulator (Dict[str, Any]): The filled accumulator (accumulate_proposal_recall).

    Returns:
        Dict[str, Any]: The 'rows' of proposal_recall.csv ('group', 'objects', 'recall_50', 'recall_70',
            'proposals_per_image'), the 'table' of the report and its 'scalars'.
    """
    levels = proposal_accumulator["levels"]
    best_overlaps = torch.cat(tensors=proposal_accumulator["best_overlaps"] + [torch.zeros(size=(0, 1 + len(levels)))])
    object_areas = torch.cat(tensors=proposal_accumulator["object_areas"] + [torch.zeros(size=(0,))])
    number_images = max(proposal_accumulator["images"], 1)
    level_proposals_per_image = {level: proposal_accumulator["level_proposals"][level] / number_images
                                 for level in levels}

    # Recall of all the objects and of every size with all the proposals, at every threshold
    all_proposal_recalls = {threshold: compute_group_recalls(
        best_overlaps=best_overlaps[:, 0], object_areas=object_areas, intersection_over_union_threshold=threshold)
        for threshold in PROPOSAL_RECALL_THRESHOLDS}
    group_names = [(ALL_OBJECTS_GROUP, ALL_OBJECTS_NAME)] + list(OBJECT_SIZE_NAMES.items())
    rows = [{"group": group, "objects": all_proposal_recalls[0.5][group]["objects"],
             "recall_50": all_proposal_recalls[0.5][group]["recall"],
             "recall_70": all_proposal_recalls[0.7][group]["recall"],
             "proposals_per_image": sum(level_proposals_per_image.values()) if group == ALL_OBJECTS_GROUP else None}
            for group, _ in group_names]
    row_names = [group_name for _, group_name in group_names]

    # Recall of all the objects with the proposals of every level alone
    for level_index, level in enumerate(levels):
        level_recalls = {threshold: compute_group_recalls(
            best_overlaps=best_overlaps[:, 1 + level_index], object_areas=object_areas,
            intersection_over_union_threshold=threshold)[ALL_OBJECTS_GROUP] for threshold in PROPOSAL_RECALL_THRESHOLDS}
        rows.append({"group": f"block_{level}", "objects": level_recalls[0.5]["objects"],
                     "recall_50": level_recalls[0.5]["recall"], "recall_70": level_recalls[0.7]["recall"],
                     "proposals_per_image": level_proposals_per_image[level]})
        row_names.append(f"Block {level} Proposals")

    # Scalars of every row, the proposals per image only where they are counted
    scalars = {}
    for row in rows:
        for column_name, scalar_name in [("recall_50", "Proposal Recall@0.5"), ("recall_70", "Proposal Recall@0.7"),
                                         ("proposals_per_image", "Proposals Per Image")]:
            if row[column_name] is not None:
                scalars[f"{scalar_name}/{row['group']}"] = row[column_name]

    # The recalls written as percentages, the objects and the proposals per image as counts
    table: EvaluationTable = {
        "title": "Proposal Recall", "columns": ["Objects", "Count", "Recall@0.5", "Recall@0.7", "Proposals\nPer Image"],
        "rows": [[row_name, row["objects"], row["recall_50"], row["recall_70"], row["proposals_per_image"]]
                 for row_name, row in zip(row_names, rows, strict=True)],
        "sort_column": None, "per_class": False, "percent_columns": ["Recall@0.5", "Recall@0.7"],
        "legend": "Share of the test objects overlapped by a proposal of the region proposal network by at least the "
                  "IoU, sizes by box area in network input pixels. A block row keeps the proposals of that block "
                  "only. — : no object of the size."}
    return {"rows": rows, "table": table, "scalars": scalars}
