"""
Inference figures of the object detection tutorials (the chapters on the anchors and proposals, on the training targets
and on the evaluation), made by tutorials/figures/object_detection/figures.py from the trained detector of the DET run
of tutorials/tutorial_sources.yaml, run on the CPU on the test image the series follows (SAMPLE_IMAGE_NAME of
detection_sources.py): the anchors of one cell of every detection level, the levels sharing a stride drawn on the same
cell, and the best anchor of every object (the data of the animated page
animations/03_object_detection/anchor_grid.html), the proposals of the image before and after the non-maximum
suppression of the proposal filter with the final detections, the labels the anchors of the image get for the region
proposal loss, and the proposals the detection heads are trained on, every count by level, with the objects the final
detections find, take for another class or miss, at the last checkpoint and at every compared evaluation; and, outside
the reused inference, the architecture facts of the detector (parameters, layers of the region proposal heads the
diagrams draw), read from the model built untrained from the configuration of the run (describe_model). The functions of
the repository do the work (forward_detection_stages, select_best_proposals, batch_assign_targets_to_anchors,
add_ground_truth_proposals, batch_assign_targets_to_proposals, sample_positive_and_negative_training_targets), so that
the figures show what the code computes. The inference is reused from the previous build unless forced, or unless that
build ran another run, other checkpoints, checkpoint files replaced since or another test image; a run read from the run
snapshots (tutorials/run_snapshots, no checkpoint) takes the inference committed under
figure_reuse/object_detection/inference of the snapshots by a build from the real runs, after checking its checkpoint
and test image.
"""

import json

import cv2
import torch

from pathlib import Path
from PIL import Image, ImageDraw
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import (SERIES_COLOURS, compute_file_fingerprint, read_reused_facts,
                                                read_run_configuration, restore_committed_figures, round_fact,
                                                write_reused_facts)
from tutorials.figures.object_detection.detection_sources import (BOX_OUTLINE_WIDTH, DETECTION_SAMPLE_ITERATIONS,
                                                                  SAMPLE_IMAGE_NAME, describe_class_name, draw_box,
                                                                  find_test_folder, hexadecimal_to_rgb, join_words,
                                                                  load_caption_fonts, load_run_model,
                                                                  read_category_names, read_image_annotations,
                                                                  read_network_image)
from tutorial_builder.layout import BACKGROUND_COLOUR as FRAME_BACKGROUND_COLOUR
from tutorial_builder.layout import CARD_COLOUR, DIM_TEXT_COLOUR, TEXT_COLOUR
from vision_training.evaluation.detection.detection_confusion import MATCHING_INTERSECTION_OVER_UNION_THRESHOLD
from vision_training.models.detection.box_operations import (clamp_boxes_to_image_boundaries,
                                                             get_intersection_over_union)
from vision_training.models.detection.region_proposer import (find_nonempty_proposals, select_best_proposals,
                                                              suppress_overlapping_proposals)
from vision_training.models.detection.target_assignment import (add_ground_truth_proposals,
                                                                batch_assign_targets_to_anchors,
                                                                batch_assign_targets_to_proposals,
                                                                sample_positive_and_negative_training_targets)
from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.trainers.checkpointing import get_checkpoint_file

# Seed of the random sampling of the anchors and proposals of the training targets, so that the counts of the figures
# stay the same from one build to the next
SAMPLING_SEED = 0

# Version of the facts of the inference, raised whenever a fact is added or removed, so that a build reusing the
# inference of an older build (inference_facts.json) makes it again instead of missing the new facts; 5 since the cell
# of the anchor scene follows the object nearest the image centre, the levels sharing a stride share their cell, and
# the objects found, taken for another class or missed are named at the last checkpoint and at every compared
# evaluation
INFERENCE_FACTS_VERSION = 5

# Objects of the line of the best anchor overlaps on the anchor scene, the rest going to a second line
OBJECT_LINE_LENGTH = 3

# Folder of the inference figures in the figures folder of the series, also the family committed under figure_reuse/
# of the run snapshots for the builds without the checkpoint
INFERENCE_FAMILY_FOLDER = "inference"

# Layout of the figures of three image panels: the side of every panel, the gap between them, and the bands of the
# headings above the panels and of their notes below them, the figure being 1720 x 700 pixels
PANEL_SIDE = 552
PANEL_GAP = 32
HEADING_HEIGHT = 64
NOTE_HEIGHT = 84
FIGURE_WIDTH = 1720
FIGURE_HEIGHT = 700

# Colour of the proposals of every detection level, the order of the levels of the diagrams (violet, blue, aqua and
# magenta, the colours of their bars in the storyboards)
LEVEL_COLOURS = {4: SERIES_COLOURS[4], 5: SERIES_COLOURS[0], 6: SERIES_COLOURS[1], 7: SERIES_COLOURS[3]}

# Colours of the labels of the anchors, drawn over the flat or textured backgrounds of the synthetic images:
# foreground, ignored and the sampled background
FOREGROUND_COLOUR = SERIES_COLOURS[0]
IGNORED_COLOUR = SERIES_COLOURS[2]
BACKGROUND_COLOUR = "#%02x%02x%02x" % FRAME_BACKGROUND_COLOUR


def read_image_objects(test_folder: Path, category_names: Dict[int, str]) -> Tuple[torch.Tensor, List[int], List[str]]:
    """
    Reads the objects of the test image the way the detection loader does: the COCO bbox [x, y, width, height] of
    every annotation turned into its [x1, y1, x2, y2] corners (select_image_objects), the image being already at the
    512 pixels of the network input, with the class of every object under the identity mapping of the synthetic shapes.

    Args:
        test_folder (Path): The test split.
        category_names (Dict[int, str]): The name of every category identifier.

    Returns:
        Tuple[torch.Tensor, List[int], List[str]]: The [N, 4] boxes, their classes and their names.
    """
    _, annotations = read_image_annotations(test_folder=test_folder, image_name=SAMPLE_IMAGE_NAME)
    boxes = torch.tensor(data=[[annotation["bbox"][0], annotation["bbox"][1],
                                annotation["bbox"][0] + annotation["bbox"][2],
                                annotation["bbox"][1] + annotation["bbox"][3]] for annotation in annotations],
                         dtype=torch.float32)
    classes = [annotation["category_id"] for annotation in annotations]
    return boxes, classes, [category_names[class_index] for class_index in classes]


def choose_anchor_cell_object(object_boxes: torch.Tensor, image_size: int) -> int:
    """
    Chooses the object of the test image whose best anchors give the cells the animated page of the anchors draws: the
    object whose box centre lies nearest the centre of the image, so that the anchors of every level grow from a cell
    in the middle of the image whatever the image shows, rather than from an object named by its class.

    Args:
        object_boxes (torch.Tensor): The [N, 4] boxes of the objects of the image.
        image_size (int): The side of the network input.

    Returns:
        int: The index of the chosen object.
    """
    box_centres = (object_boxes[:, :2] + object_boxes[:, 2:]) / 2.0
    distances = torch.linalg.vector_norm(input=box_centres - image_size / 2.0, dim=1)
    return int(distances.argmin())


def describe_level_anchors(model: TwoStageDetector, object_boxes: torch.Tensor,
                           object_names: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Describes the anchors of every detection level for the animated page of the anchors: the stride, cells, scales and
    anchor count of the level, with its anchors per cell and its numbers of scales and of aspect ratios that the
    narration counts, the anchors of one cell in the order they are stored (ratio by ratio, every scale within a
    ratio), and for every object of the image its best anchor over the levels with its overlap, the foreground
    threshold telling whether it gets a foreground anchor by overlap or only through the fallback of the assignment.
    The cell is that of the best anchor of the object nearest the image centre (choose_anchor_cell_object) in the first
    level of every stride, the following levels of the same stride (blocks 6 and 7 without the last max pooling)
    drawing their anchors on that same cell, as the page shows them; every level gets the index of its stride group.

    Args:
        model (TwoStageDetector): The trained detector.
        object_boxes (torch.Tensor): The [N, 4] boxes of the objects of the image.
        object_names (List[str]): Their class names.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]: The description of every level and of every object,
            and the index of the object whose cell is drawn.
    """
    level_anchors = model.get_level_anchors()
    cell_object_index = choose_anchor_cell_object(object_boxes=object_boxes, image_size=model.input_image_size)
    level_descriptions = []
    best_overlaps = {}
    stride_cells = {}
    for level, anchors in level_anchors.items():
        region_proposer = model.region_proposer_dictionary[str(level)]
        feature_map_size = region_proposer.feature_map_size
        stride = model.input_image_size // feature_map_size
        anchors_per_cell = region_proposer.number_anchors_per_location
        overlaps = get_intersection_over_union(boxes_1=object_boxes, boxes_2=anchors)
        best_overlaps[level] = overlaps.max(dim=1)

        # Cell of the best anchor of the chosen object in the first level of the stride, the anchors of a cell being
        # consecutive, kept for the following levels of the same stride
        if stride not in stride_cells:
            stride_cells[stride] = int(overlaps[cell_object_index].argmax()) // anchors_per_cell
        cell_index = stride_cells[stride]
        cell_anchors = anchors[cell_index * anchors_per_cell:(cell_index + 1) * anchors_per_cell]
        ratios = [float(ratio) for ratio in region_proposer.aspect_ratios for _ in region_proposer.scales]
        scales = [float(scale) for _ in region_proposer.aspect_ratios for scale in region_proposer.scales]
        level_descriptions.append({
            "name": str(level), "stride": stride, "cells": feature_map_size, "anchor_count": int(anchors.shape[0]),
            "anchors_per_cell": anchors_per_cell, "scale_count": len(region_proposer.scales),
            "ratio_count": len(region_proposer.aspect_ratios), "group": list(stride_cells).index(stride),
            "scales": [float(scale) for scale in region_proposer.scales],
            "cell": [cell_index % feature_map_size, cell_index // feature_map_size],
            "anchors": [{"box": [round_fact(value=coordinate) for coordinate in box.tolist()], "scale": scale,
                         "ratio": ratio} for box, scale, ratio in zip(cell_anchors, scales, ratios)]})

    # Best anchor of every object over the levels, the first level winning a tie
    object_descriptions = []
    for object_index, object_name in enumerate(object_names):
        best_level, best_value = None, -1.0
        for level in sorted(level_anchors):
            if float(best_overlaps[level].values[object_index]) > best_value:
                best_level, best_value = level, float(best_overlaps[level].values[object_index])
        best_anchor = level_anchors[best_level][int(best_overlaps[best_level].indices[object_index])]
        object_descriptions.append({
            "name": object_name, "box": [round_fact(value=value) for value in object_boxes[object_index].tolist()],
            "best_level": str(best_level), "best_iou": round_fact(value=best_value),
            "best_anchor": [round_fact(value=value) for value in best_anchor.tolist()],
            "foreground": best_value >= model.region_proposal_configuration["foreground_iou_threshold"]["min"],
            "level_iou": {str(level): round_fact(value=float(best_overlaps[level].values[object_index]))
                          for level in level_anchors}})
    return level_descriptions, object_descriptions, cell_object_index


def count_by_level(levels: torch.Tensor) -> Dict[str, int]:
    """
    Counts the proposals or anchors of every detection level, the breakdown the proposal filter figures and diagrams
    show.

    Args:
        levels (torch.Tensor): The level (block number) of every proposal or anchor.

    Returns:
        Dict[str, int]: The count of every level, keyed by block number.
    """
    return {str(level): int((levels == level).sum()) for level in sorted(LEVEL_COLOURS)}


def classify_image_objects(object_boxes: torch.Tensor, object_classes: List[int], detection_boxes: torch.Tensor,
                           detection_labels: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Tells for every object of the test image what the detections reaching the deployment threshold make of it, the
    outcomes the chapters name on the sample images and on the proposal figure, matched at the overlap of the confusion
    matrix (MATCHING_INTERSECTION_OVER_UNION_THRESHOLD): found when a detection of its class overlaps it that much,
    taken for another class when only detections of other classes do (the class of the most overlapping one given), and
    missed otherwise.

    Args:
        object_boxes (torch.Tensor): The [N, 4] boxes of the objects.
        object_classes (List[int]): Their classes.
        detection_boxes (torch.Tensor): The [D, 4] boxes of the detections reaching the deployment threshold.
        detection_labels (torch.Tensor): Their [D] classes.

    Returns:
        List[Dict[str, Any]]: For every object, its 'outcome' ("found", "wrong_class" or "missed") and, for a wrong
            class, the 'detected_class' of its most overlapping detection.
    """
    # No detection misses every object, the overlap of an empty set of boxes having no maximum
    if detection_boxes.shape[0] == 0:
        return [{"outcome": "missed"} for _ in object_classes]

    # Overlap of every object with every detection, those of the class of the object and those of the other classes
    overlaps = get_intersection_over_union(boxes_1=object_boxes, boxes_2=detection_boxes)
    same_class = torch.tensor(data=object_classes, dtype=torch.int64).unsqueeze(dim=1) == detection_labels.unsqueeze(
        dim=0)
    matched_overlaps = torch.where(same_class, overlaps, torch.zeros_like(input=overlaps))
    other_overlaps = torch.where(same_class, torch.zeros_like(input=overlaps), overlaps)
    outcomes = []
    for object_index in range(len(object_classes)):
        if float(matched_overlaps[object_index].max()) >= MATCHING_INTERSECTION_OVER_UNION_THRESHOLD:
            outcomes.append({"outcome": "found"})
        elif float(other_overlaps[object_index].max()) >= MATCHING_INTERSECTION_OVER_UNION_THRESHOLD:
            outcomes.append({"outcome": "wrong_class", "detected_class": int(
                detection_labels[int(other_overlaps[object_index].argmax())])})
        else:
            outcomes.append({"outcome": "missed"})
    return outcomes


def run_detections(model: TwoStageDetector, input_tensor: torch.Tensor) -> Tuple[Dict[str, Any], int]:
    """
    Runs every stage of a detector in evaluation mode on the test image and gives the outputs with the number of
    detections reaching the deployment threshold, the detections the sample images of the evaluations draw.

    Args:
        model (TwoStageDetector): The detector of a checkpoint (load_run_model).
        input_tensor (torch.Tensor): The [1, 3, H, W] input of the image (read_network_image).

    Returns:
        Tuple[Dict[str, Any], int]: The stages of forward_detection_stages and the number of detections.
    """
    with torch.no_grad():
        stages = model.forward_detection_stages(input_tensor=input_tensor)
    return stages, int(stages["model_output_dictionary"]["number_detections"][0])


def describe_sample_iterations(run_folder: Path, input_tensor: torch.Tensor, object_boxes: torch.Tensor,
                               object_classes: List[int], category_names: Dict[int, str]) -> Dict[str, Any]:
    """
    Runs the trained detector of every compared evaluation (DETECTION_SAMPLE_ITERATIONS of detection_sources.py) on the
    test image, as the sample images of those evaluations draw it, and gives the number of its detections reaching the
    deployment threshold and the objects they find, in the order of the iterations, so that the evaluation chapter
    names the shapes found at every compared iteration.

    Args:
        run_folder (Path): The folder of the DET run.
        input_tensor (torch.Tensor): The input of the test image.
        object_boxes (torch.Tensor): The [N, 4] boxes of its objects.
        object_classes (List[int]): Their classes.
        category_names (Dict[int, str]): The name of every category identifier.

    Returns:
        Dict[str, Any]: The facts, named inference.samples.*: the detection count, the found object count and the found
            objects as said (describe_class_name, joined), "none" when no object is found.
    """
    detection_counts, found_counts, found_texts = [], [], []
    for iteration in DETECTION_SAMPLE_ITERATIONS:
        stages, detection_count = run_detections(model=load_run_model(run_folder=run_folder, iteration=iteration),
                                                 input_tensor=input_tensor)
        outputs = stages["model_output_dictionary"]
        outcomes = classify_image_objects(
            object_boxes=object_boxes, object_classes=object_classes,
            detection_boxes=outputs["final_detection_boxes"][0][:detection_count].float(),
            detection_labels=outputs["final_detection_labels"][0][:detection_count])
        found_names = [describe_class_name(class_name=category_names[object_class]) for object_class, outcome in
                       zip(object_classes, outcomes) if outcome["outcome"] == "found"]
        detection_counts.append(detection_count)
        found_counts.append(len(found_names))
        found_texts.append(join_words(words=found_names) if found_names else "none")
    return {"inference.samples.iterations": list(DETECTION_SAMPLE_ITERATIONS),
            "inference.samples.detection_counts": detection_counts, "inference.samples.found_counts": found_counts,
            "inference.samples.found_texts": found_texts}


def filter_image_proposals(aggregated_proposals: Dict[str, torch.Tensor], pre_nms_proposals: int,
                           post_nms_proposals: int, nms_iou_threshold: float, input_image_size: int,
                           proposal_selection: str, level_sizes: List[int]) -> Dict[str, Any]:
    """
    Runs the proposal filter of the detector on the proposals of one image, with the counts of one mode (training or
    evaluation) and the selection of the run shown ('RegionProposal.proposal_selection', read from the trained
    detector): for every group of anchors (all the levels at once with "joint", every level on its own with
    "per_level"), the pre_nms_proposals best proposals by objectness, clamped to the image, as select_best_proposals
    takes them, and the survivors of their non-maximum suppression (suppress_overlapping_proposals, one call per image
    and group, as the eager filter runs it), "per_level" leaving out the proposals clamped to an empty box
    (find_nonempty_proposals); then the post_nms_proposals kept by select_best_proposals itself, with their valid mask.

    Args:
        aggregated_proposals (Dict[str, torch.Tensor]): The proposals of all the levels of the image
            (aggregate_proposals_and_anchors), with a batch of one image.
        pre_nms_proposals (int): The proposals kept before the suppression, in every group.
        post_nms_proposals (int): The proposals kept after it.
        nms_iou_threshold (float): The overlap of the suppression, RegionProposal.nms_iou_threshold.
        input_image_size (int): The side of the network input.
        proposal_selection (str): The selection of the run, one of PROPOSAL_SELECTIONS.
        level_sizes (List[int]): The number of anchors of every level, in the order of the concatenation of the
            proposals (RegionProposalFilter.level_sizes).

    Returns:
        Dict[str, Any]: The boxes and levels before the suppression, the survivors, and the kept boxes, levels and
            valid mask, with their counts by level.
    """
    scores = aggregated_proposals["classification_scores"].detach()
    boxes = aggregated_proposals["proposal_boxes"].detach()
    levels = aggregated_proposals["convolution_block_origin"]

    # Groups of anchors ranked and suppressed on their own, as select_best_proposals forms them
    group_sizes = [sum(level_sizes)] if proposal_selection == "joint" else level_sizes
    group_ends = [sum(group_sizes[:group_index + 1]) for group_index in range(len(group_sizes))]

    # Best proposals of every group before the suppression, clamped to the image, and the survivors of its suppression
    group_top_boxes, group_top_levels, group_kept_masks = [], [], []
    for group_start, group_end in zip([0] + group_ends[:-1], group_ends):
        top_scores, top_indices = torch.topk(input=torch.sigmoid(input=scores[:, group_start:group_end]),
                                             k=min(pre_nms_proposals, group_end - group_start), dim=1)
        top_indices = top_indices + group_start
        top_boxes = clamp_boxes_to_image_boundaries(boxes=torch.gather(input=boxes, dim=1, index=top_indices.unsqueeze(
            dim=-1).expand(-1, -1, 4)), input_image_size=input_image_size)
        nonempty_mask = find_nonempty_proposals(boxes=top_boxes, proposal_selection=proposal_selection)
        kept_mask = suppress_overlapping_proposals(
            boxes=top_boxes, scores=torch.where(condition=nonempty_mask, input=top_scores, other=-1.0),
            nms_iou_threshold=nms_iou_threshold, suppress_per_image=True)
        group_top_boxes.append(top_boxes)
        group_top_levels.append(torch.gather(input=levels, dim=1, index=top_indices))
        group_kept_masks.append(kept_mask & nonempty_mask)
    top_boxes, top_levels = torch.cat(tensors=group_top_boxes, dim=1), torch.cat(tensors=group_top_levels, dim=1)
    kept_mask = torch.cat(tensors=group_kept_masks, dim=1)

    # Proposals the filter keeps, computed by the function of the repository itself
    selected_boxes, _, selected_levels, valid_mask = select_best_proposals(
        proposal_scores=scores, proposal_boxes=boxes, proposal_levels=levels, level_sizes=level_sizes,
        pre_nms_proposals=pre_nms_proposals, post_nms_proposals=post_nms_proposals,
        nms_iou_threshold=nms_iou_threshold, input_image_size=input_image_size, suppress_per_image=True,
        proposal_selection=proposal_selection)
    return {"top_boxes": top_boxes[0], "top_levels": top_levels[0], "top_by_level": count_by_level(
                levels=top_levels[0]),
            "survivor_count": int(kept_mask.sum()), "survivors_by_level": count_by_level(
                levels=top_levels[0][kept_mask[0]]),
            "kept_boxes": selected_boxes[0], "kept_levels": selected_levels[0], "valid_mask": valid_mask[0],
            "kept_by_level": count_by_level(levels=selected_levels[0][valid_mask[0]]),
            "valid_count": int(valid_mask.sum())}


def compute_anchor_labels(model: TwoStageDetector, aggregated_proposals: Dict[str, torch.Tensor],
                          object_boxes: torch.Tensor) -> Dict[str, Any]:
    """
    Labels the anchors of the image for the region proposal loss the way assign_region_proposal_targets does
    (batch_assign_targets_to_anchors with the RegionProposal thresholds of the run, then the sampling of
    number_training_positives among total_training_samples anchors), and counts the foreground, background and ignored
    anchors, the foreground ones by level and those that reach the foreground threshold by their overlap, and the
    ignored ones in the band between the background maximum and the foreground minimum apart from those below the
    background minimum (the anchors overlapping no object when that minimum is above zero).

    Args:
        model (TwoStageDetector): The trained detector, whose configuration gives the thresholds and counts.
        aggregated_proposals (Dict[str, torch.Tensor]): The anchors and levels of the image.
        object_boxes (torch.Tensor): The [N, 4] boxes of the objects of the image.

    Returns:
        Dict[str, Any]: The label of every anchor, the sampled masks, and the counts.
    """
    configuration = model.region_proposal_configuration
    _, anchor_labels = batch_assign_targets_to_anchors(
        batched_ground_truth_boxes=[object_boxes], batched_anchors=aggregated_proposals["anchors"],
        background_iou_threshold=configuration["background_iou_threshold"],
        foreground_iou_threshold=configuration["foreground_iou_threshold"])

    # Sampled anchors of the loss, drawn with a fixed seed so that the counts do not change between builds
    torch.manual_seed(seed=SAMPLING_SEED)
    sampled_positive_mask, sampled_negative_mask = sample_positive_and_negative_training_targets(
        labels=anchor_labels, desired_positives=configuration["number_training_positives"],
        desired_total=configuration["total_training_samples"])

    # Foreground anchors reaching the threshold by their overlap, the others coming from the best anchor fallback
    overlaps = get_intersection_over_union(boxes_1=object_boxes, boxes_2=aggregated_proposals["anchors"][0])
    best_overlaps = overlaps.max(dim=0).values
    foreground_mask = anchor_labels[0] == 1.0
    overlap_foreground = int((best_overlaps >= configuration["foreground_iou_threshold"]["min"]).sum())

    # Ignored anchors below the background minimum, the others lying in the band below the foreground minimum
    ignored_mask = anchor_labels[0] == -1.0
    ignored_low_mask = ignored_mask & (best_overlaps < configuration["background_iou_threshold"]["min"])
    return {"labels": anchor_labels[0], "sampled_negative_mask": sampled_negative_mask[0],
            "ignored_band_mask": ignored_mask & ~ignored_low_mask,
            "foreground": int(foreground_mask.sum()), "background": int((anchor_labels[0] == 0.0).sum()),
            "ignored": int(ignored_mask.sum()), "ignored_low": int(ignored_low_mask.sum()),
            "ignored_band": int((ignored_mask & ~ignored_low_mask).sum()), "overlap_foreground": overlap_foreground,
            "fallback_foreground": int(foreground_mask.sum()) - overlap_foreground,
            "foreground_by_level": count_by_level(levels=aggregated_proposals["convolution_block_origin"][0][
                foreground_mask]),
            "sampled_positives": int(sampled_positive_mask.sum()),
            "sampled_negatives": int(sampled_negative_mask.sum())}


def compute_detector_targets(model: TwoStageDetector, training_filter: Dict[str, Any], object_boxes: torch.Tensor,
                             object_classes: List[int]) -> Dict[str, Any]:
    """
    Builds the training targets of the detection heads for the image the way _prepare_detector_inputs does, from the
    proposals of the training filter: the ground truth boxes written into the last slots of the proposals at the level
    of their best anchor (add_ground_truth_proposals), the class of every proposal overlapping an object by the
    foreground threshold of the detector (batch_assign_targets_to_proposals), and the sampling of
    number_training_positives among total_training_samples proposals.

    Args:
        model (TwoStageDetector): The trained detector, whose configuration gives the thresholds and counts.
        training_filter (Dict[str, Any]): The proposals of the training filter (filter_image_proposals).
        object_boxes (torch.Tensor): The [N, 4] boxes of the objects.
        object_classes (List[int]): Their classes.

    Returns:
        Dict[str, Any]: The counts of the positive proposals before and after the sampling, of the sampled negatives,
            and of the sampled proposals by level.
    """
    configuration = model.detector_configuration
    proposals, proposal_levels, _ = add_ground_truth_proposals(
        batched_ground_truth_boxes=[object_boxes], batched_proposals=training_filter["kept_boxes"].unsqueeze(dim=0),
        batched_proposal_levels=training_filter["kept_levels"].unsqueeze(dim=0),
        level_anchors=model.get_level_anchors())
    _, proposal_labels, _ = batch_assign_targets_to_proposals(
        batched_ground_truth_boxes=[object_boxes], batched_proposals=proposals,
        batched_ground_truth_labels=[torch.tensor(data=object_classes, dtype=torch.int64)],
        foreground_iou_threshold=configuration["foreground_iou_threshold"])

    # Sampled proposals of the detection losses, with a fixed seed
    torch.manual_seed(seed=SAMPLING_SEED)
    sampled_positive_mask, sampled_negative_mask = sample_positive_and_negative_training_targets(
        labels=proposal_labels, desired_positives=configuration["number_training_positives"],
        desired_total=configuration["total_training_samples"])
    sampled_mask = sampled_positive_mask | sampled_negative_mask
    return {"written_ground_truths": int(object_boxes.shape[0]), "proposal_count": int(proposals.shape[1]),
            "positives": int((proposal_labels > 0).sum()), "sampled_positives": int(sampled_positive_mask.sum()),
            "sampled_negatives": int(sampled_negative_mask.sum()),
            "sampled_by_level": count_by_level(levels=proposal_levels[sampled_mask])}


def draw_panel_figure(image: Image.Image, panels: List[Dict[str, Any]], output_path: Path) -> List[List[int]]:
    """
    Draws a figure of three panels of the test image (the proposal filter and the anchor label figures of the chapters
    on the proposals and on the training targets), each with its heading above it, its note below it, and its
    boxes: those with a transparency drawn thin on a transparent layer (the hundreds of proposals or anchors of a
    panel), the others opaque with their caption (the objects and detections). The figure has the size of the card of
    an image scene.

    Args:
        image (Image.Image): The RGB test image, at the network input size.
        panels (List[Dict[str, Any]]): Every panel: 'heading', 'note' and 'boxes', a list of dictionaries with the
            'box' in network input pixels, its RGB 'colour', and either an 'alpha' (with an optional line 'width') or
            a 'caption'.
        output_path (Path): The PNG file of the figure.

    Returns:
        List[List[int]]: The box of every panel image in the figure, for the highlights and views of the scenes.
    """
    canvas = Image.new(mode="RGB", size=(FIGURE_WIDTH, FIGURE_HEIGHT), color=CARD_COLOUR)
    heading_font, caption_font = load_caption_fonts()
    scale = PANEL_SIDE / image.width
    panel_boxes = []
    for panel_index, panel in enumerate(panels):
        left = panel_index * (PANEL_SIDE + PANEL_GAP)
        canvas.paste(im=image.resize(size=(PANEL_SIDE, PANEL_SIDE), resample=Image.Resampling.LANCZOS),
                     box=(left, HEADING_HEIGHT))

        # Thin boxes with a transparency on a layer of the size of the panel, which clips the boxes leaving the
        # image, then the opaque captioned boxes on top
        layer = Image.new(mode="RGBA", size=(PANEL_SIDE, PANEL_SIDE), color=(0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(im=layer)
        for box_entry in [entry for entry in panel["boxes"] if "alpha" in entry]:
            box = box_entry["box"]
            layer_draw.rectangle(xy=(box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale),
                                 outline=box_entry["colour"] + (int(255 * box_entry["alpha"]),),
                                 width=box_entry.get("width", 1))
        panel_region = (left, HEADING_HEIGHT, left + PANEL_SIDE, HEADING_HEIGHT + PANEL_SIDE)
        canvas.paste(im=Image.alpha_composite(im1=canvas.crop(box=panel_region).convert(mode="RGBA"),
                                              im2=layer).convert(mode="RGB"), box=panel_region[:2])
        draw = ImageDraw.Draw(im=canvas)
        for box_entry in [entry for entry in panel["boxes"] if "alpha" not in entry]:
            draw_box(draw=draw, box=box_entry["box"], colour=box_entry["colour"], width=BOX_OUTLINE_WIDTH,
                     caption=box_entry.get("caption", ""), font=caption_font, offset=(left, HEADING_HEIGHT),
                     scale=scale, caption_below=box_entry.get("caption_below", False))

        # Heading above the panel and note below it
        draw.text(xy=(left, HEADING_HEIGHT // 2), text=panel["heading"], fill=TEXT_COLOUR, font=heading_font,
                  anchor="lm")
        draw.text(xy=(left, HEADING_HEIGHT + PANEL_SIDE + NOTE_HEIGHT // 2), text=panel["note"], fill=DIM_TEXT_COLOUR,
                  font=caption_font, anchor="lm")
        panel_boxes.append([left, HEADING_HEIGHT, left + PANEL_SIDE, HEADING_HEIGHT + PANEL_SIDE])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fp=output_path)
    return panel_boxes


def format_level_counts(counts: Dict[str, int]) -> str:
    """
    Writes the counts of the levels as the note of a figure panel, short enough for four levels under a panel, e.g.
    "blocks 4/5/6/7: 1000 / 1000 / 384 / 576".

    Args:
        counts (Dict[str, int]): The count of every level.

    Returns:
        str: The note.
    """
    return f"blocks {'/'.join(counts)}: {' / '.join(str(count) for count in counts.values())}"


def count_parameters(module: torch.nn.Module) -> int:
    """
    Counts the parameters of a component of the detector, the numbers of the model size panel of the launch.

    Args:
        module (torch.nn.Module): The component.

    Returns:
        int: Its number of parameters.
    """
    return sum(parameter.numel() for parameter in module.parameters())


def describe_model(model: TwoStageDetector) -> Dict[str, Any]:
    """
    Gives the facts of the architecture of the detector of the DET run: the parameters of every component (the neck
    among them) and of the heads of every level, the RoIAlign settings and the flattened size of the detection heads,
    the channels of the maps the heads of every level read, the kernels and output channels of the layers of the region
    proposal heads that the diagrams of the first chapter draw, and the stride of the max pooling dividing the
    resolution of the backbone from one block to the next. None of them depends on the weights, so figures.py gives it
    the model built untrained from the configuration of the run (load_untrained_run_model), outside the reused
    inference, and the facts come out the same with or without the checkpoint.

    Args:
        model (TwoStageDetector): The detector of the run, trained or untrained.

    Returns:
        Dict[str, Any]: The facts, named inference.model.*.

    Raises:
        ValueError: If the backbone has no max pooling layer, whose stride the narration quotes.
    """
    detection_head = model.detector_heads_dictionary["5"]
    region_proposal_head = model.region_proposer_dictionary["5"]

    # Max pooling of the backbone blocks, every block but the last halving the resolution with the same layer
    pooling_layers = [module for module in model.backbone.modules() if isinstance(module, torch.nn.MaxPool2d)]
    if not pooling_layers:
        raise ValueError("The backbone of the detector has no max pooling layer, whose stride the chapter on the "
                         "levels quotes.")
    pooling_stride = pooling_layers[0].stride

    # Parameters of every component and of the heads of block 5, the RoIAlign and flattened size of the detection
    # heads, the classes, the stride of the max pooling and the region proposal layers the diagrams draw
    facts = {"inference.model.total_parameters": count_parameters(module=model),
             "inference.model.backbone_parameters": count_parameters(module=model.backbone),
             "inference.model.region_proposal_parameters": count_parameters(module=model.region_proposer_dictionary),
             "inference.model.region_proposal_level_parameters": count_parameters(
                 module=model.region_proposer_dictionary["5"]),
             "inference.model.detection_parameters": count_parameters(module=model.detector_heads_dictionary),
             "inference.model.detection_level_parameters": count_parameters(module=detection_head),
             "inference.model.roi_align_pool_size": detection_head.roi_align_pool_size,
             "inference.model.flattened_dimension": detection_head.roi_align_pool_size ** 2 *
                                                    detection_head.convolutional_block.output_channels,
             "inference.model.number_classes": model.number_classes,
             "inference.model.backbone_pooling_stride": int(pooling_stride if isinstance(pooling_stride, int) else
                                                            pooling_stride[0]),
             # Kernel and output channels of the last convolution of the region proposal head of block 5, then of its
             # objectness and box regression output layers
             "inference.model.region_proposal_last_kernel_size":
                 region_proposal_head.region_proposal_last_convolution.kernel_size[0],
             "inference.model.region_proposal_last_channels":
                 region_proposal_head.region_proposal_last_convolution.out_channels,
             "inference.model.region_proposal_output_kernel_size":
                 region_proposal_head.region_proposal_classifier.kernel_size[0],
             "inference.model.region_proposal_objectness_channels":
                 region_proposal_head.region_proposal_classifier.out_channels,
             "inference.model.region_proposal_regression_channels":
                 region_proposal_head.region_proposal_bounding_box_regressor.out_channels}

    # Neck between the backbone blocks and the heads, without parameters when its type is none
    facts["inference.model.neck_type"] = model.neck.neck_type
    facts["inference.model.neck_parameters"] = count_parameters(module=model.neck)

    # Spatial scale and pool of the RoIAlign of every level, the feature map side over the input side, and the channels
    # of the map the heads of the level read, after the neck (those of the backbone block without neck)
    for level, head in model.detector_heads_dictionary.items():
        facts[f"inference.model.level_{level}.spatial_scale"] = round_fact(value=head.spatial_scale)
        facts[f"inference.model.level_{level}.roi_align_pool_size"] = head.roi_align_pool_size
        facts[f"inference.model.level_{level}.feature_map_size"] = head.feature_map_size
        facts[f"inference.model.level_{level}.head_input_channels"] = model.detection_input_information[level][
            "number_channels"]
    return facts


def read_committed_inference_facts(output_folder: Path, facts_path: Path, checkpoint_name: str) -> Dict[str, Any]:
    """
    Gives the inference of the DET run for a build without its checkpoint (a build from the run snapshots of
    tutorials/run_snapshots): the inference figures and facts committed under figure_reuse/object_detection/inference
    of the snapshots by a build from the real runs, copied into the figures folder (restore_committed_figures) and read
    back like a reused inference, so that the facts and figures are those of the real checkpoint.

    Args:
        output_folder (Path): The figures folder of the series.
        facts_path (Path): The inference_facts.json of the inference figures in it.
        checkpoint_name (str): The run folder name and checkpoint iteration the build reads, as
            <run name>/Iteration_<iteration>.

    Returns:
        Dict[str, Any]: The facts of the inference, named inference.*.

    Raises:
        FileNotFoundError: If no committed inference is found, or one of the figures it lists is missing.
        ValueError: If the inference found was written by another version of the facts than INFERENCE_FACTS_VERSION,
            from another checkpoint or on another test image than SAMPLE_IMAGE_NAME.
    """
    restore_committed_figures(output_folder=output_folder, family_folder=INFERENCE_FAMILY_FOLDER)
    reused_facts = read_reused_facts(facts_path=facts_path, required_paths=[], force=False)
    if reused_facts is None:
        raise FileNotFoundError(f"No checkpoint in the DET run and no committed inference for {facts_path}: build the "
                                f"object_detection figures from the real runs, then write the snapshots again "
                                f"(tutorials/make_run_snapshots.py --series object_detection).")

    # A committed inference of another version of the facts would give missing or stale facts
    if reused_facts["inference.facts_version"] != INFERENCE_FACTS_VERSION:
        raise ValueError(f"The inference of {facts_path} has facts version {reused_facts['inference.facts_version']}, "
                         f"not {INFERENCE_FACTS_VERSION}: empty the figures folder of the build, or build the "
                         f"object_detection figures from the real runs and write the snapshots again.")

    # A committed inference of another checkpoint or test image would describe what the chapters no longer show
    if (reused_facts["inference.checkpoint_name"] != checkpoint_name
            or reused_facts["inference.image_name"] != SAMPLE_IMAGE_NAME):
        raise ValueError(f"The inference of {facts_path} was made from {reused_facts['inference.checkpoint_name']} on "
                         f"{reused_facts['inference.image_name']}, not from {checkpoint_name} on {SAMPLE_IMAGE_NAME}: "
                         f"build the object_detection figures from the real runs and write the snapshots again.")
    return reused_facts


def make_detector_inference_figures(output_folder: Path, run_folder: Path, iteration: int,
                                    force: bool) -> Dict[str, Any]:
    """
    Runs the trained detector of the DET run on the CPU on the test image of the series, draws the proposal filter and
    anchor label figures, writes the image the animated page of the anchors shows, names the objects the detections
    find, take for another class or miss (at the checkpoint and at every compared evaluation), and gives their facts,
    reused from the previous build unless forced; a run without its checkpoint (read from the run snapshots) gives the
    inference committed in the snapshots instead (read_committed_inference_facts).

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the DET run of tutorial_sources.yaml.
        iteration (int): The checkpoint whose trained weights are run, the last evaluated one of the run.
        force (bool): Whether the inference is run again even when the previous build left its figures.

    Returns:
        Dict[str, Any]: The facts of the inference, named inference.*.
    """
    figure_folder = output_folder / INFERENCE_FAMILY_FOLDER
    facts_path = figure_folder / "inference_facts.json"

    # A run without the checkpoint (read from the run snapshots) takes the inference committed by a build from the
    # real runs, whose facts keep the fingerprint of the real checkpoint
    checkpoint_path = get_checkpoint_file(weights_directory=run_folder / "Weights", iteration=iteration)
    checkpoint_name = f"{run_folder.name}/Iteration_{iteration}"
    if not checkpoint_path.exists():
        if force:
            print(f"No checkpoint {checkpoint_path}: --force cannot run the detector again, the committed inference "
                  f"of the run snapshots is reused")
        return read_committed_inference_facts(output_folder=output_folder, facts_path=facts_path,
                                              checkpoint_name=checkpoint_name)

    # An inference left by a build writing an older version of the facts, or made from another run, another checkpoint,
    # checkpoint files replaced since (a run trained again under the same name) or another test image, is made again,
    # as when forced; the checkpoints of the compared evaluations are run too
    checkpoint_fingerprint = compute_file_fingerprint(file_paths=[checkpoint_path] + [
        get_checkpoint_file(weights_directory=run_folder / "Weights", iteration=sample_iteration)
        for sample_iteration in DETECTION_SAMPLE_ITERATIONS])
    inference_key = {"inference.facts_version": INFERENCE_FACTS_VERSION, "inference.checkpoint_name": checkpoint_name,
                     "inference.checkpoint_fingerprint": checkpoint_fingerprint,
                     "inference.image_name": SAMPLE_IMAGE_NAME}
    previous_facts = json.loads(facts_path.read_text(encoding="utf-8"))["facts"] if facts_path.exists() else {}
    outdated = facts_path.exists() and any(previous_facts.get(fact_name) != value
                                           for fact_name, value in inference_key.items())
    reused_facts = read_reused_facts(facts_path=facts_path, required_paths=[], force=force or outdated)
    if reused_facts is not None:
        return reused_facts

    # The trained detector, the test image and its objects
    model = load_run_model(run_folder=run_folder, iteration=iteration)
    test_folder = find_test_folder(run_configuration=read_run_configuration(run_folder=run_folder))
    category_names = read_category_names(test_folder=test_folder)
    object_boxes, object_classes, object_names = read_image_objects(test_folder=test_folder,
                                                                    category_names=category_names)
    image, input_tensor = read_network_image(test_folder=test_folder, image_name=SAMPLE_IMAGE_NAME)
    rgb_image = Image.fromarray(obj=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB))
    figure_folder.mkdir(parents=True, exist_ok=True)
    image_path = figure_folder / SAMPLE_IMAGE_NAME
    rgb_image.save(fp=image_path)

    # Every stage of the detector in evaluation mode, then the proposals of all the levels of the image
    stages, detection_count = run_detections(model=model, input_tensor=input_tensor)
    with torch.no_grad():
        aggregated_proposals = model.aggregate_proposals_and_anchors(
            region_proposal_output_tensor_dictionary=model._forward_region_proposals(
                backbone_output_tensor_dictionary=stages["backbone_output_tensor_dictionary"]))
    outputs = stages["model_output_dictionary"]

    # Proposal filter of the image with the evaluation counts, the ones of the detections, and with the training ones,
    # in the proposal selection of the run over the anchors of its levels
    filter_configuration = model.region_proposal_configuration
    proposal_selection = model.region_proposal_filter.proposal_selection
    level_sizes = model.region_proposal_filter.level_sizes
    evaluation_filter = filter_image_proposals(
        aggregated_proposals=aggregated_proposals, pre_nms_proposals=filter_configuration["inference"][
            "pre_nms_proposals"], post_nms_proposals=filter_configuration["inference"]["post_nms_proposals"],
        nms_iou_threshold=filter_configuration["nms_iou_threshold"], input_image_size=model.input_image_size,
        proposal_selection=proposal_selection, level_sizes=level_sizes)
    training_filter = filter_image_proposals(
        aggregated_proposals=aggregated_proposals, pre_nms_proposals=filter_configuration["training"][
            "pre_nms_proposals"], post_nms_proposals=filter_configuration["training"]["post_nms_proposals"],
        nms_iou_threshold=filter_configuration["nms_iou_threshold"], input_image_size=model.input_image_size,
        proposal_selection=proposal_selection, level_sizes=level_sizes)

    # Anchors of the levels, labels of the anchors and targets of the detection heads
    level_descriptions, object_descriptions, cell_object_index = describe_level_anchors(
        model=model, object_boxes=object_boxes, object_names=object_names)
    anchor_labels = compute_anchor_labels(model=model, aggregated_proposals=aggregated_proposals,
                                          object_boxes=object_boxes)
    detector_targets = compute_detector_targets(model=model, training_filter=training_filter,
                                                object_boxes=object_boxes, object_classes=object_classes)

    # Final detections reaching the deployment threshold, and what they make of every object
    detections = [{"name": category_names[int(label)], "score": float(score), "box": box.tolist()}
                  for box, score, label in zip(outputs["final_detection_boxes"][0][:detection_count],
                                               outputs["final_detection_scores"][0][:detection_count],
                                               outputs["final_detection_labels"][0][:detection_count])]
    object_outcomes = classify_image_objects(
        object_boxes=object_boxes, object_classes=object_classes,
        detection_boxes=outputs["final_detection_boxes"][0][:detection_count].float(),
        detection_labels=outputs["final_detection_labels"][0][:detection_count])

    # Proposal filter figure: the best proposals before the suppression, those kept after it, and the detections
    white = TEXT_COLOUR
    object_entries = [{"box": box.tolist(), "colour": white, "caption": ""} for box in object_boxes]
    top_entries = [{"box": box.tolist(), "colour": hexadecimal_to_rgb(colour=LEVEL_COLOURS[int(level)]), "alpha": 0.55}
                   for box, level in zip(evaluation_filter["top_boxes"], evaluation_filter["top_levels"])]
    kept_entries = [{"box": box.tolist(), "colour": hexadecimal_to_rgb(colour=LEVEL_COLOURS[int(level)]), "alpha": 0.8}
                    for box, level, valid in zip(evaluation_filter["kept_boxes"], evaluation_filter["kept_levels"],
                                                 evaluation_filter["valid_mask"]) if bool(valid)]
    detection_entries = [{"box": detection["box"], "colour": hexadecimal_to_rgb(colour=SERIES_COLOURS[2]),
                          "caption": f"{detection['name']} {detection['score']:.2f}"} for detection in detections]

    # Caption of a detection whose box starts within a few pixels of a more confident one written below its box
    for detection_index, entry in enumerate(detection_entries):
        entry["caption_below"] = any(abs(entry["box"][0] - other["box"][0]) < 24 and
                                     abs(entry["box"][1] - other["box"][1]) < 24
                                     for other in detection_entries[:detection_index])

    # Heading of the best proposals naming the per-level selection, whose best proposals are those of every level
    selection_wording = "by objectness" if proposal_selection == "joint" else "by level"
    proposal_panel_boxes = draw_panel_figure(image=rgb_image, panels=[
        {"heading": f"{evaluation_filter['top_boxes'].shape[0]} best of {aggregated_proposals['anchors'].shape[1]} "
                    f"{selection_wording}", "note": format_level_counts(counts=evaluation_filter["top_by_level"]),
         "boxes": top_entries},
        {"heading": f"{evaluation_filter['valid_count']} kept after the NMS at "
                    f"{filter_configuration['nms_iou_threshold']}",
         "note": format_level_counts(counts=evaluation_filter["kept_by_level"]), "boxes": kept_entries},
        {"heading": f"{detection_count} detections, score >= {model.detection_score_threshold:.2f}",
         "note": "white: the ground truth boxes", "boxes": object_entries + detection_entries}],
        output_path=figure_folder / "proposal_filter.png")

    # Anchor label figure: the foreground, ignored and sampled background anchors of the image, their notes quoting the
    # thresholds of the run; the ignored panel draws the band below the foreground minimum, the anchors below the
    # background minimum (those overlapping no object) being only counted in its note
    anchors = aggregated_proposals["anchors"][0]
    foreground_minimum = filter_configuration["foreground_iou_threshold"]["min"]
    background_minimum = filter_configuration["background_iou_threshold"]["min"]
    background_maximum = filter_configuration["background_iou_threshold"]["max"]
    foreground_entries = [{"box": box.tolist(), "colour": hexadecimal_to_rgb(colour=FOREGROUND_COLOUR), "alpha": 1.0,
                           "width": 3} for box in anchors[anchor_labels["labels"] == 1.0]]
    ignored_entries = [{"box": box.tolist(), "colour": hexadecimal_to_rgb(colour=IGNORED_COLOUR), "alpha": 0.9,
                        "width": 2} for box in anchors[anchor_labels["ignored_band_mask"]]]
    ignored_note = f"IoU {background_maximum:g} to {foreground_minimum:g}: {anchor_labels['ignored_band']}, no loss"
    if background_minimum > 0.0:
        ignored_note += f" · below {background_minimum:g}: {anchor_labels['ignored_low']}"
    negative_entries = [{"box": box.tolist(), "colour": hexadecimal_to_rgb(colour=BACKGROUND_COLOUR), "alpha": 0.55}
                        for box in anchors[anchor_labels["sampled_negative_mask"]]]

    # Three panels over the objects: the foreground anchors, the ignored band and the sampled background anchors
    label_panel_boxes = draw_panel_figure(image=rgb_image, panels=[
        {"heading": f"{anchor_labels['foreground']} foreground anchors",
         "note": f"IoU >= {foreground_minimum:g}: {anchor_labels['overlap_foreground']} · best anchor only: "
                 f"{anchor_labels['fallback_foreground']}", "boxes": object_entries + foreground_entries},
        {"heading": f"{anchor_labels['ignored']} ignored anchors", "note": ignored_note,
         "boxes": object_entries + ignored_entries},
        {"heading": f"{anchor_labels['sampled_negatives']} background of {anchor_labels['background']}",
         "note": f"sampled: {anchor_labels['sampled_positives']} positives + {anchor_labels['sampled_negatives']} "
                 f"negatives", "boxes": object_entries + negative_entries}],
        output_path=figure_folder / "anchor_labels.png")

    # Facts of the anchors, the proposal filter, the targets and the detections
    facts = {"inference.object_count": len(object_names),
             "inference.anchor_total": int(anchors.shape[0]),
             "inference.anchor_page": {"image": f"figures:object_detection/inference/{SAMPLE_IMAGE_NAME}",
                                       "image_size": model.input_image_size, "levels": level_descriptions,
                                       "objects": object_descriptions},
             "inference.fallback_objects": [entry["name"] for entry in object_descriptions if not entry["foreground"]],
             "inference.proposal_filter.panel_boxes": proposal_panel_boxes,
             "inference.anchor_labels.panel_boxes": label_panel_boxes,
             "inference.detection_count": detection_count,
             "inference.found_object_count": sum(outcome["outcome"] == "found" for outcome in object_outcomes),
             "inference.level_count": len(level_descriptions),
             "inference.level_names_text": join_words(words=[entry["name"] for entry in level_descriptions]),
             "inference.anchor_cell_object_text": describe_class_name(class_name=object_names[cell_object_index]),
             **inference_key}

    # Objects matched by the fallback or by their overlap, the boxes of the panels of both figures, and the anchor
    # settings of every level
    facts["inference.fallback_count"] = len(facts["inference.fallback_objects"])
    facts["inference.overlap_object_count"] = len(object_descriptions) - facts["inference.fallback_count"]
    for panel_index, (proposal_box, label_box) in enumerate(zip(proposal_panel_boxes, label_panel_boxes)):
        facts[f"inference.proposal_filter.panel_{panel_index + 1}_box"] = proposal_box
        facts[f"inference.anchor_labels.panel_{panel_index + 1}_box"] = label_box
    for level_description in level_descriptions:
        for key in ["stride", "cells", "anchor_count", "anchors_per_cell", "scale_count", "ratio_count"]:
            facts[f"inference.level_{level_description['name']}.{key}"] = level_description[key]

    # Every object with its best anchor overlap, on the lines of the anchor scene, and the objects of the fallback in
    # the order of the manifest, as said
    object_texts = [f"{describe_class_name(class_name=entry['name'])} {entry['best_iou']:.2f}"
                    for entry in object_descriptions]
    facts["inference.object_lines"] = "\n".join(" · ".join(object_texts[start:start + OBJECT_LINE_LENGTH])
                                                for start in range(0, len(object_texts), OBJECT_LINE_LENGTH))
    for object_index, object_description in enumerate(object_descriptions):
        prefix = f"inference.object_{object_index + 1}"
        facts.update({f"{prefix}.name": object_description["name"], f"{prefix}.best_iou": object_description[
            "best_iou"], f"{prefix}.best_level": object_description["best_level"]})
    fallback_descriptions = [entry for entry in object_descriptions if not entry["foreground"]]
    for fallback_index, fallback_description in enumerate(fallback_descriptions):
        facts[f"inference.fallback_{fallback_index + 1}.name_text"] = describe_class_name(
            class_name=fallback_description["name"])
        facts[f"inference.fallback_{fallback_index + 1}.best_iou"] = fallback_description["best_iou"]

    # Objects found, taken for another class and missed by the detections of the checkpoint, as said
    for outcome_name in ["found", "wrong_class", "missed"]:
        outcome_indices = [index for index, outcome in enumerate(object_outcomes) if outcome["outcome"] == outcome_name]
        facts[f"inference.{outcome_name}_count"] = len(outcome_indices)
        outcome_object_names = [describe_class_name(class_name=object_names[index]) for index in outcome_indices]
        facts[f"inference.{outcome_name}_text"] = join_words(words=outcome_object_names) if outcome_object_names else ""
    facts["inference.wrong_class_detected_text"] = join_words(words=[
        describe_class_name(class_name=category_names[outcome["detected_class"]]) for outcome in object_outcomes
        if outcome["outcome"] == "wrong_class"]) if facts["inference.wrong_class_count"] else ""

    # Objects found at every compared evaluation, each checkpoint run on the image
    facts.update(describe_sample_iterations(run_folder=run_folder, input_tensor=input_tensor,
                                            object_boxes=object_boxes, object_classes=object_classes,
                                            category_names=category_names))

    # Counts of the proposal filter in evaluation and in training, in all and by level
    for mode_name, mode_filter in [("evaluation", evaluation_filter), ("training", training_filter)]:
        facts[f"inference.{mode_name}_filter.top_count"] = int(mode_filter["top_boxes"].shape[0])
        facts[f"inference.{mode_name}_filter.survivor_count"] = mode_filter["survivor_count"]
        facts[f"inference.{mode_name}_filter.valid_count"] = mode_filter["valid_count"]
        for level, count in mode_filter["top_by_level"].items():
            facts[f"inference.{mode_name}_filter.top_level_{level}"] = count
        for level, count in mode_filter["kept_by_level"].items():
            facts[f"inference.{mode_name}_filter.kept_level_{level}"] = count

    # Counts of the anchor labels and of the detector targets, in all and by level, and every detection by name
    for key in ["foreground", "background", "ignored", "ignored_low", "ignored_band", "overlap_foreground",
                "fallback_foreground", "sampled_positives", "sampled_negatives"]:
        facts[f"inference.anchor_labels.{key}"] = anchor_labels[key]
    for level, count in anchor_labels["foreground_by_level"].items():
        facts[f"inference.anchor_labels.foreground_level_{level}"] = count
    for key in ["written_ground_truths", "proposal_count", "positives", "sampled_positives", "sampled_negatives"]:
        facts[f"inference.detector_targets.{key}"] = detector_targets[key]
    for level, count in detector_targets["sampled_by_level"].items():
        facts[f"inference.detector_targets.sampled_level_{level}"] = count
    for detection_index, detection in enumerate(detections):
        facts[f"inference.detection_{detection_index + 1}.name"] = detection["name"]
        facts[f"inference.detection_{detection_index + 1}.score"] = round_fact(value=detection["score"])

    # Facts written next to the figures, so that the next build reuses the inference while its key holds
    write_reused_facts(facts_path=facts_path, figure_paths=[image_path, figure_folder / "proposal_filter.png",
                                                            figure_folder / "anchor_labels.png"], facts=facts)
    return facts
