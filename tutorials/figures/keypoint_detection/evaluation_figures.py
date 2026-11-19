"""
Figures and facts of the third chapter of the keypoint detection series of the tutorial videos (from candidates to
objects, the metrics and the export), made by tutorials/figures/keypoint_detection/figures.py: the heatmaps of the
trained detector on the test image at both strides and their average with the candidates extracted from it (for the
classes of the single object and of the pair of select_story_objects in keypoint_sources.py), the grouping of the
candidates of the pair keypoint by keypoint (the data of the grouping animation, checked against
group_keypoint_candidates), the object keypoint similarity of one keypoint against its distance for the boxes of the
largest and the smallest object of the image, and, from the evaluations in the Checkpoint_Evaluation folder of the run
(EVALUATION_FOLDER_NAME), the metrics of every evaluated checkpoint read from its TensorBoard logs, its worst classes
and keypoints read from its class_metrics.csv and keypoint_metrics.csv, and the boxes of the objects in its sample
image. Every property of the image the chapters narrate (the keypoints of the pair all found and partly exchanged,
the corners of the cut object of the pair placed on the border, the missing corner of the single object) is checked,
the build stopping when the image no longer shows it. The rates of the metrics grid and of the worst classes and
keypoints are written as the percentages of the reports of the repository ("94.03 %"), the facts the narration quotes
keeping them between 0 and 1; the similarity of one object against its distance stays between 0 and 1, like the
similarity thresholds it is compared with.
"""

import csv
import math
import re

import matplotlib
import numpy
import torch

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter, MultipleLocator
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorials.figures.keypoint_detection.dataset_figures import HEATMAP_COLOUR_MAP
from tutorials.figures.keypoint_detection.keypoint_inference import INFERENCE_ARRAYS_NAME
from tutorials.figures.keypoint_detection.keypoint_sources import (CHECKPOINT_ITERATION, EVALUATED_ITERATIONS,
                                                                  EVALUATION_FOLDER_NAME, OBJECT_COLOURS,
                                                                  SAMPLE_IMAGE_NAME, find_labelled_positions,
                                                                  format_class_label, join_spoken_names, write_json)
from tutorials.figures.keypoint_detection.scalar_logs import read_scalar_values
from tutorial_builder.layout import NAMED_COLOURS
from vision_training.evaluation.detection.object_sizes import compute_box_areas
from vision_training.evaluation.keypoints.object_keypoint_similarity import (CORRECT_KEYPOINT_FRACTION,
                                                                             compute_object_keypoint_similarity)
from vision_training.evaluation.report_formatting import format_percentage, format_percentage_tick, round_percentage
from vision_training.models.keypoints.hrnet import BRANCH_SCALE_FACTOR
from vision_training.models.keypoints.keypoint_detector import HEATMAP_STRIDE
from vision_training.models.keypoints.keypoint_grouping import group_keypoint_candidates, match_candidates_to_objects

# Roles of the rows of the heatmap figure, the class of the single object then the one of the pair, every row cropping
# the region of the labelled keypoints of its class with a margin, at least the minimum width, in image pixels
HEATMAP_ROW_ROLES = ["single", "pair"]
HEATMAP_CROP_MARGIN = 40.0
HEATMAP_MINIMUM_CROP_WIDTH = 160.0

# Layout of the heatmap figure: the column titles above two rows of four panels
HEATMAP_TITLE_HEIGHT = 44
HEATMAP_ROW_GAP = 12
HEATMAP_COLUMN_GAP = 16

# Largest distance between a candidate and a labelled keypoint of its channel for the candidate to lie on it, in image
# pixels: one cell of the quarter resolution heatmaps
MATCH_RADIUS = float(HEATMAP_STRIDE)

# Largest distance of a candidate to a side of the image for it to lie on the border, in image pixels
BORDER_DISTANCE = 2.0

# Distances of a keypoint to its ground truth drawn by the object keypoint similarity curves, in pixels
OKS_DISTANCES = list(range(0, 31))

# Metrics of the evaluated checkpoints shown by the metrics grid, as logged in Checkpoint_Evaluation/Tensorboard/Test
# and Test_EMA
GRID_METRICS = {"OKS mAP@0.5/area": "mAP@0.5", "OKS mAP@0.75/area": "mAP@0.75", "OKS mAP@0.95/area": "mAP@0.95",
                "OKS mAP@[0.5:0.95]/area": "mAP@[.5:.95]", "Keypoint Accuracy/PCK@0.05": "PCK@0.05",
                "Keypoint Accuracy/Mean OKS": "Mean OKS", "Keypoint Accuracy/Matched Objects": "Matched"}
ERROR_METRIC = "Keypoint Accuracy/Mean Keypoint Error (pixels)"

# Smallest lead of the averaged weights over the trained ones in OKS mAP@0.95 at the first evaluated checkpoint for
# the narration to call them well ahead
AVERAGED_WEIGHTS_LEAD = 0.03

# Number of classes and keypoints of the dot plots of the worst classes and keypoints
WORST_COUNT = 12

# Start of the rate axis of the dot plots, lowered by steps of DOT_PLOT_AXIS_STEP when a value falls below it
DOT_PLOT_AXIS_START = 0.8
DOT_PLOT_AXIS_STEP = 0.05


def crop_limits(centre: Tuple[float, float], crop_width: float, aspect_ratio: float) -> Tuple[float, float, float,
                                                                                              float]:
    """
    Gives the region of the image a panel of the heatmap figure shows, around a centre, with the proportions of the
    panel.

    Args:
        centre (Tuple[float, float]): The centre of the region in image pixels.
        crop_width (float): The width of the region in image pixels.
        aspect_ratio (float): The width over the height of the panel.

    Returns:
        Tuple[float, float, float, float]: The left, right, bottom and top of the region (the image y axis pointing
            down, so the bottom is the larger y).
    """
    crop_height = crop_width / aspect_ratio
    return (centre[0] - crop_width / 2, centre[0] + crop_width / 2, centre[1] + crop_height / 2,
            centre[1] - crop_height / 2)


def class_channels(channel_table: numpy.ndarray, class_index: int) -> List[int]:
    """
    Gives the heatmap channels of the keypoints of one class, in their keypoint order.

    Args:
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        class_index (int): The class.

    Returns:
        List[int]: The channels.
    """
    return [int(channel) for channel in channel_table[class_index] if channel >= 0]


def find_class_crop(objects: List[Dict[str, Any]], boxes: List[List[float]], class_index: int, image_size: int,
                    aspect_ratio: float) -> Tuple[Tuple[float, float], float]:
    """
    Gives the region of the image a row of the heatmap figure crops for one class: around the boxes of its objects
    with HEATMAP_CROP_MARGIN, wide enough for their height at the proportions of the panels and at least
    HEATMAP_MINIMUM_CROP_WIDTH, moved inside the image when it sticks out of it (an object cut by the border).

    Args:
        objects (List[Dict[str, Any]]): The objects of the inference facts.
        boxes (List[List[float]]): The (x1, y1, x2, y2) box of every object, in the order of the objects.
        class_index (int): The class of the row.
        image_size (int): The side of the square image.
        aspect_ratio (float): The width over the height of the panels.

    Returns:
        Tuple[Tuple[float, float], float]: The centre of the region and its width, in image pixels.
    """
    class_boxes = numpy.array([box for item, box in zip(objects, boxes) if item["class_index"] == class_index])
    left, top = class_boxes[:, :2].min(axis=0) - HEATMAP_CROP_MARGIN
    right, bottom = class_boxes[:, 2:].max(axis=0) + HEATMAP_CROP_MARGIN
    crop_width = min(max(right - left, (bottom - top) * aspect_ratio, HEATMAP_MINIMUM_CROP_WIDTH), image_size)
    crop_height = crop_width / aspect_ratio

    # Centre moved so that the region stays inside the image along both axes
    centre_x = min(max((left + right) / 2, crop_width / 2), image_size - crop_width / 2)
    centre_y = min(max((top + bottom) / 2, crop_height / 2), image_size - crop_height / 2)
    return (float(centre_x), float(centre_y)), float(crop_width)


def describe_candidates(arrays: Any, objects: List[Dict[str, Any]], keypoint_threshold: float) -> Dict[str, Any]:
    """
    Sorts the candidates of the whole image above the keypoint threshold into those lying on a labelled keypoint of
    their channel (within MATCH_RADIUS) and the others, for the narration of the heatmap scene: the bound of the error
    of the first ones and the best score of the others, which must stay below every candidate on a keypoint.

    Args:
        arrays (Any): The arrays of the inference, with the candidates of every channel.
        objects (List[Dict[str, Any]]): The objects of the inference facts.
        keypoint_threshold (float): 'KeypointHead.inference.keypoint_threshold' of the run.

    Returns:
        Dict[str, Any]: The facts of the candidates.

    Raises:
        ValueError: If no candidate lies on a keypoint, or if a candidate off the keypoints scores as high as one on
            them.
    """
    labelled_positions = {}
    for item in objects:
        for channel, position, visibility in zip(item["channels"], item["positions"], item["visibilities"]):
            if visibility > 0:
                labelled_positions.setdefault(channel, []).append(position)

    # Distance of every candidate above the threshold to the nearest labelled keypoint of its channel
    distances, scores = [], []
    for channel, candidate_index in zip(*numpy.nonzero(arrays["candidate_scores"] > keypoint_threshold)):
        position = arrays["candidate_positions"][channel, candidate_index]
        channel_positions = numpy.array(labelled_positions.get(int(channel), numpy.zeros(shape=(0, 2))))
        distances.append(float(numpy.linalg.norm(channel_positions - position, axis=1).min())
                         if len(channel_positions) else math.inf)
        scores.append(float(arrays["candidate_scores"][channel, candidate_index]))
    distances, scores = numpy.array(distances), numpy.array(scores)
    on_keypoints = distances <= MATCH_RADIUS
    if not on_keypoints.any():
        raise ValueError(f"No candidate of {SAMPLE_IMAGE_NAME} lies on a labelled keypoint.")
    other_best_score = float(scores[~on_keypoints].max()) if (~on_keypoints).any() else 0.0
    if other_best_score >= scores[on_keypoints].min():
        raise ValueError(f"A candidate off the labelled keypoints scores {other_best_score:.3f}, as much as one on "
                         f"them ({scores[on_keypoints].min():.3f}): the heatmap scene says they all score lower.")

    # Bound of the distances on the keypoints, the next tenth of a pixel above the largest one, which the narration
    # quotes as "less than"
    largest_error = float(distances[on_keypoints].max())
    return {"heatmaps_candidates_on_keypoints": int(on_keypoints.sum()),
            "heatmaps_error_bound": round_fact(value=math.floor(largest_error * 10) / 10 + 0.1),
            "heatmaps_mean_error": round_fact(value=float(distances[on_keypoints].mean())),
            "heatmaps_other_candidates": int((~on_keypoints).sum()),
            "heatmaps_other_best_score": round_fact(value=other_best_score),
            "heatmaps_lowest_score_on_keypoints": round_fact(value=float(scores[on_keypoints].min()))}


def draw_heatmap_figure(output_folder: Path, sample_record: Dict[str, Any], inference_facts: Dict[str, Any],
                        channel_table: numpy.ndarray, keypoint_threshold: float,
                        story_objects: Dict[str, Any]) -> Dict[str, Any]:
    """
    Draws the heatmaps of the trained detector on the test image for the class of the single object and the class of
    the pair: the image, the maximum of the heatmaps of the class at stride 4 and at stride 2, and their average at
    stride 2 the candidates are taken on (the stride 4 heatmaps upsampled bilinearly, as extract_keypoint_candidates
    does), with the candidates above the keypoint threshold and the labelled keypoints, and sorts the candidates of the
    whole image (describe_candidates).

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        keypoint_threshold (float): 'KeypointHead.inference.keypoint_threshold' of the run.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.

    Returns:
        Dict[str, Any]: The facts of the heatmaps and the boxes of the rows of the figure, named by the role of their
            class.
    """
    register_fonts()
    arrays = numpy.load(file=output_folder / INFERENCE_ARRAYS_NAME)
    shown_channels = arrays["channels"].tolist()
    image = sample_record["sample"]["images"].permute(1, 2, 0).numpy()[..., ::-1]
    image_size = image.shape[0]
    objects = inference_facts["inference_objects"]
    role_classes = {"single": objects[story_objects["single"]]["class_index"],
                    "pair": objects[story_objects["pair"][0]]["class_index"]}

    # Panels of the figure: four columns, two rows under the column titles
    panel_width = (CHART_WIDTH - 3 * HEATMAP_COLUMN_GAP) / 4
    panel_height = (CHART_HEIGHT - HEATMAP_TITLE_HEIGHT - HEATMAP_ROW_GAP) / 2
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    column_titles = ["Test image", f"Heat maps at stride {HEATMAP_STRIDE}",
                     f"Heat maps at stride {HEATMAP_STRIDE // BRANCH_SCALE_FACTOR}",
                     f"Average, candidates above {keypoint_threshold:g}"]
    for column_index, column_title in enumerate(column_titles):
        figure.text(x=column_index * (panel_width + HEATMAP_COLUMN_GAP) / CHART_WIDTH,
                    y=1.0 - 20 / CHART_HEIGHT, s=column_title, color=TEXT_PRIMARY_COLOUR,
                    fontsize=22 * POINTS_PER_PIXEL, ha="left", va="center")

    # Every row: the channels of its class, their heatmaps at both strides (stored in float16) and their average
    facts = {}
    for row_index, role in enumerate(HEATMAP_ROW_ROLES):
        class_index = role_classes[role]
        channels = class_channels(channel_table=channel_table, class_index=class_index)
        positions = [shown_channels.index(channel) for channel in channels]
        quarter_heatmaps = torch.from_numpy(arrays["heatmaps"][positions].astype(numpy.float32))[None]
        half_heatmaps = torch.from_numpy(arrays["higher_resolution_heatmaps"][positions].astype(numpy.float32))[None]
        averaged = (torch.nn.functional.interpolate(input=quarter_heatmaps, scale_factor=BRANCH_SCALE_FACTOR,
                                                    mode="bilinear", align_corners=False) + half_heatmaps) / 2.0
        panels = [None, quarter_heatmaps[0].max(dim=0).values.numpy(), half_heatmaps[0].max(dim=0).values.numpy(),
                  averaged[0].max(dim=0).values.numpy()]
        crop_centre, crop_width = find_class_crop(objects=objects,
                                                  boxes=sample_record["sample"]["bounding_boxes"].tolist(),
                                                  class_index=class_index, image_size=image_size,
                                                  aspect_ratio=panel_width / panel_height)
        limits = crop_limits(centre=crop_centre, crop_width=crop_width, aspect_ratio=panel_width / panel_height)
        panel_top = HEATMAP_TITLE_HEIGHT + row_index * (panel_height + HEATMAP_ROW_GAP)

        # Candidates of the class above the threshold, and its labelled keypoints
        candidate_scores = arrays["candidate_scores"][channels]
        candidate_positions = arrays["candidate_positions"][channels][candidate_scores > keypoint_threshold]
        labelled_positions = numpy.concatenate([find_labelled_positions(sample_object=item) for item in objects
                                                if item["class_index"] == class_index])
        for column_index, panel in enumerate(panels):
            panel_left = column_index * (panel_width + HEATMAP_COLUMN_GAP)
            axes = figure.add_axes(rect=(panel_left / CHART_WIDTH, 1.0 - (panel_top + panel_height) / CHART_HEIGHT,
                                         panel_width / CHART_WIDTH, panel_height / CHART_HEIGHT))
            if panel is None:
                axes.imshow(X=image, interpolation="lanczos")
            else:
                axes.imshow(X=panel, cmap=HEATMAP_COLOUR_MAP, vmin=0.0, vmax=1.0, interpolation="nearest",
                            extent=(0, image_size, image_size, 0))
            if column_index == 3:
                axes.scatter(labelled_positions[:, 0], labelled_positions[:, 1], marker="+", s=(14 * POINTS_PER_PIXEL)
                             ** 2, color=TEXT_PRIMARY_COLOUR, linewidths=2 * POINTS_PER_PIXEL, zorder=3)
                axes.scatter(candidate_positions[:, 0], candidate_positions[:, 1], s=(9 * POINTS_PER_PIXEL) ** 2,
                             facecolors="none", edgecolors="#%02x%02x%02x" % NAMED_COLOURS["accent"],
                             linewidths=2 * POINTS_PER_PIXEL, zorder=4)
            axes.set_xlim(left=limits[0], right=limits[1])
            axes.set_ylim(bottom=limits[2], top=limits[3])
            axes.set_xticks(ticks=[])
            axes.set_yticks(ticks=[])
            for spine in axes.spines.values():
                spine.set_color("#%02x%02x%02x" % NAMED_COLOURS["border"])
        facts[f"heatmaps_{role}_row_box"] = [0, round(panel_top) - 4, CHART_WIDTH, round(panel_top + panel_height) + 4]
        facts[f"heatmaps_{role}_average_box"] = [round(3 * (panel_width + HEATMAP_COLUMN_GAP)) - 4,
                                                 round(panel_top) - 4, CHART_WIDTH, round(panel_top + panel_height) + 4]

        # Peaks of both strides on the class, and the number of its candidates above the threshold
        facts[f"heatmaps_{role}_peak_stride_4"] = round_fact(value=float(panels[1].max()))
        facts[f"heatmaps_{role}_peak_stride_2"] = round_fact(value=float(panels[2].max()))
        facts[f"heatmaps_{role}_candidates"] = int(len(candidate_positions))
    save_figure(figure=figure, output_path=output_folder / "evaluation" / "heatmaps.png")
    pyplot.close(fig=figure)
    facts["heatmaps_candidates_above_threshold"] = inference_facts["inference_candidates_above_threshold"]
    facts["heatmaps_candidate_count"] = inference_facts["inference_candidate_count"]
    facts.update(describe_candidates(arrays=arrays, objects=objects, keypoint_threshold=keypoint_threshold))
    return facts


def trace_class_grouping(candidate_positions: numpy.ndarray, candidate_scores: numpy.ndarray,
                         candidate_tags: numpy.ndarray, channels: List[int], keypoint_threshold: float,
                         tag_threshold: float) -> Dict[str, Any]:
    """
    Groups the candidates of one class keypoint by keypoint the way group_keypoint_candidates does, with its
    match_candidates_to_objects, recording every step: the candidates of every keypoint with the object they join and
    their rounded tag distance to every object formed before the step (the term of the matching cost), and the mean
    tag of every object after the step.

    Args:
        candidate_positions (numpy.ndarray): The [C, N, 2] candidates of every channel.
        candidate_scores (numpy.ndarray): The [C, N] scores of the candidates.
        candidate_tags (numpy.ndarray): The [C, N] tags of the candidates.
        channels (List[int]): The channels of the keypoints of the class, in their order.
        keypoint_threshold (float): The lowest score of a grouped candidate.
        tag_threshold (float): The largest tag distance between a keypoint and the mean tag of its object.

    Returns:
        Dict[str, Any]: The 'steps' (candidates with their object, mean tags after the step) and the 'objects' (the
            keypoint positions and scores of every object formed).
    """
    object_tags, object_keypoints, object_scores, steps = [], [], [], []
    for keypoint_index, channel in enumerate(channels):
        kept = candidate_scores[channel] > keypoint_threshold
        positions, scores, tags = candidate_positions[channel][kept], candidate_scores[channel][kept], \
            candidate_tags[channel][kept]
        assignments = match_candidates_to_objects(candidate_tags=tags, candidate_scores=scores,
                                                  object_tags=object_tags, tag_threshold=tag_threshold)
        means_before = [float(numpy.mean(tags_of_object)) for tags_of_object in object_tags]

        # Every candidate joins its object or starts a new one, as in group_keypoint_candidates
        step_candidates = []
        for candidate_index, object_index in enumerate(assignments):
            starts_object = object_index < 0
            if starts_object:
                object_index = len(object_tags)
                object_tags.append([])
                object_keypoints.append([None] * len(channels))
                object_scores.append([0.0] * len(channels))
            object_tags[object_index].append(float(tags[candidate_index]))
            object_keypoints[object_index][keypoint_index] = positions[candidate_index].tolist()
            object_scores[object_index][keypoint_index] = float(scores[candidate_index])
            step_candidates.append({"tag": round_fact(value=tags[candidate_index]),
                                    "score": round_fact(value=scores[candidate_index]),
                                    "position": [round_fact(value=value) for value in positions[candidate_index]],
                                    "object": object_index, "starts_object": starts_object,
                                    "rounded_distances": [int(numpy.round(abs(float(tags[candidate_index]) - mean)))
                                                          for mean in means_before]})
        steps.append({"keypoint_index": keypoint_index, "candidates": step_candidates,
                      "object_means": [round_fact(value=numpy.mean(tags_of_object)) for tags_of_object in object_tags]})
    return {"steps": steps, "objects": [{"keypoints": keypoints, "scores": scores,
                                         "score": round_fact(value=sum(scores) / len(channels))}
                                        for keypoints, scores in zip(object_keypoints, object_scores)]}


def attribute_candidates(trace: Dict[str, Any], pair_objects: List[Dict[str, Any]]) -> numpy.ndarray:
    """
    Gives every candidate of the traced grouping of the pair the labelled object of the pair it lies on, the one whose
    labelled keypoint of the same name lies within MATCH_RADIUS of it (-1 for a candidate on none, such as a corner
    placed on the border where the keypoint outside the image is not labelled), and counts the candidates every
    grouped object took from every labelled object.

    Args:
        trace (Dict[str, Any]): The traced grouping of trace_class_grouping, whose candidates receive their
            'labelled_object' and 'distance'.
        pair_objects (List[Dict[str, Any]]): The two labelled objects of the pair, from the inference facts.

    Returns:
        numpy.ndarray: The [grouped objects, labelled objects] counts of the candidates lying on a labelled object.
    """
    labelled_counts = numpy.zeros(shape=(len(trace["objects"]), len(pair_objects)), dtype=int)
    for step in trace["steps"]:
        keypoint_index = step["keypoint_index"]
        for candidate in step["candidates"]:
            distances = numpy.array([numpy.linalg.norm(numpy.array(item["positions"][keypoint_index]) -
                                                       numpy.array(candidate["position"]))
                                     if item["visibilities"][keypoint_index] > 0 else math.inf
                                     for item in pair_objects])
            candidate["labelled_object"] = int(distances.argmin()) if distances.min() <= MATCH_RADIUS else -1
            candidate["distance"] = round_fact(value=distances.min()) if numpy.isfinite(distances.min()) else None
            if candidate["labelled_object"] >= 0:
                labelled_counts[candidate["object"], candidate["labelled_object"]] += 1
    return labelled_counts


def write_grouping_data(output_folder: Path, inference_facts: Dict[str, Any], channel_table: numpy.ndarray,
                        inference_configuration: Dict[str, Any], story_objects: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes the data of the grouping animation for the pair of the test image (its two objects of one class), every
    candidate with the labelled object it lies on (attribute_candidates), checked against the objects of
    group_keypoint_candidates, and gives the facts of the grouping of the whole image. Every property the chapters
    narrate is checked: every labelled keypoint of the pair found, the candidates on them at a rounded tag distance of
    zero from both objects and some of them exchanged between the two objects, the keypoints of the cut object of the
    pair outside the image placed on the border by the object of that cut object, the weakest object a single candidate
    dropped by the sample images, and the keypoints of the single object outside the image missing from its best
    object.

    Args:
        output_folder (Path): The figures folder of the series.
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        inference_configuration (Dict[str, Any]): 'KeypointHead.inference' of the run.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.

    Returns:
        Dict[str, Any]: The facts of the grouping.

    Raises:
        ValueError: If the traced grouping differs from group_keypoint_candidates, or if the image no longer shows a
            property the chapters narrate.
    """
    arrays = numpy.load(file=output_folder / INFERENCE_ARRAYS_NAME)
    positions, scores, tags = arrays["candidate_positions"], arrays["candidate_scores"], arrays["candidate_tags"]
    objects = inference_facts["inference_objects"]
    pair_objects = [objects[index] for index in story_objects["pair"]]
    class_index = pair_objects[0]["class_index"]
    channels = class_channels(channel_table=channel_table, class_index=class_index)
    trace = trace_class_grouping(candidate_positions=positions, candidate_scores=scores, candidate_tags=tags,
                                 channels=channels, keypoint_threshold=inference_configuration["keypoint_threshold"],
                                 tag_threshold=inference_configuration["tag_threshold"])

    # The objects of the class found by group_keypoint_candidates, every object kept, against the traced ones
    grouped = group_keypoint_candidates(candidate_positions=positions, candidate_scores=scores, candidate_tags=tags,
                                        channel_table=channel_table,
                                        keypoint_threshold=inference_configuration["keypoint_threshold"],
                                        tag_threshold=inference_configuration["tag_threshold"],
                                        object_score_threshold=0.0)
    grouped_keypoints = grouped["keypoints"][grouped["labels"] == class_index][:, :len(channels)]
    traced_keypoints = numpy.array([[point if point is not None else [numpy.nan, numpy.nan]
                                     for point in item["keypoints"]] for item in trace["objects"]], dtype=numpy.float32)
    order = numpy.argsort([-item["score"] for item in trace["objects"]], kind="stable")
    if not numpy.allclose(traced_keypoints[order], grouped_keypoints, equal_nan=True):
        raise ValueError(f"The traced grouping of the {pair_objects[0]['class_name']} pair differs from "
                         f"group_keypoint_candidates.")

    # Every grouped object belongs to the labelled object most of its candidates lie on, the others being exchanged;
    # every candidate lying on a labelled keypoint is at a rounded tag distance of zero from every object before it
    labelled_counts = attribute_candidates(trace=trace, pair_objects=pair_objects)
    object_owners = labelled_counts.argmax(axis=1)
    exchanged, labelled_distances = [], []
    for step in trace["steps"]:
        for candidate in step["candidates"]:
            owner = int(object_owners[candidate["object"]])
            candidate["exchanged"] = bool(candidate["labelled_object"] >= 0 and candidate["labelled_object"] != owner)
            keypoint_name = format_class_label(class_name=pair_objects[0]["keypoint_names"][step["keypoint_index"]])
            if candidate["labelled_object"] >= 0:
                labelled_distances += candidate["rounded_distances"]
            if candidate["exchanged"] and keypoint_name not in exchanged:
                exchanged.append(keypoint_name)
    if not exchanged or any(labelled_distances):
        raise ValueError(f"{len(exchanged)} keypoints of the {pair_objects[0]['class_name']} pair are exchanged, the "
                         f"candidates on its keypoints at rounded tag distances {labelled_distances}, but the grouping "
                         f"scene says some are exchanged, all of them at a rounded distance of zero from both objects.")

    # Every labelled keypoint of the pair found by a candidate lying on it
    labelled_keypoints = sum(int(numpy.sum(numpy.array(item["visibilities"]) > 0)) for item in pair_objects)
    found_keypoints = len({(candidate["labelled_object"], step["keypoint_index"]) for step in trace["steps"]
                           for candidate in step["candidates"] if candidate["labelled_object"] >= 0})
    if found_keypoints != labelled_keypoints:
        raise ValueError(f"{found_keypoints} of the {labelled_keypoints} labelled keypoints of the pair are found, but "
                         f"the sample scene says every one is.")

    # The keypoints of the cut object of the pair outside the image, placed on the border by the object owning it
    image_size = int(inference_facts["inference_shapes"]["high_resolution_network"]["output"][-1]) * HEATMAP_STRIDE
    cut_member = story_objects["pair"].index(story_objects["cut_pair_member"])
    cut_object_index = int(numpy.flatnonzero(object_owners == cut_member)[0])
    cut_keypoints = [keypoint_index for keypoint_index, visibility
                     in enumerate(pair_objects[cut_member]["visibilities"]) if visibility == 0]
    cut_object = trace["objects"][cut_object_index]
    border_points = [cut_object["keypoints"][keypoint_index] for keypoint_index in cut_keypoints]
    border_scores = [cut_object["scores"][keypoint_index] for keypoint_index in cut_keypoints]
    on_border = [point is not None and (min(point) <= BORDER_DISTANCE or max(point) >= image_size - BORDER_DISTANCE)
                 for point in border_points]
    if not all(on_border) or cut_object["score"] < inference_configuration["object_score_threshold"]:
        raise ValueError(f"The keypoints outside the image of the cut {pair_objects[0]['class_name']} are "
                         f"{border_points} in an object of score {cut_object['score']}, but the sample scene says that "
                         f"its shown object places all of them on the border.")

    # The weakest object of the pair class, a single candidate left over, dropped by the sample images
    weakest = trace["objects"][int(numpy.argmin([item["score"] for item in trace["objects"]]))]
    weakest_keypoints = sum(point is not None for point in weakest["keypoints"])
    if (len(trace["objects"]) <= len(pair_objects) or weakest_keypoints != 1 or
            weakest["score"] >= inference_configuration["object_score_threshold"]):
        raise ValueError(f"The weakest {pair_objects[0]['class_name']} object holds {weakest_keypoints} keypoints with "
                         f"a score of {weakest['score']}, but the grouping scene says it is a single candidate left "
                         f"over, below the threshold of the sample images.")
    write_json(data={"keypoint_names": [format_class_label(class_name=name)
                                        for name in pair_objects[0]["keypoint_names"]],
                     "steps": trace["steps"], "objects": trace["objects"],
                     "labelled_colours": [OBJECT_COLOURS[index] for index in story_objects["pair"]],
                     "tag_threshold": inference_configuration["tag_threshold"],
                     "keypoint_threshold": inference_configuration["keypoint_threshold"]},
               output_path=output_folder / "evaluation" / "grouping.json")

    # Best object of the class of the single object, its keypoints outside the image not found
    single_object = objects[story_objects["single"]]
    single_mask = grouped["labels"] == single_object["class_index"]
    best_single = int(numpy.argmax(numpy.where(single_mask, grouped["scores"], -1.0)))
    unlabelled = [index for index, visibility in enumerate(single_object["visibilities"]) if visibility == 0]
    if not numpy.isnan(grouped["keypoints"][best_single][unlabelled]).all():
        raise ValueError(f"The best {single_object['class_name']} object finds keypoints outside the image, but the "
                         f"sample scene says they are missing.")

    # Objects of the whole image, kept by the evaluation or by the sample images and the deployment
    kept_objects = int((grouped["scores"] >= inference_configuration["object_score_threshold"]).sum())
    return {"grouping_exchanged_keypoints": join_spoken_names(names=exchanged),
            "grouping_exchanged_count": len(exchanged), "grouping_found_objects": len(trace["objects"]),
            "grouping_pair_labelled_keypoints": labelled_keypoints,
            "grouping_exchanged_per_object": [int(labelled_counts[object_index].sum() -
                                                  labelled_counts[object_index].max())
                                              for object_index in range(len(trace["objects"]))],
            "grouping_pair_scores": [item["score"] for item in trace["objects"]],
            "grouping_cut_border_keypoints": len(cut_keypoints),
            "grouping_cut_border_best_score": round_fact(value=max(border_scores)),
            "grouping_weakest_score": weakest["score"],
            "grouping_single_best_score": round_fact(value=float(grouped["scores"][best_single])),
            "grouping_single_missing_keypoints": len(unlabelled),
            "grouping_all_objects": int(len(grouped["scores"])), "grouping_kept_objects": kept_objects,
            "grouping_object_score_threshold": inference_configuration["object_score_threshold"]}


def describe_object_keypoint_similarity(sample_record: Dict[str, Any], inference_facts: Dict[str, Any],
                                        default_sigma: float, story_objects: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the object keypoint similarity of a one-keypoint object against the distance of its keypoint, with
    compute_object_keypoint_similarity and the box areas of compute_box_areas, for the largest and the smallest object
    of the test image by box area: the curves of the OKS scene, the distance at which the similarity falls to 0.5 and
    the PCK tolerance of both boxes.

    Args:
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.
        default_sigma (float): 'evaluation.default_keypoint_sigma' of the run.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.

    Returns:
        Dict[str, Any]: The facts of the similarity curves, named oks_large_* and oks_small_*.

    Raises:
        ValueError: If a similarity of one half lies beyond the distances of the chart.
    """
    sample = sample_record["sample"]
    areas = compute_box_areas(boxes=sample["bounding_boxes"].to(dtype=torch.float64))
    distances = torch.tensor(data=OKS_DISTANCES, dtype=torch.float64)
    facts = {}
    for name, object_index in [("large", story_objects["largest"]), ("small", story_objects["smallest"])]:
        area = areas[object_index]
        facts[f"oks_{name}_label"] = format_class_label(
            class_name=inference_facts["inference_objects"][object_index]["class_name"])
        similarities = compute_object_keypoint_similarity(
            squared_distances=(distances ** 2)[:, None], visibilities=torch.full(size=(len(OKS_DISTANCES), 1),
                                                                                  fill_value=2),
            areas=area.expand(len(OKS_DISTANCES)), sigmas=torch.full(size=(len(OKS_DISTANCES), 1),
                                                                      fill_value=default_sigma))
        fine_distances = torch.linspace(start=0.0, end=float(OKS_DISTANCES[-1]), steps=3001, dtype=torch.float64)
        fine_similarities = compute_object_keypoint_similarity(
            squared_distances=(fine_distances ** 2)[:, None], visibilities=torch.full(size=(3001, 1), fill_value=2),
            areas=area.expand(3001), sigmas=torch.full(size=(3001, 1), fill_value=default_sigma))
        box = sample["bounding_boxes"][object_index]
        longest_side = float(max(box[2] - box[0], box[3] - box[1]))
        facts.update({f"oks_{name}_points": [[int(distance), round(float(similarity), 4)]
                                             for distance, similarity in zip(OKS_DISTANCES, similarities)],
                      f"oks_{name}_area": int(round(float(area))),
                      f"oks_{name}_half_distance": round_fact(
                          value=float(fine_distances[fine_similarities >= 0.5][-1])),
                      f"oks_{name}_pck_tolerance": round_fact(value=CORRECT_KEYPOINT_FRACTION * longest_side),
                      f"oks_{name}_longest_side": round_fact(value=longest_side)})
        if fine_similarities[-1] >= 0.5:
            raise ValueError(f"The similarity of the {facts[f'oks_{name}_label']} is still one half at "
                             f"{OKS_DISTANCES[-1]} pixels, the end of the OKS chart.")
    facts["oks_default_sigma"] = default_sigma
    facts["oks_constant"] = round_fact(value=2 * default_sigma)
    return facts


def read_results_header(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the number of test images and of test objects with keypoints of the evaluations of the run from the first
    report block of its Checkpoint_Evaluation/Results.txt.

    Args:
        run_folder (Path): The folder of the keypoint run.

    Returns:
        Dict[str, Any]: The 'metrics_test_images' and 'metrics_test_objects' facts.

    Raises:
        ValueError: If the report does not give them.
    """
    report = (run_folder / EVALUATION_FOLDER_NAME / "Results.txt").read_text(encoding="utf-8")
    images_match = re.search(pattern=r"Iteration \d+ · (\d+) Test Images", string=report)
    objects_match = re.search(pattern=r"^\s+All\s+(\d+)\s", string=report, flags=re.MULTILINE)
    if images_match is None or objects_match is None:
        raise ValueError(f"No test image or object count in {run_folder / EVALUATION_FOLDER_NAME / 'Results.txt'}.")
    return {"metrics_test_images": int(images_match.group(1)), "metrics_test_objects": int(objects_match.group(1))}


def describe_metrics(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the metrics of every evaluated checkpoint of the run, for the trained and the averaged weights, from its
    Checkpoint_Evaluation/Tensorboard/Test and Test_EMA logs: the values of the metrics grid, every one a rate written
    as the percentage of the reports, and the facts the narration quotes, between 0 and 1.

    Args:
        run_folder (Path): The folder of the keypoint run.

    Returns:
        Dict[str, Any]: The facts of the metrics.

    Raises:
        ValueError: If the metrics no longer rise or the averaged weights no longer lead, as the narration says.
    """
    tags = [*GRID_METRICS, ERROR_METRIC]
    log_root = run_folder / EVALUATION_FOLDER_NAME / "Tensorboard"
    weights_values = {"trained": read_scalar_values(log_folder=log_root / "Test", tags=tags),
                      "ema": read_scalar_values(log_folder=log_root / "Test_EMA", tags=tags)}

    # Grid of the metrics as percentages rounded like the reports, one column per checkpoint and kind of weights
    columns = [(iteration, weights) for iteration in EVALUATED_ITERATIONS for weights in ["trained", "ema"]]
    facts = {"metrics_grid_percentages": [[round_percentage(value=weights_values[weights][tag][iteration])
                                           for iteration, weights in columns] for tag in GRID_METRICS],
             "metrics_grid_row_labels": list(GRID_METRICS.values()),
             "metrics_grid_column_labels": [f"{iteration}" + (" EMA" if weights == "ema" else "")
                                            for iteration, weights in columns]}
    for weights, values in weights_values.items():
        for tag, label in [*GRID_METRICS.items(), (ERROR_METRIC, "error")]:
            name = re.sub(pattern=r"[^a-z0-9]+", repl="_", string=label.lower()).strip("_")
            for iteration in EVALUATED_ITERATIONS:
                facts[f"metrics_{weights}_{iteration}_{name}"] = values[tag][iteration]
    facts.update(read_results_header(run_folder=run_folder))
    facts["metrics_first_iteration"] = EVALUATED_ITERATIONS[0]

    # The order the narration states: both mAP of the trained weights rise from the first to the last checkpoint, and
    # the averaged weights are well ahead at 0.95 on the first checkpoint
    first, last = EVALUATED_ITERATIONS[0], EVALUATED_ITERATIONS[-1]
    rising = all(facts[f"metrics_trained_{last}_{name}"] > facts[f"metrics_trained_{first}_{name}"]
                 for name in ["map_0_5", "map_0_95"])
    lead = facts[f"metrics_ema_{first}_map_0_95"] - facts[f"metrics_trained_{first}_map_0_95"]
    if not rising or lead < AVERAGED_WEIGHTS_LEAD:
        raise ValueError(f"The metrics of the run no longer behave as the narration says: mAP of the trained weights "
                         f"rising {rising}, lead of the averaged weights at 0.95 on iteration {first} {lead:.4f} (at "
                         f"least {AVERAGED_WEIGHTS_LEAD}).")
    return facts


def read_metric_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Reads the rows of a metrics file of an evaluation of the run (class_metrics.csv or keypoint_metrics.csv).

    Args:
        csv_path (Path): The CSV file.

    Returns:
        List[Dict[str, str]]: Its rows keyed by column.
    """
    with csv_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(f=csv_file))


def read_class_average_precision(row: Dict[str, str]) -> float:
    """
    Gives the OKS AP@0.5 of a row of class_metrics.csv, the sort key of the worst classes.

    Args:
        row (Dict[str, str]): The row.

    Returns:
        float: Its OKS AP@0.5.
    """
    return float(row["oks_average_precision_50"])


def read_keypoint_correctness(row: Dict[str, str]) -> float:
    """
    Gives the PCK@0.05 of a row of keypoint_metrics.csv, the sort key of the worst keypoints, a keypoint never labelled
    (empty value) coming last.

    Args:
        row (Dict[str, str]): The row.

    Returns:
        float: Its PCK@0.05, 2 for a keypoint without value.
    """
    return float(row["percentage_correct_keypoints"]) if row["percentage_correct_keypoints"] else 2.0


def draw_dot_plot(axes: matplotlib.axes.Axes, labels: List[str], values: List[float], value_labels: List[str],
                  colour: str, title: str, x_label: str) -> None:
    """
    Draws one dot plot of the worst classes or keypoints figure: one row per entry, worst at the top, its dot at its
    value on an axis of rates from DOT_PLOT_AXIS_START (lower when a value falls below it, by steps of five points) to
    1.0 labelled as percentages, and its value written after it.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        labels (List[str]): The names of the rows.
        values (List[float]): Their values.
        value_labels (List[str]): The texts written after the dots.
        colour (str): The colour of the dots.
        title (str): The title of the panel.
        x_label (str): The name of the value axis.
    """
    style_chart_panel(axes=axes, title=title, x_label=x_label, y_label=None)
    rows = numpy.arange(len(labels))
    axis_start = min(DOT_PLOT_AXIS_START, math.floor(min(values) / DOT_PLOT_AXIS_STEP) * DOT_PLOT_AXIS_STEP)
    axes.hlines(y=rows, xmin=axis_start, xmax=values, color="#%02x%02x%02x" % NAMED_COLOURS["border"],
                linewidth=2 * POINTS_PER_PIXEL)
    axes.scatter(values, rows, s=(12 * POINTS_PER_PIXEL) ** 2, color=colour, zorder=3)
    for row, value, value_label in zip(rows, values, value_labels):
        axes.text(x=value + 0.02 * (1.0 - axis_start), y=row, s=value_label, color=TEXT_SECONDARY_COLOUR,
                  va="center", ha="left", fontsize=17 * POINTS_PER_PIXEL)
    axes.set_yticks(ticks=rows, labels=labels)
    axes.set_ylim(bottom=len(labels) - 0.5, top=-0.5)
    # Value axis of rates every five points, labelled as percentages
    axes.set_xlim(left=axis_start, right=1.0)
    axes.xaxis.set_major_locator(locator=MultipleLocator(base=DOT_PLOT_AXIS_STEP))
    axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_percentage_tick))
    axes.tick_params(axis="y", labelcolor=TEXT_PRIMARY_COLOUR, labelsize=17 * POINTS_PER_PIXEL)
    axes.grid(axis="y", visible=False)


def draw_worst_figure(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Draws the worst classes by OKS AP@0.5 and the worst keypoints by PCK@0.05 of the evaluation of the last checkpoint
    of the run, from its class_metrics.csv and keypoint_metrics.csv (trained weights), the rates written as
    percentages and the mean errors in pixels.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the keypoint run.

    Returns:
        Dict[str, Any]: The facts of the worst class and keypoints.
    """
    register_fonts()
    evaluation_folder = run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{CHECKPOINT_ITERATION}"
    class_rows = sorted(read_metric_rows(csv_path=evaluation_folder / "class_metrics.csv"),
                        key=read_class_average_precision)[:WORST_COUNT]
    keypoint_rows = sorted(read_metric_rows(csv_path=evaluation_folder / "keypoint_metrics.csv"),
                           key=read_keypoint_correctness)[:WORST_COUNT]

    # Two dot plots side by side
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    class_axes = figure.add_axes(rect=(0.15, 0.12, 0.30, 0.78))
    keypoint_axes = figure.add_axes(rect=(0.66, 0.12, 0.30, 0.78))
    draw_dot_plot(axes=class_axes, labels=[row["class"].replace("_", " ") for row in class_rows],
                  values=[float(row["oks_average_precision_50"]) for row in class_rows],
                  value_labels=[format_percentage(value=float(row["oks_average_precision_50"])) for row in class_rows],
                  colour=SERIES_COLOURS[0], title="Worst classes, OKS AP@0.5", x_label="AP@0.5")
    draw_dot_plot(axes=keypoint_axes, labels=[f"{row['class'].replace('_', ' ')} · {row['keypoint'].replace('_', ' ')}"
                                              for row in keypoint_rows],
                  values=[float(row["percentage_correct_keypoints"]) for row in keypoint_rows],
                  value_labels=[f"{format_percentage(value=float(row['percentage_correct_keypoints']))} · "
                                f"{float(row['mean_error_pixels']):.1f} px" for row in keypoint_rows],
                  colour=SERIES_COLOURS[1], title="Worst key points, PCK@0.05 · mean error", x_label="PCK@0.05")
    save_figure(figure=figure, output_path=output_folder / "evaluation" / "worst_classes_and_keypoints.png")
    pyplot.close(fig=figure)
    return {"worst_class": class_rows[0]["class"].replace("_", " "),
            "worst_class_ap": round_fact(value=float(class_rows[0]["oks_average_precision_50"])),
            "worst_class_error": round_fact(value=float(class_rows[0]["mean_error_pixels"])),
            "worst_keypoint_class": keypoint_rows[0]["class"].replace("_", " "),
            "worst_keypoint_name": keypoint_rows[0]["keypoint"].replace("_", " "),
            "worst_keypoint_pck": round_fact(value=float(keypoint_rows[0]["percentage_correct_keypoints"])),
            "worst_keypoint_error": round_fact(value=float(keypoint_rows[0]["mean_error_pixels"])),
            "worst_classes_count": len(read_metric_rows(csv_path=evaluation_folder / "class_metrics.csv")),
            "worst_keypoints_count": len(read_metric_rows(csv_path=evaluation_folder / "keypoint_metrics.csv"))}


def find_panel_separator(image: numpy.ndarray) -> int:
    """
    Finds the column of the separator between the Ground Truth and Prediction panels of a keypoint sample image, the
    one grey column of uniform colour across the panels.

    Args:
        image (numpy.ndarray): The [H, W, 3] sample image.

    Returns:
        int: The first column of the separator.

    Raises:
        ValueError: If no uniform grey column lies in the middle of the image.
    """
    panel_rows = image[image.shape[0] // 4:3 * image.shape[0] // 4]
    for column in range(image.shape[1] // 3, 2 * image.shape[1] // 3):
        column_pixels = panel_rows[:, column].astype(int)
        if (column_pixels == column_pixels[0]).all() and 60 < column_pixels[0].mean() < 200:
            return column
    raise ValueError("No separator between the panels of the sample image.")


def describe_sample_image(run_folder: Path, sample_record: Dict[str, Any], inference_facts: Dict[str, Any],
                          story_objects: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the boxes of the objects of the test image in the sample image of the last evaluation, in its Ground Truth
    and Prediction panels, for the highlights of the sample scene: the panels are found from the separator between
    them, the pair from its labelled keypoints, and the single object and the cut object of the pair from their boxes,
    which reach the border where their keypoints outside the image lie; every box is kept inside its panel.

    Args:
        run_folder (Path): The folder of the keypoint run.
        sample_record (Dict[str, Any]): The test image of load_test_sample, with the boxes of its objects.
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.

    Returns:
        Dict[str, Any]: The facts of the sample image, its boxes named by the role of their objects.
    """
    sample_path = (run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{CHECKPOINT_ITERATION}" / "Samples" /
                   SAMPLE_IMAGE_NAME)
    with Image.open(fp=sample_path) as sample_image:
        image = numpy.asarray(sample_image.convert(mode="RGB"))
    separator = find_panel_separator(image=image)
    panel_width = separator
    caption_height = (image.shape[0] - panel_width) // 2
    facts = {"sample_width": int(image.shape[1]), "sample_height": int(image.shape[0])}
    objects = inference_facts["inference_objects"]

    # Regions of the image of every role, from the labelled keypoints of the pair or the boxes of the cut objects, with
    # a margin
    boxes = sample_record["sample"]["bounding_boxes"]
    single_box, cut_box = boxes[story_objects["single"]].tolist(), boxes[story_objects["cut_pair_member"]].tolist()
    regions = {"pair": numpy.concatenate([find_labelled_positions(sample_object=objects[index])
                                          for index in story_objects["pair"]]),
               "single": numpy.array([single_box[:2], single_box[2:]]),
               "cut": numpy.array([cut_box[:2], cut_box[2:]])}
    for role, positions in regions.items():
        left, top = numpy.clip(a=positions.min(axis=0) - 24, a_min=0, a_max=panel_width)
        right, bottom = numpy.clip(a=positions.max(axis=0) + 24, a_min=0, a_max=panel_width)
        for panel_name, panel_left in [("ground_truth", 0), ("prediction", image.shape[1] - panel_width)]:
            facts[f"sample_{panel_name}_{role}_box"] = [int(panel_left + left), int(caption_height + top),
                                                        int(panel_left + right), int(caption_height + bottom)]
    return facts
