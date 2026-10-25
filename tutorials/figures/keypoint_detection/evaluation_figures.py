"""
Figures and facts of the third chapter of the keypoint detection series of the tutorial videos (from candidates to
objects, the metrics and the export), made by tutorials/figures/keypoint_detection/figures.py: the heatmaps of the
trained detector on the test image at both strides and their average with the candidates extracted from it, the
grouping of the candidates of the two hollow capsules keypoint by keypoint (the data of the grouping animation, checked
against group_keypoint_candidates), the object keypoint similarity of one keypoint against its distance for the boxes
of the image, the metrics of every evaluated checkpoint read from the TensorBoard logs of the run, its worst classes and
keypoints read from its class_metrics.csv and keypoint_metrics.csv, and the boxes of the objects in its sample image.
"""

import csv
import re

import matplotlib
import numpy
import torch

from matplotlib import pyplot
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
                                                                  OBJECT_COLOURS, SAMPLE_IMAGE_NAME, write_json)
from tutorials.figures.keypoint_detection.scalar_logs import read_scalar_values
from tutorials.tutorial_builder.layout import NAMED_COLOURS
from vision_training.evaluation.detection.object_sizes import compute_box_areas
from vision_training.evaluation.keypoints.object_keypoint_similarity import (CORRECT_KEYPOINT_FRACTION,
                                                                             compute_object_keypoint_similarity)
from vision_training.models.keypoints.hrnet import BRANCH_SCALE_FACTOR
from vision_training.models.keypoints.keypoint_grouping import group_keypoint_candidates, match_candidates_to_objects

# Classes of the test image shown by the heatmap figure, one row each, and the region of the image every row crops
HEATMAP_ROWS = [{"class_name": "trapezoid", "centre": (272.5, 401.5), "crop_width": 200.0},
                {"class_name": "hollow_capsule", "centre": (367.0, 215.5), "crop_width": 170.0}]

# Layout of the heatmap figure: the column titles above two rows of four panels
HEATMAP_TITLE_HEIGHT = 44
HEATMAP_ROW_GAP = 12
HEATMAP_COLUMN_GAP = 16

# Class whose grouping the animation follows, the two capsules of the test image
GROUPING_CLASS_NAME = "hollow_capsule"

# Distances of a keypoint to its ground truth drawn by the object keypoint similarity curves, in pixels
OKS_DISTANCES = list(range(0, 31))

# Metrics of the evaluated checkpoints shown by the metrics grid, as logged in Tensorboard/Test and Test_EMA
GRID_METRICS = {"OKS mAP@0.5/area": "mAP@0.5", "OKS mAP@0.75/area": "mAP@0.75", "OKS mAP@0.95/area": "mAP@0.95",
                "OKS mAP@[0.5:0.95]/area": "mAP@[.5:.95]", "Keypoint Accuracy/PCK@0.05": "PCK@0.05",
                "Keypoint Accuracy/Mean OKS": "Mean OKS", "Keypoint Accuracy/Matched Objects": "Matched"}
ERROR_METRIC = "Keypoint Accuracy/Mean Keypoint Error (pixels)"

# Number of classes and keypoints of the dot plots of the worst classes and keypoints
WORST_COUNT = 12


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


def draw_heatmap_figure(output_folder: Path, sample_record: Dict[str, Any], inference_facts: Dict[str, Any],
                        channel_table: numpy.ndarray, keypoint_threshold: float) -> Dict[str, Any]:
    """
    Draws the heatmaps of the trained detector on the test image for the trapezoid and the hollow capsules: the image,
    the maximum of the heatmaps of the class at stride 4 and at stride 2, and their average at stride 2 the candidates
    are taken on (the stride 4 heatmaps upsampled bilinearly, as extract_keypoint_candidates does), with the
    candidates above the keypoint threshold and the labelled keypoints.

    Args:
        output_folder (Path): The figures folder of the series.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        keypoint_threshold (float): 'KeypointHead.inference.keypoint_threshold' of the run.

    Returns:
        Dict[str, Any]: The facts of the heatmaps and the boxes of the rows of the figure.
    """
    register_fonts()
    arrays = numpy.load(file=output_folder / INFERENCE_ARRAYS_NAME)
    shown_channels = arrays["channels"].tolist()
    image = sample_record["sample"]["images"].permute(1, 2, 0).numpy()[..., ::-1]
    image_size = image.shape[0]
    objects = inference_facts["inference_objects"]

    # Panels of the figure: four columns, two rows under the column titles
    panel_width = (CHART_WIDTH - 3 * HEATMAP_COLUMN_GAP) / 4
    panel_height = (CHART_HEIGHT - HEATMAP_TITLE_HEIGHT - HEATMAP_ROW_GAP) / 2
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    column_titles = ["Test image", "Heat maps at stride 4", "Heat maps at stride 2",
                     f"Average, candidates above {keypoint_threshold:g}"]
    for column_index, column_title in enumerate(column_titles):
        figure.text(x=column_index * (panel_width + HEATMAP_COLUMN_GAP) / CHART_WIDTH,
                    y=1.0 - 20 / CHART_HEIGHT, s=column_title, color=TEXT_PRIMARY_COLOUR,
                    fontsize=22 * POINTS_PER_PIXEL, ha="left", va="center")

    # Every row: the channels of its class, their heatmaps at both strides and their average
    facts = {}
    for row_index, row in enumerate(HEATMAP_ROWS):
        class_index = next(item["class_index"] for item in objects if item["class_name"] == row["class_name"])
        channels = class_channels(channel_table=channel_table, class_index=class_index)
        positions = [shown_channels.index(channel) for channel in channels]
        quarter_heatmaps = torch.from_numpy(arrays["heatmaps"][positions])[None]
        half_heatmaps = torch.from_numpy(arrays["higher_resolution_heatmaps"][positions])[None]
        averaged = (torch.nn.functional.interpolate(input=quarter_heatmaps, scale_factor=BRANCH_SCALE_FACTOR,
                                                    mode="bilinear", align_corners=False) + half_heatmaps) / 2.0
        panels = [None, quarter_heatmaps[0].max(dim=0).values.numpy(), half_heatmaps[0].max(dim=0).values.numpy(),
                  averaged[0].max(dim=0).values.numpy()]
        limits = crop_limits(centre=row["centre"], crop_width=row["crop_width"],
                             aspect_ratio=panel_width / panel_height)
        panel_top = HEATMAP_TITLE_HEIGHT + row_index * (panel_height + HEATMAP_ROW_GAP)

        # Candidates of the class above the threshold, and its labelled keypoints
        candidate_scores = arrays["candidate_scores"][channels]
        candidate_positions = arrays["candidate_positions"][channels][candidate_scores > keypoint_threshold]
        labelled_positions = numpy.concatenate([numpy.array(item["positions"]) for item in objects
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
        facts[f"heatmaps_{row['class_name']}_row_box"] = [0, round(panel_top) - 4, CHART_WIDTH,
                                                         round(panel_top + panel_height) + 4]
        facts[f"heatmaps_{row['class_name']}_average_box"] = [round(3 * (panel_width + HEATMAP_COLUMN_GAP)) - 4,
                                                             round(panel_top) - 4, CHART_WIDTH,
                                                             round(panel_top + panel_height) + 4]

        # Peaks of both strides at the labelled keypoints of the class, and the distance of its candidates to them
        facts[f"heatmaps_{row['class_name']}_peak_stride_4"] = round_fact(value=float(panels[1].max()))
        facts[f"heatmaps_{row['class_name']}_peak_stride_2"] = round_fact(value=float(panels[2].max()))
        distances = numpy.linalg.norm(candidate_positions[:, None, :] - labelled_positions[None, :, :], axis=2)
        facts[f"heatmaps_{row['class_name']}_candidates"] = int(len(candidate_positions))
        facts[f"heatmaps_{row['class_name']}_largest_error"] = round_fact(value=float(distances.min(axis=1).max()))
        facts[f"heatmaps_{row['class_name']}_mean_error"] = round_fact(value=float(distances.min(axis=1).mean()))
    save_figure(figure=figure, output_path=output_folder / "evaluation" / "heatmaps.png")
    pyplot.close(fig=figure)
    facts["heatmaps_candidates_above_threshold"] = inference_facts["inference_candidates_above_threshold"]
    facts["heatmaps_candidate_count"] = inference_facts["inference_candidate_count"]
    return facts


def trace_class_grouping(candidate_positions: numpy.ndarray, candidate_scores: numpy.ndarray,
                         candidate_tags: numpy.ndarray, channels: List[int], keypoint_threshold: float,
                         tag_threshold: float) -> Dict[str, Any]:
    """
    Groups the candidates of one class keypoint by keypoint the way group_keypoint_candidates does, with its
    match_candidates_to_objects, recording every step: the candidates of every keypoint with the object they join, and
    the mean tag of every object after the step.

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
                                    "object": object_index, "starts_object": starts_object})
        steps.append({"keypoint_index": keypoint_index, "candidates": step_candidates,
                      "object_means": [round_fact(value=numpy.mean(tags_of_object)) for tags_of_object in object_tags]})
    return {"steps": steps, "objects": [{"keypoints": keypoints, "scores": scores,
                                         "score": round_fact(value=sum(scores) / len(channels))}
                                        for keypoints, scores in zip(object_keypoints, object_scores)]}


def write_grouping_data(output_folder: Path, inference_facts: Dict[str, Any], channel_table: numpy.ndarray,
                        inference_configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Writes the data of the grouping animation for the hollow capsules of the test image, every candidate with the
    labelled capsule it lies on (the nearest labelled keypoint of the same name), checked against the objects of
    group_keypoint_candidates, and gives the facts of the grouping of the whole image.

    Args:
        output_folder (Path): The figures folder of the series.
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        inference_configuration (Dict[str, Any]): 'KeypointHead.inference' of the run.

    Returns:
        Dict[str, Any]: The facts of the grouping.

    Raises:
        ValueError: If the traced grouping differs from group_keypoint_candidates.
    """
    arrays = numpy.load(file=output_folder / INFERENCE_ARRAYS_NAME)
    positions, scores, tags = arrays["candidate_positions"], arrays["candidate_scores"], arrays["candidate_tags"]
    objects = inference_facts["inference_objects"]
    capsules = [item for item in objects if item["class_name"] == GROUPING_CLASS_NAME]
    class_index = capsules[0]["class_index"]
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
        raise ValueError("The traced grouping of the hollow capsules differs from group_keypoint_candidates.")

    # The labelled capsule under every candidate, by the nearest labelled keypoint of the same name
    capsule_positions = numpy.array([item["positions"] for item in capsules])
    capsule_counts = numpy.zeros(shape=(len(trace["objects"]), len(capsules)), dtype=int)
    for step in trace["steps"]:
        keypoint_index = step["keypoint_index"]
        for candidate in step["candidates"]:
            distances = numpy.linalg.norm(capsule_positions[:, keypoint_index] - numpy.array(candidate["position"]),
                                          axis=1)
            candidate["capsule"] = int(distances.argmin())
            candidate["distance"] = round_fact(value=distances.min())
            capsule_counts[candidate["object"], candidate["capsule"]] += 1

    # Every object belongs to the capsule most of its keypoints lie on, the others being exchanged keypoints
    object_capsules = capsule_counts.argmax(axis=1)
    exchanged = []
    for step in trace["steps"]:
        for candidate in step["candidates"]:
            candidate["exchanged"] = bool(candidate["capsule"] != object_capsules[candidate["object"]])
            keypoint_name = capsules[0]["keypoint_names"][step["keypoint_index"]].replace("_", " ")
            if candidate["exchanged"] and keypoint_name not in exchanged:
                exchanged.append(keypoint_name)
    write_json(data={"keypoint_names": [name.replace("_", " ") for name in capsules[0]["keypoint_names"]],
                     "steps": trace["steps"], "objects": trace["objects"],
                     "capsule_colours": [OBJECT_COLOURS[objects.index(item)] for item in capsules],
                     "tag_threshold": inference_configuration["tag_threshold"],
                     "keypoint_threshold": inference_configuration["keypoint_threshold"]},
               output_path=output_folder / "evaluation" / "grouping.json")

    # Objects of the whole image, kept by the evaluation or by the sample images and the deployment, and the score of
    # the best object of every class of the image
    kept_objects = int((grouped["scores"] >= inference_configuration["object_score_threshold"]).sum())
    class_scores = {}
    for item in objects:
        class_objects = grouped["scores"][grouped["labels"] == item["class_index"]]
        class_scores[f"grouping_{item['class_name']}_best_score"] = round_fact(value=float(class_objects.max()))
    return {"grouping_exchanged_keypoints": ", ".join(exchanged), "grouping_exchanged_count": len(exchanged),
            "grouping_capsule_objects": len(trace["objects"]),
            "grouping_exchanged_per_object": [int(capsule_counts[object_index].sum() -
                                                  capsule_counts[object_index].max())
                                              for object_index in range(len(trace["objects"]))],
            "grouping_capsule_scores": [item["score"] for item in trace["objects"]],
            "grouping_all_objects": int(len(grouped["scores"])), "grouping_kept_objects": kept_objects, **class_scores,
            "grouping_object_score_threshold": inference_configuration["object_score_threshold"]}


def describe_object_keypoint_similarity(sample_record: Dict[str, Any], inference_facts: Dict[str, Any],
                                        default_sigma: float) -> Dict[str, Any]:
    """
    Computes the object keypoint similarity of a one-keypoint object against the distance of its keypoint, with
    compute_object_keypoint_similarity and the box areas of compute_box_areas, for the trapezoid and the first capsule
    of the test image: the curves of the OKS scene, the distance at which the similarity falls to 0.5 and the PCK
    tolerance of both boxes.

    Args:
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.
        default_sigma (float): 'evaluation.default_keypoint_sigma' of the run.

    Returns:
        Dict[str, Any]: The facts of the similarity curves.
    """
    sample = sample_record["sample"]
    areas = compute_box_areas(boxes=sample["bounding_boxes"].to(dtype=torch.float64))
    distances = torch.tensor(data=OKS_DISTANCES, dtype=torch.float64)
    facts = {}
    for name, class_name in [("trapezoid", "trapezoid"), ("capsule", "hollow_capsule")]:
        object_index = next(index for index, item in enumerate(inference_facts["inference_objects"])
                            if item["class_name"] == class_name)
        area = areas[object_index]
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
    facts["oks_default_sigma"] = default_sigma
    facts["oks_constant"] = round_fact(value=2 * default_sigma)
    return facts


def read_results_header(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the number of test images and of test objects with keypoints of the evaluations of the run from the first
    report block of its Results.txt.

    Args:
        run_folder (Path): The folder of the keypoint run.

    Returns:
        Dict[str, Any]: The 'metrics_test_images' and 'metrics_test_objects' facts.

    Raises:
        ValueError: If the report does not give them.
    """
    report = (run_folder / "Results.txt").read_text(encoding="utf-8")
    images_match = re.search(pattern=r"Iteration \d+ · (\d+) Test Images", string=report)
    objects_match = re.search(pattern=r"^\s+All\s+(\d+)\s", string=report, flags=re.MULTILINE)
    if images_match is None or objects_match is None:
        raise ValueError(f"No test image or object count in {run_folder / 'Results.txt'}.")
    return {"metrics_test_images": int(images_match.group(1)), "metrics_test_objects": int(objects_match.group(1))}


def describe_metrics(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the metrics of every evaluated checkpoint of the run, for the trained and the averaged weights, from its
    Tensorboard/Test and Tensorboard/Test_EMA logs: the values of the metrics grid and the facts the narration quotes.

    Args:
        run_folder (Path): The folder of the keypoint run.

    Returns:
        Dict[str, Any]: The facts of the metrics.
    """
    tags = [*GRID_METRICS, ERROR_METRIC]
    weights_values = {"trained": read_scalar_values(log_folder=run_folder / "Tensorboard" / "Test", tags=tags),
                      "ema": read_scalar_values(log_folder=run_folder / "Tensorboard" / "Test_EMA", tags=tags)}

    # Grid of the metrics, one column per checkpoint and kind of weights
    columns = [(iteration, weights) for iteration in EVALUATED_ITERATIONS for weights in ["trained", "ema"]]
    facts = {"metrics_grid_values": [[round(weights_values[weights][tag][iteration], 4)
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
    value on an axis from 0.8 to 1.0, and its value written after it.

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
    axes.hlines(y=rows, xmin=0.8, xmax=values, color="#%02x%02x%02x" % NAMED_COLOURS["border"],
                linewidth=2 * POINTS_PER_PIXEL)
    axes.scatter(values, rows, s=(12 * POINTS_PER_PIXEL) ** 2, color=colour, zorder=3)
    for row, value, value_label in zip(rows, values, value_labels):
        axes.text(x=value + 0.004, y=row, s=value_label, color=TEXT_SECONDARY_COLOUR, va="center", ha="left",
                  fontsize=17 * POINTS_PER_PIXEL)
    axes.set_yticks(ticks=rows, labels=labels)
    axes.set_ylim(bottom=len(labels) - 0.5, top=-0.5)
    axes.set_xlim(left=0.8, right=1.0)
    axes.tick_params(axis="y", labelcolor=TEXT_PRIMARY_COLOUR, labelsize=17 * POINTS_PER_PIXEL)
    axes.grid(axis="y", visible=False)


def draw_worst_figure(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Draws the worst classes by OKS AP@0.5 and the worst keypoints by PCK@0.05 of the evaluation of the last checkpoint
    of the run, from its class_metrics.csv and keypoint_metrics.csv (trained weights).

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the keypoint run.

    Returns:
        Dict[str, Any]: The facts of the worst class and keypoints.
    """
    register_fonts()
    evaluation_folder = run_folder / "Evaluation" / f"Iteration_{CHECKPOINT_ITERATION}"
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
                  value_labels=[f"{float(row['oks_average_precision_50']):.3f}" for row in class_rows],
                  colour=SERIES_COLOURS[0], title=f"Worst classes, OKS AP@0.5", x_label="AP@0.5")
    draw_dot_plot(axes=keypoint_axes, labels=[f"{row['class'].replace('_', ' ')} · {row['keypoint'].replace('_', ' ')}"
                                              for row in keypoint_rows],
                  values=[float(row["percentage_correct_keypoints"]) for row in keypoint_rows],
                  value_labels=[f"{float(row['percentage_correct_keypoints']):.3f} · "
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


def describe_sample_image(run_folder: Path, inference_facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the boxes of the objects of the test image in the sample image of the last evaluation, in its Ground Truth
    and Prediction panels, for the highlights of the sample scene: the panels are found from the separator between
    them, the objects from their labelled keypoints.

    Args:
        run_folder (Path): The folder of the keypoint run.
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.

    Returns:
        Dict[str, Any]: The facts of the sample image.
    """
    sample_path = run_folder / "Evaluation" / f"Iteration_{CHECKPOINT_ITERATION}" / "Samples" / SAMPLE_IMAGE_NAME
    with Image.open(fp=sample_path) as sample_image:
        image = numpy.asarray(sample_image.convert(mode="RGB"))
    separator = find_panel_separator(image=image)
    panel_width = separator
    caption_height = (image.shape[0] - panel_width) // 2
    facts = {"sample_width": int(image.shape[1]), "sample_height": int(image.shape[0])}

    # Box of every class of the image in both panels, from its labelled keypoints with a margin
    for class_name in ["trapezoid", "hollow_capsule"]:
        positions = numpy.concatenate([numpy.array(item["positions"]) for item in inference_facts["inference_objects"]
                                       if item["class_name"] == class_name])
        left, top = positions.min(axis=0) - 24
        right, bottom = positions.max(axis=0) + 24
        for panel_name, panel_left in [("ground_truth", 0), ("prediction", image.shape[1] - panel_width)]:
            facts[f"sample_{panel_name}_{class_name}_box"] = [int(panel_left + left), int(caption_height + top),
                                                             int(panel_left + right), int(caption_height + bottom)]
    return facts
