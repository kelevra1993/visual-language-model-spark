"""
Figures and facts of the second chapter of the keypoint detection series of the tutorial videos (the network and its
losses), made by tutorials/figures/keypoint_detection/figures.py: the shapes of the stem, of the branches of the high
resolution network and of both heads, recorded on the test image by the inference of keypoint_inference.py, and the
parameters of every component (the labels of the architecture diagrams and the data of the exchange animation); the
heatmap loss of the test image split into the part of the keypoint cells and of the other cells, with every cell at the
prior probability of the untrained heads and with the trained weights; the tags of the trained detector at the
keypoints of the objects of the image (the data of the tag animation), with the distance of the mean tags of its two
objects of one class; and the training curves of the run read from its TensorBoard logs.
"""

import math
import re

import matplotlib
import numpy
import torch

from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                REVEAL_MARGIN, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                compute_pixel_box, format_thousands,
                                                register_fonts, round_fact, save_figure, style_chart_panel)
from tutorials.figures.keypoint_detection.keypoint_inference import build_sample_targets, split_heatmap_loss
from tutorials.figures.keypoint_detection.keypoint_sources import OBJECT_COLOURS, write_json
from tutorials.figures.keypoint_detection.scalar_logs import (SMOOTHING_WEIGHT, draw_smoothed_curve, read_scalars,
                                                             smooth_values)
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.keypoints.hrnet import (BRANCH_SCALE_FACTOR, HigherResolutionBranch,
                                                    HighResolutionNetwork)
from vision_training.models.keypoints.keypoint_detector import HEATMAP_PRIOR_PROBABILITY, HEATMAP_STRIDE

# Scalars of the training curves, logged by the keypoint trainer in Tensorboard/Train, with their colours: the total
# loss alone on the first panel (the heatmap loss, almost all of it, is read for the facts only), the tag losses on the
# second without their raw values, which jump between 0 and their maximum from one batch to the next
LOSS_CURVES = {"Total Loss": SERIES_COLOURS[0]}
TAG_CURVES = {"Tag Pull Loss": SERIES_COLOURS[2], "Tag Push Loss": SERIES_COLOURS[3]}
FACT_ONLY_CURVES = ["Heatmap Loss"]

# Iterations between two labelled ticks of the training curves, also the window of the mean tag losses the narration
# compares at the start and at the end of the run
ITERATION_TICK_STEP = 1000

# Top of the total loss panel (log scale), above the first losses of a run, the lowest top of the tag loss panel, and
# the iterations after which the smoothed tag losses must stay inside their panel
LOSS_PANEL_TOP = 20.0
TAG_PANEL_TOP = 0.02
SMOOTHING_SETTLED_ITERATIONS = 100

# Shares the narration of the training curves states: the heatmap loss is almost all of the last smoothed total loss,
# the pull loss of the last window is less than half the one of the first window ("falls"), the push loss of the last
# window more than this share of the one of the first window ("barely falls")
HEATMAP_LOSS_SHARE = 0.9
PULL_LOSS_FALL_SHARE = 0.5
PUSH_LOSS_KEPT_SHARE = 0.8


def format_shape(shape: List[int]) -> str:
    """
    Writes the shape of a tensor without its batch dimension, as the labels of the architecture diagrams show it.

    Args:
        shape (List[int]): The [B, C, H, W] shape.

    Returns:
        str: The "[C, H, W]" text.
    """
    return "[" + ", ".join(str(size) for size in shape[1:]) + "]"


def describe_network(inference_facts: Dict[str, Any], configuration: Dict[str, Any],
                     output_folder: Path) -> Dict[str, Any]:
    """
    Turns the shapes and parameters recorded by the inference into the facts quoted by the architecture diagrams (with
    the channels of the predictions and of the higher resolution branch, the stages and branches of the network, the
    stride of its lowest branch and the layers of the refinement of the branch), and writes the data of the exchange
    animation: the widths and resolutions of the branches of every stage and the number of exchange modules of every
    stage.

    Args:
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.
        configuration (Dict[str, Any]): The repository configuration, with the HRNet section of the run.
        output_folder (Path): The figures folder of the series.

    Returns:
        Dict[str, Any]: The flat facts of the network.
    """
    shapes = inference_facts["inference_shapes"]
    image_size = configuration["Data"]["image_settings"]["size"]
    facts = {"network_input_shape": format_shape(shape=[1, 3, image_size, image_size]),
             "network_stem_shape": format_shape(shape=shapes["stem"]["output"][0]),
             "network_features_shape": format_shape(shape=shapes["high_resolution_network"]["output"]),
             "network_predictions_shape": format_shape(shape=shapes["prediction_convolution"]["output"]),
             "network_branch_input_shape": format_shape(shape=shapes["transposed_convolution"]["input"]),
             "network_branch_upsampled_shape": format_shape(shape=shapes["transposed_convolution"]["output"]),
             "network_branch_heatmaps_shape": format_shape(shape=shapes["higher_resolution_heatmap_head"]["output"])}

    # Branches of every stage, entering its exchange modules
    stages = []
    modules_per_stage = configuration["HRNet"]["modules_per_stage"]
    for stage_index, number_modules in enumerate(modules_per_stage):
        branch_shapes = shapes[f"transition_{stage_index + 1}"]["output"]
        stages.append({"modules": number_modules, "branches": [{"channels": shape[1], "height": shape[2],
                                                                "width": shape[3]} for shape in branch_shapes]})
        for branch_index, shape in enumerate(branch_shapes):
            facts[f"network_branch_{branch_index + 1}_shape"] = format_shape(shape=shape)
    write_json(data={"stages": stages, "stem": shapes["stem"]["output"][0][1:],
                     "image_size": image_size, "base_width": configuration["HRNet"]["base_width"]},
               output_path=output_folder / "network" / "high_resolution_network.json")

    # Channels of the stem, of the quarter resolution predictions and of the input of the higher resolution branch, the
    # branches of the last stage, the stride of the lowest one and the side of the quarter resolution maps
    prediction_channels = shapes["prediction_convolution"]["output"][1]
    last_branch_shapes = shapes[f"transition_{len(modules_per_stage)}"]["output"]
    facts.update({"network_stem_channels": shapes["stem"]["output"][0][1],
                  "network_prediction_channels": prediction_channels,
                  "network_branch_input_channels": shapes["transposed_convolution"]["input"][1],
                  "network_stage_count": len(modules_per_stage), "network_branch_count": len(last_branch_shapes),
                  "network_lowest_stride": image_size // last_branch_shapes[-1][2],
                  "heatmap_size": shapes["high_resolution_network"]["output"][2]})

    # Layers of the refinement of the higher resolution branch, read from the branch built as the detector builds it
    network_configuration = configuration["HRNet"]
    higher_resolution_branch = HigherResolutionBranch(
        feature_channels=network_configuration["base_width"], prediction_channels=prediction_channels,
        heatmap_channels=prediction_channels // 2, hidden_width=configuration["KeypointHead"]["hidden_width"],
        blocks_per_branch=network_configuration["blocks_per_branch"], modules=network_configuration["modules"],
        normalization=network_configuration["normalization"], device=torch.device("cpu"), dtype=torch.float32)
    facts["branch_refinement_layers"] = higher_resolution_branch.refinement.number_layers

    # Parameters of every component, as the model size panel of the trainer counts them
    parameters = inference_facts["inference_parameters"]
    facts.update({"network_modules_per_stage": str(modules_per_stage),
                  "network_total_modules": sum(modules_per_stage),
                  "parameters_stem": parameters["Backbone"],
                  "parameters_high_resolution_network": parameters["High Resolution Network"],
                  "parameters_head": parameters["Keypoint Head"],
                  "parameters_branch": parameters["Higher Resolution Branch"],
                  "parameters_total": parameters["Total Model"],
                  "parameters_megabytes": round_fact(value=parameters["Total Model"] * 4 / 1024 ** 2),
                  "parameters_values": [parameters["Backbone"], parameters["High Resolution Network"],
                                        parameters["Keypoint Head"], parameters["Higher Resolution Branch"]],
                  "parameters_labels": [f"{parameters['Backbone']:,}", f"{parameters['High Resolution Network']:,}",
                                        f"{parameters['Keypoint Head']:,}",
                                        f"{parameters['Higher Resolution Branch']:,}"]})
    return facts


def describe_architecture_constants(configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the values of the architecture of the keypoint detector that the chapters quote and that are derived from
    the constants of the repository or checked across the configuration rather than read from one key: the strides of
    the two sets of heatmaps and their fractions of the image (HEATMAP_STRIDE, then BRANCH_SCALE_FACTOR finer), the
    cells the finer heatmaps hold per cell of the quarter resolution ones, the stem blocks the checks require
    (check_keypoint_stem_configuration: the stem must end at HEATMAP_STRIDE, every block but the last followed by a
    pooling of BRANCH_SCALE_FACTOR unless Backbone.last_max_pooling adds one), the blocks of the stem of the
    configuration, and the weight of both tag losses, which the narration quotes once.

    Args:
        configuration (Dict[str, Any]): The repository configuration, with the model sections of the run.

    Returns:
        Dict[str, Any]: The flat facts of the architecture.

    Raises:
        ValueError: If the pull and push tag losses have different weights, the chapters saying both are weighted alike.
    """
    head_configuration = configuration["KeypointHead"]
    if head_configuration["tag_pull_weight"] != head_configuration["tag_push_weight"]:
        raise ValueError(f"KeypointHead.tag_pull_weight {head_configuration['tag_pull_weight']} differs from "
                         f"tag_push_weight {head_configuration['tag_push_weight']}: the network chapter says both tag "
                         f"losses share one weight, reword it.")

    # Poolings the stem needs to reach the heatmap stride, and the blocks giving them without or with a last pooling
    stem_poolings = round(math.log(HEATMAP_STRIDE) / math.log(BRANCH_SCALE_FACTOR))
    branch_stride = HEATMAP_STRIDE // BRANCH_SCALE_FACTOR
    return {"branch_stride": branch_stride, "heatmap_fraction": f"1/{HEATMAP_STRIDE}",
            "branch_fraction": f"1/{branch_stride}", "branch_cell_ratio": BRANCH_SCALE_FACTOR ** 2,
            "stem_blocks_without_last_pooling": stem_poolings + 1, "stem_blocks_with_last_pooling": stem_poolings,
            "stem_block_count": len(configuration["Backbone"]["convolutions"]),
            "tag_loss_weight": head_configuration["tag_pull_weight"]}


def describe_heatmap_losses(inference_facts: Dict[str, Any], sample_record: Dict[str, Any],
                            channel_table: numpy.ndarray, heatmap_sigma: float) -> Dict[str, Any]:
    """
    Splits the heatmap loss of the test image into the part of the keypoint cells and of all the other cells, at both
    strides, for the heads at their initial prior probability of 0.01 on every cell and for the trained heads, the
    bars of the loss scene.

    Args:
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference, with the split of the trained heads.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        heatmap_sigma (float): 'KeypointHead.heatmap_sigma' of the run.

    Returns:
        Dict[str, Any]: The flat facts of the heatmap losses.
    """
    number_channels = int(channel_table.max()) + 1
    targets = build_sample_targets(sample=sample_record["sample"], channel_table=channel_table,
                                   number_channels=number_channels, heatmap_sigma=heatmap_sigma)

    # Every cell at the prior probability, the logit the bias of the heatmap channels starts at
    prior_logit = math.log(HEATMAP_PRIOR_PROBABILITY / (1.0 - HEATMAP_PRIOR_PROBABILITY))
    prior_splits = {f"stride_{stride}": split_heatmap_loss(logits=torch.full_like(input=stride_targets,
                                                                                  fill_value=prior_logit),
                                                           targets=stride_targets)
                    for stride, stride_targets in zip([4, 2], targets["heatmaps"])}
    trained_splits = inference_facts["inference_heatmap_loss_splits"]

    # Bars of both kinds of cells at both strides, before and after the training
    facts = {"loss_prior_logit": round_fact(value=prior_logit)}
    for state, splits in [("prior", prior_splits), ("trained", trained_splits)]:
        values = [splits["stride_4"]["keypoint_cells"], splits["stride_4"]["other_cells"],
                  splits["stride_2"]["keypoint_cells"], splits["stride_2"]["other_cells"]]
        facts[f"loss_{state}_values"] = values
        facts[f"loss_{state}_labels"] = [f"{value:.3f}" for value in values]
        facts[f"loss_{state}_total"] = round_fact(value=sum(values))
        for stride in [4, 2]:
            for part in ["keypoint_cells", "other_cells"]:
                facts[f"loss_{state}_stride_{stride}_{part}"] = splits[f"stride_{stride}"][part]
    facts["loss_maximum"] = max(facts["loss_prior_values"])
    facts["loss_positive_cells"] = prior_splits["stride_4"]["positive_cells"]
    facts["loss_cells_stride_4"] = prior_splits["stride_4"]["cells"]
    facts["loss_cells_stride_2"] = prior_splits["stride_2"]["cells"]
    facts["loss_trained_model_heatmap_loss"] = inference_facts["inference_heatmap_loss"]
    return facts


def describe_tags(inference_facts: Dict[str, Any], story_objects: Dict[str, Any],
                  output_folder: Path) -> Dict[str, Any]:
    """
    Writes the data of the tag animation, the tags of the trained detector at the labelled keypoints of every object
    of the test image, and gives the facts of the tag losses of the image, with the distance of the mean tags of the
    pair (its two objects of one class), which the chapters say the training did not separate.

    Args:
        inference_facts (Dict[str, Any]): The facts of run_keypoint_inference.
        story_objects (Dict[str, Any]): The objects named by their role, from select_story_objects.
        output_folder (Path): The figures folder of the series.

    Returns:
        Dict[str, Any]: The flat facts of the tags.

    Raises:
        ValueError: If the mean tags of the pair are half a unit apart or more, the grouping then telling them apart.
    """
    objects = []
    facts = {}
    for object_index, sample_object in enumerate(inference_facts["inference_objects"]):
        tags = numpy.array(sample_object["tags"])
        objects.append({"name": sample_object["class_name"].replace("_", " "),
                        "class_index": sample_object["class_index"],
                        "colour": OBJECT_COLOURS[object_index], "tags": sample_object["tags"],
                        "keypoint_names": sample_object["keypoint_names"], "mean": round_fact(value=tags.mean())})
        facts[f"tags_object_{object_index + 1}_mean"] = round_fact(value=tags.mean())
        facts[f"tags_object_{object_index + 1}_spread"] = round_fact(value=numpy.abs(tags - tags.mean()).max())
        facts[f"tags_object_{object_index + 1}_name"] = sample_object["class_name"].replace("_", " ")

    # The pair, the only objects the push loss compares, not separated when their mean tags round to the same unit
    first_index, second_index = story_objects["pair"]
    pair_distance = abs(objects[first_index]["mean"] - objects[second_index]["mean"])
    if pair_distance >= 0.5:
        raise ValueError(f"The mean tags of the two {objects[first_index]['name']} objects are {pair_distance:.3f} "
                         f"apart, so the trained detector separates them: reword the tag and grouping scenes, which "
                         f"say it does not.")
    facts.update({"tags_pull_loss": inference_facts["inference_tag_pull_loss"],
                  "tags_push_loss": inference_facts["inference_tag_push_loss"],
                  "tags_weighted_pull_loss": inference_facts["inference_weighted_tag_pull_loss"],
                  "tags_weighted_push_loss": inference_facts["inference_weighted_tag_push_loss"],
                  "tags_pair_distance": round_fact(value=pair_distance),
                  "tags_minimum": round_fact(value=min(min(item["tags"]) for item in objects)),
                  "tags_maximum": round_fact(value=max(max(item["tags"]) for item in objects))})
    write_json(data={"objects": objects, "pull_loss": facts["tags_pull_loss"], "push_loss": facts["tags_push_loss"]},
               output_path=output_folder / "network" / "tags.json")
    return facts


def style_curve_panel(axes: matplotlib.axes.Axes, title: str, last_iteration: int, y_label: str) -> None:
    """
    Gives a panel of the training curves the anatomy of the charts of the videos, with the iterations of the run on
    its horizontal axis.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        title (str): The title of the panel.
        last_iteration (int): The last iteration of the run.
        y_label (str): The name of the value axis.
    """
    style_chart_panel(axes=axes, title=title, x_label="Iteration", y_label=y_label)
    axes.set_xlim(left=0, right=last_iteration)
    axes.set_xticks(ticks=list(range(0, last_iteration + 1, ITERATION_TICK_STEP)))
    axes.xaxis.set_major_formatter(formatter=FuncFormatter(func=format_thousands))


def draw_training_curves(output_folder: Path, run_folder: Path, with_curves: bool,
                         tag_loss_weight: float) -> Dict[str, Any]:
    """
    Draws the training curves of the keypoint run, its total and heatmap losses (log scale) and its weighted tag
    losses, read from Tensorboard/Train: the empty axes for the base image of the scene, or the whole chart for its
    overlay, whose curves the scene uncovers. The facts give the first, last and last smoothed value of every loss and
    the mean of every loss over the first and the last ITERATION_TICK_STEP iterations, and the shares the narration
    states are checked.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the keypoint run.
        with_curves (bool): Whether the curves are drawn (the overlay) or only the axes (the base image).
        tag_loss_weight (float): The weight of both tag losses (describe_architecture_constants), named in the title of
            their panel.

    Returns:
        Dict[str, Any]: The facts of the curves and the boxes of the panels.

    Raises:
        ValueError: If the total loss leaves its panel, or if the losses no longer behave as the narration says (the
            heatmap loss almost all of the total, the pull loss falling, the push loss barely falling).
    """
    register_fonts()
    log_folder = run_folder / "Tensorboard" / "Train"
    curves = {tag: read_scalars(log_folder=log_folder, tag=tag) for tag in [*LOSS_CURVES, *TAG_CURVES,
                                                                              *FACT_ONLY_CURVES]}
    last_iteration = int(curves["Total Loss"][0][-1])
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    loss_axes = figure.add_axes(rect=(0.07, 0.14, 0.40, 0.74))
    tag_axes = figure.add_axes(rect=(0.57, 0.14, 0.40, 0.74))
    style_curve_panel(axes=loss_axes, title="Total loss (log scale)", last_iteration=last_iteration, y_label="Loss")
    style_curve_panel(axes=tag_axes, title=f"Tag losses, weighted by {tag_loss_weight:g}",
                      last_iteration=last_iteration, y_label="Loss")

    # Total loss panel from the power of ten below its lowest raw value, tag panel above the highest smoothed value
    # once the smoothing has settled (the first smoothed values, of a few batches, may leave the panel)
    total_values = curves["Total Loss"][1]
    if total_values.max() > LOSS_PANEL_TOP:
        raise ValueError(f"The total loss of the run reaches {total_values.max():.2f}, above the top {LOSS_PANEL_TOP} "
                         f"of its panel.")
    loss_axes.set_yscale(value="log")
    loss_axes.set_ylim(bottom=10.0 ** math.floor(math.log10(total_values.min())), top=LOSS_PANEL_TOP)
    largest_tag_loss = max(smooth_values(values=curves[tag][1], weight=SMOOTHING_WEIGHT)[
        curves[tag][0] > SMOOTHING_SETTLED_ITERATIONS].max() for tag in TAG_CURVES)
    tag_axes.set_ylim(bottom=0.0, top=max(TAG_PANEL_TOP, math.ceil(1.25 * largest_tag_loss / 0.005) * 0.005))

    # Curves of both panels, smoothed like TensorBoard, with their legend
    facts = {"curves_last_iteration": last_iteration}
    for axes, panel_curves in [(loss_axes, LOSS_CURVES), (tag_axes, TAG_CURVES)]:
        for tag, colour in panel_curves.items():
            iterations, values = curves[tag]
            fact_name = tag.lower().replace(" ", "_")
            legend_label = tag.replace("Heatmap", "Heat map")
            if with_curves:
                draw_smoothed_curve(axes=axes, iterations=iterations, values=values, colour=colour,
                                    label=legend_label, smoothing=True, raw_opacity=0.22 if axes is loss_axes else 0.0)
            else:
                # The base image only holds the legend entry of the curve, its line uncovered by the reveal
                axes.plot([], [], color=colour, linewidth=3 * POINTS_PER_PIXEL, label=legend_label)
            smoothed = smooth_values(values=values, weight=SMOOTHING_WEIGHT)
            facts[f"curves_{fact_name}_first"] = round_fact(value=values[0])
            facts[f"curves_{fact_name}_last"] = round_fact(value=values[-1])
            facts[f"curves_{fact_name}_last_smoothed"] = round_fact(value=smoothed[-1])
        axes.legend(loc="upper right", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)

    # The heatmap loss, read for the facts only
    for tag in FACT_ONLY_CURVES:
        iterations, values = curves[tag]
        fact_name = tag.lower().replace(" ", "_")
        facts[f"curves_{fact_name}_first"] = round_fact(value=values[0])
        facts[f"curves_{fact_name}_last"] = round_fact(value=values[-1])
        facts[f"curves_{fact_name}_last_smoothed"] = round_fact(value=smooth_values(values=values,
                                                                                    weight=SMOOTHING_WEIGHT)[-1])

    # Mean of every loss over the first and the last window of iterations, steadier than the smoothed curves for the
    # tag losses, which jump between 0 and their maximum from one batch to the next
    for tag, (iterations, values) in curves.items():
        fact_name = tag.lower().replace(" ", "_")
        facts[f"curves_{fact_name}_first_window_mean"] = round_fact(
            value=values[iterations <= ITERATION_TICK_STEP].mean())
        facts[f"curves_{fact_name}_last_window_mean"] = round_fact(
            value=values[iterations > last_iteration - ITERATION_TICK_STEP].mean())
    facts["curves_window_iterations"] = ITERATION_TICK_STEP

    # The shares the narration states
    heatmap_share = facts["curves_heatmap_loss_last_smoothed"] / facts["curves_total_loss_last_smoothed"]
    pull_share = facts["curves_tag_pull_loss_last_window_mean"] / facts["curves_tag_pull_loss_first_window_mean"]
    push_share = facts["curves_tag_push_loss_last_window_mean"] / facts["curves_tag_push_loss_first_window_mean"]
    if heatmap_share < HEATMAP_LOSS_SHARE or pull_share > PULL_LOSS_FALL_SHARE or push_share < PUSH_LOSS_KEPT_SHARE:
        raise ValueError(f"The training curves no longer behave as the narration says: heatmap share of the total "
                         f"{heatmap_share:.3f} (at least {HEATMAP_LOSS_SHARE}), pull loss of the last window "
                         f"{pull_share:.3f} of the first (at most {PULL_LOSS_FALL_SHARE}), push loss {push_share:.3f} "
                         f"(at least {PUSH_LOSS_KEPT_SHARE}).")
    figure.canvas.draw()
    facts["curves_loss_panel_box"] = compute_pixel_box(figure=figure, axes=loss_axes, with_labels=True, margin=6)
    facts["curves_tag_panel_box"] = compute_pixel_box(figure=figure, axes=tag_axes, with_labels=True, margin=6)
    facts["curves_loss_reveal_region"] = compute_pixel_box(figure=figure, axes=loss_axes, with_labels=False,
                                                           margin=REVEAL_MARGIN)
    facts["curves_tag_reveal_region"] = compute_pixel_box(figure=figure, axes=tag_axes, with_labels=False,
                                                          margin=REVEAL_MARGIN)
    figure_name = "training_curves.png" if with_curves else "training_curves_axes.png"
    save_figure(figure=figure, output_path=output_folder / "network" / figure_name)
    pyplot.close(fig=figure)
    return facts


def describe_stem_and_run(configuration: Dict[str, Any], run_folder: Path) -> Dict[str, Any]:
    """
    Gives the facts of the stem and of the training of the run that the network chapter quotes: the receptive field
    of the stem (compute_receptive_field_information of the backbone built from the configuration), the residual
    shortcuts every stem block and every branch of the high resolution network adds (count_residual_additions, none
    in a block changing its channel count with two layers), and the throughput of the training read from the summaries
    of its training.log.

    Args:
        configuration (Dict[str, Any]): The repository configuration, with the model sections of the run.
        run_folder (Path): The folder of the keypoint run, whose parent folder holds the training.log of the run.

    Returns:
        Dict[str, Any]: The flat facts of the stem and of the training.

    Raises:
        FileNotFoundError: If the training.log of the run is missing.
    """
    image_size = configuration["Data"]["image_settings"]["size"]
    stem = build_backbone(backbone_configuration=configuration["Backbone"], input_image_size=image_size,
                          device=torch.device("cpu"), dtype=torch.float32)
    receptive_field = stem.compute_receptive_field_information()[-1]["receptive_field"]
    network_configuration = configuration["HRNet"]
    network = HighResolutionNetwork(input_channels=list(configuration["Backbone"]["convolutions"].values())[-1][1],
                                    base_width=network_configuration["base_width"],
                                    modules_per_stage=network_configuration["modules_per_stage"],
                                    blocks_per_branch=network_configuration["blocks_per_branch"],
                                    modules=network_configuration["modules"],
                                    normalization=network_configuration["normalization"], device=torch.device("cpu"),
                                    dtype=torch.float32)

    # Throughput of every summary of the training log, in images per second
    log_path = run_folder.parents[1] / "training.log"
    if not log_path.exists():
        raise FileNotFoundError(f"No training log {log_path} of the keypoint run.")
    throughputs = [float(value) for value in re.findall(pattern=r"Throughput: ([\d.]+) Images Per Second",
                                                        string=log_path.read_text(encoding="utf-8"))]
    return {"stem_receptive_field": int(receptive_field),
            "stem_shortcuts": [block.count_residual_additions() for block in stem.convolution_blocks],
            "branch_shortcuts": network.stages[0][0].branches[0].count_residual_additions(),
            "training_throughput": round_fact(value=sum(throughputs) / len(throughputs)),
            "training_throughput_minimum": min(throughputs), "training_throughput_maximum": max(throughputs)}
