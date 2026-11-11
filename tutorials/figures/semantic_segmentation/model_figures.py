"""
Figures and facts of the model scenes of the semantic segmentation series of the tutorial videos, made by
tutorials/figures/semantic_segmentation/figures.py: the shapes of the skip connections of the encoder and of every level
of the decoder recorded on the test image by the inference of segmentation_inference.py, the parameters of the
components and the receptive field of the encoder (the labels of the U-Net diagram), the residual shortcuts every block
adds once residual_nets is turned on (the caveat of the decoder), and the figure of what every level computes, the mean
absolute activation of the skip connections and of the decoder levels next to the predicted class map.
"""

import copy

import matplotlib
import numpy
import torch

from matplotlib import pyplot
from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SURFACE_COLOUR, TEXT_PRIMARY_COLOUR, TEXT_SECONDARY_COLOUR,
                                                register_fonts, round_fact, save_figure)
from tutorials.figures.semantic_segmentation.segmentation_inference import INFERENCE_ARRAYS_NAME
from tutorials.figures.semantic_segmentation.segmentation_sources import build_class_palette
from tutorial_builder.layout import NAMED_COLOURS
from vision_training.models.backbones.backbone import build_backbone
from vision_training.models.segmentation.segmentation_decoder import SegmentationDecoder

# Layout of the figure of the levels: two rows of six square panels under their titles
LEVEL_PANEL_SIDE = 270
LEVEL_PANEL_GAP = 20
LEVEL_TITLE_HEIGHT = 58
LEVEL_ROW_GAP = 20

# Colour map of the activations: the card surface for none, the aqua of the palette, then the primary text grey
ACTIVATION_COLOURS = [SURFACE_COLOUR, "#%02x%02x%02x" % NAMED_COLOURS["aqua"], TEXT_PRIMARY_COLOUR]


def format_shape(shape: List[int]) -> str:
    """
    Writes the shape of a tensor without its batch dimension, as the labels of the U-Net diagram show it.

    Args:
        shape (List[int]): The [B, C, H, W] shape.

    Returns:
        str: The "[C, H, W]" text.
    """
    return "[" + ", ".join(str(size) for size in shape[1:]) + "]"


def count_shortcuts_with_residuals(configuration: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Counts the residual shortcuts every block of the encoder and every level of the decoder adds with residual_nets
    turned on in both (count_residual_additions of the convolution blocks, built from the configuration of the run
    without weights), the caveat of the segmentation: a block changing its channel count with two layers adds none,
    and a decoder level reads the skip and the upsampled channels, never its own width.

    Args:
        configuration (Dict[str, Any]): The configuration of the run.

    Returns:
        Dict[str, List[int]]: The shortcuts of every backbone block and of every decoder level, from the deepest.
    """
    backbone_configuration = copy.deepcopy(configuration["Backbone"])
    backbone_configuration["modules"]["residual_nets"] = True
    backbone = build_backbone(backbone_configuration=backbone_configuration,
                              input_image_size=configuration["Data"]["image_settings"]["size"],
                              device=torch.device("cpu"), dtype=torch.float32)
    decoder_configuration = copy.deepcopy(configuration["SegmentationDecoder"])
    decoder_configuration["modules"]["residual_nets"] = True
    decoder = SegmentationDecoder(decoder_configuration=decoder_configuration,
                                  block_channels=[block.output_channels for block in backbone.convolution_blocks],
                                  last_max_pooling=backbone.convolution_blocks[-1].add_pooling,
                                  device=torch.device("cpu"), dtype=torch.float32)
    return {"backbone": [block.count_residual_additions() for block in backbone.convolution_blocks],
            "decoder": [block.count_residual_additions() for block in decoder.convolution_blocks.values()]}


def describe_model(inference_facts: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turns the shapes and parameters recorded by the inference into the facts quoted by the U-Net diagram and the
    model scenes.

    Args:
        inference_facts (Dict[str, Any]): The facts of run_segmentation_inference.
        configuration (Dict[str, Any]): The configuration of the run.

    Returns:
        Dict[str, Any]: The flat facts of the model.
    """
    shapes = inference_facts["inference_shapes"]
    facts = {f"model_{name}_shape": format_shape(shape=shape) for name, shape in shapes.items()}
    parameters = inference_facts["inference_parameters"]
    facts.update({"model_parameters_backbone": parameters["Backbone"],
                  "model_parameters_decoder": parameters["Segmentation Decoder"],
                  "model_parameters_head": parameters["Segmentation Head"],
                  "model_parameters_total": parameters["Total Model"],
                  "model_parameters_megabytes": round_fact(value=parameters["Total Model"] * 4 / 1024 ** 2),
                  "model_receptive_field": inference_facts["inference_receptive_field"],
                  "model_head": inference_facts["inference_head"],
                  "model_upsampling_level_5": inference_facts["inference_upsampling_layers"]["5"],
                  "model_skip_count": sum(1 for name in shapes if name.startswith("skip_")) - 1,
                  "model_level_count": sum(1 for name in shapes if name.startswith("level_"))})

    # Shortcuts of the run, and those residual_nets would add, block by block
    shortcuts = count_shortcuts_with_residuals(configuration=configuration)
    facts["model_backbone_shortcuts"] = str(shortcuts["backbone"])
    facts["model_decoder_shortcuts"] = str(shortcuts["decoder"])
    facts["model_backbone_blocks_with_shortcuts"] = ", ".join(str(index + 1) for index, count in
                                                              enumerate(shortcuts["backbone"]) if count > 0)
    facts["model_decoder_total_shortcuts"] = sum(shortcuts["decoder"])
    return facts


def draw_level_panel(figure: Any, left: float, top: float, image: numpy.ndarray, title: str, subtitle: str) -> None:
    """
    Draws one panel of the figure of the levels, an activation map or the class map, under its title and subtitle.

    Args:
        figure (Any): The matplotlib figure.
        left (float): The left edge of the panel in pixels.
        top (float): The top edge of its title in pixels.
        image (numpy.ndarray): The [H, W] activation map within [0, 1], or the [H, W, 3] class map.
        title (str): The title.
        subtitle (str): The line under the title, the shape of the map.
    """
    figure.text(x=left / CHART_WIDTH, y=1.0 - (top + 12) / CHART_HEIGHT, s=title, color=TEXT_PRIMARY_COLOUR,
                fontsize=20 * POINTS_PER_PIXEL, ha="left", va="center")
    figure.text(x=left / CHART_WIDTH, y=1.0 - (top + 38) / CHART_HEIGHT, s=subtitle, color=TEXT_SECONDARY_COLOUR,
                fontsize=17 * POINTS_PER_PIXEL, ha="left", va="center")
    axes = figure.add_axes(rect=(left / CHART_WIDTH,
                                 1.0 - (top + LEVEL_TITLE_HEIGHT + LEVEL_PANEL_SIDE) / CHART_HEIGHT,
                                 LEVEL_PANEL_SIDE / CHART_WIDTH, LEVEL_PANEL_SIDE / CHART_HEIGHT))
    if image.ndim == 2:
        colour_map = matplotlib.colors.LinearSegmentedColormap.from_list(name="tutorial_activation",
                                                                         colors=ACTIVATION_COLOURS)
        axes.imshow(X=image, cmap=colour_map, vmin=0.0, vmax=1.0, interpolation="nearest")
    else:
        axes.imshow(X=image, interpolation="nearest")
    axes.set_xticks(ticks=[])
    axes.set_yticks(ticks=[])
    for spine in axes.spines.values():
        spine.set_color("#%02x%02x%02x" % NAMED_COLOURS["border"])


def draw_level_figure(output_folder: Path, inference_facts: Dict[str, Any], number_classes: int) -> Dict[str, Any]:
    """
    Draws what every level of the trained segmenter computes on the test image: the mean absolute activation of the
    skip connections of the encoder and of its bottleneck on the first row, of the decoder levels from the deepest and
    the predicted class map on the second, every map scaled to its own 99th percentile.

    Args:
        output_folder (Path): The figures folder of the series.
        inference_facts (Dict[str, Any]): The facts of run_segmentation_inference.
        number_classes (int): The number of classes, the background included.

    Returns:
        Dict[str, Any]: The boxes of the two rows of the figure.
    """
    register_fonts()
    arrays = numpy.load(file=output_folder / INFERENCE_ARRAYS_NAME)
    shapes = inference_facts["inference_shapes"]
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    rows = [[(f"skip_{block}", f"skip of block {block}") for block in range(1, 6)] + [("bottleneck", "bottleneck")],
            [(f"level_{block}", f"decoder level {block}") for block in range(5, 0, -1)] + [("class_map", "class map")]]
    facts = {}
    for row_index, row in enumerate(rows):
        top = row_index * (LEVEL_TITLE_HEIGHT + LEVEL_PANEL_SIDE + LEVEL_ROW_GAP)
        for column_index, (name, title) in enumerate(row):
            left = column_index * (LEVEL_PANEL_SIDE + LEVEL_PANEL_GAP)
            if name == "class_map":
                palette = build_class_palette(number_classes=number_classes)
                image = palette[arrays["class_map"]]
                subtitle = f"argmax of [{number_classes}, {image.shape[0]}, {image.shape[1]}]"
            else:
                activation = arrays[f"map_{name}"].astype(numpy.float32)
                image = numpy.clip(a=activation / max(float(numpy.percentile(a=activation, q=99)), 1e-6), a_min=0.0,
                                   a_max=1.0)
                subtitle = format_shape(shape=shapes[name])
            draw_level_panel(figure=figure, left=left, top=top, image=image, title=title, subtitle=subtitle)
        facts[f"levels_row_{row_index + 1}_box"] = [0, top, CHART_WIDTH, top + LEVEL_TITLE_HEIGHT + LEVEL_PANEL_SIDE]
    save_figure(figure=figure, output_path=output_folder / "model" / "levels.png")
    pyplot.close(fig=figure)
    facts["levels_pixel_accuracy"] = inference_facts["inference_pixel_accuracy"]
    return facts
