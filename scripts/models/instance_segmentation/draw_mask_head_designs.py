"""
When run, draws the mask head designs compared by stages 3 and 3b of the instance segmentation benchmark
(documentation/architecture_benchmarking/instance_segmentation_experiment_plan.md) into one figure of the plan, so that
the convolutions before the mask and the paths of the finer inputs are seen before the pairs are trained: the mask
heads of stages 0 to 2 (three convolutions of 32 channels), those of stage 3 (four of 64), and the three arms of stage
3b on the heads of stage 3, design A (an image branch on the box crop of the input image), design B (U-Net skips from
the backbone blocks before their pooling) and the control (one fusion convolution alone). Every design is built as the
training builds it (the instance segmenter of vision_training/models/model_registry.py, from the configuration file of
the benchmark with the settings of the design), the figure reading its layers, channels, strides and residual
shortcuts from the built mask heads, and its multiply-accumulates per box are counted on those layers by forward hooks
on one box, so that the figure follows the code.
Usage: uv run python -m scripts.models.instance_segmentation.draw_mask_head_designs
--configuration_path=vision_training/configurations/instance_segmentation/run-1.yaml
--output_path=documentation/architecture_benchmarking/figures/instance_segmentation/mask_head_designs.png
"""

import argparse
import copy
import torch

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path
from torch import nn
from typing import Any, Dict, List, Tuple

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.utilities.os_utilities import format_path_link, print_color

# Designs drawn, in order: their title, the [layers, channels] of the mask head of every level, and the settings of
# the finer inputs in the 'MaskHead' section (the settings left out stay off)
DESIGNS = [{"title": "Reference: the mask heads of stages 0 to 2, convolutions [3, 32]", "convolutions": [3, 32],
            "fine_inputs": {}},
           {"title": "Stage 3: convolutions [4, 64], the reference of stage 3b", "convolutions": [4, 64],
            "fine_inputs": {}},
           {"title": "Stage 3b, design A: image branch concatenated at the mask resolution", "convolutions": [4, 64],
            "fine_inputs": {"image_branch_convolutions": [5, 32], "fusion_convolutions": [1, 64]}},
           {"title": "Stage 3b, design B: U-Net skips from the backbone blocks before their pooling",
            "convolutions": [4, 64],
            "fine_inputs": {"skip_blocks_at_pool_size": [4], "skip_blocks_at_mask_resolution": [3],
                            "fusion_convolutions": [1, 64]}},
           {"title": "Stage 3b, control: one fusion convolution at the mask resolution, without finer inputs",
            "convolutions": [4, 64], "fine_inputs": {"fusion_convolutions": [1, 64]}}]

# Finer inputs of the 'MaskHead' section turned off, given to every design before its own settings, so that a design
# leaving a setting out draws it off whatever the configuration file holds (the launch file of a later stage keeping
# the image branch of design A, for instance)
FINE_INPUTS_OFF = {"skip_blocks_at_pool_size": [], "image_branch_convolutions": [0, 0],
                   "skip_blocks_at_mask_resolution": [], "fusion_convolutions": [0, 0]}

# Colours of the paths: the level map, the image branch, the backbone skips, the fusion and the output
COLOURS = {"feature": "#2a6fdb", "image": "#e8590c", "skip": "#2f9e44", "fusion": "#7048e8", "output": "#495057"}

# Size of a block, horizontal step between two blocks, and vertical step between two rows of a design
BOX_WIDTH, BOX_HEIGHT, STEP, ROW_STEP = 1.42, 0.62, 1.72, 1.1

# Left centre of the first block of a row, and height of the cost line under a design
FIRST_CENTRE, COST_LINE_HEIGHT = 0.9, -0.6

Point = Tuple[float, float]


def build_design_model(model_configuration: Dict[str, Any], design: Dict[str, Any]) -> nn.Module:
    """
    Builds the instance segmenter of a design on the CPU, as the training builds it, so that the figure and the counts
    of operations read the layers the code gives the design and not a copy of their rules.

    Args:
        model_configuration (Dict[str, Any]): The model configuration of the benchmark (load_experiment_configuration).
        design (Dict[str, Any]): The design (DESIGNS).

    Returns:
        nn.Module: The instance segmenter of the design, in evaluation mode.
    """
    # The mask head of every level gets the convolutions of the design, the 'MaskHead' section its finer inputs, those
    # the design leaves out being off
    configuration = copy.deepcopy(model_configuration)
    for level_configuration in configuration["DetectionLevels"].values():
        level_configuration["mask_head"]["convolutions"] = list(design["convolutions"])
    configuration["MaskHead"].update(copy.deepcopy({**FINE_INPUTS_OFF, **design["fine_inputs"]}))
    torch.manual_seed(seed=0)
    model = TASK_MODEL_CLASSES["instance_segmentation"](configuration=configuration, device=torch.device("cpu"),
                                                        dtype=torch.float32)
    return model.eval()


def record_multiply_accumulates(module: nn.Module, arguments: Tuple[Any, ...], keyword_arguments: Dict[str, Any],
                                output: torch.Tensor) -> None:
    """
    Forward hook storing on a convolution its multiply-accumulates for every box of the batch it received, counted from
    its kernel and the size of its input or output, the bias additions left out as in the usual counts.

    Args:
        module (nn.Module): The nn.Conv2d or nn.ConvTranspose2d layer.
        arguments (Tuple[Any, ...]): Its positional arguments, the [R, C_in, H, W] input when given by position.
        keyword_arguments (Dict[str, Any]): Its keyword arguments, the input under 'input' when given by name.
        output (torch.Tensor): Its [R, C_out, H', W'] output.

    Raises:
        TypeError: If the hook is attached to another layer than a 2D convolution or transposed convolution.
    """
    kernel_operations = module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels // \
        module.groups
    # A convolution computes every output position, a transposed convolution spreads every input position
    if isinstance(module, nn.ConvTranspose2d):
        input_tensor = arguments[0] if arguments else keyword_arguments["input"]
        module.recorded_multiply_accumulates = input_tensor.shape[2] * input_tensor.shape[3] * kernel_operations
    elif isinstance(module, nn.Conv2d):
        module.recorded_multiply_accumulates = output.shape[2] * output.shape[3] * kernel_operations
    else:
        raise TypeError(f"Expected a 2D convolution or transposed convolution, got {type(module).__name__}.")


def count_operations_per_box(model: nn.Module) -> int:
    """
    Counts the multiply-accumulates the mask branch spends on one box: the mask head of the first detection level and
    the finer inputs shared by the levels, run on one box of zero feature maps of the sizes the model was built for.
    Every level has the same layers, its map size changing only the RoIAlign, so one level gives the cost of any box.

    Args:
        model (nn.Module): The instance segmenter (build_design_model).

    Returns:
        int: The multiply-accumulates of the convolutions of the mask branch for one box.
    """
    level_key = next(iter(model.mask_heads_dictionary))
    mask_head = model.mask_heads_dictionary[level_key]
    convolutions = [module for module in [*model.mask_fine_inputs.modules(), *mask_head.modules()]
                    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))]
    hooks = [module.register_forward_hook(hook=record_multiply_accumulates, with_kwargs=True)
             for module in convolutions]

    # One box of a quarter of the image on zero maps: the level map of the head and every backbone skip output
    level_information = model.detection_input_information[level_key]
    level_map = torch.zeros(size=(1, level_information["number_channels"], level_information["feature_map_size"],
                                  level_information["feature_map_size"]))
    skip_feature_maps = {key: torch.zeros(size=(1, information["number_channels"], information["feature_map_size"],
                                                information["feature_map_size"]))
                         for key, information in model.skip_output_information.items()}
    box = torch.tensor(data=[[0.0, 0.0, 0.0, model.input_image_size / 4, model.input_image_size / 4]])
    with torch.no_grad():
        fine_features = model.mask_fine_inputs(proposal_boxes=box, skip_feature_maps=skip_feature_maps)
        mask_head(proposal_boxes=box, input_tensor=level_map, skip_feature_maps=skip_feature_maps,
                  fine_features=fine_features)

    # Every convolution must have run once, or a path of the design was not counted
    for hook in hooks:
        hook.remove()
    missing_layers = [module for module in convolutions if not hasattr(module, "recorded_multiply_accumulates")]
    if missing_layers:
        raise RuntimeError(f"{len(missing_layers)} convolutions of the mask branch did not run on the box.")
    return sum(module.recorded_multiply_accumulates for module in convolutions)


def describe_skip(model: nn.Module, skip_key: str) -> Dict[str, int]:
    """
    Gives the block, stride and channels of a backbone skip output read by the mask heads, for its block on the figure.

    Args:
        model (nn.Module): The instance segmenter.
        skip_key (str): The key of the skip output, 'skip_<block>'.

    Returns:
        Dict[str, int]: The 'block', 'stride' and 'channels' of the skip output.
    """
    information = model.skip_output_information[skip_key]
    return {"block": int(skip_key.split("_")[-1]), "stride": model.input_image_size // information["feature_map_size"],
            "channels": information["number_channels"]}


def describe_design(model: nn.Module) -> Dict[str, Any]:
    """
    Reads from the built model everything the figure draws for a design: the level maps, the RoIAlign of the mask
    heads with its skips, the convolution block and its residual shortcuts, the doublings, the image branch and the
    skips at the mask resolution, the fusion convolutions and the predictor.

    Args:
        model (nn.Module): The instance segmenter of the design (build_design_model).

    Returns:
        Dict[str, Any]: The description of the design, with its multiply-accumulates and parameters.
    """
    # Mask head of the first level, every level having the same layers, and the strides of all the level maps
    level_key = next(iter(model.mask_heads_dictionary))
    mask_head = model.mask_heads_dictionary[level_key]
    convolution_block = mask_head.convolutional_block
    level_channels = model.detection_input_information[level_key]["number_channels"]
    level_strides = sorted({model.input_image_size // information["feature_map_size"]
                            for information in model.detection_input_information.values()})
    fusion_convolutions = model.mask_head_configuration["fusion_convolutions"]

    # Parameters of the mask heads of every level and of the fine inputs they share
    parameters = sum(parameter.numel() for parameter in [*model.mask_heads_dictionary.parameters(),
                                                         *model.mask_fine_inputs.parameters()])
    return {"levels": list(model.mask_heads_dictionary), "level_channels": level_channels,
            "level_strides": level_strides, "pool_size": mask_head.roi_align_pool_size,
            "mask_size": mask_head.mask_size,
            "skips_at_pool_size": [describe_skip(model=model, skip_key=key) for key in mask_head.skip_keys],
            "block_input_channels": convolution_block.input_channels,
            "block_channels": convolution_block.output_channels, "block_layers": convolution_block.number_layers,
            "residual": convolution_block.residual, "residual_parity": convolution_block.residual_parity,
            "number_upsamplings": mask_head.number_upsamplings, "upsampling": mask_head.upsampling,
            "image_branch": model.mask_fine_inputs.image_branch_convolutions,
            "skips_at_mask_size": [describe_skip(model=model, skip_key=key)
                                   for key in model.mask_fine_inputs.skip_keys],
            "fine_channels": model.mask_fine_inputs.number_channels, "fusion": fusion_convolutions,
            "number_classes": mask_head.number_classes,
            "operations": count_operations_per_box(model=model), "parameters": parameters}


def draw_block(axis: Axes, centre: Point, title: str, shape: str, colour: str) -> None:
    """
    Draws one layer or tensor of a design as a rounded box with its name and the tensor it outputs for one box.

    Args:
        axis (Axes): The axis of the design.
        centre (Point): The centre of the box.
        title (str): The name of the layer or tensor.
        shape (str): The tensor it outputs for one box.
        colour (str): The colour of its path.
    """
    axis.add_patch(FancyBboxPatch(xy=(centre[0] - BOX_WIDTH / 2, centre[1] - BOX_HEIGHT / 2), width=BOX_WIDTH,
                                  height=BOX_HEIGHT, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.6,
                                  edgecolor=colour, facecolor=colour + "1f"))
    axis.text(x=centre[0], y=centre[1] + 0.11, s=title, ha="center", va="center", fontsize=8.4, color="#212529",
              weight="bold")
    axis.text(x=centre[0], y=centre[1] - 0.15, s=shape, ha="center", va="center", fontsize=7.8, color="#495057")


def draw_arrow(axis: Axes, start: Point, end: Point, colour: str) -> None:
    """
    Draws the arrow from the right edge of a block to the left edge of the next one.

    Args:
        axis (Axes): The axis of the design.
        start (Point): The centre of the source block.
        end (Point): The centre of the target block.
        colour (str): The colour of the path.
    """
    axis.add_patch(FancyArrowPatch(posA=(start[0] + BOX_WIDTH / 2, start[1]), posB=(end[0] - BOX_WIDTH / 2, end[1]),
                                   arrowstyle="-|>", mutation_scale=11, linewidth=1.4, color=colour))


def draw_shortcut(axis: Axes, start: Point, end: Point, colour: str) -> None:
    """
    Draws a residual shortcut over a pair of convolutions, from the block whose output enters the pair to the layer
    closing it, where the shortcut is added.

    Args:
        axis (Axes): The axis of the design.
        start (Point): The centre of the block whose output enters the pair.
        end (Point): The centre of the layer closing the pair.
        colour (str): The colour of the path.
    """
    axis.add_patch(FancyArrowPatch(posA=(start[0] + 0.25, start[1] + BOX_HEIGHT / 2),
                                   posB=(end[0] - 0.05, end[1] + BOX_HEIGHT / 2), arrowstyle="-|>", mutation_scale=9,
                                   linewidth=1.1, color=colour, connectionstyle="arc3,rad=-0.35"))
    axis.text(x=end[0] - 0.05, y=end[1] + BOX_HEIGHT / 2 + 0.12, s="+", ha="center", va="center", fontsize=11,
              color=colour, weight="bold")


def draw_feature_path(axis: Axes, description: Dict[str, Any], height: float) -> Point:
    """
    Draws the path of the level map through a mask head: the map, its RoIAlign at the pool size with the skips
    concatenated to it, every convolution of the block with the residual shortcuts placed as ConvolutionBlock places
    them, and the doublings.

    Args:
        axis (Axes): The axis of the design.
        description (Dict[str, Any]): The description of the design (describe_design).
        height (float): The height of the row.

    Returns:
        Point: The centre of the last doubling, whose features reach the mask resolution.
    """
    colour, pool_size = COLOURS["feature"], description["pool_size"]
    strides = description["level_strides"]
    centres = [(FIRST_CENTRE, height), (FIRST_CENTRE + STEP, height)]
    draw_block(axis=axis, centre=centres[0], title=f"level map {'-'.join(description['levels'])}",
               shape=f"stride {strides[0]}-{strides[-1]}, {description['level_channels']} ch", colour=colour)
    skip_channels = "".join(f"+{skip['channels']}" for skip in description["skips_at_pool_size"])
    draw_block(axis=axis, centre=centres[1], title=f"RoIAlign {pool_size}",
               shape=f"{pool_size}x{pool_size}x" + (f"({description['level_channels']}{skip_channels})"
                                                    if skip_channels else f"{description['level_channels']}"),
               colour=colour)
    draw_arrow(axis=axis, start=centres[0], end=centres[1], colour=colour)

    # Convolutions of the block, a shortcut closing every pair as in ConvolutionBlock.forward
    for layer_index in range(description["block_layers"]):
        centre = (centres[-1][0] + STEP, height)
        layer_input = description["block_input_channels"] if layer_index == 0 else description["block_channels"]
        draw_block(axis=axis, centre=centre, title=f"conv {layer_index + 1} 3x3",
                   shape=f"{layer_input}->{description['block_channels']}, {pool_size}x{pool_size}", colour=colour)
        draw_arrow(axis=axis, start=centres[-1], end=centre, colour=colour)
        if description["residual"] and layer_index != 0 and \
                layer_index % 2 == 1 - description["residual_parity"]:
            draw_shortcut(axis=axis, start=centres[-2], end=centre, colour=colour)
        centres.append(centre)

    # Doublings of the pooled features up to the mask resolution
    for upsampling_index in range(description["number_upsamplings"]):
        side = pool_size * 2 ** (upsampling_index + 1)
        centre = (centres[-1][0] + STEP, height)
        title = "transposed 2x2" if description["upsampling"] == "transposed" else "bilinear + conv"
        draw_block(axis=axis, centre=centre, title=title, shape=f"{side}x{side}x{description['block_channels']}",
                   colour=colour)
        draw_arrow(axis=axis, start=centres[-1], end=centre, colour=colour)
        centres.append(centre)
    return centres[-1]


def draw_skips_at_pool_size(axis: Axes, description: Dict[str, Any], height: float, feature_height: float) -> None:
    """
    Draws the backbone blocks pooled at the pool size of the mask heads, under the level map, each joining the RoIAlign
    of the level map, whose features they are concatenated to.

    Args:
        axis (Axes): The axis of the design.
        description (Dict[str, Any]): The description of the design (describe_design).
        height (float): The height of their row.
        feature_height (float): The height of the row of the level map.
    """
    colour = COLOURS["skip"]
    roi_align_centre = (FIRST_CENTRE + STEP, feature_height)
    for skip_index, skip in enumerate(description["skips_at_pool_size"]):
        centre = (FIRST_CENTRE + skip_index * STEP, height)
        draw_block(axis=axis, centre=centre, title=f"backbone block {skip['block']}",
                   shape=f"stride {skip['stride']}, {skip['channels']} ch", colour=colour)
        axis.add_patch(FancyArrowPatch(posA=(centre[0] + 0.3, centre[1] + BOX_HEIGHT / 2),
                                       posB=(roi_align_centre[0] - 0.3, roi_align_centre[1] - BOX_HEIGHT / 2),
                                       arrowstyle="-|>", mutation_scale=11, linewidth=1.4, color=colour))
        axis.text(x=centre[0] + 0.55, y=centre[1] + 0.62, s=f"RoIAlign {description['pool_size']}", fontsize=7.6,
                  color=colour, ha="left")


def draw_fine_inputs(axis: Axes, description: Dict[str, Any], rows: Dict[str, float],
                     concatenation_left: float) -> List[Tuple[Point, str]]:
    """
    Draws the fine inputs shared by the levels at the mask resolution: the image branch (the box crop of the
    normalized input image and its plain convolutions) and the backbone blocks pooled at the mask resolution.

    Args:
        axis (Axes): The axis of the design.
        description (Dict[str, Any]): The description of the design (describe_design).
        rows (Dict[str, float]): The height of the row of the image branch ('image') and of the skips ('skip').
        concatenation_left (float): The centre of the block right before the concatenation, under which the skips
            end.

    Returns:
        List[Tuple[Point, str]]: The last block of every fine input and its colour, concatenated in this order.
    """
    mask_size, sources = description["mask_size"], []

    # Image branch: the crop of the normalized image at the mask resolution, then its plain convolutions
    image_layers, image_channels = description["image_branch"]
    if image_layers > 0:
        colour, height = COLOURS["image"], rows["image"]
        centres = [(FIRST_CENTRE + index * STEP, height) for index in range(3)]
        draw_block(axis=axis, centre=centres[0], title="normalized image", shape="stride 1, 3 ch", colour=colour)
        draw_block(axis=axis, centre=centres[1], title=f"RoIAlign {mask_size}", shape=f"{mask_size}x{mask_size}x3",
                   colour=colour)
        draw_block(axis=axis, centre=centres[2], title=f"{image_layers} plain conv 3x3",
                   shape=f"{mask_size}x{mask_size}x{image_channels}", colour=colour)
        draw_arrow(axis=axis, start=centres[0], end=centres[1], colour=colour)
        draw_arrow(axis=axis, start=centres[1], end=centres[2], colour=colour)
        sources.append((centres[2], colour))

    # Backbone blocks pooled at the mask resolution, ending under the last doubling
    for skip_index, skip in enumerate(description["skips_at_mask_size"]):
        colour, height = COLOURS["skip"], rows["skip"] - skip_index * ROW_STEP
        block_centre = (concatenation_left - STEP, height)
        pooled_centre = (concatenation_left, height)
        draw_block(axis=axis, centre=block_centre, title=f"backbone block {skip['block']}",
                   shape=f"stride {skip['stride']}, {skip['channels']} ch", colour=colour)
        draw_block(axis=axis, centre=pooled_centre, title=f"RoIAlign {mask_size}",
                   shape=f"{mask_size}x{mask_size}x{skip['channels']}", colour=colour)
        draw_arrow(axis=axis, start=block_centre, end=pooled_centre, colour=colour)
        sources.append((pooled_centre, colour))
    return sources


def draw_end(axis: Axes, description: Dict[str, Any], start: Point, input_channels: int) -> None:
    """
    Draws the end of a design: its fusion convolutions when it has some, the 1 x 1 predictor and the mask of the class
    of the box.

    Args:
        axis (Axes): The axis of the design.
        description (Dict[str, Any]): The description of the design (describe_design).
        start (Point): The centre of the block feeding the end.
        input_channels (int): The channels entering the end.
    """
    mask_size, (fusion_layers, fusion_channels) = description["mask_size"], description["fusion"]
    centre = start
    if fusion_layers > 0:
        fusion_centre = (centre[0] + STEP, centre[1])
        draw_block(axis=axis, centre=fusion_centre, title=f"{fusion_layers} fusion conv 3x3",
                   shape=f"{input_channels}->{fusion_channels}, {mask_size}x{mask_size}", colour=COLOURS["fusion"])
        draw_arrow(axis=axis, start=centre, end=fusion_centre, colour=COLOURS["fusion"])
        centre = fusion_centre
    predictor_centre = (centre[0] + STEP, centre[1])
    mask_centre = (centre[0] + 2 * STEP, centre[1])
    draw_block(axis=axis, centre=predictor_centre, title="1x1 predictor",
               shape=f"{mask_size}x{mask_size}x{description['number_classes']}", colour=COLOURS["output"])
    draw_block(axis=axis, centre=mask_centre, title="mask of its class", shape=f"{mask_size}x{mask_size} logits",
               colour=COLOURS["output"])
    draw_arrow(axis=axis, start=centre, end=predictor_centre, colour=COLOURS["output"])
    draw_arrow(axis=axis, start=predictor_centre, end=mask_centre, colour=COLOURS["output"])


def count_design_rows(description: Dict[str, Any]) -> int:
    """
    Gives the number of rows of a design: the level map, then a row for the skips (at the pool size on the left, at
    the mask resolution on the right), one per further skip at the mask resolution, and one for the image branch.

    Args:
        description (Dict[str, Any]): The description of the design (describe_design).

    Returns:
        int: The number of rows of the design.
    """
    skip_rows = max(1 if description["skips_at_pool_size"] else 0, len(description["skips_at_mask_size"]))
    return 1 + skip_rows + (1 if description["image_branch"][0] > 0 else 0)


def draw_design(axis: Axes, design: Dict[str, Any], description: Dict[str, Any],
                reference_operations: int) -> None:
    """
    Draws one design: the path of the level map at the top, the skips and the image branch in the rows below, the
    concatenation of the fine features at the mask resolution, the end, and the line of its cost under it.

    Args:
        axis (Axes): The axis of the design.
        design (Dict[str, Any]): The design (DESIGNS).
        description (Dict[str, Any]): Its description (describe_design).
        reference_operations (int): The multiply-accumulates per box of the reference design, for the ratio.
    """
    # Rows from the top: the level map, the skips, then the image branch at the bottom
    number_rows = count_design_rows(description=description)
    feature_height = 0.45 + (number_rows - 1) * ROW_STEP
    rows = {"skip": feature_height - ROW_STEP, "image": 0.45}
    upsampled = draw_feature_path(axis=axis, description=description, height=feature_height)
    if description["skips_at_pool_size"]:
        draw_skips_at_pool_size(axis=axis, description=description, height=rows["skip"],
                                feature_height=feature_height)

    # Fine features concatenated after the doublings, the end following at the height of the concatenation
    sources = draw_fine_inputs(axis=axis, description=description, rows=rows, concatenation_left=upsampled[0])
    end_start, end_channels = upsampled, description["block_channels"]
    if sources:
        concatenation = (upsampled[0] + STEP, (feature_height + min(source[1] for source, _ in sources)) / 2)
        fine_channels = "".join(f"+{channels}" for channels in
                                ([description["image_branch"][1]] if description["image_branch"][0] > 0 else []) +
                                [skip["channels"] for skip in description["skips_at_mask_size"]])
        mask_size = description["mask_size"]
        draw_block(axis=axis, centre=concatenation, title="concatenate",
                   shape=f"{mask_size}x{mask_size}x({description['block_channels']}{fine_channels})",
                   colour=COLOURS["fusion"])
        for source, colour in [(upsampled, COLOURS["feature"])] + sources:
            draw_arrow(axis=axis, start=source, end=concatenation, colour=colour)
        end_start, end_channels = concatenation, description["block_channels"] + description["fine_channels"]
    draw_end(axis=axis, description=description, start=end_start, input_channels=end_channels)

    # Cost line: the convolutions before the mask, the multiply-accumulates per box and the parameters
    image_layers = description["image_branch"][0]
    convolutions_before_mask = description["block_layers"] + description["fusion"][0]
    cost_line = (f"{convolutions_before_mask} convolutions 3x3 on the features before the mask"
                 + (f", {image_layers} on the image" if image_layers > 0 else "")
                 + f". {description['operations'] / 1e6:.1f} M multiply-accumulates per box "
                 f"({description['operations'] / reference_operations:.1f}x the reference), "
                 f"{description['parameters']:,} parameters in the mask heads of the {len(description['levels'])} "
                 f"levels" + (" and the fine inputs shared by them." if description["fine_channels"] else "."))
    axis.set_xlim(left=0, right=21.4)
    axis.set_ylim(bottom=COST_LINE_HEIGHT - 0.15, top=feature_height + 0.85)
    axis.axis("off")
    axis.set_title(label=design["title"], loc="left", fontsize=12, weight="bold", color="#212529")
    axis.text(x=0.15, y=COST_LINE_HEIGHT, s=cost_line, fontsize=8.6, color="#495057", ha="left", va="center")


def draw_mask_head_designs(model_configuration: Dict[str, Any], output_path: Path) -> List[Dict[str, Any]]:
    """
    Builds every design of DESIGNS from the configuration of the benchmark, draws them one under the other with the
    legend of the paths, and writes the figure.

    Args:
        model_configuration (Dict[str, Any]): The model configuration of the benchmark (load_experiment_configuration).
        output_path (Path): The PNG file written.

    Returns:
        List[Dict[str, Any]]: The description of every design (describe_design), in the order of DESIGNS.
    """
    descriptions = [describe_design(model=build_design_model(model_configuration=model_configuration, design=design))
                    for design in DESIGNS]
    height_ratios = [count_design_rows(description=description) + 0.6 for description in descriptions]
    figure = Figure(figsize=(19, 2.4 * sum(height_ratios) / 1.6 + 0.6), facecolor="white")
    FigureCanvasAgg(figure=figure)
    axes = figure.subplots(nrows=len(DESIGNS), ncols=1, gridspec_kw={"height_ratios": height_ratios})
    for axis, design, description in zip(axes, DESIGNS, descriptions):
        draw_design(axis=axis, design=design, description=description,
                    reference_operations=descriptions[0]["operations"])

    # Legend of the paths above the designs
    legend_items = [("level map path", COLOURS["feature"]), ("image path", COLOURS["image"]),
                    ("backbone skip path", COLOURS["skip"]), ("concatenation and fusion", COLOURS["fusion"]),
                    ("output", COLOURS["output"]), ("+ residual shortcut", COLOURS["feature"])]
    for item_index, (label, colour) in enumerate(legend_items):
        figure.text(x=0.04 + 0.16 * item_index, y=0.993, s=("" if label.startswith("+") else "■ ") + label,
                    color=colour, fontsize=10, va="top")
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=90, facecolor="white")
    return descriptions


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Draws the mask head designs of stages 3 and 3b of the "
                                                          "instance segmentation benchmark.")
    argument_parser.add_argument("--configuration_path", type=Path, required=True,
                                 help="Configuration file of the benchmark, run-1.yaml of its launches.")
    argument_parser.add_argument("--output_path", type=Path, required=True, help="PNG file of the figure.")
    arguments = argument_parser.parse_args()
    _, loaded_model_configuration = load_experiment_configuration(configuration_path=arguments.configuration_path)
    design_descriptions = draw_mask_head_designs(model_configuration=loaded_model_configuration,
                                                 output_path=arguments.output_path)

    # Cost of every design in the console, for the tables of the plan
    for drawn_design, design_description in zip(DESIGNS, design_descriptions):
        print_color(color="blue", output=f"{drawn_design['title']} : {design_description['operations']:,} "
                                         f"Multiply-Accumulates Per Box, {design_description['parameters']:,} "
                                         f"Parameters")
    print_color(color="green", output=f"Figure Written : {format_path_link(path=arguments.output_path)}")
