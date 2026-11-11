"""
Backbone figures of the foundations tutorials (the backbone and its modules chapter), made by
tutorials/figures/foundations/figures.py from the last checkpoint of the classification run, read on the CPU, so that
the scenes show what the trained backbone really computes: the input of a test star and the feature maps it gives at
the output of three blocks, the most active channels of each on a grid of tiles. The facts are the shapes and values of
those maps, the place of the strongest final feature (the centre of the receptive field squares) and the
squeeze-and-excitation gates of one layer for the star. The checkpoint inference is reused from the previous build
unless forced.
"""

import functools

import cv2
import numpy
import torch

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (STAR_IMAGE_NAME, find_star_image, load_classification_model,
                                                read_network_input, read_reused_facts, read_run_configuration,
                                                round_fact, write_network_input, write_reused_facts)
from tutorial_builder.layout import CARD_COLOUR, NAMED_COLOURS

# Blocks whose output feature maps are drawn, with the most active channels of each laid out on a grid of tiles
SHOWN_BLOCK_NUMBERS = [1, 3, 6]
SHOWN_CHANNEL_COUNT = 8
GRID_COLUMN_COUNT = 4
TILE_SIZE = 256
TILE_GAP = 8

# Squeeze-and-excitation layer whose gates are given: the second layer of the last block, the layer whose gates spread
# the most for the test star in this run (the gates of the earlier layers stay near 0.5 after 1500 iterations)
GATE_BLOCK_NUMBER = 6
GATE_LAYER_INDEX = 1

# Sequential ramp of the feature maps, from the card colour for no activation through blue to near white for the
# strongest activation of a channel, validated against the card
FEATURE_RAMP = numpy.array([CARD_COLOUR, NAMED_COLOURS["blue"], (232, 240, 255)], dtype=numpy.float64)

# Name of the JSON file keeping the facts of the figures for their reuse
FACTS_FILE_NAME = "backbone_facts.json"


def keep_first_output(kept_outputs: List[torch.Tensor], module: torch.nn.Module, inputs: Any,
                      output: torch.Tensor) -> None:
    """
    Forward hook of the second linear layer of a squeeze-and-excitation module keeping its output for the first image
    of the batch, the gates before their sigmoid.

    Args:
        kept_outputs (List[torch.Tensor]): The kept outputs, one per call of the layer.
        module (torch.nn.Module): The linear layer, given by PyTorch.
        inputs (Any): The inputs of the layer, given by PyTorch.
        output (torch.Tensor): The [B, C] output of the layer.
    """
    kept_outputs.append(output[0].detach())


def colour_feature_map(feature_map: numpy.ndarray) -> numpy.ndarray:
    """
    Colours one channel of a feature map with the sequential ramp of the videos, from the card colour for no
    activation to near white for the strongest activation of the channel, so that every tile uses its whole range.

    Args:
        feature_map (numpy.ndarray): The [H, W] activations of the channel, non-negative after the ReLU.

    Returns:
        numpy.ndarray: The [H, W, 3] RGB tile.
    """
    maximum = float(feature_map.max())
    normalized = feature_map / maximum if maximum > 0 else numpy.zeros_like(feature_map)

    # Piecewise linear ramp: the first half from the card to blue, the second half from blue to near white
    position = numpy.clip(normalized, 0.0, 1.0) * (len(FEATURE_RAMP) - 1)
    lower_index = numpy.minimum(numpy.floor(position).astype(int), len(FEATURE_RAMP) - 2)
    fraction = (position - lower_index)[..., None]
    colours = FEATURE_RAMP[lower_index] * (1.0 - fraction) + FEATURE_RAMP[lower_index + 1] * fraction
    return numpy.round(colours).astype(numpy.uint8)


def draw_feature_map_grid(block_output: numpy.ndarray, output_path: Path) -> List[int]:
    """
    Draws the most active channels of the output of one backbone block for the test star on a grid of tiles, every
    tile scaled up with the nearest pixel so that the resolution of the block stays visible.

    Args:
        block_output (numpy.ndarray): The [C, H, W] output of the block.
        output_path (Path): The PNG file of the grid.

    Returns:
        List[int]: The channels shown, from the most active one.

    Raises:
        OSError: If the figure cannot be written.
    """
    # Channels of the strongest mean activation for this image
    shown_channels = numpy.argsort(-block_output.mean(axis=(1, 2)))[:SHOWN_CHANNEL_COUNT].tolist()
    row_count = (len(shown_channels) + GRID_COLUMN_COUNT - 1) // GRID_COLUMN_COUNT
    grid = numpy.zeros(shape=(row_count * TILE_SIZE + (row_count - 1) * TILE_GAP,
                              GRID_COLUMN_COUNT * TILE_SIZE + (GRID_COLUMN_COUNT - 1) * TILE_GAP, 3), dtype=numpy.uint8)
    grid[...] = FEATURE_RAMP[0].astype(numpy.uint8)

    # Every channel coloured, scaled to the tile size and placed on its row and column
    for tile_index, channel in enumerate(shown_channels):
        tile = cv2.resize(src=colour_feature_map(feature_map=block_output[channel]), dsize=(TILE_SIZE, TILE_SIZE),
                          interpolation=cv2.INTER_NEAREST)
        top = (tile_index // GRID_COLUMN_COUNT) * (TILE_SIZE + TILE_GAP)
        left = (tile_index % GRID_COLUMN_COUNT) * (TILE_SIZE + TILE_GAP)
        grid[top:top + TILE_SIZE, left:left + TILE_SIZE] = tile
    if not cv2.imwrite(filename=str(output_path), img=cv2.cvtColor(src=grid, code=cv2.COLOR_RGB2BGR)):
        raise OSError(f"The figure {output_path} could not be written.")
    return shown_channels


def make_backbone_figures(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Runs the backbone of the last checkpoint of the classification run on the test star, writes the input and the
    feature map grids of the shown blocks into the model folder of the series, and gives the facts of the maps and of
    the squeeze-and-excitation gates; the figures of the previous build are reused unless forced or incomplete.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.
        force (bool): Whether the checkpoint is run again even when the previous build left the figures.

    Returns:
        Dict[str, Any]: The facts of the figures, named feature_maps.* (with the shown blocks and the number of channels
            drawn for each) and squeeze_excitation.*.

    Raises:
        RuntimeError: If the squeeze-and-excitation layer is not called once by the forward pass.
    """
    # Figures of the previous build
    model_folder = output_folder / "model"
    facts_path = model_folder / FACTS_FILE_NAME
    figure_paths = [model_folder / "input_star.png"] + [model_folder / f"feature_maps_block_{block_number}.png"
                                                        for block_number in SHOWN_BLOCK_NUMBERS]
    reused_facts = read_reused_facts(facts_path=facts_path, required_paths=figure_paths, force=force)
    if reused_facts is not None:
        return reused_facts

    # Trained model of the run and the test star as the network receives it
    model, checkpoint_iteration = load_classification_model(run_folder=run_folder)
    run_configuration = read_run_configuration(run_folder=run_folder)
    image_settings = run_configuration["Data"]["image_settings"]
    star_tensor = read_network_input(image_path=find_star_image(run_configuration=run_configuration),
                                     image_size=image_settings["size"], keep_ratio=image_settings["keep_ratio"])

    # Output of every block before its pooling, the gates of one squeeze-and-excitation layer kept by a hook on the
    # layer called as a module by its block
    gate_logits: List[torch.Tensor] = []
    gate_layer = model.backbone.convolution_blocks[GATE_BLOCK_NUMBER - 1].squeeze_and_excitation_layers[
        GATE_LAYER_INDEX].second_adaptive_recalibrator
    hook_handle = gate_layer.register_forward_hook(hook=functools.partial(keep_first_output, gate_logits))
    with torch.no_grad():
        _, pre_pooling_outputs = model.backbone(input_tensor=star_tensor, return_pre_pooling_outputs=True)
    hook_handle.remove()
    if len(gate_logits) != 1:
        raise RuntimeError(f"The squeeze-and-excitation layer {GATE_LAYER_INDEX + 1} of block {GATE_BLOCK_NUMBER} was "
                           f"called {len(gate_logits)} times by the forward pass of the backbone instead of once.")

    # Output of every shown block, the blocks running their layers and their pooling apart (no module hook sees it),
    # each followed here by the pooling of its block
    with torch.no_grad():
        block_outputs = {block_number: model.backbone.convolution_blocks[block_number - 1].apply_pooling(
            input_tensor=pre_pooling_outputs[str(block_number)])[0].numpy() for block_number in SHOWN_BLOCK_NUMBERS}

    # Input of the network and the feature maps of the shown blocks, with their shapes and values, the blocks shown
    # and the number of channels drawn for each being those the scene names
    write_network_input(input_tensor=star_tensor, output_path=model_folder / "input_star.png")
    facts = {"feature_maps.checkpoint_iteration": checkpoint_iteration, "feature_maps.input_image": STAR_IMAGE_NAME,
             "feature_maps.input_size": image_settings["size"], "feature_maps.input_channels": star_tensor.shape[1],
             "feature_maps.shown_blocks": SHOWN_BLOCK_NUMBERS, "feature_maps.shown_channel_count": SHOWN_CHANNEL_COUNT}
    for block_number in SHOWN_BLOCK_NUMBERS:
        block_output = block_outputs[block_number]
        shown_channels = draw_feature_map_grid(block_output=block_output,
                                               output_path=model_folder / f"feature_maps_block_{block_number}.png")
        facts.update({f"feature_maps.block_{block_number}.output_height": block_output.shape[1],
                      f"feature_maps.block_{block_number}.output_width": block_output.shape[2],
                      f"feature_maps.block_{block_number}.output_channels": block_output.shape[0],
                      f"feature_maps.block_{block_number}.shown_channels": shown_channels,
                      f"feature_maps.block_{block_number}.maximum_activation": round_fact(value=block_output.max()),
                      f"feature_maps.block_{block_number}.zero_fraction": round_fact(value=(block_output == 0).mean())})

    # Place of the strongest feature of the last shown block and the centre of its receptive field in the input, a
    # final position covering a square of the input of the stride of the backbone
    final_activation = block_outputs[SHOWN_BLOCK_NUMBERS[-1]].sum(axis=0)
    row, column = numpy.unravel_index(indices=int(final_activation.argmax()), shape=final_activation.shape)
    stride = image_settings["size"] / final_activation.shape[1]
    facts.update({"feature_maps.strongest_final_position.row": int(row),
                  "feature_maps.strongest_final_position.column": int(column),
                  "feature_maps.strongest_final_position.stride": round_fact(value=stride),
                  "feature_maps.strongest_final_position.receptive_field_centre_x": round_fact(
                      value=column * stride + (stride - 1) / 2),
                  "feature_maps.strongest_final_position.receptive_field_centre_y": round_fact(
                      value=row * stride + (stride - 1) / 2)})

    # Squeeze-and-excitation gates of the star, after their sigmoid
    gates = torch.sigmoid(input=gate_logits[0]).numpy()
    facts.update({"squeeze_excitation.block": GATE_BLOCK_NUMBER, "squeeze_excitation.layer": GATE_LAYER_INDEX + 1,
                  "squeeze_excitation.gate_count": len(gates),
                  "squeeze_excitation.gates": [round_fact(value=gate) for gate in gates],
                  "squeeze_excitation.mean_gate": round_fact(value=gates.mean()),
                  "squeeze_excitation.minimum_gate": round_fact(value=gates.min()),
                  "squeeze_excitation.maximum_gate": round_fact(value=gates.max())})
    write_reused_facts(facts_path=facts_path, figure_paths=figure_paths, facts=facts)
    return facts
