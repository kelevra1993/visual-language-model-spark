"""
Backbone figures of the foundations tutorials (the backbone and its modules chapter), made by
tutorials/figures/foundations/figures.py from the last checkpoint of the classification run, read on the CPU, so that
the scenes show what the trained backbone really computes: the input of a test star and the feature maps it gives at
the output of three blocks, the most active channels of each on a grid of tiles. The facts are the shapes and values of
those maps, the place of the strongest final feature (the centre of the receptive field squares) and the
squeeze-and-excitation gates of one layer for the star; the claims the chapter makes about the star (flat colours, held
by the final receptive field, every shown final channel peaking on it) are checked on the input and the maps. The
checkpoint inference is reused from the previous build unless forced; a build from the committed run snapshots, which
hold no checkpoint, reuses the outputs of this family committed in figure_reuse/foundations/model/ of the snapshots.
"""

import functools

import cv2
import numpy
import torch

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (STAR_IMAGE_NAME, find_star_image, has_last_checkpoint,
                                                load_classification_model, read_network_input, read_reused_facts,
                                                read_run_configuration, restore_committed_figures, round_fact,
                                                write_network_input, write_reused_facts)
from tutorial_builder.layout import CARD_COLOUR, NAMED_COLOURS

# Blocks whose output feature maps are drawn, with the most active channels of each laid out on a grid of tiles
SHOWN_BLOCK_NUMBERS = [1, 3, 6]
SHOWN_CHANNEL_COUNT = 8
GRID_COLUMN_COUNT = 4
TILE_SIZE = 256
TILE_GAP = 8

# Squeeze-and-excitation layer whose gates are given: the second layer of the last block, the layer whose gates spread
# the most for the test star in this run (a standard deviation of 0.24 at checkpoint 2000, from 0.04 to 0.16 for the
# gates of the other layers)
GATE_BLOCK_NUMBER = 6
GATE_LAYER_INDEX = 1

# Largest difference, in levels of any channel, between a pixel of the border of the test star image and the median
# colour of that border, and between a pixel inside the star and the median colour of the star, for the background and
# the star to count as flat, as the block 1 sentence of the backbone chapter describes them
FLAT_COLOUR_TOLERANCE = 8

# Difference from the background colour, in levels of any channel, above which a pixel belongs to the star, and the
# pixels the star mask is eroded by before its colour is measured, its anti-aliased edge left out
STAR_PIXEL_THRESHOLD = 30
STAR_EDGE_EROSION = 3

# Sequential ramp of the feature maps, from the card colour for no activation through blue to near white for the
# strongest activation of a channel, validated against the card
FEATURE_RAMP = numpy.array([CARD_COLOUR, NAMED_COLOURS["blue"], (232, 240, 255)], dtype=numpy.float64)

# Name of the JSON file keeping the facts of the figures for their reuse
FACTS_FILE_NAME = "backbone_facts.json"

# Folder of the family in the figures folder of the series, and in figure_reuse/foundations/ of the run snapshots,
# whose committed outputs (the figure_reuse entry of snapshot_files.yaml) a build without the checkpoint reuses
MODEL_FAMILY_FOLDER = "model"


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


def check_star_claims(star_image: numpy.ndarray, final_block_output: numpy.ndarray, shown_channels: List[int],
                      receptive_field: int, centre_x: float, centre_y: float, stride: float) -> None:
    """
    Checks on the test star the claims the backbone chapter makes about it, so that another STAR_IMAGE_NAME or a
    retrained run stops the build instead of narrating a property the image or the model lost: the background and the
    star are flat colours (the block 1 sentence), the receptive field square of the strongest final value holds the
    whole star (the receptive field scene), and every shown channel of the last block peaks on or next to the star (the
    last block sentence).

    Args:
        star_image (numpy.ndarray): The [H, W, 3] network input of the test star, BGR in [0, 255].
        final_block_output (numpy.ndarray): The [C, h, w] output of the last shown block.
        shown_channels (List[int]): The channels of that block drawn by the scene.
        receptive_field (int): The side of the receptive field after the last convolution of the backbone, in pixels.
        centre_x (float): The column of the centre of the receptive field squares in the input, in pixels.
        centre_y (float): The row of that centre, in pixels.
        stride (float): The input pixels between two neighbouring values of the final map.

    Raises:
        ValueError: If one of the claims does not hold for the test star.
    """
    # Background colour from the border of the image, which must be flat
    border_pixels = numpy.concatenate([star_image[0], star_image[-1], star_image[:, 0], star_image[:, -1]], axis=0)
    background_colour = numpy.median(border_pixels, axis=0)
    if numpy.abs(border_pixels - background_colour).max() > FLAT_COLOUR_TOLERANCE:
        raise ValueError(f"The border of the test star {STAR_IMAGE_NAME} is not a flat colour, while the backbone "
                         f"chapter says that block 1 responds to the flat colours of the star and of the background.")

    # Star pixels, those far from the background colour, whose core must be flat too
    star_mask = numpy.abs(star_image - background_colour).max(axis=2) > STAR_PIXEL_THRESHOLD
    if not star_mask.any():
        raise ValueError(f"No pixel of the test star {STAR_IMAGE_NAME} differs from its background.")
    core_mask = cv2.erode(src=star_mask.astype(numpy.uint8), kernel=numpy.ones((3, 3), dtype=numpy.uint8),
                          iterations=STAR_EDGE_EROSION).astype(bool)
    core_pixels = star_image[core_mask]
    if len(core_pixels) == 0 or numpy.abs(core_pixels - numpy.median(core_pixels, axis=0)).max() > \
            FLAT_COLOUR_TOLERANCE:
        raise ValueError(f"The star of {STAR_IMAGE_NAME} is not one flat colour, while the backbone chapter says that "
                         f"block 1 responds to the flat colours of the star and of the background.")

    # Whole star inside the receptive field square of the strongest final value
    star_rows, star_columns = numpy.nonzero(star_mask)
    half_side = receptive_field / 2
    if star_columns.min() < centre_x - half_side or star_columns.max() > centre_x + half_side or \
            star_rows.min() < centre_y - half_side or star_rows.max() > centre_y + half_side:
        raise ValueError(f"The star of {STAR_IMAGE_NAME} (columns {star_columns.min()} to {star_columns.max()}, rows "
                         f"{star_rows.min()} to {star_rows.max()}) leaves the receptive field square of "
                         f"{receptive_field} pixels centred on ({centre_x}, {centre_y}), which the receptive field "
                         f"scene says holds it.")

    # Every shown channel of the last block peaking on the star: the stride square of its strongest final value, grown
    # by one final value on every side (the tiles show a blob of neighbouring values), touches the star
    for channel in shown_channels:
        row, column = numpy.unravel_index(indices=int(final_block_output[channel].argmax()),
                                          shape=final_block_output[channel].shape)
        square = star_mask[max(int((row - 1) * stride), 0):int((row + 2) * stride),
                           max(int((column - 1) * stride), 0):int((column + 2) * stride)]
        if not square.any():
            raise ValueError(f"Channel {channel} of the last shown block peaks at row {row}, column {column}, away "
                             f"from the star of {STAR_IMAGE_NAME}, while the backbone chapter says every channel "
                             f"lights up where the star is.")


def make_backbone_figures(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Runs the backbone of the last checkpoint of the classification run on the test star, writes the input and the
    feature map grids of the shown blocks into the model folder of the series, and gives the facts of the maps and of
    the squeeze-and-excitation gates; the figures of the previous build are reused unless forced or incomplete. A run
    without its checkpoint (read from the committed run snapshots) reuses the outputs committed in figure_reuse/ of the
    snapshots instead (restore_committed_figures), whatever the force flag.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.
        force (bool): Whether the checkpoint is run again even when the previous build left the figures.

    Returns:
        Dict[str, Any]: The facts of the figures, named feature_maps.* (with the shown blocks and the number of channels
            drawn for each) and squeeze_excitation.*.

    Raises:
        FileNotFoundError: If the run has no checkpoint and the run snapshots hold no committed outputs of the family.
        RuntimeError: If the squeeze-and-excitation layer is not called once by the forward pass.
        ValueError: If the test star or the maps lose a property the chapter narrates (check_star_claims).
    """
    model_folder = output_folder / MODEL_FAMILY_FOLDER
    facts_path = model_folder / FACTS_FILE_NAME
    figure_paths = [model_folder / "input_star.png"] + [model_folder / f"feature_maps_block_{block_number}.png"
                                                        for block_number in SHOWN_BLOCK_NUMBERS]

    # A run without its checkpoint (read from the run snapshots) takes the committed outputs of the family, which it
    # cannot make again, even when forced; a run with its checkpoint behaves as before
    checkpoint_available = has_last_checkpoint(run_folder=run_folder)
    if not checkpoint_available and not restore_committed_figures(output_folder=output_folder,
                                                                  family_folder=MODEL_FAMILY_FOLDER):
        raise FileNotFoundError(f"No checkpoint in {run_folder / 'Weights'} and no committed outputs of the "
                                f"{MODEL_FAMILY_FOLDER} family in the run snapshots: build the foundations figures "
                                f"from the real run, then write them with tutorials/make_run_snapshots.py.")

    # Figures of the previous build, or the committed ones
    reused_facts = read_reused_facts(facts_path=facts_path, required_paths=figure_paths,
                                     force=force and checkpoint_available)
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

    # Claims of the chapter about the star checked on the input and the maps, the receptive field after the last
    # convolution of the backbone being the side of the largest square of the receptive field scene
    final_receptive_field = [layer for layer in model.backbone.compute_receptive_field_information()
                             if not layer["name"].startswith("Max Pooling")][-1]["receptive_field"]
    final_block_number = SHOWN_BLOCK_NUMBERS[-1]
    check_star_claims(star_image=star_tensor[0].permute(1, 2, 0).numpy(),
                      final_block_output=block_outputs[final_block_number],
                      shown_channels=facts[f"feature_maps.block_{final_block_number}.shown_channels"],
                      receptive_field=int(final_receptive_field), centre_x=column * stride + (stride - 1) / 2,
                      centre_y=row * stride + (stride - 1) / 2, stride=stride)

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
