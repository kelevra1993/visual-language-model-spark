"""
Draws the animation of the Tasks section of the README (README/sequence_pooling_mechanism.gif) that explains why a
sequence task whose class is a motion (M2O moving shapes) needs ClassificationHead.global_pooling: flatten: a shape
moves across eight frames, a small convolution layer (four fixed edge filters, ReLU, then a 16 x 16 average to an 8 x 8
grid, as a backbone would downsample) turns every frame into a feature map, and the frame vector the temporal model
reads is taken either by average pooling (one mean per channel) or by flatten (every cell of every channel). The
average pooled vector is the same in every frame, so their cosine similarities are all 1.00, while the flattened
vectors follow the shape and drift apart, the pattern the trained runs of README/sequence_pooling_similarity.png show.
Everything is computed with torch from the drawn frames; no run or dataset is read. From the repository root:

    uv run python -m documentation.presentation.figures.draw_pooling_similarity_animation \
        --output_path README/sequence_pooling_mechanism.gif
"""

import argparse
import io
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, Tuple

from vision_training.utilities.os_utilities import format_path_link

# Frames of the sequence: side, number of frames, background and shape grey levels, and the triangle drawn in them
FRAME_SIDE, NUMBER_FRAMES = 128, 8
BACKGROUND_LEVEL, SHAPE_LEVEL = 30, 230
TRIANGLE_SIDE, TRIANGLE_TOP, FIRST_LEFT, STEP_PIXELS = 24, 52, 8, 12

# Side of the cells of the feature map, the downsampling of the backbone (128 / 16 = an 8 x 8 grid), and the rows of
# the map the triangle crosses, the only ones whose flattened values are not 0
CELL_PIXELS = 16
SHAPE_ROWS = list(range(TRIANGLE_TOP // CELL_PIXELS, (TRIANGLE_TOP + TRIANGLE_SIDE) // CELL_PIXELS + 1))

# Fixed 3 x 3 edge filters standing for the first convolutions of a backbone, one output channel each
EDGE_FILTERS = {"vertical edges": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                "horizontal edges": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                "rising diagonals": [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                "falling diagonals": [[2, 1, 0], [1, 0, -1], [0, -1, -2]]}

# Dark theme of the neighbouring README figure: background, text, muted text, average pooling orange, flatten blue
BACKGROUND_COLOUR, TEXT_COLOUR, MUTED_COLOUR = "#0F1626", "#E6E9EF", "#9AA3B2"
POOLING_COLOUR, FLATTEN_COLOUR = "#D08A0A", "#3A86E0"

# Durations of the animation frames in milliseconds: every sequence frame, then the last one held
STEP_DURATION, LAST_DURATION = 1100, 4000


def draw_sequence_frames() -> np.ndarray:
    """
    Draws the frames of the moving shape: one triangle moving right by STEP_PIXELS per frame on a plain background,
    fully inside every frame so that only its position changes.

    Returns:
        np.ndarray: The frames, [NUMBER_FRAMES, FRAME_SIDE, FRAME_SIDE], float32 in [0, 1].
    """
    frames = []
    for frame_index in range(NUMBER_FRAMES):
        frame = Image.new(mode="L", size=(FRAME_SIDE, FRAME_SIDE), color=BACKGROUND_LEVEL)
        left = FIRST_LEFT + STEP_PIXELS * frame_index
        ImageDraw.Draw(frame).polygon(
            xy=[(left, TRIANGLE_TOP + TRIANGLE_SIDE), (left + TRIANGLE_SIDE // 2, TRIANGLE_TOP),
                (left + TRIANGLE_SIDE, TRIANGLE_TOP + TRIANGLE_SIDE)], fill=SHAPE_LEVEL)
        frames.append(np.asarray(frame, dtype=np.float32) / 255.0)
    return np.stack(frames)


def compute_feature_maps(frames: np.ndarray) -> torch.Tensor:
    """
    Turns every frame into a feature map as a backbone would: the edge filters, a ReLU, and an average over cells of
    CELL_PIXELS so that every cell of the map describes one region of the frame.

    Args:
        frames (np.ndarray): The frames, [N, H, W] in [0, 1].

    Returns:
        torch.Tensor: The feature maps, [N, channels, H / CELL_PIXELS, W / CELL_PIXELS].
    """
    filters = torch.tensor(list(EDGE_FILTERS.values()), dtype=torch.float32).unsqueeze(1)
    # Replicate the border pixels so that the plain background gives no response at the frame border
    padded_frames = functional.pad(input=torch.from_numpy(frames).unsqueeze(1), pad=(1, 1, 1, 1), mode="replicate")
    responses = functional.relu(functional.conv2d(input=padded_frames, weight=filters))
    return functional.avg_pool2d(input=responses, kernel_size=CELL_PIXELS)


def compute_frame_vectors(feature_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gives the frame vector of every frame with both global poolings of ClassificationHead: average pooling keeps one
    mean per channel, flatten keeps every cell of every channel in order.

    Args:
        feature_maps (torch.Tensor): The feature maps, [N, C, H, W].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The average pooled vectors [N, C] and the flattened vectors [N, C x H x W].
    """
    return feature_maps.mean(dim=(2, 3)), feature_maps.flatten(start_dim=1)


def compute_cosine_similarities(vectors: torch.Tensor) -> np.ndarray:
    """
    Gives the cosine similarity of the frame vectors of every pair of frames, the matrix the README figures draw.

    Args:
        vectors (torch.Tensor): The frame vectors, [N, D].

    Returns:
        np.ndarray: The similarities, [N, N].
    """
    normalized = functional.normalize(input=vectors, dim=1)
    return (normalized @ normalized.T).numpy()


def draw_similarity_matrix(axis: plt.Axes, similarities: np.ndarray, shown_frames: int, colour: str,
                           title: str) -> None:
    """
    Draws the similarity matrix of one pooling, filled up to the frames already played, every cell written with its
    value.

    Args:
        axis (plt.Axes): The axis of the matrix.
        similarities (np.ndarray): The similarities, [N, N].
        shown_frames (int): The number of frames played so far, whose rows and columns are filled.
        colour (str): The colour of the pooling, its high end.
        title (str): The title above the matrix.
    """
    colour_map = LinearSegmentedColormap.from_list(name="pooling", colors=[BACKGROUND_COLOUR, colour])
    colour_map.set_bad(color=BACKGROUND_COLOUR)
    shown = np.full(similarities.shape, np.nan)
    shown[:shown_frames, :shown_frames] = similarities[:shown_frames, :shown_frames]
    axis.imshow(shown, cmap=colour_map, vmin=0.0, vmax=1.0)

    # Every filled cell written with its similarity, the empty cells outlined
    for row in range(similarities.shape[0]):
        for column in range(similarities.shape[1]):
            if row < shown_frames and column < shown_frames:
                axis.text(x=column, y=row, s=f"{similarities[row, column]:.2f}", ha="center", va="center", fontsize=8,
                          color=TEXT_COLOUR if similarities[row, column] < 0.75 else BACKGROUND_COLOUR)
            else:
                axis.add_patch(plt.Rectangle(xy=(column - 0.5, row - 0.5), width=1, height=1, fill=False,
                                             edgecolor="#26304A", linewidth=0.8))
    axis.set_xticks(range(similarities.shape[1]))
    axis.set_yticks(range(similarities.shape[0]))
    axis.tick_params(colors=MUTED_COLOUR, labelsize=9, length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(label=title, color=TEXT_COLOUR, fontsize=12, pad=8)


def style_axis(axis: plt.Axes, title: str) -> None:
    """
    Gives an image axis the dark theme: no ticks or frame, a title in the text colour.

    Args:
        axis (plt.Axes): The axis to style.
        title (str): Its title.
    """
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor("#26304A")
    axis.set_title(label=title, color=TEXT_COLOUR, fontsize=11.5, pad=6)


def draw_animation_frame(frames: np.ndarray, feature_maps: torch.Tensor, pooled_vectors: torch.Tensor,
                         flattened_vectors: torch.Tensor, pooled_similarities: np.ndarray,
                         flattened_similarities: np.ndarray, frame_index: int) -> Image.Image:
    """
    Draws one step of the animation: the current frame, its feature maps, its two frame vectors next to those of the
    frames before it, and both similarity matrices filled up to it.

    Args:
        frames (np.ndarray): The sequence frames.
        feature_maps (torch.Tensor): Their feature maps, [N, C, H, W].
        pooled_vectors (torch.Tensor): The average pooled vectors, [N, C].
        flattened_vectors (torch.Tensor): The flattened vectors, [N, C x H x W].
        pooled_similarities (np.ndarray): The similarities of the average pooled vectors.
        flattened_similarities (np.ndarray): The similarities of the flattened vectors.
        frame_index (int): The frame drawn.

    Returns:
        Image.Image: The animation frame.
    """
    figure = plt.figure(figsize=(12.0, 7.4), dpi=100, facecolor=BACKGROUND_COLOUR)
    grid = figure.add_gridspec(nrows=2, ncols=4, height_ratios=[1.0, 1.25], width_ratios=[1.0, 1.0, 1.0, 1.6],
                               hspace=0.42, wspace=0.45, left=0.03, right=0.98, top=0.90, bottom=0.04)
    figure.suptitle(t=f"A shape moving right, frame {frame_index} of 0 to {NUMBER_FRAMES - 1}: what each pooling hands "
                      f"to the temporal model", color=TEXT_COLOUR, fontsize=14, y=0.975)

    # Current frame, its cells outlined as the grid of the feature map
    frame_axis = figure.add_subplot(grid[0, 0])
    frame_axis.imshow(frames[frame_index], cmap="gray", vmin=0.0, vmax=1.0)
    for cell_edge in range(CELL_PIXELS, FRAME_SIDE, CELL_PIXELS):
        frame_axis.axvline(x=cell_edge - 0.5, color="#3A4560", linewidth=0.6)
        frame_axis.axhline(y=cell_edge - 0.5, color="#3A4560", linewidth=0.6)
    style_axis(axis=frame_axis, title=f"frame {frame_index} (128 pixels)")

    # Feature maps of the four channels in a 2 x 2 mosaic, the cells lit where the shape is
    channels, rows, columns = feature_maps.shape[1:]
    mosaic = np.full((2 * rows + 1, 2 * columns + 1), np.nan)
    for channel_index in range(channels):
        top, left = (channel_index // 2) * (rows + 1), (channel_index % 2) * (columns + 1)
        mosaic[top:top + rows, left:left + columns] = feature_maps[frame_index, channel_index].numpy()
    map_axis = figure.add_subplot(grid[0, 1])
    map_colours = LinearSegmentedColormap.from_list(name="maps", colors=["#16203A", FLATTEN_COLOUR, "#DCEBFF"])
    map_colours.set_bad(color=BACKGROUND_COLOUR)
    map_axis.imshow(mosaic, cmap=map_colours, vmin=0.0, vmax=float(feature_maps.max()))
    style_axis(axis=map_axis, title=f"feature map: {channels} channels x {rows} x {columns}")

    # Average pooled vector: the bars of every frame so far drawn on top of each other, all at the same heights
    pooled_axis = figure.add_subplot(grid[0, 2], facecolor=BACKGROUND_COLOUR)
    for previous_index in range(frame_index + 1):
        pooled_axis.bar(x=range(channels), height=pooled_vectors[previous_index].numpy(), width=0.6,
                        color=POOLING_COLOUR, alpha=1.0 if previous_index == frame_index else 0.25)
    pooled_axis.set_ylim(bottom=0.0, top=1.35 * float(pooled_vectors.max()))
    pooled_axis.set_xticks(range(channels))
    pooled_axis.set_yticks([])
    pooled_axis.tick_params(colors=MUTED_COLOUR, labelsize=9, length=0)
    for spine in pooled_axis.spines.values():
        spine.set_visible(False)
    pooled_axis.set_title(label=f"average pooling: {channels} values", color=TEXT_COLOUR, fontsize=11.5, pad=6)
    pooled_axis.text(x=0.5, y=-0.16, s="frames so far overlaid, same 4\nvalues in every frame: the position\n"
                                       "of the shape is lost",
                     transform=pooled_axis.transAxes, ha="center", va="top", fontsize=9.5, color=MUTED_COLOUR)

    # Flattened vector of every frame so far, one row per frame: its values of the map rows the shape crosses, in the
    # order of the vector (channel, row, column), the others being 0
    shown_indices = [channel_index * rows * columns + row * columns + column for channel_index in range(channels)
                     for row in SHAPE_ROWS for column in range(columns)]
    flattened_rows = np.full((NUMBER_FRAMES, len(shown_indices)), np.nan)
    flattened_rows[:frame_index + 1] = flattened_vectors[:frame_index + 1, shown_indices].numpy()

    # Rows drawn as an image, the channels split by vertical lines and named under the middle of their cells
    flatten_axis = figure.add_subplot(grid[0, 3])
    flatten_colours = LinearSegmentedColormap.from_list(name="flatten", colors=["#16203A", FLATTEN_COLOUR, "#DCEBFF"])
    flatten_colours.set_bad(color=BACKGROUND_COLOUR)
    flatten_axis.imshow(flattened_rows, aspect="auto", interpolation="nearest", cmap=flatten_colours, vmin=0.0,
                        vmax=float(flattened_vectors.max()))
    for channel_index in range(1, channels):
        flatten_axis.axvline(x=channel_index * len(SHAPE_ROWS) * columns - 0.5, color=MUTED_COLOUR, linewidth=0.8)
    flatten_axis.set_yticks(range(NUMBER_FRAMES))
    flatten_axis.set_xticks([(channel_index + 0.5) * len(SHAPE_ROWS) * columns - 0.5
                             for channel_index in range(channels)])
    flatten_axis.set_xticklabels([f"channel {channel_index}" for channel_index in range(channels)])
    flatten_axis.tick_params(colors=MUTED_COLOUR, labelsize=9, length=0)
    flatten_axis.set_ylabel(ylabel="frame", color=MUTED_COLOUR, fontsize=10)
    for spine in flatten_axis.spines.values():
        spine.set_visible(False)

    # Title with the length of the whole vector, and the note under the axis on the part of it shown
    flatten_axis.set_title(label=f"flatten: {flattened_vectors.shape[1]} values, one row per frame",
                           color=TEXT_COLOUR, fontsize=11.5, pad=6)
    flatten_axis.text(x=0.5, y=-0.16, s=f"the {len(shown_indices)} values of the rows the shape crosses\n"
                                         f"(the rest is 0): the lit cells move with the shape",
                      transform=flatten_axis.transAxes, ha="center", va="top", fontsize=9.5, color=MUTED_COLOUR)

    # Similarity matrices of both poolings, filled up to the current frame
    draw_similarity_matrix(axis=figure.add_subplot(grid[1, 0:2]), similarities=pooled_similarities,
                           shown_frames=frame_index + 1, colour=POOLING_COLOUR,
                           title="average pooling: cosine similarity of the frame vectors")
    draw_similarity_matrix(axis=figure.add_subplot(grid[1, 2:4]), similarities=flattened_similarities,
                           shown_frames=frame_index + 1, colour=FLATTEN_COLOUR,
                           title="flatten: cosine similarity of the frame vectors")

    # Render the figure to an image held in memory
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor=BACKGROUND_COLOUR)
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def draw_pooling_similarity_animation(output_path: Path) -> None:
    """
    Computes the frame vectors of the moving shape with both poolings and writes the animation, one step per frame,
    the last one held.

    Args:
        output_path (Path): The GIF to write.
    """
    frames = draw_sequence_frames()
    feature_maps = compute_feature_maps(frames=frames)
    pooled_vectors, flattened_vectors = compute_frame_vectors(feature_maps=feature_maps)
    pooled_similarities = compute_cosine_similarities(vectors=pooled_vectors)
    flattened_similarities = compute_cosine_similarities(vectors=flattened_vectors)
    print(f"Average pooled vectors equal in every frame: {torch.allclose(pooled_vectors, pooled_vectors[0:1])}, "
          f"smallest similarity {pooled_similarities.min():.4f}; flatten: neighbours "
          f"{np.diag(flattened_similarities, k=1).round(2).tolist()}, "
          f"first to last {flattened_similarities[0, -1]:.2f}")

    # One animation frame per sequence frame, written as a looping GIF
    animation_frames: List[Image.Image] = [
        draw_animation_frame(frames=frames, feature_maps=feature_maps, pooled_vectors=pooled_vectors,
                             flattened_vectors=flattened_vectors, pooled_similarities=pooled_similarities,
                             flattened_similarities=flattened_similarities, frame_index=frame_index)
        for frame_index in range(NUMBER_FRAMES)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation_frames[0].save(output_path, save_all=True, append_images=animation_frames[1:], loop=0, optimize=True,
                             duration=[STEP_DURATION] * (NUMBER_FRAMES - 1) + [LAST_DURATION])
    print(f"Wrote {format_path_link(path=output_path)} ({len(animation_frames)} frames)")


if __name__ == "__main__":
    # Output path from the command line
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--output_path", type=Path, required=True)
    arguments = argument_parser.parse_args()
    draw_pooling_similarity_animation(output_path=arguments.output_path)
