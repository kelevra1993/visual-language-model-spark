"""
Draws the launch flow of the "Launching A Run" subsection of the README Training section (README/launch_flow.png):
the steps of `vision_training.main` from the configuration file to the training loop, each numbered, with the stop on
an invalid configuration and the choice between a new run and a resume made by the last checkpoint of the run folder.
It only draws boxes and text, the run name being the one of the README classifier. From the repository root:

    uv run python -m documentation.presentation.figures.draw_launch_flow --output_path README/launch_flow.png
"""

import argparse
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import Circle, FancyBboxPatch
from pathlib import Path

from vision_training.utilities.os_utilities import format_path_link

# Text, arrow and fill colours, the edge colour of every box going with its fill
TEXT_COLOUR, ARROW_COLOUR, NUMBER_COLOUR = "#1a202c", "#6b7280", "#1e293b"
GREY_BOX, BLUE_BOX, RED_BOX = ("#f7f9fb", "#374151"), ("#eef5ff", "#2563eb"), ("#fdf2f2", "#dc2626")
QUESTION_BOX, NEW_RUN_BOX, RESUME_BOX = ("#ffffff", "#9ca3af"), ("#effaf8", "#0f766e"), ("#f5f3ff", "#7c3aed")


def draw_box(axes: plt.Axes, left: float, bottom: float, width: float, height: float, colours: tuple, title: str,
             lines: list, step: str = "") -> None:
    """
    Draws one step of the launch flow as a rounded box with its bold title and its lines, and its step number in a
    dark disc on its top-left corner, so that the README reader follows the steps in order.

    Args:
        axes (plt.Axes): The axes of the figure, in pixels of the output image.
        left (float): The left edge of the box.
        bottom (float): The bottom edge of the box.
        width (float): The width of the box.
        height (float): The height of the box.
        colours (tuple): The fill and edge colours.
        title (str): The bold first line.
        lines (list): The lines under the title, as (text, is_code) pairs.
        step (str): The step number written on the corner, empty for none.
    """
    axes.add_patch(FancyBboxPatch(xy=(left, bottom), width=width, height=height, boxstyle="round,pad=0,rounding_size=18",
                                  facecolor=colours[0], edgecolor=colours[1], linewidth=3))
    # Title and lines stacked from the top, centred in the box
    total_height = 46 + 42 * len(lines)
    current_y = bottom + height / 2 + total_height / 2 - 23
    axes.text(x=left + width / 2, y=current_y, s=title, ha="center", va="center", fontsize=23, fontweight="bold",
              color=TEXT_COLOUR)
    for text, is_code in lines:
        current_y -= 42
        axes.text(x=left + width / 2, y=current_y, s=text, ha="center", va="center", fontsize=19,
                  color=TEXT_COLOUR, family="monospace" if is_code else "sans-serif")
    # Step number in a disc over the top-left corner
    if step:
        axes.add_patch(Circle(xy=(left + 6, bottom + height - 6), radius=30, facecolor=NUMBER_COLOUR, zorder=4))
        axes.text(x=left + 6, y=bottom + height - 6, s=step, ha="center", va="center", fontsize=22, fontweight="bold",
                  color="white", zorder=5)


def draw_arrow(axes: plt.Axes, points: list, colour: str = ARROW_COLOUR, dashed: bool = False) -> None:
    """
    Draws an arrow along a polyline, its head at the last point, for the transitions between the steps.

    Args:
        axes (plt.Axes): The axes of the figure.
        points (list): The (x, y) points of the polyline.
        colour (str): The colour of the arrow.
        dashed (bool): Whether the line is dashed (the stop on an invalid configuration).
    """
    for start, end in zip(points[:-2], points[1:-1]):
        axes.plot([start[0], end[0]], [start[1], end[1]], color=colour, linewidth=3, linestyle="--" if dashed else "-")
    axes.annotate("", xy=points[-1], xytext=points[-2],
                  arrowprops={"arrowstyle": "-|>,head_width=0.6,head_length=1.2", "color": colour, "linewidth": 3,
                              "linestyle": "--" if dashed else "-", "shrinkA": 0, "shrinkB": 0})


def draw_launch_flow(output_path: Path) -> None:
    """
    Draws the whole launch flow into a 2200 x 1160 pixel PNG file.

    Args:
        output_path (Path): The PNG file written.
    """
    figure = plt.figure(figsize=(22, 11.6), dpi=100)
    axes = figure.add_axes(rect=(0, 0, 1, 1))
    axes.set_xlim(0, 2200)
    axes.set_ylim(0, 1160)
    axes.axis("off")
    # First row: from the configuration file to the run folder
    draw_box(axes=axes, left=40, bottom=870, width=470, height=250, colours=GREY_BOX, title="Configuration file",
             lines=[("copy of the example of", False), ("the task, every key", False), ("--configuration_path", True)],
             step="1")
    draw_box(axes=axes, left=600, bottom=870, width=470, height=250, colours=GREY_BOX, title="Configuration check",
             lines=[("every key and value,", False), ("their combinations and", False), ("removed settings", False)],
             step="2")
    draw_box(axes=axes, left=1160, bottom=870, width=470, height=250, colours=BLUE_BOX, title="Run name",
             lines=[("built from the settings", False), ("CLS_IN-KAR-512_SE-…_MC35", True), ("+ run_name_suffix", False)],
             step="3")
    draw_box(axes=axes, left=1720, bottom=870, width=440, height=250, colours=BLUE_BOX, title="Run folder",
             lines=[("<experiment_folder>/", True), ("<run name>/", True)], step="4")
    for left_end, right_start in [(510, 600), (1070, 1160), (1630, 1720)]:
        draw_arrow(axes=axes, points=[(left_end + 4, 995), (right_start - 6, 995)])
    # Stop on an invalid configuration, below the check
    draw_box(axes=axes, left=615, bottom=595, width=440, height=175, colours=RED_BOX, title="Stops",
             lines=[("listing every problem,", False), ("nothing created", False)])
    draw_arrow(axes=axes, points=[(835, 866), (835, 774)], colour="#dc2626", dashed=True)
    # Last checkpoint question, below the run folder
    draw_box(axes=axes, left=1160, bottom=595, width=1000, height=175, colours=QUESTION_BOX,
             title="Last checkpoint of the run folder?",
             lines=[("Weights/full-checkpoint", True), ("or the newest complete Weights/Iteration_<n>/", False)], step="5")
    draw_arrow(axes=axes, points=[(1940, 866), (1940, 774)])
    # New run or resume, then the training loop
    draw_box(axes=axes, left=40, bottom=205, width=1040, height=265, colours=NEW_RUN_BOX, title="New run",
             lines=[("starts at iteration 1 from random weights,", False), ("or from components of other runs", False),
                    ("<Section>.pretrained when set", False)], step="6")
    draw_box(axes=axes, left=1200, bottom=205, width=960, height=265, colours=RESUME_BOX, title="Resume",
             lines=[("weights, optimizer state and moving", False), ("average of iteration n loaded,", False),
                    ("starts at iteration n + 1", False), ("pretrained components not read again", False)], step="6")
    draw_arrow(axes=axes, points=[(1280, 591), (1280, 530), (560, 530), (560, 474)])
    axes.text(x=920, y=548, s="none", ha="center", va="center", fontsize=20, style="italic", color="#4b5563")
    draw_arrow(axes=axes, points=[(1680, 591), (1680, 474)])
    axes.text(x=1700, y=535, s="iteration n", ha="left", va="center", fontsize=20, style="italic", color="#4b5563")
    axes.add_patch(FancyBboxPatch(xy=(40, 30), width=2120, height=110, boxstyle="round,pad=0,rounding_size=18",
                                  facecolor="#1e293b", edgecolor="#1e293b"))
    axes.text(x=1100, y=85, s="Training loop up to number_iterations, the total of the run · Ctrl+C or the last "
              "iteration saves a last checkpoint", ha="center", va="center", fontsize=20, color="white")
    axes.add_patch(Circle(xy=(46, 134), radius=30, facecolor="#ffffff", edgecolor="#1e293b", linewidth=3, zorder=4))
    axes.text(x=46, y=134, s="7", ha="center", va="center", fontsize=22, fontweight="bold", color="#1e293b", zorder=5)
    for x in (560, 1680):
        draw_arrow(axes=axes, points=[(x, 201), (x, 144)])
    figure.savefig(output_path, dpi=100, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    # Path of the PNG file, README/launch_flow.png for the README
    parser = argparse.ArgumentParser(description="Draw the launch flow figure of the README.")
    parser.add_argument("--output_path", type=Path, required=True)
    arguments = parser.parse_args()
    draw_launch_flow(output_path=arguments.output_path)
    print(f"Wrote {format_path_link(path=arguments.output_path)}")
