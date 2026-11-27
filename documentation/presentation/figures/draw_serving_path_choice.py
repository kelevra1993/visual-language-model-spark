"""
Draws the decision figure of the "Which One When" subsection of the README Deployment section
(README/serving_path_choice.png): three questions asked top to bottom, each answered "yes" by a serving path on its
right (eager, TorchScript, ONNX with TensorRT FP16) and the last "no" leading to ONNX Runtime, with a band under them
saying that every exported file can be served by Triton. It only draws boxes and text; the throughputs come from the
serving path table of the README (serving benchmark, 9-class classifier, batch 16). From the repository root:

    uv run python -m documentation.presentation.figures.draw_serving_path_choice --output_path README/serving_path_choice.png
"""

import argparse
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from vision_training.utilities.os_utilities import format_path_link

# Colours of the questions, of the answers and of the Triton band, light enough for black text
QUESTION_COLOUR, ANSWER_COLOUR, TRITON_COLOUR, EDGE_COLOUR = "#eef2f7", "#dcefe3", "#e6e9fb", "#4a5568"

# Every row: the question, the serving path chosen on "yes", and the file and throughput written under it
DECISION_ROWS = [("Debugging, evaluating or\nchanging the model?", "Eager", "the checkpoint · reference of every path"),
                 ("Does the program running the\nmodel link PyTorch?", "TorchScript", "model.pt · 355 images per second"),
                 ("Most images per second\non one known GPU?", "ONNX with TensorRT FP16",
                  "model.onnx · 1,215 images per second\ncheck its test metric first")]
FINAL_ANSWER = ("ONNX Runtime", "model.onnx · CPU or GPU\n435 images per second on the GPU")


def draw_box(axes: plt.Axes, centre_x: float, centre_y: float, width: float, height: float, colour: str,
             title: str, subtitle: str = "") -> None:
    """
    Draws one rounded box of the decision figure, a question or a serving path, with its bold title and its grey
    subtitle, so that the README reader sees each choice as one card.

    Args:
        axes (plt.Axes): The axes of the figure, in data units of 0 to 10 across.
        centre_x (float): The horizontal centre of the box.
        centre_y (float): The vertical centre of the box.
        width (float): The width of the box.
        height (float): The height of the box.
        colour (str): The fill colour.
        title (str): The bold text of the box.
        subtitle (str): The smaller grey text under the title, empty for none.
    """
    box = FancyBboxPatch(xy=(centre_x - width / 2, centre_y - height / 2), width=width, height=height,
                         boxstyle="round,pad=0.02,rounding_size=0.15", facecolor=colour, edgecolor=EDGE_COLOUR,
                         linewidth=1.2)
    axes.add_patch(box)
    # Title above the centre when a subtitle follows, centred otherwise
    title_y = centre_y + (0.17 if subtitle else 0.0)
    axes.text(x=centre_x, y=title_y, s=title, ha="center", va="center", fontsize=12, fontweight="bold", color="#1a202c")
    if subtitle:
        axes.text(x=centre_x, y=centre_y - 0.22, s=subtitle, ha="center", va="center", fontsize=9.5, color="#4a5568")


def draw_arrow(axes: plt.Axes, start: tuple, end: tuple, label: str) -> None:
    """
    Draws one labelled arrow between two boxes of the decision figure.

    Args:
        axes (plt.Axes): The axes of the figure.
        start (tuple): The (x, y) point the arrow leaves.
        end (tuple): The (x, y) point the arrow reaches.
        label (str): The answer written on the arrow, "yes" or "no".
    """
    axes.annotate(text="", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": EDGE_COLOUR, "lw": 1.4})
    middle_x, middle_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    axes.text(x=middle_x + (0.12 if start[0] == end[0] else 0.0), y=middle_y + (0.0 if start[0] == end[0] else 0.12),
              s=label, ha="left" if start[0] == end[0] else "center", va="center", fontsize=10, style="italic",
              color="#2d3748")


def draw_serving_path_choice(output_path: Path) -> None:
    """
    Draws the whole decision figure and writes it, so that the README shows which serving path answers which
    situation in one glance.

    Args:
        output_path (Path): The PNG file to write.
    """
    figure, axes = plt.subplots(figsize=(10, 6.4), dpi=200)
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 6.4)
    axes.axis("off")

    # Questions on the left, one per row, their "yes" answer on the right
    question_x, answer_x, row_height, box_height = 2.3, 7.2, 1.45, 0.95
    row_centres = [5.6 - index * row_height for index in range(len(DECISION_ROWS) + 1)]
    for index, (question, answer, details) in enumerate(DECISION_ROWS):
        draw_box(axes=axes, centre_x=question_x, centre_y=row_centres[index], width=3.8, height=box_height,
                 colour=QUESTION_COLOUR, title=question)
        draw_box(axes=axes, centre_x=answer_x, centre_y=row_centres[index], width=4.6, height=box_height,
                 colour=ANSWER_COLOUR, title=answer, subtitle=details)
        draw_arrow(axes=axes, start=(question_x + 1.9, row_centres[index]), end=(answer_x - 2.3, row_centres[index]),
                   label="yes")
        # "No" leads down to the next question, or to the last answer after the last question
        draw_arrow(axes=axes, start=(question_x, row_centres[index] - box_height / 2),
                   end=(question_x, row_centres[index + 1] + box_height / 2), label="no")

    # Last answer, under the last question
    draw_box(axes=axes, centre_x=question_x, centre_y=row_centres[-1], width=3.8, height=box_height,
             colour=ANSWER_COLOUR, title=FINAL_ANSWER[0], subtitle=FINAL_ANSWER[1])

    # Triton band beside the last answer: it serves any exported file
    draw_box(axes=axes, centre_x=answer_x, centre_y=row_centres[-1], width=4.6, height=box_height,
             colour=TRITON_COLOUR, title="Many clients or models?",
             subtitle="serve model.pt or model.onnx in Triton,\nwhich batches the requests")

    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {format_path_link(path=output_path)}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Draws README/serving_path_choice.png.")
    argument_parser.add_argument("--output_path", type=str, required=True, help="PNG file to write.")
    arguments = argument_parser.parse_args()
    draw_serving_path_choice(output_path=Path(arguments.output_path))
