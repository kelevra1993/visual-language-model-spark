"""
Figures and facts of the evaluation scenes of the semantic segmentation series of the tutorial videos, made by
tutorials/figures/semantic_segmentation/figures.py from the evaluation folders and logs of the synthetic segmentation
run, both checkpoints evaluated again with the current code in its Checkpoint_Evaluation folder
(EVALUATION_FOLDER_NAME): the sample image of the test image of the chapters at both evaluated checkpoints, one above
the other; the shares of the pixel confusion matrix of the last checkpoint (confusion_matrix.csv) for the grid of the
storyboard; the IoU and boundary IoU of every class (class_metrics.csv); and the headline metrics of both checkpoints
read from its Tensorboard/Test.
"""

import csv

import numpy

from matplotlib import pyplot
from PIL import Image
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, POINTS_PER_PIXEL,
                                                SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, register_fonts, round_fact, save_figure,
                                                style_chart_panel)
from tutorials.figures.keypoint_detection.scalar_logs import read_scalar_values
from tutorials.figures.semantic_segmentation.segmentation_sources import (CHECKPOINT_ITERATION, EVALUATED_ITERATIONS,
                                                                         EVALUATION_FOLDER_NAME, SAMPLE_IMAGE_NAME)
from tutorial_builder.layout import CARD_COLOUR, NAMED_COLOURS

# Gap between the two sample images stacked in the samples figure
SAMPLE_GAP = 24

# Headline metrics of the evaluations, as logged in Checkpoint_Evaluation/Tensorboard/Test, and their labels in the grid
HEADLINE_METRICS = {"Mean IoU": "Mean IoU", "Mean Dice": "Mean Dice", "Mean Boundary IoU": "Boundary IoU",
                    "Mean Boundary F1": "Boundary F1", "Pixel Accuracy": "Pixel accuracy",
                    "Frequency Weighted IoU": "Freq. weighted IoU", "Macro ROC AUC": "Macro ROC AUC",
                    "Macro Average Precision": "Macro AP"}

# Pairs of true and predicted classes whose share of the pixels the narration quotes
CONFUSION_PAIRS = [("hexagon", "pentagon"), ("hexagon", "hexagon"), ("octagon", "hexagon"), ("octagon", "circle"),
                   ("circle", "octagon")]


def stack_sample_images(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Stacks the sample image of the test image of the chapters at both evaluated checkpoints, the first one above the
    second, into the figure of the samples scene, and gives the boxes of their prediction and error panels for the
    views of the scene.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the segmentation run.

    Returns:
        Dict[str, Any]: The boxes of the rows and of their right halves, and the size of the figure.
    """
    images = []
    for iteration in EVALUATED_ITERATIONS:
        with Image.open(fp=run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{iteration}" / "Samples" /
                        SAMPLE_IMAGE_NAME) as sample_image:
            images.append(sample_image.convert(mode="RGB"))
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + SAMPLE_GAP * (len(images) - 1)
    stacked = Image.new(mode="RGB", size=(width, height), color=CARD_COLOUR)
    facts = {"samples_width": width, "samples_height": height}
    top = 0
    for iteration, image in zip(EVALUATED_ITERATIONS, images):
        stacked.paste(im=image, box=(0, top))
        facts[f"samples_{iteration}_box"] = [0, top, image.width, top + image.height]
        facts[f"samples_{iteration}_right_box"] = [image.width // 2, top, image.width, top + image.height]
        facts[f"samples_{iteration}_left_box"] = [0, top, image.width // 2, top + image.height]
        top += image.height + SAMPLE_GAP
    output_path = output_folder / "evaluation" / "samples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stacked.save(fp=output_path)
    return facts


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Reads the rows of a metrics file of an evaluation of the run.

    Args:
        csv_path (Path): The CSV file.

    Returns:
        List[Dict[str, str]]: Its rows keyed by column.
    """
    with csv_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(f=csv_file))


def describe_confusion_matrix(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the pixel confusion matrix of the last evaluation of the run (pixel counts, rows true classes) and gives the
    share of every true class in percent for the grid of the storyboard, with the shares the narration quotes.

    Args:
        run_folder (Path): The folder of the segmentation run.

    Returns:
        Dict[str, Any]: The facts of the confusion matrix.
    """
    csv_path = run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{CHECKPOINT_ITERATION}" / "confusion_matrix.csv"
    with csv_path.open(mode="r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    class_names = rows[0][1:]
    counts = numpy.array([[int(value) for value in row[1:]] for row in rows[1:]], dtype=numpy.float64)
    shares = counts / counts.sum(axis=1, keepdims=True) * 100.0
    facts = {"confusion_values": [[round(float(value), 1) for value in row] for row in shares],
             "confusion_labels": [name.replace("_", " ") for name in class_names],
             "confusion_total_pixels": int(counts.sum())}
    for true_name, predicted_name in CONFUSION_PAIRS:
        true_index, predicted_index = class_names.index(true_name), class_names.index(predicted_name)
        facts[f"confusion_{true_name}_to_{predicted_name}"] = round_fact(value=shares[true_index, predicted_index])
        facts[f"confusion_{true_name}_to_{predicted_name}_row"] = true_index
        facts[f"confusion_{true_name}_to_{predicted_name}_column"] = predicted_index
    return facts


def draw_boundary_figure(output_folder: Path, run_folder: Path, boundary_tolerance: int) -> Dict[str, Any]:
    """
    Draws the IoU and the boundary IoU of every class of the last evaluation of the run, read from its
    class_metrics.csv, as one row per class sorted by IoU, the two values joined, so that the gap between the regions
    and their contours shows.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the segmentation run.
        boundary_tolerance (int): 'ExperimentConfiguration.evaluation.boundary_tolerance' of the run, in pixels, named
            by the legend.

    Returns:
        Dict[str, Any]: The facts of the region and boundary metrics of every class.
    """
    register_fonts()
    rows = read_csv_rows(csv_path=run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{CHECKPOINT_ITERATION}" /
                         "class_metrics.csv")
    rows = sorted(rows, key=read_intersection_over_union, reverse=True)
    figure = pyplot.figure(figsize=(CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
                           facecolor=SURFACE_COLOUR)
    axes = figure.add_axes(rect=(0.12, 0.12, 0.84, 0.78))
    style_chart_panel(axes=axes, title=f"Region and contour of every class, iteration {CHECKPOINT_ITERATION}",
                      x_label="Value", y_label=None)
    positions = numpy.arange(len(rows))
    iou_values = [float(row["iou"]) for row in rows]
    boundary_values = [float(row["boundary_iou"]) for row in rows]
    f1_values = [float(row["boundary_f1"]) for row in rows]
    axes.hlines(y=positions, xmin=boundary_values, xmax=iou_values, color="#%02x%02x%02x" % NAMED_COLOURS["border"],
                linewidth=3 * POINTS_PER_PIXEL)
    axes.scatter(iou_values, positions, s=(13 * POINTS_PER_PIXEL) ** 2, color=SERIES_COLOURS[0], zorder=3,
                 label="IoU")
    axes.scatter(boundary_values, positions, s=(13 * POINTS_PER_PIXEL) ** 2, color=SERIES_COLOURS[2], zorder=3,
                 label=f"boundary IoU, {boundary_tolerance} pixel tolerance")
    axes.scatter(f1_values, positions, s=(9 * POINTS_PER_PIXEL) ** 2, color=SERIES_COLOURS[4], zorder=3, marker="D",
                 label="boundary F1")
    axes.set_yticks(ticks=positions, labels=[row["class"].replace("_", " ") for row in rows])
    axes.set_ylim(bottom=len(rows) - 0.5, top=-0.5)
    axes.set_xlim(left=0.0, right=1.02)
    axes.tick_params(axis="y", labelcolor=TEXT_PRIMARY_COLOUR, labelsize=18 * POINTS_PER_PIXEL)
    axes.grid(axis="y", visible=False)
    axes.legend(loc="lower left", frameon=False, labelcolor=TEXT_SECONDARY_COLOUR, fontsize=18 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_folder / "evaluation" / "boundary_metrics.png")
    pyplot.close(fig=figure)

    # Facts of every class by name, and of the classes at both ends
    facts = {}
    for row in rows:
        name = row["class"]
        facts[f"class_{name}_iou"] = round_fact(value=float(row["iou"]))
        facts[f"class_{name}_boundary_iou"] = round_fact(value=float(row["boundary_iou"]))
        facts[f"class_{name}_boundary_f1"] = round_fact(value=float(row["boundary_f1"]))
        facts[f"class_{name}_pixels"] = int(row["pixels"])
    shapes = [row for row in rows if row["class"] != "background"]
    facts.update({"class_best": shapes[0]["class"].replace("_", " "),
                  "class_worst": shapes[-1]["class"].replace("_", " "),
                  "class_second_worst": shapes[-2]["class"].replace("_", " ")})
    return facts


def read_intersection_over_union(row: Dict[str, str]) -> float:
    """
    Gives the IoU of a row of class_metrics.csv, the sort key of the boundary figure.

    Args:
        row (Dict[str, str]): The row.

    Returns:
        float: Its IoU.
    """
    return float(row["iou"])


def describe_headline_metrics(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the headline metrics of both evaluated checkpoints of the run from Checkpoint_Evaluation/Tensorboard/Test, the
    grid of the storyboard and the values the narration quotes.

    Args:
        run_folder (Path): The folder of the segmentation run.

    Returns:
        Dict[str, Any]: The facts of the headline metrics.
    """
    values = read_scalar_values(log_folder=run_folder / EVALUATION_FOLDER_NAME / "Tensorboard" / "Test",
                                tags=list(HEADLINE_METRICS))
    facts = {"headline_values": [[round(values[tag][iteration], 4) for iteration in EVALUATED_ITERATIONS]
                                 for tag in HEADLINE_METRICS],
             "headline_row_labels": list(HEADLINE_METRICS.values()),
             "headline_column_labels": [f"iteration {iteration}" for iteration in EVALUATED_ITERATIONS]}
    for tag in HEADLINE_METRICS:
        name = tag.lower().replace(" ", "_")
        for iteration in EVALUATED_ITERATIONS:
            facts[f"headline_{name}_{iteration}"] = values[tag][iteration]
    return facts


def write_evaluation_listing(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Writes the listing of the evaluation folder of the last checkpoint of the run, every file with its size and the
    sample images counted, and counts the scalars and images its evaluations wrote to its Tensorboard/Test.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the segmentation run.

    Returns:
        Dict[str, Any]: The facts of the evaluation folder and of the test logs.
    """
    evaluation_folder = run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{CHECKPOINT_ITERATION}"
    entries = sorted(evaluation_folder.iterdir(), key=read_entry_name)
    lines = [f"{EVALUATION_FOLDER_NAME}/{evaluation_folder.name}/"]
    for entry_index, entry in enumerate(entries):
        branch = "└── " if entry_index == len(entries) - 1 else "├── "
        if entry.is_dir():
            lines.append(f"{branch}{entry.name}/   {len(list(entry.iterdir()))} images")
        else:
            lines.append(f"{branch}{entry.name:<30}{entry.stat().st_size:>9,} bytes")
    listing_path = output_folder / "evaluation" / "evaluation_folder.txt"
    listing_path.parent.mkdir(parents=True, exist_ok=True)
    listing_path.write_text(data="\n".join(lines) + "\n", encoding="utf-8")

    # Scalars and images of the test logs
    accumulator = EventAccumulator(path=str(run_folder / EVALUATION_FOLDER_NAME / "Tensorboard" / "Test"),
                                   size_guidance={"scalars": 0, "images": 0})
    accumulator.Reload()
    return {"evaluation_files": len(entries), "evaluation_test_scalars": len(accumulator.Tags()["scalars"]),
            "evaluation_test_images": len(accumulator.Tags()["images"]),
            "evaluation_samples": len(list((evaluation_folder / "Samples").iterdir()))}


def read_entry_name(entry: Path) -> str:
    """
    Gives the name of an entry of the evaluation folder in lower case, the sort key of its listing.

    Args:
        entry (Path): The file or folder.

    Returns:
        str: Its name in lower case.
    """
    return entry.name.lower()
