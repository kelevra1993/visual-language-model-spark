"""
Evaluation figures of the frame selection tutorials (the chapter on the evaluation and the deployment), made at every
build by tutorials/figures/frame_selection/figures.py from the standalone evaluation of the real frame selection run
(its Checkpoint_Evaluation folder, EVALUATION_FOLDER_NAME): the listing of the evaluation block of iteration 1000 in its
Results.txt, the rates written as percentages ("99.50 %") and read back between 0 and 1 for the facts, and the
probability of every frame chosen by the decoding of the run (selections.csv) for every wanted class at iterations 1000
and 2000, the right and wrong choices apart, against the confidence threshold of the class in class_thresholds.json,
found with that decoding at the target precision written as a percentage ("99 %"), so that the scene shows which wrong
choices the deployed decoder flags as low confidence and which go through.
"""

import json
import re

import matplotlib
import numpy

from matplotlib import pyplot
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box, register_fonts, round_fact,
                                                save_figure, style_chart_panel)
from tutorials.figures.frame_selection.selection_figures import EVALUATION_FOLDER_NAME, read_selection_rows
from tutorials.figures.sequence_labelling.sequence_utilities import (RATE_DECIMALS, parse_block_metrics,
                                                                     read_results_block, read_run_sections,
                                                                     write_listing)
from tutorial_builder.layout import NAMED_COLOURS
from vision_training.evaluation.report_formatting import PERCENT_SIGN, format_percentage_label
from vision_training.evaluation.sequence.frame_selection_metrics import compute_selection_metrics
from vision_training.models.sequence.frame_selection_decoding import NO_FRAME, VALID_DECODINGS

# Iterations of the evaluations compared, the one of the wrong picks shown (SELECTION_ITERATION of the selection
# figures) and the last one
THRESHOLD_ITERATIONS = [1000, 2000]

# Iteration of the evaluation block listed in the report scene
LISTED_ITERATION = 1000

# Characters kept of every listed line, the header line ending with a rule as wide as the console
LISTED_LINE_LENGTH = 118

# Title of the per-class table of a block, whose first row after its rule is the worst class of the decoding of the run
PER_CLASS_TABLE_TITLE = "Per Class Selection"

# Row of the per-class table: the class, its true frames and its Hungarian selection accuracy written as a percentage
PER_CLASS_ROW_PATTERN = re.compile(pattern=r"^\s+(\S+)\s+(\d+)\s+(\d+\.\d+)" + re.escape(PERCENT_SIGN))

# Colours of the right and wrong choices, and of the thresholds
RIGHT_COLOUR = SERIES_COLOURS[1]
WRONG_COLOUR = "#%02x%02x%02x" % NAMED_COLOURS["red"]
THRESHOLD_COLOUR = SERIES_COLOURS[2]


def draw_threshold_panel(axes: matplotlib.axes.Axes, selection_rows: Dict[Any, Dict[str, str]],
                         thresholds: Dict[str, Any], class_names: List[str], iteration: int) -> Dict[str, Any]:
    """
    Draws one evaluation on a panel: for every wanted class (one row), the probability of every frame the decoding of
    the thresholds (the decoding of the run) chose for it, right choices and wrong choices in their colours, and the
    threshold of the class as a tick with its value, and gives the facts of the panel.

    Args:
        axes (matplotlib.axes.Axes): The panel.
        selection_rows (Dict[Any, Dict[str, str]]): The rows of selections.csv of the evaluation.
        thresholds (Dict[str, Any]): The class_thresholds.json of the evaluation, with the decoding they were found
            with.
        class_names (List[str]): The wanted classes, one row each from the top.
        iteration (int): The evaluated iteration.

    Returns:
        Dict[str, Any]: The thresholds, recalls and choices of every class, named thresholds.iteration_<n>_*, and over
            every class the wrong choices, those flagged and not flagged, and the right choices flagged.
    """
    facts = {}
    random_generator = numpy.random.default_rng(seed=0)
    decoding = thresholds["decoding"]
    totals = {"wrong": 0, "wrong_flagged": 0, "wrong_unflagged": 0, "right_flagged": 0}
    for row_index, class_name in enumerate(class_names):
        # Choices of the decoding the thresholds were found with, in the columns of selections.csv named after it
        rows = [row for row in selection_rows.values() if row["class"] == class_name]
        chosen_probabilities = numpy.array([float(row[f"{decoding}_probabilities"]) for row in rows])
        right_choices = numpy.array([row[f"{decoding}_right_selections"] == "1" for row in rows])
        flagged_choices = numpy.array([row[f"{decoding}_low_confidence"] == "1" for row in rows])

        # Every right choice as a dot jittered around the row of its class, the wrong ones larger, on the row and on top
        row_position = len(class_names) - 1 - row_index
        jitter = random_generator.uniform(low=-0.18, high=0.18, size=len(rows))
        axes.scatter(x=chosen_probabilities[right_choices], y=row_position + jitter[right_choices], s=60, alpha=0.6,
                     color=RIGHT_COLOUR, edgecolors="none", zorder=3)
        axes.scatter(x=chosen_probabilities[~right_choices], y=numpy.full(shape=int((~right_choices).sum()),
                                                                         fill_value=float(row_position)),
                     s=260, color=WRONG_COLOUR, edgecolors=SURFACE_COLOUR, linewidths=2, zorder=4)

        # Threshold of the class, the probability from which its choices reach the target precision keeping the most true
        # frames
        class_threshold = thresholds["classes"][class_name]
        if class_threshold is not None:
            axes.plot([class_threshold["threshold"]] * 2, [row_position - 0.34, row_position + 0.34],
                      color=THRESHOLD_COLOUR, linewidth=4 * POINTS_PER_PIXEL, zorder=5)
            axes.text(x=class_threshold["threshold"], y=row_position + 0.38, s=f"{class_threshold['threshold']:.4f}",
                      color=THRESHOLD_COLOUR, fontsize=18 * POINTS_PER_PIXEL, ha="center", va="bottom", zorder=5)
        facts.update({
            f"thresholds.iteration_{iteration}_{class_name}_threshold":
                None if class_threshold is None else round_fact(value=class_threshold["threshold"]),
            f"thresholds.iteration_{iteration}_{class_name}_recall":
                None if class_threshold is None else round_fact(value=class_threshold["recall"]),
            f"thresholds.iteration_{iteration}_{class_name}_wrong": int((~right_choices).sum()),
            f"thresholds.iteration_{iteration}_{class_name}_flagged": int(flagged_choices.sum()),
            f"thresholds.iteration_{iteration}_{class_name}_minimum_right": round_fact(
                value=chosen_probabilities[right_choices].min())})

        # Wrong choices flagged for a human or going through, and right choices flagged with them, over every class
        totals["wrong"] += int((~right_choices).sum())
        totals["wrong_flagged"] += int((~right_choices & flagged_choices).sum())
        totals["wrong_unflagged"] += int((~right_choices & ~flagged_choices).sum())
        totals["right_flagged"] += int((right_choices & flagged_choices).sum())
    facts.update({f"thresholds.iteration_{iteration}_{name}": value for name, value in totals.items()})

    # Rows named after their class, the probability axis from 0 to 1
    style_chart_panel(axes=axes, title=f"Iteration {iteration}: probability of the chosen frames",
                      x_label="Probability of the chosen frame for its class", y_label=None)
    axes.set_xlim(left=-0.02, right=1.02)
    axes.set_ylim(bottom=-0.6, top=len(class_names) - 0.2)
    axes.set_yticks(ticks=list(range(len(class_names))), labels=list(reversed(class_names)))
    axes.tick_params(axis="y", labelsize=20 * POINTS_PER_PIXEL, labelcolor=TEXT_PRIMARY_COLOUR)
    return facts


def read_worst_class(block_lines: List[str], fact_prefix: str) -> Dict[str, Any]:
    """
    Reads the first row of the per-class table of an evaluation block of Results.txt, the worst class of the decoding
    of the run (the table puts the lowest Hungarian selection accuracy first), so that the narration names the class
    the report puts first instead of typing it.

    Args:
        block_lines (List[str]): The lines of the block (read_results_block).
        fact_prefix (str): The prefix of the fact names, e.g. "evaluation".

    Returns:
        Dict[str, Any]: The class and its Hungarian selection accuracy between 0 and 1, named <prefix>.worst_class and
            <prefix>.worst_class_accuracy.

    Raises:
        ValueError: If the block has no per-class table or no row under its rule.
    """
    table_start = next((line_index for line_index, line in enumerate(block_lines)
                        if line.strip() == PER_CLASS_TABLE_TITLE), None)
    if table_start is not None:
        # First row matching a class, its true frames and its accuracy after the title and the column headers
        for line in block_lines[table_start + 1:]:
            row_match = PER_CLASS_ROW_PATTERN.match(string=line)
            if row_match is not None:
                return {f"{fact_prefix}.worst_class": row_match.group(1),
                        f"{fact_prefix}.worst_class_accuracy": round_fact(value=float(row_match.group(3)) / 100.0)}
    raise ValueError(f"Expected a '{PER_CLASS_TABLE_TITLE}' table with a class row in the evaluation block starting "
                     f"'{block_lines[0][:40]}', but found none.")


def count_pick_kinds(selection_rows: Dict[Tuple[str, str], Dict[str, str]],
                     block_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Counts the kind of every pick of both decodings from selections.csv of the listed iteration with
    compute_selection_metrics of the repository: the share of the picks that took a frame to leave unselected
    (background picks) and the share that took the true frame of another wanted class (swapped picks). The run was
    evaluated before these two shares took the place of the unselected precision and recall in the report, so that its
    block does not hold them; the selection accuracy counted the same way must give the one of the block.

    Args:
        selection_rows (Dict[Tuple[str, str], Dict[str, str]]): The rows of selections.csv, one per sequence and
            wanted class.
        block_metrics (Dict[str, float]): The metrics of the listed block (parse_block_metrics), between 0 and 1.

    Returns:
        Dict[str, Any]: The facts evaluation.picks.<decoding>_background, _swapped (between 0 and 1) and the counts
            evaluation.picks.<decoding>_background_count, _swapped_count and evaluation.picks.count.

    Raises:
        ValueError: If the counted selection accuracy of a decoding differs from the one of the block.
    """
    class_names = sorted({class_name for _, class_name in selection_rows})
    sequence_names = sorted({sequence_name for sequence_name, _ in selection_rows})
    unselected_class_index = len(class_names)
    facts = {}
    for decoding in VALID_DECODINGS:
        # Every frame named in the rows of a sequence gets a position, the true frames their class, the others the
        # unselected class, which is what compute_selection_metrics reads at the true and chosen frames
        frame_labels, selected_frames = [], []
        for sequence_name in sequence_names:
            rows = [selection_rows[(sequence_name, class_name)] for class_name in class_names]
            true_frames = [[frame for frame in row["true_frames"].split(";") if frame] for row in rows]
            chosen_frames = [[frame for frame in row[f"{decoding}_frames"].split(";") if frame] for row in rows]
            frame_positions = {frame: position for position, frame in enumerate(sorted(
                {frame for frames in true_frames + chosen_frames for frame in frames}))}
            sequence_labels = numpy.full(shape=len(frame_positions), fill_value=unselected_class_index)
            for class_index, frames in enumerate(true_frames):
                sequence_labels[[frame_positions[frame] for frame in frames]] = class_index
            row_length = max(len(frames) for frames in chosen_frames)
            selected_frames.append(numpy.array(object=[[frame_positions[frame] for frame in frames] + [NO_FRAME] * (
                row_length - len(frames)) for frames in chosen_frames]))
            frame_labels.append(sequence_labels)

        # Shares of the picks of the decoding, checked against the selection accuracy of the block
        metrics = compute_selection_metrics(selected_frames=selected_frames, frame_labels=frame_labels,
                                            wanted_class_indices=list(range(len(class_names))),
                                            unselected_class_index=unselected_class_index)
        block_accuracy = block_metrics[f"evaluation.{decoding}_selection_accuracy"]
        if abs(metrics["selection_accuracy"] - block_accuracy) > 10 ** -RATE_DECIMALS:
            raise ValueError(f"Expected the {decoding} selection accuracy counted from selections.csv to be the "
                             f"{block_accuracy} of the report, got {metrics['selection_accuracy']}.")

        # Counts of the picks behind both shares, every pick being one chosen frame
        number_picks = sum(int((selection != NO_FRAME).sum()) for selection in selected_frames)
        for kind in ["background", "swapped"]:
            facts[f"evaluation.picks.{decoding}_{kind}"] = round_fact(value=metrics[f"{kind}_picks"])
            facts[f"evaluation.picks.{decoding}_{kind}_count"] = round(metrics[f"{kind}_picks"] * number_picks)
        facts["evaluation.picks.count"] = number_picks
    return facts


def draw_thresholds_figure(run_folder: Path, output_path: Path) -> Dict[str, Any]:
    """
    Draws the chosen frames and thresholds of the evaluations of THRESHOLD_ITERATIONS side by side and gives their
    facts and the boxes of the panels.

    Args:
        run_folder (Path): The folder of the frame selection run.
        output_path (Path): The PNG file.

    Returns:
        Dict[str, Any]: The facts, named thresholds.*, with the iterations of the panels.
    """
    register_fonts()
    figure, panel_axes = pyplot.subplots(nrows=1, ncols=len(THRESHOLD_ITERATIONS), figsize=(
        CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH, sharey=True)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.13, wspace=0.08)

    # Thresholds of every panel, the header writing the decoding and the target precision they were found with, as a
    # percentage like the headers of the report
    panel_thresholds = {iteration: json.loads((run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{iteration}" /
                                               "class_thresholds.json").read_text(encoding="utf-8"))
                        for iteration in THRESHOLD_ITERATIONS}
    first_thresholds = panel_thresholds[THRESHOLD_ITERATIONS[0]]
    figure.text(x=0.012, y=0.955, s=f"{EVALUATION_FOLDER_NAME}/Iteration_<n>/selections.csv · class_thresholds.json · "
                                    f"{first_thresholds['decoding']} decoding, target precision "
                                    f"{format_percentage_label(value=first_thresholds['target_precision'])}",
                color=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL, va="center",
                family=MONOSPACE_FONT_FAMILY)
    figure.text(x=0.012, y=0.905, s="Green: right choice · red: wrong choice · yellow tick: threshold of the class, "
                                    "a choice under it is flagged low confidence", color=TEXT_SECONDARY_COLOUR,
                fontsize=18 * POINTS_PER_PIXEL, va="center")

    # One panel per compared evaluation, its chosen frames read from its selections.csv, with the pixel box of the
    # panel and its labels for the highlights of the scene
    facts = {}
    for axes, iteration in zip(panel_axes, THRESHOLD_ITERATIONS):
        evaluation_folder = run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{iteration}"
        thresholds = panel_thresholds[iteration]
        facts.update(draw_threshold_panel(axes=axes, selection_rows=read_selection_rows(
            selections_path=evaluation_folder / "selections.csv"), thresholds=thresholds,
            class_names=list(thresholds["classes"]), iteration=iteration))
        facts[f"thresholds.iteration_{iteration}_panel_box"] = compute_pixel_box(figure=figure, axes=axes,
                                                                                 with_labels=True, margin=0)
        facts["thresholds.target_precision"] = thresholds["target_precision"]

    # Iterations of the panels, left to right, quoted by the narration instead of typed numbers
    facts["thresholds.iterations"] = THRESHOLD_ITERATIONS
    save_figure(figure=figure, output_path=output_path)
    pyplot.close(fig=figure)
    return facts


def make_evaluation_figures(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Makes the evaluation listing and the thresholds figure of the frame selection series and gives their facts, with
    the top-k of the evaluation of the run.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the frame selection run.

    Returns:
        Dict[str, Any]: The facts, named evaluation.* and thresholds.*.
    """
    # Block of the listed iteration in the Results.txt of the standalone evaluation, its rates in percent
    results_path = run_folder / EVALUATION_FOLDER_NAME / "Results.txt"
    block_lines = read_results_block(results_path=results_path, iteration=LISTED_ITERATION)
    write_listing(lines=[line[:LISTED_LINE_LENGTH] for line in block_lines],
                  listing_path=output_folder / "listings" / "results_block.txt")
    facts = draw_thresholds_figure(run_folder=run_folder, output_path=output_folder / "evaluation" /
                                   "thresholds.png")

    # Top-k of the evaluation of the run, naming its top-k selection accuracy and the candidates of its decoder
    experiment_configuration, _ = read_run_sections(run_folder=run_folder)
    facts.update({"evaluation.listed_iteration": LISTED_ITERATION,
                  "evaluation.last_iteration": THRESHOLD_ITERATIONS[-1],
                  "evaluation.results_listing_lines": len(block_lines),
                  "evaluation.top_k": int(experiment_configuration["evaluation"]["top_k"])})
    block_metrics = parse_block_metrics(block_lines=block_lines, fact_prefix="evaluation")
    facts.update(block_metrics)
    facts.update(read_worst_class(block_lines=block_lines, fact_prefix="evaluation"))

    # Background and swapped picks of the listed iteration, counted from its selections
    facts.update(count_pick_kinds(selection_rows=read_selection_rows(
        selections_path=run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{LISTED_ITERATION}" / "selections.csv"),
        block_metrics=block_metrics))

    # Headline metrics and worst class of the last evaluation
    last_block_lines = read_results_block(results_path=results_path, iteration=THRESHOLD_ITERATIONS[-1])
    last_prefix = f"evaluation.iteration_{THRESHOLD_ITERATIONS[-1]}"
    facts.update(parse_block_metrics(block_lines=last_block_lines, fact_prefix=last_prefix))
    facts.update(read_worst_class(block_lines=last_block_lines, fact_prefix=last_prefix))
    return facts
