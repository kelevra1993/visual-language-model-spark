"""
Draws the moving average figure of the object detection benchmark: the test mAP@[0.5:0.95] of the trained and of the
averaged weights of its reference pair (run_21 and run_22, stage 7, proposal selection per level), read from the results
copy committed next to the benchmark plan (documentation/architecture_benchmarking/object_detection_results.csv).
Without --experiments_folder it draws the image of the moving average slide of the presentation deck
(documentation/presentation/figures/detection_moving_average.png), uploaded to the Slides artifact of the deck as an
asset. With it, it draws the image of the Moving Average Of The Weights section of the README
(README/moving_average_of_the_weights.png), which also shades the iterations the average looks back at the last
evaluation, 1 / (1 - decay): with the decay the launches were trained with, read from the training_configuration.yaml
saved in their run folders, and with the smaller decay the warm-up of ExponentialMovingAverage.get_current_decay
(vision_training/trainers/weight_averaging.py) still gave their last update. From the repository root, the README image
is drawn from the run snapshots committed for the tutorials:

    uv run python -m documentation.presentation.figures.plot_detection_moving_average \
        --results_path documentation/architecture_benchmarking/object_detection_results.csv \
        --output_path README/moving_average_of_the_weights.png \
        --experiments_folder tutorials/run_snapshots/hyperparameter_experiments/object_detection
"""

import argparse
import csv
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.trainers.weight_averaging import compute_update_decay
from vision_training.utilities.os_utilities import format_path_link

# Launches of the reference pair of the detection benchmark, the settings shipped in the detection configurations
REFERENCE_LAUNCHES = ["run_21", "run_22"]

# Colours of the averaged and the trained weights, the accent of the deck for the averaged ones
AVERAGED_COLOUR, TRAINED_COLOUR = "#1F77B4", "#E07B39"

# Colours of the windows shaded on the README image: the window of the decay of the launches in grey, the shorter
# window of the decay of their last update in the colour of the averaged weights, their labels in the dark text grey
CONFIGURED_WINDOW_COLOUR, WINDOW_TEXT_COLOUR = "#7F7F7F", "#333333"

# Heights of the brackets under the windows, in percent of test mAP: below the curves, which stay above 65 % over the
# last third of the training
CONFIGURED_WINDOW_BRACKET_HEIGHT, LAST_UPDATE_WINDOW_BRACKET_HEIGHT = 50.0, 37.0


def read_launch_curves(results_path: Path, launches: List[str]) -> Dict[Tuple[str, str], Dict[int, float]]:
    """
    Reads the test mAP@[0.5:0.95] of every evaluation of the given launches, for their trained and averaged weights,
    from the results file the benchmark driver writes, so that the figure follows the committed numbers.

    Args:
        results_path (Path): The results file, one row per launch and evaluated weights ('trained' or 'ema').
        launches (List[str]): The launches to read.

    Returns:
        Dict[Tuple[str, str], Dict[int, float]]: For every (launch, evaluated weights), the score by iteration.
    """
    curves = {}
    with results_path.open() as results_file:
        for row in csv.DictReader(results_file):
            if row["launch"] not in launches:
                continue
            # Every column map_50_95_<iteration> holding a value is one evaluation of the launch
            curves[(row["launch"], row["evaluated_weights"])] = {
                int(column.rsplit("_", 1)[1]): float(value) for column, value in row.items()
                if column.startswith("map_50_95_") and column.rsplit("_", 1)[1].isdigit() and value}
    for launch in launches:
        for evaluated_weights in ["trained", "ema"]:
            if (launch, evaluated_weights) not in curves:
                raise ValueError(f"{format_path_link(path=results_path)} holds no {evaluated_weights} row for "
                                 f"{launch}.")
    return curves


def read_launch_decay(results_path: Path, experiments_folder: Path, launches: List[str]) -> float:
    """
    Reads the decay of the moving average the launches were trained with, from the training_configuration.yaml of
    their run folders (<experiments folder>/<launch>/<run folder of the results file>/), the record of their recipe,
    since the launch files of the benchmark are not in git; the README image shades the window of that decay.

    Args:
        results_path (Path): The detection results file of the benchmark, whose run_folder column names the run
            folder of every launch.
        experiments_folder (Path): The folder holding one folder per launch, the hyperparameter_experiments/
            object_detection folder of the runs or its copy in the run snapshots of the tutorials.
        launches (List[str]): The launches, sharing one decay.

    Returns:
        float: The decay of the launches.

    Raises:
        ValueError: If a launch is missing from the results file, kept no moving average, or if the launches were
            trained with different decays.
    """
    run_folders = {}
    with results_path.open() as results_file:
        for row in csv.DictReader(results_file):
            if row["launch"] in launches:
                run_folders[row["launch"]] = row["run_folder"]
    missing_launches = [launch for launch in launches if launch not in run_folders]
    if missing_launches:
        raise ValueError(f"{format_path_link(path=results_path)} holds no row for {missing_launches}.")

    # Decay of every launch, from the moving average section of its saved configuration
    decays = {}
    for launch in launches:
        configuration_path = experiments_folder / launch / run_folders[launch] / "training_configuration.yaml"
        moving_average_configuration = load_configuration(
            configuration_path=configuration_path)["Optimization"]["exponential_moving_average"]
        if not moving_average_configuration["activated"]:
            raise ValueError(f"{format_path_link(path=configuration_path)} keeps no moving average of the weights.")
        decays[launch] = float(moving_average_configuration["decay"])
    if len(set(decays.values())) != 1:
        raise ValueError(f"The launches {launches} were trained with different decays {decays}.")
    return decays[launches[0]]


def draw_window_bracket(axis: plt.Axes, start_iteration: float, end_iteration: float, height: float, colour: str,
                        lines: List[str]) -> None:
    """
    Draws the bracket under a shaded window of the README image, spanning its iterations, with its label above it,
    right-aligned on the last evaluation so that a window narrower than its label stays readable.

    Args:
        axis (plt.Axes): The panel.
        start_iteration (float): The first iteration of the window.
        end_iteration (float): The last iteration of the window, the last evaluation.
        height (float): The height of the bracket, in percent of test mAP.
        colour (str): The colour of the bracket, that of its window.
        lines (List[str]): The lines of the label, the formula first.
    """
    axis.annotate(text="", xy=(start_iteration, height), xytext=(end_iteration, height),
                  arrowprops={"arrowstyle": "|-|", "color": colour, "linewidth": 1.5, "mutation_scale": 6,
                              "shrinkA": 0, "shrinkB": 0})
    axis.text(x=end_iteration, y=height + 1.0, s="\n".join(lines), ha="right", va="bottom", fontsize=12.5,
              color=WINDOW_TEXT_COLOUR, linespacing=1.4)


def draw_averaging_windows(axis: plt.Axes, last_iteration: int, decay: float) -> Dict[str, float]:
    """
    Shades, on the README image, the iterations the average looks back at the last evaluation, 1 / (1 - decay): the
    window of the decay of the launches, and the shorter window of the decay of their last update, which the warm-up
    still held below it, both ending on the last evaluation and labelled with the formula and its value.

    Args:
        axis (plt.Axes): The panel.
        last_iteration (int): The last evaluated iteration, the updates made by then (one per iteration from the first
            one, the launches keeping their average from the start).
        decay (float): The decay of the launches.

    Returns:
        Dict[str, float]: The decay of the last update and both windows, in iterations.
    """
    # Decay of the last update of the launches, lowered by the warm-up of the trainer (update index counted from 0)
    last_update_decay = compute_update_decay(decay=decay, update_index=last_iteration - 1)
    configured_window, last_update_window = 1.0 / (1.0 - decay), 1.0 / (1.0 - last_update_decay)

    # Both windows shaded behind the curves, the longer one in grey, the one of the last update in the averaged colour
    axis.axvspan(xmin=last_iteration - configured_window, xmax=last_iteration, color=CONFIGURED_WINDOW_COLOUR,
                 alpha=0.10, linewidth=0, zorder=0)
    axis.axvspan(xmin=last_iteration - last_update_window, xmax=last_iteration, color=AVERAGED_COLOUR, alpha=0.14,
                 linewidth=0, zorder=0)

    # Brackets with the formula and its value; the decay of the last update, a fraction such as 15,000 / 15,009, is
    # written with 4 decimals and its window rounded to the tens, which both its exact value and the written decay give
    draw_window_bracket(axis=axis, start_iteration=last_iteration - configured_window, end_iteration=last_iteration,
                        height=CONFIGURED_WINDOW_BRACKET_HEIGHT, colour=CONFIGURED_WINDOW_COLOUR,
                        lines=[f"1 / (1 − {decay:g}) = {configured_window:,.0f} iterations",
                               f"decay {decay:g} of the launches"])
    draw_window_bracket(axis=axis, start_iteration=last_iteration - last_update_window, end_iteration=last_iteration,
                        height=LAST_UPDATE_WINDOW_BRACKET_HEIGHT, colour=AVERAGED_COLOUR,
                        lines=[f"1 / (1 − {last_update_decay:.4f}) ≈ {round(last_update_window, -1):,.0f} iterations",
                               f"decay of update {last_iteration:,}, lowered by the warm-up"])
    return {"last_update_decay": last_update_decay, "configured_window": configured_window,
            "last_update_window": last_update_window}


def plot_detection_moving_average(results_path: Path, output_path: Path, experiments_folder: Optional[Path]) -> None:
    """
    Plots the pair means of the trained and averaged weights over the iterations, the two launches as thin lines and
    the gap at the last evaluation written on the figure, and prints the numbers quoted on the slide and in the README.
    With the run folders of the launches, it also shades the windows the average looks back at the last evaluation,
    the image of the README.

    Args:
        results_path (Path): The detection results file of the benchmark.
        output_path (Path): The PNG file to write.
        experiments_folder (Optional[Path]): The folder holding the run folders of the launches (see
            read_launch_decay) for the README image, None for the image of the deck.
    """
    curves = read_launch_curves(results_path=results_path, launches=REFERENCE_LAUNCHES)
    iterations = sorted(set.intersection(*[set(curve) for curve in curves.values()]))

    # Pair mean of every evaluation, for both weights, in percent
    pair_means = {evaluated_weights: [100 * sum(curves[(launch, evaluated_weights)][iteration]
                                                for launch in REFERENCE_LAUNCHES) / len(REFERENCE_LAUNCHES)
                                      for iteration in iterations]
                  for evaluated_weights in ["trained", "ema"]}

    figure, axis = plt.subplots(figsize=(10.8, 6.0), dpi=200)

    # Every launch as a thin line, the pair mean as a thick one, the gap between the means shaded
    for evaluated_weights, colour in [("ema", AVERAGED_COLOUR), ("trained", TRAINED_COLOUR)]:
        for launch in REFERENCE_LAUNCHES:
            axis.plot(iterations, [100 * curves[(launch, evaluated_weights)][iteration] for iteration in iterations],
                      color=colour, linewidth=1, alpha=0.35)
        label = "averaged weights (EMA)" if evaluated_weights == "ema" else "trained weights"
        axis.plot(iterations, pair_means[evaluated_weights], color=colour, linewidth=3, label=f"{label}, pair mean")
    axis.fill_between(iterations, pair_means["trained"], pair_means["ema"], color=AVERAGED_COLOUR, alpha=0.08)

    # Gap at the last evaluation, the one the benchmark compares
    last_iteration, last_ema, last_trained = iterations[-1], pair_means["ema"][-1], pair_means["trained"][-1]
    axis.annotate(text="", xy=(last_iteration, last_ema), xytext=(last_iteration, last_trained),
                  arrowprops={"arrowstyle": "<->", "color": "#333333", "linewidth": 1.5})
    axis.text(x=last_iteration - 300, y=(last_ema + last_trained) / 2, s=f"+{last_ema - last_trained:.2f} points",
              ha="right", va="center", fontsize=15, fontweight="bold", color="#333333")

    # Windows of the average on the README image, with thousands separators on its iterations like its labels, its
    # legend moved to the empty top left corner above the first curves
    windows = None
    if experiments_folder is not None:
        decay = read_launch_decay(results_path=results_path, experiments_folder=experiments_folder,
                                  launches=REFERENCE_LAUNCHES)
        windows = draw_averaging_windows(axis=axis, last_iteration=last_iteration, decay=decay)
        axis.xaxis.set_major_formatter(formatter=StrMethodFormatter(fmt="{x:,.0f}"))

    # Axes, legend and title sized for a slide
    axis.set_xlabel(xlabel="Iteration", fontsize=14)
    axis.set_ylabel(ylabel="Test mAP@[0.5:0.95] (%)", fontsize=14)
    axis.set_title(label="Object detection benchmark, reference pair (run_21, run_22), 10,000 test images",
                   fontsize=15)
    axis.tick_params(labelsize=12)
    axis.set_ylim(bottom=30, top=90)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right" if windows is None else "upper left", fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(fig=figure)

    # Numbers quoted on the slide and in the README
    largest_gap, largest_gap_iteration = max((ema - trained, iteration) for ema, trained, iteration
                                             in zip(pair_means["ema"], pair_means["trained"], iterations))
    print(f"Last iteration {last_iteration}: averaged {last_ema:.2f} %, trained {last_trained:.2f} %, "
          f"gap {last_ema - last_trained:.2f} points; largest gap {largest_gap:.2f} points at {largest_gap_iteration}")
    if windows is not None:
        print(f"Decay {decay:g}: window {windows['configured_window']:,.1f} iterations; decay of update "
              f"{last_iteration} {windows['last_update_decay']:.6f}: window {windows['last_update_window']:,.1f} "
              f"iterations")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Draws the moving average figure of the presentation deck, "
                                                          "or of the README with --experiments_folder.")
    argument_parser.add_argument("--results_path", type=Path, required=True)
    argument_parser.add_argument("--output_path", type=Path, required=True)
    argument_parser.add_argument("--experiments_folder", type=Path, default=None,
                                 help="Folder of the run folders of run_21 and run_22, whose saved configurations give "
                                      "the decay: draws the README image, the averaging windows shaded.")
    arguments = argument_parser.parse_args()
    plot_detection_moving_average(results_path=arguments.results_path, output_path=arguments.output_path,
                                  experiments_folder=arguments.experiments_folder)
