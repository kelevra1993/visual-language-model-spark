"""
Draws the figure of the moving average slide of the presentation deck (documentation/presentation): the test
mAP@[0.5:0.95] of the trained and of the averaged weights of the reference pair of the object detection benchmark
(run_21 and run_22, stage 7, proposal selection per level), read from the results copy committed next to the
benchmark plan (documentation/architecture_benchmarking/object_detection_results.csv). The image is uploaded to the
Slides artifact of the deck as an asset and shown on the moving-average slide.
"""

import argparse
import csv
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Tuple

# Launches of the reference pair of the detection benchmark, the settings shipped in the detection configurations
REFERENCE_LAUNCHES = ["run_21", "run_22"]

# Colours of the averaged and the trained weights, the accent of the deck for the averaged ones
AVERAGED_COLOUR, TRAINED_COLOUR = "#1F77B4", "#E07B39"


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
                raise ValueError(f"{results_path} holds no {evaluated_weights} row for {launch}.")
    return curves


def plot_detection_moving_average(results_path: Path, output_path: Path) -> None:
    """
    Plots the pair means of the trained and averaged weights over the iterations, the two launches as thin lines and
    the gap at the last evaluation written on the figure, and prints the numbers quoted on the slide.

    Args:
        results_path (Path): The detection results file of the benchmark.
        output_path (Path): The PNG file to write.
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

    # Axes, legend and title sized for a slide
    axis.set_xlabel(xlabel="Iteration", fontsize=14)
    axis.set_ylabel(ylabel="Test mAP@[0.5:0.95] (%)", fontsize=14)
    axis.set_title(label="Object detection benchmark, reference pair (run_21, run_22), 10,000 test images",
                   fontsize=15)
    axis.tick_params(labelsize=12)
    axis.set_ylim(bottom=30, top=90)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right", fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(fig=figure)

    # Numbers quoted on the slide
    largest_gap, largest_gap_iteration = max((ema - trained, iteration) for ema, trained, iteration
                                             in zip(pair_means["ema"], pair_means["trained"], iterations))
    print(f"Last iteration {last_iteration}: averaged {last_ema:.2f} %, trained {last_trained:.2f} %, "
          f"gap {last_ema - last_trained:.2f} points; largest gap {largest_gap:.2f} points at {largest_gap_iteration}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Draws the moving average figure of the presentation deck.")
    argument_parser.add_argument("--results_path", type=Path, required=True)
    argument_parser.add_argument("--output_path", type=Path, required=True)
    arguments = argument_parser.parse_args()
    plot_detection_moving_average(results_path=arguments.results_path, output_path=arguments.output_path)
