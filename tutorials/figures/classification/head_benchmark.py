"""
Measured effects of the classification head for the classification tutorials (the chapter judging and shipping a
classifier), read at every build by tutorials/figures/classification/figures.py from the results of the architecture
benchmark of the repository (benchmark_results of tutorials/tutorial_sources.yaml,
documentation/architecture_benchmarking/results.csv, its stages described in experiment_plan.md): the test accuracy of
the launches comparing the normalization of the hidden layers, the global pooling, the class weighting and the
learning rate, averaged over the two launches of every setting the plan compares, so that the narration quotes the
benchmark the way the plan reads it, with the image size and classes of the benchmark read from its run names, the
launches per setting and the evaluated iterations the narration names. Nothing is drawn: the scene draws the values
as bars.
"""

import csv
import re

import numpy

from typing import Any, Dict, List

from tutorials.figures.figure_utilities import find_repository_file, round_fact
from tutorials.figures.foundations.benchmark import count_launches_per_setting, describe_benchmark_run_name

# Launches of every compared setting, those the stages of experiment_plan.md average (two launches per setting), with
# the column of results.csv the narration quotes: the accuracy at iteration 500 for the normalization, which changes
# how fast the head learns, and the mean of the last four evaluations (iterations 1250 to 2000) for the others
COMPARED_SETTINGS = {
    "normalization.none": (["run_3", "run_4"], "accuracy_500"),
    "normalization.batch": (["run_11", "run_12"], "accuracy_500"),
    "normalization.layer": (["run_13", "run_14"], "accuracy_500"),
    "normalization_last_4.none": (["run_3", "run_4"], "accuracy_last_4_mean"),
    "normalization_last_4.batch": (["run_11", "run_12"], "accuracy_last_4_mean"),
    "normalization_last_4.layer": (["run_13", "run_14"], "accuracy_last_4_mean"),
    "pooling.flatten": (["run_13", "run_14"], "accuracy_last_4_mean"),
    "pooling.average": (["run_19", "run_20"], "accuracy_last_4_mean"),
    "weighting.balanced": (["run_23", "run_24"], "accuracy_last_4_mean"),
    "weighting.none": (["run_29", "run_30"], "accuracy_last_4_mean")}

# Launches whose spread shows the noise between two launches of the same settings: the two launches without class
# weighting, compared by the mean of their last four evaluations like the settings
NOISE_LAUNCHES = ["run_29", "run_30"]

# Launches of the learning rates from 5e-4 up, which the plan finds stalled at chance in their first evaluations, and
# the column of results.csv read for them
HIGH_LEARNING_RATE_LAUNCHES = ["run_5", "run_6", "run_7", "run_8", "run_9", "run_10"]
HIGH_LEARNING_RATE_COLUMN = "accuracy_500"

# Columns of results.csv: the test accuracy at an evaluated iteration, and the mean of the last evaluations
ITERATION_COLUMN_PATTERN = re.compile(pattern=r"accuracy_(\d+)")
LAST_EVALUATIONS_COLUMN_PATTERN = re.compile(pattern=r"accuracy_last_(\d+)_mean")


def read_benchmark_rows() -> Dict[str, Dict[str, str]]:
    """
    Reads the rows of the launches of the benchmark results, keyed by their launch name, the mean rows left out.

    Returns:
        Dict[str, Dict[str, str]]: The row of every launch.
    """
    with open(file=find_repository_file(file_key="benchmark_results"), encoding="utf-8", newline="") as results_file:
        return {row["launch"]: row for row in csv.DictReader(results_file) if row["launch"].startswith("run_")}


def average_launches(rows: Dict[str, Dict[str, str]], launches: List[str], column: str) -> float:
    """
    Averages one accuracy column over launches of the benchmark, refusing a missing launch or value.

    Args:
        rows (Dict[str, Dict[str, str]]): The rows of the launches.
        launches (List[str]): The launch names.
        column (str): The column, e.g. "accuracy_500".

    Returns:
        float: The mean value.

    Raises:
        KeyError: If a launch is not in the results or has no value in the column.
    """
    missing = [launch for launch in launches if launch not in rows or rows[launch][column] == ""]
    if missing:
        raise KeyError(f"The launches {missing} have no {column} in the benchmark results.")
    return float(numpy.mean([float(rows[launch][column]) for launch in launches]))


def make_head_benchmark_facts() -> Dict[str, Any]:
    """
    Gives the measured effects of the head settings in the benchmark, the noise between two launches of the same
    settings, and the largest accuracy at iteration 500 of the launches from a learning rate of 5e-4 up, with what the
    narration says of the benchmark: the image size and classes of its launches (from their run names), the launches
    of every compared setting, the last evaluated iteration, the iteration of the normalization comparison and of the
    high learning rates, and the number of last evaluations averaged.

    Returns:
        Dict[str, Any]: The facts, named benchmark.*.

    Raises:
        ValueError: If the compared launches differ in image size or classes, or a column name is not one of results.csv.
    """
    rows = read_benchmark_rows()
    facts = {f"benchmark.{name}": round_fact(value=average_launches(rows=rows, launches=launches, column=column))
             for name, (launches, column) in COMPARED_SETTINGS.items()}

    # Spread between two launches of the same settings, and the stalled launches of the high learning rates
    noise_means = [float(rows[launch]["accuracy_last_4_mean"]) for launch in NOISE_LAUNCHES]
    facts.update({"benchmark.noise.first_mean": round_fact(value=noise_means[0]),
                  "benchmark.noise.second_mean": round_fact(value=noise_means[1]),
                  "benchmark.high_learning_rates.largest_accuracy_500": round_fact(
                      value=max(float(rows[launch][HIGH_LEARNING_RATE_COLUMN]) for launch in
                                HIGH_LEARNING_RATE_LAUNCHES)),
                  "benchmark.high_learning_rates.smallest_learning_rate": min(
                      float(rows[launch]["learning_rate"]) for launch in HIGH_LEARNING_RATE_LAUNCHES),
                  "benchmark.noise.launch_count": len(NOISE_LAUNCHES)})

    # Image size and classes shared by the compared launches, read from their run names
    compared_launches = sorted({launch for launches, _ in COMPARED_SETTINGS.values() for launch in launches})
    run_settings = [describe_benchmark_run_name(run_name=rows[launch]["run_folder"]) for launch in compared_launches]
    if any((settings["image_size"], settings["class_count"]) != (run_settings[0]["image_size"],
                                                                  run_settings[0]["class_count"])
           for settings in run_settings):
        raise ValueError(f"The compared launches {compared_launches} differ in image size or classes.")

    # Iterations named by the columns read: the last evaluation of results.csv, the evaluation of the normalization
    # comparison and of the high learning rates, and the number of last evaluations of the other comparisons
    evaluated_iterations = [int(match.group(1)) for match in
                            (ITERATION_COLUMN_PATTERN.fullmatch(string=column) for column in rows[compared_launches[0]])
                            if match is not None]
    normalization_match = ITERATION_COLUMN_PATTERN.fullmatch(string=COMPARED_SETTINGS["normalization.none"][1])
    high_learning_rate_match = ITERATION_COLUMN_PATTERN.fullmatch(string=HIGH_LEARNING_RATE_COLUMN)
    last_evaluations_match = LAST_EVALUATIONS_COLUMN_PATTERN.fullmatch(string=COMPARED_SETTINGS["pooling.flatten"][1])
    if not evaluated_iterations or None in (normalization_match, high_learning_rate_match, last_evaluations_match):
        raise ValueError("The columns of results.csv read by the head benchmark facts are not accuracy_<iteration> "
                         "and accuracy_last_<count>_mean columns.")
    facts.update({"benchmark.image_size": run_settings[0]["image_size"],
                  "benchmark.class_count": run_settings[0]["class_count"],
                  "benchmark.launches_per_setting": count_launches_per_setting(
                      launch_lists=[launches for launches, _ in COMPARED_SETTINGS.values()]),
                  "benchmark.last_iteration": max(evaluated_iterations),
                  "benchmark.normalization_iteration": int(normalization_match.group(1)),
                  "benchmark.high_learning_rates.iteration": int(high_learning_rate_match.group(1)),
                  "benchmark.last_evaluation_count": int(last_evaluations_match.group(1))})
    return facts
