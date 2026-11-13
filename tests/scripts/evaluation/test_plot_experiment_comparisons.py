"""
Tests of scripts/evaluation/plot_experiment_comparisons.py, the comparison figures of the hyperparameter experiments:
the launch curves read from a results file (mean rows left out, unlabelled weights counted as trained), the mean of the
launches of a setting on the iterations they all evaluated, the gap to the reference setting, the trained weights of a
setting without moving average standing in the averaged-weights column, and the figures written for the stages of a
registry, with one column per kind of weights evaluated for every launch of the stage, whatever its number of launches,
the rates of the classification and sequence labelling written as percentages with their gaps in points, and the
reference stage of the instance segmentation experiment drawn from its test mask mAP.
"""

import csv
import pytest
import yaml

from matplotlib.figure import Figure
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List

from scripts.evaluation.plot_experiment_comparisons import (LAUNCH_LINE_STYLES, collect_setting_curves,
                                                            compute_gap_curve, compute_mean_curve, draw_weights_column,
                                                            plot_registry, read_results_curves, wrap_verdict)

# Columns of the fake results file, as collect_experiment_results.py writes them: the metric at every evaluation, its
# means, the swept settings and the run folder
RESULTS_COLUMNS = ["launch", "evaluated_weights", "accuracy_250", "accuracy_500", "accuracy_750", "accuracy_mean",
                   "accuracy_last_4_mean", "finished", "learning_rate", "run_folder"]

# Launch rows of two settings of two launches each, with the trained and averaged weights, the second setting stopped
# at 500 for one launch, and the mean rows that the figures leave out
TWO_SETTING_ROWS = [
    {"launch": "run_1", "evaluated_weights": "trained", "accuracy_250": "0.5", "accuracy_500": "0.6",
     "accuracy_750": "0.7", "learning_rate": "0.001"},
    {"launch": "run_2", "evaluated_weights": "trained", "accuracy_250": "0.3", "accuracy_500": "0.4",
     "accuracy_750": "0.5", "learning_rate": "0.001"},
    {"launch": "mean", "evaluated_weights": "trained", "accuracy_250": "0.4", "accuracy_500": "0.5",
     "accuracy_750": "0.6", "learning_rate": "0.001"},
    {"launch": "run_1", "evaluated_weights": "ema", "accuracy_250": "0.55", "accuracy_500": "0.65",
     "accuracy_750": "0.75", "learning_rate": "0.001"},
    {"launch": "run_2", "evaluated_weights": "ema", "accuracy_250": "0.35", "accuracy_500": "0.45",
     "accuracy_750": "0.55", "learning_rate": "0.001"},
    {"launch": "run_3", "evaluated_weights": "trained", "accuracy_250": "0.6", "accuracy_500": "0.7",
     "accuracy_750": "0.8", "learning_rate": "0.002"},
    {"launch": "run_4", "evaluated_weights": "trained", "accuracy_250": "0.4", "accuracy_500": "0.5",
     "accuracy_750": "", "learning_rate": "0.002"},
    {"launch": "run_3", "evaluated_weights": "ema", "accuracy_250": "0.65", "accuracy_500": "0.75",
     "accuracy_750": "0.85", "learning_rate": "0.002"},
    {"launch": "run_4", "evaluated_weights": "ema", "accuracy_250": "0.45", "accuracy_500": "0.55",
     "accuracy_750": "", "learning_rate": "0.002"}]

# Stage of the fake registry comparing both settings at 750, the first being the reference
TWO_SETTING_COMPARISON = {
    "experiment": "classification", "results_file": "results.csv", "stage": "2", "title": "Learning Rate",
    "plan_lines": "1-2", "last_iteration": 750, "compared_weights": "ema",
    "arms": [{"label": "1e-3", "launches": ["run_1", "run_2"], "reference": True, "trained_as_averaged": False},
             {"label": "2e-3", "launches": ["run_3", "run_4"], "reference": False, "trained_as_averaged": False}],
    "kept": "1e-3", "verdict": "1e-3 kept.", "notes": ""}


def write_results_file(results_path: Path, rows: List[Dict[str, str]], columns: List[str]) -> None:
    """
    Writes a fake results file, the cells a row does not give left empty, as collect_experiment_results.py leaves the
    settings a stage did not record.

    Args:
        results_path (Path): The file to write.
        rows (List[Dict[str, str]]): The rows, by column.
        columns (List[str]): The header of the file.
    """
    with results_path.open(mode="w", encoding="utf-8", newline="") as results_file:
        writer = csv.DictWriter(f=results_file, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_registry(registry_path: Path, comparisons: List[Dict[str, Any]]) -> None:
    """
    Writes a fake comparison registry next to the results files it reads.

    Args:
        registry_path (Path): The registry to write.
        comparisons (List[Dict[str, Any]]): Its stages.
    """
    with registry_path.open(mode="w", encoding="utf-8") as registry_file:
        yaml.safe_dump(data={"comparisons": comparisons}, stream=registry_file)


def test_read_results_curves_leaves_out_means_and_counts_unlabelled_weights_as_trained(tmp_path: Path) -> None:
    """
    The launch rows of a results file give one curve per launch and weights, without the mean rows, the empty cells of a
    stopped launch and the iterations of the mean columns, the rows written before the averaged weights counting as
    trained weights.
    """
    rows = [{"launch": "run_1", "accuracy_250": "0.5", "accuracy_500": "0.6", "accuracy_mean": "0.55"},
            {"launch": "run_2", "accuracy_250": "0.3", "accuracy_500": "", "accuracy_mean": "0.3"},
            {"launch": "mean", "accuracy_250": "0.4", "accuracy_500": "0.6", "accuracy_mean": "0.45"}]
    write_results_file(results_path=tmp_path / "results.csv", rows=rows, columns=RESULTS_COLUMNS)

    metric_prefix, launch_curves = read_results_curves(results_path=tmp_path / "results.csv")

    assert metric_prefix == "accuracy"
    assert launch_curves == {("run_1", "trained"): {250: 0.5, 500: 0.6}, ("run_2", "trained"): {250: 0.3}}


def test_read_results_curves_refuses_a_file_of_two_metrics(tmp_path: Path) -> None:
    """
    A results file holding the columns of two metrics is refused, since the figure would mix them.
    """
    write_results_file(results_path=tmp_path / "results.csv", rows=[{"launch": "run_1"}],
                       columns=["launch", "evaluated_weights", "accuracy_250", "macro_f1_250"])

    with pytest.raises(ValueError, match="single metric"):
        read_results_curves(results_path=tmp_path / "results.csv")


def test_compute_mean_curve_averages_the_iterations_of_every_launch() -> None:
    """
    The mean of a setting covers only the iterations all its launches evaluated, a launch stopped earlier ending it.
    """
    mean_curve = compute_mean_curve(launch_curves=[{250: 0.5, 500: 0.7, 750: 0.9}, {250: 0.7, 500: 0.9}])

    assert list(mean_curve) == [250, 500]
    assert mean_curve[250] == pytest.approx(0.6)
    assert mean_curve[500] == pytest.approx(0.8)


def test_compute_gap_curve_subtracts_the_reference_on_common_iterations() -> None:
    """
    The gap of a setting is its mean minus the mean of the reference, positive where it leads, on the iterations both
    evaluated.
    """
    gap_curve = compute_gap_curve(setting_curve={250: 0.6, 500: 0.8}, reference_curve={250: 0.5, 500: 0.9, 750: 1.0})

    assert list(gap_curve) == [250, 500]
    assert gap_curve[250] == pytest.approx(0.1)
    assert gap_curve[500] == pytest.approx(-0.1)


def test_plot_registry_draws_both_weights_when_every_launch_has_them(tmp_path: Path) -> None:
    """
    A stage whose launches all have trained and averaged weights gets a figure of two columns, named after its stage
    and title, in the folder of its experiment.
    """
    write_results_file(results_path=tmp_path / "results.csv", rows=TWO_SETTING_ROWS, columns=RESULTS_COLUMNS)
    write_registry(registry_path=tmp_path / "experiment_comparisons.yaml", comparisons=[TWO_SETTING_COMPARISON])

    figure_paths = plot_registry(registry_path=tmp_path / "experiment_comparisons.yaml",
                                 output_folder=tmp_path / "figures")

    assert figure_paths == [tmp_path / "figures" / "classification" / "stage_2_learning_rate.png"]
    with Image.open(fp=figure_paths[0]) as figure_image:
        two_column_width = figure_image.width

    # Without the averaged weights of one launch, the same stage gets the trained weights alone, half as wide
    rows_without_one_average = [row for row in TWO_SETTING_ROWS
                                if not (row["launch"] == "run_4" and row["evaluated_weights"] == "ema")]
    write_results_file(results_path=tmp_path / "results.csv", rows=rows_without_one_average, columns=RESULTS_COLUMNS)
    plot_registry(registry_path=tmp_path / "experiment_comparisons.yaml", output_folder=tmp_path / "figures")
    with Image.open(fp=figure_paths[0]) as figure_image:
        assert figure_image.width == pytest.approx(two_column_width / 2, abs=2)


def test_plot_registry_refuses_a_stage_without_one_reference(tmp_path: Path) -> None:
    """
    A stage must have exactly one reference setting, the one the gaps are measured to.
    """
    write_results_file(results_path=tmp_path / "results.csv", rows=TWO_SETTING_ROWS, columns=RESULTS_COLUMNS)
    comparison_without_reference = {**TWO_SETTING_COMPARISON,
                                     "arms": [{**arm, "reference": False} for arm in TWO_SETTING_COMPARISON["arms"]]}
    write_registry(registry_path=tmp_path / "experiment_comparisons.yaml", comparisons=[comparison_without_reference])

    with pytest.raises(ValueError, match="one reference"):
        plot_registry(registry_path=tmp_path / "experiment_comparisons.yaml", output_folder=tmp_path / "figures")


def test_plot_registry_refuses_a_filter_matching_no_stage(tmp_path: Path) -> None:
    """
    An experiment or stage asked for but absent from the registry is an error rather than an empty run.
    """
    write_registry(registry_path=tmp_path / "experiment_comparisons.yaml", comparisons=[TWO_SETTING_COMPARISON])

    with pytest.raises(ValueError, match="No comparison"):
        plot_registry(registry_path=tmp_path / "experiment_comparisons.yaml", output_folder=tmp_path / "figures",
                      stage="9")


def test_collect_setting_curves_reads_the_trained_weights_of_a_setting_without_average() -> None:
    """
    A setting trained without moving average (trained_as_averaged) shows its trained weights in the averaged-weights
    column, its label saying so, while the other setting shows its averaged weights, as stage 13 of the classification
    experiment compares them.
    """
    launch_curves = {("run_1", "trained"): {250: 0.5}, ("run_2", "trained"): {250: 0.3},
                     ("run_3", "trained"): {250: 0.6}, ("run_3", "ema"): {250: 0.7},
                     ("run_4", "trained"): {250: 0.4}, ("run_4", "ema"): {250: 0.5}}
    comparison = {**TWO_SETTING_COMPARISON,
                  "arms": [{**TWO_SETTING_COMPARISON["arms"][0], "trained_as_averaged": True},
                           TWO_SETTING_COMPARISON["arms"][1]]}

    setting_curves = collect_setting_curves(comparison=comparison, launch_curves=launch_curves, weights="ema")

    assert setting_curves[0]["label"] == "1e-3 (trained weights, no average)"
    assert setting_curves[0]["mean_curve"][250] == pytest.approx(0.4)
    assert setting_curves[1]["label"] == "2e-3"
    assert setting_curves[1]["mean_curve"][250] == pytest.approx(0.6)


def test_plot_registry_draws_a_setting_of_more_launches_than_line_styles(tmp_path: Path) -> None:
    """
    A setting repeated over more launches than there are line styles (six in the classification repeat stage) is drawn,
    the styles repeating, rather than failing.
    """
    repeated_launches = [f"run_1{letter}" for letter in "abcdef"][:len(LAUNCH_LINE_STYLES) + 2]
    rows = [{"launch": launch, "evaluated_weights": "trained", "accuracy_250": "0.5", "accuracy_500": "0.6",
             "accuracy_750": "0.7"} for launch in repeated_launches + ["run_2"]]
    write_results_file(results_path=tmp_path / "results.csv", rows=rows, columns=RESULTS_COLUMNS)
    comparison = {**TWO_SETTING_COMPARISON, "compared_weights": "trained",
                  "arms": [{"label": "off", "launches": repeated_launches, "reference": True,
                            "trained_as_averaged": False},
                           {"label": "on", "launches": ["run_2"], "reference": False, "trained_as_averaged": False}]}
    write_registry(registry_path=tmp_path / "experiment_comparisons.yaml", comparisons=[comparison])

    figure_paths = plot_registry(registry_path=tmp_path / "experiment_comparisons.yaml",
                                 output_folder=tmp_path / "figures")

    assert figure_paths[0].is_file()


@pytest.mark.parametrize(argnames="in_percent, expected_labels, expected_gaps, expected_ticks",
                         argvalues=[(True, ["1e-3 40.00 %", "2e-3 60.00 %"], ["+20.00 points"], ["40 %", "+5"]),
                                    (False, ["1e-3 0.4000", "2e-3 0.6000"], ["+0.2000"], ["0.40", "0.05"])])
def test_draw_weights_column_writes_rates_as_percentages(in_percent: bool, expected_labels: List[str],
                                                         expected_gaps: List[str], expected_ticks: List[str]) -> None:
    """
    Verifies the labels of a column of a comparison figure: the classification accuracy and the sequence labelling macro
    F1 written as percentages with their gaps in points and percent axes, the detection mAP with 4 decimals.

    Args:
        in_percent (bool): Whether the metric is written as a percentage.
        expected_labels (List[str]): The expected direct labels of the settings.
        expected_gaps (List[str]): The expected gap labels.
        expected_ticks (List[str]): The expected labels of the ticks 0.4 of the curve axis and 0.05 of the gap axis.
    """
    launch_curves = {("run_1", "ema"): {250: 0.5}, ("run_2", "ema"): {250: 0.3}, ("run_3", "ema"): {250: 0.7},
                     ("run_4", "ema"): {250: 0.5}}
    setting_curves = collect_setting_curves(comparison=TWO_SETTING_COMPARISON, launch_curves=launch_curves,
                                            weights="ema")
    figure = Figure()
    curve_axis, gap_axis = figure.subplots(nrows=2)
    _, _, gap_texts = draw_weights_column(curve_axis=curve_axis, gap_axis=gap_axis, setting_curves=setting_curves,
                                          last_iteration=250, column_title="Averaged weights", metric_title="accuracy",
                                          in_percent=in_percent)

    # Direct labels of the settings, gap labels and tick labels of both panels
    assert [annotation.get_text() for annotation in curve_axis.texts] == expected_labels
    assert gap_texts == expected_gaps
    tick_labels = [curve_axis.yaxis.get_major_formatter()(0.4, 0), gap_axis.yaxis.get_major_formatter()(0.05, 0)]
    if in_percent:
        assert tick_labels == expected_ticks
        assert gap_axis.get_ylabel() == "Gap to 1e-3\n(points)"


def test_verdict_lines_keep_every_number_with_its_unit() -> None:
    """
    Verifies the wrapping of a verdict under the title of a comparison figure: a line never ends on a percentage or a
    gap without its unit, the words staying those of the verdict.
    """
    verdict = ("layer kept: the highest mean, 89.69 % against 85.81 % for none, a gap of about 3.6 points over the "
               "last four evaluations, 97.93 % against 97.91 % for batch.")
    lines = wrap_verdict(verdict=verdict, width=44).split("\n")
    assert len(lines) > 3
    assert not any(line.rstrip().endswith(("89.69", "85.81", "3.6", "97.93", "97.91")) for line in lines)
    assert " ".join(lines).replace("\u00a0", " ") == verdict


def test_plot_registry_draws_an_instance_segmentation_reference_of_one_setting(tmp_path: Path) -> None:
    """
    The reference stage of the instance segmentation experiment, one setting of two launches read from a results file
    of the test mask mAP@[0.5:0.95] (mask_map_50_95 columns), gets its figure in the folder of the experiment.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    columns = ["launch", "evaluated_weights", "mask_map_50_95_1000", "mask_map_50_95_2000", "mask_map_50_95_mean",
               "mask_map_50_95_last_4_mean", "finished", "learning_rate", "run_folder"]
    rows = [{"launch": launch, "evaluated_weights": weights, "mask_map_50_95_1000": first_value,
             "mask_map_50_95_2000": second_value, "learning_rate": "0.001"}
            for launch, weights, first_value, second_value in [("run_1", "trained", "0.16", "0.37"),
                                                               ("run_2", "trained", "0.15", "0.36"),
                                                               ("run_1", "ema", "0.17", "0.42"),
                                                               ("run_2", "ema", "0.16", "0.39")]]
    write_results_file(results_path=tmp_path / "instance_segmentation_results.csv", rows=rows, columns=columns)
    write_registry(registry_path=tmp_path / "experiment_comparisons.yaml", comparisons=[{
        "experiment": "instance_segmentation", "results_file": "instance_segmentation_results.csv", "stage": "0",
        "title": "Reference", "plan_lines": "1-2", "last_iteration": 2000, "compared_weights": "ema",
        "arms": [{"label": "pool 14", "launches": ["run_1", "run_2"], "reference": True,
                  "trained_as_averaged": False}],
        "kept": "pool 14", "verdict": "Reference pair.", "notes": ""}])

    figure_paths = plot_registry(registry_path=tmp_path / "experiment_comparisons.yaml",
                                 output_folder=tmp_path / "figures")

    assert figure_paths == [tmp_path / "figures" / "instance_segmentation" / "stage_0_reference.png"]
    assert figure_paths[0].is_file()
