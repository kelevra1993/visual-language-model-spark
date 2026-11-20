"""
Tests of vision_training/evaluation/evaluation_curves.py, the evaluation summary and the evaluation curves kept up to
date after every test evaluation: the trained weights start the row of their iteration, replacing an earlier
evaluation of it, and the averaged weights add their 'ema_' columns; the best iterations keep their ties; the shaded
windows of the moving average follow the decay of its last update, given by its number of updates during the warm-up
of the decay, those of the highest averaged evaluation whatever the order of the evaluations, the configured window
of 1 / (1 − decay) iterations carrying about 63 % of its weight once the warm-up is over; a tiny classification run
keeping a moving average writes its summary, with the number of updates of its averaged weights, and draws its curves
after each evaluation, and a run evaluated before the summary existed starts it from its TensorBoard test curves.
"""

import csv
import math
import pytest
import torch
import yaml

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image
from torch import nn
from typing import Dict, List, Optional, Tuple

from tests.trainers.test_base_trainer import build_dummy_trainer
from tests.trainers.test_classification_trainer import build_trainer, write_smoke_configuration
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.evaluation.evaluation_curves import (CONFIGURED_WINDOW_COLOR, CURVES_DOTS_PER_INCH,
                                                          CURVES_FIGURE_SIZE, EVALUATION_CURVES_FIGURE_NAME,
                                                          EVALUATION_SUMMARY_FILE_NAME, WEIGHTS_COLORS, WEIGHTS_LABELS,
                                                          MovingAverageState, describe_averaging_windows,
                                                          draw_evaluation_curves, find_best_iterations,
                                                          read_evaluation_summary, read_moving_average_state,
                                                          read_summary_curves, read_test_curves, write_summary_columns)
from vision_training.trainers.weight_averaging import (ExponentialMovingAverage, compute_recent_updates_weight,
                                                       compute_update_decay)

# Decay of the moving average of the tiny classification run: a window of 10 iterations, longer than its 8 iterations
TINY_RUN_DECAY = 0.9

# Curves of a long run evaluated every 20,000 iterations, its moving average past the warm-up of its decay at its last
# evaluation, iteration 60,000
LONG_RUN_CURVES = {"trained": {20000: 0.5, 40000: 0.625, 60000: 0.75},
                   "ema": {20000: 0.375, 40000: 0.625, 60000: 0.8125}}


def read_summary_rows(summary_path: Path) -> List[Dict[str, str]]:
    """
    Reads the rows of an evaluation summary as a spreadsheet would, independently of the module under test.

    Args:
        summary_path (Path): The evaluation_summary.csv file.

    Returns:
        List[Dict[str, str]]: The rows, in file order.
    """
    with open(file=summary_path, mode="r", newline="") as summary_file:
        return list(csv.DictReader(f=summary_file))


def test_summary_rows_are_written_and_replaced(tmp_path: Path) -> None:
    """
    Verifies the rows of the evaluation summary: the trained weights start the row of their iteration and the averaged
    weights add their 'ema_' columns to it, the rows are kept in increasing iteration order whatever the order of the
    evaluations, a column met later is added after the others and left empty in the earlier rows, and a second
    evaluation of an iteration (a resumed run evaluating a checkpoint again) replaces its row, the averaged columns of
    the earlier evaluation included, without leaving a partial file. A file not written by the trainer is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    summary_path = tmp_path / "Evaluation" / EVALUATION_SUMMARY_FILE_NAME

    # Iteration 2000 with both weights, then iteration 1000 with the trained weights only, then a new column at 3000
    write_summary_columns(summary_path=summary_path, iteration=2000, columns={"Accuracy": 0.5, "Recall/cat": 0.25},
                          starts_row=True)
    write_summary_columns(summary_path=summary_path, iteration=2000,
                          columns={"ema_Accuracy": 0.625, "ema_Recall/cat": 0.375}, starts_row=False)
    write_summary_columns(summary_path=summary_path, iteration=1000, columns={"Accuracy": 0.125, "Recall/cat": 0.0},
                          starts_row=True)
    write_summary_columns(summary_path=summary_path, iteration=3000, columns={"Accuracy": 0.75, "Recall/dog": 1.0},
                          starts_row=True)
    column_names, _ = read_evaluation_summary(summary_path=summary_path)
    assert column_names == ["iteration", "Accuracy", "Recall/cat", "ema_Accuracy", "ema_Recall/cat", "Recall/dog"]
    assert read_summary_rows(summary_path=summary_path) == [
        {"iteration": "1000", "Accuracy": "0.125", "Recall/cat": "0.0", "ema_Accuracy": "", "ema_Recall/cat": "",
         "Recall/dog": ""},
        {"iteration": "2000", "Accuracy": "0.5", "Recall/cat": "0.25", "ema_Accuracy": "0.625",
         "ema_Recall/cat": "0.375", "Recall/dog": ""},
        {"iteration": "3000", "Accuracy": "0.75", "Recall/cat": "", "ema_Accuracy": "", "ema_Recall/cat": "",
         "Recall/dog": "1.0"}]

    # Iteration 2000 evaluated again: its trained weights replace the whole row, then its averaged weights complete it
    write_summary_columns(summary_path=summary_path, iteration=2000, columns={"Accuracy": 0.5625, "Recall/cat": 0.5},
                          starts_row=True)
    assert read_summary_rows(summary_path=summary_path)[1] == {
        "iteration": "2000", "Accuracy": "0.5625", "Recall/cat": "0.5", "ema_Accuracy": "", "ema_Recall/cat": "",
        "Recall/dog": ""}
    write_summary_columns(summary_path=summary_path, iteration=2000,
                          columns={"ema_Accuracy": 0.6875, "ema_Recall/cat": 0.4375}, starts_row=False)
    summary_rows = read_summary_rows(summary_path=summary_path)
    assert [row["iteration"] for row in summary_rows] == ["1000", "2000", "3000"]
    assert (summary_rows[1]["Accuracy"], summary_rows[1]["ema_Accuracy"]) == ("0.5625", "0.6875")
    assert sorted(path.name for path in summary_path.parent.iterdir()) == [EVALUATION_SUMMARY_FILE_NAME]

    # A file whose first column is not the iteration was not written by the trainer
    summary_path.write_text(data="Accuracy,iteration\n0.5,1000\n")
    with pytest.raises(expected_exception=ValueError, match="to start with the 'iteration' column"):
        write_summary_columns(summary_path=summary_path, iteration=4000, columns={"Accuracy": 0.5}, starts_row=True)


def test_windows_follow_the_highest_averaged_evaluation_whatever_the_order(tmp_path: Path) -> None:
    """
    Verifies which averaged weights give the shaded windows when the checkpoints are evaluated in the order of
    evaluate_checkpoint.py with "all", from the last one down (12, 9, 6, 3): those of the highest iteration, with the
    number of updates of its row, and not those evaluated last; no window without moving average, before the first
    averaged evaluation, or when the row of the highest averaged evaluation holds no number of updates (a row started
    from TensorBoard).

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    summary_path = tmp_path / "Checkpoint_Evaluation" / EVALUATION_SUMMARY_FILE_NAME
    for iteration in [12, 9, 6, 3]:
        write_summary_columns(summary_path=summary_path, iteration=iteration, columns={"Accuracy": iteration / 20.0},
                              starts_row=True)
        write_summary_columns(summary_path=summary_path, iteration=iteration,
                              columns={"ema_Accuracy": iteration / 24.0, "ema_number_updates": iteration},
                              starts_row=False)
    column_names, summary_rows = read_evaluation_summary(summary_path=summary_path)
    assert [row["iteration"] for row in summary_rows] == ["3", "6", "9", "12"]
    curves = read_summary_curves(column_names=column_names, summary_rows=summary_rows, metric_name="Accuracy")
    assert read_moving_average_state(averaged_curve=curves["ema"], summary_rows=summary_rows,
                                     moving_average_decay=0.9998) == {"decay": 0.9998, "number_updates": 12,
                                                                       "iteration": 12}
    assert read_moving_average_state(averaged_curve=curves["ema"], summary_rows=summary_rows,
                                     moving_average_decay=None) is None
    assert read_moving_average_state(averaged_curve={}, summary_rows=summary_rows, moving_average_decay=0.9998) is None

    # A higher averaged evaluation without number of updates, as the rows started from TensorBoard
    write_summary_columns(summary_path=summary_path, iteration=15, columns={"Accuracy": 0.8}, starts_row=True)
    write_summary_columns(summary_path=summary_path, iteration=15, columns={"ema_Accuracy": 0.8}, starts_row=False)
    column_names, summary_rows = read_evaluation_summary(summary_path=summary_path)
    curves = read_summary_curves(column_names=column_names, summary_rows=summary_rows, metric_name="Accuracy")
    assert read_moving_average_state(averaged_curve=curves["ema"], summary_rows=summary_rows,
                                     moving_average_decay=0.9998) is None


def test_best_iterations_keep_their_ties_and_skip_missing_values() -> None:
    """
    Verifies the best evaluations of a headline metric: every iteration sharing the highest value is kept in the order
    of the values, a value that is not a number never wins, and a metric without any value has no best iteration.
    """
    assert find_best_iterations(values={4: 0.5, 8: math.nan, 16: 0.75, 12: 0.75}) == ([16, 12], 0.75)
    assert find_best_iterations(values={4: math.nan, 8: 0.25}) == ([8], 0.25)
    assert find_best_iterations(values={4: math.nan}) is None
    assert find_best_iterations(values={}) is None


def test_window_of_the_last_update_follows_the_warm_up_of_the_decay() -> None:
    """
    Verifies the windows of the moving average shaded on the evaluation curves, read from its number of updates: the
    decay of 0.9998 is reached at the update of index 44,990 only, so that after 2,000 and 15,000 updates the decay of
    the last update gives windows of about 220 and 1,670 iterations, shaded in front of the configured window of 5,000
    iterations, and after 60,000 updates the configured window alone, labelled with its formula and the 63 % of the
    weight of the average it carries.
    """
    assert compute_update_decay(decay=0.9998, update_index=0) == pytest.approx(expected=0.1)
    assert compute_update_decay(decay=0.9998, update_index=44989) < 0.9998
    assert compute_update_decay(decay=0.9998, update_index=44990) == 0.9998

    # During the warm-up, the configured window behind the shorter window of the last update
    for number_updates, expected_length, expected_label in [
            (2000, 2009 / 9, "EMA window of the last update ≈ 1/(1 − d) = 223 iterations\n(d = 0.99552 at update "
                             "2,000, lowered by the warm-up)"),
            (15000, 15009 / 9, "EMA window of the last update ≈ 1/(1 − d) = 1,668 iterations\n(d = 0.9994 at update "
                               "15,000, lowered by the warm-up)")]:
        configured_window, last_update_window = describe_averaging_windows(decay=0.9998, number_updates=number_updates)
        assert configured_window["length"] == pytest.approx(expected=5000.0)
        assert configured_window["label"] == ("Configured EMA window ≈ 1/(1 − decay) = 5,000 iterations\n(decay "
                                              "0.9998, reached once the warm-up is over)")
        assert configured_window["color"] == CONFIGURED_WINDOW_COLOR
        assert last_update_window["length"] == pytest.approx(expected=expected_length)
        assert last_update_window["label"] == expected_label
        assert last_update_window["color"] == WEIGHTS_COLORS["ema"]

    # Warm-up over: the configured window alone, with the share of the weight of the average it carries
    assert describe_averaging_windows(decay=0.9998, number_updates=60000) == [
        {"length": pytest.approx(expected=5000.0), "color": WEIGHTS_COLORS["ema"],
         "label": "EMA window ≈ 1/(1 − decay) = 5,000 iterations\n(decay 0.9998, 63 % of the weight of the average)"}]
    assert describe_averaging_windows(decay=0.5, number_updates=100)[0]["label"] == (
        "EMA window ≈ 1/(1 − decay) = 2 iterations\n(decay 0.5, 75 % of the weight of the average)")


@pytest.mark.parametrize(argnames=["number_updates", "recent_updates", "expected_weight"],
                         argvalues=[(60000, 5000, 1.0 - math.exp(-1.0)), (15000, 1668, 0.654), (2000, 223, 0.654)])
def test_last_updates_carry_their_share_of_the_average(number_updates: int, recent_updates: int,
                                                       expected_weight: float) -> None:
    """
    Verifies numerically, with the moving average of the trainer, the share of the weight of the average carried by
    the updates of a window: an average of a weight of 0 for every earlier update, then of a weight of 1 for the
    updates of the window, is the share of these updates, about 1 − 1/e ≈ 63 % for the 5,000 last updates past the
    warm-up at a decay of 0.9998, and about 65 % for the window of the last update during the warm-up.

    Args:
        number_updates (int): The number of updates of the average, the last ones being those of the window.
        recent_updates (int): The number of updates of the window.
        expected_weight (float): The share of the weight of the average expected in the window, within 0.001.
    """
    model = nn.Linear(in_features=1, out_features=1, bias=False)
    nn.init.zeros_(tensor=model.weight)
    moving_average = ExponentialMovingAverage(model=model, decay=0.9998)
    moving_average.number_updates = number_updates - recent_updates

    # The updates of the window, with a weight of 1
    with torch.no_grad():
        model.weight.fill_(value=1.0)
    for _ in range(recent_updates):
        moving_average.update(model=model)
    window_weight = float(moving_average.averaged_model.weight)
    computed_weight = compute_recent_updates_weight(decay=0.9998, number_updates=number_updates,
                                                    recent_updates=recent_updates)
    assert window_weight == pytest.approx(expected=computed_weight, abs=1e-4)
    assert computed_weight == pytest.approx(expected=expected_weight, abs=1e-3)
    with pytest.raises(expected_exception=ValueError, match="last updates of the moving average"):
        compute_recent_updates_weight(decay=0.9998, number_updates=number_updates, recent_updates=number_updates + 1)


@pytest.mark.parametrize(argnames=["curves", "moving_average", "expected_spans"], argvalues=[
    (LONG_RUN_CURVES, {"decay": 0.9998, "number_updates": 60000, "iteration": 60000}, [(55000, 60000)]),
    ({"trained": {5000: 0.5, 15000: 0.625}, "ema": {5000: 0.375, 15000: 0.6875}},
     {"decay": 0.9998, "number_updates": 15000, "iteration": 15000}, [(10000, 15000), (15000 - 15009 / 9, 15000)]),
    ({"trained": {4000: 0.5, 12000: 0.625}, "ema": {12000: 0.6875}},
     {"decay": 0.9998, "number_updates": 2000, "iteration": 12000}, [(7000, 12000), (12000 - 2009 / 9, 12000)]),
    ({"trained": {500: 0.25, 1000: 0.5}}, {"decay": 0.9998, "number_updates": 1000, "iteration": 1000},
     [(0, 1000), (1000 - 1009 / 9, 1000)]),
    (LONG_RUN_CURVES, None, [])])
def test_shaded_windows_end_at_the_evaluation_of_the_averaged_weights(
        curves: Dict[str, Dict[int, float]], moving_average: Optional[MovingAverageState],
        expected_spans: List[Tuple[float, float]]) -> None:
    """
    Verifies the spans shaded on the evaluation curves, ending at the evaluation of the averaged weights: the 5,000
    iterations of a decay of 0.9998 once its warm-up is over, the configured window behind the window of the last
    update during the warm-up, the latter given by the number of updates of the average and not by the iteration (an
    average restarted at a resume, 2,000 updates at iteration 12,000), the configured window starting at iteration 0
    when the run is shorter, every window named in the legend, and none without moving average.

    Args:
        curves (Dict[str, Dict[int, float]]): The drawn curves of the trained and averaged weights.
        moving_average (Optional[MovingAverageState]): The moving average evaluated last, None when it is off.
        expected_spans (List[Tuple[float, float]]): The first and last iterations of every shaded span, the longest
            first.
    """
    figure = Figure()
    FigureCanvasAgg(figure=figure)
    axis = figure.subplots()
    draw_evaluation_curves(axis=axis, curves=curves, marked_points=[], metric_label="Accuracy", in_percent=True,
                           moving_average=moving_average)

    # The shaded spans, then their legend entries after those of the curves
    windows = [patch for patch in axis.patches if isinstance(patch, Rectangle)]
    assert [(window.get_x(), window.get_x() + window.get_width()) for window in windows] == [
        pytest.approx(expected=expected_span) for expected_span in expected_spans]
    legend_texts = [text.get_text() for text in axis.get_legend().get_texts()]
    expected_curve_labels = [WEIGHTS_LABELS[weights] for weights in curves]
    assert legend_texts[:len(curves)] == expected_curve_labels
    window_labels = legend_texts[len(curves):]
    assert len(window_labels) == len(expected_spans)
    if len(expected_spans) == 1:
        assert window_labels[0].startswith("EMA window ≈ 1/(1 − decay) = 5,000 iterations")
    if len(expected_spans) == 2:
        assert window_labels[0].startswith("Configured EMA window ≈ 1/(1 − decay) = 5,000 iterations")
        assert window_labels[1].startswith("EMA window of the last update ≈ 1/(1 − d) = ")
        assert f"at update {moving_average['number_updates']:,}" in window_labels[1]


def test_best_points_are_ringed_with_their_value() -> None:
    """
    Verifies the marked points of the evaluation curves: every tied iteration ringed, a single label next to the
    first one, the labels written in the order of the marked points, and the ticks of a rate axis kept within 0 and
    100 % with the room left above the curves for the labels.
    """
    figure = Figure()
    FigureCanvasAgg(figure=figure)
    axis = figure.subplots()
    marked_points = [{"weights": "trained", "iterations": [60000], "label": "Trained best: iteration 60,000, 75 %"},
                     {"weights": "ema", "iterations": [40000, 60000],
                      "label": "EMA best: iteration 40,000 (+1 tied), 81.25 %"}]
    tied_curves = {"trained": LONG_RUN_CURVES["trained"], "ema": {20000: 0.375, 40000: 0.8125, 60000: 0.8125}}
    draw_evaluation_curves(axis=axis, curves=tied_curves, marked_points=marked_points, metric_label="Accuracy",
                           in_percent=True, moving_average={"decay": 0.9998, "number_updates": 60000,
                                                            "iteration": 60000})

    # One ring per marked iteration, one label per marked point, the highest above its ring
    rings = [line for line in axis.get_lines() if line.get_linestyle() == "None"]
    assert [list(ring.get_xdata()) for ring in rings] == [[60000], [40000, 60000]]
    labels = {annotation.get_text(): annotation for annotation in axis.texts}
    assert list(labels) == [marked_point["label"] for marked_point in marked_points]
    assert labels["EMA best: iteration 40,000 (+1 tied), 81.25 %"].get_verticalalignment() == "bottom"
    assert labels["Trained best: iteration 60,000, 75 %"].get_verticalalignment() == "top"
    assert all(0.0 <= tick <= 1.0 for tick in axis.get_yticks())
    assert axis.get_ylim()[1] > 0.8125


def test_tiny_run_draws_its_evaluation_curves_after_every_evaluation(tmp_path: Path,
                                                                      capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the evaluation summary and curves of a tiny classification run keeping a moving average (8 iterations,
    evaluated at 4 and 8, without the per-class outputs, sample images and predictions, whose figures would take most
    of the time of the test): the summary holds one row per evaluated iteration with every scalar of both weights, the
    averaged ones in 'ema_' columns, the same values as the TensorBoard test curves, and the figure is drawn next to
    Results.txt, its link printed after each of the four evaluations.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=64,
                             split_sizes={"train": 16, "validation": 8, "test": 6}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")
    configuration_path = write_smoke_configuration(
        temporary_directory=tmp_path, dataset_folder=dataset_folder, number_iterations=8, head_settings={},
        evaluation_settings={"per_class_report": False, "number_sample_images": 0, "save_predictions": False})

    # The moving average of the weights turned on, with a window of 10 iterations
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["Optimization"]["exponential_moving_average"] = {"activated": True, "decay": TINY_RUN_DECAY}
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))
    trainer = build_trainer(configuration_path=configuration_path)
    trainer.run_training_loop()

    # One row per evaluated iteration, the scalars of both weights equal to their TensorBoard curves
    summary_path = trainer.experiment_folder / "Evaluation" / EVALUATION_SUMMARY_FILE_NAME
    summary_rows = read_summary_rows(summary_path=summary_path)
    assert [row["iteration"] for row in summary_rows] == ["4", "8"]
    trained_curves = read_test_curves(test_folder=trainer.tensorboard_directory / "Test")
    averaged_curves = read_test_curves(test_folder=trainer.tensorboard_directory / "Test_EMA")
    assert set(summary_rows[0]) == {"iteration", *trained_curves, *(f"ema_{name}" for name in averaged_curves),
                                    "ema_number_updates"}
    for row in summary_rows:
        for curve_name, curve in trained_curves.items():
            assert float(row[curve_name]) == pytest.approx(expected=curve[int(row["iteration"])], abs=1e-6)
        for curve_name, curve in averaged_curves.items():
            assert float(row[f"ema_{curve_name}"]) == pytest.approx(expected=curve[int(row["iteration"])], abs=1e-6)

        # The moving average started with the run, one update per training iteration
        assert row["ema_number_updates"] == row["iteration"]

    # The figure next to Results.txt, at the size and resolution of the evaluation curves
    figure_path = trainer.experiment_folder / EVALUATION_CURVES_FIGURE_NAME
    with Image.open(figure_path) as image:
        assert image.size == (round(CURVES_FIGURE_SIZE[0] * CURVES_DOTS_PER_INCH),
                              round(CURVES_FIGURE_SIZE[1] * CURVES_DOTS_PER_INCH))
    console_output = capsys.readouterr().out
    assert console_output.count("Evaluation Curves : ") == 4
    assert "Evaluation Summary Started From The TensorBoard Test Curves" not in console_output
    assert "Error encountered during iteration" not in console_output


def test_run_evaluated_before_the_summary_starts_it_from_tensorboard(tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a run evaluated before the evaluation summary existed (its summary removed after its evaluation of
    iteration 4) starts it at its next evaluation from its TensorBoard test curves: the row of iteration 4 holds the
    float32 values of its Test and Test_EMA curves, the row of iteration 8 the exact values of its evaluation, and the
    console says where the earlier rows come from.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0, moving_average_activated=True)
    trainer.run_training_loop()
    summary_path = trainer.evaluation_root / EVALUATION_SUMMARY_FILE_NAME
    exact_first_row = read_summary_rows(summary_path=summary_path)[0]
    summary_path.unlink()
    capsys.readouterr()

    # The resumed run evaluates iteration 8, its summary started from the evaluation of iteration 4 in TensorBoard
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=9, view_input_images=False,
                                          number_sample_images=0, moving_average_activated=True)
    resumed_trainer.run_training_loop()
    summary_rows = read_summary_rows(summary_path=summary_path)
    assert [row["iteration"] for row in summary_rows] == ["4", "8"]
    assert set(summary_rows[0]) == set(exact_first_row)
    for column_name, exact_value in exact_first_row.items():
        if column_name == "ema_number_updates":
            continue
        assert float(summary_rows[0][column_name]) == pytest.approx(expected=float(exact_value), abs=1e-6)
        assert float(summary_rows[0][column_name]) == float(torch.tensor(data=float(exact_value), dtype=torch.float32))

    # TensorBoard holds no number of updates, which only the evaluations written by the trainer give
    assert (exact_first_row["ema_number_updates"], summary_rows[0]["ema_number_updates"]) == ("4", "")
    assert summary_rows[1]["ema_Accuracy"] != "" and summary_rows[1]["ema_number_updates"] == "8"
    assert (resumed_trainer.experiment_folder / EVALUATION_CURVES_FIGURE_NAME).is_file()
    assert "Evaluation Summary Started From The TensorBoard Test Curves Of 1 Evaluated Iterations" in (
        capsys.readouterr().out)
