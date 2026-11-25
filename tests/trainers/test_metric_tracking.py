"""
Tests of MetricTracker of vision_training/trainers/metric_tracking.py: interval averages of the tracked metrics over the
batches really accumulated, the real frames counted for the frame throughput, the validation changes of the interval
summary compared with the last interval that ran validation, the off-diagonal correlations improving when they fall,
the rates of the task printed as percentages with their changes in percent, the values, changes and best markers of a
summary aligned in columns, and the counts and speeds of the header and interval lines with their digits grouped by
three within separator lines as wide as the header.
"""

import re

import pytest
import torch

from vision_training.trainers.metric_tracking import INTERVAL_BATCHES_KEY, MetricTracker
from vision_training.utilities.os_utilities import color_text


def build_metric_tracker(sample_name: str, counts_frames: bool) -> MetricTracker:
    """
    Builds a metric tracker following the total loss (improving when it decreases) and the accuracy (improving when it
    increases), for batches of 2 samples and a validation batch every 2 iterations.

    Args:
        sample_name (str): "Images", "Sequences" or "Pairs".
        counts_frames (bool): Whether the real frames of the training batches are counted, False for images.

    Returns:
        MetricTracker: The tracker.
    """
    return MetricTracker(tracked_metrics_mapping={"total_loss": "Total Loss", "accuracy": "Accuracy"}, batch_size=2,
                         validation_iteration_ratio=2, sample_name=sample_name, counts_frames=counts_frames,
                         percent_metric_keys=["accuracy"])


def test_interval_average_over_the_accumulated_batches() -> None:
    """
    Verifies that the tracked metrics of an interval are summed with their number of batches, then averaged over the
    batches really accumulated (3 here, as in the short first interval of a resumed run), that untracked outputs are
    ignored, that nothing is tracked without trackers, and that an interval without batches keeps its zeros.
    """
    tracker = build_metric_tracker(sample_name="Images", counts_frames=False)
    trackers = tracker.create_trackers()

    for loss_value, accuracy_value in [(1.0, 0.5), (2.0, 0.25), (6.0, 0.75)]:
        MetricTracker.accumulate(tracker_dictionary=trackers,
                                 metric_dictionary={"total_loss": torch.tensor(data=loss_value),
                                                    "accuracy": torch.tensor(data=accuracy_value),
                                                    "untracked_loss": torch.tensor(data=5.0)})
    MetricTracker.accumulate(tracker_dictionary=None, metric_dictionary={"total_loss": torch.tensor(data=9.0)})
    assert trackers[INTERVAL_BATCHES_KEY] == 3 and trackers["total_loss"] == pytest.approx(expected=9.0)
    assert "untracked_loss" not in trackers

    tracker.average_trackers(tracker_dictionary=trackers)
    assert trackers["total_loss"] == pytest.approx(expected=3.0)
    assert trackers["accuracy"] == pytest.approx(expected=0.5)

    # An interval without batches, such as a validation never run, keeps its zeros
    empty_trackers = tracker.create_trackers()
    tracker.average_trackers(tracker_dictionary=empty_trackers)
    assert empty_trackers["total_loss"] == 0.0 and empty_trackers[INTERVAL_BATCHES_KEY] == 0


def test_frame_throughput_counts_the_real_frames() -> None:
    """
    Verifies that the sequence tasks count the real frames of their training batches from the sequence mask, whatever
    the number of padding frames, and divide them by the duration of the interval, while the image tasks count no
    frames and have no frame throughput.
    """
    tracker = build_metric_tracker(sample_name="Sequences", counts_frames=True)
    trackers = tracker.create_trackers()

    # Two batches padded to their longest sequence, holding 5 and 7 real frames
    padded_mask = torch.tensor(data=[[True, True, True, False, False, False], [True, True, False, False, False, False]])
    cut_mask = torch.tensor(data=[[True, True, True, True], [True, True, True, False]])
    for sequence_mask in [padded_mask, cut_mask]:
        tracker.count_frames(tracker_dictionary=trackers, batch={"sequence_mask": sequence_mask})
    assert trackers["interval_frames"] == 12
    assert tracker.compute_frame_throughput(training_tracker_dictionary=trackers, duration=4.0) == pytest.approx(
        expected=3.0)
    assert tracker.create_trackers()["interval_frames"] == 0

    # The image tasks neither count frames nor report a frame throughput
    image_tracker = build_metric_tracker(sample_name="Images", counts_frames=False)
    image_trackers = image_tracker.create_trackers()
    image_tracker.count_frames(tracker_dictionary=image_trackers, batch={"images": torch.zeros(size=(2, 3, 8, 8))})
    assert "interval_frames" not in image_trackers
    assert image_tracker.compute_frame_throughput(training_tracker_dictionary=image_trackers, duration=4.0) is None


def test_valid_changes_skip_intervals_without_validation(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that an interval running no validation batch prints no Valid line, and that the next Valid lines compare
    with the last interval that ran validation.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    tracker = build_metric_tracker(sample_name="Images", counts_frames=False)
    validation_trackers_of_summaries = [
        {"start_time": 0.0, "total_loss": 1.0, "accuracy": 0.5, INTERVAL_BATCHES_KEY: 2},
        {"start_time": 0.0, "total_loss": 0.0, "accuracy": 0.0, INTERVAL_BATCHES_KEY: 0},
        {"start_time": 0.0, "total_loss": 0.8, "accuracy": 0.5, INTERVAL_BATCHES_KEY: 2}]

    # Print three summaries, the second one without validation batches
    for validation_trackers in validation_trackers_of_summaries:
        tracker.print_interval_summary(iterations=8,
                                       training_tracker_dictionary={"start_time": 0.0, "total_loss": 1.0,
                                                                    "accuracy": 0.5, INTERVAL_BATCHES_KEY: 4},
                                       validation_tracker_dictionary=validation_trackers)

    console_output = capsys.readouterr().out
    assert console_output.count("Moving Average of Valid Total Loss") == 2
    assert color_text(color="green", text="   -0.2000") in console_output


@pytest.mark.parametrize(argnames="metric_key",
                         argvalues=["cross_correlation_off_diagonal", "hidden_feature_correlation"])
def test_correlations_improve_when_they_decrease(metric_key: str) -> None:
    """
    Verifies that the mean absolute off-diagonal correlations (Barlow Twins, decorrelation), which do not end in _loss,
    improve when they fall: a falling value is green and marked best, a rising one red without the best marker, while
    the diagonal correlation of Barlow Twins still improves when it rises.

    Args:
        metric_key (str): The off-diagonal correlation metric tracked, of Barlow Twins or of the decorrelation.
    """
    tracker = MetricTracker(tracked_metrics_mapping={metric_key: "Correlation"}, batch_size=2,
                            validation_iteration_ratio=0, sample_name="Images", counts_frames=False,
                            percent_metric_keys=[])
    assert MetricTracker.is_lower_better(metric_key=metric_key)
    assert not MetricTracker.is_lower_better(metric_key="cross_correlation_diagonal")

    # A falling correlation is an improvement and the best value, a rising one a degradation
    tracker.format_metric_line(split_name="Train", metric_key=metric_key, display_name="Correlation", current_value=0.2)
    falling_line = tracker.format_metric_line(split_name="Train", metric_key=metric_key, display_name="Correlation",
                                              current_value=0.1)
    rising_line = tracker.format_metric_line(split_name="Train", metric_key=metric_key, display_name="Correlation",
                                             current_value=0.3)
    assert color_text(color="green", text="   -0.1000") in falling_line
    assert color_text(color="blue", text="Best") in falling_line
    assert color_text(color="red", text="   +0.2000") in rising_line
    assert color_text(color="blue", text="Best") not in rising_line


def test_rates_are_printed_as_percentages_with_changes_in_percent() -> None:
    """
    Verifies the lines of a rate listed in the percent keys (the accuracy of the classification-like tasks): its value
    written "50.00 %", its change in percentage points written "+10.00 %", green when it rises, neutral "+0.00 %" when
    it rounds to zero, while the loss keeps 4 decimals; a percent key that is not tracked is refused.
    """
    tracker = build_metric_tracker(sample_name="Images", counts_frames=False)
    first_line = tracker.format_metric_line(split_name="Train", metric_key="accuracy", display_name="Accuracy",
                                            current_value=0.5)
    rising_line = tracker.format_metric_line(split_name="Train", metric_key="accuracy", display_name="Accuracy",
                                             current_value=0.6)
    still_line = tracker.format_metric_line(split_name="Train", metric_key="accuracy", display_name="Accuracy",
                                            current_value=0.6000001)
    loss_line = tracker.format_metric_line(split_name="Train", metric_key="total_loss", display_name="Total Loss",
                                           current_value=0.25)

    # Value in percent, its change in percent coloured by its direction, the loss with 4 decimals
    assert first_line == color_text(color="blue", text=f"Moving Average of Train {'Accuracy':30} :    50.00 %")
    assert color_text(color="green", text="  +10.00 %") in rising_line
    assert color_text(color="blue", text="Best") in rising_line
    assert color_text(color="blue", text="   +0.00 %") in still_line
    assert "  0.2500" in loss_line and "%" not in loss_line and "points" not in rising_line

    # A percent key must be one of the tracked keys
    with pytest.raises(ValueError, match="percent keys"):
        MetricTracker(tracked_metrics_mapping={"total_loss": "Total Loss"}, batch_size=2, validation_iteration_ratio=0,
                      sample_name="Images", counts_frames=False, percent_metric_keys=["accuracy"])


def test_values_changes_and_best_markers_are_aligned_in_columns() -> None:
    """
    Verifies the columns of the console summary of a task printing rates: the decimal points of the 4-decimal values
    (up to the thousands of a Barlow Twins loss) and of the percentages fall in one column, so do the ones of the
    4-decimal changes and of the changes in percent of one or two digits, and every best marker starts in one column,
    one space after its change, while a line without a best marker ends with its change.
    """
    tracker = MetricTracker(tracked_metrics_mapping={"total_loss": "Total Loss", "accuracy": "Accuracy",
                                                     "top_five_accuracy": "Top-5 Accuracy",
                                                     "barlow_twins_loss": "Barlow Twins Loss"},
                            batch_size=2, validation_iteration_ratio=0, sample_name="Images", counts_frames=False,
                            percent_metric_keys=["accuracy", "top_five_accuracy"])

    # Two summaries where every value improves, the second lines holding a change and a best marker
    for values in [{"total_loss": 2.4567, "accuracy": 0.0412, "top_five_accuracy": 0.80,
                    "barlow_twins_loss": 2345.6789},
                   {"total_loss": 2.3456, "accuracy": 0.0512, "top_five_accuracy": 0.9403,
                    "barlow_twins_loss": 1234.5678}]:
        lines = [re.sub(pattern=r"\x1b\[[0-9;]*m", repl="", string=tracker.format_metric_line(
            split_name="Train", metric_key=metric_key, display_name=display_name, current_value=values[metric_key]))
            for metric_key, display_name in tracker.tracked_metrics_mapping.items()]
    assert lines[0].endswith(":     2.3456    -0.1111 Best")
    assert lines[1].endswith(":     5.12 %    +1.00 % Best")
    assert lines[2].endswith(":    94.03 %   +14.03 % Best")
    assert lines[3].endswith(":  1234.5678 -1111.1111 Best")

    # One column for the decimal points of the values, one for those of the changes, one for the best markers
    value_columns = {line.index(".") for line in lines}
    change_columns = {line.index(".", line.index(".") + 1) for line in lines}
    assert len(value_columns) == 1 and len(change_columns) == 1
    assert len({line.index("Best") for line in lines}) == 1

    # A change that is not the best value ends its line
    worse_line = tracker.format_metric_line(split_name="Train", metric_key="total_loss", display_name="Total Loss",
                                            current_value=2.5)
    assert worse_line.endswith(color_text(color="red", text="   +0.1544"))


def test_summary_counts_group_their_digits_within_the_separator_width(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the header and interval lines of the console summary write their counts and speeds with the digits
    grouped by three ("6 656 000"), that the header of the image tasks stays 120 characters wide with its bars in the
    same place from the first interval to the largest expected values, that the frame throughput of the sequence tasks
    is grouped too, and that the separator lines printed around the header are exactly as wide as it.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    tracker = MetricTracker(tracked_metrics_mapping={"total_loss": "Total Loss"}, batch_size=256,
                            validation_iteration_ratio=4, sample_name="Images", counts_frames=False,
                            percent_metric_keys=[])

    # Header of a long run, of a first interval and at the largest expected values, all 120 characters wide
    header = tracker.format_summary_header(iterations=26000, total_samples=6656000, interval_samples=256000,
                                           samples_per_second=114.6, frames_per_second=None)
    assert header == ("|  Iteration: 26 000  |  Total Images: 6 656 000   |  Interval Images: 256 000 |  Throughput: "
                      "114.6 Images Per Second  |")
    first_header = tracker.format_summary_header(iterations=4, total_samples=32, interval_samples=32,
                                                 samples_per_second=8.6, frames_per_second=None)
    largest_header = tracker.format_summary_header(iterations=999_999, total_samples=999_999_999,
                                                   interval_samples=999_999, samples_per_second=9999.9,
                                                   frames_per_second=None)
    assert "Throughput: 9 999.9 Images Per Second" in largest_header
    bar_columns = [[column for column, character in enumerate(line) if character == "|"]
                   for line in [header, first_header, largest_header]]
    assert len(header) == 120 and bar_columns[0] == bar_columns[1] == bar_columns[2]

    # Interval lines with grouped counts
    assert tracker.format_interval_duration_line(interval_iterations=1000, duration_seconds=1007.0) == \
        "Interval Of 1 000 Iterations Took 00:16:47."
    assert tracker.format_validation_count_line(validation_interval_count=1250) == \
        "Validation Iterations In This Interval: 1 250 (Ratio: Every 4 Training Iterations)."

    # Frame throughput of the sequence tasks grouped as well
    sequence_tracker = build_metric_tracker(sample_name="Sequences", counts_frames=True)
    sequence_header = sequence_tracker.format_summary_header(iterations=26000, total_samples=52000,
                                                             interval_samples=2000, samples_per_second=1234.5,
                                                             frames_per_second=5730.0)
    assert [cell.strip() for cell in sequence_header.split("|")[1:-1]] == [
        "Iteration: 26 000", "Total Sequences: 52 000", "Interval Sequences: 2 000",
        "Throughput: 1 234.5 Sequences Per Second", "5 730.0 Frames Per Second"]

    # Printed summary: separator lines as wide as the header, interval lines grouped
    tracker.print_interval_summary(iterations=26000,
                                   training_tracker_dictionary={"start_time": 0.0, "total_loss": 1.0,
                                                                INTERVAL_BATCHES_KEY: 1000},
                                   validation_tracker_dictionary={"start_time": 0.0, "total_loss": 1.0,
                                                                  INTERVAL_BATCHES_KEY: 250})
    console_lines = capsys.readouterr().out.splitlines()
    header_line = next(line for line in console_lines if line.startswith("|"))
    assert "Iteration: 26 000" in header_line and "Total Images: 6 656 000" in header_line
    assert "Interval Images: 256 000" in header_line
    assert console_lines[0] == console_lines[-1] == "=" * len(header_line) and len(header_line) == 120
    assert console_lines.count("-" * len(header_line)) == 2
    assert any(line.startswith("Interval Of 1 000 Iterations Took ") for line in console_lines)
    assert "Validation Iterations In This Interval: 250 (Ratio: Every 4 Training Iterations)." in console_lines
