"""
Tests of MetricTracker of vision_training/trainers/metric_tracking.py: interval averages of the tracked metrics over
the batches really accumulated, the real frames counted for the frame throughput, the time interval format, the
interval summary header with the iterations of the interval, spanning its separator lines with centred cells whose
bars keep their place, and the changes since the previous summary (metric changes and their colours, best marker).
"""

import pytest
import torch

from vision_training.trainers.metric_tracking import (INTERVAL_BATCHES_KEY, SUMMARY_MAXIMUM_FRAMES_PER_SECOND,
                                                      SUMMARY_MAXIMUM_INTERVAL_SAMPLES, SUMMARY_MAXIMUM_ITERATIONS,
                                                      SUMMARY_MAXIMUM_SAMPLES_PER_SECOND,
                                                      SUMMARY_MAXIMUM_TOTAL_SAMPLES, SUMMARY_MINIMUM_WIDTH,
                                                      MetricTracker)
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
                         validation_iteration_ratio=2, sample_name=sample_name, counts_frames=counts_frames)


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

    # A batch padded to the sequence length and a batch cut to its longest window, holding 5 and 7 real frames
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


def test_time_interval_format() -> None:
    """
    Verifies the HH:MM:SS format of the interval durations.
    """
    assert MetricTracker.format_time_interval(duration_seconds=3725.4) == "01:02:05"


@pytest.mark.parametrize(argnames=("sample_name", "counts_frames", "expected_fragments"), argvalues=[
    ("Images", False, ["Total Images", "Images Per Second"]),
    ("Sequences", True, ["Total Sequences", "Sequences Per Second", "Frames Per Second"]),
    ("Pairs", True, ["Total Pairs", "Pairs Per Second", "Frames Per Second"])])
def test_interval_summary_header(capsys: pytest.CaptureFixture, sample_name: str, counts_frames: bool,
                                 expected_fragments: list) -> None:
    """
    Verifies that the console summary counts images for the image tasks, sequences (or pairs of sequences) with their
    frame throughput for the sequence tasks, that it counts the samples and iterations the interval really ran, that
    its header spans the separator lines from end to end, at least SUMMARY_MINIMUM_WIDTH characters, and that it
    returns fresh trackers for the next interval.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
        sample_name (str): "Images", "Sequences" or "Pairs".
        counts_frames (bool): Whether the real frames of the training batches are counted, False for images.
        expected_fragments (list): Text fragments that must appear in the summary.
    """
    tracker = build_metric_tracker(sample_name=sample_name, counts_frames=counts_frames)
    training_trackers = {"start_time": 0.0, "total_loss": 1.5, "accuracy": 0.5, INTERVAL_BATCHES_KEY: 3}
    if counts_frames:
        training_trackers["interval_frames"] = 64

    fresh_trackers = tracker.print_interval_summary(iterations=8, training_tracker_dictionary=training_trackers,
                                                    validation_tracker_dictionary=None)

    console_output = capsys.readouterr().out
    for expected_fragment in expected_fragments + [f"Interval {sample_name}: 6 ", "Interval Of 3 Iterations Took"]:
        assert expected_fragment in console_output

    # Header between an equal line and a dash line of its width, its first and last bars at their ends
    console_lines = console_output.splitlines()
    summary_width = len(console_lines[1])
    assert summary_width >= SUMMARY_MINIMUM_WIDTH and console_lines[1][0] == console_lines[1][-1] == "|"
    assert console_lines[0] == "=" * summary_width and console_lines[2] == "-" * summary_width
    assert console_lines[-1] == "=" * summary_width
    assert fresh_trackers["total_loss"] == 0.0 and fresh_trackers[INTERVAL_BATCHES_KEY] == 0


@pytest.mark.parametrize(argnames="counts_frames", argvalues=[False, True])
def test_summary_header_bars_keep_their_place(counts_frames: bool) -> None:
    """
    Verifies that the header cells are sized for the largest expected values, so that the header of the first
    interval and the one at those values have the same width and their bars at the same places, that the texts are
    centred in their cells, and that a value larger than expected widens its cell instead of being cut.

    Args:
        counts_frames (bool): Whether the frame cell of the sequence tasks is shown.
    """
    tracker = build_metric_tracker(sample_name="Sequences" if counts_frames else "Images", counts_frames=counts_frames)
    first_header = tracker.format_summary_header(iterations=45, total_samples=2880, interval_samples=320,
                                                 samples_per_second=42.9,
                                                 frames_per_second=1431.2 if counts_frames else None)
    largest_header = tracker.format_summary_header(
        iterations=SUMMARY_MAXIMUM_ITERATIONS, total_samples=SUMMARY_MAXIMUM_TOTAL_SAMPLES,
        interval_samples=SUMMARY_MAXIMUM_INTERVAL_SAMPLES, samples_per_second=SUMMARY_MAXIMUM_SAMPLES_PER_SECOND,
        frames_per_second=SUMMARY_MAXIMUM_FRAMES_PER_SECOND if counts_frames else None)

    bar_positions = [character_index for character_index, character in enumerate(first_header) if character == "|"]
    assert len(first_header) == len(largest_header) and len(bar_positions) == (6 if counts_frames else 5)
    assert bar_positions == [character_index for character_index, character in enumerate(largest_header)
                             if character == "|"]

    # Every text in the middle of its cell, at most one space more on one side
    for left_bar, right_bar in zip(bar_positions[:-1], bar_positions[1:]):
        cell = first_header[left_bar + 1:right_bar]
        assert abs((len(cell) - len(cell.lstrip())) - (len(cell) - len(cell.rstrip()))) <= 1

    # An iteration count beyond the expected ones widens the first cell by its extra digit, taken from the spare space
    # of the image header, while the wider sequence header has none and grows by one character
    longer_header = tracker.format_summary_header(
        iterations=SUMMARY_MAXIMUM_ITERATIONS * 10, total_samples=2880, interval_samples=320, samples_per_second=42.9,
        frames_per_second=1431.2 if counts_frames else None)
    assert f" Iteration: {SUMMARY_MAXIMUM_ITERATIONS * 10} " in longer_header
    assert len(longer_header) == len(first_header) + int(counts_frames)


def test_metric_changes_and_best_marker() -> None:
    """
    Verifies the change printed after every value: none on the first summary, green when the value improves (a loss
    decreasing, an accuracy increasing), red when it gets worse, in the line colour and as +0 when it rounds to zero,
    and the best marker only on a value improving on every previous summary.
    """
    tracker = build_metric_tracker(sample_name="Images", counts_frames=False)
    loss_values_and_fragments = [(1.0, None), (0.5, color_text(color="green", text=" -0.5000")),
                                 (0.75, color_text(color="red", text=" +0.2500")),
                                 (0.75000001, color_text(color="blue", text=" +0.0000"))]

    # Follow the training loss over four summaries, the best marker only appearing when it improves on the best value
    for summary_index, (loss_value, expected_change) in enumerate(loss_values_and_fragments):
        loss_line = tracker.format_metric_line(split_name="Train", metric_key="total_loss", display_name="Total Loss",
                                               current_value=loss_value)
        if expected_change is None:
            assert loss_line == color_text(color="blue", text=f"Moving Average of Train {'Total Loss':30} :   1.0000")
        else:
            assert expected_change in loss_line
        assert (color_text(color="blue", text="Best") in loss_line) == (summary_index == 1)

    # An accuracy improves when it increases, so its decrease is red
    tracker.format_metric_line(split_name="Train", metric_key="accuracy", display_name="Accuracy",
                               current_value=0.6)
    accuracy_line = tracker.format_metric_line(split_name="Train", metric_key="accuracy", display_name="Accuracy",
                                               current_value=0.5)
    assert color_text(color="red", text=" -0.1000") in accuracy_line


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
    assert color_text(color="green", text=" -0.2000") in console_output
