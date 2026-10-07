"""
Tests of MetricTracker of vision_training/trainers/metric_tracking.py: interval averages of the tracked metrics,
validation sums divided by their count, the time interval format, the interval summary header, and the changes since
the previous summary (metric changes and their colours, best marker).
"""

import pytest
import torch

from vision_training.trainers.metric_tracking import MetricTracker
from vision_training.utilities.os_utilities import color_text


def build_metric_tracker(sample_name: str, frames_per_sample: int | None) -> MetricTracker:
    """
    Builds a metric tracker following the total loss (improving when it decreases) and the accuracy (improving when it
    increases), over intervals of 4 iterations of batches of 2 samples.

    Args:
        sample_name (str): "Images" or "Sequences".
        frames_per_sample (int | None): The number of frames of a sequence, or None for images.

    Returns:
        MetricTracker: The tracker.
    """
    return MetricTracker(tracked_metrics_mapping={"total_loss": "Total Loss", "accuracy": "Accuracy"},
                         information_dump=4, batch_size=2, validation_iteration_ratio=2, sample_name=sample_name,
                         frames_per_sample=frames_per_sample)


def test_interval_average_of_the_tracked_metrics() -> None:
    """
    Verifies that the tracked metrics of the batches of an interval add up to their mean, that untracked outputs are
    ignored, and that nothing is tracked without trackers or divisor.
    """
    tracker = build_metric_tracker(sample_name="Images", frames_per_sample=None)
    trackers = tracker.create_trackers()

    for loss_value in [1.0, 2.0, 3.0, 6.0]:
        MetricTracker.accumulate(tracker_dictionary=trackers,
                                 metric_dictionary={"total_loss": torch.tensor(data=loss_value),
                                                    "untracked_loss": torch.tensor(data=5.0)}, tracker_divisor=4)
    assert trackers["total_loss"] == pytest.approx(expected=3.0)
    assert "untracked_loss" not in trackers

    # Nothing is tracked without divisor, and missing trackers are accepted
    MetricTracker.accumulate(tracker_dictionary=trackers, metric_dictionary={"total_loss": torch.tensor(data=9.0)},
                             tracker_divisor=0)
    MetricTracker.accumulate(tracker_dictionary=None, metric_dictionary={"total_loss": torch.tensor(data=9.0)},
                             tracker_divisor=4)
    assert trackers["total_loss"] == pytest.approx(expected=3.0)


def test_validation_sums_are_divided_by_their_count() -> None:
    """
    Verifies that the validation sums of an interval become means once divided by the number of validation batches.
    """
    tracker = build_metric_tracker(sample_name="Images", frames_per_sample=None)
    validation_trackers = {"start_time": 0.0, "total_loss": 9.0, "accuracy": 1.5}
    tracker.average_validation_trackers(validation_trackers=validation_trackers, validation_interval_count=3)
    assert validation_trackers["total_loss"] == pytest.approx(expected=3.0)
    assert validation_trackers["accuracy"] == pytest.approx(expected=0.5)


def test_time_interval_format() -> None:
    """
    Verifies the HH:MM:SS format of the interval durations.
    """
    assert MetricTracker.format_time_interval(duration_seconds=3725.4) == "01:02:05"


@pytest.mark.parametrize(argnames=("sample_name", "frames_per_sample", "expected_fragments"), argvalues=[
    ("Images", None, ["Total Images", "imgs/sec"]),
    ("Sequences", 16, ["Total Sequences", "seqs/sec", "frames/sec"]),
    ("Pairs", 8, ["Total Pairs", "pairs/sec", "frames/sec"])])
def test_interval_summary_header(capsys: pytest.CaptureFixture, sample_name: str, frames_per_sample: int | None,
                                 expected_fragments: list) -> None:
    """
    Verifies that the console summary counts images for the image tasks, sequences (or pairs of sequences) with their
    frame throughput for the sequence tasks, and that it returns fresh trackers for the next interval.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
        sample_name (str): "Images", "Sequences" or "Pairs".
        frames_per_sample (int | None): The number of frames of a sequence (or of a pair), or None for images.
        expected_fragments (list): Text fragments that must appear in the summary.
    """
    tracker = build_metric_tracker(sample_name=sample_name, frames_per_sample=frames_per_sample)
    training_trackers = {"start_time": 0.0, "total_loss": 1.5, "accuracy": 0.5}

    fresh_trackers = tracker.print_interval_summary(iterations=8, training_tracker_dictionary=training_trackers,
                                                    validation_tracker_dictionary=None, validation_interval_count=0)

    console_output = capsys.readouterr().out
    for expected_fragment in expected_fragments:
        assert expected_fragment in console_output
    assert fresh_trackers["total_loss"] == 0.0


def test_metric_changes_and_best_marker() -> None:
    """
    Verifies the change printed after every value: none on the first summary, green when the value improves (a loss
    decreasing, an accuracy increasing), red when it gets worse, in the line colour and as +0 when it rounds to zero,
    and the best marker only on a value improving on every previous summary.
    """
    tracker = build_metric_tracker(sample_name="Images", frames_per_sample=None)
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
    tracker = build_metric_tracker(sample_name="Images", frames_per_sample=None)
    validation_values_and_counts = [({"start_time": 0.0, "total_loss": 1.0, "accuracy": 0.5}, 2),
                                    ({"start_time": 0.0, "total_loss": 0.0, "accuracy": 0.0}, 0),
                                    ({"start_time": 0.0, "total_loss": 0.8, "accuracy": 0.5}, 2)]

    # Print three summaries, the second one without validation batches
    for validation_trackers, validation_interval_count in validation_values_and_counts:
        tracker.print_interval_summary(iterations=8,
                                       training_tracker_dictionary={"start_time": 0.0, "total_loss": 1.0,
                                                                    "accuracy": 0.5},
                                       validation_tracker_dictionary=validation_trackers,
                                       validation_interval_count=validation_interval_count)

    console_output = capsys.readouterr().out
    assert console_output.count("Moving Average of Valid Total Loss") == 2
    assert color_text(color="green", text=" -0.2000") in console_output
