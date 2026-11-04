"""
Tests of MetricTracker of vision_training/trainers/metric_tracking.py: interval averages of the tracked metrics over the
batches really accumulated, the real frames counted for the frame throughput, the validation changes of the interval
summary compared with the last interval that ran validation, and the off-diagonal correlations improving when they fall.
"""

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
    assert color_text(color="green", text=" -0.2000") in console_output


@pytest.mark.parametrize("metric_key", ["cross_correlation_off_diagonal", "hidden_feature_correlation"])
def test_correlations_improve_when_they_decrease(metric_key: str) -> None:
    """
    Verifies that the mean absolute off-diagonal correlations (Barlow Twins, decorrelation), which do not end in _loss,
    improve when they fall: a falling value is green and marked best, a rising one red without the best marker, while
    the diagonal correlation of Barlow Twins still improves when it rises.
    """
    tracker = MetricTracker(tracked_metrics_mapping={metric_key: "Correlation"}, batch_size=2,
                            validation_iteration_ratio=0, sample_name="Images", counts_frames=False)
    assert MetricTracker.is_lower_better(metric_key=metric_key)
    assert not MetricTracker.is_lower_better(metric_key="cross_correlation_diagonal")

    # A falling correlation is an improvement and the best value, a rising one a degradation
    tracker.format_metric_line(split_name="Train", metric_key=metric_key, display_name="Correlation", current_value=0.2)
    falling_line = tracker.format_metric_line(split_name="Train", metric_key=metric_key, display_name="Correlation",
                                              current_value=0.1)
    rising_line = tracker.format_metric_line(split_name="Train", metric_key=metric_key, display_name="Correlation",
                                             current_value=0.3)
    assert color_text(color="green", text=" -0.1000") in falling_line
    assert color_text(color="blue", text="Best") in falling_line
    assert color_text(color="red", text=" +0.2000") in rising_line
    assert color_text(color="blue", text="Best") not in rising_line
