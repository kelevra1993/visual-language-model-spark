import pytest
import torch

from vision_training.trainers.metric_tracking import MetricTracker


def build_metric_tracker(sample_name: str, frames_per_sample: int | None) -> MetricTracker:
    """
    Builds a metric tracker following the total loss, over intervals of 4 iterations of batches of 2 samples.

    Args:
        sample_name (str): "Images" or "Sequences".
        frames_per_sample (int | None): The number of frames of a sequence, or None for images.

    Returns:
        MetricTracker: The tracker.
    """
    return MetricTracker(tracked_metrics_mapping={"total_loss": "Total Loss"}, information_dump=4, batch_size=2,
                         validation_iteration_ratio=2, sample_name=sample_name, frames_per_sample=frames_per_sample)


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
    validation_trackers = {"start_time": 0.0, "total_loss": 9.0}
    tracker.average_validation_trackers(validation_trackers=validation_trackers, validation_interval_count=3)
    assert validation_trackers["total_loss"] == pytest.approx(expected=3.0)


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
    training_trackers = {"start_time": 0.0, "total_loss": 1.5}

    fresh_trackers = tracker.print_interval_summary(iterations=8, training_tracker_dictionary=training_trackers,
                                                    validation_tracker_dictionary=None, validation_interval_count=0)

    console_output = capsys.readouterr().out
    for expected_fragment in expected_fragments:
        assert expected_fragment in console_output
    assert fresh_trackers["total_loss"] == 0.0
