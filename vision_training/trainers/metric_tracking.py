"""
Metric tracking of the general trainer: the rolling averages of the training and validation losses and batch metrics
over every console interval, the console summary printed at every information dump (with the change of every value
since the previous summary) and the TensorBoard curves.
"""

import time
import torch

from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional

from vision_training.utilities.os_utilities import color_text

# Throughput unit printed for each kind of sample: single images for the image tasks, sequences for the sequence tasks,
# pairs of sequences for the sequence comparison task
THROUGHPUT_UNITS = {"Images": "imgs/sec", "Sequences": "seqs/sec", "Pairs": "pairs/sec"}

# Colour of the metric lines of each split in the console summary
SPLIT_COLORS = {"Train": "blue", "Valid": "yellow"}

# Suffix of the tracked keys that improve when they decrease (the losses), every other key (accuracy) improves when it
# increases
LOWER_IS_BETTER_SUFFIX = "_loss"

# Number of decimals of the metric values and of their changes, and width of a value or change column
METRIC_DECIMALS = 4
METRIC_COLUMN_WIDTH = 8

# Text of the marker of a value that is the best of the run for its split
BEST_MARKER = "Best"


class MetricTracker:
    """
    Tracks the metrics of the training and validation batches for the general trainer: rolling averages over every
    console log interval, the console summary printed at every information dump, and the TensorBoard curves written at
    every iteration. The tracked metrics are given by the task, as a mapping from model output keys (its losses and
    batch metrics, plus the total loss) to their display names.

    The tracker keeps the values of the previous summary, so that every summary shows the change of each value, green
    when it improved and red when it got worse, the best value of the run marked, and the Valid-Train gap of every
    metric. Nothing is kept across a resume, so the first summary of a resumed run has no change.
    """

    def __init__(self, tracked_metrics_mapping: Dict[str, str], information_dump: int, batch_size: int,
                 validation_iteration_ratio: int, sample_name: str, frames_per_sample: Optional[int]) -> None:
        """
        Keeps the settings of the console summaries.

        Args:
            tracked_metrics_mapping (Dict[str, str]): The display name of every tracked metric key.
            information_dump (int): The number of iterations of a console log interval.
            batch_size (int): The number of samples of a batch.
            validation_iteration_ratio (int): How often a validation batch is evaluated (0 for never).
            sample_name (str): "Images" for the image tasks, "Sequences" for the sequence tasks, "Pairs" for the
                sequence comparison task.
            frames_per_sample (Optional[int]): The number of frames of a sequence (or of a pair), to also print the
                frame throughput, or None for the image tasks.
        """
        self.tracked_metrics_mapping = tracked_metrics_mapping
        self.information_dump = information_dump
        self.batch_size = batch_size
        self.validation_iteration_ratio = validation_iteration_ratio
        self.sample_name = sample_name
        self.frames_per_sample = frames_per_sample

        # Values of the last summary that printed them, and best values of the run, for every split and metric key.
        # The Valid values are only updated by the summaries whose interval ran validation batches.
        self.previous_values = {split_name: {} for split_name in SPLIT_COLORS}
        self.best_values = {split_name: {} for split_name in SPLIT_COLORS}
        self.previous_gaps = {}
        self.previous_samples_per_second = None

    def create_trackers(self) -> Dict[str, Any]:
        """
        Initializes a dictionary accumulating the rolling averages of the tracked metrics over a console log interval.

        Returns:
            Dict[str, Any]: A dictionary with every metric key set to 0.0, and the start time of the interval.
        """
        tracker_dictionary = {"start_time": time.time()}
        for metric_key in self.tracked_metrics_mapping.keys():
            tracker_dictionary[metric_key] = 0.0

        return tracker_dictionary

    @staticmethod
    def accumulate(tracker_dictionary: Optional[Dict[str, Any]], metric_dictionary: Dict[str, torch.Tensor],
                   tracker_divisor: int) -> None:
        """
        Adds the metrics of one batch to the rolling averages of the interval: every value is divided by the number of
        batches contributing to the average, so that the accumulated values are the interval means.

        Args:
            tracker_dictionary (Optional[Dict[str, Any]]): The trackers of the interval, or None to track nothing.
            metric_dictionary (Dict[str, torch.Tensor]): The scalar metrics of the batch.
            tracker_divisor (int): The number of batches contributing to the average (0 to track nothing).
        """
        if tracker_dictionary is not None and tracker_divisor > 0:
            for metric_key, metric_value in metric_dictionary.items():
                if metric_key in tracker_dictionary:
                    tracker_dictionary[metric_key] += metric_value.item() / tracker_divisor

    def average_validation_trackers(self, validation_trackers: Dict[str, Any], validation_interval_count: int) -> None:
        """
        Turns the validation sums of an interval into means: validation batches are accumulated with a divisor of 1,
        since their number in the interval is only known at its end.

        Args:
            validation_trackers (Dict[str, Any]): The validation trackers of the interval, modified in place.
            validation_interval_count (int): The number of validation batches of the interval.
        """
        for metric_key in self.tracked_metrics_mapping.keys():
            validation_trackers[metric_key] /= validation_interval_count

    def log_to_tensorboard(self, writer: Optional[SummaryWriter], iteration: int,
                           metric_dictionary: Dict[str, torch.Tensor]) -> None:
        """
        Writes the tracked metrics of one batch to TensorBoard, under their display names.

        Args:
            writer (Optional[SummaryWriter]): The TensorBoard writer to use. If None, writing is skipped.
            iteration (int): The current iteration index.
            metric_dictionary (Dict[str, torch.Tensor]): The scalar metrics of the batch.
        """
        if writer is None:
            return

        for metric_key, display_name in self.tracked_metrics_mapping.items():
            writer.add_scalar(display_name, metric_dictionary[metric_key].item(), iteration)

    @staticmethod
    def format_time_interval(duration_seconds: float) -> str:
        """
        Formats a time duration in seconds into a human-readable string with hours, minutes, and seconds.

        Args:
            duration_seconds (float): The time duration in seconds.

        Returns:
            str: The formatted time string.
        """
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = duration_seconds % 60

        # Format the time into a consistent HH:MM:SS layout
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}"

    @staticmethod
    def is_lower_better(metric_key: str) -> bool:
        """
        Gives the direction in which a tracked metric improves, which sets the colour of its changes in the console
        summary: the losses improve when they decrease, the batch metrics (accuracy) when they increase.

        Args:
            metric_key (str): The model output key of the metric.

        Returns:
            bool: True for the losses, False for the other metrics.
        """
        return metric_key.endswith(LOWER_IS_BETTER_SUFFIX)

    @staticmethod
    def choose_change_color(change: float, lower_is_better: bool, neutral_color: Optional[str],
                            decimals: int) -> Optional[str]:
        """
        Chooses the colour of a change between two console summaries, so that the progress of the training reads at a
        glance: green when the change goes in the improving direction, red when it goes in the other one.

        Args:
            change (float): The current value minus the previous one.
            lower_is_better (bool): True when the value improves by decreasing (losses), False otherwise.
            neutral_color (Optional[str]): The colour of a change rounding to zero, or None for no colour.
            decimals (int): The number of printed decimals, below which a change rounds to zero.

        Returns:
            Optional[str]: "green", "red", or the neutral colour.
        """
        # A change rounding to zero at the printed decimals is neither an improvement nor a degradation
        if round(change, decimals) == 0:
            return neutral_color

        improved = change < 0 if lower_is_better else change > 0
        return "green" if improved else "red"

    @staticmethod
    def format_change(change: float, lower_is_better: bool, neutral_color: Optional[str], decimals: int,
                      width: int) -> str:
        """
        Formats the signed change of a console summary value since the previous summary, in the colour given by
        choose_change_color.

        Args:
            change (float): The current value minus the previous one.
            lower_is_better (bool): True when the value improves by decreasing (losses), False otherwise.
            neutral_color (Optional[str]): The colour of a change rounding to zero, or None for no colour.
            decimals (int): The number of printed decimals.
            width (int): The minimum width of the printed change, sign included.

        Returns:
            str: The coloured signed change.
        """
        change_color = MetricTracker.choose_change_color(change=change, lower_is_better=lower_is_better,
                                                         neutral_color=neutral_color, decimals=decimals)

        # Print a change rounding to zero as +0 rather than -0
        displayed_change = 0.0 if round(change, decimals) == 0 else change
        change_text = f"{displayed_change:>+{width}.{decimals}f}"
        return change_text if change_color is None else color_text(color=change_color, text=change_text)

    def format_metric_line(self, split_name: str, metric_key: str, display_name: str, current_value: float,
                           gap_value: Optional[float]) -> str:
        """
        Builds one metric line of the console summary, the main training feedback read at every information dump: the
        interval mean, its change since the last summary that printed this split, the best marker when it is the best
        value of the run, and on the Valid lines the Valid-Train gap. Keeps the values for the next summary.

        Args:
            split_name (str): "Train" or "Valid".
            metric_key (str): The model output key of the metric.
            display_name (str): The name of the metric printed in the line.
            current_value (float): The mean of the metric over the interval.
            gap_value (Optional[float]): The Valid value minus the Train value of the interval, or None on Train lines.

        Returns:
            str: The coloured line.
        """
        line_color = SPLIT_COLORS[split_name]
        lower_is_better = self.is_lower_better(metric_key=metric_key)
        line = color_text(color=line_color, text=f"Moving Average of {split_name} {display_name:30} : "
                                                 f"{current_value:>{METRIC_COLUMN_WIDTH}.{METRIC_DECIMALS}f}")

        # Add the change since the previous summary and the best marker, once there is a value to compare with
        is_best = False
        if metric_key in self.previous_values[split_name]:
            change = current_value - self.previous_values[split_name][metric_key]
            line += " " + self.format_change(change=change, lower_is_better=lower_is_better, neutral_color=line_color,
                                             decimals=METRIC_DECIMALS, width=METRIC_COLUMN_WIDTH)
            best_value = self.best_values[split_name][metric_key]
            is_best = current_value < best_value if lower_is_better else current_value > best_value

            # Keep an empty marker column before the gap, so that the gaps of every Valid line are aligned
            if is_best or gap_value is not None:
                line += " " + color_text(color=line_color, text=BEST_MARKER if is_best else " " * len(BEST_MARKER))

        # Keep the gap aligned on the first summary, which has neither change nor marker
        elif gap_value is not None:
            line += " " * (METRIC_COLUMN_WIDTH + len(BEST_MARKER) + 2)

        # Add the Valid-Train gap, red when it widened in the degrading direction since the previous summary
        if gap_value is not None:
            gap_color = line_color
            if metric_key in self.previous_gaps:
                gap_color = self.choose_change_color(change=gap_value - self.previous_gaps[metric_key],
                                                     lower_is_better=lower_is_better, neutral_color=line_color,
                                                     decimals=METRIC_DECIMALS)
            line += " " + color_text(color=gap_color, text=f"Valid-Train Gap {gap_value:+.{METRIC_DECIMALS}f}")
            self.previous_gaps[metric_key] = gap_value

        # Keep the value for the change of the next summary, and the best value of the run
        self.previous_values[split_name][metric_key] = current_value
        if is_best or metric_key not in self.best_values[split_name]:
            self.best_values[split_name][metric_key] = current_value

        return line

    def print_interval_summary(self, iterations: int, training_tracker_dictionary: Dict[str, Any],
                               validation_tracker_dictionary: Optional[Dict[str, Any]],
                               validation_interval_count: int) -> Dict[str, Any]:
        """
        Prints the rolling average of the training and validation metrics at the end of a console log interval.
        It is the primary training feedback of the pipeline, showing the throughput and every tracked metric with their
        changes since the previous summary, so that the user can monitor the convergence at a glance.

        Args:
            iterations (int): Current global training iteration.
            training_tracker_dictionary (Dict[str, Any]): Rolling average trackers for training.
            validation_tracker_dictionary (Optional[Dict[str, Any]]): Rolling average trackers for validation.
            validation_interval_count (int): The number of validation forward passes that occurred in this
                console-log interval. Used to inform the user and to gate validation metric display.

        Returns:
            Dict[str, Any]: A fresh training tracker dictionary for the next interval.
        """
        print_length = 100
        print("=" * print_length)

        # Calculate sample scale and throughput metrics
        total_samples = iterations * self.batch_size
        interval_samples = self.information_dump * self.batch_size
        duration = time.time() - training_tracker_dictionary["start_time"]
        samples_per_second = interval_samples / max(duration, 0.001)

        # Show the throughput changes since the previous summary, green when faster and red when slower (width 1 for no
        # padding)
        samples_change_text, frames_change_text = "", ""
        if self.previous_samples_per_second is not None:
            samples_change = samples_per_second - self.previous_samples_per_second
            samples_change_text = " (" + self.format_change(change=samples_change, lower_is_better=False,
                                                            neutral_color=None, decimals=1, width=1) + ")"
            if self.frames_per_sample is not None:
                frames_change_text = " (" + self.format_change(change=samples_change * self.frames_per_sample,
                                                               lower_is_better=False, neutral_color=None, decimals=1,
                                                               width=1) + ")"
        self.previous_samples_per_second = samples_per_second

        # Display the formatted summary header, with the frame throughput of the sequence tasks
        header = (f"| Iteration: {iterations:<6} | Total {self.sample_name}: {total_samples:<8} "
                  f"| Interval {self.sample_name}: {interval_samples:<6} "
                  f"| Throughput: {samples_per_second:.1f} {THROUGHPUT_UNITS[self.sample_name]}{samples_change_text} |")
        if self.frames_per_sample is not None:
            header += f" {samples_per_second * self.frames_per_sample:.1f} frames/sec{frames_change_text} |"
        print(header)
        print("-" * print_length)

        for metric_key, display_name in self.tracked_metrics_mapping.items():
            train_value = training_tracker_dictionary[metric_key]
            print(self.format_metric_line(split_name="Train", metric_key=metric_key, display_name=display_name,
                                          current_value=train_value, gap_value=None))

            # Only display validation metrics if at least one validation iteration ran in this interval, so that the
            # Valid changes compare with the last interval that ran validation
            if validation_tracker_dictionary is not None and validation_interval_count > 0:
                validation_value = validation_tracker_dictionary[metric_key]
                print(self.format_metric_line(split_name="Valid", metric_key=metric_key, display_name=display_name,
                                              current_value=validation_value, gap_value=validation_value - train_value))

        print("-" * print_length)
        formatted_duration = self.format_time_interval(duration_seconds=duration)
        print(f"Interval Of {self.information_dump} Iterations Took {formatted_duration}.")

        # Show how many validation iterations contributed to the reported averages
        if self.validation_iteration_ratio > 0:
            print(f"Validation Iterations In This Interval: {validation_interval_count} "
                  f"(Ratio: Every {self.validation_iteration_ratio} Training Iterations).")

        print("=" * print_length)

        return self.create_trackers()
