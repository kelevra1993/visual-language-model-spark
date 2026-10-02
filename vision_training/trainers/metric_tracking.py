import time
import torch

from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional

from vision_training.utilities.os_utilities import print_color

# Throughput unit printed for each kind of sample: single images for the image tasks, sequences for the sequence tasks
THROUGHPUT_UNITS = {"Images": "imgs/sec", "Sequences": "seqs/sec"}


class MetricTracker:
    """
    Tracks the metrics of the training and validation batches for the general trainer: rolling averages over every
    console log interval, the console summary printed at every information dump, and the TensorBoard curves written at
    every iteration. The tracked metrics are given by the task, as a mapping from model output keys (its losses and
    batch metrics, plus the total loss) to their display names.
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
            sample_name (str): "Images" for the image tasks, "Sequences" for the sequence tasks.
            frames_per_sample (Optional[int]): The number of frames of a sequence, to also print the frame throughput,
                or None for the image tasks.
        """
        self.tracked_metrics_mapping = tracked_metrics_mapping
        self.information_dump = information_dump
        self.batch_size = batch_size
        self.validation_iteration_ratio = validation_iteration_ratio
        self.sample_name = sample_name
        self.frames_per_sample = frames_per_sample

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

    def print_interval_summary(self, iterations: int, training_tracker_dictionary: Dict[str, Any],
                               validation_tracker_dictionary: Optional[Dict[str, Any]],
                               validation_interval_count: int) -> Dict[str, Any]:
        """
        Prints the rolling average of the training and validation metrics at the end of a console log interval.
        It is the primary training feedback of the pipeline, showing the throughput and every tracked metric so that
        the user can monitor the convergence.

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

        # Display the formatted summary header, with the frame throughput of the sequence tasks
        header = (f"| Iteration: {iterations:<6} | Total {self.sample_name}: {total_samples:<8} "
                  f"| Interval {self.sample_name}: {interval_samples:<6} "
                  f"| Throughput: {samples_per_second:.1f} {THROUGHPUT_UNITS[self.sample_name]} |")
        if self.frames_per_sample is not None:
            header += f" {samples_per_second * self.frames_per_sample:.1f} frames/sec |"
        print(header)
        print("-" * print_length)

        for metric_key, display_name in self.tracked_metrics_mapping.items():
            train_value = training_tracker_dictionary[metric_key]
            message = f"Moving Average of Train {display_name:30} : {train_value:.4f}"
            print_color(color="blue", output=message)

            # Only display validation metrics if at least one validation iteration ran in this interval
            if validation_tracker_dictionary is not None and validation_interval_count > 0:
                validation_value = validation_tracker_dictionary[metric_key]
                validation_message = f"Moving Average of Valid {display_name:30} : {validation_value:.4f}"
                print_color(color="yellow", output=validation_message)

        print("-" * print_length)
        formatted_duration = self.format_time_interval(duration_seconds=duration)
        print(f"Interval Of {self.information_dump} Iterations Took {formatted_duration}.")

        # Show how many validation iterations contributed to the reported averages
        if self.validation_iteration_ratio > 0:
            print(f"Validation Iterations In This Interval: {validation_interval_count} "
                  f"(Ratio: Every {self.validation_iteration_ratio} Training Iterations).")

        print("=" * print_length)

        return self.create_trackers()
