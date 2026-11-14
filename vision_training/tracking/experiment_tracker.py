"""
Interface of the experiment tracking of the training runs and of the standalone checkpoint evaluations, next to
TensorBoard and Results.txt: the ExperimentTracker methods called by the general trainer (trainers/base_trainer.py) and
by evaluate_checkpoints (evaluation/checkpoint_evaluation.py), the DisabledTracker doing nothing when the tracking is
off or unavailable, the record of the MLflow run kept in the run folder for the resumptions, and the factory building
the MLflow tracker of mlflow_tracker.py only when 'ExperimentConfiguration.tracking.activated' is True and the mlflow
package is installed, so that a missing or broken MLflow never stops a training.
"""

import importlib
import importlib.util
import json

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from vision_training.utilities.os_utilities import print_color

# File of the run folder recording the MLflow run of the training, so that a resumed training continues the same run
RUN_RECORD_FILE_NAME = "mlflow_run.json"

# Keys of the run record, every one a string
RUN_RECORD_KEYS = ["tracking_uri", "experiment_id", "run_id"]

# Local backend of the experiments root folder, used when no tracking server is set: the sqlite database and the
# folder of the artifacts
DATABASE_FILE_NAME = "mlflow.db"
ARTIFACTS_FOLDER_NAME = "mlartifacts"

# Module of the MLflow tracker, imported only when the tracking is on, since importing mlflow takes about a second
MLFLOW_TRACKER_MODULE = "vision_training.tracking.mlflow_tracker"

# Final status of a run: completed, stopped by the user (Ctrl+C), or stopped by an error
FINISHED_STATUS = "FINISHED"
KILLED_STATUS = "KILLED"
FAILED_STATUS = "FAILED"

# Prefix of the MLflow test metrics of the trained weights, whose best value is tagged as 'best_iteration'
TRAINED_WEIGHTS_METRIC_PREFIX = "test"


class ExperimentTracker(ABC):
    """
    Tracking of one run (a training, or a standalone evaluation of checkpoints) in an experiment tracking service,
    next to the TensorBoard curves and Results.txt. The trainer only calls these methods, which must return at once
    and never raise, whatever the state of the tracking service: the MLflow tracker queues the events for a background
    thread, and the disabled tracker drops them.
    """

    @abstractmethod
    def start(self, parameters: Dict[str, Any], tags: Dict[str, str], resumed_iteration: Optional[int]) -> None:
        """
        Starts the tracking of the run: the run is created, or reopened when a resumed training continues a tracked
        run, with its parameters and tags. Called once, before any other method.

        Args:
            parameters (Dict[str, Any]): The nested settings of the run (configuration sections and derived values).
            tags (Dict[str, str]): The tags identifying the run (task, run name, run folder, source runs...).
            resumed_iteration (Optional[int]): The checkpoint iteration a resumed training starts from, None for a run
                starting from scratch.
        """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Logs metric values at a training iteration (interval averages, learning rate, test metrics...).

        Args:
            metrics (Dict[str, float]): The value of every metric, keyed by its name.
            step (int): The training iteration of the values.
        """

    @abstractmethod
    def update_best_metric(self, metric_name: str, value: float, step: int, metric_prefix: str) -> None:
        """
        Follows the best value of the headline metric of the evaluated checkpoints (the higher the better), separately
        for the trained weights ('test') and the averaged weights ('test_ema'): logs the best value so far as
        '<metric prefix>/best_<metric name>' and tags its iteration as 'best_iteration' for 'test' and
        'best_iteration_<metric prefix>' otherwise.

        Args:
            metric_name (str): The name of the headline metric.
            value (float): The value of the evaluated checkpoint.
            step (int): The training iteration of the evaluated checkpoint.
            metric_prefix (str): The prefix of the evaluated weights, 'test' or 'test_ema'.
        """

    @abstractmethod
    def log_artifact(self, file_path: Path, artifact_folder: Optional[str],
                     artifact_name: Optional[str] = None) -> None:
        """
        Sends a file of the run as an artifact, a missing file being skipped.

        Args:
            file_path (Path): The file to send.
            artifact_folder (Optional[str]): The folder of the artifact in the run, None for its root.
            artifact_name (Optional[str]): The name of the artifact, None to keep the file name.
        """

    @abstractmethod
    def get_run_id(self) -> Optional[str]:
        """
        Gives the identifier of the tracked run, stored in the checkpoint metadata.

        Returns:
            Optional[str]: The run identifier, None while the run is not created yet or without tracking.
        """

    @abstractmethod
    def close(self, status: str) -> None:
        """
        Ends the tracking of the run with its final status, after sending the waiting events within a bounded time.

        Args:
            status (str): FINISHED_STATUS, KILLED_STATUS or FAILED_STATUS.
        """


class DisabledTracker(ExperimentTracker):
    """
    Tracker doing nothing, used when the tracking is off, when MLflow is not installed or could not start, and by the
    trainers that never train (the input viewer, the trainer of a standalone evaluation), so that the trainer code
    calls the tracker without any condition.
    """

    def start(self, parameters: Dict[str, Any], tags: Dict[str, str], resumed_iteration: Optional[int]) -> None:
        """
        Does nothing.

        Args:
            parameters (Dict[str, Any]): Ignored.
            tags (Dict[str, str]): Ignored.
            resumed_iteration (Optional[int]): Ignored.
        """

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Does nothing.

        Args:
            metrics (Dict[str, float]): Ignored.
            step (int): Ignored.
        """

    def update_best_metric(self, metric_name: str, value: float, step: int, metric_prefix: str) -> None:
        """
        Does nothing.

        Args:
            metric_name (str): Ignored.
            value (float): Ignored.
            step (int): Ignored.
            metric_prefix (str): Ignored.
        """

    def log_artifact(self, file_path: Path, artifact_folder: Optional[str],
                     artifact_name: Optional[str] = None) -> None:
        """
        Does nothing.

        Args:
            file_path (Path): Ignored.
            artifact_folder (Optional[str]): Ignored.
            artifact_name (Optional[str]): Ignored.
        """

    def get_run_id(self) -> Optional[str]:
        """
        Gives no run identifier, nothing being tracked.

        Returns:
            Optional[str]: None.
        """
        return None

    def close(self, status: str) -> None:
        """
        Does nothing.

        Args:
            status (str): Ignored.
        """


def resolve_tracking_uri(tracking_configuration: Dict[str, Any], tracking_root: Path) -> Tuple[str, Optional[str]]:
    """
    Gives the address of the tracking backend of the runs: the server of 'tracking.tracking_uri' when it is set,
    otherwise the local sqlite database of the experiments root folder, shared by every run trained in it, with its
    artifacts in the 'mlartifacts' folder next to it.

    Args:
        tracking_configuration (Dict[str, Any]): The 'ExperimentConfiguration.tracking' section.
        tracking_root (Path): The experiments root folder holding the local database.

    Returns:
        Tuple[str, Optional[str]]: The tracking URI, and the root URI of the artifacts of the local database (None
            for a server, which stores the artifacts itself).
    """
    if tracking_configuration["tracking_uri"] is not None:
        return tracking_configuration["tracking_uri"], None
    tracking_root = Path(tracking_root).absolute()
    return f"sqlite:///{tracking_root / DATABASE_FILE_NAME}", (tracking_root / ARTIFACTS_FOLDER_NAME).as_uri()


def read_run_record(run_record_path: Optional[Path]) -> Optional[Dict[str, str]]:
    """
    Reads the record of the MLflow run of a training from its run folder, to continue the same run on a resume and to
    tag the standalone evaluations with the run they evaluate.

    Args:
        run_record_path (Optional[Path]): The 'mlflow_run.json' file of the run folder, None for a run without record.

    Returns:
        Optional[Dict[str, str]]: The tracking URI, experiment and run identifiers, or None when the file is missing
            or invalid.
    """
    if run_record_path is None or not Path(run_record_path).is_file():
        return None
    try:
        run_record = json.loads(Path(run_record_path).read_text())
    except (OSError, ValueError):
        return None

    # A record missing a key or holding another type is not trusted
    if not isinstance(run_record, dict) or not all(isinstance(run_record.get(key), str) for key in RUN_RECORD_KEYS):
        return None
    return {key: run_record[key] for key in RUN_RECORD_KEYS}


def write_run_record(run_record_path: Path, run_record: Dict[str, str]) -> None:
    """
    Writes the record of the MLflow run of a training in its run folder, as soon as the run exists, so that a resumed
    training reopens it.

    Args:
        run_record_path (Path): The 'mlflow_run.json' file of the run folder.
        run_record (Dict[str, str]): The tracking URI, experiment and run identifiers.
    """
    Path(run_record_path).write_text(data=json.dumps(obj=run_record, indent=4))


def create_experiment_tracker(tracking_configuration: Dict[str, Any], tracking_root: Path, run_name: str, task: str,
                              run_record_path: Optional[Path]) -> ExperimentTracker:
    """
    Builds the tracker of a training run or of a standalone evaluation, the single entry point of the tracking used by
    the trainer: the MLflow tracker when 'tracking.activated' is True, a DisabledTracker otherwise. The mlflow package
    is only imported here, when the tracking is on and the package is installed, so that the runs without tracking do
    not pay its import time and a missing or broken install gives one warning instead of stopping the training.

    Args:
        tracking_configuration (Dict[str, Any]): The 'ExperimentConfiguration.tracking' section.
        tracking_root (Path): The experiments root folder, holding the local database when no server is set.
        run_name (str): The name of the MLflow run, the name of the run folder.
        task (str): The task of the run, naming the MLflow experiment when 'tracking.experiment_name' is null.
        run_record_path (Optional[Path]): The 'mlflow_run.json' file of the run folder, None for a run that is never
            continued (a standalone evaluation).

    Returns:
        ExperimentTracker: The tracker of the run.
    """
    if not tracking_configuration["activated"]:
        return DisabledTracker()

    # Tracking asked without the package: one warning, and the training runs without tracking
    if importlib.util.find_spec(name="mlflow") is None:
        print_color(color="yellow", output="MLflow Tracking Is Activated But The 'mlflow' Package Is Not Installed, "
                                           "The Run Is Not Tracked.")
        return DisabledTracker()

    try:
        # The MLflow experiment groups the runs of the task unless another name is set
        tracking_uri, artifact_root = resolve_tracking_uri(tracking_configuration=tracking_configuration,
                                                           tracking_root=tracking_root)
        experiment_name = tracking_configuration["experiment_name"]
        if experiment_name is None:
            experiment_name = task

        mlflow_tracker_module = importlib.import_module(name=MLFLOW_TRACKER_MODULE)
        return mlflow_tracker_module.MlflowTracker(tracking_uri=tracking_uri, artifact_root=artifact_root,
                                                   experiment_name=experiment_name, run_name=run_name,
                                                   run_record_path=run_record_path)
    except Exception as error:
        print_color(color="yellow", output=f"MLflow Tracking Could Not Start, The Run Is Not Tracked : {error!r}")
        return DisabledTracker()
