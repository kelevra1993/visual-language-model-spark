"""
MLflow tracker of the training runs and of the standalone checkpoint evaluations, loaded by create_experiment_tracker
(experiment_tracker.py) only when 'ExperimentConfiguration.tracking.activated' is True. The calls of the trainer only
put events in a bounded queue and return at once; a daemon worker thread owning the MLflow client creates or reopens
the MLflow run and writes the parameters, tags, metrics and artifacts in batches, retrying with a growing delay while
MLflow is unavailable, so that the training is never blocked nor stopped by MLflow. Also holds the conversion of the
metric names and of the configuration to the names and values accepted by MLflow.
"""

import functools
import json
import logging
import math
import os
import queue
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unicodedata

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mlflow.entities import Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.utils.validation import (MAX_ENTITY_KEY_LENGTH, MAX_METRICS_PER_BATCH, MAX_PARAM_VAL_LENGTH,
                                     MAX_PARAMS_TAGS_PER_BATCH, MAX_TAG_VAL_LENGTH)

from vision_training.tracking.experiment_tracker import (FINISHED_STATUS, ExperimentTracker, read_run_record,
                                                         write_run_record)
from vision_training.utilities.os_utilities import print_color

# Bound of the events waiting for the worker (a few MB), the newer events being dropped and counted beyond it
MAXIMUM_QUEUE_SIZE = 100000

# The worker writes the waiting metrics every 2 seconds, or as soon as 500 metric values wait, and checks the queue
# every 0.1 second so that a close is never kept waiting
FLUSH_INTERVAL_SECONDS = 2.0
FLUSH_METRIC_COUNT = 500
QUEUE_POLL_SECONDS = 0.1

# Delay before writing again after a failed write, doubled after every failure up to 5 minutes
INITIAL_RETRY_DELAY_SECONDS = 5.0
MAXIMUM_RETRY_DELAY_SECONDS = 300.0

# Longest wait of a close for the worker to send the last events, the process exiting anyway
CLOSE_TIMEOUT_SECONDS = 10.0

# Timeout of the requests to a tracking server, which MLflow otherwise retries for minutes, and of the git commands
# run by start() in the training thread
HTTP_REQUEST_TIMEOUT_SECONDS = 10
GIT_TIMEOUT_SECONDS = 5

# Configuration dictionaries with more entries than this (the label dictionary of many classes) are logged as one
# parameter instead of one parameter per entry
MAXIMUM_FLATTENED_ENTRIES = 64

# Deepest cause of an error searched for its reason, and the MLflow logger of the database errors, silenced since
# it prints a whole traceback at every failed write while the tracker prints one warning for the whole outage
MAXIMUM_CAUSE_DEPTH = 10
MLFLOW_DATABASE_LOGGER = "mlflow.store.db.utils"

# End of a value cut to the length accepted by MLflow
TRUNCATION_MARKER = "..."

# MLflow errors that writing again cannot fix (invalid or changed value, deleted run): the write is dropped instead of
# retried
PERMANENT_ERROR_CODES = {"INVALID_PARAMETER_VALUE", "RESOURCE_ALREADY_EXISTS", "RESOURCE_DOES_NOT_EXIST", "BAD_REQUEST"}

# Status of a reopened run, and lifecycle stage of a run or experiment that is not deleted
RUNNING_STATUS = "RUNNING"
ACTIVE_LIFECYCLE_STAGE = "active"

# Artifact giving the original name of every logged metric, sent when the run is closed
METRIC_NAMES_FILE_NAME = "metric_names.json"

# Outcome of a write: written, rejected by MLflow (dropped), or given up because the run is closed while MLflow is
# still unavailable
OPERATION_DONE = "done"
OPERATION_REJECTED = "rejected"
OPERATION_ABANDONED = "abandoned"

# Root folder of the repository, whose git commit is tagged on every run
REPOSITORY_FOLDER = Path(__file__).absolute().parents[2]


def sanitise_metric_name(metric_name: str) -> str:
    """
    Converts a metric name of the repository (e.g. 'test/AP@0.5/<class name>' or 'test/mAP@[0.5:0.95]/area') to a
    name accepted by MLflow, which only allows letters, digits, '_', '-', '.', '/' and spaces, without empty, '.' or
    '..' segments between the slashes, and at most 250 characters. Also used for the keys of the parameters, which
    follow the same rule. The original names are kept in the 'metric_names.json' artifact of the run.

    Args:
        metric_name (str): The name used by the repository.

    Returns:
        str: The name accepted by MLflow.
    """
    # Readable replacements of the characters of the evaluation scalars: AP@0.5 -> AP_at_0.5, [0.5:0.95] -> 0.5-0.95
    sanitised_name = metric_name.replace("@", "_at_").replace("[", "").replace("]", "").replace(":", "-")

    # Accented letters reduced to their base letter, every other unsupported character replaced by an underscore
    decomposed_name = unicodedata.normalize("NFKD", sanitised_name)
    sanitised_name = "".join(character for character in decomposed_name if not unicodedata.combining(character))
    sanitised_name = re.sub(pattern=r"[^A-Za-z0-9_\-./ ]", repl="_", string=sanitised_name)[:MAX_ENTITY_KEY_LENGTH]

    # MLflow refuses the names that are not normalised paths: empty, "." and ".." segments
    segments = [segment if segment not in {".", ".."} else "_" for segment in sanitised_name.split("/") if segment]
    return "/".join(segments) if segments else "_"


def describe_error(error: Exception) -> str:
    """
    Gives the reason of an MLflow or system error in one line for the warnings of the tracker: the first line of the
    message of its deepest cause (e.g. the sqlite error wrapped by SQLAlchemy and MLflow, whose messages hold notes and
    SQL statements), cut to 300 characters and preceded by its type.

    Args:
        error (Exception): The error.

    Returns:
        str: The type and the first line of the message of the deepest cause, or its representation without message.
    """
    # The deepest cause holds the readable reason, within a few levels of wrapping
    root_error = error
    for _ in range(MAXIMUM_CAUSE_DEPTH):
        if root_error.__cause__ is None:
            break
        root_error = root_error.__cause__
    message = str(root_error).strip()
    return f"{type(root_error).__name__}: {message.splitlines()[0][:300]}" if message else repr(root_error)


def truncate_value(value: str, maximum_length: int) -> str:
    """
    Cuts a parameter or tag value to the length accepted by MLflow, marking the cut, the whole value staying in the
    configuration artifact of the run.

    Args:
        value (str): The value.
        maximum_length (int): The longest value accepted by MLflow.

    Returns:
        str: The value, cut and ending with '...' when it was too long.
    """
    if len(value) <= maximum_length:
        return value
    return value[:maximum_length - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


def flatten_configuration(configuration: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """
    Flattens the nested settings of a run to the MLflow parameters searched and compared in the MLflow table, with
    dotted keys such as 'ExperimentConfiguration.batch_size' or 'Optimization.learning_rate_schedule'. Every value is
    written as a string (paths, lists, torch types...), cut to the length accepted by MLflow. A dictionary whose keys
    are not all strings, or with more than 64 entries (the label dictionary of many classes), is one parameter.

    Args:
        configuration (Dict[str, Any]): The nested settings.
        prefix (str): The dotted key of the dictionary within the whole settings, empty at the top.

    Returns:
        Dict[str, str]: The value of every parameter, keyed by its sanitised dotted key.
    """
    parameters = {}
    for key, value in configuration.items():
        parameter_key = f"{prefix}{key}"

        # A section of settings gives one parameter per setting, through its own sub-sections
        if isinstance(value, dict) and value and len(value) <= MAXIMUM_FLATTENED_ENTRIES and \
                all(isinstance(entry_key, str) for entry_key in value):
            parameters.update(flatten_configuration(configuration=value, prefix=f"{parameter_key}."))
        else:
            parameters[sanitise_metric_name(metric_name=parameter_key)] = truncate_value(
                value=str(value), maximum_length=MAX_PARAM_VAL_LENGTH)
    return parameters


def read_git_state(repository_folder: Path) -> Dict[str, str]:
    """
    Reads the commit of the repository and whether its tracked files hold uncommitted changes, tagged on every run so
    that its code can be found again. Nothing is tagged when git is missing or the folder is not a repository. Called
    in the training thread only: a process started from the worker thread while the training thread forks the loader
    workers would make them inherit the pipes it waits on, and hang until they exit.

    Args:
        repository_folder (Path): The root folder of the repository.

    Returns:
        Dict[str, str]: The 'git_commit' and 'git_dirty' tags, or an empty dictionary.
    """
    try:
        commit = subprocess.run(args=["git", "rev-parse", "HEAD"], cwd=repository_folder, capture_output=True,
                                text=True, timeout=GIT_TIMEOUT_SECONDS, check=True).stdout.strip()
        status = subprocess.run(args=["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository_folder,
                                capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"git_commit": commit, "git_dirty": str(bool(status.strip()))}


class MlflowTracker(ExperimentTracker):
    """
    Tracker of a run in MLflow that never blocks nor stops the training. The public methods, called by the training
    thread, only put events in a bounded queue (an event is dropped and counted when the queue is full) and catch every
    error. A daemon worker thread, started by start(), owns the MLflow client: it gets or creates the experiment,
    creates the run or reopens the run of the record of a resumed training, then writes the events in batches. While
    MLflow is unavailable (locked or read-only database, unreachable server...) it waits 5 s, then 10 s, 20 s... up
    to 5 minutes before writing again, keeping the events, with one warning at the first failure and one line when
    MLflow is back. The writes MLflow rejects are dropped. close() waits at most 10 s for the last events.

    Only the worker thread reads and writes the state of the MLflow run (its client, the logged steps and parameters,
    the best value and the metric names); the training thread only reads the run identifier.
    """

    def __init__(self, tracking_uri: str, artifact_root: Optional[str], experiment_name: str, run_name: str,
                 run_record_path: Optional[Path], log_artifacts: bool) -> None:
        """
        Keeps the settings of the tracking, without any MLflow call: the run is opened by the worker thread started by
        start().

        Args:
            tracking_uri (str): The MLflow tracking URI, the local sqlite database or a server.
            artifact_root (Optional[str]): The root URI of the artifacts of the local database, None for a server.
            experiment_name (str): The MLflow experiment of the run, created when missing.
            run_name (str): The name of the MLflow run.
            run_record_path (Optional[Path]): The 'mlflow_run.json' file recording the run in its run folder, None
                for a run that is never continued.
            log_artifacts (bool): Whether the files of the run are sent as artifacts.
        """
        self.tracking_uri = tracking_uri
        self.artifact_root = artifact_root
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_record_path = run_record_path
        self.log_artifacts = log_artifacts

        # A tracking server must answer quickly and only once, since the worker retries itself; the variables set by
        # the user are kept
        if tracking_uri.startswith(("http://", "https://")):
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(HTTP_REQUEST_TIMEOUT_SECONDS))
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")

        # A locked or read-only database is reported once by the tracker instead of a traceback per failed write
        logging.getLogger(name=MLFLOW_DATABASE_LOGGER).setLevel(level=logging.CRITICAL)

        # Events of the training thread waiting for the worker, and the state of the tracker seen by both threads
        self.event_queue = queue.Queue(maxsize=MAXIMUM_QUEUE_SIZE)
        self.close_requested = threading.Event()
        self.worker_thread = None
        self.started = False
        self.closed = False
        self.accepting_events = True
        self.dropped_event_count = 0
        self.run_id = None
        self.final_status = FINISHED_STATUS

        # Settings of the run given by start()
        self.parameters = {}
        self.tags = {}
        self.resumed_iteration = None

        # State of the MLflow run, used by the worker thread only
        self.client = None
        self.experiment_id = None
        self.run_reopened = False
        self.logged_parameters = {}
        self.previous_metric_values = {}
        self.previous_resume_count = 0
        self.last_logged_steps = {}
        self.best_values = {}
        self.sanitised_names = {}
        self.metric_names = {}
        self.unavailable = False
        self.retry_delay = INITIAL_RETRY_DELAY_SECONDS
        self.rejected_write_count = 0

    # ------------------------------------------------------------------------------------------------------------------
    # Public methods, called by the training thread: they only queue events and never raise
    # ------------------------------------------------------------------------------------------------------------------

    def start(self, parameters: Dict[str, Any], tags: Dict[str, str], resumed_iteration: Optional[int]) -> None:
        """
        Flattens the parameters of the run, adds the host and git tags, and starts the worker thread, which opens the
        MLflow run in the background, so that a slow or locked database never delays the training. The git commands
        run here, in the training thread, never at the same time as the forks of the loader workers (see
        read_git_state).

        Args:
            parameters (Dict[str, Any]): The nested settings of the run (configuration sections and derived values).
            tags (Dict[str, str]): The tags identifying the run.
            resumed_iteration (Optional[int]): The checkpoint iteration a resumed training starts from, None for a run
                starting from scratch.
        """
        try:
            if self.started:
                return
            # The settings are copied as strings, so that the worker never reads objects the trainer may change
            self.started = True
            self.parameters = flatten_configuration(configuration=parameters)
            self.tags = {key: str(value) for key, value in tags.items()}
            self.resumed_iteration = resumed_iteration

            # Machine and code of the run, read in the training thread which also forks the loader workers
            self.tags.update({"host": socket.gethostname(), **read_git_state(repository_folder=REPOSITORY_FOLDER)})
            self.worker_thread = threading.Thread(target=self.run_worker, name="mlflow-tracker", daemon=True)
            self.worker_thread.start()
        except Exception as error:
            self.accepting_events = False
            print_color(color="yellow", output=f"MLflow Tracking Could Not Start, The Run Is Not Tracked : {error!r}")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Queues metric values of a training iteration.

        Args:
            metrics (Dict[str, float]): The value of every metric, keyed by its name in the repository.
            step (int): The training iteration of the values.
        """
        try:
            self.put_event(event={"type": "metrics", "step": int(step), "timestamp": int(time.time() * 1000),
                                  "values": {name: float(value) for name, value in metrics.items()}})
        except Exception as error:
            self.report_queue_error(error=error)

    def update_best_metric(self, metric_name: str, value: float, step: int) -> None:
        """
        Queues the value of the headline metric of an evaluated checkpoint, compared with the best value of the run by
        the worker.

        Args:
            metric_name (str): The name of the headline metric.
            value (float): The value of the evaluated checkpoint.
            step (int): The training iteration of the evaluated checkpoint.
        """
        try:
            self.put_event(event={"type": "best_metric", "metric_name": metric_name, "value": float(value),
                                  "step": int(step), "timestamp": int(time.time() * 1000)})
        except Exception as error:
            self.report_queue_error(error=error)

    def log_artifact(self, file_path: Path, artifact_folder: Optional[str],
                     artifact_name: Optional[str] = None) -> None:
        """
        Queues a file to send as an artifact, copied by the worker, when the artifacts are on.

        Args:
            file_path (Path): The file to send.
            artifact_folder (Optional[str]): The folder of the artifact in the run, None for its root.
            artifact_name (Optional[str]): The name of the artifact, None to keep the file name.
        """
        try:
            if self.log_artifacts:
                self.put_event(event={"type": "artifact", "file_path": Path(file_path),
                                      "artifact_folder": artifact_folder, "artifact_name": artifact_name})
        except Exception as error:
            self.report_queue_error(error=error)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Queues tags of the run.

        Args:
            tags (Dict[str, str]): The value of every tag.
        """
        try:
            self.put_event(event={"type": "tags", "values": {key: str(value) for key, value in tags.items()}})
        except Exception as error:
            self.report_queue_error(error=error)

    def get_run_id(self) -> Optional[str]:
        """
        Gives the identifier of the MLflow run, known once the worker has opened it.

        Returns:
            Optional[str]: The run identifier, None while the run is not opened.
        """
        return self.run_id

    def close(self, status: str) -> None:
        """
        Asks the worker to send the waiting events and to end the run with its final status, waiting for it at most
        10 seconds, then reports the events that could not be sent.

        Args:
            status (str): FINISHED_STATUS, KILLED_STATUS or FAILED_STATUS.
        """
        try:
            if self.worker_thread is None or self.closed:
                return
            # The queue receives nothing more, and the worker empties it then ends the run
            self.closed = True
            self.final_status = status
            self.close_requested.set()
            self.worker_thread.join(timeout=CLOSE_TIMEOUT_SECONDS)

            # Events that never reached MLflow
            if self.worker_thread.is_alive():
                print_color(color="yellow", output=f"MLflow Did Not Finish Writing Within {CLOSE_TIMEOUT_SECONDS:.0f} "
                                                   f"s, The Last Events Of The Run Are Not Tracked.")
            if self.dropped_event_count > 0:
                print_color(color="yellow", output=f"MLflow Tracking : {self.dropped_event_count} Events Dropped, The "
                                                   f"Queue Being Full.")
            if self.rejected_write_count > 0:
                print_color(color="yellow", output=f"MLflow Tracking : {self.rejected_write_count} Writes Rejected By "
                                                   f"MLflow.")
        except Exception as error:
            print_color(color="yellow", output=f"MLflow Tracking Could Not Close The Run : {error!r}")

    def put_event(self, event: Dict[str, Any]) -> None:
        """
        Puts an event in the queue of the worker without ever waiting: nothing is queued before start(), after
        close() or once the worker has stopped, and the event is dropped and counted when the queue is full.

        Args:
            event (Dict[str, Any]): The event, with its 'type'.
        """
        if not self.started or self.closed or not self.accepting_events:
            return

        # A full queue never makes the training wait
        try:
            self.event_queue.put_nowait(item=event)
        except queue.Full:
            self.dropped_event_count += 1

    @staticmethod
    def report_queue_error(error: Exception) -> None:
        """
        Reports an error raised while queuing an event (e.g. a value that is not a number), the training continuing.

        Args:
            error (Exception): The error.
        """
        print_color(color="yellow", output=f"MLflow Tracking Ignored An Event : {error!r}")

    # ------------------------------------------------------------------------------------------------------------------
    # Worker thread: the only one calling MLflow
    # ------------------------------------------------------------------------------------------------------------------

    def run_worker(self) -> None:
        """
        Body of the worker thread: opens the MLflow run, writes its parameters and tags, then writes the queued events
        in batches until the close, and ends the run with its final status. Any unexpected error stops the tracking
        only, with one warning.
        """
        try:
            # The run is opened first, retried while MLflow is unavailable, and the tracking stops if MLflow rejects it
            if self.execute_with_retry(operation=self.open_run) != OPERATION_DONE:
                return
            for operation in self.build_start_operations():
                if self.execute_with_retry(operation=operation) == OPERATION_ABANDONED:
                    return

            # Batches of events until the close, after which the queue receives nothing more and is emptied
            while True:
                closing = self.close_requested.is_set()
                for operation in self.build_write_operations(events=self.collect_events(closing=closing)):
                    if self.execute_with_retry(operation=operation) == OPERATION_ABANDONED:
                        return
                if closing and self.event_queue.empty():
                    break

            # Original metric names and final status of the run
            for operation in self.build_finish_operations():
                if self.execute_with_retry(operation=operation) == OPERATION_ABANDONED:
                    return
        except Exception as error:
            print_color(color="yellow", output=f"MLflow Tracking Stopped By An Unexpected Error, The Training "
                                               f"Continues Without It : {error!r}")
        finally:
            self.accepting_events = False

    def execute_with_retry(self, operation: Callable[[], None]) -> str:
        """
        Runs one MLflow operation of the worker until it succeeds: a rejected operation (invalid value, deleted run)
        is dropped, and a failed one (locked database, unreachable server, full disk) is retried after a delay doubling
        from 5 s to 5 minutes, the events waiting in the queue. Once the run is closed, a failed operation is given up.
        One warning is printed when MLflow becomes unavailable, and one line when it is back.

        Args:
            operation (Callable[[], None]): The operation, a method or a functools.partial of the worker.

        Returns:
            str: OPERATION_DONE, OPERATION_REJECTED or OPERATION_ABANDONED.
        """
        while True:
            # Rejected writes are dropped, the other MLflow and system errors are retried, any other error is a bug
            # stopping the worker
            try:
                operation()
            except MlflowException as error:
                if error.error_code in PERMANENT_ERROR_CODES:
                    self.report_rejected_write(error=error)
                    return OPERATION_REJECTED
                failure = error
            except OSError as error:
                failure = error
            else:
                if self.unavailable:
                    self.unavailable = False
                    self.retry_delay = INITIAL_RETRY_DELAY_SECONDS
                    print_color(color="green", output="MLflow Available Again, The Waiting Events Are Sent.")
                return OPERATION_DONE

            # A single warning for the whole unavailability, whatever the number of failed writes
            if not self.unavailable:
                self.unavailable = True
                print_color(color="yellow", output=f"MLflow Unavailable, The Training Continues Without It And The "
                                                   f"Tracking Retries In The Background : "
                                                   f"{describe_error(error=failure)}")

            # The close leaves no time for more retries, otherwise the next try waits, woken early by a close
            if self.close_requested.is_set():
                return OPERATION_ABANDONED
            self.close_requested.wait(timeout=self.retry_delay)
            self.retry_delay = min(self.retry_delay * 2.0, MAXIMUM_RETRY_DELAY_SECONDS)

    def report_rejected_write(self, error: MlflowException) -> None:
        """
        Counts a write rejected by MLflow, printing the first one, the others being counted in the closing message.

        Args:
            error (MlflowException): The rejection.
        """
        self.rejected_write_count += 1
        if self.rejected_write_count == 1:
            print_color(color="yellow", output=f"MLflow Rejected A Write, Which Is Dropped : "
                                               f"{describe_error(error=error)}")

    def open_run(self) -> None:
        """
        Opens the MLflow run: gets or creates the experiment, reopens the run of the record of a resumed training
        when it is still there, otherwise creates a new run (tagged with the previous run of the record), and records
        the run in the run folder. Retried as a whole while MLflow is unavailable, without creating a second run.
        """
        # The folder of the local database must exist before sqlite creates it
        if self.tracking_uri.startswith("sqlite:///"):
            Path(self.tracking_uri[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
        if self.client is None:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
        if self.experiment_id is None:
            self.experiment_id = self.get_or_create_experiment()

        # The run of the previous sessions of a resumed training, or a new run
        run_record = read_run_record(run_record_path=self.run_record_path)
        if self.run_id is None:
            self.run_reopened = self.reopen_run(run_record=run_record)
            if not self.run_reopened:
                if run_record is not None:
                    self.tags["previous_run_id"] = run_record["run_id"]
                self.run_id = self.client.create_run(experiment_id=self.experiment_id,
                                                     run_name=self.run_name).info.run_id

        # Record of the run in its run folder, read by the next resume
        if self.run_record_path is not None:
            write_run_record(run_record_path=self.run_record_path,
                             run_record={"tracking_uri": self.tracking_uri, "experiment_id": self.experiment_id,
                                         "run_id": self.run_id})

        # Where to browse the run
        if self.tracking_uri.startswith("sqlite:///"):
            browse_hint = f"mlflow ui --backend-store-uri {self.tracking_uri}"
        else:
            browse_hint = f"{self.tracking_uri.rstrip('/')}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
        print_color(color="blue", output=f"MLflow Run {'Reopened' if self.run_reopened else 'Created'} In The "
                                         f"Experiment {self.experiment_name!r} : {self.run_id} (Browse : "
                                         f"{browse_hint})")

    def get_or_create_experiment(self) -> str:
        """
        Gets the MLflow experiment of the run, restoring it when it was deleted, or creates it with its artifacts in
        the artifacts folder of the local database (the server choosing for itself).

        Returns:
            str: The identifier of the experiment.
        """
        # A new experiment keeps its artifacts next to the local database, in a folder named after it
        experiment = self.client.get_experiment_by_name(name=self.experiment_name)
        if experiment is None:
            artifact_location = None
            if self.artifact_root is not None:
                artifact_location = f"{self.artifact_root}/{self.experiment_name}"
            try:
                return self.client.create_experiment(name=self.experiment_name, artifact_location=artifact_location)
            except MlflowException as error:
                # Another training created it at the same time
                if error.error_code != "RESOURCE_ALREADY_EXISTS":
                    raise
                experiment = self.client.get_experiment_by_name(name=self.experiment_name)
        # An experiment deleted in the interface is restored, MLflow refusing new runs in it
        if experiment.lifecycle_stage != ACTIVE_LIFECYCLE_STAGE:
            self.client.restore_experiment(experiment_id=experiment.experiment_id)
        return experiment.experiment_id

    def reopen_run(self, run_record: Optional[Dict[str, str]]) -> bool:
        """
        Reopens the run of the record of a resumed training, when it is in the same backend and experiment and not
        deleted: its status goes back to RUNNING, and its logged parameters, last logged steps, metric values and
        resume count are read so that the resumed training only adds to it.

        Args:
            run_record (Optional[Dict[str, str]]): The record of the run folder, None when there is none.

        Returns:
            bool: True when the run was reopened, False when a new run must be created.
        """
        # Only a resumed training continues a run, of the same backend, still existing in the same experiment
        if self.resumed_iteration is None or run_record is None or run_record["tracking_uri"] != self.tracking_uri:
            return False
        try:
            run = self.client.get_run(run_id=run_record["run_id"])
        except MlflowException as error:
            if error.error_code == "RESOURCE_DOES_NOT_EXIST":
                return False
            raise
        if run.info.lifecycle_stage != ACTIVE_LIFECYCLE_STAGE or run.info.experiment_id != self.experiment_id:
            return False

        # Last logged step of every metric, so that the iterations run again after the checkpoint are not logged twice
        last_logged_steps = {}
        for metric_key in run.data.metrics:
            metric_history = self.client.get_metric_history(run_id=run.info.run_id, key=metric_key)
            last_logged_steps[metric_key] = max(metric.step for metric in metric_history)

        # The run is running again, and keeps what the previous sessions logged
        self.client.update_run(run_id=run.info.run_id, status=RUNNING_STATUS)
        self.last_logged_steps = last_logged_steps
        self.logged_parameters = dict(run.data.params)
        self.previous_metric_values = dict(run.data.metrics)
        resume_count = run.data.tags["resume_count"] if "resume_count" in run.data.tags else "0"
        self.previous_resume_count = int(resume_count) if resume_count.isdigit() else 0
        self.run_id = run.info.run_id
        return True

    def build_start_operations(self) -> List[Callable[[], None]]:
        """
        Builds the writes of the parameters and tags of the opened run. A reopened run keeps the parameters it has,
        MLflow refusing to change them: only the new parameters are logged, and the changed values are recorded as
        'resumed_<iteration>.<key>' tags. A resumed training updates the 'resume_count' and 'last_resumed_iteration'
        tags.

        Returns:
            List[Callable[[], None]]: The write operations.
        """
        # Tags of the trainer, the machine and the code of the run
        tags = dict(self.tags)

        # New parameters logged, changed ones kept as tags of the resumption
        parameters = {}
        for parameter_key, parameter_value in self.parameters.items():
            if parameter_key not in self.logged_parameters:
                parameters[parameter_key] = parameter_value
            elif self.logged_parameters[parameter_key] != parameter_value:
                tags[f"resumed_{self.resumed_iteration}.{parameter_key}"] = parameter_value

        # Resumptions of the training, counted over the sessions of the run
        if self.resumed_iteration is not None:
            tags["resume_count"] = str(self.previous_resume_count + 1)
            tags["last_resumed_iteration"] = str(self.resumed_iteration)
        return self.build_batch_operations(parameters=parameters, tags=tags, metrics=[])

    def collect_events(self, closing: bool) -> List[Dict[str, Any]]:
        """
        Takes the next batch of events from the queue: during the training, the events of the next 2 seconds, or
        fewer once 500 metric values wait or the close is asked; at the close, every waiting event up to 500 metric
        values.

        Args:
            closing (bool): True when the close was asked before this batch, the queue receiving nothing more.

        Returns:
            List[Dict[str, Any]]: The events, in their order.
        """
        events = []
        number_metric_values = 0
        deadline = time.monotonic() + FLUSH_INTERVAL_SECONDS
        while number_metric_values < FLUSH_METRIC_COUNT:
            try:
                # At the close the queue only empties, otherwise it is polled briefly so that a close is seen at once
                if closing:
                    event = self.event_queue.get_nowait()
                else:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0 or self.close_requested.is_set():
                        break
                    event = self.event_queue.get(timeout=min(remaining_seconds, QUEUE_POLL_SECONDS))
            except queue.Empty:
                if closing:
                    break
                continue
            # The metric values count for the size of the batch, every other event for one
            events.append(event)
            number_metric_values += len(event["values"]) if event["type"] == "metrics" else 1
        return events

    def build_write_operations(self, events: List[Dict[str, Any]]) -> List[Callable[[], None]]:
        """
        Turns a batch of events into MLflow writes: the metric values (without the steps already logged by the
        previous sessions of a reopened run), the best value of the headline metric with its iteration, the tags in
        batches, then one copy per artifact.

        Args:
            events (List[Dict[str, Any]]): The events of the batch.

        Returns:
            List[Callable[[], None]]: The write operations.
        """
        # Metric values and tags gathered into batches, artifacts copied one by one after them
        metrics, tags, artifact_operations = [], {}, []
        for event in events:
            if event["type"] == "metrics":
                for metric_name, metric_value in event["values"].items():
                    self.add_metric(metrics=metrics, metric_name=metric_name, value=metric_value, step=event["step"],
                                    timestamp=event["timestamp"])
            elif event["type"] == "best_metric":
                self.add_best_metric(metrics=metrics, tags=tags, event=event)
            elif event["type"] == "tags":
                tags.update(event["values"])
            elif event["type"] == "artifact":
                artifact_operations.append(functools.partial(
                    self.write_artifact, file_path=event["file_path"], artifact_folder=event["artifact_folder"],
                    artifact_name=event["artifact_name"]))
        return self.build_batch_operations(parameters={}, tags=tags, metrics=metrics) + artifact_operations

    def get_sanitised_name(self, metric_name: str) -> str:
        """
        Gives the MLflow name of a metric, the same for the whole run: its sanitised name, followed by '_2', '_3'...
        when another metric already has it. Keeps the original name for the 'metric_names.json' artifact.

        Args:
            metric_name (str): The name used by the repository.

        Returns:
            str: The MLflow name.
        """
        if metric_name in self.sanitised_names:
            return self.sanitised_names[metric_name]

        # Two names of the repository giving the same MLflow name are told apart by a number
        sanitised_name = sanitise_metric_name(metric_name=metric_name)
        candidate_name, duplicate_index = sanitised_name, 2
        while candidate_name in self.metric_names:
            suffix = f"_{duplicate_index}"
            candidate_name = sanitised_name[:MAX_ENTITY_KEY_LENGTH - len(suffix)] + suffix
            duplicate_index += 1
        self.sanitised_names[metric_name] = candidate_name
        self.metric_names[candidate_name] = metric_name
        return candidate_name

    def add_metric(self, metrics: List[Metric], metric_name: str, value: float, step: int, timestamp: int) -> None:
        """
        Adds a metric value to a batch, unless a previous session of the reopened run already logged this step.

        Args:
            metrics (List[Metric]): The metric values of the batch, extended in place.
            metric_name (str): The name used by the repository.
            value (float): The value.
            step (int): The training iteration.
            timestamp (int): The time of the value, in milliseconds.
        """
        metric_key = self.get_sanitised_name(metric_name=metric_name)
        if metric_key in self.last_logged_steps and step <= self.last_logged_steps[metric_key]:
            return
        metrics.append(Metric(key=metric_key, value=value, timestamp=timestamp, step=step))

    def add_best_metric(self, metrics: List[Metric], tags: Dict[str, str], event: Dict[str, Any]) -> None:
        """
        Compares the headline metric of an evaluated checkpoint with the best value of the run (the higher the
        better, a reopened run starting from its logged best value), adds the best value so far as
        'test/best_<metric name>' and tags 'best_iteration' when the checkpoint is the new best.

        Args:
            metrics (List[Metric]): The metric values of the batch, extended in place.
            tags (Dict[str, str]): The tags of the batch, extended in place.
            event (Dict[str, Any]): The 'best_metric' event.
        """
        best_metric_name = f"test/best_{event['metric_name']}"
        best_metric_key = self.get_sanitised_name(metric_name=best_metric_name)
        step, value = event["step"], event["value"]

        # An evaluation already logged by a previous session of a reopened run was already compared
        if best_metric_key in self.last_logged_steps and step <= self.last_logged_steps[best_metric_key]:
            return

        # Best value of the previous sessions of a reopened run
        if best_metric_key not in self.best_values and best_metric_key in self.previous_metric_values:
            self.best_values[best_metric_key] = self.previous_metric_values[best_metric_key]

        # A checkpoint whose value is not a number never becomes the best one
        if not math.isnan(value) and (best_metric_key not in self.best_values or
                                      math.isnan(self.best_values[best_metric_key]) or
                                      value > self.best_values[best_metric_key]):
            self.best_values[best_metric_key] = value
            tags["best_iteration"] = str(step)
        if best_metric_key in self.best_values:
            self.add_metric(metrics=metrics, metric_name=best_metric_name, value=self.best_values[best_metric_key],
                            step=step, timestamp=event["timestamp"])

    def build_batch_operations(self, parameters: Dict[str, str], tags: Dict[str, str],
                               metrics: List[Metric]) -> List[Callable[[], None]]:
        """
        Splits parameters, tags and metric values into the 'log_batch' writes accepted by MLflow (at most 100
        parameters or tags, or 1000 metric values per write), the tag values being cut to the accepted length. Every
        write is retried on its own, so that a failure never writes a batch twice.

        Args:
            parameters (Dict[str, str]): The parameters to log.
            tags (Dict[str, str]): The tags to set.
            metrics (List[Metric]): The metric values to log.

        Returns:
            List[Callable[[], None]]: The write operations.
        """
        # MLflow entities, the parameter values being already cut by flatten_configuration
        parameter_list = [Param(key=key, value=value) for key, value in parameters.items()]
        tag_list = [RunTag(key=key, value=truncate_value(value=value, maximum_length=MAX_TAG_VAL_LENGTH))
                    for key, value in tags.items()]
        # One write per batch of the size accepted by MLflow
        operations = []
        for start_index in range(0, len(parameter_list), MAX_PARAMS_TAGS_PER_BATCH):
            operations.append(functools.partial(
                self.client.log_batch, run_id=self.run_id,
                params=parameter_list[start_index:start_index + MAX_PARAMS_TAGS_PER_BATCH]))
        for start_index in range(0, len(tag_list), MAX_PARAMS_TAGS_PER_BATCH):
            operations.append(functools.partial(self.client.log_batch, run_id=self.run_id,
                                                tags=tag_list[start_index:start_index + MAX_PARAMS_TAGS_PER_BATCH]))
        for start_index in range(0, len(metrics), MAX_METRICS_PER_BATCH):
            operations.append(functools.partial(self.client.log_batch, run_id=self.run_id,
                                                metrics=metrics[start_index:start_index + MAX_METRICS_PER_BATCH]))
        return operations

    def write_artifact(self, file_path: Path, artifact_folder: Optional[str], artifact_name: Optional[str]) -> None:
        """
        Copies a file of the run into its MLflow artifacts, under another name through a temporary copy when asked. A
        missing file is skipped.

        Args:
            file_path (Path): The file to send.
            artifact_folder (Optional[str]): The folder of the artifact in the run, None for its root.
            artifact_name (Optional[str]): The name of the artifact, None to keep the file name.
        """
        # A file removed since it was queued is skipped, a file keeping its name is copied as it is
        if not file_path.is_file():
            return
        if artifact_name is None or artifact_name == file_path.name:
            self.client.log_artifact(run_id=self.run_id, local_path=str(file_path), artifact_path=artifact_folder)
            return
        # MLflow names an artifact after its file, so a renamed artifact goes through a temporary copy
        with tempfile.TemporaryDirectory() as temporary_directory:
            renamed_path = Path(temporary_directory) / artifact_name
            shutil.copyfile(src=file_path, dst=renamed_path)
            self.client.log_artifact(run_id=self.run_id, local_path=str(renamed_path), artifact_path=artifact_folder)

    def write_metric_names(self) -> None:
        """
        Sends the 'metric_names.json' artifact, giving the original name of every metric logged in this session from
        its MLflow name, the repository names holding characters MLflow refuses (e.g. 'AP@0.5/<class name>').
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            metric_names_path = Path(temporary_directory) / METRIC_NAMES_FILE_NAME
            metric_names_path.write_text(data=json.dumps(obj=self.metric_names, indent=4, ensure_ascii=False))
            self.client.log_artifact(run_id=self.run_id, local_path=str(metric_names_path))

    def build_finish_operations(self) -> List[Callable[[], None]]:
        """
        Builds the last writes of the run: the original metric names when the artifacts are on, then the final status.

        Returns:
            List[Callable[[], None]]: The write operations.
        """
        operations = []
        if self.log_artifacts and self.metric_names:
            operations.append(self.write_metric_names)
        operations.append(functools.partial(self.client.set_terminated, run_id=self.run_id, status=self.final_status))
        return operations
