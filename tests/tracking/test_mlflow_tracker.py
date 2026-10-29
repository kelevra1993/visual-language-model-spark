"""
Tests of the MLflow tracker of vision_training/tracking/mlflow_tracker.py: the metric names and the configuration are
converted to the names and values MLflow accepts, the warnings give the deepest cause of an error, a run is written to
the local sqlite database and read back with its parameters, tags, metrics and artifacts, a resumed training reopens
its run without logging a step twice and records its changed parameters as tags, the git commands run in the training
thread, and the training is never blocked by an unavailable, hanging or unreachable backend, a full queue or rejected
writes.
"""

import functools
import json
import os
import posixpath
import pytest
import re
import sqlite3
import threading
import time

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, TEMPORARILY_UNAVAILABLE
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Any, Dict

from vision_training.tracking import mlflow_tracker
from vision_training.tracking.experiment_tracker import (ARTIFACTS_FOLDER_NAME, FINISHED_STATUS, KILLED_STATUS,
                                                         RUN_RECORD_FILE_NAME, create_experiment_tracker,
                                                         read_run_record)
from vision_training.tracking.mlflow_tracker import (MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH, TRUNCATION_MARKER,
                                                     MlflowTracker, describe_error, flatten_configuration,
                                                     sanitise_metric_name)

# Tracking settings of the tests, on with the local database
TRACKING_CONFIGURATION = {"activated": True, "tracking_uri": None, "experiment_name": None}

# Longest wait of a blocked request of the fake clients, released by the test long before
BLOCKED_REQUEST_TIMEOUT_SECONDS = 30.0

# Characters MLflow accepts in the metric names and parameter keys
MLFLOW_NAME_PATTERN = re.compile(pattern=r"^[A-Za-z0-9_\-./ ]+$")


class UnavailableClient:
    """
    MLflow client of a backend that is always unavailable (locked or read-only database, server down): its first
    request, and so every request, fails with a temporary error.
    """

    def __init__(self, tracking_uri: str) -> None:
        """
        Keeps the tracking URI like the real client.

        Args:
            tracking_uri (str): The tracking URI given by the tracker.
        """
        self.tracking_uri = tracking_uri

    def get_experiment_by_name(self, name: str) -> None:
        """
        Fails like a locked database.

        Args:
            name (str): The experiment name.

        Raises:
            MlflowException: Always, with a temporary error code.
        """
        raise MlflowException(message="database is locked", error_code=TEMPORARILY_UNAVAILABLE)


class BlockedClient:
    """
    MLflow client of a backend that does not answer: its first request hangs until the test releases it, then fails
    with a temporary error.
    """

    def __init__(self, tracking_uri: str, release_event: threading.Event) -> None:
        """
        Keeps the tracking URI and the event releasing the blocked request.

        Args:
            tracking_uri (str): The tracking URI given by the tracker.
            release_event (threading.Event): The event set by the test to release the request.
        """
        self.tracking_uri = tracking_uri
        self.release_event = release_event

    def get_experiment_by_name(self, name: str) -> None:
        """
        Hangs until released, then fails like a request that timed out.

        Args:
            name (str): The experiment name.

        Raises:
            MlflowException: Always, with a temporary error code.
        """
        self.release_event.wait(timeout=BLOCKED_REQUEST_TIMEOUT_SECONDS)
        raise MlflowException(message="request timed out", error_code=TEMPORARILY_UNAVAILABLE)


class RejectingClient(MlflowClient):
    """
    MLflow client of a working backend that rejects every batch of parameters, tags or metrics, like values MLflow
    refuses: the experiment and the run are created, and the run is closed.
    """

    def log_batch(self, run_id: str, **batch: Any) -> None:
        """
        Rejects the batch.

        Args:
            run_id (str): The run identifier.
            **batch (Any): The parameters, tags or metrics of the batch.

        Raises:
            MlflowException: Always, with a permanent error code.
        """
        raise MlflowException(message="invalid value", error_code=INVALID_PARAMETER_VALUE)


class GitStateRecorder:
    """
    Replacement of read_git_state recording the thread calling it.
    """

    def __init__(self) -> None:
        """
        Starts with no recorded call.
        """
        self.calling_threads: list[threading.Thread] = []

    def __call__(self, repository_folder: Path) -> Dict[str, str]:
        """
        Records the calling thread and gives a fixed git state.

        Args:
            repository_folder (Path): The root folder of the repository.

        Returns:
            Dict[str, str]: The 'git_commit' and 'git_dirty' tags.
        """
        self.calling_threads.append(threading.current_thread())
        return {"git_commit": "abc", "git_dirty": "False"}


def build_tracker(tracking_root: Path, run_folder: Path) -> MlflowTracker:
    """
    Builds the MLflow tracker of a run folder through the factory, on the local database of the experiments root
    folder, its experiment named after the dummy task.

    Args:
        tracking_root (Path): The experiments root folder holding the local database.
        run_folder (Path): The run folder receiving the run record.

    Returns:
        MlflowTracker: The tracker, not started.
    """
    run_folder.mkdir(parents=True, exist_ok=True)
    tracker = create_experiment_tracker(tracking_configuration=TRACKING_CONFIGURATION, tracking_root=tracking_root,
                                        run_name=run_folder.name, task="dummy_task",
                                        run_record_path=run_folder / RUN_RECORD_FILE_NAME)
    assert isinstance(tracker, MlflowTracker)
    return tracker


def read_metric_history(client: MlflowClient, run_id: str, metric_key: str) -> list[tuple[int, float]]:
    """
    Reads the logged values of a metric of a run, in the order of their steps.

    Args:
        client (MlflowClient): The client of the backend.
        run_id (str): The run identifier.
        metric_key (str): The MLflow name of the metric.

    Returns:
        list[tuple[int, float]]: The step and value of every logged point.
    """
    return sorted((metric.step, metric.value) for metric in client.get_metric_history(run_id=run_id, key=metric_key))


def test_metric_names_are_sanitised_for_mlflow() -> None:
    """
    Verifies the conversion of the metric names of the repository to the names MLflow accepts: '@' and the brackets
    of the IoU ranges written readably, accents removed, other characters replaced, empty, '.' and '..' segments
    removed, and the length cut to the MLflow limit.
    """
    expected_names = {"test_class/AP@0.5/[unk]": "test_class/AP_at_0.5/unk",
                      "test/mAP@[0.5:0.95]/area": "test/mAP_at_0.5-0.95/area",
                      "test_class/Recall/café": "test_class/Recall/cafe", "test_class/F1/ship (large)":
                      "test_class/F1/ship _large_", "train/Total Loss": "train/Total Loss",
                      "test/Top-2 Accuracy": "test/Top-2 Accuracy", "a//b/./c/": "a/b/_/c", "/..": "_"}
    for metric_name, expected_name in expected_names.items():
        assert sanitise_metric_name(metric_name=metric_name) == expected_name

    # Every converted name is a normalised path of accepted characters within the length limit
    for metric_name in list(expected_names) + ["x" * 400, "AP@0.5/北京", "..", "a/../b"]:
        sanitised_name = sanitise_metric_name(metric_name=metric_name)
        assert MLFLOW_NAME_PATTERN.match(string=sanitised_name) is not None
        assert posixpath.normpath(sanitised_name) == sanitised_name and not sanitised_name.startswith("..")
        assert len(sanitised_name) <= MAX_ENTITY_KEY_LENGTH


def test_error_is_described_by_its_deepest_cause() -> None:
    """
    Verifies the one-line reason of the tracker warnings: the first line of the deepest cause of an error wrapped by
    MLflow (e.g. the sqlite error under the SQLAlchemy one, whose message starts with a note), with its type, and the
    representation of an error without message.
    """
    try:
        try:
            try:
                raise sqlite3.OperationalError("attempt to write a readonly database")
            except sqlite3.OperationalError as database_error:
                raise RuntimeError("(raised as a result of Query-invoked autoflush)\n[SQL: INSERT]") from database_error
        except RuntimeError as wrapped_error:
            raise MlflowException(message="SQLAlchemy error", error_code=TEMPORARILY_UNAVAILABLE) from wrapped_error
    except MlflowException as mlflow_error:
        assert describe_error(error=mlflow_error) == "OperationalError: attempt to write a readonly database"
    assert describe_error(error=MlflowException(message="database is locked")) == (
        "MlflowException: database is locked")
    assert describe_error(error=TimeoutError()) == "TimeoutError()"


def test_configuration_is_flattened_to_parameters(tmp_path: Path) -> None:
    """
    Verifies the flattening of the nested settings to MLflow parameters: dotted keys through the sections, every value
    written as a string (paths, lists, booleans, null), the dictionaries keyed by numbers or too long kept as one
    parameter, and the long values cut to the MLflow limit.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    configuration = {"ExperimentConfiguration": {"batch_size": 4, "experiment_folder": tmp_path,
                                                 "run_name_suffix": None, "evaluation": {"per_class_report": True}},
                     "Backbone": {"convolutions": {"1": [1, 16], "2": [2, 32]}},
                     "Data": {"label_dictionary": {1: 0, 2: 1},
                              "class_names": {f"class_{index}": index for index in range(100)},
                              "long_value": "x" * (MAX_PARAM_VAL_LENGTH + 10)}}
    parameters = flatten_configuration(configuration=configuration)

    assert parameters["ExperimentConfiguration.batch_size"] == "4"
    assert parameters["ExperimentConfiguration.experiment_folder"] == str(tmp_path)
    assert parameters["ExperimentConfiguration.evaluation.per_class_report"] == "True"
    assert parameters["ExperimentConfiguration.run_name_suffix"] == "None"
    assert parameters["Backbone.convolutions.1"] == "[1, 16]" and parameters["Backbone.convolutions.2"] == "[2, 32]"
    assert parameters["Data.label_dictionary"] == "{1: 0, 2: 1}"
    assert parameters["Data.class_names"].startswith("{'class_0': 0")
    assert len(parameters["Data.long_value"]) == MAX_PARAM_VAL_LENGTH
    assert parameters["Data.long_value"].endswith(TRUNCATION_MARKER)
    assert len(parameters) == 9


def test_run_is_written_to_the_local_database(tracking_root: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies a whole run in the local sqlite database: the run is created in the experiment of the task with the run
    name, recorded in its run folder, and holds its parameters, tags (with the host and the git commit), metrics under
    their sanitised names, best headline metric with its iteration, artifacts (renamed ones included, a missing file
    skipped) and the original metric names, under the artifacts folder of the experiments root, with its final status.

    Args:
        tracking_root (Path): The experiments root folder holding the empty database.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    run_folder = tracking_root / "dummy_run"
    configuration_path = run_folder / "training_configuration.yaml"
    tracker = build_tracker(tracking_root=tracking_root, run_folder=run_folder)
    configuration_path.write_text(data="ExperimentConfiguration: {}\n")

    tracker.start(parameters={"ExperimentConfiguration": {"batch_size": 4},
                              "Backbone": {"convolutions": {"1": [1, 16]}}},
                  tags={"task": "dummy_task", "run_name": "dummy_run"}, resumed_iteration=None)
    tracker.log_metrics(metrics={"train/Total Loss": 1.0, "learning_rate": 0.01}, step=2)
    tracker.log_metrics(metrics={"train/Total Loss": 0.5, "learning_rate": 0.01}, step=4)
    tracker.log_metrics(metrics={"test/Accuracy": 0.5, "test_class/AP@0.5/[unk]": 0.25}, step=4)
    tracker.update_best_metric(metric_name="Accuracy", value=0.5, step=4, metric_prefix="test")
    tracker.update_best_metric(metric_name="Accuracy", value=0.4, step=8, metric_prefix="test")
    tracker.update_best_metric(metric_name="Accuracy", value=0.3, step=4, metric_prefix="test_ema")
    tracker.update_best_metric(metric_name="Accuracy", value=0.6, step=8, metric_prefix="test_ema")
    tracker.set_tags(tags={"note": "smoke"})
    tracker.log_artifact(file_path=configuration_path, artifact_folder="configuration")
    tracker.log_artifact(file_path=configuration_path, artifact_folder="configuration",
                         artifact_name="training_configuration_resume_4.yaml")
    tracker.log_artifact(file_path=run_folder / "missing.txt", artifact_folder=None)
    tracker.close(status=FINISHED_STATUS)

    # The run recorded in its run folder, in the experiment of the task
    run_record = read_run_record(run_record_path=run_folder / RUN_RECORD_FILE_NAME)
    assert run_record["run_id"] == tracker.get_run_id()
    client = MlflowClient(tracking_uri=run_record["tracking_uri"])
    run = client.get_run(run_id=run_record["run_id"])
    assert run.info.status == FINISHED_STATUS and run.info.run_name == "dummy_run"
    assert client.get_experiment(experiment_id=run_record["experiment_id"]).name == "dummy_task"

    # Parameters and tags
    assert run.data.params == {"ExperimentConfiguration.batch_size": "4", "Backbone.convolutions.1": "[1, 16]"}
    assert {"task", "run_name", "host", "note", "best_iteration", "git_commit", "git_dirty"} <= set(run.data.tags)
    assert run.data.tags["best_iteration"] == "4" and run.data.tags["note"] == "smoke"
    assert run.data.tags["best_iteration_test_ema"] == "8"

    # Metrics under their sanitised names, the best value of the trained weights staying the one of iteration 4, the
    # averaged weights following their own best value
    run_id = run.info.run_id
    assert read_metric_history(client=client, run_id=run_id, metric_key="train/Total Loss") == [(2, 1.0), (4, 0.5)]
    assert read_metric_history(client=client, run_id=run_id, metric_key="test_class/AP_at_0.5/unk") == [(4, 0.25)]
    assert read_metric_history(client=client, run_id=run_id, metric_key="test/best_Accuracy") == [(4, 0.5), (8, 0.5)]
    assert read_metric_history(client=client, run_id=run_id, metric_key="test_ema/best_Accuracy") == [(4, 0.3),
                                                                                                    (8, 0.6)]

    # Artifacts in the artifacts folder of the experiments root, with the original metric names
    assert run.info.artifact_uri.startswith((tracking_root / ARTIFACTS_FOLDER_NAME / "dummy_task").as_uri())
    assert sorted(artifact.path for artifact in client.list_artifacts(run_id=run_id)) == ["configuration",
                                                                                           "metric_names.json"]
    assert sorted(artifact.path for artifact in client.list_artifacts(run_id=run_id, path="configuration")) == [
        "configuration/training_configuration.yaml", "configuration/training_configuration_resume_4.yaml"]
    metric_names_path = Path(client.download_artifacts(run_id=run_id, path="metric_names.json",
                                                       dst_path=str(tracking_root)))
    assert json.loads(metric_names_path.read_text())["test_class/AP_at_0.5/unk"] == "test_class/AP@0.5/[unk]"

    console_output = capsys.readouterr().out
    assert "MLflow Run Created" in console_output and "MLflow Unavailable" not in console_output


def test_resumed_training_continues_its_run(tracking_root: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a resumed training reopens the run of its record: the steps already logged before the stop are not
    logged again, the parameters keep their first values with the changed ones as 'resumed_<iteration>.<key>' tags and
    the new ones added, the resumptions are counted, and the best headline metric continues from the logged one. A
    training starting from scratch in the same run folder creates a new run tagged with the previous one.

    Args:
        tracking_root (Path): The experiments root folder holding the empty database.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    run_folder = tracking_root / "dummy_run"

    # First session: trained up to iteration 6, its last checkpoint being the one of iteration 4, stopped by the user
    first_tracker = build_tracker(tracking_root=tracking_root, run_folder=run_folder)
    first_tracker.start(parameters={"ExperimentConfiguration": {"batch_size": 4, "learning_rate": 0.01}},
                        tags={"task": "dummy_task"}, resumed_iteration=None)
    for step, loss in [(2, 1.0), (4, 0.8), (6, 0.6)]:
        first_tracker.log_metrics(metrics={"train/Total Loss": loss}, step=step)
    first_tracker.update_best_metric(metric_name="Accuracy", value=0.5, step=4, metric_prefix="test")
    first_tracker.close(status=KILLED_STATUS)

    # Second session: resumed from the checkpoint of iteration 4 with a larger batch and a new setting
    second_tracker = build_tracker(tracking_root=tracking_root, run_folder=run_folder)
    second_tracker.start(parameters={"ExperimentConfiguration": {"batch_size": 8, "learning_rate": 0.01},
                                     "Optimization": {"optimizer": "adam"}},
                         tags={"task": "dummy_task"}, resumed_iteration=4)
    for step, loss in [(6, 0.7), (8, 0.5)]:
        second_tracker.log_metrics(metrics={"train/Total Loss": loss}, step=step)
    second_tracker.update_best_metric(metric_name="Accuracy", value=0.4, step=8, metric_prefix="test")
    second_tracker.close(status=FINISHED_STATUS)

    run_id = first_tracker.get_run_id()
    assert second_tracker.get_run_id() == run_id
    client = MlflowClient(tracking_uri=read_run_record(run_record_path=run_folder / RUN_RECORD_FILE_NAME)[
        "tracking_uri"])
    run = client.get_run(run_id=run_id)
    assert run.info.status == FINISHED_STATUS
    assert run.data.params == {"ExperimentConfiguration.batch_size": "4",
                               "ExperimentConfiguration.learning_rate": "0.01", "Optimization.optimizer": "adam"}
    assert run.data.tags["resumed_4.ExperimentConfiguration.batch_size"] == "8"
    assert (run.data.tags["resume_count"], run.data.tags["last_resumed_iteration"]) == ("1", "4")
    assert run.data.tags["best_iteration"] == "4"
    assert read_metric_history(client=client, run_id=run_id, metric_key="train/Total Loss") == [
        (2, 1.0), (4, 0.8), (6, 0.6), (8, 0.5)]
    assert read_metric_history(client=client, run_id=run_id, metric_key="test/best_Accuracy") == [(4, 0.5), (8, 0.5)]
    assert "MLflow Run Reopened" in capsys.readouterr().out

    # Third session: a training from scratch in the same run folder starts a new run, linked to the previous one
    new_tracker = build_tracker(tracking_root=tracking_root, run_folder=run_folder)
    new_tracker.start(parameters={}, tags={"task": "dummy_task"}, resumed_iteration=None)
    new_tracker.close(status=FINISHED_STATUS)
    assert new_tracker.get_run_id() not in {None, run_id}
    assert client.get_run(run_id=new_tracker.get_run_id()).data.tags["previous_run_id"] == run_id
    assert read_run_record(run_record_path=run_folder / RUN_RECORD_FILE_NAME)["run_id"] == new_tracker.get_run_id()


def test_tracker_does_nothing_outside_its_run(tmp_path: Path) -> None:
    """
    Verifies that the calls made before start() are ignored, and that closing a tracker never started does nothing.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    tracker = MlflowTracker(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=None,
                            experiment_name="dummy_task", run_name="dummy_run", run_record_path=None)
    tracker.log_metrics(metrics={"train/Total Loss": 1.0}, step=1)
    tracker.set_tags(tags={"note": "ignored"})
    tracker.close(status=FINISHED_STATUS)
    assert tracker.event_queue.qsize() == 0 and tracker.get_run_id() is None
    assert list(tmp_path.iterdir()) == []


def test_git_state_is_read_in_the_training_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the git commands run in the thread calling start(), the training thread, and never in the worker
    thread: a process started by the worker while the training thread forks the loader workers would hang until they
    exit, the workers inheriting the pipes it waits on.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the MLflow client and read_git_state.
    """
    git_state_recorder = GitStateRecorder()
    monkeypatch.setattr(target=mlflow_tracker, name="MlflowClient", value=UnavailableClient)
    monkeypatch.setattr(target=mlflow_tracker, name="read_git_state", value=git_state_recorder)
    tracker = MlflowTracker(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=None,
                            experiment_name="dummy_task", run_name="dummy_run", run_record_path=None)
    tracker.start(parameters={}, tags={"task": "dummy_task"}, resumed_iteration=None)
    tracker.close(status=FINISHED_STATUS)

    assert git_state_recorder.calling_threads == [threading.current_thread()]
    assert tracker.tags["git_commit"] == "abc" and "host" in tracker.tags


def test_unavailable_backend_never_blocks_the_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a backend failing on every request (locked database) never slows the training: 1000 calls return in
    well under a second, the close returns at once instead of waiting for the retries, and a single warning is printed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the MLflow client.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    monkeypatch.setattr(target=mlflow_tracker, name="MlflowClient", value=UnavailableClient)
    tracker = MlflowTracker(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=None,
                            experiment_name="dummy_task", run_name="dummy_run",
                            run_record_path=tmp_path / "record.json")
    tracker.start(parameters={"ExperimentConfiguration": {"batch_size": 4}}, tags={}, resumed_iteration=None)

    start_time = time.monotonic()
    for step in range(1000):
        tracker.log_metrics(metrics={"train/Total Loss": 1.0, "learning_rate": 0.01}, step=step)
    assert time.monotonic() - start_time < 1.0

    close_start_time = time.monotonic()
    tracker.close(status=FINISHED_STATUS)
    assert time.monotonic() - close_start_time < 2.0
    assert not tracker.worker_thread.is_alive() and tracker.get_run_id() is None
    assert capsys.readouterr().out.count("MLflow Unavailable") == 1
    assert not (tmp_path / "record.json").exists()


def test_hanging_backend_close_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a backend that does not answer never blocks the training nor its end: the calls return at once while
    the worker hangs, and the close gives up after its timeout with a warning.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the MLflow client and the close timeout.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    release_event = threading.Event()
    monkeypatch.setattr(target=mlflow_tracker, name="MlflowClient",
                        value=functools.partial(BlockedClient, release_event=release_event))
    monkeypatch.setattr(target=mlflow_tracker, name="CLOSE_TIMEOUT_SECONDS", value=0.5)
    tracker = MlflowTracker(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=None,
                            experiment_name="dummy_task", run_name="dummy_run", run_record_path=None)
    tracker.start(parameters={}, tags={}, resumed_iteration=None)

    start_time = time.monotonic()
    for step in range(1000):
        tracker.log_metrics(metrics={"train/Total Loss": 1.0}, step=step)
    tracker.close(status=FINISHED_STATUS)
    assert time.monotonic() - start_time < 1.5
    assert "MLflow Did Not Finish Writing" in capsys.readouterr().out

    # The released request fails, and the worker stops since the run is closed
    release_event.set()
    tracker.worker_thread.join(timeout=5.0)
    assert not tracker.worker_thread.is_alive()


def test_unreachable_server_never_blocks_the_training(monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the future server case on a closed port: the HTTP timeout and retries of MLflow are shortened unless the
    user set them, the calls return at once, the close returns within its timeout and a single warning is printed.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture isolating the MLflow HTTP environment variables.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    monkeypatch.delenv(name="MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv(name="MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    tracker = MlflowTracker(tracking_uri="http://127.0.0.1:9", artifact_root=None, experiment_name="dummy_task",
                            run_name="dummy_run", run_record_path=None)
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "10" and os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"
    tracker.start(parameters={"ExperimentConfiguration": {"batch_size": 4}}, tags={}, resumed_iteration=None)

    start_time = time.monotonic()
    for step in range(1000):
        tracker.log_metrics(metrics={"train/Total Loss": 1.0}, step=step)
    assert time.monotonic() - start_time < 1.0
    tracker.close(status=FINISHED_STATUS)
    assert time.monotonic() - start_time < mlflow_tracker.CLOSE_TIMEOUT_SECONDS
    assert capsys.readouterr().out.count("MLflow Unavailable") == 1


def test_full_queue_drops_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the events beyond the bound of the queue are dropped and counted without blocking, while the worker
    hangs on a backend that does not answer, and that the close reports them.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the MLflow client, the queue bound and the
            close timeout.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    release_event = threading.Event()
    monkeypatch.setattr(target=mlflow_tracker, name="MlflowClient",
                        value=functools.partial(BlockedClient, release_event=release_event))
    monkeypatch.setattr(target=mlflow_tracker, name="MAXIMUM_QUEUE_SIZE", value=5)
    monkeypatch.setattr(target=mlflow_tracker, name="CLOSE_TIMEOUT_SECONDS", value=0.5)
    tracker = MlflowTracker(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root=None,
                            experiment_name="dummy_task", run_name="dummy_run", run_record_path=None)
    tracker.start(parameters={}, tags={}, resumed_iteration=None)

    start_time = time.monotonic()
    for step in range(100):
        tracker.log_metrics(metrics={"train/Total Loss": 1.0}, step=step)
    assert time.monotonic() - start_time < 0.5
    assert tracker.dropped_event_count == 95
    tracker.close(status=FINISHED_STATUS)
    assert "95 Events Dropped" in capsys.readouterr().out

    release_event.set()
    tracker.worker_thread.join(timeout=5.0)
    assert not tracker.worker_thread.is_alive()


def test_rejected_writes_are_dropped(tracking_root: Path, monkeypatch: pytest.MonkeyPatch,
                                     capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the writes MLflow rejects are dropped instead of retried: the tracking continues, the run is still
    closed with its status, the first rejection is printed and the close reports their number, without any
    unavailability warning.

    Args:
        tracking_root (Path): The experiments root folder holding the empty database.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the MLflow client.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    monkeypatch.setattr(target=mlflow_tracker, name="MlflowClient", value=RejectingClient)
    tracker = build_tracker(tracking_root=tracking_root, run_folder=tracking_root / "dummy_run")
    tracker.start(parameters={"ExperimentConfiguration": {"batch_size": 4}}, tags={"task": "dummy_task"},
                  resumed_iteration=None)
    tracker.log_metrics(metrics={"train/Total Loss": 1.0}, step=2)
    tracker.close(status=FINISHED_STATUS)

    # One rejected batch of parameters, of tags and of metrics
    assert tracker.rejected_write_count == 3
    run = MlflowClient(tracking_uri=tracker.tracking_uri).get_run(run_id=tracker.get_run_id())
    assert run.info.status == FINISHED_STATUS and run.data.params == {}
    console_output = capsys.readouterr().out
    assert console_output.count("MLflow Rejected A Write") == 1 and "3 Writes Rejected By MLflow" in console_output
    assert "MLflow Unavailable" not in console_output
