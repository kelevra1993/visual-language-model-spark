"""
Tests of vision_training/tracking/experiment_tracker.py: the disabled tracker does nothing and never raises, the
factory gives a disabled tracker when the tracking is off, when mlflow is not installed or when the MLflow tracker
cannot be built, the tracking URI points to the local database of the experiments root folder unless a server is set,
and the run record of a run folder is written and read back, an invalid record being ignored.
"""

import pytest

from pathlib import Path
from unittest.mock import patch

from vision_training.tracking.experiment_tracker import (ARTIFACTS_FOLDER_NAME, DATABASE_FILE_NAME, FINISHED_STATUS,
                                                         RUN_RECORD_FILE_NAME, DisabledTracker,
                                                         create_experiment_tracker, read_run_record,
                                                         resolve_tracking_uri, write_run_record)

# Tracking settings of the tests, on with the local database
ACTIVATED_TRACKING_CONFIGURATION = {"activated": True, "tracking_uri": None, "experiment_name": None,
                                    "log_artifacts": True}


def test_disabled_tracker_does_nothing(tmp_path: Path) -> None:
    """
    Verifies that every method of the disabled tracker can be called in any order without raising, and that it gives
    no run identifier.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    tracker = DisabledTracker()
    tracker.log_metrics(metrics={"train/Total Loss": 1.0}, step=1)
    tracker.start(parameters={"ExperimentConfiguration": {"batch_size": 4}}, tags={"task": "dummy_task"},
                  resumed_iteration=None)
    tracker.update_best_metric(metric_name="Accuracy", value=0.5, step=2, metric_prefix="test")
    tracker.log_artifact(file_path=tmp_path / "missing.txt", artifact_folder=None)
    tracker.set_tags(tags={"best_iteration": "2"})
    tracker.close(status=FINISHED_STATUS)
    tracker.close(status=FINISHED_STATUS)
    assert tracker.get_run_id() is None
    assert list(tmp_path.iterdir()) == []


def test_factory_gives_a_disabled_tracker_without_tracking(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the factory gives a disabled tracker when the tracking is off, and when it is on but mlflow is not
    installed or the MLflow tracker cannot be built, with one warning in the last two cases and without creating any
    file.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    disabled_tracker = create_experiment_tracker(
        tracking_configuration={**ACTIVATED_TRACKING_CONFIGURATION, "activated": False}, tracking_root=tmp_path,
        run_name="dummy_run", task="dummy_task", run_record_path=tmp_path / RUN_RECORD_FILE_NAME)
    assert isinstance(disabled_tracker, DisabledTracker) and capsys.readouterr().out == ""

    # The mlflow package missing from the environment
    with patch(target="vision_training.tracking.experiment_tracker.importlib.util.find_spec", return_value=None):
        missing_package_tracker = create_experiment_tracker(
            tracking_configuration=ACTIVATED_TRACKING_CONFIGURATION, tracking_root=tmp_path, run_name="dummy_run",
            task="dummy_task", run_record_path=tmp_path / RUN_RECORD_FILE_NAME)
    assert isinstance(missing_package_tracker, DisabledTracker)
    assert "'mlflow' Package Is Not Installed" in capsys.readouterr().out

    # A broken install, failing on import
    with patch(target="vision_training.tracking.experiment_tracker.importlib.import_module",
               side_effect=ImportError("broken mlflow")):
        broken_package_tracker = create_experiment_tracker(
            tracking_configuration=ACTIVATED_TRACKING_CONFIGURATION, tracking_root=tmp_path, run_name="dummy_run",
            task="dummy_task", run_record_path=tmp_path / RUN_RECORD_FILE_NAME)
    assert isinstance(broken_package_tracker, DisabledTracker)
    assert "MLflow Tracking Could Not Start" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_tracking_uri_is_the_local_database_unless_a_server_is_set(tmp_path: Path) -> None:
    """
    Verifies that the runs are tracked in the sqlite database of the experiments root folder, with the artifacts next
    to it, when no server is set, and in the server otherwise, which stores the artifacts itself.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    tracking_uri, artifact_root = resolve_tracking_uri(tracking_configuration=ACTIVATED_TRACKING_CONFIGURATION,
                                                       tracking_root=tmp_path)
    assert tracking_uri == f"sqlite:///{tmp_path.absolute() / DATABASE_FILE_NAME}"
    assert artifact_root == (tmp_path.absolute() / ARTIFACTS_FOLDER_NAME).as_uri()

    server_configuration = {**ACTIVATED_TRACKING_CONFIGURATION, "tracking_uri": "http://192.168.1.20:5000"}
    assert resolve_tracking_uri(tracking_configuration=server_configuration, tracking_root=tmp_path) == (
        "http://192.168.1.20:5000", None)


def test_run_record_is_written_and_read_back(tmp_path: Path) -> None:
    """
    Verifies that the run record of a run folder is read back as written, and that a missing, unreadable or incomplete
    record is ignored.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    run_record_path = tmp_path / RUN_RECORD_FILE_NAME
    assert read_run_record(run_record_path=run_record_path) is None
    assert read_run_record(run_record_path=None) is None

    run_record = {"tracking_uri": "sqlite:///mlflow.db", "experiment_id": "1", "run_id": "abc"}
    write_run_record(run_record_path=run_record_path, run_record=run_record)
    assert read_run_record(run_record_path=run_record_path) == run_record

    # Invalid records: not JSON, a missing key, a key of another type
    for invalid_text in ["{not json", '{"tracking_uri": "sqlite:///mlflow.db", "run_id": "abc"}',
                         '{"tracking_uri": "sqlite:///mlflow.db", "experiment_id": 1, "run_id": "abc"}']:
        run_record_path.write_text(data=invalid_text)
        assert read_run_record(run_record_path=run_record_path) is None
