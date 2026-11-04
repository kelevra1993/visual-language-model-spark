"""
Tests of vision_training/tracking/experiment_tracker.py: the factory gives a disabled tracker when the tracking is off,
when mlflow is not installed or when the MLflow tracker cannot be built, and the run record of a run folder is written
and read back, an invalid record being ignored.
"""

import pytest

from pathlib import Path
from unittest.mock import patch

from vision_training.tracking.experiment_tracker import (RUN_RECORD_FILE_NAME, DisabledTracker,
                                                         create_experiment_tracker, read_run_record, write_run_record)

# Tracking settings of the tests, on with the local database
ACTIVATED_TRACKING_CONFIGURATION = {"activated": True, "tracking_uri": None, "experiment_name": None}


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
