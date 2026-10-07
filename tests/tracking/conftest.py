"""
Fixtures of the tracking tests of tests/tracking: an empty MLflow sqlite database created once per test session and
copied into the experiments root folder of every test that needs the local backend, creating the tables of a new
database taking several seconds.
"""

import pytest
import shutil

from mlflow.tracking import MlflowClient
from pathlib import Path

from vision_training.tracking.experiment_tracker import DATABASE_FILE_NAME


@pytest.fixture(scope="session")
def empty_database_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Creates an empty MLflow sqlite database once for the whole test session, its tables being created by the first
    request of a client.

    Args:
        tmp_path_factory (pytest.TempPathFactory): The pytest factory of the session temporary directories.

    Returns:
        Path: The empty database file.
    """
    database_path = tmp_path_factory.mktemp(basename="empty_tracking_database") / DATABASE_FILE_NAME
    MlflowClient(tracking_uri=f"sqlite:///{database_path}").search_experiments()
    return database_path


@pytest.fixture
def tracking_root(tmp_path: Path, empty_database_path: Path) -> Path:
    """
    Gives the experiments root folder of a test, holding a copy of the empty MLflow database used as the local backend
    of its trackers.

    Args:
        tmp_path (Path): The pytest temporary directory of the test.
        empty_database_path (Path): The empty database of the session.

    Returns:
        Path: The experiments root folder.
    """
    experiments_folder = tmp_path / "experiments"
    experiments_folder.mkdir()
    shutil.copyfile(src=empty_database_path, dst=experiments_folder / DATABASE_FILE_NAME)
    return experiments_folder
