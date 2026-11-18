"""
Tests of the MLflow tracking of the trainings: on the dummy task, a recording tracker checks the hooks of the general
trainer (run opened at the start of the loop, interval averages, test evaluations, configuration, training data
analysis, data record, input samples and Results.txt artifacts, run identifier in the checkpoints, status of the end of
the training).
"""

import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

from tests.trainers.test_base_trainer import build_dummy_trainer
from vision_training.tracking.experiment_tracker import (FAILED_STATUS, FINISHED_STATUS, RUN_RECORD_FILE_NAME,
                                                         DisabledTracker, ExperimentTracker)
from vision_training.trainers.input_samples import INPUT_SAMPLES_FOLDER_NAME, INPUT_SAMPLES_RECORD_FILE_NAME

# Factory of the trackers as called by the general trainer, replaced by the recording tracker
TRACKER_FACTORY_TARGET = "vision_training.trainers.base_trainer.create_experiment_tracker"


class RecordingTracker(ExperimentTracker):
    """
    Tracker recording every call of the trainer, to check its tracking hooks without MLflow.
    """

    # Run identifier given to the trainer, stored in its checkpoints
    RUN_ID = "recorded_run"

    def __init__(self) -> None:
        """
        Starts with no recorded call.
        """
        self.started_with = None
        self.logged_metrics: List[Tuple[int, Dict[str, float]]] = []
        self.best_metrics: List[Tuple[str, float, int, str]] = []
        self.artifacts: List[Tuple[Path, Optional[str], Optional[str]]] = []
        self.close_statuses: List[str] = []

    def start(self, parameters: Dict[str, Any], tags: Dict[str, str], resumed_iteration: Optional[int]) -> None:
        """
        Records the settings of the run.

        Args:
            parameters (Dict[str, Any]): The nested settings of the run.
            tags (Dict[str, str]): The tags of the run.
            resumed_iteration (Optional[int]): The checkpoint iteration a resumed training starts from.
        """
        self.started_with = {"parameters": parameters, "tags": tags, "resumed_iteration": resumed_iteration}

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Records metric values.

        Args:
            metrics (Dict[str, float]): The value of every metric.
            step (int): The training iteration of the values.
        """
        self.logged_metrics.append((step, dict(metrics)))

    def update_best_metric(self, metric_name: str, value: float, step: int, metric_prefix: str) -> None:
        """
        Records a value of the headline metric.

        Args:
            metric_name (str): The name of the headline metric.
            value (float): The value of the evaluated checkpoint.
            step (int): The training iteration of the evaluated checkpoint.
            metric_prefix (str): The prefix of the evaluated weights, 'test' or 'test_ema'.
        """
        self.best_metrics.append((metric_name, value, step, metric_prefix))

    def log_artifact(self, file_path: Path, artifact_folder: Optional[str],
                     artifact_name: Optional[str] = None) -> None:
        """
        Records a file sent as an artifact.

        Args:
            file_path (Path): The file.
            artifact_folder (Optional[str]): The folder of the artifact.
            artifact_name (Optional[str]): The name of the artifact.
        """
        self.artifacts.append((Path(file_path), artifact_folder, artifact_name))

    def get_run_id(self) -> Optional[str]:
        """
        Gives the recorded run identifier.

        Returns:
            Optional[str]: RUN_ID.
        """
        return self.RUN_ID

    def close(self, status: str) -> None:
        """
        Records the final status.

        Args:
            status (str): The final status of the run.
        """
        self.close_statuses.append(status)


def test_training_loop_feeds_the_tracker(tmp_path: Path) -> None:
    """
    Verifies the tracking hooks of the general trainer on the dummy task: the tracker is disabled until the training
    loop opens the run with the settings, derived values and tags of the run, then receives the training and
    validation averages, learning rate and throughput of every console summary, the test scalars of the evaluated
    checkpoint with the per-class ones and its headline metric, the configuration, the files of the analysis of the
    training data, the record of the data and the input samples (their record and the images of every split, in their
    split folders) at the start and Results.txt at the end, and is closed as finished, its run identifier being stored
    in the checkpoints. A resumed training opens the run as resumed and sends its configuration and its analysis files
    under the name of the resumption, but neither the record of the data nor the input samples, both unchanged.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    assert isinstance(trainer.experiment_tracker, DisabledTracker)

    # A file of the analysis of the training data, as the detection tasks write their anchor recall at launch
    assert trainer.training_data_analysis_files == {}
    analysis_file_path = trainer.experiment_folder / "analysis_example.csv"
    analysis_file_path.write_text(data="objects,count\nall,8\n")
    trainer.training_data_analysis_files = {"Analysis Example": analysis_file_path}
    recording_tracker = RecordingTracker()
    with patch(target=TRACKER_FACTORY_TARGET, return_value=recording_tracker) as factory_mock:
        trainer.run_training_loop()
    factory_mock.assert_called_once_with(tracking_configuration=trainer.tracking_configuration,
                                         tracking_root=tmp_path / "experiments", run_name="dummy_run",
                                         task="dummy_task", run_record_path=trainer.experiment_folder /
                                         RUN_RECORD_FILE_NAME)

    # Settings, derived values and tags of the run
    parameters = recording_tracker.started_with["parameters"]
    assert list(parameters) == ["ExperimentConfiguration", "Data", "Backbone", "Optimization", "Derived"]
    assert parameters["Derived"]["number_training_samples"] == 8
    assert parameters["Derived"]["iterations_per_epoch"] == 4
    assert parameters["Derived"]["number_parameters"] == sum(parameter.numel()
                                                             for parameter in trainer.model.parameters())
    assert recording_tracker.started_with["tags"] == {"task": "dummy_task", "run_name": "dummy_run",
                                                      "run_folder": str(trainer.experiment_folder.absolute()),
                                                      "pretrained_backbone": "None"}
    assert recording_tracker.started_with["resumed_iteration"] is None

    # Averages of the console summaries of iterations 2 and 4, validation included
    interval_metrics = [(step, metrics) for step, metrics in recording_tracker.logged_metrics
                        if "learning_rate" in metrics]
    display_names = trainer.tracked_metrics_mapping.values()
    expected_interval_keys = ({f"train/{display_name}" for display_name in display_names} |
                              {f"validation/{display_name}" for display_name in display_names} |
                              {"train/throughput", "train/interval_seconds", "learning_rate"})
    assert [step for step, _ in interval_metrics] == [2, 4]
    assert all(set(metrics) == expected_interval_keys for _, metrics in interval_metrics)
    assert interval_metrics[0][1]["learning_rate"] == pytest.approx(expected=0.01)

    # Evaluation of the checkpoint of iteration 4, with the per-class scalars of the per-class report
    test_metrics = [(step, metrics) for step, metrics in recording_tracker.logged_metrics
                    if "test/Accuracy" in metrics]
    assert [step for step, _ in test_metrics] == [4]
    assert any(metric_name.startswith("test_class/Recall/") for metric_name in test_metrics[0][1])
    assert recording_tracker.best_metrics == [("Accuracy", test_metrics[0][1]["test/Accuracy"], 4, "test")]

    # Configuration, analysis of the training data, record of the data and input samples (the images of the 8
    # training, 4 validation and 6 test samples, then their record) at the start, Results.txt at the end, then the run
    # closed as finished
    input_samples_folder = trainer.experiment_folder / INPUT_SAMPLES_FOLDER_NAME
    input_sample_artifacts = [(input_samples_folder / split_name / f"sample_{position:02d}.png",
                               f"{INPUT_SAMPLES_FOLDER_NAME}/{split_name}", None)
                              for split_name, number_samples in [("Test", 6), ("Training", 8), ("Validation", 4)]
                              for position in range(number_samples)]
    assert recording_tracker.artifacts == [(trainer.configuration_path, "configuration", None),
                                           (analysis_file_path, "analysis", None),
                                           (trainer.data_record_path, "data", None), *input_sample_artifacts,
                                           (input_samples_folder / INPUT_SAMPLES_RECORD_FILE_NAME,
                                            INPUT_SAMPLES_FOLDER_NAME, None),
                                           (trainer.experiment_folder / "Results.txt", None, None)]
    assert recording_tracker.close_statuses == [FINISHED_STATUS]
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                            weights_only=True)
    assert checkpoint["mlflow_run_id"] == RecordingTracker.RUN_ID

    # A resumed training, from the checkpoint of iteration 5
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=7, view_input_images=False,
                                          number_sample_images=0)
    resumed_trainer.training_data_analysis_files = {"Analysis Example": analysis_file_path}
    resumed_tracker = RecordingTracker()
    with patch(target=TRACKER_FACTORY_TARGET, return_value=resumed_tracker):
        resumed_trainer.run_training_loop()
    assert resumed_tracker.started_with["resumed_iteration"] == 5
    assert resumed_tracker.artifacts[:2] == [
        (resumed_trainer.configuration_path, "configuration", "training_configuration_resume_5.yaml"),
        (analysis_file_path, "analysis", "analysis_example_resume_5.csv")]

    # The configuration and the data did not change, so neither the record of the data nor the input samples, kept as
    # they were, are sent again
    assert all(artifact_folder != "data" and not str(artifact_folder).startswith(INPUT_SAMPLES_FOLDER_NAME)
               for _, artifact_folder, _ in resumed_tracker.artifacts)
    assert resumed_tracker.close_statuses == [FINISHED_STATUS]


@pytest.mark.parametrize(argnames=["patched_method", "raised_error", "expected_status", "propagated_error"],
                         argvalues=[("save_model", RuntimeError("disk full"), FAILED_STATUS, RuntimeError)])
def test_run_status_follows_the_end_of_the_training(tmp_path: Path, patched_method: str,
                                                    raised_error: BaseException | type, expected_status: str,
                                                    propagated_error: Optional[type]) -> None:
    """
    Verifies that the MLflow run is closed once with the status of the end of the training: FAILED when the final save
    fails, the error stopping the final save being raised again once the run is closed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        patched_method (str): The method of the trainer made to fail.
        raised_error (BaseException | type): The error it raises on every call.
        expected_status (str): The expected final status of the run.
        propagated_error (Optional[type]): The error expected out of the training loop, None when the loop handles it.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    recording_tracker = RecordingTracker()
    with patch(target=TRACKER_FACTORY_TARGET, return_value=recording_tracker), \
            patch.object(target=trainer, attribute=patched_method, side_effect=raised_error):
        if propagated_error is None:
            trainer.run_training_loop(maximum_consecutive_errors=2)
        else:
            with pytest.raises(expected_exception=propagated_error):
                trainer.run_training_loop(maximum_consecutive_errors=2)
    assert recording_tracker.close_statuses == [expected_status]
