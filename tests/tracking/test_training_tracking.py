"""
Tests of the MLflow tracking of the trainings and of the standalone evaluations: on the dummy task, a recording tracker
checks the hooks of the general trainer (run opened at the start of the loop, interval averages, test evaluations,
configuration, training data analysis, data record and Results.txt artifacts, run identifier in the checkpoints, status
of the end of the
training); on a synthetic classification dataset, a training, its resumption and a standalone evaluation of its
checkpoints are written to the local MLflow database and read back.
"""

import pytest
import torch
import yaml

from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

from tests.trainers.test_base_trainer import build_dummy_trainer
from tests.trainers.test_classification_trainer import SMOKE_IMAGE_SIZE, build_trainer, write_smoke_configuration
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.evaluation.checkpoint_evaluation import evaluate_checkpoints
from vision_training.models.model_transfer import disable_pretrained_settings
from vision_training.tracking.experiment_tracker import (FAILED_STATUS, FINISHED_STATUS, KILLED_STATUS,
                                                         RUN_RECORD_FILE_NAME, DisabledTracker, ExperimentTracker,
                                                         read_run_record)
from vision_training.trainers.classification_trainer import ClassificationTrainer

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
        self.tags: Dict[str, str] = {}
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

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Records tags.

        Args:
            tags (Dict[str, str]): The value of every tag.
        """
        self.tags.update(tags)

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


def enable_tracking(configuration_path: Path) -> None:
    """
    Turns the MLflow tracking on in a configuration file, on the local database of its experiments root folder.

    Args:
        configuration_path (Path): The configuration file, modified in place.
    """
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["ExperimentConfiguration"]["tracking"]["activated"] = True
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))


def read_metric_steps(client: MlflowClient, run_id: str, metric_key: str) -> List[int]:
    """
    Reads the steps at which a metric of a run was logged.

    Args:
        client (MlflowClient): The client of the backend.
        run_id (str): The run identifier.
        metric_key (str): The MLflow name of the metric.

    Returns:
        List[int]: The sorted steps.
    """
    return sorted(metric.step for metric in client.get_metric_history(run_id=run_id, key=metric_key))


def test_training_loop_feeds_the_tracker(tmp_path: Path) -> None:
    """
    Verifies the tracking hooks of the general trainer on the dummy task: the tracker is disabled until the training
    loop opens the run with the settings, derived values and tags of the run, then receives the training and
    validation averages, learning rate and throughput of every console summary, the test scalars of the evaluated
    checkpoint with the per-class ones and its headline metric, the configuration and the files of the analysis of
    the training data at the start and Results.txt at the end, and is closed as finished, its run identifier being
    stored in the checkpoints. A resumed training opens the run as resumed and sends its configuration and its analysis
    files under the name of the resumption.

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

    # Configuration, analysis of the training data and record of the data at the start, Results.txt at the end, then
    # the run closed as finished
    assert recording_tracker.artifacts == [(trainer.configuration_path, "configuration", None),
                                           (analysis_file_path, "analysis", None),
                                           (trainer.data_record_path, "data", None),
                                           (trainer.experiment_folder / "Results.txt", None, None)]
    assert recording_tracker.close_statuses == [FINISHED_STATUS]
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                            weights_only=True)
    assert checkpoint["mlflow_run_id"] == RecordingTracker.RUN_ID

    # A resumed training, from the checkpoint of iteration 5
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                          number_sample_images=0)
    resumed_trainer.training_data_analysis_files = {"Analysis Example": analysis_file_path}
    resumed_tracker = RecordingTracker()
    with patch(target=TRACKER_FACTORY_TARGET, return_value=resumed_tracker):
        resumed_trainer.run_training_loop()
    assert resumed_tracker.started_with["resumed_iteration"] == 5
    assert resumed_tracker.artifacts[:2] == [
        (resumed_trainer.configuration_path, "configuration", "training_configuration_resume_5.yaml"),
        (analysis_file_path, "analysis", "analysis_example_resume_5.csv")]

    # The data did not change, so its record is not sent again
    assert all(artifact_folder != "data" for _, artifact_folder, _ in resumed_tracker.artifacts)
    assert resumed_tracker.close_statuses == [FINISHED_STATUS]


@pytest.mark.parametrize(argnames=["patched_method", "raised_error", "expected_status", "propagated_error"],
                         argvalues=[("forward_pass", KeyboardInterrupt, KILLED_STATUS, None),
                                    ("forward_pass", RuntimeError("broken batch"), FAILED_STATUS, None),
                                    ("save_model", RuntimeError("disk full"), FAILED_STATUS, RuntimeError),
                                    ("save_model", KeyboardInterrupt, KILLED_STATUS, KeyboardInterrupt)])
def test_run_status_follows_the_end_of_the_training(tmp_path: Path, patched_method: str,
                                                    raised_error: BaseException | type, expected_status: str,
                                                    propagated_error: Optional[type]) -> None:
    """
    Verifies that the MLflow run is closed once with the status of the end of the training: KILLED when the user stops
    the loop (Ctrl+C) or the final save, FAILED after too many consecutive errors or when the final save fails, the
    error stopping the final save being raised again once the run is closed.

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


def test_classification_training_and_evaluation_are_tracked(tmp_path: Path, tracking_root: Path,
                                                            capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the tracking of a real classification training in the local MLflow database: the run of the training
    holds its settings, tags, interval and test metrics, best headline metric and evaluation artifacts and is finished;
    its resumption continues the same run and records the changed number of iterations; a standalone evaluation of its
    checkpoints gets its own run tagged with the training run, which stays unchanged.

    Args:
        tmp_path (Path): The pytest temporary directory.
        tracking_root (Path): The experiments root folder of the smoke configuration, holding the empty database.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 16, "validation": 8, "test": 6}, maximum_objects=1,
                             minimum_frames=1, maximum_frames=1, number_workers=2, task="classification")

    # Training of 5 iterations: console summaries at 2 and 4, evaluation of the checkpoint of iteration 4
    configuration_path = write_smoke_configuration(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                   number_iterations=5, head_settings={}, evaluation_settings={})
    enable_tracking(configuration_path=configuration_path)
    trainer = build_trainer(configuration_path=configuration_path)
    assert trainer.experiment_configuration["experiment_folder"] == tracking_root
    trainer.run_training_loop()

    run_record = read_run_record(run_record_path=trainer.experiment_folder / RUN_RECORD_FILE_NAME)
    client = MlflowClient(tracking_uri=run_record["tracking_uri"])
    training_run = client.get_run(run_id=run_record["run_id"])
    training_run_id = training_run.info.run_id
    assert training_run.info.status == FINISHED_STATUS and training_run.info.run_name == trainer.run_name
    assert client.get_experiment(experiment_id=training_run.info.experiment_id).name == "classification"
    assert training_run.data.params["ExperimentConfiguration.batch_size"] == "4"
    assert training_run.data.params["Derived.number_training_samples"] == "16"
    assert training_run.data.tags["task"] == "classification" and training_run.data.tags["best_iteration"] == "4"

    # Interval averages, test metrics and the best headline metric at their iterations
    for metric_key in ["train/Total Loss", "validation/Total Loss", "learning_rate", "train/throughput"]:
        assert read_metric_steps(client=client, run_id=training_run_id, metric_key=metric_key) == [2, 4]
    for metric_key in ["test/Accuracy", "test/best_Accuracy", f"test_class/Recall/{trainer.class_names[0]}"]:
        assert read_metric_steps(client=client, run_id=training_run_id, metric_key=metric_key) == [4]

    # Artifacts: configuration, record of the data, small evaluation files, Results.txt and the original metric names
    assert sorted(artifact.path for artifact in client.list_artifacts(run_id=training_run_id)) == [
        "Results.txt", "configuration", "data", "evaluation", "metric_names.json"]
    assert sorted(artifact.path for artifact in client.list_artifacts(run_id=training_run_id,
                                                                      path="evaluation/iteration_4")) == [
        "evaluation/iteration_4/class_metrics.csv", "evaluation/iteration_4/class_thresholds.json",
        "evaluation/iteration_4/confusion_matrix.png"]
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                            weights_only=True)
    assert checkpoint["mlflow_run_id"] == training_run_id

    # Resumption from the checkpoint of iteration 5 for 2 more iterations: the same run, continued
    resumed_configuration_path = write_smoke_configuration(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                           number_iterations=2, head_settings={},
                                                           evaluation_settings={})
    enable_tracking(configuration_path=resumed_configuration_path)
    resumed_trainer = build_trainer(configuration_path=resumed_configuration_path)
    resumed_trainer.run_training_loop()
    resumed_run = client.get_run(run_id=training_run_id)
    assert read_run_record(run_record_path=trainer.experiment_folder / RUN_RECORD_FILE_NAME)["run_id"] == (
        training_run_id)
    assert resumed_run.info.status == FINISHED_STATUS
    assert (resumed_run.data.tags["resume_count"], resumed_run.data.tags["last_resumed_iteration"]) == ("1", "5")
    assert resumed_run.data.tags["resumed_5.ExperimentConfiguration.number_iterations"] == "2"
    assert read_metric_steps(client=client, run_id=training_run_id, metric_key="train/Total Loss") == [2, 4, 6]
    assert "configuration/training_configuration_resume_5.yaml" in [
        artifact.path for artifact in client.list_artifacts(run_id=training_run_id, path="configuration")]
    assert [artifact.path for artifact in client.list_artifacts(run_id=training_run_id, path="data")] == [
        "data/data.json"]

    # Standalone evaluation of two checkpoints, built like scripts/evaluation/evaluate_checkpoint.py
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=trainer.configuration_path)
    experiment_configuration["experiment_folder"] = tracking_root / "evaluations"
    disable_pretrained_settings(model_configuration=model_configuration)
    evaluation_trainer = ClassificationTrainer(experiment_configuration=experiment_configuration,
                                               model_configuration=model_configuration,
                                               configuration_path=trainer.configuration_path)
    evaluate_checkpoints(trainer=evaluation_trainer, experiment_folder=trainer.experiment_folder, iterations=[4, 7])

    # Its own run in the experiment of the training, linked to the training run
    runs = client.search_runs(experiment_ids=[training_run.info.experiment_id])
    assert len(runs) == 2
    evaluation_run = next(run for run in runs if run.info.run_id != training_run_id)
    assert evaluation_run.info.run_name == f"{trainer.experiment_folder.name}-evaluation"
    assert evaluation_run.info.status == FINISHED_STATUS
    assert evaluation_run.data.tags["evaluation_of"] == training_run_id
    assert evaluation_run.data.tags["evaluated_experiment_folder"] == str(trainer.experiment_folder.absolute())
    assert evaluation_run.data.tags["evaluated_iterations"] == "4, 7"
    assert read_metric_steps(client=client, run_id=evaluation_run.info.run_id, metric_key="test/Accuracy") == [4, 7]
    assert "evaluation_summary.csv" in [artifact.path for artifact in client.list_artifacts(
        run_id=evaluation_run.info.run_id)]

    # The training run is only read by the evaluation
    assert read_metric_steps(client=client, run_id=training_run_id, metric_key="test/Accuracy") == [4]
    console_output = capsys.readouterr().out
    for unexpected_message in ["Error encountered during iteration", "Critical Error", "MLflow Unavailable",
                               "MLflow Rejected", "MLflow Did Not Finish"]:
        assert unexpected_message not in console_output
