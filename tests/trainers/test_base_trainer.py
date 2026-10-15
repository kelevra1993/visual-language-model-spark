"""
Tests of BaseTrainer of vision_training/trainers/base_trainer.py on the dummy task: a model iteration sums the losses
and tracks the metrics, training saves, evaluates and resumes from its checkpoints, the interval averages of a run
resumed between two summaries cover the iterations it really ran, a resumed run reports and keeps the configuration
and the data it changed, the final checkpoint is evaluated on the weights it holds, the evaluation draws the fixed
sample images, the moving average of the weights is saved, evaluated next to the trained weights and resumed, and the
input viewer creates no run folder.
"""

import copy
import cv2
import json
import pytest
import torch

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Callable, Dict, Optional, Tuple
from unittest.mock import patch

from tests.fixtures.dummy_task import DummyDataset, DummyTrainer, build_dummy_configurations
from tests.fixtures.evaluation_checks import read_results_blocks
from vision_training.configuration_processing.configuration_changes import MISSING_SETTING
from vision_training.core.contracts import EvaluationResults
from vision_training.trainers.checkpointing import CHECKPOINT_FORMAT_VERSION, EXPONENTIAL_MOVING_AVERAGE_STATE_KEY
from vision_training.trainers.metric_tracking import INTERVAL_BATCHES_KEY
from vision_training.utilities.os_utilities import format_path_link


class EvaluationRecorder:
    """
    Replaces the test evaluation of a trainer to record the weights of the model at every evaluated iteration, then
    runs the original evaluation, so that a test can compare the evaluated weights with the saved checkpoints.
    """

    def __init__(self, trainer: DummyTrainer) -> None:
        """
        Keeps the trainer and its original evaluation method.

        Args:
            trainer (DummyTrainer): The trainer whose evaluations are recorded.
        """
        self.trainer = trainer
        self.original_evaluation: Callable[[int], Tuple[EvaluationResults, Optional[EvaluationResults]]] = (
            trainer.run_test_evaluation)
        self.evaluated_weights: Dict[int, Dict[str, torch.Tensor]] = {}

    def __call__(self, iteration: int) -> Tuple[EvaluationResults, Optional[EvaluationResults]]:
        """
        Records the weights of the evaluated model, then evaluates it.

        Args:
            iteration (int): The training iteration of the evaluated checkpoint.

        Returns:
            Tuple[EvaluationResults, Optional[EvaluationResults]]: The results of the original evaluation, for the
                trained and the averaged weights.
        """
        self.evaluated_weights[iteration] = copy.deepcopy(self.trainer.model.state_dict())
        return self.original_evaluation(iteration=iteration)


def build_dummy_trainer(temporary_directory: Path, number_iterations: int, view_input_images: bool,
                        number_sample_images: int, configuration_text: str = "dummy: true\n",
                        moving_average_activated: bool = False, number_training_samples: int = 8) -> DummyTrainer:
    """
    Builds the trainer of the dummy task in a temporary directory. Its test split has 6 samples but only 4 distinct
    ones, its last batch repeating the first two samples, which must be evaluated only once.

    Args:
        temporary_directory (Path): The pytest temporary directory, receiving the configuration and the run folder.
        number_iterations (int): The number of training iterations of the run.
        view_input_images (bool): Whether the input viewer runs, which stops at its first samples and exits.
        number_sample_images (int): The number of fixed test samples drawn at every evaluation, 0 for none.
        configuration_text (str): The YAML text of the launched configuration file, copied into the run folder and
            compared with the saved one when the run resumes.
        moving_average_activated (bool): Whether the run keeps a moving average of its weights, at a decay of 0.5.
        number_training_samples (int): The number of samples of the training split. Defaults to 8.

    Returns:
        DummyTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    configuration_path = temporary_directory / "configuration.yaml"
    configuration_path.write_text(data=configuration_text)
    experiment_configuration, model_configuration = build_dummy_configurations(
        experiment_folder=temporary_directory / "experiments", number_iterations=number_iterations)
    experiment_configuration["diagnosis_modes"]["view_input_images"] = view_input_images
    experiment_configuration["evaluation"]["number_sample_images"] = number_sample_images
    model_configuration["Optimization"]["exponential_moving_average"] = {"activated": moving_average_activated,
                                                                         "decay": 0.5}
    datasets = {"train": DummyDataset(sample_identifiers=list(range(number_training_samples)), seed=0),
                "validation": DummyDataset(sample_identifiers=list(range(4)), seed=1),
                "test": DummyDataset(sample_identifiers=[0, 1, 2, 3, 0, 1], seed=2)}
    return DummyTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                        configuration_path=configuration_path, datasets=datasets)


def test_model_iteration_sums_the_losses_and_tracks_the_metrics(tmp_path: Path) -> None:
    """
    Verifies the generic step of the trainer: every '_loss' output of the model is summed into the total loss, and the
    losses and batch metrics are added to the sums of the trackers, which count the batch.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=1, view_input_images=False,
                                  number_sample_images=0)
    batch = next(iter(trainer.train_dataloader))
    tracker_dictionary = trainer.metric_tracker.create_trackers()

    with torch.no_grad():
        total_loss, model_outputs = trainer.run_model_iteration(batch=batch, writer=None, iteration=1,
                                                                tracker_dictionary=tracker_dictionary,
                                                                apply_augmentation=False)

    assert torch.isclose(input=total_loss, other=model_outputs["first_loss"] + model_outputs["second_loss"])
    assert tracker_dictionary[INTERVAL_BATCHES_KEY] == 1
    assert tracker_dictionary["total_loss"] == pytest.approx(expected=total_loss.item())
    assert tracker_dictionary["first_loss"] == pytest.approx(expected=model_outputs["first_loss"].item())
    assert tracker_dictionary["second_loss"] == pytest.approx(expected=model_outputs["second_loss"].item())
    assert 0.0 <= tracker_dictionary["accuracy"] <= 1.0


def test_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole generic pipeline on the dummy task: 5 iterations save the intermediate, evaluated and final
    checkpoints with the task and run name, evaluate every test sample exactly once into Results.txt, and write the
    three TensorBoard folders; a second run with the same settings resumes from the last checkpoint with exactly its
    weights and continues the registry. No training step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0)
    assert trainer.start_iteration == 1
    trainer.run_training_loop()

    # Checkpoints at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run)
    weights_directory = trainer.weights_directory
    assert sorted(path.name for path in weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                 "Iteration_5"]
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_5"'
    checkpoint = torch.load(f=weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                            weights_only=True)
    assert (checkpoint["task"], checkpoint["run_name"], checkpoint["iteration"]) == ("dummy_task", "dummy_run", 5)
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION and checkpoint["mlflow_run_id"] is None

    # The test split has 6 samples but only 4 distinct ones, each evaluated exactly once
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [(4, "4 Test Samples")]
    assert "Accuracy" in results_blocks[0]["text"] and "Per Class Recall" in results_blocks[0]["text"]
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train", "Validation"]

    # A second run with the same settings resumes from iteration 5 with exactly its weights
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                          number_sample_images=0)
    assert resumed_trainer.start_iteration == 6
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=checkpoint["model_state"][parameter_name])

    # The resumed run trains iterations 6 (intermediate save) and 7 (end of the run)
    resumed_trainer.run_training_loop()
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_7"'
    assert {"Iteration_6", "Iteration_7"} <= {path.name for path in weights_directory.glob("Iteration_*")}

    # The training loop catches errors and keeps going, so make sure that no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_resumed_interval_averages_the_iterations_it_ran(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a run resumed between two summaries (saved at iteration 3, a summary every 2 iterations) averages
    its first interval over the single iteration 4 it ran instead of over information_dump iterations, and its next
    interval over iterations 5 and 6, for the console summary and the MLflow values, the validation averaging its
    batches the same way.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    build_dummy_trainer(temporary_directory=tmp_path, number_iterations=3, view_input_images=False,
                        number_sample_images=0).run_training_loop()
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=3, view_input_images=False,
                                          number_sample_images=0)
    assert resumed_trainer.start_iteration == 4
    capsys.readouterr()

    # Record the metrics of every batch written to TensorBoard and the interval averages sent to MLflow
    with patch.object(target=resumed_trainer.metric_tracker, attribute="log_to_tensorboard",
                      wraps=resumed_trainer.metric_tracker.log_to_tensorboard) as tensorboard_mock, \
            patch.object(target=resumed_trainer, attribute="log_interval_metrics",
                         wraps=resumed_trainer.log_interval_metrics) as interval_mock:
        resumed_trainer.run_training_loop()

    # Total loss of every training and validation batch, keyed by iteration
    batch_losses = {"training": {}, "validation": {}}
    for tensorboard_call in tensorboard_mock.call_args_list:
        is_training_batch = tensorboard_call.kwargs["writer"] is resumed_trainer.training_writer
        batch_losses["training" if is_training_batch else "validation"][tensorboard_call.kwargs["iteration"]] = \
            tensorboard_call.kwargs["metric_dictionary"]["total_loss"].item()
    assert sorted(batch_losses["training"]) == [4, 5, 6] and sorted(batch_losses["validation"]) == [4, 6]

    # Summaries at iterations 4 (iteration 4 alone) and 6 (iterations 5 and 6), one validation batch each
    interval_calls = {interval_call.kwargs["iteration"]: interval_call.kwargs
                      for interval_call in interval_mock.call_args_list}
    assert sorted(interval_calls) == [4, 6]
    for iteration, interval_iterations in [(4, [4]), (6, [5, 6])]:
        training_trackers = interval_calls[iteration]["training_trackers"]
        assert training_trackers[INTERVAL_BATCHES_KEY] == len(interval_iterations)
        assert training_trackers["total_loss"] == pytest.approx(expected=sum(
            batch_losses["training"][interval_iteration] for interval_iteration in interval_iterations) /
            len(interval_iterations))
        validation_trackers = interval_calls[iteration]["validation_trackers"]
        assert validation_trackers[INTERVAL_BATCHES_KEY] == 1
        assert validation_trackers["total_loss"] == pytest.approx(expected=batch_losses["validation"][iteration])

    # The console summaries count the iterations and samples of their own interval
    console_output = capsys.readouterr().out
    assert "Interval Of 1 Iterations Took" in console_output and "Interval Of 2 Iterations Took" in console_output
    assert "Interval Images: 2 " in console_output and "Interval Images: 4 " in console_output


def test_resumed_run_reports_and_keeps_its_changed_configuration(tmp_path: Path,
                                                                  capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the record of the settings changed between two starts of a run: a new run compares nothing, a run resumed
    with a changed configuration prints every changed, added and removed setting and keeps the previous configuration
    as 'training_configuration_until_iteration_<n>.yaml' before the copy of the new one replaces it, a second resume
    from the same checkpoint keeps that first file, and a run resumed with an unchanged configuration archives nothing.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    first_configuration_text = "dummy: true\nlearning_rate: 0.001\nremoved_setting: 1\n"
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                  number_sample_images=0, configuration_text=first_configuration_text)
    assert trainer.configuration_changes is None
    trainer.run_training_loop()
    capsys.readouterr()

    # Resumed from iteration 2 with a changed learning rate, an added and a removed setting
    second_configuration_text = "dummy: true\nlearning_rate: 0.0005\nadded_setting: null\n"
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                          number_sample_images=0, configuration_text=second_configuration_text)
    archived_configuration_path = trainer.experiment_folder / "training_configuration_until_iteration_2.yaml"
    assert resumed_trainer.configuration_changes == {
        "resumed_iteration": 2, "archived_configuration_path": archived_configuration_path,
        "changed_settings": [("learning_rate", 0.001, 0.0005), ("added_setting", MISSING_SETTING, None),
                             ("removed_setting", 1, MISSING_SETTING)]}
    assert archived_configuration_path.read_text() == first_configuration_text
    assert resumed_trainer.configuration_path.read_text() == second_configuration_text
    console_lines = capsys.readouterr().out.splitlines()
    for expected_line in [f"- Previous Configuration  : {format_path_link(path=archived_configuration_path)}",
                          "- Changed Settings        : Since Iteration 2",
                          "    - learning_rate   : 0.001 -> 0.0005", "    - added_setting   : <absent> -> null",
                          "    - removed_setting : 1 -> <absent>"]:
        assert any(expected_line in console_line for console_line in console_lines), expected_line

    # Resumed again from the same checkpoint: the configuration that trained the run up to it is kept
    build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                        number_sample_images=0, configuration_text="dummy: true\nlearning_rate: 0.0001\n")
    assert archived_configuration_path.read_text() == first_configuration_text

    # Resumed with the configuration saved by the previous start: nothing changed, nothing archived
    capsys.readouterr()
    unchanged_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                            number_sample_images=0,
                                            configuration_text="dummy: true\nlearning_rate: 0.0001\n")
    assert unchanged_trainer.configuration_changes == {"resumed_iteration": 2, "changed_settings": [],
                                                       "archived_configuration_path": None}
    assert "- Changed Settings        : None Since Iteration 2" in capsys.readouterr().out
    assert sorted(path.name for path in trainer.experiment_folder.glob("training_configuration*.yaml")) == [
        "training_configuration.yaml", "training_configuration_until_iteration_2.yaml"]


def test_run_records_its_data_and_keeps_the_previous_data(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the record of the data of a run ('data.json'): written by a new run with the images of every split and no
    class names (the dummy task has none), kept with its date by a resume with the same data, kept as
    'data_until_iteration_<n>.json' by a resume whose training split changed before the new record replaces it, and that
    file kept by a second resume from the same checkpoint.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                  number_sample_images=0)
    data_record_path = trainer.experiment_folder / "data.json"
    first_record = json.loads(s=data_record_path.read_text())
    assert trainer.data_record_changes is None
    assert first_record["class_names"] is None
    assert {split_name: split_record["images"] for split_name, split_record in first_record["splits"].items()} == {
        "train": [f"sample_{index}" for index in range(8)], "validation": [f"sample_{index}" for index in range(4)],
        "test": ["sample_0", "sample_1", "sample_2", "sample_3", "sample_0", "sample_1"]}
    trainer.run_training_loop()
    capsys.readouterr()

    # Resumed from iteration 2 with the same data: the record of the first start is kept
    unchanged_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                            number_sample_images=0)
    assert unchanged_trainer.data_record_changes == {"resumed_iteration": 2, "changed_splits": [],
                                                     "archived_data_record_path": None}
    assert json.loads(s=data_record_path.read_text()) == first_record
    assert "- Changed Data Splits     : None Since Iteration 2" in capsys.readouterr().out

    # Resumed with fewer training samples: the first record is kept as the data that trained the run up to iteration 2
    changed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                          number_sample_images=0, number_training_samples=6)
    archived_data_record_path = trainer.experiment_folder / "data_until_iteration_2.json"
    assert changed_trainer.data_record_changes == {"resumed_iteration": 2, "changed_splits": ["train"],
                                                   "archived_data_record_path": archived_data_record_path}
    assert json.loads(s=archived_data_record_path.read_text()) == first_record
    assert json.loads(s=data_record_path.read_text())["splits"]["train"]["number_images"] == 6
    console_lines = capsys.readouterr().out.splitlines()
    for expected_line in [f"- Previous Data Record    : {format_path_link(path=archived_data_record_path)}",
                          "- Changed Data Splits     : train Since Iteration 2"]:
        assert any(expected_line in console_line for console_line in console_lines), expected_line

    # Resumed again from the same checkpoint with other data: the record of the data up to it is kept
    build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                        number_sample_images=0, number_training_samples=4)
    assert json.loads(s=archived_data_record_path.read_text()) == first_record
    assert json.loads(s=data_record_path.read_text())["splits"]["train"]["number_images"] == 4


def test_final_checkpoint_is_evaluated_on_its_weights(tmp_path: Path) -> None:
    """
    Verifies that a run ending on a weight saving iteration (4 iterations, saved and evaluated every 4) evaluates its
    final checkpoint once, after the final save, on exactly the weights that checkpoint holds, instead of evaluating
    the weights before the last training step and overwriting their checkpoint.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=4, view_input_images=False,
                                  number_sample_images=0)
    evaluation_recorder = EvaluationRecorder(trainer=trainer)
    trainer.run_test_evaluation = evaluation_recorder
    trainer.run_training_loop()

    # One evaluation of iteration 4, written once in Results.txt
    assert list(evaluation_recorder.evaluated_weights) == [4]
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [block["iteration"] for block in results_blocks] == [4]

    # The evaluated weights are the weights of the saved checkpoint of iteration 4
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_4" / "model_000004.pt", map_location="cpu",
                            weights_only=True)
    for parameter_name, parameter_value in evaluation_recorder.evaluated_weights[4].items():
        assert torch.equal(input=parameter_value.cpu(), other=checkpoint["model_state"][parameter_name])


def test_input_viewer_creates_no_run_folder(tmp_path: Path) -> None:
    """
    Verifies that the input viewer, which exits without training, creates no run folder for a new run, and leaves the
    configuration and the record of the data saved in the folder of an existing run untouched.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # A new run only viewed: the program exits before any folder exists
    with pytest.raises(expected_exception=SystemExit):
        build_dummy_trainer(temporary_directory=tmp_path, number_iterations=1, view_input_images=True,
                            number_sample_images=0)
    assert not (tmp_path / "experiments").exists()

    # An existing run viewed again: its saved configuration is not replaced by the copy of the viewer launch
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=1, view_input_images=False,
                                  number_sample_images=0)
    trainer.configuration_path.write_text(data="saved: configuration\n")
    trainer.data_record_path.write_text(data="saved record")
    with pytest.raises(expected_exception=SystemExit):
        build_dummy_trainer(temporary_directory=tmp_path, number_iterations=1, view_input_images=True,
                            number_sample_images=0)
    assert trainer.configuration_path.read_text() == "saved: configuration\n"
    assert trainer.data_record_path.read_text() == "saved record"


def test_evaluation_draws_the_fixed_sample_images(tmp_path: Path) -> None:
    """
    Verifies the sample images of the evaluation on the dummy task: the fixed test samples are chosen once, one per
    class of the 4 distinct test samples, and every evaluation draws each of them exactly once (the repeated samples of
    the last batch being left out), saves it in the 'Samples' folder of the evaluated checkpoint, reports it with a
    note and its folder, and writes it to the Test TensorBoard writer under 'Samples/<sample name>'.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=1, view_input_images=False,
                                  number_sample_images=2)
    test_labels = {sample_identifier: int(trainer.datasets["test"].labels[sample_identifier])
                   for sample_identifier in range(4)}
    assert len(trainer.sample_image_identifiers) == 2
    assert {test_labels[sample_identifier] for sample_identifier in trainer.sample_image_identifiers} == \
        set(test_labels.values())

    # Two evaluations draw the same samples, each saved once per evaluated checkpoint
    for iteration in [1, 2]:
        results, _ = trainer.run_test_evaluation(iteration=iteration)
        expected_names = sorted(f"sample_{sample_identifier}" for sample_identifier in trainer.sample_image_identifiers)
        assert sorted(results["sample_images"]) == [f"Samples/{sample_name}" for sample_name in expected_names]
        samples_folder = trainer.experiment_folder / "Evaluation" / f"Iteration_{iteration}" / "Samples"
        assert sorted(path.stem for path in samples_folder.iterdir()) == expected_names
        assert results["files"]["Sample Images"] == samples_folder
        assert results["notes"][-1] == "Sample Images : 2 fixed test images"

        # The saved image is the BGR version of the RGB image sent to TensorBoard
        saved_image = cv2.imread(filename=str(samples_folder / f"{expected_names[0]}.png"))
        assert (cv2.cvtColor(src=saved_image, code=cv2.COLOR_BGR2RGB) ==
                results["sample_images"][f"Samples/{expected_names[0]}"]).all()

    # One image per sample and evaluation in the Test TensorBoard writer
    test_events = EventAccumulator(path=str(trainer.tensorboard_directory / "Test"))
    test_events.Reload()
    assert sorted(test_events.Tags()["images"]) == sorted(results["sample_images"])
    assert [event.step for event in test_events.Images(tag=sorted(results["sample_images"])[0])] == [1, 2]


def test_moving_average_is_saved_evaluated_and_resumed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the moving average of the weights on the dummy task: 5 iterations update it at every step and store it in
    the checkpoints, the checkpoint of iteration 4 is evaluated on the trained weights, then on the averaged weights
    with their own Results.txt block, sample images folder and Test_EMA TensorBoard folder, the trained model is given
    back to the training, and a resumed run restores the average of its checkpoint.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=2, moving_average_activated=True)
    trainer.run_training_loop()
    assert trainer.model is not trainer.exponential_moving_average.averaged_model
    assert not trainer.evaluating_averaged_weights

    # Two blocks for the checkpoint of iteration 4: the trained weights, then the averaged weights
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [
        (4, "4 Test Samples"), (4, "4 Test Samples · EMA Weights")]
    evaluation_folder = trainer.experiment_folder / "Evaluation" / "Iteration_4"
    assert len(list((evaluation_folder / "Samples").iterdir())) == 2
    assert len(list((evaluation_folder / "EMA" / "Samples").iterdir())) == 2
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Test_EMA", "Train",
                                                                                   "Validation"]
    averaged_events = EventAccumulator(path=str(trainer.tensorboard_directory / "Test_EMA"))
    averaged_events.Reload()
    assert [event.step for event in averaged_events.Scalars(tag="Accuracy")] == [4]

    # The final checkpoint holds the average of the 5 updates, different from the trained weights
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_5" / "model_000005.pt", map_location="cpu",
                            weights_only=True)
    average_state = checkpoint[EXPONENTIAL_MOVING_AVERAGE_STATE_KEY]
    assert average_state["number_updates"] == 5
    assert any(not torch.equal(input=average_state["model_state"][parameter_name], other=parameter_value)
               for parameter_name, parameter_value in checkpoint["model_state"].items())

    # A resumed run restores the average of its checkpoint
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                          number_sample_images=0, moving_average_activated=True)
    assert resumed_trainer.exponential_moving_average.number_updates == 5
    for parameter_name, parameter_value in resumed_trainer.exponential_moving_average.averaged_model.state_dict(
            ).items():
        assert torch.equal(input=parameter_value.cpu(), other=average_state["model_state"][parameter_name])

    # No training step may fail silently
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_moving_average_turned_on_when_a_run_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a run turning the moving average of its weights on when it resumes from a checkpoint saved without
    one restarts the average from the loaded weights, with a message, and evaluates it at its next evaluation.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                  number_sample_images=0)
    trainer.run_training_loop()
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_2" / "model_000002.pt", map_location="cpu",
                            weights_only=True)
    assert EXPONENTIAL_MOVING_AVERAGE_STATE_KEY not in checkpoint

    # The resumed run starts its average from the weights of iteration 2
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                          number_sample_images=0, moving_average_activated=True)
    assert "Holds No Moving Average Of The Weights" in capsys.readouterr().out
    assert resumed_trainer.exponential_moving_average.number_updates == 0
    for parameter_name, parameter_value in resumed_trainer.exponential_moving_average.averaged_model.state_dict(
            ).items():
        assert torch.equal(input=parameter_value.cpu(), other=checkpoint["model_state"][parameter_name])

    # Iterations 3 and 4 update it, and the final checkpoint of iteration 4 is evaluated on both weights
    resumed_trainer.run_training_loop()
    results_blocks = read_results_blocks(results_path=resumed_trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [
        (4, "4 Test Samples"), (4, "4 Test Samples · EMA Weights")]
