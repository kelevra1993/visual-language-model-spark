"""
Tests of BaseTrainer of vision_training/trainers/base_trainer.py on the dummy task: training saves, evaluates and
resumes from its checkpoints, the interval averages of a run resumed between two summaries cover the iterations it
really ran, a resumed run records and keeps the data it changed, the final checkpoint is evaluated on the weights it
holds, and the moving average of the weights is saved, evaluated next to the trained weights and resumed. A resumed
run keeps the classes saved in its folder, a fresh start moves the previous training of the same settings aside,
unreadable images and failed evaluations never stop the training, a training whose data vanished ends with a final
checkpoint, and a split without batch is refused. A training without validation builds no validation loader, which
the input viewer still gets, and a task whose validation loader does not follow that setting is refused.
"""

import copy
import json
import pytest
import torch

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

from tests.fixtures.dummy_task import DummyDataset, DummyTrainer, build_dummy_configurations
from scripts.evaluation.evaluate_checkpoint import point_to_checkpoint_evaluation
from tests.fixtures.evaluation_checks import read_results_blocks
from vision_training.core.contracts import EvaluationResults
from vision_training.data.label_mapping import RUN_LABEL_DICTIONARY_FILE_NAME
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


class VanishingDataset(DummyDataset):
    """
    Training split of the dummy task whose data disappears during the run, as when its mount drops: every sample read
    after the first ones fails, so that a test can check what the training does once no batch can be loaded anymore.
    """

    def __init__(self, sample_identifiers: List[int], seed: int, readable_samples: int) -> None:
        """
        Draws the images and labels of the samples, and counts the samples read.

        Args:
            sample_identifiers (List[int]): The identifier of every sample.
            seed (int): The seed of the random images and labels.
            readable_samples (int): The number of samples read before the data disappears.
        """
        super().__init__(sample_identifiers=sample_identifiers, seed=seed)
        self.readable_samples = readable_samples
        self.number_read_samples = 0

    def __getitem__(self, sample_index: int) -> Dict[str, Any]:
        """
        Returns one sample while the data is there.

        Args:
            sample_index (int): The index of the sample.

        Returns:
            Dict[str, Any]: The image, label and sample identifier of the sample.

        Raises:
            OSError: Once readable_samples samples were read.
        """
        self.number_read_samples += 1
        if self.number_read_samples > self.readable_samples:
            raise OSError("The data folder is not mounted")
        return super().__getitem__(sample_index=sample_index)


def build_dummy_trainer(temporary_directory: Path, number_iterations: int, view_input_images: bool,
                        number_sample_images: int, configuration_text: str = "dummy: true\n",
                        moving_average_activated: bool = False, number_training_samples: int = 8,
                        allow_data_change_on_resume: bool = False, run_name: Optional[str] = None,
                        label_dictionary: Optional[Dict[Any, int]] = None,
                        training_dataset: Optional[Dataset] = None,
                        evaluated_run_folder: Optional[Path] = None,
                        validation_iteration_ratio: int = 2) -> DummyTrainer:
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
        allow_data_change_on_resume (bool): Whether the run may resume on changed labels. Defaults to False.
        run_name (Optional[str]): The name of the run folder, as given by a standalone evaluation, None for the name
            of the dummy trainer. Defaults to None.
        label_dictionary (Optional[Dict[Any, int]]): The label dictionary given by the mapping file, None for the one
            of the dummy configuration. Defaults to None.
        training_dataset (Optional[Dataset]): The training split, None for number_training_samples readable samples.
            Defaults to None.
        evaluated_run_folder (Optional[Path]): The run folder evaluated by a standalone evaluation, whose trainer is
            pointed to its Checkpoint_Evaluation folder as scripts/evaluation/evaluate_checkpoint.py does
            (point_to_checkpoint_evaluation), None for a training. Defaults to None.
        validation_iteration_ratio (int): The number of training iterations per validation batch, 0 for a training
            without validation, whose validation loader is not built. Defaults to 2.

    Returns:
        DummyTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    configuration_path = temporary_directory / "configuration.yaml"
    configuration_path.write_text(data=configuration_text)
    experiment_configuration, model_configuration = build_dummy_configurations(
        experiment_folder=temporary_directory / "experiments", number_iterations=number_iterations)
    experiment_configuration["diagnosis_modes"]["view_input_images"] = view_input_images
    experiment_configuration["evaluation"]["number_sample_images"] = number_sample_images
    experiment_configuration["allow_data_change_on_resume"] = allow_data_change_on_resume
    experiment_configuration["validation_iteration_ratio"] = validation_iteration_ratio
    experiment_configuration["run_name"] = run_name
    if evaluated_run_folder is not None:
        point_to_checkpoint_evaluation(experiment_configuration=experiment_configuration,
                                       experiment_folder=evaluated_run_folder, folder_suffix="")
    if label_dictionary is not None:
        model_configuration["Data"]["label_dictionary"] = label_dictionary
    model_configuration["Optimization"]["exponential_moving_average"] = {"activated": moving_average_activated,
                                                                         "decay": 0.5}
    if training_dataset is None:
        training_dataset = DummyDataset(sample_identifiers=list(range(number_training_samples)), seed=0)
    datasets = {"train": training_dataset,
                "validation": DummyDataset(sample_identifiers=list(range(4)), seed=1),
                "test": DummyDataset(sample_identifiers=[0, 1, 2, 3, 0, 1], seed=2)}
    return DummyTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                        configuration_path=configuration_path, datasets=datasets)


def test_training_saves_evaluates_and_resumes(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the whole generic pipeline on the dummy task: 5 iterations save the intermediate, evaluated and final
    checkpoints with the task and run name, each saved after the step of its iteration (the checkpoint of iteration k
    holding k optimizer steps), evaluate every test sample exactly once into Results.txt, and write the three
    TensorBoard folders; a second run with the same settings and a total of 7 iterations resumes from the last
    checkpoint with exactly its weights, trains the iterations 6 and 7 and continues the registry, and a third one
    finds the run complete and trains nothing. No training step may fail silently.

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

    # Every checkpoint is saved after the step of its iteration: the one of iteration k holds k optimizer steps
    for iteration in [2, 4, 5]:
        saved_checkpoint = torch.load(f=weights_directory / f"Iteration_{iteration}" / f"model_{iteration:06}.pt",
                                      map_location="cpu", weights_only=True)
        assert all(parameter_state["step"] == iteration
                   for parameter_state in saved_checkpoint["optimizer_state"]["state"].values())

    # The test split has 6 samples but only 4 distinct ones, each evaluated exactly once
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [(4, "4 Test Samples")]
    assert "Accuracy" in results_blocks[0]["text"] and "Per Class Recall" in results_blocks[0]["text"]
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train", "Validation"]

    # A second run with the same settings and a total of 7 iterations resumes from iteration 5 with exactly its weights
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=7, view_input_images=False,
                                          number_sample_images=0)
    assert resumed_trainer.start_iteration == 6
    for parameter_name, parameter_value in resumed_trainer.model.state_dict().items():
        assert torch.equal(input=parameter_value.cpu(), other=checkpoint["model_state"][parameter_name])

    # The resumed run trains iterations 6 (intermediate save) and 7 (end of the run)
    resumed_trainer.run_training_loop()
    assert (weights_directory / "full-checkpoint").read_text().splitlines()[0] == 'model_checkpoint_path: "Iteration_7"'
    assert {"Iteration_6", "Iteration_7"} <= {path.name for path in weights_directory.glob("Iteration_*")}

    # A third run with the same total finds the run complete and trains nothing
    complete_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=7, view_input_images=False,
                                           number_sample_images=0)
    assert complete_trainer.start_iteration == 8
    registry_text = (weights_directory / "full-checkpoint").read_text()
    complete_trainer.run_training_loop()
    assert (weights_directory / "full-checkpoint").read_text() == registry_text

    # The training loop catches errors and keeps going, so make sure that no iteration failed along the way
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
    assert "The Run Is Already Complete: Its Last Checkpoint Is At Iteration 7" in console_output


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
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=6, view_input_images=False,
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


def test_run_records_its_data_and_keeps_the_previous_data(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the record of the data of a run ('data.json'): written by a new run with the images of every split and no
    class names (the dummy task has none), kept with its date by a resume with the same data, a resume whose training
    split changed refused with an error leaving the run folder untouched, kept as 'data_until_iteration_<n>.json' by
    that resume once allowed ('allow_data_change_on_resume') before the new record replaces it, and that file kept by a
    second allowed resume from the same checkpoint.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                  number_sample_images=0)
    data_record_path = trainer.experiment_folder / "data.json"
    first_record = json.loads(s=data_record_path.read_text())
    assert trainer.data_record_changes is None
    assert first_record["classes"] is None
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

    # Resumed with fewer training samples: refused before the configuration copy and the record are replaced
    run_folder_files = {path: path.read_bytes() for path in trainer.experiment_folder.glob("*") if path.is_file()}
    with pytest.raises(expected_exception=ValueError, match=r"labels.json of the train split\(s\) changed since "
                                                            r"iteration 2 .* needs a new run"):
        build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                            number_sample_images=0, number_training_samples=6,
                            configuration_text="dummy: true\nlearning_rate: 0.0001\n")
    assert {path: path.read_bytes() for path in trainer.experiment_folder.glob("*") if path.is_file()} == \
        run_folder_files

    # The same resume once allowed: the first record is kept as the data that trained the run up to iteration 2
    changed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                          number_sample_images=0, number_training_samples=6,
                                          allow_data_change_on_resume=True)
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
                        number_sample_images=0, number_training_samples=4, allow_data_change_on_resume=True)
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


def test_resumed_run_keeps_its_classes_when_the_mapping_changes(tmp_path: Path,
                                                                capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the label dictionary of a run is saved in its folder ('label_dictionary.json') and in its
    checkpoints, that a run resumed after its mapping file swapped the two classes keeps the classes it was trained
    with, with a message, and that a new run, in another folder, takes the classes of the edited mapping file.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trained_label_dictionary = {"negative": 0, "positive": 1}
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2, view_input_images=False,
                                  number_sample_images=0)
    trainer.run_training_loop()
    run_label_dictionary_path = trainer.experiment_folder / RUN_LABEL_DICTIONARY_FILE_NAME
    assert json.loads(s=run_label_dictionary_path.read_text()) == trained_label_dictionary
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_2" / "model_000002.pt", map_location="cpu",
                            weights_only=True)
    assert checkpoint["label_dictionary"] == trained_label_dictionary

    # The mapping file swaps the two classes: the resumed run keeps its own and saves them again
    swapped_label_dictionary = {"negative": 1, "positive": 0}
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=4, view_input_images=False,
                                          number_sample_images=0, label_dictionary=swapped_label_dictionary)
    assert resumed_trainer.start_iteration == 3
    assert resumed_trainer.label_dictionary == trained_label_dictionary
    assert resumed_trainer.data_configuration["label_dictionary"] == trained_label_dictionary
    assert json.loads(s=run_label_dictionary_path.read_text()) == trained_label_dictionary
    assert "The Label Mapping Changed Since The Run Started, The Run Keeps Its Own Classes" in capsys.readouterr().out

    # A new run, in another folder, trains the classes of the edited mapping file
    new_run_directory = tmp_path / "new_run"
    new_run_directory.mkdir()
    new_trainer = build_dummy_trainer(temporary_directory=new_run_directory, number_iterations=4,
                                      view_input_images=False, number_sample_images=0,
                                      label_dictionary=swapped_label_dictionary)
    assert new_trainer.label_dictionary == swapped_label_dictionary
    assert json.loads(s=(new_trainer.experiment_folder / RUN_LABEL_DICTIONARY_FILE_NAME).read_text()) == \
        swapped_label_dictionary


def test_unreadable_images_drop_their_batches_without_stopping_the_training(tmp_path: Path,
                                                                            capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a training batch holding an unreadable image is dropped without stopping the training: the image is
    named once in the run however many times its batch comes back, the dropped batches are counted in the next console
    summary and only there, and every iteration still trains on a batch.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    # Unshuffled batches [0, 1], [2, 3], [4, 5], [6, 7]: the batch of the image 3 is dropped at iterations 2 and 5
    training_dataset = DummyDataset(sample_identifiers=list(range(8)), seed=0, unreadable_identifiers=[3])
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=6, view_input_images=False,
                                  number_sample_images=0, training_dataset=training_dataset)
    trainer.run_training_loop()
    console_output = capsys.readouterr().out
    assert console_output.count("Training Batch Dropped : The image cannot be read : 3.png") == 1
    assert console_output.count("- Dropped Batches Since The Last Summary : Training 1 (their loading failed") == 2
    assert "Error encountered during iteration" not in console_output

    # Every iteration took its step, the last checkpoint holding the 6 steps
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_6" / "model_000006.pt", map_location="cpu",
                            weights_only=True)
    assert all(parameter_state["step"] == 6 for parameter_state in checkpoint["optimizer_state"]["state"].values())


def test_failed_evaluation_keeps_its_checkpoint_and_the_training_going(tmp_path: Path,
                                                                       capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a test evaluation failing at a weight saving iteration (e.g. an unreadable test image) is counted as
    an error of its iteration without stopping the training: the checkpoint of the iteration, saved right after its
    step and before the evaluation, holds its 4 steps, the next iterations train and save as usual, and the last
    checkpoint, whose evaluation failed too, is announced as not evaluated instead of being saved again.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=8, view_input_images=False,
                                  number_sample_images=0)
    with patch.object(target=trainer, attribute="run_test_evaluation",
                      side_effect=ValueError("The image cannot be read : test.png")):
        trainer.run_training_loop()
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration 4." in console_output
    assert "Error encountered during iteration 8." in console_output
    assert "The Checkpoint Of Iteration 8 Was Not Evaluated" in console_output
    assert sorted(path.name for path in trainer.weights_directory.glob(pattern="Iteration_*")) == [
        "Iteration_2", "Iteration_4", "Iteration_6", "Iteration_8"]
    assert (trainer.weights_directory / "full-checkpoint").read_text().splitlines()[0] == (
        'model_checkpoint_path: "Iteration_8"')
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_4" / "model_000004.pt", map_location="cpu",
                            weights_only=True)
    assert all(parameter_state["step"] == 4 for parameter_state in checkpoint["optimizer_state"]["state"].values())


def test_vanished_data_ends_the_training_with_a_final_checkpoint(tmp_path: Path,
                                                                capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a training whose data disappears (every sample read failing after the third batch) stops after its
    consecutive errors instead of leaving the process, and saves a final checkpoint of the steps taken since its last
    one: the third step, after the intermediate checkpoint of iteration 2.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    # 20 batches of 2 samples per pass, the first 3 readable
    training_dataset = VanishingDataset(sample_identifiers=list(range(40)), seed=0, readable_samples=6)
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=20, view_input_images=False,
                                  number_sample_images=0, training_dataset=training_dataset)
    trainer.run_training_loop()
    console_output = capsys.readouterr().out
    assert "10 fetches of a Training batch failed in a row, the last one with: The data folder is not mounted" in (
        console_output)
    assert "Critical Error" in console_output
    assert "Last Checkpoint Saved At Iteration 3. Exiting Training." in console_output
    checkpoint = torch.load(f=trainer.weights_directory / "Iteration_3" / "model_000003.pt", map_location="cpu",
                            weights_only=True)
    assert all(parameter_state["step"] == 3 for parameter_state in checkpoint["optimizer_state"]["state"].values())


def test_split_without_batch_is_refused(tmp_path: Path) -> None:
    """
    Verifies that a split giving no batch (e.g. every sample left out by the label mapping) is refused before the
    training with the split named, instead of waiting forever for a batch, and that a loader whose passes keep ending
    without any batch raises after two passes.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=1, view_input_images=False,
                                  number_sample_images=0)
    empty_loader = DataLoader(dataset=[], batch_size=2)
    trainer.validation_dataloader = empty_loader
    with pytest.raises(expected_exception=ValueError,
                       match="The 'validation_data_folder' split gives no batch: its dataset keeps 0 samples"):
        trainer.check_loaders_give_batches()
    with pytest.raises(expected_exception=RuntimeError,
                       match="The Validation loader gives no batch: 2 passes over its split ended without any."):
        trainer.get_next_batch(dataloader_iterator=iter(empty_loader), dataloader=empty_loader,
                               split_name="Validation")


def test_training_without_validation_builds_no_validation_loader(tmp_path: Path,
                                                                 capsys: pytest.CaptureFixture) -> None:
    """
    Verifies a training without validation ('validation_iteration_ratio' 0) on the dummy task: its validation loader is
    not built (None, announced at the start, and left out of the loaders whose workers are stopped), the state of the
    training loop holds no validation iterator nor trackers, and the 5 iterations still save their checkpoints and
    evaluate the test split, without any Validation TensorBoard folder. No training step may fail silently.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5, view_input_images=False,
                                  number_sample_images=0, validation_iteration_ratio=0)
    assert trainer.builds_validation_loader is False and trainer.validation_dataloader is None
    assert trainer.list_dataloaders() == [trainer.train_dataloader, trainer.test_dataloader]

    # The state of the training loop holds no validation iterator, which would start the workers of the loader
    _, validation_iterator, validation_enabled, _, validation_trackers, _ = trainer.initialize_training_loop_state()
    assert (validation_iterator, validation_enabled, validation_trackers) == (None, False, None)

    # Checkpoints at iterations 2 (intermediate), 4 (with test evaluation) and 5 (end of the run), as with validation
    trainer.run_training_loop()
    assert sorted(path.name for path in trainer.weights_directory.glob("Iteration_*")) == ["Iteration_2", "Iteration_4",
                                                                                         "Iteration_5"]
    results_blocks = read_results_blocks(results_path=trainer.experiment_folder / "Results.txt")
    assert [(block["iteration"], block["evaluated_samples"]) for block in results_blocks] == [(4, "4 Test Samples")]
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train"]

    # The missing loader is announced, and the training loop catches errors and keeps going, so none may have failed
    console_output = capsys.readouterr().out
    assert "Validation Loader Not Built : 'validation_iteration_ratio' Is 0, No Validation Batch Is Loaded" in (
        console_output)
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output


def test_validation_loader_follows_the_validation_setting(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the general trainer refuses a task whose validation loader does not follow builds_validation_loader
    (a loader built for a training without validation, or none for a training with validation), so that a task
    forgetting to pass the setting to its loaders stops at its launch, and that the input viewer, a diagnosis mode
    ending the program, still gets the validation loader of a training without validation and shows its samples.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    # One folder per trainer, each receiving its configuration and its run
    for folder_name in ["without_validation", "with_validation", "input_viewer"]:
        (tmp_path / folder_name).mkdir()

    # A validation loader built for a training without validation is refused
    trainer_without_validation = build_dummy_trainer(temporary_directory=tmp_path / "without_validation",
                                                     number_iterations=1, view_input_images=False,
                                                     number_sample_images=0, validation_iteration_ratio=0)
    trainer_without_validation.validation_dataloader = DataLoader(
        dataset=trainer_without_validation.datasets["validation"], batch_size=2)
    with pytest.raises(expected_exception=ValueError,
                       match="The validation loader of DummyTrainer.build_dataloaders is built while "
                             "builds_validation_loader is False"):
        trainer_without_validation.check_loaders_give_batches()

    # A missing validation loader for a training with validation is refused
    trainer_with_validation = build_dummy_trainer(temporary_directory=tmp_path / "with_validation",
                                                  number_iterations=1, view_input_images=False,
                                                  number_sample_images=0, validation_iteration_ratio=2)
    trainer_with_validation.validation_dataloader = None
    with pytest.raises(expected_exception=ValueError,
                       match="The validation loader of DummyTrainer.build_dataloaders is not built while "
                             "builds_validation_loader is True"):
        trainer_with_validation.check_loaders_give_batches()

    # The input viewer of a training without validation shows its validation samples, then ends the program
    with pytest.raises(expected_exception=SystemExit):
        build_dummy_trainer(temporary_directory=tmp_path / "input_viewer", number_iterations=1, view_input_images=True,
                            number_sample_images=0, validation_iteration_ratio=0)
    assert "View Of The Training And Validation Images That Are Fed To The Neural Network" in capsys.readouterr().out
