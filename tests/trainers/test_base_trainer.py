"""
Tests of BaseTrainer of vision_training/trainers/base_trainer.py on the dummy task: a model iteration sums the losses
and tracks the metrics, and training saves, evaluates and resumes from its checkpoints.
"""

import pytest
import torch

from pathlib import Path

from tests.fixtures.dummy_task import DummyDataset, DummyTrainer, build_dummy_configurations
from vision_training.trainers.checkpointing import CHECKPOINT_FORMAT_VERSION


def build_dummy_trainer(temporary_directory: Path, number_iterations: int) -> DummyTrainer:
    """
    Builds the trainer of the dummy task in a temporary directory. Its test split has 6 samples but only 4 distinct
    ones, its last batch repeating the first two samples like the DALI loaders do.

    Args:
        temporary_directory (Path): The pytest temporary directory, receiving the configuration and the run folder.
        number_iterations (int): The number of training iterations of the run.

    Returns:
        DummyTrainer: The trainer, resumed from the last checkpoint of its run folder when there is one.
    """
    configuration_path = temporary_directory / "configuration.yaml"
    configuration_path.write_text(data="dummy: true\n")
    experiment_configuration, model_configuration = build_dummy_configurations(
        experiment_folder=temporary_directory / "experiments", number_iterations=number_iterations)
    datasets = {"train": DummyDataset(sample_identifiers=list(range(8)), seed=0),
                "validation": DummyDataset(sample_identifiers=list(range(4)), seed=1),
                "test": DummyDataset(sample_identifiers=[0, 1, 2, 3, 0, 1], seed=2)}
    return DummyTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                        configuration_path=configuration_path, datasets=datasets)


def test_model_iteration_sums_the_losses_and_tracks_the_metrics(tmp_path: Path) -> None:
    """
    Verifies the generic step of the trainer: every '_loss' output of the model is summed into the total loss, and the
    losses and batch metrics are accumulated into the trackers divided by the number of batches of the interval.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=1)
    batch = next(iter(trainer.train_dataloader))
    tracker_dictionary = trainer.metric_tracker.create_trackers()

    with torch.no_grad():
        total_loss, model_outputs = trainer.run_model_iteration(batch=batch, writer=None, iteration=1,
                                                                tracker_dictionary=tracker_dictionary,
                                                                apply_augmentation=False, tracker_divisor=2)

    assert torch.isclose(input=total_loss, other=model_outputs["first_loss"] + model_outputs["second_loss"])
    assert tracker_dictionary["total_loss"] == pytest.approx(expected=total_loss.item() / 2)
    assert tracker_dictionary["first_loss"] == pytest.approx(expected=model_outputs["first_loss"].item() / 2)
    assert tracker_dictionary["second_loss"] == pytest.approx(expected=model_outputs["second_loss"].item() / 2)
    assert 0.0 <= tracker_dictionary["accuracy"] <= 0.5


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
    trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=5)
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
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION

    # The test split has 6 samples but only 4 distinct ones, each evaluated exactly once
    assert (trainer.experiment_folder / "Results.txt").read_text().startswith("Iteration 4 : 4 Test Samples")
    assert sorted(path.name for path in trainer.tensorboard_directory.iterdir()) == ["Test", "Train", "Validation"]

    # A second run with the same settings resumes from iteration 5 with exactly its weights
    resumed_trainer = build_dummy_trainer(temporary_directory=tmp_path, number_iterations=2)
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
