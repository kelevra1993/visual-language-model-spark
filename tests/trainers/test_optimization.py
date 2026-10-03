"""
Tests of the optimization settings of the trainers: the learning rate factor of
vision_training/trainers/learning_rate_schedule.py, the optimizer, schedule and gradient clipping during training, and
mixed precision training.
"""

import math
import pytest
import torch
import yaml

from pathlib import Path
from typing import Any, Dict

from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset
from vision_training.trainers.learning_rate_schedule import compute_learning_rate_factor
from tests.fixtures.dummy_task import DummyDataset, DummyTrainer, build_dummy_configurations
from tests.trainers.test_classification_trainer import SMOKE_IMAGE_SIZE, build_trainer, write_smoke_configuration


def get_optimization_configuration(learning_rate_schedule: str) -> Dict[str, Any]:
    """
    Supplies an 'Optimization' section with a warm-up of 4 iterations, a cosine decay of 10 iterations down to 10 %
    of the learning rate, and the given schedule.

    Args:
        learning_rate_schedule (str): "constant" or "cosine".

    Returns:
        Dict[str, Any]: The optimization settings.
    """
    return {"optimizer": "adam", "weight_decay": 0.0, "learning_rate_schedule": learning_rate_schedule,
            "warmup_iterations": 4, "decay_iterations": 10, "minimum_learning_rate_ratio": 0.1,
            "gradient_clipping_norm": None, "mixed_precision": False}


@pytest.mark.parametrize(argnames=("iteration", "learning_rate_schedule", "expected_factor"), argvalues=[
    (1, "constant", 0.25), (4, "constant", 1.0), (1000, "constant", 1.0),
    (2, "cosine", 0.5), (4, "cosine", 1.0), (9, "cosine", 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * 0.5))),
    (14, "cosine", 0.1), (1000, "cosine", 0.1)])
def test_learning_rate_factor(iteration: int, learning_rate_schedule: str, expected_factor: float) -> None:
    """
    Verifies the learning rate schedule: a linear warm-up reaching the full learning rate at its last iteration, then
    a constant learning rate, or a cosine decay reaching its minimum after the decay iterations and staying there.

    Args:
        iteration (int): The training iteration.
        learning_rate_schedule (str): "constant" or "cosine".
        expected_factor (float): The expected learning rate factor.
    """
    assert compute_learning_rate_factor(iteration=iteration, optimization_configuration=get_optimization_configuration(
        learning_rate_schedule=learning_rate_schedule)) == pytest.approx(expected=expected_factor)


@pytest.mark.parametrize(argnames=("optimizer_name", "optimizer_class"), argvalues=[
    ("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW), ("sgd", torch.optim.SGD)])
def test_optimizer_schedule_and_clipping_in_training(tmp_path: Path, optimizer_name: str,
                                                     optimizer_class: type) -> None:
    """
    Verifies the optimization of the general trainer: the configured optimizer with its weight decay, the learning
    rate of the schedule at every step (a warm-up of 2 iterations then a cosine decay of 3), and the gradients clipped
    to the configured norm before the step.

    Args:
        tmp_path (Path): The pytest temporary directory.
        optimizer_name (str): The configured optimizer.
        optimizer_class (type): The expected optimizer class.
    """
    configuration_path = tmp_path / "configuration.yaml"
    configuration_path.write_text(data="dummy: true\n")
    experiment_configuration, model_configuration = build_dummy_configurations(
        experiment_folder=tmp_path / "experiments", number_iterations=5)
    model_configuration["Optimization"].update({"optimizer": optimizer_name, "weight_decay": 0.01,
                                                "learning_rate_schedule": "cosine", "warmup_iterations": 2,
                                                "decay_iterations": 3, "gradient_clipping_norm": 0.5})
    datasets = {split_name: DummyDataset(sample_identifiers=list(range(4)), seed=seed)
                for seed, split_name in enumerate(["train", "validation", "test"])}
    trainer = DummyTrainer(experiment_configuration=experiment_configuration, model_configuration=model_configuration,
                           configuration_path=configuration_path, datasets=datasets)
    assert type(trainer.optimizer) is optimizer_class
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.01

    # A training step with large gradients: the step clips them and applies the warm-up learning rate
    trainer.optimizer.zero_grad()
    (1000.0 * sum(parameter.sum() for parameter in trainer.model.parameters())).backward()
    trainer.step_optimizer(iteration=1)
    gradient_norm = torch.sqrt(sum(parameter.grad.pow(2).sum() for parameter in trainer.model.parameters()))
    assert float(gradient_norm) <= 0.5 + 1e-5
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected=0.5 * 0.01)

    # The training loop follows the schedule, down to the minimum learning rate at the end of the decay
    trainer.run_training_loop()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(expected=0.01 * 0.01)


def test_mixed_precision_training(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies a short classification training with mixed precision: the run name carries no token for it (a run can
    be resumed with or without mixed precision), the forward passes run in bfloat16 without failing any iteration, and
    the test evaluation, in float32, is written to Results.txt.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output of the trainer.
    """
    dataset_folder = tmp_path / "synthetic_shapes"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=SMOKE_IMAGE_SIZE,
                             split_sizes={"train": 8, "validation": 4, "test": 4}, maximum_objects=1,
                             number_frames=1, number_workers=2, task="classification")
    configuration_path = write_smoke_configuration(temporary_directory=tmp_path, dataset_folder=dataset_folder,
                                                   number_iterations=5, head_settings={}, evaluation_settings={})
    configuration = yaml.safe_load(stream=configuration_path.read_text())
    configuration["Optimization"]["mixed_precision"] = True
    configuration_path.write_text(data=yaml.safe_dump(data=configuration))

    trainer = build_trainer(configuration_path=configuration_path)
    assert "BF16" not in trainer.run_name and trainer.run_name.endswith("_MC4")
    trainer.run_training_loop()

    assert (trainer.experiment_folder / "Results.txt").read_text().startswith("Iteration 4 : 4 Test Images")
    console_output = capsys.readouterr().out
    assert "Error encountered during iteration" not in console_output
    assert "Critical Error" not in console_output
