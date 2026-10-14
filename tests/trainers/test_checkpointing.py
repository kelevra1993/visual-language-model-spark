"""
Tests of vision_training/trainers/checkpointing.py: the checkpoint registry naming the last checkpoint and listing every
checkpoint once, saved checkpoints found and loaded, and missing checkpoints reported.
"""

import pytest
import torch

from pathlib import Path
from torch import nn

from vision_training.trainers.checkpointing import (CHECKPOINT_FORMAT_VERSION, CHECKPOINT_REGISTRY_NAME,
                                                    get_checkpoint_iterations, get_checkpoint_path,
                                                    get_last_checkpoint_iteration, load_checkpoint, save_checkpoint,
                                                    update_checkpoint_registry)


def test_registry_names_the_last_checkpoint_and_keeps_the_history(tmp_path: Path) -> None:
    """
    Verifies that the registry of a run names its last checkpoint on its first line and lists every checkpoint after
    it, and that a run without registry has no last checkpoint.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the 'Weights' folder.
    """
    assert get_last_checkpoint_iteration(weights_directory=tmp_path) == 0

    for iteration in [2, 4, 10]:
        update_checkpoint_registry(weights_directory=tmp_path, iteration=iteration)

    assert (tmp_path / CHECKPOINT_REGISTRY_NAME).read_text().splitlines() == [
        'model_checkpoint_path: "Iteration_10"', 'all_model_checkpoint_paths: "Iteration_2"',
        'all_model_checkpoint_paths: "Iteration_4"', 'all_model_checkpoint_paths: "Iteration_10"']
    assert get_last_checkpoint_iteration(weights_directory=tmp_path) == 10


def test_registry_lists_every_checkpoint_once(tmp_path: Path) -> None:
    """
    Verifies that every iteration recorded in the registry is listed once and in order, even when a resumed run saved
    an iteration again, and that a run without registry has no checkpoint.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the 'Weights' folder.
    """
    assert get_checkpoint_iterations(weights_directory=tmp_path) == []
    for iteration in [250, 500, 750, 500, 1000]:
        update_checkpoint_registry(weights_directory=tmp_path, iteration=iteration)
    assert get_checkpoint_iterations(weights_directory=tmp_path) == [250, 500, 750, 1000]


def test_saved_checkpoint_is_found_and_loaded(tmp_path: Path) -> None:
    """
    Verifies that a saved checkpoint is written atomically (no temporary file left), found as the last checkpoint of
    its run, and loaded back with its weights, its metadata and its format version.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the run folder.
    """
    model = nn.Linear(in_features=2, out_features=2)
    optimizer = torch.optim.Adam(params=model.parameters())
    weights_directory = tmp_path / "Weights"

    checkpoint_folder = save_checkpoint(weights_directory=weights_directory, iteration=3, model=model,
                                        optimizer=optimizer, metadata={"task": "dummy_task"},
                                        exponential_moving_average_state=None)

    assert checkpoint_folder == weights_directory / "Iteration_3"
    assert not list(checkpoint_folder.glob("*.tmp"))
    assert get_checkpoint_path(experiment_folder=tmp_path, iteration=None) == checkpoint_folder / "model_000003.pt"

    checkpoint = load_checkpoint(weights_directory=weights_directory, iteration=3, device=torch.device("cpu"))
    assert (checkpoint["task"], checkpoint["iteration"]) == ("dummy_task", 3)
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert torch.equal(input=checkpoint["model_state"]["weight"], other=model.weight)


def test_missing_checkpoints_are_reported(tmp_path: Path) -> None:
    """
    Verifies that asking for the last checkpoint of a run without registry, or for an iteration that was never
    saved, raises an error naming the missing file.

    Args:
        tmp_path (Path): The pytest temporary directory, used as an empty run folder.
    """
    with pytest.raises(expected_exception=FileNotFoundError, match="No checkpoint registry found"):
        get_checkpoint_path(experiment_folder=tmp_path, iteration=None)
    with pytest.raises(expected_exception=FileNotFoundError, match="No checkpoint found"):
        get_checkpoint_path(experiment_folder=tmp_path, iteration=7)
