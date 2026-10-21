"""
Tests of vision_training/trainers/checkpointing.py: the checkpoint registry naming the last checkpoint and listing every
checkpoint once, saved checkpoints found and loaded (with the auxiliary heads of a run and their optimizer states),
missing checkpoints reported, and a registry that cannot be read replaced by the complete checkpoint folders.
"""

import pytest
import torch

from pathlib import Path
from torch import nn

from vision_training.trainers.checkpointing import (AUXILIARY_HEADS_STATE_KEY, CHECKPOINT_FORMAT_VERSION,
                                                    CHECKPOINT_REGISTRY_NAME, get_checkpoint_iterations,
                                                    get_checkpoint_path, get_last_checkpoint_iteration,
                                                    load_checkpoint, save_checkpoint, update_checkpoint_registry)


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
    its run, and loaded back with its weights, its metadata and its format version, and that the auxiliary heads of a
    run are stored with their optimizer states under their own key, absent from the checkpoints of a run without them.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the run folder.
    """
    model = nn.Linear(in_features=2, out_features=2)
    optimizer = torch.optim.Adam(params=model.parameters())
    weights_directory = tmp_path / "Weights"

    checkpoint_folder = save_checkpoint(weights_directory=weights_directory, iteration=3, model=model,
                                        optimizer=optimizer, metadata={"task": "dummy_task"},
                                        exponential_moving_average_state=None, auxiliary_heads_state=None)

    assert checkpoint_folder == weights_directory / "Iteration_3"
    assert not list(checkpoint_folder.glob("*.tmp"))
    assert get_checkpoint_path(experiment_folder=tmp_path, iteration=None) == checkpoint_folder / "model_000003.pt"

    checkpoint = load_checkpoint(weights_directory=weights_directory, iteration=3, device=torch.device("cpu"))
    assert (checkpoint["task"], checkpoint["iteration"]) == ("dummy_task", 3)
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert torch.equal(input=checkpoint["model_state"]["weight"], other=model.weight)
    assert AUXILIARY_HEADS_STATE_KEY not in checkpoint

    # An auxiliary head trained one step, saved with its optimizer state and read back by the loader of the weights
    auxiliary_head = nn.Linear(in_features=2, out_features=3)
    auxiliary_optimizer = torch.optim.AdamW(params=auxiliary_head.parameters())
    auxiliary_head(input=torch.ones(size=(1, 2))).sum().backward()
    auxiliary_optimizer.step()
    save_checkpoint(weights_directory=weights_directory, iteration=4, model=model, optimizer=optimizer,
                    metadata={"task": "dummy_task"}, exponential_moving_average_state=None,
                    auxiliary_heads_state={"dummy_head": {"head_state": auxiliary_head.state_dict(),
                                                          "optimizer_state": auxiliary_optimizer.state_dict()}})
    checkpoint = load_checkpoint(weights_directory=weights_directory, iteration=4, device=torch.device("cpu"))
    saved_head_state = checkpoint[AUXILIARY_HEADS_STATE_KEY]["dummy_head"]
    assert torch.equal(input=saved_head_state["head_state"]["weight"], other=auxiliary_head.weight)
    assert saved_head_state["optimizer_state"]["state"][0]["step"] == 1


def test_missing_checkpoints_are_reported(tmp_path: Path) -> None:
    """
    Verifies that asking for the last checkpoint of a run without any checkpoint, or for an iteration that was never
    saved, raises an error naming the missing folder or file.

    Args:
        tmp_path (Path): The pytest temporary directory, used as an empty run folder.
    """
    with pytest.raises(expected_exception=FileNotFoundError, match="No checkpoint found in the run"):
        get_checkpoint_path(experiment_folder=tmp_path, iteration=None)
    with pytest.raises(expected_exception=FileNotFoundError, match="No checkpoint found"):
        get_checkpoint_path(experiment_folder=tmp_path, iteration=7)


def test_unreadable_registry_falls_back_to_the_checkpoint_folders(tmp_path: Path) -> None:
    """
    Verifies that a registry left empty or cut by a process killed while writing it no longer prevents the resume: the
    newest complete checkpoint folder gives the last checkpoint, every complete folder the list of checkpoints, a
    folder holding only a temporary weight file being left out, and the next save rebuilds the registry from the
    folders without leaving its temporary file.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the run folder.
    """
    model = nn.Linear(in_features=2, out_features=2)
    optimizer = torch.optim.Adam(params=model.parameters())
    weights_directory = tmp_path / "Weights"
    for iteration in [2, 4]:
        save_checkpoint(weights_directory=weights_directory, iteration=iteration, model=model, optimizer=optimizer,
                        metadata={}, exponential_moving_average_state=None, auxiliary_heads_state=None)

    # A checkpoint whose weight file was never renamed from its temporary file is not complete
    (weights_directory / "Iteration_6").mkdir()
    (weights_directory / "Iteration_6" / "model_000006.pt.tmp").write_bytes(data=b"cut")

    # An empty registry, then one cut in the middle of a line
    registry_path = weights_directory / CHECKPOINT_REGISTRY_NAME
    for registry_text in ["", 'model_checkpoint_path: "Iteration_4"\nall_model_checkpoint_paths: "Itera']:
        registry_path.write_text(data=registry_text)
        assert get_last_checkpoint_iteration(weights_directory=weights_directory) == 4
        assert get_checkpoint_iterations(weights_directory=weights_directory) == [2, 4]
        assert get_checkpoint_path(experiment_folder=tmp_path, iteration=None) == (
            weights_directory / "Iteration_4" / "model_000004.pt")

    # The next save writes a whole registry again, from the complete checkpoint folders
    save_checkpoint(weights_directory=weights_directory, iteration=8, model=model, optimizer=optimizer, metadata={},
                    exponential_moving_average_state=None, auxiliary_heads_state=None)
    assert registry_path.read_text().splitlines() == [
        'model_checkpoint_path: "Iteration_8"', 'all_model_checkpoint_paths: "Iteration_2"',
        'all_model_checkpoint_paths: "Iteration_4"', 'all_model_checkpoint_paths: "Iteration_8"']
    assert not list(weights_directory.glob("*.tmp"))
