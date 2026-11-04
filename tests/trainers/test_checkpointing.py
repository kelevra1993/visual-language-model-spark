"""
Tests of vision_training/trainers/checkpointing.py: the checkpoint registry naming the last checkpoint and keeping the
history, and a registry that cannot be read replaced by the complete checkpoint folders.
"""

import torch

from pathlib import Path
from torch import nn

from vision_training.trainers.checkpointing import (CHECKPOINT_REGISTRY_NAME, get_checkpoint_iterations,
                                                    get_checkpoint_path, get_last_checkpoint_iteration,
                                                    save_checkpoint, update_checkpoint_registry)


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
                        metadata={}, exponential_moving_average_state=None)

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
                    exponential_moving_average_state=None)
    assert registry_path.read_text().splitlines() == [
        'model_checkpoint_path: "Iteration_8"', 'all_model_checkpoint_paths: "Iteration_2"',
        'all_model_checkpoint_paths: "Iteration_4"', 'all_model_checkpoint_paths: "Iteration_8"']
    assert not list(weights_directory.glob("*.tmp"))
