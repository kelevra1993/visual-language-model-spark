"""
Tests of vision_training/trainers/checkpointing.py: the checkpoint registry naming the last checkpoint and keeping the
history, a registry that cannot be read replaced by the complete checkpoint folders, and a failed save leaving no
temporary file.
"""

import errno
import pytest
import torch

from pathlib import Path
from torch import nn
from unittest.mock import patch

from vision_training.trainers import checkpointing
from vision_training.trainers.checkpointing import (CHECKPOINT_REGISTRY_NAME, get_checkpoint_iterations,
                                                    get_checkpoint_path, get_last_checkpoint_iteration,
                                                    save_checkpoint, update_checkpoint_registry)

# The real replace_durably, called by the stand-in that fails only for the registry
ORIGINAL_REPLACE_DURABLY = checkpointing.replace_durably


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


def write_partial_file_then_fail(obj: object, f: Path) -> None:
    """
    Stands in for torch.save on a disk filling up during the save of a checkpoint: the first bytes of the weight file
    reach its temporary file, then the write fails as on a full disk.

    Args:
        obj (object): The checkpoint to save, left unwritten.
        f (Path): The temporary weight file.

    Raises:
        OSError: Always, with the error number of a full disk.
    """
    Path(f).write_bytes(data=b"partial")
    raise OSError(errno.ENOSPC, "No space left on device")


def fail_registry_replacement(temporary_file_path: Path, file_path: Path) -> None:
    """
    Stands in for replace_durably on a disk filling up between the weight file and the registry: the weight file
    reaches its final name, then the registry written to its temporary file fails to replace the previous one.

    Args:
        temporary_file_path (Path): The written temporary file.
        file_path (Path): The final file.

    Raises:
        OSError: For the registry, with the error number of a full disk.
    """
    if file_path.name == CHECKPOINT_REGISTRY_NAME:
        raise OSError(errno.ENOSPC, "No space left on device")
    ORIGINAL_REPLACE_DURABLY(temporary_file_path=temporary_file_path, file_path=file_path)


def test_failed_save_leaves_no_temporary_file(tmp_path: Path) -> None:
    """
    Verifies that a save failing on a full disk raises its error without leaving any temporary file: a weight file cut
    by the full disk is removed with the registry untouched, and a registry that cannot be written is removed too, the
    registry on the disk still naming the previous checkpoint.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the run folder.
    """
    model = nn.Linear(in_features=2, out_features=2)
    optimizer = torch.optim.Adam(params=model.parameters())
    weights_directory = tmp_path / "Weights"
    save_checkpoint(weights_directory=weights_directory, iteration=2, model=model, optimizer=optimizer, metadata={},
                    exponential_moving_average_state=None)

    # The weight file cut by the full disk is removed, and the registry keeps naming the checkpoint of iteration 2
    with patch.object(target=checkpointing.torch, attribute="save", side_effect=write_partial_file_then_fail):
        with pytest.raises(expected_exception=OSError, match="No space left on device"):
            save_checkpoint(weights_directory=weights_directory, iteration=4, model=model, optimizer=optimizer,
                            metadata={}, exponential_moving_average_state=None)
    assert not list(weights_directory.rglob(pattern="*.tmp"))
    assert get_checkpoint_iterations(weights_directory=weights_directory) == [2]

    # The registry that cannot replace the previous one is removed, which still names the checkpoint of iteration 2
    with patch.object(target=checkpointing, attribute="replace_durably", side_effect=fail_registry_replacement):
        with pytest.raises(expected_exception=OSError, match="No space left on device"):
            save_checkpoint(weights_directory=weights_directory, iteration=6, model=model, optimizer=optimizer,
                            metadata={}, exponential_moving_average_state=None)
    assert not list(weights_directory.rglob(pattern="*.tmp"))
    assert get_last_checkpoint_iteration(weights_directory=weights_directory) == 2
