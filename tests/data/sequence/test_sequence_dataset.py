"""
Tests of SequenceDataset of vision_training/data/sequence/sequence_dataset.py: the window starts, the windows of
sequences with one class per sequence or one class per frame, the batches cut to their longest window by
collate_longest_windows, and the window counts of vision_training/data/split_description.py.
"""

import json
import cv2
import numpy as np
import pytest
import torch

from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any, Dict, List

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.sequence.sequence_dataset import (SequenceDataset, collate_longest_windows,
                                                            compute_window_starts)
from vision_training.data.split_description import count_split_sequences


def write_sequence_split(split_folder: Path, sequences: List[Dict[str, Any]]) -> None:
    """
    Writes a sequence split in the layout read by the sequence loader: the 32 pixel frames of every sequence in
    'data/<sequence name>/', the frame of index i of a sequence being filled with the value i + 1 so that the tests
    can tell which frames a window holds, and the 'labels.json' manifest.

    Args:
        split_folder (Path): The folder of the split.
        sequences (List[Dict[str, Any]]): The 'id', 'name', number of frames ('number_frames') and labels
            ('category_id' and / or 'frame_category_ids') of every sequence.
    """
    manifest_sequences = []
    for sequence in sequences:
        (split_folder / "data" / sequence["name"]).mkdir(parents=True)
        frame_files = [f"{sequence['name']}/{frame_index:06d}.png" for frame_index in range(sequence["number_frames"])]
        for frame_index, frame_file in enumerate(frame_files):
            cv2.imwrite(filename=str(split_folder / "data" / frame_file),
                        img=np.full(shape=(32, 32, 3), fill_value=frame_index + 1, dtype=np.uint8))
        manifest_sequence = {key: value for key, value in sequence.items() if key != "number_frames"}
        manifest_sequences.append({**manifest_sequence, "frames": frame_files})

    categories = [{"id": 1, "name": "left"}, {"id": 2, "name": "right"}, {"id": 3, "name": "still"}]
    (split_folder / "labels.json").write_text(data=json.dumps(obj={"categories": categories,
                                                                   "sequences": manifest_sequences}))


@pytest.mark.parametrize(argnames=("number_frames", "sequence_length", "window_stride", "short_sequences",
                                   "expected_starts"), argvalues=[
    (8, 4, 4, "pad", [0, 4]),
    (10, 4, 4, "pad", [0, 4, 6]),
    (6, 4, 1, "pad", [0, 1, 2]),
    (4, 4, 2, "skip", [0]),
    (3, 4, 4, "pad", [0]),
    (3, 4, 4, "skip", [])])
def test_window_starts(number_frames: int, sequence_length: int, window_stride: int, short_sequences: str,
                       expected_starts: List[int]) -> None:
    """
    Verifies how a sequence is cut into windows: one window every 'window_stride' frames, a last window ending on the
    last frame when the stride does not reach it, and a sequence shorter than a window padded or skipped.

    Args:
        number_frames (int): The number of frames of the sequence.
        sequence_length (int): The number of frames of a window.
        window_stride (int): The number of frames between two window starts.
        short_sequences (str): "pad" or "skip".
        expected_starts (List[int]): The expected first frame of every window.
    """
    assert compute_window_starts(number_frames=number_frames, sequence_length=sequence_length,
                                 window_stride=window_stride, short_sequences=short_sequences) == expected_starts


def test_sequence_windows_with_one_class_per_sequence(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the many to one dataset: every sequence is cut into windows carrying its class, a short sequence gives
    one window padded with black frames marked by the sequence mask, the sequences of unmapped categories are left out
    with a warning, the class counts count windows, every window has the class of its sequence, and a window is named
    by its sequence and its real frames from its identifier.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "long", "number_frames": 8, "category_id": 1},
        {"id": 2, "name": "short", "number_frames": 2, "category_id": 2},
        {"id": 3, "name": "unmapped", "number_frames": 12, "category_id": 3}])
    sequence_settings = {"sequence_length": 4, "window_stride": 4, "short_sequences": "pad"}

    dataset = SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                              image_size=16, keep_ratio=True, label_dictionary={1: 0, 2: 1},
                              sequence_settings=sequence_settings, label_type="sequence", shuffle=False)
    assert "1 sequences of unmapped categories or shorter than 4 frames left out" in capsys.readouterr().out
    assert dataset.class_names == {0: "left", 1: "right"}

    # The long sequence is cut into the windows [0, 1, 2, 3] and [4, 5, 6, 7]
    assert len(dataset) == 3 and dataset.get_class_counts() == [2, 1]
    assert dataset.get_sample_class_indices() == {0: [0], 1: [0], 2: [1]}
    first_window, second_window, short_window = dataset[0], dataset[1], dataset[2]
    assert first_window["images"].shape == (4, 3, 16, 16) and first_window["images"].dtype == torch.uint8
    assert first_window["images"][:, 0, 8, 8].tolist() == [1, 2, 3, 4]
    assert second_window["images"][:, 0, 8, 8].tolist() == [5, 6, 7, 8]
    assert first_window["labels"].dtype == torch.int64 and int(first_window["labels"]) == 0
    assert first_window["sample_identifiers"] != second_window["sample_identifiers"]

    # The short sequence keeps its frames 0 and 1, padded with two black frames
    assert short_window["sequence_mask"].tolist() == [True, True, False, False]
    assert short_window["images"][:, 0, 8, 8].tolist() == [1, 2, 0, 0]
    assert int(short_window["labels"]) == 1

    # Every window is named by its sequence and its first and last real frames
    assert dataset.describe_sample(sample_identifier=second_window["sample_identifiers"]) == {
        "sequence": "long", "first_frame": "long/000004.png", "last_frame": "long/000007.png"}
    assert dataset.get_window_frame_names(sample_identifier=short_window["sample_identifiers"]) == [
        "short/000000.png", "short/000001.png"]


def test_sequence_windows_with_one_class_per_frame(tmp_path: Path) -> None:
    """
    Verifies the many to one dataset with one class per frame: every frame gets its class, the frames of unmapped
    categories and the padding frames get the ignored label -100, and the class counts and the classes of the window
    count the labelled frames only.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "changing", "number_frames": 3, "frame_category_ids": [1, 3, 2]}])
    sequence_settings = {"sequence_length": 4, "window_stride": 4, "short_sequences": "pad"}

    dataset = SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                              image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                              sequence_settings=sequence_settings, label_type="frame", shuffle=False)
    window = dataset[0]
    assert window["labels"].tolist() == [0, IGNORED_LABEL_INDEX, 1, IGNORED_LABEL_INDEX]
    assert window["sequence_mask"].tolist() == [True, True, True, False]
    assert dataset.get_class_counts() == [1, 1]
    assert dataset.get_sample_class_indices() == {0: [0, 1]}


@pytest.mark.parametrize(argnames="label_type", argvalues=["sequence", "frame"])
def test_collate_longest_windows_cuts_the_batches_to_their_longest_window(tmp_path: Path, label_type: str) -> None:
    """
    Verifies that the batches of windows of 2, 3 and 5 real frames, padded to 8 frames by the dataset, are cut to the
    5 frames of their longest window by collate_longest_windows: the frames, the sequence mask and the frame labels
    are the first 5 positions of the batch of the default collate, while the labels of one class per window and the
    window identifiers are unchanged.

    Args:
        tmp_path (Path): The pytest temporary directory.
        label_type (str): "sequence" (one class per window) or "frame" (one class per frame).
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "two", "number_frames": 2, "category_id": 1, "frame_category_ids": [1, 2]},
        {"id": 2, "name": "three", "number_frames": 3, "category_id": 2, "frame_category_ids": [2, 1, 2]},
        {"id": 3, "name": "five", "number_frames": 5, "category_id": 1, "frame_category_ids": [1, 1, 2, 2, 1]}])
    sequence_settings = {"sequence_length": 8, "window_stride": 8, "short_sequences": "pad"}
    dataset = SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                              image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                              sequence_settings=sequence_settings, label_type=label_type, shuffle=False)

    # The same windows batched by the default collate and by the collate of the longest window
    padded_batch = next(iter(DataLoader(dataset=dataset, batch_size=3, shuffle=False)))
    cut_batch = next(iter(DataLoader(dataset=dataset, batch_size=3, shuffle=False,
                                     collate_fn=partial(collate_longest_windows, members_per_sample=1))))

    assert padded_batch["images"].shape == (3, 8, 3, 16, 16)
    assert cut_batch["images"].shape == (3, 5, 3, 16, 16)
    assert cut_batch["sequence_mask"].sum(dim=1).tolist() == [2, 3, 5]
    assert torch.equal(input=cut_batch["images"], other=padded_batch["images"][:, :5])
    assert torch.equal(input=cut_batch["sequence_mask"], other=padded_batch["sequence_mask"][:, :5])
    assert torch.equal(input=cut_batch["sample_identifiers"], other=padded_batch["sample_identifiers"])
    if label_type == "frame":
        assert torch.equal(input=cut_batch["labels"], other=padded_batch["labels"][:, :5])
    else:
        assert torch.equal(input=cut_batch["labels"], other=padded_batch["labels"])


def test_split_description_counts_windows(tmp_path: Path) -> None:
    """
    Verifies that the split description counts the sequences, the frames and the windows, the windows being the
    samples of an epoch, a short sequence giving no window when the short sequences are skipped.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "long", "number_frames": 12, "category_id": 1},
        {"id": 2, "name": "short", "number_frames": 2, "category_id": 2}])
    sequence_settings = {"sequence_length": 4, "window_stride": 4, "short_sequences": "skip"}
    assert count_split_sequences(split_data_folder=tmp_path, sequence_settings=sequence_settings) == (2, 14, 3)
