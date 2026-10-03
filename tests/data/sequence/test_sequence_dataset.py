import json
import cv2
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.sequence.sequence_dataset import SequenceDataset, compute_window_starts
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
    Verifies the many to one dataset: the frame stride keeps one frame every two, every sequence is cut into windows
    carrying its class, a short sequence gives one window padded with black frames marked by the sequence mask, the
    sequences of unmapped categories are left out with a warning, the class counts count windows, and a window is
    named by its sequence and its real frames from its identifier.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "long", "number_frames": 12, "category_id": 1},
        {"id": 2, "name": "short", "number_frames": 4, "category_id": 2},
        {"id": 3, "name": "unmapped", "number_frames": 12, "category_id": 3}])
    sequence_settings = {"sequence_length": 4, "frame_stride": 2, "window_stride": 4, "short_sequences": "pad"}

    dataset = SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                              image_size=16, keep_ratio=True, label_dictionary={1: 0, 2: 1},
                              sequence_settings=sequence_settings, label_type="sequence", shuffle=False)
    assert "1 sequences of unmapped categories or shorter than 4 frames left out" in capsys.readouterr().out
    assert dataset.class_names == {0: "left", 1: "right"}

    # The long sequence keeps frames 0, 2, ..., 10, cut into the windows [0, 2, 4, 6] and [4, 6, 8, 10]
    assert len(dataset) == 3 and dataset.get_class_counts() == [2, 1]
    first_window, second_window, short_window = dataset[0], dataset[1], dataset[2]
    assert first_window["images"].shape == (4, 3, 16, 16) and first_window["images"].dtype == torch.uint8
    assert first_window["images"][:, 0, 8, 8].tolist() == [1, 3, 5, 7]
    assert second_window["images"][:, 0, 8, 8].tolist() == [5, 7, 9, 11]
    assert first_window["labels"].dtype == torch.int64 and int(first_window["labels"]) == 0
    assert first_window["sample_identifiers"] != second_window["sample_identifiers"]

    # The short sequence keeps frames 0 and 2, padded with two black frames
    assert short_window["sequence_mask"].tolist() == [True, True, False, False]
    assert short_window["images"][:, 0, 8, 8].tolist() == [1, 3, 0, 0]
    assert int(short_window["labels"]) == 1

    # Every window is named by its sequence and its first and last real frames
    assert dataset.describe_sample(sample_identifier=second_window["sample_identifiers"]) == {
        "sequence": "long", "first_frame": "long/000004.png", "last_frame": "long/000010.png"}
    assert dataset.get_window_frame_names(sample_identifier=short_window["sample_identifiers"]) == [
        "short/000000.png", "short/000002.png"]


def test_sequence_windows_with_one_class_per_frame(tmp_path: Path) -> None:
    """
    Verifies the many to one dataset with one class per frame: every frame gets its class, the frames of unmapped
    categories and the padding frames get the ignored label -100, and the class counts count the labelled frames.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "changing", "number_frames": 3, "frame_category_ids": [1, 3, 2]}])
    sequence_settings = {"sequence_length": 4, "frame_stride": 1, "window_stride": 4, "short_sequences": "pad"}

    dataset = SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                              image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                              sequence_settings=sequence_settings, label_type="frame", shuffle=False)
    window = dataset[0]
    assert window["labels"].tolist() == [0, IGNORED_LABEL_INDEX, 1, IGNORED_LABEL_INDEX]
    assert window["sequence_mask"].tolist() == [True, True, True, False]
    assert dataset.get_class_counts() == [1, 1]


def test_split_description_counts_windows(tmp_path: Path) -> None:
    """
    Verifies that the split description counts the sequences, the frames and the windows cut after the frame stride,
    the windows being the samples of an epoch.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "long", "number_frames": 12, "category_id": 1},
        {"id": 2, "name": "short", "number_frames": 2, "category_id": 2}])
    sequence_settings = {"sequence_length": 4, "frame_stride": 2, "window_stride": 4, "short_sequences": "skip"}
    assert count_split_sequences(split_data_folder=tmp_path, sequence_settings=sequence_settings) == (2, 14, 2)
