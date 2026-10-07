"""
Tests of SequencePairDataset of vision_training/data/sequence/sequence_pair_dataset.py for the sequence comparison task:
the member frame indices, pairs read as the before then the after member, short or empty members left out, the pair
counts and pair batches split into their members.
"""

import json
import cv2
import numpy as np
import pytest
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any, Dict, List

from vision_training.data.sequence.sequence_pair_dataset import (SequencePairDataset, compute_member_frame_indices,
                                                                 split_pair_members)
from vision_training.data.split_description import count_split_pairs

# Pixel value of the first frame of every member, the frame of index i holding the value plus i, so that the tests
# tell the frames of the two members apart
MEMBER_FIRST_VALUES = {"before": 1, "after": 101}


def write_pair_split(split_folder: Path, pairs: List[Dict[str, Any]]) -> None:
    """
    Writes a sequence comparison split in the layout read by the sequence pair dataset: the 16 pixel frames of the
    members of every pair in 'data/<pair name>/before/' and '.../after/', the before frame of index i being filled with
    the value i + 1 and the after frame of index i with the value i + 101, so that the tests can tell which frames a
    sample holds, and the 'labels.json' manifest.

    Args:
        split_folder (Path): The folder of the split.
        pairs (List[Dict[str, Any]]): The 'id', 'name', 'category_id' and numbers of frames of the members
            ('before_frames' and 'after_frames') of every pair.
    """
    manifest_pairs = []
    for pair in pairs:
        manifest_pair = {"id": pair["id"], "name": pair["name"]}

        # Frames of every member, in their own folder
        for member_name, first_value in MEMBER_FIRST_VALUES.items():
            (split_folder / "data" / pair["name"] / member_name).mkdir(parents=True)
            frame_files = [f"{pair['name']}/{member_name}/{frame_index:06d}.png"
                           for frame_index in range(pair[f"{member_name}_frames"])]
            for frame_index, frame_file in enumerate(frame_files):
                cv2.imwrite(filename=str(split_folder / "data" / frame_file),
                            img=np.full(shape=(16, 16, 3), fill_value=frame_index + first_value, dtype=np.uint8))
            manifest_pair[member_name] = frame_files
        manifest_pair["category_id"] = pair["category_id"]
        manifest_pairs.append(manifest_pair)

    categories = [{"id": 1, "name": "loss_of_object"}, {"id": 2, "name": "same_number_of_objects"},
                  {"id": 3, "name": "unknown"}]
    (split_folder / "labels.json").write_text(data=json.dumps(obj={"categories": categories, "pairs": manifest_pairs}))


@pytest.mark.parametrize(argnames=("number_frames", "sequence_length", "short_sequences", "expected_indices"),
                         argvalues=[(3, 4, "pad", [0, 1, 2]), (3, 4, "skip", []), (0, 4, "pad", []),
                                    (4, 4, "skip", [0, 1, 2, 3]), (10, 4, "pad", [1, 3, 6, 8]),
                                    (8, 4, "pad", [1, 3, 5, 7]), (6, 4, "pad", [0, 2, 3, 5]), (5, 1, "pad", [2])])
def test_member_frame_indices(number_frames: int, sequence_length: int, short_sequences: str,
                              expected_indices: List[int]) -> None:
    """
    Verifies the frames chosen for a member window: every frame of a member that fits, the middle frame of each of the
    equal parts of a longer member, and none for an empty member or for a short member when short members are skipped.

    Args:
        number_frames (int): The number of frames of the member.
        sequence_length (int): The maximum number of frames of a member window.
        short_sequences (str): "pad" or "skip".
        expected_indices (List[int]): The expected indices of the kept frames.
    """
    assert compute_member_frame_indices(number_frames=number_frames, sequence_length=sequence_length,
                                        short_sequences=short_sequences) == expected_indices


def test_pairs_are_read_as_before_then_after_members(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the samples of the sequence pair dataset: the before window followed by the after window along the frame
    axis, a longer member reduced to evenly spaced frames, a shorter one padded with black frames marked by the
    sequence mask, the frame stride applied before, the class of the pair, distinct identifiers naming the pairs, the
    class names, counts and the class of every pair, and the pairs of unmapped categories left out with a warning.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    write_pair_split(split_folder=tmp_path, pairs=[
        {"id": 1, "name": "pair_a", "category_id": 1, "before_frames": 6, "after_frames": 2},
        {"id": 2, "name": "pair_b", "category_id": 2, "before_frames": 4, "after_frames": 4},
        {"id": 3, "name": "pair_c", "category_id": 3, "before_frames": 2, "after_frames": 2}])
    sequence_settings = {"sequence_length": 4, "frame_stride": 1, "short_sequences": "pad"}
    dataset = SequencePairDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                  image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                                  sequence_settings=sequence_settings, shuffle=False)

    assert len(dataset) == 2
    assert "1 pairs of unmapped categories" in capsys.readouterr().out
    assert dataset.class_names == {0: "loss_of_object", 1: "same_number_of_objects"}
    assert dataset.get_class_counts() == [1, 1]
    assert dataset.get_sample_class_indices() == {0: [0], 1: [1]}

    # The before member keeps the middle frames of 4 equal parts (frames 0, 2, 3 and 5), the after member is padded
    first_sample = dataset[0]
    assert first_sample["images"].shape == (8, 3, 16, 16) and first_sample["images"].dtype == torch.uint8
    assert first_sample["images"][:, 0, 0, 0].tolist() == [1, 3, 4, 6, 101, 102, 0, 0]
    assert first_sample["sequence_mask"].tolist() == [True] * 6 + [False] * 2
    assert first_sample["labels"].dtype == torch.int64 and int(first_sample["labels"]) == 0
    assert [dataset[pair_index]["sample_identifiers"] for pair_index in range(2)] == [0, 1]
    assert dataset.describe_sample(sample_identifier=1) == {"pair": "pair_b"}

    # With a frame stride of 2, the before member keeps frames 0, 2 and 4, then is padded
    strided_dataset = SequencePairDataset(data_directory=str(tmp_path / "data"),
                                          labels_file=str(tmp_path / "labels.json"), image_size=16, keep_ratio=False,
                                          label_dictionary={1: 0, 2: 1},
                                          sequence_settings={**sequence_settings, "frame_stride": 2}, shuffle=False)
    assert strided_dataset[0]["images"][:4, 0, 0, 0].tolist() == [1, 3, 5, 0]


def test_pairs_with_a_short_or_empty_member_are_left_out(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a pair is left out when one of its members is shorter than the window and short members are skipped,
    and always when one of its members is empty, a member window needing at least one real frame.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warnings.
    """
    write_pair_split(split_folder=tmp_path, pairs=[
        {"id": 1, "name": "short", "category_id": 1, "before_frames": 4, "after_frames": 2},
        {"id": 2, "name": "empty", "category_id": 2, "before_frames": 0, "after_frames": 3},
        {"id": 3, "name": "full", "category_id": 2, "before_frames": 5, "after_frames": 4}])

    # Short members skipped, then padded
    datasets = {short_sequences: SequencePairDataset(
        data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"), image_size=16,
        keep_ratio=False, label_dictionary={1: 0, 2: 1},
        sequence_settings={"sequence_length": 4, "frame_stride": 1, "short_sequences": short_sequences}, shuffle=False)
        for short_sequences in ["skip", "pad"]}

    assert len(datasets["skip"]) == 1 and len(datasets["pad"]) == 2
    printed_output = capsys.readouterr().out
    assert ("2 pairs of unmapped categories, with an empty member or with a member shorter than 4 frames" in
            printed_output)
    assert "1 pairs of unmapped categories, with an empty member" in printed_output


def test_split_description_counts_pairs(tmp_path: Path) -> None:
    """
    Verifies the split description of the comparison task: the pairs of the manifest, the frames of both members, and
    the pairs kept as the dataset keeps them.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_pair_split(split_folder=tmp_path, pairs=[
        {"id": 1, "name": "short", "category_id": 1, "before_frames": 4, "after_frames": 2},
        {"id": 2, "name": "empty", "category_id": 2, "before_frames": 0, "after_frames": 3},
        {"id": 3, "name": "full", "category_id": 2, "before_frames": 5, "after_frames": 4}])

    for short_sequences, expected_kept_pairs in [("skip", 1), ("pad", 2)]:
        assert count_split_pairs(split_data_folder=tmp_path, sequence_settings={
            "sequence_length": 4, "frame_stride": 1, "short_sequences": short_sequences}) == (3, 18,
                                                                                              expected_kept_pairs)


def test_pair_batches_split_into_members(tmp_path: Path) -> None:
    """
    Verifies that the default collate stacks the pairs into [B, 2T] batches, which split back into the before and
    after members given to the comparison model, and that a batch holding another number of frames is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_pair_split(split_folder=tmp_path, pairs=[
        {"id": 1, "name": "pair_a", "category_id": 1, "before_frames": 3, "after_frames": 2},
        {"id": 2, "name": "pair_b", "category_id": 2, "before_frames": 4, "after_frames": 1}])
    dataset = SequencePairDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                  image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                                  sequence_settings={"sequence_length": 4, "frame_stride": 1,
                                                     "short_sequences": "pad"}, shuffle=False)
    batch = next(iter(DataLoader(dataset=dataset, batch_size=2, shuffle=False)))

    before_images, after_images = split_pair_members(pair_tensor=batch["images"], sequence_length=4)
    before_mask, after_mask = split_pair_members(pair_tensor=batch["sequence_mask"], sequence_length=4)

    assert batch["images"].shape == (2, 8, 3, 16, 16)
    assert before_images.shape == after_images.shape == (2, 4, 3, 16, 16)
    assert before_mask.tolist() == [[True, True, True, False], [True, True, True, True]]
    assert after_mask.tolist() == [[True, True, False, False], [True, False, False, False]]
    assert after_images[:, 0, 0, 0, 0].tolist() == [101, 101]
    with pytest.raises(expected_exception=ValueError, match="2 x 3 frames"):
        split_pair_members(pair_tensor=batch["images"], sequence_length=3)
