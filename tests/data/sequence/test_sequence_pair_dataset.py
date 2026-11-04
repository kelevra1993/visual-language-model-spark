"""
Tests of SequencePairDataset of vision_training/data/sequence/sequence_pair_dataset.py for the sequence comparison task:
pairs read as the before then the after member padded to the longer member, pairs with a member longer than the
maximum sequence length or empty left out, pair batches split into their members, and pair batches padded to their
longest member by collate_padded_sequences.
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

from vision_training.data.sequence.sequence_dataset import collate_padded_sequences
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS, SequencePairDataset, split_pair_members

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


def test_pairs_are_read_as_before_then_after_members(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the samples of the sequence pair dataset: the whole before member followed by the whole after member
    along the frame axis, the shorter member padded with black frames to the longer one and marked by the sequence
    mask, the class of the pair, distinct identifiers naming the pairs, the class names, counts and the class of every
    pair, and the pairs of unmapped categories left out with a warning.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    write_pair_split(split_folder=tmp_path, pairs=[
        {"id": 1, "name": "pair_a", "category_id": 1, "before_frames": 6, "after_frames": 2},
        {"id": 2, "name": "pair_b", "category_id": 2, "before_frames": 4, "after_frames": 4},
        {"id": 3, "name": "pair_c", "category_id": 3, "before_frames": 2, "after_frames": 2}])
    dataset = SequencePairDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                  image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                                  sequence_settings={"maximum_sequence_length": 6}, split_name="Training",
                                  shuffle=False)

    assert len(dataset) == 2
    assert "1 pairs of unmapped categories or with an empty member left out" in capsys.readouterr().out
    assert dataset.class_names == {0: "loss_of_object", 1: "same_number_of_objects"}
    assert dataset.get_class_counts() == [1, 1]
    assert dataset.get_sample_class_indices() == {0: [0], 1: [1]}

    # Both members hold 6 frames, the after member being padded with 4 black frames
    first_sample = dataset[0]
    assert first_sample["images"].shape == (12, 3, 16, 16) and first_sample["images"].dtype == torch.uint8
    assert first_sample["images"][:, 0, 0, 0].tolist() == [1, 2, 3, 4, 5, 6, 101, 102, 0, 0, 0, 0]
    assert first_sample["sequence_mask"].tolist() == [True] * 8 + [False] * 4
    assert first_sample["labels"].dtype == torch.int64 and int(first_sample["labels"]) == 0
    assert dataset[1]["images"].shape == (8, 3, 16, 16)
    assert [dataset[pair_index]["sample_identifiers"] for pair_index in range(2)] == [0, 1]
    assert dataset.describe_sample(sample_identifier=1) == {"pair": "pair_b"}


def test_pairs_with_a_long_or_empty_member_are_left_out(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that a pair is left out with a red warning of its own and a total when one of its members is longer than
    the maximum sequence length, and with the yellow warning when one of its members is empty, a member needing at
    least one real frame.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warnings.
    """
    write_pair_split(split_folder=tmp_path, pairs=[
        {"id": 1, "name": "long", "category_id": 1, "before_frames": 2, "after_frames": 5},
        {"id": 2, "name": "empty", "category_id": 2, "before_frames": 0, "after_frames": 3},
        {"id": 3, "name": "full", "category_id": 2, "before_frames": 4, "after_frames": 4}])
    dataset = SequencePairDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                  image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                                  sequence_settings={"maximum_sequence_length": 4}, split_name="Validation",
                                  shuffle=False)

    assert len(dataset) == 1 and dataset.get_sample_name(sample_identifier=0) == "full"
    printed_output = capsys.readouterr().out
    assert ("Pair 'long' Of The Validation Split Has A Member Of 5 Frames, More Than The Maximum Sequence Length (4): "
            "It Is Left Out Of The Validation." in printed_output)
    assert ("1 Pairs Of The Validation Split Have A Member Longer Than The Maximum Sequence Length (4) And Are Left "
            "Out Of The Validation." in printed_output)
    assert "1 pairs of unmapped categories or with an empty member left out" in printed_output


def test_pair_batches_split_into_members(tmp_path: Path) -> None:
    """
    Verifies that pairs of the same member length stack into [B, 2T] batches with the default collate, which split
    back into the before and after members given to the comparison model, and that a batch whose frame axis cannot
    hold two members of the same number of frames is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_pair_split(split_folder=tmp_path, pairs=[
        {"id": 1, "name": "pair_a", "category_id": 1, "before_frames": 4, "after_frames": 2},
        {"id": 2, "name": "pair_b", "category_id": 2, "before_frames": 1, "after_frames": 4}])
    dataset = SequencePairDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                  image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                                  sequence_settings={"maximum_sequence_length": 4}, split_name="Training",
                                  shuffle=False)
    batch = next(iter(DataLoader(dataset=dataset, batch_size=2, shuffle=False)))

    before_images, after_images = split_pair_members(pair_tensor=batch["images"])
    before_mask, after_mask = split_pair_members(pair_tensor=batch["sequence_mask"])

    assert batch["images"].shape == (2, 8, 3, 16, 16)
    assert before_images.shape == after_images.shape == (2, 4, 3, 16, 16)
    assert before_mask.tolist() == [[True, True, True, True], [True, False, False, False]]
    assert after_mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert after_images[:, 0, 0, 0, 0].tolist() == [101, 101]
    with pytest.raises(expected_exception=ValueError, match="got 7 frames per pair"):
        split_pair_members(pair_tensor=batch["images"][:, :7])


def test_pair_batches_padded_to_their_longest_member(tmp_path: Path) -> None:
    """
    Verifies that collate_padded_sequences pads both members of every pair of a batch to the longest member of the
    batch: pairs of (3, 2) and (1, 2) real frames, stored as members of 3 and 2 frames by the dataset, give
    [B, 2 x 3] batches whose before and after members keep their real frames first.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_pair_split(split_folder=tmp_path, pairs=[
        {"id": 1, "name": "pair_a", "category_id": 1, "before_frames": 3, "after_frames": 2},
        {"id": 2, "name": "pair_b", "category_id": 2, "before_frames": 1, "after_frames": 2}])
    dataset = SequencePairDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                  image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                                  sequence_settings={"maximum_sequence_length": 6}, split_name="Training",
                                  shuffle=False)
    batch = next(iter(DataLoader(dataset=dataset, batch_size=2, shuffle=False,
                                 collate_fn=partial(collate_padded_sequences, members_per_sample=len(PAIR_MEMBERS)))))

    assert batch["images"].shape == (2, 6, 3, 16, 16)
    before_images, after_images = split_pair_members(pair_tensor=batch["images"])
    before_mask, after_mask = split_pair_members(pair_tensor=batch["sequence_mask"])
    assert before_mask.sum(dim=1).tolist() == [3, 1] and after_mask.sum(dim=1).tolist() == [2, 2]
    assert before_images[1, :, 0, 0, 0].tolist() == [1, 0, 0]
    assert after_images[1, :, 0, 0, 0].tolist() == [101, 102, 0]
    assert batch["labels"].tolist() == [0, 1]
