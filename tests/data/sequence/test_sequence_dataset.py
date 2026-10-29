"""
Tests of SequenceDataset of vision_training/data/sequence/sequence_dataset.py: the whole sequences with one class per
sequence or one class per frame, the sequences longer than the maximum sequence length left out with red warnings, the
whole sequences of the frame selection with their unselected frames and their check, the batches padded to their
longest sequence by collate_padded_sequences, the sequence counts of vision_training/data/split_description.py, and
the training loader of a head needing batch statistics leaving out its last incomplete batch.
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
from vision_training.data.loaders import get_sequence_dataloaders
from vision_training.data.sequence.sequence_dataset import SequenceDataset, collate_padded_sequences
from vision_training.data.split_description import count_split_sequences


def write_sequence_split(split_folder: Path, sequences: List[Dict[str, Any]]) -> None:
    """
    Writes a sequence split in the layout read by the sequence loader: the 32 pixel frames of every sequence in
    'data/<sequence name>/', the frame of index i of a sequence being filled with the value i + 1 so that the tests
    can tell which frames a sample holds, and the 'labels.json' manifest.

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


def test_whole_sequences_with_one_class_per_sequence(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the many to one dataset: every sequence is one sample holding all its frames with the class of the
    sequence, the sequences of unmapped categories or without frames are left out with a yellow warning, a sequence
    longer than the maximum sequence length is left out with a red warning of its own and a total, the class counts
    count sequences, and a sequence is named by its frames from its identifier.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warnings.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "long", "number_frames": 8, "category_id": 1},
        {"id": 2, "name": "short", "number_frames": 2, "category_id": 2},
        {"id": 3, "name": "unmapped", "number_frames": 3, "category_id": 3},
        {"id": 4, "name": "empty", "number_frames": 0, "category_id": 1},
        {"id": 5, "name": "too_long", "number_frames": 9, "category_id": 1}])

    dataset = SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                              image_size=16, keep_ratio=True, label_dictionary={1: 0, 2: 1},
                              sequence_settings={"maximum_sequence_length": 8}, label_type="sequence",
                              split_name="Training", shuffle=False)
    printed_output = capsys.readouterr().out
    assert "2 sequences of unmapped categories or without frames left out" in printed_output
    assert ("Sequence 'too_long' Of The Training Split Has 9 Frames, More Than The Maximum Sequence Length (8): It Is "
            "Left Out Of The Training." in printed_output)
    assert ("1 Sequences Of The Training Split Are Longer Than The Maximum Sequence Length (8) And Are Left Out Of The "
            "Training." in printed_output)
    assert dataset.class_names == {0: "left", 1: "right"}

    # Every kept sequence is one sample of all its frames, without padding frames
    assert len(dataset) == 2 and dataset.get_class_counts() == [1, 1]
    assert dataset.get_sample_class_indices() == {0: [0], 1: [1]}
    long_sequence, short_sequence = dataset[0], dataset[1]
    assert long_sequence["images"].shape == (8, 3, 16, 16) and long_sequence["images"].dtype == torch.uint8
    assert long_sequence["images"][:, 0, 8, 8].tolist() == list(range(1, 9))
    assert long_sequence["sequence_mask"].tolist() == [True] * 8
    assert long_sequence["labels"].dtype == torch.int64 and int(long_sequence["labels"]) == 0
    assert short_sequence["images"].shape == (2, 3, 16, 16) and int(short_sequence["labels"]) == 1

    # Every sequence is named by its first and last frames, and lists all its frames
    assert dataset.describe_sample(sample_identifier=long_sequence["sample_identifiers"]) == {
        "sequence": "long", "first_frame": "long/000000.png", "last_frame": "long/000007.png"}
    assert dataset.get_sequence_frame_names(sample_identifier=short_sequence["sample_identifiers"]) == [
        "short/000000.png", "short/000001.png"]
    assert dataset.get_sample_name(sample_identifier=short_sequence["sample_identifiers"]) == "short"


def test_whole_sequences_with_one_class_per_frame(tmp_path: Path) -> None:
    """
    Verifies the many to many dataset: every frame gets its class, the frames of unmapped categories get the ignored
    label -100, and the class counts and the classes of the sequence count the labelled frames only.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "changing", "number_frames": 3, "frame_category_ids": [1, 3, 2]}])

    dataset = SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                              image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                              sequence_settings={"maximum_sequence_length": 50}, label_type="frame",
                              split_name="Training", shuffle=False)
    sequence = dataset[0]
    assert sequence["labels"].tolist() == [0, IGNORED_LABEL_INDEX, 1]
    assert sequence["sequence_mask"].tolist() == [True, True, True]
    assert dataset.get_class_counts() == [1, 1]
    assert dataset.get_sample_class_indices() == {0: [0, 1]}


@pytest.mark.parametrize(argnames="label_type", argvalues=["sequence", "frame"])
def test_collate_pads_the_batches_to_their_longest_sequence(tmp_path: Path, label_type: str) -> None:
    """
    Verifies that the sequences of 2, 3 and 5 frames are padded to the 5 frames of their longest sequence by
    collate_padded_sequences: the real frames first, then black frames out of the sequence mask, the frame labels
    padded with the ignored label -100, while the labels of one class per sequence and the identifiers are unchanged.

    Args:
        tmp_path (Path): The pytest temporary directory.
        label_type (str): "sequence" (one class per sequence) or "frame" (one class per frame).
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "two", "number_frames": 2, "category_id": 1, "frame_category_ids": [1, 2]},
        {"id": 2, "name": "three", "number_frames": 3, "category_id": 2, "frame_category_ids": [2, 1, 2]},
        {"id": 3, "name": "five", "number_frames": 5, "category_id": 1, "frame_category_ids": [1, 1, 2, 2, 1]}])
    dataset = SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                              image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                              sequence_settings={"maximum_sequence_length": 8}, label_type=label_type,
                              split_name="Training", shuffle=False)
    batch = next(iter(DataLoader(dataset=dataset, batch_size=3, shuffle=False,
                                 collate_fn=partial(collate_padded_sequences, members_per_sample=1))))

    assert batch["images"].shape == (3, 5, 3, 16, 16)
    assert batch["sequence_mask"].tolist() == [[True] * 2 + [False] * 3, [True] * 3 + [False] * 2, [True] * 5]
    assert batch["images"][0, :, 0, 8, 8].tolist() == [1, 2, 0, 0, 0]
    assert batch["images"][2, :, 0, 8, 8].tolist() == [1, 2, 3, 4, 5]
    assert batch["sample_identifiers"].tolist() == [0, 1, 2]
    if label_type == "frame":
        assert batch["labels"][0].tolist() == [0, 1] + [IGNORED_LABEL_INDEX] * 3
        assert batch["labels"][1].tolist() == [1, 0, 1] + [IGNORED_LABEL_INDEX] * 2
    else:
        assert batch["labels"].tolist() == [0, 1, 0]


def test_collate_pads_every_member_of_the_pairs() -> None:
    """
    Verifies that collate_padded_sequences pads both members of every pair to the longest member of the batch: pairs
    stored with members of 2 and 3 frames give [B, 2 x 3] batches, the padding frames of every member following its
    real frames.
    """
    samples = [{"images": torch.arange(start=1, end=2 * member_length + 1).reshape(2 * member_length, 1),
                "sequence_mask": torch.ones(size=(2 * member_length,), dtype=torch.bool),
                "labels": torch.tensor(data=member_length, dtype=torch.int64), "sample_identifiers": member_length}
               for member_length in [2, 3]]
    batch = collate_padded_sequences(samples=samples, members_per_sample=2)

    assert batch["images"].shape == (2, 6, 1)
    assert batch["images"][0, :, 0].tolist() == [1, 2, 0, 3, 4, 0]
    assert batch["images"][1, :, 0].tolist() == [1, 2, 3, 4, 5, 6]
    assert batch["sequence_mask"][0].tolist() == [True, True, False, True, True, False]
    assert batch["labels"].tolist() == [2, 3]


def test_split_description_counts_kept_sequences(tmp_path: Path) -> None:
    """
    Verifies that the split description counts the sequences, the frames and the kept sequences, the samples of an
    epoch, a sequence without frames or longer than the maximum sequence length not being kept.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "long", "number_frames": 12, "category_id": 1},
        {"id": 2, "name": "short", "number_frames": 2, "category_id": 2},
        {"id": 3, "name": "empty", "number_frames": 0, "category_id": 2}])
    assert count_split_sequences(split_data_folder=tmp_path,
                                 sequence_settings={"maximum_sequence_length": 10}) == (3, 14, 1)
    assert count_split_sequences(split_data_folder=tmp_path,
                                 sequence_settings={"maximum_sequence_length": 12}) == (3, 14, 2)


def test_selection_sequences_are_whole_with_unselected_frames(tmp_path: Path) -> None:
    """
    Verifies the frame selection dataset: every sequence is one sample, the unselected class is found by the name of
    its category, the frames of unmapped categories become unselected frames, the sequences are named after their
    folder, and the classes of a sequence used to spread the sample images leave the unselected frames out.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Category 1 is wanted (class 0), category 3 is the unselected one (class 1), category 2 is unmapped
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "first", "number_frames": 5, "frame_category_ids": [3, 1, 2, 3, 3]},
        {"id": 2, "name": "second", "number_frames": 2, "frame_category_ids": [1, 2]}])
    dataset = SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                              image_size=16, keep_ratio=False, label_dictionary={1: 0, 3: 1},
                              sequence_settings={"maximum_sequence_length": 6}, label_type="selection",
                              split_name="Training",
                              selection_settings={"unselected_category": "still", "frames_per_class": 1},
                              shuffle=False)

    assert dataset.unselected_class_index == 1 and len(dataset) == 2
    assert dataset[0]["labels"].tolist() == [1, 0, 1, 1, 1]
    assert dataset[1]["labels"].tolist() == [0, 1]
    assert dataset[0]["sequence_mask"].sum() == 5
    assert dataset.get_sample_name(sample_identifier=0) == "first"
    assert dataset.get_sample_class_indices() == {0: [0], 1: [0]}
    assert count_split_sequences(split_data_folder=tmp_path,
                                 sequence_settings={"maximum_sequence_length": 6}) == (2, 7, 2)


def test_selection_sequences_that_do_not_fit_are_all_listed(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the frame selection dataset refuses the sequences it cannot train on, listing all of them in one
    error: a sequence holding a wanted class twice and one missing a wanted class, a sequence longer than the maximum
    sequence length being left out with a red warning instead.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warnings.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "valid", "number_frames": 3, "frame_category_ids": [1, 2, 3]},
        {"id": 2, "name": "twice", "number_frames": 3, "frame_category_ids": [1, 1, 2]},
        {"id": 3, "name": "missing", "number_frames": 2, "frame_category_ids": [3, 2]},
        {"id": 4, "name": "long", "number_frames": 5, "frame_category_ids": [1, 2, 3, 3, 3]}])

    with pytest.raises(expected_exception=ValueError) as raised_error:
        SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                        image_size=16, keep_ratio=False, label_dictionary={1: 0, 2: 1, 3: 2},
                        sequence_settings={"maximum_sequence_length": 4}, label_type="selection",
                        split_name="Validation",
                        selection_settings={"unselected_category": "still", "frames_per_class": 1})
    error_message = str(raised_error.value)
    assert error_message.startswith("2 sequences cannot be used for the frame selection")
    assert " - twice : left 2 frames" in error_message
    assert error_message.endswith(" - missing : left 0 frames")
    assert "valid" not in error_message and "long" not in error_message
    assert "Sequence 'long' Of The Validation Split Has 5 Frames" in capsys.readouterr().out


@pytest.mark.parametrize(argnames=("unselected_category", "label_dictionary", "expected_message"), argvalues=[
    ("discarded", {1: 0, 3: 1}, "No category is named 'discarded'"),
    ("still", {1: 0, 2: 1}, "The unselected category 'still' (identifier 3) is not mapped")])
def test_selection_needs_a_mapped_unselected_category(tmp_path: Path, unselected_category: str,
                                                      label_dictionary: Dict[int, int], expected_message: str) -> None:
    """
    Verifies that the frame selection dataset refuses an unselected category missing from the manifest or left out
    of the label mapping, whose frames would have no class.

    Args:
        tmp_path (Path): The pytest temporary directory.
        unselected_category (str): The name of the unselected category of the selection settings.
        label_dictionary (Dict[int, int]): The label mapping of the dataset.
        expected_message (str): The start of the expected error.
    """
    write_sequence_split(split_folder=tmp_path, sequences=[
        {"id": 1, "name": "first", "number_frames": 2, "frame_category_ids": [1, 3]}])
    with pytest.raises(expected_exception=ValueError, match=expected_message.replace("(", r"\(").replace(")", r"\)")):
        SequenceDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                        image_size=16, keep_ratio=False, label_dictionary=label_dictionary,
                        sequence_settings={"maximum_sequence_length": 4}, label_type="selection",
                        split_name="Training",
                        selection_settings={"unselected_category": unselected_category, "frames_per_class": 1})


def test_loaders_needing_batch_statistics_drop_the_last_training_batch(tmp_path: Path) -> None:
    """
    Verifies the sequence loaders of a classifier whose head is batch normalized (drop_last_training_batch): the
    training loader leaves out its last incomplete batch, which a single sequence would fill and break, the validation
    loader keeps every sequence, and a training split smaller than a batch stops the launch with its size instead of
    giving no batch.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    split_sizes = {"train": 3, "validation": 3, "test": 2}
    for split_name, number_sequences in split_sizes.items():
        write_sequence_split(split_folder=tmp_path / split_name, sequences=[
            {"id": sequence_index + 1, "name": f"sequence_{sequence_index}", "number_frames": 2,
             "category_id": sequence_index % 2 + 1} for sequence_index in range(number_sequences)])
    experiment_configuration = {f"{split_name}_data_folder": str(tmp_path / split_name) for split_name in split_sizes}
    data_configuration = {"image_settings": {"size": 32, "keep_ratio": True}, "label_dictionary": {1: 0, 2: 1},
                          "sequence_settings": {"maximum_sequence_length": 8}}
    model_configuration = {"Data": data_configuration}

    # Batches of 2 sequences: the third training sequence is left out only for the head needing batch statistics
    for drop_last_training_batch, expected_training_batches in [(True, [2]), (False, [2, 1])]:
        training_loader, validation_loader, _ = get_sequence_dataloaders(
            experiment_configuration=experiment_configuration, model_configuration=model_configuration, batch_size=2,
            label_type="sequence", drop_last_training_batch=drop_last_training_batch)
        assert [batch["images"].shape[0] for batch in training_loader] == expected_training_batches
        assert [batch["images"].shape[0] for batch in validation_loader] == [2, 1]

    # Batches of 4 sequences, more than the training split holds
    with pytest.raises(expected_exception=ValueError, match="The 'train_data_folder' split holds 3 samples, fewer than "
                                                            "the batch size 4"):
        get_sequence_dataloaders(experiment_configuration=experiment_configuration,
                                 model_configuration=model_configuration, batch_size=4, label_type="sequence",
                                 drop_last_training_batch=True)
