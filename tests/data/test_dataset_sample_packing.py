"""
Tests of the packed records of the classification, segmentation, sequence and sequence pair datasets (SerializedRecords
of vision_training/data/serialized_records.py): every dataset keeps its samples, sequences or pairs in its packed
records, without any Python dictionary or list per sample, the objects a forked loader worker copies page by page by
reading them (copy-on-access), and still names a sample from its identifier once its samples are shuffled.
"""

import gc
import json
import pytest

from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Tuple

from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.segmentation.segmentation_dataset import SegmentationDataset
from vision_training.data.sequence.sequence_dataset import SequenceDataset
from vision_training.data.sequence.sequence_pair_dataset import SequencePairDataset

# Number of samples of every manifest, so that one Python object per sample would stand out of the few objects a
# dataset keeps
NUMBER_SAMPLES = 2000

# Identifier of the sample named after the shuffle, an identifier of the manifest or the position of a sequence or pair
NAMED_IDENTIFIER = 1234

# Categories of every manifest and their classes
CATEGORIES = [{"id": 1, "name": "circle"}, {"id": 2, "name": "square"}]
LABEL_DICTIONARY = {1: 0, 2: 1}


def write_manifest(split_folder: Path, manifest: Dict[str, Any]) -> str:
    """
    Writes the 'labels.json' manifest of a split, no image being read by the construction of the datasets.

    Args:
        split_folder (Path): The folder of the split.
        manifest (Dict[str, Any]): The manifest, with its categories.

    Returns:
        str: The path of the written manifest.
    """
    split_folder.mkdir(parents=True, exist_ok=True)
    labels_path = split_folder / "labels.json"
    labels_path.write_text(data=json.dumps(obj={"categories": CATEGORIES, **manifest}))
    return str(labels_path)


def build_classification_dataset(split_folder: Path) -> Tuple[Dataset, Dict[str, str]]:
    """
    Builds a classification dataset of NUMBER_SAMPLES samples, with groups and their categories alternating.

    Args:
        split_folder (Path): The folder of the split.

    Returns:
        Tuple[Dataset, Dict[str, str]]: The dataset, and the description of the sample NAMED_IDENTIFIER.
    """
    labels_path = write_manifest(split_folder=split_folder, manifest={"samples": [
        {"id": sample_identifier, "file_name": f"image_{sample_identifier}.png",
         "category_id": sample_identifier % 2 + 1, "groups": {"camera": f"c{sample_identifier % 3}"}}
        for sample_identifier in range(NUMBER_SAMPLES)]})
    dataset = ClassificationDataset(data_directory=str(split_folder / "data"), labels_file=labels_path, image_size=32,
                                    keep_ratio=True, label_dictionary=LABEL_DICTIONARY, output_mode="multi_class")
    return dataset, {"sample": f"image_{NAMED_IDENTIFIER}.png"}


def build_segmentation_dataset(split_folder: Path) -> Tuple[Dataset, Dict[str, str]]:
    """
    Builds a semantic segmentation dataset of NUMBER_SAMPLES samples, each with its mask file.

    Args:
        split_folder (Path): The folder of the split.

    Returns:
        Tuple[Dataset, Dict[str, str]]: The dataset, and the description of the sample NAMED_IDENTIFIER.
    """
    labels_path = write_manifest(split_folder=split_folder, manifest={"samples": [
        {"id": sample_identifier, "file_name": f"image_{sample_identifier}.png",
         "mask_file_name": f"mask_{sample_identifier}.png"} for sample_identifier in range(NUMBER_SAMPLES)]})
    dataset = SegmentationDataset(data_directory=str(split_folder / "data"),
                                  masks_directory=str(split_folder / "masks"), labels_file=labels_path, image_size=32,
                                  keep_ratio=True, label_mode="multi_class", label_dictionary=LABEL_DICTIONARY,
                                  tiling_configuration={"activated": False, "scale": 1.0, "overlap": 0},
                                  tile_sampling="random")
    return dataset, {"sample": f"image_{NAMED_IDENTIFIER}.png"}


def build_sequence_dataset(split_folder: Path) -> Tuple[Dataset, Dict[str, str]]:
    """
    Builds a sequence classification dataset of NUMBER_SAMPLES sequences of three frames, every sequence kept, so that
    the identifier of a sequence is its position in the manifest.

    Args:
        split_folder (Path): The folder of the split.

    Returns:
        Tuple[Dataset, Dict[str, str]]: The dataset, and the description of the sequence NAMED_IDENTIFIER.
    """
    labels_path = write_manifest(split_folder=split_folder, manifest={"sequences": [
        {"id": sequence_index + 1, "name": f"sequence_{sequence_index}", "category_id": sequence_index % 2 + 1,
         "frames": [f"sequence_{sequence_index}/frame_{frame_index}.png" for frame_index in range(3)]}
        for sequence_index in range(NUMBER_SAMPLES)]})
    dataset = SequenceDataset(data_directory=str(split_folder / "data"), labels_file=labels_path, image_size=32,
                              keep_ratio=True, label_dictionary=LABEL_DICTIONARY,
                              sequence_settings={"maximum_sequence_length": 8}, label_type="sequence",
                              split_name="Training")
    return dataset, {"sequence": f"sequence_{NAMED_IDENTIFIER}",
                     "first_frame": f"sequence_{NAMED_IDENTIFIER}/frame_0.png",
                     "last_frame": f"sequence_{NAMED_IDENTIFIER}/frame_2.png"}


def build_sequence_pair_dataset(split_folder: Path) -> Tuple[Dataset, Dict[str, str]]:
    """
    Builds a sequence comparison dataset of NUMBER_SAMPLES pairs of two-frame members, every pair kept, so that the
    identifier of a pair is its position in the manifest.

    Args:
        split_folder (Path): The folder of the split.

    Returns:
        Tuple[Dataset, Dict[str, str]]: The dataset, and the description of the pair NAMED_IDENTIFIER.
    """
    labels_path = write_manifest(split_folder=split_folder, manifest={"pairs": [
        {"id": pair_index + 1, "name": f"pair_{pair_index}", "category_id": pair_index % 2 + 1,
         "before": [f"pair_{pair_index}/before_{frame_index}.png" for frame_index in range(2)],
         "after": [f"pair_{pair_index}/after_{frame_index}.png" for frame_index in range(2)]}
        for pair_index in range(NUMBER_SAMPLES)]})
    dataset = SequencePairDataset(data_directory=str(split_folder / "data"), labels_file=labels_path, image_size=32,
                                  keep_ratio=True, label_dictionary=LABEL_DICTIONARY,
                                  sequence_settings={"maximum_sequence_length": 8}, split_name="Training")
    return dataset, {"pair": f"pair_{NAMED_IDENTIFIER}"}


@pytest.mark.parametrize(argnames="build_dataset", argvalues=[build_classification_dataset, build_segmentation_dataset,
                                                              build_sequence_dataset, build_sequence_pair_dataset],
                         ids=["classification", "segmentation", "sequence", "sequence_pair"])
def test_dataset_keeps_no_python_object_per_sample(tmp_path: Path,
                                                   build_dataset: Callable[[Path], Tuple[Dataset, Dict[str, str]]]
                                                   ) -> None:
    """
    Verifies that every dataset keeps its samples in its packed records, without any Python dictionary or list per
    sample: the objects followed by the garbage collector grow by a few objects only for 2000 samples, and the shuffled
    dataset still names a sample from its identifier.

    Args:
        tmp_path (Path): The pytest temporary directory.
        build_dataset (Callable[[Path], Tuple[Dataset, Dict[str, str]]]): Writes the manifest of a split and builds its
            dataset.
    """
    # Objects followed by the garbage collector before and after the dataset is built, the dataset kept alive
    gc.collect()
    object_count_before = len(gc.get_objects())
    dataset, named_description = build_dataset(split_folder=tmp_path)
    gc.collect()
    assert len(dataset) == NUMBER_SAMPLES
    assert len(gc.get_objects()) - object_count_before < 100

    # The sample of an identifier found among the shuffled samples
    assert dataset.describe_sample(sample_identifier=NAMED_IDENTIFIER) == named_description
