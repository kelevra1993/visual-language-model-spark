"""
Tests of the GroupedBatchSampler of vision_training/data/grouped_batch_sampler.py, the batches of the self-supervised
task gathering blocks of images sharing an attribute of their "groups", the hard negatives of each other: every batch
is full, the images of a block share their block value and a grouped batch its batch value, no image comes twice in a
pass, the images left over form random batches, the length is the number of batches yielded, a seed reproduces the
batches, which are rebuilt at every pass, the split stays random without groups and stops with groups in some samples
only, ClassificationDataset keeps the groups of its samples, and the loaders of the self-supervised task give grouped
batches of the batch size.
"""

import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.data.classification.test_classification_dataset import write_split
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.grouped_batch_sampler import GroupedBatchSampler, build_grouped_batch_sampler
from vision_training.data.loaders import get_classification_dataloaders

# Images of every camera of every site of the tests. With blocks of 4 images and batches of 8, the site s0 gives 4
# blocks (2 grouped batches) and 1 leftover image, the site s1 2 blocks (1 grouped batch) and 3 leftover images, and the
# single block of the site s2, which cannot fill a batch, goes to the leftover pool with its 3 other images: the 11
# leftover images give 1 random batch, 3 of them being dropped at every pass
SITE_CAMERA_IMAGES = {("s0", "c0"): 9, ("s0", "c1"): 8, ("s1", "c2"): 5, ("s1", "c3"): 6, ("s2", "c4"): 7}
BATCH_SIZE = 8
IMAGES_PER_BLOCK = 4

# Grouped batches section of the tests, blocks of one camera within batches of one site
GROUPED_BATCHES_CONFIGURATION = {"activated": True, "block_key": "camera", "images_per_block": IMAGES_PER_BLOCK,
                                 "batch_key": "site"}


def build_samples() -> List[Dict[str, Any]]:
    """
    Builds the sample records of the tests, the site and the camera of every image in its "groups", in a shuffled
    order so that the images of a group are not neighbours in the dataset.

    Returns:
        List[Dict[str, Any]]: The sample records, with their identifier and their groups.
    """
    group_values = [group_value for group_value, number_images in SITE_CAMERA_IMAGES.items()
                    for _ in range(number_images)]
    order = torch.randperm(n=len(group_values), generator=torch.Generator().manual_seed(0)).tolist()
    return [{"identifier": sample_index,
             "groups": {"site": group_values[position][0], "camera": group_values[position][1]}}
            for sample_index, position in enumerate(order)]


def build_sampler(samples: List[Dict[str, Any]], batch_key: Optional[str]) -> GroupedBatchSampler:
    """
    Builds the sampler of the test samples, with the camera as block key.

    Args:
        samples (List[Dict[str, Any]]): The sample records.
        batch_key (Optional[str]): The key shared by every image of a batch, None for blocks of any site in a batch.

    Returns:
        GroupedBatchSampler: The sampler, with batches of BATCH_SIZE and blocks of IMAGES_PER_BLOCK images.
    """
    group_values = [(None if batch_key is None else sample["groups"][batch_key], sample["groups"]["camera"])
                    for sample in samples]
    return GroupedBatchSampler(group_values=group_values, batch_size=BATCH_SIZE, images_per_block=IMAGES_PER_BLOCK)


def test_grouped_batches_are_full_and_share_their_values() -> None:
    """
    Verifies the batches of several passes with a batch key: every batch is full, no image comes twice in a pass, the
    length is the number of batches yielded, the 3 grouped batches (2 of the site s0, 1 of the site s1) are made of
    blocks of one camera, and the leftover images form a random batch of several sites, which holds every image of the
    site s2 used in the pass; blocks that do not divide the batches are refused.
    """
    samples = build_samples()
    sampler = build_sampler(samples=samples, batch_key="site")
    assert (sampler.number_grouped_batches, sampler.number_leftover_batches) == (3, 1)

    torch.manual_seed(seed=0)
    for _ in range(5):
        batches = list(sampler)
        assert len(batches) == len(sampler) == len(samples) // BATCH_SIZE
        assert all(len(batch) == BATCH_SIZE for batch in batches)
        used_indices = [sample_index for batch in batches for sample_index in batch]
        assert len(used_indices) == len(set(used_indices))

        # Grouped batches, of one site, made of blocks of one camera
        batch_sites = [{samples[sample_index]["groups"]["site"] for sample_index in batch} for batch in batches]
        grouped_batches = [batch for batch, sites in zip(batches, batch_sites) if len(sites) == 1]
        assert sorted(next(iter(sites)) for sites in batch_sites if len(sites) == 1) == ["s0", "s0", "s1"]
        for batch in grouped_batches:
            blocks = [batch[start:start + IMAGES_PER_BLOCK] for start in range(0, BATCH_SIZE, IMAGES_PER_BLOCK)]
            assert all(len({samples[sample_index]["groups"]["camera"] for sample_index in block}) == 1
                       for block in blocks)

        # The random batch of the leftover images, holding the used images of the site s2
        leftover_batches = [batch for batch, sites in zip(batches, batch_sites) if len(sites) > 1]
        assert len(leftover_batches) == 1
        assert {sample_index for sample_index in used_indices
                if samples[sample_index]["groups"]["site"] == "s2"} <= set(leftover_batches[0])

    # Blocks that do not divide the batches, which would leave the grouped batches incomplete
    with pytest.raises(expected_exception=ValueError, match="The blocks of 3 images must divide the batches of 8"):
        GroupedBatchSampler(group_values=[(None, "c0")] * 16, batch_size=BATCH_SIZE, images_per_block=3)


def test_blocks_of_every_value_share_the_batches_without_batch_key() -> None:
    """
    Verifies the batches without batch key, every block sharing a single batch value: the 7 blocks of all the cameras
    give 3 grouped batches whose blocks are of one camera, the blocks of different cameras being joined in a batch, and
    the remaining block goes to the leftover pool, which gives 1 random batch.
    """
    samples = build_samples()
    sampler = build_sampler(samples=samples, batch_key=None)
    assert (sampler.number_grouped_batches, sampler.number_leftover_batches) == (3, 1)

    # Batches whose two blocks are each of one camera, some of them joining two cameras over the passes
    torch.manual_seed(seed=1)
    joined_cameras = 0
    for _ in range(10):
        batches = list(sampler)
        assert len(batches) == 4 and all(len(batch) == BATCH_SIZE for batch in batches)
        block_cameras = [[{samples[sample_index]["groups"]["camera"] for sample_index in batch[start:start + 4]}
                          for start in range(0, BATCH_SIZE, IMAGES_PER_BLOCK)] for batch in batches]
        grouped_blocks = [cameras for cameras in block_cameras if all(len(block) == 1 for block in cameras)]
        assert len(grouped_blocks) >= 3
        joined_cameras += sum(cameras[0] != cameras[1] for cameras in grouped_blocks)
    assert joined_cameras > 0


def test_same_seed_gives_the_same_batches_rebuilt_at_every_pass() -> None:
    """
    Verifies that the batches are drawn with the PyTorch generator: the same seed gives the same passes, while two
    passes after one seed differ, the batches being rebuilt at every pass.
    """
    sampler = build_sampler(samples=build_samples(), batch_key="site")
    torch.manual_seed(seed=3)
    first_passes = [list(sampler) for _ in range(2)]
    torch.manual_seed(seed=3)
    second_passes = [list(sampler) for _ in range(2)]
    assert first_passes == second_passes
    assert first_passes[0] != first_passes[1]


def test_builder_keeps_random_batches_without_groups_and_stops_with_some(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the sampler built for a split: none with the section off; a sampler, reported with its keys, their number
    of values and the batches of a pass, when every sample holds the keys; none, with a warning naming the split, when
    no sample holds them; and an error giving how many samples miss each key when only some samples hold them, or when
    a value is not a string.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    samples = build_samples()

    # Section off, then on with the keys in every sample
    assert build_grouped_batch_sampler(samples=samples,
                                       grouped_batches_configuration={**GROUPED_BATCHES_CONFIGURATION,
                                                                      "activated": False},
                                       batch_size=BATCH_SIZE, split_name="train_data_folder") is None
    sampler = build_grouped_batch_sampler(samples=samples, grouped_batches_configuration=GROUPED_BATCHES_CONFIGURATION,
                                          batch_size=BATCH_SIZE, split_name="train_data_folder")
    assert isinstance(sampler, GroupedBatchSampler) and len(sampler) == 4
    assert ("Grouped batches of the 'train_data_folder' split : block key 'camera' (5 values), batch key 'site' (3 "
            "values), 3 grouped and 1 leftover batches per pass") in capsys.readouterr().out

    # No sample with the keys: random batches, with a warning naming the split
    samples_without_groups = [{"identifier": sample["identifier"], "groups": {}} for sample in samples]
    assert build_grouped_batch_sampler(samples=samples_without_groups,
                                       grouped_batches_configuration=GROUPED_BATCHES_CONFIGURATION,
                                       batch_size=BATCH_SIZE, split_name="validation_data_folder") is None
    assert ("No sample of the 'validation_data_folder' split holds the keys ['camera', 'site'] of "
            "'Data.grouped_batches': its batches stay random.") in capsys.readouterr().out

    # The keys in some samples only, then a value that is not a string
    samples[0]["groups"] = {}
    del samples[1]["groups"]["site"]
    with pytest.raises(expected_exception=ValueError, match=r"Only some samples of the 'train_data_folder' split hold "
                                                            r"the keys \['camera', 'site'\] of 'Data.grouped_batches': "
                                                            r"of its 35 samples, 1 miss 'camera', 2 miss 'site'\."):
        build_grouped_batch_sampler(samples=samples, grouped_batches_configuration=GROUPED_BATCHES_CONFIGURATION,
                                    batch_size=BATCH_SIZE, split_name="train_data_folder")
    samples = build_samples()
    samples[2]["groups"]["camera"] = 42
    with pytest.raises(expected_exception=ValueError, match="The 'camera' group of the sample 2 of the "
                                                            "'train_data_folder' split must be a string, got 42."):
        build_grouped_batch_sampler(samples=samples, grouped_batches_configuration=GROUPED_BATCHES_CONFIGURATION,
                                    batch_size=BATCH_SIZE, split_name="train_data_folder")


def test_dataset_keeps_the_groups_of_its_samples(tmp_path: Path) -> None:
    """
    Verifies that ClassificationDataset keeps the "groups" of every sample of labels.json in its record, an empty
    dictionary for a sample without them, and refuses groups that are not a dictionary.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    categories = [{"id": 1, "name": "circle"}]
    write_split(split_folder=tmp_path / "grouped", categories=categories,
                samples=[{"id": 1, "file_name": "1.png", "groups": {"camera": "c1", "site": "s1"}},
                         {"id": 2, "file_name": "2.png"}])
    dataset = ClassificationDataset(data_directory=str(tmp_path / "grouped" / "data"),
                                    labels_file=str(tmp_path / "grouped" / "labels.json"), image_size=32,
                                    keep_ratio=True, label_dictionary={1: 0}, output_mode="multi_class", shuffle=False,
                                    keep_unlabelled_samples=True)
    assert [sample["groups"] for sample in dataset.samples] == [{"camera": "c1", "site": "s1"}, {}]

    write_split(split_folder=tmp_path / "wrong", categories=categories,
                samples=[{"id": 1, "file_name": "1.png", "groups": "c1"}])
    with pytest.raises(expected_exception=ValueError, match="The 'groups' of the sample 1 of .* must be a dictionary"):
        ClassificationDataset(data_directory=str(tmp_path / "wrong" / "data"),
                              labels_file=str(tmp_path / "wrong" / "labels.json"), image_size=32, keep_ratio=True,
                              label_dictionary={1: 0}, output_mode="multi_class", keep_unlabelled_samples=True)


def test_self_supervised_loaders_give_grouped_batches(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the loaders of the self-supervised task with the grouped batches on: the training split, whose 12 images
    hold two cameras, gives 3 full batches of 4 images made of two blocks of 2 images of one camera, every pass; the
    validation split without groups keeps its random full batches, with a warning naming it; and the test split keeps
    every image.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed messages.
    """
    # A training split of 12 images alternating between two cameras, and validation and test splits without groups
    categories = [{"id": 1, "name": "circle"}]
    cameras = {sample_identifier: f"c{sample_identifier % 2}" for sample_identifier in range(1, 13)}
    write_split(split_folder=tmp_path / "train", categories=categories,
                samples=[{"id": sample_identifier, "file_name": f"{sample_identifier}.png", "category_id": 1,
                          "groups": {"camera": camera}} for sample_identifier, camera in cameras.items()])
    for split_name, number_samples in [("validation", 6), ("test", 3)]:
        write_split(split_folder=tmp_path / split_name, categories=categories,
                    samples=[{"id": sample_identifier, "file_name": f"{sample_identifier}.png", "category_id": 1}
                             for sample_identifier in range(1, number_samples + 1)])
    # Build the self-supervised loaders with blocks of 2 images of one camera, and check the printed summary
    experiment_configuration = {f"{split_name}_data_folder": str(tmp_path / split_name)
                                for split_name in ["train", "validation", "test"]}
    model_configuration = {"Data": {"image_settings": {"size": 32, "keep_ratio": True}, "label_dictionary": {1: 0},
                                    "loading": {"number_workers": 0, "prefetch_batches": 2},
                                    "grouped_batches": {"activated": True, "block_key": "camera",
                                                        "images_per_block": 2, "batch_key": None}}}
    training_loader, validation_loader, test_loader = get_classification_dataloaders(
        experiment_configuration=experiment_configuration, model_configuration=model_configuration, batch_size=4,
        self_supervised=True)
    printed_output = capsys.readouterr().out
    assert ("Grouped batches of the 'train_data_folder' split : block key 'camera' (2 values), 3 grouped and 0 "
            "leftover batches per pass") in printed_output
    assert "No sample of the 'validation_data_folder' split holds the keys ['camera']" in printed_output

    # Full training batches of two blocks of one camera, every image once per pass
    assert isinstance(training_loader.batch_sampler, GroupedBatchSampler) and len(training_loader) == 3
    for _ in range(2):
        identifiers = [batch["sample_identifiers"].tolist() for batch in training_loader]
        assert [len(batch_identifiers) for batch_identifiers in identifiers] == [4, 4, 4]
        assert sorted(identifier for batch_identifiers in identifiers for identifier in batch_identifiers) == list(
            range(1, 13))
        assert all(cameras[batch_identifiers[start]] == cameras[batch_identifiers[start + 1]]
                   for batch_identifiers in identifiers for start in [0, 2])

    # Random full validation batches, and every test image
    assert not isinstance(validation_loader.batch_sampler, GroupedBatchSampler) and validation_loader.drop_last
    assert [batch["images"].shape[0] for batch in validation_loader] == [4]
    assert [batch["images"].shape[0] for batch in test_loader] == [3]
