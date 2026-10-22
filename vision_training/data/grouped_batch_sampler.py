"""
GroupedBatchSampler, the batches of the self-supervised task built from groups of related images, so that the
negatives of an image (the other images of its batch, which its views must be told apart from) include many images
sharing its nuisance attributes: the same camera, scene or device type. The groups come from the optional "groups"
dictionary of every sample of the 'labels.json' manifests, kept in its sample record by ClassificationDataset, and the
keys from the 'Data.grouped_batches' section. Built by get_classification_dataloaders of data/loaders.py for the
training and validation splits of the self-supervised task and given to their DataLoader as its batch sampler, the
batches staying random when the section is off or when no sample of the split holds its keys. The input viewer of the
self-supervised trainer reads which batches of a pass are grouped, for its batch mosaic.
"""

import torch

from torch.utils.data import Sampler
from typing import Any, Dict, Iterator, List, Optional, Tuple

from vision_training.utilities.os_utilities import print_color


class GroupedBatchSampler(Sampler[List[int]]):
    """
    Yields the dataset indices of the batches of one pass over a split, rebuilt at every pass:
    1. the images are grouped by their (batch value, block value), every group is shuffled and cut into blocks of
       images_per_block images, its remaining images going to a leftover pool;
    2. the blocks of every batch value are shuffled and joined into full batches of batch_size / images_per_block
       blocks, the blocks that cannot fill a batch going to the leftover pool;
    3. the leftover pool is shuffled and cut into full batches of random images;
    4. the order of all the batches is shuffled, the last incomplete batch of the leftover pool being dropped.
    Every image is used at most once per pass, and every pass gives floor(N / batch_size) batches, like the random
    batches without their last incomplete one, since every grouped batch is full and the leftover pool holds every other
    image. The draws of every pass use a PyTorch generator seeded from the global PyTorch generator, like the shuffling
    of the DataLoader, so that a seed reproduces the batches. Whether every batch of the current pass is grouped is
    kept, in the order of the pass, for the batch mosaic of the input viewer ('view_input_images').
    """

    def __init__(self, group_values: List[Tuple[Optional[str], str]], batch_size: int, images_per_block: int) -> None:
        """
        Groups the dataset indices by their batch and block values, and counts the grouped and leftover batches of
        every pass, the same at every pass since the shuffles keep the sizes of the groups.

        Args:
            group_values (List[Tuple[Optional[str], str]]): The (batch value, block value) of every sample of the
                dataset, in its order, the batch value being None when the batches only gather blocks.
            batch_size (int): The number of images of a batch.
            images_per_block (int): The number of images of a block, dividing the batch size.

        Raises:
            ValueError: If the blocks do not divide the batches, which would leave the grouped batches incomplete.
        """
        super().__init__()
        if images_per_block < 1 or batch_size % images_per_block != 0:
            raise ValueError(f"The blocks of {images_per_block} images must divide the batches of {batch_size} "
                             f"images.")
        self.batch_size = batch_size
        self.images_per_block = images_per_block
        self.blocks_per_batch = batch_size // images_per_block
        self.number_samples = len(group_values)

        # Dataset indices of every (batch value, block value) group, in the order of the dataset
        self.group_indices: Dict[Tuple[Optional[str], str], List[int]] = {}
        for sample_index, group_key in enumerate(group_values):
            self.group_indices.setdefault(group_key, []).append(sample_index)

        # Full blocks of every batch value, of which every full set of blocks_per_batch gives a grouped batch, the
        # leftover pool giving the other batches
        blocks_per_batch_value: Dict[Optional[str], int] = {}
        for (batch_value, _), sample_indices in self.group_indices.items():
            blocks_per_batch_value[batch_value] = (blocks_per_batch_value.get(batch_value, 0) +
                                                   len(sample_indices) // images_per_block)
        self.number_grouped_batches = sum(number_blocks // self.blocks_per_batch
                                          for number_blocks in blocks_per_batch_value.values())
        self.number_leftover_batches = self.number_samples // batch_size - self.number_grouped_batches

        # Whether every batch of the current pass, in the order it is yielded, is a grouped batch (False for a random
        # batch of the leftover pool), set when a pass starts
        self.grouped_batch_flags: List[bool] = []

    def __len__(self) -> int:
        """
        Gives the number of batches of every pass, read by the DataLoader for its length.

        Returns:
            int: The grouped and leftover batches of a pass, floor(N / batch_size).
        """
        return self.number_grouped_batches + self.number_leftover_batches

    def __iter__(self) -> Iterator[List[int]]:
        """
        Draws the batches of one pass over the split, called by the DataLoader at the start of every pass.

        Returns:
            Iterator[List[int]]: The dataset indices of every batch of the pass, in a random order.
        """
        # Generator of the pass, seeded from the global PyTorch generator so that a seed reproduces every pass
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty(size=(), dtype=torch.int64).random_().item()))

        # Every group shuffled and cut into blocks, its remaining images going to the leftover pool
        blocks_per_batch_value: Dict[Optional[str], List[List[int]]] = {}
        leftover_indices = []
        for (batch_value, _), sample_indices in self.group_indices.items():
            shuffled_indices = [sample_indices[position]
                                for position in torch.randperm(n=len(sample_indices), generator=generator).tolist()]
            number_blocks = len(shuffled_indices) // self.images_per_block
            blocks_per_batch_value.setdefault(batch_value, []).extend(
                shuffled_indices[block_index * self.images_per_block:(block_index + 1) * self.images_per_block]
                for block_index in range(number_blocks))
            leftover_indices += shuffled_indices[number_blocks * self.images_per_block:]

        # The blocks of every batch value shuffled and joined into full batches, the blocks that cannot fill a batch
        # going to the leftover pool
        batches = []
        for blocks in blocks_per_batch_value.values():
            shuffled_blocks = [blocks[position]
                               for position in torch.randperm(n=len(blocks), generator=generator).tolist()]
            number_batches = len(shuffled_blocks) // self.blocks_per_batch
            for batch_index in range(number_batches):
                batch_blocks = shuffled_blocks[batch_index * self.blocks_per_batch:
                                               (batch_index + 1) * self.blocks_per_batch]
                batches.append([sample_index for block in batch_blocks for sample_index in block])
            leftover_indices += [sample_index for block in shuffled_blocks[number_batches * self.blocks_per_batch:]
                                 for sample_index in block]

        # The leftover pool shuffled and cut into full batches of random images, its last incomplete batch dropped
        shuffled_leftover_indices = [leftover_indices[position] for position in
                                     torch.randperm(n=len(leftover_indices), generator=generator).tolist()]
        batches += [shuffled_leftover_indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
                    for batch_index in range(len(shuffled_leftover_indices) // self.batch_size)]

        # Grouped and random batches in a random order, the grouped ones being the first of the list
        batch_order = torch.randperm(n=len(batches), generator=generator).tolist()
        self.grouped_batch_flags = [position < self.number_grouped_batches for position in batch_order]
        for position in batch_order:
            yield batches[position]


def build_grouped_batch_sampler(samples: List[Dict[str, Any]], grouped_batches_configuration: Dict[str, Any],
                                batch_size: int, split_name: str) -> Optional[GroupedBatchSampler]:
    """
    Builds the grouped batch sampler of a training or validation split of the self-supervised task from the "groups"
    of its samples, or gives None for random batches: when the section is off, or when no sample of the split holds
    the keys of the section, with a warning naming the split, so that a dataset without groups keeps its random
    batches. Prints the keys, their number of values and the grouped and leftover batches of every pass.

    Args:
        samples (List[Dict[str, Any]]): The sample records of the dataset of the split, in its order, each with its
            "groups" dictionary (empty for a sample without groups).
        grouped_batches_configuration (Dict[str, Any]): The 'Data.grouped_batches' section.
        batch_size (int): The number of images of a batch.
        split_name (str): The configuration key of the split, e.g. "train_data_folder", named in the messages.

    Returns:
        Optional[GroupedBatchSampler]: The sampler of the split, or None for random batches.

    Raises:
        ValueError: If only some samples of the split hold the keys, giving how many samples miss each key, or if the
            value of a key is not a string.
    """
    if not grouped_batches_configuration["activated"]:
        return None

    # Key shared by the images of a block and, when given, the key shared by every image of a batch
    block_key = grouped_batches_configuration["block_key"]
    batch_key = grouped_batches_configuration["batch_key"]
    group_keys = [block_key] if batch_key is None else [block_key, batch_key]

    # Random batches when no sample holds the keys, an error when only some samples hold them
    missing_counts = {group_key: sum(group_key not in sample["groups"] for sample in samples)
                      for group_key in group_keys}
    if all(missing_count == len(samples) for missing_count in missing_counts.values()):
        print_color(color="yellow", output=f"No sample of the '{split_name}' split holds the keys {group_keys} of "
                                           f"'Data.grouped_batches': its batches stay random.")
        return None
    if any(missing_counts.values()):
        missing_descriptions = ", ".join(f"{missing_count} miss '{group_key}'"
                                         for group_key, missing_count in missing_counts.items())
        raise ValueError(f"Only some samples of the '{split_name}' split hold the keys {group_keys} of "
                         f"'Data.grouped_batches': of its {len(samples)} samples, {missing_descriptions}.")

    # Every value is a string attribute, compared between samples
    for sample in samples:
        for group_key in group_keys:
            if not isinstance(sample["groups"][group_key], str):
                raise ValueError(f"The '{group_key}' group of the sample {sample['identifier']} of the '{split_name}' "
                                 f"split must be a string, got {sample['groups'][group_key]!r}.")

    # Batch value (None without batch key) and block value of every sample, in the order of the dataset
    group_values = [(None if batch_key is None else sample["groups"][batch_key], sample["groups"][block_key])
                    for sample in samples]
    sampler = GroupedBatchSampler(group_values=group_values, batch_size=batch_size,
                                  images_per_block=grouped_batches_configuration["images_per_block"])

    # Keys with their number of values, and the batches of every pass
    key_descriptions = [f"{key_role} key '{group_key}' ({len({sample['groups'][group_key] for sample in samples})} "
                        f"values)" for key_role, group_key in zip(["block", "batch"], group_keys)]
    print_color(color="blue", output=f"Grouped batches of the '{split_name}' split : {', '.join(key_descriptions)}, "
                                     f"{sampler.number_grouped_batches} grouped and "
                                     f"{sampler.number_leftover_batches} leftover batches per pass")
    return sampler
