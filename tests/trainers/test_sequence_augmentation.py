import random
import torch

from vision_training.data.augmentations import AUGMENTATION_TYPES, DataAugmenter
from vision_training.trainers.sequence_comparison_trainer import SequenceComparisonTrainer
from vision_training.trainers.sequence_labelling_trainer import SequenceLabellingTrainer


def build_shuffling_augmenter() -> DataAugmenter:
    """
    Builds an augmenter that shuffles every window and applies no other augmentation.

    Returns:
        DataAugmenter: The augmenter.
    """
    return DataAugmenter(augmentation_configuration={
        "activated": True, "probability": 1.0, "independent_frames": False,
        "shuffle_frames": {"activated": True, "probability": 1.0},
        "types": {name: {"activated": False, "probability": 0.0} for name in AUGMENTATION_TYPES}})


def test_frame_labels_follow_their_shuffled_frames() -> None:
    """
    Verifies that the augmentation of a sequence labelling batch moves the class of every frame along with its frame
    when the frames are shuffled, and keeps the sequence mask. The trainer is created without its datasets, only its
    augmenter and sequence settings being used by augment_batch.
    """
    torch.manual_seed(seed=0)
    random.seed(0)
    trainer = SequenceLabellingTrainer.__new__(SequenceLabellingTrainer)
    trainer.data_augmenter = build_shuffling_augmenter()
    trainer.sequence_settings = {"sequence_length": 6}

    # Frames filled with their position, labelled with their position, the last two frames of the second window
    # being padding frames with the ignored label
    frames = torch.arange(6, dtype=torch.float32)[None, :, None, None, None].expand(2, 6, 3, 8, 8).clone()
    labels = torch.tensor(data=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, -100, -100]])
    sequence_mask = labels != -100
    augmented_batch = trainer.augment_batch(batch={"images": frames, "labels": labels, "sequence_mask": sequence_mask})

    frame_values = augmented_batch["images"][:, :, 0, 0, 0].to(dtype=torch.int64)
    assert torch.equal(input=frame_values[sequence_mask], other=augmented_batch["labels"][sequence_mask])
    assert augmented_batch["labels"][1, 4:].tolist() == [-100, -100]
    assert torch.equal(input=augmented_batch["sequence_mask"], other=sequence_mask)
    assert not torch.equal(input=augmented_batch["labels"], other=labels)


def test_pair_members_are_shuffled_within_their_windows() -> None:
    """
    Verifies that the augmentation of a comparison batch shuffles every member window on its own: the before frames
    stay in the before window and the after frames in the after window, and the class of the pair is unchanged.
    """
    torch.manual_seed(seed=1)
    random.seed(1)
    trainer = SequenceComparisonTrainer.__new__(SequenceComparisonTrainer)
    trainer.data_augmenter = build_shuffling_augmenter()
    trainer.sequence_settings = {"sequence_length": 4}

    frames = torch.arange(8, dtype=torch.float32)[None, :, None, None, None].expand(3, 8, 3, 8, 8).clone()
    labels = torch.tensor(data=[0, 1, 0])
    augmented_batch = trainer.augment_batch(batch={"images": frames, "labels": labels,
                                                   "sequence_mask": torch.ones(size=(3, 8), dtype=torch.bool)})

    frame_values = augmented_batch["images"][:, :, 0, 0, 0]
    assert bool((frame_values[:, :4] < 4).all()) and bool((frame_values[:, 4:] >= 4).all())
    assert torch.equal(input=augmented_batch["labels"], other=labels)
    assert not torch.equal(input=augmented_batch["images"], other=frames)
