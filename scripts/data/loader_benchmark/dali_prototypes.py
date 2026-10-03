"""
DALI prototypes of the loader benchmark of implementation_dali_loaders.md (phase 1): the pipelines and batch loaders
of the classification task and of the sequence tasks (windows of frames, and pairs of windows), decoding the images on
the GPU and yielding the batch dictionaries of the Native loaders of vision_training/data with the preprocessing of
preprocess_image. Used by run_loader_benchmark.py, and moved to vision_training/data if the benchmark keeps them.
"""

import math
import os
import numpy as np
import torch
import nvidia.dali.fn as dali_function
import nvidia.dali.types as dali_types

from nvidia.dali import pipeline_def as pipeline_definition
from nvidia.dali.data_node import DataNode
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from typing import Dict, Iterator, List, Optional, Tuple, Union

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.sequence.sequence_dataset import SequenceDataset
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS, SequencePairDataset

# Position of a resized image in its black square, just below the centre: DALI rounds the centre position up, while
# preprocess_image leaves the extra pixel of an odd padding at the bottom and on the right ((size - new size) // 2)
PADDING_CROP_POSITION = 0.4999


def resize_and_pad_images(images: DataNode, image_size: int, keep_ratio: bool) -> DataNode:
    """
    Resizes the decoded images of a DALI pipeline into the square network input exactly like preprocess_image, used
    by the Native loaders and by the deployment clients: linear interpolation without antialiasing, the image mapped
    onto its rounded size like cv2.resize ('subpixel_scale=False'), centred on a black square when the ratio is kept,
    then laid out as uint8 [3, size, size] BGR tensors.

    Args:
        images (DataNode): The decoded [H, W, 3] BGR images.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether to keep the aspect ratio, padding the image with black, or to squash it.

    Returns:
        DataNode: The uint8 [3, image_size, image_size] images.
    """
    # Longest side resized to image_size, then the image centred on a black square
    if keep_ratio:
        images = dali_function.resize(images, resize_x=image_size, resize_y=image_size, mode="not_larger",
                                      interp_type=dali_types.INTERP_LINEAR, antialias=False, subpixel_scale=False)
        return dali_function.crop_mirror_normalize(images, dtype=dali_types.UINT8, output_layout="CHW",
                                                   crop=(image_size, image_size), crop_pos_x=PADDING_CROP_POSITION,
                                                   crop_pos_y=PADDING_CROP_POSITION, out_of_bounds_policy="pad",
                                                   fill_values=0)

    # Image squashed to the square
    images = dali_function.resize(images, resize_x=image_size, resize_y=image_size,
                                  interp_type=dali_types.INTERP_LINEAR, antialias=False, subpixel_scale=False)
    return dali_function.crop_mirror_normalize(images, dtype=dali_types.UINT8, output_layout="CHW")


@pipeline_definition
def classification_pipeline(file_paths: List[str], sample_positions: List[int], image_size: int, keep_ratio: bool,
                            shuffle: bool) -> Tuple[DataNode, DataNode]:
    """
    DALI pipeline of the classification prototype: the image files of the samples read by the C++ file reader,
    decoded on the GPU (the Huffman decoding running on the CPU threads of the pipeline) and resized like the Native
    loader. The label read with every file is the position of its sample in the dataset, from which the loader looks
    up the real labels and identifiers.

    Args:
        file_paths (List[str]): The image file of every sample, in dataset order.
        sample_positions (List[int]): The position of every sample in the dataset, read as its label.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether to keep the aspect ratio of the images, padding them.
        shuffle (bool): Whether to shuffle the whole dataset after every epoch, a full permutation (the buffer of
            'random_shuffle' would mix only neighbouring samples of manifests sorted by class).

    Returns:
        Tuple[DataNode, DataNode]: The uint8 [3, size, size] images and the sample positions.
    """
    encoded_images, positions = dali_function.readers.file(files=file_paths, labels=sample_positions,
                                                           shuffle_after_epoch=shuffle, pad_last_batch=True,
                                                           name="Reader")
    images = dali_function.decoders.image(encoded_images, device="mixed", output_type=dali_types.BGR)
    return resize_and_pad_images(images=images, image_size=image_size, keep_ratio=keep_ratio), positions


class ClassificationDALILoader:
    """
    Batch loader of the classification prototype, yielding the batches of the Native classification loader: uint8
    [B, 3, S, S] BGR images (already on the GPU), the int64 class indices or float32 multi-hot labels, and the
    sample identifiers. Every sample is given exactly once per epoch, the last batch being smaller, and the Native
    dataset stays available as '.dataset' for the metadata read by the trainers.
    """

    def __init__(self, dataset: ClassificationDataset, batch_size: int, number_threads: int,
                 prefetch_queue_depth: int, shuffle: bool, device_id: int) -> None:
        """
        Builds the label and identifier tables of the samples and the DALI pipeline reading their files.

        Args:
            dataset (ClassificationDataset): The Native dataset of the split, giving the samples and their labels.
            batch_size (int): The number of samples of a batch.
            number_threads (int): The number of CPU threads of the pipeline (file reading, Huffman decoding).
            prefetch_queue_depth (int): The number of batches prepared in advance by the pipeline.
            shuffle (bool): Whether to shuffle the samples at every epoch.
            device_id (int): The index of the GPU.
        """
        self.dataset = dataset
        self.batch_size = batch_size

        # Label and identifier of every sample, indexed by its position in the dataset
        if dataset.output_mode == "multi_label":
            self.label_table = torch.zeros(size=(len(dataset), dataset.number_classes), dtype=torch.float32)
            for position, sample in enumerate(dataset.samples):
                self.label_table[position, sample["class_indices"]] = 1.0
        else:
            self.label_table = torch.tensor(data=[sample["class_indices"][0] for sample in dataset.samples],
                                            dtype=torch.int64)
        self.identifier_table = torch.tensor(data=[sample["identifier"] for sample in dataset.samples],
                                             dtype=torch.int64)

        # Pipeline reading the image files in dataset order, the epoch ending on a smaller last batch
        file_paths = [os.path.join(dataset.data_directory, sample["file_name"]) for sample in dataset.samples]
        pipeline = classification_pipeline(file_paths=file_paths, sample_positions=list(range(len(dataset))),
                                           image_size=dataset.image_size, keep_ratio=dataset.keep_ratio,
                                           shuffle=shuffle, batch_size=batch_size, num_threads=number_threads,
                                           device_id=device_id, prefetch_queue_depth=prefetch_queue_depth)
        pipeline.build()
        self.dali_iterator = DALIGenericIterator(pipelines=[pipeline], output_map=["images", "positions"],
                                                 reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL,
                                                 auto_reset=True)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yields the batches of an epoch, the labels and identifiers looked up from the sample positions.

        Yields:
            Dict[str, torch.Tensor]: The 'images' (on the GPU), 'labels' and 'sample_identifiers' of a batch.
        """
        for outputs in self.dali_iterator:
            positions = outputs[0]["positions"].reshape(-1).to(dtype=torch.int64)
            yield {"images": outputs[0]["images"], "labels": self.label_table[positions],
                   "sample_identifiers": self.identifier_table[positions]}

    def __len__(self) -> int:
        """
        Gives the number of batches of an epoch.

        Returns:
            int: The number of batches.
        """
        return math.ceil(len(self.dataset) / self.batch_size)


def build_sample_frame_slots(dataset: Union[SequenceDataset, SequencePairDataset]) -> List[List[Optional[str]]]:
    """
    Lists the frame slots of every sample of a sequence dataset in the layout of its Native samples, so that the DALI
    sequence loader rebuilds exactly the same frame tensors: the real frames of a window first, then padding slots up
    to the sequence length, and for a pair the before window followed by the after window.

    Args:
        dataset (Union[SequenceDataset, SequencePairDataset]): The Native dataset of the split.

    Returns:
        List[List[Optional[str]]]: The path of the frame of every slot of every sample, None for the padding slots.
    """
    # Members of a pair padded one by one, the before window first
    if isinstance(dataset, SequencePairDataset):
        sample_slots = []
        for pair in dataset.pairs:
            slots = []
            for member_name in PAIR_MEMBERS:
                member_paths = [os.path.join(dataset.data_directory, frame) for frame in pair[member_name]]
                slots.extend(member_paths + [None] * (dataset.sequence_length - len(member_paths)))
            sample_slots.append(slots)
        return sample_slots

    # Windows of frames padded at their end
    sample_slots = []
    for window in dataset.windows:
        frame_paths = [os.path.join(dataset.data_directory, frame) for frame in window["frames"]]
        sample_slots.append(frame_paths + [None] * (dataset.sequence_length - len(frame_paths)))
    return sample_slots


def build_sequence_label_table(dataset: Union[SequenceDataset, SequencePairDataset]) -> torch.Tensor:
    """
    Gathers the label of every sample of a sequence dataset, as its Native samples give it, so that the DALI sequence
    loader looks them up from the sample positions: the class of a pair, the class of a window, or the class of every
    frame of a window with IGNORED_LABEL_INDEX on its padding frames.

    Args:
        dataset (Union[SequenceDataset, SequencePairDataset]): The Native dataset of the split.

    Returns:
        torch.Tensor: The int64 [N] labels, or [N, sequence_length] frame labels.
    """
    # One class per pair, or per window
    if isinstance(dataset, SequencePairDataset):
        return torch.tensor(data=[pair["label"] for pair in dataset.pairs], dtype=torch.int64)
    if dataset.label_type == "sequence":
        return torch.tensor(data=[window["frame_labels"][0] for window in dataset.windows], dtype=torch.int64)

    # One class per frame, the padding frames being ignored
    label_table = torch.full(size=(len(dataset), dataset.sequence_length), fill_value=IGNORED_LABEL_INDEX,
                             dtype=torch.int64)
    for position, window in enumerate(dataset.windows):
        label_table[position, :len(window["frame_labels"])] = torch.tensor(data=window["frame_labels"],
                                                                         dtype=torch.int64)
    return label_table


class WindowFrameSource:
    """
    Source of the DALI sequence pipeline, called in the reader processes of the parallel external source: for every
    batch of an epoch, it reads the encoded bytes of the real frames of its samples, with the slot of every frame in
    the batch and the position of its sample in the dataset. The padding slots are not read nor decoded. The sample
    order is a permutation drawn from the seed and the epoch index, so it changes at every epoch while every reader
    process computes the same one. It is a top-level class so that the spawned reader processes can unpickle it.
    """

    def __init__(self, sample_slots: List[List[Optional[str]]], frames_per_sample: int, batch_size: int,
                 shuffle: bool, seed: int) -> None:
        """
        Keeps the frame slots of the samples and the batching settings.

        Args:
            sample_slots (List[List[Optional[str]]]): The frame path of every slot of every sample, None for padding.
            frames_per_sample (int): The number of slots of a sample (the sequence length, twice it for a pair).
            batch_size (int): The number of samples of a batch.
            shuffle (bool): Whether to shuffle the samples at every epoch.
            seed (int): The seed of the permutations.
        """
        self.sample_slots = sample_slots
        self.frames_per_sample = frames_per_sample
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch_index = -1
        self.epoch_order = np.arange(len(sample_slots))

    def get_epoch_order(self, epoch_index: int) -> np.ndarray:
        """
        Gives the order of the samples in an epoch, computed once per epoch in every reader process.

        Args:
            epoch_index (int): The index of the epoch.

        Returns:
            np.ndarray: The sample positions in their order of the epoch.
        """
        if epoch_index != self.epoch_index:
            self.epoch_index = epoch_index
            number_samples = len(self.sample_slots)
            self.epoch_order = (np.random.default_rng(seed=[self.seed, epoch_index]).permutation(number_samples)
                                if self.shuffle else np.arange(number_samples))
        return self.epoch_order

    def __call__(self, batch_information: dali_types.BatchInfo) -> Tuple[List[np.ndarray], List[np.ndarray],
                                                                        List[np.ndarray]]:
        """
        Reads the real frames of one batch of the epoch.

        Args:
            batch_information (dali_types.BatchInfo): The index of the batch in its epoch and the index of the epoch.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: The encoded bytes of every real frame, its slot
                in the batch (sample position in the batch x frames_per_sample + frame position) and the position of
                its sample in the dataset.

        Raises:
            StopIteration: When the epoch has no sample left.
        """
        epoch_order = self.get_epoch_order(epoch_index=batch_information.epoch_idx)
        first_position = batch_information.iteration * self.batch_size
        if first_position >= len(epoch_order):
            raise StopIteration

        # Encoded bytes of the real frames of the samples of the batch, with their slots and sample positions
        encoded_frames, slot_indices, sample_positions = [], [], []
        batch_order = epoch_order[first_position:first_position + self.batch_size]
        for batch_position, sample_position in enumerate(batch_order):
            for frame_position, frame_path in enumerate(self.sample_slots[sample_position]):
                if frame_path is not None:
                    encoded_frames.append(np.fromfile(file=frame_path, dtype=np.uint8))
                    slot_indices.append(np.array([batch_position * self.frames_per_sample + frame_position],
                                                 dtype=np.int64))
                    sample_positions.append(np.array([sample_position], dtype=np.int64))
        return encoded_frames, slot_indices, sample_positions


@pipeline_definition
def sequence_pipeline(source: WindowFrameSource, image_size: int, keep_ratio: bool,
                      source_prefetch_depth: int) -> Tuple[DataNode, DataNode, DataNode]:
    """
    DALI pipeline of the sequence prototype: the real frames of every batch read by the reader processes of the
    parallel external source, decoded on the GPU and resized like the Native loader. A DALI batch is the set of real
    frames of a batch of samples, whose number varies from batch to batch.

    Args:
        source (WindowFrameSource): The source of the frames of every batch.
        image_size (int): The size of the square network input.
        keep_ratio (bool): Whether to keep the aspect ratio of the frames, padding them.
        source_prefetch_depth (int): The number of batches read in advance by the reader processes.

    Returns:
        Tuple[DataNode, DataNode, DataNode]: The uint8 [3, size, size] frames, their slots in the batch and the
            positions of their samples.
    """
    encoded_frames, slot_indices, sample_positions = dali_function.external_source(
        source=source, num_outputs=3, batch=True, batch_info=True, parallel=True,
        prefetch_queue_depth=source_prefetch_depth, dtype=[dali_types.UINT8, dali_types.INT64, dali_types.INT64],
        ndim=[1, 1, 1])
    frames = dali_function.decoders.image(encoded_frames, device="mixed", output_type=dali_types.BGR)
    return resize_and_pad_images(images=frames, image_size=image_size, keep_ratio=keep_ratio), slot_indices, \
        sample_positions


class SequenceDALILoader:
    """
    Batch loader of the sequence prototype, yielding the batches of the Native sequence loaders: uint8
    [B, F, 3, S, S] BGR frames (on the GPU) whose padding frames are exact zeros, the [B, F] sequence mask, the labels
    (window class, frame classes or pair class) and the sample identifiers, F being the sequence length or twice it
    for the pairs. Every sample is given exactly once per epoch, and the Native dataset stays available as
    '.dataset'.
    """

    def __init__(self, dataset: Union[SequenceDataset, SequencePairDataset], batch_size: int, number_threads: int,
                 number_reader_processes: int, prefetch_queue_depth: int, shuffle: bool, seed: int,
                 device_id: int) -> None:
        """
        Builds the label and identifier tables of the samples and the DALI pipeline reading their frames.

        Args:
            dataset (Union[SequenceDataset, SequencePairDataset]): The Native dataset of the split.
            batch_size (int): The number of samples (windows or pairs) of a batch.
            number_threads (int): The number of CPU threads of the pipeline (Huffman decoding).
            number_reader_processes (int): The number of processes reading the frame files.
            prefetch_queue_depth (int): The number of batches prepared in advance by the pipeline.
            shuffle (bool): Whether to shuffle the samples at every epoch.
            seed (int): The seed of the permutations of the epochs.
            device_id (int): The index of the GPU.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.frames_per_sample = dataset.sequence_length * (len(PAIR_MEMBERS) if isinstance(
            dataset, SequencePairDataset) else 1)

        # Label and identifier of every sample, indexed by its position in the dataset
        records = dataset.pairs if isinstance(dataset, SequencePairDataset) else dataset.windows
        self.label_table = build_sequence_label_table(dataset=dataset)
        self.identifier_table = torch.tensor(data=[record["identifier"] for record in records], dtype=torch.int64)

        # Pipeline whose DALI batches hold at most every slot of a batch of samples
        source = WindowFrameSource(sample_slots=build_sample_frame_slots(dataset=dataset),
                                   frames_per_sample=self.frames_per_sample, batch_size=batch_size, shuffle=shuffle,
                                   seed=seed)
        pipeline = sequence_pipeline(source=source, image_size=dataset.image_size, keep_ratio=dataset.keep_ratio,
                                     source_prefetch_depth=max(prefetch_queue_depth, number_reader_processes),
                                     batch_size=batch_size * self.frames_per_sample, num_threads=number_threads,
                                     device_id=device_id, prefetch_queue_depth=prefetch_queue_depth,
                                     py_num_workers=number_reader_processes, py_start_method="spawn")
        pipeline.build()
        self.dali_iterator = DALIGenericIterator(pipelines=[pipeline],
                                                 output_map=["frames", "slot_indices", "sample_positions"],
                                                 last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yields the batches of an epoch: the decoded real frames written into their slots of a zero tensor, and the
        mask, labels and identifiers rebuilt from the slots and sample positions.

        Yields:
            Dict[str, torch.Tensor]: The 'images' (on the GPU), 'sequence_mask', 'labels' and 'sample_identifiers' of a
                batch.
        """
        for outputs in self.dali_iterator:
            frames = outputs[0]["frames"]
            slot_indices = outputs[0]["slot_indices"].reshape(-1)
            sample_positions = outputs[0]["sample_positions"].reshape(-1)

            # Position of every sample in the batch, every sample having at least one real frame
            batch_positions = torch.div(input=slot_indices, other=self.frames_per_sample, rounding_mode="floor")
            number_samples = int(batch_positions.max()) + 1
            batch_sample_positions = torch.zeros(size=(number_samples,), dtype=torch.int64)
            batch_sample_positions[batch_positions] = sample_positions

            # Real frames in their slots, the padding slots staying black, and the mask of the real frames
            images = frames.new_zeros(size=(number_samples * self.frames_per_sample, *frames.shape[1:]))
            images[slot_indices.to(device=frames.device)] = frames
            sequence_mask = torch.zeros(size=(number_samples * self.frames_per_sample,), dtype=torch.bool)
            sequence_mask[slot_indices] = True

            yield {"images": images.view(number_samples, self.frames_per_sample, *frames.shape[1:]),
                   "sequence_mask": sequence_mask.view(number_samples, self.frames_per_sample),
                   "labels": self.label_table[batch_sample_positions],
                   "sample_identifiers": self.identifier_table[batch_sample_positions]}

    def __len__(self) -> int:
        """
        Gives the number of batches of an epoch.

        Returns:
            int: The number of batches.
        """
        return math.ceil(len(self.dataset) / self.batch_size)
