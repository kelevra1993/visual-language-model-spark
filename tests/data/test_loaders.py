"""
Tests of the Native loader settings of vision_training/data/loaders.py: build_native_loader with worker processes
(kept alive between epochs, pinned memory on a discrete GPU only, prefetch, a single OpenCV thread per worker, also
once OpenCV has run its thread pool in the main process) and without them, and the choice of the pinned memory by the
kind of GPU (uses_pinned_memory).
"""

import cv2
import numpy as np
import pytest
import torch

from types import SimpleNamespace
from typing import Optional

from torch.utils.data import Dataset

from vision_training.data.loaders import (NUMBER_LOADER_WORKERS, PREFETCH_BATCHES, build_native_loader,
                                          uses_pinned_memory)


class OpenCVThreadsDataset(Dataset):
    """
    Tiny dataset whose samples give the number of OpenCV threads of the process loading them, to verify the
    initialisation of the loader workers.
    """

    def __len__(self) -> int:
        """
        Gives the number of samples.

        Returns:
            int: The number of samples.
        """
        return 4

    def __getitem__(self, sample_index: int) -> torch.Tensor:
        """
        Gives the number of OpenCV threads of the loading process.

        Args:
            sample_index (int): The index of the sample.

        Returns:
            torch.Tensor: The number of OpenCV threads.
        """
        return torch.tensor(data=cv2.getNumThreads())


def test_native_loader_with_workers() -> None:
    """
    Verifies that the loader of a dataset uses the fixed loading settings of the trainings: NUMBER_LOADER_WORKERS worker
    processes kept alive between epochs, PREFETCH_BATCHES batches prefetched by each, pinned memory with a discrete GPU
    only (uses_pinned_memory, so none on the integrated GPU of the DGX Spark), and workers decoding with a single
    OpenCV thread, also when OpenCV has already run its thread pool in the main process (setting the threads inside a
    forked worker then blocks forever).
    """
    number_threads = cv2.getNumThreads()

    # OpenCV thread pool started in this process before the loader exists
    cv2.resize(src=np.zeros(shape=(2000, 2000, 3), dtype=np.uint8), dsize=(500, 500))
    loader = build_native_loader(dataset=OpenCVThreadsDataset(), batch_size=2)
    assert loader.num_workers == NUMBER_LOADER_WORKERS
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == PREFETCH_BATCHES
    assert loader.pin_memory is uses_pinned_memory()

    # Two epochs read by the same workers, every one of them with a single OpenCV thread
    for _ in range(2):
        assert torch.cat(tensors=list(loader)).tolist() == [1, 1, 1, 1]

    # OpenCV threads of the test process restored once the workers are gone
    del loader
    cv2.setNumThreads(nthreads=number_threads)


def test_native_loader_without_workers() -> None:
    """
    Verifies that 0 workers loads the samples in the main process, without the settings that need worker processes.
    """
    loader = build_native_loader(dataset=OpenCVThreadsDataset(), batch_size=2, number_workers=0)
    assert loader.num_workers == 0
    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None
    assert len(list(loader)) == 2


def test_native_loader_keeps_the_sample_order_without_shuffle() -> None:
    """
    Verifies that a loader built with shuffle False gives the samples in their order at every pass, as the embedding
    extraction of scripts/embedding/cluster_image_embeddings.py needs to write every batch to its rows, while the
    default loader shuffles them.
    """
    dataset = torch.utils.data.TensorDataset(torch.arange(end=64))
    ordered_loader = build_native_loader(dataset=dataset, batch_size=8, shuffle=False, number_workers=0)
    for _ in range(2):
        assert torch.cat(tensors=[batch[0] for batch in ordered_loader]).tolist() == list(range(64))

    # The default loader shuffles, 64 samples left in their order by chance having a probability of 1 / 64!
    shuffled_loader = build_native_loader(dataset=dataset, batch_size=8, number_workers=0)
    assert torch.cat(tensors=[batch[0] for batch in shuffled_loader]).tolist() != list(range(64))


@pytest.mark.parametrize(argnames=["cuda_available", "integrated", "expected_pinned"],
                         argvalues=[(True, 0, True), (True, 1, False), (False, None, False)])
def test_batches_are_pinned_for_a_discrete_gpu_only(monkeypatch: pytest.MonkeyPatch, cuda_available: bool,
                                                    integrated: Optional[int], expected_pinned: bool) -> None:
    """
    Verifies that the loaders pin their batches with a discrete GPU, and neither with an integrated GPU sharing the
    host memory (the GB10 of the DGX Spark, whose pinned copies only take memory) nor without any GPU, both in
    uses_pinned_memory and in the loader built by build_native_loader.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the CUDA availability and the properties of the current device.
        cuda_available (bool): Whether a CUDA GPU is available.
        integrated (Optional[int]): The is_integrated property of the GPU (1 for an integrated one), None without GPU.
        expected_pinned (bool): Whether the batches must be pinned.
    """
    # Availability and kind of the current CUDA device, its properties read only when a GPU is available
    monkeypatch.setattr(target=torch.cuda, name="is_available", value=lambda: cuda_available)
    monkeypatch.setattr(target=torch.cuda, name="current_device", value=lambda: 0)
    monkeypatch.setattr(target=torch.cuda, name="get_device_properties",
                        value=lambda device: SimpleNamespace(is_integrated=integrated))

    # The choice and the loader of a training follow the kind of GPU
    assert uses_pinned_memory() is expected_pinned
    loader = build_native_loader(dataset=OpenCVThreadsDataset(), batch_size=2, number_workers=0)
    assert loader.pin_memory is expected_pinned
