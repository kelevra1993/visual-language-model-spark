"""
Tests of the Native loader settings of vision_training/data/loaders.py: build_native_loader with worker processes
(kept alive between epochs, pinned memory, prefetch, a single OpenCV thread per worker, also once OpenCV has run its
thread pool in the main process) and without them.
"""

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from vision_training.data.loaders import build_native_loader


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
    Verifies that the loader of a dataset uses the 'Data.loading' settings: worker processes kept alive between
    epochs, the prefetch depth, pinned memory, and workers decoding with a single OpenCV thread, also when OpenCV has
    already run its thread pool in the main process (setting the threads inside a forked worker then blocks forever).
    """
    number_threads = cv2.getNumThreads()

    # OpenCV thread pool started in this process before the loader exists
    cv2.resize(src=np.zeros(shape=(2000, 2000, 3), dtype=np.uint8), dsize=(500, 500))
    loader = build_native_loader(dataset=OpenCVThreadsDataset(), batch_size=2,
                                 loading_configuration={"number_workers": 2, "prefetch_batches": 3})
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3
    assert loader.pin_memory is True

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
    loader = build_native_loader(dataset=OpenCVThreadsDataset(), batch_size=2,
                                 loading_configuration={"number_workers": 0, "prefetch_batches": 2})
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
    ordered_loader = build_native_loader(dataset=dataset, batch_size=8, shuffle=False,
                                         loading_configuration={"number_workers": 0, "prefetch_batches": 2})
    for _ in range(2):
        assert torch.cat(tensors=[batch[0] for batch in ordered_loader]).tolist() == list(range(64))

    # The default loader shuffles, 64 samples left in their order by chance having a probability of 1 / 64!
    shuffled_loader = build_native_loader(dataset=dataset, batch_size=8,
                                          loading_configuration={"number_workers": 0, "prefetch_batches": 2})
    assert torch.cat(tensors=[batch[0] for batch in shuffled_loader]).tolist() != list(range(64))
