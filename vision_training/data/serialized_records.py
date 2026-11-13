"""
Storage of the per-image records of a dataset (the image entry and the annotations of every image of a COCO manifest)
as one read-only NumPy byte buffer of pickled records with their offsets, instead of millions of Python dictionaries and
lists. The DataLoader workers, forked from the main process, read a record by slicing the buffer and unpickling it, so
they never write to the reference counts of Python objects of the main process and never copy its pages (the
copy-on-access of Python objects that the PyTorch documentation of multi-process data loading warns about, issue
13246): a worker reading every record of the 50,000 validation images of the hard instance segmentation dataset copied
0.64 GB of the main process with the parsed manifest, and copies 0.05 GB with the packed records. Used by
CocoDetectionDataset of data/detection/coco_dataset.py.
"""

import io
import numpy as np
import pickle

from typing import Any, Iterable


class SerializedRecords:
    """
    Records of a dataset pickled one after the other into a single read-only NumPy byte buffer, read back one at a time
    by their index, the buffer and its offsets being the only Python objects the loader workers touch.
    """

    def __init__(self, records: Iterable[Any]) -> None:
        """
        Pickles the records one after the other into one byte stream, keeping the end offset of every record, the memo
        of the pickler being cleared after each record so that every record unpickles alone. The stream grows in place
        and becomes the buffer without any copy, allocated by the C library rather than by NumPy, whose huge page advice
        stalls the first writes of a large array while the kernel compacts fragmented memory (3.7 seconds to fill the
        0.31 GB of the validation split of the hard instance segmentation dataset while two trainings ran). A generator
        of records lets the caller free every source record as soon as it is pickled.

        Args:
            records (Iterable[Any]): The picklable records, in the order of their indices.
        """
        # Every record pickled with the highest protocol, the fastest and most compact one, and its end offset kept
        stream = io.BytesIO()
        pickler = pickle.Pickler(file=stream, protocol=pickle.HIGHEST_PROTOCOL)
        record_ends = []
        for record in records:
            pickler.dump(record)
            pickler.clear_memo()
            record_ends.append(stream.tell())

        # The bytes of the stream, trimmed to their size, taken as the read-only buffer, and the offsets as one array
        self.buffer = np.frombuffer(buffer=stream.getvalue(), dtype=np.uint8)
        self.record_ends = np.array(object=record_ends, dtype=np.int64)

    def __len__(self) -> int:
        """
        Gives the number of records, the number of samples of the dataset holding them.

        Returns:
            int: The number of records.
        """
        return len(self.record_ends)

    def __getitem__(self, record_index: int) -> Any:
        """
        Unpickles one record from its bytes in the buffer, a new Python object of the calling process, so that the
        loader workers never touch the objects of the main process.

        Args:
            record_index (int): The index of the record, from 0 to the number of records minus one.

        Returns:
            Any: The record, as it was given.

        Raises:
            IndexError: If the index is outside the records.
        """
        if not 0 <= record_index < len(self.record_ends):
            raise IndexError(f"Record index {record_index} is outside the {len(self.record_ends)} records.")

        # Bytes of the record, between the end of the previous record and its own end
        record_start = int(self.record_ends[record_index - 1]) if record_index > 0 else 0
        return pickle.loads(self.buffer[record_start:int(self.record_ends[record_index])].tobytes())
