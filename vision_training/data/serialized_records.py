"""
Storage of the per-sample records of a dataset (the image entry and the annotations of every image of a COCO manifest,
the samples of a classification or segmentation manifest, the sequences and the sequence pairs) as one read-only NumPy
byte buffer of pickled records with their offsets, instead of millions of Python dictionaries and lists. The DataLoader
workers, forked from the main process, read a record by slicing the buffer and unpickling it, so they never write to the
reference counts of Python objects of the main process and never copy its pages (the copy-on-access of Python objects
that the PyTorch documentation of multi-process data loading warns about, issue 13246): a worker reading every record
of the 50,000 validation images of the hard instance segmentation dataset copied 0.64 GB of the main process with the
parsed manifest, and copies 0.05 GB with the packed records; one reading the 1.27 million training samples of
ImageNet-1k copied 0.40 GB, and copies nothing. Used by CocoDetectionDataset of data/detection/coco_dataset.py, and by
ClassificationDataset, SegmentationDataset, SequenceDataset and SequencePairDataset, which also find a record by its
identifier to name the evaluated samples, and list the records of the smallest identifiers, the input samples drawn
at the start of every training launch (trainers/input_samples.py).
"""

import io
import numpy as np
import pickle

from collections.abc import Sequence
from typing import Any, Callable, Iterable, List, Optional, Tuple


class SerializedRecords(Sequence):
    """
    Records of a dataset pickled one after the other into a single read-only NumPy byte buffer, read back like a
    read-only list (by index, by slice or in order) or by their identifier, the buffer, its offsets and the sorted
    identifiers being the only Python objects the loader workers touch.
    """

    # Records packed without identifier key have no identifiers, find_record refusing them
    sorted_identifiers: Optional[np.ndarray] = None
    sorted_record_indices: Optional[np.ndarray] = None

    def __init__(self, records: Iterable[Any], identifier_key: Optional[str | Callable[[Any], int]] = None) -> None:
        """
        Pickles the records one after the other into one byte stream, keeping the end offset of every record, the memo
        of the pickler being cleared after each record so that every record unpickles alone. The stream grows in place
        and becomes the buffer without any copy, allocated by the C library rather than by NumPy, whose huge page advice
        stalls the first writes of a large array while the kernel compacts fragmented memory (3.7 seconds to fill the
        0.31 GB of the validation split of the hard instance segmentation dataset while two trainings ran). A generator
        of records lets the caller free every source record as soon as it is pickled. With an identifier key, the
        identifiers of the records are sorted once with the index of their record, for find_record.

        Args:
            records (Iterable[Any]): The picklable records, in the order of their indices.
            identifier_key (Optional[str | Callable[[Any], int]]): The key of the integer identifier of every record,
                a dictionary, by which find_record finds it ("identifier" for the samples, the identifier given in the
                batches), or the function reading it from a record of another type (the COCO identifier of an image
                record, a tuple); None for records only read by their index. Defaults to None.

        Raises:
            ValueError: If the identifier of a record is not an integer.
        """
        # Every record pickled with the highest protocol, the fastest and most compact one, and its end offset kept
        stream = io.BytesIO()
        pickler = pickle.Pickler(file=stream, protocol=pickle.HIGHEST_PROTOCOL)
        record_ends = []
        record_identifiers = []
        for record in records:
            pickler.dump(record)
            pickler.clear_memo()
            record_ends.append(stream.tell())
            if callable(identifier_key):
                record_identifiers.append(identifier_key(record))
            elif identifier_key is not None:
                record_identifiers.append(record[identifier_key])

        # The bytes of the stream, trimmed to their size, taken as the read-only buffer, and the offsets as one array
        self.buffer = np.frombuffer(buffer=stream.getvalue(), dtype=np.uint8)
        self.record_ends = np.array(object=record_ends, dtype=np.int64)

        # Identifiers sorted with the index of their record, the stable sort keeping the records of a shared
        # identifier in their order
        if identifier_key is not None:
            identifiers = (np.array(object=record_identifiers) if record_identifiers else
                           np.zeros(shape=(0,), dtype=np.int64))
            if identifiers.dtype.kind != "i":
                example = next((identifier for identifier in record_identifiers if type(identifier) is not int),
                               record_identifiers[0])
                key_name = getattr(identifier_key, "__name__", identifier_key)
                raise ValueError(f"The '{key_name}' of every record must be an integer, the identifier of its sample "
                                 f"in the batches, got {example!r}.")
            self.sorted_record_indices = np.argsort(a=identifiers, kind="stable")
            self.sorted_identifiers = identifiers[self.sorted_record_indices]

    def __len__(self) -> int:
        """
        Gives the number of records, the number of samples of the dataset holding them.

        Returns:
            int: The number of records.
        """
        return len(self.record_ends)

    def __getitem__(self, record_index: int | slice) -> Any:
        """
        Unpickles one record from its bytes in the buffer, a new Python object of the calling process, so that the
        loader workers never touch the objects of the main process; a slice gives the list of its records, and the
        passes over every record (for record in records) read them one by one.

        Args:
            record_index (int | slice): The index of the record, from 0 to the number of records minus one, or a slice
                of indices.

        Returns:
            Any: The record, as it was given, or the list of the records of the slice.

        Raises:
            IndexError: If the index is outside the records.
        """
        # The records of a slice, read one by one
        if isinstance(record_index, slice):
            return [self[index] for index in range(*record_index.indices(len(self.record_ends)))]
        if not 0 <= record_index < len(self.record_ends):
            raise IndexError(f"Record index {record_index} is outside the {len(self.record_ends)} records.")

        # Bytes of the record, between the end of the previous record and its own end
        record_start = int(self.record_ends[record_index - 1]) if record_index > 0 else 0
        return pickle.loads(self.buffer[record_start:int(self.record_ends[record_index])].tobytes())

    def find_record(self, identifier: int) -> Any:
        """
        Unpickles the record of an identifier, the sample of a prediction named in the evaluation outputs or the image
        of a batch shown by an input viewer, by a binary search of the sorted identifiers; of several records sharing
        the identifier, the last one is given.

        Args:
            identifier (int): The identifier of the record, as given in the 'sample_identifiers' of a batch.

        Returns:
            Any: The record.

        Raises:
            ValueError: If the records were packed without identifier key.
            KeyError: If no record has the identifier.
        """
        if self.sorted_identifiers is None:
            raise ValueError("The records were packed without identifier key: they are only read by their index.")

        # Last position of the identifier among the sorted identifiers, right before the first larger one
        position = int(np.searchsorted(a=self.sorted_identifiers, v=identifier, side="right")) - 1
        if position < 0 or self.sorted_identifiers[position] != identifier:
            raise KeyError(f"No record has the identifier {identifier}.")
        return self[int(self.sorted_record_indices[position])]

    def list_first_records(self, number_records: int) -> List[Tuple[int, int]]:
        """
        Lists the records of the smallest identifiers with their index, so that the input samples written at the start
        of every training launch (trainers/input_samples.py) show the same samples of a split at every launch, whatever
        the order the dataset shuffled its records into. Records sharing an identifier keep their order.

        Args:
            number_records (int): The number of records listed, all of them when the records are fewer.

        Returns:
            List[Tuple[int, int]]: The index and the identifier of every listed record, by increasing identifier.

        Raises:
            ValueError: If the records were packed without identifier key.
        """
        if self.sorted_identifiers is None:
            raise ValueError("The records were packed without identifier key: they are only read by their index.")
        return [(int(record_index), int(identifier))
                for record_index, identifier in zip(self.sorted_record_indices[:number_records],
                                                    self.sorted_identifiers[:number_records])]
