"""
Tests of SerializedRecords (vision_training/data/serialized_records.py), the read-only byte buffer of pickled records
holding the per-image records of CocoDetectionDataset, so that the forked loader workers never copy Python objects of
the main process: every record comes back equal to the one given, as a new object, in any order and alone, even when
records share objects, the records are kept as NumPy arrays only, an empty set of records gives an empty buffer, and an
index outside the records is refused.
"""

import numpy as np
import pytest

from typing import Any, List

from vision_training.data.serialized_records import SerializedRecords


def build_coco_records() -> List[Any]:
    """
    Builds records shaped like those of CocoDetectionDataset, an image entry and its annotations, with the values a COCO
    manifest holds: integers, floats, strings with accents, nested dictionaries and lists, an image without annotations.

    Returns:
        List[Any]: The records.
    """
    return [({"id": 7, "file_name": "shapes/image_7.png", "width": 40, "height": 20},
             [{"id": 1, "image_id": 7, "category_id": 2, "iscrowd": 0, "area": 12.5, "bbox": [1.5, 2.0, 3.0, 4.25],
               "segmentation": {"size": [20, 40], "counts": "Xk1<P0Ql0"}},
              {"id": 2, "image_id": 7, "category_id": 1, "iscrowd": 1,
               "segmentation": [[10.0, 20.0, 30.0, 20.0, 30.0, 50.0]]}]),
            ({"id": 9, "file_name": "café/image_9.png", "width": 40, "height": 20}, []),
            ({"id": 3, "file_name": "image_3.png", "width": 64, "height": 48, "license": None},
             [{"id": 3, "image_id": 3, "category_id": 1, "iscrowd": 0, "bbox": [0, 0, 64, 48]}])]


def test_records_come_back_equal_as_new_objects() -> None:
    """
    Verifies that every record comes back equal to the one given, in its order, as a new object of the reading process,
    from a read-only uint8 buffer and int64 offsets, the only attributes kept.
    """
    records = build_coco_records()
    serialized_records = SerializedRecords(records=(record for record in records))

    assert len(serialized_records) == len(records)
    assert [serialized_records[record_index] for record_index in range(len(records))] == records
    assert serialized_records[0] is not records[0] and serialized_records[0][1] is not records[0][1]

    # Only NumPy arrays, which the forked workers read without writing to any reference count of the main process
    assert sorted(vars(serialized_records)) == ["buffer", "record_ends"]
    assert serialized_records.buffer.dtype == np.uint8 and not serialized_records.buffer.flags.writeable
    assert serialized_records.record_ends.dtype == np.int64


def test_records_read_alone_in_any_order_even_sharing_objects() -> None:
    """
    Verifies that every record unpickles alone, the last one first, even when records hold the same objects (a shared
    category dictionary, the same strings), which a pickler keeping its memo across records would write as references
    to the previous records.
    """
    shared_category = {"id": 1, "name": "circle", "supercategory": "shape"}
    records = [[shared_category, "circle"], [shared_category, "circle"], {"category": shared_category}]
    serialized_records = SerializedRecords(records=records)

    assert [serialized_records[record_index] for record_index in [2, 1, 0]] == [records[2], records[1], records[0]]


def test_empty_records_give_an_empty_buffer() -> None:
    """
    Verifies that a split without images gives no records, an empty buffer, and refuses any index.
    """
    serialized_records = SerializedRecords(records=[])

    assert len(serialized_records) == 0 and serialized_records.buffer.shape == (0,)
    with pytest.raises(expected_exception=IndexError, match="outside the 0 records"):
        serialized_records[0]


@pytest.mark.parametrize(argnames="record_index", argvalues=[-1, 3, 10])
def test_index_outside_the_records_is_refused(record_index: int) -> None:
    """
    Verifies that an index outside the records, negative or past the last record, is refused instead of reading the
    bytes of another record.

    Args:
        record_index (int): The index read.
    """
    serialized_records = SerializedRecords(records=build_coco_records())

    with pytest.raises(expected_exception=IndexError, match=f"Record index {record_index} is outside the 3 records"):
        serialized_records[record_index]
