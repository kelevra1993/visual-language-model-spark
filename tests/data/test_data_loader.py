import json
import pytest

from pathlib import Path
from typing import Any, Dict

from vision_training.data.data_loader import count_split_images_and_annotations


@pytest.mark.parametrize(argnames=("labels", "expected_counts"), argvalues=[
    ({"images": [{"id": 1}, {"id": 2}, {"id": 3}],
      "annotations": [{"id": 1, "image_id": 1}, {"id": 2, "image_id": 1}, {"id": 3, "image_id": 3},
                      {"id": 4, "image_id": 3}],
      "categories": [{"id": 1, "name": "square"}]}, (3, 4)),
    ({"images": [{"id": 1}, {"id": 2}], "categories": [{"id": 1, "name": "square"}]}, (2, 0))])
def test_split_images_and_annotations_are_counted(tmp_path: Path, labels: Dict[str, Any],
                                                  expected_counts: tuple) -> None:
    """
    Verifies that the images and annotations of a split are counted from its labels.json, and that an unlabelled
    split, without an 'annotations' list, counts 0 annotations.

    Args:
        tmp_path (Path): The pytest temporary directory, used as the split folder.
        labels (Dict[str, Any]): The COCO content of the split's labels.json.
        expected_counts (tuple): The expected numbers of images and annotations.
    """
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    assert count_split_images_and_annotations(split_data_folder=tmp_path) == expected_counts
