"""
Tests of get_class_names and build_label_dictionary of vision_training/data/label_mapping.py: the classes are named
after the categories of a manifest, merged categories joined with '+', a class named by no category keeps its name in
the label dictionary (the background and foreground classes of the detector) or else its index, and the categories
mapped to the background of a detector name no class.
"""

import json

from pathlib import Path

from vision_training.data.label_mapping import build_label_dictionary, get_class_names

# Categories of a manifest of the tests
CATEGORIES = [{"id": 1, "name": "square"}, {"id": 2, "name": "circle"}, {"id": 3, "name": "star"}]


def test_classes_are_named_after_their_categories() -> None:
    """
    Verifies the names of the classes of a classifier: one category per class, two categories merged into one class
    joined with '+', and a class without category keeping its index.
    """
    assert get_class_names(categories=CATEGORIES, label_dictionary={1: 0, 2: 1, 3: 2}) == {0: "square", 1: "circle",
                                                                                           2: "star"}
    assert get_class_names(categories=CATEGORIES, label_dictionary={1: 0, 2: 0, 3: 1}) == {0: "square+circle",
                                                                                           1: "star"}
    assert get_class_names(categories=CATEGORIES[:1], label_dictionary={1: 0, 4: 1}) == {0: "square", 1: "1"}


def test_detector_classes_keep_their_dictionary_names() -> None:
    """
    Verifies the names of the classes of the detector: its background class keeps its name, its foreground classes are
    named after their categories, and the single foreground class of the binary mode keeps its name.
    """
    assert get_class_names(categories=CATEGORIES, label_dictionary={"background": 0, 1: 1, 2: 2, 3: 3}) == {
        0: "background", 1: "square", 2: "circle", 3: "star"}
    assert get_class_names(categories=CATEGORIES, label_dictionary={"background": 0, "foreground": 1}) == {
        0: "background", 1: "foreground"}


def test_categories_mapped_to_the_background_leave_it_its_name(tmp_path: Path) -> None:
    """
    Verifies the categories mapped to 0 to leave them out of a detector: the label dictionary built for the object
    detection names the background class even when a category is mapped to 0, that category names no class, the
    background keeping its name, and a classifier, which has no background class, gets no background key.

    Args:
        tmp_path (Path): The pytest temporary directory, holding the mapping file.
    """
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.write_text(data=json.dumps(obj={"1": 1, "2": 0, "3": 2}))
    mapping_configuration = {"mode": "multi_class", "mapping_file_path": str(mapping_path)}
    label_dictionary = build_label_dictionary(label_mapping_configuration=mapping_configuration,
                                              task="object_detection")
    assert label_dictionary == {1: 1, 2: 0, 3: 2, "background": 0}
    assert get_class_names(categories=CATEGORIES, label_dictionary=label_dictionary) == {
        0: "background", 1: "square", 2: "star"}
    assert "background" not in build_label_dictionary(label_mapping_configuration=mapping_configuration,
                                                      task="classification")
