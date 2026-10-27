"""
Tests of the classification data of vision_training/data/classification: ClassificationDataset loading multi-class and
multi-label manifests, keeping the unlabelled samples for the self-supervised task, whose loaders keep full training
and validation batches, as the training loader of a classifier needing batch statistics does, the console report of
the samples without mapped category of the three splits, the corrupted images named in their error, the label mapping
without background class, the manifests written by create_class_folder_manifests from class folders, and the square
crop moving at random on the training images only.
"""

import json
import cv2
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, List

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.classification.class_folder_manifests import create_class_folder_manifests
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.data.label_mapping import build_label_dictionary
from vision_training.data.loaders import get_classification_dataloaders
from vision_training.utilities.os_utilities import INVISIBLE_SEQUENCE_PATTERN


def write_split(split_folder: Path, categories: List[Dict[str, Any]], samples: List[Dict[str, Any]]) -> None:
    """
    Writes a classification split in the layout read by the classification loader: one 32 pixel image per sample in
    'data/' and the 'labels.json' manifest listing the categories and the samples.

    Args:
        split_folder (Path): The folder of the split.
        categories (List[Dict[str, Any]]): The 'id' and 'name' of every category of the manifest.
        samples (List[Dict[str, Any]]): The samples of the manifest, with their 'id', 'file_name' and categories.
    """
    (split_folder / "data").mkdir(parents=True)
    for sample in samples:
        cv2.imwrite(filename=str(split_folder / "data" / sample["file_name"]),
                    img=np.full(shape=(32, 48, 3), fill_value=sample["id"], dtype=np.uint8))
    (split_folder / "labels.json").write_text(data=json.dumps(obj={"categories": categories, "samples": samples}))


def test_multi_class_manifest_loading(tmp_path: Path) -> None:
    """
    Verifies that the dataset reads a multi-class manifest: the images are resized to the network input, the raw
    categories become class indices, the samples of unmapped categories are left out and counted for the report of the
    loaders, the class names come from the categories (joined when the label dictionary merges two categories), the
    class counts and the class of every sample match, and a sample is named by its file name from its identifier.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    categories = [{"id": 1, "name": "circle"}, {"id": 2, "name": "square"}, {"id": 3, "name": "box"},
                  {"id": 4, "name": "star"}]
    samples = [{"id": 1, "file_name": "a.png", "category_id": 1}, {"id": 2, "file_name": "b.png", "category_id": 2},
               {"id": 3, "file_name": "c.png", "category_id": 3}, {"id": 4, "file_name": "d.png", "category_id": 4}]
    write_split(split_folder=tmp_path, categories=categories, samples=samples)

    # Squares and boxes are merged into class 1, stars are not mapped
    dataset = ClassificationDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                    image_size=64, keep_ratio=True, label_dictionary={1: 0, 2: 1, 3: 1},
                                    output_mode="multi_class", shuffle=False)
    assert len(dataset) == 3 and dataset.number_unlabelled_samples == 1
    assert dataset.class_names == {0: "circle", 1: "square+box"}
    assert dataset.get_class_counts() == [1, 2]
    assert dataset.get_sample_class_indices() == {1: [0], 2: [1], 3: [1]}

    sample = dataset[1]
    assert sample["images"].shape == (3, 64, 64) and sample["images"].dtype == torch.uint8
    assert sample["labels"].dtype == torch.int64 and int(sample["labels"]) == 1
    assert sample["sample_identifiers"] == 2
    assert dataset.describe_sample(sample_identifier=sample["sample_identifiers"]) == {"sample": "b.png"}


def test_multi_label_manifest_loading(tmp_path: Path) -> None:
    """
    Verifies that the dataset reads a multi-label manifest: every sample gets the multi-hot vector of its mapped
    categories, a sample without any mapped category being kept as a negative of every class, without any class among
    the classes of the samples.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    categories = [{"id": 1, "name": "red"}, {"id": 2, "name": "round"}, {"id": 3, "name": "large"}]
    samples = [{"id": 1, "file_name": "a.png", "category_ids": [1, 2]}, {"id": 2, "file_name": "b.png",
                                                                         "category_ids": [3]}]
    write_split(split_folder=tmp_path, categories=categories, samples=samples)

    dataset = ClassificationDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                    image_size=32, keep_ratio=False, label_dictionary={1: 0, 2: 1},
                                    output_mode="multi_label", shuffle=False)
    assert len(dataset) == 2
    assert dataset[0]["labels"].tolist() == [1.0, 1.0] and dataset[0]["labels"].dtype == torch.float32
    assert dataset[1]["labels"].tolist() == [0.0, 0.0]
    assert dataset.get_class_counts() == [1, 1]
    assert dataset.get_sample_class_indices() == {1: [0, 1], 2: []}


def test_unlabelled_samples_are_kept_for_the_self_supervised_task(tmp_path: Path) -> None:
    """
    Verifies that the self-supervised task, which trains on every image, keeps the samples without category, those of
    unmapped categories and the multi-label samples of several mapped categories with the ignored label and no class,
    counted for the report of the loaders, while the samples of a single mapped category, given by 'category_id' or by
    the 'category_ids' of a multi-label manifest, keep their class for the k-nearest neighbour evaluation.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    categories = [{"id": 1, "name": "circle"}, {"id": 4, "name": "star"}, {"id": 5, "name": "triangle"}]
    samples = [{"id": 1, "file_name": "a.png", "category_id": 1}, {"id": 2, "file_name": "b.png", "category_id": 4},
               {"id": 3, "file_name": "c.png"}, {"id": 4, "file_name": "d.png", "category_ids": [5, 4]},
               {"id": 5, "file_name": "e.png", "category_ids": [1, 5]}]
    write_split(split_folder=tmp_path, categories=categories, samples=samples)

    dataset = ClassificationDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                    image_size=32, keep_ratio=True, label_dictionary={1: 0, 5: 1},
                                    output_mode="multi_class", shuffle=False, keep_unlabelled_samples=True)
    assert len(dataset) == 5 and dataset.number_unlabelled_samples == 3
    assert dataset.get_sample_class_indices() == {1: [0], 2: [], 3: [], 4: [1], 5: []}
    assert [int(dataset[sample_index]["labels"]) for sample_index in range(5)] == [
        0, IGNORED_LABEL_INDEX, IGNORED_LABEL_INDEX, 1, IGNORED_LABEL_INDEX]
    assert dataset.categories == categories


def test_self_supervised_loaders_keep_full_batches(tmp_path: Path) -> None:
    """
    Verifies the loaders of the self-supervised task: every image is kept, the training and validation loaders leave
    out their last incomplete batch (the losses comparing the views of a batch with each other, a lone image would give
    a loss without gradient) while the test loader encodes every image, and a training or validation split smaller
    than a batch stops the launch instead of looping forever without batch.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    categories = [{"id": 1, "name": "circle"}]
    split_sizes = {"train": 5, "validation": 3, "test": 3}
    for split_name, number_samples in split_sizes.items():
        write_split(split_folder=tmp_path / split_name, categories=categories,
                    samples=[{"id": sample_identifier, "file_name": f"{sample_identifier}.png"}
                             for sample_identifier in range(1, number_samples + 1)])
    experiment_configuration = {f"{split_name}_data_folder": str(tmp_path / split_name) for split_name in split_sizes}
    model_configuration = {"Data": {"image_settings": {"size": 32, "keep_ratio": True, "square_crop": False},
                                    "label_dictionary": {1: 0},
                                    "loading": {"number_workers": 0, "prefetch_batches": 2},
                                    "grouped_batches": {"activated": False, "block_key": "camera",
                                                        "images_per_block": 2, "batch_key": None}}}

    # Batches of 2 images: 2 full training batches, 1 full validation batch, and the 3 test images in 2 batches
    training_loader, validation_loader, test_loader = get_classification_dataloaders(
        experiment_configuration=experiment_configuration, model_configuration=model_configuration, batch_size=2,
        self_supervised=True)
    assert [len(loader.dataset) for loader in [training_loader, validation_loader, test_loader]] == [5, 3, 3]
    assert [loader.drop_last for loader in [training_loader, validation_loader, test_loader]] == [True, True, False]
    assert [batch["images"].shape[0] for batch in training_loader] == [2, 2]
    assert [batch["images"].shape[0] for batch in test_loader] == [2, 1]

    # Batches of 4 images, more than the validation split holds
    with pytest.raises(expected_exception=ValueError, match="'validation_data_folder' split holds 3 images, fewer "
                                                            "than the batch size 4"):
        get_classification_dataloaders(experiment_configuration=experiment_configuration,
                                       model_configuration=model_configuration, batch_size=4, self_supervised=True)


def test_loaders_report_the_samples_without_mapped_category(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the console report of the classification loaders: once the three splits are built, the samples without a
    single mapped category are counted in one block, one line per split holding some, titled as left out for the
    classifier and as kept without class for the self-supervised task, a split without them having no line.

    Args:
        tmp_path (Path): The pytest temporary directory.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed report.
    """
    # Splits of 4 images, of which 2 training images and 1 validation image are of the unmapped category 2
    categories = [{"id": 1, "name": "circle"}, {"id": 2, "name": "square"}]
    for split_name, number_unmapped_samples in [("train", 2), ("validation", 1), ("test", 0)]:
        write_split(split_folder=tmp_path / split_name, categories=categories,
                    samples=[{"id": sample_identifier, "file_name": f"{sample_identifier}.png",
                              "category_id": 2 if sample_identifier <= number_unmapped_samples else 1}
                             for sample_identifier in range(1, 5)])
    experiment_configuration = {f"{split_name}_data_folder": str(tmp_path / split_name)
                                for split_name in ["train", "validation", "test"]}
    model_configuration = {"Data": {"image_settings": {"size": 32, "keep_ratio": True, "square_crop": False},
                                    "label_dictionary": {1: 0},
                                    "loading": {"number_workers": 0, "prefetch_batches": 2},
                                    "grouped_batches": {"activated": False, "block_key": "camera",
                                                        "images_per_block": 2, "batch_key": None}},
                           "ClassificationHead": {"output_mode": "multi_class"}}

    # The classifier leaves the samples out, the self-supervised task keeps them, both reporting them in one block
    titles = {False: "Samples Of Unmapped Categories, Left Out",
              True: "Samples Without A Single Mapped Category, Kept Without Class"}
    for self_supervised, expected_sizes in [(False, [2, 3, 4]), (True, [4, 4, 4])]:
        loaders = get_classification_dataloaders(experiment_configuration=experiment_configuration,
                                                 model_configuration=model_configuration, batch_size=2,
                                                 self_supervised=self_supervised)
        assert [len(loader.dataset) for loader in loaders] == expected_sizes
        printed_output = INVISIBLE_SEQUENCE_PATTERN.sub(repl="", string=capsys.readouterr().out)
        assert printed_output == (f"{titles[self_supervised]} :\n  - Train : 2 samples (labels.json)\n"
                                  f"  - Validation : 1 samples (labels.json)\n")


def test_classifier_needing_batch_statistics_drops_its_last_training_batch(tmp_path: Path) -> None:
    """
    Verifies the loaders of a classifier whose training needs batch statistics (a batch normalized head or the view
    losses, drop_last_training_batch): the training loader leaves out its last incomplete batch, which a single image
    would fill and break, while the validation and test loaders keep every image, and the training loader of the other
    classifiers keeps it too.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    categories = [{"id": 1, "name": "circle"}]
    split_sizes = {"train": 5, "validation": 3, "test": 3}
    for split_name, number_samples in split_sizes.items():
        write_split(split_folder=tmp_path / split_name, categories=categories,
                    samples=[{"id": sample_identifier, "file_name": f"{sample_identifier}.png", "category_id": 1}
                             for sample_identifier in range(1, number_samples + 1)])
    experiment_configuration = {f"{split_name}_data_folder": str(tmp_path / split_name) for split_name in split_sizes}
    model_configuration = {"Data": {"image_settings": {"size": 32, "keep_ratio": True, "square_crop": False},
                                    "label_dictionary": {1: 0},
                                    "loading": {"number_workers": 0, "prefetch_batches": 2},
                                    "grouped_batches": {"activated": False, "block_key": "camera",
                                                        "images_per_block": 2, "batch_key": None}},
                           "ClassificationHead": {"output_mode": "multi_class"}}

    # Batches of 2 images: the fifth training image is left out only for the training needing batch statistics
    for drop_last_training_batch, expected_training_batches in [(True, [2, 2]), (False, [2, 2, 1])]:
        training_loader, validation_loader, test_loader = get_classification_dataloaders(
            experiment_configuration=experiment_configuration, model_configuration=model_configuration, batch_size=2,
            drop_last_training_batch=drop_last_training_batch)
        assert [loader.drop_last for loader in [training_loader, validation_loader, test_loader]] == [
            drop_last_training_batch, False, False]
        assert [batch["images"].shape[0] for batch in training_loader] == expected_training_batches
        assert [batch["images"].shape[0] for batch in validation_loader] == [2, 1]


def test_unreadable_image_is_named(tmp_path: Path) -> None:
    """
    Verifies that a corrupted image of a classification split stops its sample with an error naming the file, which
    the training loop prints before dropping the batch, instead of an unrelated error of the resizing.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    samples = [{"id": 1, "file_name": "a.png", "category_id": 1}, {"id": 2, "file_name": "b.png", "category_id": 1}]
    write_split(split_folder=tmp_path, categories=[{"id": 1, "name": "circle"}], samples=samples)
    (tmp_path / "data" / "b.png").write_bytes(data=b"not an image")
    dataset = ClassificationDataset(data_directory=str(tmp_path / "data"), labels_file=str(tmp_path / "labels.json"),
                                    image_size=32, keep_ratio=True, label_dictionary={1: 0}, output_mode="multi_class",
                                    shuffle=False)

    # The readable image loads, the corrupted one is named
    assert dataset[0]["images"].shape[-2:] == (32, 32)
    with pytest.raises(expected_exception=ValueError, match="The image cannot be read : ") as error:
        dataset[1]
    assert "b.png" in str(error.value)


def test_label_mapping_without_background_class(tmp_path: Path) -> None:
    """
    Verifies that the label dictionary of the classification task keeps the class indices of the mapping file without
    adding the background class of the object detector, which names its class 0 'background' even when the mapping
    file maps a category to 0 to leave it out, and that the binary mode, which only makes sense next to a background
    class, gives no dictionary for it.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.write_text(data=json.dumps(obj={"1": 0, "2": 1}))
    label_mapping_configuration = {"mode": "multi_class", "mapping_file_path": str(mapping_path)}

    assert build_label_dictionary(label_mapping_configuration=label_mapping_configuration,
                                  task="classification") == {1: 0, 2: 1}
    assert build_label_dictionary(label_mapping_configuration=label_mapping_configuration,
                                  task="object_detection") == {1: 0, 2: 1, "background": 0}
    assert build_label_dictionary(label_mapping_configuration={"mode": "binary", "mapping_file_path": None},
                                  task="classification") is None


def test_class_folder_manifests(tmp_path: Path) -> None:
    """
    Verifies the conversion of a dataset stored as one folder per class: every split gets a 'data' link to its class
    folders and a manifest of its images, the categories (the class folders of all the splits, in alphabetical order)
    are the same for every split even when a split lacks a class, the label mapping gives the classes 0 to K-1, and
    the classification dataset reads the converted split.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Training split with two classes, validation split without circles
    for split_name, class_images in [("training_data", {"squares": 2, "circles": 1}),
                                     ("validation_data", {"squares": 1})]:
        for class_name, number_images in class_images.items():
            class_folder = tmp_path / split_name / class_name
            class_folder.mkdir(parents=True)
            for image_index in range(number_images):
                cv2.imwrite(filename=str(class_folder / f"image_{image_index}.png"),
                            img=np.zeros(shape=(16, 16, 3), dtype=np.uint8))
            (class_folder / "notes.txt").write_text(data="not an image")

    output_folder = tmp_path / "manifests"
    create_class_folder_manifests(split_folders={"train": str(tmp_path / "training_data"),
                                                 "validation": str(tmp_path / "validation_data")},
                                  output_folder=str(output_folder))

    train_labels = json.loads((output_folder / "train" / "labels.json").read_text())
    validation_labels = json.loads((output_folder / "validation" / "labels.json").read_text())
    assert train_labels["categories"] == validation_labels["categories"] == [{"id": 1, "name": "circles"},
                                                                             {"id": 2, "name": "squares"}]
    assert [sample["file_name"] for sample in train_labels["samples"]] == ["circles/image_0.png",
                                                                           "squares/image_0.png",
                                                                           "squares/image_1.png"]
    assert [sample["category_id"] for sample in validation_labels["samples"]] == [2]
    assert (output_folder / "train" / "data").is_symlink()
    assert json.loads((output_folder / "label_mapping.json").read_text()) == {"1": 0, "2": 1}

    # A second conversion replaces the links and manifests
    create_class_folder_manifests(split_folders={"train": str(tmp_path / "training_data")},
                                  output_folder=str(output_folder))

    dataset = ClassificationDataset(data_directory=str(output_folder / "train" / "data"),
                                    labels_file=str(output_folder / "train" / "labels.json"), image_size=16,
                                    keep_ratio=True, label_dictionary={1: 0, 2: 1}, output_mode="multi_class",
                                    shuffle=False)
    assert [int(dataset[sample_index]["labels"]) for sample_index in range(len(dataset))] == [0, 1, 1]


@pytest.mark.parametrize(argnames="square_crop", argvalues=[True, False])
def test_square_crop_moves_on_the_training_images_only(tmp_path: Path, square_crop: bool) -> None:
    """
    Verifies the square crop of the classification loaders: the 32 x 64 images, whose shorter side is already the
    input size, are cut to a 32 pixel square without any padding, at a random position along their width at every
    reading of a training image and at the centre for the validation and test images. Without the square crop, the
    same images are padded with black bands, which the crop removes.

    Args:
        tmp_path (Path): The pytest temporary directory.
        square_crop (bool): Whether 'Data.image_settings.square_crop' is on.
    """
    # One image per split whose column c holds the value c + 10, so that the first column of the input gives the
    # position of the square and a black pixel can only come from a padding
    categories = [{"id": 1, "name": "circle"}]
    column_image = np.tile(A=np.arange(start=10, stop=74, dtype=np.uint8)[None, :, None], reps=(32, 1, 3))
    for split_name in ["train", "validation", "test"]:
        write_split(split_folder=tmp_path / split_name, categories=categories,
                    samples=[{"id": 1, "file_name": "1.png", "category_id": 1}])
        cv2.imwrite(filename=str(tmp_path / split_name / "data" / "1.png"), img=column_image)
    experiment_configuration = {f"{split_name}_data_folder": str(tmp_path / split_name)
                                for split_name in ["train", "validation", "test"]}
    model_configuration = {"Data": {"image_settings": {"size": 32, "keep_ratio": True, "square_crop": square_crop},
                                    "label_dictionary": {1: 0},
                                    "loading": {"number_workers": 0, "prefetch_batches": 2},
                                    "grouped_batches": {"activated": False, "block_key": "camera",
                                                        "images_per_block": 2, "batch_key": None}},
                           "ClassificationHead": {"output_mode": "multi_class"}}
    loaders = get_classification_dataloaders(experiment_configuration=experiment_configuration,
                                             model_configuration=model_configuration, batch_size=1)

    # Read every image 300 times, keeping the first column of the input and whether it holds any padding
    first_columns, padded_images = [], []
    for loader in loaders:
        images = [loader.dataset[0]["images"] for _ in range(300)]
        first_columns.append({int(image[0, 0, 0]) for image in images})
        padded_images.append(any(bool((image == 0).all(dim=0).any()) for image in images))

    if square_crop:
        # The 33 positions 0 to 32 of the square for the training image, the centre 16 for the other splits
        assert first_columns == [set(range(10, 43)), {26}, {26}]
        assert padded_images == [False, False, False]
    else:
        # The letterbox of the whole image, the same at every reading
        assert padded_images == [True, True, True]
