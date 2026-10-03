"""
Tests of the loader benchmark of scripts/data/loader_benchmark (implementation_dali_loaders.md, phase 1): a smoke run
of a classification case and of a sequence comparison case on tiny generated JPEG datasets, with every stage and one
variant of each loader family (measurement rows, parity of the DALI prototypes with the Native datasets, reports), and
the frame slots and label tables of the DALI sequence loader against the Native windows and pairs.
"""

import json
import cv2
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, List

from scripts.data.loader_benchmark.benchmark_datasets import (create_imagenet_sequence_dataset,
                                                              create_sequence_pair_dataset)
from scripts.data.loader_benchmark.benchmark_measurements import write_benchmark_reports
from scripts.data.loader_benchmark.dali_prototypes import build_sample_frame_slots, build_sequence_label_table
from scripts.data.loader_benchmark.run_loader_benchmark import CONFIGURATIONS_FOLDER, run_case_in_process
from vision_training.data.label_mapping import write_label_mapping
from vision_training.data.sequence.sequence_dataset import SequenceDataset
from vision_training.data.sequence.sequence_pair_dataset import SequencePairDataset

# Categories of the generated classification images
TEST_CATEGORIES = [{"id": 1, "name": "first_category"}, {"id": 2, "name": "second_category"}]

# Sizes (height, width) of the generated images, landscape, portrait and square, with odd paddings once resized
IMAGE_SIZES = [(90, 120), (120, 90), (100, 100), (75, 133), (133, 75)]

# Number of images of every classification split, and of sequences and pairs of every split: the training splits
# span several batches, so that the measurements of the smoke run stay within their first epoch
CLASSIFICATION_SPLIT_SIZES = {"train": 200, "validation": 20, "test": 20}
SEQUENCE_SPLIT_SIZES = {"train": 40, "validation": 4, "test": 4}

# Tiny benchmark settings: one repeat, a few batches, and one variant of each loader family
TEST_SETTINGS = {"number_repeats": 1, "number_warmup_batches": 1, "number_measured_batches": 2,
                 "number_ceiling_warmup_iterations": 1, "number_ceiling_iterations": 2,
                 "number_end_to_end_iterations": 2, "number_evaluation_batches": 2, "number_parity_samples": 8,
                 "native_worker_counts": [1], "native_prefetch_factors": [2], "dali_thread_counts": [2],
                 "dali_reader_process_counts": [1], "dali_prefetch_queue_depth": 2}


def write_classification_split(split_folder: Path, number_images: int, random_generator: np.random.Generator) -> None:
    """
    Writes a classification split of smooth JPEG images (a gradient and a disc, whose JPEG decodings by the GPU and
    by OpenCV stay close) of several sizes and aspect ratios, with its manifest.

    Args:
        split_folder (Path): The folder of the split.
        number_images (int): The number of images.
        random_generator (np.random.Generator): The generator of the colours.
    """
    (split_folder / "data").mkdir(parents=True)
    samples = []
    for image_index in range(number_images):
        height, width = IMAGE_SIZES[image_index % len(IMAGE_SIZES)]

        # Horizontal gradient with a disc of a random colour
        gradient = np.linspace(start=40, stop=200, num=width, dtype=np.float32)[None, :, None]
        image = np.repeat(a=np.repeat(a=gradient, repeats=height, axis=0), repeats=3, axis=2).astype(np.uint8)
        colour = tuple(int(value) for value in random_generator.integers(low=0, high=256, size=3))
        cv2.circle(img=image, center=(width // 2, height // 2), radius=min(height, width) // 3, color=colour,
                   thickness=-1)

        file_name = f"image_{image_index:04d}.jpg"
        cv2.imwrite(filename=str(split_folder / "data" / file_name), img=image,
                    params=[cv2.IMWRITE_JPEG_QUALITY, 95])
        samples.append({"id": image_index + 1, "file_name": file_name, "category_id": 1 + image_index % 2})
    (split_folder / "labels.json").write_text(data=json.dumps(obj={"categories": TEST_CATEGORIES,
                                                                   "samples": samples}))


def create_test_datasets(dataset_root: Path) -> None:
    """
    Generates the tiny datasets of the tests: a classification dataset, then a sequence dataset and a pair dataset
    built from its training images by the dataset builders of the benchmark ('classification', 'sequences' and 'pairs'
    under dataset_root, with their label mappings).

    Args:
        dataset_root (Path): The folder of the datasets.
    """
    random_generator = np.random.default_rng(seed=0)
    for split_name, number_images in CLASSIFICATION_SPLIT_SIZES.items():
        write_classification_split(split_folder=dataset_root / "classification" / split_name,
                                   number_images=number_images, random_generator=random_generator)
    write_label_mapping(categories=TEST_CATEGORIES, output_folder=dataset_root / "classification")
    create_imagenet_sequence_dataset(imagenet_split_folder=dataset_root / "classification" / "train",
                                     output_folder=dataset_root / "sequences", split_sizes=SEQUENCE_SPLIT_SIZES,
                                     minimum_length=2, maximum_length=4, seed=0)
    create_sequence_pair_dataset(sequence_dataset_folder=dataset_root / "sequences",
                                 output_folder=dataset_root / "pairs", split_sizes=SEQUENCE_SPLIT_SIZES, seed=0)


def build_test_case(template: Path, dataset_folder: Path, cold_cache: bool) -> Dict[str, Any]:
    """
    Builds the definition of a benchmark case on a generated dataset, in the format of build_benchmark_cases.

    Args:
        template (Path): The YAML of the repository giving the task and the model.
        dataset_folder (Path): The root folder of the generated dataset (train, validation and test splits).
        cold_cache (bool): Whether the loaders are also measured with a cold page cache.

    Returns:
        Dict[str, Any]: The definition of the case.
    """
    return {"template": template,
            "split_folders": {f"{split_name}_data_folder": dataset_folder / split_name
                              for split_name in ["train", "validation", "test"]},
            "label_mapping_path": dataset_folder / "label_mapping.json", "cold_cache": cold_cache,
            "frame_label_parity": False}


def check_case_rows(rows: List[Dict[str, Any]], cold_cache: bool) -> None:
    """
    Verifies the rows of a case: every stage of every loader family measured with positive throughputs, the cold
    measurements when asked, and DALI samples identical to the Native ones in labels and masks with close pixels.

    Args:
        rows (List[Dict[str, Any]]): The rows of the case.
        cold_cache (bool): Whether the case was also measured with a cold page cache.
    """
    measured = {(row["stage"], row["variant"], row["cache"]) for row in rows}
    for variant in ["native_current", "native_tuned", "dali"]:
        for stage in ["A", "C", "D"]:
            assert (stage, variant, "warm") in measured
        assert (("A", variant, "cold") in measured) == cold_cache
    assert ("B", "gpu", "") in measured
    for row in rows:
        if row["stage"] in ["A", "B", "C", "D"]:
            assert row["samples_per_second"] > 0

    # DALI samples equal to the Native ones
    parity_rows = [row for row in rows if row["stage"] == "E"]
    assert len(parity_rows) == 1
    assert parity_rows[0]["compared_samples"] > 0
    assert parity_rows[0]["label_mismatches"] == 0
    assert parity_rows[0]["mask_mismatches"] == 0
    assert parity_rows[0]["mean_pixel_difference"] < 2.0


def test_loader_benchmark_cases(tmp_path: Path) -> None:
    """
    Verifies that the benchmark measures a classification case (also with a cold page cache) and a sequence
    comparison case in their own processes, with every stage and loader family and DALI outputs equal to the Native
    ones, and writes its reports.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_test_datasets(dataset_root=tmp_path)
    run_folder = tmp_path / "run"
    cases = {"C-TEST": build_test_case(template=CONFIGURATIONS_FOLDER / "classification" /
                                       "configuration_synthetic_shapes.yaml",
                                       dataset_folder=tmp_path / "classification", cold_cache=True),
             "P-TEST": build_test_case(template=CONFIGURATIONS_FOLDER / "sequence_comparison" /
                                       "configuration_synthetic_pairs.yaml", dataset_folder=tmp_path / "pairs",
                                       cold_cache=False)}

    # Every case in its own process
    rows = []
    for case_name, case_definition in cases.items():
        case_rows = run_case_in_process(case_name=case_name, case_definition=case_definition, settings=TEST_SETTINGS,
                                        case_output_folder=run_folder / case_name)
        check_case_rows(rows=case_rows, cold_cache=case_definition["cold_cache"])
        rows += case_rows

    # Parity images and reports
    assert len(list((run_folder / "Parity").glob(pattern="*.png"))) == len(cases)
    results_path, summary_path = write_benchmark_reports(rows=rows, output_folder=run_folder,
                                                         environment_lines=["- Test run"])
    assert results_path.read_text().count("\n") == len(rows) + 1
    summary = summary_path.read_text()
    for case_name in cases:
        assert f"## {case_name}" in summary


def test_sequence_frame_slots_and_labels(tmp_path: Path) -> None:
    """
    Verifies that the frame slots and label tables of the DALI sequence loader follow the Native samples: real frames
    first then padding slots in every window (each member window of a pair), and the window classes, the frame
    classes with the ignored index on the padding frames, and the pair classes.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    create_test_datasets(dataset_root=tmp_path)
    sequence_settings = {"sequence_length": 4, "frame_stride": 1, "window_stride": 4, "short_sequences": "pad"}
    datasets = [SequenceDataset(data_directory=str(tmp_path / "sequences" / "train" / "data"),
                                labels_file=str(tmp_path / "sequences" / "train" / "labels.json"), image_size=32,
                                keep_ratio=True, label_dictionary={1: 0, 2: 1}, sequence_settings=sequence_settings,
                                label_type=label_type) for label_type in ["sequence", "frame"]]
    datasets.append(SequencePairDataset(data_directory=str(tmp_path / "pairs" / "train" / "data"),
                                        labels_file=str(tmp_path / "pairs" / "train" / "labels.json"), image_size=32,
                                        keep_ratio=True, label_dictionary={1: 0, 2: 1},
                                        sequence_settings=sequence_settings))

    # Every sample of every dataset, its slots giving its mask and its table row giving its label
    for dataset in datasets:
        sample_slots = build_sample_frame_slots(dataset=dataset)
        label_table = build_sequence_label_table(dataset=dataset)
        for position in range(len(dataset)):
            native_sample = dataset[position]
            slot_mask = torch.tensor(data=[frame_path is not None for frame_path in sample_slots[position]])
            assert torch.equal(input=slot_mask, other=native_sample["sequence_mask"])
            assert torch.equal(input=label_table[position], other=native_sample["labels"])
