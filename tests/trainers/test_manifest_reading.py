"""
Tests of the manifests parsed once per launch (vision_training/data/manifest_cache.py) for the trainer of every task of
vision_training/trainers: built within read_manifests_once from the smoke configuration of its task test, the trainer
parses the 'labels.json' of each of its three splits once, although its datasets, the record of its data, the
description of its splits and the task readers (test manifest, keypoint definitions) all read them.
"""

import json
import pytest

from collections import Counter
from pathlib import Path
from typing import Any, Dict

from tests.trainers.test_optional_validation_loader import TASK_SMOKE_SETTINGS
from vision_training.data.manifest_cache import read_manifests_once
from vision_training.data.synthetic.create_synthetic_dataset import create_synthetic_dataset


@pytest.mark.parametrize(argnames="task_name", argvalues=list(TASK_SMOKE_SETTINGS))
def test_launch_parses_every_manifest_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_name: str) -> None:
    """
    Verifies, for the trainer of every task, that its construction within read_manifests_once parses the 'labels.json'
    of each of its three splits once, every JSON file parsed by the framework being followed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture following the JSON parser.
        task_name (str): The task whose trainer is built, a key of TASK_SMOKE_SETTINGS.
    """
    task_settings: Dict[str, Any] = TASK_SMOKE_SETTINGS[task_name]
    smoke_module = task_settings["module"]

    # Small generated dataset of the task and its smoke configuration, with validation
    dataset_folder = tmp_path / "synthetic_dataset"
    create_synthetic_dataset(output_folder=str(dataset_folder), image_size=smoke_module.SMOKE_IMAGE_SIZE,
                             number_workers=1, **task_settings["dataset_settings"])
    configuration_path = smoke_module.write_smoke_configuration(temporary_directory=tmp_path,
                                                                dataset_folder=dataset_folder, number_iterations=2,
                                                                **task_settings["configuration_settings"])

    # Every manifest parsed during the launch, whatever reads it
    parsed_manifests = []
    original_load = json.load
    monkeypatch.setattr(target=json, name="load", value=lambda fp, **keyword_arguments: (
        parsed_manifests.append(Path(fp.name).resolve()) if Path(fp.name).name == "labels.json" else None)
        or original_load(fp=fp, **keyword_arguments))
    with read_manifests_once():
        smoke_module.build_trainer(configuration_path=configuration_path)

    # The manifests of the three splits, each parsed once
    assert sorted(path.parent.name for path in parsed_manifests) == ["test", "train", "validation"]
    assert set(Counter(parsed_manifests).values()) == {1}
