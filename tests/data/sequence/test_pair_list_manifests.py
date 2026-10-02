import json
import cv2
import numpy as np

from pathlib import Path
from typing import List

from vision_training.data.sequence.pair_list_manifests import create_pair_list_manifests
from vision_training.data.sequence.sequence_pair_dataset import SequencePairDataset


def write_member_frames(frame_folder: Path, frame_names: List[str]) -> None:
    """
    Writes black 16 pixel frames with the given names in a folder of frames of the source dataset.

    Args:
        frame_folder (Path): The folder of the frames of one sequence.
        frame_names (List[str]): The file names of the frames, in the order they are written.
    """
    frame_folder.mkdir(parents=True)
    for frame_name in frame_names:
        cv2.imwrite(filename=str(frame_folder / frame_name), img=np.zeros(shape=(16, 16, 3), dtype=np.uint8))


def test_pair_lists_are_converted_to_manifests(tmp_path: Path) -> None:
    """
    Verifies the conversion of a dataset described by pair list files: categories shared by every split and numbered
    in alphabetical order, the frames of every member listed by file name relative to the linked source folder, the
    label mapping, the outputs replaced when converting again, and a converted split read by the sequence pair
    dataset.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    # Source folder holding one folder of frames per sequence, the frames written out of order
    source_folder = tmp_path / "source"
    write_member_frames(frame_folder=source_folder / "visit_1", frame_names=["frame_b.png", "frame_a.png"])
    write_member_frames(frame_folder=source_folder / "visit_2", frame_names=["frame_a.png"])
    write_member_frames(frame_folder=source_folder / "visit_3", frame_names=["frame_a.png", "frame_c.png",
                                                                             "frame_b.png"])

    # One pair list file per split, the validation split naming a label of its own
    pair_list_files = {}
    for split_name, entries in [("train", [{"before": "visit_1", "after": "visit_2", "label": "same",
                                            "sequence": "pair_1"},
                                           {"before": "visit_2", "after": "visit_3", "label": "lost",
                                            "sequence": "pair_2"}]),
                                ("validation", [{"before": "visit_3", "after": "visit_1", "label": "moved",
                                                 "sequence": "pair_3"}])]:
        pair_list_path = tmp_path / f"{split_name}.json"
        pair_list_path.write_text(data=json.dumps(obj={"source": str(source_folder), "data": entries}))
        pair_list_files[split_name] = str(pair_list_path)

    # Converting twice replaces the outputs of the first conversion
    output_folder = tmp_path / "manifests"
    for _ in range(2):
        create_pair_list_manifests(pair_list_files=pair_list_files, output_folder=str(output_folder))

    train_labels = json.loads((output_folder / "train" / "labels.json").read_text())
    assert train_labels["categories"] == [{"id": 1, "name": "lost"}, {"id": 2, "name": "moved"},
                                          {"id": 3, "name": "same"}]
    assert train_labels["pairs"][0] == {"id": 1, "name": "pair_1",
                                        "before": ["visit_1/frame_a.png", "visit_1/frame_b.png"],
                                        "after": ["visit_2/frame_a.png"], "category_id": 3}
    assert train_labels["pairs"][1]["after"] == ["visit_3/frame_a.png", "visit_3/frame_b.png", "visit_3/frame_c.png"]
    assert (output_folder / "train" / "data").resolve() == source_folder.resolve()
    assert json.loads((output_folder / "label_mapping.json").read_text()) == {"1": 0, "2": 1, "3": 2}

    # The converted training split is read by the dataset of the comparison task
    dataset = SequencePairDataset(data_directory=str(output_folder / "train" / "data"),
                                  labels_file=str(output_folder / "train" / "labels.json"), image_size=16,
                                  keep_ratio=True, label_dictionary={1: 0, 2: 1, 3: 2},
                                  sequence_settings={"sequence_length": 2, "frame_stride": 1,
                                                     "short_sequences": "pad"}, shuffle=False)
    assert len(dataset) == 2 and dataset[0]["sequence_mask"].tolist() == [True, True, True, False]
