"""
Tests of vision_training/evaluation/detection/anchor_recall_figure.py, the anchor_recall.png figure written at the
first launch of a detection run: the tables of the training and test splits drawn one under the other into a readable
PNG image, and an empty list of tables refused.
"""

import numpy as np
import pytest
import torch

from pathlib import Path
from PIL import Image

from vision_training.evaluation.detection.anchor_recall_figure import draw_anchor_recall_figure
from vision_training.evaluation.detection.region_recall import summarise_anchor_recall


def test_figure_stacks_the_tables_of_every_split(tmp_path: Path) -> None:
    """
    Verifies that the anchor recall tables of two splits are drawn into one PNG image taller than wide, holding red
    and green shaded cells (the recalls of 0 and 1), and that drawing without any table is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    tables = [summarise_anchor_recall(
        best_anchor_overlaps={3: torch.tensor(data=[0.8, 0.2, 0.9]), 4: torch.tensor(data=[0.1, 0.1, 0.75])},
        object_areas=torch.tensor(data=[100.0, 2500.0, 40000.0]), image_side=512, intersection_over_union_threshold=0.7,
        split_name=split_name, threshold_setting="RegionProposal.foreground_iou_threshold.min")["table"]
        for split_name in ["Training", "Test"]]
    figure_path = draw_anchor_recall_figure(tables=tables, output_path=tmp_path / "run" / "anchor_recall.png")
    assert figure_path == tmp_path / "run" / "anchor_recall.png"

    # One image holding both tables, its shaded cells red for the missed sizes and green for the covered ones
    with Image.open(fp=figure_path) as image:
        assert image.format == "PNG" and image.height > image.width > 1000
        red, green, blue = np.moveaxis(np.asarray(image.convert(mode="RGB"), dtype=np.int64), source=-1,
                                       destination=0)
        assert bool(((red > 200) & (green < 170) & (blue < 170)).any())
        assert bool(((green > 180) & (red < 190) & (blue < 170)).any())

    with pytest.raises(ValueError, match="at least one anchor recall table"):
        draw_anchor_recall_figure(tables=[], output_path=tmp_path / "empty.png")
