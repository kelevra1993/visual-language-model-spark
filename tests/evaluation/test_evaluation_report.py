"""
Tests of vision_training/evaluation/evaluation_report.py: the report of an evaluated checkpoint writes the global
scalars, and the per-class scalars and images only with the per-class report, appends one block per evaluation to
Results.txt, and the per-class metrics file is written in class index order with its values formatted.
"""

import csv
import numpy as np
import pytest

from pathlib import Path
from typing import Any, Dict, List, Tuple

from vision_training.core.contracts import EvaluationResults
from vision_training.evaluation.evaluation_report import report_evaluation_results, write_class_metrics_file
from tests.fixtures.evaluation_checks import read_results_blocks


class RecordingWriter:
    """
    Stand-in of the Test TensorBoard writer recording the scalars and images it receives.
    """

    def __init__(self) -> None:
        """
        Starts with no scalar nor image.
        """
        self.scalars: List[Tuple[str, float, int]] = []
        self.images: List[Tuple[str, Tuple[int, ...], int, str]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """
        Records a scalar.

        Args:
            tag (str): The name of the scalar.
            scalar_value (float): Its value.
            global_step (int): The iteration it belongs to.
        """
        self.scalars.append((tag, scalar_value, global_step))

    def add_image(self, tag: str, img_tensor: np.ndarray, global_step: int, dataformats: str) -> None:
        """
        Records an image by its shape.

        Args:
            tag (str): The name of the image.
            img_tensor (np.ndarray): The image.
            global_step (int): The iteration it belongs to.
            dataformats (str): The layout of the image dimensions.
        """
        self.images.append((tag, img_tensor.shape, global_step, dataformats))

    def flush(self) -> None:
        """
        Does nothing, nothing is buffered.
        """


def build_results() -> EvaluationResults:
    """
    Builds the evaluation results of the tests: two global scalars, two per-class scalars, a per-class table and a
    confusion matrix image.

    Returns:
        EvaluationResults: The results.
    """
    return {"evaluated_samples": "6 Test Images", "metrics": {"Accuracy": 0.5, "Macro F1": 0.4}, "notes": [],
            "tables": [{"title": "Per Class Metrics", "columns": ["Class", "Samples", "F1"],
                        "rows": [["circle", 3, 0.8], ["square", 3, 0.0]], "sort_column": "F1", "per_class": True,
                        "legend": ""}],
            "scalars": {"Accuracy": 0.5, "Macro F1": 0.4}, "class_scalars": {"F1/circle": 0.8, "F1/square": 0.0},
            "images": {"Confusion Matrix": np.zeros(shape=(8, 10, 3), dtype=np.uint8)}, "files": {},
            "headline_metric": "Accuracy"}


@pytest.mark.parametrize(argnames="per_class_report", argvalues=[True, False])
def test_report_follows_the_per_class_report(tmp_path: Path, per_class_report: bool) -> None:
    """
    Verifies that the report writes the global scalars at every evaluation, and the per-class scalars, images and
    table only with the per-class report, and that every evaluation appends its own block to Results.txt.

    Args:
        tmp_path (Path): The pytest temporary directory.
        per_class_report (bool): Whether the per-class outputs are reported.
    """
    writer = RecordingWriter()
    results_path = tmp_path / "Results.txt"
    for iteration in [4, 8]:
        report_evaluation_results(results=build_results(), iteration=iteration, test_writer=writer,
                                  results_path=results_path, per_class_report=per_class_report)

    expected_tags = ["Accuracy", "Macro F1"] + (["F1/circle", "F1/square"] if per_class_report else [])
    assert [tag for tag, _, global_step in writer.scalars if global_step == 4] == expected_tags
    assert writer.images == ([("Confusion Matrix", (8, 10, 3), 4, "HWC"), ("Confusion Matrix", (8, 10, 3), 8, "HWC")]
                             if per_class_report else [])

    results_blocks = read_results_blocks(results_path=results_path)
    assert [block["iteration"] for block in results_blocks] == [4, 8]
    assert all(("Per Class Metrics" in block["text"]) == per_class_report for block in results_blocks)


def test_class_metrics_file(tmp_path: Path) -> None:
    """
    Verifies the per-class metrics file: one row per class in the given (class index) order, floats with 6 decimals and
    missing values as empty cells.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    class_rows: List[Dict[str, Any]] = [
        {"class_index": 0, "class": "circle", "samples": 3, "f1": 0.8, "threshold_at_target_precision": 0.912345678},
        {"class_index": 1, "class": "square", "samples": 0, "f1": 0.0, "threshold_at_target_precision": None}]
    class_metrics_path = write_class_metrics_file(class_rows=class_rows, evaluation_folder=tmp_path / "Iteration_4")

    assert class_metrics_path == tmp_path / "Iteration_4" / "class_metrics.csv"
    with open(file=class_metrics_path, mode="r", newline="") as class_metrics_file:
        rows = list(csv.reader(class_metrics_file))
    assert rows == [["class_index", "class", "samples", "f1", "threshold_at_target_precision"],
                    ["0", "circle", "3", "0.800000", "0.912346"], ["1", "square", "0", "0.000000", ""]]
