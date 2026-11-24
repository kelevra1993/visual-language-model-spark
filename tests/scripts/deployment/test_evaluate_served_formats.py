"""
Tests of scripts/deployment/evaluate_served_formats.py, the test metric of the served formats of the serving benchmark:
test_metrics.csv keeps every evaluated row so that a relaunch skips it. The requests and conversions of its rows are
those of the test split parity of the export, tested in tests/deployment/test_export_parity.py.
"""

from pathlib import Path

from scripts.deployment.evaluate_served_formats import METRIC_COLUMNS, append_row_metrics, read_measured_rows


def test_metrics_file_gives_back_the_evaluated_rows(tmp_path: Path) -> None:
    """
    Every scalar of an evaluated row is one line of test_metrics.csv, the headline metric flagged, and the row is
    found again at a relaunch.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    metrics_path = tmp_path / "test_metrics.csv"
    assert read_measured_rows(metrics_path=metrics_path) == set()
    results = {"scalars": {"Accuracy": 0.95, "Balanced Accuracy": 0.94}, "headline_metric": "Accuracy",
               "evaluated_samples": "50000 Test Images"}
    append_row_metrics(metrics_path=metrics_path, model_name="classification", row_name="triton_tensorrt_fp16",
                       results=results, duration_seconds=12.0)
    lines = metrics_path.read_text().splitlines()
    assert lines[0].split(",") == METRIC_COLUMNS
    assert len(lines) == 3
    assert lines[1].startswith("classification,triton_tensorrt_fp16,Accuracy,0.95,True,")
    assert read_measured_rows(metrics_path=metrics_path) == {("classification", "triton_tensorrt_fp16")}
