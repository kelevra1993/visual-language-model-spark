"""
Tests of scripts/deployment/check_half_precision_parity.py, the trial of an FP16 TorchScript classifier against the
eager FP32 one: the image comparison counts the predicted classes and the deployed decisions that moved and the
probability differences, the table says when no threshold is deployed, the deployed thresholds are read in the class
order of the run (a file of other classes refused), and the half precision classifier runs its network in FP16 while its temperature and probabilities stay in
FP32, close to the eager classifier, traced into a TorchScript file that loads and runs.
"""

import json
import numpy as np
import pytest
import torch

from pathlib import Path

from scripts.deployment.check_half_precision_parity import (HalfPrecisionClassifier, build_parity_table,
                                                            compare_half_precision_outputs, read_deployed_thresholds)
from tests.deployment.test_export_specifications import build_test_model
from vision_training.deployment.export_formats import export_torchscript
from vision_training.deployment.export_specifications import build_inference_model


def test_comparison_counts_the_moved_predictions_and_decisions() -> None:
    """
    Verifies that the comparison counts an image whose predicted class changes, an image whose deployed decision
    changes for one class without its predicted class changing, and gives the largest and mean probability differences.
    """
    eager_probabilities = np.array([[0.95, 0.05], [0.40, 0.60], [0.10, 0.90], [0.80, 0.20]], dtype=np.float32)
    half_probabilities = np.array([[0.95, 0.05], [0.55, 0.45], [0.10, 0.90], [0.70, 0.30]], dtype=np.float32)
    comparison = compare_half_precision_outputs(eager_probabilities=eager_probabilities,
                                                half_probabilities=half_probabilities,
                                                thresholds=np.array([0.75, 0.5]))
    assert comparison["compared_images"] == 4 and comparison["predicted_class_changes"] == 1
    assert comparison["decision_changes"] == 2 and comparison["decision_changes_per_class"] == [1, 1]
    assert comparison["largest_difference"] == pytest.approx(0.15, abs=1e-6)
    assert comparison["mean_difference"] == pytest.approx((0.15 * 2 + 0.1 * 2) / 8, abs=1e-6)
    with pytest.raises(expected_exception=ValueError, match="of the same shape"):
        compare_half_precision_outputs(eager_probabilities=eager_probabilities,
                                       half_probabilities=half_probabilities[:2], thresholds=np.array([0.75, 0.5]))


def test_table_without_deployed_thresholds() -> None:
    """
    Verifies that without deployed thresholds (use_saved_temperature off) the comparison counts no decision and the
    table says that no threshold is deployed, the metrics of both rows and their change in points still given.
    """
    comparison = compare_half_precision_outputs(eager_probabilities=np.array([[0.7, 0.3]], dtype=np.float32),
                                                half_probabilities=np.array([[0.69, 0.31]], dtype=np.float32),
                                                thresholds=None)
    assert comparison["decision_changes"] is None and comparison["decision_changes_per_class"] is None
    rows = [{"scalars": {"Balanced Accuracy": balanced_accuracy, "Accuracy": 0.9, "Macro F1": 0.8},
             "evaluated_samples": "1 Test Images"} for balanced_accuracy in [0.95, 0.9448]]
    table = build_parity_table(summary={
        "run": "/runs/CLS_TEST_MC2", "checkpoint": "/runs/CLS_TEST_MC2/Weights/Iteration_10/model_000010.pt",
        "iteration": 10, "weights": "trained weights", "test_data_folder": "/data/test", "logit_temperature": 1.0,
        "thresholds_path": None, "random_batch_check": [], "rows": rows, "comparison": comparison,
        "agreement": {"within_tolerance": 1, "compared_samples": 1}, "file_sizes": {"fp32": 2e6, "fp16": 1e6}})
    assert "thresholds not deployed" in table and "| Deployed decision changed | | no threshold deployed | |" in table
    assert "| Balanced Accuracy | 95.00 % | 94.48 % | -0.52 points |" in table
    assert "| TorchScript file | 2.0 MB | 1.0 MB | |" in table


def test_deployed_thresholds_follow_the_class_order(tmp_path: Path) -> None:
    """
    Verifies that the deployed thresholds are read in the class order of the run, whatever the order of the file, and
    that a file of other classes is refused.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    thresholds_path = tmp_path / "saved_temperature_thresholds.json"
    thresholds_path.write_text(data=json.dumps(obj={"logit_temperature": 3.0, "target_precision": 0.99, "classes": {
        "no_event": {"threshold": 0.93, "recall": 0.67}, "event": {"threshold": 0.9, "recall": 0.87}}}))
    assert read_deployed_thresholds(thresholds_path=thresholds_path,
                                    class_names=["event", "no_event"]).tolist() == [0.9, 0.93]
    with pytest.raises(expected_exception=ValueError, match="Expected the thresholds of the classes"):
        read_deployed_thresholds(thresholds_path=thresholds_path, class_names=["event", "other"])


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The FP16 network runs on the GPU.")
def test_half_precision_classifier_keeps_float32_probabilities(tmp_path: Path) -> None:
    """
    Verifies that the half precision classifier runs its network in FP16, keeps the logit temperature of the eager
    classifier in FP32 and returns FP32 probabilities close to the eager ones, and that its traced TorchScript file
    loads and gives the same probabilities.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    inference_model = build_inference_model(task="classification", model=build_test_model(
        task="classification", temporal_type=None, fusion_type=None, global_pooling="average").cuda())
    inference_model.set_logit_temperature(logit_temperature=3.0)
    half_model = HalfPrecisionClassifier(inference_model=inference_model).eval()
    assert all(parameter.dtype == torch.float16 for parameter in half_model.model.parameters())
    assert half_model.logit_temperature.dtype == torch.float32 and half_model.logit_temperature.item() == 3.0

    # Probabilities of a random batch of the input contract, FP32 and close to the eager ones
    images = torch.rand(size=(4, 3, inference_model.input_image_size, inference_model.input_image_size),
                        generator=torch.Generator().manual_seed(0)).cuda() * 255
    with torch.no_grad():
        eager_probabilities, half_probabilities = inference_model(images=images), half_model(images=images)
    assert half_probabilities.dtype == torch.float32
    assert torch.allclose(input=half_probabilities, other=eager_probabilities, atol=1e-2)

    # The traced file loads and gives the probabilities of the half precision classifier
    export_torchscript(inference_model=half_model, example_inputs={"images": images}, output_path=tmp_path / "model.pt")
    with torch.no_grad():
        traced_probabilities = torch.jit.load(str(tmp_path / "model.pt"), map_location="cuda")(images=images)
    assert torch.allclose(input=traced_probabilities, other=half_probabilities, atol=1e-6)
