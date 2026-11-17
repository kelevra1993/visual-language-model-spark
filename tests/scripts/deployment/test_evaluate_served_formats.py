"""
Tests of scripts/deployment/evaluate_served_formats.py, the test metric of the served formats of the serving benchmark:
the test batches reach the model in requests no larger than the served models take, the frames of every sequence go
alone and without their padding, the outputs of a served detector become the predictions its trainer scores, and
test_metrics.csv keeps every evaluated row so that a relaunch skips it.
"""

import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Dict, List

from scripts.deployment.evaluate_served_formats import (METRIC_COLUMNS, append_row_metrics, convert_served_outputs,
                                                        predict_served_batch, read_measured_rows)

# Device of the trainer in the tests, the conversion being the same on the GPU
DEVICE = torch.device("cpu")


def run_fake_classifier(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Stands for a served classifier: the probability of class 0 of every image is its mean value, and the one of class
    1 the number of images of the request, which shows how the batch was split.

    Args:
        inputs (Dict[str, np.ndarray]): The [B, 3, H, W] 'images' of the request.

    Returns:
        List[np.ndarray]: The [B, 2] class probabilities.
    """
    images = inputs["images"]
    image_means = images.reshape(images.shape[0], -1).mean(axis=1)
    return [np.stack([image_means, np.full(shape=images.shape[0], fill_value=images.shape[0])], axis=1)
            .astype(np.float32)]


def run_fake_sequence_labeller(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Stands for a served sequence labeller: the probability of class 0 of every frame is its mean value, and the one of
    class 1 the number of frames of the request, which shows whether the padding was sent.

    Args:
        inputs (Dict[str, np.ndarray]): The [B, T, 3, H, W] 'images' of the request.

    Returns:
        List[np.ndarray]: The [B, T, 2] frame probabilities.
    """
    images = inputs["images"]
    frame_means = images.reshape(images.shape[0], images.shape[1], -1).mean(axis=2)
    return [np.stack([frame_means, np.full(shape=frame_means.shape, fill_value=images.shape[1])], axis=2)
            .astype(np.float32)]


def test_images_are_sent_in_requests_of_the_largest_served_batch() -> None:
    """
    A test batch of 5 images reaches a model taking at most 2 images per request in requests of 2, 2 and 1 images,
    and the probabilities come back in the order of the batch.
    """
    images = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1).expand(5, 3, 2, 2).contiguous()
    predictions = predict_served_batch(batch={"images": images}, task="classification",
                                       run_model=run_fake_classifier, output_names=["class_probabilities"],
                                       maximum_request_size=2, device=DEVICE)
    probabilities = predictions["class_probabilities"]
    assert probabilities[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert probabilities[:, 1].tolist() == [2.0, 2.0, 2.0, 2.0, 1.0]


def test_sequences_are_sent_alone_without_their_padding() -> None:
    """
    Every sequence of a padded batch is sent alone with its real frames only, and the probabilities of its padding
    frames stay at 0 for the trainer, which ignores the labels of the padding.
    """
    images = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1, 1).expand(2, 4, 3, 2, 2).contiguous()
    sequence_mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    predictions = predict_served_batch(batch={"images": images, "sequence_mask": sequence_mask},
                                       task="sequence_labelling", run_model=run_fake_sequence_labeller,
                                       output_names=["class_probabilities"], maximum_request_size=16, device=DEVICE)
    probabilities = predictions["class_probabilities"]
    assert probabilities.shape == (2, 4, 2)
    assert probabilities[0, :, 0].tolist() == [0.0, 1.0, 2.0, 0.0]
    assert probabilities[0, :, 1].tolist() == [3.0, 3.0, 3.0, 0.0]
    assert probabilities[1, :, 0].tolist() == [4.0, 0.0, 0.0, 0.0]
    assert probabilities[1, :, 1].tolist() == [1.0, 0.0, 0.0, 0.0]

    # Padding before a real frame cannot be removed by keeping the first frames
    with pytest.raises(ValueError, match="real frames of sequence 1 first"):
        predict_served_batch(batch={"images": images,
                                    "sequence_mask": torch.tensor([[True, True, True, False],
                                                                   [False, True, False, False]])},
                             task="sequence_labelling", run_model=run_fake_sequence_labeller,
                             output_names=["class_probabilities"], maximum_request_size=16, device=DEVICE)


def test_served_detections_become_the_predictions_of_the_trainer() -> None:
    """
    The detections of a served detector become the final detections the detection trainer scores, the number of
    detections of every image (a scalar per image in the served model) as a [B] vector, and the proposals, which the
    served detector does not return, as empty tensors.
    """
    served_outputs = {"detection_boxes": np.zeros(shape=(2, 100, 4), dtype=np.float32),
                      "detection_scores": np.zeros(shape=(2, 100), dtype=np.float32),
                      "detection_labels": np.zeros(shape=(2, 100), dtype=np.int64),
                      "number_detections": np.array([3, 0], dtype=np.int64)}
    predictions = convert_served_outputs(task="object_detection", served_outputs=served_outputs, device=DEVICE)
    assert predictions["final_detection_boxes"].shape == (2, 100, 4)
    assert predictions["number_detections"].tolist() == [3, 0]
    assert predictions["proposal_boxes"].shape == (2, 0, 4)
    assert predictions["valid_proposal_mask"].dtype == torch.bool
    with pytest.raises(ValueError, match="Expected classification"):
        convert_served_outputs(task="keypoint_detection", served_outputs=served_outputs, device=DEVICE)


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
