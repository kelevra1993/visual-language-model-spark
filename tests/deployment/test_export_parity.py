"""
Tests of the test split parity of an export (vision_training/deployment/export_parity.py and its use by
export_experiment): the test batches reach the model of a row as a deployed client sends them (requests no larger than
the exported batch size, the real frames of every sequence and of both members of every pair alone, the tiles of a
segmentation trained on tiles), the outputs become the predictions the trainer of every task scores, every request of a
served row is compared with the eager model sample by sample, the stored evaluation of the exported iteration is read,
the summary is written with the changes of every row against the eager row, and an export with the parity writes its
summary even when its random batch check fails, the check alone deciding the installation. An FP16 TorchScript row is
named with its precision and its random batch check with its eager FP16 reference. The Triton container is described
and started with its own name and ports (vision_training/deployment/triton_server.py).
"""

import csv
import functools
import json
import re
import numpy as np
import pytest
import torch

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from tests.deployment import test_input_contract
from tests.deployment.test_export_model import (FORCED_MISMATCH, ORIGINAL_GET_EXPORT_SPECIFICATION,
                                                get_mismatching_specification, read_repository_files)
from vision_training.deployment import export_model, triton_server
from vision_training.deployment.export_model import STAGING_FOLDER_PREFIX, export_experiment
from vision_training.deployment.export_parity import (EAGER_ROW_NAME, SAMPLES_FILE_NAME, SUMMARY_JSON_NAME,
                                                      SUMMARY_MARKDOWN_NAME, build_summary_markdown,
                                                      convert_inference_outputs,
                                                      create_output_comparison, encode_named_reference_images,
                                                      get_parity_folder, get_row_label, measure_sample_difference,
                                                      predict_exported_batch, read_reference_scalars,
                                                      summarise_sample_agreement, write_parity_summary)
from vision_training.deployment.export_specifications import (compare_detections, compare_probabilities,
                                                              compare_representations)
from vision_training.deployment.triton_server import (build_triton_server, check_triton_available, find_free_ports,
                                                      start_triton_server)
from vision_training.trainers.self_supervised_trainer import SelfSupervisedTrainer

# Device of the stand-in trainers, the conversion being the same on the GPU
DEVICE = torch.device("cpu")


class StandInTrainer:
    """
    Stand-in for the trainer of a run, holding what the parity reads from it: the task, the device, the tiling of a
    segmentation run and the grouping of the keypoint candidates, which it records instead of grouping them.
    """

    def __init__(self, task_name: str, tiling_configuration: Optional[Dict[str, Any]] = None,
                 input_image_size: int = 4) -> None:
        """
        Sets the task of the stand-in trainer.

        Args:
            task_name (str): The task of the run.
            tiling_configuration (Optional[Dict[str, Any]]): The 'Data.tiling' of a segmentation run, None otherwise.
                Defaults to None.
            input_image_size (int): The network input size, the side of a tile. Defaults to 4.
        """
        self.TASK_NAME = task_name
        self.device = DEVICE
        self.tiling_configuration = tiling_configuration
        self.input_image_size = input_image_size
        self.grouped_predictions = []

    def group_candidates(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Records the keypoint candidates given to the grouping and returns them with an empty grouping.

        Args:
            predictions (Dict[str, Any]): The keypoint candidates of a batch.

        Returns:
            Dict[str, Any]: The candidates, with no grouped object.
        """
        self.grouped_predictions.append(predictions)
        return {**predictions, "grouped_objects": []}


def run_fake_classifier(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Stands for an exported classifier: the probability of class 0 of every image is its mean value, and the one of
    class 1 the number of images of the request, which shows how the batch was split.

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
    Stands for an exported sequence labeller: the probability of class 0 of every frame is its mean value, and the one
    of class 1 the number of frames of the request, which shows whether the padding was sent.

    Args:
        inputs (Dict[str, np.ndarray]): The [B, T, 3, H, W] 'images' of the request.

    Returns:
        List[np.ndarray]: The [B, T, 2] frame probabilities.
    """
    images = inputs["images"]
    frame_means = images.reshape(images.shape[0], images.shape[1], -1).mean(axis=2)
    return [np.stack([frame_means, np.full(shape=frame_means.shape, fill_value=images.shape[1])], axis=2)
            .astype(np.float32)]


def run_fake_sequence_classifier(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Stands for an exported sequence classifier: the probability of class 0 of the sequence is the mean of its frames,
    and the one of class 1 its number of frames.

    Args:
        inputs (Dict[str, np.ndarray]): The [1, T, 3, H, W] 'images' of the request.

    Returns:
        List[np.ndarray]: The [1, 2] class probabilities.
    """
    images = inputs["images"]
    return [np.array([[images.mean(), images.shape[1]]], dtype=np.float32)]


def run_fake_comparator(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Stands for an exported sequence comparator: the probabilities of the pair are the numbers of frames of its before
    and after members, which shows whether their padding was sent.

    Args:
        inputs (Dict[str, np.ndarray]): The [1, T_before, 3, H, W] 'before_images' and [1, T_after, 3, H, W]
            'after_images' of the request.

    Returns:
        List[np.ndarray]: The [1, 2] class probabilities.
    """
    return [np.array([[inputs["before_images"].shape[1], inputs["after_images"].shape[1]]], dtype=np.float32)]


def run_fake_tile_segmenter(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Stands for an exported segmenter: every pixel of a tile gets the number of tiles of the request as the probability
    of class 1 and 0 for class 0, after the [N, S, S] class map.

    Args:
        inputs (Dict[str, np.ndarray]): The [N, 3, S, S] 'images' tiles of the request.

    Returns:
        List[np.ndarray]: The [N, S, S] class map and the [N, 2, S, S] class probabilities.
    """
    tiles = inputs["images"]
    probabilities = np.zeros(shape=(tiles.shape[0], 2) + tiles.shape[2:], dtype=np.float32)
    probabilities[:, 1] = tiles.shape[0]
    return [np.ones(shape=(tiles.shape[0],) + tiles.shape[2:], dtype=np.int64), probabilities]


def run_fake_detector(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Stands for an exported detector giving two detections per image, of scores 0.9 and 0.6 in this order.

    Args:
        inputs (Dict[str, np.ndarray]): The [B, 3, H, W] 'images' of the request.

    Returns:
        List[np.ndarray]: The [B, 2, 4] boxes, [B, 2] scores, [B, 2] labels and [B] numbers of detections.
    """
    number_images = inputs["images"].shape[0]
    boxes = np.tile(np.array([[[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 3.0, 3.0]]], dtype=np.float32), (number_images, 1, 1))
    scores = np.tile(np.array([[0.9, 0.6]], dtype=np.float32), (number_images, 1))
    labels = np.tile(np.array([[1, 2]], dtype=np.int64), (number_images, 1))
    return [boxes, scores, labels, np.full(shape=number_images, fill_value=2, dtype=np.int64)]


def run_fake_detector_swapped(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Stands for the export of run_fake_detector returning its two detections in the other order, as a rounding can
    order two tied detections, the score of the second one moved by 0.01 for the images of mean value above 1.

    Args:
        inputs (Dict[str, np.ndarray]): The [B, 3, H, W] 'images' of the request.

    Returns:
        List[np.ndarray]: The outputs of run_fake_detector, the detections in the other order.
    """
    boxes, scores, labels, numbers = run_fake_detector(inputs=inputs)
    scores = scores.copy()
    image_means = inputs["images"].reshape(inputs["images"].shape[0], -1).mean(axis=1)
    scores[image_means > 1.0, 1] += 0.01
    return [boxes[:, ::-1].copy(), scores[:, ::-1].copy(), labels[:, ::-1].copy(), numbers]


def run_fake_encoder(inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """
    Stands for an exported self-supervised encoder: the four values of the representation of every image are its mean
    value.

    Args:
        inputs (Dict[str, np.ndarray]): The [B, 3, H, W] 'images' of the request.

    Returns:
        List[np.ndarray]: The [B, 4] representations.
    """
    image_means = inputs["images"].reshape(inputs["images"].shape[0], -1).mean(axis=1)
    return [np.repeat(image_means[:, None], repeats=4, axis=1).astype(np.float32)]


def name_test_sample(sample_identifier: int) -> str:
    """
    Stands for get_test_sample_name of a trainer, naming a test sample after its identifier.

    Args:
        sample_identifier (int): The identifier of the sample in the batches.

    Returns:
        str: The name of the sample, e.g. 'image_20' for 20.
    """
    return f"image_{sample_identifier}"


class NamedSamples:
    """
    Stand-in for the dataset of a loader, naming its samples after a prefix of its own, so that a test tells the
    datasets of two loaders apart.
    """

    def __init__(self, prefix: str) -> None:
        """
        Sets the prefix of the sample names.

        Args:
            prefix (str): The prefix of every sample name.
        """
        self.prefix = prefix

    def get_sample_name(self, sample_identifier: int) -> str:
        """
        Names a sample of the dataset.

        Args:
            sample_identifier (int): The identifier of the sample in the batches.

        Returns:
            str: The prefix followed by the identifier.
        """
        return f"{self.prefix}_{sample_identifier}"


class StandInLoader:
    """
    Stand-in for a DataLoader, giving fixed batches and holding its dataset.
    """

    def __init__(self, batches: List[Dict[str, Any]], dataset: NamedSamples) -> None:
        """
        Sets the batches and the dataset of the loader.

        Args:
            batches (List[Dict[str, Any]]): The batches given at every pass.
            dataset (NamedSamples): The dataset of the loader.
        """
        self.batches, self.dataset = batches, dataset

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Gives the batches of the loader.

        Returns:
            Iterator[Dict[str, Any]]: The batches, in order.
        """
        return iter(self.batches)

    def __len__(self) -> int:
        """
        Gives the number of batches of the loader.

        Returns:
            int: The number of batches.
        """
        return len(self.batches)


def write_evaluation_summary(summary_path: Path, rows: List[Dict[str, str]]) -> None:
    """
    Writes an evaluation_summary.csv as the trainers write it, one row per evaluated iteration.

    Args:
        summary_path (Path): The summary file, its folder created.
        rows (List[Dict[str, str]]): The rows, every one holding every column.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, mode="w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_test_summary(experiment_folder: Path, checkpoint_path: Path, formats: List[str],
                       random_batch_checks: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Builds the summary of a parity of a classifier as run_test_split_parity gathers it, an eager row, a TorchScript
    row 2 points of accuracy below it and a failed ONNX row, for the tests of the summary files and of the export.

    Args:
        experiment_folder (Path): The exported run folder.
        checkpoint_path (Path): The exported checkpoint.
        formats (List[str]): The exported formats.
        random_batch_checks (Dict[str, List[str]]): The mismatches of the random batch check of every format.

    Returns:
        Dict[str, Any]: The summary, without its 'installation'.
    """
    parity_folder = get_parity_folder(experiment_folder=experiment_folder, checkpoint_path=checkpoint_path,
                                      use_averaged_weights=False)
    common_row = {"headline_metric": "Accuracy", "percent_metrics": ["Accuracy"], "evaluated_samples": "4 Images",
                  "duration_seconds": 1.0, "error": None}
    worst_record = {"split": "test", "sample_identifier": 7, "sample": "circle_image_000007",
                    "within_tolerance": False, "largest_difference": 0.25,
                    "mismatches": "class probabilities differ by up to 2.50e-01"}
    rows = [{"name": EAGER_ROW_NAME, "label": "Eager", "status": "evaluated", "agreement": None,
             "scalars": {"Accuracy": 0.9, "Effective Rank": 3.5}, **common_row},
            {"name": "torchscript", "label": "TorchScript", "status": "evaluated",
             "agreement": summarise_sample_agreement(records=[worst_record]),
             "scalars": {"Accuracy": 0.88, "Effective Rank": 3.25}, **common_row},
            {"name": "onnx", "label": "ONNX (TensorRT FP16)", "status": "failed", "agreement": None, "scalars": {},
             **common_row, "error": "engine build crashed", "evaluated_samples": None}]
    return {"run": str(experiment_folder), "task": "classification", "checkpoint": str(checkpoint_path),
            "iteration": 1, "weights": "trained weights", "test_data_folder": "test", "formats": formats,
            "tensorrt_precision": "FP16", "torchscript_precision": "FP32", "maximum_batch_size": 4, "model_version": 1,
            "logit_temperature": 1.0,
            "headline_metric": "Accuracy", "random_batch_checks": random_batch_checks,
            "difference_description": "largest difference of a class probability",
            "reference": {"source": "stored.csv", "scalars": {"Accuracy": 0.9}, "comparable": True, "notes": []},
            "rows": rows, "created_at": "2026-11-24T12:00:00", "parity_folder": str(parity_folder)}


class ParityRecorder:
    """
    Stand-in for run_test_split_parity in the tests of the export: it records the formats it was given with their
    random batch checks and returns the summary of build_test_summary, without serving anything.
    """

    def __init__(self) -> None:
        """
        Starts with no recorded call.
        """
        self.calls = []

    def __call__(self, **arguments: Any) -> Any:
        """
        Records a call and returns the summary of the parity with one sample record.

        Args:
            **arguments (Any): The keyword arguments of run_test_split_parity.

        Returns:
            Any: The summary and the sample records of every served row.
        """
        self.calls.append(arguments)
        parity_folder = get_parity_folder(experiment_folder=arguments["experiment_folder"],
                                          checkpoint_path=arguments["checkpoint_path"],
                                          use_averaged_weights=arguments["use_averaged_weights"])
        parity_folder.mkdir(parents=True, exist_ok=True)
        summary = build_test_summary(experiment_folder=arguments["experiment_folder"],
                                     checkpoint_path=arguments["checkpoint_path"], formats=arguments["formats"],
                                     random_batch_checks=arguments["random_batch_checks"])
        return summary, {"torchscript": [summary["rows"][1]["agreement"]["worst_samples"][0]]}


def test_images_are_sent_in_requests_of_the_largest_exported_batch() -> None:
    """
    A test batch of 5 images reaches a model taking at most 2 images per request in requests of 2, 2 and 1 images,
    and the probabilities come back in the order of the batch.
    """
    images = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1).expand(5, 3, 2, 2).contiguous()
    predictions = predict_exported_batch(batch={"images": images, "sample_identifiers": torch.arange(5)},
                                         trainer=StandInTrainer(task_name="classification"),
                                         run_model=run_fake_classifier, output_names=["class_probabilities"],
                                         maximum_request_size=2, comparison=None)
    probabilities = predictions["class_probabilities"]
    assert probabilities[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert probabilities[:, 1].tolist() == [2.0, 2.0, 2.0, 2.0, 1.0]


def test_sequences_are_sent_alone_without_their_padding() -> None:
    """
    Every sequence of a padded batch is sent alone with its real frames only: the frame probabilities of the padding
    of a sequence labeller stay at 0 for the trainer, which ignores the labels of the padding, and the sequence
    probabilities of a sequence classifier are stacked in the order of the batch. Padding before a real frame is
    refused.
    """
    images = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1, 1).expand(2, 4, 3, 2, 2).contiguous()
    sequence_mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    batch = {"images": images, "sequence_mask": sequence_mask, "sample_identifiers": torch.tensor([10, 11])}
    predictions = predict_exported_batch(batch=batch, trainer=StandInTrainer(task_name="sequence_labelling"),
                                         run_model=run_fake_sequence_labeller, output_names=["class_probabilities"],
                                         maximum_request_size=16, comparison=None)
    probabilities = predictions["class_probabilities"]
    assert probabilities.shape == (2, 4, 2)
    assert probabilities[0, :, 0].tolist() == [0.0, 1.0, 2.0, 0.0]
    assert probabilities[0, :, 1].tolist() == [3.0, 3.0, 3.0, 0.0]
    assert probabilities[1, :, 0].tolist() == [4.0, 0.0, 0.0, 0.0]
    assert probabilities[1, :, 1].tolist() == [1.0, 0.0, 0.0, 0.0]

    # One probability vector per sequence for the sequence classifier
    predictions = predict_exported_batch(batch=batch, trainer=StandInTrainer(task_name="sequence_classification"),
                                         run_model=run_fake_sequence_classifier,
                                         output_names=["class_probabilities"], maximum_request_size=16,
                                         comparison=None)
    assert predictions["class_probabilities"].tolist() == [[1.0, 3.0], [4.0, 1.0]]

    # Padding before a real frame cannot be removed by keeping the first frames
    with pytest.raises(ValueError, match="real frames of sequence 1 first"):
        predict_exported_batch(batch={**batch, "sequence_mask": torch.tensor([[True, True, True, False],
                                                                              [False, True, False, False]])},
                               trainer=StandInTrainer(task_name="sequence_labelling"),
                               run_model=run_fake_sequence_labeller, output_names=["class_probabilities"],
                               maximum_request_size=16, comparison=None)


def test_pairs_send_both_members_alone_without_their_padding() -> None:
    """
    The before and after members of every pair, padded to the longest member of the batch, reach the exported
    comparator with their real frames only, each pair in a request of its own.
    """
    images = torch.zeros(size=(2, 6, 3, 2, 2))
    sequence_mask = torch.tensor([[True, True, False, True, True, True], [True, False, False, True, True, False]])
    predictions = predict_exported_batch(batch={"images": images, "sequence_mask": sequence_mask,
                                                "sample_identifiers": torch.tensor([0, 1])},
                                         trainer=StandInTrainer(task_name="sequence_comparison"),
                                         run_model=run_fake_comparator, output_names=["class_probabilities"],
                                         maximum_request_size=16, comparison=None)
    assert predictions["class_probabilities"].tolist() == [[2.0, 3.0], [1.0, 2.0]]


def test_tiles_of_a_tiled_segmentation_go_in_requests_of_the_largest_exported_batch() -> None:
    """
    A test image of a segmentation run trained on tiles is cut into its tiles, sent in requests of at most the exported
    batch size, and the probabilities of the overlapping tiles are averaged back into the image: an image of 6 x 4
    pixels gives two tiles of 4 pixels overlapping by 2 rows, sent alone with a largest request of 1.
    """
    trainer = StandInTrainer(task_name="semantic_segmentation", tiling_configuration={"activated": True, "overlap": 2},
                             input_image_size=4)
    predictions = predict_exported_batch(batch={"images": torch.zeros(size=(1, 3, 6, 4)),
                                                "sample_identifiers": torch.tensor([3])},
                                         trainer=trainer, run_model=run_fake_tile_segmenter,
                                         output_names=["class_map", "class_probabilities"], maximum_request_size=1,
                                         comparison=None)
    assert predictions["class_probabilities"].shape == (1, 2, 6, 4)
    assert torch.all(predictions["class_probabilities"][:, 1] == 1.0)


def test_exported_outputs_become_the_predictions_of_every_task() -> None:
    """
    The detections of an exported detector become the final detections its trainer scores (the number of detections
    as a [B] vector, the proposals it does not return as empty tensors), the instance segmenter adds its masks and
    their scores, the keypoint candidates are grouped by the trainer, the encoder gives its representations, and a task
    without exported model is refused.
    """
    detection_outputs = {"detection_boxes": np.zeros(shape=(2, 100, 4), dtype=np.float32),
                         "detection_scores": np.zeros(shape=(2, 100), dtype=np.float32),
                         "detection_labels": np.zeros(shape=(2, 100), dtype=np.int64),
                         "number_detections": np.array([[3], [0]], dtype=np.int64)}
    predictions = convert_inference_outputs(trainer=StandInTrainer(task_name="object_detection"),
                                            outputs=detection_outputs)
    assert predictions["final_detection_boxes"].shape == (2, 100, 4)
    assert predictions["number_detections"].tolist() == [3, 0]
    assert predictions["proposal_boxes"].shape == (2, 0, 4)
    assert predictions["valid_proposal_mask"].dtype == torch.bool
    assert "final_detection_masks" not in predictions

    # Instance segmentation: the masks of the detected classes and their mask scores
    predictions = convert_inference_outputs(
        trainer=StandInTrainer(task_name="instance_segmentation"),
        outputs={**detection_outputs, "detection_masks": np.ones(shape=(2, 100, 7, 7), dtype=np.float32),
                 "mask_scores": np.full(shape=(2, 100), fill_value=0.5, dtype=np.float32)})
    assert predictions["final_detection_masks"].shape == (2, 100, 7, 7)
    assert predictions["final_detection_mask_scores"].shape == (2, 100)

    # Keypoint detection: the candidates grouped by the trainer of the run
    keypoint_trainer = StandInTrainer(task_name="keypoint_detection")
    keypoint_outputs = {"candidate_positions": np.zeros(shape=(1, 3, 5, 2), dtype=np.float32),
                        "candidate_scores": np.zeros(shape=(1, 3, 5), dtype=np.float32),
                        "candidate_tags": np.zeros(shape=(1, 3, 5), dtype=np.float32)}
    predictions = convert_inference_outputs(trainer=keypoint_trainer, outputs=keypoint_outputs)
    assert predictions["grouped_objects"] == []
    assert set(keypoint_trainer.grouped_predictions[0]) == set(keypoint_outputs)

    # Self-supervised learning: the representations of the encoder
    predictions = convert_inference_outputs(trainer=StandInTrainer(task_name="self_supervised"),
                                            outputs={"representations": np.ones(shape=(2, 8), dtype=np.float32)})
    assert predictions["representations"].shape == (2, 8)
    with pytest.raises(ValueError, match="Expected a task with an exported inference model"):
        convert_inference_outputs(trainer=StandInTrainer(task_name="unknown_task"), outputs=detection_outputs)


def test_served_requests_are_compared_with_the_eager_model_sample_by_sample() -> None:
    """
    Every request of a served row is also run by the eager model and every sample compared with the comparison of the
    export specification: detections swapped by a rounding agree and differ by 0 once their scores are sorted, a score
    moved beyond the tolerance is a mismatch of its own sample, named by its identifier and its test sample name, and
    the summary of the row ranks the samples by their largest difference.
    """
    images = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1).expand(3, 3, 2, 2).contiguous()
    comparison = create_output_comparison(task="object_detection", run_reference=run_fake_detector,
                                          specification={"compare_outputs": compare_detections,
                                                         "comparison_tolerances": {"box_tolerance": 1e-2,
                                                                                   "score_tolerance": 1e-4}},
                                          name_sample=name_test_sample)
    predict_exported_batch(batch={"images": images, "sample_identifiers": torch.tensor([20, 21, 22])},
                           trainer=StandInTrainer(task_name="object_detection"), run_model=run_fake_detector_swapped,
                           output_names=["detection_boxes", "detection_scores", "detection_labels",
                                         "number_detections"], maximum_request_size=2, comparison=comparison)
    records = {record["sample_identifier"]: record for record in comparison["records"]}
    assert list(records) == [20, 21, 22]
    assert (records[22]["split"], records[22]["sample"]) == ("test", "image_22")
    assert records[20]["within_tolerance"] and records[21]["within_tolerance"]
    assert records[20]["largest_difference"] == 0.0
    assert not records[22]["within_tolerance"]
    assert records[22]["largest_difference"] == pytest.approx(0.01)
    assert "without a candidate" in records[22]["mismatches"]
    agreement = summarise_sample_agreement(records=comparison["records"])
    assert (agreement["compared_samples"], agreement["within_tolerance"]) == (3, 2)
    assert agreement["worst_samples"][0]["sample_identifier"] == 22


def test_reference_images_of_a_self_supervised_row_are_named_from_their_own_split() -> None:
    """
    The k-nearest neighbour reference images of a served self-supervised row, training images whose identifiers are
    those of the training manifest, are recorded under the reference split and named from the dataset of their loader,
    and the test batches that follow are named from the test dataset again.
    """
    trainer = SelfSupervisedTrainer.__new__(SelfSupervisedTrainer)
    trainer.device, trainer.dtype, trainer.evaluating_averaged_weights = DEVICE, torch.float32, False
    reference_batch = {"images": torch.zeros(size=(2, 3, 2, 2)), "labels": torch.tensor([0, 1]),
                       "sample_identifiers": torch.tensor([5, 6])}
    trainer.reference_dataloader = StandInLoader(batches=[reference_batch], dataset=NamedSamples(prefix="training"))
    trainer.test_dataloader = StandInLoader(batches=[], dataset=NamedSamples(prefix="test"))
    comparison = create_output_comparison(task="self_supervised", run_reference=run_fake_encoder,
                                          specification={"compare_outputs": compare_representations,
                                                         "comparison_tolerances": {"tolerance": 1e-4}},
                                          name_sample=trainer.get_test_sample_name)
    trainer.predict_batch = functools.partial(predict_exported_batch, trainer=trainer, run_model=run_fake_encoder,
                                              output_names=["representations"], maximum_request_size=4,
                                              comparison=comparison)
    representations, labels = encode_named_reference_images(trainer=trainer, comparison=comparison)
    assert representations.shape == (2, 4) and labels.tolist() == [0, 1]
    trainer.predict_batch(batch={"images": torch.ones(size=(1, 3, 2, 2)), "sample_identifiers": torch.tensor([5])})
    assert [(record["split"], record["sample"]) for record in comparison["records"]] == [
        ("reference", "training_5"), ("reference", "training_6"), ("test", "test_5")]
    assert all(record["within_tolerance"] for record in comparison["records"])


def test_sample_difference_reads_the_probabilities_or_the_sorted_scores() -> None:
    """
    The largest difference of a sample is the largest probability difference of a classifier, the largest difference
    of the sorted candidate scores of a keypoint detector, and infinite for outputs of other shapes.
    """
    reference = [np.array([[0.2, 0.8]], dtype=np.float32)]
    assert measure_sample_difference(task="classification", reference=reference,
                                     candidate=[np.array([[0.25, 0.75]], dtype=np.float32)]) == pytest.approx(0.05)
    assert measure_sample_difference(task="classification", reference=reference,
                                     candidate=[np.zeros(shape=(1, 3), dtype=np.float32)]) == float("inf")
    keypoint_scores = np.array([[[0.9, 0.5, 0.1]]], dtype=np.float32)
    positions = np.zeros(shape=(1, 1, 3, 2), dtype=np.float32)
    assert measure_sample_difference(task="keypoint_detection", reference=[positions, keypoint_scores],
                                     candidate=[positions, keypoint_scores[:, :, ::-1]]) == 0.0
    assert compare_probabilities(reference=reference, candidate=reference, tolerance=1e-4) == []


def test_reference_scalars_come_from_the_checkpoint_evaluation_then_the_training(tmp_path: Path) -> None:
    """
    The stored evaluation of the exported iteration is read from Checkpoint_Evaluation first, then from the Evaluation
    of the training, the averaged weights from their 'ema_' columns and the empty cells left out; an iteration no
    evaluation holds gives no reference.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    experiment_folder = tmp_path / "CLS_TEST_MC3"
    write_evaluation_summary(summary_path=experiment_folder / "Evaluation" / "evaluation_summary.csv",
                             rows=[{"iteration": "1000", "Accuracy": "0.8", "ema_Accuracy": "0.81"},
                                   {"iteration": "2000", "Accuracy": "0.9", "ema_Accuracy": ""}])
    write_evaluation_summary(summary_path=experiment_folder / "Checkpoint_Evaluation" / "evaluation_summary.csv",
                             rows=[{"iteration": "2000", "Accuracy": "0.91", "Macro F1": ""}])
    reference = read_reference_scalars(experiment_folder=experiment_folder, iteration=2000,
                                       use_averaged_weights=False)
    assert reference == {"source": str(experiment_folder / "Checkpoint_Evaluation" / "evaluation_summary.csv"),
                         "scalars": {"Accuracy": 0.91}}
    reference = read_reference_scalars(experiment_folder=experiment_folder, iteration=1000, use_averaged_weights=True)
    assert reference["scalars"] == {"Accuracy": 0.81}
    assert read_reference_scalars(experiment_folder=experiment_folder, iteration=3000,
                                  use_averaged_weights=False) == {"source": None, "scalars": {}}


def test_summary_gives_every_row_against_the_eager_row(tmp_path: Path) -> None:
    """
    The parity summary writes the metrics of every evaluated row with their change against the eager row (in points
    for a rate), the stored evaluation, the samples of largest difference and the failed rows with their error, its
    JSON twin the same summary, and parity_samples.csv one line per compared sample.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    experiment_folder = tmp_path / "CLS_TEST_MC3"
    checkpoint_path = experiment_folder / "Weights" / "Iteration_1" / "model_000001.pt"
    summary = build_test_summary(experiment_folder=experiment_folder, checkpoint_path=checkpoint_path,
                                 formats=["torchscript", "onnx"], random_batch_checks={"torchscript": [], "onnx": []})
    summary["installation"] = "installed into served_models, version 1"
    parity_folder = Path(summary["parity_folder"])
    assert parity_folder == experiment_folder / "Export_Parity" / "Iteration_1"
    parity_folder.mkdir(parents=True)
    sample_records = {"torchscript": summary["rows"][1]["agreement"]["worst_samples"]}
    summary_paths = write_parity_summary(parity_folder=parity_folder, summary=summary, sample_records=sample_records)
    assert [path.name for path in summary_paths] == [SUMMARY_MARKDOWN_NAME, SUMMARY_JSON_NAME]
    markdown = summary_paths[0].read_text()
    assert "| **Accuracy** | 90.00 % | 88.00 % | -2.00 points | 90.00 % |" in markdown
    assert "| Effective Rank | 3.5000 | 3.2500 | -0.2500 | — |" in markdown
    assert "- ONNX (TensorRT FP16): engine build crashed" in markdown
    assert ("| circle_image_000007 | 7 | 2.500e-01 | no | class probabilities differ by up to 2.50e-01 |"
            in markdown)
    assert "Installation: installed into served_models, version 1" in markdown
    assert json.loads(summary_paths[1].read_text())["rows"][1]["scalars"]["Accuracy"] == 0.88
    sample_lines = (parity_folder / SAMPLES_FILE_NAME).read_text().splitlines()
    assert sample_lines[0] == "row,split,sample_identifier,sample,within_tolerance,largest_difference,mismatches"
    assert sample_lines[1].startswith("torchscript,test,7,circle_image_000007,False,0.25,")


def test_summary_of_a_detector_leaves_out_its_proposal_scalars(tmp_path: Path) -> None:
    """
    The metrics table of the summary of a detector leaves out its proposal scalars, which read 0 in every row as an
    exported detector returns no proposal, and says so; the metrics table of another task keeps every scalar.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    experiment_folder = tmp_path / "DET_TEST_MC3"
    summary = build_test_summary(experiment_folder=experiment_folder,
                                 checkpoint_path=experiment_folder / "Weights" / "Iteration_1" / "model_000001.pt",
                                 formats=["torchscript", "onnx"], random_batch_checks={"torchscript": [], "onnx": []})
    summary["installation"] = "installed into served_models, version 1"
    for row in summary["rows"][:2]:
        row["scalars"].update({"Proposal Recall@0.5/all": 0.0, "Proposals Per Image/all": 0.0})
    assert "| Proposal Recall@0.5/all |" in build_summary_markdown(summary=summary)
    summary["task"] = "object_detection"
    markdown = build_summary_markdown(summary=summary)
    assert "Proposal Recall@0.5/all" not in markdown and "Proposals Per Image/all |" not in markdown
    assert "- 2 proposal scalars left out" in markdown
    assert "| **Accuracy** | 90.00 % | 88.00 % | -2.00 points |" in markdown


def test_summary_names_the_precision_of_an_fp16_torchscript_row(tmp_path: Path) -> None:
    """
    Verifies that the summary of an export with an FP16 TorchScript file names its row and its random batch check with
    the precision and the eager FP16 reference of the check, the ONNX row keeping its own, and that an FP32 TorchScript
    row keeps the plain label of the summaries written before.

    Args:
        tmp_path (Path): The temporary folder of the test.
    """
    experiment_folder = tmp_path / "CLS_TEST_MC3"
    summary = build_test_summary(experiment_folder=experiment_folder,
                                 checkpoint_path=experiment_folder / "Weights" / "Iteration_1" / "model_000001.pt",
                                 formats=["torchscript", "onnx"], random_batch_checks={"torchscript": [], "onnx": []})
    summary.update({"torchscript_precision": "FP16", "installation": "installed into served_models, version 1"})
    markdown = build_summary_markdown(summary=summary)
    assert "- Formats: TorchScript (FP16), ONNX (TensorRT FP16), maximum batch size 4" in markdown
    assert "- TorchScript (FP16): matches the eager FP16 copy of the network" in markdown
    assert "- ONNX (TensorRT FP16): matches the eager model" in markdown
    assert get_row_label(row_name="torchscript", tensorrt_precision=None, torchscript_precision="FP32") == "TorchScript"
    assert get_row_label(row_name="torchscript", tensorrt_precision="FP16",
                         torchscript_precision="FP16") == "TorchScript (FP16)"


def test_triton_server_of_the_parity_has_its_own_name_and_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A server is described by its container name and ports, the parity asks the system for free distinct ports, the
    container is started listening on those ports, and a machine without Docker is refused before an export starts.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the Docker commands and the readiness request.
    """
    ports = find_free_ports(number_ports=3)
    assert len(set(ports)) == 3
    server = build_triton_server(container_name="export-parity-triton-1", http_port=ports[0], grpc_port=ports[1],
                                 metrics_port=ports[2])
    assert server["grpc_address"] == f"localhost:{ports[1]}"

    # The command of the container, recorded instead of run
    commands = []
    monkeypatch.setattr(triton_server.subprocess, "run", lambda args, **keywords: commands.append(args))
    monkeypatch.setattr(triton_server, "is_server_ready", lambda server: True)
    start_triton_server(server=server, repository=Path("/repository"), engine_cache_folder=Path("/tmp"),
                        environment={})
    assert commands[0] == ["docker", "rm", "--force", "export-parity-triton-1"]
    assert f"--http-port={ports[0]}" in commands[1] and f"--grpc-port={ports[1]}" in commands[1]
    assert "/repository:/models:ro" in commands[1]

    # No docker command on the machine
    monkeypatch.setattr(triton_server.shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="docker command is not found"):
        check_triton_available()


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason="The export runs the model on the GPU.")
def test_export_with_parity_writes_its_summary_whatever_the_random_batch_check(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that an export with test_split_parity scores every format even when the random batch check of the first
    one fails, writes the summary into the parity folder of the run with the refused installation, raises the mismatch
    and leaves the repository without any model, and that a checked export installs its models with the summary next
    to every one of them.

    Args:
        tmp_path (Path): The pytest temporary directory.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture standing in for the Triton parts of the parity.
    """
    experiment_folder, repository = tmp_path / "CLS_TEST_MC3", tmp_path / "served_models"
    test_input_contract.write_trained_model(experiment_folder=experiment_folder, task="classification",
                                            keep_ratio=True, tiling_configuration=None)
    parity_recorder = ParityRecorder()
    monkeypatch.setattr(target=export_model, name="check_triton_available", value=lambda: None)
    monkeypatch.setattr(target=export_model, name="run_test_split_parity", value=parity_recorder)
    export_settings = {"experiment_folder": experiment_folder, "iteration": test_input_contract.EXPORTED_ITERATION,
                       "use_averaged_weights": False, "formats": ["torchscript", "onnx"], "repository": repository,
                       "model_version": 1, "maximum_batch_size": test_input_contract.TEST_MAXIMUM_BATCH_SIZE,
                       "tensorrt_precision": "FP16", "torchscript_precision": "FP32", "use_saved_temperature": False,
                       "saved_thresholds_path": None, "test_split_parity": True}

    # Both formats fail their check: both are still scored, the summary records the refused installation
    monkeypatch.setattr(target=export_model, name="get_export_specification", value=get_mismatching_specification)
    with pytest.raises(expected_exception=RuntimeError, match=re.escape(pattern=FORCED_MISMATCH)):
        export_experiment(**export_settings)
    assert parity_recorder.calls[0]["formats"] == ["torchscript", "onnx"]
    assert parity_recorder.calls[0]["random_batch_checks"] == {"torchscript": [FORCED_MISMATCH],
                                                               "onnx": [FORCED_MISMATCH]}
    parity_folder = experiment_folder / "Export_Parity" / f"Iteration_{test_input_contract.EXPORTED_ITERATION}"
    markdown = (parity_folder / SUMMARY_MARKDOWN_NAME).read_text()
    assert "Installation: refused, the random batch check of torchscript, onnx failed" in markdown
    assert f"- TorchScript: mismatch: {FORCED_MISMATCH}" in markdown
    assert read_repository_files(repository=repository) == {}
    assert not list(tmp_path.glob(pattern=f"{STAGING_FOLDER_PREFIX}*"))

    # The checked export installs its models with the summary next to them
    monkeypatch.setattr(target=export_model, name="get_export_specification",
                        value=ORIGINAL_GET_EXPORT_SPECIFICATION)
    export_experiment(**export_settings)
    repository_files = read_repository_files(repository=repository)
    for model_folder in ["image_classifier_torchscript/1", "image_classifier_onnx/1"]:
        assert f"{model_folder}/{SUMMARY_MARKDOWN_NAME}" in repository_files
        assert f"{model_folder}/{SUMMARY_JSON_NAME}" in repository_files
    assert b"Installation: installed into" in repository_files[f"image_classifier_onnx/1/{SUMMARY_MARKDOWN_NAME}"]
    assert not list(tmp_path.glob(pattern=f"{STAGING_FOLDER_PREFIX}*"))

