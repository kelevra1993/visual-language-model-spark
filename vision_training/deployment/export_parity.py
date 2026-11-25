"""
Test split parity of an export (deployment/export_model.py with test_split_parity): the eager inference model in this
process, the reference, then every exported format served by a temporary Triton container on the staged models,
exactly as it will be served (TorchScript in FP32 or FP16, ONNX with its TensorRT precision), are scored on the whole
test split of the run by the test evaluation of the trainer of its task, whose predict_batch is replaced by the
requests to the model of the row (predict_exported_batch). Every request of a served row also goes through the eager
model, and every sample is compared with the comparison and tolerances of the export specification, so that the parity
gives both the metrics of every format and the samples whose outputs moved. The metrics of every row, their changes
against the eager row, the stored evaluation of the exported iteration where it is comparable and the agreement of the
samples are written to '<run>/Export_Parity/Iteration_<n>[_EMA][_T<temperature>]_<precisions>/' (parity_summary.md,
parity_summary.json, parity_samples.csv, the evaluation files of every row and the server log), a folder of its own for
every checkpoint, weights, classifier temperature and precision of every exported format. The parity only reports: the
installation of the models stays decided by the random batch check of the export. The request and conversion functions
also serve the test metric rows of the serving benchmark (scripts/deployment/evaluate_served_formats.py).
"""

import csv
import functools
import json
import os
import shutil
import time
import urllib.error
import numpy as np
import torch
import tritonclient.grpc as triton_client

from datetime import datetime
from pathlib import Path
from torch import nn
from tritonclient.utils import InferenceServerException
from typing import Any, Callable, Dict, List, Optional, Tuple

from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.core.contracts import SAMPLE_IDENTIFIERS_KEY
from vision_training.core.task_registry import get_trainer_class
from vision_training.data.manifest_cache import read_manifests_once
from vision_training.data.sequence.sequence_pair_dataset import split_pair_members
from vision_training.deployment.deployment_utilities import (get_triton_output_names, run_local_inference,
                                                             run_triton_inference)
from vision_training.deployment.triton_configuration import EXPORT_FORMATS
from vision_training.deployment.triton_server import (TritonServerError, build_triton_server, find_free_ports,
                                                      load_served_model, start_triton_server, stop_triton_server,
                                                      unload_served_model)
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.evaluation_curves import (AVERAGED_WEIGHTS_COLUMN_PREFIX, ITERATION_COLUMN,
                                                          read_evaluation_summary)
from vision_training.evaluation.report_formatting import format_points, format_report_value
from vision_training.models.model_transfer import disable_pretrained_settings
from vision_training.models.segmentation.sliding_window import segment_image_by_tiles
from vision_training.trainers.base_trainer import EVALUATION_FOLDER_NAME, BaseTrainer
from vision_training.trainers.detection_trainer import DetectionTrainer
from vision_training.trainers.self_supervised_trainer import SelfSupervisedTrainer
from vision_training.utilities.os_utilities import format_path_link, print_color

# Folder of the parities of a run, next to its evaluations, holding one folder per exported checkpoint, weights,
# classifier temperature and precisions (get_parity_folder), its files, and the folder of the evaluation files of every
# row
PARITY_FOLDER_NAME = "Export_Parity"
AVERAGED_WEIGHTS_SUFFIX = "_EMA"
ROW_EVALUATIONS_FOLDER_NAME = "row_evaluations"
SUMMARY_MARKDOWN_NAME = "parity_summary.md"
SUMMARY_JSON_NAME = "parity_summary.json"
SAMPLES_FILE_NAME = "parity_samples.csv"
SERVER_LOG_NAME = "triton_server.log"

# Row of the eager inference model in the process, the reference of every served row
EAGER_ROW_NAME = "eager"

# Columns of parity_samples.csv, one line per compared sample of every served row: its split ('test', or 'reference'
# for the k-nearest neighbour reference images of a self-supervised run, training images), its identifier in the
# batches and its name (the file, sequence or pair name of the evaluation outputs)
SAMPLE_COLUMNS = ["row", "split", "sample_identifier", "sample", "within_tolerance", "largest_difference",
                  "mismatches"]
TEST_SPLIT_NAME = "test"
REFERENCE_SPLIT_NAME = "reference"

# Samples of largest difference listed per row in the summary
WORST_SAMPLES_SHOWN = 10

# Change of the headline metric of a served row against the eager row from which the console writes it in yellow, for
# the eye only: the parity never decides anything
HIGHLIGHTED_HEADLINE_CHANGE = 0.005

# Prefix of the name of the Triton container of a parity, followed by the process identifier
PARITY_CONTAINER_PREFIX = "export-parity-triton"

# Tasks whose test samples are sequences sent alone, and the tasks giving the probabilities of every frame
SEQUENCE_SAMPLE_TASKS = ["sequence_classification", "sequence_labelling", "frame_selection"]
FRAME_PROBABILITY_TASKS = ["sequence_labelling", "frame_selection"]

# Tasks scored on class probabilities, and the detection tasks
PROBABILITY_TASKS = ["classification", "sequence_classification", "sequence_labelling", "sequence_comparison",
                     "frame_selection"]
DETECTION_TASKS = ["object_detection", "instance_segmentation"]

# Start of the names of the proposal scalars of the detectors (evaluation/detection/region_recall.py), left out of the
# metrics table of the summary as an exported detector returns no proposal (their rows read 0), kept in its JSON twin
PROPOSAL_SCALAR_PREFIX = "Proposal"

# What the largest difference of a compared sample measures, for every task, written in the summary
DIFFERENCE_DESCRIPTIONS = {
    **{task: "largest difference of a class probability" for task in PROBABILITY_TASKS},
    "semantic_segmentation": "largest difference of a class probability of a pixel",
    "self_supervised": "largest difference of a value of the normalised representation",
    "object_detection": "largest difference of the detection scores sorted by value (a detection missing in one "
                        "output counting its whole score)",
    "instance_segmentation": "largest difference of the detection scores sorted by value (a detection missing in one "
                             "output counting its whole score)",
    "keypoint_detection": "largest difference of the candidate scores of a heatmap channel sorted by value"}

# Names of the export formats in the summary
FORMAT_LABELS = {"torchscript": "TorchScript", "onnx": "ONNX"}


def get_parity_folder(experiment_folder: Path, checkpoint_path: Path, use_averaged_weights: bool, task: str,
                      logit_temperature: float, formats: List[str], tensorrt_precision: Optional[str],
                      torchscript_precision: str) -> Path:
    """
    Names the folder of the parity of an export, next to the evaluations of the run, after everything its rows depend
    on, so that no parity replaces that of another export: '<run>/Export_Parity/Iteration_<n>[_EMA][_T<temperature>]_
    <precisions>/', the temperature built into a classifier (the only task with one, 'T1' without a saved temperature)
    and the precision of every exported format in the order of EXPORT_FORMATS: 'TorchScript-FP32' or
    'TorchScript-FP16', then 'TensorRT-FP32', 'TensorRT-FP16' or 'ONNXRuntime-CUDA' for ONNX without TensorRT, e.g.
    'Iteration_5000_EMA_T3_TorchScript-FP16_TensorRT-FP32'. A parity of the same export replaces it.

    Args:
        experiment_folder (Path): The exported run folder.
        checkpoint_path (Path): The exported checkpoint, in its 'Iteration_<n>' folder.
        use_averaged_weights (bool): Whether the moving average of the weights is exported.
        task (str): The task of the run.
        logit_temperature (float): The logit temperature built into an exported classifier, 1 for the other tasks.
        formats (List[str]): The exported formats.
        tensorrt_precision (Optional[str]): The TensorRT precision of the ONNX model, None for ONNX Runtime CUDA.
        torchscript_precision (str): The precision of the network traced into the TorchScript file.

    Returns:
        Path: The parity folder.
    """
    weights_suffix = AVERAGED_WEIGHTS_SUFFIX if use_averaged_weights else ""
    temperature_suffix = f"_T{logit_temperature:g}" if task == "classification" else ""

    # Precision of every exported format, in a fixed order whatever the order of the formats asked
    precision_names = []
    for export_format in EXPORT_FORMATS:
        if export_format not in formats:
            continue
        if export_format == "torchscript":
            precision_names.append(f"TorchScript-{torchscript_precision}")
        else:
            precision_names.append("ONNXRuntime-CUDA" if tensorrt_precision is None else
                                   f"TensorRT-{tensorrt_precision}")
    return experiment_folder / PARITY_FOLDER_NAME / (f"{checkpoint_path.parent.name}{weights_suffix}"
                                                     f"{temperature_suffix}_{'_'.join(precision_names)}")


def build_evaluation_trainer(experiment_folder: Path, test_data_folder: Path, evaluation_folder: Path,
                             standalone_evaluation_folder: str, batch_size: int) -> BaseTrainer:
    """
    Builds the trainer of an exported or benchmarked run for its test evaluation only, like
    scripts/evaluation/evaluate_checkpoint.py but writing into the given folder instead of the run folder: its
    configuration is read from the run, the tracking, the predictions file, the error images, the per-class report and
    the sample images are turned off (they read outputs the exported models do not return, such as the class logits),
    and its test loader gives batches of the largest batch size of the exported models. The weights of its model are
    never used: every row brings its own model through predict_batch.

    Args:
        experiment_folder (Path): The trained run holding 'training_configuration.yaml'.
        test_data_folder (Path): The test split evaluated.
        evaluation_folder (Path): The folder receiving the evaluation files.
        standalone_evaluation_folder (str): The folder of the trainer inside evaluation_folder, holding its archived
            configuration and the default evaluation files.
        batch_size (int): The batch size of the test loader, the largest one of the exported models.

    Returns:
        BaseTrainer: The trainer of the task of the run.
    """
    configuration_path = experiment_folder / "training_configuration.yaml"
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    experiment_configuration["test_data_folder"] = test_data_folder
    experiment_configuration["batch_size"] = batch_size
    experiment_configuration["tracking"]["activated"] = False
    # Outputs the exported models cannot feed (class logits, error images drawing the proposals) turned off, the tasks
    # without a predictions file having no 'save_predictions' setting
    if "save_predictions" in experiment_configuration["evaluation"]:
        experiment_configuration["evaluation"]["save_predictions"] = False
    experiment_configuration["evaluation"]["per_class_report"] = False
    if "error_dumps" in experiment_configuration["evaluation"]:
        experiment_configuration["evaluation"]["error_dumps"]["activated"] = False
    # Evaluation files written in '<evaluation folder>/<standalone evaluation folder>/'
    experiment_configuration["experiment_folder"] = evaluation_folder.parent
    experiment_configuration["run_name"] = evaluation_folder.name
    experiment_configuration["standalone_evaluation_folder"] = standalone_evaluation_folder
    disable_pretrained_settings(model_configuration=model_configuration)

    # Trainer of the task, every 'labels.json' parsed once, without the fixed sample images
    trainer_class = get_trainer_class(task=experiment_configuration["task"])
    with read_manifests_once():
        trainer = trainer_class(experiment_configuration=experiment_configuration,
                                model_configuration=model_configuration, configuration_path=configuration_path)
    trainer.sample_image_identifiers = []
    return trainer


def point_trainer_to_row(trainer: BaseTrainer, row_folder: Path) -> None:
    """
    Sends the evaluation files of the next row (Results.txt, evaluation_summary.csv, evaluation_curves.png, the
    iteration folder and the Test TensorBoard events) to the folder of that row, so that the rows of an export never
    write into one another.

    Args:
        trainer (BaseTrainer): The trainer of the run, from build_evaluation_trainer.
        row_folder (Path): The folder of the row, created.
    """
    row_folder.mkdir(parents=True, exist_ok=True)
    trainer.experiment_folder = row_folder
    trainer.evaluation_root = row_folder
    trainer.tensorboard_directory = row_folder / "Tensorboard"

    # The Test writers of the previous row closed, the next evaluation creating its own in the new folder
    for writer_name in ["test_writer", "averaged_test_writer"]:
        writer = getattr(trainer, writer_name)
        if writer is not None:
            writer.close()
            setattr(trainer, writer_name, None)


def convert_inference_outputs(trainer: BaseTrainer, outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Turns the outputs of an inference model (eager or served) for a test batch into the predictions the trainer of its
    task scores: the class probabilities of the classifiers, sequence models and segmenter, the representations of the
    encoder, the final detections of the detectors with no proposals (the exported detector does not return them, so
    the proposal recall of the report reads 0 and is not compared) and the masks of the instance segmenter, and the
    keypoint candidates grouped into objects by the trainer as in its own evaluation.

    Args:
        trainer (BaseTrainer): The trainer of the run.
        outputs (Dict[str, np.ndarray]): The outputs of the batch, keyed by the output names of the export
            specification.

    Returns:
        Dict[str, Any]: The predictions of predict_batch.

    Raises:
        ValueError: If the task of the trainer has no exported inference model.
    """
    task, device = trainer.TASK_NAME, trainer.device
    tensors = {output_name: torch.from_numpy(np.ascontiguousarray(output)).to(device=device)
               for output_name, output in outputs.items()}
    if task in PROBABILITY_TASKS or task == "semantic_segmentation":
        return {"class_probabilities": tensors["class_probabilities"]}
    if task == "self_supervised":
        return {"representations": tensors["representations"]}
    if task == "keypoint_detection":
        return trainer.group_candidates(predictions={output_name: tensors[output_name] for output_name in
                                                     ["candidate_positions", "candidate_scores", "candidate_tags"]})
    if task not in DETECTION_TASKS:
        raise ValueError(f"Expected a task with an exported inference model, got {task}.")

    # Final detections of every image, and empty proposals for the proposal recall
    number_images = tensors["detection_boxes"].shape[0]
    predictions = {"final_detection_boxes": tensors["detection_boxes"],
                   "final_detection_scores": tensors["detection_scores"],
                   "final_detection_labels": tensors["detection_labels"],
                   "number_detections": tensors["number_detections"].reshape(-1),
                   "proposal_boxes": torch.zeros(size=(number_images, 0, 4), device=device),
                   "proposal_levels": torch.zeros(size=(number_images, 0), dtype=torch.int64, device=device),
                   "valid_proposal_mask": torch.zeros(size=(number_images, 0), dtype=torch.bool, device=device)}

    # Masks of the detected class of every detection, pasted by the trainer, and their scores when the masks are scored
    if task == "instance_segmentation":
        predictions["final_detection_masks"] = tensors["detection_masks"]
        if "mask_scores" in tensors:
            predictions["final_detection_mask_scores"] = tensors["mask_scores"]
    return predictions


def convert_to_request_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Turns a batch of images or frames of the trainer (on its device, in its precision) into the contiguous float32
    array of a request, the type of the inputs of the exported models.

    Args:
        tensor (torch.Tensor): The images or frames.

    Returns:
        np.ndarray: The float32 array.
    """
    return np.ascontiguousarray(tensor.detach().float().cpu().numpy())


def measure_sample_difference(task: str, reference: List[np.ndarray], candidate: List[np.ndarray]) -> float:
    """
    Gives the largest difference between the outputs of one sample of the eager model and of an exported model, the
    number ranking the samples of a row in the summary (DIFFERENCE_DESCRIPTIONS): the class probabilities, the
    representation values, and for the detectors and the keypoint detector the scores sorted by value, so that two
    detections or candidates swapped by a rounding do not count as a difference.

    Args:
        task (str): The task of the model.
        reference (List[np.ndarray]): The eager outputs of the sample, each with a batch dimension of one.
        candidate (List[np.ndarray]): The exported outputs of the sample, in the same order.

    Returns:
        float: The largest difference, infinite when the outputs do not have the same shapes.
    """
    # Output compared: the detection or candidate scores, the pixel probabilities of the segmenter, or the only output
    output_index = 1 if task in DETECTION_TASKS + ["keypoint_detection", "semantic_segmentation"] else 0
    reference_output, candidate_output = reference[output_index], candidate[output_index]
    if reference_output.shape != candidate_output.shape:
        return float("inf")
    if task in DETECTION_TASKS + ["keypoint_detection"]:
        reference_output = -np.sort(a=-reference_output, axis=-1)
        candidate_output = -np.sort(a=-candidate_output, axis=-1)
    return float(np.abs(reference_output.astype(np.float64) - candidate_output.astype(np.float64)).max(initial=0.0))


def create_output_comparison(task: str, run_reference: Callable[..., List[np.ndarray]],
                             specification: Dict[str, Any], name_sample: Callable[..., str]) -> Dict[str, Any]:
    """
    Prepares the sample comparison of a served row: every request sent to the served model is also run by the eager
    model, and every sample of the request compared with the comparison and tolerances of the export specification
    and named, so that the summary points at the files whose outputs moved.

    Args:
        task (str): The task of the model.
        run_reference (Callable[..., List[np.ndarray]]): Runs one request ('inputs' keyword) through the eager model.
        specification (Dict[str, Any]): The export specification of the task.
        name_sample (Callable[..., str]): Names a test sample from its identifier ('sample_identifier' keyword), the
            get_test_sample_name of the trainer.

    Returns:
        Dict[str, Any]: The 'task', 'run_reference', 'compare_outputs', 'tolerances', the 'split' of the requests
            ('test', switched to 'reference' by encode_named_reference_images) with its 'name_sample', and the empty
            'records' list.
    """
    return {"task": task, "run_reference": run_reference, "compare_outputs": specification["compare_outputs"],
            "tolerances": specification["comparison_tolerances"], "split": TEST_SPLIT_NAME, "name_sample": name_sample,
            "records": []}


def encode_named_reference_images(trainer: SelfSupervisedTrainer,
                                  comparison: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replaces encode_reference_images of a self-supervised trainer during a served row, so that the k-nearest neighbour
    reference images (training images, sent to the served model like the test images) are recorded under their own
    split and named from their own dataset, their identifiers being those of the training manifest.

    Args:
        trainer (SelfSupervisedTrainer): The trainer of the run.
        comparison (Dict[str, Any]): The comparison of the served row, its 'split' and 'name_sample' set back to the
            test split afterwards.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The representations and class indices of the reference images, from
            encode_reference_images.
    """
    comparison["split"] = REFERENCE_SPLIT_NAME
    comparison["name_sample"] = trainer.reference_dataloader.dataset.get_sample_name
    try:
        return SelfSupervisedTrainer.encode_reference_images(self=trainer)
    finally:
        comparison["split"], comparison["name_sample"] = TEST_SPLIT_NAME, trainer.get_test_sample_name


def compare_request_outputs(comparison: Dict[str, Any], inputs: Dict[str, np.ndarray],
                            candidate_outputs: List[np.ndarray], sample_identifiers: List[int]) -> None:
    """
    Runs a request through the eager model and records, for every sample of the request, its split and name, whether
    the exported outputs agree with the eager ones within the tolerances of the export specification, their largest
    difference and the description of every mismatch.

    Args:
        comparison (Dict[str, Any]): The comparison of the row, from create_output_comparison, its 'records' extended.
        inputs (Dict[str, np.ndarray]): The inputs of the request, keyed by input name.
        candidate_outputs (List[np.ndarray]): The outputs of the served model for the request, in the order of the
            export specification.
        sample_identifiers (List[int]): The identifier of every sample of the request, in order.
    """
    reference_outputs = comparison["run_reference"](inputs=inputs)
    for sample_index, sample_identifier in enumerate(sample_identifiers):
        reference_sample = [output[sample_index:sample_index + 1] for output in reference_outputs]
        candidate_sample = [output[sample_index:sample_index + 1] for output in candidate_outputs]
        mismatches = comparison["compare_outputs"](reference=reference_sample, candidate=candidate_sample,
                                                   **comparison["tolerances"])
        comparison["records"].append({"split": comparison["split"], "sample_identifier": sample_identifier,
                                      "sample": comparison["name_sample"](sample_identifier=sample_identifier),
                                      "within_tolerance": not mismatches,
                                      "largest_difference": measure_sample_difference(
                                          task=comparison["task"], reference=reference_sample,
                                          candidate=candidate_sample),
                                      "mismatches": "; ".join(mismatches)})


def run_request(inputs: Dict[str, np.ndarray], run_model: Callable[..., List[np.ndarray]],
                comparison: Optional[Dict[str, Any]], sample_identifiers: List[int]) -> List[np.ndarray]:
    """
    Sends one request to the model of a row, then, for a served row, compares its outputs with those of the eager model
    sample by sample.

    Args:
        inputs (Dict[str, np.ndarray]): The inputs of the request, keyed by input name.
        run_model (Callable[..., List[np.ndarray]]): Runs one request ('inputs' keyword), the eager or served model.
        comparison (Optional[Dict[str, Any]]): The comparison of a served row, None for the eager row.
        sample_identifiers (List[int]): The identifier of every sample of the request, in order.

    Returns:
        List[np.ndarray]: The outputs of the model, in the order of the export specification.
    """
    outputs = run_model(inputs=inputs)
    if comparison is not None:
        compare_request_outputs(comparison=comparison, inputs=inputs, candidate_outputs=outputs,
                                sample_identifiers=sample_identifiers)
    return outputs


def extract_real_frames(frames: torch.Tensor, sequence_mask: torch.Tensor, sample_index: int) -> np.ndarray:
    """
    Gives the real frames of one sequence of a padded batch as the request of a deployed sequence model, which takes no
    sequence mask: the frames of the sequence alone, without the padding after them.

    Args:
        frames (torch.Tensor): The [B, T, 3, H, W] frames of the batch, or of one member of a batch of pairs.
        sequence_mask (torch.Tensor): The [B, T] booleans, True for the real frames.
        sample_index (int): The index of the sequence in the batch.

    Returns:
        np.ndarray: The [1, T_real, 3, H, W] float32 frames.

    Raises:
        ValueError: If the real frames of the sequence are not its first frames.
    """
    sequence_length = int(sequence_mask[sample_index].sum())
    if not bool(sequence_mask[sample_index, :sequence_length].all()):
        raise ValueError(f"Expected the real frames of sequence {sample_index} first, got the mask "
                         f"{sequence_mask[sample_index].tolist()}.")
    return convert_to_request_array(tensor=frames[sample_index:sample_index + 1, :sequence_length])


def predict_exported_tiles(tiles: torch.Tensor, run_model: Callable[..., List[np.ndarray]],
                           probability_index: int, maximum_request_size: int, comparison: Optional[Dict[str, Any]],
                           sample_identifier: int) -> torch.Tensor:
    """
    Gives the class probabilities of a batch of tiles of a test image of a segmentation run trained on tiles, sent to
    the model of the row in requests of at most maximum_request_size tiles, for segment_image_by_tiles.

    Args:
        tiles (torch.Tensor): The [N, 3, S, S] tiles.
        run_model (Callable[..., List[np.ndarray]]): Runs one request ('inputs' keyword), the eager or served model.
        probability_index (int): The index of the class probabilities among the outputs of the model.
        maximum_request_size (int): The largest batch size of a request.
        comparison (Optional[Dict[str, Any]]): The comparison of a served row, None for the eager row.
        sample_identifier (int): The identifier of the image, given to every compared tile.

    Returns:
        torch.Tensor: The [N, K, S, S] class probabilities, on the device of the tiles.
    """
    probabilities = []
    for request_tiles in torch.split(tiles, split_size_or_sections=maximum_request_size):
        outputs = run_request(inputs={"images": convert_to_request_array(tensor=request_tiles)}, run_model=run_model,
                              comparison=comparison, sample_identifiers=[sample_identifier] * request_tiles.shape[0])
        probabilities.append(outputs[probability_index])
    return torch.from_numpy(np.concatenate(probabilities, axis=0)).to(device=tiles.device)


def predict_exported_batch(batch: Dict[str, Any], trainer: BaseTrainer, run_model: Callable[..., List[np.ndarray]],
                           output_names: List[str], maximum_request_size: int,
                           comparison: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replaces predict_batch of the trainer for one row: the test batch is sent to the model of the row as a deployed
    client sends it, and its outputs are turned into the predictions of the task (convert_inference_outputs). The
    images go in requests of at most maximum_request_size images, the frames of every sequence alone and without their
    padding (a deployed sequence model taking no sequence mask), the frame probabilities of the padding being left at
    0, the two members of every pair of the comparison alone in the same way, and the tiles of every image of a
    segmentation run trained on tiles in requests of at most maximum_request_size tiles, averaged as in its
    evaluation. Every request of a served row is also compared with the eager model (comparison).

    Args:
        batch (Dict[str, Any]): The prepared test batch.
        trainer (BaseTrainer): The trainer of the run.
        run_model (Callable[..., List[np.ndarray]]): Runs one request ('inputs' keyword), the eager or served model.
        output_names (List[str]): The output names of the model, in the order run_model returns them.
        maximum_request_size (int): The largest batch size of a request.
        comparison (Optional[Dict[str, Any]]): The comparison of a served row, None for the eager row.

    Returns:
        Dict[str, Any]: The predictions of the batch.
    """
    task, images = trainer.TASK_NAME, batch["images"]
    sample_identifiers = [int(sample_identifier) for sample_identifier in batch[SAMPLE_IDENTIFIERS_KEY]]

    # Sequences: the real frames of every sequence in a request of its own, the frame probabilities scattered back
    # into the padded batch
    if task in SEQUENCE_SAMPLE_TASKS:
        sequence_outputs = [run_request(inputs={"images": extract_real_frames(
            frames=images, sequence_mask=batch["sequence_mask"], sample_index=sample_index)}, run_model=run_model,
            comparison=comparison, sample_identifiers=[sample_identifier])[0][0]
            for sample_index, sample_identifier in enumerate(sample_identifiers)]
        if task not in FRAME_PROBABILITY_TASKS:
            return convert_inference_outputs(trainer=trainer, outputs={"class_probabilities": np.stack(
                sequence_outputs)})
        probabilities = np.zeros(shape=(images.shape[0], images.shape[1], sequence_outputs[0].shape[-1]),
                                 dtype=np.float32)
        for sample_index, sequence_probabilities in enumerate(sequence_outputs):
            probabilities[sample_index, :sequence_probabilities.shape[0]] = sequence_probabilities
        return convert_inference_outputs(trainer=trainer, outputs={"class_probabilities": probabilities})

    # Pairs: the real frames of the before and after members of every pair in a request of its own
    if task == "sequence_comparison":
        before_frames, after_frames = split_pair_members(pair_tensor=images)
        before_mask, after_mask = split_pair_members(pair_tensor=batch["sequence_mask"])
        pair_outputs = [run_request(inputs={
            "before_images": extract_real_frames(frames=before_frames, sequence_mask=before_mask,
                                                 sample_index=sample_index),
            "after_images": extract_real_frames(frames=after_frames, sequence_mask=after_mask,
                                                sample_index=sample_index)},
            run_model=run_model, comparison=comparison, sample_identifiers=[sample_identifier])[0][0]
            for sample_index, sample_identifier in enumerate(sample_identifiers)]
        return convert_inference_outputs(trainer=trainer, outputs={"class_probabilities": np.stack(pair_outputs)})

    # Segmentation trained on tiles: the tiles of every image through the model of the row, averaged as in training
    if task == "semantic_segmentation" and trainer.tiling_configuration["activated"]:
        probability_index = output_names.index("class_probabilities")
        image_probabilities = [segment_image_by_tiles(
            image=image, predict_tile_probabilities=functools.partial(
                predict_exported_tiles, run_model=run_model, probability_index=probability_index,
                maximum_request_size=maximum_request_size, comparison=comparison,
                sample_identifier=sample_identifier),
            tile_size=trainer.input_image_size, overlap=trainer.tiling_configuration["overlap"],
            tiles_per_batch=maximum_request_size) for image, sample_identifier in zip(images, sample_identifiers)]
        return {"class_probabilities": torch.stack(tensors=image_probabilities)}

    # Images: requests of at most maximum_request_size images, their outputs joined in the order of the batch
    request_outputs = [run_request(inputs={"images": convert_to_request_array(tensor=request_images)},
                                   run_model=run_model, comparison=comparison,
                                   sample_identifiers=sample_identifiers[request_start:request_start +
                                                                         request_images.shape[0]])
                       for request_start, request_images in zip(
                           range(0, images.shape[0], maximum_request_size),
                           torch.split(images, split_size_or_sections=maximum_request_size))]
    outputs = {output_name: np.concatenate([request_output[output_index] for request_output in request_outputs],
                                           axis=0)
               for output_index, output_name in enumerate(output_names)}
    return convert_inference_outputs(trainer=trainer, outputs=outputs)


def evaluate_parity_row(trainer: BaseTrainer, run_model: Callable[..., List[np.ndarray]], output_names: List[str],
                        maximum_request_size: int, iteration: int, row_folder: Path,
                        comparison: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Scores one row on the whole test split with the test evaluation of the trainer, its predict_batch running the
    model of the row (predict_exported_batch), its evaluation files written to the folder of the row. The detectors
    keep their deployment score threshold (the one written into their export) instead of the lower threshold of the
    training evaluation, as the exported detector returns only those detections.

    Args:
        trainer (BaseTrainer): The trainer of the run, from build_evaluation_trainer.
        run_model (Callable[..., List[np.ndarray]]): Runs one request ('inputs' keyword), the eager or served model.
        output_names (List[str]): The output names of the model, in the order run_model returns them.
        maximum_request_size (int): The largest batch size of a request.
        iteration (int): The exported iteration, which names the iteration folder of the evaluation.
        row_folder (Path): The folder of the evaluation files of the row.
        comparison (Optional[Dict[str, Any]]): The comparison of a served row with the eager model, None for the eager
            row.

    Returns:
        Dict[str, Any]: The 'scalars', 'headline_metric', 'percent_metrics' and 'evaluated_samples' of the evaluation,
            and its 'duration_seconds'.
    """
    point_trainer_to_row(trainer=trainer, row_folder=row_folder)
    trainer.predict_batch = functools.partial(predict_exported_batch, trainer=trainer, run_model=run_model,
                                              output_names=output_names, maximum_request_size=maximum_request_size,
                                              comparison=comparison)
    if isinstance(trainer, DetectionTrainer):
        trainer.evaluation_score_threshold = trainer.deployment_score_threshold

    # The k-nearest neighbour reference images of a served self-supervised row recorded under their own split
    if isinstance(trainer, SelfSupervisedTrainer) and comparison is not None:
        trainer.encode_reference_images = functools.partial(encode_named_reference_images, trainer=trainer,
                                                            comparison=comparison)
    evaluation_start = time.time()
    results = trainer.evaluate_test_split(iteration=iteration)
    return {"scalars": {name: float(value) for name, value in results["scalars"].items()},
            "headline_metric": results["headline_metric"], "percent_metrics": list(results["percent_metrics"]),
            "evaluated_samples": results["evaluated_samples"], "duration_seconds": time.time() - evaluation_start}


def summarise_sample_agreement(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarises the sample comparison of a served row: the number of compared samples, those within the tolerances of
    the export specification, the largest and mean largest difference, and the samples of largest difference.

    Args:
        records (List[Dict[str, Any]]): The records of compare_request_outputs.

    Returns:
        Dict[str, Any]: The 'compared_samples', 'within_tolerance', 'largest_difference',
            'mean_largest_difference' and the 'worst_samples' records.
    """
    differences = [record["largest_difference"] for record in records]
    return {"compared_samples": len(records),
            "within_tolerance": sum(record["within_tolerance"] for record in records),
            "largest_difference": max(differences, default=0.0),
            "mean_largest_difference": float(np.mean(differences)) if differences else 0.0,
            "worst_samples": sorted(records, key=lambda record: -record["largest_difference"])[:WORST_SAMPLES_SHOWN]}


def read_reference_scalars(experiment_folder: Path, iteration: int, use_averaged_weights: bool) -> Dict[str, Any]:
    """
    Reads the scalars of the stored evaluation of the exported iteration and weights, the reference the eager row is
    expected to reproduce: the evaluation_summary.csv of '<run>/Checkpoint_Evaluation/' (scripts/evaluation/
    evaluate_checkpoint.py, on the test split of the run), else the one of '<run>/Evaluation/' (the training), the
    averaged weights read from their 'ema_' columns.

    Args:
        experiment_folder (Path): The exported run folder.
        iteration (int): The exported iteration.
        use_averaged_weights (bool): Whether the moving average of the weights is exported.

    Returns:
        Dict[str, Any]: The 'source' summary file and its 'scalars', or None and no scalar when no stored evaluation
            holds the iteration.
    """
    column_prefix = AVERAGED_WEIGHTS_COLUMN_PREFIX if use_averaged_weights else ""
    for evaluation_folder_name in [CHECKPOINT_EVALUATION_FOLDER_NAME, EVALUATION_FOLDER_NAME]:
        summary_path = experiment_folder / evaluation_folder_name / "evaluation_summary.csv"
        columns, rows = read_evaluation_summary(summary_path=summary_path)
        iteration_rows = [row for row in rows if row[ITERATION_COLUMN] == str(iteration)]
        if not iteration_rows:
            continue

        # Columns of the weights of the export, their prefix removed, the empty cells left out
        scalars = {column[len(column_prefix):]: float(iteration_rows[-1][column]) for column in columns
                   if column != ITERATION_COLUMN and column.startswith(column_prefix) and iteration_rows[-1][column]
                   and (use_averaged_weights or not column.startswith(AVERAGED_WEIGHTS_COLUMN_PREFIX))}
        if scalars:
            return {"source": str(summary_path), "scalars": scalars}
    return {"source": None, "scalars": {}}


def describe_reference_difference(trainer: BaseTrainer, logit_temperature: float) -> Optional[str]:
    """
    Tells why the stored evaluation of the exported iteration cannot be compared with the rows, its values being then
    left out of the summary: the detectors are scored at their deployment score threshold, the stored evaluation at
    the lower evaluation one, and a classifier exported at another temperature than 1 has other thresholds, curves and
    areas than its stored evaluation.

    Args:
        trainer (BaseTrainer): The trainer of the run, before the rows change its score threshold.
        logit_temperature (float): The logit temperature built into the exported classifier, 1 for the other tasks.

    Returns:
        Optional[str]: The reason, None when the stored evaluation is comparable.
    """
    if isinstance(trainer, DetectionTrainer):
        return (f"Stored evaluation left out: the exported detector returns the detections scored at least "
                f"{trainer.deployment_score_threshold:g} (Detector.inference.score_threshold), while the stored "
                f"evaluation scores those above {trainer.evaluation_score_threshold:g} (evaluation.score_threshold).")
    if trainer.TASK_NAME == "classification" and logit_temperature != 1.0:
        return (f"Stored evaluation left out: the classifier is exported at the logit temperature "
                f"{logit_temperature:g}, while the stored evaluation scored its probabilities at a temperature of 1.")
    return None


def get_row_label(row_name: str, tensorrt_precision: Optional[str], torchscript_precision: str) -> str:
    """
    Names a row in the summary and the console: the eager model, TorchScript with the precision of its network when
    it is FP16, or ONNX with the runtime Triton serves it with.

    Args:
        row_name (str): EAGER_ROW_NAME or an export format.
        tensorrt_precision (Optional[str]): The TensorRT precision of the ONNX model, None for ONNX Runtime CUDA.
        torchscript_precision (str): The precision of the network traced into the TorchScript file, 'FP32' or 'FP16'.

    Returns:
        str: The label, e.g. 'ONNX (TensorRT FP16)' or 'TorchScript (FP16)'.
    """
    if row_name == EAGER_ROW_NAME:
        return "Eager"
    if row_name == "onnx":
        return f"ONNX ({'ONNX Runtime CUDA' if tensorrt_precision is None else f'TensorRT {tensorrt_precision}'})"
    # An FP32 TorchScript row keeps the plain label of the summaries written before the FP16 TorchScript export
    return FORMAT_LABELS[row_name] if torchscript_precision == "FP32" else f"{FORMAT_LABELS[row_name]} (FP16)"


def format_metric_change(change: float, in_percent: bool) -> str:
    """
    Writes the change of a metric of a served row against the eager row: in points for a rate, with four decimals
    otherwise.

    Args:
        change (float): The difference, served row minus eager row.
        in_percent (bool): Whether the metric is a rate written as a percentage.

    Returns:
        str: The change, e.g. '-0.52 points'.
    """
    return format_points(difference=change) if in_percent else f"{change:+.4f}"


def build_summary_markdown(summary: Dict[str, Any]) -> str:
    """
    Writes the parity summary read by a person: the exported checkpoint and test split, the installation, the random
    batch check of every format, the metrics of every row with their change against the eager row and the stored
    evaluation, the agreement of the samples of every served row with its samples of largest difference, and the rows
    that failed with their error.

    Args:
        summary (Dict[str, Any]): The summary of run_test_split_parity, with its 'installation'.

    Returns:
        str: The Markdown text.
    """
    evaluated_rows = [row for row in summary["rows"] if row["status"] == "evaluated"]
    eager_row = next(row for row in summary["rows"] if row["name"] == EAGER_ROW_NAME)
    percent_metrics = set(eager_row["percent_metrics"])
    format_labels = {export_format: get_row_label(row_name=export_format,
                                                  tensorrt_precision=summary["tensorrt_precision"],
                                                  torchscript_precision=summary["torchscript_precision"])
                     for export_format in summary["formats"]}
    lines = [f"# Export Parity · {Path(summary['run']).name}", "",
             f"- Checkpoint: `{summary['checkpoint']}` (iteration {summary['iteration']}, {summary['weights']})",
             f"- Test split: `{summary['test_data_folder']}`, {eager_row['evaluated_samples']}",
             f"- Formats: {', '.join(format_labels.values())}, maximum batch size {summary['maximum_batch_size']}, "
             f"version {summary['model_version']}",
             f"- Installation: {summary['installation']}", "", "## Random Batch Check", ""]

    # Random batch check of every format against the eager model, or against its eager FP16 copy for an FP16
    # TorchScript file, which decides the installation
    for export_format, mismatches in summary["random_batch_checks"].items():
        half_precision = export_format == "torchscript" and summary["torchscript_precision"] == "FP16"
        reference_text = ("the eager FP16 copy of the network, at tolerances loosened for FP16" if half_precision
                          else "the eager model")
        check_text = f"matches {reference_text}" if not mismatches else "mismatch: " + "; ".join(mismatches)
        lines.append(f"- {format_labels[export_format]}: {check_text}")

    # Metrics of every row, the headline first, with the change of every served row against the eager row
    reference_scalars = summary["reference"]["scalars"] if summary["reference"]["comparable"] else {}
    header = ["Metric"] + [cell for row in evaluated_rows for cell in
                           ([row["label"]] if row["name"] == EAGER_ROW_NAME else [row["label"], "Change"])]
    header += ["Stored Evaluation"] if reference_scalars else []
    lines += ["", "## Test Metrics", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    metric_names = [summary["headline_metric"]] + [name for name in eager_row["scalars"]
                                                   if name != summary["headline_metric"]]
    # The proposal scalars of a detector left out, its exported model returning no proposal
    proposal_names = [name for name in metric_names if summary["task"] in DETECTION_TASKS
                      and name.startswith(PROPOSAL_SCALAR_PREFIX)]
    metric_names = [name for name in metric_names if name not in proposal_names]
    for metric_name in metric_names:
        in_percent = metric_name in percent_metrics
        eager_value = eager_row["scalars"][metric_name]
        cells = [f"**{metric_name}**" if metric_name == summary["headline_metric"] else metric_name]
        for row in evaluated_rows:
            value = row["scalars"][metric_name]
            cells.append(format_report_value(value=value, in_percent=in_percent))
            if row["name"] != EAGER_ROW_NAME:
                cells.append(format_metric_change(change=value - eager_value, in_percent=in_percent))
        if reference_scalars:
            cells.append(format_report_value(value=reference_scalars.get(metric_name), in_percent=in_percent))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [""] + [f"- {note}" for note in summary["reference"]["notes"]]
    if proposal_names:
        lines.append(f"- {len(proposal_names)} proposal scalars left out (Proposal Recall, Proposals Per Image): the "
                     f"exported detector returns no proposal, they read 0 in every row ({SUMMARY_JSON_NAME} keeps "
                     f"them).")
    if summary["reference"]["source"] is not None and summary["reference"]["comparable"]:
        lines.append(f"- Stored evaluation: `{summary['reference']['source']}`")

    # Agreement of the samples of every served row with the eager model, then the samples of largest difference
    served_rows = [row for row in evaluated_rows if row["agreement"] is not None]
    if served_rows:
        lines += ["", "## Sample Agreement", "",
                  f"Every request of a served row also went through the eager model. Within tolerances: the outputs "
                  f"agree within the FP32 tolerances of the random batch check of the export. Largest difference: "
                  f"{summary['difference_description']}.", "",
                  "| Row | Compared Samples | Within Tolerances | Largest Difference | Mean Largest Difference |",
                  "|---|---|---|---|---|"]
        for row in served_rows:
            agreement = row["agreement"]
            within_share = agreement["within_tolerance"] / max(1, agreement["compared_samples"])
            lines.append(f"| {row['label']} | {agreement['compared_samples']} | {agreement['within_tolerance']} "
                         f"({format_report_value(value=within_share, in_percent=True)}) | "
                         f"{agreement['largest_difference']:.3e} | {agreement['mean_largest_difference']:.3e} |")
        for row in served_rows:
            lines += ["", f"### Largest Differences · {row['label']}", "",
                      "| Sample | Identifier | Largest Difference | Within Tolerances | Mismatches |",
                      "|---|---|---|---|---|"]
            # The k-nearest neighbour reference images of a self-supervised run marked as such
            lines += [f"| {record['sample']}"
                      f"{'' if record['split'] == TEST_SPLIT_NAME else ' (' + record['split'] + ')'} | "
                      f"{record['sample_identifier']} | {record['largest_difference']:.3e} | "
                      f"{'yes' if record['within_tolerance'] else 'no'} | {record['mismatches'] or '—'} |"
                      for record in row["agreement"]["worst_samples"]]

    # Rows whose model could not be served or evaluated
    failed_rows = [row for row in summary["rows"] if row["status"] == "failed"]
    if failed_rows:
        lines += ["", "## Failed Rows", ""] + [f"- {row['label']}: {row['error']}" for row in failed_rows]
    return "\n".join(lines) + "\n"


def write_parity_summary(parity_folder: Path, summary: Dict[str, Any],
                         sample_records: Dict[str, List[Dict[str, Any]]]) -> List[Path]:
    """
    Writes the summary of a parity into its folder: parity_summary.md for a person, parity_summary.json for the
    scripts, and parity_samples.csv with the comparison of every sample of every served row.

    Args:
        parity_folder (Path): The parity folder, from get_parity_folder.
        summary (Dict[str, Any]): The summary of run_test_split_parity, with its 'installation'.
        sample_records (Dict[str, List[Dict[str, Any]]]): The records of compare_request_outputs of every served row,
            keyed by row name.

    Returns:
        List[Path]: The Markdown and JSON summaries, copied next to the installed models by the export.
    """
    markdown_path, json_path = parity_folder / SUMMARY_MARKDOWN_NAME, parity_folder / SUMMARY_JSON_NAME
    markdown_path.write_text(data=build_summary_markdown(summary=summary))
    json_path.write_text(data=json.dumps(obj=summary, indent=2, default=str))
    with open(parity_folder / SAMPLES_FILE_NAME, mode="w", newline="") as samples_file:
        writer = csv.DictWriter(samples_file, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for row_name, records in sample_records.items():
            for record in records:
                writer.writerow({"row": row_name, **record})
    return [markdown_path, json_path]


def print_parity_headlines(summary: Dict[str, Any]) -> None:
    """
    Prints the headline metric of every row with its change against the eager row, a change of at least
    HIGHLIGHTED_HEADLINE_CHANGE in yellow and a failed row in red, for the eye only.

    Args:
        summary (Dict[str, Any]): The summary of run_test_split_parity.
    """
    headline_metric = summary["headline_metric"]
    eager_row = next(row for row in summary["rows"] if row["name"] == EAGER_ROW_NAME)
    in_percent = headline_metric in eager_row["percent_metrics"]
    eager_value = eager_row["scalars"][headline_metric]
    print_color(color="blue", output=f"Test Split Parity : {headline_metric}", add_separators=True)
    for row in summary["rows"]:
        if row["status"] == "failed":
            print_color(color="red", output=f"{row['label']} : failed, {row['error']}", indent=1)
            continue
        value = row["scalars"][headline_metric]
        value_text = format_report_value(value=value, in_percent=in_percent)
        if row["name"] == EAGER_ROW_NAME:
            print_color(color="green", output=f"{row['label']} : {value_text}", indent=1)
            continue
        change = value - eager_value
        agreement = row["agreement"]
        change_text = format_metric_change(change=change, in_percent=in_percent)
        print_color(color="yellow" if abs(change) >= HIGHLIGHTED_HEADLINE_CHANGE else "green",
                    output=f"{row['label']} : {value_text} ({change_text}), {agreement['within_tolerance']} of "
                           f"{agreement['compared_samples']} samples within the tolerances", indent=1)


def run_test_split_parity(experiment_folder: Path, checkpoint_path: Path, iteration: int, use_averaged_weights: bool,
                          inference_model: nn.Module, specification: Dict[str, Any], staged_repository: Path,
                          engine_cache_folder: Path, formats: List[str], tensorrt_precision: Optional[str],
                          torchscript_precision: str, maximum_batch_size: int, model_version: int,
                          logit_temperature: float,
                          random_batch_checks: Dict[str, List[str]]) -> Tuple[Dict[str, Any],
                                                                              Dict[str, List[Dict[str, Any]]]]:
    """
    Scores the eager inference model and every exported format, served by a temporary Triton container on the staged
    models with their exported configuration, on the whole test split of the run, and gathers the summary of the
    parity. The eager row comes first, the reference; every served row then compares each of its requests with the
    eager model. A served row whose model Triton fails to load or to run (a TensorRT engine crashing the server, for
    instance) is recorded as failed with its error, the server is started again, and the other rows go on, so that
    every format is evaluated whatever the others gave. The folder of an earlier parity of the same export (checkpoint,
    weights, temperature and precisions, get_parity_folder) is replaced.

    Args:
        experiment_folder (Path): The exported run folder.
        checkpoint_path (Path): The exported checkpoint.
        iteration (int): The exported iteration.
        use_averaged_weights (bool): Whether the moving average of the weights is exported.
        inference_model (nn.Module): The eager inference model of the export, on the GPU, with the exported weights
            (and logit temperature).
        specification (Dict[str, Any]): The export specification of the task.
        staged_repository (Path): The model repository of the staged export, served read only.
        engine_cache_folder (Path): The folder receiving the TensorRT engines of the served ONNX model.
        formats (List[str]): The exported formats.
        tensorrt_precision (Optional[str]): The TensorRT precision of the ONNX model, None for ONNX Runtime CUDA.
        torchscript_precision (str): The precision of the network traced into the TorchScript file, 'FP32' or 'FP16'.
        maximum_batch_size (int): The largest batch size of the exported models, the size of the test batches.
        model_version (int): The Triton version of the exported models.
        logit_temperature (float): The logit temperature built into an exported classifier, 1 for the other tasks.
        random_batch_checks (Dict[str, List[str]]): The mismatches of the random batch check of every format.

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]: The summary of the parity, with its 'parity_folder'
            and without its 'installation', and the records of compare_request_outputs of every served row, keyed by
            row name.
    """
    experiment_configuration, _ = load_experiment_configuration(
        configuration_path=experiment_folder / "training_configuration.yaml")
    parity_folder = get_parity_folder(experiment_folder=experiment_folder, checkpoint_path=checkpoint_path,
                                      use_averaged_weights=use_averaged_weights, task=experiment_configuration["task"],
                                      logit_temperature=logit_temperature, formats=formats,
                                      tensorrt_precision=tensorrt_precision,
                                      torchscript_precision=torchscript_precision)
    if parity_folder.exists():
        shutil.rmtree(path=parity_folder)
    parity_folder.mkdir(parents=True)
    test_data_folder = Path(experiment_configuration["test_data_folder"])
    print_color(color="blue", output=f"Test Split Parity : {format_path_link(path=test_data_folder)}",
                add_separators=True)

    # Trainer of the run, its evaluation files in the folder of every row, and the reason its stored evaluation cannot
    # be compared, read before the rows set the detectors at their deployment threshold
    row_evaluations_folder = parity_folder / ROW_EVALUATIONS_FOLDER_NAME
    trainer = build_evaluation_trainer(experiment_folder=experiment_folder, test_data_folder=test_data_folder,
                                       evaluation_folder=parity_folder,
                                       standalone_evaluation_folder=ROW_EVALUATIONS_FOLDER_NAME,
                                       batch_size=maximum_batch_size)
    reference_note = describe_reference_difference(trainer=trainer, logit_temperature=logit_temperature)
    output_names = [output["name"] for output in specification["outputs"]]
    run_eager_model = functools.partial(run_local_inference, inference_model=inference_model, decoding_settings=None)

    # Reference: the eager inference model of the export in this process
    eager_label = get_row_label(row_name=EAGER_ROW_NAME, tensorrt_precision=None, torchscript_precision="FP32")
    print_color(color="blue", output=f"Row : {eager_label}", indent=1)
    eager_results = evaluate_parity_row(trainer=trainer, run_model=run_eager_model, output_names=output_names,
                                        maximum_request_size=maximum_batch_size, iteration=iteration,
                                        row_folder=row_evaluations_folder / EAGER_ROW_NAME, comparison=None)
    rows = [{"name": EAGER_ROW_NAME, "label": eager_label, "status": "evaluated", "error": None, "agreement": None,
             **eager_results}]
    sample_records = {}

    # Served rows: one Triton container on free ports serving the staged models one at a time
    server_ports = find_free_ports(number_ports=3)
    server = build_triton_server(container_name=f"{PARITY_CONTAINER_PREFIX}-{os.getpid()}",
                                 http_port=server_ports[0], grpc_port=server_ports[1], metrics_port=server_ports[2])
    start_triton_server(server=server, repository=staged_repository, engine_cache_folder=engine_cache_folder,
                        environment={})
    # The server log is kept and the container removed whatever ends the rows
    try:
        for export_format in formats:
            row_label = get_row_label(row_name=export_format, tensorrt_precision=tensorrt_precision,
                                      torchscript_precision=torchscript_precision)
            model_name = f"{specification['model_name']}_{export_format}"
            print_color(color="blue", output=f"Row : {row_label}", indent=1)
            comparison = create_output_comparison(task=trainer.TASK_NAME, run_reference=run_eager_model,
                                                  specification=specification,
                                                  name_sample=trainer.get_test_sample_name)
            try:
                load_served_model(server=server, model_name=model_name, configuration=None)
                client = triton_client.InferenceServerClient(url=server["grpc_address"])
                # Served names of the outputs ('<name>__<index>' for TorchScript), in the order of output_names
                served_output_names = get_triton_output_names(client=client, model_name=model_name,
                                                              output_names=output_names)
                row_results = evaluate_parity_row(
                    trainer=trainer, run_model=functools.partial(run_triton_inference, client=client,
                                                                 model_name=model_name,
                                                                 output_names=served_output_names,
                                                                 shared_memory_handle=None, region_name=None),
                    output_names=output_names, maximum_request_size=maximum_batch_size, iteration=iteration,
                    row_folder=row_evaluations_folder / export_format, comparison=comparison)
                unload_served_model(server=server, model_name=model_name)
                rows.append({"name": export_format, "label": row_label, "status": "evaluated", "error": None,
                             "agreement": summarise_sample_agreement(records=comparison["records"]), **row_results})
                sample_records[export_format] = comparison["records"]

            # A model Triton refuses to load or fails to run is recorded with its error, the server, which it may
            # have crashed, started again with the log of the failure kept for the next rows
            except (TritonServerError, InferenceServerException, urllib.error.URLError) as error:
                error_text = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
                print_color(color="red", output=f"{row_label} failed : {error_text}", indent=1)
                rows.append({"name": export_format, "label": row_label, "status": "failed", "error": error_text,
                             "agreement": None, "scalars": {}, "headline_metric": eager_results["headline_metric"],
                             "percent_metrics": eager_results["percent_metrics"], "evaluated_samples": None,
                             "duration_seconds": None})
                stop_triton_server(server=server, log_path=parity_folder / f"triton_server_{export_format}_failure.log")
                start_triton_server(server=server, repository=staged_repository,
                                    engine_cache_folder=engine_cache_folder, environment={})
    finally:
        stop_triton_server(server=server, log_path=parity_folder / SERVER_LOG_NAME)

    # Stored evaluation of the exported iteration, compared with the rows unless the export scores otherwise
    reference = read_reference_scalars(experiment_folder=experiment_folder, iteration=iteration,
                                       use_averaged_weights=use_averaged_weights)
    reference_notes = [reference_note] if reference_note is not None else []
    if reference["source"] is None:
        reference_notes.append(f"No stored evaluation of iteration {iteration} in {CHECKPOINT_EVALUATION_FOLDER_NAME} "
                               f"or {EVALUATION_FOLDER_NAME}.")
    if trainer.TASK_NAME == "self_supervised":
        reference_notes.append("The exported encoder returns L2 normalised representations: the rank metrics of the "
                               "rows differ from the stored evaluation of the raw representations by construction.")
    summary = {"run": str(experiment_folder), "task": trainer.TASK_NAME, "checkpoint": str(checkpoint_path),
               "iteration": iteration, "weights": "EMA weights" if use_averaged_weights else "trained weights",
               "test_data_folder": str(test_data_folder), "formats": formats,
               "tensorrt_precision": tensorrt_precision, "torchscript_precision": torchscript_precision,
               "maximum_batch_size": maximum_batch_size,
               "model_version": model_version, "logit_temperature": logit_temperature,
               "headline_metric": eager_results["headline_metric"], "random_batch_checks": random_batch_checks,
               "difference_description": DIFFERENCE_DESCRIPTIONS[trainer.TASK_NAME],
               "reference": {**reference, "comparable": reference_note is None and reference["source"] is not None,
                             "notes": reference_notes},
               "rows": rows, "created_at": datetime.now().isoformat(timespec="seconds"),
               "parity_folder": str(parity_folder)}
    print_parity_headlines(summary=summary)
    del trainer
    torch.cuda.empty_cache()
    return summary, sample_records
