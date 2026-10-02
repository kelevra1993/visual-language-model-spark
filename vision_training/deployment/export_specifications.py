import numpy as np
import torch

from torch import nn
from typing import Any, Dict, List, Sequence

from vision_training.models.classification.inference_classifier import InferenceClassifier
from vision_training.models.detection.inference_detector import InferenceDetector
from vision_training.models.sequence.inference_sequence_model import InferenceSequenceModel
from vision_training.models.task_model import TaskModel

# Inference model of every task, wrapping the trained model for deployment and local inference
INFERENCE_MODEL_CLASSES = {"object_detection": InferenceDetector, "classification": InferenceClassifier,
                           "sequence_classification": InferenceSequenceModel,
                           "sequence_labelling": InferenceSequenceModel}

# Triton model name of every task, followed by '_<export format>' in the model repository
TASK_MODEL_NAMES = {"object_detection": "object_detector", "classification": "image_classifier",
                    "sequence_classification": "sequence_classifier", "sequence_labelling": "sequence_labeller"}

# Tasks whose inputs are windows of frames [B, T, 3, H, W] instead of images [B, 3, H, W]
SEQUENCE_TASKS = ["sequence_classification", "sequence_labelling"]


def build_inference_model(task: str, model: TaskModel) -> nn.Module:
    """
    Wraps a trained model into the inference model of its task, in evaluation mode, the network exported to Triton
    and run by the local inference.

    Args:
        task (str): The task of the trained model.
        model (TaskModel): The trained model, from load_trained_model.

    Returns:
        nn.Module: The inference model of the task.
    """
    return INFERENCE_MODEL_CLASSES[task](model=model).eval()


def convert_outputs_to_numpy(outputs: torch.Tensor | Sequence[torch.Tensor]) -> List[np.ndarray]:
    """
    Turns the outputs of an inference model (a tuple of tensors for the detector, a single tensor for the classifiers
    and sequence models) into a list of arrays, the format compared by the export specifications.

    Args:
        outputs (torch.Tensor | Sequence[torch.Tensor]): The model outputs.

    Returns:
        List[np.ndarray]: The outputs as arrays on the CPU, in order.
    """
    output_tensors = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
    return [tensor.detach().cpu().numpy() for tensor in output_tensors]


def sort_detections(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Orders the detections of one image canonically (by score, then label and box coordinates) so that two exports
    can be compared even when detections with exactly the same score come out in a different order.

    Args:
        boxes (np.ndarray): The [K, 4] detection boxes.
        scores (np.ndarray): The [K] detection scores.
        labels (np.ndarray): The [K] detection labels.

    Returns:
        np.ndarray: The [K, 6] detections as (score, label, x1, y1, x2, y2) rows in canonical order.
    """
    detections = np.concatenate([scores[:, None], labels[:, None].astype(np.float64), boxes], axis=1)
    order = np.lexsort(keys=tuple(detections[:, column] for column in reversed(range(detections.shape[1]))))
    return detections[order]


def compare_detections(reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray], box_tolerance: float,
                       score_tolerance: float) -> List[str]:
    """
    Compares the outputs of an exported detector with the eager inference detector, image by image, independently of
    the order of detections with tied scores.

    Args:
        reference (Sequence[np.ndarray]): The eager boxes, scores, labels and numbers of detections.
        candidate (Sequence[np.ndarray]): The exported model outputs, in the same order.
        box_tolerance (float): The largest accepted box coordinate difference in pixels.
        score_tolerance (float): The largest accepted score difference.

    Returns:
        List[str]: A description of every mismatch, empty when the outputs agree.
    """
    mismatches = []
    reference_boxes, reference_scores, reference_labels, reference_numbers = reference
    candidate_boxes, candidate_scores, candidate_labels, candidate_numbers = candidate

    if not np.array_equal(reference_numbers.reshape(-1), candidate_numbers.reshape(-1)):
        mismatches.append(f"numbers of detections differ: {reference_numbers.tolist()} vs {candidate_numbers.tolist()}")

    # Compare the canonically ordered detections of every image
    for image_index in range(reference_boxes.shape[0]):
        reference_detections = sort_detections(boxes=reference_boxes[image_index], scores=reference_scores[image_index],
                                               labels=reference_labels[image_index])
        candidate_detections = sort_detections(boxes=candidate_boxes[image_index], scores=candidate_scores[image_index],
                                               labels=candidate_labels[image_index])
        score_difference = np.abs(reference_detections[:, 0] - candidate_detections[:, 0]).max()
        box_difference = np.abs(reference_detections[:, 2:] - candidate_detections[:, 2:]).max()
        labels_equal = np.array_equal(reference_detections[:, 1], candidate_detections[:, 1])
        if score_difference > score_tolerance or box_difference > box_tolerance or not labels_equal:
            mismatches.append(f"image {image_index}: score difference {score_difference:.2e}, box difference "
                              f"{box_difference:.2e} px, labels equal {labels_equal}")
    return mismatches


def compare_probabilities(reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray],
                          tolerance: float) -> List[str]:
    """
    Compares the class probabilities of an exported classifier or sequence model with those of the eager inference
    model.

    Args:
        reference (Sequence[np.ndarray]): The eager class probabilities, as a one-element sequence.
        candidate (Sequence[np.ndarray]): The exported model class probabilities, as a one-element sequence.
        tolerance (float): The largest accepted probability difference.

    Returns:
        List[str]: A description of the mismatch, empty when the outputs agree.
    """
    if reference[0].shape != candidate[0].shape:
        return [f"class probability shapes differ: {reference[0].shape} vs {candidate[0].shape}"]
    probability_difference = np.abs(reference[0] - candidate[0]).max()
    if probability_difference > tolerance:
        return [f"class probabilities differ by up to {probability_difference:.2e}"]
    return []


def get_export_specification(task: str, inference_model: nn.Module) -> Dict[str, Any]:
    """
    Describes how the inference model of a task is exported and served, so that the export, the Triton configuration
    and the throughput measurement are written once for every task: the input, which is always named 'images', the
    outputs in the order the inference model returns them, the batches used to trace and check the exports, and how
    the exported outputs are compared with the eager ones.

    Args:
        task (str): The task of the inference model.
        inference_model (nn.Module): The inference model of the task, from build_inference_model.

    Returns:
        Dict[str, Any]: The specification:
            - 'model_name' (str): the Triton model name, followed by '_<export format>' in the repository;
            - 'input_dimensions' (List[int]): the dimensions of 'images' without the batch one, -1 for the number of
              frames of the sequence tasks, which may change from one request to the next;
            - 'dynamic_frames' (bool): whether the second dimension of 'images' is a number of frames;
            - 'maximum_frames' (int): the largest number of frames, the sequence length of the training (1 for the
              image tasks);
            - 'outputs' (List[Dict[str, Any]]): the 'name', 'dimensions' (without the batch one), Triton 'data_type'
              and 'reshape_to_scalar' flag of every output;
            - 'example_shape' and 'check_shape' (List[int]): the shapes of the batch traced at export, and of the
              batch of another size (and another number of frames) the exports are checked on;
            - 'compare_outputs' (Callable) and 'comparison_tolerances' (Dict[str, float]): the comparison of the
              exported outputs with the eager ones, and its tolerances.
    """
    image_size = inference_model.input_image_size
    image_dimensions = [3, image_size, image_size]

    # Object detection: a fixed number of detections per image, compared independently of the order of tied scores
    if task == "object_detection":
        maximum_detections = inference_model.maximum_detections
        return {"model_name": TASK_MODEL_NAMES[task], "input_dimensions": image_dimensions, "dynamic_frames": False,
                "maximum_frames": 1,
                "outputs": [{"name": "detection_boxes", "dimensions": [maximum_detections, 4], "data_type": "TYPE_FP32",
                             "reshape_to_scalar": False},
                            {"name": "detection_scores", "dimensions": [maximum_detections], "data_type": "TYPE_FP32",
                             "reshape_to_scalar": False},
                            {"name": "detection_labels", "dimensions": [maximum_detections],
                             "data_type": "TYPE_INT64", "reshape_to_scalar": False},
                            {"name": "number_detections", "dimensions": [1], "data_type": "TYPE_INT64",
                             "reshape_to_scalar": True}],
                "example_shape": [2] + image_dimensions, "check_shape": [3] + image_dimensions,
                "compare_outputs": compare_detections,
                "comparison_tolerances": {"box_tolerance": 1e-2, "score_tolerance": 1e-4}}

    # Classification: the class probabilities of every image
    number_classes = inference_model.number_classes
    if task not in SEQUENCE_TASKS:
        return {"model_name": TASK_MODEL_NAMES[task], "input_dimensions": image_dimensions, "dynamic_frames": False,
                "maximum_frames": 1,
                "outputs": [{"name": "class_probabilities", "dimensions": [number_classes], "data_type": "TYPE_FP32",
                             "reshape_to_scalar": False}],
                "example_shape": [2] + image_dimensions, "check_shape": [3] + image_dimensions,
                "compare_outputs": compare_probabilities, "comparison_tolerances": {"tolerance": 1e-4}}

    # Sequence tasks: windows of any number of frames up to the sequence length, checked with one frame less than
    # traced, and the class probabilities of every window or of every frame
    sequence_length = inference_model.sequence_length
    output_dimensions = [-1, number_classes] if task == "sequence_labelling" else [number_classes]
    return {"model_name": TASK_MODEL_NAMES[task], "input_dimensions": [-1] + image_dimensions, "dynamic_frames": True,
            "maximum_frames": sequence_length,
            "outputs": [{"name": "class_probabilities", "dimensions": output_dimensions, "data_type": "TYPE_FP32",
                         "reshape_to_scalar": False}],
            "example_shape": [2, sequence_length] + image_dimensions,
            "check_shape": [3, max(1, sequence_length - 1)] + image_dimensions,
            "compare_outputs": compare_probabilities, "comparison_tolerances": {"tolerance": 1e-4}}
