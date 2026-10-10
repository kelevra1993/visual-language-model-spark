"""
Export specifications of every task: the inference model wrapping a trained model, the input and output names and
dynamic dimensions, the random input batches, and the comparisons of the exported outputs with the eager model. Read
by deployment/export_model.py, the Triton configuration and the deployment utilities.
"""

import numpy as np
import torch

from torch import nn
from typing import Any, Dict, List, Optional, Sequence

from vision_training.models.classification.inference_classifier import InferenceClassifier
from vision_training.models.detection.inference_detector import InferenceDetector
from vision_training.models.instance_segmentation.inference_instance_segmenter import InferenceInstanceSegmenter
from vision_training.models.segmentation.inference_segmenter import InferenceSegmenter
from vision_training.models.sequence.inference_sequence_comparator import InferenceSequenceComparator
from vision_training.models.sequence.inference_sequence_model import InferenceSequenceModel
from vision_training.models.task_model import TaskModel

# Inference model of every task, wrapping the trained model for deployment and local inference
INFERENCE_MODEL_CLASSES = {"object_detection": InferenceDetector, "instance_segmentation": InferenceInstanceSegmenter,
                           "classification": InferenceClassifier,
                           "semantic_segmentation": InferenceSegmenter,
                           "sequence_classification": InferenceSequenceModel,
                           "sequence_labelling": InferenceSequenceModel,
                           "sequence_comparison": InferenceSequenceComparator,
                           "frame_selection": InferenceSequenceModel}

# Triton model name of every task, followed by '_<export format>' in the model repository (and by '_decoder' and
# '_<export format>_ensemble' for the decoder and the ensembles of the frame selector)
TASK_MODEL_NAMES = {"object_detection": "object_detector", "instance_segmentation": "instance_segmenter",
                    "classification": "image_classifier",
                    "semantic_segmentation": "semantic_segmenter", "sequence_classification": "sequence_classifier",
                    "sequence_labelling": "sequence_labeller", "sequence_comparison": "sequence_comparator",
                    "frame_selection": "frame_selector"}

# Tasks whose inputs are windows of frames [B, T, 3, H, W] instead of images [B, 3, H, W]
SEQUENCE_TASKS = ["sequence_classification", "sequence_labelling", "sequence_comparison", "frame_selection"]

# Tasks whose model gives the class probabilities of every frame [B, T, K] instead of every window or pair [B, K]
FRAME_OUTPUT_TASKS = ["sequence_labelling", "frame_selection"]

# Names of the inputs of the inference model of every task, in the order of the arguments of its forward method:
# they name the inputs of the ONNX exports, and match the argument names kept by the TorchScript exports, which the
# Triton PyTorch backend uses to pass every input to its argument
TASK_INPUT_NAMES = {"object_detection": ["images"], "instance_segmentation": ["images"], "classification": ["images"],
                    "semantic_segmentation": ["images"], "sequence_classification": ["images"],
                    "sequence_labelling": ["images"],
                    "sequence_comparison": ["before_images", "after_images"], "frame_selection": ["images"]}


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
    Turns the outputs of an inference model (a tuple of tensors for the detector and the segmenters, a single tensor
    for the classifiers and sequence models) into a list of arrays, the format compared by the export specifications.

    Args:
        outputs (torch.Tensor | Sequence[torch.Tensor]): The model outputs.

    Returns:
        List[np.ndarray]: The outputs as arrays on the CPU, in order.
    """
    output_tensors = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
    return [tensor.detach().cpu().numpy() for tensor in output_tensors]


def match_detections(reference_detections: Sequence[np.ndarray], candidate_detections: Sequence[np.ndarray],
                     score_tolerance: float) -> np.ndarray:
    """
    Pairs the detections of one image of two models (the eager inference model and an export), so that their outputs
    are compared independently of the order of the detections whose scores are tied: two detections with the same
    score in the eager model can differ by a rounding error in the export and come out in the other order. Every
    reference detection, from the most confident one, takes the unpaired candidate of the same label whose score lies
    within the tolerance and whose box is the closest.

    Args:
        reference_detections (Sequence[np.ndarray]): The [K, 4] boxes, [K] scores and [K] labels of the reference.
        candidate_detections (Sequence[np.ndarray]): The [K, 4] boxes, [K] scores and [K] labels of the candidate.
        score_tolerance (float): The largest score difference of two paired detections.

    Returns:
        np.ndarray: The [K] index of the candidate paired with every reference detection, -1 when none fits.
    """
    reference_boxes, reference_scores, reference_labels = reference_detections
    candidate_boxes, candidate_scores, candidate_labels = candidate_detections

    # Cost of every pair: the largest box coordinate difference, infinite for another label or a distant score
    score_differences = np.abs(reference_scores[:, None] - candidate_scores[None, :])
    box_differences = np.abs(reference_boxes[:, None, :] - candidate_boxes[None, :, :]).max(axis=-1)
    compatible_pairs = (reference_labels[:, None] == candidate_labels[None, :]) & (score_differences <= score_tolerance)
    pair_costs = np.where(compatible_pairs, box_differences, np.inf)

    # Greedy pairing from the most confident reference detection
    paired_indices = np.full(shape=(reference_scores.shape[0],), fill_value=-1, dtype=np.int64)
    paired_candidates = np.zeros(shape=(candidate_scores.shape[0],), dtype=bool)
    for reference_index in np.argsort(-reference_scores, kind="stable"):
        available_costs = np.where(paired_candidates, np.inf, pair_costs[reference_index])
        best_candidate = int(np.argmin(available_costs))
        if np.isfinite(available_costs[best_candidate]):
            paired_indices[reference_index] = best_candidate
            paired_candidates[best_candidate] = True
    return paired_indices


def compare_detections(reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray], box_tolerance: float,
                       score_tolerance: float) -> List[str]:
    """
    Compares the outputs of an exported detector with the eager inference detector, image by image, the detections
    being paired by match_detections independently of the order of the detections with tied scores.

    Args:
        reference (Sequence[np.ndarray]): The eager boxes, scores, labels and numbers of detections.
        candidate (Sequence[np.ndarray]): The exported model outputs, in the same order.
        box_tolerance (float): The largest accepted box coordinate difference in pixels.
        score_tolerance (float): The largest accepted score difference.

    Returns:
        List[str]: A description of every mismatch, empty when the outputs agree.
    """
    mismatches = []
    reference_boxes, reference_scores, reference_labels, reference_numbers = reference[:4]
    candidate_boxes, candidate_scores, candidate_labels, candidate_numbers = candidate[:4]

    if not np.array_equal(reference_numbers.reshape(-1), candidate_numbers.reshape(-1)):
        mismatches.append(f"numbers of detections differ: {reference_numbers.tolist()} vs {candidate_numbers.tolist()}")

    # Compare the paired detections of every image
    for image_index in range(reference_boxes.shape[0]):
        paired_indices = match_detections(
            reference_detections=[reference_boxes[image_index], reference_scores[image_index],
                                  reference_labels[image_index]],
            candidate_detections=[candidate_boxes[image_index], candidate_scores[image_index],
                                  candidate_labels[image_index]], score_tolerance=score_tolerance)
        if (paired_indices < 0).any():
            mismatches.append(f"image {image_index}: {int((paired_indices < 0).sum())} detections without a "
                              f"candidate of the same label and score")
            continue
        box_difference = np.abs(reference_boxes[image_index] - candidate_boxes[image_index][paired_indices]).max()
        if box_difference > box_tolerance:
            mismatches.append(f"image {image_index}: box difference {box_difference:.2e} px")
    return mismatches


def compare_instance_segmentations(reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray],
                                   box_tolerance: float, score_tolerance: float, mask_tolerance: float) -> List[str]:
    """
    Compares the outputs of an exported instance segmenter with the eager inference instance segmenter: the
    detections as for the detector, then the mask of every detection with the mask of its paired detection, within
    the mask tolerance.

    Args:
        reference (Sequence[np.ndarray]): The eager boxes, scores, labels, numbers of detections and masks.
        candidate (Sequence[np.ndarray]): The exported model outputs, in the same order.
        box_tolerance (float): The largest accepted box coordinate difference in pixels.
        score_tolerance (float): The largest accepted score difference.
        mask_tolerance (float): The largest accepted mask probability difference.

    Returns:
        List[str]: A description of every mismatch, empty when the outputs agree.
    """
    mismatches = compare_detections(reference=reference, candidate=candidate, box_tolerance=box_tolerance,
                                    score_tolerance=score_tolerance)
    if mismatches or reference[4].shape != candidate[4].shape:
        return mismatches + ([] if reference[4].shape == candidate[4].shape else
                             [f"mask shapes differ: {reference[4].shape} vs {candidate[4].shape}"])

    # Mask of every detection against the mask of its paired detection
    for image_index in range(reference[0].shape[0]):
        paired_indices = match_detections(
            reference_detections=[reference[0][image_index], reference[1][image_index], reference[2][image_index]],
            candidate_detections=[candidate[0][image_index], candidate[1][image_index], candidate[2][image_index]],
            score_tolerance=score_tolerance)
        mask_difference = np.abs(reference[4][image_index] - candidate[4][image_index][paired_indices]).max()
        if mask_difference > mask_tolerance:
            mismatches.append(f"image {image_index}: mask difference {mask_difference:.2e}")
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


def compare_segmentations(reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray], tolerance: float,
                          minimum_equal_share: float) -> List[str]:
    """
    Compares the outputs of an exported segmenter with those of the eager inference segmenter: the class probabilities
    of every pixel within the tolerance, and the class maps on almost every pixel, a pixel whose two most probable
    classes are tied up to the rounding of the export possibly taking the other class.

    Args:
        reference (Sequence[np.ndarray]): The eager class maps and class probabilities.
        candidate (Sequence[np.ndarray]): The exported model outputs, in the same order.
        tolerance (float): The largest accepted probability difference.
        minimum_equal_share (float): The smallest accepted share of pixels with the same class in both class maps.

    Returns:
        List[str]: A description of every mismatch, empty when the outputs agree.
    """
    mismatches = compare_probabilities(reference=reference[1:], candidate=candidate[1:], tolerance=tolerance)
    if reference[0].shape != candidate[0].shape:
        return mismatches + [f"class map shapes differ: {reference[0].shape} vs {candidate[0].shape}"]
    equal_share = float(np.mean(a=reference[0] == candidate[0]))
    if equal_share < minimum_equal_share:
        mismatches.append(f"class maps agree on {100.0 * equal_share:.3f}% of the pixels only")
    return mismatches


def build_input_specification(input_name: str, image_size: int, sequence_length: Optional[int],
                              input_index: int) -> Dict[str, Any]:
    """
    Describes one input of an exported model, read by the export (names and dynamic dimensions), the Triton
    configuration (one input line) and the checks of the exports (random batches): one image per sample for the image
    tasks, or one window of up to sequence length frames per sample for the sequence tasks. The batches the exports
    are checked on hold one frame less than the traced ones for the first input, and one more frame less for every
    following input, so that the inputs of the comparison model are checked with different numbers of frames.

    Args:
        input_name (str): The name of the input, from TASK_INPUT_NAMES.
        image_size (int): The side of the square network input in pixels.
        sequence_length (Optional[int]): The largest number of frames of a window, the sequence length of the
            training, or None for the image tasks.
        input_index (int): The position of the input among the inputs of the model.

    Returns:
        Dict[str, Any]: The 'name', 'dimensions' (without the batch one, -1 for the number of frames, which may change
            from one request to the next), Triton 'data_type', 'dynamic_frames' flag, 'maximum_frames' (1 for the
            image tasks), and the 'example_shape' and 'check_shape' of the batches traced at export and checked after.
    """
    image_dimensions = [3, image_size, image_size]

    # Images: traced on batches of 2 images and checked on batches of 3
    if sequence_length is None:
        return {"name": input_name, "dimensions": image_dimensions, "data_type": "TYPE_FP32", "dynamic_frames": False,
                "maximum_frames": 1, "example_shape": [2] + image_dimensions, "check_shape": [3] + image_dimensions}

    # Windows of frames: traced with the sequence length and checked with fewer frames, a different number per input
    check_frames = max(1, sequence_length - 1 - input_index)
    return {"name": input_name, "dimensions": [-1] + image_dimensions, "data_type": "TYPE_FP32",
            "dynamic_frames": True, "maximum_frames": sequence_length,
            "example_shape": [2, sequence_length] + image_dimensions,
            "check_shape": [3, check_frames] + image_dimensions}


def create_random_inputs(specification: Dict[str, Any], shape_name: str, maximum_batch_size: int,
                         device: torch.device, generator: Optional[torch.Generator]) -> Dict[str, torch.Tensor]:
    """
    Draws random float BGR inputs in [0, 255] for every input of an export specification, the batches traced at
    export or the batches the exports are compared with the eager model on, their batch size capped at the largest
    batch size of the exports. The inputs are keyed by name, in the order of the forward arguments of the inference
    model, so that they are passed by name to the eager and TorchScript models and to ONNX Runtime.

    Args:
        specification (Dict[str, Any]): The export specification, from get_export_specification.
        shape_name (str): "example_shape" for the traced batches, "check_shape" for the checked batches.
        maximum_batch_size (int): The largest batch size of the exports.
        device (torch.device): The device of the inputs.
        generator (Optional[torch.Generator]): The random generator on the device of the inputs, or None for the
            global one.

    Returns:
        Dict[str, torch.Tensor]: The random batch of every input, keyed by input name.
    """
    random_inputs = {}
    for input_specification in specification["inputs"]:
        shape = input_specification[shape_name]
        random_inputs[input_specification["name"]] = torch.rand(
            size=[min(shape[0], maximum_batch_size)] + shape[1:], generator=generator, device=device) * 255.0
    return random_inputs


def get_export_specification(task: str, inference_model: nn.Module) -> Dict[str, Any]:
    """
    Describes how the inference model of a task is exported and served, so that the export, the Triton configuration
    and the throughput measurement are written once for every task: the inputs in the order of the forward arguments
    of the inference model ('images', or 'before_images' and 'after_images' for the comparison), the outputs in the
    order the inference model returns them, the batches used to trace and check the exports, and how the exported
    outputs are compared with the eager ones.

    Args:
        task (str): The task of the inference model.
        inference_model (nn.Module): The inference model of the task, from build_inference_model.

    Returns:
        Dict[str, Any]: The specification:
            - 'model_name' (str): the Triton model name, followed by '_<export format>' in the repository;
            - 'inputs' (List[Dict[str, Any]]): the description of every input (see build_input_specification);
            - 'outputs' (List[Dict[str, Any]]): the 'name', 'dimensions' (without the batch one), Triton 'data_type'
              and 'reshape_to_scalar' flag of every output;
            - 'compare_outputs' (Callable) and 'comparison_tolerances' (Dict[str, float]): the comparison of the
              exported outputs with the eager ones, and its tolerances.
    """
    # Inputs of the task, windows of frames for the sequence tasks and images otherwise
    sequence_length = inference_model.sequence_length if task in SEQUENCE_TASKS else None
    inputs = [build_input_specification(input_name=input_name, image_size=inference_model.input_image_size,
                                        sequence_length=sequence_length, input_index=input_index)
              for input_index, input_name in enumerate(TASK_INPUT_NAMES[task])]

    # Object detection: a fixed number of detections per image, compared independently of the order of tied scores
    if task in ["object_detection", "instance_segmentation"]:
        maximum_detections = inference_model.maximum_detections
        detection_outputs = [{"name": "detection_boxes", "dimensions": [maximum_detections, 4],
                              "data_type": "TYPE_FP32", "reshape_to_scalar": False},
                             {"name": "detection_scores", "dimensions": [maximum_detections],
                              "data_type": "TYPE_FP32", "reshape_to_scalar": False},
                             {"name": "detection_labels", "dimensions": [maximum_detections],
                              "data_type": "TYPE_INT64", "reshape_to_scalar": False},
                             {"name": "number_detections", "dimensions": [1], "data_type": "TYPE_INT64",
                              "reshape_to_scalar": True}]
        if task == "object_detection":
            return {"model_name": TASK_MODEL_NAMES[task], "inputs": inputs, "outputs": detection_outputs,
                    "compare_outputs": compare_detections,
                    "comparison_tolerances": {"box_tolerance": 1e-2, "score_tolerance": 1e-4}}

        # Instance segmentation: the M x M mask probabilities of the detected class of every detection, pasted into
        # the image inside its box by the client
        mask_size = inference_model.mask_size
        mask_output = {"name": "detection_masks", "dimensions": [maximum_detections, mask_size, mask_size],
                       "data_type": "TYPE_FP32", "reshape_to_scalar": False}
        return {"model_name": TASK_MODEL_NAMES[task], "inputs": inputs, "outputs": detection_outputs + [mask_output],
                "compare_outputs": compare_instance_segmentations,
                "comparison_tolerances": {"box_tolerance": 1e-2, "score_tolerance": 1e-4, "mask_tolerance": 1e-3}}

    # Number of classes of the segmenter, the classifiers and the sequence models
    number_classes = inference_model.number_classes

    # Semantic segmentation: the class map and the class probabilities of every pixel at the network resolution
    if task == "semantic_segmentation":
        image_size = inference_model.input_image_size
        return {"model_name": TASK_MODEL_NAMES[task], "inputs": inputs,
                "outputs": [{"name": "class_map", "dimensions": [image_size, image_size], "data_type": "TYPE_INT64",
                             "reshape_to_scalar": False},
                            {"name": "class_probabilities", "dimensions": [number_classes, image_size, image_size],
                             "data_type": "TYPE_FP32", "reshape_to_scalar": False}],
                "compare_outputs": compare_segmentations,
                "comparison_tolerances": {"tolerance": 1e-4, "minimum_equal_share": 0.999}}

    # Classification and sequence tasks: the class probabilities of every image, window or pair, or of every frame
    # for the sequence labeller and the frame selector
    output_dimensions = [-1, number_classes] if task in FRAME_OUTPUT_TASKS else [number_classes]
    return {"model_name": TASK_MODEL_NAMES[task], "inputs": inputs,
            "outputs": [{"name": "class_probabilities", "dimensions": output_dimensions, "data_type": "TYPE_FP32",
                         "reshape_to_scalar": False}],
            "compare_outputs": compare_probabilities, "comparison_tolerances": {"tolerance": 1e-4}}
