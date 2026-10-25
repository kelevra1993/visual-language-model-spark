"""
Run of the trained keypoint detector on the CPU for the figures of the keypoint detection series of the tutorial
videos, the expensive step of tutorials/figures/keypoint_detection/figures.py (about two minutes on a busy machine),
reused between builds unless forced: the checkpoint of iteration 6000 of the synthetic keypoint run, on the test image
of the chapters with its ground truth, gives in one forward pass the heatmaps of both strides, the tags, the losses of
the image, the keypoint candidates (extract_keypoint_candidates, as in the evaluation and the exports) and, through
forward hooks, the shape of every tensor of the network and the parameters of its components. The arrays the figures
draw are kept for the channels of the classes of the image only.
"""

import numpy
import torch

from pathlib import Path
from torch import nn
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import read_reused_facts, round_fact, write_reused_facts
from tutorials.figures.keypoint_detection.keypoint_sources import (CHECKPOINT_ITERATION, REPOSITORY_CONFIGURATION_PATH,
                                                                  describe_sample_objects)
from tutorials.tutorial_builder.settings import THREAD_COUNT
from vision_training.data.keypoints.heatmap_targets import build_keypoint_targets
from vision_training.losses.keypoint_losses import compute_heatmap_loss, compute_tag_losses
from vision_training.models.keypoints.keypoint_detector import HEATMAP_STRIDE, extract_keypoint_candidates
from vision_training.models.model_loading import load_trained_model
from vision_training.trainers.checkpointing import get_checkpoint_path

# Files of the inference in the figures folder, the arrays and the facts that let the next build reuse them
INFERENCE_ARRAYS_NAME = "inference/keypoint_inference.npz"
INFERENCE_FACTS_NAME = "inference/keypoint_inference_facts.json"

# Exponents of the focal heatmap loss, those of compute_heatmap_loss, used to split its value into the part of the
# keypoint cells and the part of the other cells; the split is checked against compute_heatmap_loss
FOCAL_EXPONENT = 2.0
PENALTY_REDUCTION_EXPONENT = 4.0

# Largest relative difference allowed between the split heatmap loss and the value of compute_heatmap_loss
LOSS_SPLIT_TOLERANCE = 1e-4


class ShapeRecorder:
    """
    Forward hook keeping the shape of the input and output tensors of the modules of the detector, so that the
    architecture diagrams quote the shapes of the real network on the real image.

    Attributes:
        shapes (Dict[str, Dict[str, Any]]): The input and output shapes recorded under the name of every module.
    """

    def __init__(self) -> None:
        """
        Starts with no recorded shape.
        """
        self.shapes: Dict[str, Dict[str, Any]] = {}

    def attach(self, module: nn.Module, name: str) -> torch.utils.hooks.RemovableHandle:
        """
        Registers the hook on a module under a name.

        Args:
            module (nn.Module): The module whose tensors are recorded.
            name (str): The name of the module in the facts.

        Returns:
            torch.utils.hooks.RemovableHandle: The handle removing the hook.
        """
        module.recorded_name = name
        return module.register_forward_hook(hook=self.record)

    def record(self, module: nn.Module, inputs: Any, output: Any) -> None:
        """
        Keeps the shapes of the first input and of the output of a module, every tensor of a list or tuple.

        Args:
            module (nn.Module): The module that ran.
            inputs (Any): Its positional inputs.
            output (Any): Its output, a tensor, or a list or tuple of tensors.
        """
        self.shapes[module.recorded_name] = {"input": describe_shapes(value=inputs[0] if inputs else None),
                                             "output": describe_shapes(value=output)}


def describe_shapes(value: Any) -> Any:
    """
    Gives the shape of a tensor as a list, or the shapes of the tensors of a list or tuple, None for anything else.

    Args:
        value (Any): A tensor, a list or tuple of tensors, or None.

    Returns:
        Any: The shape as a list of integers, a list of such shapes, or None.
    """
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [describe_shapes(value=item) for item in value]
    return None


def split_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Splits the focal heatmap loss of one stride into the part of the keypoint cells (target 1) and the part of all
    the other cells, each divided by the number of keypoint cells like the loss, with the formulas of
    compute_heatmap_loss, and checks that both parts add up to the value of compute_heatmap_loss.

    Args:
        logits (torch.Tensor): The [B, C, h, w] heatmap logits.
        targets (torch.Tensor): The [B, C, h, w] target heatmaps of build_keypoint_targets.

    Returns:
        Dict[str, float]: The 'keypoint_cells' and 'other_cells' parts, the 'total', the number of 'positive_cells' and
            of 'cells'.

    Raises:
        ValueError: If the two parts do not add up to compute_heatmap_loss.
    """
    logits = logits.to(dtype=torch.float32)
    probabilities = torch.sigmoid(logits)
    positive_cells = targets == 1.0
    number_positives = positive_cells.sum().clamp(min=1).to(dtype=torch.float32)

    # Terms of the keypoint cells and of the others, as in compute_heatmap_loss
    positive_terms = -(1.0 - probabilities) ** FOCAL_EXPONENT * torch.nn.functional.logsigmoid(logits)
    negative_terms = (-(1.0 - targets) ** PENALTY_REDUCTION_EXPONENT * probabilities ** FOCAL_EXPONENT *
                      torch.nn.functional.logsigmoid(-logits))
    keypoint_part = float(positive_terms[positive_cells].sum() / number_positives)
    other_part = float(negative_terms[~positive_cells].sum() / number_positives)

    # Both parts against the loss of the repository
    total = float(compute_heatmap_loss(predicted_logits=logits, target_heatmaps=targets))
    if abs(keypoint_part + other_part - total) > LOSS_SPLIT_TOLERANCE * max(abs(total), 1.0):
        raise ValueError(f"The split heatmap loss {keypoint_part} + {other_part} differs from the value {total} of "
                         f"compute_heatmap_loss.")
    return {"keypoint_cells": round_fact(value=keypoint_part), "other_cells": round_fact(value=other_part),
            "total": round_fact(value=total), "positive_cells": int(positive_cells.sum()),
            "cells": int(targets.numel())}


def build_sample_targets(sample: Dict[str, Any], channel_table: numpy.ndarray, number_channels: int,
                         heatmap_sigma: float) -> Dict[str, Any]:
    """
    Builds the training targets of the test image with build_keypoint_targets, at the strides 4 and 2 of the detector
    with its higher resolution branch, as the forward pass of the training builds them.

    Args:
        sample (Dict[str, Any]): The dataset sample of the test image.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        number_channels (int): The number of heatmap channels.
        heatmap_sigma (float): 'KeypointHead.heatmap_sigma' of the run.

    Returns:
        Dict[str, Any]: The targets of build_keypoint_targets for a batch of this one image.
    """
    image_height, image_width = sample["images"].shape[1:]
    return build_keypoint_targets(keypoints=[sample["keypoints"]], labels=[sample["labels"]],
                                  channel_table=torch.from_numpy(channel_table), number_channels=number_channels,
                                  image_height=image_height, image_width=image_width,
                                  heatmap_strides=[HEATMAP_STRIDE, HEATMAP_STRIDE // 2], tag_stride=HEATMAP_STRIDE,
                                  heatmap_sigma=heatmap_sigma)


def attach_shape_recorders(model: nn.Module, recorder: ShapeRecorder) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Registers the shape recorder on the parts of the detector the architecture diagram shows: the stem, the
    transitions of every stage, the high resolution network, the prediction convolution and the parts of the higher
    resolution branch.

    Args:
        model (nn.Module): The keypoint detector.
        recorder (ShapeRecorder): The recorder.

    Returns:
        List[torch.utils.hooks.RemovableHandle]: The handles of the hooks.
    """
    handles = [recorder.attach(module=model.backbone, name="stem"),
               recorder.attach(module=model.high_resolution_network, name="high_resolution_network"),
               recorder.attach(module=model.keypoint_head["prediction_convolution"], name="prediction_convolution")]
    for stage_index, transition in enumerate(model.high_resolution_network.transitions):
        handles.append(recorder.attach(module=transition, name=f"transition_{stage_index + 1}"))
    branch = model.keypoint_head["higher_resolution_branch"]
    handles += [recorder.attach(module=branch.transposed_convolution, name="transposed_convolution"),
                recorder.attach(module=branch.refinement, name="refinement"),
                recorder.attach(module=branch.heatmap_head, name="higher_resolution_heatmap_head")]
    return handles


def run_keypoint_inference(output_folder: Path, run_folder: Path, sample_record: Dict[str, Any],
                           channel_table: numpy.ndarray, keypoint_categories: Dict[int, Dict[str, Any]],
                           configuration: Dict[str, Any], force: bool) -> Dict[str, Any]:
    """
    Runs the trained detector on the test image once, or reuses the previous run, and gives its facts: the shapes of
    the network, the parameters of its components, the losses of the image (split by keypoint and other cells), the
    tag losses, the candidates above the keypoint threshold. The arrays are saved in INFERENCE_ARRAYS_NAME.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the keypoint run.
        sample_record (Dict[str, Any]): The test image of load_test_sample.
        channel_table (numpy.ndarray): The channel of every keypoint of every class.
        keypoint_categories (Dict[int, Dict[str, Any]]): The keypoint definition of every class.
        configuration (Dict[str, Any]): The repository configuration, with the KeypointHead settings of the run.
        force (bool): Whether the inference runs again even when the previous build left its files.

    Returns:
        Dict[str, Any]: The facts of the inference, keyed by their names without the series prefix.
    """
    arrays_path = output_folder / INFERENCE_ARRAYS_NAME
    facts_path = output_folder / INFERENCE_FACTS_NAME
    reused_facts = read_reused_facts(facts_path=facts_path, required_paths=[arrays_path], force=force)
    if reused_facts is not None:
        return reused_facts

    # Trained detector of the checkpoint on the CPU, its threads limited so that the trainings keep the processors
    torch.set_num_threads(THREAD_COUNT)
    checkpoint_path = get_checkpoint_path(experiment_folder=run_folder, iteration=CHECKPOINT_ITERATION)
    model = load_trained_model(configuration_path=REPOSITORY_CONFIGURATION_PATH, checkpoint_path=checkpoint_path,
                               device=torch.device("cpu"), use_averaged_weights=False)
    recorder = ShapeRecorder()
    handles = attach_shape_recorders(model=model, recorder=recorder)

    # Heatmap logits of both strides, kept by hooks on the last convolutions of both heads
    logit_recorder = LogitRecorder()
    handles.append(model.keypoint_head["prediction_convolution"].register_forward_hook(hook=logit_recorder.record_head))
    handles.append(model.keypoint_head["higher_resolution_branch"].register_forward_hook(
        hook=logit_recorder.record_branch))

    # One forward pass with the ground truth: heatmaps, tags and losses, then the candidates of the evaluation
    sample = sample_record["sample"]
    images = sample["images"][None].to(dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor=images, ground_truth_keypoints=[sample["keypoints"]],
                        ground_truth_labels=[sample["labels"]])
        candidates = extract_keypoint_candidates(heatmaps=outputs["heatmaps"], tags=outputs["tags"],
                                                 higher_resolution_heatmaps=outputs["higher_resolution_heatmaps"],
                                                 maximum_candidates=model.maximum_candidates)
    for handle in handles:
        handle.remove()

    # Targets of the image, the heatmap loss of both strides split by kind of cell, and the tag losses
    head_configuration = configuration["KeypointHead"]
    targets = build_sample_targets(sample=sample, channel_table=channel_table, number_channels=model.number_channels,
                                   heatmap_sigma=head_configuration["heatmap_sigma"])
    number_channels = model.number_channels
    loss_splits = {"stride_4": split_heatmap_loss(logits=logit_recorder.head_logits[:, :number_channels],
                                                  targets=targets["heatmaps"][0]),
                   "stride_2": split_heatmap_loss(logits=logit_recorder.branch_logits, targets=targets["heatmaps"][1])}
    pull_loss, push_loss = compute_tag_losses(tag_maps=outputs["tags"], tag_indices=targets["tag_indices"],
                                              tag_valid=targets["tag_valid"], object_labels=targets["object_labels"])

    # Tags of every labelled keypoint of every object, read at its cell like the tag losses
    keypoint_tags = outputs["tags"].reshape(1, -1).gather(dim=1, index=targets["tag_indices"].reshape(1, -1)).reshape(
        targets["tag_indices"].shape)[0]
    objects = describe_sample_objects(sample=sample, keypoint_categories=keypoint_categories,
                                      channel_table=channel_table)
    for object_index, sample_object in enumerate(objects):
        valid = targets["tag_valid"][0, object_index]
        sample_object["tags"] = [round_fact(value=tag) for tag in keypoint_tags[object_index][valid].tolist()]

    # Channels of the classes of the image, whose heatmaps and tags the figures draw
    shown_channels = sorted({channel for sample_object in objects for channel in sample_object["channels"]})
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(file=arrays_path, channels=numpy.array(shown_channels),
                           heatmaps=outputs["heatmaps"][0, shown_channels].numpy(),
                           higher_resolution_heatmaps=outputs["higher_resolution_heatmaps"][0, shown_channels].numpy(),
                           tags=outputs["tags"][0, shown_channels].numpy(),
                           candidate_positions=candidates["candidate_positions"][0].numpy(),
                           candidate_scores=candidates["candidate_scores"][0].numpy(),
                           candidate_tags=candidates["candidate_tags"][0].numpy())

    # Facts of the network, its parameters, its losses on the image and its candidates
    keypoint_threshold = head_configuration["inference"]["keypoint_threshold"]
    facts = {"inference_shapes": recorder.shapes,
             "inference_parameters": {name: sum(parameter.numel() for parameter in component.parameters())
                                      for name, component in model.get_size_components().items()},
             "inference_heatmap_loss": round_fact(value=outputs["heatmap_loss"]),
             "inference_heatmap_loss_splits": loss_splits,
             "inference_tag_pull_loss": round_fact(value=pull_loss),
             "inference_tag_push_loss": round_fact(value=push_loss),
             "inference_weighted_tag_pull_loss": round_fact(value=outputs["tag_pull_loss"]),
             "inference_weighted_tag_push_loss": round_fact(value=outputs["tag_push_loss"]),
             "inference_objects": objects,
             "inference_candidates_above_threshold": int((candidates["candidate_scores"] > keypoint_threshold).sum()),
             "inference_candidate_count": int(candidates["candidate_scores"].numel()),
             "inference_checkpoint_iteration": CHECKPOINT_ITERATION}
    write_reused_facts(facts_path=facts_path, figure_paths=[arrays_path], facts=facts)
    return facts


class LogitRecorder:
    """
    Forward hooks keeping the heatmap logits of both heads of the detector, the quarter resolution prediction
    convolution (heatmap logits then tags) and the higher resolution branch, so that the heatmap loss of the image is
    split from the logits the loss reads.

    Attributes:
        head_logits (torch.Tensor): The [1, 2C, h, w] output of the prediction convolution.
        branch_logits (torch.Tensor): The [1, C, 2h, 2w] output of the higher resolution branch.
    """

    def __init__(self) -> None:
        """
        Starts with no recorded logit.
        """
        self.head_logits = None
        self.branch_logits = None

    def record_head(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        """
        Keeps the output of the prediction convolution.

        Args:
            module (nn.Module): The prediction convolution.
            inputs (Any): Its inputs.
            output (torch.Tensor): Its output, the heatmap logits followed by the tags.
        """
        self.head_logits = output.detach()

    def record_branch(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        """
        Keeps the output of the higher resolution branch.

        Args:
            module (nn.Module): The higher resolution branch.
            inputs (Any): Its inputs.
            output (torch.Tensor): Its heatmap logits.
        """
        self.branch_logits = output.detach()
