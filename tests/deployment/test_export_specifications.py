"""
Tests of the model exports of vision_training/deployment/export_model.py and export_specifications.py: the TorchScript
and ONNX exports of the recurrent comparison model and of the comparison model with sequence normalization match the
eager model, a second recurrent export keeps its dynamic frames, the ONNX outputs of the bidirectional sequence
labeller and frame selector keep their dynamic frames, the comparisons of the segmenter and keypoint detector outputs
report their differences, the inference models match the training path on full sequences, the logit temperature of
the classifier softens its probabilities in the eager model and both exports, the exports of the detector keep its
DIoU-NMS, its greedy NMS beyond the size at which batched_nms changes its algorithm and its selection of the proposals
level by level, its ONNX suppressions see the boxes of the image only, and the exports of the instance segmenter keep
the finer inputs of its mask heads.
"""

import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from pathlib import Path
from torch import nn
from typing import Optional

from vision_training.deployment.export_formats import (export_onnx, export_torchscript, find_fixed_dynamic_dimensions,
                                                       hide_known_export_warnings, restore_dynamic_output_frames)
from vision_training.deployment.export_specifications import (FRAME_OUTPUT_TASKS, TASK_INPUT_NAMES,
                                                              build_inference_model, compare_keypoints,
                                                              compare_segmentations,
                                                              convert_outputs_to_numpy, create_random_inputs,
                                                              get_export_specification)
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from tests.models.instance_segmentation.test_instance_segmenter import MIXED_MASK_HEAD_SETTINGS
from tests.testing_utilities import build_test_model as build_test_detector
from tests.fixtures.configuration_fixtures import (get_classification_model_configuration,
                                                   get_keypoint_model_configuration,
                                                   get_segmentation_model_configuration,
                                                   get_self_supervised_model_configuration,
                                                   get_sequence_comparison_model_configuration,
                                                   get_sequence_model_configuration)

# Largest batch size of the test exports, above the batch sizes of the traced and checked batches
TEST_MAXIMUM_BATCH_SIZE = 4


def build_test_model(task: str, temporal_type: Optional[str], fusion_type: Optional[str], global_pooling: str,
                     pre_activation: bool = False) -> TaskModel:
    """
    Builds a small untrained model of a task on the CPU, in evaluation mode, with seeded weights: bidirectional
    recurrent modules, a causal transformer for the sequence labeller, the selection settings of the frame selector,
    and a keypoint head with its higher resolution branch giving heatmaps and tags of a trained scale.

    Args:
        task (str): "classification", "semantic_segmentation", "sequence_classification", "sequence_labelling",
            "sequence_comparison", "frame_selection", "self_supervised" or "keypoint_detection".
        temporal_type (Optional[str]): The temporal module of the sequence models, None for the classifier, the
            segmenter, the self-supervised model and the keypoint detector.
        fusion_type (Optional[str]): The fusion of the members of the comparison model, None for the other tasks.
        global_pooling (str): The reduction of the final feature map, "average" or "max", or "flatten" for the
            classifier only.
        pre_activation (bool): Whether the residual shortcuts of the backbone of the classifier use the pre-activation
            placement (residual_nets_v2).

    Returns:
        TaskModel: The model of the task.
    """
    # Seed the weights and take the test configuration of the task, the classifier with the pooling and shortcuts asked
    torch.manual_seed(seed=0)
    if task == "semantic_segmentation":
        configuration, device, dtype = get_segmentation_model_configuration(upsampling="transposed")
    elif task == "self_supervised":
        configuration, device, dtype = get_self_supervised_model_configuration(method="simclr",
                                                                               global_pooling=global_pooling)
    elif task == "keypoint_detection":
        configuration, device, dtype = get_keypoint_model_configuration(higher_resolution=True, hidden_width=16)
    elif temporal_type is None:
        configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
        configuration["ClassificationHead"]["global_pooling"] = global_pooling
        configuration["Backbone"]["modules"]["residual_nets_v2"] = pre_activation
    else:
        # The sequence models: bidirectional recurrent modules, a causal transformer for the labeller, and the
        # selection settings read by the frame selector
        if task == "sequence_comparison":
            configuration, device, dtype = get_sequence_comparison_model_configuration(temporal_type=temporal_type,
                                                                                       fusion_type=fusion_type)
        else:
            configuration, device, dtype = get_sequence_model_configuration(temporal_type=temporal_type)
        configuration["TemporalModel"]["bidirectional"] = temporal_type in ["gru", "lstm"]
        configuration["TemporalModel"]["causal"] = task == "sequence_labelling" and temporal_type == "transformer"
        configuration["Selection"] = {"unselected_category": "unselected", "decoding": "hungarian",
                                      "frames_per_class": 1, "selection_loss_weight": 1.0}
    # Build the model of the task in evaluation mode, so that the exports trace its inference behaviour
    model = TASK_MODEL_CLASSES[task](configuration=configuration, device=device, dtype=dtype).eval()

    # The keypoint head starts with heatmaps of about 1e-5, below every tolerance, so both of its final convolutions,
    # behind the 3x3 hidden convolutions, are drawn at a trained scale
    if task == "keypoint_detection":
        nn.init.normal_(tensor=model.keypoint_head["prediction_convolution"][-1].weight, mean=0.0, std=25.0)
        nn.init.normal_(tensor=model.keypoint_head["higher_resolution_branch"].heatmap_head[-1].weight, mean=0.0,
                        std=25.0)
    return model


@pytest.mark.parametrize(argnames=("task", "temporal_type", "fusion_type", "global_pooling", "pre_activation"),
                         argvalues=[("sequence_comparison", "gru", "cross_attention", "average", False)])
def test_exports_match_the_eager_model(tmp_path: Path, task: str, temporal_type: Optional[str],
                                       fusion_type: Optional[str], global_pooling: str, pre_activation: bool) -> None:
    """
    Verifies the TorchScript and ONNX exports of the inference model of the recurrent comparison model with cross
    attention: both keep the input names of the specification and reproduce the eager class probabilities on a batch
    of another size than the traced one and with other numbers of frames, so that Triton can batch requests and
    receive sequences of any length. The before and after members of the comparison model are checked with different
    numbers of frames.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The task of the exported model.
        temporal_type (Optional[str]): The temporal module of the sequence models, None for the classifier and the
            segmenter.
        fusion_type (Optional[str]): The fusion of the members of the comparison model, None for the other tasks.
        global_pooling (str): The reduction of the final feature map of the model.
        pre_activation (bool): Whether the classifier uses pre-activation residual shortcuts.
    """
    hide_known_export_warnings()
    inference_model = build_inference_model(task=task, model=build_test_model(task=task, temporal_type=temporal_type,
                                                                              fusion_type=fusion_type,
                                                                              global_pooling=global_pooling,
                                                                              pre_activation=pre_activation))
    specification = get_export_specification(task=task, inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                        maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                        generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))

    # The inputs follow the forward arguments, the members of the comparison holding 3 and 2 of the 4 traced frames
    assert list(check_inputs) == TASK_INPUT_NAMES[task]
    if task == "sequence_comparison":
        assert [input_batch.shape[1] for input_batch in example_inputs.values()] == [4, 4]
        assert [input_batch.shape[1] for input_batch in check_inputs.values()] == [3, 2]

    # One vector of class probabilities per sample, or per frame for the sequence labeller and the frame selector, a
    # class map and the class probabilities of every pixel for the segmenter, the representation of every image, or
    # the 5 candidates of the 7 heatmap channels of every image for the keypoint detector
    check_shape = specification["inputs"][0]["check_shape"]
    if task == "semantic_segmentation":
        assert [output.shape for output in reference_outputs] == [(3, 64, 64), (3, 3, 64, 64)]
    elif task == "self_supervised":
        assert reference_outputs[0].shape == (3, 32)
    elif task == "keypoint_detection":
        assert check_shape == [3, 3, 64, 64]
        assert [output.shape for output in reference_outputs] == [(3, 7, 5, 2), (3, 7, 5), (3, 7, 5)]
    else:
        expected_shape = (check_shape[0], check_shape[1], 3) if task in FRAME_OUTPUT_TASKS else (3, 3)
        assert reference_outputs[0].shape == expected_shape

    # TorchScript export, traced on the example batch, keeping the input names as the names of its arguments, and run
    # on the check batch with the inputs passed by name
    export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                       output_path=tmp_path / "model.pt")
    torchscript_model = torch.jit.load(str(tmp_path / "model.pt"))
    assert [argument.name for argument in torchscript_model.forward.schema.arguments[1:]] == TASK_INPUT_NAMES[task]
    with torch.no_grad():
        torchscript_outputs = convert_outputs_to_numpy(outputs=torchscript_model(**check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=torchscript_outputs,
                                            **specification["comparison_tolerances"]) == []

    # ONNX export with dynamic batch (and frames), run by ONNX Runtime on the check batch with the inputs passed by name
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)
    session = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    assert [session_input.name for session_input in session.get_inputs()] == TASK_INPUT_NAMES[task]
    onnx_outputs = session.run(None, {input_name: input_batch.numpy()
                                      for input_name, input_batch in check_inputs.items()})
    assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                            **specification["comparison_tolerances"]) == []


def test_classifier_exports_divide_the_logits_by_the_logit_temperature(tmp_path: Path) -> None:
    """
    Verifies the logit temperature of the inference classifier: it divides the class logits before the softmax,
    softening the probabilities without changing the predicted class, the TorchScript and ONNX exports reproduce the
    softened probabilities and hold the temperature (a buffer of the traced module, a named constant of the ONNX
    graph), and a temperature that is not above 0 is refused.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    model = build_test_model(task="classification", temporal_type=None, fusion_type=None, global_pooling="average")
    inference_model = build_inference_model(task="classification", model=model)
    specification = get_export_specification(task="classification", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                        maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                        generator=torch.Generator().manual_seed(2))

    # Probabilities at the temperature 1 (those of the classifier) and at the temperature 3
    with torch.no_grad():
        class_logits = model(input_tensor=check_inputs["images"])["class_logits"]
        model_probabilities = inference_model(**check_inputs)
        inference_model.set_logit_temperature(logit_temperature=3.0)
        softened_probabilities = inference_model(**check_inputs)
    assert torch.allclose(input=model_probabilities, other=torch.softmax(input=class_logits, dim=-1), atol=1e-7)
    assert torch.allclose(input=softened_probabilities, other=torch.softmax(input=class_logits / 3.0, dim=-1),
                          atol=1e-7)
    assert torch.equal(input=softened_probabilities.argmax(dim=1), other=model_probabilities.argmax(dim=1))
    assert (softened_probabilities.max(dim=1).values < model_probabilities.max(dim=1).values).all()

    # Both exports traced at the temperature 3 reproduce the softened probabilities on another batch size
    reference_outputs = convert_outputs_to_numpy(outputs=softened_probabilities)
    export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                       output_path=tmp_path / "model.pt")
    torchscript_model = torch.jit.load(str(tmp_path / "model.pt"))
    assert float(torchscript_model.logit_temperature) == 3.0
    with torch.no_grad():
        torchscript_outputs = convert_outputs_to_numpy(outputs=torchscript_model(**check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=torchscript_outputs,
                                            **specification["comparison_tolerances"]) == []
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)
    onnx_temperatures = [float(onnx.numpy_helper.to_array(tensor=initializer))
                         for initializer in onnx.load(str(tmp_path / "model.onnx")).graph.initializer
                         if "logit_temperature" in initializer.name]
    assert onnx_temperatures == [3.0]
    session = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": check_inputs["images"].numpy()})
    assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                            **specification["comparison_tolerances"]) == []

    # A temperature that is not above 0 is refused, the temperature of the model being kept
    with pytest.raises(expected_exception=ValueError, match="must be above 0"):
        inference_model.set_logit_temperature(logit_temperature=0.0)
    assert float(inference_model.logit_temperature) == 3.0


def test_detector_exports_keep_the_distance_nms(tmp_path: Path) -> None:
    """
    Verifies the TorchScript and ONNX exports of the inference detector whose final NMS is DIoU-NMS
    ('Detector.nms_criterion' "distance_iou" with the exponent 0.6, suppress_every_pair_by_distance):
    both reproduce the detections of the eager inference detector on a batch of another size than the traced one, and
    these detections differ from those of the greedy NMS on the same weights, so that the exported graphs run DIoU-NMS.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    inference_models = {nms_criterion: build_inference_model(task="object_detection", model=build_test_detector(
        dtype=torch.float32, nms_criterion=nms_criterion, nms_distance_exponent=0.6))
        for nms_criterion in ["iou", "distance_iou"]}
    inference_model = inference_models["distance_iou"]
    specification = get_export_specification(task="object_detection", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                        maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                        generator=torch.Generator().manual_seed(2))

    # Detections of DIoU-NMS, different from those of the greedy NMS of the same detector
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))
        greedy_outputs = convert_outputs_to_numpy(outputs=inference_models["iou"](**check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=greedy_outputs,
                                            **specification["comparison_tolerances"]) != []

    # TorchScript export, traced on the example batch and run on the check batch
    export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                       output_path=tmp_path / "model.pt")
    with torch.no_grad():
        torchscript_outputs = convert_outputs_to_numpy(outputs=torch.jit.load(str(tmp_path / "model.pt"))(
            **check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=torchscript_outputs,
                                            **specification["comparison_tolerances"]) == []

    # Dynamo ONNX export with a dynamic batch, run by ONNX Runtime on the check batch
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)
    session = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": check_inputs["images"].numpy()})
    assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                            **specification["comparison_tolerances"]) == []


def test_instance_segmenter_exports_keep_the_fine_inputs_of_the_mask_heads(tmp_path: Path) -> None:
    """
    Verifies the TorchScript and ONNX exports of the inference instance segmenter whose mask heads read every finer
    input (the image branch on the crop of the input image, backbone skips at the pool size and at the mask
    resolution, fusion convolutions, and the doubled features of the pool of 5 resized to the fine features): both
    reproduce the detections and masks of the eager inference segmenter on a batch of another size than the traced
    one, and these masks differ from those of the same weights without fine inputs, so that the exported graphs run
    them.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    fine_input_settings = {"MaskHead": {**MIXED_MASK_HEAD_SETTINGS["MaskHead"], "skip_blocks_at_pool_size": [4],
                                        "image_branch_convolutions": [2, 8], "skip_blocks_at_mask_resolution": [0, 2],
                                        "fusion_convolutions": [1, 16]},
                           "level_mask_heads": MIXED_MASK_HEAD_SETTINGS["level_mask_heads"]}
    inference_model = build_inference_model(task="instance_segmentation", model=build_test_detector(
        dtype=torch.float32, instance_segmentation=True, mask_head_settings=fine_input_settings))
    specification = get_export_specification(task="instance_segmentation", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                        maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                        generator=torch.Generator().manual_seed(2))

    # Masks of the fine inputs, different from those of the same detector without them
    plain_model = build_inference_model(task="instance_segmentation", model=build_test_detector(
        dtype=torch.float32, instance_segmentation=True, mask_head_settings=MIXED_MASK_HEAD_SETTINGS))
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))
        plain_outputs = convert_outputs_to_numpy(outputs=plain_model(**check_inputs))
    assert not np.allclose(reference_outputs[4], plain_outputs[4], atol=1e-2)

    # TorchScript export, traced on the example batch and run on the check batch
    export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                       output_path=tmp_path / "model.pt")
    with torch.no_grad():
        torchscript_outputs = convert_outputs_to_numpy(outputs=torch.jit.load(str(tmp_path / "model.pt"))(
            **check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=torchscript_outputs,
                                            **specification["comparison_tolerances"]) == []

    # Dynamo ONNX export with a dynamic batch, run by ONNX Runtime on the check batch
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)
    session = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": check_inputs["images"].numpy()})
    assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                            **specification["comparison_tolerances"]) == []


def test_detector_exports_keep_the_greedy_nms_of_many_boxes(tmp_path: Path) -> None:
    """
    Verifies the TorchScript and ONNX exports of the inference detector whose greedy final NMS gets more than the 4000
    coordinates per image from which batched_nms of torchvision loops over the groups on the CPU (250 proposals kept
    of 300, 5 classes, 5000 coordinates per image): the dynamo ONNX export of the dynamic batch goes through, the
    groups being suppressed in a single call (suppress_stacked_groups: moved apart in TorchScript, one
    NonMaxSuppression batch per group in ONNX), and both exports reproduce the detections of the eager inference
    detector on a batch of another size than the traced one. Before, the export stopped on the loop over the
    classes, as the export of a detector of 35 classes on the GPU did from 3 images on.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    model = build_test_detector(dtype=torch.float32)
    model.region_proposal_filter.inference_pre_nms_proposals = 300
    model.region_proposal_filter.inference_post_nms_proposals = 250
    inference_model = build_inference_model(task="object_detection", model=model)
    specification = get_export_specification(task="object_detection", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                        maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                        generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))

    # TorchScript export, traced on the example batch and run on the check batch
    export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                       output_path=tmp_path / "model.pt")
    with torch.no_grad():
        torchscript_outputs = convert_outputs_to_numpy(outputs=torch.jit.load(str(tmp_path / "model.pt"))(
            **check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=torchscript_outputs,
                                            **specification["comparison_tolerances"]) == []

    # Dynamo ONNX export with a dynamic batch, run by ONNX Runtime on the check batch
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)
    session = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": check_inputs["images"].numpy()})
    assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                            **specification["comparison_tolerances"]) == []


def test_detector_exports_keep_the_proposal_selection_per_level(tmp_path: Path) -> None:
    """
    Verifies the TorchScript and ONNX exports of the inference detector selecting its proposals level by level
    ('RegionProposal.proposal_selection' "per_level"): both reproduce the detections of the eager inference detector on
    a batch of another size than the traced one, and these detections differ from those of the selection over all the
    levels on the same weights, so that the exported graphs run the loop over the levels. The inference keeps 20
    proposals before the suppression, of all the anchors or of every level, so that the selection over all the levels
    gives fewer than the 50 proposals kept after it.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    inference_models = {}
    for proposal_selection in ["joint", "per_level"]:
        model = build_test_detector(dtype=torch.float32, proposal_selection=proposal_selection)
        model.region_proposal_filter.inference_pre_nms_proposals = 20
        inference_models[proposal_selection] = build_inference_model(task="object_detection", model=model)
    inference_model = inference_models["per_level"]
    specification = get_export_specification(task="object_detection", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                        maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                        generator=torch.Generator().manual_seed(2))

    # Detections of the selection level by level, different from those of the selection over all the levels
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))
        joint_outputs = convert_outputs_to_numpy(outputs=inference_models["joint"](**check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=joint_outputs,
                                            **specification["comparison_tolerances"]) != []

    # TorchScript export, traced on the example batch and run on the check batch
    export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                       output_path=tmp_path / "model.pt")
    with torch.no_grad():
        torchscript_outputs = convert_outputs_to_numpy(outputs=torch.jit.load(str(tmp_path / "model.pt"))(
            **check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=torchscript_outputs,
                                            **specification["comparison_tolerances"]) == []

    # Dynamo ONNX export with a dynamic batch, run by ONNX Runtime on the check batch
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)
    session = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": check_inputs["images"].numpy()})
    assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                            **specification["comparison_tolerances"]) == []


def test_detector_onnx_suppression_keeps_the_boxes_in_the_image(tmp_path: Path) -> None:
    """
    Verifies that the ONNX export of the greedy detector gives its NonMaxSuppression nodes (the proposals and the
    detections) the boxes of the image only, one ONNX batch per group, without the offsets of the groups moved apart,
    which a TensorRT FP16 engine overflowed beyond 65,504 from the second image of a batch of the detection benchmark
    detector (35 classes at 1024 pixels), and still reproduces the detections of the eager inference detector.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    inference_model = build_inference_model(task="object_detection", model=build_test_detector(dtype=torch.float32))
    specification = get_export_specification(task="object_detection", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = {"images": torch.rand(size=(TEST_MAXIMUM_BATCH_SIZE,) + tuple(example_inputs["images"].shape[1:]),
                                         generator=torch.Generator().manual_seed(2)) * 255.0}
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)

    # Boxes reaching every suppression added to the outputs of the graph
    model_proto = onnx.load(f=str(tmp_path / "model.onnx"))
    suppression_boxes = [node.input[0] for node in model_proto.graph.node if node.op_type == "NonMaxSuppression"]
    assert len(suppression_boxes) == 2
    for boxes_name in suppression_boxes:
        model_proto.graph.output.append(onnx.helper.make_tensor_value_info(name=boxes_name,
                                                                          elem_type=onnx.TensorProto.FLOAT, shape=None))
    onnx.save(proto=model_proto, f=str(tmp_path / "model_with_suppression_boxes.onnx"))

    # Every box of the batch of the largest size stays within the image, in as many batches as groups, and the
    # detections are those of the eager detector
    session = onnxruntime.InferenceSession(str(tmp_path / "model_with_suppression_boxes.onnx"),
                                           providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"images": check_inputs["images"].numpy()})
    image_size = inference_model.input_image_size
    proposal_boxes, detection_boxes = outputs[-2:]
    assert proposal_boxes.shape[0] == TEST_MAXIMUM_BATCH_SIZE
    assert detection_boxes.shape[0] % TEST_MAXIMUM_BATCH_SIZE == 0 and detection_boxes.shape[0] > TEST_MAXIMUM_BATCH_SIZE
    for boxes in [proposal_boxes, detection_boxes]:
        assert boxes.min() >= 0.0 and boxes.max() <= image_size
    assert specification["compare_outputs"](reference=reference_outputs, candidate=outputs[:len(reference_outputs)],
                                            **specification["comparison_tolerances"]) == []


def test_segmentation_comparison_reports_the_differences() -> None:
    """
    Verifies that the comparison of the segmenter outputs accepts a few pixels of tied classes taking the other class,
    and reports class maps disagreeing on more pixels and class probabilities beyond the tolerance.
    """
    class_maps = np.zeros(shape=(1, 40, 50), dtype=np.int64)
    class_probabilities = np.full(shape=(1, 2, 40, 50), fill_value=0.5, dtype=np.float32)

    # One pixel of 2000 taking the other class is accepted, one in ten is reported
    almost_equal_maps = class_maps.copy()
    almost_equal_maps[0, 0, 0] = 1
    assert compare_segmentations(reference=[class_maps, class_probabilities],
                                 candidate=[almost_equal_maps, class_probabilities], tolerance=1e-4,
                                 minimum_equal_share=0.999) == []
    different_maps = class_maps.copy()
    different_maps[0, :4] = 1
    different_probabilities = class_probabilities + 0.01
    assert compare_segmentations(reference=[class_maps, class_probabilities],
                                 candidate=[different_maps, different_probabilities], tolerance=1e-4,
                                 minimum_equal_share=0.999) == ["class probabilities differ by up to 1.00e-02",
                                                                "class maps agree on 90.000% of the pixels only"]


def test_keypoint_comparison_reports_the_differences() -> None:
    """
    Verifies that the comparison of the keypoint detector outputs accepts one candidate of 200 moved by a tied
    maximum with another tag, the zero-score candidates filling a channel at other cells with other tags and a tied
    cell becoming a maximum in the exported model only, which shifts the ranks of the following candidates, and
    reports more moved candidates, candidate tags and scores beyond their tolerances, and outputs of another shape.
    """
    positions = np.zeros(shape=(4, 10, 5, 2), dtype=np.float32)
    scores = np.full(shape=(4, 10, 5), fill_value=0.5, dtype=np.float32)
    tags = np.zeros(shape=(4, 10, 5), dtype=np.float32)
    tolerances = {"position_tolerance": 1e-2, "score_tolerance": 1e-4, "tag_tolerance": 1e-3,
                  "minimum_equal_share": 0.95}

    # One candidate of 200 moved by a whole heatmap cell with another tag is accepted
    almost_equal_positions, almost_equal_tags = positions.copy(), tags.copy()
    almost_equal_positions[0, 0, 0], almost_equal_tags[0, 0, 0] = [4.0, 0.0], 1.0
    assert compare_keypoints(reference=[positions, scores, tags],
                             candidate=[almost_equal_positions, scores, almost_equal_tags], **tolerances) == []

    # The zero-score candidates after the first two of every channel, tied, may come at other cells with other tags
    filled_scores, filled_positions, filled_tags = scores.copy(), positions.copy(), tags.copy()
    filled_scores[:, :, 2:], filled_positions[:, :, 2:], filled_tags[:, :, 2:] = 0.0, 4.0, 1.0
    assert compare_keypoints(reference=[positions, filled_scores, tags],
                             candidate=[filled_positions, filled_scores, filled_tags], **tolerances) == []

    # A tied cell becoming a maximum in one channel of the exported model only shifts the ranks of the candidates
    # after it, the last one leaving the top-k, and is accepted although the scores at every rank differ
    ranked_positions = np.stack(arrays=[np.arange(5) * 4.0, np.zeros(shape=5)], axis=-1).astype(np.float32)
    ranked_positions = np.broadcast_to(array=ranked_positions, shape=positions.shape).copy()
    ranked_scores = np.broadcast_to(array=np.linspace(start=0.5, stop=0.1, num=5), shape=scores.shape).copy()
    inserted_positions, inserted_scores, inserted_tags = ranked_positions.copy(), ranked_scores.copy(), tags.copy()
    inserted_positions[0, 0, 2:], inserted_scores[0, 0, 2:] = ranked_positions[0, 0, 1:4], ranked_scores[0, 0, 1:4]
    inserted_positions[0, 0, 1], inserted_scores[0, 0, 1], inserted_tags[0, 0, 1] = [2.0, 0.0], 0.45, 1.0
    assert compare_keypoints(reference=[ranked_positions, ranked_scores, tags],
                             candidate=[inserted_positions, inserted_scores, inserted_tags], **tolerances) == []

    # One candidate in four moved and one in four with another tag are reported, with the shifted scores
    different_positions, different_tags = positions.copy(), tags.copy()
    different_positions[0], different_tags[1] = 4.0, 0.01
    assert compare_keypoints(reference=[positions, scores, tags],
                             candidate=[different_positions, scores + 0.01, different_tags], **tolerances) == [
        "candidate scores differ by up to 1.00e-02",
        "candidate positions and tags agree on 50.0% of the candidates only"]

    # Outputs of another shape are reported before any value is compared
    assert compare_keypoints(reference=[positions, scores, tags], candidate=[positions[:, :9], scores, tags],
                             **tolerances) == ["candidate position shapes differ: (4, 10, 5, 2) vs (4, 9, 5, 2)"]


def test_second_recurrent_export_keeps_dynamic_frames(tmp_path: Path) -> None:
    """
    Verifies that exporting a recurrent sequence model twice to ONNX in the same process keeps the number of frames
    dynamic in both files: the second export must not reuse the decomposition of the GRU cached by the first one with
    its sizes, which would write a fixed number of frames.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    inference_model = build_inference_model(task="sequence_classification", model=build_test_model(
        task="sequence_classification", temporal_type="gru", fusion_type=None, global_pooling="average"))
    specification = get_export_specification(task="sequence_classification", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                        maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                        generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))

    # Both exports keep their dynamic dimensions and reproduce the eager model with one frame less than traced
    for export_index in range(2):
        model_path = tmp_path / f"model_{export_index}.onnx"
        export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=model_path,
                    specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)
        assert find_fixed_dynamic_dimensions(model_proto=onnx.load(f=str(model_path)),
                                             specification=specification) == []
        session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        onnx_outputs = session.run(None, {"images": check_inputs["images"].numpy()})
        assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                                **specification["comparison_tolerances"]) == []


@pytest.mark.parametrize(argnames=("task", "temporal_type"),
                         argvalues=[("sequence_labelling", "lstm"), ("sequence_labelling", "gru"),
                                    ("frame_selection", "gru")])
def test_frame_outputs_of_the_onnx_export_keep_dynamic_frames(tmp_path: Path, task: str, temporal_type: str) -> None:
    """
    Verifies that the ONNX export of the bidirectional sequence labeller and frame selector declares the frames of
    their output with the frame dimension of their input, where the dynamo export writes the traced frames after the
    recurrent layer (Triton then refusing the model against the dynamic frames of its config.pbtxt), and that the
    dimension check of the export reports an output frame axis left fixed.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): The sequence task with one output row per frame.
        temporal_type (str): The bidirectional recurrent module of the model.
    """
    hide_known_export_warnings()
    inference_model = build_inference_model(task=task, model=build_test_model(
        task=task, temporal_type=temporal_type, fusion_type=None, global_pooling="average"))
    specification = get_export_specification(task=task, inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    model_path = tmp_path / "model.onnx"
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=model_path,
                specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)

    # The output takes the batch and frame dimensions of the input, and a second restore changes nothing
    model_proto = onnx.load(f=str(model_path))
    input_dimensions = model_proto.graph.input[0].type.tensor_type.shape.dim
    output_dimensions = model_proto.graph.output[0].type.tensor_type.shape.dim
    assert [dimension.dim_param or dimension.dim_value for dimension in output_dimensions] == [
        input_dimensions[0].dim_param, input_dimensions[1].dim_param, specification["outputs"][0]["dimensions"][1]]
    assert restore_dynamic_output_frames(model_proto=model_proto, specification=specification) == []

    # An output frame axis fixed again is reported by the check, then restored with the intermediate shapes dropped
    output_dimensions[1].dim_value = 4
    assert find_fixed_dynamic_dimensions(model_proto=model_proto, specification=specification) == [
        "axis 1 of the output 'class_probabilities' is fixed to 4"]
    assert restore_dynamic_output_frames(model_proto=model_proto, specification=specification) == [
        "class_probabilities axis 1: 4"]
    assert output_dimensions[1].dim_param == input_dimensions[1].dim_param
    assert len(model_proto.graph.value_info) == 0


@pytest.mark.parametrize(argnames=("task", "temporal_type"), argvalues=[("sequence_labelling", "transformer")])
def test_inference_path_matches_the_training_path_on_full_sequences(task: str, temporal_type: str) -> None:
    """
    Verifies that the path of the sequence models without sequence mask, run by the inference models, gives the
    predictions of the masked path used in training and evaluation when every frame of the sequences is real.

    Args:
        task (str): The sequence task of the model.
        temporal_type (str): The temporal module of the model.
    """
    model = build_test_model(task=task, temporal_type=temporal_type, fusion_type=None, global_pooling="average")
    frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(3)) * 255.0
    with torch.no_grad():
        masked_probabilities = model(input_tensor=frames, sequence_mask=torch.ones(size=(2, 4), dtype=torch.bool))[
            "class_probabilities"]
        inference_probabilities = build_inference_model(task=task, model=model)(images=frames)
    assert torch.allclose(input=inference_probabilities, other=masked_probabilities, atol=1e-6)


@pytest.mark.parametrize(argnames=("temporal_type", "fusion_type"), argvalues=[("transformer", "cross_attention")])
def test_inference_comparator_matches_the_training_path_on_full_sequences(temporal_type: str, fusion_type: str) -> None:
    """
    Verifies that the inference comparator, which encodes and summarises every member on its own without sequence
    masks, gives the predictions of the masked path used in training and evaluation, which runs both members as one
    batch, when every frame of both members is real.

    Args:
        temporal_type (str): The temporal module of the model.
        fusion_type (str): The fusion of the members of the model.
    """
    model = build_test_model(task="sequence_comparison", temporal_type=temporal_type, fusion_type=fusion_type,
                             global_pooling="average")
    before_frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(4)) * 255.0
    after_frames = torch.rand(size=(2, 4, 3, 64, 64), generator=torch.Generator().manual_seed(5)) * 255.0
    full_mask = torch.ones(size=(2, 4), dtype=torch.bool)
    with torch.no_grad():
        masked_probabilities = model(before_images=before_frames, after_images=after_frames, before_mask=full_mask,
                                     after_mask=full_mask)["class_probabilities"]
        inference_probabilities = build_inference_model(task="sequence_comparison", model=model)(
            before_images=before_frames, after_images=after_frames)
    assert torch.allclose(input=inference_probabilities, other=masked_probabilities, atol=1e-6)


@pytest.mark.parametrize(argnames=("task", "normalization_scope"), argvalues=[("sequence_comparison", "pair")])
def test_sequence_normalization_exports_match_the_eager_model(tmp_path: Path, task: str,
                                                              normalization_scope: Optional[str]) -> None:
    """
    Verifies the exports of the sequence models whose backbone uses sequence normalization: the statistics of every
    sequence (or of every pair, with the pair scope of the comparator) are computed in the exported graphs too, which
    reproduce the eager probabilities on batches of another size and with other numbers of frames than traced, the
    before and after members of the comparator having different numbers of frames.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): "sequence_classification" or "sequence_comparison".
        normalization_scope (Optional[str]): The scope of the statistics of the comparator, None for the classifier.
    """
    # A seeded sequence model whose backbone normalizes its feature maps with the statistics of every sequence
    hide_known_export_warnings()
    torch.manual_seed(seed=0)
    if task == "sequence_comparison":
        configuration, device, dtype = get_sequence_comparison_model_configuration(temporal_type="gru",
                                                                                   fusion_type="difference")
        configuration["Fusion"]["sequence_normalization_scope"] = normalization_scope
    else:
        configuration, device, dtype = get_sequence_model_configuration(temporal_type="gru")
    configuration["Backbone"]["normalization"]["feature_map_normalization"] = "sequence"
    model = TASK_MODEL_CLASSES[task](configuration=configuration, device=device, dtype=dtype).eval()
    inference_model = build_inference_model(task=task, model=model)
    # Trace with the example shape and compare on the check shape, of another batch size and number of frames
    specification = get_export_specification(task=task, inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    check_inputs = create_random_inputs(specification=specification, shape_name="check_shape",
                                        maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                        generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        reference_outputs = convert_outputs_to_numpy(outputs=inference_model(**check_inputs))

    # TorchScript export, run with the inputs passed by name
    export_torchscript(inference_model=inference_model, example_inputs=example_inputs,
                       output_path=tmp_path / "model.pt")
    with torch.no_grad():
        torchscript_outputs = convert_outputs_to_numpy(
            outputs=torch.jit.load(str(tmp_path / "model.pt"))(**check_inputs))
    assert specification["compare_outputs"](reference=reference_outputs, candidate=torchscript_outputs,
                                            **specification["comparison_tolerances"]) == []

    # ONNX export with dynamic batch and frames
    export_onnx(inference_model=inference_model, example_inputs=example_inputs, output_path=tmp_path / "model.onnx",
                specification=specification, maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE)
    session = onnxruntime.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {input_name: input_batch.numpy()
                                      for input_name, input_batch in check_inputs.items()})
    assert specification["compare_outputs"](reference=reference_outputs, candidate=onnx_outputs,
                                            **specification["comparison_tolerances"]) == []
