"""
Tests of the model exports of vision_training/deployment/export_model.py and export_specifications.py: the TorchScript
and ONNX exports of every task (the image classifier with a pooled or flattened feature map or pre-activation residual
shortcuts, the semantic segmenter, the frame selector, the encoder of the self-supervised task) match the eager model,
keep their dynamic frames, the comparison of the segmenter outputs reports their differences, and the inference models
match the training path on full windows.
"""

import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from pathlib import Path
from typing import Optional

from vision_training.deployment.export_model import (export_onnx, export_torchscript, find_fixed_dynamic_dimensions,
                                                     hide_known_export_warnings)
from vision_training.deployment.export_specifications import (FRAME_OUTPUT_TASKS, TASK_INPUT_NAMES,
                                                              build_inference_model, compare_segmentations,
                                                              convert_outputs_to_numpy, create_random_inputs,
                                                              get_export_specification)
from vision_training.models.model_registry import TASK_MODEL_CLASSES
from vision_training.models.task_model import TaskModel
from tests.fixtures.configuration_fixtures import (get_classification_model_configuration,
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
    recurrent modules, a causal transformer for the sequence labeller, and the selection settings of the frame
    selector.

    Args:
        task (str): "classification", "semantic_segmentation", "sequence_classification", "sequence_labelling",
            "sequence_comparison", "frame_selection" or "self_supervised".
        temporal_type (Optional[str]): The temporal module of the sequence models, None for the classifier, the
            segmenter and the self-supervised model.
        fusion_type (Optional[str]): The fusion of the members of the comparison model, None for the other tasks.
        global_pooling (str): The reduction of the final feature map, "average" or "max", or "flatten" for the
            classifier only.
        pre_activation (bool): Whether the residual shortcuts of the backbone of the classifier use the pre-activation
            placement (residual_nets_v2).

    Returns:
        TaskModel: The model of the task.
    """
    torch.manual_seed(seed=0)
    if task == "semantic_segmentation":
        configuration, device, dtype = get_segmentation_model_configuration(upsampling="transposed")
    elif task == "self_supervised":
        configuration, device, dtype = get_self_supervised_model_configuration(method="simclr",
                                                                               global_pooling=global_pooling)
    elif temporal_type is None:
        configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
        configuration["ClassificationHead"]["global_pooling"] = global_pooling
        configuration["Backbone"]["modules"]["residual_nets_v2"] = pre_activation
    else:
        if task == "sequence_comparison":
            configuration, device, dtype = get_sequence_comparison_model_configuration(temporal_type=temporal_type,
                                                                                       fusion_type=fusion_type)
        else:
            configuration, device, dtype = get_sequence_model_configuration(temporal_type=temporal_type)
        configuration["TemporalModel"]["bidirectional"] = temporal_type in ["gru", "lstm"]
        configuration["TemporalModel"]["causal"] = task == "sequence_labelling" and temporal_type == "transformer"
        configuration["Selection"] = {"unselected_category": "unselected", "decoding": "hungarian",
                                      "frames_per_class": 1, "selection_loss_weight": 1.0}
    return TASK_MODEL_CLASSES[task](configuration=configuration, device=device, dtype=dtype).eval()


@pytest.mark.parametrize(argnames=("task", "temporal_type", "fusion_type", "global_pooling", "pre_activation"),
                         argvalues=[("classification", None, None, "average", False),
                                    ("classification", None, None, "flatten", False),
                                    ("classification", None, None, "average", True),
                                    ("semantic_segmentation", None, None, "average", False),
                                    ("sequence_classification", "gru", None, "average", False),
                                    ("sequence_classification", "transformer", None, "average", False),
                                    ("sequence_labelling", "lstm", None, "average", False),
                                    ("sequence_labelling", "transformer", None, "average", False),
                                    ("sequence_labelling", "temporal_convolution", None, "average", False),
                                    ("sequence_comparison", "gru", "cross_attention", "average", False),
                                    ("sequence_comparison", "transformer", "difference", "average", False),
                                    ("frame_selection", "gru", None, "average", False),
                                    ("self_supervised", None, None, "average", False)])
def test_exports_match_the_eager_model(tmp_path: Path, task: str, temporal_type: Optional[str],
                                       fusion_type: Optional[str], global_pooling: str, pre_activation: bool) -> None:
    """
    Verifies the TorchScript and ONNX exports of the inference models of the classification, segmentation and
    sequence tasks: both keep the input names of the specification and reproduce the eager class probabilities (and
    class maps of the segmenter) on a batch of another size than the traced one and, for the sequence models, with
    another number of frames, so that Triton can batch
    requests and receive windows of any length. The before and after windows of the comparison model are checked
    with different numbers of frames, and its recurrent case, exported after the recurrent sequence classifier in the
    same process, also checks that a second recurrent export keeps its numbers of frames dynamic.

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

    # The inputs follow the forward arguments, the windows of the comparison holding 3 and 2 of the 4 traced frames
    assert list(check_inputs) == TASK_INPUT_NAMES[task]
    if task == "sequence_comparison":
        assert [input_batch.shape[1] for input_batch in example_inputs.values()] == [4, 4]
        assert [input_batch.shape[1] for input_batch in check_inputs.values()] == [3, 2]

    # One vector of class probabilities per sample, or per frame for the sequence labeller and the frame selector, a
    # class map and the class probabilities of every pixel for the segmenter, or the representation of every image
    check_shape = specification["inputs"][0]["check_shape"]
    if task == "semantic_segmentation":
        assert [output.shape for output in reference_outputs] == [(3, 64, 64), (3, 3, 64, 64)]
    elif task == "self_supervised":
        assert reference_outputs[0].shape == (3, 32)
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


def test_fixed_dynamic_dimensions_are_reported(tmp_path: Path) -> None:
    """
    Verifies that the guard of the ONNX exports reports every input dimension the specification declares dynamic but
    the file holds as a fixed value, here the batch dimension and the number of frames of a sequence model exported
    without dynamic shapes.

    Args:
        tmp_path (Path): The pytest temporary directory.
    """
    hide_known_export_warnings()
    inference_model = build_inference_model(task="sequence_classification", model=build_test_model(
        task="sequence_classification", temporal_type="mean", fusion_type=None, global_pooling="average"))
    specification = get_export_specification(task="sequence_classification", inference_model=inference_model)
    example_inputs = create_random_inputs(specification=specification, shape_name="example_shape",
                                          maximum_batch_size=TEST_MAXIMUM_BATCH_SIZE, device=torch.device("cpu"),
                                          generator=torch.Generator().manual_seed(1))
    torch.onnx.export(inference_model, args=tuple(example_inputs.values()), f=str(tmp_path / "fixed.onnx"),
                      input_names=list(example_inputs), opset_version=18, dynamo=True, verbose=False)

    assert find_fixed_dynamic_dimensions(model_proto=onnx.load(f=str(tmp_path / "fixed.onnx")),
                                         specification=specification) == [
        "the batch dimension of 'images' is fixed to 2", "the frames dimension of 'images' is fixed to 4"]


@pytest.mark.parametrize(argnames=("task", "temporal_type"), argvalues=[
    ("sequence_classification", "gru"), ("sequence_classification", "attention_pooling"),
    ("sequence_labelling", "transformer"), ("sequence_labelling", "temporal_convolution")])
def test_inference_path_matches_the_training_path_on_full_windows(task: str, temporal_type: str) -> None:
    """
    Verifies that the path of the sequence models without sequence mask, run by the inference models, gives the
    predictions of the masked path used in training and evaluation when every frame of the windows is real.

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


@pytest.mark.parametrize(argnames=("temporal_type", "fusion_type"), argvalues=[
    ("gru", "concatenation"), ("transformer", "cross_attention")])
def test_inference_comparator_matches_the_training_path_on_full_windows(temporal_type: str, fusion_type: str) -> None:
    """
    Verifies that the inference comparator, which encodes and summarises every member on its own without sequence
    masks, gives the predictions of the masked path used in training and evaluation, which runs both members as one
    batch, when every frame of both windows is real.

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


@pytest.mark.parametrize(argnames=("task", "normalization_scope"), argvalues=[
    ("sequence_classification", None), ("sequence_comparison", "member"), ("sequence_comparison", "pair")])
def test_sequence_normalization_exports_match_the_eager_model(tmp_path: Path, task: str,
                                                              normalization_scope: Optional[str]) -> None:
    """
    Verifies the exports of the sequence models whose backbone uses sequence normalization: the statistics of every
    window (or of every pair, with the pair scope of the comparator) are computed in the exported graphs too, which
    reproduce the eager probabilities on batches of another size and with other numbers of frames than traced, the
    before and after windows of the comparator having different numbers of frames.

    Args:
        tmp_path (Path): The pytest temporary directory.
        task (str): "sequence_classification" or "sequence_comparison".
        normalization_scope (Optional[str]): The scope of the statistics of the comparator, None for the classifier.
    """
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
