"""
Tests of the temporal modules of vision_training/models/sequence/temporal_modules.py: output shapes with padding frames,
poolings ignoring the frame order, ordered modules depending on it, causal and non-causal look-ahead and the module
summary.
"""

import pytest
import torch

from typing import Any, Dict, Tuple

from vision_training.models.sequence.temporal_modules import (ORDERED_TEMPORAL_TYPES, POOLING_TEMPORAL_TYPES,
                                                              build_temporal_module, print_sequence_module_summary)

# Frame vectors of the tests: batches of 3 sequences of 6 frames of 8 features
BATCH_SIZE, SEQUENCE_LENGTH, INPUT_FEATURES = 3, 6, 8


def get_temporal_configuration(temporal_type: str, bidirectional: bool, causal: bool) -> Dict[str, Any]:
    """
    Supplies a small 'TemporalModel' section for the temporal module tests.

    Args:
        temporal_type (str): The type of the temporal module.
        bidirectional (bool): Whether the recurrent modules read the sequence in both directions.
        causal (bool): Whether the transformer and temporal convolution only see the past frames.

    Returns:
        Dict[str, Any]: The temporal module settings.
    """
    return {"type": temporal_type, "hidden_size": 12, "number_layers": 2, "number_heads": 3, "kernel_size": 3,
            "dropout_rate": 0.0, "bidirectional": bidirectional, "causal": causal}


def build_test_module(temporal_type: str, keep_time_axis: bool, bidirectional: bool = False,
                      causal: bool = False) -> torch.nn.Module:
    """
    Builds a temporal module of the tests in evaluation mode, on the CPU in float64 so that the comparisons are exact
    up to rounding.

    Args:
        temporal_type (str): The type of the temporal module.
        keep_time_axis (bool): Whether the module returns one vector per frame.
        bidirectional (bool): Whether the recurrent modules read the sequence in both directions.
        causal (bool): Whether the transformer and temporal convolution only see the past frames.

    Returns:
        torch.nn.Module: The temporal module.
    """
    torch.manual_seed(seed=0)
    module = build_temporal_module(temporal_configuration=get_temporal_configuration(
        temporal_type=temporal_type, bidirectional=bidirectional, causal=causal), input_features=INPUT_FEATURES,
        maximum_length=SEQUENCE_LENGTH, keep_time_axis=keep_time_axis, device=torch.device(device="cpu"),
        dtype=torch.float64)
    return module.eval()


def get_test_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draws seeded frame vectors and a sequence mask with sequences of 6, 4 and 1 real frames.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The [3, 6, 8] float64 frame vectors and the [3, 6] boolean sequence
            mask.
    """
    frame_features = torch.randn(size=(BATCH_SIZE, SEQUENCE_LENGTH, INPUT_FEATURES),
                                 generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    sequence_mask = torch.arange(end=SEQUENCE_LENGTH).unsqueeze(dim=0) < torch.tensor(data=[[6], [4], [1]])
    return frame_features, sequence_mask


@pytest.mark.parametrize(argnames=("temporal_type", "keep_time_axis", "bidirectional"), argvalues=[
    *[(temporal_type, False, False) for temporal_type in POOLING_TEMPORAL_TYPES + ORDERED_TEMPORAL_TYPES],
    *[(temporal_type, True, False) for temporal_type in ORDERED_TEMPORAL_TYPES],
    ("gru", False, True), ("lstm", True, True)])
def test_output_shapes_and_padding_frames_are_ignored(temporal_type: str, keep_time_axis: bool,
                                                      bidirectional: bool) -> None:
    """
    Verifies that every temporal module gives [B, F'] sequence vectors (or [B, T, F'] frame vectors when keeping the
    time axis) with F' its 'output_features', and that the padding frames are ignored: replacing their vectors by
    large random values changes nothing on the real frames.

    Args:
        temporal_type (str): The type of the temporal module.
        keep_time_axis (bool): Whether the module returns one vector per frame.
        bidirectional (bool): Whether the recurrent modules read the sequence in both directions.
    """
    module = build_test_module(temporal_type=temporal_type, keep_time_axis=keep_time_axis,
                               bidirectional=bidirectional)
    frame_features, sequence_mask = get_test_inputs()

    with torch.no_grad():
        outputs = module(frame_features=frame_features, sequence_mask=sequence_mask)
        noisy_features = torch.where(condition=sequence_mask.unsqueeze(dim=-1), input=frame_features,
                                     other=100.0 * torch.randn(size=frame_features.shape, dtype=torch.float64))
        noisy_outputs = module(frame_features=noisy_features, sequence_mask=sequence_mask)

    expected_shape = (BATCH_SIZE, SEQUENCE_LENGTH, module.output_features) if keep_time_axis else (
        BATCH_SIZE, module.output_features)
    assert outputs.shape == expected_shape
    if keep_time_axis:
        assert torch.allclose(input=outputs[sequence_mask], other=noisy_outputs[sequence_mask])
    else:
        assert torch.allclose(input=outputs, other=noisy_outputs)


@pytest.mark.parametrize(argnames="temporal_type", argvalues=POOLING_TEMPORAL_TYPES)
def test_poolings_ignore_the_frame_order(temporal_type: str) -> None:
    """
    Verifies that the poolings give the same sequence vector whatever the order of the real frames, which is why they
    serve as the baseline telling whether a task needs the order of the frames.

    Args:
        temporal_type (str): The type of the pooling.
    """
    module = build_test_module(temporal_type=temporal_type, keep_time_axis=False)
    frame_features, sequence_mask = get_test_inputs()
    with torch.no_grad():
        outputs = module(frame_features=frame_features, sequence_mask=sequence_mask)
        reversed_outputs = module(frame_features=frame_features.flip(dims=[1]), sequence_mask=sequence_mask.flip(
            dims=[1]))
    assert torch.allclose(input=outputs[0], other=reversed_outputs[0])


@pytest.mark.parametrize(argnames="temporal_type", argvalues=ORDERED_TEMPORAL_TYPES)
def test_ordered_modules_depend_on_the_frame_order(temporal_type: str) -> None:
    """
    Verifies that the modules modelling the frame order give a different sequence vector when the frames are
    reversed.

    Args:
        temporal_type (str): The type of the temporal module.
    """
    module = build_test_module(temporal_type=temporal_type, keep_time_axis=False)
    frame_features, sequence_mask = get_test_inputs()
    with torch.no_grad():
        outputs = module(frame_features=frame_features, sequence_mask=sequence_mask)
        reversed_outputs = module(frame_features=frame_features.flip(dims=[1]), sequence_mask=sequence_mask)
    assert not torch.allclose(input=outputs[0], other=reversed_outputs[0])


@pytest.mark.parametrize(argnames=("temporal_type", "causal"), argvalues=[("gru", False), ("lstm", False),
                                                                          ("transformer", True),
                                                                          ("temporal_convolution", True)])
def test_causal_modules_do_not_look_ahead(temporal_type: str, causal: bool) -> None:
    """
    Verifies that the causal modules (unidirectional recurrent modules, causal transformer and temporal convolution)
    never look at the future frames: changing the frames after a frame leaves the outputs of that frame and of the
    frames before it unchanged.

    Args:
        temporal_type (str): The type of the temporal module.
        causal (bool): The causal flag of the transformer and temporal convolution.
    """
    module = build_test_module(temporal_type=temporal_type, keep_time_axis=True, causal=causal)
    frame_features = torch.randn(size=(1, SEQUENCE_LENGTH, INPUT_FEATURES), generator=torch.Generator().manual_seed(2),
                                 dtype=torch.float64)
    sequence_mask = torch.ones(size=(1, SEQUENCE_LENGTH), dtype=torch.bool)

    # Change every frame from frame 3 on
    changed_features = frame_features.clone()
    changed_features[:, 3:] = torch.randn(size=(1, SEQUENCE_LENGTH - 3, INPUT_FEATURES), dtype=torch.float64)
    with torch.no_grad():
        outputs = module(frame_features=frame_features, sequence_mask=sequence_mask)
        changed_outputs = module(frame_features=changed_features, sequence_mask=sequence_mask)
    assert torch.allclose(input=outputs[:, :3], other=changed_outputs[:, :3])
    assert not torch.allclose(input=outputs[:, 3:], other=changed_outputs[:, 3:])


@pytest.mark.parametrize(argnames=("temporal_type", "bidirectional"), argvalues=[("gru", True), ("transformer", False),
                                                                                 ("temporal_convolution", False)])
def test_non_causal_modules_look_ahead(temporal_type: str, bidirectional: bool) -> None:
    """
    Verifies that the non-causal modules (bidirectional recurrent modules, transformer and temporal convolution
    without the causal flag) use the future frames, the counterpart of the no-look-ahead test.

    Args:
        temporal_type (str): The type of the temporal module.
        bidirectional (bool): Whether the recurrent module reads the sequence in both directions.
    """
    module = build_test_module(temporal_type=temporal_type, keep_time_axis=True, bidirectional=bidirectional)
    frame_features = torch.randn(size=(1, SEQUENCE_LENGTH, INPUT_FEATURES), generator=torch.Generator().manual_seed(3),
                                 dtype=torch.float64)
    sequence_mask = torch.ones(size=(1, SEQUENCE_LENGTH), dtype=torch.bool)
    changed_features = frame_features.clone()
    changed_features[:, 3:] += 1.0
    with torch.no_grad():
        outputs = module(frame_features=frame_features, sequence_mask=sequence_mask)
        changed_outputs = module(frame_features=changed_features, sequence_mask=sequence_mask)
    assert not torch.allclose(input=outputs[:, 2], other=changed_outputs[:, 2])


def test_module_summary_leaves_out_the_pretrained_settings(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the summary panel of a temporal module lists its settings and its output size, but not its
    'pretrained' settings, whose source run is printed with the experiment information.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed panel.
    """
    temporal_configuration = {**get_temporal_configuration(temporal_type="gru", bidirectional=True, causal=False),
                              "pretrained": {"experiment_folder": "/runs/source", "iteration": None, "freeze": True,
                                             "learning_rate_multiplier": 1.0}}
    print_sequence_module_summary(title="Temporal Module", module_configuration=temporal_configuration,
                                  output_features=24, border_style="blue")

    printed_panel = capsys.readouterr().out
    assert "Hidden Size" in printed_panel and "Output Features" in printed_panel and "24" in printed_panel
    assert "Pretrained" not in printed_panel and "/runs/source" not in printed_panel
