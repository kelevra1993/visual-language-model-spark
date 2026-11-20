"""
Tests of the temporal modules of vision_training/models/sequence/temporal_modules.py: output shapes with padding frames,
the transformer without positional embedding ignoring the frame order, ordered modules depending on it and causal
modules not looking ahead.
"""

import pytest
import torch

from typing import Any, Dict, Tuple

from vision_training.models.sequence.temporal_modules import build_temporal_module

# Frame vectors of the tests: batches of 3 sequences of 6 frames of 8 features
BATCH_SIZE, SEQUENCE_LENGTH, INPUT_FEATURES = 3, 6, 8


def get_temporal_configuration(temporal_type: str, bidirectional: bool, causal: bool,
                               positional_embedding: bool) -> Dict[str, Any]:
    """
    Supplies a small 'TemporalModel' section for the temporal module tests.

    Args:
        temporal_type (str): The type of the temporal module.
        bidirectional (bool): Whether the recurrent modules read the sequence in both directions.
        causal (bool): Whether the transformer and temporal convolution only see the past frames.
        positional_embedding (bool): Whether the transformer gives every frame its position.

    Returns:
        Dict[str, Any]: The temporal module settings.
    """
    return {"type": temporal_type, "hidden_size": 12, "number_layers": 2, "number_heads": 3, "kernel_size": 3,
            "dropout_rate": 0.0, "bidirectional": bidirectional, "causal": causal,
            "positional_embedding": positional_embedding}


def build_test_module(temporal_type: str, keep_time_axis: bool, bidirectional: bool = False,
                      causal: bool = False, positional_embedding: bool = True) -> torch.nn.Module:
    """
    Builds a temporal module of the tests in evaluation mode, on the CPU in float64 so that the comparisons are exact
    up to rounding.

    Args:
        temporal_type (str): The type of the temporal module.
        keep_time_axis (bool): Whether the module returns one vector per frame.
        bidirectional (bool): Whether the recurrent modules read the sequence in both directions.
        causal (bool): Whether the transformer and temporal convolution only see the past frames.
        positional_embedding (bool): Whether the transformer gives every frame its position.

    Returns:
        torch.nn.Module: The temporal module.
    """
    torch.manual_seed(seed=0)
    module = build_temporal_module(temporal_configuration=get_temporal_configuration(
        temporal_type=temporal_type, bidirectional=bidirectional, causal=causal,
        positional_embedding=positional_embedding), input_features=INPUT_FEATURES, maximum_length=SEQUENCE_LENGTH,
        keep_time_axis=keep_time_axis, device=torch.device(device="cpu"), dtype=torch.float64)
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


@pytest.mark.parametrize(argnames=("temporal_type", "keep_time_axis", "bidirectional"),
                         argvalues=[("attention_pooling", False, False), ("transformer", True, False)])
def test_output_shapes_and_padding_frames_are_ignored(temporal_type: str, keep_time_axis: bool,
                                                      bidirectional: bool) -> None:
    """
    Verifies that a pooling gives [B, F'] sequence vectors and a transformer keeping the time axis [B, T, F'] frame
    vectors, with F' its 'output_features', and that the padding frames are ignored: replacing their vectors by
    large random values changes nothing on the real frames.

    Args:
        temporal_type (str): The type of the temporal module.
        keep_time_axis (bool): Whether the module returns one vector per frame.
        bidirectional (bool): Whether the recurrent modules read the sequence in both directions.
    """
    module = build_test_module(temporal_type=temporal_type, keep_time_axis=keep_time_axis, bidirectional=bidirectional)
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


def test_transformer_without_positional_embedding_ignores_the_frame_order() -> None:
    """
    Verifies that the transformer without positional embedding treats a sequence as a set of frames: it holds no
    embedding parameter, its weights are the same whatever the maximum number of frames, shuffling the real frames of
    a sequence leaves its sequence vector unchanged (many to one), and the frame vectors follow their frames (many to
    many).
    """
    frame_features, sequence_mask = get_test_inputs()
    permutation = torch.tensor(data=[3, 0, 5, 1, 4, 2])

    # No positional embedding among the weights, which no longer depend on the maximum number of frames
    sequence_module = build_test_module(temporal_type="transformer", keep_time_axis=False, positional_embedding=False)
    assert sequence_module.positional_embedding is None
    assert all("positional_embedding" not in parameter_name for parameter_name, _ in
               sequence_module.named_parameters())
    longer_module = build_temporal_module(temporal_configuration=get_temporal_configuration(
        temporal_type="transformer", bidirectional=False, causal=False, positional_embedding=False),
        input_features=INPUT_FEATURES, maximum_length=4 * SEQUENCE_LENGTH, keep_time_axis=False,
        device=torch.device(device="cpu"), dtype=torch.float64)
    assert {name: tensor.shape for name, tensor in longer_module.state_dict().items()} == {
        name: tensor.shape for name, tensor in sequence_module.state_dict().items()}

    # Same sequence vector for the shuffled frames of the full sequence, frame vectors shuffled with their frames
    frame_module = build_test_module(temporal_type="transformer", keep_time_axis=True, positional_embedding=False)
    with torch.no_grad():
        sequence_vectors = sequence_module(frame_features=frame_features, sequence_mask=sequence_mask)
        shuffled_sequence_vectors = sequence_module(frame_features=frame_features[:, permutation],
                                                    sequence_mask=sequence_mask[:, permutation])
        frame_vectors = frame_module(frame_features=frame_features, sequence_mask=sequence_mask)
        shuffled_frame_vectors = frame_module(frame_features=frame_features[:, permutation],
                                              sequence_mask=sequence_mask[:, permutation])
    assert torch.allclose(input=shuffled_sequence_vectors[0], other=sequence_vectors[0])
    assert torch.allclose(input=shuffled_frame_vectors[0], other=frame_vectors[0, permutation])

    # The transformer with its positional embedding depends on the order of the same frames
    ordered_module = build_test_module(temporal_type="transformer", keep_time_axis=False)
    with torch.no_grad():
        ordered_vectors = ordered_module(frame_features=frame_features, sequence_mask=sequence_mask)
        shuffled_ordered_vectors = ordered_module(frame_features=frame_features[:, permutation],
                                                  sequence_mask=sequence_mask[:, permutation])
    assert not torch.allclose(input=shuffled_ordered_vectors[0], other=ordered_vectors[0])


@pytest.mark.parametrize(argnames="temporal_type", argvalues=["transformer"])
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


@pytest.mark.parametrize(argnames=("temporal_type", "causal"), argvalues=[("transformer", True),
                                                                          ("temporal_convolution", True)])
def test_causal_modules_do_not_look_ahead(temporal_type: str, causal: bool) -> None:
    """
    Verifies that the causal modules (causal transformer and temporal convolution) never look at the future frames:
    changing the frames after a frame leaves the outputs of that frame and of the frames before it unchanged.

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


def test_unknown_temporal_type_is_refused() -> None:
    """
    Verifies that a misspelled temporal module type reaching the builder without the configuration checks (a script,
    a test or an older saved configuration) stops with the list of the valid types, instead of building a temporal
    convolution.
    """
    with pytest.raises(expected_exception=ValueError, match="'TemporalModel.type' must be one of .* got 'transformr'"):
        build_test_module(temporal_type="transformr", keep_time_axis=True)
