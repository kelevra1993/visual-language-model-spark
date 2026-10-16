"""
Tests of the model of the self-supervised task, SelfSupervisedModel of
vision_training/models/self_supervised/self_supervised_model.py, and of InferenceEncoder, its exported version: the
sizes of the representations and embeddings of every pooling, the loss and batch metrics of each method with second
views, a projection head without hidden layer, the L2 normalized representations of the encoder, which leave the
projection head out, and an unknown method refused.
"""

import pytest
import torch

from torch.nn import functional as torch_functional

from vision_training.models.self_supervised.inference_encoder import InferenceEncoder
from vision_training.models.self_supervised.self_supervised_model import SelfSupervisedModel
from tests.fixtures.configuration_fixtures import get_self_supervised_model_configuration

# Keys of the loss and of the batch metrics returned by the model with second views, for each method
METHOD_OUTPUT_KEYS = {"simclr": ["embeddings", "info_nce_loss", "positive_pair_accuracy",
                                 "embedding_standard_deviation"],
                      "barlow_twins": ["embeddings", "barlow_twins_loss", "cross_correlation_diagonal",
                                       "cross_correlation_off_diagonal", "embedding_standard_deviation"]}


def create_test_images(number_images: int) -> torch.Tensor:
    """
    Creates float BGR images in [0, 255] of random content at the 64 pixel input size of the test configuration.

    Args:
        number_images (int): The number of images of the batch.

    Returns:
        torch.Tensor: The [N, 3, 64, 64] images.
    """
    generator = torch.Generator().manual_seed(0)
    return torch.rand(size=(number_images, 3, 64, 64), generator=generator) * 255.0


@pytest.mark.parametrize(argnames=("global_pooling", "representation_size"),
                         argvalues=[("average", 32), ("max", 32), ("flatten", 32 * 8 * 8)])
def test_model_gives_representations_and_embeddings(global_pooling: str, representation_size: int) -> None:
    """
    Verifies the outputs of the model without second views, as in the evaluation: the representation of every image
    (the 32 channels of the final feature map pooled, or its 32 x 8 x 8 values flattened) and its embedding of 8.

    Args:
        global_pooling (str): The reduction of the final feature map.
        representation_size (int): The expected size of the representations.
    """
    configuration, device, dtype = get_self_supervised_model_configuration(method="simclr",
                                                                           global_pooling=global_pooling)
    model = SelfSupervisedModel(configuration=configuration, device=device, dtype=dtype).eval()
    assert model.representation_size == representation_size
    with torch.no_grad():
        outputs = model(input_tensor=create_test_images(number_images=3))
    assert sorted(outputs) == ["embeddings", "representations"]
    assert outputs["representations"].shape == (3, representation_size) and outputs["embeddings"].shape == (3, 8)


@pytest.mark.parametrize(argnames="method", argvalues=["simclr", "barlow_twins"])
def test_model_computes_the_loss_of_its_method(method: str) -> None:
    """
    Verifies the outputs of the model with second views, as in training and validation: the embeddings of the first
    views, the loss of the method and its batch metrics, the loss training every parameter of the backbone and the
    projection head.

    Args:
        method (str): The self-supervised method.
    """
    configuration, device, dtype = get_self_supervised_model_configuration(method=method, global_pooling="average")
    model = SelfSupervisedModel(configuration=configuration, device=device, dtype=dtype).train()
    images = create_test_images(number_images=4)
    outputs = model(input_tensor=images, second_view_tensor=images.flip(dims=(3,)))
    assert sorted(outputs) == sorted(METHOD_OUTPUT_KEYS[method])
    assert outputs["embeddings"].shape == (4, 8)

    # The loss reaches the backbone and the projection head
    loss_key = "info_nce_loss" if method == "simclr" else "barlow_twins_loss"
    outputs[loss_key].backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_projection_head_without_hidden_layer() -> None:
    """
    Verifies a projection head holding the embedding layer only, without normalization: the embedding is a linear map
    of the representation.
    """
    configuration, device, dtype = get_self_supervised_model_configuration(method="barlow_twins",
                                                                           global_pooling="average")
    configuration["ProjectionHead"].update({"fully_connected": [8], "fully_connected_normalization": "none"})
    model = SelfSupervisedModel(configuration=configuration, device=device, dtype=dtype).eval()
    assert sum(isinstance(module, torch.nn.Linear) for module in model.projection_head.modules()) == 1
    with torch.no_grad():
        outputs = model(input_tensor=create_test_images(number_images=2))
    assert outputs["embeddings"].shape == (2, 8)


def test_inference_encoder_gives_the_normalized_representations() -> None:
    """
    Verifies that the exported encoder returns the L2 normalized representations of the model, the pooled final
    feature map before the projection head, the vectors compared by the k-nearest neighbour evaluation.
    """
    configuration, device, dtype = get_self_supervised_model_configuration(method="simclr", global_pooling="average")
    model = SelfSupervisedModel(configuration=configuration, device=device, dtype=dtype).eval()
    encoder = InferenceEncoder(model=model).eval()
    images = create_test_images(number_images=3)
    with torch.no_grad():
        encoded_images = encoder(images=images)
        expected_representations = torch_functional.normalize(input=model(input_tensor=images)["representations"],
                                                              dim=1)
    assert encoder.representation_size == 32 and encoded_images.shape == (3, 32)
    assert torch.allclose(input=encoded_images, other=expected_representations)
    assert torch.allclose(input=encoded_images.norm(dim=1), other=torch.ones(size=(3,)))


def test_unknown_method_is_refused() -> None:
    """
    Verifies that a model of an unknown method stops at its construction instead of silently training Barlow Twins.
    """
    configuration, device, dtype = get_self_supervised_model_configuration(method="byol", global_pooling="average")
    with pytest.raises(expected_exception=ValueError, match="Unknown self-supervised method 'byol'"):
        SelfSupervisedModel(configuration=configuration, device=device, dtype=dtype)
