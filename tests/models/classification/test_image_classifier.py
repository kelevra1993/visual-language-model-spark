"""
Tests of ImageClassifier and of the classification head of vision_training/models/classification: output shapes and
probabilities, learning a tiny batch, the class weights buffer and the global pooling shapes.
"""

import pytest
import torch

from typing import List

from vision_training.models.classification.classification_head import ClassificationHead, build_global_pooling
from vision_training.models.classification.image_classifier import ImageClassifier
from tests.fixtures.configuration_fixtures import get_classification_model_configuration


@pytest.mark.parametrize(argnames="output_mode", argvalues=["multi_class", "multi_label"])
def test_classifier_output_shapes_and_probabilities(output_mode: str) -> None:
    """
    Verifies the outputs of the image classifier: [B, K] logits and probabilities, the probabilities summing to 1 over
    the classes with one class per image (softmax) and lying in [0, 1] independently with several (sigmoid), and the
    classification loss returned only when labels are given.

    Args:
        output_mode (str): The output mode of the classification head.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode=output_mode)
    model = ImageClassifier(configuration=configuration, device=device, dtype=dtype)
    model.eval()
    images = torch.rand(size=(4, 3, 64, 64), generator=torch.Generator().manual_seed(0)) * 255.0

    outputs = model(input_tensor=images)
    assert outputs["class_logits"].shape == outputs["class_probabilities"].shape == (4, 3)
    assert "classification_loss" not in outputs
    if output_mode == "multi_class":
        assert torch.allclose(input=outputs["class_probabilities"].sum(dim=1), other=torch.ones(size=(4,)))
    else:
        assert torch.equal(input=outputs["class_probabilities"], other=torch.sigmoid(input=outputs["class_logits"]))

    # Labels: class indices with one class per image, multi-hot vectors with several
    labels = torch.tensor(data=[0, 1, 2, 1]) if output_mode == "multi_class" else torch.tensor(
        data=[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    outputs = model(input_tensor=images, labels=labels)
    assert outputs["classification_loss"].ndim == 0 and torch.isfinite(input=outputs["classification_loss"])


def test_classifier_learns_a_tiny_batch() -> None:
    """
    Verifies that the classifier can fit a tiny batch: the classification loss of 6 images with fixed labels drops
    well below its initial value within a few dozen optimizer steps.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    model = ImageClassifier(configuration=configuration, device=device, dtype=dtype)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
    images = torch.rand(size=(6, 3, 64, 64), generator=torch.Generator().manual_seed(1)) * 255.0
    labels = torch.tensor(data=[0, 1, 2, 0, 1, 2])

    initial_loss = None
    for _ in range(40):
        optimizer.zero_grad()
        loss = model(input_tensor=images, labels=labels)["classification_loss"]
        initial_loss = float(loss.detach()) if initial_loss is None else initial_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = float(model(input_tensor=images, labels=labels)["classification_loss"])
    assert final_loss < 0.25 * initial_loss


def test_class_weights_buffer_weights_the_loss() -> None:
    """
    Verifies that the class weights buffer of the classifier (set by the trainer for balanced classes) weights the
    classes of the loss, and that it is saved in the state dictionary of the checkpoints.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    model = ImageClassifier(configuration=configuration, device=device, dtype=dtype)
    model.eval()
    images = torch.rand(size=(3, 3, 64, 64), generator=torch.Generator().manual_seed(2)) * 255.0
    labels = torch.tensor(data=[0, 1, 2])

    with torch.no_grad():
        class_logits = model(input_tensor=images)["class_logits"]
        model.class_weights.copy_(torch.tensor(data=[1.0, 2.0, 3.0]))
        weighted_loss = model(input_tensor=images, labels=labels)["classification_loss"]

    expected_loss = torch.nn.functional.cross_entropy(input=class_logits, target=labels,
                                                      weight=torch.tensor(data=[1.0, 2.0, 3.0]))
    assert torch.allclose(input=weighted_loss, other=expected_loss)
    assert "class_weights" in model.state_dict()


@pytest.mark.parametrize(argnames=("global_pooling", "expected_vector"), argvalues=[("average", [2.5, 6.5]),
                                                                                   ("max", [4.0, 8.0])])
def test_global_pooling_and_head_shapes(global_pooling: str, expected_vector: List[float]) -> None:
    """
    Verifies that the global pooling reduces a feature map to one vector per image with its average or maximum, and
    that the classification head applies to the last dimension, so that it also classifies every frame of a sequence.

    Args:
        global_pooling (str): The global pooling under test.
        expected_vector (List[float]): The expected pooled vector of the test feature map.
    """
    feature_map = torch.arange(start=1.0, end=9.0).reshape(1, 2, 2, 2)
    pooled_features = build_global_pooling(global_pooling=global_pooling)(input=feature_map)
    assert torch.equal(input=pooled_features, other=torch.tensor(data=[expected_vector]))

    head = ClassificationHead(input_features=2, fully_connected=[], dropout_rate=0.0, number_classes=5,
                              device=torch.device(device="cpu"), dtype=torch.float32)
    assert head(features=pooled_features).shape == (1, 5)
    assert head(features=torch.zeros(size=(2, 7, 2))).shape == (2, 7, 5)
