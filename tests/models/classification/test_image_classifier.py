"""
Tests of ImageClassifier and of the classification head of vision_training/models/classification: output shapes and
probabilities, learning a tiny batch, the class weights buffer, the final feature maps of both views returned for the
view losses, the decorrelation loss of the hidden features of the head, the global pooling shapes, the head sized by
the pooled or flattened feature map, and the summaries printing the projectors of the view losses.
"""

import pytest
import torch

from typing import Dict, List

from vision_training.models.classification.classification_head import (ClassificationHead, build_global_pooling,
                                                                       compute_pooled_features)
from vision_training.models.classification.image_classifier import ImageClassifier
from vision_training.models.classification.view_losses import build_view_loss_projectors
from vision_training.models.task_model import TOTAL_WITH_AUXILIARY_HEADS_NAME
from tests.fixtures.configuration_fixtures import get_classification_model_configuration
from tests.models.classification.test_view_losses import build_view_losses_configuration


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


def test_second_views_give_the_final_feature_maps_of_both_views() -> None:
    """
    Verifies that with second views, for the view losses of the trainer, the classifier returns the final feature maps
    of the images and of their second views, while its logits and classification loss stay those of the images
    classified alone, the head never reading the second views. The classifier is in evaluation mode, where every image
    goes through the backbone independently of the others in its batch.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    model = ImageClassifier(configuration=configuration, device=device, dtype=dtype)
    model.eval()
    images = torch.rand(size=(4, 3, 64, 64), generator=torch.Generator().manual_seed(3)) * 255.0
    second_views = torch.rand(size=(4, 3, 64, 64), generator=torch.Generator().manual_seed(4)) * 255.0
    labels = torch.tensor(data=[0, 1, 2, 1])

    with torch.no_grad():
        outputs = model(input_tensor=images, labels=labels)
        view_outputs = model(input_tensor=images, labels=labels, second_view_tensor=second_views)
        image_feature_map, _ = model.backbone(input_tensor=images)
        second_view_feature_map, _ = model.backbone(input_tensor=second_views)

    # The classification of the images is unchanged, the predictions without second views returning no feature map
    assert "final_feature_map" not in outputs and "second_view_final_feature_map" not in outputs
    assert torch.allclose(input=view_outputs["class_logits"], other=outputs["class_logits"], atol=1e-5)
    assert torch.allclose(input=view_outputs["classification_loss"], other=outputs["classification_loss"], atol=1e-6)

    # The final feature maps of both views, those of the backbone run on each view alone
    assert view_outputs["final_feature_map"].shape == image_feature_map.shape == (4, 32, 8, 8)
    assert torch.allclose(input=view_outputs["final_feature_map"], other=image_feature_map, atol=1e-5)
    assert torch.allclose(input=view_outputs["second_view_final_feature_map"], other=second_view_feature_map, atol=1e-5)


def build_decorrelated_classifier(activated: bool) -> ImageClassifier:
    """
    Builds the small image classifier of the tests with two hidden layers in its head, the second one decorrelated
    with a weight of 0.5 when asked, from the same random seed so that two classifiers share their weights. It stays
    in training mode, where the batch statistics of the backbone spread the features of the images over the batch as
    in a training; with the running statistics of a new backbone, every feature of the head varies less over a batch
    than the epsilon of the normalization of the correlation (1e-5), which then counts it as constant.

    Args:
        activated (bool): Whether 'ClassificationHead.decorrelation' is on.

    Returns:
        ImageClassifier: The classifier in training mode, without dropout.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    configuration["ClassificationHead"].update({"fully_connected": [16, 12], "fully_connected_normalization": "layer",
                                                "decorrelation": {"activated": activated, "weight": 0.5,
                                                                  "hidden_layer": 2}})
    return ImageClassifier(configuration=configuration, device=device, dtype=dtype).train()


def test_decorrelation_loss_of_the_hidden_features() -> None:
    """
    Verifies the decorrelation loss of the image classifier: returned with the labels only, equal to the weight times
    the sum of the squared off-diagonal correlations of the features of the chosen hidden layer as its ReLU receives
    them, its tracked mean absolute correlation, the logits and the classification loss left unchanged, and gradients
    reaching the layers up to that hidden layer only, not the class logits layer.
    """
    model = build_decorrelated_classifier(activated=True)
    reference_model = build_decorrelated_classifier(activated=False)
    images = torch.rand(size=(10, 3, 64, 64), generator=torch.Generator().manual_seed(4)) * 255.0
    labels = torch.tensor(data=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    # Features received by the ReLU of the second hidden layer, read by a hook on that ReLU
    relu_layers = [layer for layer in model.classification_head.fully_connected_block
                   if isinstance(layer, torch.nn.ReLU)]
    received_features = []
    hook = relu_layers[1].register_forward_pre_hook(hook=lambda module, inputs: received_features.append(
        inputs[0].detach().clone()))
    outputs = model(input_tensor=images, labels=labels)
    hook.remove()

    # Weight times the squared off-diagonal terms of the correlation matrix of the 12 features over the 10 images, the
    # features varying enough over the batch for the epsilon of the normalization to change them by less than 1 %
    correlation = torch.corrcoef(input=received_features[0].T)
    off_diagonal = correlation - torch.diag(input=torch.diagonal(input=correlation))
    assert received_features[0].shape == (10, 12) and (received_features[0] < 0).any()
    assert received_features[0].var(dim=0, unbiased=False).min() > 1e-3
    assert torch.allclose(input=outputs["decorrelation_loss"], other=0.5 * off_diagonal.pow(exponent=2).sum(),
                          rtol=1e-2)
    assert torch.allclose(input=outputs["hidden_feature_correlation"], other=off_diagonal.abs().sum() / (12 * 11),
                          rtol=1e-2)
    assert not outputs["hidden_feature_correlation"].requires_grad

    # Same logits and classification loss as the classifier without decorrelation, and no decorrelation without labels
    reference_outputs = reference_model(input_tensor=images, labels=labels)
    assert torch.allclose(input=outputs["class_logits"], other=reference_outputs["class_logits"], atol=1e-6)
    assert torch.allclose(input=outputs["classification_loss"], other=reference_outputs["classification_loss"])
    assert "decorrelation_loss" not in reference_outputs
    assert "decorrelation_loss" not in model(input_tensor=images) and "hidden_feature_correlation" not in model(
        input_tensor=images)

    # Gradients of the decorrelation loss reach the backbone and the hidden layers, not the class logits layer
    outputs["decorrelation_loss"].backward()
    assert model.classification_head.classifier.weight.grad is None
    first_hidden_layer = model.classification_head.fully_connected_block[1]
    assert first_hidden_layer.weight.grad is not None and first_hidden_layer.weight.grad.abs().sum() > 0
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.backbone.parameters())


def test_minimising_the_decorrelation_loss_decorrelates_the_hidden_features() -> None:
    """
    Verifies that the decorrelation loss does what it is meant for: a few optimizer steps on it alone lower the mean
    absolute correlation of the features of the decorrelated hidden layer on a fixed batch.
    """
    model = build_decorrelated_classifier(activated=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
    images = torch.rand(size=(32, 3, 64, 64), generator=torch.Generator().manual_seed(5)) * 255.0
    labels = torch.zeros(size=(32,), dtype=torch.int64)

    correlations = []
    for _ in range(30):
        optimizer.zero_grad()
        outputs = model(input_tensor=images, labels=labels)
        correlations.append(float(outputs["hidden_feature_correlation"]))
        outputs["decorrelation_loss"].backward()
        optimizer.step()
    assert correlations[-1] < 0.5 * correlations[0]


@pytest.mark.parametrize(argnames=("global_pooling", "expected_vector"),
                         argvalues=[("average", [2.5, 6.5]), ("max", [4.0, 8.0]),
                                    ("flatten", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])])
def test_global_pooling_and_head_shapes(global_pooling: str, expected_vector: List[float]) -> None:
    """
    Verifies that the global pooling reduces a feature map to one vector per image with its average or maximum, or
    keeps all its values in channel, row and column order when flattening, and that the classification head applies
    to the last dimension, so that it also classifies every frame of a sequence.

    Args:
        global_pooling (str): The global pooling under test.
        expected_vector (List[float]): The expected pooled vector of the test feature map.
    """
    feature_map = torch.arange(start=1.0, end=9.0).reshape(1, 2, 2, 2)
    pooled_features = build_global_pooling(global_pooling=global_pooling)(input=feature_map)
    assert torch.equal(input=pooled_features, other=torch.tensor(data=[expected_vector]))

    head = ClassificationHead(input_features=len(expected_vector), fully_connected=[],
                              fully_connected_normalization="none", batch_normalization_momentum=0.1, dropout_rate=0.0,
                              number_classes=5, device=torch.device(device="cpu"), dtype=torch.float32)
    assert head(features=pooled_features).shape == (1, 5)
    assert head(features=torch.zeros(size=(2, 7, len(expected_vector)))).shape == (2, 7, 5)


@pytest.mark.parametrize(argnames=("global_pooling", "image_size", "last_max_pooling", "expected_features"),
                         argvalues=[("average", 64, False, 32), ("flatten", 64, False, 32 * 8 * 8),
                                    ("flatten", 64, True, 32 * 4 * 4), ("flatten", 32, False, 32 * 4 * 4)])
def test_head_input_follows_the_pooled_feature_map(global_pooling: str, image_size: int, last_max_pooling: bool,
                                                   expected_features: int) -> None:
    """
    Verifies that the head of the image classifier reads the vector the global pooling makes from the final feature
    map: its 32 channels for a pooling, whatever the image size, and its 32 channels times its height and width when
    flattening, which follow the image size and the last max pooling (the 4 test blocks halving 64 pixels to 8 without
    it, to 4 with it), and that the flattened classifier gives one logit per class of every image.

    Args:
        global_pooling (str): The reduction of the final feature map.
        image_size (int): The side of the input images.
        last_max_pooling (bool): Whether the last backbone block ends with a max pooling.
        expected_features (int): The expected number of input features of the head.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    configuration["Data"]["image_settings"]["size"] = image_size
    configuration["Backbone"]["last_max_pooling"] = last_max_pooling
    configuration["ClassificationHead"]["global_pooling"] = global_pooling
    model = ImageClassifier(configuration=configuration, device=device, dtype=dtype)
    model.eval()
    assert model.classification_head.input_features == expected_features

    # One logit per class of every image, and when flattening as many head inputs as values in the final feature map
    images = torch.rand(size=(2, 3, image_size, image_size), generator=torch.Generator().manual_seed(3)) * 255.0
    with torch.no_grad():
        final_feature_map, _ = model.backbone(input_tensor=images)
        outputs = model(input_tensor=images)
    assert outputs["class_logits"].shape == (2, 3)
    if global_pooling == "flatten":
        assert final_feature_map[0].numel() == expected_features


def read_model_size_parameters(summary: str) -> Dict[str, int]:
    """
    Reads the number of parameters of every line of the Model Size panel printed by print_model_size_summary, the
    other panels of the output being skipped.

    Args:
        summary (str): The captured console output holding the panel.

    Returns:
        Dict[str, int]: The number of parameters of every line of the panel, in its order.
    """
    parameters = {}
    component_name = None
    inside_panel = False

    # Walk the lines of the panel, its line names followed by their parameter counts
    for line in summary.splitlines():
        if "Model Size" in line:
            inside_panel = True
            continue
        if not inside_panel:
            continue
        if line.startswith("╰"):
            break
        text = line.strip("│ ")
        if text.startswith("- Parameters"):
            parameters[component_name] = int(text.split(":")[1].replace(",", ""))
        elif text and not text.startswith("-"):
            component_name = text
    return parameters


def test_summaries_print_the_view_loss_projectors_given_as_auxiliary_heads(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the summaries of the classifier with the projectors of its view losses, the auxiliary heads of the
    classification trainer: without them, the architecture summary ends with the classification head and the model
    size lists the classifier only; with them, the View Losses panel follows the head, and the model size lists every
    projector after the classifier, then the classifier with its projectors, whose parameters add up.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    configuration["ClassificationHead"]["view_losses"] = build_view_losses_configuration(
        barlow_twins_activated=True, contrastive_activated=True, positives="views")
    model = ImageClassifier(configuration=configuration, device=device, dtype=dtype)
    projectors = build_view_loss_projectors(
        view_losses_configuration=configuration["ClassificationHead"]["view_losses"],
        input_features=compute_pooled_features(global_pooling="average", backbone=model.backbone), device=device,
        dtype=dtype)

    # Without projectors, the head is printed alone and the model size gives the classifier only
    model.print_summary()
    model.print_model_size_summary()
    summary = capsys.readouterr().out
    assert "Classification Head" in summary and "View Losses" not in summary
    assert list(read_model_size_parameters(summary=summary)) == ["Backbone", "Classification Head", "Total Model"]

    # With them, the View Losses panel follows the head, and every projector and the total join the model size
    model.print_summary(auxiliary_heads=projectors)
    model.print_model_size_summary(auxiliary_heads=projectors)
    summary = capsys.readouterr().out
    assert summary.index("Classification Head") < summary.index("View Losses") < summary.index("Model Size")
    assert "Barlow Twins Projector : [32, 32, 16]" in summary and "Contrastive Projector  : [8]" in summary
    parameters = read_model_size_parameters(summary=summary)
    assert list(parameters) == ["Backbone", "Classification Head", "Total Model", "Barlow Twins Projector",
                                "Contrastive Projector", TOTAL_WITH_AUXILIARY_HEADS_NAME]
    assert parameters["Barlow Twins Projector"] == sum(
        parameter.numel() for parameter in projectors["barlow_twins_projector"].parameters())
    assert parameters[TOTAL_WITH_AUXILIARY_HEADS_NAME] == (parameters["Total Model"] +
                                                           parameters["Barlow Twins Projector"] +
                                                           parameters["Contrastive Projector"])
