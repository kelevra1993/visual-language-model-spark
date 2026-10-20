"""
Tests of vision_training/models/classification/view_losses.py, the view losses of the image classifier: the projectors
are built for the losses turned on only, batch normalized when they have hidden layers, the view losses are the
weighted Barlow Twins and contrastive losses of the projected views of both images, pooled from their final feature
maps by the pooling of the view losses, with the batch metrics of their methods, the positives "classes" reading the
labels, the gradients reach the feature maps and the projectors, and the summary panel lists the representation and
the losses with their projectors.
"""

import pytest
import torch

from torch import nn
from typing import Any, Dict

from vision_training.losses.self_supervised_losses import compute_barlow_twins_loss, compute_info_nce_loss
from vision_training.models.classification.classification_head import build_global_pooling
from vision_training.models.classification.view_losses import (build_view_loss_projectors, compute_view_losses,
                                                               print_view_losses_summary)

# Channels, height and width of the final feature maps of the tests, the channels giving the size of their average or
# max pooled representations
FEATURE_MAP_SHAPE = (24, 3, 3)
REPRESENTATION_SIZE = 24


def build_view_losses_configuration(barlow_twins_activated: bool, contrastive_activated: bool, positives: str,
                                    global_pooling: str = "average") -> Dict[str, Any]:
    """
    Writes a 'ClassificationHead.view_losses' section for the tests, with small projectors.

    Args:
        barlow_twins_activated (bool): Whether Barlow Twins is on.
        contrastive_activated (bool): Whether the contrastive loss is on.
        positives (str): The positives of the contrastive loss, "views" or "classes".
        global_pooling (str): The pooling giving the representation of a view. Defaults to "average".

    Returns:
        Dict[str, Any]: The view losses section.
    """
    return {"global_pooling": global_pooling,
            "view_augmentation": {"random_resized_crop": {"scale": [0.5, 1.0], "aspect_ratio": [0.75, 1.3333]},
                                  "left_right_flip": {"probability": 0.5}, "up_down_flip": {"probability": 0.0},
                                  "rotation": {"probability": 0.0, "maximum_degrees": 0.0},
                                  "colour_jitter": {"probability": 0.8, "brightness": 0.4, "contrast": 0.4,
                                                    "saturation": 0.4, "hue": 0.1},
                                  "grayscale": {"probability": 0.2},
                                  "gaussian_blur": {"probability": 0.5, "kernel_size": 9, "sigma": [0.1, 2.0]}},
            "barlow_twins": {"activated": barlow_twins_activated, "weight": 0.01, "off_diagonal_weight": 0.005,
                             "projector": [32, 32, 16]},
            "contrastive": {"activated": contrastive_activated, "weight": 0.5, "temperature": 0.1,
                            "positives": positives, "projector": [8]}}


def test_projectors_are_built_for_the_losses_turned_on() -> None:
    """
    Verifies that only the losses turned on get a projector, named after the loss, with batch normalized hidden layers
    and the embedding size last, and a projector of a single layer made of its linear embedding layer alone.
    """
    projectors = build_view_loss_projectors(
        view_losses_configuration=build_view_losses_configuration(barlow_twins_activated=True,
                                                                  contrastive_activated=False, positives="views"),
        input_features=REPRESENTATION_SIZE, device=torch.device("cpu"), dtype=torch.float32)
    assert list(projectors) == ["barlow_twins_projector"]
    barlow_twins_projector = projectors["barlow_twins_projector"]
    assert sum(isinstance(module, nn.BatchNorm1d) for module in barlow_twins_projector.modules()) == 2
    assert barlow_twins_projector(representations=torch.randn(size=(4, REPRESENTATION_SIZE))).shape == (4, 16)

    # Both losses on, the contrastive projector being its embedding layer alone
    projectors = build_view_loss_projectors(
        view_losses_configuration=build_view_losses_configuration(barlow_twins_activated=True,
                                                                  contrastive_activated=True, positives="views"),
        input_features=REPRESENTATION_SIZE, device=torch.device("cpu"), dtype=torch.float32)
    assert list(projectors) == ["barlow_twins_projector", "contrastive_projector"]
    contrastive_layers = [module for module in projectors["contrastive_projector"].modules()
                          if not isinstance(module, (nn.Sequential, type(projectors["contrastive_projector"])))]
    assert len(contrastive_layers) == 1 and isinstance(contrastive_layers[0], nn.Linear)
    assert build_view_loss_projectors(
        view_losses_configuration=build_view_losses_configuration(barlow_twins_activated=False,
                                                                  contrastive_activated=False, positives="views"),
        input_features=REPRESENTATION_SIZE, device=torch.device("cpu"), dtype=torch.float32) == {}


@pytest.mark.parametrize(argnames=("positives", "global_pooling"),
                         argvalues=[("views", "average"), ("classes", "max"), ("classes", "flatten")])
def test_view_losses_are_the_weighted_losses_of_the_projected_views(positives: str, global_pooling: str) -> None:
    """
    Verifies that the view losses are the Barlow Twins and contrastive losses of the embeddings of both views, pooled
    from their final feature maps by the pooling of the view losses and projected together, multiplied by their
    weights, with the batch metrics of their methods, and that the contrastive loss reads the labels with the positives
    "classes" only.

    Args:
        positives (str): The positives of the contrastive loss.
        global_pooling (str): The pooling giving the representation of a view.
    """
    torch.manual_seed(seed=0)
    view_losses_configuration = build_view_losses_configuration(barlow_twins_activated=True,
                                                                contrastive_activated=True, positives=positives,
                                                                global_pooling=global_pooling)
    first_feature_maps = torch.randn(size=(6, *FEATURE_MAP_SHAPE))
    second_feature_maps = first_feature_maps + 0.1 * torch.randn(size=(6, *FEATURE_MAP_SHAPE))

    # Representations pooled by hand: the mean or the maximum of every channel over the positions, or every value
    both_feature_maps = torch.cat(tensors=[first_feature_maps, second_feature_maps], dim=0)
    if global_pooling == "average":
        expected_representations = both_feature_maps.mean(dim=(2, 3))
    elif global_pooling == "max":
        expected_representations = both_feature_maps.amax(dim=(2, 3))
    else:
        expected_representations = both_feature_maps.flatten(start_dim=1)
    projectors = build_view_loss_projectors(view_losses_configuration=view_losses_configuration,
                                            input_features=expected_representations.shape[1],
                                            device=torch.device("cpu"), dtype=torch.float32)
    labels = torch.tensor(data=[0, 1, 0, 2, 1, 3])
    outputs = compute_view_losses(first_feature_maps=first_feature_maps, second_feature_maps=second_feature_maps,
                                  labels=labels, global_pooling=build_global_pooling(global_pooling=global_pooling),
                                  projectors=projectors, view_losses_configuration=view_losses_configuration)
    assert sorted(outputs) == ["barlow_twins_loss", "contrastive_loss", "cross_correlation_diagonal",
                               "cross_correlation_off_diagonal", "positive_pair_accuracy"]

    # The same losses computed by hand from the embeddings of both views, projected in one pass
    expected_outputs = {}
    for loss_name in ["barlow_twins", "contrastive"]:
        embeddings = projectors[f"{loss_name}_projector"](representations=expected_representations)
        expected_outputs[loss_name] = (embeddings[:6], embeddings[6:])
    barlow_twins_outputs = compute_barlow_twins_loss(first_embeddings=expected_outputs["barlow_twins"][0],
                                                     second_embeddings=expected_outputs["barlow_twins"][1],
                                                     off_diagonal_weight=0.005)
    contrastive_outputs = compute_info_nce_loss(first_embeddings=expected_outputs["contrastive"][0],
                                                second_embeddings=expected_outputs["contrastive"][1], temperature=0.1,
                                                labels=labels if positives == "classes" else None)
    assert torch.allclose(input=outputs["barlow_twins_loss"], other=0.01 * barlow_twins_outputs["barlow_twins_loss"])
    assert torch.allclose(input=outputs["contrastive_loss"], other=0.5 * contrastive_outputs["info_nce_loss"])
    assert torch.equal(input=outputs["positive_pair_accuracy"], other=contrastive_outputs["positive_pair_accuracy"])
    assert torch.equal(input=outputs["cross_correlation_diagonal"],
                       other=barlow_twins_outputs["cross_correlation_diagonal"])

    # The labels only change the contrastive loss with the positives "classes", two images sharing class 0 and 1
    unlabelled_outputs = compute_info_nce_loss(first_embeddings=expected_outputs["contrastive"][0],
                                               second_embeddings=expected_outputs["contrastive"][1], temperature=0.1,
                                               labels=None)
    assert torch.allclose(input=outputs["contrastive_loss"], other=0.5 * unlabelled_outputs["info_nce_loss"]) == \
        (positives == "views")


def test_view_losses_train_the_projectors_and_the_feature_maps() -> None:
    """
    Verifies that the gradients of the view losses reach every parameter of the projectors and the final feature maps
    of both views, so that the backbone learns from them, and that only the losses turned on are computed.
    """
    torch.manual_seed(seed=0)
    view_losses_configuration = build_view_losses_configuration(barlow_twins_activated=False,
                                                                contrastive_activated=True, positives="classes")
    projectors = build_view_loss_projectors(view_losses_configuration=view_losses_configuration,
                                            input_features=REPRESENTATION_SIZE, device=torch.device("cpu"),
                                            dtype=torch.float32)
    first_feature_maps = torch.randn(size=(6, *FEATURE_MAP_SHAPE), requires_grad=True)
    second_feature_maps = torch.randn(size=(6, *FEATURE_MAP_SHAPE), requires_grad=True)
    outputs = compute_view_losses(first_feature_maps=first_feature_maps, second_feature_maps=second_feature_maps,
                                  labels=torch.tensor(data=[0, 1, 0, 2, 1, 3]),
                                  global_pooling=build_global_pooling(global_pooling="average"), projectors=projectors,
                                  view_losses_configuration=view_losses_configuration)
    assert sorted(outputs) == ["contrastive_loss", "positive_pair_accuracy"]
    outputs["contrastive_loss"].backward()
    assert first_feature_maps.grad.abs().sum() > 0 and second_feature_maps.grad.abs().sum() > 0
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in projectors["contrastive_projector"].parameters())


def test_summary_lists_the_view_losses_with_their_projectors(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies the summary panel: nothing when every view loss is off, otherwise the pooling and size of the
    representation of a view, the second views, the setting of every loss turned on with the layers and number of
    parameters of its projector, and 'off' for the other one.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the console output.
    """
    view_losses_configuration = build_view_losses_configuration(barlow_twins_activated=False,
                                                                contrastive_activated=False, positives="classes")
    print_view_losses_summary(view_losses_configuration=view_losses_configuration, projectors={})
    assert capsys.readouterr().out == ""

    # The contrastive loss on: its projector of 24 x 8 weights and 8 biases
    view_losses_configuration["contrastive"]["activated"] = True
    projectors = build_view_loss_projectors(view_losses_configuration=view_losses_configuration,
                                            input_features=REPRESENTATION_SIZE, device=torch.device("cpu"),
                                            dtype=torch.float32)
    print_view_losses_summary(view_losses_configuration=view_losses_configuration, projectors=projectors)
    summary = capsys.readouterr().out
    assert "View Losses" in summary and "Representation         : average pooling, 24 features" in summary
    assert "Second View Crop       : 0.5 to 1 of the image area" in summary
    assert "Barlow Twins           : off" in summary
    assert "Contrastive            : weight 0.5, temperature 0.1, positives classes" in summary
    assert "Contrastive Projector  : [8], 200 parameters" in summary
