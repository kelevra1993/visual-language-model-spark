"""
Tests of the backbone of vision_training/models/backbones/backbone.py: printing its summary, done by the trainers after
a backbone transfer, leaves its weights, statistics and training mode unchanged, and its panel lists the residual
shortcuts only for the blocks adding one.
"""

import copy
import pytest
import torch

from vision_training.models.backbones.backbone import build_backbone
from tests.fixtures.configuration_fixtures import get_classification_model_configuration


def test_summary_leaves_the_backbone_untouched() -> None:
    """
    Verifies that printing the backbone summary, done by the trainers after a backbone transfer, leaves every weight
    and batch normalization statistic unchanged and keeps the training mode, its dummy forward pass running in
    evaluation mode.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    backbone = build_backbone(backbone_configuration=configuration["Backbone"],
                              input_image_size=configuration["Data"]["image_settings"]["size"], device=device,
                              dtype=dtype).train()
    state_before = copy.deepcopy(backbone.state_dict())

    backbone.print_summary()

    assert backbone.training
    for parameter_name, parameter_value in backbone.state_dict().items():
        assert torch.equal(input=parameter_value, other=state_before[parameter_name]), parameter_name


def test_summary_lists_the_shortcuts_of_the_blocks_adding_one(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the backbone panel lists the residual shortcuts only for the blocks adding one. Of the 4 blocks of
    the classification configuration of the tests, with residual shortcuts from the second block on, only block 3 (two
    layers of 16 channels) adds one: block 2 has a single layer, and the first of the two layers of block 4 changes the
    channels from 16 to 32, which moves its first pair to a third layer it does not have. Every block keeps its
    Squeeze-and-Excitation.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    backbone = build_backbone(backbone_configuration=configuration["Backbone"],
                              input_image_size=configuration["Data"]["image_settings"]["size"], device=device,
                              dtype=dtype)

    backbone.print_summary()

    # Read the modules line of every block of the panel, the text of a block ending where the next one starts
    summary_text = capsys.readouterr().out
    block_texts = summary_text.split("Convolution Block ")[1:]
    module_lines = [next(line for line in block_text.splitlines() if "Modules" in line) for block_text in block_texts]
    assert len(module_lines) == 4
    assert ["Residual Shortcuts" in module_line for module_line in module_lines] == [False, False, True, False]
    assert all("Squeeze-and-Excitation" in module_line for module_line in module_lines)
