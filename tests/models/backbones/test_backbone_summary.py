import copy
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
