from textwrap import indent

import torch
import yaml
from model.model import Model
from utilities.os_utilities import print_green, print_blue
from utilities.tensor_utilities import print_tensor_status, print_tensor_shape


def debug_model() -> None:
    """
    Instantiates the complete Model and runs a dummy forward pass to verify
    the outputs of the backbone and subsequently the region proposals.
    """
    print_blue(output="Prepared Model configuration:", add_separators=True)

    configuration = {'Data': {'image_settings': {'size': 1024}},
                     'Backbone': {
                         'input_channels': 3,
                         'convolutions': {'1': [2, 16],
                                          '2': [2, 32],
                                          '3': [2, 32],
                                          '4': [2, 64],
                                          '5': [2, 64],
                                          '6': [2, 64],
                                          '7': [2, 64]},
                         'modules': {},
                         'last_max_pooling': True,
                         'normalization': {'feature_map_normalization': 'none'},
                         'enhancer_convolution_indices': [5, 6, 7]},
                     'Anchors': {
                         'scales_and_ratios': {'5': {'scales': [32, 64], 'aspect_ratios': [0.5, 1.0, 2.0]},
                                               '6': {'scales': [32, 64], 'aspect_ratios': [0.5, 1.0, 2.0]},
                                               '7': {'scales': [32, 64], 'aspect_ratios': [0.5, 1.0, 2.0]}}}}

    device = torch.device(device="cpu")
    dtype = torch.float32

    model = Model(configuration=configuration, device=device, dtype=dtype)

    print_green(output="Model successfully instantiated and ready for testing!", add_separators=True)

    batch_size = 2
    input_channels = 3
    input_image_size = 1024

    input_tensor = torch.randn(size=(batch_size, input_channels, input_image_size, input_image_size), dtype=dtype,
                               device=device)

    print_tensor_shape(tensor=input_tensor, name="input_tensor")

    print_blue(output="Executing forward pass with mock tensor...", add_separators=True)

    (final_backbone_tensor,
     backbone_output_tensor_dictionary,
     region_proposal_output_tensor_dictionary,
     aggregated_proposals_dictionary) = model(input_tensor=input_tensor)

    print_tensor_shape(tensor=final_backbone_tensor, name="final_backbone_tensor")

    print_blue(output="Backbone Output Tensor Dictionary:", add_separators=True, indent=1)

    for block_index, tensor in backbone_output_tensor_dictionary.items():
        print_tensor_shape(tensor=tensor, name=f"backbone_output_block_{block_index}", indent=1)

    print_blue(output="Region Proposal Output Tensor Dictionary For Anchor Modifications:", add_separators=True)

    for block_index, predictions in region_proposal_output_tensor_dictionary.items():
        print_blue(output=f"Block {block_index} Predictions:", add_separators=True)
        print_tensor_shape(tensor=predictions["classification_scores"], name="classification_scores", indent=1)
        print_tensor_shape(tensor=predictions["bounding_box_regressions"], name="bounding_box_regressions", indent=1)
        print_tensor_shape(tensor=predictions["proposal_boxes"], name="proposal_boxes", indent=1)

    print_blue(output="Aggregated Proposals and Anchors Dictionary:", add_separators=True)
    for key, tensor in aggregated_proposals_dictionary.items():
        print_tensor_shape(tensor=tensor, name=f"aggregated_{key}", indent=1)


if __name__ == "__main__":
    debug_model()
