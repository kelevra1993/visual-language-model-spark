from textwrap import indent

import torch
import yaml
from model.model import Model
from scripts.utilities.debugging_utilities import print_bounding_boxes
from utilities.os_utilities import print_green, print_blue
from utilities.tensor_utilities import print_tensor_status, print_tensor_shape, print_tensor_list


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
                                               '7': {'scales': [32, 64], 'aspect_ratios': [0.5, 1.0, 2.0]}}},
                     'RegionProposal': {
                         'nms_iou_threshold': 0.7,
                         'training': {'pre_nms_proposals': 12000, 'post_nms_proposals': 2000},
                         'inference': {'pre_nms_proposals': 6000, 'post_nms_proposals': 1000}
                     }}

    device = torch.device(device="cpu")
    dtype = torch.float32

    model = Model(configuration=configuration, mode="training", device=device, dtype=dtype)

    print_green(output="Model successfully instantiated and ready for testing!", add_separators=True)

    batch_size = 2
    input_channels = 3
    input_image_size = 1024

    input_tensor = torch.randn(size=(batch_size, input_channels, input_image_size, input_image_size), dtype=dtype,
                               device=device)

    print_tensor_shape(tensor=input_tensor, name="input_tensor")

    print_blue(output="Executing forward pass with mock tensor...", add_separators=True)

    (final_backbone_tensor, backbone_output_tensor_dictionary, region_proposal_output_tensor_dictionary,
     aggregated_proposals_dictionary, filtered_proposals_dictionary) = model(input_tensor=input_tensor)

    print_tensor_shape(tensor=final_backbone_tensor, name="final_backbone_tensor")

    print_blue(output="Backbone Output Tensor Dictionary:", add_separators=True)

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

    print_blue(output="Filtered Region Proposals Dictionary (Post-NMS):", add_separators=True)
    for key, tensor in filtered_proposals_dictionary.items():
        print_tensor_shape(tensor=tensor, name=key, indent=1)
        if "score" in key:
            print_tensor_list(tensor=tensor[0, :15])
        else:
            print_bounding_boxes(boxes=tensor[0],number_of_boxes=5)



if __name__ == "__main__":
    debug_model()
