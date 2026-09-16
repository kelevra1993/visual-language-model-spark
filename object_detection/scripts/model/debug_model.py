from textwrap import indent

import torch
import yaml
from model.model import Model
import numpy as np
from scripts.utilities.debugging_utilities import print_bounding_boxes, generate_random_bounding_boxes, \
    visualize_anchor_target_assignments, visualize_foreground_and_background_anchors
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
                                          '7': [2, 64],
                                          '8': [2, 64],
                                          '9': [2, 64],
                                          },
                         'modules': {},
                         'last_max_pooling': True,
                         'normalization': {'feature_map_normalization': 'none'},
                         'enhancer_convolution_indices': [5, 7, 8]},
                     'Anchors': {
                         'scales_and_ratios': {
                             '5': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]},
                             '7': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]},
                             '8': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]}
                         }},
                     'RegionProposal': {
                         'nms_iou_threshold': 0.7,
                         'training': {'pre_nms_proposals': 500, 'post_nms_proposals': 250},
                         'inference': {'pre_nms_proposals': 100, 'post_nms_proposals': 50},
                         'foreground_iou_threshold': {"min": 0.35, "max": 1.0},
                         'background_iou_threshold': {"min": 0.2, "max": 0.3},
                         'strict_fallback_assignment': False,
                         'number_training_positives': 128,
                         'total_training_samples': 256

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

    print_blue(output="Generating random ground truth boxes...", add_separators=True)
    scales = [64.0, 128.0]
    aspect_ratios = [0.5, 1.0, 2.0]
    ground_truth_bounding_boxes = []

    for batch_index in range(batch_size):
        # Generate a realistic random quantity of boxes (e.g., between 10 and 20 objects per image)
        random_number_of_boxes = int(np.random.randint(low=10, high=20))
        mock_boxes = generate_random_bounding_boxes(
            scales=scales, aspect_ratios=aspect_ratios,
            input_image_size=input_image_size,
            number_of_boxes=random_number_of_boxes,
            device=device, dtype=dtype)
        ground_truth_bounding_boxes.append(mock_boxes)

    for index, boxes in enumerate(ground_truth_bounding_boxes):
        print_tensor_shape(tensor=boxes, name=f"ground_truth_boxes_image_{index}", indent=1)

    print_blue(output="Executing forward pass with mock tensor...", add_separators=True)

    model_output_dictionary = model(
        input_tensor=input_tensor, ground_truth_bounding_boxes=ground_truth_bounding_boxes)

    # Only run for the first batch to visualise the ground truth iou's with proposals.
    visualize_foreground_and_background_anchors(
        ground_truth_boxes=ground_truth_bounding_boxes[0],
        anchors=model.anchors,
        background_iou_threshold=configuration["RegionProposal"]["background_iou_threshold"],
        foreground_iou_threshold=configuration["RegionProposal"]["foreground_iou_threshold"],
        input_image_size=input_image_size)

    # Unpack the dictionary for the subsequent debugging output
    final_backbone_tensor = model_output_dictionary["final_backbone_tensor"]
    backbone_output_tensor_dictionary = model_output_dictionary["backbone_output_tensor_dictionary"]
    region_proposal_output_tensor_dictionary = model_output_dictionary["region_proposal_output_tensor_dictionary"]
    aggregated_proposals_dictionary = model_output_dictionary["aggregated_proposals_dictionary"]
    filtered_proposals_dictionary = model_output_dictionary["filtered_proposals_dictionary"]
    region_proposal_anchor_targets = model_output_dictionary["region_proposal_anchor_targets"]
    region_proposal_anchor_labels = model_output_dictionary["region_proposal_anchor_labels"]

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
            print_bounding_boxes(boxes=tensor[0], number_of_boxes=5)

    print_blue(output="Visualizing Anchor Target Assignments...", add_separators=True)
    visualize_anchor_target_assignments(ground_truth_bounding_boxes=ground_truth_bounding_boxes,
                                        batched_target_ground_truth_boxes=region_proposal_anchor_targets,
                                        batched_labels=region_proposal_anchor_labels,
                                        batched_anchors=aggregated_proposals_dictionary["anchors"],
                                        input_image_size=input_image_size)


if __name__ == "__main__":
    debug_model()
