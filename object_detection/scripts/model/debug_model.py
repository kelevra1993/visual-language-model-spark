from textwrap import indent

import torch
import yaml
from model.model import Model
import numpy as np
from scripts.utilities.debugging_utilities import print_bounding_boxes, generate_random_bounding_boxes, \
    visualize_anchor_target_assignments, visualize_foreground_and_background_anchors
from utilities.os_utilities import print_green, print_blue
from utilities.tensor_utilities import print_tensor_status, print_tensor_shape, print_tensor_list

from typing import Tuple, List, Dict, Any


def get_model_configuration() -> Tuple[Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies the default configuration dictionary, device, and dtype for testing the object detection Model pipeline.
    
    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The FPN F-RCNN configuration mapping, compute device, and tensor dtype.
    """
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
                         'enhancer_convolution_indices': [6, 7, 8]},
                     'Anchors': {
                         'scales_and_ratios': {
                             '6': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]},
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
                         'total_training_samples': 256,
                         'localisation_loss_beta': 1 / 9

                     }}
    device = torch.device(device="cpu")
    dtype = torch.float32

    return configuration, device, dtype


def get_dummy_input_tensor(device: torch.device, dtype: torch.dtype) -> Tuple[int, int, int, torch.Tensor]:
    """
    Generates a dummy batched input image tensor to feed the object detection model during debugging.
    
    Args:
        device (torch.device): The device (CPU/GPU) to allocate the tensor on.
        dtype (torch.dtype): The specific precision type for the tensor.
        
    Returns:
        Tuple[int, int, int, torch.Tensor]: A tuple containing the batch size, channel dimension, 
                                            spatial resolution, and the generated mock tensor itself.
    """
    batch_size = 2
    input_channels = 3
    input_image_size = 1024

    input_tensor = torch.randn(size=(batch_size, input_channels, input_image_size, input_image_size), dtype=dtype,
                               device=device)

    return batch_size, input_channels, input_image_size, input_tensor


def get_dummy_ground_truth_boxes(batch_size: int, input_image_size: int, configuration: Dict[str, Any],
                                 device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
    """
    Generates a realistic list of randomly positioned ground truth bounding boxes for each image in the batch.
    
    It extracts the unique bounding box scales and aspect ratios directly from the provided model configuration 
    to assure the randomly generated boxes closely resemble objects the Region Proposal Network expects.
    
    Args:
        batch_size (int): Number of images in the batch to generate boxes for.
        input_image_size (int): The maximum spatial resolution constraint for the random boxes.
        configuration (Dict[str, Any]): The master model configuration dict to dynamically extract scales and ratios.
        device (torch.device): The compute device allocation.
        dtype (torch.dtype): The specific precision type allocation.
        
    Returns:
        List[torch.Tensor]: A list containing the generated ground truth bounding boxes for each image in the batch.
    """
    # Dynamically retrieve a unique list of all scales and ratios from the FPN anchors configuration
    scales_and_ratios = configuration.get("Anchors", {}).get("scales_and_ratios", {})

    all_scales = set()
    all_aspect_ratios = set()
    for scale_level, config in scales_and_ratios.items():
        all_scales.update(config.get("scales", []))
        all_aspect_ratios.update(config.get("aspect_ratios", []))

    scales = list(all_scales)
    aspect_ratios = list(all_aspect_ratios)

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

    return ground_truth_bounding_boxes


def debug_model() -> None:
    """
    Instantiates the complete Model and runs a dummy forward pass to verify
    the outputs of the backbone and subsequently the region proposals.
    """

    configuration, device, dtype = get_model_configuration()

    model = Model(configuration=configuration, mode="training", device=device, dtype=dtype)

    print_green(output="Model Configuration + Instantiation Done!", add_separators=True)
    model.print_summary()

    batch_size, input_channels, input_image_size, input_tensor = get_dummy_input_tensor(device=device, dtype=dtype)

    ground_truth_bounding_boxes = get_dummy_ground_truth_boxes(batch_size=batch_size,
                                                               input_image_size=input_image_size,
                                                               configuration=configuration,
                                                               device=device,
                                                               dtype=dtype)

    print_blue(output="Visualising Ground Truths with Possitive and Negative Anchors...", add_separators=True)
    # Only run for the first batch to visualise the ground truth iou's with proposals.
    # visualize_foreground_and_background_anchors(
    #     ground_truth_boxes=ground_truth_bounding_boxes[0],
    #     anchors=model.anchors,
    #     background_iou_threshold=configuration["RegionProposal"]["background_iou_threshold"],
    #     foreground_iou_threshold=configuration["RegionProposal"]["foreground_iou_threshold"],
    #     input_image_size=input_image_size)

    for index, boxes in enumerate(ground_truth_bounding_boxes):
        print_tensor_shape(tensor=boxes, name=f"ground_truth_boxes_image_{index}", indent=1)

    model_output_dictionary = model(input_tensor=input_tensor, ground_truth_bounding_boxes=ground_truth_bounding_boxes)

    # Unpack the dictionary for the subsequent debugging output
    final_backbone_tensor = model_output_dictionary["final_backbone_tensor"]
    backbone_output_tensor_dictionary = model_output_dictionary["backbone_output_tensor_dictionary"]
    aggregated_proposals_dictionary = model_output_dictionary["aggregated_proposals_dictionary"]
    filtered_proposals_dictionary = model_output_dictionary["filtered_proposals_dictionary"]


if __name__ == "__main__":
    debug_model()
