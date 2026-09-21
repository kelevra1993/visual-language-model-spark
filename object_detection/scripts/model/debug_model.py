from textwrap import indent

import torch
import yaml
from model.model import Model
import numpy as np
from scripts.utilities.debugging_utilities import print_bounding_boxes, generate_random_bounding_boxes, \
    visualize_anchor_target_assignments, visualize_foreground_and_background_anchors, get_model_configuration, \
    get_dummy_input_tensor, get_dummy_ground_truth_boxes, get_dummy_ground_truth_labels
from utilities.os_utilities import print_green, print_blue
from utilities.tensor_utilities import print_tensor_status, print_tensor_shape, print_tensor_list

from typing import Tuple, List, Dict, Any


def debug_model() -> None:
    """
    Instantiates the complete Model and runs a dummy forward pass to verify
    the outputs of the backbone and subsequently the region proposals.
    """

    configuration, device, dtype = get_model_configuration()
    number_classes = configuration.get("Data").get("number_classes")

    model = Model(configuration=configuration, device=device, dtype=dtype, verbose=True)

    print_green(output="Model Configuration + Instantiation Done!", add_separators=True)
    model.print_summary()

    batch_size, input_channels, input_image_size, input_tensor = get_dummy_input_tensor(device=device, dtype=dtype)

    ground_truth_bounding_boxes = get_dummy_ground_truth_boxes(batch_size=batch_size,
                                                               input_image_size=input_image_size,
                                                               configuration=configuration,
                                                               device=device,
                                                               dtype=dtype)

    ground_truth_labels = get_dummy_ground_truth_labels(ground_truth_boxes=ground_truth_bounding_boxes,
                                                        number_classes=number_classes,
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

    model_output_dictionary = model(input_tensor=input_tensor,
                                    ground_truth_bounding_boxes=ground_truth_bounding_boxes,
                                    ground_truth_labels=ground_truth_labels)


if __name__ == "__main__":
    debug_model()
