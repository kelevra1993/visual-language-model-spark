"""
Tests of vision_training/models/detection/region_proposer.py: RegionProposal against the reference outputs, through
check_nn_module_method of tests/testing_utilities.py, and the batched proposal filter, which keeps the best
proposals of every image after the suppression and pads the images left short with copies of their own survivors,
the per-image suppression of the training keeping the same proposals as the single call of the deployment.
"""

import torch
from pathlib import Path

from vision_training.models.detection.region_proposer import (RegionProposal, RegionProposalFilter,
                                                               suppress_overlapping_proposals)
from tests.testing_utilities import check_nn_module_method, create_deterministic_tensor


def test_region_proposer() -> None:
    # Define batch size for the test
    batch_size = 4
    input_channels = 256
    feature_map_size = 6
    feature_map_shape = (input_channels, feature_map_size, feature_map_size)

    # Create input tensors and populate the simple and batched dictionaries
    input_tensor = create_deterministic_tensor(shape=(1,) + feature_map_shape)

    simple_input_dictionary = {"input_tensor": input_tensor}

    batched_input_tensor = input_tensor.broadcast_to(size=(batch_size,) + feature_map_shape)

    batched_input_dictionary = {"input_tensor": batched_input_tensor}

    # Initialize device and dtype for configurations
    device = torch.device(device="cpu")
    dtype = torch.float32

    # Create a dictionary of configurations to test various paths of the RegionProposal
    region_proposal_configurations = {
        "basic_region_proposal": {
            "convolutions": [2, 64],
            "input_channels": input_channels,
            "scales": [32.0, 64.0],
            "aspect_ratios": [0.5, 1.0, 2.0],
            "input_image_size": 1024,
            "feature_map_size": feature_map_size,
            "feature_map_normalization": "batch",
            "batch_normalization_momentum": 0.1,
            "device": device,
            "dtype": dtype},
        "complex_region_proposal": {
            "convolutions": [3, 32],
            "input_channels": input_channels,
            "scales": [16.0, 32.0, 64.0, 128.0],
            "aspect_ratios": [0.5, 1.0, 2.0],
            "input_image_size": 1024,
            "feature_map_size": feature_map_size,
            "feature_map_normalization": "layer",
            "batch_normalization_momentum": 0.1,
            "device": device,
            "dtype": dtype}}

    # Iterate through the defined configurations to test each structural permutation
    for configuration_name, configuration_dictionary in region_proposal_configurations.items():
        print(f"Testing RegionProposal Configuration: {configuration_name}")

        # Instantiate the region proposal module with the current configuration parameters
        region_proposal_module = RegionProposal(**configuration_dictionary)

        # Call the testing utility to verify the forward pass deterministically
        # RegionProposal returns the objectness scores, the box transformations and the proposal boxes
        check_nn_module_method(
            module=region_proposal_module,
            simple_input_dictionary=simple_input_dictionary,
            output_tensor_names=[f"region_proposal_{configuration_name}_scores",
                                 f"region_proposal_{configuration_name}_box_transformations",
                                 f"region_proposal_{configuration_name}_boxes"],
            reference_folder=Path(__file__).parents[1] / "components" / "reference_values",
            batch_size=batch_size,
            batched_input_dictionary=batched_input_dictionary,
            save_output=False)

        print(f" - {configuration_name} Test Completed Successfully.")


def test_proposal_filter_keeps_the_best_survivors_of_every_image() -> None:
    """
    Verifies the filtered proposals of a batch of two images: the duplicates of a better proposal are suppressed, the
    survivors of every image come in decreasing score order with their score and level, a proposal outside the
    pre-suppression limit is never kept, and the image left with one survivor fills its other slots with copies of it,
    the mask of the survivors leaving these copies out.
    """
    proposal_filter = RegionProposalFilter(configuration={
        "training": {"pre_nms_proposals": 4, "post_nms_proposals": 3},
        "inference": {"pre_nms_proposals": 4, "post_nms_proposals": 3}, "nms_iou_threshold": 0.5}).train()
    boxes = torch.tensor(data=[[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 9.0], [20.0, 20.0, 30.0, 30.0],
                                [40.0, 40.0, 50.0, 50.0], [60.0, 60.0, 70.0, 70.0]],
                               [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 9.5], [0.0, 0.0, 9.5, 10.0],
                                [1.0, 0.0, 10.0, 10.0], [60.0, 60.0, 70.0, 70.0]]])
    logits = torch.tensor(data=[[3.0, 2.0, 1.0, -5.0, -6.0], [3.0, 2.5, 2.2, 2.1, -6.0]])
    origins = torch.tensor(data=[[5, 6, 7, 5, 6], [7, 6, 5, 7, 6]])

    filtered_boxes, filtered_scores, filtered_origins, valid_proposal_mask = proposal_filter(
        proposal_boxes=boxes, proposal_scores=logits, proposal_origins=origins, input_image_size=100)
    assert filtered_boxes[0].tolist() == [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0], [40.0, 40.0, 50.0, 50.0]]
    assert torch.equal(input=filtered_scores[0], other=torch.sigmoid(input=logits[0, [0, 2, 3]]))
    assert filtered_origins.tolist() == [[5, 7, 5], [7, 7, 7]]
    assert filtered_boxes[1].tolist() == [[0.0, 0.0, 10.0, 10.0]] * 3
    assert valid_proposal_mask.tolist() == [[True, True, True], [True, False, False]]


def test_per_image_and_batched_suppressions_keep_the_same_proposals() -> None:
    """
    Verifies that the suppression of the training, one call per image, and that of the deployed detector, a single
    call over the batch, keep exactly the same proposals of every image.
    """
    generator = torch.Generator().manual_seed(0)
    corners = torch.rand(size=(4, 300, 2), generator=generator) * 400
    boxes = torch.cat(tensors=[corners, corners + 8 + torch.rand(size=(4, 300, 2), generator=generator) * 100], dim=2)
    scores = torch.rand(size=(4, 300), generator=generator)
    per_image_mask = suppress_overlapping_proposals(boxes=boxes, scores=scores, nms_iou_threshold=0.5,
                                                    suppress_per_image=True)
    batched_mask = suppress_overlapping_proposals(boxes=boxes, scores=scores, nms_iou_threshold=0.5,
                                                  suppress_per_image=False)
    assert torch.equal(input=per_image_mask, other=batched_mask)
    assert 0 < int(per_image_mask.sum()) < per_image_mask.numel()


if __name__ == "__main__":
    test_region_proposer()
