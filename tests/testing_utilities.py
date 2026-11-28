"""
Shared helpers of the model tests: checking an nn.Module method against reference outputs, creating deterministic
tensors and building the small seeded detector (or instance segmenter) used by the detector baseline, inference and
instance segmentation tests, and the weighted centroid with which the data tests check that the resized and augmented
masks stay aligned with their images.
"""

import copy
import os
import torch
from torch import nn
from typing import Any, List, Dict, Union, Optional, Tuple
from pathlib import Path

from vision_training.models.detection.two_stage_detector import TwoStageDetector
from vision_training.models.instance_segmentation.instance_segmenter import InstanceSegmenter
from tests.fixtures.configuration_fixtures import get_detection_model_configuration


def check_nn_module_method(module: nn.Module, simple_input_dictionary: Dict[str, torch.Tensor],
                           output_tensor_names: List[str], reference_folder: Union[str, Path], batch_size: int,
                           batched_input_dictionary: Optional[Dict[str, torch.Tensor]] = None, use_kwargs: bool = True,
                           save_output: bool = False) -> None:
    """
    Performs a deterministic regression test on a PyTorch module's forward pass to validate structural consistency
    within the testing pipeline.

    This utility ensures architectural consistency by comparing a module's output
    against saved reference tensors. It eliminates non-determinism by setting the module to evaluation mode,
    converting to double precision, and overwriting parameters deterministically. The function validates
    both a single-instance pass and a batched pass using broadcasting to ensure the module
    correctly handles batch dimensions.

    Args:
        module (nn.Module): The neural network module to be tested.
        simple_input_dictionary (Dict[str, torch.Tensor]): A dictionary mapping input argument names
        to their respective input tensors.
        output_tensor_names (List[str]): A list of names for the expected output tensors, matching the filenames in the
        reference folder.
        reference_folder (Union[str, Path]): The path to the directory containing the '.pt' reference files.
        batch_size (int): The size of the batch to simulate for the batched consistency check.
        batched_input_dictionary (Optional[Dict[str, torch.Tensor]]): An optional dictionary of pre-batched inputs.
         If None, it broadcasts the simple inputs.
        use_kwargs (bool): Whether to pass inputs to the module as keyword arguments.
        save_output (bool): If True, saves the computed outputs to the reference_folder instead of asserting.
    """
    # Set the module to evaluation mode to disable non-deterministic operations such as dropout or batch normalization
    module.eval()

    # Convert the module parameters to double precision to minimize numerical drift during testing
    module.double()

    # Iterate through all module parameters and overwrite them deterministically to ensure reproducible test outputs
    with torch.no_grad():
        for parameter in module.parameters():
            # Replace the parameter values with a linearly spaced tensor matching its original shape
            parameter.copy_(torch.linspace(start=-1, end=1, steps=parameter.numel()).reshape(shape=parameter.shape))

    if batched_input_dictionary is None:
        # Broadcast the simple input tensors to match the simulated batch size for consistency verification
        batched_input_dictionary = {input_name: input_tensor.broadcast_to(size=(batch_size,) + input_tensor.shape)
                                    for input_name, input_tensor in simple_input_dictionary.items()}

    # Execute the forward pass for both the simple and batched inputs to retrieve the module outputs
    if use_kwargs:
        simple_output = module(**simple_input_dictionary)
        batched_output = module(**batched_input_dictionary)
    else:
        simple_output = module(simple_input_dictionary).values()
        batched_output = module(batched_input_dictionary).values()

    # Normalize the output format to lists to uniformly handle single and multiple output scenarios
    if isinstance(simple_output, torch.Tensor):
        simple_output = [simple_output]

    if isinstance(batched_output, torch.Tensor):
        batched_output = [batched_output]

    # Generate the expected file paths for the reference tensors based on the provided output names
    output_file_names = [f'{reference_folder}/{output_name}.pt' for output_name in output_tensor_names]
    batched_output_file_names = [f'{reference_folder}/{output_name}_batched.pt' for output_name in output_tensor_names]

    # Iterate over the produced outputs and their corresponding reference files to validate correctness
    for output_tensor, batched_output_tensor, output_filename, batched_output_filename, output_tensor_name in zip(
            simple_output, batched_output, output_file_names, batched_output_file_names, output_tensor_names):

        if save_output:
            # Create directories if they do not exist to ensure saving succeeds
            os.makedirs(name=Path(output_filename).parent, exist_ok=True)
            # Save the computed tensors directly to disk and skip assertions
            torch.save(obj=output_tensor, f=output_filename)
            torch.save(obj=batched_output_tensor, f=batched_output_filename)
            print(f"Saved reference values for {output_tensor_name}")
            continue

        # Load the expected reference tensor for the simple forward pass and verify it matches the computed output
        expected_output_tensor = torch.load(f=output_filename, weights_only=True)
        assert torch.allclose(input=output_tensor, other=expected_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Simple Check.'

        # Determine the expected batched reference tensor by either loading it from disk or broadcasting the simple
        # reference
        if os.path.exists(path=batched_output_filename):
            expected_batched_output_tensor = torch.load(f=batched_output_filename, weights_only=True)
        else:
            expected_output_batch_shape = (batch_size,) + expected_output_tensor.shape
            expected_batched_output_tensor = expected_output_tensor.unsqueeze(dim=0).broadcast_to(
                size=expected_output_batch_shape)

        # Verify that the computed batched output matches the expected batched reference tensor
        assert torch.allclose(input=batched_output_tensor, other=expected_batched_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Batched Check.'


def create_deterministic_tensor(shape: Tuple[int, ...], dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Creates a deterministic tensor with a specific shape and data type using a linear space.

    This function ensures that the input tensors used for testing are identical across
    different runs and environments by populating them with linearly spaced values
    between -total_elements and total_elements, rather than using random initialization.

    Args:
        shape (Tuple[int, ...]): The desired shape of the output tensor.
        dtype (torch.dtype): The desired data type of the output tensor.
    """
    # Calculate the total number of elements required to fill the specified shape
    total_elements = torch.empty(size=shape).numel()

    # Generate a linearly spaced 1D tensor and reshape it to the requested dimensions
    deterministic_tensor = torch.linspace(start=-total_elements, end=total_elements, steps=total_elements, dtype=dtype)
    return deterministic_tensor.reshape(shape=shape)


def compute_weighted_centroid(weights: torch.Tensor) -> torch.Tensor:
    """
    Gives the centroid of a [H, W] map of pixel weights, for the tests checking that a mask resized or moved like its
    image stays on the image pixels: the centroid of the mask pixels of an object is compared with that of the
    brightness of the object in the resampled image, both in pixel index coordinates.

    Args:
        weights (torch.Tensor): The [H, W] non-negative weights, e.g. a binary mask or the brightness of an image.

    Returns:
        torch.Tensor: The float64 (x, y) centroid, in pixel indices.
    """
    float_weights = weights.to(dtype=torch.float64)
    row_indices, column_indices = torch.meshgrid(torch.arange(end=weights.shape[0], dtype=torch.float64),
                                                 torch.arange(end=weights.shape[1], dtype=torch.float64), indexing="ij")
    return torch.stack(tensors=[(column_indices * float_weights).sum(), (row_indices * float_weights).sum()]) / \
        float_weights.sum()


def build_test_model(dtype: torch.dtype, instance_segmentation: bool = False,
                     mask_head_settings: Optional[Dict[str, Any]] = None, neck_type: str = "none",
                     box_loss: str = "smooth_l1", nms_criterion: str = "iou",
                     nms_distance_exponent: float = 1.0, proposal_selection: str = "joint",
                     shared_region_proposal: bool = False, shared_detector: bool = False) -> TwoStageDetector:
    """
    Builds a small untrained detector from the debug configuration, shared by the detector tests (export path and
    behaviour baseline), with 512 pixel inputs (the smallest size fitting its 9 pooling blocks) and a score threshold
    of zero so that the compared outputs always contain detections. The weights are drawn from a fixed seed, so every
    call returns exactly the same detector. With instance_segmentation, the instance segmenter of the same detector
    with mask heads is built instead, for the instance segmentation tests.

    Args:
        dtype (torch.dtype): The precision of the parameters: float32 like the exported models, or float64 for the
            behaviour baseline, whose recorded values must not depend on float32 rounding.
        instance_segmentation (bool): Whether to build the InstanceSegmenter instead of the TwoStageDetector.
            Defaults to False.
        mask_head_settings (Optional[Dict[str, Any]]): The mask geometry replacing the one of the fixture: the
            'MaskHead' settings under 'MaskHead' (e.g. 'mask_resolution', 'upsampling') and the mask head of every
            detection level under 'level_mask_heads', None to keep them. Defaults to None.
        neck_type (str): The 'Neck.type' between the backbone blocks and the heads. Defaults to "none", the heads
            reading the blocks, as in the behaviour baseline.
        box_loss (str): The 'Detector.box_loss' of the detector loss, the region proposal network always using the
            smooth L1. Defaults to "smooth_l1", as in the behaviour baseline.
        nms_criterion (str): The 'Detector.nms_criterion' of the final NMS. Defaults to "iou", the greedy NMS of the
            behaviour baseline.
        nms_distance_exponent (float): The 'Detector.nms_distance_exponent' of "distance_iou". Defaults to 1.0.
        proposal_selection (str): The 'RegionProposal.proposal_selection' of the proposal filter. Defaults to "joint",
            the selection over all the levels of the behaviour baseline.
        shared_region_proposal (bool): The 'RegionProposal.shared', one region proposal head applied to every level.
            Defaults to False, one head per level as in the behaviour baseline.
        shared_detector (bool): The 'Detector.shared', one detection head reading the proposals of every level.
            Defaults to False, one head per level as in the behaviour baseline.

    Returns:
        TwoStageDetector: The detector (or instance segmenter) in evaluation mode, on the CPU.
    """
    torch.manual_seed(seed=0)
    configuration, device, _ = get_detection_model_configuration(mask_head=instance_segmentation, neck_type=neck_type,
                                                                 box_loss=box_loss)
    configuration: Dict[str, Any] = copy.deepcopy(x=configuration)
    configuration["Data"]["image_settings"]["size"] = 512
    configuration["Detector"]["inference"]["score_threshold"] = 0.0
    configuration["Detector"].update({"nms_criterion": nms_criterion, "nms_distance_exponent": nms_distance_exponent})
    configuration["RegionProposal"].update({"proposal_selection": proposal_selection,
                                            "shared": shared_region_proposal})
    configuration["Detector"]["shared"] = shared_detector
    # Replace the mask geometry of the fixture: the settings shared by the levels and the mask head of every level
    if mask_head_settings is not None:
        configuration["MaskHead"].update(mask_head_settings["MaskHead"])
        for level_key, level_mask_head in mask_head_settings["level_mask_heads"].items():
            configuration["DetectionLevels"][level_key]["mask_head"] = level_mask_head

    # Move the model to its device and precision after building it, as the Trainer does, since a few layers (the
    # region proposal output layers) are created in the default float32 precision
    model_class = InstanceSegmenter if instance_segmentation else TwoStageDetector
    model = model_class(configuration=configuration, device=device, dtype=dtype)
    return model.to(device=device, dtype=dtype).eval()


def build_whole_input_windows(images: torch.Tensor) -> torch.Tensor:
    """
    Gives the image windows of a batch of square images filling the whole network input, [0, 0, S, S] for every image,
    the windows the detector tests give with their random images (which have no padding), so that the detections are
    clipped to the input as before the windows existed.

    Args:
        images (torch.Tensor): The [B, 3, S, S] images.

    Returns:
        torch.Tensor: The [B, 4] float32 windows, on the device of the images.
    """
    image_size = images.shape[-1]
    return torch.tensor(data=[[0.0, 0.0, image_size, image_size]], dtype=torch.float32,
                        device=images.device).expand(images.shape[0], -1).contiguous()
