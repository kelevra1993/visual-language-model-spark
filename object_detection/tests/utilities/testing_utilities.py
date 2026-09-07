import os
import torch
from torch import nn
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path


def check_nn_module_method(module: nn.Module, input_tensor_dictionary: Dict[str, torch.Tensor],
                           output_tensor_names: List[str],
                           reference_folder: Union[str, Path], batch_size: int,
                           batched_input_dictionary: Optional[Dict[str, torch.Tensor]] = None,
                           use_kwargs: bool = True) -> None:
    """
    Performs a deterministic regression test on a PyTorch module's forward pass to validate structural consistency
    within the testing pipeline.

    This utility ensures architectural consistency by comparing a module's output
    against saved reference tensors. It eliminates non-determinism by setting the module to evaluation mode, 
    converting to double precision, and overwriting parameters deterministically. The function validates 
    both a single-instance pass and a batched pass using broadcasting to ensure the module correctly
    handles batch dimensions.

    Args:
        module (nn.Module): The neural network module to be tested.
        input_tensor_dictionary (Dict[str, torch.Tensor]): A dictionary mapping input argument names to their respective
        input tensors.
        output_tensor_names (List[str]): A list of names for the expected output tensors, matching the filenames in the
        reference folder.
        reference_folder (Union[str, Path]): The path to the directory containing the '.pt' reference files.
        batch_size (int): The size of the batch to simulate for the batched consistency check.
        batched_input_dictionary (Optional[Dict[str, torch.Tensor]]): An optional dictionary of pre-batched inputs.
         If None, it broadcasts the simple inputs.
        use_kwargs (bool): Whether to pass inputs to the module as keyword arguments.
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

    # Construct the dictionaries for both simple and batched inputs to test both processing scenarios
    simple_input_dictionary = {input_name: input_tensor for input_name, input_tensor in input_tensor_dictionary.items()}
    if batched_input_dictionary is None:
        # Broadcast the simple input tensors to match the simulated batch size for consistency verification
        batched_input_dictionary = {input_name: input_tensor.broadcast_to(size=(batch_size,) + input_tensor.shape)
                                    for input_name, input_tensor in input_tensor_dictionary.items()}

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
            simple_output, batched_output,
            output_file_names, batched_output_file_names,
            output_tensor_names):

        # Load the expected reference tensor for the simple forward pass and verify it matches the computed output
        expected_output_tensor = torch.load(f=output_filename, weights_only=True)
        assert torch.allclose(input=output_tensor, other=expected_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Simple Check.'

        # Determine the expected batched reference tensor by either loading it from disk or broadcasting the simple reference
        if os.path.exists(path=batched_output_filename):
            expected_batched_output_tensor = torch.load(f=batched_output_filename, weights_only=True)
        else:
            expected_output_batch_shape = (batch_size,) + expected_output_tensor.shape
            expected_batched_output_tensor = expected_output_tensor.unsqueeze(dim=0).broadcast_to(size=expected_output_batch_shape)

        # Verify that the computed batched output matches the expected batched reference tensor
        assert torch.allclose(input=batched_output_tensor, other=expected_batched_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Batched Check.'

def create_deterministic_tensor(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
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
