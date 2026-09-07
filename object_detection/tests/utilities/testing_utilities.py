import os
import torch
from torch import nn
from typing import List, Dict, Union
from pathlib import Path


def check_nn_module_method(module: nn.Module, input_tensor_dictionary: Dict[str, torch.Tensor],
                           output_tensor_names: List[str],
                           reference_folder: str | Path, batch_size: int,
                           batched_input_tensor_dictionary: Dict[str, torch.Tensor] = None,
                           use_kwargs=True):
    """
    Performs a deterministic regression test on a PyTorch module's forward pass.

    This utility ensures architectural consistency by comparing a module's output
    against saved reference tensors. It eliminates non-determinism by:
    1. Setting the module to evaluation mode (disabling Dropout, BatchNorm, etc.).
    2. Converting the module and inputs to double precision (float64) to minimize numerical drift.
    3. Overwriting all module parameters with a fixed, deterministic sequence (torch.linspace).

    The function validates both a single-instance "simple" pass and a "batched" pass
    using broadcasting to ensure the module correctly handles batch dimensions.

    :param module: The nn.Module to be tested.
    :param input_tensor_dictionary: A dictionary mapping input argument names to their
                                    respective input tensors.
    :param output_tensor_names: A list of names for the expected output tensors,
                                matching the filenames in the reference_folder.
    :param reference_folder: Path to the directory containing the '.pt' reference files.
    :param batch_size: The size of the batch to simulate for the batched consistency check.
    :param batched_input_tensor_dictionary: Optional dictionary of pre-batched inputs. If None, it broadcasts the simple inputs.
    """
    # First We Just Modify The Parameters Of The Module To Be Fixed
    # Set To eval() To Disable Dropout Or Any Batch Normalisation (anything non deterministic)
    module.eval()
    module.double()

    # Make module parameters to always be the same in order to have the exact same outputs for testing.
    with torch.no_grad():
        for param in module.parameters():
            # Set to the same parameters.
            param.copy_(torch.linspace(-1, 1, param.numel()).reshape(param.shape))

    # Simple Input And Batched Input
    simple_input_dictionary = {input_name: input_tensor for input_name, input_tensor in input_tensor_dictionary.items()}
    if batched_input_tensor_dictionary is not None:
        batched_input_dictionary = batched_input_tensor_dictionary
    else:
        batched_input_dictionary = {input_name: input_tensor.broadcast_to((batch_size,) + input_tensor.shape)
                                    for input_name, input_tensor in input_tensor_dictionary.items()}

    # Simple Output And Batched Output
    if use_kwargs:
        simple_output = module(**simple_input_dictionary)
        batched_output = module(**batched_input_dictionary)
    else:
        simple_output = module(simple_input_dictionary).values()
        batched_output = module(batched_input_dictionary).values()

    if isinstance(simple_output, torch.Tensor):
        simple_output = [simple_output]

    if isinstance(batched_output, torch.Tensor):
        batched_output = [batched_output]

    out_file_names = [f'{reference_folder}/{out_name}.pt' for out_name in output_tensor_names]
    batched_out_file_names = [f'{reference_folder}/{out_name}_batched.pt' for out_name in output_tensor_names]

    for output_tensor, batched_output_tensor, output_filename, batched_output_filename, output_tensor_name in zip(
            simple_output, batched_output,
            out_file_names, batched_out_file_names,
            output_tensor_names):

        expected_output_tensor = torch.load(output_filename, weights_only=True)
        assert torch.allclose(output_tensor, expected_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Simple Check.'

        if os.path.exists(batched_output_filename):
            expected_batched_output_tensor = torch.load(batched_output_filename, weights_only=True)
        else:
            expected_out_batch_shape = (batch_size,) + expected_output_tensor.shape
            expected_batched_output_tensor = expected_output_tensor.unsqueeze(0).broadcast_to(expected_out_batch_shape)

        assert torch.allclose(batched_output_tensor, expected_batched_output_tensor, atol=1e-5), \
            f'Problem With output {output_tensor_name} For Batched Check.'
