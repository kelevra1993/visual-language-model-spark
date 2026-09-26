import torch
import numpy as np
from typing import Optional, Any

from utilities.os_utilities import print_blue, print_yellow, print_green, get_variable_name


def get_device() -> torch.device:
    """
    Identifies and returns the most efficient available hardware device for tensor computations.

    This utility ensures that the project remains cross-platform compatible by prioritizing
    CUDA (NVIDIA GPUs), then MPS (Apple Silicon GPUs), and falling back to CPU if no
    accelerators are detected. It is used globally across all modules to maintain
    device consistency.

    Returns:
        torch.device: The detected torch.device (e.g., 'cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def print_tensor_shape(tensor: torch.Tensor, name: Optional[str] = None, indent: int = 0) -> None:
    """
    Logs the shape of a tensor to the console in a formatted blue string.

    In the global project context, this utility is used during development and debugging
    to verify that tensor dimensions align with expected shapes (e.g., MSA or Pair representations)
    after complex transformations or contractions.

    Args:
        tensor (torch.Tensor): The torch.Tensor whose shape will be printed. Shape: (*).
        name (Optional[str], optional): An optional label to identify the tensor in the output.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.
    """
    if not name:
        name = get_variable_name(target_variable=tensor)
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    print_blue(output=f"{indentation_string}Tensor {name} Is Of Shape : {list(tensor.shape)}")


def print_tensor_type(tensor: torch.Tensor, name: Optional[str] = None, indent: int = 0) -> None:
    """
    Logs the data type of a tensor to the console in a formatted yellow string.

    Ensures that tensors maintain consistent dtypes (e.g., torch.float32) across different
    architectural modules, preventing type mismatch errors during multi-platform
    execution (CPU/CUDA/MPS).

    Args:
        tensor (torch.Tensor): The torch.Tensor whose dtype will be printed. Shape: (*).
        name (Optional[str], optional): An optional label to identify the tensor in the output.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.
    """
    if not name:
        name = get_variable_name(target_variable=tensor)
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    print_yellow(output=f"{indentation_string}Tensor {name} Is Of Type : {tensor.dtype}")


def print_tensor_average(tensor: torch.Tensor, name: Optional[str] = None, indent: int = 0) -> None:
    """
    Logs the mean (average) value of a tensor to the console.

    This utility is crucial for debugging the center of mass of tensor distributions,
    ensuring that normalization layers and gradients are behaving consistently and have not
    drifted significantly.

    Args:
        tensor (torch.Tensor): The torch.Tensor to compute the average value for.
        name (Optional[str], optional): An optional label to identify the tensor in the output.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.
    """
    if not name:
        name = get_variable_name(target_variable=tensor)
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    
    # Safely compute mean as float (to handle int/bool tensors gracefully)
    average_value = tensor.float().mean().item()
    print_yellow(output=f"{indentation_string}Tensor {name} Average Is : {average_value:.6f}")


def print_tensor_min_max(tensor: torch.Tensor, name: Optional[str] = None, indent: int = 0) -> None:
    """
    Logs the minimum and maximum values of a tensor to the console.

    This utility is crucial for debugging potential exploding or vanishing gradients and
    verifying normalization steps within the processing pipeline, ensuring tensor values remain
    within expected mathematical bounds.

    Args:
        tensor (torch.Tensor): The torch.Tensor to compute the min and max values for.
        name (Optional[str], optional): An optional label to identify the tensor in the output.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.
    """
    if not name:
        name = get_variable_name(target_variable=tensor)
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    print_yellow(output=f"{indentation_string}Tensor {name} Maximum Is : {tensor.max()}")
    print_yellow(output=f"{indentation_string}Tensor {name} Minimum Is : {tensor.min()}")


def print_tensor_device(tensor: torch.Tensor, name: Optional[str] = None, indent: int = 0) -> None:
    """
    Logs the hardware device of a tensor to the console in a formatted green string.

    Crucial for identifying and resolving device placement issues, ensuring all tensors
    participating in an operation reside on the same hardware (CUDA, MPS, or CPU) as
    mandated by the project's cross-platform compatibility guidelines.

    Args:
        tensor (torch.Tensor): The torch.Tensor whose device will be printed. Shape: (*).
        name (Optional[str], optional): An optional label to identify the tensor in the output.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.
    """
    if not name:
        name = get_variable_name(target_variable=tensor)
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    print_green(output=f"{indentation_string}Tensor {name} Is On : {tensor.device}")


def print_tensor_status(tensor: torch.Tensor, name: Optional[str] = None, indent: int = 0) -> None:
    """
    Provides a comprehensive log of a tensor's shape, type, and device.

    Aggregates individual printing utilities to offer a single-point snapshot of a tensor's
    state. This is particularly useful for debugging

    Args:
        tensor (torch.Tensor): The torch.Tensor to inspect. Shape: (*).
        name (Optional[str], optional): An optional label to identify the tensor in the output.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.
    """
    if not name:
        name = get_variable_name(target_variable=tensor)
    indentation_string = (indent * 2 * " ") if indent > 0 else ""
    print_blue(output=f"{indentation_string}" + 60*'-')
    print_tensor_shape(tensor=tensor, name=name, indent=indent)
    print_tensor_type(tensor=tensor, name=name, indent=indent)
    print_tensor_average(tensor=tensor, name=name, indent=indent)
    print_tensor_min_max(tensor=tensor, name=name, indent=indent)
    print_tensor_device(tensor=tensor, name=name, indent=indent)
    print_blue(output=f"{indentation_string}" + 60*'-')


def print_tensor_list(tensor: torch.Tensor, round: int = 4) -> None:
    """
    Converts a tensor to a list and prints it with specified rounding precision.

    Useful for inspecting the numerical values of small tensors or intermediate results
    during the development and testing of architectural components.

    Args:
        tensor (torch.Tensor): The torch.Tensor to print.
        round (int): Number of decimal places for rounding.
    """
    print(np.round(tensor.tolist(), round))


def unsqueeze_tensor(input: torch.Tensor, direction: str, number: int = 1) -> torch.Tensor:
    """
    Expands the dimensions of a tensor by adding singleton dimensions at the specified end.

    This utility is frequently used to prepare tensors for broadcasting or to match
    the expected input shapes of various neural network layers in our architectures.

    Args:
        input (torch.Tensor): The source torch.Tensor.
        direction (str): Where to add dimensions; must be "left" (start) or "right" (end).
        number (int): The number of dimensions to add.

    Returns:
        torch.Tensor: The transformed torch.Tensor with 'number' additional dimensions.
             - If direction is "left": shape (1, ..., input.shape)
             - If direction is "right": shape (input.shape, ..., 1)
    """
    if direction not in ["left", "right"]:
        print("Warning : direction must either be left or right")
        raise NotImplementedError

    for i in range(number):
        input = input.unsqueeze(dim=-1 if direction == "right" else 0)

    return input
