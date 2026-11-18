"""
Tensor helpers: selection of the training device, used by the base trainer, the copy of the random states of Python
and PyTorch under which the trainers draw images without changing the draws of the training (the views of the fixed
self-supervised test images, the input samples of every training launch), and printing of the shape of a tensor,
used by the box shape check of box_operations.py.
"""

import random
import torch

from contextlib import contextmanager
from typing import Iterator, Optional

from vision_training.utilities.os_utilities import print_color, get_variable_name


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


@contextmanager
def fork_random_state(device: torch.device) -> Iterator[None]:
    """
    Runs a block drawing random values on copies of the random states of Python and PyTorch (the CPU and the device of
    the model), the original states being restored when the block ends, even on an error: the self-supervised trainer
    draws the views of its fixed test images under it, and the base trainer the input samples written at the start of
    every training launch, so that the augmentations, the shuffling of the loaders and every other draw of the
    training stay those it would have made without the block.

    Args:
        device (torch.device): The device of the model, whose random state is copied along with the CPU one.

    Yields:
        None: The block runs on the copied states.
    """
    python_random_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[] if device.type == "cpu" else [device]):
            yield
    finally:
        random.setstate(python_random_state)


def print_tensor_shape(tensor: torch.Tensor, name: Optional[str] = None, indent: int = 0) -> None:
    """
    Logs the shape of a tensor to the console in a formatted blue string.

    In the global project context, this utility is used during development and debugging, by the box checks of
    box_operations.py, to verify that tensor dimensions align with the expected shapes (feature maps, proposals,
    detections) after the reshapes of the detection pipeline.

    Args:
        tensor (torch.Tensor): The torch.Tensor whose shape will be printed. Shape: (*).
        name (Optional[str], optional): An optional label to identify the tensor in the output.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.
    """
    if not name:
        name = get_variable_name(target_variable=tensor)
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    print_color(color="blue", output=f"{indentation_string}Tensor {name} Is Of Shape : {list(tensor.shape)}")

