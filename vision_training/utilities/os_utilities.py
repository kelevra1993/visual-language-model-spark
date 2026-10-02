import json
from json import JSONDecodeError

import torch
import numpy as np

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import inspect


# No printing of scientific notations
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000, threshold=np.inf)
torch.set_printoptions(linewidth=1000)


def read_json(path: str) -> dict[str, Any] | list[Any]:
    """
    Reads and parses a JSON file.

    This function is used throughout the project to load records, manifests,
    and other metadata stored in JSON format.

    Args:
        path (str): The file path to the .json file.

    Returns:
        Dict[str, Any] | List[Any]: The parsed JSON data.
    """
    json_data = None
    try:
        with open(path, "r") as file:
            json_data = json.loads(file.read())
        return json_data
    except JSONDecodeError:
        return json_data


# ANSI escape codes of the console colours available to print_color
COLOR_CODES = {"blue": "\033[94m", "green": "\033[32m", "yellow": "\033[93m", "red": "\033[91m"}


def print_color(color: str, output: str, add_separators: bool = False, indent: int = 0) -> None:
    """
    Prints a string to the console in a bold colour, the single console output helper of the project.

    The colour carries the meaning of the message across the pipeline: blue for information and progress, green
    for successful operations (saved weights, completed steps), yellow for warnings that need attention without
    stopping the run, and red for errors and critical failures.

    Args:
        color (str): The colour of the message, one of "blue", "green", "yellow" or "red".
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal separators for better visibility.
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.

    Raises:
        ValueError: If the colour is not one of the supported colours.
    """
    if color not in COLOR_CODES:
        raise ValueError(f"Unsupported color {color!r}, expected one of {list(COLOR_CODES)}.")
    color_code = COLOR_CODES[color]

    # Prefix every line with the indentation and a bullet when an indentation level is requested
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    indented_output = "\n".join(f"{indentation_string}{line}" for line in output.split("\n"))

    # Print the message in bold colour, optionally framed by separators as long as its longest line
    if add_separators:
        length = max(len(line) for line in output.split("\n")) + 1
        print(indentation_string + color_code + "\033[1m" + str(length * "-") + "\033[0m")
        print(color_code + "\033[1m" + indented_output + "\033[0m")
        print(indentation_string + color_code + "\033[1m" + str(length * "-") + "\033[0m")
    else:
        print(color_code + "\033[1m" + indented_output + "\033[0m")


def print_dictionary(dictionary: Dict[str, Any], indent: int = 4) -> None:
    """
    Prints a dictionary to the console in a formatted JSON-like style.

    This utility is used to display configuration parameters, experiment
    summaries, or manifest data in a readable format during execution.

    Args:
        dictionary (Dict[str, Any]): The dictionary to be printed.
        indent (int): The number of spaces to use for indentation.
    """
    print(json.dumps(dictionary, indent=indent))


def get_random_color() -> Tuple[int, int, int]:
    """
    Generates a random RGB color tuple.
    
    This utility is useful for randomly assigning colors to different bounding boxes
    or classes during visualization, ensuring distinct visual representations.
    
    Args:
        None
        
    Returns:
        Tuple[int, int, int]: A tuple containing random (B, G, R) values from 0 to 255.
    """
    return (int(np.random.randint(low=0, high=256)),
            int(np.random.randint(low=0, high=256)),
            int(np.random.randint(low=0, high=256)))


def get_variable_name(target_variable: Any, namespace: Optional[Dict[str, Any]] = None) -> str:
    """
    Retrieves the string name of a variable by checking for its exact memory identity in a given namespace.
    If no namespace is provided, it automatically inspects the caller's frame (two levels up) to find 
    the original variable name, removing the need for manual typing.
    
    Args:
        target_variable (Any): The variable whose name we want to find.
        namespace (Optional[Dict[str, Any]]): The dictionary mapping variable names to values. 
                                              Defaults to None, triggering automatic frame inspection.
        
    Returns:
        str: The name of the variable as a string, or 'Unknown_Variable' if not found.
    """
    if namespace is None:
        try:
            # We go back 2 frames: 1 frame for this function, 1 frame for the caller (e.g., print_tensor_shape)
            # This allows us to access the locals dictionary of the script/function that actually passed the variable.
            frame = inspect.currentframe().f_back.f_back
            namespace = frame.f_locals
        except Exception:
            namespace = {}

    for name, value in namespace.items():
        # Using 'is' checks for exact memory identity, not just equal value
        if value is target_variable:
            return name

    return "Unknown_Variable"


def get_image_paths(folder: Path) -> List[Path]:
    """
    Lists the images of a folder, e.g. the 'data' folder of a dataset split, to feed them to the deployed detector
    in the inference client, or a class folder whose images are listed in a classification manifest.

    Args:
        folder (Path): The folder holding the images, which is not searched recursively.

    Returns:
        List[Path]: The sorted image paths.

    Raises:
        FileNotFoundError: If the folder holds no image, e.g. a split folder given instead of its 'data' subfolder.
    """
    # Image file extensions listed by get_image_paths
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    image_paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    # Fail loudly instead of running on nothing
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image found in : file://{folder.absolute()}")
    return image_paths


def link_folder(link_path: Path, target_folder: Path) -> None:
    """
    Points 'link_path' to 'target_folder' with a symbolic link, replacing a previous link, so that the manifests built
    from an existing dataset (class folders, pair lists) read its images where they are instead of copying them into
    the 'data' folder of every split.

    Args:
        link_path (Path): The path of the link, e.g. the 'data' folder of a split.
        target_folder (Path): The folder the link points to.

    Raises:
        FileExistsError: If 'link_path' exists and is not a symbolic link, so that a real folder is never replaced.
    """
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"The data folder of the split already exists and is not a link to the dataset "
                              f"images : file://{link_path.absolute()}")
    link_path.symlink_to(target=target_folder.absolute(), target_is_directory=True)
