"""
General helpers used across the framework, the scripts and the tests: JSON reading, coloured console output,
dictionary printing, random colours, image path listing, folder linking, the CUDA allocator setting of the training
launch and the release of the freed heap memory once a dataset has packed its manifest and at the end of the launch.
"""

import ctypes
import json
import os
import re
import sys
from json import JSONDecodeError

import torch
import numpy as np

from pathlib import Path
from typing import Dict, Any, List, Optional
import inspect


# No printing of scientific notations
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=1000, threshold=np.inf)
torch.set_printoptions(linewidth=1000)


def read_json(path: str) -> dict[str, Any] | list[Any]:
    """
    Reads and parses a JSON file: the data records of the runs (data.json, compared when a run resumes and read by the
    transfers between runs), the dataset manifests read by the data scripts, and the pair lists of the comparison
    manifests. A missing file raises FileNotFoundError, and a malformed one (a data.json truncated by an interrupted
    write) raises with the file and the position of the error instead of returning None to callers that index it.

    Args:
        path (str): The file path to the .json file.

    Returns:
        Dict[str, Any] | List[Any]: The parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON, naming the file and the line and column of the error.
    """
    with open(path, "r") as file:
        file_content = file.read()

    # A parse error names the file, the position alone not telling which of the run files is broken
    try:
        return json.loads(file_content)
    except JSONDecodeError as error:
        raise ValueError(f"The JSON file {format_path_link(path=path)} cannot be parsed: {error}.") from error


# ANSI escape codes of the console colours available to color_text and print_color
COLOR_CODES = {"blue": "\033[94m", "green": "\033[32m", "yellow": "\033[93m", "red": "\033[91m"}


def color_text(color: str, text: str) -> str:
    """
    Wraps a string in the bold colour codes of the console, the building block of print_color and of the lines mixing
    several colours, such as the metric lines of the training console summary whose changes are green or red.

    Args:
        color (str): The colour of the text, one of "blue", "green", "yellow" or "red".
        text (str): The string to colour.

    Returns:
        str: The text between the colour and bold codes and the reset code.

    Raises:
        ValueError: If the colour is not one of the supported colours.
    """
    if color not in COLOR_CODES:
        raise ValueError(f"Unsupported color {color!r}, expected one of {list(COLOR_CODES)}.")

    return COLOR_CODES[color] + "\033[1m" + text + "\033[0m"


# OSC 8 escape sequences of a terminal hyperlink: the opener is followed by the link target, both ended by the string
# terminator, and an opener with an empty target closes the link
HYPERLINK_OPENER = "\033]8;;"
HYPERLINK_TERMINATOR = "\033\\"

# Select Graphic Rendition codes underlining the label of a hyperlink so that the clickable text stands out, the
# closing code ending the underline only, so that the colour and bold of the surrounding message go on after the link
UNDERLINE_OPENER = "\033[4m"
UNDERLINE_TERMINATOR = "\033[24m"

# Pattern of the escape sequences taking no width in the console: the opening and closing sequences of the hyperlinks
# and the colour and style codes, such as the underline of their labels
INVISIBLE_SEQUENCE_PATTERN = re.compile(pattern=r"\033\]8;;[^\033]*\033\\|\033\[[0-9;]*m")


def format_path_link(path: Path | str, label: Optional[str] = None) -> str:
    """
    Turns a file or folder path into an OSC 8 terminal hyperlink with an underlined label, so that the console output
    of the pipeline (the experiment information printed at launch, the saved files, the errors) shows a short text,
    recognisable as clickable, instead of the whole file:// path. Terminals without hyperlink support, such as the
    PyCharm Run window without "Emulate terminal in output console", show the underlined label alone.

    Args:
        path (Path | str): The file or folder opened by the link, made absolute since a file URI needs an absolute
            path.
        label (Optional[str]): The text shown in the console. Defaults to the name of the file or folder.

    Returns:
        str: The underlined label between the sequence opening the hyperlink to the file URI of the path and the one
            closing it.
    """
    absolute_path = Path(path).absolute()

    # The name of the file or folder by default, the whole path for the root folder which has no name
    link_label = label if label is not None else (absolute_path.name or str(absolute_path))
    return (f"{HYPERLINK_OPENER}{absolute_path.as_uri()}{HYPERLINK_TERMINATOR}"
            f"{UNDERLINE_OPENER}{link_label}{UNDERLINE_TERMINATOR}{HYPERLINK_OPENER}{HYPERLINK_TERMINATOR}")


def print_color(color: str, output: str, add_separators: bool = False, indent: int = 0) -> None:
    """
    Prints a string to the console in a bold colour, the single colour console output helper of the project (lines
    mixing several colours are built with color_text).

    The colour carries the meaning of the message across the pipeline: blue for information and progress, green
    for successful operations (saved weights, completed steps), yellow for warnings that need attention without
    stopping the run, and red for errors and critical failures.

    Args:
        color (str): The colour of the message, one of "blue", "green", "yellow" or "red".
        output (str): The string to be printed.
        add_separators (bool): If True, wraps the output with horizontal separators for better visibility, as long
            as its longest line shows in the console (the underlined hyperlinks of format_path_link counting as their
            labels).
        indent (int, optional): The number of indentation levels (2 spaces each). Defaults to 0.

    Raises:
        ValueError: If the colour is not one of the supported colours.
    """
    # Prefix every line with the indentation and a bullet when an indentation level is requested
    indentation_string = (indent * 2 * " ") + "- " if indent > 0 else ""
    indented_output = "\n".join(f"{indentation_string}{line}" for line in output.split("\n"))

    # Print the message in bold colour, optionally framed by separators as long as its longest line
    if add_separators:
        visible_width = max(len(INVISIBLE_SEQUENCE_PATTERN.sub(repl="", string=line)) for line in output.split("\n"))
        separator = color_text(color=color, text=str((visible_width + 1) * "-"))
        print(indentation_string + separator)
        print(color_text(color=color, text=indented_output))
        print(indentation_string + separator)
    else:
        print(color_text(color=color, text=indented_output))


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


def get_image_paths(folder: Path, search_subfolders: bool = False) -> List[Path]:
    """
    Lists the images of a folder, e.g. the 'data' folder of a dataset split, to feed them to the deployed detector
    in the inference client, a class folder whose images are listed in a classification manifest, or a folder of plain
    images embedded by scripts/embedding/cluster_image_embeddings.py, searched with its subfolders.

    Args:
        folder (Path): The folder holding the images.
        search_subfolders (bool): Whether to also list the images of every subfolder, the links to folders being
            followed. Defaults to False.

    Returns:
        List[Path]: The sorted image paths.

    Raises:
        FileNotFoundError: If the folder holds no image, e.g. a split folder given instead of its 'data' subfolder.
    """
    # Image file extensions listed by get_image_paths
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    # The files of the folder, or of the folder and every subfolder, the linked folders included
    if search_subfolders:
        candidate_paths = [Path(current_folder) / file_name
                           for current_folder, _, file_names in os.walk(top=folder, followlinks=True)
                           for file_name in file_names]
    else:
        candidate_paths = list(folder.iterdir())
    image_paths = sorted(path for path in candidate_paths if path.suffix.lower() in IMAGE_EXTENSIONS)

    # Fail loudly instead of running on nothing
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image found in : {format_path_link(path=folder)}")
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
                              f"images : {format_path_link(path=link_path)}")
    link_path.symlink_to(target=target_folder.absolute(), target_is_directory=True)


def enable_expandable_cuda_segments() -> None:
    """
    Turns on the expandable segments of the CUDA caching allocator at the start of the training launch
    (vision_training/main.py), before anything is placed on the GPU. The batches of whole sequences are padded to their
    longest sequence, so their shapes change from batch to batch, and the default allocator keeps reserving blocks of
    new sizes during the test evaluations (about 4 Gigabytes at a time) until two parallel runs fill the unified memory
    of the machine; expandable segments grow and reuse the same blocks instead, keeping the GPU memory flat. Setting it
    in the code means no launch can forget the PYTORCH_CUDA_ALLOC_CONF environment variable. Allocator settings given
    by hand in that variable are kept, the expandable segments being added unless the variable sets them itself.

    Raises:
        RuntimeError: If CUDA is already initialised, the allocator then ignoring the setting.
    """
    # The allocator reads its settings at the first CUDA call, so a later setting would be silently ignored
    if torch.cuda.is_initialized():
        raise RuntimeError("The expandable segments of the CUDA allocator must be turned on before CUDA is "
                           "initialised, at the start of the launch")

    # Add the expandable segments to the allocator settings given by hand, if any
    allocator_settings = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in allocator_settings:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(filter(None, [allocator_settings, "expandable_segments:True"]))


def release_freed_heap_memory() -> None:
    """
    Gives back to the system the memory freed in the heap of the C allocator (malloc_trim of the GNU C library), so that
    the main process of a training does not keep resident, nor hand down to the loader workers it forks, the memory of
    the manifests parsed at launch, which freed Python strings and lists otherwise keep (0.26 GB left resident by a
    parse of the manifest of the 50,000 validation images of the instance segmentation benchmark dataset): called by
    CocoDetectionDataset once it has packed its manifest, by the base trainer at the end of its launch, and by
    read_manifests_once of data/manifest_cache.py once the manifests shared during the launch are freed. Nothing is
    done outside Linux, malloc_trim being a function of the GNU C library.
    """
    # Only the GNU C library of Linux trims its heap, other systems returning freed memory their own way
    if not sys.platform.startswith("linux"):
        return None

    # The C library already loaded by the interpreter, opened by its file name, gives back every freed page of its heap
    c_library = ctypes.CDLL(name="libc.so.6")
    c_library.malloc_trim(0)
    return None
