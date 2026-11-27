"""
Downloads raw detection datasets (Open Images V7, COCO 2017) from the FiftyOne Zoo with the requested splits and label
types, the source of the COCO data used by the object detection configurations. Run as a script with the dataset name
and the settings given as arguments (parse_retriever_arguments), its help printed when none is given. FiftyOne comes
with the datasets dependency group, which a plain "uv sync" installs and the Docker image of the trainer leaves out: it
is imported only when a dataset is retrieved, so that the help and the argument checks run without it.
"""

import argparse
import importlib
import multiprocessing
import sys
from typing import Any, Optional, List, Literal
from pathlib import Path

# Datasets of the FiftyOne Zoo the retriever covers
DATASET_NAMES = ["open-images-v7", "coco-2017"]

# Splits both datasets publish
DATASET_SPLITS = ["train", "validation", "test"]

# Folder receiving the downloaded datasets unless the command line gives another one: the 'datasets' folder of the
# repository, the same on every machine
DEFAULT_DATASET_DIRECTORY = Path(__file__).absolute().parents[2] / "datasets"

# Settings taken when the command line of the script (parse_retriever_arguments) does not give them: 1,000 images per
# split, the full datasets needing about 25 GB and a few hours (COCO 2017) or well over 500 GB and several days (Open
# Images V7)
DEFAULT_RETRIEVER_SETTINGS = {"maximum_samples": 1000, "splits": list(DATASET_SPLITS),
                              "label_types": ["detections", "segmentations"]}


def retrieve_dataset(dataset_name: Literal["open-images-v7", "coco-2017"], dataset_directory: str,
                     maximum_samples: Optional[int] = None, splits: Optional[List[str]] = None,
                     label_types: Optional[List[str]] = None) -> Any:
    """
    Downloads and constructs a dataset using the FiftyOne Zoo API.

    This function fetches the specified dataset from the FiftyOne Zoo with the desired splits
    and label types. It bypasses an internal directory bug in FiftyOne by setting the global config
    and leverages multiprocessing to maximize the download speed.

    Args:
        dataset_name (Literal["open-images-v7", "coco-2017"]): The name of the FiftyOne dataset to download.
        dataset_directory (str): The absolute path where the raw dataset files will be stored.
        maximum_samples (Optional[int]): The maximum number of images to download per split. Pass `None` to download
            all.
        splits (Optional[List[str]]): A list of dataset splits to download. Defaults to ["train", "validation", "test"].
        label_types (Optional[List[str]]): A list of label types to retrieve. Defaults to ["detections",
            "segmentations"].

    Returns:
        Any: The instantiated FiftyOne dataset object (fiftyone.Dataset) containing the downloaded data.
    """
    if splits is None:
        splits = list(DEFAULT_RETRIEVER_SETTINGS["splits"])
    if label_types is None:
        label_types = list(DEFAULT_RETRIEVER_SETTINGS["label_types"])

    # FiftyOne loaded only here, an optional heavy package of the datasets group, so that the parser runs without it
    fiftyone = importlib.import_module(name="fiftyone")
    fiftyone_zoo = importlib.import_module(name="fiftyone.zoo")

    # Bypass FiftyOne's internal dataset_dir kwarg bug by globally overriding the target directory
    fiftyone.config.dataset_zoo_dir = dataset_directory

    # Dynamically allocate 80% of CPU cores for multithreaded downloading to keep the system responsive
    number_of_workers = int(0.8 * multiprocessing.cpu_count())

    dataset = fiftyone_zoo.load_zoo_dataset(name_or_url=dataset_name, splits=splits, label_types=label_types,
                                            max_samples=maximum_samples, num_workers=number_of_workers,
                                            persistent=True)

    return dataset


def read_maximum_samples_argument(value: str) -> Optional[int]:
    """
    Converts the --maximum_samples argument of the script into the maximum_samples of retrieve_dataset, 'None'
    downloading every image of the asked splits, so that the command line reaches the full datasets.

    Args:
        value (str): The text given on the command line.

    Returns:
        Optional[int]: None for 'None', otherwise the number of images per split.

    Raises:
        argparse.ArgumentTypeError: If the text is neither 'None' nor a positive integer, argparse then printing the
            usage and the error.
    """
    if value == "None":
        return None
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"expected None or a positive integer, got {value!r}")
    return int(value)


def parse_retriever_arguments(arguments: List[str]) -> argparse.Namespace:
    """
    Reads the settings of a dataset download from the command line of the script, so that a dataset is retrieved
    without editing the file: the dataset name is required, the folder defaults to DEFAULT_DATASET_DIRECTORY (the
    'datasets' folder of the repository) and the other settings to DEFAULT_RETRIEVER_SETTINGS. Launched without any
    argument, the script prints the help of every argument and stops without downloading anything. A '~' left by the
    shell at the start of the folder ('--dataset_directory=~/Data/...') is expanded to the home folder.

    Args:
        arguments (List[str]): The command line arguments, without the script name.

    Returns:
        argparse.Namespace: The arguments of retrieve_dataset, the folder as a Path.

    Raises:
        SystemExit: After printing the help when no argument is given, or the usage and the error when the dataset
            name is missing or a value is not one the retriever accepts.
    """
    # The description and the example keep their line breaks, argparse wrapping the help of every argument
    parser = argparse.ArgumentParser(
        prog="uv run python -m vision_training.data.data_retriever",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Downloads COCO 2017 or Open Images V7 from the FiftyOne Zoo. The full datasets need about 25 GB\n"
                    "and a few hours (COCO 2017), or well over 500 GB and several days (Open Images V7).",
        epilog="Example, every image of the COCO 2017 splits:\n"
               "  uv run python -m vision_training.data.data_retriever --dataset_name coco-2017 --maximum_samples None")
    # The dataset and the folder receiving it
    parser.add_argument("--dataset_name", choices=DATASET_NAMES, required=True,
                        help="Dataset of the FiftyOne Zoo to download (required).")
    parser.add_argument("--dataset_directory", type=Path, default=DEFAULT_DATASET_DIRECTORY,
                        help="Folder receiving the raw dataset files (default: the datasets folder of the "
                             "repository).")
    # The images, splits and labels downloaded
    parser.add_argument("--maximum_samples", type=read_maximum_samples_argument,
                        default=DEFAULT_RETRIEVER_SETTINGS["maximum_samples"],
                        help="Images downloaded per split, None for all of them (default: %(default)s).")
    parser.add_argument("--splits", nargs="+", choices=DATASET_SPLITS,
                        default=list(DEFAULT_RETRIEVER_SETTINGS["splits"]),
                        help=f"Splits downloaded (default: {' '.join(DEFAULT_RETRIEVER_SETTINGS['splits'])}).")
    parser.add_argument("--label_types", nargs="+", default=list(DEFAULT_RETRIEVER_SETTINGS["label_types"]),
                        help=f"Label types downloaded, among those the FiftyOne Zoo gives the dataset (default: "
                             f"{' '.join(DEFAULT_RETRIEVER_SETTINGS['label_types'])}).")

    # Without any argument, the script describes its arguments instead of failing on the missing dataset name
    if not arguments:
        parser.print_help()
        parser.exit(status=0)
    parsed_arguments = parser.parse_args(args=arguments)

    # A '~' the shell leaves after '=' points to the home folder, as in the other scripts of the repository
    parsed_arguments.dataset_directory = parsed_arguments.dataset_directory.expanduser()
    return parsed_arguments


if __name__ == "__main__":
    # Dataset and download settings read from the command line, the help printed when none is given
    retriever_arguments = parse_retriever_arguments(arguments=sys.argv[1:])

    retrieve_dataset(dataset_name=retriever_arguments.dataset_name,
                     dataset_directory=str(retriever_arguments.dataset_directory),
                     maximum_samples=retriever_arguments.maximum_samples, splits=retriever_arguments.splits,
                     label_types=retriever_arguments.label_types)
