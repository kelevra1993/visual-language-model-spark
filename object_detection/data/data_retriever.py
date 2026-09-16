import multiprocessing
import fiftyone as fo
import fiftyone.zoo as foz
from typing import Optional, List, Literal


def retrieve_dataset(dataset_name: Literal["open-images-v7", "coco-2017"],
                     dataset_directory: str,
                     maximum_samples: Optional[int] = None,
                     splits: Optional[List[str]] = None,
                     label_types: Optional[List[str]] = None) -> fo.Dataset:
    """
    Downloads and constructs a dataset using the FiftyOne Zoo API.

    This function fetches the specified dataset from the FiftyOne Zoo with the desired splits 
    and label types. It bypasses an internal directory bug in FiftyOne by setting the global config
    and leverages multiprocessing to maximize the download speed.

    Args:
        dataset_name (Literal["open-images-v7", "coco-2017"]): The name of the FiftyOne dataset to download.
        dataset_directory (str): The absolute path where the raw dataset files will be stored.
        maximum_samples (Optional[int]): The maximum number of images to download. Pass `None` to download all.
        splits (Optional[List[str]]): A list of dataset splits to download. Defaults to ["train", "validation", "test"].
        label_types (Optional[List[str]]): A list of label types to retrieve. Defaults to ["detections", "segmentations"].

    Returns:
        fo.Dataset: The instantiated FiftyOne dataset object containing the downloaded data.
    """
    if splits is None:
        splits = ["train", "validation", "test"]
    if label_types is None:
        label_types = ["detections", "segmentations"]

    # Bypass FiftyOne's internal dataset_dir kwarg bug by globally overriding the target directory
    fo.config.dataset_zoo_dir = dataset_directory

    # Dynamically allocate 80% of CPU cores for multithreaded downloading to keep the system responsive
    number_of_workers = int(0.8 * multiprocessing.cpu_count())

    dataset = foz.load_zoo_dataset(name_or_url=dataset_name,
                                   splits=splits,
                                   label_types=label_types,
                                   max_samples=maximum_samples,
                                   num_workers=number_of_workers,
                                   persistent=True)

    return dataset


if __name__ == "__main__":
    """
    [NOTE] For now, this retriever only officially covers 'open-images-v7' and 'coco-2017'.
    Please be aware of the following resource constraints when omitting `maximum_samples`:
    - COCO 2017: Requires approximately 25 GB of disk space. Depending on your internet connection,
      downloading the entire dataset may take a few hours.
      
    - Open Images V7: Requires well over 500 GB of disk space for the full dataset. Downloading 
      this dataset in its entirety can take several days. Ensure you have sufficient storage and bandwidth.
      
    """
    retrieve_dataset(dataset_name="coco-2017",
                     dataset_directory="/home/robert_kelevra/Data/object_detection_data",
                     maximum_samples=None)
