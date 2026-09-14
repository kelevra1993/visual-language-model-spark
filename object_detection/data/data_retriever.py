import multiprocessing
import fiftyone as fo
import fiftyone.zoo as foz
from typing import Optional


def retrieve_open_images_dataset(dataset_directory: str, maximum_samples: Optional[int]) -> fo.Dataset:
    """
    Downloads and constructs the Open Images V7 dataset using the FiftyOne Zoo API.

    This function fetches the training, validation, and test splits across all available 
    classes. It strictly filters the label types to only retrieve bounding box detections. 
    Multiprocessing is leveraged to maximize the download speed.

    Args:
        dataset_directory (str): The absolute path where the raw dataset files will be stored.
        maximum_samples (Optional[int]): The maximum number of images to download.
        Pass `None` to download the entire dataset.

    Returns:
        fo.Dataset: The instantiated FiftyOne dataset object containing the downloaded data.
    """
    # Bypass FiftyOne's internal dataset_dir kwarg bug by globally overriding the target directory
    fo.config.dataset_zoo_dir = dataset_directory

    # Dynamically allocate 80% of CPU cores for multithreaded downloading to keep the system responsive
    number_of_workers = int(0.8 * multiprocessing.cpu_count())

    dataset = foz.load_zoo_dataset(name_or_url="open-images-v7",
                                   splits=["train", "validation", "test"], label_types=["detections"],
                                   max_samples=maximum_samples, num_workers=number_of_workers, persistent=True)

    return dataset


if __name__ == "__main__":
    retrieve_open_images_dataset(dataset_directory="/home/robert_kelevra/Data/object_detection_data",
                                 maximum_samples=100)
