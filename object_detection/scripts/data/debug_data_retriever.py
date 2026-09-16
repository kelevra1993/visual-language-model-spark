import fiftyone as fo
from utilities.os_utilities import print_green, print_blue, print_red


def debug_data_retriever(dataset_name: str = "open-images-v7") -> None:
    """
    Launches the FiftyOne application to visually inspect an already downloaded dataset.

    This debugging script assumes the data has already been fetched by `data_retriever.py`.
    It strictly connects to the FiftyOne database to load the existing dataset and opens 
    the visualization dashboard, ensuring no accidental downloads occur.

    Args:
        dataset_name (str): The name of the dataset in the FiftyOne database.
    """
    print_blue(output=f"Looking for dataset '{dataset_name}' in the local FiftyOne database...", add_separators=True)

    available_datasets = fo.list_datasets()

    print("Available datasets: ", available_datasets)

    if not available_datasets:
        print_red(output="No datasets found in the local FiftyOne database!")
        print_red(output="Please execute data_retriever.py first to download and register the dataset.")
        return

    if dataset_name not in available_datasets:
        # Fallback to the most recently created dataset if a naming collision occurred during download
        print_red(output=f"Dataset '{dataset_name}' not found. Falling back to the most recent dataset: '{available_datasets[-1]}'")
        dataset_name = available_datasets[-1]

    dataset = fo.load_dataset(name=dataset_name)

    print_green(output="Dataset loaded successfully! Launching FiftyOne visualizer...", add_separators=True)

    # Launch the interactive web app and wait for the user to close it
    session = fo.launch_app(dataset=dataset)
    session.wait()


if __name__ == "__main__":
    debug_data_retriever(dataset_name="coco-2017-train-validation-test-None")
