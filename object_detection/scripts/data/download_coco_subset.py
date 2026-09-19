import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.data_retriever import retrieve_dataset

def download():
    dataset_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets'))
    print(f"Downloading to {dataset_directory}")
    dataset = retrieve_dataset(
        dataset_name="coco-2017",
        dataset_directory=dataset_directory,
        maximum_samples=2000,
        splits=["train"],
        label_types=["detections"]
    )
    print(f"Dataset downloaded/loaded successfully. Total samples: {len(dataset)}")

if __name__ == "__main__":
    download()
