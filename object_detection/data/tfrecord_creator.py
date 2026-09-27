import os
import sys
import argparse

from object_detection.data.data_loader import create_tfrecord, create_tfrecord_sharded


def parse_arguments() -> argparse.Namespace:
    """
    Parses the command line arguments required to generate TFRecord files in the background.
    
    This function explicitly defines all the parameters needed to configure the dataset conversion process,
    such as paths, image scaling dimensions, and whether to partition the records into multiple shards.

    Returns:
        argparse.Namespace: The parsed command line arguments containing configuration flags.
    """
    parser = argparse.ArgumentParser(description="Background worker to generate TFRecords sequentially.")
    parser.add_argument("--image_size", type=int, required=True, help="The target spatial dimension for image scaling.")
    parser.add_argument("--keep_ratio", type=lambda x: (str(x).lower() == 'true'), required=True,
                        help="Boolean flag indicating if the aspect ratio should be padded.")
    parser.add_argument("--is_sharded", type=lambda x: (str(x).lower() == 'true'), required=True,
                        help="Boolean flag indicating if the records should be chunked into multiple shards.")
    parser.add_argument("--number_of_shards", type=int, default=10, help="The exact number of shards to split the data into.")
    parser.add_argument("--prefix", type=str, required=True, help="The standardized prefix for the dataset files.")
    parser.add_argument("--train_data_folder", type=str, required=True, help="Root folder for the training split (contains data/ and labels.json).")
    parser.add_argument("--validation_data_folder", type=str, required=True, help="Root folder for the validation split (contains data/ and labels.json).")
    parser.add_argument("--test_data_folder", type=str, required=True, help="Root folder for the test split (contains data/ and labels.json).")

    return parser.parse_args()


def main() -> None:
    """
    The main execution block that orchestrates the sequential creation of TFRecords.
    
    This script is intended to be spawned as a completely detached background process. It sequentially processes
    the training, validation, and testing dataset splits to preemptively encode them into highly optimized 
    TFRecord archives without blocking the primary training loop.
    """
    arguments = parse_arguments()

    # Consolidate the dataset splits into an iterable list to process them systematically
    dataset_splits = [
        ("Train", arguments.train_data_folder),
        ("Validation", arguments.validation_data_folder),
        ("Test", arguments.test_data_folder)
    ]

    # Systematically iterate through each split, directing the workflow to either the sharded or monolithic generator
    for split_name, split_data_folder in dataset_splits:
        # Derive the physical image directory and annotation file from the split folder convention
        data_directory = os.path.join(split_data_folder, "data")
        labels_file = os.path.join(split_data_folder, "labels.json")

        if arguments.is_sharded:
            # Construct the absolute path for the directory that will house the multiple shard fragments
            output_directory = os.path.join(split_data_folder, f"{arguments.prefix}Sharded-Records-{split_name}")
            create_tfrecord_sharded(
                data_directory=data_directory,
                labels_file=labels_file,
                output_directory=output_directory,
                image_size=arguments.image_size,
                keep_ratio=arguments.keep_ratio,
                number_of_shards=arguments.number_of_shards,
                prefix=arguments.prefix
            )
        else:
            # Construct the absolute path for a singular monolithic TFRecord archive file
            output_file = os.path.join(split_data_folder, f"{arguments.prefix}{split_name}.tfrecord")
            create_tfrecord(
                data_directory=data_directory,
                labels_file=labels_file,
                output_tensorflow_record=output_file,
                image_size=arguments.image_size,
                keep_ratio=arguments.keep_ratio
            )


if __name__ == "__main__":
    main()
