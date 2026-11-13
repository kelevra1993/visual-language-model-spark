"""
Shared base of the trainers reading COCO-format splits whose test images are scored and drawn image by image, the
object detection trainer (and the instance segmentation trainer extending it) and the keypoint detection trainer: the
class names and colours and the file name of every test image read from the test labels.json, the description of a
split by its images and annotations, and the names and classes of the fixed test images drawn at every evaluation.
"""

from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

from vision_training.data.label_mapping import get_class_names
from vision_training.data.manifest_cache import read_manifest
from vision_training.data.split_description import count_split_images_and_annotations
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.utilities.visualization import build_class_colours


class CocoSplitTrainer(BaseTrainer):
    """
    Base of the trainers of the COCO-format tasks evaluated image by image: the subclasses call read_test_manifest
    when building their loaders, and inherit the class names, the split description and the test sample names.
    """

    def read_test_manifest(self) -> None:
        """
        Reads the test labels.json before the loaders are built: names the classes after the categories of the test
        split for the per-class report, with a colour per class for the sample images, and keeps the file name of
        every test image, which names the sample images.
        """
        test_labels_path = Path(self.experiment_configuration["test_data_folder"]) / "labels.json"
        test_labels = read_manifest(labels_path=test_labels_path)
        self.class_names = get_class_names(categories=test_labels["categories"], label_dictionary=self.label_dictionary)
        self.class_colours = build_class_colours(number_classes=len(self.class_names))
        self.test_image_file_names = {image["id"]: image["file_name"] for image in test_labels["images"]}

    def list_class_names(self) -> Optional[List[str]]:
        """
        Gives the class names taken from the test manifest by read_test_manifest, the background first, for the record
        of the data of the run and the transfer of the heads (detection, mask or keypoint heads) to a run adding
        classes after them.

        Returns:
            Optional[List[str]]: The class names, in class index order.
        """
        return [self.class_names[class_index] for class_index in range(len(self.class_names))]

    def describe_split(self, split_data_folder: Path) -> Tuple[int, str]:
        """
        Counts the images and annotations of a COCO split from its labels.json.

        Args:
            split_data_folder (Path): The root folder of the split.

        Returns:
            Tuple[int, str]: The number of images and their description with the number of annotations.
        """
        number_images, number_annotations = count_split_images_and_annotations(split_data_folder=split_data_folder)
        return number_images, f"{number_images} images, {number_annotations} annotations"

    def get_test_sample_classes(self) -> Dict[int, List[int]]:
        """
        Lists the test images without their classes, so that the fixed test images drawn at every evaluation are
        evenly spread over the test split.

        Returns:
            Dict[int, List[int]]: An empty list for every COCO identifier of the test images.
        """
        return {image_identifier: [] for image_identifier in self.test_image_file_names}

    def get_test_sample_name(self, sample_identifier: int) -> str:
        """
        Gives the short name of a test image from the test labels.json: its file name without extension, its folders
        joined with underscores.

        Args:
            sample_identifier (int): The COCO identifier of the image, as given in the 'sample_identifiers' of a batch.

        Returns:
            str: The name of the image, e.g. 'image_000012' for 'image_000012.png'.
        """
        return str(PurePosixPath(self.test_image_file_names[sample_identifier]).with_suffix("")).replace("/", "_")
