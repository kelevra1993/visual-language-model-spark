"""
View figures of the self-supervised series of the tutorial videos (chapter 1, the two views, and chapter 2, the InfoNCE
matrix), made by tutorials/figures/self_supervised/figures.py: four sample images of the evaluations of the SimCLR run
(a star, a circle, a triangle and a square of its test split), each written with the two views the DataAugmenter of the
run draws for it from the seed of its sample image, and the augmentations drawn for both views of the star as the input
viewer lists them (describe_report), the listings of the views scene. The views of the star are compared with the
sample image the run saved (views.star_saved_difference), 0 only while the augmentation strengths of the code are those
the run was trained with.
"""

import cv2
import numpy
import torch

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import round_fact
from tutorials.figures.self_supervised.ssl_sources import (build_view_augmenter, draw_sample_views, find_sample,
                                                          list_evaluated_iterations, read_image_settings,
                                                          read_images, read_test_samples)

# Sample images of the evaluations whose views the figures show, in the order of the InfoNCE matrix: the star whose
# views the first chapter details, then a circle, a triangle and a square
VIEW_SAMPLE_NAMES = ["test_000000.png", "test_000006.png", "test_000012.png", "test_000008.png"]

# Height of the caption bar above the image and views of a sample image of the evaluation (CAPTION_BAR_HEIGHT of
# vision_training/utilities/visualization.py), cut before comparing the drawn views with the saved ones
SAMPLE_CAPTION_HEIGHT = 24


def write_image(image: torch.Tensor, output_path: Path) -> None:
    """
    Writes a [1, 3, H, W] float BGR image in [0, 255] as a PNG file, a thumbnail of the views scenes.

    Args:
        image (torch.Tensor): The image or view.
        output_path (Path): The PNG file, its folder created if missing.

    Raises:
        OSError: If the file cannot be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = image[0].permute(1, 2, 0).numpy().clip(min=0, max=255).astype(numpy.uint8)
    if not cv2.imwrite(filename=str(output_path), img=array):
        raise OSError(f"The figure {output_path} could not be written.")


def compare_with_saved_sample(images: List[torch.Tensor], saved_path: Path) -> float:
    """
    Measures how far the image and views drawn on the CPU are from the sample image the run saved at an evaluation:
    the mean absolute difference of their pixels, 0 when the views are those of the run.

    Args:
        images (List[torch.Tensor]): The image, the first view and the second view, each [1, 3, H, W].
        saved_path (Path): The sample image of the run, its caption bar above the image and its views side by side.

    Returns:
        float: The mean absolute difference in intensity levels.

    Raises:
        FileNotFoundError: If the sample image is missing.
    """
    saved_image = cv2.imread(filename=str(saved_path))
    if saved_image is None:
        raise FileNotFoundError(f"No sample image {saved_path} in the run.")
    drawn_image = numpy.concatenate([image[0].permute(1, 2, 0).numpy().clip(min=0, max=255).astype(numpy.uint8)
                                     for image in images], axis=1)
    saved_panels = saved_image[SAMPLE_CAPTION_HEIGHT:, :drawn_image.shape[1]]
    return float(numpy.abs(drawn_image.astype(numpy.float64) - saved_panels.astype(numpy.float64)).mean())


def make_view_figures(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Writes the image and the two views of every sample image of VIEW_SAMPLE_NAMES into views/, the listings of the
    augmentations drawn for both views of the star into listings/, and gives the facts the views scenes quote.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The facts, named views.*.
    """
    samples, _ = read_test_samples(run_folder=run_folder)
    image_size, keep_ratio = read_image_settings(run_folder=run_folder)
    augmenter = build_view_augmenter(run_folder=run_folder)
    last_iteration = list_evaluated_iterations(run_folder=run_folder)[-1]
    facts = {"views.image_size": image_size}

    # Every sample image with its two views, as the evaluation draws them
    for sample_index, file_name in enumerate(VIEW_SAMPLE_NAMES):
        sample = find_sample(samples=samples, file_name=file_name)
        image = read_images(samples=[sample], image_size=image_size, keep_ratio=keep_ratio)
        first_view, second_view, first_report, second_report = draw_sample_views(
            augmenter=augmenter, image=image, sample_identifier=sample["identifier"])
        stem = Path(file_name).stem
        for suffix, picture in [("image", image), ("view_1", first_view), ("view_2", second_view)]:
            write_image(image=picture, output_path=output_folder / "views" / f"{stem}_{suffix}.png")
        facts[f"views.sample_{sample_index}.name"] = stem
        facts[f"views.sample_{sample_index}.class"] = sample["class_name"].replace("_", " ")

        # The star: its augmentation lists and its match with the sample image saved by the run
        if sample_index == 0:
            for view_name, report in [("view_1", first_report), ("view_2", second_report)]:
                listing_path = output_folder / "listings" / f"{stem}_{view_name}_augmentations.txt"
                listing_path.parent.mkdir(parents=True, exist_ok=True)
                listing_path.write_text(data="\n".join(augmenter.describe_report(report=report)) + "\n",
                                        encoding="utf-8")
            facts["views.star_saved_difference"] = round_fact(value=compare_with_saved_sample(
                images=[image, first_view, second_view],
                saved_path=run_folder / "Evaluation" / f"Iteration_{last_iteration}" / "Samples" / file_name))
    return facts
