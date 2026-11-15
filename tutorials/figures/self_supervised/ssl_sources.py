"""
Sources of the figures of the self-supervised series of the tutorial videos (tutorials/figures/self_supervised/): the
SimCLR run of tutorial_sources.yaml (key self_supervised) with its saved configuration, its test split and its classes,
the reference training images its k-NN evaluation classifies the test images by (select_reference_samples of the
repository), its checkpoints rebuilt on the CPU, the representations of images encoded by the exported InferenceEncoder,
and the two random views of an image drawn by the DataAugmenter of the run exactly as the sample images of its
evaluations draw them (SAMPLE_VIEWS_SEED plus the sample identifier). The figure families of this folder (views,
training curves, evaluation, representations, configuration checks) all read the run through these helpers.
"""

import json
import random

import numpy
import torch

from pathlib import Path
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import read_network_input, read_run_configuration
from tutorial_builder.figure_context import read_thread_count
from vision_training.data.augmentations import AugmentationReport, DataAugmenter
from vision_training.data.classification.classification_dataset import ClassificationDataset
from vision_training.evaluation.self_supervised.representation_metrics import select_reference_samples
from vision_training.models.components.convolution_block import BATCH_NORMALIZATION_MOMENTUM
from vision_training.models.model_loading import load_trained_model
from vision_training.models.self_supervised.inference_encoder import InferenceEncoder
from vision_training.models.self_supervised.self_supervised_model import SelfSupervisedModel
from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.trainers.self_supervised_trainer import SAMPLE_VIEWS_SEED

# Name of the run in tutorial_sources.yaml
RUN_KEY = "self_supervised"

# Name of the training log written next to the run folder by the launch of the tutorial run
LOG_FILE_NAME = "self_supervised_tutorial.log"

# Number of test images encoded per forward pass on the CPU, small enough for a machine shared with trainings
ENCODING_BATCH_SIZE = 32


def list_evaluated_iterations(run_folder: Path) -> List[int]:
    """
    Lists the iterations whose checkpoint the run evaluated on its test split, from the folders of its Evaluation
    folder, in increasing order: the checkpoints the evaluation figures compare.

    Args:
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        List[int]: The evaluated iterations.

    Raises:
        FileNotFoundError: If the run has no evaluated checkpoint.
    """
    iterations = sorted(int(folder.name.split("_")[1]) for folder in (run_folder / "Evaluation").glob("Iteration_*")
                        if folder.is_dir())
    if not iterations:
        raise FileNotFoundError(f"No Evaluation/Iteration_<n> folder in {run_folder}: the run evaluated no checkpoint.")
    return iterations


def describe_evaluation_steps(run_folder: Path) -> Dict[str, int]:
    """
    Gives the steps between the checkpoints the run evaluated on its test split, read from its evaluation folders, for
    the sentence of the first chapter saying how often the run was evaluated: the run of the series evaluated a
    checkpoint every weight_saving_iterations of its first configuration, then, its training extended with a larger
    weight_saving_iterations, at that larger step, its saved configuration keeping only the last one.

    Args:
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, int]: The first step, from iteration 0 ('run.early_evaluation_step'), the last iteration evaluated at
            that step ('run.evaluation_step_change_iteration') and the step after it ('run.late_evaluation_step').

    Raises:
        ValueError: If the evaluated iterations do not follow one step and then a single other one, the two steps the
            sentence of the first chapter describes.
    """
    evaluated_iterations = list_evaluated_iterations(run_folder=run_folder)
    steps = [later - earlier for earlier, later in zip([0] + evaluated_iterations[:-1], evaluated_iterations)]

    # The first step that differs from the first one, every later step being equal to it
    change_indices = [step_index for step_index, step in enumerate(steps) if step != steps[0]]
    if not change_indices or len(set(steps[change_indices[0]:])) != 1:
        raise ValueError(f"The evaluated iterations of {run_folder}, {evaluated_iterations}, do not follow one step "
                         f"then another: reword the run sentence of the first self-supervised chapter, which says they "
                         f"do.")
    return {"run.early_evaluation_step": steps[0],
            "run.evaluation_step_change_iteration": evaluated_iterations[change_indices[0] - 1],
            "run.late_evaluation_step": steps[change_indices[0]]}


def read_test_samples(run_folder: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Reads the test split of the run from its labels.json manifest, in the order of the file, with the class index of
    every image given by the label dictionary saved in the run folder, the classes the k-NN evaluation scores.

    Args:
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Tuple[List[Dict[str, Any]], List[str]]: Every test sample (its 'identifier', 'path', 'class_index' and
            'class_name'), and the class names in class index order.

    Raises:
        ValueError: If a test image has a category missing from the label dictionary of the run.
    """
    run_configuration = read_run_configuration(run_folder=run_folder)
    test_folder = Path(run_configuration["ExperimentConfiguration"]["test_data_folder"])
    manifest = json.loads((test_folder / "labels.json").read_text(encoding="utf-8"))
    label_dictionary = {int(category): int(class_index) for category, class_index in
                        json.loads((run_folder / "label_dictionary.json").read_text(encoding="utf-8")).items()}

    # Class names in class index order, from the categories of the manifest
    category_names = {category["id"]: category["name"] for category in manifest["categories"]}
    class_names = [""] * len(label_dictionary)
    for category, class_index in label_dictionary.items():
        class_names[class_index] = category_names[category]

    # Every test image with its class index
    samples = []
    for sample in manifest["samples"]:
        if sample["category_id"] not in label_dictionary:
            raise ValueError(f"The test image {sample['file_name']} has the category {sample['category_id']}, missing "
                             f"from the label dictionary of {run_folder}.")
        class_index = label_dictionary[sample["category_id"]]
        samples.append({"identifier": int(sample["id"]), "path": test_folder / "data" / sample["file_name"],
                        "class_index": class_index, "class_name": class_names[class_index]})
    return samples, class_names


def read_reference_samples(run_folder: Path, class_names: List[str]) -> List[Dict[str, Any]]:
    """
    Chooses the reference images of the k-NN evaluation of the run as its trainer does (build_reference_dataloader):
    the training split read by the ClassificationDataset of the loaders, every image kept, then at most
    'evaluation.number_reference_images_per_class' labelled images of every class picked by select_reference_samples
    of the repository, the labelled training images every labelled test image is compared with.

    Args:
        run_folder (Path): The folder of the self-supervised run.
        class_names (List[str]): The class names in class index order, from read_test_samples.

    Returns:
        List[Dict[str, Any]]: Every reference image (its 'identifier', 'path', 'class_index' and 'class_name'), by
            class index then file name hash, the order of the reference loader of the trainer.

    Raises:
        ValueError: If the training split holds no labelled image, the k-NN of the run having then no reference image.
    """
    run_configuration = read_run_configuration(run_folder=run_folder)
    train_folder = Path(run_configuration["ExperimentConfiguration"]["train_data_folder"])
    number_images_per_class = int(run_configuration["ExperimentConfiguration"]["evaluation"][
        "number_reference_images_per_class"])
    label_dictionary = {int(category): int(class_index) for category, class_index in
                        json.loads((run_folder / "label_dictionary.json").read_text(encoding="utf-8")).items()}
    image_settings = run_configuration["Data"]["image_settings"]

    # The training split as the loaders of the run read it, every image kept, unshuffled
    training_dataset = ClassificationDataset(data_directory=str(train_folder / "data"),
                                             labels_file=str(train_folder / "labels.json"),
                                             image_size=int(image_settings["size"]),
                                             keep_ratio=bool(image_settings["keep_ratio"]),
                                             label_dictionary=label_dictionary, output_mode="multi_class",
                                             shuffle=False, keep_unlabelled_samples=True)
    reference_records = select_reference_samples(samples=training_dataset.samples,
                                                 number_images_per_class=number_images_per_class)
    if not reference_records:
        raise ValueError(f"The training split {train_folder} holds no labelled image: the k-NN evaluation of "
                         f"{run_folder} has no reference image.")
    return [{"identifier": int(record["identifier"]), "path": train_folder / "data" / record["file_name"],
             "class_index": record["class_indices"][0], "class_name": class_names[record["class_indices"][0]]}
            for record in reference_records]


def read_image_settings(run_folder: Path) -> Tuple[int, bool]:
    """
    Reads the input size and the aspect ratio setting of the run, with which its loaders read every image.

    Args:
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Tuple[int, bool]: 'Data.image_settings.size' and 'Data.image_settings.keep_ratio'.
    """
    image_settings = read_run_configuration(run_folder=run_folder)["Data"]["image_settings"]
    return int(image_settings["size"]), bool(image_settings["keep_ratio"])


def load_self_supervised_model(run_folder: Path, iteration: int) -> SelfSupervisedModel:
    """
    Rebuilds the trained self-supervised model of a checkpoint of the run on the CPU, in evaluation mode, with the
    loader of the evaluation and export scripts, its trained weights (the run has no moving average). The threads of
    PyTorch are limited to those of the build.

    Args:
        run_folder (Path): The folder of the self-supervised run.
        iteration (int): The iteration of the checkpoint.

    Returns:
        SelfSupervisedModel: The model in evaluation mode.
    """
    torch.set_num_threads(read_thread_count())
    checkpoint_path = get_checkpoint_path(experiment_folder=run_folder, iteration=iteration)
    return load_trained_model(configuration_path=run_folder / "training_configuration.yaml",
                              checkpoint_path=checkpoint_path, device=torch.device("cpu"), use_averaged_weights=False)


def read_images(samples: List[Dict[str, Any]], image_size: int, keep_ratio: bool) -> torch.Tensor:
    """
    Reads test or reference training images the way the loaders of the run do, into one [N, 3, H, W] float BGR batch
    in [0, 255].

    Args:
        samples (List[Dict[str, Any]]): The samples, from read_test_samples or read_reference_samples.
        image_size (int): The input size of the run.
        keep_ratio (bool): Whether the run keeps the aspect ratio.

    Returns:
        torch.Tensor: The images.
    """
    return torch.cat(tensors=[read_network_input(image_path=sample["path"], image_size=image_size,
                                                 keep_ratio=keep_ratio) for sample in samples], dim=0)


def encode_images(model: SelfSupervisedModel, samples: List[Dict[str, Any]], image_size: int,
                  keep_ratio: bool) -> numpy.ndarray:
    """
    Encodes test or reference training images with the InferenceEncoder of the model, the network exported as
    image_encoder: the L2 normalized representations the k-NN evaluation compares by cosine similarity, read batch by
    batch on the CPU.

    Args:
        model (SelfSupervisedModel): The trained model, in evaluation mode.
        samples (List[Dict[str, Any]]): The samples to encode.
        image_size (int): The input size of the run.
        keep_ratio (bool): Whether the run keeps the aspect ratio.

    Returns:
        numpy.ndarray: The [N, F] float32 normalized representations, in the order of the samples.
    """
    encoder = InferenceEncoder(model=model).eval()
    representations = []
    with torch.no_grad():
        for batch_start in range(0, len(samples), ENCODING_BATCH_SIZE):
            images = read_images(samples=samples[batch_start:batch_start + ENCODING_BATCH_SIZE],
                                 image_size=image_size, keep_ratio=keep_ratio)
            representations.append(encoder(images=images).numpy())
    return numpy.concatenate(representations, axis=0).astype(numpy.float32)


def build_view_augmenter(run_folder: Path) -> DataAugmenter:
    """
    Builds the DataAugmenter of the run with the self-supervised strengths, from its saved augmentation section, the
    augmenter that drew its views and the views of its sample images.

    Args:
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        DataAugmenter: The augmenter of the views.
    """
    augmentation_configuration = read_run_configuration(run_folder=run_folder)["Data"]["data_augmentation"]
    return DataAugmenter(augmentation_configuration=augmentation_configuration, self_supervised=True)


def draw_sample_views(augmenter: DataAugmenter, image: torch.Tensor, sample_identifier: int) -> Tuple[
                      torch.Tensor, torch.Tensor, AugmentationReport, AugmentationReport]:
    """
    Draws the two views of a test image as SelfSupervisedTrainer.build_sample_image draws them for the sample images
    of every evaluation: Python's random module and PyTorch seeded with SAMPLE_VIEWS_SEED plus the sample identifier,
    the random state of the caller restored afterwards. The parameters drawn with Python's random module match those
    of the run; the noise and plasma maps drawn by PyTorch differ from the GPU run, its generator being another one.

    Args:
        augmenter (DataAugmenter): The augmenter of the views of the run.
        image (torch.Tensor): The [1, 3, H, W] float BGR image in [0, 255].
        sample_identifier (int): The identifier of the image in the test manifest.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, AugmentationReport, AugmentationReport]: The first and second [1, 3, H, W]
            views, and the report of the augmentations drawn for each.
    """
    views_seed = SAMPLE_VIEWS_SEED + sample_identifier
    caller_random_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed=views_seed)
            random.seed(views_seed)
            first_view, second_view, first_reports, second_reports = augmenter.create_views(images=image)
    finally:
        random.setstate(caller_random_state)
    return first_view, second_view, first_reports[0], second_reports[0]


def find_sample(samples: List[Dict[str, Any]], file_name: str) -> Dict[str, Any]:
    """
    Finds a test sample by its file name, such as the star test_000000.png of the sample images.

    Args:
        samples (List[Dict[str, Any]]): The test samples.
        file_name (str): The file name of the image.

    Returns:
        Dict[str, Any]: The sample.

    Raises:
        KeyError: If no test sample has this file name.
    """
    for sample in samples:
        if sample["path"].name == file_name:
            return sample
    raise KeyError(f"No test sample {file_name} in the test split of the self-supervised run.")


def describe_run_settings(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the settings of the run the narration quotes, read from its saved configuration: the batch size, the input
    size, the iterations, the checkpoint and evaluation steps, the learning rate, the temperature, the number of
    neighbours of the k-NN and of its reference training images per class, and the augmentation types activated for
    the views; with the steps between its evaluated checkpoints read from its evaluation folders
    (describe_evaluation_steps), its configuration keeping only the last one.

    Args:
        run_folder (Path): The folder of the self-supervised run.

    Returns:
        Dict[str, Any]: The facts, named run.*.
    """
    configuration = read_run_configuration(run_folder=run_folder)
    experiment_configuration = configuration["ExperimentConfiguration"]
    augmentation_types = configuration["Data"]["data_augmentation"]["types"]
    return {**describe_evaluation_steps(run_folder=run_folder),
            "run.batch_size": int(experiment_configuration["batch_size"]),
            "run.image_size": int(configuration["Data"]["image_settings"]["size"]),
            "run.number_iterations": int(float(experiment_configuration["number_iterations"])),
            "run.weight_saving_iterations": int(experiment_configuration["weight_saving_iterations"]),
            "run.information_dump": int(experiment_configuration["information_dump"]),
            "run.validation_iteration_ratio": int(experiment_configuration["validation_iteration_ratio"]),
            "run.learning_rate": float(experiment_configuration["learning_rate"]),
            "run.temperature": float(configuration["ProjectionHead"]["simclr"]["temperature"]),
            "run.number_neighbours": int(experiment_configuration["evaluation"]["number_neighbours"]),
            "run.number_reference_images_per_class": int(
                experiment_configuration["evaluation"]["number_reference_images_per_class"]),
            "run.number_sample_images": int(experiment_configuration["evaluation"]["number_sample_images"]),
            "run.activated_type_count": sum(bool(settings["activated"]) for settings in augmentation_types.values()),
            "run.type_count": len(augmentation_types),
            "run.backbone_momentum": BATCH_NORMALIZATION_MOMENTUM}
