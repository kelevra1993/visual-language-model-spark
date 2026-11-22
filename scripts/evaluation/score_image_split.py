"""
When run, scores every image of a large classification split with one checkpoint of a trained classifier, for the
mining of rare classes: the probability of every class is written to a predictions file with the columns of the
predictions.csv of scripts/evaluation/evaluate_checkpoint.py ('sample' and 'probability_<class>'), read by the
scripts of a project that ranks the images by the probability of a class. Unlike evaluate_checkpoint.py,
which runs the whole evaluation of the task with one loader worker and both the trained and the averaged weights, it
only runs the forward pass of the chosen weights, the images decoded by many loader workers through the test dataset of
the run (the same reading, resize and padding as its evaluation), the batches written as they come so that a run stopped
by its time budget or interrupted resumes where it stopped, a predictions file written for other classes or whose last
row was cut by an interruption being refused. The images of the split are scored in their order in its labels.json, so
that a split ordered by priority scores its most promising images first.

The forward pass can run in bfloat16 autocast, the precision of the training, after checking on the labelled test split
of the run that the predicted classes and the balanced accuracy match the float32 pass (--compare-precisions).

Run from the repository root:
    uv run python -m scripts.evaluation.score_image_split --experiment-folder <run> --iteration 1500 \
        --use-averaged-weights --test-data-folder <split> --output-path <file.csv> --time-budget-hours 3
"""
import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

from scripts.evaluation.evaluate_checkpoint import apply_test_data_folder, point_to_checkpoint_evaluation
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.core.contracts import SAMPLE_IDENTIFIERS_KEY
from vision_training.core.task_registry import get_trainer_class
from vision_training.data.loaders import build_native_loader
from vision_training.data.manifest_cache import read_manifests_once
from vision_training.models.model_transfer import disable_pretrained_settings
from vision_training.trainers.base_trainer import BaseTrainer
from vision_training.trainers.checkpointing import get_checkpoint_file

# Precisions of the forward pass
AUTOCAST_DTYPES = {"float32": None, "bfloat16": torch.bfloat16}

# Column naming the image in the predictions file, as in the predictions.csv of the evaluation
SAMPLE_COLUMN = "sample"

# Suffix of the writing folder of the trainer when the scored split is the test split of the run
SCORING_TEST_SPLIT_FOLDER_SUFFIX = "_score_image_split_test"


def parse_arguments() -> argparse.Namespace:
    """
    Reads the run, the checkpoint and its weights, the split to score, the output file, the precision, the loader
    settings and the time budget.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    argument_parser = argparse.ArgumentParser(description="Score every image of a split with one checkpoint.")
    argument_parser.add_argument("--experiment-folder", required=True, help="Run folder of the classifier.")
    argument_parser.add_argument("--iteration", type=int, required=True, help="Checkpoint iteration.")
    argument_parser.add_argument("--use-averaged-weights", action="store_true",
                                 help="Score with the moving average of the weights saved with the checkpoint.")
    argument_parser.add_argument("--test-data-folder", default=None,
                                 help="Split to score (labels.json and data/), any placeholder class; required "
                                      "unless --compare-precisions only.")
    argument_parser.add_argument("--output-path", default=None, help="Predictions file, appended when it exists.")
    argument_parser.add_argument("--precision", choices=list(AUTOCAST_DTYPES), default="float32",
                                 help="Precision of the forward pass.")
    argument_parser.add_argument("--batch-size", type=int, default=128, help="Images per forward pass.")
    argument_parser.add_argument("--number-workers", type=int, default=12, help="Loader worker processes.")
    argument_parser.add_argument("--time-budget-hours", type=float, default=None,
                                 help="Stops after this time, the file then resuming at the next launch.")
    argument_parser.add_argument("--compare-precisions", action="store_true",
                                 help="Scores the labelled test split of the run in every precision first and prints "
                                      "the agreement and the speed, then stops unless --test-data-folder is given.")
    return argument_parser.parse_args()


def build_scoring_trainer(experiment_folder: Path, test_data_folder: Optional[str]) -> BaseTrainer:
    """
    Builds the trainer of the run with its test split replaced by the scored split, its writing folder being the
    standalone evaluation folder of that split ('<run>/Checkpoint_Evaluation_split_<name>/') so that nothing of the run
    itself changes, the pretrained components not being loaded since the checkpoint gives the weights.

    Args:
        experiment_folder (Path): Run folder of the classifier.
        test_data_folder (Optional[str]): Split to score, None for the test split of the run.

    Returns:
        BaseTrainer: The trainer, its test dataset reading the scored split.
    """
    configuration_path = experiment_folder / "training_configuration.yaml"
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=configuration_path)
    split_folder_suffix = apply_test_data_folder(experiment_configuration=experiment_configuration,
                                                 test_data_folder=test_data_folder)
    experiment_configuration["tracking"]["activated"] = False

    # The test split of the run itself (the precision comparison) is read from a folder of its own, so that the
    # 'Checkpoint_Evaluation' folder of its standalone evaluations is never touched
    point_to_checkpoint_evaluation(experiment_configuration=experiment_configuration,
                                   experiment_folder=experiment_folder,
                                   folder_suffix=split_folder_suffix or SCORING_TEST_SPLIT_FOLDER_SUFFIX)
    disable_pretrained_settings(model_configuration=model_configuration)
    trainer_class = get_trainer_class(task=experiment_configuration["task"])
    with read_manifests_once():
        return trainer_class(experiment_configuration=experiment_configuration,
                             model_configuration=model_configuration, configuration_path=configuration_path)


def load_checkpoint_weights(trainer: BaseTrainer, experiment_folder: Path, iteration: int,
                            use_averaged_weights: bool) -> torch.nn.Module:
    """
    Loads the weights of a checkpoint into the model of the trainer, and gives the scored model: the trained weights,
    or their moving average saved with the checkpoint.

    Args:
        trainer (BaseTrainer): The trainer of the run.
        experiment_folder (Path): Run folder holding 'Weights'.
        iteration (int): Checkpoint iteration.
        use_averaged_weights (bool): Whether to score with the averaged weights.

    Returns:
        torch.nn.Module: The scored model, in evaluation mode.

    Raises:
        ValueError: When the averaged weights are asked for a run without moving average.
    """
    checkpoint_path = get_checkpoint_file(weights_directory=experiment_folder / "Weights", iteration=iteration)
    checkpoint = torch.load(f=checkpoint_path, map_location=trainer.device, weights_only=True)
    trainer.model.load_state_dict(state_dict=checkpoint["model_state"])
    trainer.load_exponential_moving_average(checkpoint=checkpoint, iteration=iteration)
    if use_averaged_weights and trainer.exponential_moving_average is None:
        raise ValueError(f"The run {experiment_folder.name} keeps no moving average of its weights.")
    model = trainer.exponential_moving_average.averaged_model if use_averaged_weights else trainer.model
    return model.eval()


def score_dataset(trainer: BaseTrainer, model: torch.nn.Module, dataset: Any, batch_size: int, number_workers: int,
                  autocast_dtype: Optional[torch.dtype], writer: Optional[Any], class_names: List[str],
                  deadline: Optional[float]) -> Tuple[Dict[int, Tuple[np.ndarray, int]], int, float]:
    """
    Runs the model on every image of a dataset, decoded by the loader workers, and writes the probabilities of every
    batch as it comes, until the deadline.

    Args:
        trainer (BaseTrainer): The trainer, preparing the batches as its evaluation does.
        model (torch.nn.Module): The scored model.
        dataset (Any): Test dataset of the trainer, or a subset of it.
        batch_size (int): Images per forward pass.
        number_workers (int): Loader worker processes.
        autocast_dtype (Optional[torch.dtype]): Autocast precision, None for float32.
        writer (Optional[Any]): CSV writer of the predictions file, None to keep the probabilities only.
        class_names (List[str]): Class names, in the order of the probabilities.
        deadline (Optional[float]): time.time() after which no new batch is scored, None for none.

    Returns:
        Tuple[Dict[int, Tuple[np.ndarray, int]], int, float]: The probabilities and the label of every sample
            identifier (empty when written), the number of scored images and the seconds taken.
    """
    loader = build_native_loader(dataset=dataset, batch_size=batch_size,
                                 collate_function=trainer.test_dataloader.collate_fn, shuffle=False,
                                 number_workers=number_workers)
    test_dataset = trainer.test_dataloader.dataset
    probabilities_by_identifier, scored_count, start_time = {}, 0, time.time()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if deadline is not None and time.time() > deadline:
                print(f"Time budget reached after {scored_count:,} images.")
                break

            # Forward pass of the prepared batch in the chosen precision, the probabilities kept in float32
            batch = trainer.prepare_batch(batch=batch)
            with torch.autocast(device_type=trainer.device.type, dtype=autocast_dtype or torch.float32,
                                enabled=autocast_dtype is not None):
                probabilities = model(input_tensor=batch["images"])["class_probabilities"].float().cpu().numpy()
            identifiers = batch[SAMPLE_IDENTIFIERS_KEY].cpu().numpy().tolist()

            # Rows written with nine significant digits, as the predictions.csv of the evaluation
            if writer is not None:
                for identifier, sample_probabilities in zip(identifiers, probabilities):
                    row = {SAMPLE_COLUMN: test_dataset.describe_sample(sample_identifier=int(identifier))[
                        SAMPLE_COLUMN]}
                    row.update({f"probability_{class_name}": f"{float(value):.9g}"
                                for class_name, value in zip(class_names, sample_probabilities)})
                    writer.writerow(row)
            else:
                labels = batch["labels"].cpu().numpy().tolist()
                probabilities_by_identifier.update(zip(identifiers, zip(probabilities, labels)))
            scored_count += len(identifiers)

            # Throughput every 100 batches
            if batch_index % 100 == 99:
                elapsed = time.time() - start_time
                print(f"{scored_count:,} images in {elapsed / 60:.1f} min, {scored_count / elapsed:.0f} images/s",
                      flush=True)
    return probabilities_by_identifier, scored_count, time.time() - start_time


def compute_balanced_accuracy(true_indices: np.ndarray, predicted_indices: np.ndarray, class_count: int) -> float:
    """
    Averages the recall of the classes present in the labels, the headline metric the best checkpoint is chosen by.

    Args:
        true_indices (np.ndarray): True class indices.
        predicted_indices (np.ndarray): Predicted class indices.
        class_count (int): Number of classes.

    Returns:
        float: The balanced accuracy.
    """
    recalls = [float(np.mean(predicted_indices[true_indices == class_index] == class_index))
               for class_index in range(class_count) if np.any(true_indices == class_index)]
    return float(np.mean(recalls))


def compare_precisions(trainer: BaseTrainer, model: torch.nn.Module, batch_size: int, number_workers: int,
                       class_names: List[str]) -> None:
    """
    Scores the labelled test split of the run in float32 and in bfloat16 and prints their speed, the largest difference
    of probability, the images whose predicted class changes and the balanced accuracy of each, so that the faster
    precision is used only when it gives the same results.

    Args:
        trainer (BaseTrainer): Trainer whose test split is the labelled one of the run.
        model (torch.nn.Module): The scored model.
        batch_size (int): Images per forward pass.
        number_workers (int): Loader worker processes.
        class_names (List[str]): Class names.
    """
    test_dataset = trainer.test_dataloader.dataset
    results = {}
    for precision, autocast_dtype in AUTOCAST_DTYPES.items():
        probabilities_by_identifier, scored_count, seconds = score_dataset(
            trainer=trainer, model=model, dataset=test_dataset, batch_size=batch_size, number_workers=number_workers,
            autocast_dtype=autocast_dtype, writer=None, class_names=class_names, deadline=None)
        results[precision] = probabilities_by_identifier
        print(f"{precision}: {scored_count:,} images in {seconds:.1f} s")
    # Agreement of the two precisions and balanced accuracy of each on the labels of the split
    identifiers = sorted(results["float32"])
    true_indices = np.array([results["float32"][identifier][1] for identifier in identifiers])
    probabilities = {precision: np.stack([result[identifier][0] for identifier in identifiers])
                     for precision, result in results.items()}
    changed_count = int(np.sum(probabilities["float32"].argmax(1) != probabilities["bfloat16"].argmax(1)))
    print(f"Largest probability difference: {np.abs(probabilities['float32'] - probabilities['bfloat16']).max():.5f}, "
          f"predicted class changed: {changed_count} of {len(identifiers):,}")
    for precision, precision_probabilities in probabilities.items():
        balanced_accuracy = compute_balanced_accuracy(true_indices=true_indices,
                                                      predicted_indices=precision_probabilities.argmax(1),
                                                      class_count=len(class_names))
        print(f"{precision}: balanced accuracy {balanced_accuracy * 100:.2f} %")


def read_scored_samples(output_path: Path, fieldnames: List[str]) -> Set[str]:
    """
    Reads the images already in the predictions file of an earlier launch, so that a scoring stopped by its time budget
    or interrupted resumes with the images left, and refuses a file the new rows cannot be appended to: one written for
    other classes, or whose last row was cut by an interruption.

    Args:
        output_path (Path): Predictions file of the scoring, absent or empty at the first launch.
        fieldnames (List[str]): Columns of the predictions file, the sample then the probability of every class.

    Returns:
        Set[str]: The images already scored, empty for an absent, empty or header-only file.

    Raises:
        ValueError: When the columns of the file differ, a row lacks a value or the file does not end with a line break.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()

    # The rows of the next launch go under the header of the file, which must name the same columns
    scored_samples, incomplete_line_numbers = set(), []
    with open(output_path, encoding="utf-8", newline="") as predictions_file:
        reader = csv.DictReader(f=predictions_file)
        if reader.fieldnames != fieldnames:
            raise ValueError(f"{output_path} has the columns {reader.fieldnames}, expected {fieldnames}: it was written "
                             f"for other classes, give another --output-path.")

        # A row cut by an interruption lacks values (None) or holds more of them (key None)
        for row in reader:
            if None in row or any(value in (None, "") for value in row.values()):
                incomplete_line_numbers.append(reader.line_num)
            scored_samples.add(row[SAMPLE_COLUMN])

    # A row cut inside its last value still has every column but no line break after it
    with open(output_path, "rb") as predictions_file:
        predictions_file.seek(-1, 2)
        ends_with_line_break = predictions_file.read(1) == b"\n"
    if incomplete_line_numbers or not ends_with_line_break:
        raise ValueError(f"{output_path} holds {len(incomplete_line_numbers)} incomplete rows (lines "
                         f"{incomplete_line_numbers[:5]}) and ends {'with' if ends_with_line_break else 'without'} a "
                         f"line break, as left by an interrupted write: remove its last line and launch again.")
    return scored_samples


def main() -> None:
    """
    Builds the trainer on the scored split, loads the checkpoint, optionally compares the precisions on the labelled
    test split, then scores the images not yet in the predictions file until the time budget ends.
    """
    arguments = parse_arguments()
    experiment_folder = Path(arguments.experiment_folder).absolute()

    # Scoring a split needs the split and its predictions file, the precision comparison alone needs neither
    scores_split = arguments.test_data_folder is not None or not arguments.compare_precisions
    if scores_split and (arguments.test_data_folder is None or arguments.output_path is None):
        raise ValueError("Scoring a split needs --test-data-folder and --output-path; only --compare-precisions runs "
                         "without them.")

    # Precisions compared on the labelled test split of the run, with the same weights
    if arguments.compare_precisions:
        trainer = build_scoring_trainer(experiment_folder=experiment_folder, test_data_folder=None)
        model = load_checkpoint_weights(trainer=trainer, experiment_folder=experiment_folder,
                                        iteration=arguments.iteration,
                                        use_averaged_weights=arguments.use_averaged_weights)
        compare_precisions(trainer=trainer, model=model, batch_size=arguments.batch_size,
                           number_workers=arguments.number_workers, class_names=list(trainer.class_names))
        if arguments.test_data_folder is None:
            return

    # Trainer reading the scored split, and the images not yet in the predictions file
    trainer = build_scoring_trainer(experiment_folder=experiment_folder, test_data_folder=arguments.test_data_folder)
    model = load_checkpoint_weights(trainer=trainer, experiment_folder=experiment_folder,
                                    iteration=arguments.iteration, use_averaged_weights=arguments.use_averaged_weights)
    class_names = list(trainer.class_names)
    test_dataset = trainer.test_dataloader.dataset
    output_path = Path(arguments.output_path)
    fieldnames = [SAMPLE_COLUMN] + [f"probability_{class_name}" for class_name in class_names]
    scored_samples = read_scored_samples(output_path=output_path, fieldnames=fieldnames)
    writes_header = not output_path.exists() or output_path.stat().st_size == 0
    remaining_indices = [index for index, sample in enumerate(test_dataset.samples)
                         if sample["file_name"] not in scored_samples]
    print(f"{len(test_dataset):,} images in the split, {len(scored_samples):,} already scored, "
          f"{len(remaining_indices):,} to score in {arguments.precision}")

    # Score the remaining images in the order of the split, appending to the predictions file
    deadline = None if arguments.time_budget_hours is None else time.time() + arguments.time_budget_hours * 3600
    with open(output_path, "a", encoding="utf-8", newline="") as predictions_file:
        writer = csv.DictWriter(f=predictions_file, fieldnames=fieldnames)

        # The header is written once, by the launch that creates the file
        if writes_header:
            writer.writeheader()
        _, scored_count, seconds = score_dataset(
            trainer=trainer, model=model, dataset=Subset(dataset=test_dataset, indices=remaining_indices),
            batch_size=arguments.batch_size, number_workers=arguments.number_workers,
            autocast_dtype=AUTOCAST_DTYPES[arguments.precision], writer=writer, class_names=class_names,
            deadline=deadline)
    print(f"Scored {scored_count:,} images in {seconds / 60:.1f} min ({scored_count / max(seconds, 1e-9):.0f} "
          f"images/s): {output_path}")


if __name__ == "__main__":
    main()
