"""
Tests of scripts/evaluation/score_image_split.py, the scoring of a large classification split by one checkpoint for the
mining of rare classes: the images are scored in the order of the split, a launch resumes after the images already in
the predictions file and writes its header only once, a predictions file of other classes or cut by an interruption is
refused, a scoring without split or predictions file is refused, the time budget stops the scoring before the next
batch, and the precision comparison prints the agreement and the balanced accuracy of both precisions. The trainer and
the checkpoint are replaced by a small split of one-value images and a model whose probabilities follow the image value.
"""

import csv
import io
import sys
import time

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

from torch.utils.data import DataLoader, Dataset

import scripts.evaluation.score_image_split as score_image_split_module
from scripts.evaluation.score_image_split import (SAMPLE_COLUMN, compare_precisions, compute_balanced_accuracy,
                                                  score_dataset)
from vision_training.core.contracts import SAMPLE_IDENTIFIERS_KEY

# Classes of the scored model and the columns of its predictions file
CLASS_NAMES = ["positive", "negative"]
FIELDNAMES = [SAMPLE_COLUMN] + [f"probability_{class_name}" for class_name in CLASS_NAMES]

# Images of the scored split, in the order of its labels.json
SPLIT_FILE_NAMES = [f"image_{index:03d}.png" for index in range(6)]


class FakeScoringDataset(Dataset):
    """
    Split of the tests read like the test dataset of a classification run: the image of every sample is its index as a
    single value, its label alternates between the two classes, and the predictions file names it by its file name.
    """

    def __init__(self, file_names: List[str]) -> None:
        """
        Keeps the file names of the split as its samples.

        Args:
            file_names (List[str]): The file names of the images, in the order of the split.
        """
        self.samples = [{"file_name": file_name} for file_name in file_names]

    def __len__(self) -> int:
        """
        Gives the number of images of the split.

        Returns:
            int: The number of images.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Gives the one-value image, the label and the sample identifier (its index) of an image.

        Args:
            index (int): The index of the image in the split.

        Returns:
            Dict[str, Any]: The image, label and identifier tensors, batched by the default collate function.
        """
        return {"images": torch.tensor([float(index)]), "labels": torch.tensor(index % 2),
                SAMPLE_IDENTIFIERS_KEY: torch.tensor(index)}

    def describe_sample(self, sample_identifier: int) -> Dict[str, str]:
        """
        Names an image in the predictions file from its identifier, as the classification dataset does.

        Args:
            sample_identifier (int): The identifier of the image, its index in the split.

        Returns:
            Dict[str, str]: The 'sample' column, the file name of the image.
        """
        return {SAMPLE_COLUMN: self.samples[sample_identifier]["file_name"]}


class FakeScoringModel(torch.nn.Module):
    """
    Classifier of the tests whose probabilities follow the image value v: the softmax of (v / 4, -v / 4), so that every
    image but the first is given the first class.
    """

    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Gives the class probabilities of a batch of one-value images.

        Args:
            input_tensor (torch.Tensor): The images, of shape (batch, 1).

        Returns:
            Dict[str, torch.Tensor]: The 'class_probabilities', of shape (batch, 2).
        """
        return {"class_probabilities": torch.softmax(torch.cat([input_tensor, -input_tensor], dim=1) / 4, dim=1)}


class FakeScoringTrainer:
    """
    Trainer of the tests, giving what score_image_split.py reads from a classification trainer: its device, its class
    names, its test loader (dataset and batching function) and its batch preparation.
    """

    def __init__(self, file_names: List[str]) -> None:
        """
        Builds the test loader of the split on the CPU.

        Args:
            file_names (List[str]): The file names of the images, in the order of the split.
        """
        self.device = torch.device("cpu")
        self.class_names = list(CLASS_NAMES)
        self.test_dataloader = DataLoader(dataset=FakeScoringDataset(file_names=file_names))

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Leaves the batch as it is, the images being already on the CPU.

        Args:
            batch (Dict[str, Any]): The batch of the loader.

        Returns:
            Dict[str, Any]: The same batch.
        """
        return batch


def build_fake_scoring_trainer(experiment_folder: Path, test_data_folder: Optional[str]) -> FakeScoringTrainer:
    """
    Replaces build_scoring_trainer of the script, giving the trainer of the split of the tests whatever the run.

    Args:
        experiment_folder (Path): The run folder given to the script, unused.
        test_data_folder (Optional[str]): The scored split given to the script, unused.

    Returns:
        FakeScoringTrainer: The trainer of the split of the tests.
    """
    return FakeScoringTrainer(file_names=SPLIT_FILE_NAMES)


def load_fake_checkpoint_weights(trainer: FakeScoringTrainer, experiment_folder: Path, iteration: int,
                                 use_averaged_weights: bool) -> torch.nn.Module:
    """
    Replaces load_checkpoint_weights of the script, giving the model of the tests whatever the checkpoint.

    Args:
        trainer (FakeScoringTrainer): The trainer of the tests, unused.
        experiment_folder (Path): The run folder given to the script, unused.
        iteration (int): The checkpoint iteration given to the script, unused.
        use_averaged_weights (bool): Whether the averaged weights were asked, unused.

    Returns:
        torch.nn.Module: The model of the tests, in evaluation mode.
    """
    return FakeScoringModel().eval()


def run_scoring(monkeypatch: pytest.MonkeyPatch, arguments: List[str]) -> None:
    """
    Runs the main function of the script with the trainer and the model of the tests, on the command line arguments
    given after the run folder, the iteration, the batch size of 2 and the loading in the main process.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the trainer, the weights and the command line.
        arguments (List[str]): The other command line arguments.
    """
    monkeypatch.setattr(score_image_split_module, "build_scoring_trainer", build_fake_scoring_trainer)
    monkeypatch.setattr(score_image_split_module, "load_checkpoint_weights", load_fake_checkpoint_weights)
    monkeypatch.setattr(sys, "argv", ["score_image_split.py", "--experiment_folder", "run", "--iteration", "10",
                                      "--batch_size", "2", "--number_workers", "0"] + arguments)
    score_image_split_module.main()


def compute_expected_probabilities(index: int) -> List[float]:
    """
    Gives the probabilities the model of the tests gives the image of an index of the split.

    Args:
        index (int): The index of the image in the split.

    Returns:
        List[float]: The probability of every class.
    """
    return FakeScoringModel()(input_tensor=torch.tensor([[float(index)]]))["class_probabilities"][0].tolist()


def test_scoring_resumes_after_the_scored_images_in_the_order_of_the_split(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A predictions file holding two images of the split, as left by a launch stopped by its time budget, keeps its rows
    and receives the four other images in the order of the split, each with the probabilities of the model.

    Args:
        tmp_path (Path): The temporary folder of the test, receiving the predictions file.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the trainer, the weights and the command line.
    """
    output_path = tmp_path / "predictions.csv"
    output_path.write_text(data=",".join(FIELDNAMES) + "\r\nimage_000.png,0.5,0.5\r\nimage_003.png,0.1,0.9\r\n",
                           encoding="utf-8")
    run_scoring(monkeypatch=monkeypatch, arguments=["--test_data_folder", "split", "--output_path", str(output_path)])

    # The two earlier rows stay first, the four other images follow in the order of the split
    with open(output_path, encoding="utf-8", newline="") as predictions_file:
        rows = list(csv.DictReader(f=predictions_file))
    assert [row[SAMPLE_COLUMN] for row in rows] == ["image_000.png", "image_003.png", "image_001.png", "image_002.png",
                                                    "image_004.png", "image_005.png"]
    assert rows[1]["probability_negative"] == "0.9"
    for row in rows[2:]:
        index = SPLIT_FILE_NAMES.index(row[SAMPLE_COLUMN])
        written_probabilities = [float(row[f"probability_{class_name}"]) for class_name in CLASS_NAMES]
        assert written_probabilities == pytest.approx(compute_expected_probabilities(index=index), rel=1e-6)


def test_header_only_file_gets_no_second_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A predictions file holding only its header, as left by a launch stopped before its first batch, receives every image
    of the split under that header, without a second header line.

    Args:
        tmp_path (Path): The temporary folder of the test, receiving the predictions file.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the trainer, the weights and the command line.
    """
    output_path = tmp_path / "predictions.csv"
    output_path.write_text(data=",".join(FIELDNAMES) + "\r\n", encoding="utf-8")
    run_scoring(monkeypatch=monkeypatch, arguments=["--test_data_folder", "split", "--output_path", str(output_path)])
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines.count(",".join(FIELDNAMES)) == 1
    assert len(lines) == 1 + len(SPLIT_FILE_NAMES)


def test_predictions_file_of_other_classes_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A predictions file written for other classes is refused before anything is appended to it.

    Args:
        tmp_path (Path): The temporary folder of the test, receiving the predictions file.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the trainer, the weights and the command line.
    """
    output_path = tmp_path / "predictions.csv"
    written_text = "sample,probability_cat,probability_dog\r\nimage_000.png,0.5,0.5\r\n"
    output_path.write_text(data=written_text, encoding="utf-8")
    with pytest.raises(ValueError, match="written for other classes"):
        run_scoring(monkeypatch=monkeypatch,
                    arguments=["--test_data_folder", "split", "--output_path", str(output_path)])
    assert output_path.read_bytes() == written_text.encode(encoding="utf-8")


@pytest.mark.parametrize("last_row", ["image_003.png,0.1", "image_003.png,0.1,0.9", "image_003.png,0.1,,\r\n"])
def test_row_cut_by_an_interruption_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, last_row: str) -> None:
    """
    A predictions file whose last row lacks a value, has no line break after its last value or holds an empty value,
    as left by an interrupted write, is refused before anything is appended to it.

    Args:
        tmp_path (Path): The temporary folder of the test, receiving the predictions file.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the trainer, the weights and the command line.
        last_row (str): The last row of the predictions file.
    """
    output_path = tmp_path / "predictions.csv"
    written_text = ",".join(FIELDNAMES) + "\r\nimage_000.png,0.5,0.5\r\n" + last_row
    output_path.write_text(data=written_text, encoding="utf-8")
    with pytest.raises(ValueError, match="interrupted write"):
        run_scoring(monkeypatch=monkeypatch,
                    arguments=["--test_data_folder", "split", "--output_path", str(output_path)])
    assert output_path.read_bytes() == written_text.encode(encoding="utf-8")


@pytest.mark.parametrize("arguments", [["--output_path", "predictions.csv"], ["--test_data_folder", "split"],
                                       ["--compare_precisions", "--test_data_folder", "split"]])
def test_scoring_without_split_or_predictions_file_is_refused(monkeypatch: pytest.MonkeyPatch,
                                                              arguments: List[str]) -> None:
    """
    A scoring needs the split and its predictions file, also after a precision comparison, and is refused before the
    trainer is built when one of them is missing.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest fixture replacing the trainer, the weights and the command line.
        arguments (List[str]): The command line arguments, missing the split or the predictions file.
    """
    with pytest.raises(ValueError, match="needs --test_data_folder and --output_path"):
        run_scoring(monkeypatch=monkeypatch, arguments=arguments)


def test_time_budget_stops_the_scoring_before_the_next_batch() -> None:
    """
    A deadline already passed scores and writes no image, while no deadline scores every image of the split.
    """
    trainer = FakeScoringTrainer(file_names=SPLIT_FILE_NAMES)
    for deadline, expected_count in [(time.time() - 1, 0), (None, len(SPLIT_FILE_NAMES))]:
        predictions_text = io.StringIO()
        writer = csv.DictWriter(f=predictions_text, fieldnames=FIELDNAMES)
        _, scored_count, _ = score_dataset(trainer=trainer, model=FakeScoringModel().eval(),
                                           dataset=trainer.test_dataloader.dataset, batch_size=2, number_workers=0,
                                           autocast_dtype=None, writer=writer, class_names=CLASS_NAMES,
                                           deadline=deadline)
        assert scored_count == expected_count
        assert len(predictions_text.getvalue().splitlines()) == expected_count


def test_precision_comparison_prints_the_agreement_and_the_balanced_accuracy(capsys: pytest.CaptureFixture) -> None:
    """
    The comparison scores the labelled split in float32 and bfloat16 and prints that no predicted class changes and
    the balanced accuracy of each: 50.00 %, the first class (labels 0, 2, 4) found every time and the second never.

    Args:
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed comparison.
    """
    trainer = FakeScoringTrainer(file_names=SPLIT_FILE_NAMES)
    compare_precisions(trainer=trainer, model=FakeScoringModel().eval(), batch_size=2, number_workers=0,
                       class_names=CLASS_NAMES)
    printed_text = capsys.readouterr().out
    assert "predicted class changed: 0 of 6" in printed_text
    assert "float32: balanced accuracy 50.00 %" in printed_text
    assert "bfloat16: balanced accuracy 50.00 %" in printed_text


def test_balanced_accuracy_averages_the_recall_of_the_classes_present() -> None:
    """
    The balanced accuracy averages the recall of the classes holding labels (1 of 2 and 3 of 4), a class without labels
    being left out.
    """
    balanced_accuracy = compute_balanced_accuracy(true_indices=np.array([0, 0, 1, 1, 1, 1]),
                                                  predicted_indices=np.array([0, 1, 1, 1, 1, 0]), class_count=3)
    assert balanced_accuracy == pytest.approx((0.5 + 0.75) / 2)
