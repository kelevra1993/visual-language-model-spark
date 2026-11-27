"""
Tests of scripts/evaluation/draw_report_figures.py, the figures of the research report of a kept classifier drawn from
its run folder: the confusion matrix, the precision-recall curves at the temperature saved from the temperature
explorer (its thresholds checked against the class logits, a temperature of 1 without saved file), the training curves
of the trained and averaged weights with the kept checkpoint, and the balanced accuracy of the groups seen and never
seen in training, read from the predictions of the evaluation and a manifest of the group of every sample. Every input
is checked before any figure is drawn, and a threshold near an edge of the recall axis is written on the other side of
its point. The script prints its help when launched without arguments, reads every setting of draw_report_figures but
the font from its command line, DEFAULT_REPORT_SETTINGS for those not given, and refuses a missing run folder or
iteration, an unknown value or a class given two names before anything is read.
"""

import csv
import inspect
import json
import pytest
import torch
import yaml

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pathlib import Path
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional

import scripts.evaluation.draw_report_figures as draw_report_figures_module
from scripts.evaluation.draw_report_figures import (DEFAULT_REPORT_SETTINGS, GROUPS_FIGURE_NAME,
                                                    PRECISION_RECALL_FIGURE_NAME, REPORT_FIGURES_FOLDER_NAME,
                                                    REPORT_VALUES_FILE_NAME, TRAINING_CURVES_FIGURE_NAME,
                                                    draw_class_precision_recall, draw_report_figures,
                                                    parse_report_arguments, read_confusion_matrix, read_sample_groups,
                                                    register_report_font, style_precision_recall_axis)
from vision_training.evaluation.classification.classification_metrics import compute_confusion_matrix
from vision_training.evaluation.classification.confusion_matrix import save_confusion_matrix_table
from vision_training.evaluation.classification.temperature_explorer import (compute_temperature_probabilities,
                                                                            compute_temperature_thresholds,
                                                                            load_class_logits, save_class_logits,
                                                                            write_saved_thresholds)

# Classes of the fake classifiers, the kept iteration and the iterations evaluated during their training, with the
# balanced accuracy of the trained and averaged weights at each
THREE_CLASS_NAMES = ["cat", "dog", "bird"]
TWO_CLASS_NAMES = ["empty", "full"]
KEPT_ITERATION = 100
TRAINED_CURVE = {50: 0.62, 100: 0.71, 150: 0.69}
AVERAGED_CURVE = {50: 0.6, 100: 0.74, 150: 0.73}

# Target precisions of the thresholds, the main one first, and the temperature saved from the temperature explorer
TARGET_PRECISIONS = [0.9, 0.8]
SAVED_TEMPERATURE = 2.0

# Trucks of the samples: the training samples come from the first four trucks only, so that the validation samples
# of the last two are never seen in training
TRUCK_NAMES = ["t1", "t2", "t3", "t4", "t5", "t6"]
TRAINING_TRUCK_NAMES = ["t1", "t2", "t3", "t4"]

# Common settings of the figures drawn by the tests
FIGURE_SETTINGS = {"evaluation_folder_name": "Checkpoint_Evaluation", "sample_kind": "images",
                   "split_name": "validation", "minimum_recall": 0.0, "file_name_key": "file_name",
                   "group_key": "truck", "group_plural": "trucks", "training_labels_path": None,
                   "predictions_path": None, "font_family": None}

# Every argument of the script, which its help must describe
REPORT_SCRIPT_ARGUMENTS = ["--experiment_folder", "--iteration", "--weights", "--evaluation_folder_name",
                           "--class_display_names", "--sample_kind", "--split_name", "--minimum_recall",
                           "--group_manifest_path", "--file_name_key", "--group_key", "--group_plural",
                           "--training_labels_path", "--predictions_path"]


def write_test_curves(test_folder: Path, curve: Dict[int, float]) -> None:
    """
    Writes the balanced accuracy curve of a fake run in a Tensorboard folder, as the trainer writes it at every
    evaluation of its training.

    Args:
        test_folder (Path): The Tensorboard/Test or Tensorboard/Test_EMA folder.
        curve (Dict[int, float]): The balanced accuracy at every evaluated iteration.
    """
    writer = SummaryWriter(log_dir=str(test_folder))
    for iteration, value in curve.items():
        writer.add_scalar(tag="Balanced Accuracy", scalar_value=value, global_step=iteration)
    writer.close()


def write_fake_run(run_folder: Path, class_names: List[str], weights: str, saved_temperature: Optional[float],
                   averaged_curve: Optional[Dict[int, float]]) -> Dict[str, Any]:
    """
    Writes a fake classification run with its evaluation of the kept checkpoint, as the trainer and
    scripts/evaluation/evaluate_checkpoint.py write them: 20 validation images per class whose logits favour their
    class with a few errors (class_logits.npz, confusion_matrix.csv, class_thresholds.json at a temperature of 1, and
    saved_temperature_thresholds.json when a temperature is saved), predictions.csv, the test curves of the training,
    its configuration pointing to a training split of 24 images, and the manifest of the truck of every image.

    Args:
        run_folder (Path): The run folder to write.
        class_names (List[str]): The classes of the classifier.
        weights (str): The weights of the evaluation, 'trained' or 'ema'.
        saved_temperature (Optional[float]): The temperature saved from the temperature explorer, None for none.
        averaged_curve (Optional[Dict[int, float]]): The balanced accuracy of the averaged weights during the
            training, None for a run without moving average.

    Returns:
        Dict[str, Any]: The 'evaluation_folder', the 'manifest_path', the 'labels' and 'predicted' class indices, the
            'file_names' and 'trucks' of the validation images, and the 'record' of the class logits.
    """
    # Logits favouring the class of every image, a few images given another class
    generator = torch.Generator().manual_seed(7)
    labels = torch.arange(len(class_names)).repeat_interleave(20)
    class_logits = torch.randn(len(labels), len(class_names), generator=generator)
    class_logits[torch.arange(len(labels)), labels] += 2.5
    predicted = class_logits.argmax(dim=1)
    file_names = [f"image_{index:03d}.jpg" for index in range(len(labels))]
    trucks = [TRUCK_NAMES[index % len(TRUCK_NAMES)] for index in range(len(labels))]

    # Evaluation folder of the kept checkpoint, its thresholds at a temperature of 1 and the saved ones
    evaluation_folder = run_folder / "Checkpoint_Evaluation" / f"Iteration_{KEPT_ITERATION}"
    evaluation_folder = evaluation_folder / "EMA" if weights == "ema" else evaluation_folder
    evaluation_folder.mkdir(parents=True)
    save_class_logits(class_logits=class_logits, labels=labels, class_names=class_names, output_mode="multi_class",
                      target_precisions=TARGET_PRECISIONS, class_f1_scores=[0.5] * len(class_names),
                      output_folder=evaluation_folder, title=f"Iteration {KEPT_ITERATION}")
    record = load_class_logits(evaluation_folder=evaluation_folder)
    save_confusion_matrix_table(confusion_matrix=compute_confusion_matrix(predicted_labels=predicted, labels=labels,
                                                                          number_classes=len(class_names)),
                                class_names=class_names, output_folder=evaluation_folder)
    (evaluation_folder / "class_thresholds.json").write_text(data=json.dumps(obj=compute_temperature_thresholds(
        class_logits_record=record, logit_temperature=1.0)))
    if saved_temperature is not None:
        write_saved_thresholds(evaluation_folder=evaluation_folder, saved_thresholds=compute_temperature_thresholds(
            class_logits_record=record, logit_temperature=saved_temperature))

    # Predictions of every validation image, as the evaluation writes them with save_predictions
    with (evaluation_folder / "predictions.csv").open(mode="w", newline="") as predictions_file:
        writer = csv.DictWriter(f=predictions_file, fieldnames=["sample", "true_class", "predicted_class", "correct"])
        writer.writeheader()
        writer.writerows([{"sample": file_name, "true_class": class_names[int(label)],
                           "predicted_class": class_names[int(predicted_index)],
                           "correct": int(label == predicted_index)}
                          for file_name, label, predicted_index in zip(file_names, labels, predicted)])

    # Test curves of the training, and the training split of the first trucks only
    write_test_curves(test_folder=run_folder / "Tensorboard" / "Test", curve=TRAINED_CURVE)
    if averaged_curve is not None:
        write_test_curves(test_folder=run_folder / "Tensorboard" / "Test_EMA", curve=averaged_curve)
    train_folder = run_folder.parent / "data" / "train"
    train_folder.mkdir(parents=True, exist_ok=True)
    training_samples = [{"id": index + 1, "file_name": f"train_{index:03d}.jpg", "category_id": 1}
                        for index in range(24)]
    (train_folder / "labels.json").write_text(data=json.dumps(obj={"categories": [], "samples": training_samples}))
    (run_folder / "training_configuration.yaml").write_text(data=yaml.safe_dump(
        data={"ExperimentConfiguration": {"task": "classification", "train_data_folder": str(train_folder)}}))

    # Manifest of the truck of every training and validation image
    manifest_path = run_folder.parent / "data" / "frames_manifest.jsonl"
    manifest_records = ([{"file_name": sample["file_name"], "truck": TRAINING_TRUCK_NAMES[index % 4]}
                         for index, sample in enumerate(training_samples)] +
                        [{"file_name": file_name, "truck": truck} for file_name, truck in zip(file_names, trucks)])
    manifest_path.write_text(data="".join(json.dumps(obj=manifest_record) + "\n"
                                          for manifest_record in manifest_records))
    return {"evaluation_folder": evaluation_folder, "manifest_path": manifest_path, "labels": labels,
            "predicted": predicted, "file_names": file_names, "trucks": trucks, "record": record}


def compute_balanced_accuracy(labels: List[int], predicted: List[int]) -> float:
    """
    Computes the mean recall of the classes present among some samples, the balanced accuracy the groups chart
    shows, written out here so that the test does not reuse the metrics of the script.

    Args:
        labels (List[int]): The true class of every sample.
        predicted (List[int]): The predicted class of every sample.

    Returns:
        float: The balanced accuracy.
    """
    recalls = [sum(1 for label, prediction in zip(labels, predicted) if label == class_index == prediction) /
               labels.count(class_index) for class_index in sorted(set(labels))]
    return sum(recalls) / len(recalls)


def test_report_figures_are_drawn_from_the_evaluation_and_the_training(tmp_path: Path) -> None:
    """
    The four figures and report_values.json are written in Report_Figures of the evaluation folder: the thresholds are
    the saved ones at their temperature, written with the fewest decimals selecting the same images, the kept
    checkpoint is the balanced accuracy of its weights during the training, and the groups chart splits the validation
    images of the trucks with training images from those of the trucks never seen.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the fake run.
    """
    fake_run = write_fake_run(run_folder=tmp_path / "run", class_names=THREE_CLASS_NAMES, weights="ema",
                              saved_temperature=SAVED_TEMPERATURE, averaged_curve=AVERAGED_CURVE)

    output_folder = draw_report_figures(experiment_folder=tmp_path / "run", iteration=KEPT_ITERATION, weights="ema",
                                        class_display_names={"bird": "Bird", "cat": "Cat", "dog": "Dog"},
                                        group_manifest_path=fake_run["manifest_path"], **FIGURE_SETTINGS)

    # Every figure written as an image, in the evaluation folder of the kept weights
    assert output_folder == fake_run["evaluation_folder"] / REPORT_FIGURES_FOLDER_NAME
    for figure_name in ["confusion_matrix.png", PRECISION_RECALL_FIGURE_NAME, TRAINING_CURVES_FIGURE_NAME,
                        GROUPS_FIGURE_NAME]:
        with Image.open(output_folder / figure_name) as image:
            assert image.width > 500 and image.height > 300
    report_values = json.loads(s=(output_folder / REPORT_VALUES_FILE_NAME).read_text())

    # Saved thresholds at their temperature, in the order of the names shown, each written to select the same images
    saved_thresholds = json.loads(s=(fake_run["evaluation_folder"] / "saved_temperature_thresholds.json").read_text())
    probabilities = compute_temperature_probabilities(class_logits=fake_run["record"]["class_logits"],
                                                      output_mode="multi_class", logit_temperature=SAVED_TEMPERATURE)
    assert report_values["logit_temperature"] == SAVED_TEMPERATURE and report_values["temperature_saved"]
    assert list(report_values["class_thresholds"]) == ["bird", "cat", "dog"]
    for class_name, class_targets in report_values["class_thresholds"].items():
        class_scores = probabilities[:, THREE_CLASS_NAMES.index(class_name)]
        saved_targets = [saved_thresholds["classes"][class_name]] + [
            target["classes"][class_name] for target in saved_thresholds["additional_targets"]]
        assert [target["target_precision"] for target in class_targets if target] == [
            precision for precision, saved_target in zip(TARGET_PRECISIONS, saved_targets) if saved_target]

        # Every reached target: the saved threshold and recall, its written text keeping the same images
        for class_target, saved_target in zip(class_targets, saved_targets):
            assert (class_target is None) == (saved_target is None)
            if class_target is None:
                continue
            assert class_target["threshold"] == saved_target["threshold"]
            assert class_target["recall"] == pytest.approx(saved_target["recall"])
            assert float(class_target["threshold_text"]) <= saved_target["threshold"]
            assert int((class_scores >= float(class_target["threshold_text"])).sum()) == int(
                (class_scores >= saved_target["threshold"]).sum())

    # Accuracy of the evaluation and balanced accuracy of the kept averaged weights during the training
    labels, predicted = fake_run["labels"].tolist(), fake_run["predicted"].tolist()
    assert report_values["accuracy"] == pytest.approx(sum(label == prediction for label, prediction
                                                          in zip(labels, predicted)) / len(labels))
    assert report_values["balanced_accuracy"] == pytest.approx(compute_balanced_accuracy(labels=labels,
                                                                                         predicted=predicted))
    assert report_values["training_kept_balanced_accuracy"] == pytest.approx(AVERAGED_CURVE[KEPT_ITERATION])

    # Groups: the images of t5 and t6 are never seen, their balanced accuracy computed on them only
    never_seen_indices = [index for index, truck in enumerate(fake_run["trucks"]) if truck in ("t5", "t6")]
    groups = report_values["groups"]
    assert groups["never_seen_groups"] == ["t5", "t6"]
    assert (groups["seen"]["groups"], groups["never_seen"]["groups"], groups["all"]["groups"]) == (4, 2, 6)
    assert groups["never_seen"]["samples"] == len(never_seen_indices) == 20
    assert groups["seen"]["samples"] + groups["never_seen"]["samples"] == len(labels)
    assert groups["never_seen"]["balanced_accuracy"] == pytest.approx(compute_balanced_accuracy(
        labels=[labels[index] for index in never_seen_indices], predicted=[predicted[index]
                                                                            for index in never_seen_indices]))
    assert groups["all"]["balanced_accuracy"] == pytest.approx(report_values["balanced_accuracy"])


def test_binary_classifier_without_saved_temperature_is_drawn_at_one(tmp_path: Path,
                                                                    capsys: pytest.CaptureFixture) -> None:
    """
    Without saved_temperature_thresholds.json, the thresholds are found from the logits at a temperature of 1 and a
    warning says so; the two classes of a binary classifier share one plot, the trained weights of a run without moving
    average giving the training curves, and no groups chart is drawn without manifest.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the fake run.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    write_fake_run(run_folder=tmp_path / "run", class_names=TWO_CLASS_NAMES, weights="trained", saved_temperature=None,
                   averaged_curve=None)

    output_folder = draw_report_figures(experiment_folder=tmp_path / "run", iteration=KEPT_ITERATION,
                                        weights="trained", class_display_names=None, group_manifest_path=None,
                                        **FIGURE_SETTINGS)

    report_values = json.loads(s=(output_folder / REPORT_VALUES_FILE_NAME).read_text())
    assert report_values["logit_temperature"] == 1.0 and not report_values["temperature_saved"]
    assert list(report_values["class_thresholds"]) == TWO_CLASS_NAMES
    assert report_values["training_kept_balanced_accuracy"] == pytest.approx(TRAINED_CURVE[KEPT_ITERATION])
    assert report_values["groups"] is None and not (output_folder / GROUPS_FIGURE_NAME).exists()
    assert "drawn at a logit temperature of 1" in capsys.readouterr().out


def test_saved_thresholds_must_be_found_again_from_the_logits(tmp_path: Path) -> None:
    """
    Saved thresholds that the class logits do not give again at their temperature are refused, the figure marking the
    thresholds deployed with the export.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the fake run.
    """
    fake_run = write_fake_run(run_folder=tmp_path / "run", class_names=THREE_CLASS_NAMES, weights="trained",
                              saved_temperature=SAVED_TEMPERATURE, averaged_curve=None)
    saved_path = fake_run["evaluation_folder"] / "saved_temperature_thresholds.json"
    saved_thresholds = json.loads(s=saved_path.read_text())
    changed_class = next(class_name for class_name, class_threshold in saved_thresholds["classes"].items()
                         if class_threshold is not None)
    saved_thresholds["classes"][changed_class]["threshold"] /= 2.0
    saved_path.write_text(data=json.dumps(obj=saved_thresholds))

    with pytest.raises(ValueError, match="differ from"):
        draw_report_figures(experiment_folder=tmp_path / "run", iteration=KEPT_ITERATION, weights="trained",
                            class_display_names=None, group_manifest_path=None, **FIGURE_SETTINGS)


def test_groups_chart_inputs_are_checked_before_any_figure(tmp_path: Path) -> None:
    """
    The groups chart needs the predictions of the evaluation itself (they give its confusion matrix), the group of
    every image, and images of trucks both seen and never seen; each of these inputs is refused before any figure is
    drawn, the folder of the figures never being created.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the fake run.
    """
    fake_run = write_fake_run(run_folder=tmp_path / "run", class_names=THREE_CLASS_NAMES, weights="trained",
                              saved_temperature=None, averaged_curve=None)
    draw_settings = {**FIGURE_SETTINGS, "experiment_folder": tmp_path / "run", "iteration": KEPT_ITERATION,
                     "weights": "trained", "class_display_names": None,
                     "group_manifest_path": fake_run["manifest_path"]}
    figures_folder = fake_run["evaluation_folder"] / REPORT_FIGURES_FOLDER_NAME
    predictions_path = fake_run["evaluation_folder"] / "predictions.csv"
    predictions_text = predictions_path.read_text()

    # Missing predictions file
    predictions_path.unlink()
    with pytest.raises(FileNotFoundError, match="--save_predictions True"):
        draw_report_figures(**draw_settings)
    assert not figures_folder.exists()

    # Predictions of another evaluation, one image predicted otherwise
    rows = list(csv.DictReader(predictions_text.splitlines()))
    rows[0]["predicted_class"] = next(class_name for class_name in THREE_CLASS_NAMES
                                      if class_name != rows[0]["predicted_class"])
    with predictions_path.open(mode="w", newline="") as predictions_file:
        writer = csv.DictWriter(f=predictions_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="do not give the confusion matrix"):
        draw_report_figures(**draw_settings)
    assert not figures_folder.exists()
    predictions_path.write_text(data=predictions_text)

    # An image without truck in the manifest
    manifest_lines = fake_run["manifest_path"].read_text().splitlines(keepends=True)
    fake_run["manifest_path"].write_text(data="".join(manifest_lines[:-1]))
    with pytest.raises(ValueError, match="1 samples have no group"):
        draw_report_figures(**draw_settings)
    assert not figures_folder.exists()

    # Every truck seen in training: nothing to compare
    fake_run["manifest_path"].write_text(data="".join(
        json.dumps(obj={**json.loads(s=line), "truck": "t1"}) + "\n" for line in manifest_lines))
    with pytest.raises(ValueError, match="0 from groups never seen"):
        draw_report_figures(**draw_settings)
    assert not figures_folder.exists()


def test_training_curves_need_the_kept_checkpoint(tmp_path: Path) -> None:
    """
    The kept checkpoint must have been evaluated during the training with its weights: an iteration never evaluated,
    or averaged weights of a run without moving average, are refused.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the fake run.
    """
    write_fake_run(run_folder=tmp_path / "run", class_names=TWO_CLASS_NAMES, weights="ema", saved_temperature=None,
                   averaged_curve=None)

    with pytest.raises(ValueError, match="evaluated no ema weights at iteration 100"):
        draw_report_figures(experiment_folder=tmp_path / "run", iteration=KEPT_ITERATION, weights="ema",
                            class_display_names=None, group_manifest_path=None, **FIGURE_SETTINGS)
    with pytest.raises(FileNotFoundError, match="No evaluation of iteration 200"):
        draw_report_figures(experiment_folder=tmp_path / "run", iteration=200, weights="ema",
                            class_display_names=None, group_manifest_path=None, **FIGURE_SETTINGS)
    with pytest.raises(ValueError, match="Expected the weights"):
        draw_report_figures(experiment_folder=tmp_path / "run", iteration=KEPT_ITERATION, weights="averaged",
                            class_display_names=None, group_manifest_path=None, **FIGURE_SETTINGS)


def test_threshold_written_near_an_edge_goes_to_the_other_side() -> None:
    """
    A threshold written right of its point near the right end of the recall axis, or left of it near the left end, is
    written on the other side, so that it stays inside the plot; one in the middle keeps its side.
    """
    figure = Figure()
    FigureCanvasAgg(figure=figure)
    axis = figure.subplots()
    style_precision_recall_axis(axis=axis, target_precisions=[0.9], minimum_precision=0.6, minimum_recall=0.0,
                                label_targets=False)
    curves = {"recalls": [0.0, 0.05, 0.5, 0.95], "precisions": [1.0, 1.0, 0.95, 0.9],
              "threshold_points": [(0.95, 0.9, "0.1000"), (0.5, 0.95, "0.5000"), (0.05, 1.0, "0.9000")]}

    draw_class_precision_recall(axis=axis, curves=curves, color="#2b5fa8", label=None, text_offset=(8, 7))
    draw_class_precision_recall(axis=axis, curves=curves, color="#2b5fa8", label=None, text_offset=(-8, -16))

    alignments = [(annotation.get_text(), annotation.get_horizontalalignment()) for annotation in axis.texts]
    assert alignments == [("0.1000", "right"), ("0.5000", "left"), ("0.9000", "left"),
                          ("0.1000", "right"), ("0.5000", "right"), ("0.9000", "left")]


def test_confusion_matrix_and_manifest_are_checked(tmp_path: Path) -> None:
    """
    A confusion matrix whose rows are not the classes of its header is refused, and so is a manifest giving a sample two
    groups or a sample without group.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the table and the manifest.
    """
    table_path = tmp_path / "confusion_matrix.csv"
    table_path.write_text(data="true \\ predicted,cat,dog\ndog,1,2\ncat,3,4\n")
    with pytest.raises(ValueError, match="one row of counts per class"):
        read_confusion_matrix(table_path=table_path)
    table_path.write_text(data="true \\ predicted,cat,dog\ncat,1,2\ndog,3,4\n")
    class_names, confusion_matrix = read_confusion_matrix(table_path=table_path)
    assert class_names == ["cat", "dog"] and confusion_matrix.tolist() == [[1, 2], [3, 4]]

    manifest_path = tmp_path / "frames_manifest.jsonl"
    manifest_path.write_text(data='{"file_name": "a.jpg", "truck": "t1"}\n{"file_name": "a.jpg", "truck": "t2"}\n')
    with pytest.raises(ValueError, match="gives a.jpg the groups"):
        read_sample_groups(manifest_path=manifest_path, file_name_key="file_name", group_key="truck")
    manifest_path.write_text(data='{"file_name": "a.jpg", "truck": "t1"}\n\n{"file_name": "b.jpg", "truck": null}\n')
    with pytest.raises(ValueError, match="Line 3"):
        read_sample_groups(manifest_path=manifest_path, file_name_key="file_name", group_key="truck")


def test_missing_report_font_keeps_the_default_font(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture) -> None:
    """
    Without the Inter font of the reports, the figures keep matplotlib's default font, a warning saying so.

    Args:
        tmp_path (Path): The temporary folder of the test, standing for a font folder without fonts.
        monkeypatch (pytest.MonkeyPatch): The pytest fixture pointing the script to that folder.
        capsys (pytest.CaptureFixture): The pytest fixture capturing the printed warning.
    """
    monkeypatch.setattr(draw_report_figures_module, "REPORT_FONT_FOLDER", tmp_path / "no_fonts")

    assert register_report_font() is None
    assert "not installed" in capsys.readouterr().out


def test_report_script_without_arguments_prints_its_help(capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the script launched without any argument prints the help of every one of its arguments and stops
    without drawing, instead of failing on the missing run folder.

    Args:
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_report_arguments(arguments=[])
    assert exit_information.value.code == 0
    help_text = capsys.readouterr().out
    assert [argument for argument in REPORT_SCRIPT_ARGUMENTS if argument not in help_text] == []


def test_report_arguments_default_to_the_report_settings() -> None:
    """
    Verifies that the script given only the run folder and the iteration draws with DEFAULT_REPORT_SETTINGS, the class
    names and no groups chart, and that every parsed argument is a parameter of draw_report_figures, the font alone
    being left to register_report_font, so that the main block passes them all.
    """
    parsed_arguments = parse_report_arguments(arguments=["--experiment_folder", "run", "--iteration",
                                                         str(KEPT_ITERATION)])
    assert vars(parsed_arguments) == {"experiment_folder": Path("run"), "iteration": KEPT_ITERATION,
                                      "class_display_names": None, "group_manifest_path": None,
                                      "training_labels_path": None, "predictions_path": None,
                                      **DEFAULT_REPORT_SETTINGS}
    assert set(vars(parsed_arguments)) == set(inspect.signature(draw_report_figures).parameters) - {"font_family"}


def test_report_arguments_reach_every_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that every setting of draw_report_figures but the font can be given on the command line of the script:
    the names shown for the classes in the order given, a name holding '=' kept whole, and paths starting with a '~'
    the shell leaves after '=' expanded to the home folder.

    Args:
        tmp_path (Path): The pytest temporary directory, standing for the home folder.
        monkeypatch (pytest.MonkeyPatch): Sets the home folder.
    """
    monkeypatch.setenv(name="HOME", value=str(tmp_path))
    parsed_arguments = parse_report_arguments(arguments=[
        "--experiment_folder=~/run", "--iteration", str(KEPT_ITERATION), "--weights", "ema",
        "--evaluation_folder_name", "Evaluation", "--class_display_names", "dog=Dog", "cat=Cat = Feline",
        "--sample_kind", "frames", "--split_name", "validation", "--minimum_recall", "0.98",
        "--group_manifest_path=~/frames_manifest.jsonl", "--file_name_key", "image", "--group_key", "truck",
        "--group_plural", "trucks", "--training_labels_path=~/labels.json", "--predictions_path=~/predictions.csv"])
    assert vars(parsed_arguments) == {"experiment_folder": tmp_path / "run", "iteration": KEPT_ITERATION,
                                      "weights": "ema", "evaluation_folder_name": "Evaluation",
                                      "class_display_names": {"dog": "Dog", "cat": "Cat = Feline"},
                                      "sample_kind": "frames", "split_name": "validation", "minimum_recall": 0.98,
                                      "group_manifest_path": tmp_path / "frames_manifest.jsonl",
                                      "file_name_key": "image", "group_key": "truck", "group_plural": "trucks",
                                      "training_labels_path": tmp_path / "labels.json",
                                      "predictions_path": tmp_path / "predictions.csv"}
    assert list(parsed_arguments.class_display_names) == ["dog", "cat"]


def test_parsed_report_arguments_draw_the_figures(tmp_path: Path) -> None:
    """
    Verifies that the arguments read from a command line draw the figures of a run as the main block passes them, the
    averaged weights of the kept checkpoint and the groups chart of its trucks.

    Args:
        tmp_path (Path): The temporary folder of the test, holding the fake run.
    """
    fake_run = write_fake_run(run_folder=tmp_path / "run", class_names=THREE_CLASS_NAMES, weights="ema",
                              saved_temperature=SAVED_TEMPERATURE, averaged_curve=AVERAGED_CURVE)
    parsed_arguments = parse_report_arguments(arguments=[
        "--experiment_folder", str(tmp_path / "run"), "--iteration", str(KEPT_ITERATION), "--weights", "ema",
        "--split_name", "validation", "--group_manifest_path", str(fake_run["manifest_path"]), "--group_key",
        "truck", "--group_plural", "trucks"])

    output_folder = draw_report_figures(**vars(parsed_arguments), font_family=None)

    report_values = json.loads(s=(output_folder / REPORT_VALUES_FILE_NAME).read_text())
    assert output_folder == fake_run["evaluation_folder"] / REPORT_FIGURES_FOLDER_NAME
    assert (report_values["weights"], report_values["logit_temperature"]) == ("ema", SAVED_TEMPERATURE)
    assert report_values["groups"]["never_seen_groups"] == ["t5", "t6"]


@pytest.mark.parametrize(argnames="arguments", argvalues=[
    ["--iteration", str(KEPT_ITERATION)],
    ["--experiment_folder", "run"],
    ["--experiment_folder", "run", "--iteration", "last"],
    ["--experiment_folder", "run", "--iteration", str(KEPT_ITERATION), "--weights", "averaged"],
    ["--experiment_folder", "run", "--iteration", str(KEPT_ITERATION), "--evaluation_folder_name", "Results"],
    ["--experiment_folder", "run", "--iteration", str(KEPT_ITERATION), "--minimum_recall", "high"],
    ["--experiment_folder", "run", "--iteration", str(KEPT_ITERATION), "--class_display_names", "cat"],
    ["--experiment_folder", "run", "--iteration", str(KEPT_ITERATION), "--class_display_names", "cat=Cat",
     "cat=Feline"]])
def test_report_script_refuses_missing_settings_or_unknown_values(arguments: List[str],
                                                                  capsys: pytest.CaptureFixture) -> None:
    """
    Verifies that the script stops with its usage and the error, before anything is read, when the run folder or the
    iteration is missing, a number, weights or evaluation folder is not one the figures accept, or a class name is
    not given as CLASS=NAME or is given twice.

    Args:
        arguments (List[str]): The command line arguments.
        capsys (pytest.CaptureFixture): The captured console output.
    """
    with pytest.raises(expected_exception=SystemExit) as exit_information:
        parse_report_arguments(arguments=arguments)
    assert exit_information.value.code == 2
    assert "error:" in capsys.readouterr().err
