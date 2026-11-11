"""
Evaluation figures and facts of the sequence labelling tutorials (the chapters on the model and on reading an
evaluation), made at every build by tutorials/figures/sequence_labelling/figures.py from the real sequence labelling
run: the listing of the evaluation block of its last iteration in Results.txt, the probability of no_change in its
predictions.csv on the first and on the later frames and by frame position (the signature of a model that learned the
position of a frame rather than its content), the macro ROC AUC computed again on the later frames only, the confidence
thresholds of class_thresholds.json, the per-class precision and recall of class_metrics.csv, the segment metrics of
the example of the tasks reference (documentation/reference/tasks_reference.md, formerly in the README) computed by the
code of the repository, and the shapes, residual blocks and parameter counts of the trained model run on the CPU on a
real padded batch.
"""

import json
import re

import numpy
import pandas
import torch

from matplotlib import pyplot
from pathlib import Path
from scipy.stats import rankdata
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import (CHART_HEIGHT, CHART_WIDTH, DOTS_PER_INCH, MONOSPACE_FONT_FAMILY,
                                                POINTS_PER_PIXEL, SERIES_COLOURS, SURFACE_COLOUR, TEXT_PRIMARY_COLOUR,
                                                TEXT_SECONDARY_COLOUR, compute_pixel_box, read_reused_facts,
                                                register_fonts, round_fact, save_figure,
                                                style_chart_panel, write_reused_facts)
from tutorials.figures.sequence_labelling.sequence_utilities import (build_test_dataset, collate_named_sequences,
                                                                     count_model_parameters, load_sequence_model,
                                                                     parse_block_metrics, predict_sequences,
                                                                     read_complete_reused_facts, read_results_block,
                                                                     read_run_sections, write_listing)
from vision_training.evaluation.sequence.segment_metrics import compute_segment_metrics
from vision_training.models.sequence.frame_encoder import encode_frames

# Iteration of the evaluation read by the report scenes, the last one of the run
EVALUATION_ITERATION = 2000

# Number of change classes rows of the per-class table kept in the listing, before the no_change row
LISTED_CHANGE_ROWS = 2

# Lines of the evaluation block kept before its per-class table: the header, the metrics and the notes
HEADLINE_LINE_COUNT = 9

# Characters kept of every listed line, the header line ending with a rule as wide as the console
LISTED_LINE_LENGTH = 118

# Example of the segment metrics of documentation/reference/tasks_reference.md (Sequence Labelling), formerly in the
# README, hence the names: a true sequence A A A A A B B B B B predicted A A B A A B B A B B, the classes A and B
# written as 0 and 1
README_TRUE_CLASSES = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
README_PREDICTED_CLASSES = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

# Bins of the histogram of the probability of no_change
PROBABILITY_BIN_WIDTH = 0.005

# Note of the segments of an evaluation block, "Segments : 58133 true and 10000 predicted in ..."
SEGMENT_NOTE_PATTERN = re.compile(pattern=r"Segments : (\d+) true and (\d+) predicted")

# Facts of the model a reused model_facts.json must hold, those added after its first version
REQUIRED_MODEL_FACTS = ["model.backbone_channels", "model.backbone_map_size", "model.flatten_features",
                        "model.temporal_features", "model.residual_blocks", "model.residual_block_count"]


def write_results_listing(run_folder: Path, listing_path: Path) -> Dict[str, Any]:
    """
    Writes the listing of the report scene: the evaluation block of the last iteration of Results.txt, its headline
    metrics and notes, then the head of its per-class table with its first change classes and the no_change row.

    Args:
        run_folder (Path): The folder of the sequence labelling run.
        listing_path (Path): The text file of the listing.

    Returns:
        Dict[str, Any]: The number of lines of the listing, named evaluation.results_listing_lines.
    """
    block_lines = read_results_block(results_path=run_folder / "Results.txt", iteration=EVALUATION_ITERATION)
    table_start = next(line_index for line_index, line in enumerate(block_lines) if "Per Class Metrics" in line)
    no_change_line = next(line for line in block_lines[table_start:] if line.strip().startswith("no_change "))
    listed_lines = block_lines[:HEADLINE_LINE_COUNT] + [""] + block_lines[table_start:table_start + 4] + \
        block_lines[table_start + 4:table_start + 4 + LISTED_CHANGE_ROWS] + ["   ..."] + [no_change_line]
    write_listing(lines=[line[:LISTED_LINE_LENGTH] for line in listed_lines], listing_path=listing_path)

    # Headline metrics of the block and its counts of segments, as facts
    facts = {"evaluation.results_listing_lines": len(listed_lines)}
    facts.update(parse_block_metrics(block_lines=block_lines, fact_prefix="evaluation"))
    segment_match = SEGMENT_NOTE_PATTERN.search(string="\n".join(block_lines))
    facts["evaluation.true_segments"] = int(segment_match.group(1))
    facts["evaluation.predicted_segments"] = int(segment_match.group(2))
    return facts


def count_sequences_without_change(run_folder: Path) -> Dict[str, Any]:
    """
    Counts the test sequences of the run whose frames are all no_change, the sequences a model giving no_change to
    every frame gets fully right, so that the sequence exact match of a degenerate solution can be read against it.

    Args:
        run_folder (Path): The folder of the sequence labelling run.

    Returns:
        Dict[str, Any]: The facts, named evaluation.sequences_without_change and evaluation.test_sequences.
    """
    experiment_configuration, _ = read_run_sections(run_folder=run_folder)
    test_labels = json.loads((Path(experiment_configuration["test_data_folder"]) / "labels.json").read_text(
        encoding="utf-8"))
    no_change_category = next(category["id"] for category in test_labels["categories"]
                              if category["name"] == "no_change")
    return {"evaluation.sequences_without_change": sum(all(category == no_change_category
                                                           for category in sequence["frame_category_ids"])
                                                       for sequence in test_labels["sequences"]),
            "evaluation.test_sequences": len(test_labels["sequences"])}


def compute_rank_roc_auc(scores: numpy.ndarray, positives: numpy.ndarray) -> float:
    """
    Computes the area under the ROC curve of one class from the ranks of its scores (the Mann-Whitney statistic,
    ties sharing their mean rank), to check the macro ROC AUC of the report on a subset of the frames.

    Args:
        scores (numpy.ndarray): The probability of the class for every frame.
        positives (numpy.ndarray): Whether every frame is of the class.

    Returns:
        float: The ROC AUC.
    """
    ranks = rankdata(scores)
    positive_count = int(positives.sum())
    negative_count = len(positives) - positive_count
    return float((ranks[positives].sum() - positive_count * (positive_count + 1) / 2) /
                 (positive_count * negative_count))


def compute_macro_roc_auc(predictions: pandas.DataFrame, class_names: List[str],
                          frame_selection: numpy.ndarray) -> float:
    """
    Averages the ROC AUC of every class over the selected frames of predictions.csv, a class needing frames of its
    own and of the others among them, as the macro ROC AUC of the report does on all the frames.

    Args:
        predictions (pandas.DataFrame): The rows of predictions.csv.
        class_names (List[str]): The names of the classes, the probability columns being probability_<name>.
        frame_selection (numpy.ndarray): Whether every frame is kept.

    Returns:
        float: The macro ROC AUC of the kept frames.
    """
    true_classes = predictions["true_class"].to_numpy()[frame_selection]
    class_areas = []
    for class_name in class_names:
        positives = true_classes == class_name
        if 0 < positives.sum() < len(positives):
            class_areas.append(compute_rank_roc_auc(
                scores=predictions[f"probability_{class_name}"].to_numpy()[frame_selection], positives=positives))
    return float(numpy.mean(class_areas))


def draw_no_change_probabilities(predictions: pandas.DataFrame, frame_positions: numpy.ndarray,
                                 output_path: Path) -> Dict[str, Any]:
    """
    Draws the probability of no_change given by the run to every labelled test frame: its histogram on the first
    frames and on the later frames (left), and its mean at every frame position next to the true share of no_change
    at that position (right), and gives the boxes of the two panels.

    Args:
        predictions (pandas.DataFrame): The rows of predictions.csv.
        frame_positions (numpy.ndarray): The position of every labelled frame in its sequence.
        output_path (Path): The PNG file.

    Returns:
        Dict[str, Any]: The boxes of the panels, named predictions.*_panel_box.
    """
    no_change_probabilities = predictions["probability_no_change"].to_numpy()
    is_no_change = (predictions["true_class"] == "no_change").to_numpy()
    first_frames = frame_positions == 0
    register_fonts()
    figure, (histogram_axes, position_axes) = pyplot.subplots(nrows=1, ncols=2, figsize=(
        CHART_WIDTH / DOTS_PER_INCH, CHART_HEIGHT / DOTS_PER_INCH), dpi=DOTS_PER_INCH,
        gridspec_kw={"width_ratios": [1.3, 1.0]})
    figure.patch.set_facecolor(SURFACE_COLOUR)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.13, wspace=0.2)
    figure.text(x=0.012, y=0.955, s=f"Evaluation/Iteration_{EVALUATION_ITERATION}/predictions.csv · "
                                    f"{len(predictions):,} labelled frames of test_10000",
                color=TEXT_SECONDARY_COLOUR, fontsize=16 * POINTS_PER_PIXEL, va="center", family=MONOSPACE_FONT_FAMILY)

    # Histogram of the probability of no_change, first frames and later frames, on a logarithmic count axis
    bins = numpy.arange(0.3, 1.0 + PROBABILITY_BIN_WIDTH, PROBABILITY_BIN_WIDTH)
    histogram_axes.hist(x=no_change_probabilities[~first_frames], bins=bins, color=SERIES_COLOURS[0],
                        label=f"later frames ({int((~first_frames).sum()):,})")
    histogram_axes.hist(x=no_change_probabilities[first_frames], bins=bins, color=SERIES_COLOURS[2],
                        label=f"first frames ({int(first_frames.sum()):,})")
    style_chart_panel(axes=histogram_axes, title="Probability of no_change given to every frame",
                      x_label="probability_no_change", y_label="Frames")
    histogram_axes.set_yscale(value="log")
    histogram_axes.set_xlim(left=0.3, right=1.02)
    histogram_axes.legend(loc="upper center", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                          fontsize=18 * POINTS_PER_PIXEL)

    # Mean probability of no_change at every position against the true share of no_change there
    positions = numpy.arange(int(frame_positions.max()) + 1)
    mean_probabilities = [no_change_probabilities[frame_positions == position].mean() for position in positions]
    true_shares = [is_no_change[frame_positions == position].mean() for position in positions]
    position_axes.plot(positions, true_shares, color=SERIES_COLOURS[1], linewidth=3 * POINTS_PER_PIXEL,
                       linestyle=(0, (6, 5)), marker="o", markersize=7 * POINTS_PER_PIXEL,
                       label="true share of no_change")
    position_axes.plot(positions, mean_probabilities, color=SERIES_COLOURS[0], linewidth=3 * POINTS_PER_PIXEL,
                       marker="o", markersize=7 * POINTS_PER_PIXEL, label="mean probability_no_change")
    style_chart_panel(axes=position_axes, title="By frame position", x_label="Frame position",
                      y_label="Probability or share")
    position_axes.set_ylim(bottom=0.0, top=1.08)
    position_axes.set_xticks(ticks=positions)
    position_axes.legend(loc="upper right", frameon=False, labelcolor=TEXT_PRIMARY_COLOUR,
                         fontsize=18 * POINTS_PER_PIXEL)
    save_figure(figure=figure, output_path=output_path)
    facts = {"predictions.histogram_panel_box": compute_pixel_box(figure=figure, axes=histogram_axes,
                                                                 with_labels=True, margin=0),
             "predictions.position_panel_box": compute_pixel_box(figure=figure, axes=position_axes,
                                                                with_labels=True, margin=0),
             "predictions.mean_by_position": [round_fact(value=value) for value in mean_probabilities],
             "predictions.true_share_by_position": [round_fact(value=value) for value in true_shares]}
    pyplot.close(fig=figure)
    return facts


def make_prediction_facts(run_folder: Path, output_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Reads predictions.csv of the last evaluation (69,918 rows of 71 probabilities), draws the probability of no_change
    and gives the facts of the degenerate solution: the probabilities on the first and later frames, the predicted
    classes, and the macro ROC AUC on all the frames and on the later frames only. Reused from the previous build
    unless forced.

    Args:
        run_folder (Path): The folder of the sequence labelling run.
        output_folder (Path): The figures folder of the series.
        force (bool): Whether the reused figure is made again.

    Returns:
        Dict[str, Any]: The facts, named predictions.*.
    """
    figure_path = output_folder / "evaluation" / "no_change_probabilities.png"
    reused_facts_path = output_folder / "evaluation" / "no_change_probabilities_facts.json"
    reused_facts = read_reused_facts(facts_path=reused_facts_path, required_paths=[], force=force)
    if reused_facts is not None:
        return reused_facts
    predictions = pandas.read_csv(filepath_or_buffer=run_folder / "Evaluation" / f"Iteration_{EVALUATION_ITERATION}" /
                                  "predictions.csv")
    frame_positions = predictions["frame"].str.extract(pat=r"/(\d+)\.png$")[0].astype(int).to_numpy()
    first_frames = frame_positions == 0
    no_change_probabilities = predictions["probability_no_change"].to_numpy()
    class_names = [column[len("probability_"):] for column in predictions.columns if column.startswith("probability_")]
    facts = draw_no_change_probabilities(predictions=predictions, frame_positions=frame_positions,
                                         output_path=figure_path)

    # Probabilities of no_change on the first and on the later frames, and the classes predicted
    later_no_change = (predictions["true_class"].to_numpy()[~first_frames] == "no_change")
    facts.update({"predictions.frame_count": len(predictions), "predictions.first_frame_count": int(first_frames.sum()),
                  "predictions.later_frame_count": int((~first_frames).sum()),
                  "predictions.first_minimum": round_fact(value=no_change_probabilities[first_frames].min()),
                  "predictions.first_mean": round_fact(value=no_change_probabilities[first_frames].mean()),
                  "predictions.later_mean": round_fact(value=no_change_probabilities[~first_frames].mean()),
                  "predictions.later_minimum": round_fact(value=no_change_probabilities[~first_frames].min()),
                  "predictions.later_maximum": round_fact(value=no_change_probabilities[~first_frames].max()),
                  "predictions.later_true_no_change_share": round_fact(value=later_no_change.mean()),
                  "predictions.later_true_no_change_count": int(later_no_change.sum()),
                  "predictions.predicted_classes": sorted(predictions["predicted_class"].unique().tolist()),
                  "predictions.macro_roc_auc_all_frames": round_fact(value=compute_macro_roc_auc(
                      predictions=predictions, class_names=class_names,
                      frame_selection=numpy.ones(shape=len(predictions), dtype=bool))),
                  "predictions.macro_roc_auc_later_frames": round_fact(value=compute_macro_roc_auc(
                      predictions=predictions, class_names=class_names, frame_selection=~first_frames))})
    write_reused_facts(facts_path=reused_facts_path, figure_paths=[figure_path], facts=facts)
    return facts


def compute_segment_facts(run_folder: Path) -> Dict[str, Any]:
    """
    Computes with compute_segment_metrics of the repository the segment metrics of the example of the tasks reference
    (its frame accuracy, edit score and segment F1 at every overlap), and reads the test frames of no_change of the
    run, the accuracy of a model giving no_change to every frame.

    Args:
        run_folder (Path): The folder of the sequence labelling run.

    Returns:
        Dict[str, Any]: The facts, named segments.*.
    """
    true_classes, predicted_classes = numpy.array(README_TRUE_CLASSES), numpy.array(README_PREDICTED_CLASSES)
    metrics = compute_segment_metrics(sequences=[{"frame_indices": numpy.arange(len(true_classes)),
                                                  "true_classes": true_classes,
                                                  "predicted_classes": predicted_classes}])
    facts = {"segments.readme_frame_accuracy": round_fact(value=numpy.mean(true_classes == predicted_classes)),
             "segments.readme_edit_score": round_fact(value=metrics["edit_score"]),
             "segments.readme_true_segments": metrics["number_true_segments"],
             "segments.readme_predicted_segments": metrics["number_predicted_segments"]}
    for overlap_threshold, segment_f1 in metrics["segment_f1"].items():
        facts[f"segments.readme_f1_at_{round(overlap_threshold * 100)}"] = round_fact(value=segment_f1)

    # Test frames of no_change and of every class in the record of the run
    data_record = json.loads((run_folder / "data.json").read_text(encoding="utf-8"))
    no_change_record = next(record for record in data_record["classes"] if record["class_name"] == "no_change")
    facts["segments.no_change_test_frames"] = no_change_record["frames"]["test"]
    facts["segments.test_frames"] = sum(record["frames"]["test"] for record in data_record["classes"])
    return facts


def compute_model_facts(run_folder: Path) -> Dict[str, Any]:
    """
    Rebuilds the trained model of the last checkpoint on the CPU, counts the parameters of its components, and runs
    the first test sequences of the run through it, padded into one batch, to give the shape of every stage: the
    backbone feature maps of the real frames, the pooled frame vectors, the frame vectors in their sequences, the
    output of the temporal module and the class logits, with the channels and side of the last feature map, the
    features a flatten pooling would give, and the backbone blocks holding a residual shortcut.

    Args:
        run_folder (Path): The folder of the sequence labelling run.

    Returns:
        Dict[str, Any]: The facts, named model.*.
    """
    model = load_sequence_model(run_folder=run_folder, iteration=EVALUATION_ITERATION)
    dataset = build_test_dataset(run_folder=run_folder, label_type="frame")
    batch = collate_named_sequences(dataset=dataset, sequence_names=[sequence["sequence_name"]
                                                                     for sequence in dataset.sequences[:4]])
    images = batch["images"].float()
    with torch.no_grad():
        backbone_maps, _ = model.backbone(input_tensor=images[batch["sequence_mask"]])
        pooled_vectors = model.global_pooling(input=backbone_maps)
        frame_features = encode_frames(backbone=model.backbone, global_pooling=model.global_pooling,
                                       input_tensor=images, sequence_mask=batch["sequence_mask"])
        temporal_features = model.temporal_module(frame_features=frame_features, sequence_mask=batch["sequence_mask"])
    outputs = predict_sequences(model=model, batch=batch)
    parameters = count_model_parameters(model=model)
    padding_zero_vectors = (frame_features.abs().sum(dim=2) == 0).sum(dim=1).tolist()

    # Blocks of the backbone closing at least one residual pair, with the condition of ConvolutionBlock.forward: a
    # block whose first layer changes the channel count shifts its pairs by one layer and closes none with two layers
    residual_blocks = [block_index + 1 for block_index, block in enumerate(model.backbone.convolution_blocks)
                       if block.residual and any(layer_index % 2 == 1 - block.residual_parity and layer_index != 0
                                                 for layer_index in range(len(block.layer_module_indices)))]
    return {"model.backbone_channels": int(backbone_maps.shape[1]),
            "model.backbone_map_size": int(backbone_maps.shape[2]),
            "model.flatten_features": int(backbone_maps[0].numel()),
            "model.temporal_features": int(temporal_features.shape[2]),
            "model.residual_blocks": residual_blocks, "model.residual_block_count": len(residual_blocks),
            "model.backbone_parameters": parameters["backbone"],
            "model.temporal_parameters": parameters["temporal_module"],
            "model.head_parameters": parameters["classification_head"], "model.total_parameters": parameters["total"],
            "model.total_megabytes": round_fact(value=parameters["total"] * 4 / 1e6),
            "model.input_shape": str(list(batch["images"].shape)),
            "model.backbone_shape": str(list(backbone_maps.shape)),
            "model.pooled_shape": str(list(pooled_vectors.shape)),
            "model.frame_features_shape": str(list(frame_features.shape)),
            "model.temporal_shape": str(list(temporal_features.shape)),
            "model.logits_shape": str(list(outputs["class_logits"].shape)),
            "model.padding_zero_vectors": padding_zero_vectors, "model.head": str(model.classification_head)}


def read_class_metric_facts(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the per-class metrics of the last evaluation (class_metrics.csv): the precision and recall of no_change and
    the best recall of a change class, which tell a degenerate solution giving no_change to every frame.

    Args:
        run_folder (Path): The folder of the sequence labelling run.

    Returns:
        Dict[str, Any]: The facts, named evaluation.no_change_precision, evaluation.no_change_recall and
            evaluation.change_class_maximum_recall.
    """
    class_metrics = pandas.read_csv(filepath_or_buffer=run_folder / "Evaluation" /
                                    f"Iteration_{EVALUATION_ITERATION}" / "class_metrics.csv")
    is_no_change = (class_metrics["class"] == "no_change").to_numpy()
    return {"evaluation.no_change_precision": round_fact(value=float(class_metrics["precision"][is_no_change].iloc[0])),
            "evaluation.no_change_recall": round_fact(value=float(class_metrics["recall"][is_no_change].iloc[0])),
            "evaluation.change_class_maximum_recall": round_fact(value=float(
                class_metrics["recall"][~is_no_change].max()))}


def make_evaluation_facts(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes the evaluation listing and figure of the sequence labelling series and gives their facts, with those of the
    thresholds file, the per-class metrics, the segment metrics and the model, the model facts being reused from the
    previous build unless forced or unless that build wrote fewer facts than the storyboards quote.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the sequence labelling run.
        force (bool): Whether the reused figures are made again.

    Returns:
        Dict[str, Any]: The facts, named evaluation.*, predictions.*, thresholds.*, segments.* and model.*.
    """
    facts = write_results_listing(run_folder=run_folder, listing_path=output_folder / "listings" /
                                  "results_block.txt")
    facts.update(make_prediction_facts(run_folder=run_folder, output_folder=output_folder, force=force))

    # Confidence thresholds of the last evaluation: no_change, and the classes that never reach the target precision
    thresholds = json.loads((run_folder / "Evaluation" / f"Iteration_{EVALUATION_ITERATION}" /
                             "class_thresholds.json").read_text(encoding="utf-8"))
    facts.update({"thresholds.target_precision": thresholds["target_precision"],
                  "thresholds.no_change_threshold": round_fact(value=thresholds["classes"]["no_change"]["threshold"]),
                  "thresholds.no_change_recall": round_fact(value=thresholds["classes"]["no_change"]["recall"]),
                  "thresholds.null_classes": sum(class_threshold is None
                                                 for class_threshold in thresholds["classes"].values())})
    facts.update(compute_segment_facts(run_folder=run_folder))
    facts.update(count_sequences_without_change(run_folder=run_folder))

    facts.update(read_class_metric_facts(run_folder=run_folder))

    # Shapes and parameters of the trained model, run on the CPU, reused from the previous build unless forced
    model_facts_path = output_folder / "evaluation" / "model_facts.json"
    model_facts = read_complete_reused_facts(facts_path=model_facts_path, required_paths=[],
                                             required_fact_names=REQUIRED_MODEL_FACTS, force=force)
    if model_facts is None:
        model_facts = compute_model_facts(run_folder=run_folder)
        write_reused_facts(facts_path=model_facts_path, figure_paths=[], facts=model_facts)
    facts.update(model_facts)
    facts["evaluation.iteration"] = EVALUATION_ITERATION
    return facts
