"""
Figures and facts of the object detection series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series object_detection) at every build, from the DET run of
tutorials/tutorial_sources.yaml (the shipped synthetic configuration) and the code of the repository: the COCO manifest
and anchor recall of its data, the trained detector of its last checkpoint run on the CPU on the test image of the
series (anchors, proposals in the proposal selection of the run, anchor labels, training targets), its training losses
and console summaries, and its evaluation (sample images at three iterations, mAP of the trained and averaged weights,
average precision of every class, confusion matrix, report tables, proposal recall), the settings and iterations of
the run quoted next to its measurements (read from its saved configuration, not from the shipped one, nor typed), and
the joint and per-level proposal selections compared by stage 7
of the detection benchmark (proposal_selection.py). Every figure family is a module of this folder writing into a
subfolder of the output folder, its facts gathered here into the facts.json the storyboards quote as
{{object_detection.<name>}}. The instance segmentation series imports the run figures of this folder for its own run.
"""

import fractions
import shutil

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts, read_run_configuration
from tutorials.figures.object_detection.dataset_figures import make_dataset_figures
from tutorials.figures.object_detection.detection_sources import SAMPLE_IMAGE_NAME, count_scalar_tags
from tutorials.figures.object_detection.detector_inference import make_detector_inference_figures
from tutorials.figures.object_detection.evaluation_figures import (draw_class_average_precision_figure,
                                                                   draw_confusion_matrix_figure, read_confusion_notes,
                                                                   read_proposal_recall, write_report_listings)
from tutorials.figures.object_detection.proposal_selection import make_proposal_selection_facts
from tutorials.figures.object_detection.run_figures import (draw_loss_curves_figure, draw_metric_iterations_figure,
                                                            draw_sample_iterations_figure,
                                                            read_console_summary_facts,
                                                            write_console_summary_listing)
from vision_training.losses.detection_losses import LOCALISATION_LOSS_BETA

# Console log of the training of the run (its output redirected to a file), next to the run folder
CONSOLE_LOG_NAME = "object_detection.log"

# Copy of the sample image of the test image at the last checkpoint, in the evaluation folder of the figures
LAST_SAMPLE_NAME = "last_sample.png"

# Losses of the run in the panels of the loss figure: the region proposal network, then the detection heads
LOSS_PANELS = [("Region proposal losses", [("RPN Classification Loss", "rpn_classification"),
                                           ("RPN Regression Loss", "rpn_regression")]),
               ("Detection head losses", [("Detection Classification Loss", "detection_classification"),
                                          ("Detection Regression Loss", "detection_regression")]),
               ("Weighted sum", [("Total Loss", "total")])]

# Headline metrics of the evaluated checkpoints
METRIC_PANELS = [("mAP@0.5/area", "mAP@0.5, area method", "map_50"),
                 ("mAP@[0.5:0.95]/area", "mAP@[0.5:0.95], area method", "map_50_95")]

# Largest denominator tried when the smooth L1 beta of the box losses is written as a fraction
BETA_MAXIMUM_DENOMINATOR = 100


def describe_box_loss_beta() -> Dict[str, Any]:
    """
    Gives the smooth L1 beta of the box losses of both stages (LOCALISATION_LOSS_BETA of
    vision_training/losses/detection_losses.py, written 1 / 9 there, an expression a {{constant:...}} placeholder
    cannot read) as the fraction the recap card shows and the denominator the narration reads ("one over nine").

    Returns:
        Dict[str, Any]: The fraction text and its denominator, named losses.box_loss_beta_*.

    Raises:
        ValueError: If the beta is not one over a whole number, which the narration could not read as written.
    """
    beta_fraction = fractions.Fraction(LOCALISATION_LOSS_BETA).limit_denominator(BETA_MAXIMUM_DENOMINATOR)
    if beta_fraction.numerator != 1 or abs(float(beta_fraction) - LOCALISATION_LOSS_BETA) > 1e-12:
        raise ValueError(f"LOCALISATION_LOSS_BETA {LOCALISATION_LOSS_BETA} is not one over a whole number: the "
                         f"training chapter of the object detection series reads it as one over its denominator.")
    return {"losses.box_loss_beta_fraction": f"{beta_fraction.numerator}/{beta_fraction.denominator}",
            "losses.box_loss_beta_denominator": beta_fraction.denominator}


def describe_run_settings(run_configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the settings of the DET run that the chapters quote next to its measurements (the detections of its sample
    images and of the test image, the labels of its anchors and the targets of its proposals, its proposal filter, its
    losses), read from the configuration saved in the run folder rather than from the shipped configuration, so that
    they stay those of the run when the shipped values move on: the deployment score threshold and final suppression,
    the overlap thresholds and sampling of the anchor labels, the neck, the proposal selection and the counts and
    suppression of the proposal filter, the sampling of the proposals of the detection heads, the learning rate, the
    iterations, the evaluation interval and the sample images of every evaluation.

    Args:
        run_configuration (Dict[str, Any]): The configuration saved in the run folder (read_run_configuration).

    Returns:
        Dict[str, Any]: The settings, named run.*.
    """
    # Sections of the saved configuration holding the settings quoted with the measurements of the run
    experiment_configuration = run_configuration["ExperimentConfiguration"]
    region_proposal_configuration = run_configuration["RegionProposal"]
    detector_configuration = run_configuration["Detector"]
    return {"run.number_iterations": experiment_configuration["number_iterations"],
            "run.evaluation_interval": experiment_configuration["weight_saving_iterations"],
            "run.learning_rate": experiment_configuration["learning_rate"],
            "run.number_sample_images": experiment_configuration["evaluation"]["number_sample_images"],
            "run.neck_type": run_configuration["Neck"]["type"],
            "run.neck_output_channels": run_configuration["Neck"]["output_channels"],
            "run.score_threshold": detector_configuration["inference"]["score_threshold"],
            "run.detector_nms_threshold": detector_configuration["nms_iou_threshold"],
            "run.nms_criterion": detector_configuration["nms_criterion"],
            "run.nms_distance_exponent": detector_configuration["nms_distance_exponent"],
            "run.anchor_foreground_minimum": region_proposal_configuration["foreground_iou_threshold"]["min"],
            "run.anchor_background_minimum": region_proposal_configuration["background_iou_threshold"]["min"],
            "run.anchor_background_maximum": region_proposal_configuration["background_iou_threshold"]["max"],
            "run.anchor_training_samples": region_proposal_configuration["total_training_samples"],
            "run.anchor_training_positives": region_proposal_configuration["number_training_positives"],
            "run.proposal_selection": region_proposal_configuration["proposal_selection"],
            "run.proposal_nms_threshold": region_proposal_configuration["nms_iou_threshold"],
            "run.training_pre_nms_proposals": region_proposal_configuration["training"]["pre_nms_proposals"],
            "run.training_post_nms_proposals": region_proposal_configuration["training"]["post_nms_proposals"],
            "run.inference_pre_nms_proposals": region_proposal_configuration["inference"]["pre_nms_proposals"],
            "run.inference_post_nms_proposals": region_proposal_configuration["inference"]["post_nms_proposals"],
            "run.detector_foreground_minimum": detector_configuration["foreground_iou_threshold"],
            "run.detector_training_samples": detector_configuration["total_training_samples"],
            "run.detector_training_positives": detector_configuration["number_training_positives"]}


def find_run_iterations(run_configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the iterations of the DET run that the figures read, from its saved configuration rather than typed, so that
    a run trained again for another number of iterations only changes its folder in tutorial_sources.yaml: the last
    checkpoint (number_iterations), whose weights the inference figures run and whose evaluation the report, class and
    confusion figures read; the three evaluated iterations of the sample images (the first evaluation, the middle of
    the run and its end), whose confusion matrix notes are quoted too; and the iterations of the console summaries
    (every information_dump iterations).

    Args:
        run_configuration (Dict[str, Any]): The configuration saved in the run folder (read_run_configuration).

    Returns:
        Dict[str, Any]: The 'checkpoint_iteration', the 'sample_iterations' and the 'console_iterations'.

    Raises:
        ValueError: If a sample iteration is not an evaluated one (a multiple of weight_saving_iterations), or the run
            prints no console summary.
    """
    # Iterations of the run, its evaluations every weight_saving_iterations
    experiment_configuration = run_configuration["ExperimentConfiguration"]
    number_iterations = experiment_configuration["number_iterations"]
    evaluation_interval = experiment_configuration["weight_saving_iterations"]
    sample_iterations = [evaluation_interval, number_iterations // 2, number_iterations]
    if any(iteration % evaluation_interval != 0 for iteration in sample_iterations):
        raise ValueError(f"The sample iterations {sample_iterations} of the DET run are not all evaluated ones, "
                         f"multiples of weight_saving_iterations {evaluation_interval}.")

    # Console summaries printed every information_dump iterations
    information_dump = experiment_configuration["information_dump"]
    console_iterations = list(range(information_dump, number_iterations + 1, information_dump))
    if not console_iterations:
        raise ValueError(f"The DET run of {number_iterations} iterations prints no console summary every "
                         f"{information_dump} iterations.")
    return {"checkpoint_iteration": number_iterations, "sample_iterations": sample_iterations,
            "console_iterations": console_iterations}


def describe_iteration_aliases(facts: Dict[str, Any], iterations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the facts named by an iteration (samples.metric_<n>, outcomes.<n>.right, metrics.<metric>.test_<n>) again
    under names that do not hold it, as lists in the order of the sample iterations or as the value of the last
    checkpoint, so that the storyboards stay true when the run is trained again for another number of iterations.

    Args:
        facts (Dict[str, Any]): The merged facts of the figure families.
        iterations (Dict[str, Any]): The iterations of the run (find_run_iterations).

    Returns:
        Dict[str, Any]: The facts samples.metrics, samples.right_detections, metrics.<metric>.test_last and
            metrics.<metric>.test_ema_last.
    """
    sample_iterations = iterations["sample_iterations"]
    aliases = {"samples.metrics": [facts[f"samples.metric_{iteration}"] for iteration in sample_iterations],
               "samples.right_detections": [facts[f"outcomes.{iteration}.right"] for iteration in sample_iterations]}
    for _, _, short_name in METRIC_PANELS:
        for folder in ["test", "test_ema"]:
            aliases[f"metrics.{short_name}.{folder}_last"] = facts[
                f"metrics.{short_name}.{folder}_{iterations['checkpoint_iteration']}"]
    return aliases


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the object detection series into its output folder and gives their facts, the checkpoint
    inference being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the expensive figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.
    """
    # Run folder, the settings it was trained and evaluated with, its iterations and its last evaluated checkpoint
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="object_detection")
    run_configuration = read_run_configuration(run_folder=run_folder)
    run_settings = describe_run_settings(run_configuration=run_configuration)
    iterations = find_run_iterations(run_configuration=run_configuration)
    checkpoint_iteration = iterations["checkpoint_iteration"]
    evaluation_folder = run_folder / "Evaluation" / f"Iteration_{checkpoint_iteration}"
    listings_folder = output_folder / "listings"

    # Sample image of the test image at the last checkpoint, as the evaluation wrote it, shown by the first chapter
    sample_path = evaluation_folder / "Samples" / SAMPLE_IMAGE_NAME
    if not sample_path.exists():
        raise FileNotFoundError(f"No sample image {sample_path} of the last checkpoint of the DET run.")
    (output_folder / "evaluation").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src=sample_path, dst=output_folder / "evaluation" / LAST_SAMPLE_NAME)

    # Console summaries of the run, read from the console log next to the run folder
    console_facts = {}
    for iteration in iterations["console_iterations"]:
        summary_lines = write_console_summary_listing(log_path=run_folder.parent / CONSOLE_LOG_NAME,
                                                      iteration=iteration,
                                                      output_path=listings_folder / f"console_{iteration}.txt")
        console_facts.update(read_console_summary_facts(summary_lines=summary_lines,
                                                        fact_prefix=f"console.{iteration}"))
    facts = merge_facts(fact_groups=[
        console_facts,
        read_confusion_notes(run_folder=run_folder, iterations=iterations["sample_iterations"]),
        make_dataset_figures(output_folder=output_folder, run_folder=run_folder),
        make_detector_inference_figures(output_folder=output_folder, run_folder=run_folder,
                                        iteration=checkpoint_iteration, force=force),
        draw_loss_curves_figure(run_folder=run_folder, panels=LOSS_PANELS,
                                output_path=output_folder / "training" / "loss_curves.png", fact_prefix="losses"),
        draw_sample_iterations_figure(run_folder=run_folder, image_name=SAMPLE_IMAGE_NAME,
                                      iterations=iterations["sample_iterations"], metric_tag="mAP@0.5/area",
                                      metric_label="mAP@0.5",
                                      output_path=output_folder / "evaluation" / "sample_iterations.png",
                                      fact_prefix="samples"),
        draw_metric_iterations_figure(run_folder=run_folder, panels=METRIC_PANELS,
                                      output_path=output_folder / "evaluation" / "map_iterations.png",
                                      fact_prefix="metrics"),
        draw_class_average_precision_figure(evaluation_folder=evaluation_folder,
                                            output_path=output_folder / "evaluation" / "class_average_precision.png"),
        draw_confusion_matrix_figure(evaluation_folder=evaluation_folder,
                                     score_threshold=run_settings["run.score_threshold"],
                                     output_path=output_folder / "evaluation" / "confusion_matrix.png"),
        write_report_listings(run_folder=run_folder, iteration=checkpoint_iteration, output_folder=listings_folder),
        read_proposal_recall(evaluation_folder=evaluation_folder),
        make_proposal_selection_facts(experiments_folder=experiments_folder),
        describe_box_loss_beta(), run_settings,
        {"run.test_scalar_count": count_scalar_tags(log_folder=run_folder / "Tensorboard" / "Test"),
         "run.checkpoint_iteration": checkpoint_iteration,
         "samples.iterations": iterations["sample_iterations"], "console.iterations": iterations["console_iterations"]}])

    # Facts named by an iteration given again under names that hold none
    return merge_facts(fact_groups=[facts, describe_iteration_aliases(facts=facts, iterations=iterations)])
