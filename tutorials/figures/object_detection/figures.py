"""
Figures and facts of the object detection series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series object_detection) at every build, from the DET run of
tutorials/tutorial_sources.yaml (the synthetic configuration resumed to 5000 iterations) and the code of the repository:
the COCO manifest and anchor recall of its data, the trained detector run on the CPU on the test image of the series
(anchors, proposals, anchor labels, training targets), its training losses and console summaries, and its evaluation
(sample images at three iterations, mAP of the trained and averaged weights, average precision of every class,
confusion matrix, report tables, proposal recall). Every figure family is a module of this folder writing into a
subfolder of the output folder, its facts gathered here into the facts.json the storyboards quote as
{{object_detection.<name>}}. The instance segmentation series imports the run figures of this folder for its own run.
"""

import fractions

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts, read_run_configuration
from tutorials.figures.object_detection.dataset_figures import make_dataset_figures
from tutorials.figures.object_detection.detection_sources import (CHECKPOINT_ITERATION, SAMPLE_IMAGE_NAME,
                                                                  SAMPLE_ITERATIONS, count_scalar_tags)
from tutorials.figures.object_detection.detector_inference import make_detector_inference_figures
from tutorials.figures.object_detection.evaluation_figures import (draw_class_average_precision_figure,
                                                                   draw_confusion_matrix_figure, read_confusion_notes,
                                                                   read_proposal_recall, write_report_listings)
from tutorials.figures.object_detection.run_figures import (draw_loss_curves_figure, draw_metric_iterations_figure,
                                                            draw_sample_iterations_figure,
                                                            read_console_summary_facts,
                                                            write_console_summary_listing)
from vision_training.losses.detection_losses import LOCALISATION_LOSS_BETA

# Console log of the first 2000 iterations of the run, written by the README screenshot launch, relative to the folder
# of the run, and the iterations whose summaries the training chapter shows
CONSOLE_LOG_NAME = "object_detection.log"
CONSOLE_SUMMARY_ITERATIONS = [1000, 2000]

# Evaluated iterations of the run, whose confusion matrix notes the evaluation chapter quotes
EVALUATED_ITERATIONS = [1000, 2000, 2500, 5000]

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
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="object_detection")
    evaluation_folder = run_folder / "Evaluation" / f"Iteration_{CHECKPOINT_ITERATION}"
    listings_folder = output_folder / "listings"

    # Console summaries of the first evaluated iterations, read from the console log next to the run folder
    console_facts = {}
    for iteration in CONSOLE_SUMMARY_ITERATIONS:
        summary_lines = write_console_summary_listing(log_path=run_folder.parent / CONSOLE_LOG_NAME,
                                                      iteration=iteration,
                                                      output_path=listings_folder / f"console_{iteration}.txt")
        console_facts.update(read_console_summary_facts(summary_lines=summary_lines,
                                                        fact_prefix=f"console.{iteration}"))
    return merge_facts(fact_groups=[
        console_facts,
        read_confusion_notes(run_folder=run_folder, iterations=EVALUATED_ITERATIONS),
        make_dataset_figures(output_folder=output_folder, run_folder=run_folder),
        make_detector_inference_figures(output_folder=output_folder, run_folder=run_folder, force=force),
        draw_loss_curves_figure(run_folder=run_folder, panels=LOSS_PANELS,
                                output_path=output_folder / "training" / "loss_curves.png", fact_prefix="losses"),
        draw_sample_iterations_figure(run_folder=run_folder, image_name=SAMPLE_IMAGE_NAME,
                                      iterations=SAMPLE_ITERATIONS, metric_tag="mAP@0.5/area",
                                      metric_label="mAP@0.5",
                                      output_path=output_folder / "evaluation" / "sample_iterations.png",
                                      fact_prefix="samples"),
        draw_metric_iterations_figure(run_folder=run_folder, panels=METRIC_PANELS,
                                      output_path=output_folder / "evaluation" / "map_iterations.png",
                                      fact_prefix="metrics"),
        draw_class_average_precision_figure(evaluation_folder=evaluation_folder,
                                            output_path=output_folder / "evaluation" / "class_average_precision.png"),
        draw_confusion_matrix_figure(evaluation_folder=evaluation_folder,
                                     output_path=output_folder / "evaluation" / "confusion_matrix.png"),
        write_report_listings(run_folder=run_folder, iteration=CHECKPOINT_ITERATION, output_folder=listings_folder),
        read_proposal_recall(evaluation_folder=evaluation_folder),
        describe_box_loss_beta(),
        {"run.test_scalar_count": count_scalar_tags(log_folder=run_folder / "Tensorboard" / "Test"),
         "run.checkpoint_iteration": CHECKPOINT_ITERATION,
         "run.number_iterations": read_run_configuration(run_folder=run_folder)["ExperimentConfiguration"][
             "number_iterations"],
         "samples.iterations": SAMPLE_ITERATIONS, "console.iterations": CONSOLE_SUMMARY_ITERATIONS}])
