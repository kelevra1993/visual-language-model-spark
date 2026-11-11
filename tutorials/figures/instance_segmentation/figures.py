"""
Figures and facts of the instance segmentation series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series instance_segmentation) at every build, from the INS run of
tutorials/tutorial_sources.yaml (the synthetic configuration resumed to 5000 iterations) and the code of the
repository: the run length encoding of the masks and its decoding, the mask target of a real proposal, the masks the
trained instance segmenter predicts on the CPU and their pasting (mask_figures.py), the mask against box average
precision and the report tables (evaluation_figures.py), and, with the run figures of the object detection series
imported from tutorials/figures/object_detection/run_figures.py, the losses, the sample images along the run, the mask
mAP of the trained and averaged weights and the console summaries, with the geometry of the mask heads and the values
the narration derives from them (describe_quoted_facts). The facts are quoted by the storyboards as
{{instance_segmentation.<name>}}.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts
from tutorials.figures.instance_segmentation.evaluation_figures import (draw_mask_box_figure,
                                                                        write_mask_report_listings)
from tutorials.figures.instance_segmentation.mask_figures import describe_mask_geometry, make_mask_figures
from tutorials.figures.object_detection.detection_sources import (CHECKPOINT_ITERATION, SAMPLE_IMAGE_NAME,
                                                                  SAMPLE_ITERATIONS, count_scalar_tags)
from tutorials.figures.object_detection.evaluation_figures import read_confusion_notes, read_proposal_recall
from tutorials.figures.object_detection.run_figures import (draw_loss_curves_figure, draw_metric_iterations_figure,
                                                            draw_sample_iterations_figure,
                                                            read_console_summary_facts,
                                                            write_console_summary_listing)

# Console logs of the run, next to its folder: the first 2000 iterations, then the resume to 5000, and the iteration
# of the summary read from each
CONSOLE_SUMMARIES = [("instance_segmentation.log", 1000), ("instance_segmentation_extended.log", 5000)]

# Losses of the run in the panels of the loss figure: the four losses of the detector, then the mask loss
LOSS_PANELS = [("Detector losses", [("RPN Classification Loss", "rpn_classification"),
                                    ("RPN Regression Loss", "rpn_regression"),
                                    ("Detection Classification Loss", "detection_classification"),
                                    ("Detection Regression Loss", "detection_regression")]),
               ("Mask loss", [("Mask Loss", "mask")])]

# Headline metrics of the evaluated checkpoints, on the masks
METRIC_PANELS = [("Mask mAP@0.5/area", "Mask mAP@0.5, area method", "mask_map_50"),
                 ("Mask mAP@[0.5:0.95]/area", "Mask mAP@[0.5:0.95], area method", "mask_map_50_95")]

# Evaluated iterations of the run, whose confusion matrix notes the chapters quote
EVALUATED_ITERATIONS = [1000, 2000, 2500, 5000]


def describe_quoted_facts(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derives from the facts of the figure families the values the narration quotes in words or as a whole, so that no
    storyboard types them: the iterations of the console summaries and of the sample images (CONSOLE_SUMMARIES and
    SAMPLE_ITERATIONS, the facts named after them), the objects of the test image found by at least one detection (a
    shape found twice counting once), and the detection level whose proposals are the most numerous.

    Args:
        facts (Dict[str, Any]): The merged facts of the figure families of the series.

    Returns:
        Dict[str, Any]: The derived facts.
    """
    # Distinct objects of the test image matched by the detections of the trained segmenter
    detection_objects = {facts[f"predictions.detection_{index + 1}.object"]
                         for index in range(facts["predictions.detection_count"])}

    # Level of the proposal recall table with the most proposals per image
    block_proposals = {int(name.split(".")[1].split("_")[1]): value for name, value in facts.items()
                       if name.startswith("proposal_recall.block_") and name.endswith(".proposals_per_image")}
    return {"console.first_iteration": CONSOLE_SUMMARIES[0][1], "console.last_iteration": CONSOLE_SUMMARIES[-1][1],
            "samples.iterations": list(SAMPLE_ITERATIONS), "predictions.found_object_count": len(detection_objects),
            "proposal_recall.busiest_block": max(block_proposals, key=block_proposals.get)}


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the instance segmentation series into its output folder and gives their facts, the
    checkpoint inference being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the expensive figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.
    """
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="instance_segmentation")
    evaluation_folder = run_folder / "Evaluation" / f"Iteration_{CHECKPOINT_ITERATION}"
    listings_folder = output_folder / "listings"

    # Console summaries of the first evaluation and of the last iteration, read from the logs next to the run folder
    console_facts = {}
    for log_name, iteration in CONSOLE_SUMMARIES:
        summary_lines = write_console_summary_listing(log_path=run_folder.parent / log_name, iteration=iteration,
                                                      output_path=listings_folder / f"console_{iteration}.txt")
        console_facts.update(read_console_summary_facts(summary_lines=summary_lines,
                                                        fact_prefix=f"console.{iteration}"))
    facts = merge_facts(fact_groups=[
        console_facts, describe_mask_geometry(run_folder=run_folder),
        make_mask_figures(output_folder=output_folder, run_folder=run_folder, force=force),
        read_confusion_notes(run_folder=run_folder, iterations=EVALUATED_ITERATIONS),
        draw_loss_curves_figure(run_folder=run_folder, panels=LOSS_PANELS,
                                output_path=output_folder / "training" / "loss_curves.png", fact_prefix="losses"),
        draw_sample_iterations_figure(run_folder=run_folder, image_name=SAMPLE_IMAGE_NAME,
                                      iterations=SAMPLE_ITERATIONS, metric_tag="Mask mAP@0.5/area",
                                      metric_label="mask mAP@0.5",
                                      output_path=output_folder / "evaluation" / "sample_iterations.png",
                                      fact_prefix="samples"),
        draw_metric_iterations_figure(run_folder=run_folder, panels=METRIC_PANELS,
                                      output_path=output_folder / "evaluation" / "mask_map_iterations.png",
                                      fact_prefix="metrics"),
        draw_mask_box_figure(evaluation_folder=evaluation_folder,
                             output_path=output_folder / "evaluation" / "mask_box_average_precision.png"),
        write_mask_report_listings(run_folder=run_folder, iteration=CHECKPOINT_ITERATION,
                                   output_folder=listings_folder),
        read_proposal_recall(evaluation_folder=evaluation_folder),
        {"run.test_scalar_count": count_scalar_tags(log_folder=run_folder / "Tensorboard" / "Test"),
         "run.checkpoint_iteration": CHECKPOINT_ITERATION}])

    # Values the narration quotes that are derived from the facts above
    return merge_facts(fact_groups=[facts, describe_quoted_facts(facts=facts)])
