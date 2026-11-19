"""
Figures and facts of the object detection series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series object_detection) at every build, from the DET run of
tutorials/tutorial_sources.yaml (the shipped synthetic configuration with a level 4 and a 384 pixel scale added,
stopped after its evaluation of DETECTION_CHECKPOINT_ITERATION) and the code of the repository: the COCO manifest and
anchor recall of its data, the trained detector of its last evaluated checkpoint run on the CPU on the test image of the
series (anchors, proposals in the proposal selection of the run, anchor labels, training targets, the objects found at
every compared evaluation; committed in the run snapshots for a build without the checkpoint), the architecture of the
detector built untrained from its configuration (parameters and layers, no checkpoint needed), its training losses and
console summaries, and its evaluation (sample images at three iterations, mAP of the trained and averaged weights, and,
from its last checkpoint evaluated again with the current code in its Checkpoint_Evaluation folder, the average
precision of every class, confusion matrix, report tables and proposal recall), the settings and iterations of the run
quoted next to its measurements (read from its saved configuration, not from the shipped one, nor typed, with the
levels and scales it adds to the shipped configuration), and the joint and per-level proposal selections compared by
stage 7 of the detection benchmark (proposal_selection.py). Every claim the chapters make about these measurements
without a number (the objects of the test image found or missed, the classes named, the shape of a loss curve, the
lead of the averaged weights, the busiest level) is checked here, so that a run trained again stops the build instead
of being described wrongly. Every figure family is a module of this folder writing into a subfolder of the output
folder, its facts gathered here into the facts.json the storyboards quote as {{object_detection.<name>}}. The instance
segmentation series imports the run figures of this folder for its own run.
"""

import fractions
import shutil

import numpy

from pathlib import Path
from typing import Any, Dict, List

from tutorial_builder.figure_context import find_repository_root
from tutorials.figures.figure_utilities import (build_console_metric_tracker, find_run_folder, merge_facts,
                                                read_run_configuration)
from tutorials.figures.object_detection.dataset_figures import make_dataset_figures
from tutorials.figures.object_detection.detection_sources import (DETECTION_CHECKPOINT_ITERATION,
                                                                  DETECTION_SAMPLE_ITERATIONS, EVALUATION_FOLDER_NAME,
                                                                  SAMPLE_IMAGE_NAME, count_scalar_tags,
                                                                  describe_class_name, join_words,
                                                                  load_untrained_run_model, read_scalar_series)
from tutorials.figures.object_detection.detector_inference import describe_model, make_detector_inference_figures
from tutorials.figures.object_detection.evaluation_figures import (draw_class_average_precision_figure,
                                                                   draw_confusion_matrix_figure, read_confusion_notes,
                                                                   read_proposal_recall, write_report_listings)
from tutorials.figures.object_detection.proposal_selection import make_proposal_selection_facts
from tutorials.figures.object_detection.run_figures import (SMOOTHING_WEIGHT, draw_loss_curves_figure,
                                                            draw_metric_iterations_figure,
                                                            draw_sample_iterations_figure,
                                                            read_console_summary_facts, smooth_values,
                                                            write_console_summary_listing)
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.losses.detection_losses import LOCALISATION_LOSS_BETA
from vision_training.trainers.detection_trainer import DetectionTrainer

# Console log of the training of the run (its output redirected to a file), next to the run folder
CONSOLE_LOG_NAME = "object_detection.log"

# Detection configuration of the repository the run was launched from, whose levels and anchor scales the chapters
# compare with those of the run (a level added, a scale added to a level)
DETECTION_CONFIGURATION_PATH = Path("vision_training/configurations/object_detection/"
                                    "configuration_synthetic_shapes.yaml")

# Number of levels and of scales the run adds to the shipped configuration, as the first chapter says: one level of
# its own (block 4) and one scale on a level of the configuration (384 pixels on block 7)
NARRATED_ADDED_LEVEL_COUNT = 1
NARRATED_ADDED_SCALE_COUNT = 1

# Objects of the test image the detections of the last checkpoint take for another class and miss, and the found
# objects of the first compared evaluation, as the chapters say
NARRATED_WRONG_CLASS_COUNT = 1
NARRATED_MISSED_COUNT = 1
NARRATED_FIRST_SAMPLE_FOUND_COUNT = 0

# Objects of the test image matched only by the fallback of the anchor assignment, named one by one by the first
# chapter
NARRATED_FALLBACK_COUNT = 2

# Smallest rise of the smoothed detection classification loss after its first low, the climb back the second chapter
# describes, and the iterations of the start of the curve within which that low falls ("drops at once")
CLASSIFICATION_LOSS_RISE = 0.1
CLASSIFICATION_LOSS_EARLY_ITERATIONS = 500

# Largest distance of the first detection classification loss from the logarithm of the number of classes, the loss
# of uniform logits the second chapter compares it with ("close to the logarithm of 36")
FIRST_CLASSIFICATION_LOSS_TOLERANCE = 0.2

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


def describe_run_levels(run_folder: Path, run_configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the detection levels and anchor scales of the DET run, read from its saved configuration, next to those of the
    detection configuration of the repository it was launched from, so that the first chapter can show the levels of
    the shipped file and say what the run adds to them: the blocks of both, the scales of every level of the run, the
    span of its anchors, the level it adds with its scales and the scale it adds to a level of the configuration,
    checking that the run adds exactly what the chapter says (NARRATED_ADDED_LEVEL_COUNT levels and
    NARRATED_ADDED_SCALE_COUNT scales, and nothing removed) and that every level of both uses the same aspect ratios.
    Also gives the name of the run folder, which the image scenes type out in their runs: paths.

    Args:
        run_folder (Path): The folder of the DET run.
        run_configuration (Dict[str, Any]): The configuration saved in the run folder (read_run_configuration).

    Returns:
        Dict[str, Any]: The facts, named run.*.

    Raises:
        ValueError: If the run removes a level or a scale of the configuration, adds other numbers of levels and
            scales than the chapter says, or uses other aspect ratios.
    """
    shipped_levels = load_configuration(configuration_path=find_repository_root() / DETECTION_CONFIGURATION_PATH)[
        "DetectionLevels"]
    run_levels = run_configuration["DetectionLevels"]
    facts = {"run.name": run_folder.name,
             "run.level_names_text": join_words(words=[str(level) for level in run_levels]),
             "run.shipped_level_names_text": join_words(words=[str(level) for level in shipped_levels])}

    # Scales of every level of the run, and the span of its anchors over all the levels
    anchor_scales = []
    for level, level_settings in run_levels.items():
        scales = level_settings["anchors"]["scales"]
        anchor_scales.extend(scales)
        facts[f"run.level_{level}.scales"] = scales
        facts[f"run.level_{level}.scales_text"] = join_words(words=[str(scale) for scale in scales])
    facts.update({"run.smallest_anchor": min(anchor_scales), "run.largest_anchor": max(anchor_scales)})

    # Levels and scales the run adds to the shipped configuration, none of the configuration being removed
    added_levels = [str(level) for level in run_levels if level not in shipped_levels]
    removed_levels = [str(level) for level in shipped_levels if level not in run_levels]
    added_scales = [(str(level), scale) for level in shipped_levels if level in run_levels
                    for scale in run_levels[level]["anchors"]["scales"]
                    if scale not in shipped_levels[level]["anchors"]["scales"]]
    removed_scales = [(str(level), scale) for level in shipped_levels if level in run_levels
                      for scale in shipped_levels[level]["anchors"]["scales"]
                      if scale not in run_levels[level]["anchors"]["scales"]]
    if (removed_levels or removed_scales or len(added_levels) != NARRATED_ADDED_LEVEL_COUNT
            or len(added_scales) != NARRATED_ADDED_SCALE_COUNT):
        raise ValueError(f"{run_folder.name} adds the levels {added_levels} and the scales {added_scales} to "
                         f"{DETECTION_CONFIGURATION_PATH} and removes the levels {removed_levels} and the scales "
                         f"{removed_scales}, where the first chapter says it adds {NARRATED_ADDED_LEVEL_COUNT} level "
                         f"and {NARRATED_ADDED_SCALE_COUNT} scale.")

    # Aspect ratios of every level of the run equal to those of the configuration, which the anchor scene quotes for all
    aspect_ratios = {tuple(level_settings["anchors"]["aspect_ratios"])
                     for levels in [shipped_levels, run_levels] for level_settings in levels.values()}
    if len(aspect_ratios) != 1:
        raise ValueError(f"The levels of {run_folder.name} and {DETECTION_CONFIGURATION_PATH} use the aspect ratios "
                         f"{sorted(aspect_ratios)}, where the first chapter quotes those of the configuration for "
                         f"every level.")
    facts.update({"run.added_level": added_levels[0],
                  "run.added_level_scales_text": facts[f"run.level_{added_levels[0]}.scales_text"],
                  "run.added_scale_level": added_scales[0][0], "run.added_scale": added_scales[0][1]})
    return facts


def find_run_iterations(run_folder: Path, run_configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives the iterations of the DET run that the figures read: the last evaluated checkpoint
    (DETECTION_CHECKPOINT_ITERATION of detection_sources.py, the run having been stopped after its evaluation), whose
    weights the inference figures run and whose evaluation the report, class and confusion figures read; the three
    evaluated iterations of the sample images (DETECTION_SAMPLE_ITERATIONS: the first evaluation, one in the middle and
    the last), whose confusion matrix notes are quoted too; and the iterations of the console summaries, the first one
    (information_dump) and the one of the last checkpoint; with the iterations the run planned. Every one is checked
    against the run, so that a run trained again for other iterations stops the build until the constants follow it.

    Args:
        run_folder (Path): The folder of the DET run.
        run_configuration (Dict[str, Any]): The configuration saved in the run folder (read_run_configuration).

    Returns:
        Dict[str, Any]: The 'checkpoint_iteration', the 'sample_iterations', the 'console_iterations' and the
            'planned_iterations'.

    Raises:
        ValueError: If the run was not evaluated last at the checkpoint before its planned end, if a sample iteration is
            not an evaluated one, or if the checkpoint has no console summary.
    """
    # Iterations planned by the run, its evaluations every weight_saving_iterations, the last one at the checkpoint
    experiment_configuration = run_configuration["ExperimentConfiguration"]
    planned_iterations = experiment_configuration["number_iterations"]
    evaluation_interval = experiment_configuration["weight_saving_iterations"]
    evaluated_iterations = sorted(int(folder.name.split("_")[1]) for folder in
                                  (run_folder / "Evaluation").glob("Iteration_*"))
    if (planned_iterations <= DETECTION_CHECKPOINT_ITERATION
            or evaluated_iterations[-1] != DETECTION_CHECKPOINT_ITERATION):
        raise ValueError(f"{run_folder.name} plans {planned_iterations} iterations and was evaluated last at "
                         f"{evaluated_iterations[-1]}, where the chapters say it was stopped after its evaluation of "
                         f"{DETECTION_CHECKPOINT_ITERATION}.")
    if (any(iteration % evaluation_interval != 0 or iteration not in evaluated_iterations for iteration in
            DETECTION_SAMPLE_ITERATIONS) or DETECTION_SAMPLE_ITERATIONS[0] != evaluation_interval
            or DETECTION_SAMPLE_ITERATIONS[-1] != DETECTION_CHECKPOINT_ITERATION):
        raise ValueError(f"The sample iterations {DETECTION_SAMPLE_ITERATIONS} of the DET run are not its first "
                         f"evaluation ({evaluation_interval}), one in the middle and its last "
                         f"({DETECTION_CHECKPOINT_ITERATION}), among the evaluated {evaluated_iterations}.")

    # Console summaries printed every information_dump iterations: the first one and that of the checkpoint
    information_dump = experiment_configuration["information_dump"]
    if DETECTION_CHECKPOINT_ITERATION % information_dump != 0:
        raise ValueError(f"The DET run prints a console summary every {information_dump} iterations, none at the "
                         f"checkpoint {DETECTION_CHECKPOINT_ITERATION}.")
    return {"checkpoint_iteration": DETECTION_CHECKPOINT_ITERATION,
            "sample_iterations": list(DETECTION_SAMPLE_ITERATIONS),
            "console_iterations": [information_dump, DETECTION_CHECKPOINT_ITERATION],
            "planned_iterations": planned_iterations}


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
    # Facts of the sample images as lists in the order of the sample iterations
    sample_iterations = iterations["sample_iterations"]
    aliases = {"samples.metrics": [facts[f"samples.metric_{iteration}"] for iteration in sample_iterations],
               "samples.right_detections": [facts[f"outcomes.{iteration}.right"] for iteration in sample_iterations]}

    # Every metric of the trained and of the averaged weights at the last checkpoint
    for _, _, short_name in METRIC_PANELS:
        for folder in ["test", "test_ema"]:
            aliases[f"metrics.{short_name}.{folder}_last"] = facts[
                f"metrics.{short_name}.{folder}_{iterations['checkpoint_iteration']}"]
    return aliases


def check_classification_loss_curve(run_folder: Path, last_iteration: int, number_classes: int) -> None:
    """
    Checks the shape the second chapter gives the detection classification loss of the run, on the curve the loss
    figure draws (smoothed like TensorBoard, up to the last evaluated checkpoint): a first value close to the logarithm
    of the number of classes, a first low within the first CLASSIFICATION_LOSS_EARLY_ITERATIONS iterations, a climb
    back of at least CLASSIFICATION_LOSS_RISE above it, then a fall below that low by the end.

    Args:
        run_folder (Path): The folder of the DET run.
        last_iteration (int): The last iteration of the curve, the checkpoint the chapters stop at.
        number_classes (int): The classes of the detection heads, the background included.

    Raises:
        ValueError: If the curve does not have that shape.
    """
    iterations, values = read_scalar_series(log_folder=run_folder / "Tensorboard" / "Train",
                                            tag="Detection Classification Loss")
    kept = iterations <= last_iteration
    iterations, smoothed_values = iterations[kept], smooth_values(values=values[kept], weight=SMOOTHING_WEIGHT)
    if abs(float(values[0]) - float(numpy.log(number_classes))) > FIRST_CLASSIFICATION_LOSS_TOLERANCE:
        raise ValueError(f"The first detection classification loss {values[0]:.3f} is not close to the logarithm of "
                         f"{number_classes}, as the second chapter says.")

    # First low of the early iterations, the highest point after it, and the end of the curve
    early = iterations <= CLASSIFICATION_LOSS_EARLY_ITERATIONS
    low_index = int(numpy.argmin(smoothed_values[early]))
    peak_value = float(smoothed_values[low_index:].max())
    if (peak_value - float(smoothed_values[low_index]) < CLASSIFICATION_LOSS_RISE
            or float(smoothed_values[-1]) >= float(smoothed_values[low_index])):
        raise ValueError(f"The smoothed detection classification loss does not drop, climb back by "
                         f"{CLASSIFICATION_LOSS_RISE} and fall below its first low ({smoothed_values[low_index]:.3f}, "
                         f"then {peak_value:.3f}, ending at {smoothed_values[-1]:.3f}), as the second chapter says.")


def check_narrated_claims(facts: Dict[str, Any], iterations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks the claims the chapters make about the measurements of the run without quoting a number, so that a run
    trained again, another test image or another checkpoint stops the build instead of being described wrongly, and
    gives the facts derived for them: the objects of the test image matched only by the fallback (two, named), found,
    taken for another class (one) and missed (one) at the last checkpoint, no object found and no detection over the
    whole test split at the first compared evaluation, every positive anchor and proposal of the image kept by the
    samplings, the block taking the most proposal slots on the image and over the test split (the same, with its
    slots), the two coarsest levels sharing their stride and one scale, the levels overlapping in the objects they
    cover, the anchor recall of the small objects coming from the finest level alone and no large object reaching an
    anchor of block 5, the averaged weights leading more from 0.5 to 0.95 than at 0.5, the classes never right at the
    deployment threshold (the one of the best AP named) and never reaching the target precision.

    Args:
        facts (Dict[str, Any]): The merged facts of the figure families of the series.
        iterations (Dict[str, Any]): The iterations of the run (find_run_iterations).

    Returns:
        Dict[str, Any]: The derived facts, named proposal_recall.busiest_block(_proposals_per_image),
            inference.busiest_level, inference.evaluation_filter.busiest_level_kept, run.shared_scale and
            confusion.never_right_best.*.

    Raises:
        ValueError: If one of the claims does not hold.
    """
    problems: List[str] = []

    # Objects of the test image: two matched by the fallback, and at the checkpoint one taken for another class and one
    # missed, the others found
    if facts["inference.fallback_count"] != NARRATED_FALLBACK_COUNT:
        problems.append(f"{facts['inference.fallback_count']} objects matched by the fallback, not "
                        f"{NARRATED_FALLBACK_COUNT}: {facts['inference.fallback_objects']}")
    if (facts["inference.wrong_class_count"] != NARRATED_WRONG_CLASS_COUNT
            or facts["inference.missed_count"] != NARRATED_MISSED_COUNT):
        problems.append(f"at the checkpoint, {facts['inference.wrong_class_count']} objects taken for another class "
                        f"and {facts['inference.missed_count']} missed, not {NARRATED_WRONG_CLASS_COUNT} and "
                        f"{NARRATED_MISSED_COUNT}")
    if facts["inference.samples.found_counts"][-1] != facts["inference.found_object_count"]:
        problems.append("the objects found at the last compared evaluation differ from those of the checkpoint")

    # First compared evaluation: no object found on the image, no detection over the whole test split
    first_iteration = iterations["sample_iterations"][0]
    if (facts["inference.samples.found_counts"][0] != NARRATED_FIRST_SAMPLE_FOUND_COUNT
            or facts["inference.samples.detection_counts"][0] != 0
            or any(facts[f"outcomes.{first_iteration}.{outcome}"] != 0
                   for outcome in ["right", "wrong_class", "false_positives"])):
        problems.append(f"detections at iteration {first_iteration}, where the chapters say none reaches the threshold")

    # Samplings keeping every positive anchor and proposal of the image
    if facts["inference.anchor_labels.sampled_positives"] != facts["inference.anchor_labels.foreground"]:
        problems.append("not every foreground anchor of the image is sampled")
    if facts["inference.detector_targets.sampled_positives"] != facts["inference.detector_targets.positives"]:
        problems.append("not every positive proposal of the image is sampled")

    # Busiest block of the proposal slots, on the image and over the test split
    image_kept = {name.split("_")[-1]: value for name, value in facts.items()
                  if name.startswith("inference.evaluation_filter.kept_level_")}
    split_proposals = {name.split(".")[1].split("_")[1]: value for name, value in facts.items()
                       if name.startswith("proposal_recall.block_") and name.endswith(".proposals_per_image")}
    busiest_level = max(image_kept, key=image_kept.get)
    busiest_block = max(split_proposals, key=split_proposals.get)
    if busiest_level != busiest_block:
        problems.append(f"block {busiest_level} takes the most slots of the image, block {busiest_block} of the split")

    # Two coarsest levels sharing their stride and one scale, whose anchors are the same boxes in both
    finer_level, coarser_level = facts["run.level_names_text"].replace(" and ", ", ").split(", ")[-2:]
    shared_scales = sorted(set(facts[f"run.level_{finer_level}.scales"]) &
                           set(facts[f"run.level_{coarser_level}.scales"]))
    if (facts[f"inference.level_{finer_level}.stride"] != facts[f"inference.level_{coarser_level}.stride"]
            or len(shared_scales) != 1):
        problems.append(f"blocks {finer_level} and {coarser_level} do not share their stride and exactly one scale "
                        f"(shared scales {shared_scales})")

    # Levels overlapping in what they cover: the recalls of the blocks alone add up beyond the recall of all of them
    if sum(facts["proposal_recall.recall_per_block"]) <= facts["proposal_recall.all.recall_50"]:
        problems.append("the proposal recalls of the blocks alone do not add up beyond the recall of all the blocks")

    # Anchor recall: the small objects covered by the finest level alone, no large object by block 5
    finest_level = facts["run.level_names_text"].split(", ")[0]
    if facts["anchor_recall.small.recall_all_levels"] != facts[f"anchor_recall.small.recall_block_{finest_level}"]:
        problems.append(f"the anchor recall of the small objects does not come from block {finest_level} alone")
    if facts["anchor_recall.large.recall_block_5"] != 0:
        problems.append("block 5 covers some large objects")

    # Averaged weights ahead at the checkpoint, more from 0.5 to 0.95 than at 0.5
    lead_50 = facts["metrics.map_50.test_ema_last"] - facts["metrics.map_50.test_last"]
    lead_50_95 = facts["metrics.map_50_95.test_ema_last"] - facts["metrics.map_50_95.test_last"]
    if not 0 < lead_50 < lead_50_95:
        problems.append(f"the lead of the averaged weights is {lead_50:.4f} at 0.5 and {lead_50_95:.4f} from 0.5 to "
                        f"0.95, where the chapter says it grows")

    # Classes never right at the threshold and never reaching the target precision
    if not facts["confusion.never_right"] or not facts["classes.without_threshold"]:
        problems.append("no class is never right at the threshold, or every class reaches the target precision")
    if problems:
        raise ValueError(f"The DET run does not show what the chapters say ({SAMPLE_IMAGE_NAME}, iteration "
                         f"{iterations['checkpoint_iteration']}): " + "; ".join(problems) + ".")
    never_right_best = max(facts["confusion.never_right"], key=lambda name: facts[f"classes.{name}.ap_50"])
    return {"proposal_recall.busiest_block": busiest_block, "inference.busiest_level": busiest_level,
            "proposal_recall.busiest_block_proposals_per_image": split_proposals[busiest_block],
            "inference.evaluation_filter.busiest_level_kept": image_kept[busiest_level],
            "run.shared_scale": shared_scales[0],
            "confusion.never_right_best.name_text": describe_class_name(class_name=never_right_best),
            "confusion.never_right_best.ap_50": facts[f"classes.{never_right_best}.ap_50"]}


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the object detection series into its output folder and gives their facts, the checkpoint
    inference being reused from the previous build unless forced or unless its checkpoint file changed (and taken from
    the run snapshots when the run has no checkpoint), the architecture facts coming from the untrained model of the
    configuration of the run.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the expensive figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.
    """
    # Run folder, the settings it was trained and evaluated with, its iterations, and its last checkpoint as the
    # training evaluated it (sample images) and as the current code evaluated it again (report, classes, confusion
    # matrix and proposal recall)
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="object_detection")
    run_configuration = read_run_configuration(run_folder=run_folder)
    run_settings = describe_run_settings(run_configuration=run_configuration)
    iterations = find_run_iterations(run_folder=run_folder, run_configuration=run_configuration)
    checkpoint_iteration = iterations["checkpoint_iteration"]
    training_evaluation_folder = run_folder / "Evaluation" / f"Iteration_{checkpoint_iteration}"
    evaluation_folder = run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{checkpoint_iteration}"
    listings_folder = output_folder / "listings"

    # Sample image of the test image at the last checkpoint, as the evaluation wrote it, shown by the first chapter
    sample_path = training_evaluation_folder / "Samples" / SAMPLE_IMAGE_NAME
    if not sample_path.exists():
        raise FileNotFoundError(f"No sample image {sample_path} of the last checkpoint of the DET run.")
    (output_folder / "evaluation").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src=sample_path, dst=output_folder / "evaluation" / LAST_SAMPLE_NAME)

    # Console summaries of the run, the first one and that of the last checkpoint, read from the console log next to
    # the run folder, their metric lines written again by the console tracker of the current detection trainer
    console_facts = {}
    metric_tracker = build_console_metric_tracker(run_folder=run_folder, trainer_class=DetectionTrainer,
                                                  trainer_attributes={})
    for position_name, iteration in zip(["first", "last"], iterations["console_iterations"]):
        summary_lines = write_console_summary_listing(log_path=run_folder.parent / CONSOLE_LOG_NAME,
                                                      iteration=iteration,
                                                      output_path=listings_folder / f"console_{position_name}.txt",
                                                      metric_tracker=metric_tracker)
        console_facts.update(read_console_summary_facts(summary_lines=summary_lines,
                                                        fact_prefix=f"console.{position_name}"))
    model = load_untrained_run_model(run_folder=run_folder)
    check_classification_loss_curve(run_folder=run_folder, last_iteration=checkpoint_iteration,
                                    number_classes=model.number_classes)
    facts = merge_facts(fact_groups=[
        console_facts,
        read_confusion_notes(run_folder=run_folder, iterations=iterations["sample_iterations"]),
        make_dataset_figures(output_folder=output_folder, run_folder=run_folder),
        make_detector_inference_figures(output_folder=output_folder, run_folder=run_folder,
                                        iteration=checkpoint_iteration, force=force),
        describe_model(model=model),
        describe_run_levels(run_folder=run_folder, run_configuration=run_configuration),
        draw_loss_curves_figure(run_folder=run_folder, panels=LOSS_PANELS,
                                output_path=output_folder / "training" / "loss_curves.png", fact_prefix="losses",
                                last_iteration=checkpoint_iteration),
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
        write_report_listings(results_path=run_folder / EVALUATION_FOLDER_NAME / "Results.txt",
                              iteration=checkpoint_iteration, output_folder=listings_folder),
        read_proposal_recall(evaluation_folder=evaluation_folder),
        make_proposal_selection_facts(experiments_folder=experiments_folder),
        describe_box_loss_beta(), run_settings,
        {"run.test_scalar_count": count_scalar_tags(log_folder=run_folder / "Tensorboard" / "Test"),
         "run.checkpoint_iteration": checkpoint_iteration, "run.planned_iterations": iterations["planned_iterations"],
         "samples.iterations": iterations["sample_iterations"],
         "console.iterations": iterations["console_iterations"]}])

    # Facts named by an iteration given again under names that hold none, then the claims of the chapters checked
    facts = merge_facts(fact_groups=[facts, describe_iteration_aliases(facts=facts, iterations=iterations)])
    return merge_facts(fact_groups=[facts, check_narrated_claims(facts=facts, iterations=iterations)])
