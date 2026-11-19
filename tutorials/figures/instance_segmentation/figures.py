"""
Figures and facts of the instance segmentation series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series instance_segmentation) at every build, from the INS run of
tutorials/tutorial_sources.yaml (the synthetic configuration with the settings the benchmarks kept, stopped after its
evaluation of 6000 iterations) and the code of the repository: the run length encoding of the masks and its decoding,
the mask target of a real proposal, the masks the trained instance segmenter predicts on the CPU and their pasting
(mask_figures.py), the mask against box average precision and the report tables of the last checkpoint evaluated
again with the current code in the Checkpoint_Evaluation folder of the run (evaluation_figures.py), and, with the run
figures of the object detection series imported from tutorials/figures/object_detection/run_figures.py, the losses,
the sample images along the run, the mask mAP of the trained and averaged weights and the console summaries, with the
geometry of the mask heads, the settings of the run the chapters name (neck, proposal selection, suppression
criterion), the deployment score threshold of the run (read from its saved configuration, the shipped one free to move
on) and the values the narration derives from them (describe_quoted_facts), every property the chapters narrate about
the test image, the losses and the report being checked there. The facts are quoted by the storyboards as
{{instance_segmentation.<name>}}.
"""

import math

from pathlib import Path
from typing import Any, Dict, List

from tutorial_builder.figure_context import find_repository_root
from tutorials.figures.figure_utilities import (build_console_metric_tracker, find_run_folder, merge_facts,
                                                read_run_configuration)
from tutorials.figures.instance_segmentation.evaluation_figures import (HOLLOW_PREFIX, draw_mask_box_figure,
                                                                        write_mask_report_listings)
from tutorials.figures.instance_segmentation.mask_figures import (FOUND_MASK_OVERLAP, describe_mask_geometry,
                                                                  make_mask_figures)
from tutorials.figures.object_detection.detection_sources import (CHECKPOINT_ITERATION, EVALUATION_FOLDER_NAME,
                                                                  SAMPLE_IMAGE_NAME, SAMPLE_ITERATIONS,
                                                                  count_scalar_tags, find_test_folder,
                                                                  read_test_manifest)
from tutorials.figures.object_detection.evaluation_figures import read_confusion_notes, read_proposal_recall
from tutorials.figures.object_detection.run_figures import (draw_loss_curves_figure, draw_metric_iterations_figure,
                                                            draw_sample_iterations_figure,
                                                            read_console_summary_facts,
                                                            write_console_summary_listing)
from vision_training.configuration_processing.configuration_loader import load_configuration
from vision_training.trainers.instance_segmentation_trainer import InstanceSegmentationTrainer

# Console log of the run, next to its folder (one launch, stopped after the evaluation of 6000), and the iterations of
# the summaries read from it: the first and the last
CONSOLE_SUMMARIES = [("instance_segmentation.log", 1000), ("instance_segmentation.log", 6000)]

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
EVALUATED_ITERATIONS = [1000, 2000, 3000, 4000, 5000, 6000]

# Detection configuration of the repository, whose neck, proposal selection and final suppression criterion (the
# winners of the detection benchmark) the first chapter says the run shown took, and those settings
DETECTION_CONFIGURATION_PATH = Path("vision_training/configurations/object_detection/"
                                    "configuration_synthetic_shapes.yaml")
DETECTOR_SETTINGS = {"neck_type": ("Neck", "type"), "proposal_selection": ("RegionProposal", "proposal_selection"),
                     "nms_criterion": ("Detector", "nms_criterion")}

# Neck the chapters describe for the run shown (a feature pyramid on the detection levels)
NARRATED_NECK_TYPE = "feature_pyramid"

# Largest distance of the first mask loss of the run from the logarithm of two, the loss of logits at zero the first
# chapter names
FIRST_MASK_LOSS_TOLERANCE = 0.02


def join_words(words: List[str]) -> str:
    """
    Joins words the way the narration lists them: commas, then "and" before the last one.

    Args:
        words (List[str]): The words, at least one.

    Returns:
        str: The list, e.g. "4, 5, 6 and 7".
    """
    return ", ".join(words[:-1]) + " and " + words[-1] if len(words) > 1 else words[0]


def describe_run_settings(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the settings of the INS run that the first chapter names next to those of the configurations (its neck,
    proposal selection and final suppression criterion, read from its saved configuration), its folder name and test
    image for the runs: path of the sample image it shows, its planned iterations and the span of its anchors, and the
    objects of its test split, checking that the run took the settings of the detection configuration of the
    repository and a feature pyramid neck, and that it was stopped after the evaluation of CHECKPOINT_ITERATION, as the
    chapter says.

    Args:
        run_folder (Path): The folder of the INS run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named run.* and data.*.

    Raises:
        ValueError: If a setting of the run differs from the detection configuration, or its neck is not a feature
            pyramid.
    """
    run_configuration = read_run_configuration(run_folder=run_folder)
    detection_configuration = load_configuration(configuration_path=find_repository_root() /
                                                 DETECTION_CONFIGURATION_PATH)

    # Settings of the detector the run shares with the detection configuration
    facts = {"run.name": run_folder.name, "run.sample_image_name": SAMPLE_IMAGE_NAME}
    for fact_name, (section, key) in DETECTOR_SETTINGS.items():
        run_value = run_configuration[section][key]
        if run_value != detection_configuration[section][key]:
            raise ValueError(f"The {section}.{key} of {run_folder.name} is {run_value!r}, not the "
                             f"{detection_configuration[section][key]!r} of {DETECTION_CONFIGURATION_PATH}, whose "
                             f"settings the first chapter says the run took.")
        facts[f"run.{fact_name}"] = run_value
    if facts["run.neck_type"] != NARRATED_NECK_TYPE:
        raise ValueError(f"The neck of {run_folder.name} is {facts['run.neck_type']!r}, where the chapters describe a "
                         f"{NARRATED_NECK_TYPE} neck.")

    # Iterations planned by the run, which was stopped after the evaluation of its last checkpoint, as the first
    # chapter says, and the span of its anchors over its detection levels
    planned_iterations = run_configuration["ExperimentConfiguration"]["number_iterations"]
    evaluated_iterations = [int(folder.name.split("_")[1])
                            for folder in (run_folder / "Evaluation").glob("Iteration_*")]
    if planned_iterations <= CHECKPOINT_ITERATION or max(evaluated_iterations) != CHECKPOINT_ITERATION:
        raise ValueError(f"{run_folder.name} plans {planned_iterations} iterations and was evaluated last at "
                         f"{max(evaluated_iterations)}, where the first chapter says it was stopped after its "
                         f"evaluation of {CHECKPOINT_ITERATION}.")
    anchor_scales = [scale for level in run_configuration["DetectionLevels"].values()
                     for scale in level["anchors"]["scales"]]
    facts.update({"run.planned_iterations": planned_iterations, "run.smallest_anchor": min(anchor_scales),
                  "run.largest_anchor": max(anchor_scales)})

    # Objects and images of the test split of the run, the split of the confusion notes of the reports
    manifest = read_test_manifest(test_folder=find_test_folder(run_configuration=run_configuration))
    facts.update({"data.test_images": len(manifest["images"]), "data.test_objects": len(manifest["annotations"])})
    return facts


def describe_test_image_story(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derives from the detections of the trained segmenter on the test image of the series the objects and detections the
    chapters name by their role, and checks every property the narration states about them: every object found by a
    detection of its class at a mask IoU of FOUND_MASK_OVERLAP or more; one object, and one only, found twice, once
    under its own class and once under another class; a detection of the right class on every detection level, the
    highest scoring one of each level being named; the right detections of the best and of the weakest mask IoU, the
    latter scoring a higher IoU on its box and being a hollow shape whose mask misses the shape mostly outside its
    detection box and adds pixels mostly in its hole; and no detection of the first compared checkpoint on any sample
    image.

    Args:
        facts (Dict[str, Any]): The merged facts of the figure families of the series.

    Returns:
        Dict[str, Any]: The derived facts, named predictions.* and samples.*.

    Raises:
        ValueError: If one of the narrated properties does not hold for the run and image of the build.
    """
    object_names = facts["predictions.object_names"]
    detections = [{"index": index, **{key: facts[f"predictions.detection_{index}.{key}"] for key in
                                      ["name", "name_text", "level", "mask_iou", "box_iou", "object", "object_index",
                                       "object_text", "score", "tile_box"]}}
                  for index in range(1, facts["predictions.detection_count"] + 1)]
    right_detections = [detection for detection in detections
                        if detection["name"] == detection["object"] and detection["mask_iou"] >= FOUND_MASK_OVERLAP]

    # Every object found by a detection of its class, as the chapters say
    found_objects = {detection["object_index"] for detection in right_detections}
    if len(found_objects) != len(object_names):
        missed_names = [name for index, name in enumerate(object_names) if index not in found_objects]
        raise ValueError(f"{SAMPLE_IMAGE_NAME}: the objects {missed_names} are not found by a detection of their class "
                         f"at a mask IoU of {FOUND_MASK_OVERLAP}, where the chapters say every object is found.")

    # One object found twice, under its class and under another one
    detections_by_object = {}
    for detection in detections:
        detections_by_object.setdefault(detection["object_index"], []).append(detection)
    duplicated = [object_detections for object_detections in detections_by_object.values()
                  if len(object_detections) > 1]
    if len(duplicated) != 1 or len(duplicated[0]) != 2:
        raise ValueError(f"{SAMPLE_IMAGE_NAME}: the chapters say one object is found twice, but the detections per "
                         f"object are {[len(value) for value in detections_by_object.values()]}.")
    duplicate_right = [detection for detection in duplicated[0] if detection in right_detections]
    duplicate_wrong = [detection for detection in duplicated[0] if detection["name"] != detection["object"]]
    if len(duplicate_right) != 1 or len(duplicate_wrong) != 1:
        raise ValueError(f"{SAMPLE_IMAGE_NAME}: the object found twice is not found once under its class and once "
                         f"under another class: {duplicated[0]}.")
    story = {"predictions.found_object_count": len(found_objects),
             "predictions.duplicate.object_text": duplicate_right[0]["object_text"],
             "predictions.duplicate.right_level": duplicate_right[0]["level"],
             "predictions.duplicate.right_tile_box": duplicate_right[0]["tile_box"],
             "predictions.duplicate.wrong_name_text": duplicate_wrong[0]["name_text"],
             "predictions.duplicate.wrong_level": duplicate_wrong[0]["level"],
             "predictions.duplicate.wrong_tile_box": duplicate_wrong[0]["tile_box"]}

    # The highest scoring right detection of every level, every level giving one
    level_names = [int(name) for name in facts["geometry.level_names"].split(", ")]
    for level in level_names:
        level_detections = [detection for detection in right_detections if detection["level"] == level]
        if not level_detections:
            raise ValueError(f"{SAMPLE_IMAGE_NAME}: no right detection from level {level}, where the chapters show a "
                             f"mask from every level.")
        best_detection = max(level_detections, key=lambda detection: detection["score"])
        story.update({f"predictions.level_{level}.name_text": best_detection["name_text"],
                      f"predictions.level_{level}.tile_box": best_detection["tile_box"]})
    story["predictions.levels_text"] = join_words(words=[str(level) for level in level_names])

    # The right detections of the best and of the weakest mask overlap, the weakest doing better on its box, and its
    # mask, that of a hollow shape, missing the shape mostly outside its detection box and adding pixels mostly in the
    # hole
    weakest_detection = min(right_detections, key=lambda detection: detection["mask_iou"])
    for role, chosen in [("best_mask", max(right_detections, key=lambda detection: detection["mask_iou"])),
                         ("weakest_mask", weakest_detection)]:
        story.update({f"predictions.{role}.name_text": chosen["name_text"],
                      f"predictions.{role}.mask_iou": chosen["mask_iou"],
                      f"predictions.{role}.box_iou": chosen["box_iou"]})
    if weakest_detection["box_iou"] <= weakest_detection["mask_iou"]:
        raise ValueError(f"{SAMPLE_IMAGE_NAME}: the weakest mask ({weakest_detection['name_text']}) does not score a "
                         f"higher IoU on its box, as the chapters say.")
    pixel_counts = {name: facts[f"predictions.detection_{weakest_detection['index']}.{name}"] for name in
                    ["missed_pixels", "missed_outside_box_pixels", "added_pixels", "added_hole_pixels"]}
    if (not weakest_detection["name"].startswith(HOLLOW_PREFIX)
            or 2 * pixel_counts["missed_outside_box_pixels"] <= pixel_counts["missed_pixels"]
            or 2 * pixel_counts["added_hole_pixels"] <= pixel_counts["added_pixels"]):
        raise ValueError(f"{SAMPLE_IMAGE_NAME}: the weakest mask ({weakest_detection['name_text']}, {pixel_counts}) is "
                         f"not that of a hollow shape missing the shape mostly outside its box and adding pixels "
                         f"mostly in its hole, as the second chapter says.")

    # The samples of the first compared checkpoint stay empty
    if facts["samples.first_empty_count"] != facts["samples.first_image_count"]:
        raise ValueError(f"{facts['samples.first_image_count'] - facts['samples.first_empty_count']} sample images of "
                         f"iteration {SAMPLE_ITERATIONS[0]} get detections, where the chapters say the samples stay "
                         f"empty.")
    return story


def check_narrated_comparisons(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks the comparisons the chapters state about the losses and the report of the run, so that a run trained again
    stops the build instead of being described wrongly: the first annotation of the test manifest compressed, the
    first mask loss at the logarithm of two, the masks ahead of the boxes at an IoU of 0.95, more small objects by mask
    area than by box area, and one object size the weakest by mAP@0.5 both by mask and by box, which it names.

    Args:
        facts (Dict[str, Any]): The merged facts of the figure families of the series.

    Returns:
        Dict[str, Any]: The weakest object size and its mask and box mAP@0.5, named report.weakest_size*.

    Raises:
        ValueError: If one of the comparisons does not hold.
    """
    if not facts["annotation.compressed"]:
        raise ValueError("The first annotation of the test manifest is not a compressed run length encoding, as the "
                         "first chapter says.")
    if abs(facts["losses.mask.first"] - math.log(2.0)) > FIRST_MASK_LOSS_TOLERANCE:
        raise ValueError(f"The first mask loss of the run is {facts['losses.mask.first']}, not the logarithm of two "
                         f"the first chapter names.")
    if facts["report.trained.mask_methods.area.2"] <= facts["report.trained.box_methods.area.2"]:
        raise ValueError("The masks do not beat the boxes at an IoU of 0.95, as the second chapter says.")
    if facts["report.trained.mask_sizes.small.0"] <= facts["report.trained.box_sizes.small.0"]:
        raise ValueError("The test objects are not more numerous among the small ones by mask than by box, as the "
                         "second chapter says.")

    # The weakest object size by mAP@0.5, the same by mask and by box ("either way")
    weakest_sizes = {}
    for table_name in ["mask_sizes", "box_sizes"]:
        size_maps = {size: facts[f"report.trained.{table_name}.{size}.1"] for size in ["small", "medium", "large"]}
        weakest_sizes[table_name] = min(size_maps, key=size_maps.get)
    if weakest_sizes["mask_sizes"] != weakest_sizes["box_sizes"]:
        raise ValueError(f"The weakest object size is not the same by mask and by box ({weakest_sizes}), where the "
                         f"second chapter says it is the weak point either way.")
    weakest_size = weakest_sizes["mask_sizes"]
    return {"report.weakest_size": weakest_size,
            "report.weakest_size_mask_map": facts[f"report.trained.mask_sizes.{weakest_size}.1"],
            "report.weakest_size_box_map": facts[f"report.trained.box_sizes.{weakest_size}.1"]}


def describe_quoted_facts(facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derives from the facts of the figure families the values the narration quotes in words or as a whole, so that no
    storyboard types them: the iterations of the console summaries and of the sample images (CONSOLE_SUMMARIES and
    SAMPLE_ITERATIONS, the facts named after them), the objects and detections of the test image named by their role
    (describe_test_image_story), and the detection level whose proposals are the most numerous; the comparisons the
    chapters state are checked, the weakest object size being named (check_narrated_comparisons).

    Args:
        facts (Dict[str, Any]): The merged facts of the figure families of the series.

    Returns:
        Dict[str, Any]: The derived facts.
    """
    # Level of the proposal recall table with the most proposals per image
    block_proposals = {int(name.split(".")[1].split("_")[1]): value for name, value in facts.items()
                       if name.startswith("proposal_recall.block_") and name.endswith(".proposals_per_image")}
    return {"console.first_iteration": CONSOLE_SUMMARIES[0][1], "console.last_iteration": CONSOLE_SUMMARIES[-1][1],
            "samples.iterations": list(SAMPLE_ITERATIONS), **describe_test_image_story(facts=facts),
            **check_narrated_comparisons(facts=facts),
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
    # Run folder and its last checkpoint as the current code evaluated it again (report, classes, proposal recall), the
    # sample images and curves staying those of the training
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="instance_segmentation")
    evaluation_folder = run_folder / EVALUATION_FOLDER_NAME / f"Iteration_{CHECKPOINT_ITERATION}"
    listings_folder = output_folder / "listings"

    # Console summaries of the first evaluation and of the last iteration, read from the log next to the run folder,
    # their metric lines written again by the console tracker of the current instance segmentation trainer
    console_facts = {}
    metric_tracker = build_console_metric_tracker(run_folder=run_folder,
                                                  trainer_class=InstanceSegmentationTrainer, trainer_attributes={})
    for log_name, iteration in CONSOLE_SUMMARIES:
        summary_lines = write_console_summary_listing(log_path=run_folder.parent / log_name, iteration=iteration,
                                                      output_path=listings_folder / f"console_{iteration}.txt",
                                                      metric_tracker=metric_tracker)
        console_facts.update(read_console_summary_facts(summary_lines=summary_lines,
                                                        fact_prefix=f"console.{iteration}"))
    facts = merge_facts(fact_groups=[
        console_facts, describe_mask_geometry(run_folder=run_folder), describe_run_settings(run_folder=run_folder),
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
        write_mask_report_listings(results_path=run_folder / EVALUATION_FOLDER_NAME / "Results.txt",
                                   iteration=CHECKPOINT_ITERATION, output_folder=listings_folder),
        read_proposal_recall(evaluation_folder=evaluation_folder),
        {"run.test_scalar_count": count_scalar_tags(log_folder=run_folder / "Tensorboard" / "Test"),
         "run.checkpoint_iteration": CHECKPOINT_ITERATION,
         "run.score_threshold": read_run_configuration(run_folder=run_folder)["Detector"]["inference"][
             "score_threshold"]}])

    # Values the narration quotes that are derived from the facts above
    return merge_facts(fact_groups=[facts, describe_quoted_facts(facts=facts)])
