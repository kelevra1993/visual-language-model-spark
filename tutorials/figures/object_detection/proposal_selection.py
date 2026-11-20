"""
Facts of the proposal selection scenes of the object detection tutorials (chapter 1, the joint selection against the
per-level one), read at every build by tutorials/figures/object_detection/figures.py from stage 7 of the detection
benchmark of the repository (documentation/architecture_benchmarking/object_detection_experiment_plan.md): the anchors
of every level and the proposal counts of the benchmark detector, built untrained on the CPU from the configuration of
a per-level launch (the anchors and counts do not depend on the weights, so no checkpoint is read); the proposals
per image and the proposal recall of every level of the averaged weights at the compared iteration, read from the test
curves of the two launches of each selection (the object_detection_benchmark_* runs of tutorials/tutorial_sources.yaml);
the test mAP of the averaged weights at every evaluation, read from the committed results of the benchmark
(object_detection_benchmark_results, object_detection_results.csv), averaged over the two launches of each selection as
the plan compares them, with the gap between the two launches of a selection (the noise of a training); and the
inference-only preview of the per-level selection on the weights of a joint launch. Nothing is drawn: the scenes draw
the values as pipelines, bars and curves, the labels of the recalls and the ticks of the mAP written as the percentages
of the reports of the repository ("94.03 %", "30 %") and the lead and gap in percentage points, the facts keeping the
rates and their differences between 0 and 1.
"""

import csv
import math
import statistics

from pathlib import Path
from typing import Any, Dict, List

from scripts.evaluation.collect_experiment_results import read_test_curves
from tutorials.figures.figure_utilities import find_repository_file, find_run_folder, round_fact
from tutorials.figures.object_detection.detection_sources import load_untrained_run_model
from vision_training.evaluation.report_formatting import (convert_to_percentage, format_percentage,
                                                          format_percentage_label)

# Launches of each selection, the run keys of tutorial_sources.yaml, and their names in the results of the benchmark
SELECTION_LAUNCHES = {"joint": {"object_detection_benchmark_joint_a": "run_9",
                                "object_detection_benchmark_joint_b": "run_10"},
                      "per_level": {"object_detection_benchmark_per_level_a": "run_21",
                                    "object_detection_benchmark_per_level_b": "run_22"}}

# Evaluation of the weights of a joint launch with the per-level selection at inference only, and that launch
PREVIEW_RUN_KEY = "object_detection_benchmark_per_level_preview"
PREVIEW_REFERENCE_RUN_KEY = "object_detection_benchmark_joint_a"

# Iteration at which the stage compares the selections, the last one of its launches
COMPARED_ITERATION = 15000

# Test curves of the averaged weights read at the compared iteration, and the compared metric
PROPOSALS_PER_IMAGE_TAG = "Proposals Per Image/{level}"
PROPOSAL_RECALL_TAG = "Proposal Recall@0.5/{level}"
MAP_TAG = "mAP@[0.5:0.95]/area"

# Iterations between two ticks of the iteration axis of the curves, and mAP points between two ticks of the lead
ITERATION_TICK_STEP = 5000
POINTS_TICK_STEP = 2


def read_selection_curves(results_path: Path, launch_names: List[str]) -> Dict[str, Dict[int, float]]:
    """
    Reads the test mAP@[0.5:0.95] of the averaged weights of launches of the detection benchmark at every evaluation,
    from the committed results of the benchmark, so that the curves of the training scene come from the numbers the
    plan compares and can be drawn on a machine without the run folders.

    Args:
        results_path (Path): The results of the benchmark, object_detection_results.csv.
        launch_names (List[str]): The launches, named as in the results (run_9, run_21...).

    Returns:
        Dict[str, Dict[int, float]]: The mAP at every evaluated iteration of every launch.

    Raises:
        KeyError: If a launch has no row of averaged weights in the results.
    """
    # Rows of the averaged weights of the launches, the iteration of every filled map_50_95_<iteration> column
    curves = {}
    with results_path.open(encoding="utf-8", newline="") as results_file:
        for row in csv.DictReader(results_file):
            if row["launch"] in launch_names and row["evaluated_weights"] == "ema":
                curves[row["launch"]] = {int(column.rsplit("_", 1)[1]): float(value) for column, value in row.items()
                                         if column.startswith("map_50_95_") and column.rsplit("_", 1)[1].isdigit()
                                         and value != ""}
    missing_launches = sorted(set(launch_names) - set(curves))
    if missing_launches:
        raise KeyError(f"No averaged weights row for {missing_launches} in {results_path}.")
    return curves


def describe_benchmark_levels(run_folder: Path) -> Dict[str, Any]:
    """
    Builds the detector of a launch of the benchmark untrained on the CPU from its saved configuration to give the
    anchors of every level and the proposal counts of its filter, the numbers the pipelines of the joint and per-level
    selections show, read from the model rather than computed from the configuration so that they follow the anchor
    generator of the repository; the anchors are not weights, so no checkpoint is needed and the facts come out the
    same from the run snapshots.

    Args:
        run_folder (Path): The run folder of a per-level launch, holding its configuration and label dictionary.

    Returns:
        Dict[str, Any]: The levels, their anchors, the evaluation and training counts, and the candidates reaching a
            suppression with each selection, named proposal_selection.*.
    """
    # Build the detector of the launch from its configuration and read the anchors of every level from its region
    # proposers
    model = load_untrained_run_model(run_folder=run_folder)
    level_names = list(model.region_proposer_dictionary)
    level_anchor_counts = [int(region_proposer.region_proposal_anchor_object.anchors.shape[0])
                           for region_proposer in model.region_proposer_dictionary.values()]
    filter_configuration = model.region_proposal_configuration

    # Candidates of an image reaching a suppression in evaluation: the best of all the anchors for the joint
    # selection, the best of every level (all of a level with fewer anchors) for the per-level one
    inference_pre_nms = filter_configuration["inference"]["pre_nms_proposals"]
    per_level_top_counts = [min(inference_pre_nms, anchor_count) for anchor_count in level_anchor_counts]
    return {"proposal_selection.image_size": model.input_image_size,
            "proposal_selection.level_count": len(level_names),
            "proposal_selection.level_names": level_names,
            "proposal_selection.level_labels": [f"level {level_name}" for level_name in level_names],
            "proposal_selection.level_anchor_counts": level_anchor_counts,
            "proposal_selection.anchor_total": sum(level_anchor_counts),
            "proposal_selection.inference_pre_nms": inference_pre_nms,
            "proposal_selection.inference_post_nms": filter_configuration["inference"]["post_nms_proposals"],
            "proposal_selection.training_pre_nms": filter_configuration["training"]["pre_nms_proposals"],
            "proposal_selection.training_post_nms": filter_configuration["training"]["post_nms_proposals"],
            "proposal_selection.per_level_top_counts": per_level_top_counts,
            "proposal_selection.per_level_candidate_count": sum(per_level_top_counts),
            "proposal_selection.joint_candidate_count": min(inference_pre_nms, sum(level_anchor_counts))}


def read_pair_level_values(run_folders: List[Path], tag_template: str, level_names: List[str]) -> List[float]:
    """
    Averages a per-level test curve of the averaged weights (proposals per image, proposal recall) over the two
    launches of a selection at the compared iteration, the way the plan compares the selections.

    Args:
        run_folders (List[Path]): The run folders of the launches of the selection.
        tag_template (str): The curve, with {level} for the block name (PROPOSALS_PER_IMAGE_TAG...).
        level_names (List[str]): The block names of the levels.

    Returns:
        List[float]: The mean over the launches of every level, in the order of the levels.

    Raises:
        KeyError: If a launch has no such curve or no evaluation at the compared iteration.
    """
    # Test curves of the averaged weights of every launch, read once for all the levels
    launch_curves = [read_test_curves(test_folder=run_folder / "Tensorboard" / "Test_EMA")
                     for run_folder in run_folders]

    # Value of every level at the compared iteration in every launch, averaged over the launches
    level_values = []
    for level_name in level_names:
        tag = tag_template.format(level=f"block_{level_name}")
        launch_values = []
        for run_folder, curves in zip(run_folders, launch_curves):
            if COMPARED_ITERATION not in curves.get(tag, {}):
                raise KeyError(f"No {tag} at iteration {COMPARED_ITERATION} in the averaged weights of {run_folder}.")
            launch_values.append(curves[tag][COMPARED_ITERATION])
        level_values.append(round_fact(value=statistics.fmean(launch_values)))
    return level_values


def describe_measured_proposals(run_folders: Dict[str, List[Path]], level_names: List[str],
                                post_nms_proposals: int) -> Dict[str, Any]:
    """
    Gives the proposals per image and the proposal recall at an IoU of 0.5 of every level for both selections, the
    bars of the measured scene: how many of the evaluation slots hold a real proposal rather than a copy, and how many
    test objects every level covers.

    Args:
        run_folders (Dict[str, List[Path]]): The run folders of the launches of every selection.
        level_names (List[str]): The block names of the levels.
        post_nms_proposals (int): The evaluation slots of an image, filled with copies when too few proposals survive.

    Returns:
        Dict[str, Any]: The values of every selection, with their labels (the recalls as percentages) and the top of
            the recall bars, named proposal_selection.*.
    """
    facts = {}
    for selection_name, selection_folders in run_folders.items():
        # Proposals of every level and their sum, the real proposals of the slots of an image
        proposals_per_level = read_pair_level_values(run_folders=selection_folders,
                                                     tag_template=PROPOSALS_PER_IMAGE_TAG, level_names=level_names)
        proposals_per_image = round_fact(value=sum(proposals_per_level))
        facts[f"proposal_selection.{selection_name}_proposals_per_level"] = proposals_per_level
        facts[f"proposal_selection.{selection_name}_proposals_per_level_labels"] = [
            f"{value:.1f}" for value in proposals_per_level]
        facts[f"proposal_selection.{selection_name}_proposals_per_image"] = proposals_per_image
        facts[f"proposal_selection.{selection_name}_copied_slots"] = round_fact(
            value=post_nms_proposals - proposals_per_image)

        # Share of the test objects covered at 0.5 by a proposal of every level, labelled as percentages
        recall_per_level = read_pair_level_values(run_folders=selection_folders, tag_template=PROPOSAL_RECALL_TAG,
                                                  level_names=level_names)
        facts[f"proposal_selection.{selection_name}_recall_per_level"] = recall_per_level
        facts[f"proposal_selection.{selection_name}_recall_per_level_labels"] = [
            format_percentage(value=value) for value in recall_per_level]

    # Top of the recall bars, the next tenth above the highest level, leaving room for the legend of the bars
    facts["proposal_selection.recall_axis_maximum"] = math.ceil(max(
        facts["proposal_selection.joint_recall_per_level"] + facts["proposal_selection.per_level_recall_per_level"])
        * 10) / 10

    # Real proposals of both selections as the two bars of the slots, joint first
    facts["proposal_selection.proposals_per_image_bars"] = [facts["proposal_selection.joint_proposals_per_image"],
                                                            facts["proposal_selection.per_level_proposals_per_image"]]
    facts["proposal_selection.proposals_per_image_bar_labels"] = [
        f"{value:.1f}" for value in facts["proposal_selection.proposals_per_image_bars"]]
    return facts


def make_axis_ticks(maximum: float, step: float, in_percent: bool) -> List[Dict[str, Any]]:
    """
    Gives the ticks of an axis of a curve scene from 0 to its maximum, the labels written as whole numbers with
    thousands separators (iterations, mAP points) or, for an axis of rates, as the percentage labels of the reports of
    the repository ("30 %"), so that the axes follow the measured values instead of numbers typed in the storyboard.

    Args:
        maximum (float): The end of the axis.
        step (float): The distance between two ticks.
        in_percent (bool): Whether the axis holds rates between 0 and 1, labelled as percentages.

    Returns:
        List[Dict[str, Any]]: The value and label of every tick.
    """
    # Ticks from 0 to the maximum, rounded so that a fractional step gives clean labels
    tick_count = int(round(maximum / step)) + 1
    tick_values = [round(tick_index * step, 6) for tick_index in range(tick_count)]
    return [{"value": tick_value,
             "label": format_percentage_label(value=tick_value) if in_percent else f"{tick_value:,.0f}"}
            for tick_value in tick_values]


def describe_training_comparison(results_path: Path) -> Dict[str, Any]:
    """
    Gives the test mAP of the averaged weights of both selections at every evaluation, averaged over their two
    launches, the lead of the per-level selection and the larger gap between the two launches of a selection (the
    noise of one training, the tie margin of the plan) at every evaluation, the first iteration from which the lead
    stays above that margin, and the axes of the two curves of the training scene (the mAP in percent, the lead and
    margin in percentage points).

    Args:
        results_path (Path): The results of the benchmark, object_detection_results.csv.

    Returns:
        Dict[str, Any]: The curves, final values, lead and margin (between 0 and 1, the curves of the lead and margin
            in percentage points), named proposal_selection.*.

    Raises:
        ValueError: If the launches of the selections were not evaluated at the same iterations up to the compared one.
    """
    launch_names = [launch_name for launches in SELECTION_LAUNCHES.values() for launch_name in launches.values()]
    curves = read_selection_curves(results_path=results_path, launch_names=launch_names)
    iterations = sorted(iteration for iteration in curves[launch_names[0]] if iteration <= COMPARED_ITERATION)
    if any(sorted(iteration for iteration in curves[launch_name] if iteration <= COMPARED_ITERATION) != iterations
           for launch_name in launch_names):
        raise ValueError(f"The launches {launch_names} were not evaluated at the same iterations up to "
                         f"{COMPARED_ITERATION} in {results_path}.")

    # Mean of the two launches of every selection and the gap between them, at every evaluation
    means, gaps = {}, {}
    for selection_name, launches in SELECTION_LAUNCHES.items():
        selection_curves = [curves[launch_name] for launch_name in launches.values()]
        means[selection_name] = [statistics.fmean(curve[iteration] for curve in selection_curves)
                                 for iteration in iterations]
        gaps[selection_name] = [max(curve[iteration] for curve in selection_curves) -
                                min(curve[iteration] for curve in selection_curves) for iteration in iterations]
    leads = [per_level_mean - joint_mean for per_level_mean, joint_mean in zip(means["per_level"], means["joint"])]
    margins = [max(joint_gap, per_level_gap) for joint_gap, per_level_gap in zip(gaps["joint"], gaps["per_level"])]

    # First evaluation from which the lead stays above the margin up to the compared iteration
    first_win_index = len(iterations)
    while first_win_index > 0 and leads[first_win_index - 1] > margins[first_win_index - 1]:
        first_win_index -= 1
    if first_win_index == len(iterations):
        raise ValueError(f"The per-level selection does not lead beyond the margin at {COMPARED_ITERATION}: the "
                         f"training scene of chapter 1 says it does.")

    # Axes of the curves: the iterations from 0, the mAP up to the next tenth (three ticks, labelled in percent), the
    # lead and margin in mAP points up to the next tick
    map_maximum = math.ceil(max(means["per_level"] + means["joint"]) * 10) / 10
    points_maximum = POINTS_TICK_STEP * math.ceil(convert_to_percentage(value=max(leads + margins)) / POINTS_TICK_STEP)

    # Launches of the selections, the mean mAP curve of each, and the lead and margin curves in percentage points
    return {"proposal_selection.compared_iteration": COMPARED_ITERATION,
            "proposal_selection.launches_per_selection": len(SELECTION_LAUNCHES["joint"]),
            "proposal_selection.joint_launches": list(SELECTION_LAUNCHES["joint"].values()),
            "proposal_selection.per_level_launches": list(SELECTION_LAUNCHES["per_level"].values()),
            "proposal_selection.joint_map_points": [[iteration, round_fact(value=value)]
                                                    for iteration, value in zip(iterations, means["joint"])],
            "proposal_selection.per_level_map_points": [[iteration, round_fact(value=value)]
                                                        for iteration, value in zip(iterations, means["per_level"])],
            "proposal_selection.lead_points_curve": [[iteration, round_fact(value=convert_to_percentage(value=lead))]
                                                     for iteration, lead in zip(iterations, leads)],
            "proposal_selection.margin_points_curve": [
                [iteration, round_fact(value=convert_to_percentage(value=margin))]
                for iteration, margin in zip(iterations, margins)],
            # Values at the compared iteration, the largest lead and the first evaluation won beyond the margin
            "proposal_selection.joint_map": round_fact(value=means["joint"][-1]),
            "proposal_selection.per_level_map": round_fact(value=means["per_level"][-1]),
            "proposal_selection.lead": round_fact(value=leads[-1]),
            "proposal_selection.margin": round_fact(value=margins[-1]),
            "proposal_selection.maximum_lead": round_fact(value=max(leads)),
            "proposal_selection.maximum_lead_iteration": iterations[leads.index(max(leads))],
            "proposal_selection.first_win_iteration": iterations[first_win_index],
            # Ranges and ticks of the iteration, mAP and percentage point axes of the two curves
            "proposal_selection.iteration_range": [0, COMPARED_ITERATION],
            "proposal_selection.iteration_ticks": make_axis_ticks(maximum=COMPARED_ITERATION, step=ITERATION_TICK_STEP,
                                                                  in_percent=False),
            "proposal_selection.map_range": [0, map_maximum],
            "proposal_selection.map_ticks": make_axis_ticks(maximum=map_maximum, step=map_maximum / 3,
                                                            in_percent=True),
            "proposal_selection.points_range": [0, points_maximum],
            "proposal_selection.points_ticks": make_axis_ticks(maximum=points_maximum, step=POINTS_TICK_STEP,
                                                               in_percent=False)}


def describe_inference_preview(experiments_folder: Path) -> Dict[str, Any]:
    """
    Gives the test mAP of the weights of a joint launch evaluated with the per-level selection at inference only,
    against the same weights with the joint selection, so that the narration can tell the gain of the selection alone
    from the gain of training with it.

    Args:
        experiments_folder (Path): The experiments folder of the build.

    Returns:
        Dict[str, Any]: Both values and the gain, named proposal_selection.preview_*.

    Raises:
        KeyError: If either evaluation has no mAP at the compared iteration.
    """
    # mAP of the averaged weights at the compared iteration, evaluated with each selection
    preview_values = {}
    for value_name, run_key in [("per_level", PREVIEW_RUN_KEY), ("joint", PREVIEW_REFERENCE_RUN_KEY)]:
        run_folder = find_run_folder(experiments_folder=experiments_folder, run_key=run_key)
        curve = read_test_curves(test_folder=run_folder / "Tensorboard" / "Test_EMA").get(MAP_TAG, {})
        if COMPARED_ITERATION not in curve:
            raise KeyError(f"No {MAP_TAG} at iteration {COMPARED_ITERATION} in the averaged weights of {run_folder}.")
        preview_values[value_name] = curve[COMPARED_ITERATION]
    gain = preview_values["per_level"] - preview_values["joint"]
    return {"proposal_selection.preview_joint_map": round_fact(value=preview_values["joint"]),
            "proposal_selection.preview_per_level_map": round_fact(value=preview_values["per_level"]),
            "proposal_selection.preview_gain": round_fact(value=gain)}


def make_proposal_selection_facts(experiments_folder: Path) -> Dict[str, Any]:
    """
    Gathers every fact of the proposal selection scenes of chapter 1 from stage 7 of the detection benchmark: the
    levels and counts of its detector, the measured proposals and recall of both selections, their training curves
    and the inference-only preview.

    Args:
        experiments_folder (Path): The experiments folder of the build, next to which the benchmark folders lie.

    Returns:
        Dict[str, Any]: The facts, named proposal_selection.*.
    """
    # Run folders of the launches of every selection, the detector levels read from the first per-level launch
    run_folders = {selection_name: [find_run_folder(experiments_folder=experiments_folder, run_key=run_key)
                                    for run_key in launches]
                   for selection_name, launches in SELECTION_LAUNCHES.items()}
    level_facts = describe_benchmark_levels(run_folder=run_folders["per_level"][0])
    return {**level_facts,
            **describe_measured_proposals(
                run_folders=run_folders, level_names=level_facts["proposal_selection.level_names"],
                post_nms_proposals=level_facts["proposal_selection.inference_post_nms"]),
            **describe_training_comparison(results_path=find_repository_file(
                file_key="object_detection_benchmark_results")),
            **describe_inference_preview(experiments_folder=experiments_folder)}
