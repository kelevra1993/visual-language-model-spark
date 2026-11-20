"""
Facts of the scene of the finer inputs of the mask heads (instance segmentation tutorials, the chapter on scoring and
shipping masks), read at every build by tutorials/figures/instance_segmentation/figures.py from stage 3b of the instance
segmentation benchmark of the repository
(documentation/architecture_benchmarking/instance_segmentation_experiment_plan.md), through its committed results
(instance_segmentation_benchmark_results of tutorials/tutorial_sources.yaml, instance_segmentation_results.csv): the
test mask mAP@[0.5:0.95] of the averaged weights at every evaluation of the
heads of stage 3, of design A (the image branch), of design B (the backbone skips) and of the control (the fusion
convolution alone), averaged over the two launches of each arm as the plan compares them, with the gap between the two
launches of an arm (the noise of a training), the mask head settings of every arm read from the same rows, and the
doubled mask resolution of stage 2 against its reference, which the narration says did not help. Every comparison the
narration states (both designs ahead of the heads of stage 3, their tie, the control behind them, the doubled
resolution not ahead) is checked, so that a changed results file stops the build instead of being described wrongly.
Nothing is drawn: the scene draws the curves and the settings as diagram elements, the axis ticks written as the
percentages of the reports of the repository.
"""

import csv
import json
import math
import statistics

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import find_repository_file, round_fact
from tutorials.figures.object_detection.proposal_selection import make_axis_ticks, read_selection_curves

# Launches of every arm of stage 3b, named as in the results of the benchmark: the heads of stage 3 (the reference of
# the stage), design A, design B and the control
MASK_INPUT_ARMS = {"heads": ["run_7", "run_8"], "image_branch": ["run_9", "run_10"],
                   "backbone_skips": ["run_11", "run_12"], "control": ["run_13", "run_14"]}

# Launches of stage 2: the reference of stages 0 to 2 and the doubled mask resolution
RESOLUTION_ARMS = {"reference": ["run_1", "run_2"], "doubled": ["run_5", "run_6"]}

# Columns of the mask mAP@[0.5:0.95] at every evaluation in the results of the benchmark
MAP_COLUMN_PREFIX = "mask_map_50_95_"

# Columns of the mask head settings in the results, and the names of their facts
SETTING_COLUMNS = {"mask_convolutions": "mask_convolutions", "mask_resolution": "mask_resolution",
                   "mask_pool_size": "pool_size", "mask_image_branch_convolutions": "image_branch_convolutions",
                   "mask_skip_blocks_at_pool_size": "skip_blocks_at_pool_size",
                   "mask_skip_blocks_at_mask_resolution": "skip_blocks_at_mask_resolution",
                   "mask_fusion_convolutions": "fusion_convolutions"}

# Finer inputs every arm turns on (a non-zero image branch, a non-empty list of skips, a non-zero fusion), the design
# the narration describes for it
ARM_INPUTS = {"heads": set(), "image_branch": {"image_branch_convolutions", "fusion_convolutions"},
              "backbone_skips": {"skip_blocks_at_pool_size", "skip_blocks_at_mask_resolution", "fusion_convolutions"},
              "control": {"fusion_convolutions"}}

# Iterations between two ticks of the iteration axis
ITERATION_TICK_STEP = 5000


def read_launch_settings(results_path: Path, launch_names: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Reads the mask head settings of launches of the instance segmentation benchmark from the rows of their averaged
    weights in the committed results, so that the settings the scene shows for every arm are those of the launches
    whose curves it draws.

    Args:
        results_path (Path): The results of the benchmark, instance_segmentation_results.csv.
        launch_names (List[str]): The launches, named as in the results (run_7, run_9...).

    Returns:
        Dict[str, Dict[str, str]]: The settings of every launch, named as the facts of SETTING_COLUMNS, written as the
            results write them ("[5, 32]").

    Raises:
        KeyError: If a launch has no row of averaged weights in the results.
    """
    # Rows of the averaged weights of the launches, their mask head settings renamed as the facts
    settings = {}
    with results_path.open(encoding="utf-8", newline="") as results_file:
        for row in csv.DictReader(results_file):
            if row["launch"] in launch_names and row["evaluated_weights"] == "ema":
                settings[row["launch"]] = {fact_name: row[column] for column, fact_name in SETTING_COLUMNS.items()}
    missing_launches = sorted(set(launch_names) - set(settings))
    if missing_launches:
        raise KeyError(f"No averaged weights row for {missing_launches} in {results_path}.")
    return settings


def describe_arm_settings(launch_settings: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """
    Gives the mask head settings of every arm of stage 3b, checking that the two launches of an arm share them, that
    every arm turns on exactly the finer inputs of its design (ARM_INPUTS), and that the arms share their convolutions,
    pool, mask resolution and fusion convolutions, so that the arms differ only by the inputs the scene names.

    Args:
        launch_settings (Dict[str, Dict[str, str]]): The settings of every launch of MASK_INPUT_ARMS.

    Returns:
        Dict[str, Any]: The convolutions of the heads, the fusion convolutions and the inputs of the designs, named
            mask_inputs.*.

    Raises:
        ValueError: If the launches of an arm differ, an arm turns on other inputs than its design, or the arms differ
            in a shared setting.
    """
    arm_settings = {}
    for arm_name, launch_names in MASK_INPUT_ARMS.items():
        # Settings of the arm, the same in its two launches
        first_settings = launch_settings[launch_names[0]]
        if any(launch_settings[launch_name] != first_settings for launch_name in launch_names[1:]):
            raise ValueError(f"The launches {launch_names} of the arm {arm_name} have different mask head settings: "
                             f"{[launch_settings[launch_name] for launch_name in launch_names]}.")

        # Finer inputs turned on: a list of skips holding a block, a list of convolutions holding a layer
        turned_on = {fact_name for fact_name in ARM_INPUTS["image_branch"] | ARM_INPUTS["backbone_skips"]
                     if any(json.loads(first_settings[fact_name]))}
        if turned_on != ARM_INPUTS[arm_name]:
            raise ValueError(f"The arm {arm_name} turns on {sorted(turned_on)}, where the scene describes "
                             f"{sorted(ARM_INPUTS[arm_name])}.")
        arm_settings[arm_name] = first_settings

    # Settings the arms share, then the inputs of the designs
    shared_names = ["mask_convolutions", "pool_size", "mask_resolution"]
    for setting_name in shared_names:
        values = {arm_name: settings[setting_name] for arm_name, settings in arm_settings.items()}
        if len(set(values.values())) != 1:
            raise ValueError(f"The arms of stage 3b differ in {setting_name}: {values}.")
    fusion_values = {arm_name: arm_settings[arm_name]["fusion_convolutions"]
                     for arm_name in ["image_branch", "backbone_skips", "control"]}
    if len(set(fusion_values.values())) != 1:
        raise ValueError(f"The designs and the control of stage 3b differ in their fusion convolutions: "
                         f"{fusion_values}.")
    return {"mask_inputs.mask_convolutions": arm_settings["heads"]["mask_convolutions"],
            "mask_inputs.fusion_convolutions": fusion_values["control"],
            "mask_inputs.image_branch_convolutions": arm_settings["image_branch"]["image_branch_convolutions"],
            "mask_inputs.skip_blocks_at_pool_size": arm_settings["backbone_skips"]["skip_blocks_at_pool_size"],
            "mask_inputs.skip_blocks_at_mask_resolution":
                arm_settings["backbone_skips"]["skip_blocks_at_mask_resolution"]}


def summarize_arms(curves: Dict[str, Dict[int, float]], arms: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """
    Averages the mask mAP of the launches of every arm at every evaluation and gives, at the last evaluation shared by
    all the launches, the mean of every arm and the gap between its launches, the way the plan compares the arms.

    Args:
        curves (Dict[str, Dict[int, float]]): The mask mAP at every evaluated iteration of every launch.
        arms (Dict[str, List[str]]): The launches of every arm.

    Returns:
        Dict[str, Dict[str, Any]]: For every arm, its mean curve as [iteration, value] points, its final mean, its
            launch gap and the compared iteration.

    Raises:
        ValueError: If the launches were not evaluated at the same iterations.
    """
    # Iterations evaluated by every launch, the same for all
    launch_names = [launch_name for launches in arms.values() for launch_name in launches]
    iterations = sorted(curves[launch_names[0]])
    if any(sorted(curves[launch_name]) != iterations for launch_name in launch_names):
        raise ValueError(f"The launches {launch_names} were not evaluated at the same iterations.")

    # Mean curve, final mean and launch gap of every arm
    summaries = {}
    for arm_name, launches in arms.items():
        means = [statistics.fmean(curves[launch_name][iteration] for launch_name in launches)
                 for iteration in iterations]
        final_values = [curves[launch_name][iterations[-1]] for launch_name in launches]
        summaries[arm_name] = {"points": [[iteration, round_fact(value=value)]
                                          for iteration, value in zip(iterations, means)],
                               "final": means[-1], "gap": max(final_values) - min(final_values),
                               "iteration": iterations[-1]}
    return summaries


def check_benchmark_comparisons(mask_input_arms: Dict[str, Dict[str, Any]],
                               resolution_arms: Dict[str, Dict[str, Any]]) -> None:
    """
    Checks the comparisons the narration of the scene states, each against the launch gaps of the arms compared: the
    doubled mask resolution not ahead of its reference, both designs ahead of the heads of stage 3, the two designs
    tied, and the control ahead of the heads by less than the designs are ahead of the control.

    Args:
        mask_input_arms (Dict[str, Dict[str, Any]]): The summaries of the arms of stage 3b (summarize_arms).
        resolution_arms (Dict[str, Dict[str, Any]]): The summaries of the arms of stage 2.

    Raises:
        ValueError: If one of the comparisons does not hold.
    """
    # The doubled resolution did not help: not ahead of the reference beyond the larger launch gap
    resolution_margin = max(arm["gap"] for arm in resolution_arms.values())
    if resolution_arms["doubled"]["final"] - resolution_arms["reference"]["final"] > resolution_margin:
        raise ValueError(f"The doubled mask resolution gains beyond its launch gaps ({resolution_arms}), where the "
                         f"scene says it did not help.")

    # Both designs ahead of the heads beyond the launch gaps, and tied with each other
    heads, image_branch = mask_input_arms["heads"], mask_input_arms["image_branch"]
    backbone_skips, control = mask_input_arms["backbone_skips"], mask_input_arms["control"]
    for design_name, design in [("image_branch", image_branch), ("backbone_skips", backbone_skips)]:
        if design["final"] - heads["final"] <= max(design["gap"], heads["gap"]):
            raise ValueError(f"The design {design_name} is not ahead of the heads of stage 3 beyond the launch gaps "
                             f"({design}, {heads}), as the scene says.")
    if abs(image_branch["final"] - backbone_skips["final"]) > max(image_branch["gap"], backbone_skips["gap"]):
        raise ValueError(f"The designs are not tied within their launch gaps ({image_branch}, {backbone_skips}), as "
                         f"the scene says.")

    # The fusion convolution alone gains less than the finer inputs add on top of it
    finer_input_gain = min(image_branch["final"], backbone_skips["final"]) - control["final"]
    if control["final"] - heads["final"] >= finer_input_gain:
        raise ValueError(f"The control gains as much as the finer inputs add to it ({control}, {heads}, "
                         f"{image_branch}, {backbone_skips}), where the scene says the finer inputs carry the gain.")


def describe_mask_input_benchmark() -> Dict[str, Any]:
    """
    Gathers every fact of the scene of the finer inputs of the mask heads from the committed results of the instance
    segmentation benchmark: the mean curves of the four arms of stage 3b, their final values, gains and launch gaps,
    their mask head settings, the doubled mask resolution of stage 2 against its reference, and the axes of the curve.

    Returns:
        Dict[str, Any]: The facts, named mask_inputs.*, the rates and their differences between 0 and 1.
    """
    # Curves and settings of the launches of stages 2 and 3b, read from the committed results
    results_path = find_repository_file(file_key="instance_segmentation_benchmark_results")
    mask_input_launches = [launch_name for launches in MASK_INPUT_ARMS.values() for launch_name in launches]
    resolution_launches = [launch_name for launches in RESOLUTION_ARMS.values() for launch_name in launches]
    curves = read_selection_curves(results_path=results_path, launch_names=mask_input_launches + resolution_launches,
                                   column_prefix=MAP_COLUMN_PREFIX)
    mask_input_arms = summarize_arms(curves=curves, arms=MASK_INPUT_ARMS)
    resolution_arms = summarize_arms(curves=curves, arms=RESOLUTION_ARMS)
    check_benchmark_comparisons(mask_input_arms=mask_input_arms, resolution_arms=resolution_arms)
    setting_facts = describe_arm_settings(launch_settings=read_launch_settings(
        results_path=results_path, launch_names=mask_input_launches))
    resolution_settings = read_launch_settings(results_path=results_path, launch_names=resolution_launches)

    # Mean curve, final value and launch gap of every arm, and the gains of the designs and of the control
    facts = {"mask_inputs.compared_iteration": mask_input_arms["heads"]["iteration"],
             "mask_inputs.launches_per_arm": len(MASK_INPUT_ARMS["heads"])}
    for arm_name, summary in mask_input_arms.items():
        facts.update({f"mask_inputs.{arm_name}_points": summary["points"],
                      f"mask_inputs.{arm_name}_map": round_fact(value=summary["final"]),
                      f"mask_inputs.{arm_name}_launch_gap": round_fact(value=summary["gap"])})
    for arm_name in ["image_branch", "backbone_skips", "control"]:
        facts[f"mask_inputs.{arm_name}_gain"] = round_fact(
            value=mask_input_arms[arm_name]["final"] - mask_input_arms["heads"]["final"])
    for design_name in ["image_branch", "backbone_skips"]:
        facts[f"mask_inputs.{design_name}_over_control"] = round_fact(
            value=mask_input_arms[design_name]["final"] - mask_input_arms["control"]["final"])
    facts["mask_inputs.design_difference"] = round_fact(
        value=mask_input_arms["image_branch"]["final"] - mask_input_arms["backbone_skips"]["final"])

    # Stage 2: the mask resolutions of the reference and of the doubled resolution, and their final values
    for arm_name, launches in RESOLUTION_ARMS.items():
        facts.update({f"mask_inputs.{arm_name}_resolution": int(resolution_settings[launches[0]]["mask_resolution"]),
                      f"mask_inputs.{arm_name}_resolution_map": round_fact(value=resolution_arms[arm_name]["final"])})

    # Axes of the curve: the iterations from 0, the mAP up to the next tenth in three ticks labelled in percent
    map_maximum = math.ceil(max(summary["final"] for summary in mask_input_arms.values()) * 10) / 10
    facts.update({"mask_inputs.iteration_range": [0, facts["mask_inputs.compared_iteration"]],
                  "mask_inputs.iteration_ticks": make_axis_ticks(maximum=facts["mask_inputs.compared_iteration"],
                                                                 step=ITERATION_TICK_STEP, in_percent=False),
                  "mask_inputs.map_range": [0, map_maximum],
                  "mask_inputs.map_ticks": make_axis_ticks(maximum=map_maximum, step=map_maximum / 3,
                                                           in_percent=True)})
    return {**facts, **setting_facts}
