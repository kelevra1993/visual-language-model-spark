"""
Evaluation curves of a run for the evaluation scene of the last foundations chapter: the evaluation_curves.png that
every training and every standalone evaluation draws next to its Results.txt (BaseTrainer.update_evaluation_curves),
drawn again by the code of the repository (draw_summary_curves of vision_training/evaluation/evaluation_curves.py) from
the evaluation summary of the detection run of tutorial_sources.yaml, the run of the series whose trained and averaged
weights are both evaluated, so that the scene shows the two curves, their best iterations ringed and the windows of
the moving average shaded. The summary and the configuration of the run are read from the real run or from the
committed run snapshots (snapshot_files.yaml of this folder), and the facts (best iterations and values, windows of
the averaged weights) are given by the same functions as the figure, so that the narration and the drawing agree.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import read_run_configuration, round_fact
from vision_training.evaluation.evaluation_curves import (EVALUATION_CURVES_FIGURE_NAME, EVALUATION_SUMMARY_FILE_NAME,
                                                          describe_averaging_windows, draw_summary_curves,
                                                          find_best_iterations, read_evaluation_summary,
                                                          read_moving_average_state, read_summary_curves)

# Headline metric of the detection task and whether the report writes it as a rate, the 'headline_metric' and
# 'percent_metrics' of the results of DetectionTrainer (vision_training/trainers/detection_trainer.py), which the
# trainer gives draw_summary_curves after every evaluation
DETECTION_HEADLINE_METRIC = "mAP@0.5/area"
DETECTION_HEADLINE_METRIC_IN_PERCENT = True

# Name of the facts of every kind of weights, by its key in the curves of the evaluation summary
WEIGHTS_FACT_NAMES = {"trained": "trained", "ema": "averaged"}


def make_evaluation_curves_figure(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Draws the evaluation curves of the detection run into evaluation_curves/ of the foundations figures, as its trainer
    drew them after its last evaluation, from the evaluation summary of its Evaluation folder and the moving average of
    its saved configuration, and gives the facts the evaluation scene quotes.

    Args:
        output_folder (Path): The figures folder of the foundations series.
        run_folder (Path): The folder of the detection run of tutorial_sources.yaml, real or read from the snapshots.

    Returns:
        Dict[str, Any]: The facts: the number of evaluations, the best iteration and value of the trained and of the
            averaged weights, and the configured window and the window of the last update of the averaged weights.

    Raises:
        ValueError: If the summary lacks a curve of the trained or averaged weights, if the run has no moving average
            evaluated with its number of updates, or if its last averaged weights are past the warm-up of the decay
            (a single window shaded), the scene describing the two windows of the warm-up.
    """
    summary_path = run_folder / "Evaluation" / EVALUATION_SUMMARY_FILE_NAME
    figure_path = output_folder / "evaluation_curves" / EVALUATION_CURVES_FIGURE_NAME

    # Decay of the moving average of the run, None when it was off, as the trainer passes it
    moving_average_configuration = read_run_configuration(run_folder=run_folder)["Optimization"][
        "exponential_moving_average"]
    moving_average_decay = (float(moving_average_configuration["decay"]) if moving_average_configuration["activated"]
                            else None)
    draw_summary_curves(summary_path=summary_path, figure_path=figure_path, metric_name=DETECTION_HEADLINE_METRIC,
                        in_percent=DETECTION_HEADLINE_METRIC_IN_PERCENT, moving_average_decay=moving_average_decay)

    # Curves and averaged weights read as draw_summary_curves reads them, for the facts of the narration
    column_names, summary_rows = read_evaluation_summary(summary_path=summary_path)
    curves = read_summary_curves(column_names=column_names, summary_rows=summary_rows,
                                 metric_name=DETECTION_HEADLINE_METRIC)
    if set(curves) != {"trained", "ema"}:
        raise ValueError(f"Expected the curves of the trained and averaged weights in {summary_path}, got "
                         f"{sorted(curves)}: the evaluation scene describes both.")
    moving_average = read_moving_average_state(averaged_curve=curves["ema"], summary_rows=summary_rows,
                                               moving_average_decay=moving_average_decay)
    if moving_average is None:
        raise ValueError(f"No averaged weights evaluated with their number of updates in {summary_path} (moving "
                         f"average decay {moving_average_decay}): the evaluation scene describes their windows.")

    # Best iteration and value of both weights, the first of tied iterations being the one labelled on the figure
    facts: Dict[str, Any] = {"evaluation_curves.evaluation_count": len(summary_rows)}
    for weights, fact_name in WEIGHTS_FACT_NAMES.items():
        best_iterations, best_value = find_best_iterations(values=curves[weights])
        facts.update({f"evaluation_curves.{fact_name}_best_iteration": best_iterations[0],
                      f"evaluation_curves.{fact_name}_best_value": round_fact(value=best_value)})

    # Windows shaded behind the curves, the configured one first, then the one of the last update while the warm-up
    # of the decay shortens it
    averaging_windows = describe_averaging_windows(decay=moving_average["decay"],
                                                   number_updates=moving_average["number_updates"])
    if len(averaging_windows) != 2:
        raise ValueError(f"The averaged weights of iteration {moving_average['iteration']} in {summary_path} are past "
                         f"the warm-up of the decay, a single window being shaded: reword the evaluation scene, which "
                         f"describes the configured window and the shorter window of the last update.")
    facts.update({"evaluation_curves.configured_window": round(averaging_windows[0]["length"]),
                  "evaluation_curves.last_update_window": round(averaging_windows[1]["length"]),
                  "evaluation_curves.last_averaged_iteration": moving_average["iteration"]})
    return facts
