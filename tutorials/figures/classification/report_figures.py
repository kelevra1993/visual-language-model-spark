"""
Report figures of the classification tutorials (the chapter judging and shipping a classifier), drawn at every build by
tutorials/figures/classification/figures.py with the functions of scripts/evaluation/draw_report_figures.py, the script
that redraws the figures of the research report of a kept classifier, on the last evaluated checkpoint of the
classification run of tutorial_sources.yaml (LAST_ITERATION of evaluation_tables.py, the one its standalone evaluation
scores), with the weights and split the script takes by default (DEFAULT_REPORT_SETTINGS): the training curves of the
test balanced accuracy at every evaluation of the training, the kept checkpoint ringed, drawn from the Test logs of
TensorBoard of the run, which the report scene shows, and the accuracy and balanced accuracy the title of the report
confusion matrix writes, computed from the confusion_matrix.csv of the standalone evaluation of that checkpoint
(Checkpoint_Evaluation). The precision-recall figure of the script is not drawn: it scores the test images again from
the class_logits.npz of the evaluation (7.4 MB, above the size limit of a run snapshot file even compressed), so the
scene names it without showing it. A build from the run snapshots reads the same two files, the scalars of the Test
logs and the confusion matrix table, and gives the same figure and facts.
"""

import matplotlib

from pathlib import Path
from typing import Any, Dict

from scripts.evaluation.draw_report_figures import (CONFUSION_MATRIX_TABLE_NAME, DEFAULT_REPORT_SETTINGS,
                                                    TRAINING_CURVES_FIGURE_NAME, draw_training_curves_figure,
                                                    find_evaluation_folder, read_balanced_accuracy_curves,
                                                    read_confusion_matrix)
from tutorials.figures.classification.evaluation_tables import LAST_ITERATION, check_last_iteration
from tutorials.figures.figure_utilities import SANS_FONT_FAMILY, register_fonts, round_fact
from vision_training.evaluation.checkpoint_evaluation import CHECKPOINT_EVALUATION_FOLDER_NAME
from vision_training.evaluation.classification.classification_metrics import compute_metrics_from_confusion_matrix

# Subfolder of the figures folder of the series holding the figure of the report scene
REPORT_FIGURES_FOLDER = "report"


def make_report_figures(output_folder: Path, run_folder: Path) -> Dict[str, Any]:
    """
    Draws the training curves of the report of the kept classifier with the function of draw_report_figures.py and
    gives the values the report scene quotes, so that the scene shows what the script writes for the last evaluated
    checkpoint of the run with its default weights and split.

    Args:
        output_folder (Path): The figures folder of the classification series in the cache of the build.
        run_folder (Path): The folder of the classification run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named report.*: the kept iteration and weights, the accuracy and balanced accuracy
            of its confusion matrix, and the balanced accuracy of the kept checkpoint during the training.

    Raises:
        ValueError: If LAST_ITERATION is not the last evaluated checkpoint of the run (check_last_iteration).
    """
    # Kept checkpoint, the last evaluated one of the run, with the weights and split the script takes by default
    check_last_iteration(run_folder=run_folder)
    iteration = LAST_ITERATION
    weights = DEFAULT_REPORT_SETTINGS["weights"]
    split_name = DEFAULT_REPORT_SETTINGS["split_name"]

    # Accuracy and balanced accuracy written in the title of the report confusion matrix, from the standalone evaluation
    evaluation_folder = find_evaluation_folder(experiment_folder=run_folder,
                                               evaluation_folder_name=CHECKPOINT_EVALUATION_FOLDER_NAME,
                                               iteration=iteration, weights=weights)
    _, confusion_matrix = read_confusion_matrix(table_path=evaluation_folder / CONFUSION_MATRIX_TABLE_NAME)
    metrics = compute_metrics_from_confusion_matrix(confusion_matrix=confusion_matrix)

    # Training curves of the report, drawn by the function of the script in the Inter font of the reports
    report_folder = output_folder / REPORT_FIGURES_FOLDER
    report_folder.mkdir(parents=True, exist_ok=True)
    register_fonts()
    training_curves = read_balanced_accuracy_curves(experiment_folder=run_folder, iteration=iteration, weights=weights)
    with matplotlib.rc_context(rc={"font.family": SANS_FONT_FAMILY}):
        kept_balanced_accuracy = draw_training_curves_figure(
            curves=training_curves, iteration=iteration, weights=weights, split_name=split_name,
            output_path=report_folder / TRAINING_CURVES_FIGURE_NAME)
    return {"report.iteration": iteration, "report.weights": weights,
            "report.accuracy": round_fact(value=metrics["accuracy"]),
            "report.balanced_accuracy": round_fact(value=metrics["balanced_accuracy"]),
            "report.kept_balanced_accuracy": round_fact(value=kept_balanced_accuracy)}
