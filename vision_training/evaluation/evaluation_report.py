"""
Reporting of the evaluation results of a checkpoint, for every task: the block of the report printed in the console and
appended to the Results.txt file of the run (the same layout, see report_formatting.py), the scalars and images written
to the Test TensorBoard writer (the sample images at every evaluation), the extra files printed as links, the per-class
outputs only with the per-class report, and the per-class metrics file saved at every evaluation. Called by the base
trainer after every evaluation and by the trainers saving their per-class metrics.
"""

import csv

from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List

from vision_training.core.contracts import EvaluationResults
from vision_training.evaluation.report_formatting import (TIMESTAMP_FORMAT, build_report_renderables, format_file_value,
                                                          print_report, render_report_text)
from vision_training.utilities.os_utilities import format_path_link, print_color


def write_class_metrics_file(class_rows: List[Dict[str, Any]], evaluation_folder: Path,
                             file_name: str = "class_metrics.csv") -> Path:
    """
    Saves the metrics of every class of an evaluated checkpoint to 'class_metrics.csv' in its evaluation folder, at
    every evaluation whatever the per-class report, so that the details of every class stay available on the disk
    even for runs with many classes: one row per class in class index order, floats with 6 decimals, missing values
    as empty cells. The other metric tables of the detection tasks are saved the same way under their own name: the
    proposal recall of a checkpoint (proposal_recall.csv) and the anchor recall of the run (anchor_recall.csv, in the
    run folder).

    Args:
        class_rows (List[Dict[str, Any]]): The metrics of every class (or of every row of the table), keyed by column
            name, in class index order.
        evaluation_folder (Path): The folder of the file, the evaluation folder of the checkpoint,
            '<run>/Evaluation/Iteration_<n>/', or the run folder for the anchor recall.
        file_name (str): The name of the file. Defaults to "class_metrics.csv".

    Returns:
        Path: The written file.
    """
    evaluation_folder.mkdir(parents=True, exist_ok=True)
    class_metrics_path = evaluation_folder / file_name
    with open(file=class_metrics_path, mode="w", newline="") as class_metrics_file:
        writer = csv.DictWriter(f=class_metrics_file, fieldnames=list(class_rows[0]) if class_rows else [])
        writer.writeheader()
        writer.writerows([{column_name: format_file_value(value=value) for column_name, value in class_row.items()}
                          for class_row in class_rows])
    return class_metrics_path


def report_evaluation_results(results: EvaluationResults, iteration: int, test_writer: SummaryWriter,
                              results_path: Path, per_class_report: bool) -> None:
    """
    Reports the evaluation of a saved checkpoint on the test split, for every task: the global scalars and the sample
    images are written to the Test TensorBoard writer to follow them over the training, with the scalars and images of
    every class when the per-class report is on, then the block of the checkpoint (header with the iteration, the
    evaluated samples and the time, global metrics, notes and tables) is appended to the Results.txt file of the run
    and printed with the same layout, the per-class tables only with the per-class report, and the extra files saved
    by the task are printed as links.

    Args:
        results (EvaluationResults): The results of the evaluation, computed by the task.
        iteration (int): The training iteration of the evaluated checkpoint.
        test_writer (SummaryWriter): The TensorBoard writer of the test evaluations.
        results_path (Path): The Results.txt file of the run.
        per_class_report (bool): Whether the per-class outputs (tables, scalars and images of every class) are
            reported, or only the global ones.
    """
    # Curves followed over the training: the global scalars and the sample images, and with the per-class report the
    # scalars of every class and the per-class images (e.g. the confusion matrix)
    for scalar_name, scalar_value in results["scalars"].items():
        test_writer.add_scalar(tag=scalar_name, scalar_value=scalar_value, global_step=iteration)
    for image_name, image in results["sample_images"].items():
        test_writer.add_image(tag=image_name, img_tensor=image, global_step=iteration, dataformats="HWC")
    if per_class_report:
        for scalar_name, scalar_value in results["class_scalars"].items():
            test_writer.add_scalar(tag=scalar_name, scalar_value=scalar_value, global_step=iteration)
        for image_name, image in results["images"].items():
            test_writer.add_image(tag=image_name, img_tensor=image, global_step=iteration, dataformats="HWC")
    test_writer.flush()

    # Block of this checkpoint, appended to the results file of the run as plain text and printed in colour
    renderables = build_report_renderables(results=results, iteration=iteration,
                                           timestamp=datetime.now().strftime(TIMESTAMP_FORMAT),
                                           per_class_report=per_class_report)
    with open(file=results_path, mode="a", encoding="utf-8") as results_file:
        results_file.write(render_report_text(renderables=renderables) + "\n")
    print_report(renderables=renderables)

    # Links to the results file and to the files saved by the task
    print_color(color="green", output=f"Results : {format_path_link(path=results_path)}", indent=1)
    for file_name, file_path in results["files"].items():
        print_color(color="green", output=f"{file_name} : {format_path_link(path=file_path)}", indent=1)
