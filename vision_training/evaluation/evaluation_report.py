"""
Reporting of the evaluation results of a checkpoint, for every task: scalars written to the Test TensorBoard
writer, lines appended to the Results.txt file of the run and printed, extra files printed as links. Called by the
base trainer after every evaluation.
"""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from vision_training.core.contracts import EvaluationResults
from vision_training.utilities.os_utilities import print_color


def report_evaluation_results(results: EvaluationResults, iteration: int, test_writer: SummaryWriter,
                              results_path: Path) -> None:
    """
    Reports the evaluation of a saved checkpoint on the test split, for every task: the scalars are written to the
    Test TensorBoard writer to follow them over the training, the lines are appended to the Results.txt file of the
    run (one block per evaluated checkpoint) and printed, and the extra files saved by the task are printed as links.

    Args:
        results (EvaluationResults): The lines, scalars and files of the evaluation, computed by the task.
        iteration (int): The training iteration of the evaluated checkpoint.
        test_writer (SummaryWriter): The TensorBoard writer of the test evaluations.
        results_path (Path): The Results.txt file of the run.
    """
    # Write the curves followed over the training
    for scalar_name, scalar_value in results["scalars"].items():
        test_writer.add_scalar(scalar_name, scalar_value, iteration)
    test_writer.flush()

    # Append the block of this checkpoint to the results file of the run
    with open(results_path, "a") as results_file:
        results_file.write("\n".join(results["lines"]) + "\n\n")

    print_color(color="green", output="\n".join(results["lines"]), add_separators=True)
    print_color(color="green", output=f"Results : file://{results_path.absolute()}", indent=1)
    for file_name, file_path in results["files"].items():
        print_color(color="green", output=f"{file_name} : file://{file_path.absolute()}", indent=1)
