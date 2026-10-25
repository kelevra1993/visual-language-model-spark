"""
Figures and facts of the sequence labelling (M2M) series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series sequence_labelling) at every build, from the real sequence labelling run of
tutorials/tutorial_sources.yaml, its synthetic changing shapes data and the code of the repository: the data figures
(a test sequence frame by frame, the class balance and the position prior, a padded batch, the labels listings), the
training curves against the references of the task, and the evaluation listing, prediction figure and facts that show
the run to be a degenerate solution. Every figure family is a module of this folder writing into a subfolder of the
output folder, its facts gathered here into the facts.json the storyboards quote as {{sequence_labelling.<name>}}.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts
from tutorials.figures.sequence_labelling.dataset_figures import make_dataset_figures
from tutorials.figures.sequence_labelling.evaluation_figures import make_evaluation_facts
from tutorials.figures.sequence_labelling.training_figures import make_training_figures


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the sequence labelling series into its output folder and gives their facts, the figures
    reading the whole training labels file or the predictions file being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the reused figures are made again.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.
    """
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="sequence_labelling")
    run_label = f"{experiments_folder.name}/{run_folder.relative_to(experiments_folder)}"
    dataset_facts = make_dataset_figures(output_folder=output_folder, run_folder=run_folder, force=force)
    return merge_facts(fact_groups=[
        dataset_facts,
        make_training_figures(output_folder=output_folder, run_folder=run_folder, run_label=run_label,
                              reference_facts=dataset_facts),
        make_evaluation_facts(output_folder=output_folder, run_folder=run_folder, force=force),
        {"run.name": run_folder.name, "run.label": run_label}])
