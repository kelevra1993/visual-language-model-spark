"""
Figures and facts of the self-supervised series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series self_supervised) at every build, from the SimCLR run of
tutorials/tutorial_sources.yaml (key self_supervised) and the code of the repository: the two views of sample images
and the augmentations drawn for them, the listing of the configuration checks, the training curves and the first
validations, the test metrics across the checkpoints with the per-class F1 and the last Results.txt block, and the
representations of the checkpoints rebuilt on the CPU (k-NN neighbourhood of a star, InfoNCE matrix of the views, the
colour cue both views share). Every figure family is a module of this folder, which writes its figures into a subfolder
of the output folder and gives its facts, gathered here into the facts.json the storyboards quote as
{{self_supervised.<name>}}.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts
from tutorials.figures.self_supervised.configuration_checks import make_check_listing
from tutorials.figures.self_supervised.evaluation_figures import make_evaluation_figures
from tutorials.figures.self_supervised.representation_figures import make_representation_figures
from tutorials.figures.self_supervised.ssl_sources import RUN_KEY, describe_run_settings
from tutorials.figures.self_supervised.training_figures import make_training_figures
from tutorials.figures.self_supervised.view_figures import make_view_figures


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the self-supervised series into its output folder and gives their facts, the checkpoint
    inference being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the expensive figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.
    """
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key=RUN_KEY)
    facts = merge_facts(fact_groups=[
        make_view_figures(output_folder=output_folder, run_folder=run_folder),
        make_check_listing(output_folder=output_folder, run_folder=run_folder),
        make_training_figures(output_folder=output_folder, run_folder=run_folder),
        make_evaluation_figures(output_folder=output_folder, run_folder=run_folder),
        make_representation_figures(output_folder=output_folder, run_folder=run_folder, force=force),
        describe_run_settings(run_folder=run_folder)])
    facts["run.name"] = run_folder.name
    return facts
