"""
Figures and facts of the classification series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series classification) at every build, from the real runs of
tutorials/tutorial_sources.yaml and the code of the repository: what is specific to the classification head, the
sample and error images of the evaluations of the classification run, its evaluation tables (confusion matrices,
per-class metrics and thresholds, headline metrics), the confidence of its predictions, its learning rate variants,
the listings of its data files, the measured effects of the head settings in the architecture benchmark, and the
outputs of its trained classifier (the logits and probabilities of a test star, the correlation of the hidden
features of the head, the balanced class weights and the parameters of every block and head layer), with the run name
tokens of the head section of the synthetic classification configuration (data_pipeline_configuration of
tutorial_sources.yaml, the file the head scene shows). Every figure family is a module of this folder, which writes
its figures into a subfolder of the output folder (evaluation/, model/, training/, listings/) and gives its facts,
gathered here into the facts.json the storyboards quote as {{classification.<name>}}.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.classification.classification_head import (describe_head_run_name_tokens,
                                                                   make_classification_head_figures)
from tutorials.figures.classification.data_listings import make_data_listings
from tutorials.figures.classification.evaluation_images import make_evaluation_figures
from tutorials.figures.classification.evaluation_tables import make_evaluation_tables
from tutorials.figures.classification.head_benchmark import make_head_benchmark_facts
from tutorials.figures.classification.learning_rates import make_learning_rate_figure
from tutorials.figures.classification.prediction_confidence import make_prediction_confidence_figure
from tutorials.figures.figure_utilities import find_repository_file, find_run_folder, merge_facts


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the classification series into its output folder and gives their facts, the checkpoint
    inference being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the expensive figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.
    """
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="classification")
    evaluation_folder = find_run_folder(experiments_folder=experiments_folder, run_key="classification_evaluation")
    return merge_facts(fact_groups=[
        make_evaluation_figures(output_folder=output_folder, run_folder=run_folder,
                                evaluation_folder=evaluation_folder),
        make_evaluation_tables(run_folder=run_folder, evaluation_folder=evaluation_folder),
        make_prediction_confidence_figure(output_folder=output_folder, run_folder=run_folder),
        make_learning_rate_figure(output_folder=output_folder, experiments_folder=experiments_folder,
                                  run_folder=run_folder),
        make_data_listings(output_folder=output_folder, run_folder=run_folder),
        make_head_benchmark_facts(),
        make_classification_head_figures(output_folder=output_folder, run_folder=run_folder, force=force),
        describe_head_run_name_tokens(configuration_path=find_repository_file(file_key="data_pipeline_configuration"),
                                      run_folder=run_folder)])
