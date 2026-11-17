"""
Figures and facts of the foundations series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series foundations) at every build, from the real runs of
tutorials/tutorial_sources.yaml and the code of the repository: the listings of the classification run, its training
curves, the learning rate schedule of a configuration, the image size figures, the augmentation examples, the
benchmark charts, the feature maps and squeeze-and-excitation gates of the trained backbone, the launch console of a
deeper backbone started from it, and the values of the code
that no placeholder of a storyboard can read (code_values.py). Every figure family is
a module of this folder, which writes its figures into a subfolder of the output folder (model/, benchmark/, ...) or at
its top and gives its facts, gathered here into the facts.json the storyboards quote as {{foundations.<name>}}.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_repository_file, find_run_folder, merge_facts
from tutorials.figures.foundations.aspect_ratio import make_aspect_ratio_figures
from tutorials.figures.foundations.augmentation_examples import make_augmentation_examples
from tutorials.figures.foundations.backbone_activations import make_backbone_figures
from tutorials.figures.foundations.backbone_transfer import make_backbone_transfer_listing
from tutorials.figures.foundations.benchmark import make_benchmark_figures
from tutorials.figures.foundations.code_values import describe_code_values
from tutorials.figures.foundations.learning_rate_schedule import make_learning_rate_schedule_figure
from tutorials.figures.foundations.listings import make_listings
from tutorials.figures.foundations.training_curves import make_training_curves_figure
from tutorial_builder.figure_context import find_repository_root


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the foundations series into its output folder and gives their facts, the expensive figures
    (the checkpoint inference, the augmentation examples) being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the expensive figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.
    """
    classification_run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="classification")
    run_label = f"{experiments_folder.name}/{classification_run_folder.relative_to(experiments_folder)}"
    return merge_facts(fact_groups=[
        make_listings(output_folder=output_folder, run_folder=classification_run_folder,
                      data_configuration_path=find_repository_file(file_key="data_pipeline_configuration")),
        make_training_curves_figure(output_folder=output_folder, run_folder=classification_run_folder,
                                    run_label=run_label),
        make_learning_rate_schedule_figure(output_folder=output_folder, configuration_path=find_repository_file(
            file_key="learning_rate_schedule_configuration")),
        make_aspect_ratio_figures(output_folder=output_folder),
        make_augmentation_examples(output_folder=output_folder, run_folder=classification_run_folder, force=force),
        make_benchmark_figures(output_folder=output_folder, results_path=find_repository_file(
            file_key="benchmark_results")),
        make_backbone_figures(output_folder=output_folder, run_folder=classification_run_folder, force=force),
        make_backbone_transfer_listing(output_folder=output_folder, run_folder=classification_run_folder),
        describe_code_values(repository_folder=find_repository_root(), configuration_path=find_repository_file(
            file_key="data_pipeline_configuration"))])
