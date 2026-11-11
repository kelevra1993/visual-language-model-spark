"""
Figures and facts of the sequence labelling (M2M) series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series sequence_labelling) at every build, from the real sequence labelling run of
tutorials/tutorial_sources.yaml, its synthetic changing shapes data and the code of the repository: the data figures
(a test sequence frame by frame, the class balance and the position prior, a padded batch, the labels listings), the
training curves against the references of the task, and the evaluation listing, prediction figure and facts that show
the run to be a degenerate solution. Every figure family is a module of this folder writing into a subfolder of the
output folder, its facts gathered here into the facts.json the storyboards quote as {{sequence_labelling.<name>}},
with the settings the run was trained with, read from its saved configuration.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts
from tutorials.figures.sequence_labelling.dataset_figures import make_dataset_figures
from tutorials.figures.sequence_labelling.evaluation_figures import make_evaluation_facts
from tutorials.figures.sequence_labelling.sequence_utilities import read_run_sections
from tutorials.figures.sequence_labelling.training_figures import make_training_figures

# Directions of a bidirectional recurrent module, whose output joins the vectors of its forward and backward passes
BIDIRECTIONAL_DIRECTIONS = 2


def read_run_setting_facts(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the settings the sequence labelling run was trained with from its saved training configuration, the batch,
    sequence, backbone, temporal module and head settings the chapter on the model quotes for the README run, so that
    the narration describes that run even when the shipped configuration changes.

    Args:
        run_folder (Path): The folder of the sequence labelling run.

    Returns:
        Dict[str, Any]: The facts, named run.*.
    """
    experiment_configuration, model_configuration = read_run_sections(run_folder=run_folder)
    convolutions = model_configuration["Backbone"]["convolutions"]
    temporal_configuration = model_configuration["TemporalModel"]
    return {"run.batch_size": experiment_configuration["batch_size"],
            "run.validation_iteration_ratio": experiment_configuration["validation_iteration_ratio"],
            "run.maximum_sequence_length": model_configuration["Data"]["sequence_settings"]["maximum_sequence_length"],
            "run.input_channels": model_configuration["Backbone"]["input_channels"],
            "run.backbone_block_count": len(convolutions),
            "run.temporal_hidden_size": temporal_configuration["hidden_size"],
            "run.temporal_layers": temporal_configuration["number_layers"],
            "run.temporal_bidirectional_features": BIDIRECTIONAL_DIRECTIONS * temporal_configuration["hidden_size"],
            "run.head_hidden_units": model_configuration["ClassificationHead"]["fully_connected"][0]}


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
        read_run_setting_facts(run_folder=run_folder),
        {"run.name": run_folder.name, "run.label": run_label}])
