"""
Figures and facts of the frame selection (SEL) series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series frame_selection) at every build, from the real frame selection run of
tutorials/tutorial_sources.yaml, its synthetic selected shapes data and the code of the repository: the data figures (a
test sequence frame by frame, the labels listings and the error of the selection check), the training curves of the
two losses against the losses of equal logits, the wrong picks of iteration 1000 drawn by the selection image of the
repository with the matrix data of the decoding animations, and the evaluation listing and thresholds figure. Every
figure family is a module of this folder writing into a subfolder of the output folder, its facts gathered here into the
facts.json the storyboards quote as {{frame_selection.<name>}}, with the settings the run was trained with. The run is
read from the experiments folder of the build, the real runs or the committed snapshots (tutorials/run_snapshots).
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import describe_run_label, find_run_folder, merge_facts
from tutorials.figures.frame_selection.dataset_figures import make_dataset_figures
from tutorials.figures.frame_selection.evaluation_figures import make_evaluation_figures
from tutorials.figures.frame_selection.selection_figures import make_selection_figures
from tutorials.figures.frame_selection.training_figures import make_training_figures
from tutorials.figures.sequence_labelling.sequence_utilities import read_run_sections


def read_run_setting_facts(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the settings the frame selection run was trained with from its saved training configuration, those the
    scenes of its model, its run name and its export quote (batch size, input channels, maximum sequence length,
    temporal module, head, frames per class and selection loss weight), so that the narration describes the README
    run even when the shipped configuration changes.

    Args:
        run_folder (Path): The folder of the frame selection run.

    Returns:
        Dict[str, Any]: The facts, named run.*.
    """
    experiment_configuration, model_configuration = read_run_sections(run_folder=run_folder)
    temporal_configuration = model_configuration["TemporalModel"]
    selection_configuration = model_configuration["Selection"]
    return {"run.batch_size": experiment_configuration["batch_size"],
            "run.input_channels": model_configuration["Backbone"]["input_channels"],
            "run.maximum_sequence_length": model_configuration["Data"]["sequence_settings"]["maximum_sequence_length"],
            "run.temporal_number_layers": temporal_configuration["number_layers"],
            "run.temporal_hidden_size": temporal_configuration["hidden_size"],
            "run.head_hidden_units": model_configuration["ClassificationHead"]["fully_connected"][0],
            "run.frames_per_class": selection_configuration["frames_per_class"],
            "run.selection_loss_weight": selection_configuration["selection_loss_weight"]}


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the frame selection series into its output folder and gives their facts, the checkpoint
    inference of the selection figures being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs, or the committed snapshots.
        force (bool): Whether the reused figures are made again.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.
    """
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key="frame_selection")
    run_label = describe_run_label(experiments_folder=experiments_folder, run_folder=run_folder)
    dataset_facts = make_dataset_figures(output_folder=output_folder, run_folder=run_folder)
    return merge_facts(fact_groups=[
        dataset_facts,
        make_training_figures(output_folder=output_folder, run_folder=run_folder, run_label=run_label,
                              reference_facts=dataset_facts),
        make_selection_figures(output_folder=output_folder, run_folder=run_folder, force=force),
        make_evaluation_figures(output_folder=output_folder, run_folder=run_folder),
        read_run_setting_facts(run_folder=run_folder), {"run.name": run_folder.name, "run.label": run_label}])
