"""
Figures and facts of the sequence comparison (CMP) series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series sequence_comparison) at every build, from the README run of
tutorials/tutorial_sources.yaml, its object loss pairs dataset and the code of the repository: the dataset figures and
listings, the animated pair batch encoded by the trained comparator on the CPU, the sizes of the components of the
comparator, and the training and evaluation figures. Every figure family is a module of this folder writing into a
subfolder of the output folder (dataset/, batch/, evaluation/, ...), its facts gathered here into the facts.json the
storyboards quote as {{sequence_comparison.<name>}}. The helpers are those of the sequence classification series.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts, read_run_configuration
from tutorials.figures.sequence_classification.sequence_figure_utilities import (add_number_words, find_console_log,
                                                                                 load_sequence_model)
from tutorials.figures.sequence_comparison.pair_batch_animation import make_pair_batch_animation
from tutorials.figures.sequence_comparison.pair_dataset_figures import make_pair_dataset_figures
from tutorials.figures.sequence_comparison.pair_evaluation_figures import make_pair_evaluation_figures
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.models.task_model import TaskModel
from vision_training.trainers.checkpointing import get_last_checkpoint_iteration

# Run of the series in tutorial_sources.yaml
RUN_KEY = "sequence_comparison"


def count_component_parameters(model: TaskModel) -> Dict[str, Any]:
    """
    Counts the parameters of every component of the trained comparator, and gives the features of its member and
    pair vectors, the sizes the comparator scenes quote.

    Args:
        model (TaskModel): The trained comparator.

    Returns:
        Dict[str, Any]: The facts, named parameters.* and features.*.
    """
    facts = {}
    for component_name in ["backbone", "temporal_module", "member_fusion", "classification_head"]:
        facts[f"parameters.{component_name}"] = sum(parameter.numel() for parameter in
                                                    getattr(model, component_name).parameters())
    facts["parameters.total"] = sum(parameter.numel() for parameter in model.parameters())
    facts["parameters.total_megabytes"] = round(facts["parameters.total"] * 4 / 1e6, 2)
    facts["features.member"] = int(model.temporal_module.output_features)
    facts["features.pair"] = int(model.member_fusion.output_features)
    return facts


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the sequence comparison series into its output folder and gives their facts, the encoding
    of the animated batch being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the expensive figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names, with the words of their small
            whole numbers.
    """
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key=RUN_KEY)
    log_path = find_console_log(experiments_folder=experiments_folder, run_key=RUN_KEY, comparison_key=None)
    iteration = get_last_checkpoint_iteration(weights_directory=run_folder / "Weights")
    run_configuration = read_run_configuration(run_folder=run_folder)
    experiment_configuration = run_configuration["ExperimentConfiguration"]
    _, model_configuration = load_experiment_configuration(configuration_path=run_folder /
                                                           "training_configuration.yaml")
    label_dictionary = model_configuration["Data"]["label_dictionary"]
    model = load_sequence_model(run_folder=run_folder, iteration=iteration)
    train_folder = Path(experiment_configuration["train_data_folder"])
    facts = {"run.name": run_folder.name, "run.batch_size": experiment_configuration["batch_size"],
             "run.image_size": run_configuration["Data"]["image_settings"]["size"], "run.iteration": iteration}
    return add_number_words(facts=merge_facts(fact_groups=[
        facts, count_component_parameters(model=model),
        make_pair_dataset_figures(output_folder=output_folder, train_folder=train_folder,
                                  label_dictionary=label_dictionary, batch_size=experiment_configuration["batch_size"]),
        make_pair_batch_animation(output_folder=output_folder, model=model, train_folder=train_folder,
                                  label_dictionary=label_dictionary,
                                  image_size=run_configuration["Data"]["image_settings"]["size"], force=force),
        make_pair_evaluation_figures(output_folder=output_folder, run_folder=run_folder, log_path=log_path,
                                     iteration=iteration,
                                     test_folder=Path(experiment_configuration["test_data_folder"]))]))
