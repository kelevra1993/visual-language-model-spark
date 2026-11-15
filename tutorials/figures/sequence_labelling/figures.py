"""
Figures and facts of the sequence labelling (M2M) series of the tutorial videos, made by make_figures.py (python -m
tutorials.figures.make_figures --series sequence_labelling) at every build, from the real sequence labelling run of
tutorials/tutorial_sources.yaml (pre-trained backbone, bidirectional LSTM, flatten pooling, balanced class weights),
its synthetic changing shapes data, the results of the sequence labelling benchmark and the code of the repository:
the data figures (a test sequence frame by frame, the class balance and the position prior, a padded batch, the labels
listings), the training curves against the references of the task, and the evaluation listings, prediction figure,
sample image and facts of the run, with the trivial value of every metric and the benchmark launches trained from
scratch it is compared with. Every figure family is a module of this folder writing into a subfolder of the output
folder, its facts gathered here into the facts.json the storyboards quote as {{sequence_labelling.<name>}}, with the
settings the run was trained with, read from its saved configuration. The build stops when the evaluation read is not
the one the chapters describe: the last checkpoint evaluated again with its averaged weights on the test subset.
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

# Run of tutorial_sources.yaml the series describes
RUN_KEY = "sequence_labelling"


def read_run_setting_facts(run_folder: Path) -> Dict[str, Any]:
    """
    Reads the settings the sequence labelling run was trained with from its saved training configuration, the batch,
    sequence, backbone, temporal module, head and loss settings the chapters quote for the run of the series, so that
    the narration describes that run even when the shipped configuration changes.

    Args:
        run_folder (Path): The folder of the sequence labelling run.

    Returns:
        Dict[str, Any]: The facts, named run.*.
    """
    experiment_configuration, model_configuration = read_run_sections(run_folder=run_folder)
    convolutions = model_configuration["Backbone"]["convolutions"]
    temporal_configuration = model_configuration["TemporalModel"]
    head_configuration = model_configuration["ClassificationHead"]
    pretrained_configuration = model_configuration["Backbone"]["pretrained"]
    exponential_moving_average = model_configuration["Optimization"]["exponential_moving_average"]

    # Passes of the temporal module, whose output joins the vectors of every pass
    temporal_directions = BIDIRECTIONAL_DIRECTIONS if temporal_configuration["bidirectional"] else 1
    return {"run.batch_size": experiment_configuration["batch_size"],
            "run.validation_iteration_ratio": experiment_configuration["validation_iteration_ratio"],
            "run.maximum_sequence_length": model_configuration["Data"]["sequence_settings"]["maximum_sequence_length"],
            "run.input_channels": model_configuration["Backbone"]["input_channels"],
            "run.backbone_block_count": len(convolutions),
            "run.temporal_hidden_size": temporal_configuration["hidden_size"],
            "run.temporal_layers": temporal_configuration["number_layers"],
            "run.temporal_bidirectional_features": temporal_directions * temporal_configuration["hidden_size"],
            "run.temporal_type": temporal_configuration["type"],
            "run.temporal_type_upper": temporal_configuration["type"].upper(),
            "run.temporal_bidirectional": bool(temporal_configuration["bidirectional"]),
            "run.temporal_directions": temporal_directions,
            "run.head_hidden_units": head_configuration["fully_connected"][0],
            "run.head_hidden_layers": len(head_configuration["fully_connected"]),
            "run.head_normalization": head_configuration["fully_connected_normalization"],
            "run.global_pooling": head_configuration["global_pooling"],
            "run.class_weighting": head_configuration["class_weighting"],
            "run.pretrained_run": Path(pretrained_configuration["experiment_folder"]).name
            if pretrained_configuration["experiment_folder"] else None,
            "run.pretrained_iteration": pretrained_configuration["iteration"],
            "run.pretrained_frozen": bool(pretrained_configuration["freeze"]),
            "run.pretrained_learning_rate_multiplier": pretrained_configuration["learning_rate_multiplier"],
            "run.averaged_weights": bool(exponential_moving_average["activated"]),
            "run.number_iterations": int(experiment_configuration["number_iterations"]),
            "run.weight_saving_iterations": int(experiment_configuration["weight_saving_iterations"]),
            "run.test_split": Path(experiment_configuration["test_data_folder"]).name}


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the sequence labelling series into its output folder and gives their facts, the figures
    reading the whole training labels file or the predictions file being reused from the previous build unless forced
    or unless the files they read changed.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs.
        force (bool): Whether the reused figures are made again.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names.

    Raises:
        ValueError: If the run or its evaluation used another test split than the subset of the chapters, or if the
            evaluation read is not the last checkpoint evaluated again with the averaged weights.
    """
    run_folder = find_run_folder(experiments_folder=experiments_folder, run_key=RUN_KEY)
    run_label = f"{experiments_folder.name}/{run_folder.relative_to(experiments_folder)}"
    dataset_facts = make_dataset_figures(output_folder=output_folder, run_folder=run_folder, force=force)
    run_setting_facts = read_run_setting_facts(run_folder=run_folder)
    evaluation_facts = make_evaluation_facts(output_folder=output_folder, run_folder=run_folder, force=force)

    # The first chapter says the run was tested on the subset drawn by the split subset script: stop if the run or the
    # evaluation read used another split
    subset_split_name = f"test_{dataset_facts['subset.sequence_count']}"
    if {run_setting_facts["run.test_split"], evaluation_facts["evaluation.test_split"]} != {subset_split_name}:
        raise ValueError(f"The sequence labelling run {run_folder.name} was tested on "
                         f"{run_setting_facts['run.test_split']} and evaluated again on "
                         f"{evaluation_facts['evaluation.test_split']}, the tutorials expect {subset_split_name}.")

    # The evaluation chapter reads the last checkpoint, evaluated again with the per-class report and with the moving
    # average of the weights: stop if the evaluation found is another one
    expected_evaluation = {"evaluation.iteration": run_setting_facts["run.number_iterations"],
                           "evaluation.reevaluated": True, "evaluation.averaged_weights": True}
    differing_names = [fact_name for fact_name, value in expected_evaluation.items()
                       if evaluation_facts[fact_name] != value]
    if differing_names:
        raise ValueError(f"The evaluation of {run_folder.name} read by the tutorials has "
                         f"{', '.join(f'{name} {evaluation_facts[name]}' for name in differing_names)}, the chapters "
                         f"expect {', '.join(f'{name} {expected_evaluation[name]}' for name in differing_names)} "
                         f"(scripts/evaluation/evaluate_checkpoint.py on the last checkpoint).")
    return merge_facts(fact_groups=[
        dataset_facts,
        make_training_figures(output_folder=output_folder, run_folder=run_folder, run_label=run_label,
                              reference_facts=dataset_facts),
        evaluation_facts, run_setting_facts, {"run.name": run_folder.name, "run.label": run_label}])
