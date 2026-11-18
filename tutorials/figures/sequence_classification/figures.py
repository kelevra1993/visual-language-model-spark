"""
Figures and facts of the sequence classification (M2O) series of the tutorial videos, made by make_figures.py (python
-m tutorials.figures.make_figures --series sequence_classification) at every build, from the README runs of
tutorials/tutorial_sources.yaml (the current flatten run at 128 pixels, and the average pooling and flatten runs at 512
pixels it is compared with), their moving shapes dataset and the code of the repository: the dataset figures and
listings, the animated padded batch drawn by the current generator, the model figures computed with the checkpoints on
the CPU, and the training and evaluation figures. Every figure family is a module of this folder writing into a
subfolder of the output folder (dataset/, batch/, model/, evaluation/, ...), its facts gathered here into the
facts.json the storyboards quote as {{sequence_classification.<name>}}. A build from the committed snapshots
(tutorials/run_snapshots, without checkpoints) builds the flatten run from its configuration for the facts of its
structure and reuses the committed outputs of the families computed with the trained weights.
"""

from pathlib import Path
from typing import Any, Dict

from tutorials.figures.figure_utilities import find_run_folder, merge_facts, read_run_configuration
from tutorials.figures.sequence_classification.batch_animation import make_batch_animation
from tutorials.figures.sequence_classification.dataset_figures import make_dataset_figures
from tutorials.figures.sequence_classification.evaluation_figures import make_evaluation_figures
from tutorials.figures.sequence_classification.model_figures import describe_export_input, make_model_figures
from tutorials.figures.sequence_classification.sequence_figure_utilities import (add_number_words,
                                                                                 find_comparison_run_folder,
                                                                                 find_console_log,
                                                                                 load_sequence_model)
from vision_training.configuration_processing.configuration_loader import load_experiment_configuration
from vision_training.models.sequence.temporal_modules import ORDERED_TEMPORAL_TYPES, POOLING_TEMPORAL_TYPES
from vision_training.trainers.checkpointing import get_checkpoint_file, get_last_checkpoint_iteration

# Run of the series in tutorial_sources.yaml, and the names of its compared runs in the facts
RUN_KEY = "sequence_classification"
COMPARISON_KEYS = {"average_512": "average_pooling_512", "flatten_512": "flatten_512"}


def make_figures(output_folder: Path, experiments_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes every figure of the sequence classification series into its output folder and gives their facts, the
    checkpoint inference being reused from the previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series in the cache of the build.
        experiments_folder (Path): The folder of the real runs, or the committed snapshots.
        force (bool): Whether the expensive figures are made again even when the previous build left them.

    Returns:
        Dict[str, Any]: The facts of every figure family, keyed by their dotted names, with the words of their small
            whole numbers.
    """
    run_folders = {"flatten_128": find_run_folder(experiments_folder=experiments_folder, run_key=RUN_KEY)}
    log_paths = {"flatten_128": find_console_log(experiments_folder=experiments_folder, run_key=RUN_KEY,
                                                 comparison_key=None)}
    for run_name, comparison_key in COMPARISON_KEYS.items():
        run_folders[run_name] = find_comparison_run_folder(experiments_folder=experiments_folder, run_key=RUN_KEY,
                                                           comparison_key=comparison_key)
        log_paths[run_name] = find_console_log(experiments_folder=experiments_folder, run_key=RUN_KEY,
                                               comparison_key=comparison_key)
    checkpoint_iterations = {run_name: get_last_checkpoint_iteration(weights_directory=run_folder / "Weights")
                             for run_name, run_folder in run_folders.items()}

    # Settings of the main run: its splits, batch size, frame size, classes and sample sequences per evaluation
    run_configuration = read_run_configuration(run_folder=run_folders["flatten_128"])
    experiment_configuration = run_configuration["ExperimentConfiguration"]
    _, model_configuration = load_experiment_configuration(
        configuration_path=run_folders["flatten_128"] / "training_configuration.yaml")
    label_dictionary = model_configuration["Data"]["label_dictionary"]
    image_size = run_configuration["Data"]["image_settings"]["size"]

    # Flatten run with its trained weights, or built from its configuration when the run is read from the committed
    # snapshots, the facts of its structure being the same and those of its weights restored from the snapshots
    flatten_model = load_sequence_model(run_folder=run_folders["flatten_128"],
                                        iteration=checkpoint_iterations["flatten_128"], allow_untrained=True)
    flatten_has_trained_weights = get_checkpoint_file(weights_directory=run_folders["flatten_128"] / "Weights",
                                                      iteration=checkpoint_iterations["flatten_128"]).exists()
    facts = {"run.name": run_folders["flatten_128"].name, "run.batch_size": experiment_configuration["batch_size"],
             "run.image_size": image_size, "run.number_iterations": checkpoint_iterations["flatten_128"],
             "run.number_sample_images": experiment_configuration["evaluation"]["number_sample_images"],
             "run.number_classes": len(set(label_dictionary.values())),
             "run.maximum_sequence_length": run_configuration["Data"]["sequence_settings"]["maximum_sequence_length"],
             "run.hidden_size": run_configuration["TemporalModel"]["hidden_size"],
             "run.number_layers": run_configuration["TemporalModel"]["number_layers"],
             "run.head_hidden_units": run_configuration["ClassificationHead"]["fully_connected"][0],
             "run.head_dropout": run_configuration["ClassificationHead"]["dropout_rate"],
             "run.weight_saving_iterations": experiment_configuration["weight_saving_iterations"],
             "run.target_precision": experiment_configuration["evaluation"]["target_precision"],
             "run.sequence_features": flatten_model.temporal_module.output_features,
             "temporal.pooling_types": len(POOLING_TEMPORAL_TYPES),
             "temporal.ordered_types": len(ORDERED_TEMPORAL_TYPES),
             "temporal.types": len(POOLING_TEMPORAL_TYPES) + len(ORDERED_TEMPORAL_TYPES)}

    # Frame size of every compared run, from its saved configuration
    for run_name, run_folder in run_folders.items():
        facts[f"image_sizes.{run_name}"] = read_run_configuration(run_folder=run_folder)["Data"]["image_settings"][
            "size"]
    return add_number_words(facts=merge_facts(fact_groups=[
        facts,
        make_dataset_figures(output_folder=output_folder,
                             train_folder=Path(experiment_configuration["train_data_folder"]),
                             batch_size=experiment_configuration["batch_size"]),
        make_batch_animation(output_folder=output_folder, model=flatten_model, label_dictionary=label_dictionary,
                             image_size=image_size, has_trained_weights=flatten_has_trained_weights,
                             keep_ratio=run_configuration["Data"]["image_settings"]["keep_ratio"]),
        make_model_figures(output_folder=output_folder, flatten_model=flatten_model, run_folders=run_folders,
                           checkpoint_iterations=checkpoint_iterations,
                           test_folder=Path(experiment_configuration["test_data_folder"]),
                           label_dictionary=label_dictionary, force=force),
        make_evaluation_figures(output_folder=output_folder, run_folders=run_folders, log_paths=log_paths,
                                evaluation_iterations=checkpoint_iterations),
        describe_export_input(model=flatten_model, task=RUN_KEY)]))
