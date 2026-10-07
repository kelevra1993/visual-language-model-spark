"""
When run, evaluates the last, a given or every checkpoint of a trained experiment on its test split (or another
labelled split) through vision_training.evaluation.checkpoint_evaluation.evaluate_checkpoints, the trainer of the task
being built from the copied configuration, the results written to a separate evaluations folder and, when the tracking
is on, tracked in an MLflow run of their own.
"""

from pathlib import Path

from vision_training.core.task_registry import get_trainer_class
from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.evaluation.checkpoint_evaluation import evaluate_checkpoints
from vision_training.models.model_transfer import disable_pretrained_settings
from vision_training.trainers.checkpointing import get_checkpoint_iterations, get_last_checkpoint_iteration

if __name__ == "__main__":
    # Run name of the trained experiment whose configuration and checkpoints are evaluated, it is only read
    run_name = "CLS_IN-KAR-256_SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64_HFLAT.512.256-CWB_MC1000"
    experiment_folder = Path(__file__).absolute().parents[2] / "experiments" / run_name
    # Checkpoint iteration to evaluate, None for the last saved one, or "all" for every checkpoint of the registry
    # (the intermediate ones included, the last saved first), compared in 'Evaluation/evaluation_summary.csv'
    iteration = "all"
    # Labelled split evaluated instead of the test split of the experiment, None to keep the experiment's one
    test_data_folder = None
    # Batch size of the evaluation, which can be larger than the training one since no gradient is kept, None to
    # keep the experiment's one
    batch_size = 128
    # Tracks the evaluation in an MLflow run of its own, named '<run name>-evaluation' in the backend of the evaluated
    # experiment and tagged with its training run ('evaluation_of'), True or False, None to keep the experiment's
    # 'tracking.activated'. The MLflow run of the training is never modified
    track_evaluation = None
    # Root folder of the evaluations, kept apart from the experiment so that its files are untouched. Like a training
    # run, Results.txt, Tensorboard/Test and the Evaluation folder are written in a subfolder of it named after the
    # run settings
    output_folder = experiment_folder.parent / "evaluations"

    # Point the copied configuration to the output folder and to the evaluated split
    configuration_path = experiment_folder / "training_configuration.yaml"
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    experiment_configuration["experiment_folder"] = output_folder
    if test_data_folder is not None:
        experiment_configuration["test_data_folder"] = test_data_folder
    if batch_size is not None:
        experiment_configuration["batch_size"] = batch_size
    if track_evaluation is not None:
        experiment_configuration["tracking"]["activated"] = track_evaluation

    # The weights come from the evaluated checkpoints, so the pretrained components the run started from are not loaded
    disable_pretrained_settings(model_configuration=model_configuration)

    trainer_class = get_trainer_class(task=experiment_configuration["task"])
    trainer = trainer_class(experiment_configuration=experiment_configuration,
                            model_configuration=model_configuration, configuration_path=configuration_path)

    # Every checkpoint of the registry from the last saved one, so that the most trained weights are evaluated first,
    # the last one, or the given one
    weights_directory = experiment_folder / "Weights"
    if iteration == "all":
        evaluated_iterations = get_checkpoint_iterations(weights_directory=weights_directory)[::-1]
    elif iteration is None:
        evaluated_iterations = [get_last_checkpoint_iteration(weights_directory=weights_directory)]
    else:
        evaluated_iterations = [iteration]
    evaluate_checkpoints(trainer=trainer, experiment_folder=experiment_folder, iterations=evaluated_iterations)
