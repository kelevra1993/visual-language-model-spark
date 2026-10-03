"""
When run, evaluates the last, a given or every checkpoint of a trained experiment on its test split (or another
labelled split) through vision_training.evaluation.checkpoint_evaluation.evaluate_checkpoints, the trainer of the task
being built from the copied configuration and the results written to a separate evaluations folder.
"""

from pathlib import Path

from vision_training.core.task_registry import get_trainer_class
from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.evaluation.checkpoint_evaluation import evaluate_checkpoints
from vision_training.models.model_transfer import disable_pretrained_settings
from vision_training.trainers.checkpointing import get_checkpoint_iterations, get_last_checkpoint_iteration

if __name__ == "__main__":
    # Trained experiment whose configuration and checkpoints are evaluated, it is only read
    experiment_folder = Path(__file__).absolute().parents[2] / "experiments" / "poseidon_experiment_4"
    # Checkpoint iteration to evaluate, None for the last saved one, or "all" for every checkpoint of the registry
    # (the intermediate ones included), compared in 'Evaluation/evaluation_summary.csv'
    iteration = None
    # Labelled split evaluated instead of the test split of the experiment, None to keep the experiment's one
    test_data_folder = Path("/home/robert_kelevra/Data/object_detection_data/synthetic_shapes/test")
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

    # The weights come from the evaluated checkpoints, so the pretrained components the run started from are not loaded
    disable_pretrained_settings(model_configuration=model_configuration)

    trainer_class = get_trainer_class(task=experiment_configuration["task"])
    trainer = trainer_class(experiment_configuration=experiment_configuration,
                            model_configuration=model_configuration, configuration_path=configuration_path)

    # Every checkpoint of the registry, the last one, or the given one
    weights_directory = experiment_folder / "Weights"
    if iteration == "all":
        evaluated_iterations = get_checkpoint_iterations(weights_directory=weights_directory)
    elif iteration is None:
        evaluated_iterations = [get_last_checkpoint_iteration(weights_directory=weights_directory)]
    else:
        evaluated_iterations = [iteration]
    evaluate_checkpoints(trainer=trainer, experiment_folder=experiment_folder, iterations=evaluated_iterations)
