import torch

from pathlib import Path

from vision_training.trainers.checkpointing import get_checkpoint_path
from vision_training.trainers.detection_trainer import DetectionTrainer
from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.utilities.os_utilities import print_color

if __name__ == "__main__":
    # Trained experiment whose configuration and checkpoint are evaluated, it is only read
    experiment_folder = Path(__file__).absolute().parents[2] / "experiments" / "poseidon_experiment_4"
    # Checkpoint iteration to evaluate, None for the last saved one
    iteration = None
    # Labelled split evaluated instead of the test split of the experiment, None to keep the experiment's one
    test_data_folder = Path("/home/robert_kelevra/Data/object_detection_data/synthetic_shapes/test")
    # Root folder of the evaluations, kept apart from the experiment so that its files are untouched. Like a training
    # run, Results.txt and Tensorboard/Test are written in a subfolder of it named after the run settings
    output_folder = experiment_folder.parent / "evaluations"

    # Point the copied configuration to the output folder and to the evaluated split
    configuration_path = experiment_folder / "training_configuration.yaml"
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=configuration_path)
    experiment_configuration["experiment_folder"] = output_folder
    if test_data_folder is not None:
        experiment_configuration["test_data_folder"] = test_data_folder

    trainer = DetectionTrainer(experiment_configuration=experiment_configuration,
                               model_configuration=model_configuration, configuration_path=configuration_path)

    # Load the weights of the evaluated checkpoint into the model of the trainer
    checkpoint_path = get_checkpoint_path(experiment_folder=experiment_folder, iteration=iteration)
    checkpoint = torch.load(f=checkpoint_path, map_location=trainer.device, weights_only=True)
    trainer.model.load_state_dict(state_dict=checkpoint["model_state"])
    print_color(color="blue", output=f"Evaluating : file://{checkpoint_path}", add_separators=True)

    trainer.run_test_evaluation(iteration=int(checkpoint_path.stem.split("_")[-1]))
