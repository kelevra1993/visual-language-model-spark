from pathlib import Path
from vision_training.configuration.configuration_loader import load_experiment_configuration
from vision_training.trainers.detection_trainer import DetectionTrainer


def main() -> None:
    """
    Entry point for the object detection training pipeline.
    This function orchestrates the setup phase by determining the appropriate hardware device,
    loading the user-specified experiment and model configurations from the YAML file, and 
    initializing the Trainer class to begin the model training or evaluation lifecycle.
    """
    # Define configuration path
    experiment_configuration_path = (Path(__file__).parent / "configurations" / "object_detection" /
                                     "configuration_example.yaml")

    # Load configuration
    experiment_configuration, model_configuration = load_experiment_configuration(
        configuration_path=experiment_configuration_path)

    # Initialize Trainer
    trainer = DetectionTrainer(experiment_configuration=experiment_configuration,
                               model_configuration=model_configuration,
                               configuration_path=experiment_configuration_path)

    # trainer.run_benchmarking_loop(benchmarking_iterations=1000)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()
