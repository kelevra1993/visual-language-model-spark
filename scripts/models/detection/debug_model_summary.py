from vision_training.models.detection.two_stage_detector import TwoStageDetector
from tests.fixtures.configuration_fixtures import get_detection_model_configuration

def debug_model_summary() -> None:
    """
    Executes a simulated instantiation to print out the full TwoStageDetector architecture summary.
    
    This function acts as a standalone utility in the model architecture pipeline, allowing
    developers to quickly inspect the constructed TwoStageDetector configuration (including the backbone,
    detection, and region proposal components) without initiating a full training loop.
    
    Args:
        None
    """
    configuration, device, dtype = get_detection_model_configuration()

    model = TwoStageDetector(configuration=configuration,
                  device=device,
                  dtype=dtype)

    model.print_summary()

if __name__ == "__main__":
    debug_model_summary()
