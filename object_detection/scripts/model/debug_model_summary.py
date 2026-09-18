import torch
from model.model import Model
from scripts.utilities.debugging_utilities import get_model_configuration

def debug_model_summary() -> None:
    """
    Executes a simulated instantiation to print out the full Model architecture summary.
    
    This function acts as a standalone utility in the model architecture pipeline, allowing
    developers to quickly inspect the constructed Model configuration (including the backbone,
    enhancer, and region proposal components) without initiating a full training loop.
    
    Args:
        None
    """
    configuration, device, dtype = get_model_configuration()

    model = Model(configuration=configuration,
                  mode="training",
                  device=device,
                  dtype=dtype)

    model.print_summary()

if __name__ == "__main__":
    debug_model_summary()
