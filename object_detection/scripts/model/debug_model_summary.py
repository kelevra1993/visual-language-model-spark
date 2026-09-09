import torch
from architecture_modules.backbone import Backbone

def debug_model_summary() -> None:
    """
    Executes a simulated forward pass to print out the Backbone architecture summary.
    
    This function acts as a standalone utility in the model architecture pipeline, allowing
    developers to quickly inspect the constructed Backbone configuration (including layer depth, 
    channel transformations, and normalization strategies) without initiating a full training loop.
    
    Args:
        None
    """
    device = torch.device(device="cpu")
    dtype = torch.float32

    convolutions = {
        "0": [2, 32],
        "1": [2, 64],
        "2": [2, 128],
    }

    modules = {}
    normalization = {"feature_map_normalization": "layer"}

    backbone = Backbone(
        input_channels=3,
        convolutions=convolutions,
        modules=modules,
        last_max_pooling=False,
        normalization=normalization,
        device=device,
        dtype=dtype
    )

    backbone.print_summary(expected_image_size=(384, 384))


if __name__ == "__main__":
    debug_model_summary()
