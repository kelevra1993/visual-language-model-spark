import torch
from architecture_modules.backbone import Backbone

def run_summary():
    device = torch.device("cpu")
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
    run_summary()
