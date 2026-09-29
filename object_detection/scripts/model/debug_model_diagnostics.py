import copy
import torch

from model.model import Model
from scripts.utilities.debugging_utilities import get_model_configuration, get_dummy_input_tensor
from utilities.os_utilities import print_blue, print_green
from utilities.tensor_utilities import print_tensor_shape


def debug_model_diagnostics() -> None:
    """
    Builds the Model with each architecture module variant and prints its receptive field and size diagnostics.

    This standalone utility exercises the optional architecture modules (squeeze-and-excite, residual shortcuts,
    inception, input and per-sample normalization) in the backbone and detection heads.
    For every variant it instantiates the Model, runs a forward pass on a dummy batch, and prints the receptive
    field and model size summaries that the Trainer displays when the matching 'diagnosis_modes' are enabled.
    """
    base_configuration, device, dtype = get_model_configuration()

    # Every variant toggles one architecture module in both the Backbone and the Detector
    variants = {"Plain": {},
                "Squeeze And Excite": {"squeeze_and_excite": True},
                "Residual Shortcuts": {"residual_nets": True},
                "Inception": {"inception": True, "inception_starting_block": 3}}

    # Generate a single dummy input batch reused by every variant
    _, _, _, input_tensor = get_dummy_input_tensor(device=device, dtype=dtype)

    for variant_name, modules in variants.items():
        print_blue(output=f"Model Variant : {variant_name}", add_separators=True)

        # Inject the variant modules and both input normalizations into a fresh copy of the configuration,
        # keeping every configuration key but disabling all module flags before enabling the variant ones
        configuration = copy.deepcopy(base_configuration)
        disabled_modules = {"squeeze_and_excite": False, "residual_nets": False, "inception": False}
        configuration["Backbone"]["modules"] = {**configuration["Backbone"]["modules"], **disabled_modules, **modules}
        configuration["Backbone"]["normalization"]["input_normalization"] = True
        configuration["Backbone"]["normalization"]["per_sample_normalization"] = True
        configuration["Detector"]["modules"] = {**configuration["Detector"]["modules"], **disabled_modules, **modules}

        model = Model(configuration=configuration, device=device, dtype=dtype)
        model.eval()

        # Run an inference forward pass to verify the variant builds a valid computational graph
        with torch.no_grad():
            model_output_dictionary = model(input_tensor=input_tensor)
        print_tensor_shape(tensor=model_output_dictionary["classification_scores"], name="classification_scores",
                           indent=1)

        # Print the diagnostics displayed by the Trainer
        model.print_receptive_field_summary()
        model.print_model_size_summary()
        print_green(output=f"Model Variant {variant_name} Successfully Built And Executed", add_separators=True)


if __name__ == "__main__":
    debug_model_diagnostics()
