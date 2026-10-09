"""
Triton Python backend model of the frame selection task (SEL), copied as 'model.py' into the version folder of the
decoder model of the Triton repository by deployment/frame_selection_deployment.py, next to the decoding module
(frame_selection_decoding.py, copied from models/sequence/) and the decoding settings of the exported run
(decoding_settings.json). Run inside the Triton server, chained after the exported frame selector by the ensembles, it
turns the per-frame class probabilities of every sequence into the chosen frames of every wanted class with the
decoding of the run, the same function as the evaluation and the local inference. It only imports what the Triton
image provides (numpy, scipy through the decoding module, and the backend utilities of Triton), so it cannot be
imported outside the server.
"""

import importlib.util
import json
import os
import numpy as np
import triton_python_backend_utils

from types import ModuleType
from typing import Any, Dict, List

# Files read from the version folder of the decoder model, next to this file
DECODING_MODULE_FILE = "frame_selection_decoding.py"
DECODING_SETTINGS_FILE = "decoding_settings.json"

# Input of the decoder, the output of the exported frame selector, and its outputs in the order of its configuration,
# which are the keys of the outputs of run_frame_selection
DECODER_INPUT_NAME = "class_probabilities"
DECODER_OUTPUT_NAMES = ["selected_frames", "selected_probabilities", "low_confidence", "candidate_frames",
                        "candidate_probabilities"]


def load_decoding_module(model_folder: str) -> ModuleType:
    """
    Loads the decoding module copied next to this file, which the Triton server does not find on its own, the
    repository package not being installed in the server.

    Args:
        model_folder (str): The version folder of the decoder model, holding this file and the decoding module.

    Returns:
        ModuleType: The loaded frame_selection_decoding module.
    """
    module_specification = importlib.util.spec_from_file_location(
        name="frame_selection_decoding", location=os.path.join(model_folder, DECODING_MODULE_FILE))
    decoding_module = importlib.util.module_from_spec(spec=module_specification)
    module_specification.loader.exec_module(decoding_module)
    return decoding_module


class TritonPythonModel:
    """
    Decoder model of the frame selection served by the Triton Python backend: every request holds the [B, T, K] class
    probabilities of a batch of sequences, and every response the chosen frames of every wanted class of every
    sequence with their probabilities and low-confidence flags, and the k most probable candidate frames of every
    class with their probabilities.
    """

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Loads the decoding module and the decoding settings of the exported run, once when Triton loads the model.

        Args:
            args (Dict[str, str]): The model arguments given by Triton (configuration, repository, version), unused:
                the files are read next to this one.
        """
        model_folder = os.path.dirname(os.path.abspath(__file__))
        self.decoding_module = load_decoding_module(model_folder=model_folder)
        with open(file=os.path.join(model_folder, DECODING_SETTINGS_FILE), mode="r") as settings_file:
            self.decoding_settings = json.load(fp=settings_file)

    def execute(self, requests: List[Any]) -> List[Any]:
        """
        Decodes the sequences of every request, one response per request in the same order.

        Args:
            requests (List[Any]): The Triton inference requests, each holding the 'class_probabilities' input.

        Returns:
            List[Any]: The Triton inference responses, each holding the outputs of DECODER_OUTPUT_NAMES stacked over
                the sequences of its request.
        """
        responses = []
        for request in requests:
            # Selection of every sequence of the request, with the decoding of the run
            probabilities = triton_python_backend_utils.get_input_tensor_by_name(
                inference_request=request, name=DECODER_INPUT_NAME).as_numpy()
            sequence_outputs = [self.decoding_module.run_frame_selection(probabilities=sequence_probabilities,
                                                                         decoding_settings=self.decoding_settings)
                                for sequence_probabilities in probabilities]

            # Every output stacked over the sequences, the Tensor binding of Triton taking positional arguments only
            output_tensors = [triton_python_backend_utils.Tensor(output_name, np.stack([
                sequence_output[output_name] for sequence_output in sequence_outputs]))
                for output_name in DECODER_OUTPUT_NAMES]
            responses.append(triton_python_backend_utils.InferenceResponse(output_tensors=output_tensors))
        return responses
