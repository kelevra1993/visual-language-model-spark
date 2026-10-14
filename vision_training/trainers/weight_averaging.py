"""
Exponential moving average (EMA) of the weights of a training, set by 'Optimization.exponential_moving_average': a
second copy of the task model, never trained itself, moved towards the trained weights after every optimizer step,
so that it averages the jitter of the last steps. The base trainer updates it, stores it in every checkpoint, restores
it on a resume and evaluates it on the test split next to the trained weights.
"""

import copy
import torch

from torch import nn
from typing import Any, Dict

# Keys of the state of the average stored in the checkpoints
AVERAGED_MODEL_STATE_KEY = "model_state"
NUMBER_UPDATES_KEY = "number_updates"

# Suffixes of the buffers averaged like the parameters: the running statistics of the batch normalization layers
AVERAGED_BUFFER_SUFFIXES = ("running_mean", "running_var")


class ExponentialMovingAverage:
    """
    Moving average of the weights of a task model, updated after every optimizer step with

        ema_weights = decay × ema_weights + (1 − decay) × current_weights

    on the parameters and the running statistics of the batch normalization layers, the other buffers (e.g. the class
    weights of a loss or the batch counters) being copied from the trained model. A weight of k steps ago counts with
    a factor decay^k, so the average follows roughly the last 1 / (1 − decay) steps: about 1,000 for 0.999 and 5,000
    for 0.9998. So that the random starting weights do not stay in the average for that long, the n-th update uses
    min(decay, (1 + n) / (10 + n)), the average of the first steps following the trained weights closely.

    The averaged model is a copy of the task model kept in evaluation mode, without gradients, on the device of the
    trained model: it is only read by the test evaluation and saved with the checkpoints.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        """
        Starts the average from the current weights of the model.

        Args:
            model (nn.Module): The trained task model, already on its device with its starting weights.
            decay (float): The decay of the average, strictly between 0 and 1.
        """
        self.decay = decay
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(requires_grad=False)
        self.number_updates = 0

    def reset(self, model: nn.Module) -> None:
        """
        Restarts the average from the current weights of the model, for a run resumed from a checkpoint saved without
        an average.

        Args:
            model (nn.Module): The trained task model.
        """
        self.averaged_model.load_state_dict(state_dict=model.state_dict())
        self.number_updates = 0

    def get_current_decay(self) -> float:
        """
        Gives the decay of the next update: the configured decay, lowered for the first updates so that the average
        forgets the random starting weights quickly.

        Returns:
            float: min(decay, (1 + n) / (10 + n)) for the n-th update, n counted from 0.
        """
        return min(self.decay, (1 + self.number_updates) / (10 + self.number_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Moves the average towards the weights of the model after an optimizer step: the parameters and the batch
        normalization running statistics by a fraction 1 − decay, the other buffers copied.

        Args:
            model (nn.Module): The trained task model, whose layers match the averaged model.
        """
        current_decay = self.get_current_decay()

        # Parameters, frozen ones included (their average stays their value)
        for averaged_parameter, parameter in zip(self.averaged_model.parameters(), model.parameters()):
            averaged_parameter.lerp_(end=parameter, weight=1.0 - current_decay)

        # Running statistics of the normalization layers averaged, every other buffer copied
        for (buffer_name, averaged_buffer), buffer in zip(self.averaged_model.named_buffers(), model.buffers()):
            if buffer_name.endswith(AVERAGED_BUFFER_SUFFIXES):
                averaged_buffer.lerp_(end=buffer, weight=1.0 - current_decay)
            else:
                averaged_buffer.copy_(buffer)
        self.number_updates += 1

    def state_dict(self) -> Dict[str, Any]:
        """
        Gives the state of the average stored in the checkpoints, so that a resumed run continues the same average.

        Returns:
            Dict[str, Any]: The averaged weights ('model_state') and the number of updates ('number_updates').
        """
        return {AVERAGED_MODEL_STATE_KEY: self.averaged_model.state_dict(), NUMBER_UPDATES_KEY: self.number_updates}

    def load_state_dict(self, state_dictionary: Dict[str, Any]) -> None:
        """
        Restores the average saved in a checkpoint.

        Args:
            state_dictionary (Dict[str, Any]): The state given by state_dict.
        """
        self.averaged_model.load_state_dict(state_dict=state_dictionary[AVERAGED_MODEL_STATE_KEY])
        self.number_updates = int(state_dictionary[NUMBER_UPDATES_KEY])
