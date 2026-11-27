"""
Exponential moving average (EMA) of the weights of a training, set by 'Optimization.exponential_moving_average': a
second copy of the task model, never trained itself, moved towards the trained weights after every optimizer step,
so that it averages the jitter of the last steps. The base trainer updates it, stores it in every checkpoint, restores
it on a resume and evaluates it on the test split next to the trained weights. The decay of every update, lowered
during a warm-up, and the share of the weight of the average carried by its last updates are given here too, for the
windows of the average shaded on the evaluation curves (evaluation/evaluation_curves.py).
"""

import copy
import math
import torch

from torch import nn
from typing import Any, Dict

# Keys of the state of the average stored in the checkpoints
AVERAGED_MODEL_STATE_KEY = "model_state"
NUMBER_UPDATES_KEY = "number_updates"

# Suffixes of the buffers averaged like the parameters: the running statistics of the batch normalization layers
AVERAGED_BUFFER_SUFFIXES = ("running_mean", "running_var")


def compute_update_decay(decay: float, update_index: int) -> float:
    """
    Gives the decay of an update of the moving average of the weights, used by every update of the trainer
    (ExponentialMovingAverage.get_current_decay) and by the windows of the average shaded on the evaluation curves:
    the configured decay, lowered during a warm-up so that the average forgets the random starting weights quickly.
    The configured decay is only reached after about 9 / (1 − decay) updates, 44,990 for a decay of 0.9998, the update
    15,000 using (1 + 14,999) / (10 + 14,999) ≈ 0.9994 instead.

    Args:
        decay (float): The configured decay of the average, strictly between 0 and 1.
        update_index (int): The index n of the update, counted from 0.

    Returns:
        float: min(decay, (1 + n) / (10 + n)).
    """
    return min(decay, (1 + update_index) / (10 + update_index))


def compute_recent_updates_weight(decay: float, number_updates: int, recent_updates: int) -> float:
    """
    Gives the share of the weight of a moving average of the weights carried by its last updates, written next to the
    window of the average on the evaluation curves: every update multiplies the weight of everything before it by its
    decay, so the starting weights and the updates before the last k keep the product of the decays of these k
    updates, about 1/e (37 %) for the last 1 / (1 − decay) updates once the warm-up is over, the last k the rest.

    Args:
        decay (float): The configured decay of the average, strictly between 0 and 1.
        number_updates (int): The number of updates of the average.
        recent_updates (int): The number k of last updates, between 1 and number_updates.

    Returns:
        float: The share of the weight of the average carried by the last k updates, between 0 and 1.

    Raises:
        ValueError: If recent_updates is not between 1 and number_updates.
    """
    if not 1 <= recent_updates <= number_updates:
        raise ValueError(f"Expected between 1 and {number_updates} last updates of the moving average, got "
                         f"{recent_updates}.")
    return 1.0 - math.prod(compute_update_decay(decay=decay, update_index=update_index)
                           for update_index in range(number_updates - recent_updates, number_updates))


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
            float: min(decay, (1 + n) / (10 + n)) for the n-th update, n counted from 0 (compute_update_decay).
        """
        return compute_update_decay(decay=self.decay, update_index=self.number_updates)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Moves the average towards the weights of the model after an optimizer step: the parameters and the batch
        normalization running statistics by a fraction 1 − decay, the other buffers copied. Every group of tensors is
        moved by one foreach call (torch._foreach_lerp_ and torch._foreach_copy_, as torch.optim.swa_utils does), a
        few GPU kernels per step instead of one per tensor (about 900 for the Open Images detectors), the values being
        exactly those of one lerp_ or copy_ per tensor.

        Args:
            model (nn.Module): The trained task model, whose layers match the averaged model.
        """
        current_decay = self.get_current_decay()

        # Parameters, frozen ones included (their average stays their value)
        averaged_parameters, trained_parameters = list(self.averaged_model.parameters()), list(model.parameters())

        # Running statistics of the normalization layers averaged, every other buffer copied
        averaged_buffers, trained_buffers, copied_buffers, source_buffers = [], [], [], []
        for (buffer_name, averaged_buffer), buffer in zip(self.averaged_model.named_buffers(), model.buffers()):
            if buffer_name.endswith(AVERAGED_BUFFER_SUFFIXES):
                averaged_buffers.append(averaged_buffer)
                trained_buffers.append(buffer)
            else:
                copied_buffers.append(averaged_buffer)
                source_buffers.append(buffer)

        # One call per group, the foreach calls refusing empty lists (e.g. a model without batch normalization)
        averaged_tensors, trained_tensors = averaged_parameters + averaged_buffers, trained_parameters + trained_buffers
        if averaged_tensors:
            torch._foreach_lerp_(self=averaged_tensors, tensors1=trained_tensors, weight=1.0 - current_decay)
        if copied_buffers:
            torch._foreach_copy_(self=copied_buffers, src=source_buffers)
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
