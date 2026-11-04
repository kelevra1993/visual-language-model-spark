"""
Tests of the exponential moving average of the weights of vision_training/trainers/weight_averaging.py, kept by the
trainers next to the trained weights: the update formula with its decay of the first updates, and the averaged and
copied buffers.
"""

import pytest
import torch

from torch import nn

from vision_training.trainers.weight_averaging import ExponentialMovingAverage


def build_model() -> nn.Sequential:
    """
    Builds a small model with a linear layer and a batch normalization layer, so that the average covers parameters,
    running statistics and an integer batch counter.

    Returns:
        nn.Sequential: The model.
    """
    torch.manual_seed(seed=0)
    return nn.Sequential(nn.Linear(in_features=4, out_features=3), nn.BatchNorm1d(num_features=3))


def change_weights(model: nn.Sequential, value: float) -> None:
    """
    Sets every parameter and running statistic of the model to a value and counts one more batch, as a training step
    would change them.

    Args:
        model (nn.Sequential): The model built by build_model.
        value (float): The value of every parameter and running statistic.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value=value)
        model[1].running_mean.fill_(value=value)
        model[1].running_var.fill_(value=value)
        model[1].num_batches_tracked += 1


def test_decay_of_the_first_updates() -> None:
    """
    Verifies that the decay is min(decay, (1 + n) / (10 + n)) at the n-th update: 0.1 at the first one, so that the
    average leaves the random starting weights quickly, then the configured decay once (1 + n) / (10 + n) exceeds it.
    """
    moving_average = ExponentialMovingAverage(model=build_model(), decay=0.99)
    assert moving_average.get_current_decay() == pytest.approx(expected=0.1)
    moving_average.number_updates = 90
    assert moving_average.get_current_decay() == pytest.approx(expected=0.91)
    moving_average.number_updates = 10000
    assert moving_average.get_current_decay() == pytest.approx(expected=0.99)


def test_update_averages_parameters_and_statistics_and_copies_the_other_buffers() -> None:
    """
    Verifies ema_weights = decay x ema_weights + (1 - decay) x current_weights on the parameters and the running
    statistics, with the decay of the update, the batch counter being copied, and the averaged model staying in
    evaluation mode without gradients.
    """
    model = build_model()
    moving_average = ExponentialMovingAverage(model=model, decay=0.99)
    starting_weight = model[0].weight.detach().clone()
    starting_variance = model[1].running_var.clone()

    # First update, at a decay of 0.1
    change_weights(model=model, value=2.0)
    moving_average.update(model=model)
    averaged_model = moving_average.averaged_model
    assert torch.allclose(input=averaged_model[0].weight, other=0.1 * starting_weight + 0.9 * 2.0)
    assert torch.allclose(input=averaged_model[1].running_var, other=0.1 * starting_variance + 0.9 * 2.0)
    assert int(averaged_model[1].num_batches_tracked) == 1
    assert moving_average.number_updates == 1

    # Second update, at a decay of 2 / 11
    change_weights(model=model, value=4.0)
    moving_average.update(model=model)
    second_decay = 2 / 11
    expected_weight = second_decay * (0.1 * starting_weight + 0.9 * 2.0) + (1 - second_decay) * 4.0
    assert torch.allclose(input=averaged_model[0].weight, other=expected_weight)
    assert int(averaged_model[1].num_batches_tracked) == 2

    # The averaged model is never trained
    assert not averaged_model.training
    assert not any(parameter.requires_grad for parameter in averaged_model.parameters())
