import math

from typing import Any, Dict


def compute_learning_rate_factor(iteration: int, optimization_configuration: Dict[str, Any]) -> float:
    """
    Gives the factor applied to the learning rate of every parameter group at a training iteration, the schedule of
    the 'Optimization' section: a linear warm-up from 0 over 'warmup_iterations', then either a constant learning
    rate or a cosine decay down to 'minimum_learning_rate_ratio' over 'decay_iterations', the learning rate staying
    at its minimum afterwards. The factor only depends on the iteration, so a resumed run continues the schedule
    exactly without storing any scheduler state.

    Args:
        iteration (int): The training iteration, starting at 1.
        optimization_configuration (Dict[str, Any]): The 'Optimization' section.

    Returns:
        float: The learning rate factor, within [0, 1].
    """
    warmup_iterations = optimization_configuration["warmup_iterations"]

    # Linear warm-up, reaching the full learning rate at the last warm-up iteration
    if iteration <= warmup_iterations:
        return iteration / warmup_iterations
    if optimization_configuration["learning_rate_schedule"] == "constant":
        return 1.0

    # Cosine decay after the warm-up, then the minimum learning rate
    decay_progress = min(1.0, (iteration - warmup_iterations) / optimization_configuration["decay_iterations"])
    minimum_ratio = optimization_configuration["minimum_learning_rate_ratio"]
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
