"""
Choice of the fixed test samples drawn at every evaluation of every task (the sample images saved in
'Evaluation/Iteration_<n>/Samples/' and sent to TensorBoard), from the classes of the test samples given by the task
trainers. Called once by the general trainer (trainers/base_trainer.py) after the loaders are built, so that the same
samples are followed over the whole training and its resumptions.
"""

import numpy as np

from typing import Dict, List, Set

# Group of the samples without class (detection and segmentation images, multi-label samples without class)
NO_CLASS_GROUP = -1


def choose_sample_identifiers(sample_classes: Dict[int, List[int]], number_samples: int) -> Set[int]:
    """
    Chooses the test samples shown at every evaluation, so that the predictions of the same samples can be followed
    over the training: every sample is grouped by its most frequent class (the lowest class index on ties, the samples
    without class forming their own group), the groups receive one sample each in turn, in class order, until the
    number of samples is reached, then the samples of each group are taken evenly spread over its sorted identifiers.
    Every class thus gets a sample when the number of samples reaches the number of classes, and the tasks giving no
    class (one group) get samples evenly spread over the whole test split. The choice only depends on the identifiers
    and the classes, not on the shuffling of the loaders.

    Args:
        sample_classes (Dict[int, List[int]]): The class indices of every test sample, keyed by sample identifier
            (several for a multi-label sample or the frames of a window, none for a sample without class).
        number_samples (int): The number of samples to choose, 0 for none.

    Returns:
        Set[int]: The identifiers of the chosen samples, all of them when the test split holds fewer samples.
    """
    # Samples of every group, in increasing identifier order
    grouped_identifiers: Dict[int, List[int]] = {}
    for sample_identifier in sorted(sample_classes):
        class_indices = sample_classes[sample_identifier]
        group = max(sorted(set(class_indices)), key=class_indices.count) if class_indices else NO_CLASS_GROUP
        grouped_identifiers.setdefault(group, []).append(sample_identifier)

    # Number of samples of every group: one per group in turn, in class order, skipping the groups already drawn
    group_order = sorted(grouped_identifiers)
    group_quotas = {group: 0 for group in group_order}
    remaining_samples = min(number_samples, len(sample_classes))
    while remaining_samples > 0:
        for group in group_order:
            if remaining_samples > 0 and group_quotas[group] < len(grouped_identifiers[group]):
                group_quotas[group] += 1
                remaining_samples -= 1

    # Samples of every group evenly spread over its sorted identifiers
    chosen_identifiers = set()
    for group in group_order:
        if group_quotas[group] == 0:
            continue
        group_identifiers = grouped_identifiers[group]
        chosen_positions = np.linspace(start=0, stop=len(group_identifiers) - 1, num=group_quotas[group]).round()
        chosen_identifiers.update(group_identifiers[int(position)] for position in chosen_positions)
    return chosen_identifiers
