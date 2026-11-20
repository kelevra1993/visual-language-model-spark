"""
Tests of vision_training/evaluation/recall_intervals.py: the exact (Clopper-Pearson) interval of a recall against its
closed forms, and the Korn-Graubard interval over groups matching the exact interval for independent samples and
widening for few groups.
"""

import numpy as np
import pytest

from vision_training.evaluation.recall_intervals import (compute_bootstrap_recall_intervals,
                                                         compute_clopper_pearson_intervals)


def test_clopper_pearson_bounds_match_their_closed_forms() -> None:
    """
    Verifies the exact interval at 95%: 0 of 10 found gives [0, 1 - 0.025^(1/10)], 10 of 10 [0.025^(1/10), 1], 5 of 10
    the symmetric [0.1871, 0.8129], and a class without true samples no interval.
    """
    lower_bounds, upper_bounds = compute_clopper_pearson_intervals(true_positives=np.array([0, 10, 5, 0]),
                                                                   trials=np.array([10, 10, 10, 0]))
    assert lower_bounds[:3] == pytest.approx(expected=[0.0, 0.025 ** 0.1, 0.18709], abs=1e-5)
    assert upper_bounds[:3] == pytest.approx(expected=[1.0 - 0.025 ** 0.1, 1.0, 0.81291], abs=1e-5)
    assert lower_bounds[3] is None and upper_bounds[3] is None


def test_bootstrap_interval_matches_the_exact_interval_for_independent_samples() -> None:
    """
    Verifies that groups of a single sample, independent, give a Korn-Graubard interval close to the exact interval of
    the same counts (700 of 1000 found: about [0.671, 0.728]).
    """
    group_trials = np.ones(shape=(1000, 1), dtype=np.int64)
    group_true_positives = np.zeros(shape=(1000, 1), dtype=np.int64)
    group_true_positives[:700] = 1
    bootstrap_bounds = compute_bootstrap_recall_intervals(group_true_positives=group_true_positives,
                                                          group_trials=group_trials)
    exact_bounds = compute_clopper_pearson_intervals(true_positives=np.array([700]), trials=np.array([1000]))
    assert bootstrap_bounds[0][0] == pytest.approx(expected=exact_bounds[0][0], abs=0.01)
    assert bootstrap_bounds[1][0] == pytest.approx(expected=exact_bounds[1][0], abs=0.01)


def test_bootstrap_interval_widens_for_few_groups() -> None:
    """
    Verifies the degrees of freedom of the Korn-Graubard interval: 5 sequences of 10 frames with recalls from 0.6 to 1
    against the same 5 sequences repeated 20 times. The standard error of the 100 sequences is about sqrt(20) times
    smaller, which a percentile bootstrap would keep as the ratio of the widths (about 5), while the Student quantile
    at 4 degrees of freedom (2.78 against 1.98 at 99) widens the interval of the 5 sequences further (about 7).
    """
    group_trials = np.full(shape=(5, 1), fill_value=10, dtype=np.int64)
    group_true_positives = np.array([[6], [8], [9], [7], [10]])
    few_lower, few_upper = compute_bootstrap_recall_intervals(group_true_positives=group_true_positives,
                                                              group_trials=group_trials)
    many_lower, many_upper = compute_bootstrap_recall_intervals(
        group_true_positives=np.tile(group_true_positives, reps=(20, 1)),
        group_trials=np.tile(group_trials, reps=(20, 1)))
    assert many_lower[0] < 0.8 < many_upper[0] and few_lower[0] < many_lower[0] and few_upper[0] > many_upper[0]
    assert (few_upper[0] - few_lower[0]) / (many_upper[0] - many_lower[0]) > 6.0


def test_bootstrap_interval_of_the_same_fractional_recall_in_every_group_uses_its_true_samples() -> None:
    """
    Verifies the interval of a class found 1 time in 2 in each of 100 sequences: every resample gives a recall of 0.5,
    so that the variance is 0 and the size is the limit of the variance formula, the 200 true frames reduced for the 99
    degrees of freedom (about [0.428, 0.572]), not the 100 sequences (about [0.398, 0.602]), and close to the interval
    of the same counts with a small variance between the sequences. A class found or missed whole in every sequence
    keeps the exact interval over its sequences, each counted as one sample.
    """
    group_trials = np.full(shape=(100, 2), fill_value=2, dtype=np.int64)
    group_true_positives = np.column_stack([np.ones(shape=100, dtype=np.int64), np.full(shape=100, fill_value=2)])
    lower_bounds, upper_bounds = compute_bootstrap_recall_intervals(group_true_positives=group_true_positives,
                                                                    group_trials=group_trials)
    sequence_lower, sequence_upper = compute_clopper_pearson_intervals(true_positives=np.array([50.0]),
                                                                       trials=np.array([100.0]))
    assert lower_bounds[0] == pytest.approx(expected=0.428, abs=0.002)
    assert upper_bounds[0] == pytest.approx(expected=0.572, abs=0.002)
    assert lower_bounds[0] > sequence_lower[0] + 0.02 and upper_bounds[0] < sequence_upper[0] - 0.02

    # Two sequences moved by one frame each way leave the recall at 0.5 with a small variance, and about the same bounds
    varying_true_positives = group_true_positives[:, :1].copy()
    varying_true_positives[0, 0], varying_true_positives[1, 0] = 0, 2
    varying_lower, varying_upper = compute_bootstrap_recall_intervals(group_true_positives=varying_true_positives,
                                                                      group_trials=group_trials[:, :1])
    assert lower_bounds[0] == pytest.approx(expected=varying_lower[0], abs=0.01)
    assert upper_bounds[0] == pytest.approx(expected=varying_upper[0], abs=0.01)

    # Every sequence finding the class whole: the exact interval of 100 of 100 sequences
    assert lower_bounds[1] == pytest.approx(expected=0.025 ** 0.01, abs=1e-6) and upper_bounds[1] == 1.0
