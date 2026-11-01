"""
Tests of vision_training/evaluation/recall_intervals.py: the exact (Clopper-Pearson) interval of a recall against its
closed forms and with fractional counts, the rejection of inconsistent counts, the counts added up per group, the
Korn-Graubard interval over groups (reproducible, holding the recall, matching the exact interval for independent
samples, widening for correlated ones and for few groups, exact over the groups without variance), and the per-class
recall table of the report.
"""

import numpy as np
import pytest

from scipy.stats import beta

from vision_training.evaluation.recall_intervals import (CLOPPER_PEARSON_DESCRIPTION, build_recall_interval_table,
                                                         compute_bootstrap_recall_intervals,
                                                         compute_clopper_pearson_intervals, describe_bootstrap_interval,
                                                         sum_counts_by_group)


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


def test_clopper_pearson_bounds_of_fractional_counts() -> None:
    """
    Verifies the exact interval of fractional true positives, as the recall times the effective sample size of the
    Korn-Graubard interval: 0.5 of 3 and 2.5 of 3 found follow the same beta quantiles as whole counts, with no
    parameter rounded.
    """
    lower_bounds, upper_bounds = compute_clopper_pearson_intervals(true_positives=np.array([0.5, 2.5]),
                                                                   trials=np.array([3, 3]))
    assert lower_bounds == pytest.approx(expected=[beta.ppf(q=0.025, a=0.5, b=3.5), beta.ppf(q=0.025, a=2.5, b=1.5)])
    assert upper_bounds == pytest.approx(expected=[beta.ppf(q=0.975, a=1.5, b=2.5), beta.ppf(q=0.975, a=3.5, b=0.5)])


def test_clopper_pearson_rejects_inconsistent_counts() -> None:
    """
    Verifies that more true positives than true samples, negative counts or counts of different shapes raise an error
    instead of giving a recall outside [0, 1].
    """
    with pytest.raises(expected_exception=ValueError, match="between 0 and the true samples"):
        compute_clopper_pearson_intervals(true_positives=np.array([3]), trials=np.array([2]))
    with pytest.raises(expected_exception=ValueError, match="between 0 and the true samples"):
        compute_clopper_pearson_intervals(true_positives=np.array([-1]), trials=np.array([2]))
    with pytest.raises(expected_exception=ValueError, match="same shape"):
        compute_clopper_pearson_intervals(true_positives=np.array([1, 1]), trials=np.array([2]))


def test_counts_are_added_up_per_group() -> None:
    """
    Verifies the counts of the samples added up per group, the groups numbered in the order of their sorted
    identifiers, a group without counts keeping zeros.
    """
    group_counts = sum_counts_by_group(group_identifiers=np.array([7, 3, 7, 3, 9]),
                                       counts=np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 0]]))
    assert group_counts.tolist() == [[1, 1], [2, 1], [0, 0]]


def test_bootstrap_interval_is_reproducible_and_holds_the_recall() -> None:
    """
    Verifies that the Korn-Graubard interval of the same counts is the same at every call (fixed seed), holds the recall
    of every class, and is None for a class without true samples.
    """
    random_generator = np.random.default_rng(seed=3)
    group_trials = random_generator.integers(low=0, high=4, size=(300, 3))
    group_trials[:, 2] = 0
    group_true_positives = random_generator.binomial(n=group_trials, p=0.6)
    first_bounds = compute_bootstrap_recall_intervals(group_true_positives=group_true_positives,
                                                      group_trials=group_trials)
    second_bounds = compute_bootstrap_recall_intervals(group_true_positives=group_true_positives,
                                                       group_trials=group_trials)
    assert first_bounds == second_bounds

    # Every present class holds its recall between its bounds, the absent one has none
    recalls = group_true_positives[:, :2].sum(axis=0) / group_trials[:, :2].sum(axis=0)
    for class_index, recall in enumerate(recalls):
        assert first_bounds[0][class_index] < recall < first_bounds[1][class_index]
    assert first_bounds[0][2] is None and first_bounds[1][2] is None


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


def test_bootstrap_interval_without_variance_is_exact_over_the_groups() -> None:
    """
    Verifies the interval of the classes whose resamples show no variance: 3 sequences whose frames of class 0 are all
    missed and of class 1 all found give the same recall in every resample, so that the percentiles of the bootstrap
    would give [0, 0] and [1, 1], while the bounds are the exact ones of 0 and 3 of 3 sequences, [0, 1 - 0.025^(1/3)]
    and [0.025^(1/3), 1].
    """
    group_trials = np.array([[2, 3], [2, 3], [2, 3]])
    group_true_positives = np.array([[0, 3], [0, 3], [0, 3]])
    lower_bounds, upper_bounds = compute_bootstrap_recall_intervals(group_true_positives=group_true_positives,
                                                                    group_trials=group_trials)
    assert lower_bounds == pytest.approx(expected=[0.0, 0.025 ** (1 / 3)])
    assert upper_bounds == pytest.approx(expected=[1.0 - 0.025 ** (1 / 3), 1.0])


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


def test_bootstrap_interval_widens_with_correlated_samples() -> None:
    """
    Verifies the reason of the bootstrap over groups: 40 sequences of 10 frames whose frames are all found (28) or all
    missed (12) hold only 40 independent outcomes, so the interval over the sequences is far wider than the exact
    interval over the 400 frames, which would treat them as independent.
    """
    group_trials = np.full(shape=(40, 1), fill_value=10, dtype=np.int64)
    group_true_positives = np.zeros(shape=(40, 1), dtype=np.int64)
    group_true_positives[:28] = 10
    bootstrap_lower, bootstrap_upper = compute_bootstrap_recall_intervals(group_true_positives=group_true_positives,
                                                                          group_trials=group_trials)
    frame_lower, frame_upper = compute_clopper_pearson_intervals(true_positives=np.array([280]),
                                                                 trials=np.array([400]))
    assert bootstrap_lower[0] < 0.7 < bootstrap_upper[0]
    assert bootstrap_upper[0] - bootstrap_lower[0] > 2.0 * (frame_upper[0] - frame_lower[0])


def test_recall_interval_table() -> None:
    """
    Verifies the per-class recall table: the support column named by the task, the recall with the bounds of its
    interval, sorted by the recall, no recall nor interval for a class without true samples, the interval described in
    the legend, and several recalls side by side as for the detection tasks.
    """
    class_rows = [{"class": "circle", "samples": 3, "recall": 2 / 3, "recall_lower": 0.09, "recall_upper": 0.99},
                  {"class": "square", "samples": 0, "recall": 0.0, "recall_lower": None, "recall_upper": None}]
    table = build_recall_interval_table(class_rows=class_rows, support_key="samples", support_column="Positives",
                                        recall_columns=[("recall", "Recall")],
                                        interval_description=CLOPPER_PEARSON_DESCRIPTION)
    assert table["title"] == "Per Class Recall" and table["per_class"] is True
    assert table["columns"] == ["Class", "Positives", "Recall", "95%\nLower", "95%\nUpper"]
    assert table["sort_column"] == "Recall"
    assert table["rows"] == [["circle", 3, 2 / 3, 0.09, 0.99], ["square", 0, None, None, None]]
    assert CLOPPER_PEARSON_DESCRIPTION in table["legend"] and "no positives of the class" in table["legend"]

    # Recalls at two overlaps, the first one sorting the table, with a description of the bootstrap
    detection_rows = [{"class": "square", "ground_truths": 4, "recall_50": 0.75, "recall_50_lower": 0.19,
                       "recall_50_upper": 0.99, "recall_75": 0.5, "recall_75_lower": 0.07, "recall_75_upper": 0.93}]
    detection_table = build_recall_interval_table(
        class_rows=detection_rows, support_key="ground_truths", support_column="Ground Truths",
        recall_columns=[("recall_50", "Recall\n@0.5"), ("recall_75", "Recall\n@0.75")],
        interval_description=describe_bootstrap_interval(group_name="test images"), title="Mask Per Class Recall",
        legend_introduction="Recalls of all the detections. ")
    assert detection_table["columns"] == ["Class", "Ground Truths", "Recall\n@0.5", "95%\nLower", "95%\nUpper",
                                          "Recall\n@0.75", "95%\nLower", "95%\nUpper"]
    assert detection_table["rows"] == [["square", 4, 0.75, 0.19, 0.99, 0.5, 0.07, 0.93]]
    assert detection_table["sort_column"] == "Recall\n@0.5" and detection_table["title"] == "Mask Per Class Recall"
    assert detection_table["legend"].startswith("Recalls of all the detections. Worst Recall @0.5 first")
    assert "Korn-Graubard" in detection_table["legend"]
    assert "1000 bootstrap resamples of the test images" in detection_table["legend"]
