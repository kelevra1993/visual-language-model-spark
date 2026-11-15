"""
Confidence intervals of the per-class recalls of an evaluated checkpoint and the per-class recall table of its report:
the exact (Clopper-Pearson) interval of a recall counted over independent samples (the images of the classification,
the sequences of the sequence classification and comparison, the ground truth objects of the detection tasks), and
over groups of correlated samples (the frames of a test sequence, the pixels of a test image) the Korn-Graubard
interval, exact at the effective sample size given by the variance of a bootstrap that keeps every group whole. The
classification, sequence, self-supervised, segmentation and detection trainers write the bounds next to every recall
of class_metrics.csv ('<recall>_lower' and '<recall>_upper') and show them in the 'Per Class Recall' table of the
per-class report.
"""

import numpy as np

from scipy.stats import beta, norm, t
from typing import Any, Dict, List, Optional, Tuple

from vision_training.core.contracts import EvaluationTable
from vision_training.evaluation.report_formatting import format_percentage_label

# Confidence level of every recall interval, the same for every task
CONFIDENCE_LEVEL = 0.95

# Resamples of the bootstrap of the Korn-Graubard intervals, drawn from a fixed seed so that an evaluation always gives
# the same interval, and the resamples drawn at once, bounding the memory of their [resamples, groups] draws (100 x
# 50,000 groups: 40 MB)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CHUNK_RESAMPLES = 100

# Spread of the resampled recalls of a class under which it shows no variance, above the rounding of equal ratios
RESAMPLED_RECALL_TOLERANCE = 1e-12

# Title of the per-class recall table of the report
RECALL_TABLE_TITLE = "Per Class Recall"

# Description of the exact interval in the legend of the recall table
CLOPPER_PEARSON_DESCRIPTION = "exact (Clopper-Pearson) interval, the counted samples being independent"


def describe_bootstrap_interval(group_name: str) -> str:
    """
    Describes the Korn-Graubard interval of a task in the legend of its recall table, naming the groups its bootstrap
    resamples, so that a reader of the report knows the interval keeps the correlated samples of a group together.

    Args:
        group_name (str): The resampled groups, e.g. "test sequences" or "test images".

    Returns:
        str: The description of the interval.
    """
    return (f"Korn-Graubard interval, exact at the effective sample size of the variance of {BOOTSTRAP_RESAMPLES} "
            f"bootstrap resamples of the {group_name} (fixed seed), or over the {group_name} holding the class when "
            f"they all have the same recall")


def check_recall_counts(true_positives: np.ndarray, trials: np.ndarray) -> None:
    """
    Checks the counts a recall interval is computed from, so that a mismatch between the true positives and the true
    samples of the evaluation fails loudly instead of giving a recall outside [0, 1].

    Args:
        true_positives (np.ndarray): The true positives of every class (and group).
        trials (np.ndarray): The true samples of every class (and group), of the same shape.

    Raises:
        ValueError: If the counts differ in shape, are negative, or count more true positives than true samples.
    """
    if true_positives.shape != trials.shape:
        raise ValueError(f"Expected true positives and true samples of the same shape, got {true_positives.shape} and "
                         f"{trials.shape}.")
    if (true_positives < 0).any() or (true_positives > trials).any():
        raise ValueError("Expected true positives between 0 and the true samples of every class.")


def compute_clopper_pearson_intervals(true_positives: np.ndarray, trials: np.ndarray,
                                      confidence_level: float = CONFIDENCE_LEVEL
                                      ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Computes the exact (Clopper-Pearson) confidence interval of the recall of every class counted over independent
    samples (the images of the classification, the ground truth objects of the detection tasks), from the quantiles of
    the beta distribution: it covers the true recall at least as often as the confidence level, at the cost of a wider
    interval for the classes with few samples, and stays within [0, 1] for a recall of 0 or 1. The counts may be
    fractional, as the recall times the effective sample size of the Korn-Graubard interval over groups.

    Args:
        true_positives (np.ndarray): The [K] true positives of every class, whole or fractional.
        trials (np.ndarray): The [K] true samples of every class, whole or fractional.
        confidence_level (float): The share of the intervals covering the true recall. Defaults to CONFIDENCE_LEVEL.

    Returns:
        Tuple[List[Optional[float]], List[Optional[float]]]: The lower and upper bounds of every class, None for a
            class without true samples.

    Raises:
        ValueError: If the counts differ in shape, are negative, or count more true positives than true samples.
    """
    true_positives, trials = np.asarray(true_positives, dtype=np.float64), np.asarray(trials, dtype=np.float64)
    check_recall_counts(true_positives=true_positives, trials=trials)
    tail_probability = (1.0 - confidence_level) / 2.0

    # Beta quantiles of every class, the lower bound being 0 without true positive and the upper bound 1 when every
    # true sample is found, where the quantile is undefined; the replaced parameters only keep those cells computable
    has_true_positives, has_missed_samples = true_positives > 0, true_positives < trials
    lower_bounds = np.where(has_true_positives,
                            beta.ppf(q=tail_probability, a=np.where(has_true_positives, true_positives, 1.0),
                                     b=trials - true_positives + 1), 0.0)
    upper_bounds = np.where(has_missed_samples,
                            beta.ppf(q=1.0 - tail_probability, a=true_positives + 1,
                                     b=np.where(has_missed_samples, trials - true_positives, 1.0)), 1.0)

    # No interval for a class without true samples
    return ([float(bound) if class_trials > 0 else None for bound, class_trials in zip(lower_bounds, trials)],
            [float(bound) if class_trials > 0 else None for bound, class_trials in zip(upper_bounds, trials)])


def sum_counts_by_group(group_identifiers: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Adds up the per-class counts of the samples of every group (the labelled frames of every test sequence), the
    groups being numbered in the order of their sorted identifiers, for the Korn-Graubard interval over the groups.

    Args:
        group_identifiers (np.ndarray): The [N] group of every sample.
        counts (np.ndarray): The [N, K] counts of every sample and class.

    Returns:
        np.ndarray: The [G, K] int64 counts of every group and class.
    """
    group_identifiers, counts = np.asarray(group_identifiers), np.asarray(counts, dtype=np.int64)
    unique_identifiers, group_indices = np.unique(group_identifiers, return_inverse=True)
    group_counts = np.zeros(shape=(len(unique_identifiers), counts.shape[1]), dtype=np.int64)
    np.add.at(group_counts, group_indices.reshape(-1), counts)
    return group_counts


def compute_bootstrap_recall_intervals(group_true_positives: np.ndarray, group_trials: np.ndarray,
                                       confidence_level: float = CONFIDENCE_LEVEL,
                                       number_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
                                       ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Computes the Korn-Graubard confidence interval of the recall of every class counted over correlated samples
    gathered in groups (the frames of a test sequence, the pixels of a test image), the interval of a proportion
    estimated from clustered samples. A bootstrap measures the variance of the recall: every resample draws as many
    groups as there are, with replacement, the recall of a class being the true positives over the true samples of the
    drawn groups, a resample without a true sample of a class not counting for it. Keeping every group whole carries
    the correlation of its samples into that variance, which an interval over the samples themselves would ignore.
    The effective sample size of the class, the independent samples that would give that variance (p (1 - p) /
    variance, at most its true samples), is reduced by the squared ratio of the normal and Student quantiles at the
    degrees of freedom of the groups holding the class minus one, so that the interval widens when few groups hold
    it, and the bounds are the exact (Clopper-Pearson) interval of the recall at that size. When every resample gives
    the same recall (a recall of 0 or 1 in every group, or a single group holding the class), the variance says
    nothing and the bounds are the exact interval over the groups holding the class, each counted as one sample, so
    that the interval never shrinks to a point.

    Args:
        group_true_positives (np.ndarray): The [G, K] true positives of every group and class.
        group_trials (np.ndarray): The [G, K] true samples of every group and class.
        confidence_level (float): The share of the intervals covering the true recall. Defaults to CONFIDENCE_LEVEL.
        number_resamples (int): The number of resamples. Defaults to BOOTSTRAP_RESAMPLES.
        seed (int): The seed of the resampling, fixed so that an evaluation always gives the same interval. Defaults
            to BOOTSTRAP_SEED.

    Returns:
        Tuple[List[Optional[float]], List[Optional[float]]]: The lower and upper bounds of every class, None for a
            class without true samples.

    Raises:
        ValueError: If the counts differ in shape, are negative, or count more true positives than true samples.
    """
    group_true_positives = np.asarray(group_true_positives, dtype=np.int64)
    group_trials = np.asarray(group_trials, dtype=np.int64)
    check_recall_counts(true_positives=group_true_positives, trials=group_trials)
    number_groups, number_classes = group_trials.shape
    lower_bounds: List[Optional[float]] = [None] * number_classes
    upper_bounds: List[Optional[float]] = [None] * number_classes
    present_classes = np.flatnonzero(group_trials.sum(axis=0) > 0)
    if len(present_classes) == 0:
        return lower_bounds, upper_bounds

    # Resamples drawn by chunks, every one counting how many times each group is drawn (a bincount of the drawn group
    # indices, offset by resample), the drawn counts of a class being the weighted sums of the groups
    random_generator = np.random.default_rng(seed=seed)
    present_true_positives = group_true_positives[:, present_classes].astype(np.float64)
    present_trials = group_trials[:, present_classes].astype(np.float64)
    resampled_recalls = []
    for chunk_start in range(0, number_resamples, BOOTSTRAP_CHUNK_RESAMPLES):
        chunk_size = min(BOOTSTRAP_CHUNK_RESAMPLES, number_resamples - chunk_start)
        drawn_groups = random_generator.integers(low=0, high=number_groups, size=(chunk_size, number_groups))
        offset_groups = drawn_groups + number_groups * np.arange(chunk_size)[:, None]
        group_draws = np.bincount(offset_groups.reshape(-1), minlength=chunk_size * number_groups).reshape(
            chunk_size, number_groups).astype(np.float64)
        drawn_true_positives, drawn_trials = group_draws @ present_true_positives, group_draws @ present_trials
        resampled_recalls.append(np.divide(drawn_true_positives, drawn_trials,
                                           out=np.full(shape=drawn_trials.shape, fill_value=np.nan),
                                           where=drawn_trials > 0))

    # Variance of the recall of every present class over the resamples holding true samples of it, none when every
    # resample gives the same recall
    resampled_recalls = np.concatenate(resampled_recalls, axis=0)
    present_recalls = present_true_positives.sum(axis=0) / present_trials.sum(axis=0)
    present_samples, present_groups = present_trials.sum(axis=0), (present_trials > 0).sum(axis=0)
    recall_variances = np.nanvar(resampled_recalls, axis=0)
    varying_classes = np.flatnonzero(np.nanmax(resampled_recalls, axis=0) - np.nanmin(resampled_recalls, axis=0) >
                                     RESAMPLED_RECALL_TOLERANCE)

    # Effective sample size of every class: the groups holding it without variance, otherwise the independent samples
    # giving its variance, at most its true samples, reduced for the degrees of freedom of its groups
    tail_probability = (1.0 - confidence_level) / 2.0
    effective_sizes = present_groups.astype(np.float64)
    if len(varying_classes) > 0:
        varying_recalls = present_recalls[varying_classes]
        independent_sizes = np.minimum(varying_recalls * (1.0 - varying_recalls) / recall_variances[varying_classes],
                                       present_samples[varying_classes])
        degrees_of_freedom_factors = (norm.ppf(q=1.0 - tail_probability) / t.ppf(
            q=1.0 - tail_probability, df=present_groups[varying_classes] - 1)) ** 2
        effective_sizes[varying_classes] = independent_sizes * degrees_of_freedom_factors

    # Exact interval of every present class at its effective sample size, its recall times that size counted as found
    present_lower_bounds, present_upper_bounds = compute_clopper_pearson_intervals(
        true_positives=present_recalls * effective_sizes, trials=effective_sizes, confidence_level=confidence_level)
    for present_index, class_index in enumerate(present_classes):
        lower_bounds[class_index] = present_lower_bounds[present_index]
        upper_bounds[class_index] = present_upper_bounds[present_index]
    return lower_bounds, upper_bounds


def build_recall_interval_table(class_rows: List[Dict[str, Any]], support_key: str, support_column: str,
                                recall_columns: List[Tuple[str, str]], interval_description: str,
                                title: str = RECALL_TABLE_TITLE, legend_introduction: str = "",
                                in_percent: bool = False) -> EvaluationTable:
    """
    Builds the per-class recall table of the report of an evaluated checkpoint, shown with the per-class report next
    to the per-class table (kept within the width of the report) and sorted by ascending first recall so that the
    classes the model misses most come first: the true samples of every class and, for every recall, its value with
    the bounds of its confidence interval, read from the rows of class_metrics.csv ('<recall>', '<recall>_lower' and
    '<recall>_upper'). A class without true samples has no recall nor interval and is listed last. With in_percent,
    which every trainer passes, the recalls and their bounds are written as percentages ("94.03 %") and the
    confidence level as "95 %"; without it they keep the earlier 4 decimals and "95%".

    Args:
        class_rows (List[Dict[str, Any]]): The metrics of every class, keyed like class_metrics.csv, in class order.
        support_key (str): The key of the true samples of every class in the rows, e.g. 'samples', 'pixels',
            'ground_truths' or 'true_frames'.
        support_column (str): The name of the column of the true samples, e.g. 'Samples', 'Positives' or 'Frames'.
        recall_columns (List[Tuple[str, str]]): The key of every recall in the rows and its displayed name, e.g.
            [('recall', 'Recall')] or [('recall_50', 'Recall\\n@0.5'), ('recall_75', 'Recall\\n@0.75')], the first one
            sorting the table.
        interval_description (str): How the intervals are computed, written in the legend
            (CLOPPER_PEARSON_DESCRIPTION or describe_bootstrap_interval).
        title (str): The title of the table. Defaults to RECALL_TABLE_TITLE.
        legend_introduction (str): The sentences put before the legend, describing the recalls, e.g. the overlaps of
            the detection tasks. Defaults to "".
        in_percent (bool): Whether the recalls and their bounds are written as percentages. Defaults to False.

    Returns:
        EvaluationTable: The per-class recall table.
    """
    # Confidence level of the headers and the legend, "95 %" with the percentages of the report, "95%" otherwise
    level_name = format_percentage_label(value=CONFIDENCE_LEVEL) if in_percent else f"{CONFIDENCE_LEVEL:.0%}"
    columns = ["Class", support_column]
    for _, recall_name in recall_columns:
        columns += [recall_name, f"{level_name}\nLower", f"{level_name}\nUpper"]

    # The recall of every class with its bounds, none for a class without true samples
    rows = []
    for class_row in class_rows:
        has_samples = class_row[support_key] > 0
        row = [class_row["class"], class_row[support_key]]
        for recall_key, _ in recall_columns:
            row += [class_row[recall_key] if has_samples else None,
                    class_row[f"{recall_key}_lower"] if has_samples else None,
                    class_row[f"{recall_key}_upper"] if has_samples else None]
        rows.append(row)
    first_recall_name = recall_columns[0][1].replace("\n", " ")
    legend = (f"{legend_introduction}Worst {first_recall_name} first, ties in class order. {level_name} Lower and "
              f"Upper : {interval_description}. — : no {support_column.lower()} of the class.")
    return {"title": title, "columns": columns, "rows": rows, "sort_column": recall_columns[0][1], "per_class": True,
            "legend": legend, "percent_columns": columns[2:] if in_percent else []}
