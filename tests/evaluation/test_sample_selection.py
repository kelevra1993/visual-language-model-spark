"""
Tests of vision_training/evaluation/sample_selection.py: the fixed test samples drawn at every evaluation are evenly
spread over the test split when the samples have no class, spread over the classes first otherwise, and their choice
does not depend on the order of the samples.
"""

from vision_training.evaluation.sample_selection import choose_sample_identifiers


def test_samples_without_class_are_evenly_spread() -> None:
    """
    Verifies that samples without class (detection and segmentation images) are chosen evenly spread over the sorted
    identifiers, the first and last ones included, and that no sample or all of them are chosen when 0 or more samples
    than the test split holds are asked.
    """
    sample_classes = {sample_identifier: [] for sample_identifier in range(10)}

    assert choose_sample_identifiers(sample_classes=sample_classes, number_samples=4) == {0, 3, 6, 9}
    assert choose_sample_identifiers(sample_classes=sample_classes, number_samples=1) == {0}
    assert choose_sample_identifiers(sample_classes=sample_classes, number_samples=0) == set()
    assert choose_sample_identifiers(sample_classes=sample_classes, number_samples=25) == set(range(10))


def test_samples_are_spread_over_the_classes() -> None:
    """
    Verifies that the classes receive one sample each in turn, in class order, so that a rare class is shown, a class
    whose samples are all drawn being skipped, and that the samples of a class are evenly spread over its identifiers.
    """
    # Class 0 has 8 samples, class 1 has 1 sample and class 2 has 3 samples
    sample_classes = {**{sample_identifier: [0] for sample_identifier in range(8)}, 8: [1],
                      **{sample_identifier: [2] for sample_identifier in range(9, 12)}}

    # One sample per class, the first of each class
    assert choose_sample_identifiers(sample_classes=sample_classes, number_samples=3) == {0, 8, 9}

    # Class 1 is exhausted after its single sample, the others keep receiving samples in turn: 3 for class 0 and 2
    # for class 2, each evenly spread over its identifiers
    assert choose_sample_identifiers(sample_classes=sample_classes, number_samples=6) == {0, 4, 7, 8, 9, 11}

    # Fewer samples than classes: the lowest classes get one
    assert choose_sample_identifiers(sample_classes=sample_classes, number_samples=2) == {0, 8}


def test_samples_are_grouped_by_their_most_frequent_class() -> None:
    """
    Verifies that a sample with several classes (the frames of a window, a multi-label sample) is grouped by its most
    frequent class, the lowest class on ties, that the samples without class form their own group, and that the order
    of the samples in the dictionary does not change the choice.
    """
    sample_classes = {5: [2, 2, 1], 3: [1, 2], 1: [], 4: [1, 1, 2]}

    # Groups: no class {1}, class 1 {3, 4} (the tie of window 3 going to class 1), class 2 {5}
    assert choose_sample_identifiers(sample_classes=sample_classes, number_samples=3) == {1, 3, 5}
    reversed_classes = dict(reversed(list(sample_classes.items())))
    assert choose_sample_identifiers(sample_classes=reversed_classes, number_samples=3) == {1, 3, 5}
    assert choose_sample_identifiers(sample_classes=sample_classes, number_samples=4) == {1, 3, 4, 5}
