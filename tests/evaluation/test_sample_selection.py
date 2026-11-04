"""
Tests of vision_training/evaluation/sample_selection.py: the fixed test samples drawn at every evaluation are spread
over the classes first, one per class in turn, and evenly over the samples of every class.
"""

from vision_training.evaluation.sample_selection import choose_sample_identifiers


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
