import numpy as np

from dgdp.metrics import interval_coverage, mean_absolute_error


def test_mean_absolute_error():
    truth = np.array([[1.0, 3.0], [2.0, 8.0]])
    pred = np.array([[2.0, 1.0], [2.0, 5.0]])

    assert mean_absolute_error(pred, truth) == 1.5


def test_interval_coverage():
    truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    lower = np.array([[0.0, 2.5], [2.5, 3.5]])
    upper = np.array([[2.0, 3.5], [3.5, 4.5]])

    assert interval_coverage(lower, upper, truth) == 0.75
