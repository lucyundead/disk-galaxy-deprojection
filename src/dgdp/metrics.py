from __future__ import annotations

import numpy as np


def mean_absolute_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(truth))))


def interval_coverage(lower: np.ndarray, upper: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth)
    covered = (np.asarray(lower) <= truth) & (truth <= np.asarray(upper))
    return float(np.mean(covered))
