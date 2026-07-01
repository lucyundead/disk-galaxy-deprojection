from __future__ import annotations

import numpy as np


def interp_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """1-D linear-interpolation weight matrix (len(dst), len(src)); dst is clamped to src range."""
    src = np.asarray(src, dtype=float)
    dst = np.clip(np.asarray(dst, dtype=float), src[0], src[-1])
    idx = np.clip(np.searchsorted(src, dst) - 1, 0, len(src) - 2)
    frac = (dst - src[idx]) / (src[idx + 1] - src[idx])
    weights = np.zeros((len(dst), len(src)))
    rows = np.arange(len(dst))
    weights[rows, idx] = 1.0 - frac
    weights[rows, idx + 1] = frac
    return weights
