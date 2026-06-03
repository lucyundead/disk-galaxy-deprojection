from __future__ import annotations

import numpy as np


def make_feature_matrix(images: np.ndarray, baseline: np.ndarray, metadata: np.ndarray) -> np.ndarray:
    image_features = np.asarray(images, dtype=np.float32).reshape(images.shape[0], -1)
    return np.concatenate(
        [
            image_features,
            np.asarray(baseline, dtype=np.float32),
            np.asarray(metadata, dtype=np.float32),
        ],
        axis=1,
    )


def standardize_train_apply(
    train: np.ndarray,
    other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return (train - mean) / scale, (other - mean) / scale, mean, scale
