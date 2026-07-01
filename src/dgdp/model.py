"""Torch-free load + inference of the bundled deprojection model.

The training MDN uses one mixture component, so its prediction is the deterministic mean head:
a 2-layer MLP trunk + a linear means layer producing the PCA scores, which the stored PCA basis
maps to the mixture weight-target vector. Only numpy is needed at inference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from dgdp.fourier_rz import r_knots


def _relu(x):
    return np.maximum(x, 0.0)


def save_bundle(path, *, cfg, mlp, pca, feat):
    """Persist a trained bundle. cfg is JSON-able (arrays -> lists); weights saved by key."""
    np.savez(
        path,
        config=json.dumps(cfg, default=lambda a: np.asarray(a).tolist()),
        W1=mlp["W1"], b1=mlp["b1"], W2=mlp["W2"], b2=mlp["b2"], Wm=mlp["Wm"], bm=mlp["bm"],
        pca_vec=pca["vec"], pca_mean=pca["mean"], y_mean=pca["y_mean"], y_std=pca["y_std"],
        feat_mean=feat["mean"], feat_scale=feat["scale"],
    )


@dataclass
class DeprojectionModel:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    Wm: np.ndarray
    bm: np.ndarray
    pca_vec: np.ndarray
    pca_mean: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    feat_mean: np.ndarray
    feat_scale: np.ndarray
    heights: np.ndarray
    rk_by_m: dict
    k_by_m: dict
    grid: dict
    image_feature_size: int
    central_pixel_scale_kpc: float
    img_mass_median: float
    base_mass_median: float

    @classmethod
    def load(cls, path: str) -> "DeprojectionModel":
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["config"]))
        heights = np.asarray(cfg["heights"], float)
        alloc = {int(m): tuple(v) for m, v in cfg["alloc"].items()}
        rk_by_m = {m: r_knots(nr, cfg["r_max"], cfg["r_min"]) for m, (nr, _k) in alloc.items()}
        k_by_m = {m: len(heights) for m in alloc}
        return cls(
            z["W1"], z["b1"], z["W2"], z["b2"], z["Wm"], z["bm"],
            z["pca_vec"], z["pca_mean"], z["y_mean"], z["y_std"],
            z["feat_mean"], z["feat_scale"], heights, rk_by_m, k_by_m,
            cfg["grid"], int(cfg["image_feature_size"]), float(cfg["central_pixel_scale_kpc"]),
            float(cfg["img_mass_median"]), float(cfg["base_mass_median"]),
        )

    def predict_weights(self, features: np.ndarray) -> np.ndarray:
        """features (rows, D) raw -> standardize -> MLP mean head (PCA scores) -> PCA inverse."""
        x = (np.asarray(features, float) - self.feat_mean) / self.feat_scale
        h = _relu(x @ self.W1.T + self.b1)
        h = _relu(h @ self.W2.T + self.b2)
        scores = h @ self.Wm.T + self.bm                       # (rows, n_pca_components)
        return (scores * self.y_std + self.y_mean) @ self.pca_vec + self.pca_mean
