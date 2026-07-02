"""Torch-free load + inference of the bundled deprojection model.

The training MDN is a 2-layer MLP trunk with a mixture head over the PCA scores of the
weight-target vector: per component a mean, a diagonal log-scale, and a mixture logit.
`predict_weights` returns the mixture-mean prediction (identical to the old deterministic
head when K=1); `sample_weights` draws posterior samples of the weight vector for
uncertainty propagation. Old bundles (mean head only) still load and predict; they carry
no scale head, so sampling raises. Only numpy is needed at inference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from dgdp.fourier_rz import r_knots


def _relu(x):
    return np.maximum(x, 0.0)


def save_bundle(path, *, cfg, mlp, pca, feat):
    """Persist a trained bundle. cfg is JSON-able (arrays -> lists); weights saved by key.

    mlp requires W1,b1,W2,b2,Wm,bm (means head, K*out rows); pass Wl,bl,Ws,bs (mixture
    logits + log-scales heads) and cfg["n_components"] to enable sampling at inference.
    """
    extra = {k: mlp[k] for k in ("Wl", "bl", "Ws", "bs") if k in mlp}
    np.savez(
        path,
        config=json.dumps(cfg, default=lambda a: np.asarray(a).tolist()),
        W1=mlp["W1"], b1=mlp["b1"], W2=mlp["W2"], b2=mlp["b2"], Wm=mlp["Wm"], bm=mlp["bm"],
        pca_vec=pca["vec"], pca_mean=pca["mean"], y_mean=pca["y_mean"], y_std=pca["y_std"],
        feat_mean=feat["mean"], feat_scale=feat["scale"],
        **extra,
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
    n_components: int = 1
    Wl: np.ndarray | None = None                     # mixture logits head (K, hidden)
    bl: np.ndarray | None = None
    Ws: np.ndarray | None = None                     # log-scales head (K*out, hidden)
    bs: np.ndarray | None = None

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
            int(cfg.get("n_components", 1)),
            *(z[k] if k in z.files else None for k in ("Wl", "bl", "Ws", "bs")),
        )

    def _mixture(self, features: np.ndarray):
        """Trunk + heads -> (pi (rows,K), means (rows,K,out), log_scales or None)."""
        x = (np.asarray(features, float) - self.feat_mean) / self.feat_scale
        h = _relu(x @ self.W1.T + self.b1)
        h = _relu(h @ self.W2.T + self.b2)
        k = self.n_components
        means = (h @ self.Wm.T + self.bm).reshape(len(h), k, -1)
        if self.Wl is None:
            return np.ones((len(h), k)) / k, means, None
        logits = h @ self.Wl.T + self.bl
        pi = np.exp(logits - logits.max(axis=1, keepdims=True))
        pi /= pi.sum(axis=1, keepdims=True)
        log_scales = None
        if self.Ws is not None:
            log_scales = np.clip((h @ self.Ws.T + self.bs).reshape(len(h), k, -1), -6.0, 3.0)
        return pi, means, log_scales

    def _scores_to_weights(self, scores: np.ndarray) -> np.ndarray:
        return (scores * self.y_std + self.y_mean) @ self.pca_vec + self.pca_mean

    def predict_weights(self, features: np.ndarray) -> np.ndarray:
        """features (rows, D) raw -> mixture-mean PCA scores -> weight-target vector."""
        pi, means, _ = self._mixture(features)
        return self._scores_to_weights((pi[:, :, None] * means).sum(axis=1))

    def sample_weights(self, features: np.ndarray, n_samples: int, rng=None) -> np.ndarray:
        """Posterior draws of the weight vector, shape (rows, n_samples, target_dim)."""
        if self.Ws is None:
            raise ValueError("bundle has no scale head -- retrain with the mixture heads "
                             "exported to enable sampling")
        rng = np.random.default_rng(rng)
        pi, means, log_scales = self._mixture(features)
        rows, k, out = means.shape
        comp = np.array([rng.choice(k, size=n_samples, p=pi[r]) for r in range(rows)])
        idx = np.arange(rows)[:, None]
        mu, sc = means[idx, comp], np.exp(log_scales)[idx, comp]     # (rows, S, out)
        scores = mu + sc * rng.standard_normal(mu.shape)
        return self._scores_to_weights(scores.reshape(-1, out)).reshape(rows, n_samples, -1)
