"""Build the feature-space OOD reference cloud (src/dgdp/models/dgdp_ood_reference.npz).

Computes the training-set feature vectors exactly as the bundle training did (make_features
on the table images/metadata/baseline masses), standardises them with the SHIPPED bundle's
feat_mean/feat_scale (the inference-time standardisation), fits a PCA, and stores everything
`dgdp.ood` needs to place a new galaxy in the training distribution: components, per-component
scales, orthogonal-residual scale, the reference rows' whitened scores + distances, galaxy ids
and inclinations. ~0.3 MB, ships with the package.

Run: PYTHONPATH=src .venv/bin/python scripts/build_ood_reference.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dgdp.deproject import _BUNDLED
from dgdp.features import make_features
from dgdp.model import DeprojectionModel

TABLE = Path("outputs/tng50_milestone2c_clean3d/density_residual_table.npz")
OUT = Path("src/dgdp/models/dgdp_ood_reference.npz")
N_COMP = 32


def main():
    m = DeprojectionModel.load(str(_BUNDLED))
    t = np.load(TABLE)
    feat = make_features(t["images"].astype(np.float32), t["metadata"].astype(np.float32),
                         baseline_grid_mass_msun=t["baseline_grid_mass_msun"].astype(np.float32),
                         image_feature_size=m.image_feature_size,
                         central_pixel_scale_kpc=m.central_pixel_scale_kpc)
    x = (feat - m.feat_mean) / m.feat_scale                    # inference-time standardisation
    mu = x.mean(axis=0)
    xc = x - mu
    _, sv, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:N_COMP]                                        # (K, D)
    scores = xc @ comps.T                                      # (rows, K)
    lam = scores.std(axis=0)
    resid = np.linalg.norm(xc - scores @ comps, axis=1)
    lam_r = float(np.sqrt((resid**2).mean()))
    white = np.column_stack([scores / lam, resid / lam_r])     # (rows, K+1) whitened coords
    d2 = (white**2).sum(axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT,
             feat_mean=m.feat_mean, feat_scale=m.feat_scale,   # identity check at load time
             mu=mu.astype(np.float32), comps=comps.astype(np.float32),
             lam=lam.astype(np.float32), lam_r=np.float32(lam_r),
             white=white.astype(np.float32), d2=d2.astype(np.float32),
             galaxy_id=t["galaxy_id"].astype(np.int64),
             incl_deg=t["metadata"].astype(np.float32)[:, 0])
    var = (sv[:N_COMP] ** 2).sum() / (sv**2).sum()
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} kB): {len(x)} rows, "
          f"{N_COMP} comps ({100*var:.1f}% var), median d2 {np.median(d2):.1f} "
          f"(~K+1={N_COMP+1}), incl range {t['metadata'][:, 0].min():.0f}-"
          f"{t['metadata'][:, 0].max():.0f} deg, {len(np.unique(t['galaxy_id']))} galaxies")


if __name__ == "__main__":
    main()
