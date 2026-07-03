"""Is this galaxy inside the model's training distribution? Feature-space diagnostic.

The MDN only ever sees a feature vector; if a query falls outside the TNG50 training cloud
the network extrapolates silently and the calibrated bands stop meaning anything. This
module places the query in the training cloud (dgdp_ood_reference.npz, built by
scripts/build_ood_reference.py): whitened PCA distance -> empirical percentile among the
training rows, plus the nearest TNG training analogs (subhalo ids) for a physical sanity
check. Pure numpy; the reference ships with the package.
"""
from __future__ import annotations

from importlib.resources import files

import numpy as np

_REF_PATH = files("dgdp.models").joinpath("dgdp_ood_reference.npz")
_REF = None


def _reference():
    global _REF
    if _REF is None:
        with _REF_PATH.open("rb") as fh:
            z = np.load(fh)
            _REF = {k: z[k] for k in z.files}
    return _REF


def ood_check(features_raw, feat_mean, feat_scale, n_neighbors=5):
    """Place raw feature rows (rows, D) in the training cloud.

    Returns a dict per the FIRST row: d2 (whitened squared distance), percentile (empirical,
    vs the training rows' own d2), nearest_train_ids + nearest_incl_deg + nearest_dist
    (whitened Euclidean, deduplicated by galaxy). Returns None when the model's feature
    standardisation does not match the shipped reference (e.g. a custom bundle) -- the
    diagnostic is only meaningful for the bundle it was built against.
    """
    try:
        ref = _reference()
    except FileNotFoundError:
        return None
    if (np.shape(features_raw)[1] != len(ref["feat_mean"])
            or not np.allclose(feat_mean, ref["feat_mean"], rtol=1e-5, atol=1e-8)
            or not np.allclose(feat_scale, ref["feat_scale"], rtol=1e-5, atol=1e-8)):
        return None
    x = (np.asarray(features_raw, float)[0] - ref["feat_mean"]) / ref["feat_scale"]
    xc = x - ref["mu"]
    s = xc @ ref["comps"].T
    r = float(np.linalg.norm(xc - s @ ref["comps"]))
    white = np.concatenate([s / ref["lam"], [r / float(ref["lam_r"])]])
    d2 = float((white**2).sum())
    percentile = float(100.0 * (ref["d2"] <= d2).mean())
    dist = np.linalg.norm(ref["white"] - white[None, :], axis=1)
    order = np.argsort(dist)
    ids, incl, dd = [], [], []
    for j in order:
        gid = int(ref["galaxy_id"][j])
        if gid not in ids:
            ids.append(gid)
            incl.append(float(ref["incl_deg"][j]))
            dd.append(float(dist[j]))
        if len(ids) >= n_neighbors:
            break
    return {"d2": d2, "percentile": percentile, "nearest_train_ids": ids,
            "nearest_incl_deg": incl, "nearest_dist": dd}
