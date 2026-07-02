"""Positive, midplane-centred sech^2 vertical mixture for the q_m parametrization.

Replaces the free-knot vertical profile with a per-galaxy-scaled mixture:

    g(z) = sum_k w_k * phi_k(z),   phi_k(z) = sech^2(z / (2 s a_k)) / (4 s a_k)

* a_k  : fixed dimensionless height RATIOS (thin..halo)
* s    : per-galaxy vertical scale [kpc]  -- training: from truth RMS|z|; inference: predicted
* w_k  : weights;  m=0 (and m>0 default) non-negative -> simplex when sum-normalised;
         m>0 may use a SIGNED fallback (boxy/peanut bars whose vertical shape changes sign).

By construction the reconstructed g is z-symmetric, non-negative (positive weights), and
integrates to 1 -- so the conserving deprojection needs no post-hoc symmetrise/renorm/taper.
(int_{-inf}^{inf} sech^2(z/2h) dz = 4h, hence the 1/(4 s a_k) normalisation.)

This module holds the per-profile primitives (used by the expressiveness de-risk and, later,
by the training target builder + reconstructor). Run as a script for a self-check.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RATIOS = np.array([0.3, 0.7, 1.5, 3.0])   # dimensionless thin -> halo
RMS_PER_H = np.pi / np.sqrt(12.0)                  # RMS|z| of one sech^2(z/2h) = 1.814 h


def kernel_matrix(zabs, s, ratios=DEFAULT_RATIOS):
    """Normalised sech^2 kernels phi_k(|z|), heights s*ratios; each int phi_k dz = 1 (full z)."""
    h = s * np.asarray(ratios, dtype=float)
    raw = 1.0 / np.cosh(zabs[:, None] / (2.0 * h[None, :])) ** 2
    return raw / (4.0 * h)[None, :]


def galaxy_scale(rmsz_R, weights=None):
    """Per-galaxy scale s [kpc] from RMS|z|(R): mean RMS|z| converted to an exp scale height."""
    w = np.ones_like(rmsz_R) if weights is None else np.asarray(weights, dtype=float)
    return float((rmsz_R * w).sum() / max(w.sum(), 1e-30) / RMS_PER_H)


def sech2_height_fit(profiles, z_grid, h_min=0.1, h_max=8.0, n_h=256):
    """Best-fit sech^2 scale height h_z [kpc] per row of `profiles` (rows, nz).

    Pure shape fit: profile and model are both unit-normalised, then h_z minimises the L2
    distance on the z grid (vectorised grid search over geomspace(h_min, h_max, n_h);
    numpy-only, ~0.9% quantisation). Convention: rho(z) ∝ sech^2(z / h_z) -- the same h_z
    the geometric baseline, the Comeron+2018 comparison scripts, and van-der-Kruit-style
    edge-on decompositions use (asymptotic tail e^{-2|z|/h_z}; exponential scale height
    = h_z/2; a single sech^2 has RMS|z| = 0.907 h_z). Unlike the RMS moment (tail-weighted,
    so thin+thick blends read thick), the fit reports what observational sech^2 fits report.
    Rows with no mass return 0.
    """
    p = np.asarray(profiles, dtype=float)
    tot = p.sum(axis=1)
    p = p / np.maximum(tot, 1e-300)[:, None]
    hs = np.geomspace(h_min, h_max, int(n_h))
    q = 1.0 / np.cosh(np.asarray(z_grid, dtype=float)[None, :] / hs[:, None]) ** 2
    q /= q.sum(axis=1, keepdims=True)
    err = ((p[:, None, :] - q[None, :, :]) ** 2).sum(axis=2)     # (rows, n_h)
    return np.where(tot > 0, hs[err.argmin(axis=1)], 0.0)


def fit_weights(profile, zabs, s, ratios=DEFAULT_RATIOS, signed=False, ridge=1e-6):
    """Best (non-negative, or signed) weights of `profile` over the scaled kernel basis.

    Returns raw weights (amplitude free -- fit quality is scale-invariant). Sum-normalise
    separately to turn the result into a unit-integral PDF.
    """
    b = kernel_matrix(zabs, s, ratios)
    if signed:
        return np.linalg.lstsq(b, profile, rcond=None)[0]
    from scipy.optimize import nnls  # optional (training only; not on the inference path)
    n = b.shape[1]
    ba = np.vstack([b, np.sqrt(ridge) * np.eye(n)])
    pa = np.concatenate([profile, np.zeros(n)])
    return nnls(ba, pa, maxiter=2000)[0]


def reconstruct_profile(zabs, s, w, ratios=DEFAULT_RATIOS):
    """Mixture vertical profile g(|z|) from weights (unit-integral if w is sum-normalised)."""
    return kernel_matrix(zabs, s, ratios) @ np.asarray(w, dtype=float)


# --- batch helpers wiring the mixture into the (R,z) conserving deprojection target -------
# A FIXED height dictionary (heights [kpc], thin->halo): thickness lives in the predicted weight
# VECTOR (which heights carry weight), not a per-galaxy scalar -- so prediction is robust the way
# the free-knot z-profile was (a single predicted scale regresses to the mean and mis-sizes OOD
# galaxies). The radial structure (knot resampling, Sigma_m anchoring) is the caller's concern;
# this module owns only the vertical PDF q_m(z;R) = sum_k w_{m,k}(R) phi_k(z).


def weights_target(a_rk, z_grid, heights, signed=False, floor_frac=1e-3):
    """Mixture weights approximating the normalised vertical profile of harmonic a_rk over a
    FIXED sech^2 height dictionary `heights` [kpc]. q = a_rk / int a_rk dz, fit per (row, knot):
    m=0 -> non-negative (NNLS); m>0 (signed=True) -> signed lstsq on real & imag parts.
    Returns weights (rows, nRk, K) complex (imag 0 for m=0); negligible-Sigma knots -> 0.
    """
    a_rk = np.asarray(a_rk)
    dz = float(z_grid[1] - z_grid[0])
    sig = a_rk.sum(axis=2) * dz                                         # (rows, nRk) ~ int a dz
    active = np.abs(sig) > floor_frac * np.abs(sig).max(axis=1, keepdims=True)
    safe = np.where(np.abs(sig) > 0, sig, 1.0)[:, :, None]
    q = np.where(active[:, :, None], a_rk / safe, 0.0)                  # (rows, nRk, nz), int q dz = 1
    b = kernel_matrix(np.abs(np.asarray(z_grid, dtype=float)), 1.0, heights)   # (nz, K) shared by all rows
    out = np.zeros((a_rk.shape[0], a_rk.shape[1], len(heights)), dtype=np.complex128)
    rr, jj = np.where(active)                                           # active (row, knot) pairs
    if rr.size:
        if signed:                                                     # batch every active RHS at once
            wr = np.linalg.lstsq(b, q[rr, jj].real.T, rcond=None)[0]   # (K, n_active)
            wi = np.linalg.lstsq(b, q[rr, jj].imag.T, rcond=None)[0]
            out[rr, jj] = (wr + 1j * wi).T
        else:
            from scipy.optimize import nnls                            # optional (training only)
            for r, j in zip(rr, jj):                                   # NNLS is per-RHS
                out[r, j] = nnls(b, q[r, j].real)[0]
    return out


def reconstruct(weights, z_grid, heights, signed=False):
    """Mixture weights (rows, nRk, K) -> normalised q_m(z;R) on z_grid (rows, nRk, nz) over the
    fixed dictionary `heights`. m=0 (signed=False): clip weights >= 0. Always enforce the discrete
    int q dz = 1 per (row, knot) so anchoring by Sigma_m(R) conserves mass exactly -- no patch.
    """
    weights = np.asarray(weights)
    dz = float(z_grid[1] - z_grid[0])
    b = kernel_matrix(np.abs(np.asarray(z_grid, dtype=float)), 1.0, heights)   # (nz, K)
    w = weights if signed else np.clip(weights.real, 0.0, None).astype(np.complex128)
    q = np.einsum("nRk,zk->nRz", w, b)                                 # (rows, nRk, nz)
    denom = q.sum(axis=2, keepdims=True) * dz                          # discrete int q dz
    return q / np.where(np.abs(denom) > 1e-12, denom, 1.0)


def _selfcheck():
    zabs = np.linspace(0.156, 4.84, 16)                       # milestone-2d positive |z| centres
    # synthetic thin+thick truth (exp scale heights 0.3 & 1.2 kpc)
    def sech2(z, h):
        return 1.0 / np.cosh(z / (2 * h)) ** 2 / (4 * h)
    truth = 0.7 * sech2(zabs, 0.3) + 0.3 * sech2(zabs, 1.2)
    rmsz = np.sqrt((truth * zabs**2).sum() / truth.sum())
    s = galaxy_scale(np.array([rmsz]))
    w = fit_weights(truth, zabs, s)
    fit = reconstruct_profile(zabs, s, w)
    rel = np.linalg.norm(fit - truth) / np.linalg.norm(truth)
    wn = w / w.sum()
    integ = (reconstruct_profile(zabs, s, wn) * (zabs[1] - zabs[0]) * 2).sum()   # ~full-z integral
    assert rel < 0.15, f"mixture fit residual too large: {rel:.3f}"   # smoke test; accuracy = de-risk
    assert (w >= 0).all(), "negative weights from NNLS"
    assert abs(integ - 1.0) < 0.1, f"normalised mixture integral off: {integ:.3f}"

    # batch round-trip on the FIXED dictionary: 2 rows x 2 radial knots, differing true thickness
    z_grid = np.linspace(-4.84, 4.84, 32)
    dz = z_grid[1] - z_grid[0]
    heights = np.geomspace(0.2, 3.5, 7)
    a_rk = np.empty((2, 2, z_grid.size))
    for n, h in enumerate((0.3, 0.9)):                                # thin row vs thick row
        a_rk[n, 0] = 5.0 * sech2(np.abs(z_grid), h)                   # thin knot
        a_rk[n, 1] = 2.0 * (0.6 * sech2(np.abs(z_grid), h) + 0.4 * sech2(np.abs(z_grid), 3 * h))
    wt = weights_target(a_rk.astype(complex), z_grid, heights)
    q = reconstruct(wt, z_grid, heights).real                        # (rows, nRk, nz)
    integ_b = q.sum(axis=2) * dz
    sym = np.abs(q - q[:, :, ::-1]).max()
    q_true = a_rk / (a_rk.sum(axis=2, keepdims=True) * dz)
    rel_b = np.linalg.norm(q - q_true) / np.linalg.norm(q_true)
    assert (wt.imag == 0).all(), "m=0 weights must be real"
    assert (q >= -1e-12).all(), "m=0 reconstruction must be non-negative"
    assert np.abs(integ_b - 1.0).max() < 1e-6, f"batch int q dz off: {np.abs(integ_b-1).max():.2e}"
    assert sym < 1e-9, f"batch reconstruction not z-symmetric: {sym:.2e}"
    assert rel_b < 0.12, f"fixed-dict round-trip residual too large: {rel_b:.3f}"
    print(f"selfcheck OK: rel-L2={rel:.4f}, int(norm)~{integ:.3f} | fixed-dict batch rel-L2={rel_b:.4f}, "
          f"int={integ_b.mean():.4f}, sym={sym:.1e}")


if __name__ == "__main__":
    _selfcheck()
