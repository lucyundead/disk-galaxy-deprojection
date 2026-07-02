from __future__ import annotations

import numpy as np

from dgdp import vertical_mixture as vm
from dgdp.util import interp_matrix

EVEN_M = (0, 2, 4)


def harmonics(field: np.ndarray, dz: float):
    """Even-m azimuthal harmonics a_m(R,z) and z-integral Sigma_m(R). field: (rows,nR,nphi,nz)."""
    coeff = np.fft.rfft(field, axis=2) / field.shape[2]
    a = {m: coeff[:, :, m, :].astype(np.complex64) for m in EVEN_M}
    sigma = {m: (a[m].sum(axis=2) * dz) for m in EVEN_M}
    return a, sigma


def r_resample(cmap: np.ndarray, r_src: np.ndarray, r_dst: np.ndarray) -> np.ndarray:
    """Resample only the R axis of (rows,R,z) maps (z handled by the mixture)."""
    return np.einsum("Rr,nrz->nRz", interp_matrix(r_src, r_dst), cmap)


def reconstruct_density(vec_rows, anchor, rk_by_m, k_by_m, heights,
                        r_grid, z_grid, phi_centers) -> np.ndarray:
    """Predicted mixture weights -> q_m(z;R) -> anchor Sigma_m(R) -> cell DENSITY (rows,nR,nphi,nz).

    q is z-symmetric, >=0 (m=0), int q dz=1 by construction; linear R-interp preserves the unit
    integral so anchoring by Sigma_m(R) conserves the column mass exactly. Clipped non-negative
    with a mass-conserving rescale: on the uniform phi grid the signed series phi-means to
    a_0(R,z) exactly, so scaling each (R,z) ring by a_0/mean(clipped) (a factor in [0,1])
    removes precisely the clipping surplus -- truncated-Fourier ringing (e.g. around the
    point-like central anchors) can no longer rectify into spurious high-|z| mass.
    """
    rows = vec_rows.shape[0]
    rho = np.zeros((rows, len(r_grid), len(phi_centers), len(z_grid)))
    i = 0
    for m in EVEN_M:
        rk, kk = rk_by_m[m], k_by_m[m]
        size = len(rk) * kk
        wre = vec_rows[:, i:i + size].reshape(rows, len(rk), kk)
        i += size
        if m == 0:
            w = wre.astype(np.complex128)
        else:
            wim = vec_rows[:, i:i + size].reshape(rows, len(rk), kk)
            i += size
            w = wre + 1j * wim
        q_rk = vm.reconstruct(w, z_grid, heights, signed=(m != 0))
        q_grid = r_resample(q_rk, rk, r_grid)
        a_m = anchor[m][:, :, None] * q_grid
        if m == 0:
            a0 = np.clip(a_m.real, 0.0, None)                  # (rows,nR,nz) phi-mean target
            rho += a_m.real[:, :, None, :]
        else:
            cos_m, sin_m = np.cos(m * phi_centers), np.sin(m * phi_centers)
            rho += 2.0 * (a_m.real[:, :, None, :] * cos_m[None, None, :, None]
                          - a_m.imag[:, :, None, :] * sin_m[None, None, :, None])
    rho = np.clip(rho, 0.0, None)
    mean_clip = rho.mean(axis=2)
    fac = np.divide(a0, mean_clip, out=np.zeros_like(mean_clip), where=mean_clip > 0)
    return rho * fac[:, :, None, :]
