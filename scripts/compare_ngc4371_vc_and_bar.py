"""NGC 4371: rotation curve and bar-strength A2(R) -- fixed TNG-learned deprojection vs MGE.

Both done consistently:
* v_c(R): build an AGAMA CylSpline potential for EACH model (the learned grid is wrapped as
  a DensityAzimuthalHarmonic via to_agama_density; the MGE from its analytic Gaussians) and
  take the AZIMUTHALLY-AVERAGED circular speed -- orientation-independent, no grid-sampling
  bias (direct-summing the MGE's compact central Gaussians on the coarse grid underestimates
  v_c, so we avoid that).
* A2(R) = |sum_phi Sigma e^{-2i phi}| / sum_phi Sigma from each model's face-on surface
  density (learned = the deprojected image anchor; MGE = analytic int rho dz).

Run: PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 \\
        .venv/bin/python scripts/compare_ngc4371_vc_and_bar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, "scripts")
try:
    import agama
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama
from dgdp.agama_density import to_agama_density
from ngc4371_mge_benchmark import build_density


def azimuthal_vc(pot, radii, nphi=24):
    """Azimuthally-averaged circular speed sqrt(<-a_R>_phi * R) at z=0."""
    ph = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    vc = np.empty(len(radii))
    for i, R in enumerate(radii):
        pos = np.column_stack([R * np.cos(ph), R * np.sin(ph), np.zeros(nphi)])
        f = pot.force(pos)
        aR = f[:, 0] * np.cos(ph) + f[:, 1] * np.sin(ph)
        vc[i] = np.sqrt(max(-aR.mean() * R, 0.0))
    return vc


def a2_profile(sig_rphi, phi):
    a0 = sig_rphi.sum(axis=1)
    a2 = np.abs((sig_rphi * np.exp(-2j * phi)[None, :]).sum(axis=1))
    return a2 / np.maximum(a0, 1e-30)


def main():
    agama.setUnits(mass=1, length=1, velocity=1)
    d = Path("outputs/real_images")
    a = np.load(d / "ngc4371_arrays.npz")
    lm, r, phi, z = a["learned_mass"], a["r_grid"], a["phi_centers"], a["z_grid"]
    nphi, dz = len(phi), float(z[1] - z[0])

    # learned density (mass/vol) -> interpolator -> AGAMA density callable -> CylSpline potential
    redge = np.concatenate([[r[0] ** 2 / np.sqrt(r[0] * r[1])], np.sqrt(r[:-1] * r[1:]),
                            [r[-1] ** 2 / np.sqrt(r[-2] * r[-1])]])
    vol = (np.pi * (redge[1:] ** 2 - redge[:-1] ** 2) / nphi) * dz
    rho = lm / vol[:, None, None]
    phi_ext = np.concatenate([phi, [phi[0] + 2 * np.pi]])
    rho_i = RegularGridInterpolator((r, phi_ext, z), np.concatenate([rho, rho[:, :1, :]], axis=1),
                                    bounds_error=False, fill_value=0.0)

    def learned_density(xyz):
        xyz = np.atleast_2d(xyz)
        R = np.hypot(xyz[:, 0], xyz[:, 1])
        P = phi[0] + np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]) - phi[0], 2 * np.pi)
        return np.clip(rho_i((np.clip(R, r[0], r[-1]), P, np.clip(xyz[:, 2], z[0], z[-1]))), 0, None)

    dens_learn = to_agama_density(learned_density, total_mass=float(lm.sum()),
                                  r_max=20.0, z_max=5.0, mmax=6)
    dens_mge = build_density(ml=1.0)
    pot_learn = agama.Potential(type="CylSpline", density=dens_learn, mmax=6,
                                rmin=0.1, zmin=0.05, rmax=30.0, zmax=15.0)
    pot_mge = agama.Potential(type="CylSpline", density=dens_mge, mmax=6,
                              rmin=0.1, zmin=0.05, rmax=30.0, zmax=15.0)

    radii = np.linspace(0.3, 18.0, 70)
    vc_learn, vc_mge = azimuthal_vc(pot_learn, radii), azimuthal_vc(pot_mge, radii)

    # A2(R): learned from the grid; MGE from analytic int rho dz on the same (r, phi)
    sig_learn = lm.sum(axis=2)
    zint = np.linspace(-6, 6, 61)
    sig_mge = np.zeros((len(r), nphi))
    for zv in zint:
        pts = np.column_stack([(r[:, None] * np.cos(phi)[None, :]).ravel(),
                               (r[:, None] * np.sin(phi)[None, :]).ravel(),
                               np.full(len(r) * nphi, zv)])
        sig_mge += dens_mge.density(pts).reshape(len(r), nphi)
    a2_learn, a2_mge = a2_profile(sig_learn, phi), a2_profile(sig_mge, phi)
    valid = (r > 0.5) & (r < 8)   # bar region; R<0.5 and R>8 are low-S/N image noise, not bar

    print(f"{'R[kpc]':>7}{'vc_learn':>9}{'vc_mge':>8}{'A2_learn':>9}{'A2_mge':>8}")
    for rq in (1, 2, 3, 5, 8, 12):
        i = np.argmin(np.abs(r - rq))
        print(f"{rq:7.0f}{np.interp(rq, radii, vc_learn):9.1f}{np.interp(rq, radii, vc_mge):8.1f}"
              f"{a2_learn[i]:9.2f}{a2_mge[i]:8.2f}")
    print(f"v_c peak: learned {vc_learn.max():.0f}, MGE {vc_mge.max():.0f} km/s (AGAMA, azimuthally averaged)")
    print(f"A2 peak (0.5<R<8): learned {a2_learn[valid].max():.2f} @ R={r[valid][a2_learn[valid].argmax()]:.1f}, "
          f"MGE {a2_mge[valid].max():.2f} @ R={r[valid][a2_mge[valid].argmax()]:.1f} kpc")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(radii, vc_learn, color="#c44e52", lw=2, label="fixed TNG-learned q_m")
    ax[0].plot(radii, vc_mge, "k--", lw=2, label="MGE (Behzad)")
    ax[0].set(xlabel="R [kpc]", ylabel="v_c [km/s]", xlim=(0, 18), ylim=(0, 200),
              title="rotation curve (AGAMA, azimuthally averaged, M/L=1)")
    ax[0].legend(fontsize=9)
    ax[1].plot(r[valid], a2_learn[valid], color="#c44e52", lw=2, label="fixed TNG-learned (= image)")
    ax[1].plot(r[valid], a2_mge[valid], "k--", lw=2, label="MGE (Behzad)")
    ax[1].set(xlabel="R [kpc]", ylabel="A2(R)  (m=2 / m=0)", xlim=(0, 8), ylim=(0, None),
              title="bar strength: deprojected face-on A2 profile (bar region)")
    ax[1].legend(fontsize=9)
    fig.suptitle("NGC 4371: rotation curve and bar strength — fixed TNG-learned vs MGE", fontsize=12)
    fig.tight_layout()
    fig.savefig(d / "ngc4371_vc_and_bar.png", dpi=130)
    print(f"wrote {d}/ngc4371_vc_and_bar.png")


if __name__ == "__main__":
    main()
