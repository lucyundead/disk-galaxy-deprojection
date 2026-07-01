"""NGC 4371: azimuthally-averaged radial surface-density profile Sigma(R) and projected
half-mass radius -- our R=64 TNG-learned deprojection vs Behzad's MGE.

Sigma(R) = <int rho dz>_phi (face-on surface density). Half-mass radius R_half is the
cylinder radius enclosing 50% of the projected mass, 2pi int_0^R R' Sigma(R') dR'.
The learned in-plane Sigma is the deprojected image, so this checks the disk concentration
that sets v_c (the report's pre-fix bug had de-concentrated it: R_half 5.0 vs image ~2.7 kpc).

Run: PYTHONPATH=src .venv/bin/python scripts/compare_ngc4371_radial_density.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
try:
    import agama  # noqa: F401
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama  # noqa: F401
from ngc4371_mge_benchmark import build_density


def r_half(radii, annulus_mass):
    """Projected half-mass radius from per-annulus mass (radii sorted ascending)."""
    cum = np.cumsum(annulus_mass)
    return float(np.interp(0.5, cum / cum[-1], radii)), cum / cum[-1]


def main():
    agama.setUnits(mass=1, length=1, velocity=1)
    d = Path("outputs/real_images")
    a = np.load(d / "ngc4371_arrays.npz")
    lm, r = a["learned_mass"], a["r_grid"]

    # learned: per-annulus mass is exact from the grid; Sigma from log-grid annulus areas
    ann_mass_learn = lm.sum(axis=(1, 2))
    redge = np.concatenate([[r[0] ** 2 / np.sqrt(r[0] * r[1])], np.sqrt(r[:-1] * r[1:]),
                            [r[-1] ** 2 / np.sqrt(r[-2] * r[-1])]])
    ann_area = np.pi * (redge[1:] ** 2 - redge[:-1] ** 2)
    sig_learn = ann_mass_learn / ann_area
    rh_learn, cum_learn = r_half(r, ann_mass_learn)

    # MGE: azimuthally-averaged Sigma(R) = <int rho dz>_phi on a fine linear R grid
    dens = build_density(ml=1.0)
    rr = np.linspace(0.05, 25.0, 250)
    phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    zz = np.linspace(-8, 8, 161)
    dz = zz[1] - zz[0]
    sig_mge = np.empty_like(rr)
    for i, R in enumerate(rr):
        x = (R * np.cos(phi))[:, None, None] * np.ones((1, 1, len(zz)))
        y = (R * np.sin(phi))[:, None, None] * np.ones((1, 1, len(zz)))
        z3 = np.ones((len(phi), 1, 1)) * zz[None, None, :]
        rho = dens.density(np.column_stack([x.ravel(), y.ravel(), z3.ravel()])).reshape(len(phi), len(zz))
        sig_mge[i] = (rho.sum(axis=1) * dz).mean()          # int dz, then <>_phi
    ann_mass_mge = 2 * np.pi * rr * sig_mge * (rr[1] - rr[0])
    rh_mge, cum_mge = r_half(rr, ann_mass_mge)

    print(f"{'R[kpc]':>7}{'Sig_learn':>11}{'Sig_mge':>11}{'ratio':>8}")
    for rq in (0.5, 1, 2, 3, 5, 8, 12):
        sl = float(np.interp(rq, r, sig_learn))
        sm = float(np.interp(rq, rr, sig_mge))
        print(f"{rq:7.1f}{sl:11.3e}{sm:11.3e}{sl / max(sm, 1e-30):8.2f}")
    print(f"\nprojected half-mass radius R_half: learned {rh_learn:.2f} kpc, MGE {rh_mge:.2f} kpc")
    frac3_learn = float(np.interp(3.0, r, cum_learn))
    frac3_mge = float(np.interp(3.0, rr, cum_mge))
    print(f"mass fraction within R<3 kpc: learned {frac3_learn:.2f}, MGE {frac3_mge:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].loglog(r, sig_learn, color="#c44e52", lw=2, label="TNG-learned (R=64)")
    ax[0].loglog(rr, sig_mge, "k--", lw=2, label="MGE (Behzad)")
    ax[0].set(xlabel="R [kpc]", ylabel=r"$\Sigma$(R) [M$_\odot$/kpc$^2$]", xlim=(0.1, 20),
              title="NGC 4371 radial surface density (azimuthally averaged)")
    ax[0].legend(fontsize=9)
    ax[1].plot(r, cum_learn, color="#c44e52", lw=2, label="TNG-learned (R=64)")
    ax[1].plot(rr, cum_mge, "k--", lw=2, label="MGE (Behzad)")
    ax[1].axhline(0.5, color="0.6", ls=":", lw=1)
    ax[1].axvline(rh_learn, color="#c44e52", ls=":", lw=1)
    ax[1].axvline(rh_mge, color="k", ls=":", lw=1)
    ax[1].set(xlabel="R [kpc]", ylabel="enclosed projected mass fraction", xlim=(0, 15), ylim=(0, 1.02),
              title=f"cumulative mass: R½ learned {rh_learn:.1f} vs MGE {rh_mge:.1f} kpc")
    ax[1].legend(fontsize=9, loc="lower right")
    fig.suptitle("NGC 4371 disk concentration: TNG-learned vs MGE", fontsize=12)
    fig.tight_layout()
    fig.savefig(d / "ngc4371_radial_density.png", dpi=130)
    print(f"wrote {d}/ngc4371_radial_density.png")


if __name__ == "__main__":
    main()
