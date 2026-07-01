"""Controlled test: how much does disk scale height change the rotation curve?

Same exponential surface density Sigma(R) = exp(-R/Rd) and same TOTAL mass, three sech^2
vertical scale heights. Midplane v_c(R) is computed by direct softened summation over
sampled particles (no AGAMA, no Fourier rep), with an AGAMA CylSpline cross-check. This
isolates the pure thin-vs-thick effect raised by the NGC 4321 comparison.

RMS|z| of sech^2(z/h) is 0.907 h, so h = 0.3 / 1.0 / 2.0 kpc -> RMS|z| 0.27 / 0.91 / 1.81 kpc
(0.3 kpc matches the NGC4321 baseline; ~2 kpc matches the learned outer disk).

Run:
    PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 \\
        .venv/bin/python scripts/thick_disk_vc_test.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import agama
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama

from dgdp.agama_density import circular_velocity

agama.setUnits(mass=1, length=1, velocity=1)
GRAV = 4.300917270e-6  # kpc (km/s)^2 / Msun


def sample_disk(n, r_d, h, mass, rng):
    """Exponential disk Sigma ~ exp(-R/Rd) (radial PDF ~ R exp(-R/Rd)) x sech^2(z/h)."""
    radius = rng.gamma(2.0, r_d, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    u = rng.uniform(1e-9, 1.0 - 1e-9, n)
    z = h * np.arctanh(2.0 * u - 1.0)  # inverse-CDF of sech^2(z/h)
    pos = np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])
    return pos, np.full(n, mass / n)


def direct_vc(pos, mass, radii, eps):
    """Midplane v_c(R) by direct softened sum at (R, 0, 0)."""
    out = np.empty(len(radii))
    for i, radius in enumerate(radii):
        dx = pos[:, 0] - radius
        inv = (dx * dx + pos[:, 1] ** 2 + pos[:, 2] ** 2 + eps * eps) ** -1.5
        accel_x = GRAV * np.sum(mass * inv * dx)  # inward (<0) for interior mass
        out[i] = np.sqrt(max(-accel_x * radius, 0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-d", type=float, default=2.6)
    ap.add_argument("--mass", type=float, default=6.0e10)
    ap.add_argument("--heights", type=float, nargs="+", default=[0.3, 1.0, 2.0])
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/real_images"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    radii = np.linspace(0.5, 12.0, 40)

    vc_direct, vc_agama = {}, {}
    for h in args.heights:
        pos, mass = sample_disk(args.n, args.r_d, h, args.mass, rng)
        vc_direct[h] = direct_vc(pos, mass, radii, args.eps)
        pot = agama.Potential(type="CylSpline", particles=(pos, mass), symmetry="axisymmetric",
                              gridSizeR=30, gridSizeZ=30, Rmin=0.1, Rmax=40.0, zmin=0.02, zmax=20.0, mmax=0)
        vc_agama[h] = circular_velocity(pot, radii)
        print(f"h={h:.1f} kpc (RMS|z|={0.907 * h:.2f}): sampled {args.n}, "
              f"v_c peak (direct) {vc_direct[h].max():.1f} km/s")

    h0 = args.heights[0]
    print(f"\nReference scale height h0={h0} kpc, Rd={args.r_d} kpc, M={args.mass:.1e} Msun")
    print(f"{'R[kpc]':>7} | " + " | ".join(f"h={h}: vc  ratio  force%" for h in args.heights[1:]))
    for rq in (1.0, 2.0, 3.0, 5.0, 8.0):
        cells = []
        v0 = float(np.interp(rq, radii, vc_direct[h0]))
        for h in args.heights[1:]:
            vh = float(np.interp(rq, radii, vc_direct[h]))
            ratio = vh / v0
            cells.append(f"{vh:6.1f}  {ratio:.3f}  {(ratio**2 - 1) * 100:+5.0f}%")
        print(f"{rq:7.1f} | v0={v0:6.1f} | " + " | ".join(cells))
    print("(force% = v_c^2 change vs the thin h0 disk, at fixed R; negative = thick disk pulls less)")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = ["#1f77b4", "#dd8452", "#c44e52", "#55a868"]
    for h, c in zip(args.heights, colors, strict=False):
        ax[0].plot(radii, vc_direct[h], "-", color=c, lw=2, label=f"h={h} kpc (RMS|z|={0.907 * h:.2f})")
        ax[0].plot(radii, vc_agama[h], "--", color=c, lw=1, alpha=0.6)
    ax[0].set_xlabel("R [kpc]")
    ax[0].set_ylabel("v_c [km/s]")
    ax[0].set_title(f"same Sigma(R)=exp(-R/{args.r_d}), same mass\nsolid=direct sum, dashed=AGAMA")
    ax[0].legend(fontsize=8)
    ax[0].set_ylim(0, None)

    for h, c in zip(args.heights[1:], colors[1:], strict=False):
        ax[1].plot(radii, vc_direct[h] / vc_direct[h0], "-", color=c, lw=2, label=f"h={h}/h={h0}")
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_xlabel("R [kpc]")
    ax[1].set_ylabel(f"v_c(h) / v_c(h={h0})")
    ax[1].set_title("rotation-curve ratio vs the thin disk")
    ax[1].legend(fontsize=8)

    fig.suptitle("Effect of disk scale height on the rotation curve (same surface density + mass)", fontsize=12)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "thick_disk_vc_test.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
