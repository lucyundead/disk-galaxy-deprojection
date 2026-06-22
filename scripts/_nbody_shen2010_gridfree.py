"""Grid-free proof of concept on Shen2010: mesh-free SPH-KDE density + superellipsoid
iso-density shape fit (the gal3d method, reimplemented with numpy/scipy, no external
dependency).

Demonstrates two things the coarse grid cannot:
  1. an adaptive SPH-KDE density (k=32 NN, cubic-spline kernel), evaluable at ANY
     point, resolves the boxy/peanut X with no fixed-resolution smearing;
  2. fitting a generalized (super)ellipsoid to iso-density surfaces gives a
     squareness exponent S<1 in the bar/bulge that captures the X as a *parameter*
     (not resolved cells), tilt-free, ~6 numbers per shell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from peanut_strength import _shen_aligned_positions


def cubic_spline_w(q):
    """Normalized 3D M4 cubic-spline kernel as a function of q=d/h (support q<=1)."""
    w = np.zeros_like(q)
    a = q <= 0.5
    b = (q > 0.5) & (q <= 1.0)
    w[a] = 1 - 6 * q[a] ** 2 + 6 * q[a] ** 3
    w[b] = 2 * (1 - q[b]) ** 3
    return 8.0 / np.pi * w


class SPHDensity:
    def __init__(self, pos, mass, k=32):
        self.tree = cKDTree(pos)
        self.pos = pos
        self.mass = mass
        self.k = k

    def __call__(self, points):
        d, idx = self.tree.query(points, k=self.k, workers=-1)
        h = d[:, -1:]
        w = cubic_spline_w(d / h) / h**3
        return (self.mass[idx] * w).sum(axis=1)


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])


def fit_superellipsoid(points):
    """Fit F(points)=1 in the fixed bar frame (x=bar, z=disk normal).

    Pin the semi-axes a,b,c to the iso-surface extents (for a superellipsoid the
    max extent along each axis IS that axis intercept), which removes the
    scale<->exponent degeneracy, then fit only the squareness exponents sa,sb,sc.
    Convention: S<1 pinched/peanut, S=1 ellipsoid, S>1 boxy/rectangular."""
    a = max(np.percentile(np.abs(points[:, 0]), 98), 1e-2)
    b = max(np.percentile(np.abs(points[:, 1]), 98), 1e-2)
    c = max(np.percentile(np.abs(points[:, 2]), 98), 1e-2)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    def resid(s):
        return ((x / a) ** 2) ** s[0] + ((y / b) ** 2) ** s[1] + ((z / c) ** 2) ** s[2] - 1.0

    res = least_squares(resid, [1.0, 1.0, 1.0], bounds=([0.2] * 3, [2.0] * 3), max_nfev=2000)
    return [a, b / a, c / b, res.x[0], res.x[1], res.x[2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--grids", type=Path, default=Path("outputs/nbody_shen2010/nbody_shen2010_grids.npz"))
    ap.add_argument("--output", type=Path, default=Path("outputs/nbody_shen2010/figures/nbody_gridfree.png"))
    ap.add_argument("--n-dir", type=int, default=400)
    ap.add_argument("--k", type=int, default=32)
    args = ap.parse_args()

    pos = _shen_aligned_positions(args.cache_dir)
    mass = np.full(pos.shape[0], 4.5e10 / pos.shape[0])
    rho = SPHDensity(pos, mass, k=args.k)
    print(f"SPH-KDE on {pos.shape[0]} particles (k={args.k}), bar frame")

    # ---- 1. mesh-free edge-on (x-z, y=0 slice), fine ----
    xx = np.arange(-6, 6.001, 0.05)
    zz = np.arange(-3, 3.001, 0.05)
    XX, ZZ = np.meshgrid(xx, zz, indexing="ij")
    pts = np.column_stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()])
    kde_map = rho(pts).reshape(XX.shape)

    # ---- 2. iso-density surfaces + superellipsoid fit per shell ----
    dirs = fib_sphere(args.n_dir)
    radii = np.geomspace(0.1, 12.0, 80)
    ray_pts = (radii[None, :, None] * dirs[:, None, :]).reshape(-1, 3)
    ray_rho = rho(ray_pts).reshape(args.n_dir, len(radii))
    rho_ref = float(np.median(ray_rho[:, np.argmin(np.abs(radii - 0.3))]))
    levels = rho_ref * np.array([0.6, 0.4, 0.25, 0.15, 0.1, 0.06, 0.04, 0.025, 0.015])

    shells = []
    for lev in levels:
        iso_r = np.full(args.n_dir, np.nan)
        for j in range(args.n_dir):
            prof = ray_rho[j]
            above = np.where(prof >= lev)[0]
            if above.size == 0 or above[-1] == len(radii) - 1:
                continue
            i1 = above[-1]
            r0, r1 = radii[i1], radii[i1 + 1]
            p0_, p1_ = prof[i1], prof[i1 + 1]
            iso_r[j] = r0 + (lev - p0_) * (r1 - r0) / (p1_ - p0_) if p1_ != p0_ else r0
        ok = np.isfinite(iso_r)
        if ok.sum() < 0.7 * args.n_dir:
            continue
        surf = iso_r[ok, None] * dirs[ok]
        p = fit_superellipsoid(surf)
        m_eff = float(np.median(np.linalg.norm(surf, axis=1)))
        shells.append({"level": lev, "m": m_eff, "surf": surf, "p": p})
        print(f"  shell m~{m_eff:5.2f} kpc: a={p[0]:.2f} eps_ab={p[1]:.2f} eps_bc={p[2]:.2f} "
              f"sa={p[3]:.2f} sb={p[4]:.2f} sc={p[5]:.2f}  (S<1 => boxy/peanut)")

    # coarse production-grid edge-on for comparison
    g = np.load(args.grids)
    re, pe, ze = g["r_edges_kpc"], g["phi_edges_rad"], g["z_edges_kpc"]
    tdens = g["truth_mass"] / (
        0.5 * (re[1:] ** 2 - re[:-1] ** 2)[:, None, None]
        * np.diff(pe)[None, :, None] * np.diff(ze)[None, None, :]
    )
    gx = np.arange(-6, 6.001, 0.1)
    gz = np.arange(-3, 3.001, 0.1)
    GX, GZ = np.meshgrid(gx, gz, indexing="ij")
    grr = np.hypot(GX, np.zeros_like(GX))
    gpp = np.arctan2(np.zeros_like(GX), GX)
    gir = np.clip(np.searchsorted(re, grr) - 1, 0, len(re) - 2)
    gip = np.clip(np.searchsorted(pe, gpp) - 1, 0, len(pe) - 2)
    grid_map = np.zeros_like(GX)
    for ii in range(GZ.shape[0]):
        for jj in range(GZ.shape[1]):
            iz = int(np.searchsorted(ze, GZ[ii, jj]) - 1)
            if 0 <= iz < len(ze) - 1 and grr[ii, jj] < re[-1]:
                grid_map[ii, jj] = tdens[gir[ii, jj], gip[ii, jj], iz]

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    vmax = float(kde_map.max())
    ax[0].imshow(kde_map.T, origin="lower", extent=[-6, 6, -3, 3], cmap="magma",
                 norm=LogNorm(vmin=vmax * 3e-3, vmax=vmax), aspect="auto")
    ax[0].contour(xx, zz, kde_map.T, levels=vmax * np.array([0.03, 0.06, 0.12, 0.25, 0.5]),
                  colors="cyan", linewidths=0.8)
    ax[0].set_title("mesh-free SPH-KDE edge-on (y=0, 0.05 kpc)\n[resolves the X, no grid]")
    ax[0].set_xlabel("x [kpc] (along bar)")
    ax[0].set_ylabel("z [kpc]")

    gvmax = float(grid_map.max())
    ax[1].imshow(grid_map.T, origin="lower", extent=[-6, 6, -3, 3], cmap="magma",
                 norm=LogNorm(vmin=gvmax * 3e-3, vmax=gvmax), aspect="auto")
    ax[1].contour(gx, gz, grid_map.T, levels=gvmax * np.array([0.03, 0.06, 0.12, 0.25, 0.5]),
                  colors="cyan", linewidths=0.8)
    ax[1].set_title("production 0.625 kpc grid edge-on\n[X smeared]")
    ax[1].set_xlabel("x [kpc] (along bar)")
    ax[1].set_ylabel("z [kpc]")

    # squareness profile + a couple of iso-surface fits (x-z) overlaid
    ms = [s["m"] for s in shells]
    ax[2].plot(ms, [s["p"][3] for s in shells], "o-", label="sa (x, along bar)")
    ax[2].plot(ms, [s["p"][4] for s in shells], "s-", label="sb (y)")
    ax[2].plot(ms, [s["p"][5] for s in shells], "^-", label="sc (z, vertical)")
    ax[2].axhline(1.0, color="k", ls=":", lw=1, label="S=1 (ellipsoid)")
    ax[2].fill_between([min(ms), max(ms)], 0.2, 1.0, color="orange", alpha=0.12)
    ax[2].set_xlabel("iso-density shell radius m [kpc]")
    ax[2].set_ylabel("squareness exponent S")
    ax[2].set_title("superellipsoid fit: S<1 = boxy/peanut")
    ax[2].set_ylim(0.2, 2.0)
    ax[2].legend(fontsize=8)
    fig.suptitle("Shen2010 grid-free: SPH-KDE density + superellipsoid iso-density shape", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)

    sc_bulge = np.median([s["p"][5] for s in shells if s["m"] < 4.0])
    print(f"\nmedian vertical squareness sc in bulge (m<4 kpc): {sc_bulge:.2f} "
          f"({'BOXY/PEANUT (S<1)' if sc_bulge < 0.9 else 'near-ellipsoidal'})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
