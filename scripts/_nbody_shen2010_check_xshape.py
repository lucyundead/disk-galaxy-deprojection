"""Boxy/peanut/X diagnostic for the Shen2010 snapshot, by iso-density SHAPE.

A boxy/peanut/X bulge is a contour-shape phenomenon: in a side-on (bar along x)
log edge-on map the iso-density contours are pinched at the waist and bulge
off-plane (the X arms). This does NOT require the vertical profile rho(z) to have
an off-plane maximum -- the thin disk fills the midplane in a slab. So the right
metric is the iso-density contour height z(x): for a peanut/X it has a central
minimum (the pinch) and off-plane shoulders at intermediate x.

This renders the truth from particles at high resolution and compares the
iso-density contour shape of truth / geometric baseline / MDN on the production
grid, to see whether the MDN reproduces the peanut.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter


def bar_frame_xz(cache_dir: Path, slab_y_kpc: float):
    pos = np.load(cache_dir / "positions_raw.npy").astype(np.float64)
    pos -= pos.mean(axis=0)
    x0, y0, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.hypot(x0, y0)
    inbar = r <= 5.0
    bar_angle = 0.5 * np.angle(np.sum((x0[inbar] + 1j * y0[inbar]) ** 2))
    c, s = np.cos(-bar_angle), np.sin(-bar_angle)
    x = c * x0 - s * y0
    y = s * x0 + c * y0
    slab = np.abs(y) < slab_y_kpc
    return x[slab], z[slab], np.rad2deg(bar_angle)


def contour_height(M, xcen, zcen, level):
    """Max |z| reaching `level` at each x column; symmetrized in x and z."""
    h = np.zeros(M.shape[0])
    for xi in range(M.shape[0]):
        above = np.flatnonzero(M[xi] >= level)
        h[xi] = np.max(np.abs(zcen[above])) if above.size else 0.0
    h = 0.5 * (h + h[::-1])
    return h


def pinch_ratio(M, xcen, zcen, frac):
    """h(x=0)/max h at a level = frac*peak. <1 => pinched waist (peanut/X)."""
    level = frac * M.max()
    h = contour_height(M, xcen, zcen, level)
    center = int(np.argmin(np.abs(xcen)))
    hmax = h.max()
    if hmax <= 0:
        return 1.0, 0.0, h
    x_at_max = float(abs(xcen[int(np.argmax(h))]))
    return float(h[center] / hmax), x_at_max, h


def grid_edgeon(mass, r_edges, phi_edges, z_edges, *, radius, voxel, slab):
    """Side-on (bar along x) slab-summed map from a cyl mass grid (shape only)."""
    density = mass
    xy = np.arange(-radius + 0.5 * voxel, radius, voxel)
    z = np.arange(-z_edges[-1] + 0.5 * voxel, z_edges[-1], voxel)
    xg, yg = np.meshgrid(xy, xy, indexing="ij")
    rr = np.hypot(xg, yg)
    pp = np.arctan2(yg, xg)
    ir = np.clip(np.searchsorted(r_edges, rr) - 1, 0, len(r_edges) - 2)
    ip = np.clip(np.searchsorted(phi_edges, pp) - 1, 0, len(phi_edges) - 2)
    valid = rr < r_edges[-1]
    keep_y = np.abs(xy) <= slab
    panel = np.zeros((len(xy), len(z)))
    for k, zv in enumerate(z):
        iz = int(np.searchsorted(z_edges, zv) - 1)
        if 0 <= iz < len(z_edges) - 1:
            cell = np.where(valid, density[ir, ip, iz], 0.0)
            panel[:, k] = cell[:, keep_y].sum(axis=1)
    return xy, z, panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--grids", type=Path, default=Path("outputs/nbody_shen2010/nbody_shen2010_grids.npz"))
    ap.add_argument("--output", type=Path, default=Path("outputs/nbody_shen2010/figures/nbody_xshape_check.png"))
    ap.add_argument("--slab-y-kpc", type=float, default=1.5)
    ap.add_argument("--rep-inc", type=float, default=40.0)
    ap.add_argument("--rep-bar", type=float, default=40.0)
    args = ap.parse_args()

    # ---- truth from particles, high resolution ----------------------------
    xs, zs, bar_deg = bar_frame_xz(args.cache_dir, args.slab_y_kpc)
    ext, zext, vox = 6.0, 3.0, 0.05
    xe = np.arange(-ext, ext + vox, vox)
    ze = np.arange(-zext, zext + vox, vox)
    xcen = 0.5 * (xe[:-1] + xe[1:])
    zcen = 0.5 * (ze[:-1] + ze[1:])
    H, _, _ = np.histogram2d(xs, zs, bins=[xe, ze])
    Hs = gaussian_filter(H, 2.0)
    print(f"bar rotated by {bar_deg:+.2f} deg; slab |y|<{args.slab_y_kpc} kpc")
    print("truth (particles) peanut pinch ratio h(0)/max_h  [<1 => peanut/X]:")
    for frac in (0.03, 0.05, 0.1, 0.2):
        p, xmax, _ = pinch_ratio(Hs, xcen, zcen, frac)
        print(f"  level={frac:.0%} of peak: pinch={p:.3f}  off-plane shoulder at |x|={xmax:.2f} kpc")

    # ---- truth / baseline / MDN on the production grid --------------------
    g = np.load(args.grids)
    meta = g["metadata"]
    rep = int(np.argmin(np.abs(meta[:, 0] - args.rep_inc) + np.abs(meta[:, 2] - args.rep_bar)))
    re, pe, zedg = g["r_edges_kpc"], g["phi_edges_rad"], g["z_edges_kpc"]
    grids = {
        "truth": g["truth_mass"],
        "baseline": g["baseline_mass"][rep],
        "MDN": g["pred_mean_mass"][rep],
    }
    panels = {}
    for name, mass in grids.items():
        gx, gz, panel = grid_edgeon(mass, re, pe, zedg, radius=6.0, voxel=0.1, slab=1.5)
        panels[name] = (gx, gz, panel)

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)

    # top-left: high-res truth edge-on (particles) with contours
    ax = axes[0, 0]
    vmax = float(Hs.max())
    ax.imshow(Hs.T, origin="lower", extent=[-ext, ext, -zext, zext], cmap="magma",
              norm=LogNorm(vmin=vmax * 3e-3, vmax=vmax), aspect="auto")
    ax.contour(xcen, zcen, Hs.T, levels=vmax * np.array([0.03, 0.06, 0.12, 0.25, 0.5]),
               colors="cyan", linewidths=0.8)
    ax.set_title("truth (particles): edge-on log + contours")
    ax.set_xlabel("x [kpc] (along bar)")
    ax.set_ylabel("z [kpc]")
    ax.set_ylim(-2.5, 2.5)

    # top-middle: contour-height z(x) for truth particles (the pinch)
    ax = axes[0, 1]
    for frac, col in zip((0.03, 0.06, 0.12), ("#1f77b4", "#2ca02c", "#d62728"), strict=True):
        _, _, h = pinch_ratio(Hs, xcen, zcen, frac)
        ax.plot(xcen, h, "-", color=col, lw=2, label=f"iso-level {frac:.0%} of peak")
    ax.set_title("truth contour height z(x): central dip = peanut waist")
    ax.set_xlabel("x [kpc] (along bar)")
    ax.set_ylabel("contour height |z| [kpc]")
    ax.set_xlim(-5, 5)
    ax.legend(fontsize=8)

    # top-right: contour-height z(x) comparison on the production grid
    ax = axes[0, 2]
    colors = {"truth": "k", "baseline": "0.6", "MDN": "#b3403c"}
    for name in ("truth", "baseline", "MDN"):
        gx, gz, panel = panels[name]
        _, _, h = pinch_ratio(panel, gx, gz, 0.12)
        ax.plot(gx, h, "o-", color=colors[name], lw=2, ms=3, label=name)
    ax.set_title("production-grid contour height z(x) (level 12%)")
    ax.set_xlabel("x [kpc] (along bar)")
    ax.set_ylabel("contour height |z| [kpc]")
    ax.set_xlim(-5, 5)
    ax.legend(fontsize=9)

    # bottom row: edge-on maps of truth / baseline / MDN on the production grid
    for ax, name in zip(axes[1], ("truth", "baseline", "MDN"), strict=True):
        gx, gz, panel = panels[name]
        vmax = float(panel.max())
        ax.imshow(panel.T, origin="lower", extent=[gx[0], gx[-1], gz[0], gz[-1]], cmap="magma",
                  norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax), aspect="auto")
        ax.contour(gx, gz, panel.T, levels=vmax * np.array([0.04, 0.1, 0.25, 0.5]),
                   colors="cyan", linewidths=0.8)
        ax.set_title(f"{name} edge-on (production grid)")
        ax.set_xlabel("x [kpc] (along bar)")
        ax.set_ylabel("z [kpc]")
        ax.set_ylim(-2.5, 2.5)

    fig.suptitle("Shen2010 N-body: boxy/peanut/X by iso-density shape", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print("\nproduction-grid pinch ratio (level 12%):")
    for name in ("truth", "baseline", "MDN"):
        gx, gz, panel = panels[name]
        p, xmax, _ = pinch_ratio(panel, gx, gz, 0.12)
        print(f"  {name:>9}: pinch={p:.3f}  shoulder |x|={xmax:.2f} kpc")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
