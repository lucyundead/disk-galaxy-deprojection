"""Diagnose boxy/peanut vertical structure for one galaxy.

The boxy/peanut signature is a vertical thickening of the bar that grows
toward the bar ends. The dramatic 'X' (two off-plane density peaks) requires
resolving the bifurcation in z; this grid has 0.625 kpc z-bins, comparable to
the X-arm separation in a few-kpc bar, so the X is smeared into a single thick
band even in the truth. The robust, resolution-tolerant signature is therefore
RMS_z as a function of position along the bar.

Produces a 2-panel figure:
  left  : truth edge-on (x-z) surface density in a slab through the bar, with
          contours to show isophote boxiness;
  right : mass-weighted RMS_z(x) along the bar for truth / baseline / MDN.

Read-only; uses existing artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from train_density_residual_pca import cylindrical_bin_volumes_from_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subhalo", type=int, required=True)
    parser.add_argument("--inclination", type=float, default=40.0)
    parser.add_argument("--bar-angle", type=float, default=0.0)
    parser.add_argument("--slab-half-kpc", type=float, default=1.5)
    parser.add_argument("--radius-kpc", type=float, default=6.0)
    parser.add_argument("--height-kpc", type=float, default=4.0)
    parser.add_argument("--voxel-kpc", type=float, default=0.15)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _cartesian_cube(density, r_edges, phi_edges, z_edges, radius, height, voxel):
    xy = np.arange(-radius + 0.5 * voxel, radius, voxel)
    z = np.arange(-height + 0.5 * voxel, height, voxel)
    xg, yg = np.meshgrid(xy, xy, indexing="ij")
    radius_grid = np.sqrt(xg**2 + yg**2)
    phi = np.arctan2(yg, xg)
    ir = np.clip(np.searchsorted(r_edges, radius_grid) - 1, 0, len(r_edges) - 2)
    iphi = np.clip(np.searchsorted(phi_edges, phi) - 1, 0, len(phi_edges) - 2)
    valid_xy = radius_grid < r_edges[-1]
    cube = np.zeros((len(xy), len(xy), len(z)))
    for k, zv in enumerate(z):
        iz = np.searchsorted(z_edges, zv) - 1
        if iz < 0 or iz >= len(z_edges) - 1:
            continue
        cube[:, :, k] = np.where(valid_xy, density[ir, iphi, iz], 0.0)
    return cube, xy, z


def _rms_z_along_x(cube, xy, z, slab_half):
    """Mass-weighted RMS_z(x) within |y| < slab_half."""
    keep_y = np.abs(xy) <= slab_half
    slab = cube[:, keep_y, :]  # (nx, ny_slab, nz)
    weight = slab.sum(axis=1)  # (nx, nz)
    total = weight.sum(axis=1)
    rms = np.sqrt(
        np.divide(
            (weight * z[None, :] ** 2).sum(axis=1),
            total,
            out=np.zeros_like(total),
            where=total > 0,
        )
    )
    return rms, total


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = np.load(args.density_table)
    preds = np.load(args.predictions)

    in_galaxy = np.flatnonzero(table["galaxy_id"] == args.subhalo)
    meta = table["metadata"][in_galaxy]
    match = np.flatnonzero(
        np.isclose(meta[:, 0], args.inclination) & np.isclose(meta[:, 2], args.bar_angle)
    )
    if match.size == 0:
        raise SystemExit(f"projection (i={args.inclination}, bar={args.bar_angle}) unavailable")
    row = int(in_galaxy[match[0]])

    r_edges = table["r_edges_kpc"].astype(np.float64)
    phi_edges = table["phi_edges_rad"].astype(np.float64)
    z_edges = table["z_edges_kpc"].astype(np.float64)
    volumes = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)
    pred_row = int(
        np.flatnonzero(
            (preds["galaxy_id"] == table["galaxy_id"][row])
            & (preds["projection_id"] == table["projection_id"][row])
        )[0]
    )
    grids = {
        "truth": table["truth_density"][row].astype(np.float64),
        "baseline": table["baseline_density"][row].astype(np.float64),
        "MDN": preds["posterior_mean_mass"][pred_row].astype(np.float64) / volumes,
    }

    rms_curves = {}
    edge_map = None
    xy = z = None
    for name, density in grids.items():
        cube, xy, z = _cartesian_cube(
            density, r_edges, phi_edges, z_edges, args.radius_kpc, args.height_kpc, args.voxel_kpc
        )
        rms_curves[name], _ = _rms_z_along_x(cube, xy, z, args.slab_half_kpc)
        if name == "truth":
            keep_y = np.abs(xy) <= args.slab_half_kpc
            edge_map = cube[:, keep_y, :].sum(axis=1) * args.voxel_kpc

    bar_length = None
    import csv

    for record in csv.DictReader(open(args.manifest, encoding="utf-8")):
        if int(record["subhalo_id"]) == args.subhalo:
            bar_length = float(record["bar_length"])
            break

    z_res = float(z_edges[1] - z_edges[0])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    vmax = float(edge_map.max())
    im = axes[0].imshow(
        edge_map.T,
        origin="lower",
        extent=[-args.radius_kpc, args.radius_kpc, -args.height_kpc, args.height_kpc],
        cmap="magma",
        norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax),
        aspect="auto",
    )
    levels = vmax * np.array([0.02, 0.05, 0.12, 0.3, 0.6])
    axes[0].contour(
        xy,
        z,
        edge_map.T,
        levels=levels,
        colors="cyan",
        linewidths=0.8,
        alpha=0.7,
    )
    axes[0].set_title(f"Truth edge-on, bar slab (|y| < {args.slab_half_kpc:g} kpc)")
    axes[0].set_xlabel("x [kpc] (along bar)")
    axes[0].set_ylabel("z [kpc]")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label=r"$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]")

    colors = {"truth": "k", "baseline": "0.6", "MDN": "#b3403c"}
    labels = {"truth": "truth", "baseline": "geometric baseline", "MDN": "MDN posterior"}
    for name, rms in rms_curves.items():
        axes[1].plot(xy, rms, color=colors[name], lw=2.2, label=labels[name])
    if bar_length:
        for sign in (-1, 1):
            axes[1].axvline(sign * bar_length, color="navy", ls="--", lw=1.0, alpha=0.6)
        axes[1].text(
            bar_length, axes[1].get_ylim()[1] * 0.95, " bar end", color="navy", fontsize=10, va="top"
        )
    axes[1].set_xlabel("x [kpc] (along bar)")
    axes[1].set_ylabel(r"mass-weighted RMS$_z$ [kpc]")
    axes[1].set_title("Vertical thickening toward the bar ends")
    axes[1].legend()
    axes[1].annotate(
        f"grid z-resolution = {z_res:.2f} kpc",
        xy=(0.5, 0.04),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        color="0.3",
    )

    fig.suptitle(
        f"Subhalo {args.subhalo}: boxy/peanut diagnostic"
        + (f" (bar radius {bar_length:.1f} kpc)" if bar_length else ""),
        fontsize=15,
    )
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {args.output}")
    center = rms_curves["truth"][np.argmin(np.abs(xy))]
    if bar_length:
        end = rms_curves["truth"][np.argmin(np.abs(xy - bar_length))]
        print(f"truth RMS_z: center {center:.2f} kpc -> bar end {end:.2f} kpc (ratio {end / center:.2f})")


if __name__ == "__main__":
    main()
