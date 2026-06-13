"""Does the retrained MDN recover a held-out galaxy's boxy/peanut X?

Reads a finer-z density residual table (|z|<5 kpc, 0.3125 kpc) and the matching
MDN predictions, and for the chosen galaxy compares truth / geometric baseline /
MDN posterior in the bar region:
  - edge-on (x-z) surface-density maps (slab through the bar);
  - bar-end vertical mass profiles, which reveal the off-plane peaks of an X.

Intended for a galaxy forced into the held-out test split, so a recovered X is
genuinely out-of-sample. Read-only; uses existing artifacts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from train_density_residual_pca import (
    _mass_grids,
    cylindrical_bin_volumes_from_edges,
    reconstruct_delta_mass_from_coefficients,
)


def _row(table, preds, subhalo, projection):
    in_gal = np.flatnonzero(
        (table["galaxy_id"] == subhalo) & (table["projection_id"] == projection)
    )
    if in_gal.size == 0:
        raise SystemExit(f"subhalo {subhalo} projection {projection} not in table")
    t_row = int(in_gal[0])
    p_row = int(
        np.flatnonzero(
            (preds["galaxy_id"] == subhalo) & (preds["projection_id"] == projection)
        )[0]
    )
    return t_row, p_row


def _bar_end_profile(mass, r_edges, phi_edges, z_edges, length):
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    pc = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    along = (np.abs(pc) < np.pi / 6) | (np.abs(np.abs(pc) - np.pi) < np.pi / 6)
    r_sel = (rc >= 0.6 * length) & (rc <= 1.1 * length)
    prof = mass[np.ix_(r_sel, along, np.arange(len(zc)))].sum(axis=(0, 1))
    prof = 0.5 * (prof + prof[::-1])
    return zc, prof


def _edgeon(density, r_edges, phi_edges, z_edges, radius, height, voxel, slab):
    xy = np.arange(-radius + 0.5 * voxel, radius, voxel)
    z = np.arange(-height + 0.5 * voxel, height, voxel)
    xg, yg = np.meshgrid(xy, xy, indexing="ij")
    rr = np.sqrt(xg**2 + yg**2)
    pp = np.arctan2(yg, xg)
    ir = np.clip(np.searchsorted(r_edges, rr) - 1, 0, len(r_edges) - 2)
    ip = np.clip(np.searchsorted(phi_edges, pp) - 1, 0, len(phi_edges) - 2)
    valid = rr < r_edges[-1]
    keep_y = np.abs(xy) <= slab
    panel = np.zeros((len(xy), len(z)))
    for k, zv in enumerate(z):
        iz = np.searchsorted(z_edges, zv) - 1
        if 0 <= iz < len(z_edges) - 1:
            panel[:, k] = np.where(valid, density[ir, ip, iz], 0.0)[:, keep_y].sum(axis=1) * voxel
    return xy, z, panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pca", type=Path, default=None, help="add a basis best-case curve")
    parser.add_argument("--subhalo", type=int, required=True)
    parser.add_argument("--projection", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    table = np.load(args.density_table)
    preds = np.load(args.predictions)
    t_row, p_row = _row(table, preds, args.subhalo, args.projection)
    split = str(table["split"][t_row])

    r_edges = table["r_edges_kpc"].astype(np.float64)
    phi_edges = table["phi_edges_rad"].astype(np.float64)
    z_edges = table["z_edges_kpc"].astype(np.float64)
    volumes = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)

    truth_mass_all, baseline_mass_all, _ = _mass_grids(table)
    grids = {
        "truth": (table["truth_density"][t_row].astype(np.float64), truth_mass_all[t_row]),
        "baseline": (
            table["baseline_density"][t_row].astype(np.float64),
            baseline_mass_all[t_row],
        ),
        "MDN": (
            preds["posterior_mean_mass"][p_row].astype(np.float64) / volumes,
            preds["posterior_mean_mass"][p_row].astype(np.float64),
        ),
    }

    length = None
    for rec in csv.DictReader(open(args.manifest, encoding="utf-8")):
        if int(rec["subhalo_id"]) == args.subhalo:
            length = float(rec["bar_length"])
            break

    colors = {"truth": "k", "baseline": "0.6", "MDN": "#b3403c"}
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), constrained_layout=True)
    for ax, name in zip(axes[:3], ("truth", "baseline", "MDN"), strict=True):
        density = grids[name][0]
        xy, z, panel = _edgeon(density, r_edges, phi_edges, z_edges, 6.0, 4.0, 0.1, 1.5)
        vmax = float(panel.max())
        ax.imshow(
            panel.T, origin="lower", extent=[-6, 6, -4, 4], cmap="magma",
            norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax), aspect="auto",
        )
        ax.contour(xy, z, panel.T, levels=vmax * np.array([0.04, 0.1, 0.25, 0.5]), colors="cyan", linewidths=0.7, alpha=0.7)
        ax.set_title(f"{name} edge-on")
        ax.set_xlabel("x [kpc] (along bar)")
        ax.set_ylabel("z [kpc]")

    dips = {}
    for name in ("truth", "baseline", "MDN"):
        zc, prof = _bar_end_profile(grids[name][1], r_edges, phi_edges, z_edges, length)
        prof = prof / prof.max()
        dip = prof[np.argmin(np.abs(zc))]
        peak_z = abs(zc[np.argmax(prof)])
        dips[name] = (dip, peak_z)
        axes[3].plot(zc, prof, "o-", color=colors[name], lw=2, label=f"{name} (dip={dip:.2f}, peak|z|={peak_z:.2f})")

    # Optional: best the basis can do, given the truth's own coefficients. Shows
    # whether a recovery failure is representational or a prediction failure.
    if args.pca is not None:
        pca = np.load(args.pca)
        scale = table["baseline_grid_mass_msun"][t_row].astype(np.float32)
        delta = reconstruct_delta_mass_from_coefficients(
            preds["true_coefficients"][p_row][None, :],
            components=pca["components"].astype(np.float32),
            mean=pca["mean"].astype(np.float32),
            scale_msun=np.array([scale], dtype=np.float32),
            grid_shape=truth_mass_all[t_row].shape,
        )[0]
        basis_mass = np.clip(baseline_mass_all[t_row] + delta, 0.0, None)
        zc, prof = _bar_end_profile(basis_mass, r_edges, phi_edges, z_edges, length)
        prof = prof / prof.max()
        dip = prof[np.argmin(np.abs(zc))]
        axes[3].plot(zc, prof, "--", color="#1f77b4", lw=2, label=f"truth via PCA basis (dip={dip:.2f})")
    axes[3].set_xlim(-3, 3)
    axes[3].set_xlabel("z [kpc]")
    axes[3].set_ylabel("normalized mass per z-bin")
    axes[3].set_title("Bar-end vertical profile")
    axes[3].legend(fontsize=9)

    fig.suptitle(
        f"Subhalo {args.subhalo} ({split} split): does the MDN recover the X? "
        f"(bar radius {length:.1f} kpc)",
        fontsize=15,
    )
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}  split={split}")
    for name, (dip, peak_z) in dips.items():
        print(f"  {name:>9}: midplane/peak={dip:.3f}  peak|z|={peak_z:.2f} kpc")


if __name__ == "__main__":
    main()
