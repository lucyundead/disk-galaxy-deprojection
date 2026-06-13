"""Does the boxy/peanut X resolve at finer z-resolution?

Compares the production truth grid (|z|<10 kpc, 0.625 kpc bins, from the density
residual table) against a cluster re-bin (|z|<5 kpc, 0.3125 kpc bins, standalone
hdf5) for one galaxy. Both are intrinsic truth in the same bar-aligned frame.

Two diagnostics:
  - native-resolution vertical mass profile in the bar-end region (no
    interpolation): if the finer grid develops off-plane peaks with a central
    dip, the X is resolved;
  - edge-on (x-z) surface-density maps side by side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from train_density_residual_pca import cylindrical_bin_volumes_from_edges


def _load_new(path: Path):
    with h5py.File(path, "r") as f:
        return (
            f["density_msun_per_kpc3"][:],
            f["r_edges_kpc"][:],
            f["phi_edges_rad"][:],
            f["z_edges_kpc"][:],
        )


def _load_old(table_path: Path, subhalo: int):
    t = np.load(table_path)
    row = int(np.flatnonzero(t["galaxy_id"] == subhalo)[0])
    return (
        t["truth_density"][row].astype(np.float64),
        t["r_edges_kpc"].astype(np.float64),
        t["phi_edges_rad"].astype(np.float64),
        t["z_edges_kpc"].astype(np.float64),
    )


def _bar_end_vertical_profile(density, r_edges, phi_edges, z_edges, r_lo, r_hi):
    """Mass per z-bin summed over bar-end radii and along-bar azimuth."""
    vol = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)
    mass = density * vol
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    pc = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    r_sel = (rc >= r_lo) & (rc <= r_hi)
    along = (np.abs(pc) < np.pi / 6) | (np.abs(np.abs(pc) - np.pi) < np.pi / 6)
    prof = mass[np.ix_(r_sel, along, np.arange(len(zc)))].sum(axis=(0, 1))
    prof = 0.5 * (prof + prof[::-1])  # symmetrize
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
        if iz < 0 or iz >= len(z_edges) - 1:
            continue
        plane = np.where(valid, density[ir, ip, iz], 0.0)
        panel[:, k] = plane[:, keep_y].sum(axis=1) * voxel
    return xy, z, panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-hdf5", type=Path, required=True)
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--subhalo", type=int, required=True)
    parser.add_argument("--bar-length-kpc", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    new = _load_new(args.new_hdf5)
    old = _load_old(args.density_table, args.subhalo)
    r_lo, r_hi = 0.6 * args.bar_length_kpc, 1.1 * args.bar_length_kpc

    z_old, p_old = _bar_end_vertical_profile(*old, r_lo, r_hi)
    z_new, p_new = _bar_end_vertical_profile(*new, r_lo, r_hi)
    p_old = p_old / p_old.max()
    p_new = p_new / p_new.max()

    def bimodality(zc, prof):
        mid = prof[np.argmin(np.abs(zc))]
        peak = prof.max()
        peak_z = abs(zc[np.argmax(prof)])
        return peak_z, mid / peak

    pzo, dipo = bimodality(z_old, p_old)
    pzn, dipn = bimodality(z_new, p_new)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)

    for ax, grid, title in (
        (axes[0], old, "Production: 0.625 kpc z-bins"),
        (axes[1], new, "Re-bin: 0.3125 kpc z-bins"),
    ):
        xy, z, panel = _edgeon(*grid, radius=6.0, height=4.0, voxel=0.1, slab=1.5)
        vmax = float(panel.max())
        ax.imshow(
            panel.T,
            origin="lower",
            extent=[-6, 6, -4, 4],
            cmap="magma",
            norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax),
            aspect="auto",
        )
        ax.contour(xy, z, panel.T, levels=vmax * np.array([0.03, 0.08, 0.2, 0.45]), colors="cyan", linewidths=0.7, alpha=0.7)
        for sign in (-1, 1):
            ax.axvline(sign * args.bar_length_kpc, color="w", ls="--", lw=0.9, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("x [kpc] (along bar)")
        ax.set_ylabel("z [kpc]")

    axes[2].plot(z_old, p_old, "o-", color="k", lw=2, label=f"0.625 kpc (peak|z|={pzo:.2f}, dip={dipo:.2f})")
    axes[2].plot(z_new, p_new, "s-", color="#b3403c", lw=2, label=f"0.3125 kpc (peak|z|={pzn:.2f}, dip={dipn:.2f})")
    axes[2].set_xlim(-3, 3)
    axes[2].set_xlabel("z [kpc]")
    axes[2].set_ylabel("normalized mass per z-bin")
    axes[2].set_title(f"Bar-end vertical profile (R in [{r_lo:.1f}, {r_hi:.1f}] kpc)")
    axes[2].legend(fontsize=9)
    axes[2].annotate(
        "dip < 1 and peak off z=0\n=> X / peanut resolved",
        xy=(0.03, 0.04),
        xycoords="axes fraction",
        fontsize=9,
        color="0.3",
    )

    fig.suptitle(f"Subhalo {args.subhalo}: does the boxy/peanut X resolve at finer z?", fontsize=15)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")
    print(f"old (0.625): peak|z|={pzo:.3f} kpc, midplane/peak={dipo:.3f}")
    print(f"new (0.3125): peak|z|={pzn:.3f} kpc, midplane/peak={dipn:.3f}")


if __name__ == "__main__":
    main()
