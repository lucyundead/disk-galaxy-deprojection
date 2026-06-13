"""3x3 'input -> truth -> recoveries, face-on and edge-on' figure for one galaxy.

Layout:
    [ inclined input image ] [ truth face-on   ] [ truth edge-on   ]
    [        (blank)        ] [ baseline face-on] [ baseline edge-on]
    [        (blank)        ] [ MDN face-on     ] [ MDN edge-on     ]

The truth 3D density is intrinsic (identical across projections), the baseline
and MDN recoveries are specific to the chosen projection. Face-on and edge-on
panels are produced by resampling the cylindrical (R, phi, z) volume-density
grids onto a Cartesian voxel cube and integrating the surface density along z
(face-on) or along y (edge-on, looking down the bar minor axis so the
boxy/peanut vertical structure is visible). All inputs are existing artifacts;
no inference is run.

Only projections present in the density table can be rendered (the milestone 2c
table stores inclinations 20/40/60 deg and bar angles 0/45/90 deg). Other
viewing geometries require re-rendering the image from particle data on the
cluster.
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

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "font.size": 12,
        "axes.titlesize": 13,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--density-table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subhalo", type=int, required=True)
    parser.add_argument("--inclination", type=float, required=True)
    parser.add_argument("--bar-angle", type=float, required=True)
    parser.add_argument("--image-pixel-scale-kpc", type=float, default=0.35)
    parser.add_argument("--radius-kpc", type=float, default=15.0)
    parser.add_argument("--height-kpc", type=float, default=5.0)
    parser.add_argument("--voxel-kpc", type=float, default=0.25)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _select_row(table, subhalo: int, inclination: float, bar_angle: float) -> int:
    in_galaxy = np.flatnonzero(table["galaxy_id"] == subhalo)
    if in_galaxy.size == 0:
        raise SystemExit(f"subhalo {subhalo} not in table")
    meta = table["metadata"][in_galaxy]
    match = np.flatnonzero(
        np.isclose(meta[:, 0], inclination) & np.isclose(meta[:, 2], bar_angle)
    )
    if match.size == 0:
        available = sorted({(float(m[0]), float(m[2])) for m in meta})
        raise SystemExit(
            f"no projection (i={inclination}, bar={bar_angle}) for subhalo {subhalo}.\n"
            f"available (inclination_deg, bar_angle_deg): {available}"
        )
    return int(in_galaxy[match[0]])


def _resample_cube(
    density: np.ndarray,
    r_edges: np.ndarray,
    phi_edges: np.ndarray,
    z_edges: np.ndarray,
    radius_kpc: float,
    voxel_kpc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a cylindrical volume-density grid onto a Cartesian voxel cube.

    Returns the cube (nx, ny, nz) of volume densities and the z-voxel centers.
    """
    xy = np.arange(-radius_kpc + 0.5 * voxel_kpc, radius_kpc, voxel_kpc)
    z = np.arange(z_edges[0] + 0.5 * voxel_kpc, z_edges[-1], voxel_kpc)
    xg, yg = np.meshgrid(xy, xy, indexing="ij")
    radius = np.sqrt(xg**2 + yg**2)
    phi = np.arctan2(yg, xg)
    ir = np.searchsorted(r_edges, radius) - 1
    iphi = np.searchsorted(phi_edges, phi) - 1
    valid_xy = (ir >= 0) & (ir < len(r_edges) - 1) & (iphi >= 0) & (iphi < len(phi_edges) - 1)
    iz = np.searchsorted(z_edges, z) - 1
    valid_z = (iz >= 0) & (iz < len(z_edges) - 1)

    cube = np.zeros((len(xy), len(xy), len(z)), dtype=np.float64)
    ir_c = np.clip(ir, 0, len(r_edges) - 2)
    iphi_c = np.clip(iphi, 0, len(phi_edges) - 2)
    for k in range(len(z)):
        if not valid_z[k]:
            continue
        plane = density[ir_c, iphi_c, iz[k]]
        cube[:, :, k] = np.where(valid_xy, plane, 0.0)
    return cube, z


def _faceon(cube: np.ndarray, voxel_kpc: float) -> np.ndarray:
    """Surface density (Msun/kpc^2) integrating volume density along z."""
    return cube.sum(axis=2) * voxel_kpc


def _edgeon(cube: np.ndarray, voxel_kpc: float, z: np.ndarray, height_kpc: float) -> np.ndarray:
    """Surface density viewing along y (bar minor axis); returns (x, z) map."""
    panel = cube.sum(axis=1) * voxel_kpc  # (nx, nz)
    keep = np.abs(z) <= height_kpc
    return panel[:, keep]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = np.load(args.density_table)
    preds = np.load(args.predictions)
    row = _select_row(table, args.subhalo, args.inclination, args.bar_angle)

    r_edges = table["r_edges_kpc"].astype(np.float64)
    phi_edges = table["phi_edges_rad"].astype(np.float64)
    z_edges = table["z_edges_kpc"].astype(np.float64)
    volumes = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)

    truth_density = table["truth_density"][row].astype(np.float64)
    baseline_density = table["baseline_density"][row].astype(np.float64)
    pred_row = int(
        np.flatnonzero(
            (preds["galaxy_id"] == table["galaxy_id"][row])
            & (preds["projection_id"] == table["projection_id"][row])
        )[0]
    )
    mdn_density = preds["posterior_mean_mass"][pred_row].astype(np.float64) / volumes

    grids = {
        "Truth": truth_density,
        "Geometric baseline": baseline_density,
        "MDN posterior": mdn_density,
    }
    faceon, edgeon = {}, {}
    for name, density in grids.items():
        cube, z = _resample_cube(density, r_edges, phi_edges, z_edges, args.radius_kpc, args.voxel_kpc)
        faceon[name] = _faceon(cube, args.voxel_kpc)
        edgeon[name] = _edgeon(cube, args.voxel_kpc, z, args.height_kpc)

    fo_max = max(float(p.max()) for p in faceon.values())
    eo_max = max(float(p.max()) for p in edgeon.values())
    fo_norm = LogNorm(vmin=fo_max * 3e-3, vmax=fo_max)
    eo_norm = LogNorm(vmin=eo_max * 3e-3, vmax=eo_max)

    half = args.radius_kpc
    fo_extent = [-half, half, -half, half]
    eo_extent = [-half, half, -args.height_kpc, args.height_kpc]

    image = table["images"][row].astype(np.float64)
    crop = int(round(args.radius_kpc / args.image_pixel_scale_kpc))
    mid = image.shape[0] // 2
    image_crop = image[mid - crop : mid + crop, mid - crop : mid + crop]
    img_extent = [-args.radius_kpc, args.radius_kpc, -args.radius_kpc, args.radius_kpc]

    bar_length = None
    with open(args.manifest, encoding="utf-8") as handle:
        import csv

        for record in csv.DictReader(handle):
            if int(record["subhalo_id"]) == args.subhalo:
                bar_length = float(record["bar_length"])
                break

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.8), constrained_layout=True)
    theta = np.linspace(0, 2 * np.pi, 200)

    def _circle(ax):
        if bar_length:
            ax.plot(bar_length * np.cos(theta), bar_length * np.sin(theta), "w--", lw=1.1, alpha=0.85)

    def _vlines(ax):
        if bar_length:
            for sign in (-1, 1):
                ax.axvline(sign * bar_length, color="w", ls="--", lw=1.0, alpha=0.7)

    # Top-left: inclined input image.
    img_vmax = float(image_crop.max())
    axes[0, 0].imshow(
        image_crop.T,
        origin="lower",
        extent=img_extent,
        cmap="bone",
        norm=LogNorm(vmin=img_vmax * 3e-3, vmax=img_vmax),
    )
    axes[0, 0].set_title(f"Input image\n(i={args.inclination:g} deg, bar PA={args.bar_angle:g} deg)")
    axes[0, 0].set_xlabel("[kpc]")
    axes[0, 0].set_ylabel("[kpc]")

    names = ["Truth", "Geometric baseline", "MDN posterior"]
    for r_i, name in enumerate(names):
        fo_ax = axes[r_i, 1]
        im_fo = fo_ax.imshow(
            faceon[name].T, origin="lower", extent=fo_extent, cmap="magma", norm=fo_norm
        )
        fo_ax.set_title(f"{name} - face-on")
        _circle(fo_ax)
        fo_ax.set_xlabel("x [kpc] (bar axis)")
        fo_ax.set_ylabel("y [kpc]")

        eo_ax = axes[r_i, 2]
        im_eo = eo_ax.imshow(
            edgeon[name].T, origin="lower", extent=eo_extent, cmap="magma", norm=eo_norm, aspect="auto"
        )
        eo_ax.set_title(f"{name} - edge-on")
        _vlines(eo_ax)
        eo_ax.set_xlabel("x [kpc] (bar axis)")
        eo_ax.set_ylabel("z [kpc]")

    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    fig.colorbar(im_fo, ax=[axes[r, 1] for r in range(3)], shrink=0.6, label=r"$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]")
    fig.colorbar(im_eo, ax=[axes[r, 2] for r in range(3)], shrink=0.6, label=r"$\Sigma_\star$ [M$_\odot$ kpc$^{-2}$]")

    fig.suptitle(
        f"Subhalo {args.subhalo}: single inclined image -> 3D recovery "
        f"(bar radius {bar_length:.1f} kpc)" if bar_length else f"Subhalo {args.subhalo}",
        fontsize=15,
    )
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output} (table row {row}, prediction row {pred_row})")


if __name__ == "__main__":
    main()
