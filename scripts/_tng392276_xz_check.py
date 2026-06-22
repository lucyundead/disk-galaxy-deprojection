"""Check 392276's vertical structure: is it a clean symmetric peanut or tilted/
asymmetric (alignment artifact)? And can a low-rank (R,z) basis capture asymmetry?

Renders the bar-frame edge-on (x-z and y-z) maps from the fine-z truth grid,
measures the z-centroid profile <z>(x) (tilt) and up/down mass asymmetry, and
compares the per-m (R,z) SVD rank needed for the raw map vs its z-symmetrized
part (a tilt/shear is high-rank in a separable u(R)v(z) basis).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def edgeon(density, r_edges, phi_edges, z_edges, *, los, radius=8.0, voxel=0.1, slab=1.5):
    """Slab-summed edge-on map. los='y' -> x-z plane (bar side-on); 'x' -> y-z."""
    a = np.arange(-radius + 0.5 * voxel, radius, voxel)
    zz = np.arange(-z_edges[-1] + 0.5 * voxel, z_edges[-1], voxel)
    ag, bg = np.meshgrid(a, a, indexing="ij")  # ag = in-plane shown axis, bg = LOS axis
    rr = np.hypot(ag, bg)
    pp = np.arctan2(bg, ag) if los == "y" else np.arctan2(ag, bg)
    ir = np.clip(np.searchsorted(r_edges, rr) - 1, 0, len(r_edges) - 2)
    ip = np.clip(np.searchsorted(phi_edges, pp) - 1, 0, len(phi_edges) - 2)
    valid = rr < r_edges[-1]
    keep = np.abs(a) <= slab
    panel = np.zeros((len(a), len(zz)))
    for k, zv in enumerate(zz):
        iz = int(np.searchsorted(z_edges, zv) - 1)
        if 0 <= iz < len(z_edges) - 1:
            cell = np.where(valid, density[ir, ip, iz], 0.0)
            panel[:, k] = cell[:, keep].sum(axis=1)
    return a, zz, panel


def rank_for_energy(m, energy=0.99):
    s = np.linalg.svd(m, compute_uv=False)
    c = np.cumsum(s**2) / max(np.sum(s**2), 1e-30)
    return int(np.searchsorted(c, energy) + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, nargs="+", default=[
        Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"),
        Path("/mnt/e/dgdp-milestone2d/density_residual_table_holdout.npz"),
    ])
    ap.add_argument("--subhalo", type=int, default=392276)
    ap.add_argument("--output", type=Path, default=Path("outputs/nbody_shen2010/figures/tng392276_xz_check.png"))
    args = ap.parse_args()

    table = None
    for t in args.tables:
        if not t.exists():
            continue
        gid = np.load(t)["galaxy_id"]
        if args.subhalo in set(np.asarray(gid).tolist()):
            table = t
            break
    if table is None:
        raise SystemExit(f"subhalo {args.subhalo} not found in {args.tables}")
    print(f"using {table}")
    data = np.load(table)
    rows = np.flatnonzero(np.asarray(data["galaxy_id"]) == args.subhalo)
    re, pe, ze = data["r_edges_kpc"], data["phi_edges_rad"], data["z_edges_kpc"]
    dens = data["truth_density"][rows[0]].astype(np.float64)  # bar-frame, geometry-independent
    print(f"rows for {args.subhalo}: {len(rows)}; grid {dens.shape}; dz={ze[1]-ze[0]:.4f} kpc")

    ax_x, zz, pxz = edgeon(dens, re, pe, ze, los="y")  # x-z (bar side-on)
    ax_y, _, pyz = edgeon(dens, re, pe, ze, los="x")   # y-z

    # z-centroid profile <z>(x) in the x-z map (tilt / asymmetry)
    zc = zz
    zcent = np.array([
        np.sum(pxz[i] * zc) / s if (s := pxz[i].sum()) > 0 else 0.0
        for i in range(pxz.shape[0])
    ])
    bar = np.abs(ax_x) <= 4.0
    slope = np.polyfit(ax_x[bar], zcent[bar], 1)[0]  # kpc per kpc
    tilt_deg = float(np.degrees(np.arctan(slope)))

    # up/down mass asymmetry from the 3D grid (bar region R<5 kpc)
    rc = 0.5 * (re[:-1] + re[1:])
    zc3 = 0.5 * (ze[:-1] + ze[1:])
    barmask = rc < 5.0
    up = dens[barmask][:, :, zc3 > 0].sum()
    dn = dens[barmask][:, :, zc3 < 0].sum()
    updown_asym = float((up - dn) / (up + dn))

    # per-m (R,z) rank: raw vs z-symmetrized part
    coeff = np.fft.rfft(dens, axis=1)
    print("per-m (R,z) map: antisymmetric energy fraction and SVD rank (99%):")
    rank_rows = []
    for m in (0, 2, 4, 6):
        cm = coeff[:, m, :]
        cm_sym = 0.5 * (cm + cm[:, ::-1])
        anti_frac = float(np.linalg.norm(cm - cm_sym) / max(np.linalg.norm(cm), 1e-30))
        r_raw = rank_for_energy(cm)
        r_sym = rank_for_energy(cm_sym)
        rank_rows.append((m, anti_frac, r_raw, r_sym))
        print(f"  m={m}: antisym_frac={anti_frac:.3f}  rank_raw={r_raw}  rank_symmetrized={r_sym}")
    print(f"x-z <z>(x) tilt slope={slope:+.3f} ({tilt_deg:+.1f} deg over |x|<4); "
          f"up/down mass asym (R<5)={updown_asym:+.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for a, panel, axis, ttl in (
        (ax[0], pxz, ax_x, "x-z (bar side-on, LOS=y)"),
        (ax[1], pyz, ax_y, "y-z (LOS=x)"),
    ):
        vmax = float(panel.max())
        a.imshow(panel.T, origin="lower", extent=[axis[0], axis[-1], zz[0], zz[-1]], cmap="magma",
                 norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax), aspect="auto")
        a.contour(axis, zz, panel.T, levels=vmax * np.array([0.04, 0.1, 0.25, 0.5]),
                  colors="cyan", linewidths=0.8)
        a.axhline(0, color="white", lw=0.6, ls=":")
        a.set_title(f"{args.subhalo}: {ttl}")
        a.set_xlabel("in-plane [kpc]")
        a.set_ylabel("z [kpc]")
        a.set_ylim(-3, 3)
    ax[2].plot(ax_x, zcent, "o-", color="#b3403c", ms=3)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_xlim(-6, 6)
    ax[2].set_ylim(-1, 1)
    ax[2].set_xlabel("x [kpc] (along bar)")
    ax[2].set_ylabel("<z>(x) [kpc]")
    ax[2].set_title(f"z-centroid: tilt {tilt_deg:+.1f} deg, up/down {updown_asym:+.2f}")
    fig.suptitle(f"Subhalo {args.subhalo}: symmetric peanut or tilted/asymmetric?", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
