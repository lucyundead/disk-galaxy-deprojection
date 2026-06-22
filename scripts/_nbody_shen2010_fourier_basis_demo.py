"""Does an azimuthal-Fourier basis (even m<=10) preserve the X while smoothing?

Motivated by the existing face-on image decomposition (keep m=0,2,4,6,8,10).
Extended to 3D: at each (R,z) keep only the even, low-order azimuthal harmonics
of the cylindrical density. This:
  - confirms the t800 X is strong at fine resolution (and that the production
    0.625 kpc grid smears it),
  - measures the azimuthal power spectrum (odd m negligible, high m small),
  - shows the even-m<=10 reconstruction keeps the peanut/X but is much smoother,
  - reports the dimension reduction relevant for a flow-matching target.
Read-only; uses the cached particles and the saved production grids.
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


def bar_frame(cache_dir: Path):
    pos = np.load(cache_dir / "positions_raw.npy").astype(np.float64)
    pos -= pos.mean(axis=0)
    x0, y0, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.hypot(x0, y0)
    inbar = r <= 5.0
    ang = 0.5 * np.angle(np.sum((x0[inbar] + 1j * y0[inbar]) ** 2))
    c, s = np.cos(-ang), np.sin(-ang)
    return c * x0 - s * y0, s * x0 + c * y0, z


def edgeon_from_cyl(density, r_edges, phi_edges, z_edges, *, radius=6.0, voxel=0.08, slab=0.8):
    xy = np.arange(-radius + 0.5 * voxel, radius, voxel)
    zz = np.arange(-z_edges[-1] + 0.5 * voxel, z_edges[-1], voxel)
    xg, yg = np.meshgrid(xy, xy, indexing="ij")
    rr = np.hypot(xg, yg)
    pp = np.arctan2(yg, xg)
    ir = np.clip(np.searchsorted(r_edges, rr) - 1, 0, len(r_edges) - 2)
    ip = np.clip(np.searchsorted(phi_edges, pp) - 1, 0, len(phi_edges) - 2)
    valid = rr < r_edges[-1]
    keep = np.abs(xy) <= slab
    panel = np.zeros((len(xy), len(zz)))
    for k, zv in enumerate(zz):
        iz = int(np.searchsorted(z_edges, zv) - 1)
        if 0 <= iz < len(z_edges) - 1:
            cell = np.where(valid, density[ir, ip, iz], 0.0)
            panel[:, k] = cell[:, keep].sum(axis=1)
    return xy, zz, panel


def pinch(panel, xcen, zcen, frac=0.1):
    level = frac * panel.max()
    h = np.array([
        np.max(np.abs(zcen[panel[xi] >= level])) if np.any(panel[xi] >= level) else 0.0
        for xi in range(panel.shape[0])
    ])
    h = 0.5 * (h + h[::-1])
    c = int(np.argmin(np.abs(xcen)))
    return float(h[c] / h.max()) if h.max() > 0 else 1.0, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--grids", type=Path, default=Path("outputs/nbody_shen2010/nbody_shen2010_grids.npz"))
    ap.add_argument("--output", type=Path, default=Path("outputs/nbody_shen2010/figures/nbody_fourier_basis_demo.png"))
    args = ap.parse_args()

    x, y, z = bar_frame(args.cache_dir)
    n = x.size
    mass = np.full(n, 4.5e10 / n)
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)

    # fine cylindrical grid (0.1 kpc vertical => resolves the X)
    r_edges = np.concatenate(([0.0], np.geomspace(0.1, 12.0, 40)))
    phi_edges = np.linspace(-np.pi, np.pi, 65)   # 64 bins, Nyquist m=32
    z_edges = np.linspace(-4.0, 4.0, 81)         # 80 bins, 0.1 kpc
    massg, _ = np.histogramdd(np.column_stack((r, phi, z)), bins=(r_edges, phi_edges, z_edges), weights=mass)
    area = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    vol = area[:, None, None] * np.diff(phi_edges)[None, :, None] * np.diff(z_edges)[None, None, :]
    dens = massg / vol
    n_phi = len(phi_edges) - 1

    # azimuthal Fourier
    coeff = np.fft.rfft(dens, axis=1)
    w = vol[:, 0, :]  # (R,z) weight
    power = np.array([float(np.sum(np.abs(coeff[:, m, :]) ** 2 * w)) for m in range(coeff.shape[1])])
    power /= power[0]

    def reconstruct(keep_m):
        mask = np.zeros(coeff.shape[1])
        mask[list(keep_m)] = 1.0
        return np.fft.irfft(coeff * mask[None, :, None], n=n_phi, axis=1)

    even_keep = [0, 2, 4, 6, 8, 10]
    dens_even = reconstruct(even_keep)
    dens_le10 = reconstruct(range(0, 11))

    def l2(a):
        return float(np.sqrt(np.sum((a - dens) ** 2 * vol) / np.sum(dens ** 2 * vol)))

    print("azimuthal power (normalized to m=0): "
          + " ".join(f"m{m}={power[m]:.4f}" for m in range(13)))
    print(f"odd-m power fraction: {np.sum(power[1:13:2]) / np.sum(power[:13]):.4f}")
    print(f"L2 recon error even-m<=10: {l2(dens_even):.4f};  all-m<=10: {l2(dens_le10):.4f}")
    full_dim = (len(r_edges) - 1) * n_phi * (len(z_edges) - 1)
    even_dim = len(even_keep) * (len(r_edges) - 1) * (len(z_edges) - 1)
    print(f"dimension: full grid {full_dim} -> even-m<=10 amplitude maps {even_dim} "
          f"({full_dim / even_dim:.1f}x reduction)")

    # production grid truth (coarse) for comparison
    g = np.load(args.grids)
    pvol = (
        0.5 * (g["r_edges_kpc"][1:] ** 2 - g["r_edges_kpc"][:-1] ** 2)[:, None, None]
        * np.diff(g["phi_edges_rad"])[None, :, None]
        * np.diff(g["z_edges_kpc"])[None, None, :]
    )
    pdens = g["truth_mass"] / pvol

    # renders
    xy, zz, p_full = edgeon_from_cyl(dens, r_edges, phi_edges, z_edges)
    _, _, p_even = edgeon_from_cyl(dens_even, r_edges, phi_edges, z_edges)
    px, pz, p_prod = edgeon_from_cyl(pdens, g["r_edges_kpc"], g["phi_edges_rad"], g["z_edges_kpc"])

    # particle thin-slab edge-on (raw strong X)
    sl = np.abs(y) < 0.5
    xe = np.arange(-6, 6.01, 0.05)
    zeh = np.arange(-3, 3.01, 0.05)
    H, _, _ = np.histogram2d(x[sl], z[sl], bins=[xe, zeh])
    H = gaussian_filter(H, 1.5)
    xc = 0.5 * (xe[:-1] + xe[1:])
    zc = 0.5 * (zeh[:-1] + zeh[1:])

    pr_full, _ = pinch(p_full, xy, zz, 0.1)
    pr_even, _ = pinch(p_even, xy, zz, 0.1)
    pr_prod, _ = pinch(p_prod, px, pz, 0.1)
    print(f"pinch ratio (0.1 level): fine-full={pr_full:.3f}  even-m<=10={pr_even:.3f}  "
          f"production-grid={pr_prod:.3f}  [<1 => peanut/X]")

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    vmax = float(H.max())
    ax[0, 0].imshow(H.T, origin="lower", extent=[-6, 6, -3, 3], cmap="magma",
                    norm=LogNorm(vmin=vmax * 3e-3, vmax=vmax), aspect="auto")
    ax[0, 0].contour(xc, zc, H.T, levels=vmax * np.array([0.03, 0.08, 0.18, 0.4]), colors="cyan", linewidths=0.8)
    ax[0, 0].set_title("truth particles, |y|<0.5 kpc (the strong X)")
    for a, panel, xx, zzz, ttl in (
        (ax[0, 1], p_full, xy, zz, f"fine grid 0.1 kpc, full (pinch {pr_full:.2f})"),
        (ax[0, 2], p_even, xy, zz, f"fine grid, even m<=10 (pinch {pr_even:.2f})"),
    ):
        vm = float(panel.max())
        a.imshow(panel.T, origin="lower", extent=[xx[0], xx[-1], zzz[0], zzz[-1]], cmap="magma",
                 norm=LogNorm(vmin=vm * 5e-3, vmax=vm), aspect="auto")
        a.contour(xx, zzz, panel.T, levels=vm * np.array([0.04, 0.1, 0.25, 0.5]), colors="cyan", linewidths=0.8)
        a.set_title(ttl)
    for a in ax[0]:
        a.set_xlabel("x [kpc] (along bar)")
        a.set_ylabel("z [kpc]")
        a.set_ylim(-2.5, 2.5)

    ax[1, 0].semilogy(range(13), power[:13], "o-", color="0.3")
    ax[1, 0].semilogy(range(0, 13, 2), power[0:13:2], "o", color="#d62728", label="even m")
    ax[1, 0].semilogy(range(1, 13, 2), power[1:13:2], "s", color="#1f77b4", label="odd m")
    ax[1, 0].axvline(10.5, color="k", ls=":", lw=1)
    ax[1, 0].set_xlabel("azimuthal order m")
    ax[1, 0].set_ylabel("relative power (norm. to m=0)")
    ax[1, 0].set_title("azimuthal power spectrum")
    ax[1, 0].legend()

    a2 = np.abs(coeff[:, 2, :]) / n_phi
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    zc2 = 0.5 * (z_edges[:-1] + z_edges[1:])
    im = ax[1, 1].pcolormesh(rc, zc2, a2.T, cmap="viridis", shading="auto")
    ax[1, 1].set_xlim(0, 8)
    ax[1, 1].set_ylim(-2.5, 2.5)
    ax[1, 1].set_xlabel("R [kpc]")
    ax[1, 1].set_ylabel("z [kpc]")
    ax[1, 1].set_title("m=2 amplitude map a_2(R,z): the bar/peanut")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)

    for panel, xx, zzz, lab, col in (
        (p_full, xy, zz, "fine full", "k"),
        (p_even, xy, zz, "even m<=10", "#d62728"),
        (p_prod, px, pz, "production 0.625 kpc", "0.5"),
    ):
        _, h = pinch(panel, xx, zzz, 0.1)
        ax[1, 2].plot(xx, h, "-", color=col, lw=2, label=lab)
    ax[1, 2].set_xlim(-5, 5)
    ax[1, 2].set_xlabel("x [kpc] (along bar)")
    ax[1, 2].set_ylabel("contour height |z| [kpc]")
    ax[1, 2].set_title("iso-density height z(x): central dip = X waist")
    ax[1, 2].legend(fontsize=8)

    fig.suptitle("Shen2010 t800: strong X, azimuthal-Fourier (even m<=10) preserves it", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
