"""Does a TNG-fit residual basis REPRESENT the Shen2010 strong X?

Fits the residual PCA basis on the TNG fine-z train rows two ways -- (a) current
global flattened PCA-32, (b) recommended Fourier-filter (even m<=10) + global
PCA -- then projects the Shen2010 residual (truth - baseline) onto each and asks
whether the reconstruction keeps the boxy/peanut X (b4 boxiness + bar-end vertical
profile + edge-on). This is a representation test (can the basis hold the X),
distinct from whether a model can predict the coefficients from an image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from analyze_mdn_x_recovery import _bar_end_profile
from build_tng50_baseline_density_grid import baseline_density_grid_from_image
from dgdp.density3d import build_cylindrical_density_grid, cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.projection import project_to_mock_image
from dgdp.types import Geometry, ParticleSet
from peanut_strength import _region_for, _shen_aligned_positions, edgeon_surface_density, peanut_strength

EVEN_M = (0, 2, 4, 6, 8, 10)


def filter_even(field):
    coeff = np.fft.rfft(field, axis=2)
    out = np.zeros_like(coeff)
    for m in EVEN_M:
        out[:, :, m, :] = coeff[:, :, m, :]
    return np.fft.irfft(out, n=field.shape[2], axis=2).astype(np.float32)


def pca_fit(x):
    mean = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt


def project(resid_flat, mean, vt, k):
    z = (resid_flat - mean) @ vt[:k].T
    return z @ vt[:k] + mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--output", type=Path, default=Path("outputs/nbody_shen2010/figures/nbody_shen2010_span_check.png"))
    ap.add_argument("--inclination", type=float, default=40.0)
    ap.add_argument("--bar-angle", type=float, default=40.0)
    args = ap.parse_args()

    # ---- fit TNG bases (raw global, filtered global) ----
    t = np.load(args.table)
    re, pe, ze = t["r_edges_kpc"], t["phi_edges_rad"], t["z_edges_kpc"]
    train = t["split"].astype(str) == "train"
    vol = cylindrical_bin_volumes(make_cylindrical_grid_spec(z_max_kpc=float(ze[-1]), n_z=len(ze) - 1)).astype(np.float32)
    tmass = t["truth_density"].astype(np.float32) * vol[None]
    resid = (tmass - t["baseline_density"].astype(np.float32) * vol[None])
    resid = (resid / tmass.sum(axis=(1, 2, 3), keepdims=True)).astype(np.float32)
    del tmass
    n = resid.shape[0]
    g_mean, g_vt = pca_fit(resid.reshape(n, -1)[train])
    residf = filter_even(resid)
    f_mean, f_vt = pca_fit(residf.reshape(n, -1)[train])
    del resid, residf
    print("fitted TNG bases (raw global, filtered global) on", int(train.sum()), "train rows")

    # ---- Shen2010 residual on the same fine grid ----
    spec = make_cylindrical_grid_spec(z_max_kpc=float(ze[-1]), n_z=len(ze) - 1)
    pos = _shen_aligned_positions(args.cache_dir)
    masses = np.full(pos.shape[0], 4.5e10 / pos.shape[0])
    parts = ParticleSet(positions_kpc=pos, masses_msun=masses)
    truth = build_cylindrical_density_grid(parts, spec).density_msun_per_kpc3.astype(np.float32)
    geom = Geometry(inclination_deg=args.inclination, disk_pa_deg=0.0, bar_angle_deg=args.bar_angle)
    image = project_to_mock_image(parts, geom, image_size=192, pixel_scale_kpc=0.35,
                                  psf_sigma_pixels=0.0, noise_sigma_fraction=0.0, seed=0).image
    baseline = baseline_density_grid_from_image(image.astype(float), geometry=geom, pixel_scale_kpc=0.35,
                                                spec=spec, vertical_scale_height_kpc=0.4).density_msun_per_kpc3.astype(np.float32)
    tmass_s = float((truth * vol).sum())
    resid_s = ((truth - baseline) * vol / tmass_s).astype(np.float32)      # raw normalized residual
    resid_sf = filter_even(resid_s[None])[0]                               # filtered

    grid_shape = truth.shape
    recon = {}
    recon["raw global PCA-32"] = project(resid_s.reshape(1, -1), g_mean, g_vt, 32)[0].reshape(grid_shape)
    recon["filtered+global PCA-32"] = project(resid_sf.reshape(1, -1), f_mean, f_vt, 32)[0].reshape(grid_shape)
    recon["filtered+global PCA-64"] = project(resid_sf.reshape(1, -1), f_mean, f_vt, 64)[0].reshape(grid_shape)

    def to_density(rec_norm):
        return np.clip(baseline + (rec_norm * tmass_s) / vol, 0.0, None)

    reg = _region_for(4.4)   # Shen2010 bar length ~4.4 kpc
    fields = {"truth": truth, "baseline": baseline}
    for k, v in recon.items():
        fields[k] = to_density(v)
    print("\nb4 boxiness (Shen2010 bar; >0 boxy/peanut):")
    box = {}
    for name, dens in fields.items():
        x, z, sig = edgeon_surface_density(dens.astype(np.float64), re, pe, ze)
        box[name] = peanut_strength(x, z, sig, **reg)["strength"]
        print(f"  {name:24s} {box[name]:+.4f}")
    # residual reconstruction rel-L2 (filtered target for filtered, raw for raw)
    e_raw = float(np.linalg.norm(recon["raw global PCA-32"] - resid_s) / np.linalg.norm(resid_s))
    e_filt = float(np.linalg.norm(recon["filtered+global PCA-32"] - resid_sf) / np.linalg.norm(resid_sf))
    print(f"\nresidual recon rel-L2: raw global-32 {e_raw:.3f} (vs raw) | filtered+global-32 {e_filt:.3f} (vs filtered)")

    # ---- figure: edge-on + bar-end vertical profile ----
    def edgeon(dens):
        d = dens
        xy = np.arange(-6 + 0.05, 6, 0.1)
        zz = np.arange(-ze[-1] + 0.05, ze[-1], 0.1)
        xg, yg = np.meshgrid(xy, xy, indexing="ij")
        rr = np.hypot(xg, yg)
        pp = np.arctan2(yg, xg)
        ir = np.clip(np.searchsorted(re, rr) - 1, 0, len(re) - 2)
        ip = np.clip(np.searchsorted(pe, pp) - 1, 0, len(pe) - 2)
        valid = rr < re[-1]
        keep = np.abs(xy) <= 1.5
        panel = np.zeros((len(xy), len(zz)))
        for k, zv in enumerate(zz):
            iz = int(np.searchsorted(ze, zv) - 1)
            if 0 <= iz < len(ze) - 1:
                panel[:, k] = np.where(valid, d[ir, ip, iz], 0.0)[:, keep].sum(axis=1)
        return xy, zz, panel

    show = ["truth", "baseline", "raw global PCA-32", "filtered+global PCA-32"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4), constrained_layout=True)
    for ax, name in zip(axes[:4], show, strict=True):
        xy, zz, panel = edgeon(fields[name])
        vmax = float(panel.max())
        ax.imshow(panel.T, origin="lower", extent=[-6, 6, -ze[-1], ze[-1]], cmap="magma",
                  norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax), aspect="auto")
        ax.contour(xy, zz, panel.T, levels=vmax * np.array([0.04, 0.1, 0.25, 0.5]), colors="cyan", linewidths=0.7)
        ax.set_title(f"{name}\nb4={box[name]:+.3f}", fontsize=10)
        ax.set_xlabel("x [kpc] (along bar)")
        ax.set_ylabel("z [kpc]")
        ax.set_ylim(-2.5, 2.5)
    colors = {"truth": "k", "baseline": "0.6", "raw global PCA-32": "#1f77b4", "filtered+global PCA-32": "#b3403c"}
    for name in show:
        zc, prof = _bar_end_profile(fields[name] * vol, re, pe, ze, 4.4)
        prof = prof / prof.max()
        axes[4].plot(zc, prof, "o-", ms=3, lw=1.8, color=colors[name],
                     label=f"{name.split(' PCA')[0]} (dip={prof[np.argmin(np.abs(zc))]:.2f})")
    axes[4].set_xlim(-3, 3)
    axes[4].set_xlabel("z [kpc]")
    axes[4].set_ylabel("normalized mass/z-bin")
    axes[4].set_title("bar-end vertical profile")
    axes[4].legend(fontsize=8)
    fig.suptitle(f"Can a TNG-fit basis represent the Shen2010 X? (i={args.inclination:.0f}, bar={args.bar_angle:.0f})", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
