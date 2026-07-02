"""Diagnostic (read-only, no training): would an explicit mass-dependent sech^2 baseline help
the MDN learn vertical structure better than the current PCA-population-mean centering?

Runs on the cluster against density_residual_table.npz + the committed R=64 bundle. Three
measurements + a verdict, printed to stdout (plots are a bonus):

  1. mass->thickness relation in TNG truth (is there a signal to exploit?)
  2. does the current model already learn it? -> error(pred RMS|z|) vs M* on the val split (decider)
  3. variance ceiling: thickness scatter M* leaves behind (radius/bar/env driven)

See docs/superpowers/specs/2026-07-01-mass-baseline-diagnostic-design.md.

Run (cluster): python scripts/diagnose_mass_baseline.py \
    --table outputs/tng50_milestone2d_rich/density_residual_table.npz \
    --bundle src/dgdp/models/dgdp_fixed_dict.npz --output-dir outputs/diag_mass_baseline
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgdp import vertical_mixture as vm
from dgdp.features import make_features
from dgdp.model import DeprojectionModel


def eff_scalar(rmsz_R, sigma_R, r_grid, rmax=12.0):
    """Sigma-weighted mean RMS|z| over the reliable inner disk (R<rmax). Trailing-axis reduce."""
    m = r_grid < rmax
    w = sigma_R[..., m]
    return (rmsz_R[..., m] * w).sum(-1) / np.maximum(w.sum(-1), 1e-30)


def truth_rmsz(mmap_arr, rows, radial_area, z_grid, chunk=1024):
    """RMS|z|(R) and mass-proportional Sigma(R) for `rows` of a (n,R,phi,z) density memmap.

    vol is constant in phi (uniform grid) and cancels in the RMS|z| ratio, so we only sum the
    density over phi (memory-light, chunked) and fold the per-R annulus area into Sigma weights.
    """
    nr, nz = len(radial_area), len(z_grid)
    dphi_sum = np.zeros((len(rows), nr, nz))                 # density summed over phi
    for i in range(0, len(rows), chunk):
        dphi_sum[i:i + chunk] = np.asarray(mmap_arr[rows[i:i + chunk]]).sum(axis=2)
    col = dphi_sum.sum(axis=2)                               # (rows,R) column density (phi-summed)
    rmsz = np.sqrt((dphi_sum * z_grid[None, None, :] ** 2).sum(axis=2) / np.maximum(col, 1e-30))
    sigma = col * radial_area[None, :]                       # mass-proportional Sigma(R) for weighting
    return rmsz, sigma


def logfit(x, y):
    """least-squares log10(y)=a*log10(x)+b; returns slope a, R^2, Pearson r, residual std (dex)."""
    lx, ly = np.log10(x), np.log10(y)
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    r = float(np.corrcoef(lx, ly)[0, 1])
    return float(a), float(b), float(r2), r, float(np.std(ly - pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, default=Path("src/dgdp/models/dgdp_fixed_dict.npz"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/diag_mass_baseline"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # login node has a per-process memory cap; mmap the two big arrays (truth_density, images)
    # from the npz so only the row subsets we index actually materialise. Small arrays load normally.
    zf = zipfile.ZipFile(args.table)
    print("== table members ==", zf.namelist())
    scratch = args.output_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def mmap_member(key):
        zf.extract(key + ".npy", scratch)                    # disk->disk, low RAM
        return np.load(scratch / (key + ".npy"), mmap_mode="r")

    truth = mmap_member("truth_density")                      # (n,R,phi,z)
    images = mmap_member("images")                            # (n,H,W)
    t = np.load(args.table)                                   # access only the small members below

    split = t["split"].astype(str)
    r_edges, z_edges = t["r_edges_kpc"], t["z_edges_kpc"]
    r_grid = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_grid = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = float(np.diff(z_edges)[0])
    radial_area = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    mstar = t["baseline_grid_mass_msun"].astype(np.float64)  # per-row total stellar mass
    n = len(mstar)
    print(f"truth {truth.shape} {truth.dtype}; images {images.shape} {images.dtype}")
    print(f"\nrows={n}  train={int((split=='train').sum())}  val={int((split=='val').sum())}")

    # galaxy grouping: prefer an explicit id, else dedup projections by exact M*
    idkey = next((k for k in t.files if any(s in k.lower() for s in ("subhalo", "galaxy", "gid"))), None)
    if idkey is not None:
        gid = t[idkey]
        _, gfirst = np.unique(gid, return_index=True)
        print(f"galaxy id key: {idkey} -> {len(gfirst)} galaxies")
    else:
        _, gfirst = np.unique(np.round(mstar, 3), return_index=True)
        print(f"no id key; deduped by M* -> {len(gfirst)} galaxies")

    # ---- 1. mass -> thickness relation in TNG truth (all galaxies) ----
    gr, gs = truth_rmsz(truth, gfirst, radial_area, z_grid)
    eff_g = eff_scalar(gr, gs, r_grid)
    mg = mstar[gfirst]
    a1, b1, r2_1, r_1, sc_1 = logfit(mg, eff_g)
    hz_at = 10 ** (a1 * np.log10([1e10, 10 ** 10.5, 1e11]) + b1)
    print("\n== 1. mass->thickness (TNG truth, per galaxy) ==")
    print(f"  eff RMS|z| range {eff_g.min():.2f}-{eff_g.max():.2f} kpc; M* {mg.min():.2e}-{mg.max():.2e}")
    print(f"  log-log fit slope={a1:.3f}  R^2={r2_1:.2f}  Pearson r={r_1:.2f}  scatter={sc_1:.3f} dex")
    print(f"  hz(M*) -> at 1e10:{hz_at[0]:.2f}  1e10.5:{hz_at[1]:.2f}  1e11:{hz_at[2]:.2f} kpc")

    # ---- 2. does the current model already learn it? (val split, the decider) ----
    m = DeprojectionModel.load(str(args.bundle))
    val = np.flatnonzero(split == "val")
    feat = make_features(np.asarray(images[val]).astype(np.float32), t["metadata"][val].astype(np.float32),
                         baseline_grid_mass_msun=mstar[val].astype(np.float32),
                         image_feature_size=m.image_feature_size,
                         central_pixel_scale_kpc=m.central_pixel_scale_kpc)
    pred = m.predict_weights(feat)
    rk0, K = m.rk_by_m[0], len(m.heights)
    n0 = len(rk0) * K
    w0 = pred[:, :n0].reshape(len(val), len(rk0), K)
    q0 = vm.reconstruct(w0, z_grid, m.heights).real                      # (val,nRk0,nz), unit-integral
    rmsz_knots = np.sqrt((q0 * (z_grid ** 2)[None, None, :]).sum(-1) * dz)
    rmsz_pred = np.array([np.interp(r_grid, rk0, row) for row in rmsz_knots])  # (val,R)
    rmsz_tv, sig_tv = truth_rmsz(truth, val, radial_area, z_grid)
    eff_pred = eff_scalar(rmsz_pred, sig_tv, r_grid)
    eff_tru = eff_scalar(rmsz_tv, sig_tv, r_grid)
    err = eff_pred - eff_tru
    a2, b2, r2_2, r_2, sc_2 = logfit(mstar[val], np.maximum(eff_pred, 1e-3))  # pred vs M* (context)
    slope_err = np.polyfit(np.log10(mstar[val]), err, 1)[0]                   # error vs log10 M*
    r_err = float(np.corrcoef(np.log10(mstar[val]), err)[0, 1])
    print("\n== 2. model residual mass bias (val split) ==")
    print(f"  val rows={len(val)}  eff RMS|z| pred {eff_pred.mean():.2f} vs truth {eff_tru.mean():.2f} kpc")
    print(f"  mean error {err.mean():+.3f}  median {np.median(err):+.3f}  MAE {np.abs(err).mean():.3f} kpc")
    print(f"  error-vs-log10(M*): slope={slope_err:+.3f} kpc/dex  Pearson r={r_err:+.2f}")
    lo, hi = mstar[val] < np.median(mstar[val]), mstar[val] >= np.median(mstar[val])
    print(f"  mean error low-mass half {err[lo].mean():+.3f}  high-mass half {err[hi].mean():+.3f} kpc")

    # ---- 3. variance ceiling: thickness scatter M* leaves behind ----
    resid = np.log10(eff_g) - (a1 * np.log10(mg) + b1)
    print("\n== 3. variance ceiling ==")
    print(f"  thickness spread total {np.std(np.log10(eff_g)):.3f} dex; explained by M* R^2={r2_1:.2f}")
    print(f"  residual thickness scatter after hz(M*): {np.std(resid):.3f} dex "
          f"(radius/bar/env driven, NOT mass-settable)")

    # ---- verdict ----
    relation_ok = abs(r_1) > 0.5 and r2_1 > 0.3
    model_bias = abs(r_err) > 0.3 and abs(slope_err) > 0.1 and np.abs(err).mean() > 0.1
    if relation_ok and model_bias:
        verdict = "RETRAIN LIKELY WORTH IT (strong mass-thickness relation AND unlearned model bias)"
    elif relation_ok and not model_bias:
        verdict = "SKIP RETRAIN (relation exists but the model already learns it; error ~flat in M*)"
    else:
        verdict = "SKIP RETRAIN (weak mass-thickness signal to begin with)"
    print(f"\n== VERDICT: {verdict} ==")

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(mg, eff_g, "o", ms=3, alpha=0.6)
    xs = np.logspace(np.log10(mg.min()), np.log10(mg.max()), 50)
    ax.loglog(xs, 10 ** (a1 * np.log10(xs) + b1), "r-", label=f"slope {a1:.2f}, R²={r2_1:.2f}")
    ax.set(xlabel="M* [Msun]", ylabel="eff RMS|z| [kpc]", title="1. TNG mass->thickness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "mass_thickness.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogx(mstar[val], err, "o", ms=3, alpha=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(xs, slope_err * np.log10(xs) + np.polyfit(np.log10(mstar[val]), err, 1)[1], "r-",
            label=f"slope {slope_err:+.2f} kpc/dex, r={r_err:+.2f}")
    ax.set(xlabel="M* [Msun]", ylabel="pred-truth eff RMS|z| [kpc]", title="2. model error vs mass (val)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_error_vs_mass.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(resid, bins=20)
    ax.set(xlabel="log10 eff RMS|z| residual after hz(M*) [dex]", ylabel="galaxies",
           title=f"3. thickness scatter M* leaves ({np.std(resid):.2f} dex)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "variance_split.png", dpi=130)
    plt.close(fig)
    shutil.rmtree(scratch, ignore_errors=True)               # drop the ~19 GB extracted mmap temp
    print(f"\nwrote plots to {args.output_dir}/")


if __name__ == "__main__":
    main()
