"""Projection-spread uncertainty proxy for the learned thickness (ideas 2+3 follow-up).

The idea-2 diagnostic showed the model's eff RMS|z| error is projection-dependent, not
mass-dependent; the idea-3 comparison identified per-galaxy uncertainty as our one real gap vs
Ding's AICp. This read-only cluster diagnostic calibrates the cheap, retrain-free proxy:

  A. variance decomposition -- how much thickness-error variance is within-galaxy
     (viewing-angle-driven) vs between-galaxy (galaxy bias)?
  B. informativeness -- does a galaxy's across-projection prediction spread track its error?
  C. deployment calibration -- |error| quantiles conditioned on INCLINATION (known at inference
     for a real galaxy), i.e. the error bar dgdp could report alongside its point estimate.

Reuses the memmap/table helpers from diagnose_mass_baseline.py. Val split = 37 galaxies x 198
projections. See docs/reports/2026-07-01-mass-baseline-diagnostic.md (side-finding) and
docs/reports/2026-07-01-ding-aicp-comparison.md (gap).

Run (cluster): env PYTHONPATH=src python scripts/diagnose_projection_uncertainty.py \
    --table outputs/tng50_milestone2d_rich/density_residual_table.npz \
    --bundle src/dgdp/models/dgdp_fixed_dict.npz --output-dir outputs/diag_projection_uncertainty
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
from diagnose_mass_baseline import eff_scalar, truth_rmsz  # noqa: E402

from dgdp import vertical_mixture as vm  # noqa: E402
from dgdp.features import make_features  # noqa: E402
from dgdp.model import DeprojectionModel  # noqa: E402


def spearman(a, b):
    """rank correlation without scipy (37 galaxies; ties negligible)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, default=Path("src/dgdp/models/dgdp_fixed_dict.npz"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/diag_projection_uncertainty"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(args.table)
    scratch = args.output_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def mmap_member(key):
        zf.extract(key + ".npy", scratch)
        return np.load(scratch / (key + ".npy"), mmap_mode="r")

    truth = mmap_member("truth_density")
    images = mmap_member("images")
    t = np.load(args.table)
    split = t["split"].astype(str)
    meta = t["metadata"]
    bmass = t["baseline_grid_mass_msun"].astype(np.float32)
    gid = t["galaxy_id"]
    r_edges, z_edges = t["r_edges_kpc"], t["z_edges_kpc"]
    r_grid = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_grid = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = float(np.diff(z_edges)[0])
    radial_area = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)

    val = np.flatnonzero(split == "val")
    incl = meta[val, 0].astype(float)
    barang = meta[val, 2].astype(float)
    gval = gid[val]
    print(f"val rows {len(val)} = {len(np.unique(gval))} galaxies; "
          f"inclinations {np.unique(np.round(incl, 1))}")

    # predicted vs truth eff RMS|z| per row (same construction as diagnose_mass_baseline meas. 2)
    m = DeprojectionModel.load(str(args.bundle))
    feat = make_features(np.asarray(images[val]).astype(np.float32), meta[val].astype(np.float32),
                         baseline_grid_mass_msun=bmass[val], image_feature_size=m.image_feature_size,
                         central_pixel_scale_kpc=m.central_pixel_scale_kpc)
    pred = m.predict_weights(feat)
    rk0, K = m.rk_by_m[0], len(m.heights)
    w0 = pred[:, :len(rk0) * K].reshape(len(val), len(rk0), K)
    q0 = vm.reconstruct(w0, z_grid, m.heights).real
    rmsz_knots = np.sqrt((q0 * (z_grid ** 2)[None, None, :]).sum(-1) * dz)
    rmsz_pred = np.array([np.interp(r_grid, rk0, row) for row in rmsz_knots])
    rmsz_tv, sig_tv = truth_rmsz(truth, val, radial_area, z_grid)
    eff_pred = eff_scalar(rmsz_pred, sig_tv, r_grid)
    eff_tru = eff_scalar(rmsz_tv, sig_tv, r_grid)
    err = eff_pred - eff_tru

    # ---- A. variance decomposition: within-galaxy (projection) vs between-galaxy ----
    galaxies = np.unique(gval)
    spread_g = np.empty(len(galaxies))      # across-projection std of the prediction
    mae_g = np.empty(len(galaxies))
    bias_g = np.empty(len(galaxies))
    tru_g = np.empty(len(galaxies))
    within = 0.0
    for i, g in enumerate(galaxies):
        e = err[gval == g]
        spread_g[i] = float(np.std(eff_pred[gval == g]))
        mae_g[i] = float(np.abs(e).mean())
        bias_g[i] = float(e.mean())
        tru_g[i] = float(eff_tru[gval == g].mean())
        within += float(((e - e.mean()) ** 2).sum())
    var_tot = float(((err - err.mean()) ** 2).sum())
    frac_within = within / max(var_tot, 1e-30)
    print("\n== A. error variance decomposition ==")
    print(f"  within-galaxy (viewing-angle-driven): {100 * frac_within:.0f}%   "
          f"between-galaxy (galaxy bias): {100 * (1 - frac_within):.0f}%")
    print(f"  median per-galaxy |bias| {np.median(np.abs(bias_g)):.3f} kpc; "
          f"median across-projection spread {np.median(spread_g):.3f} kpc")

    # ---- B. is the spread informative of the error? + coverage of spread-as-1-sigma ----
    rho_sm = spearman(spread_g, mae_g)
    cover = float(np.mean(np.abs(err) <= spread_g[np.searchsorted(galaxies, gval)]))
    print("\n== B. spread informativeness ==")
    print(f"  Spearman(spread_g, MAE_g) = {rho_sm:+.2f}  (n={len(galaxies)} galaxies)")
    print(f"  coverage of |err| <= spread(galaxy): {100 * cover:.0f}%  (Gaussian 1-sigma ~ 68%)")

    # ---- C. inclination-conditioned calibration (deployable error bar) ----
    print("\n== C. |error| quantiles by inclination (kpc; frac = /truth) ==")
    print(f"{'incl':>6}{'rows':>7}{'bias':>8}{'P50':>7}{'P68':>7}{'P90':>7}{'P68 frac':>10}")
    uq = np.unique(np.round(incl, 1))
    if len(uq) <= 12:                     # discrete mock grid: one bin per inclination value
        rows_of = [(f"{u:.0f}", np.isclose(np.round(incl, 1), u)) for u in uq]
    else:                                 # continuous: 6 quantile bins
        edges = np.percentile(incl, np.linspace(0, 100, 7))
        edges[-1] += 1e-6
        rows_of = [(f"{lo:.0f}-{hi:.0f}", (incl >= lo) & (incl < hi))
                   for lo, hi in zip(edges[:-1], edges[1:])]
    calib = {}
    for name, sel in rows_of:
        ae = np.abs(err[sel])
        p50, p68, p90 = np.percentile(ae, [50, 68, 90])
        frac68 = np.percentile(ae / np.maximum(eff_tru[sel], 1e-3), 68)
        calib[name] = (p50, p68, p90, frac68)
        print(f"{name:>6}{sel.sum():>7d}{err[sel].mean():>+8.3f}{p50:>7.3f}{p68:>7.3f}{p90:>7.3f}"
              f"{100 * frac68:>9.0f}%")
    ae_all = np.abs(err)
    print(f"{'ALL':>6}{len(err):>7d}{err.mean():>+8.3f}{np.percentile(ae_all, 50):>7.3f}"
          f"{np.percentile(ae_all, 68):>7.3f}{np.percentile(ae_all, 90):>7.3f}"
          f"{100 * np.percentile(ae_all / np.maximum(eff_tru, 1e-3), 68):>9.0f}%")
    print(f"  secondary drivers: Spearman(|err|, incl) = {spearman(ae_all, incl):+.2f}, "
          f"(|err|, bar angle) = {spearman(ae_all, barang):+.2f}")

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(spread_g, mae_g, "o", ms=5, alpha=0.7)
    lim = max(spread_g.max(), mae_g.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k:", lw=0.8)
    ax.set(xlabel="across-projection spread of prediction [kpc]", ylabel="MAE vs truth [kpc]",
           xlim=(0, lim), ylim=(0, lim),
           title=f"B. spread vs error per val galaxy (Spearman {rho_sm:+.2f})")
    fig.tight_layout()
    fig.savefig(args.output_dir / "spread_vs_error.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = np.arange(len(calib))
    for j, lbl in ((0, "P50"), (1, "P68"), (2, "P90")):
        ax.plot(xs, [calib[k][j] for k in calib], "o-", label=lbl)
    ax.set_xticks(xs, list(calib))
    ax.set(xlabel="inclination [deg]", ylabel="|eff RMS|z| error| quantile [kpc]",
           title="C. inclination-conditioned error (the deployable error bar)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "error_vs_inclination.png", dpi=130)
    plt.close(fig)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"\nwrote plots to {args.output_dir}/")


if __name__ == "__main__":
    main()
