"""Idea 3 benchmark: how accurately does our TNG-learned deprojection recover the 3D density,
in the metric Ding et al. report (RMS of log10 density, bar 0.23 / outer 0.15)?

For each TNG val projection we rebuild our model density exactly as the pipeline does
(image features -> predicted q_m weights; in-plane anchor Sigma_m from the geometric baseline;
reconstruct_density) and compare to the TNG truth in log10 space, per radial region. Mass-weighted
(by truth mass) and unweighted-above-floor RMS are both reported. Runs on the cluster, memmap'd,
chunked. Caveat: different simulation/galaxy/viewing-angle than Ding, and per-cell vs their radial
profile -- indicative, not a like-for-like head-to-head.

Run (cluster): env PYTHONPATH=src python scripts/benchmark_density_accuracy.py \
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

from dgdp.features import make_features
from dgdp.harmonics import harmonics, reconstruct_density
from dgdp.model import DeprojectionModel

BAR_MAX, OUT_MAX = 8.0, 22.0            # radial regions [kpc]; bar r<8, outer 8<=r<22 (Ding-style)
CHUNK = 384


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, default=Path("src/dgdp/models/dgdp_fixed_dict.npz"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/diag_mass_baseline"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(args.table)
    scratch = args.output_dir / "_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    def mmap_member(key):
        zf.extract(key + ".npy", scratch)
        return np.load(scratch / (key + ".npy"), mmap_mode="r")

    truth = mmap_member("truth_density")
    baseline = mmap_member("baseline_density")
    images = mmap_member("images")
    t = np.load(args.table)
    split = t["split"].astype(str)
    meta = t["metadata"]
    bmass = t["baseline_grid_mass_msun"].astype(np.float32)
    r_edges, z_edges, phi_edges = t["r_edges_kpc"], t["z_edges_kpc"], t["phi_edges_rad"]
    r_grid = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_grid = 0.5 * (z_edges[:-1] + z_edges[1:])
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    dz = float(np.diff(z_edges)[0])
    vol = (0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)[:, None, None]
           * np.diff(phi_edges)[None, :, None] * np.diff(z_edges)[None, None, :]).astype(np.float64)

    m = DeprojectionModel.load(str(args.bundle))
    val = np.flatnonzero(split == "val")
    regions = {"bar (r<8)": r_grid < BAR_MAX, "outer (8-22)": (r_grid >= BAR_MAX) & (r_grid < OUT_MAX)}
    mw_num = {k: 0.0 for k in regions}
    mw_den = {k: 0.0 for k in regions}
    uw_num = {k: 0.0 for k in regions}
    uw_cnt = {k: 0 for k in regions}
    prof_true = prof_model = None                       # sample radial profile for the figure

    for c in range(0, len(val), CHUNK):
        idx = val[c:c + CHUNK]
        feat = make_features(np.asarray(images[idx]).astype(np.float32), meta[idx].astype(np.float32),
                             baseline_grid_mass_msun=bmass[idx], image_feature_size=m.image_feature_size,
                             central_pixel_scale_kpc=m.central_pixel_scale_kpc)
        vec = m.predict_weights(feat)
        _, sigma_base = harmonics(np.asarray(baseline[idx]).astype(np.float32), dz)
        rho_m = reconstruct_density(vec, sigma_base, m.rk_by_m, m.k_by_m, m.heights, r_grid, z_grid, phi)
        rho_t = np.asarray(truth[idx]).astype(np.float64)
        # remove any global normalisation offset (compare the density DISTRIBUTION, not total mass)
        tot_t = (rho_t * vol).sum(axis=(1, 2, 3))
        tot_m = np.maximum((rho_m * vol).sum(axis=(1, 2, 3)), 1e-30)
        rho_m *= (tot_t / tot_m)[:, None, None, None]

        floor = 1e-4 * rho_t.max(axis=(1, 2, 3), keepdims=True)
        resid = np.log10(np.maximum(rho_m, floor)) - np.log10(np.maximum(rho_t, floor))
        sq = resid ** 2
        w = rho_t * vol                                 # truth mass weight
        fm = rho_t > floor                              # cells with real signal
        for name, Rmask in regions.items():
            sel = np.zeros(rho_t.shape, bool)
            sel[:, Rmask] = True
            mw_num[name] += float((w * sq * sel).sum())
            mw_den[name] += float((w * sel).sum())
            fsel = sel & fm
            uw_num[name] += float((sq * fsel).sum())
            uw_cnt[name] += int(fsel.sum())
        if prof_true is None:                           # azimuth+z-averaged log-density profile, row 0
            colt = (rho_t[0] * vol).sum(axis=(1, 2))
            colm = (rho_m[0] * vol).sum(axis=(1, 2))
            prof_true, prof_model = colt, colm

    print(f"val projections: {len(val)}  regions bar r<{BAR_MAX} / outer {BAR_MAX}-{OUT_MAX} kpc")
    print(f"{'region':>14}{'RMS log rho (mass-wt)':>24}{'RMS log rho (>floor)':>22}{'cells':>12}")
    for name in regions:
        mw = np.sqrt(mw_num[name] / max(mw_den[name], 1e-30))
        uw = np.sqrt(uw_num[name] / max(uw_cnt[name], 1))
        print(f"{name:>14}{mw:>24.3f}{uw:>22.3f}{uw_cnt[name]:>12d}")
    print("\nDing et al. (N-body, non-parametric + AICp): RMS log rho  bar 0.23 / outer 0.15")
    print("NOTE: different sim/galaxy/viewing angle and per-cell vs their radial profile -> indicative only.")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(r_grid, np.log10(np.maximum(prof_model, 1e-30)), "r-", label="model (ours)")
    ax.plot(r_grid, np.log10(np.maximum(prof_true, 1e-30)), "b--", label="true (TNG)")
    ax.axvline(BAR_MAX, color="0.6", ls=":")
    ax.set(xlabel="R [kpc]", ylabel="log10 column mass(R) [Msun]", xlim=(0, OUT_MAX),
           title="density recovery, sample val galaxy (cf. Ding radial profile)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "density_accuracy_profile.png", dpi=130)
    plt.close(fig)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"wrote {args.output_dir}/density_accuracy_profile.png")


if __name__ == "__main__":
    main()
