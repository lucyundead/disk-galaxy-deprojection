"""Mass-conserving, image-anchored, smooth Fourier x (R,z) deprojection.

Re-parameterize each azimuthal harmonic as a_m(R,z) = Sigma_m(R) * q_m(z;R),
int q_m dz = 1, where:
  - Sigma_m(R) (radial azimuthal-Fourier structure) is MEASURED from the deprojected
    image (geometric baseline) on the FINE grid R - high radial resolution for free,
    so the rotation curve is not degraded by coarse radial knots;
  - q_m(z;R) (normalized vertical profiles) is PREDICTED on the compact, smooth
    power-allocated (R,z) knots (Task-1 allocation) - a smooth, CylSpline-ready field.

Total mass is carried only by m=0 and int q_0 dz = 1, so the output total mass is a
controlled scalar, not a 3D-shape-driven drift. Two conserving variants:
  - "image": total = image mass exactly (pure conservation of the observation);
  - "+masscorr": total = image mass * exp(predicted log M_truth/M_image) - a 1-scalar
    Sigma_0 correction that hits the truth mass while staying exactly enforced.

Compares held-out 3D recovery + mass conservation/accuracy vs the current PCA-on-grid
pipeline. Reconstructs on the cylindrical grid (eval scaffold).

Run:
    .venv/bin/python scripts/deproject_fourier_rz_conserving.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.fourier_rz import _trapezoid_weights, r_knots, z_knots
from deproject_fourier_rz_compare import _interp_matrix, fit_pca, train_mdn
from train_density_residual_pca import make_density_residual_features, standardize_with_train

EVEN_M = (0, 2, 4)


def harmonics(field, dz):
    """Even-m azimuthal harmonics a_m(R,z) and their z-integral Sigma_m(R) on the grid."""
    coeff = np.fft.rfft(field, axis=2) / field.shape[2]
    a = {m: coeff[:, :, m, :].astype(np.complex64) for m in EVEN_M}
    sigma = {m: (a[m].sum(axis=2) * dz) for m in EVEN_M}
    return a, sigma


def sep_resample(cmap, r_src, z_src, r_dst, z_dst):
    """Separable bilinear resample of (rows, R, z) maps between knot/grid axes."""
    w_r, w_z = _interp_matrix(r_src, r_dst), _interp_matrix(z_src, z_dst)
    return np.einsum("Rr,nrz,Zz->nRZ", w_r, cmap, w_z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--allocation", type=Path, default=Path("outputs/nbody_shen2010/fourier_rz_allocation.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    ap.add_argument("--n-comp", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()

    payload = json.loads(args.allocation.read_text(encoding="utf-8"))
    alloc = {int(m): (int(a), int(b)) for m, (a, b) in payload["allocation"].items()}
    kp = payload["knot_params"]
    knots = {m: (r_knots(nr, kp["r_max"], kp["r_min"]), z_knots(nzh, kp["z_max"], kp["z_min"]))
             for m, (nr, nzh) in alloc.items()}
    wz = {m: _trapezoid_weights(knots[m][1]) for m in alloc}

    table = np.load(args.table)
    split = table["split"].astype(str)
    train, val, test = split == "train", split == "val", split == "test"
    z_edges, r_edges, phi_edges = table["z_edges_kpc"], table["r_edges_kpc"], table["phi_edges_rad"]
    dz = float(np.diff(z_edges)[0])
    r_grid = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_grid = 0.5 * (z_edges[:-1] + z_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    vol = cylindrical_bin_volumes(make_cylindrical_grid_spec(z_max_kpc=float(z_edges[-1]), n_z=len(z_edges) - 1)).astype(np.float32)

    truth = table["truth_density"].astype(np.float32)
    baseline = table["baseline_density"].astype(np.float32)
    truth_mass = truth * vol[None]
    baseline_mass = baseline * vol[None]
    baseline_total = baseline_mass.sum(axis=(1, 2, 3))
    truth_total = truth_mass.sum(axis=(1, 2, 3))

    a_true, sigma_true = harmonics(truth, dz)
    _, sigma_img = harmonics(baseline, dz)  # image-anchored radial structure (fine grid R)
    print(f"rows: train {train.sum()} val {val.sum()} test {test.sum()}; grid {truth.shape[1:]}")

    feat = make_density_residual_features(
        table["images"].astype(np.float32), table["metadata"].astype(np.float32),
        baseline_grid_mass_msun=table["baseline_grid_mass_msun"].astype(np.float32),
        image_feature_size=24, central_pixel_scale_kpc=0.35,
    )
    x_all, _, _ = standardize_with_train(feat, train)

    # ---- (step a) radial-anchor correction heads: predict Sigma_m(R) corrections ----
    # The diagnosed bottleneck is the image radial anchor Sigma_m(R). Dedicated heads
    # (so m=0 capacity isn't diluted): a real log-ratio for m=0 (subsumes the total-mass
    # correction), a complex ratio c_m(R)=Sigma_m_true/Sigma_m_image for the bar m=2,4.
    n_r_grid = sigma_img[0].shape[1]

    def predict_head(target_arr, n_pca, seed_off):
        cc, cv, ccm = fit_pca(target_arr.astype(np.float32), train, min(n_pca, args.n_comp))
        yc, ym, ys = standardize_with_train(cc, train)
        h = train_mdn(x_all, yc, train, val, seed=args.seed + seed_off, epochs=args.epochs)
        with torch.no_grad():
            return (h.sample(torch.tensor(x_all[test]), args.n_samples).numpy().mean(axis=1) * ys + ym) @ cv + ccm

    s0i, s0t = sigma_img[0].real, sigma_true[0].real
    f0 = 1e-3 * np.abs(s0i).max(axis=1, keepdims=True)
    logr0_tgt = np.clip(np.where((np.abs(s0i) > f0) & (s0t > 0),
                                 np.log(np.maximum(s0t, 1e-30) / np.maximum(s0i, 1e-30)), 0.0), -1.5, 1.5)
    s0_corr = (sigma_img[0][test].real * np.exp(predict_head(logr0_tgt, 16, 7))).astype(np.complex128)

    c24_parts = []
    for m in (2, 4):
        si, st = sigma_img[m], sigma_true[m]
        fm = 1e-2 * np.abs(si).max(axis=1, keepdims=True)
        c = np.where(np.abs(si) > fm, st / np.where(np.abs(si) > 0, si, 1.0), 1.0 + 0j)
        c24_parts += [np.clip(c.real, -3.0, 5.0), np.clip(c.imag, -4.0, 4.0)]
    pc = predict_head(np.concatenate(c24_parts, axis=1), 24, 11)
    s2_corr = sigma_img[2][test] * (pc[:, :n_r_grid] + 1j * pc[:, n_r_grid:2 * n_r_grid])
    s4_corr = sigma_img[4][test] * (pc[:, 2 * n_r_grid:3 * n_r_grid] + 1j * pc[:, 3 * n_r_grid:4 * n_r_grid])

    truth_norm = np.linalg.norm((truth_mass[test] / vol[None]).reshape(test.sum(), -1), axis=1)

    def recovery(pred_mass):
        pred_total = pred_mass.sum(axis=(1, 2, 3))
        return {
            "cell_mass_mae": float(np.mean(np.abs(pred_mass - truth_mass[test]))),
            "rel_l2_density": float(np.mean(np.linalg.norm(((pred_mass - truth_mass[test]) / vol[None]).reshape(test.sum(), -1), axis=1)
                                            / np.maximum(truth_norm, 1e-30))),
            "mass_cons_vs_image": float(np.mean(np.abs(pred_total - baseline_total[test]) / baseline_total[test])),
            "mass_acc_vs_truth": float(np.mean(np.abs(pred_total - truth_total[test]) / truth_total[test])),
        }

    def rescale(pred_mass, target_total):
        total = pred_mass.sum(axis=(1, 2, 3))
        return pred_mass * (target_total / np.maximum(total, 1e-30))[:, None, None, None]

    results = {"geom_baseline": recovery(baseline_mass[test])}

    # ---- current PCA-on-grid pipeline (non-conserving residual) ----
    resid = ((truth_mass - baseline_mass) / baseline_total[:, None, None, None]).astype(np.float32).reshape(truth.shape[0], -1)
    coeff, vec, mean = fit_pca(resid, train, args.n_comp)
    y_all, y_mean, y_std = standardize_with_train(coeff, train)
    mdn = train_mdn(x_all, y_all, train, val, seed=args.seed, epochs=args.epochs)
    with torch.no_grad():
        delta = ((mdn.sample(torch.tensor(x_all[test]), args.n_samples).numpy().mean(axis=1) * y_std + y_mean) @ vec + mean)
    grid_pred = baseline_mass[test] + delta.reshape(test.sum(), *truth.shape[1:]) * baseline_total[test, None, None, None]
    results["grid_mdn"] = recovery(grid_pred)
    del resid

    # ---- (step 3) smooth-knot conserving target: q_m(z;R) on power-allocated knots ----
    floor = 1e-3 * np.abs(sigma_true[0]).max(axis=1, keepdims=True)
    target_parts, q_true_knot = [], {}
    for m in EVEN_M:
        rk, zk = knots[m]
        a_knot = sep_resample(a_true[m], r_grid, z_grid, rk, zk)          # (rows, nRk, nzk)
        sig_knot = (a_knot * wz[m][None, None, :]).sum(axis=2)            # int a dz on knots -> (rows, nRk)
        mask = (np.abs(sig_knot) > floor)[:, :, None]
        q = np.where(mask, a_knot / np.where(np.abs(sig_knot) > 0, sig_knot, 1.0)[:, :, None], 0.0)
        q_true_knot[m] = q
        target_parts.append(q.real.reshape(q.shape[0], -1))
        if m != 0:
            target_parts.append(q.imag.reshape(q.shape[0], -1))
    target = np.concatenate(target_parts, axis=1).astype(np.float32)
    coeff, vec, mean = fit_pca(target, train, args.n_comp)
    y_all, y_mean, y_std = standardize_with_train(coeff, train)
    mdn = train_mdn(x_all, y_all, train, val, seed=args.seed, epochs=args.epochs)
    with torch.no_grad():
        pred_scores = mdn.sample(torch.tensor(x_all[test]), args.n_samples).numpy().mean(axis=1)
    pred_vec = (pred_scores * y_std + y_mean) @ vec + mean

    def reconstruct_smooth(vec_rows, anchor):
        """Predicted q on knots -> renormalize -> anchor radial Sigma_m(R) -> grid density.

        anchor: {m: (rows, nR_grid) complex}. Returns (clipped_mass, preclip_total); the
        m>0 harmonics carry zero net mass, so preclip_total is the controlled m=0 total.
        """
        rows = vec_rows.shape[0]
        rho = np.zeros((rows, len(r_grid), len(phi_centers), len(z_grid)))
        i = 0
        for m in EVEN_M:
            rk, zk = knots[m]
            size = len(rk) * len(zk)
            re = vec_rows[:, i:i + size].reshape(rows, len(rk), len(zk))
            i += size
            if m == 0:
                q = np.clip(re, 0.0, None).astype(np.complex128)
            else:
                im = vec_rows[:, i:i + size].reshape(rows, len(rk), len(zk))
                i += size
                q = re + 1j * im
            denom = (q * wz[m][None, None, :]).sum(axis=2)[:, :, None]      # int q dz on knots
            q = q / np.where(np.abs(denom) > 1e-12, denom, 1.0)            # enforce int q dz = 1
            q_grid = sep_resample(q, rk, zk, r_grid, z_grid)              # smooth knot -> grid
            a_m = anchor[m][:, :, None] * q_grid                          # anchor radial Sigma_m(R)
            if m == 0:
                rho += a_m.real[:, :, None, :]
            else:
                cos_m, sin_m = np.cos(m * phi_centers), np.sin(m * phi_centers)
                rho += 2.0 * (a_m.real[:, :, None, :] * cos_m[None, None, :, None]
                              - a_m.imag[:, :, None, :] * sin_m[None, None, :, None])
        preclip_total = (rho * vol[None]).sum(axis=(1, 2, 3))
        return np.clip(rho, 0.0, None) * vol[None], preclip_total

    def variant(anchor):
        clipped, ptot = reconstruct_smooth(pred_vec, anchor)
        return recovery(rescale(clipped, ptot))

    results["fourier_cons_image_anchor"] = variant({m: sigma_img[m][test] for m in EVEN_M})
    results["fourier_cons_radial_m0"] = variant({0: s0_corr, 2: sigma_img[2][test], 4: sigma_img[4][test]})
    results["fourier_cons_radial_all"] = variant({0: s0_corr, 2: s2_corr, 4: s4_corr})
    results["fourier_cons_oracle_anchor"] = variant({m: sigma_true[m][test] for m in EVEN_M})

    # ---- report ----
    order = ["geom_baseline", "grid_mdn", "fourier_cons_image_anchor", "fourier_cons_radial_m0",
             "fourier_cons_radial_all", "fourier_cons_oracle_anchor"]
    ntgt = dict.fromkeys(order, target.shape[1])
    ntgt["geom_baseline"], ntgt["grid_mdn"] = "-", int(truth[0].size)
    print(f"\n{'method':>32} {'n_target':>9} {'cellMAE':>10} {'relL2':>7} {'massAcc':>8}")
    for name in order:
        r = results[name]
        print(f"{name:>32} {str(ntgt[name]):>9} {r['cell_mass_mae']:>10.3e} {r['rel_l2_density']:>7.3f} "
              f"{r['mass_acc_vs_truth']:>8.4f}")
    print("\nAnchors share the SAME predicted vertical profiles q_m; only Sigma_m(R) differs: image "
          "(measured),\nradial_m0 (predicted Sigma_0(R) correction), radial_all (+ complex Sigma_2,4(R) bar "
          "corrections),\noracle (true Sigma_m = ceiling). massAcc=|M_out-M_truth|/M_truth. Gap image->oracle "
          "= radial-anchor headroom.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fourier_rz_conserving_metrics.json").write_text(
        json.dumps({"n_comp": args.n_comp, "n_target_conserving": int(target.shape[1]), "results": results},
                   indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output_dir / 'fourier_rz_conserving_metrics.json'}")

    # ---- figure ----
    labels = ["geom\nbaseline", "grid MDN", "cons.\nimage anchor", "cons.\n+radial m=0",
              "cons.\n+radial all", "cons.\noracle anchor"]
    colors = ["#999999", "#4c72b0", "#c44e52", "#dd8452", "#8172b3", "#55a868"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    for axis, key, ylab, title in (
        (ax[0], "cell_mass_mae", "3D cell-mass MAE [Msun]", "Held-out 3D recovery"),
        (ax[1], "rel_l2_density", "3D density rel-L2 vs truth", "Held-out density rel-L2"),
        (ax[2], "mass_acc_vs_truth", "|M_out - M_truth| / M_truth", "Total-mass accuracy vs truth"),
    ):
        axis.bar(labels, [results[k][key] for k in order], color=colors)
        axis.set_ylabel(ylab)
        axis.set_title(f"{title} (lower better)")
        axis.tick_params(axis="x", labelsize=8)
    fig.suptitle("Radial-anchor correction (a): image -> +Sigma_0(R) -> +Sigma_0,2,4(R) -> oracle "
                 "(shared vertical profiles)", fontsize=12)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "figures" / "fourier_rz_conserving_compare.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
