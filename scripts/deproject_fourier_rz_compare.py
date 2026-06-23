"""Head-to-head deprojection: even-m Fourier x (R,z) target vs the PCA-on-grid target.

Everything is held equal except the *target representation* the model predicts:

  - PCA-on-grid (the current pipeline): flatten the cylindrical residual grid,
    then cross-galaxy PCA-K;
  - Fourier x (R,z) (new, potential-ready): azimuthally FFT the residual grid,
    keep even m<=10 on a power-weighted per-harmonic (R,z) allocation (the few-
    hundred-coefficient compact target from compress_fourier_rz_target.py), then
    cross-galaxy PCA-K of those coefficients.

The residual is delta_mass / baseline_total (test-time computable). Both targets
get the same 586-dim image+geometry features, the same galaxy-level milestone-2d
split, and the same 1-Gaussian MDN; the Fourier target additionally gets a
conditional flow. Held-out recovery reconstructs each predicted residual back onto
the cylindrical grid (the eval scaffold), adds the geometric baseline, and scores
3D cell-mass MAE and density rel-L2 against the TNG truth.

Run:
    .venv/bin/python scripts/deproject_fourier_rz_compare.py
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
from dgdp.fourier_rz import (
    EVEN_M,
    derive_power_allocation_multi,
    fit_fourier_rz_from_grid,
    r_knots,
    z_knots,
)
from dgdp.models.mdn import SummaryResidualMDN
from flow_matching_prototype import FlowVel, batches, flow_sample
from train_density_residual_pca import make_density_residual_features, standardize_with_train


# --------------------------------------------------------------------------- #
# Separable bilinear resampling between the grid (r,z) and the Fourier knots.
# --------------------------------------------------------------------------- #
def _interp_matrix(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """1-D linear-interpolation weight matrix (len(dst), len(src)); dst is clamped."""
    src = np.asarray(src, dtype=float)
    dst = np.clip(np.asarray(dst, dtype=float), src[0], src[-1])
    idx = np.clip(np.searchsorted(src, dst) - 1, 0, len(src) - 2)
    frac = (dst - src[idx]) / (src[idx + 1] - src[idx])
    weights = np.zeros((len(dst), len(src)))
    rows = np.arange(len(dst))
    weights[rows, idx] = 1.0 - frac
    weights[rows, idx + 1] = frac
    return weights


def build_fourier_coeffs(resid_grid, r_centers, z_centers, alloc, r_max, z_max):
    """Per-row even-m Fourier x (R,z) coefficient vectors of a residual grid.

    ``resid_grid``: (rows, nR, nphi, nz). Returns ``(coeffs (rows, n_coeff), knots)``.
    """
    fourier = np.fft.rfft(resid_grid, axis=2) / resid_grid.shape[2]  # (rows, nR, K, nz)
    parts, knots = [], {}
    for m in sorted(alloc):
        n_r_m, n_zh_m = alloc[m]
        rk, zk = r_knots(n_r_m, r_max), z_knots(n_zh_m, z_max)
        w_r, w_z = _interp_matrix(r_centers, rk), _interp_matrix(z_centers, zk)
        compact = np.einsum("Rr,nrz,Zz->nRZ", w_r, fourier[:, :, m, :], w_z)  # (rows, nRk, nzk)
        knots[m] = (rk, zk)
        parts.append(compact.real.reshape(compact.shape[0], -1))
        if m != 0:
            parts.append(compact.imag.reshape(compact.shape[0], -1))
    return np.concatenate(parts, axis=1).astype(np.float32), knots


def reconstruct_fourier_on_grid(coeffs, knots, r_centers, phi_centers, z_centers):
    """Inverse of :func:`build_fourier_coeffs`: signed residual field on the grid."""
    rows = coeffs.shape[0]
    field = np.zeros((rows, len(r_centers), len(phi_centers), len(z_centers)), dtype=np.float64)
    i = 0
    for m in sorted(knots):
        rk, zk = knots[m]
        size = len(rk) * len(zk)
        re = coeffs[:, i : i + size].reshape(rows, len(rk), len(zk))
        i += size
        if m == 0:
            compact = re.astype(np.complex128)
        else:
            im = coeffs[:, i : i + size].reshape(rows, len(rk), len(zk))
            i += size
            compact = re + 1j * im
        w_r, w_z = _interp_matrix(rk, r_centers), _interp_matrix(zk, z_centers)
        on_grid = np.einsum("rR,nRZ,zZ->nrz", w_r, compact, w_z)  # (rows, nR, nz) complex
        if m == 0:
            field += on_grid.real[:, :, None, :]
        else:
            cos_m, sin_m = np.cos(m * phi_centers), np.sin(m * phi_centers)
            field += 2.0 * (on_grid.real[:, :, None, :] * cos_m[None, None, :, None]
                            - on_grid.imag[:, :, None, :] * sin_m[None, None, :, None])
    return field


# --------------------------------------------------------------------------- #
# Shared modelling pieces.
# --------------------------------------------------------------------------- #
def fit_pca(target, train_mask, n_comp):
    mean = target[train_mask].mean(axis=0)
    _, _, vt = np.linalg.svd(target[train_mask] - mean, full_matrices=False)
    components = vt[:n_comp]
    coeff = (target - mean) @ components.T
    return coeff.astype(np.float32), components.astype(np.float32), mean.astype(np.float32)


def train_mdn(x_all, y_all, train, val, *, seed, epochs, patience=60):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xt, yt = torch.tensor(x_all), torch.tensor(y_all)
    tr, va = np.flatnonzero(train), np.flatnonzero(val)
    mdn = SummaryResidualMDN(input_dim=x_all.shape[1], output_dim=y_all.shape[1], hidden_dim=128, n_components=1)
    opt = torch.optim.Adam(mdn.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, wait = 1e9, None, 0
    for _ in range(epochs):
        mdn.train()
        for b in batches(len(tr), 64, rng):
            opt.zero_grad()
            mdn.negative_log_likelihood(xt[tr[b]], yt[tr[b]]).backward()
            opt.step()
        mdn.eval()
        with torch.no_grad():
            vl = float(mdn.negative_log_likelihood(xt[va], yt[va]))
        if vl < best - 1e-4:
            best, best_state, wait = vl, {k: v.clone() for k, v in mdn.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    mdn.load_state_dict(best_state)
    return mdn


def train_flow(x_all, y_all, train, val, *, seed, epochs, patience=80):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    xt, yt = torch.tensor(x_all), torch.tensor(y_all)
    tr, va = np.flatnonzero(train), np.flatnonzero(val)
    k = y_all.shape[1]
    flow = FlowVel(k, x_all.shape[1], hidden=256)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    v_x0 = torch.randn(len(va), k, generator=g)
    v_t = torch.rand(len(va), 1, generator=g)
    v_xt = (1 - v_t) * v_x0 + v_t * yt[va]
    v_tgt = yt[va] - v_x0
    best, best_state, wait = 1e9, None, 0
    for _ in range(epochs):
        flow.train()
        for b in batches(len(tr), 128, rng):
            yb = yt[tr[b]]
            x0 = torch.randn_like(yb)
            tt = torch.rand(yb.shape[0], 1)
            xtt = (1 - tt) * x0 + tt * yb
            opt.zero_grad()
            ((flow(xtt, tt, xt[tr[b]]) - (yb - x0)) ** 2).mean().backward()
            opt.step()
        flow.eval()
        with torch.no_grad():
            vl = float(((flow(v_xt, v_t, xt[va]) - v_tgt) ** 2).mean())
        if vl < best - 1e-5:
            best, best_state, wait = vl, {k2: v.clone() for k2, v in flow.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    flow.load_state_dict(best_state)
    return flow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    ap.add_argument("--n-comp", type=int, default=32)
    ap.add_argument("--capture", type=float, default=0.99)
    ap.add_argument("--r-max", type=float, default=15.0)
    ap.add_argument("--z-max", type=float, default=4.0)
    ap.add_argument("--n-alloc-rows", type=int, default=48)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    table = np.load(args.table)
    split = table["split"].astype(str)
    train, val, test = split == "train", split == "val", split == "test"
    z_edges, r_edges, phi_edges = table["z_edges_kpc"], table["r_edges_kpc"], table["phi_edges_rad"]
    vol = cylindrical_bin_volumes(
        make_cylindrical_grid_spec(z_max_kpc=float(z_edges[-1]), n_z=len(z_edges) - 1)
    ).astype(np.float32)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    truth_mass = (table["truth_density"].astype(np.float32) * vol[None]).astype(np.float32)
    baseline_mass = (table["baseline_density"].astype(np.float32) * vol[None]).astype(np.float32)
    baseline_total = baseline_mass.sum(axis=(1, 2, 3))
    resid = ((truth_mass - baseline_mass) / baseline_total[:, None, None, None]).astype(np.float32)
    print(f"rows: train {train.sum()} val {val.sum()} test {test.sum()}; grid {resid.shape[1:]}")

    # ---- features (shared) ----
    feat = make_density_residual_features(
        table["images"].astype(np.float32), table["metadata"].astype(np.float32),
        baseline_grid_mass_msun=table["baseline_grid_mass_msun"].astype(np.float32),
        image_feature_size=24, central_pixel_scale_kpc=0.35,
    )
    x_all, _, _ = standardize_with_train(feat, train)

    # ---- derive the Fourier allocation on the residual itself ----
    sample_idx = rng.choice(np.flatnonzero(train), size=min(args.n_alloc_rows, int(train.sum())), replace=False)
    alloc_models = [
        fit_fourier_rz_from_grid(resid[i], r_centers, z_centers, r_max=args.r_max, z_max=args.z_max)
        for i in sample_idx
    ]
    alloc, alloc_info = derive_power_allocation_multi(alloc_models, capture=args.capture)
    alloc = dict(sorted(alloc.items()))
    n_fourier_coeff = int(sum((1 if m == 0 else 2) * nr * (2 * nzh + 1) for m, (nr, nzh) in alloc.items()))
    print("residual aggregate power fraction: "
          + "  ".join(f"m{m}={alloc_info['power_fraction'].get(m, 0):.2e}" for m in EVEN_M))
    print(f"Fourier allocation (capture {args.capture}): "
          + "  ".join(f"m{m}={alloc[m]}" for m in sorted(alloc)) + f"  -> {n_fourier_coeff} coeff")

    # ---- two targets ----
    grid_target = resid.reshape(resid.shape[0], -1)
    fourier_target, knots = build_fourier_coeffs(resid, r_centers, z_centers, alloc, args.r_max, args.z_max)

    # ---- recovery metric helpers (reconstruct residual -> grid -> +baseline -> vs truth) ----
    truth_norm = np.linalg.norm((truth_mass[test] / vol[None]).reshape(test.sum(), -1), axis=1)
    base_cellmae = float(np.mean(np.abs(baseline_mass[test] - truth_mass[test])))
    base_rel = float(np.mean(
        np.linalg.norm(((baseline_mass[test] - truth_mass[test]) / vol[None]).reshape(test.sum(), -1), axis=1)
        / np.maximum(truth_norm, 1e-30)))

    def phys(mass):
        """Smooth physical summaries: vertical RMS-z [kpc] and global bar m=2 amplitude."""
        total = np.maximum(mass.sum(axis=(1, 2, 3)), 1e-30)
        rms_z = np.sqrt((mass.sum(axis=(1, 2)) * (z_centers ** 2)[None]).sum(1) / total)
        radial_phi = mass.sum(axis=3)
        m2 = (np.abs((radial_phi * np.exp(2j * phi_centers)[None, None, :]).sum(axis=(1, 2)))
              / np.maximum(radial_phi.sum(axis=(1, 2)), 1e-30))
        return rms_z, m2

    truth_rms_z, truth_m2 = phys(truth_mass[test])

    def recovery(field_test):
        """field_test: (n_test, nR, nphi, nz) predicted residual fraction -> recovery scores."""
        pred_mass = baseline_mass[test] + field_test * baseline_total[test, None, None, None]
        cellmae = float(np.mean(np.abs(pred_mass - truth_mass[test])))
        rel = float(np.mean(
            np.linalg.norm(((pred_mass - truth_mass[test]) / vol[None]).reshape(test.sum(), -1), axis=1)
            / np.maximum(truth_norm, 1e-30)))
        rms_z, m2 = phys(pred_mass)
        return {"cell_mass_mae": cellmae, "rel_l2_density": rel,
                "rms_z_mae": float(np.mean(np.abs(rms_z - truth_rms_z))),
                "m2_amp_mae": float(np.mean(np.abs(m2 - truth_m2)))}

    def to_grid(coeff_vec, basis):
        if basis == "grid":
            return coeff_vec.reshape(coeff_vec.shape[0], *resid.shape[1:])
        return reconstruct_fourier_on_grid(coeff_vec, knots, r_centers, phi_centers, z_centers)

    base_rms_z, base_m2 = phys(baseline_mass[test])
    results = {"baseline": {"cell_mass_mae": base_cellmae, "rel_l2_density": base_rel,
                            "rms_z_mae": float(np.mean(np.abs(base_rms_z - truth_rms_z))),
                            "m2_amp_mae": float(np.mean(np.abs(base_m2 - truth_m2)))}}
    summary_rows = []

    def resid_rel(field):
        return float(np.linalg.norm(field - resid[test]) / np.linalg.norm(resid[test]))

    def coverage(samp, std_coeff):
        lo, hi = np.quantile(samp, 0.16, axis=1), np.quantile(samp, 0.84, axis=1)
        return float(np.mean((std_coeff >= lo) & (std_coeff <= hi)))

    for basis, target in (("grid", grid_target), ("fourier", fourier_target)):
        coeff, vec, mean = fit_pca(target, train, args.n_comp)
        y_all, y_mean, y_std = standardize_with_train(coeff, train)
        evr = float(1.0 - np.sum((target[train] - (coeff[train] @ vec + mean)) ** 2)
                    / np.sum((target[train] - mean) ** 2))
        std_coeff = (coeff[test] - y_mean) / y_std

        oracle_field = to_grid((coeff[test] @ vec + mean), basis)
        oracle = recovery(oracle_field) | {"resid_rel_l2": resid_rel(oracle_field)}

        mdn = train_mdn(x_all, y_all, train, val, seed=args.seed, epochs=args.epochs)
        with torch.no_grad():
            mdn_samp = mdn.sample(torch.tensor(x_all[test]), args.n_samples).numpy()
        mdn_field = to_grid((mdn_samp.mean(axis=1) * y_std + y_mean) @ vec + mean, basis)
        mdn = recovery(mdn_field) | {"resid_rel_l2": resid_rel(mdn_field),
                                     "coeff_coverage68": coverage(mdn_samp, std_coeff)}

        results[f"{basis}_mdn"] = {"n_target": int(target.shape[1]), "pca_evr_train": evr,
                                   "oracle": oracle, "mdn": mdn}
        summary_rows.append((f"{basis} MDN", target.shape[1], evr, mdn))
        print(f"\n[{basis}] target {target.shape[1]} -> PCA{args.n_comp} (EVR {evr:.3f})")
        print(f"  oracle: cellMAE {oracle['cell_mass_mae']:.3e} relL2 {oracle['rel_l2_density']:.3f} "
              f"residRel {oracle['resid_rel_l2']:.3f} RMSz-MAE {oracle['rms_z_mae']:.3f} m2-MAE {oracle['m2_amp_mae']:.4f}")
        print(f"  MDN:    cellMAE {mdn['cell_mass_mae']:.3e} relL2 {mdn['rel_l2_density']:.3f} "
              f"residRel {mdn['resid_rel_l2']:.3f} RMSz-MAE {mdn['rms_z_mae']:.3f} m2-MAE {mdn['m2_amp_mae']:.4f} cov68 {mdn['coeff_coverage68']:.3f}")

        if basis == "fourier":
            flow = train_flow(x_all, y_all, train, val, seed=args.seed, epochs=args.epochs)
            flow_samp = flow_sample(flow, torch.tensor(x_all[test]), args.n_comp, args.n_samples)
            flow_field = to_grid((flow_samp.mean(axis=1) * y_std + y_mean) @ vec + mean, basis)
            flow = recovery(flow_field) | {"resid_rel_l2": resid_rel(flow_field),
                                           "coeff_coverage68": coverage(flow_samp, std_coeff)}
            results["fourier_flow"] = flow
            summary_rows.append(("fourier flow", target.shape[1], evr, flow))
            print(f"  flow:   cellMAE {flow['cell_mass_mae']:.3e} relL2 {flow['rel_l2_density']:.3f} "
                  f"residRel {flow['resid_rel_l2']:.3f} RMSz-MAE {flow['rms_z_mae']:.3f} m2-MAE {flow['m2_amp_mae']:.4f} cov68 {flow['coeff_coverage68']:.3f}")

    # ---- headline table ----
    hdr = f"\n{'method':>14} {'n_tgt':>6} {'cellMAE':>10} {'relL2':>7} {'residRel':>9} {'RMSz_MAE':>9} {'m2_MAE':>8} {'cov68':>6}"
    print(hdr)
    print(f"{'geom baseline':>14} {'-':>6} {base_cellmae:>10.3e} {base_rel:>7.3f} {'-':>9} "
          f"{results['baseline']['rms_z_mae']:>9.3f} {results['baseline']['m2_amp_mae']:>8.4f} {'-':>6}")
    for name, n_t, _evr, mt in summary_rows:
        print(f"{name:>14} {n_t:>6} {mt['cell_mass_mae']:>10.3e} {mt['rel_l2_density']:>7.3f} "
              f"{mt['resid_rel_l2']:>9.3f} {mt['rms_z_mae']:>9.3f} {mt['m2_amp_mae']:>8.4f} {mt['coeff_coverage68']:>6.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_comp": args.n_comp, "capture": args.capture, "n_fourier_coeff": n_fourier_coeff,
        "fourier_allocation": {str(m): [int(a), int(b)] for m, (a, b) in alloc.items()},
        "results": results,
    }
    (args.output_dir / "fourier_rz_deprojection_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.output_dir / 'fourier_rz_deprojection_metrics.json'}")

    # ---- figure: cell-mass MAE, density rel-L2, vertical RMS-z error ----
    mdn_metrics = [
        ("geom\nbaseline", results["baseline"], None),
        (f"grid MDN\n({grid_target.shape[1]})", results["grid_mdn"]["mdn"], results["grid_mdn"]["oracle"]),
        (f"Fourier MDN\n({n_fourier_coeff})", results["fourier_mdn"]["mdn"], results["fourier_mdn"]["oracle"]),
        (f"Fourier flow\n({n_fourier_coeff})", results["fourier_flow"], None),
    ]
    labels = [m[0] for m in mdn_metrics]
    colors = ["#999999", "#4c72b0", "#c44e52", "#dd8452"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for axis, key, title, ylab in (
        (ax[0], "cell_mass_mae", "Held-out 3D recovery", "3D cell-mass MAE [Msun]"),
        (ax[1], "rel_l2_density", "Held-out density rel-L2", "3D density rel-L2 vs truth"),
        (ax[2], "rms_z_mae", "Vertical RMS-z error", "RMS-z MAE [kpc]"),
    ):
        axis.bar(labels, [m[1][key] for m in mdn_metrics], color=colors)
        axis.axhline(results["grid_mdn"]["oracle"][key], ls="--", color="#4c72b0", lw=1, label="grid oracle")
        axis.axhline(results["fourier_mdn"]["oracle"][key], ls="--", color="#c44e52", lw=1, label="Fourier oracle")
        axis.set_ylabel(ylab)
        axis.set_title(f"{title} (lower better)")
        axis.legend(fontsize=8)
    fig.suptitle(f"Deprojection target: Fourier x (R,z) vs PCA-on-grid (TNG fine-z test, PCA-{args.n_comp})", fontsize=13)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "figures" / "fourier_rz_deprojection_compare.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
