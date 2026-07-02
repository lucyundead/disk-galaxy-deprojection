"""Train the fixed-dict q_m mixture head on the R=64 TNG milestone-2d table and export the
torch-free deprojection bundle (dgdp/models/dgdp_fixed_dict.npz).

Maintainer-only: needs the ~31 GB table + torch/scipy (the `[train]` extra). Reuses the exact
training path of the conserving/NGC scripts (features -> fixed-dict weights target -> PCA -> MDN)
so the bundled model matches the validated numbers, then extracts the MDN mean head to numpy.

Run (on the cluster, via `_milestone2d_richgrid_cluster.py train-bundle`):
    python scripts/train_deprojection_model.py --table outputs/.../density_residual_table.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dgdp import vertical_mixture as vm
from dgdp.density3d import make_cylindrical_grid_spec
from dgdp.features import make_features
from dgdp.fourier_rz import r_knots
from dgdp.harmonics import EVEN_M, harmonics, r_resample
from dgdp.model import save_bundle

# scripts/ is on sys.path when run as `python scripts/train_deprojection_model.py`
from deproject_fourier_rz_compare import fit_pca, train_mdn  # noqa: E402
from train_density_residual_pca import standardize_with_train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--allocation", type=Path, default=Path("configs/fourier_rz_mixture.json"))
    ap.add_argument("--out", type=Path, default=Path("src/dgdp/models/dgdp_fixed_dict.npz"))
    ap.add_argument("--n-comp", type=int, default=32)
    ap.add_argument("--n-components", type=int, default=5, help="MDN mixture components")
    ap.add_argument("--calib-samples", type=int, default=64,
                    help="posterior draws per val row for the rms|z| calibration check")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260630)
    args = ap.parse_args()

    payload = json.loads(args.allocation.read_text(encoding="utf-8"))
    alloc = {int(m): (int(a), int(b)) for m, (a, b) in payload["allocation"].items()}
    kp = payload["knot_params"]
    heights = np.asarray(payload["heights"], dtype=float)
    rk_by_m = {m: r_knots(nr, kp["r_max"], kp["r_min"]) for m, (nr, _k) in alloc.items()}

    table = np.load(args.table)
    split = table["split"].astype(str)
    train, val = split == "train", split == "val"
    z_edges, r_edges, phi_edges = table["z_edges_kpc"], table["r_edges_kpc"], table["phi_edges_rad"]
    dz = float(np.diff(z_edges)[0])
    r_grid = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_grid = 0.5 * (z_edges[:-1] + z_edges[1:])
    truth = table["truth_density"].astype(np.float32)
    a_true, _ = harmonics(truth, dz)

    # grid params for the bundle; assert they reproduce the table edges (same spec fn as the build)
    grid = dict(r_min=0.05, r_max=30.0, n_r=len(r_edges) - 1, n_phi=len(phi_edges) - 1,
                z_max=float(z_edges[-1]), n_z=len(z_edges) - 1)
    chk = make_cylindrical_grid_spec(r_min_kpc=grid["r_min"], r_max_kpc=grid["r_max"],
                                     n_r=grid["n_r"], n_phi=grid["n_phi"],
                                     z_max_kpc=grid["z_max"], n_z=grid["n_z"])
    assert np.allclose(chk.r_edges_kpc, r_edges), "grid r-edges do not reproduce the table"
    assert np.allclose(chk.z_edges_kpc, z_edges), "grid z-edges do not reproduce the table"

    feat = make_features(table["images"].astype(np.float32), table["metadata"].astype(np.float32),
                         baseline_grid_mass_msun=table["baseline_grid_mass_msun"].astype(np.float32),
                         image_feature_size=24, central_pixel_scale_kpc=0.35)
    x_all, feat_mean, feat_scale = standardize_with_train(feat, train)

    target_parts = []
    for m in EVEN_M:
        a_rk = r_resample(a_true[m], r_grid, rk_by_m[m])
        w = vm.weights_target(a_rk, z_grid, heights, signed=(m != 0))
        target_parts.append(w.real.reshape(w.shape[0], -1))
        if m != 0:
            target_parts.append(w.imag.reshape(w.shape[0], -1))
    target = np.concatenate(target_parts, axis=1).astype(np.float32)
    coeff, vec, mean = fit_pca(target, train, args.n_comp)
    y_all, y_mean, y_std = standardize_with_train(coeff, train)

    import torch

    xt_val = torch.tensor(x_all[val])
    yt_val = torch.tensor(y_all[val])
    mdn1 = train_mdn(x_all, y_all, train, val, seed=args.seed, epochs=args.epochs)
    with torch.no_grad():
        nll1 = float(mdn1.negative_log_likelihood(xt_val, yt_val))
    mdn = mdn1
    if args.n_components > 1:
        mdn = train_mdn(x_all, y_all, train, val, seed=args.seed, epochs=args.epochs,
                        n_components=args.n_components)
    with torch.no_grad():
        nllk = float(mdn.negative_log_likelihood(xt_val, yt_val))
    print(f"val NLL: K=1 {nll1:.3f}  K={args.n_components} {nllk:.3f}")

    mlp = dict(W1=mdn.net[0].weight.detach().numpy(), b1=mdn.net[0].bias.detach().numpy(),
               W2=mdn.net[2].weight.detach().numpy(), b2=mdn.net[2].bias.detach().numpy(),
               Wm=mdn.means.weight.detach().numpy(), bm=mdn.means.bias.detach().numpy(),
               Wl=mdn.logits.weight.detach().numpy(), bl=mdn.logits.bias.detach().numpy(),
               Ws=mdn.log_scales.weight.detach().numpy(), bs=mdn.log_scales.bias.detach().numpy())
    cfg = dict(heights=heights.tolist(),
               alloc={m: [nr, len(heights)] for m, (nr, _k) in alloc.items()},
               r_min=kp["r_min"], r_max=kp["r_max"], grid=grid,
               n_components=int(mdn.n_components),
               image_feature_size=24, central_pixel_scale_kpc=0.35,
               img_mass_median=float(np.median(table["images"].astype(np.float64).sum(axis=(1, 2))[train])),
               base_mass_median=float(np.median(table["baseline_grid_mass_msun"].astype(np.float64)[train])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(str(args.out), cfg=cfg, mlp=mlp,
                pca=dict(vec=vec, mean=mean, y_mean=y_mean, y_std=y_std),
                feat=dict(mean=feat_mean, scale=feat_scale))
    print(f"wrote bundle {args.out} (target {target.shape[1]}, rows train {int(train.sum())}, "
          f"K={int(mdn.n_components)})")

    calib = calibrate_rms_z(mdn, x_all[val], truth[val], vec, mean, y_mean, y_std,
                            rk_by_m[0], heights, r_grid, z_grid,
                            n_samples=args.calib_samples, seed=args.seed)
    calib["val_nll_k1"], calib[f"val_nll_k{args.n_components}"] = nll1, nllk
    calib_path = args.out.with_suffix("").with_suffix(".calibration.json")
    calib_path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    print(f"wrote {calib_path}: cov68={calib['coverage68_aggregate']:.3f} "
          f"cov90={calib['coverage90_aggregate']:.3f}")


def calibrate_rms_z(mdn, x_val, truth_val, vec, mean, y_mean, y_std, rk0, heights,
                    r_grid, z_grid, *, n_samples, seed, chunk=256):
    """Coverage of the TRUE RMS|z| within the posterior bands on the val split.

    Uses the m=0 mixture block only: after the conserving clip the phi-mean of the
    reconstruction equals a_0 = Sigma_0 q_0 exactly, and per-R RMS|z| depends on q_0 alone
    -- so knot-level rms from the sampled m=0 weights is the exact phi-averaged statistic.
    The aggregate is truth-ring-mass-weighted over knots with 0.3 < rk < 12 kpc.
    """
    import torch

    n0, kh = len(rk0), len(heights)
    m_rz = truth_val.sum(axis=2)                                        # (rows, nR, nz)
    tot = np.maximum(m_rz.sum(axis=2), 1e-30)
    rms_true = np.sqrt((m_rz * np.asarray(z_grid)[None, None, :] ** 2).sum(axis=2) / tot)
    ring_mass = tot                                                     # (rows, nR)
    sel_k = (rk0 > 0.3) & (rk0 < 12.0)
    rms_true_k = np.stack([np.interp(rk0, r_grid, row) for row in rms_true])
    w_k = np.stack([np.interp(rk0, r_grid, row) for row in ring_mass])
    agg_true = (rms_true_k[:, sel_k] * w_k[:, sel_k]).sum(1) / w_k[:, sel_k].sum(1)

    torch.manual_seed(seed + 1)
    hit68_k, hit90_k, hit68_a, hit90_a = [], [], [], []
    for lo in range(0, len(x_val), chunk):
        xs = torch.tensor(x_val[lo:lo + chunk])
        draws = mdn.sample(xs, n_samples).numpy()                       # (rows, S, n_pca)
        wvec = (draws * y_std + y_mean) @ vec + mean                    # (rows, S, target)
        w0 = np.clip(wvec[..., : n0 * kh].reshape(-1, n0, kh), 0.0, None)
        q = vm.reconstruct(w0.astype(np.complex128), z_grid, heights).real
        rms_s = np.sqrt((q * np.asarray(z_grid)[None, None, :] ** 2).sum(2)
                        / np.maximum(q.sum(2), 1e-30))                  # (rows*S, n0)
        rms_s = rms_s.reshape(len(xs), n_samples, n0)
        ww = w_k[lo:lo + chunk][:, None, sel_k]
        agg_s = (rms_s[:, :, sel_k] * ww).sum(2) / ww.sum(2)            # (rows, S)
        qtl = (rms_s < rms_true_k[lo:lo + chunk][:, None, :]).mean(axis=1)   # (rows, n0)
        hit68_k.append((qtl >= 0.16) & (qtl <= 0.84))
        hit90_k.append((qtl >= 0.05) & (qtl <= 0.95))
        qa = (agg_s < agg_true[lo:lo + chunk][:, None]).mean(axis=1)
        hit68_a.append((qa >= 0.16) & (qa <= 0.84))
        hit90_a.append((qa >= 0.05) & (qa <= 0.95))
    hit68_k, hit90_k = np.concatenate(hit68_k), np.concatenate(hit90_k)
    return dict(
        n_samples=int(n_samples), n_val_rows=int(len(x_val)),
        knots_kpc=np.asarray(rk0).tolist(),
        coverage68_per_knot=hit68_k.mean(axis=0).tolist(),
        coverage90_per_knot=hit90_k.mean(axis=0).tolist(),
        coverage68_aggregate=float(np.concatenate(hit68_a).mean()),
        coverage90_aggregate=float(np.concatenate(hit90_a).mean()),
    )


if __name__ == "__main__":
    main()
