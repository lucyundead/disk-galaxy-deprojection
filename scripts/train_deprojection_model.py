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
    mdn = train_mdn(x_all, y_all, train, val, seed=args.seed, epochs=args.epochs)

    mlp = dict(W1=mdn.net[0].weight.detach().numpy(), b1=mdn.net[0].bias.detach().numpy(),
               W2=mdn.net[2].weight.detach().numpy(), b2=mdn.net[2].bias.detach().numpy(),
               Wm=mdn.means.weight.detach().numpy(), bm=mdn.means.bias.detach().numpy())
    cfg = dict(heights=heights.tolist(),
               alloc={m: [nr, len(heights)] for m, (nr, _k) in alloc.items()},
               r_min=kp["r_min"], r_max=kp["r_max"], grid=grid,
               image_feature_size=24, central_pixel_scale_kpc=0.35,
               img_mass_median=float(np.median(table["images"].astype(np.float64).sum(axis=(1, 2))[train])),
               base_mass_median=float(np.median(table["baseline_grid_mass_msun"].astype(np.float64)[train])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(str(args.out), cfg=cfg, mlp=mlp,
                pca=dict(vec=vec, mean=mean, y_mean=y_mean, y_std=y_std),
                feat=dict(mean=feat_mean, scale=feat_scale))
    print(f"wrote bundle {args.out} (target {target.shape[1]}, rows train {int(train.sum())})")


if __name__ == "__main__":
    main()
