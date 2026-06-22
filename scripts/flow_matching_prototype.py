"""Flow-matching prototype vs the 1-Gaussian MDN, on the TNG fine-z residual.

Head-to-head with everything held equal except the generator:
  - target: filter (even m<=10) + global PCA-K coefficients of the residual;
  - features: the standard 586-dim image+geometry features;
  - train/val/test: the galaxy-level milestone-2d split.
Trains (a) the adopted SummaryResidualMDN (1 Gaussian) and (b) a conditional
flow-matching velocity net, then compares held-out posterior-mean accuracy,
coefficient coverage (calibration), and residual reconstruction. Answers whether
a flexible (possibly non-Gaussian) posterior beats the Gaussian on this target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.models.mdn import SummaryResidualMDN
from train_density_residual_pca import make_density_residual_features, standardize_with_train

EVEN_M = (0, 2, 4, 6, 8, 10)


def filter_even(field):
    coeff = np.fft.rfft(field, axis=2)
    out = np.zeros_like(coeff)
    for m in EVEN_M:
        out[:, :, m, :] = coeff[:, :, m, :]
    return np.fft.irfft(out, n=field.shape[2], axis=2).astype(np.float32)


class FlowVel(nn.Module):
    def __init__(self, dim, feat_dim, hidden=256):
        super().__init__()
        self.fe = nn.Sequential(nn.Linear(feat_dim, hidden), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(dim + hidden + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t, feat):
        return self.net(torch.cat([x, self.fe(feat), t], dim=-1))


def batches(n, bs, rng):
    idx = rng.permutation(n)
    return [idx[i:i + bs] for i in range(0, n, bs)]


@torch.no_grad()
def flow_sample(net, feat, dim, n_samples, n_steps=100):
    net.eval()
    out = []
    for _ in range(n_samples):
        x = torch.randn(feat.shape[0], dim)
        for i in range(n_steps):
            t = torch.full((feat.shape[0], 1), i / n_steps)
            x = x + net(x, t, feat) * (1.0 / n_steps)
        out.append(x.numpy())
    return np.stack(out, axis=1)  # (rows, n_samples, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    ap.add_argument("--n-comp", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260614)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t = np.load(args.table)
    ze = t["z_edges_kpc"]
    split = t["split"].astype(str)
    train, val, test = split == "train", split == "val", split == "test"
    vol = cylindrical_bin_volumes(make_cylindrical_grid_spec(z_max_kpc=float(ze[-1]), n_z=len(ze) - 1)).astype(np.float32)

    # ---- target: filter + global PCA-K coefficients ----
    tmass = t["truth_density"].astype(np.float32) * vol[None]
    resid = ((tmass - t["baseline_density"].astype(np.float32) * vol[None]) / tmass.sum(axis=(1, 2, 3), keepdims=True)).astype(np.float32)
    del tmass
    residf = filter_even(resid).reshape(resid.shape[0], -1)
    del resid
    pmean = residf[train].mean(axis=0)
    _, _, vt = np.linalg.svd(residf[train] - pmean, full_matrices=False)
    V = vt[: args.n_comp]
    coeff = (residf - pmean) @ V.T                     # true target coefficients

    # ---- features ----
    feat = make_density_residual_features(
        t["images"].astype(np.float32), t["metadata"].astype(np.float32),
        baseline_grid_mass_msun=t["baseline_grid_mass_msun"].astype(np.float32),
        image_feature_size=24, central_pixel_scale_kpc=0.35,
    )
    x_all, x_mean, x_std = standardize_with_train(feat, train)
    y_all, y_mean, y_std = standardize_with_train(coeff, train)
    xt = torch.tensor(x_all, dtype=torch.float32)
    yt = torch.tensor(y_all, dtype=torch.float32)
    n_feat, K = x_all.shape[1], args.n_comp
    print(f"feat {n_feat}, K {K}; train {train.sum()} val {val.sum()} test {test.sum()}")

    tr_idx, va_idx = np.flatnonzero(train), np.flatnonzero(val)

    # ---- train MDN (1 Gaussian) ----
    mdn = SummaryResidualMDN(input_dim=n_feat, output_dim=K, hidden_dim=128, n_components=1)
    opt = torch.optim.Adam(mdn.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, wait = 1e9, None, 0
    for ep in range(args.epochs):
        mdn.train()
        for b in batches(len(tr_idx), 64, rng):
            opt.zero_grad()
            loss = mdn.negative_log_likelihood(xt[tr_idx[b]], yt[tr_idx[b]])
            loss.backward()
            opt.step()
        mdn.eval()
        with torch.no_grad():
            vl = float(mdn.negative_log_likelihood(xt[va_idx], yt[va_idx]))
        if vl < best - 1e-4:
            best, best_state, wait = vl, {k: v.clone() for k, v in mdn.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= 60:
            break
    mdn.load_state_dict(best_state)
    print(f"MDN best val NLL {best:.3f} (epoch {ep - wait + 1})")

    # ---- train flow (conditional flow matching) ----
    flow = FlowVel(K, n_feat, hidden=256)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(args.seed)
    v_x0 = torch.randn(len(va_idx), K, generator=g)
    v_t = torch.rand(len(va_idx), 1, generator=g)
    v_xt = (1 - v_t) * v_x0 + v_t * yt[va_idx]
    v_tgt = yt[va_idx] - v_x0
    best_f, best_fs, wait = 1e9, None, 0
    for ep in range(args.epochs):
        flow.train()
        for b in batches(len(tr_idx), 128, rng):
            yb = yt[tr_idx[b]]
            x0 = torch.randn_like(yb)
            tt = torch.rand(yb.shape[0], 1)
            xtt = (1 - tt) * x0 + tt * yb
            opt.zero_grad()
            loss = ((flow(xtt, tt, xt[tr_idx[b]]) - (yb - x0)) ** 2).mean()
            loss.backward()
            opt.step()
        flow.eval()
        with torch.no_grad():
            vl = float(((flow(v_xt, v_t, xt[va_idx]) - v_tgt) ** 2).mean())
        if vl < best_f - 1e-5:
            best_f, best_fs, wait = vl, {k: v.clone() for k, v in flow.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= 80:
            break
    flow.load_state_dict(best_fs)
    print(f"flow best val CFM loss {best_f:.4f} (epoch {ep - wait + 1})")

    # ---- sample posteriors on test ----
    te_idx = np.flatnonzero(test)
    with torch.no_grad():
        mdn_s = mdn.sample(xt[te_idx], args.n_samples).numpy()        # (rows, S, K) standardized
    flow_s = flow_sample(flow, xt[te_idx], K, args.n_samples)

    def unstd(z):
        return z * y_std[None, None, :] + y_mean[None, None, :]

    mdn_c, flow_c = unstd(mdn_s), unstd(flow_s)
    true_c = coeff[te_idx]

    def metrics(samp):
        mean = samp.mean(axis=1)
        rmse = float(np.sqrt(np.mean((mean - true_c) ** 2)))
        lo, hi = np.quantile(samp, 0.16, axis=1), np.quantile(samp, 0.84, axis=1)
        cov = float(np.mean((true_c >= lo) & (true_c <= hi)))
        recon = mean @ V + pmean
        rel = float(np.linalg.norm(recon - residf[te_idx]) / np.linalg.norm(residf[te_idx]))
        return mean, rmse, cov, rel

    m_mean, m_rmse, m_cov, m_rel = metrics(mdn_c)
    f_mean, f_rmse, f_cov, f_rel = metrics(flow_c)
    norm_true = np.linalg.norm(residf[te_idx])
    oracle_rel = float(np.linalg.norm((true_c @ V + pmean) - residf[te_idx]) / norm_true)
    mean_coeff_train = np.broadcast_to(coeff[train].mean(axis=0), (len(te_idx), K))
    zero_rel = float(np.linalg.norm((mean_coeff_train @ V + pmean) - residf[te_idx]) / norm_true)

    print("\n=== held-out (test) comparison ===")
    print(f"  coeff posterior-mean RMSE:  MDN {m_rmse:.4f} | flow {f_rmse:.4f}")
    print(f"  coeff 68% coverage:         MDN {m_cov:.3f} | flow {f_cov:.3f}  (target 0.68)")
    print(f"  residual recon rel-L2:      MDN {m_rel:.4f} | flow {f_rel:.4f}  "
          f"(oracle/basis ceiling {oracle_rel:.4f}, predict-mean {zero_rel:.4f})")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    ax[0].bar(["MDN", "flow"], [m_rmse, f_rmse], color=["#4c72b0", "#b3403c"])
    ax[0].set_ylabel("coeff posterior-mean RMSE")
    ax[0].set_title("accuracy (lower better)")
    ax[1].bar(["MDN", "flow"], [m_cov, f_cov], color=["#4c72b0", "#b3403c"])
    ax[1].axhline(0.68, color="k", ls="--", label="target 0.68")
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("68% coefficient coverage")
    ax[1].set_title("calibration")
    ax[1].legend()
    # posterior shape for the 2 highest-variance coeffs of one example row
    row = 0
    for j, col in zip((0, 1), ("#1f77b4", "#2ca02c"), strict=True):
        ax[2].hist(flow_c[row, :, j], bins=24, density=True, alpha=0.5, color=col, label=f"flow c{j}")
        mu, sd = mdn_c[row, :, j].mean(), mdn_c[row, :, j].std()
        xs = np.linspace(mu - 4 * sd, mu + 4 * sd, 100)
        ax[2].plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=col, lw=2, ls="--")
        ax[2].axvline(true_c[row, j], color=col, lw=2)
    ax[2].set_title("posterior shape (hist=flow, dashed=MDN Gaussian, line=truth)")
    ax[2].set_xlabel("coefficient value")
    ax[2].legend(fontsize=8)
    fig.suptitle(f"Flow matching vs 1-Gaussian MDN (TNG fine-z, filter+global PCA-{K})", fontsize=13)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figures" / "flow_vs_mdn.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_dir / 'figures' / 'flow_vs_mdn.png'}")


if __name__ == "__main__":
    main()
