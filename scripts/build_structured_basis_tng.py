"""Hybrid PCA-per-m structured basis for the 3D residual (foundation, Phase 1).

Target = density residual (truth - baseline), normalised by truth grid mass, on
the fine-z (0.3125 kpc) TNG grid. The structured basis is:
  1. azimuthal Fourier, keep even m in {0,2,4,6,8,10} (enforces bar symmetry,
     drops odd/high-m local noise);
  2. PCA per harmonic on the (R,z) amplitude maps a_m(R,z) (fit on TRAIN rows).
This reports how many components each harmonic needs (the flow-matching target
dimension) and the held-out reconstruction error, compared to the current global
flattened PCA at matched dimension.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVEN_M = (0, 2, 4, 6, 8, 10)


def rel_l2(recon, true, mask):
    num = np.sqrt(np.sum((recon[mask] - true[mask]) ** 2))
    den = np.sqrt(np.sum(true[mask] ** 2))
    return float(num / den)


def pca_fit(x_train):
    mean = x_train.mean(axis=0)
    u, s, vt = np.linalg.svd(x_train - mean, full_matrices=False)
    return mean, vt, s


def cum_var(s):
    v = s**2
    return np.cumsum(v) / v.sum()


def k_for(s, frac):
    return int(np.searchsorted(cum_var(s), frac) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    ap.add_argument("--var-frac", type=float, default=0.95)
    args = ap.parse_args()

    t = np.load(args.table)
    re, pe, ze = t["r_edges_kpc"], t["phi_edges_rad"], t["z_edges_kpc"]
    split = t["split"].astype(str)
    train = split == "train"
    test = split == "test"
    vol = (
        0.5 * (re[1:] ** 2 - re[:-1] ** 2)[:, None, None]
        * np.diff(pe)[None, :, None]
        * np.diff(ze)[None, None, :]
    ).astype(np.float32)
    truth_mass = t["truth_density"].astype(np.float32) * vol[None]
    resid = (truth_mass - t["baseline_density"].astype(np.float32) * vol[None])
    truth_total = truth_mass.sum(axis=(1, 2, 3), keepdims=True)
    resid = (resid / truth_total).astype(np.float32)   # fractional residual
    del truth_mass
    n, nR, nphi, nz = resid.shape
    print(f"rows {n} (train {train.sum()}, test {test.sum()}); grid {nR}x{nphi}x{nz}")

    # ---- global flattened PCA (the current approach) ----
    flat = resid.reshape(n, -1)
    g_mean, g_vt, g_s = pca_fit(flat[train])
    print(f"\nglobal PCA: comps for {args.var_frac:.0%} var = {k_for(g_s, args.var_frac)} "
          f"(of {g_vt.shape[0]})")

    # ---- structured hybrid PCA-per-m ----
    coeff = np.fft.rfft(resid, axis=2)   # (n, R, m, z) complex
    del resid, flat
    per_m = {}
    total_k = 0
    print("\nper-harmonic PCA (even m<=10):")
    for m in EVEN_M:
        cm = coeff[:, :, m, :]                       # (n, R, z) complex
        feat = np.concatenate([cm.real.reshape(n, -1), cm.imag.reshape(n, -1)], axis=1).astype(np.float32)
        mean, vt, s = pca_fit(feat[train])
        k = k_for(s, args.var_frac)
        total_k += k
        per_m[m] = (mean, vt, s, k)
        print(f"  m={m:2d}: power={np.mean(np.abs(cm) ** 2):.3e}  k({args.var_frac:.0%})={k:3d}  "
              f"k(99%)={k_for(s, 0.99):3d}")
    print(f"  -> structured basis dimension ({args.var_frac:.0%}) = {total_k}")

    # ---- held-out reconstruction error vs global PCA at matched dim ----
    truth_resid = np.fft.irfft(coeff, n=nphi, axis=2).astype(np.float32)  # = original resid

    def structured_recon(k_override=None):
        out = np.zeros_like(coeff)
        for m in EVEN_M:
            mean, vt, s, k = per_m[m]
            kk = k if k_override is None else k_override.get(m, k)
            cm = coeff[:, :, m, :]
            feat = np.concatenate([cm.real.reshape(n, -1), cm.imag.reshape(n, -1)], axis=1).astype(np.float32)
            z = (feat - mean) @ vt[:kk].T
            rec = z @ vt[:kk] + mean
            half = rec.shape[1] // 2
            out[:, :, m, :] = (rec[:, :half] + 1j * rec[:, half:]).reshape(n, nR, nz)
        return np.fft.irfft(out, n=nphi, axis=2).astype(np.float32)

    struct = structured_recon()
    err_struct = rel_l2(struct, truth_resid, test)

    g_flat = truth_resid.reshape(n, -1)
    z = (g_flat[test] - g_mean) @ g_vt[:total_k].T
    g_rec = (z @ g_vt[:total_k] + g_mean)
    err_global_matched = rel_l2(g_rec, g_flat[test], np.ones(test.sum(), bool))
    z32 = (g_flat[test] - g_mean) @ g_vt[:32].T
    g_rec32 = (z32 @ g_vt[:32] + g_mean)
    err_global32 = rel_l2(g_rec32, g_flat[test], np.ones(test.sum(), bool))

    # constructive option: Fourier-filter (even m<=10) for symmetry+smoothness,
    # THEN global PCA for compactness. Error is vs the filtered (smooth) target.
    coeff_f = np.zeros_like(coeff)
    for m in EVEN_M:
        coeff_f[:, :, m, :] = coeff[:, :, m, :]
    resid_f = np.fft.irfft(coeff_f, n=nphi, axis=2).astype(np.float32).reshape(n, -1)
    f_mean, f_vt, f_s = pca_fit(resid_f[train])
    zf = (resid_f[test] - f_mean) @ f_vt[:32].T
    err_filt32 = rel_l2(zf @ f_vt[:32] + f_mean, resid_f[test], np.ones(test.sum(), bool))
    filt_dropped = rel_l2(resid_f, truth_resid.reshape(n, -1), test)   # what the filter discards
    print(f"\nFourier filter (even m<=10) discards {filt_dropped:.1%} of the raw residual "
          f"(odd/high-m local noise); filtered PCA-32 var captured = {k_for(f_s, 0.95)} comps for 95%")

    print("\nheld-out (test) relative-L2 reconstruction error of the residual field:")
    print(f"  structured PCA-per-m   (dim {total_k}): {err_struct:.4f}")
    print(f"  global flattened PCA   (dim {total_k}): {err_global_matched:.4f}")
    print(f"  global flattened PCA   (dim 32, current): {err_global32:.4f}")
    print(f"  filtered+global PCA    (dim 32, vs smooth target): {err_filt32:.4f}")

    np.savez_compressed(
        args.output_dir / "structured_basis_tng.npz",
        **{f"mean_m{m}": per_m[m][0] for m in EVEN_M},
        **{f"vt_m{m}": per_m[m][1][: per_m[m][3]] for m in EVEN_M},
        k_per_m=np.array([per_m[m][3] for m in EVEN_M]),
        even_m=np.array(EVEN_M),
        r_edges_kpc=re, phi_edges_rad=pe, z_edges_kpc=ze,
        var_frac=np.float32(args.var_frac), total_dim=np.int32(total_k),
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for m in EVEN_M:
        ax[0].plot(np.arange(1, len(per_m[m][2]) + 1), cum_var(per_m[m][2]), label=f"m={m}")
    ax[0].axhline(args.var_frac, color="k", ls=":", lw=1)
    ax[0].set_xlim(0, 60)
    ax[0].set_xlabel("PCA components")
    ax[0].set_ylabel("cumulative variance")
    ax[0].set_title("per-harmonic PCA spectrum")
    ax[0].legend(fontsize=8)
    bars = ["structured\n(per-m)", f"global\n(dim {total_k})", "global\n(dim 32)"]
    ax[1].bar(bars, [err_struct, err_global_matched, err_global32], color=["#b3403c", "0.6", "0.4"])
    ax[1].set_ylabel("test residual reconstruction rel-L2")
    ax[1].set_title(f"reconstruction (structured dim = {total_k})")
    fig.suptitle("Hybrid PCA-per-m structured residual basis (TNG fine-z)", fontsize=13)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "figures" / "structured_basis_tng.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.output_dir / 'structured_basis_tng.npz'} and figure")


if __name__ == "__main__":
    main()
