"""Expressiveness pre-check: can a small symmetric, non-negative mixture represent the
milestone-2d truth vertical profiles as well as the current free-knot representation?

For each galaxy x R-bin we take the truth vertical profile (m=0 azimuthal average, and the
m=2 bar harmonic), fold to |z|, normalise, and compare the fit residual of:
  * free-knot : the CURRENT representation -- linear interp through the geomspace |z| knots
                {0,0.12,0.24,0.48,0.97,1.95,4.0} kpc (m=0 n_z_half=6).
  * sech2 best-2 / best-3 : best non-negative combo of K sech^2(z/2h) kernels (z=0 centred).
  * sech2 full / gauss full : best non-negative combo over the whole height dictionary
                (the positive-mixture ceiling) + median # of components actually used.
Mixtures are fit by NNLS over a fixed dictionary of scale heights (convex -> no local
minima, unlike free-height least-squares). All candidates are symmetric & non-negative by
construction. Metrics: profile rel-L2 and RMS|z| fractional error (the physical moment).

Decision: if a 2-3 component mixture matches the free-knot residual, the constrained
parametrization is safe to bake into training. Read-only; nothing in the pipeline changes.

Run: .venv/bin/python scripts/test_mixture_vs_freeknot.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

ZD = np.geomspace(0.15, 4.5, 10)   # scale-height dictionary [kpc]


def basis(zabs, kind):
    if kind == "sech2":
        return 1.0 / np.cosh(zabs[:, None] / (2 * ZD[None, :])) ** 2
    return np.exp(-(zabs[:, None] ** 2) / (2 * ZD[None, :] ** 2))


def freeknot_recon(p, zabs):
    knots = np.concatenate([[0.0], np.geomspace(0.12, 4.0, 6)])
    return np.interp(zabs, knots, np.interp(knots, zabs, p))


def _nnls(B, p, lam=1e-6):
    """Ridge-stabilised NNLS (the dictionaries are collinear -> plain NNLS can cycle)."""
    n = B.shape[1]
    Ba = np.vstack([B, np.sqrt(lam) * np.eye(n)])
    pa = np.concatenate([p, np.zeros(n)])
    return nnls(Ba, pa, maxiter=2000)[0]


def nnls_full(B, p):
    w = _nnls(B, p)
    frac = w * ZD / max((w * ZD).sum(), 1e-30)   # mass fraction per component (int sech2 ~ h)
    return B @ w, int((frac > 0.02).sum())


def nnls_bestk(B, p, k):
    best_r, best_fit = np.inf, None
    for cols in combinations(range(B.shape[1]), k):
        sub = B[:, cols]
        fit = sub @ _nnls(sub, p)
        r = np.linalg.norm(fit - p)
        if r < best_r:
            best_r, best_fit = r, fit
    return best_fit


def metrics(fit, p, zabs):
    rel = np.linalg.norm(fit - p) / max(np.linalg.norm(p), 1e-30)
    rms_t = np.sqrt((p * zabs**2).sum() / max(p.sum(), 1e-30))
    rms_f = np.sqrt((fit * zabs**2).sum() / max(fit.sum(), 1e-30))
    return rel, abs(rms_f - rms_t) / max(rms_t, 1e-30)


def main():
    z = np.load("/mnt/e/dgdp-milestone2d/density_residual_table.npz")
    td, gid = z["truth_density"], z["galaxy_id"]
    zc = 0.5 * (z["z_edges_kpc"][:-1] + z["z_edges_kpc"][1:])
    phi = 0.5 * (z["phi_edges_rad"][:-1] + z["phi_edges_rad"][1:])
    npos = zc.size // 2
    zabs = zc[npos:]
    Bs, Bg = basis(zabs, "sech2"), basis(zabs, "gauss")
    _, first = np.unique(gid, return_index=True)
    gals = np.sort(first)[::4]   # ~46-galaxy subsample (robust statistics, fast)

    models = ["free-knot", "sech2 best-2", "sech2 best-3", "sech2 full", "gauss full"]
    res = {m: {"rel": [], "rms": []} for m in models}
    res2 = {m: {"rel": [], "rms": []} for m in ("free-knot", "sech2 best-3")}
    neff, examples = [], []
    for gi in gals:
        g = td[gi].astype(np.float64)
        rho0 = g.mean(axis=1)
        a2 = np.abs((g * np.exp(-2j * phi)[None, :, None]).mean(axis=1))
        col = rho0.sum(axis=1)
        for ri in np.where(col > 0.01 * col.max())[0]:
            p = rho0[ri, npos:] + rho0[ri, npos - 1::-1]
            p = p / max(p.max(), 1e-30)
            full_fit, ne = nnls_full(Bs, p)
            neff.append(ne)
            fits = {"free-knot": freeknot_recon(p, zabs),
                    "sech2 best-2": nnls_bestk(Bs, p, 2),
                    "sech2 best-3": nnls_bestk(Bs, p, 3),
                    "sech2 full": full_fit,
                    "gauss full": nnls_full(Bg, p)[0]}
            for m in models:
                rl, rm = metrics(fits[m], p, zabs)
                res[m]["rel"].append(rl)
                res[m]["rms"].append(rm)
            p2 = a2[ri, npos:] + a2[ri, npos - 1::-1]
            if p2.max() > 0:
                p2 = p2 / p2.max()
                for m, f in (("free-knot", freeknot_recon(p2, zabs)),
                             ("sech2 best-3", nnls_bestk(Bs, p2, 3))):
                    rl, rm = metrics(f, p2, zabs)
                    res2[m]["rel"].append(rl)
                    res2[m]["rms"].append(rm)
            if len(examples) < 4 and ri in (12, 20):
                examples.append((zabs, p, fits))

    n = len(res["free-knot"]["rel"])
    print(f"n profiles (m=0): {n};  median # mixture components used (sech2 full): {int(np.median(neff))}\n")
    print(f"{'model':<14}{'relL2 med':>10}{'relL2 p90':>10}{'RMSz med':>10}{'RMSz p90':>10}")
    for m in models:
        rel, rms = np.array(res[m]["rel"]), np.array(res[m]["rms"])
        print(f"{m:<14}{np.median(rel):>10.3f}{np.percentile(rel, 90):>10.3f}"
              f"{np.median(rms):>10.3f}{np.percentile(rms, 90):>10.3f}")
    print(f"\nm=2 bar harmonic (n={len(res2['sech2 best-3']['rel'])}):")
    for m in ("free-knot", "sech2 best-3"):
        rel, rms = np.array(res2[m]["rel"]), np.array(res2[m]["rms"])
        print(f"{m:<14}{np.median(rel):>10.3f}{np.percentile(rel, 90):>10.3f}"
              f"{np.median(rms):>10.3f}{np.percentile(rms, 90):>10.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cols = {"free-knot": "k", "sech2 best-2": "C0", "sech2 best-3": "C3",
            "sech2 full": "C1", "gauss full": "C2"}
    if examples:
        za, p, fits = examples[-1]
        ax[0].plot(za, p, "ko", ms=4, label="truth")
        for m in ("free-knot", "sech2 best-3", "gauss full"):
            ax[0].plot(za, fits[m], color=cols[m], lw=1.5, label=m)
        ax[0].set(xlabel="|z| [kpc]", ylabel="norm. density", yscale="log", ylim=(1e-3, 1.5),
                  title="example vertical profile fit")
        ax[0].legend(fontsize=7)
    for m in models:
        ax[1].plot(np.sort(res[m]["rel"]), np.linspace(0, 1, len(res[m]["rel"])), color=cols[m], lw=2, label=m)
        ax[2].plot(np.sort(res[m]["rms"]), np.linspace(0, 1, len(res[m]["rms"])), color=cols[m], lw=2, label=m)
    ax[1].set(xlabel="profile rel-L2", ylabel="CDF", xlim=(0, 0.3), title="m=0 profile residual")
    ax[2].set(xlabel="RMS|z| fractional error", ylabel="CDF", xlim=(0, 0.2), title="m=0 RMS|z| error")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].grid(alpha=0.3)
    out = Path("outputs/tng50_vertical_scale")
    fig.tight_layout()
    fig.savefig(out / "mixture_vs_freeknot.png", dpi=130)
    print(f"\nwrote {out}/mixture_vs_freeknot.png")


if __name__ == "__main__":
    main()
