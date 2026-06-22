"""External cross-check of the boxy/peanut census on real TNG galaxies.

The b4 boxiness metric is validated on synthetic ground truth, but its ranking of
real galaxies is hard to confirm by eye. Here we test it against two independent
expectations of a genuine boxy/peanut bulge:
  1. it should be vertically THICKER in the bar than outside it (B/P thickening);
  2. it should come from a STRONGER bar (B/P forms by buckling) -> catalog A2.
A positive correlation with both supports that the metric captures real B/P.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def bar_thickening_ratio(dens, r_edges, phi_edges, z_edges, bar_length):
    """RMS|z| in the bar (R<L) divided by RMS|z| in the outer disk (1.5L<R<2.5L)."""
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    mass_rz = dens.sum(axis=1)  # (R, z) after summing over phi

    def rms(mask_r):
        w = mass_rz[mask_r].sum(axis=0)
        s = w.sum()
        return np.sqrt(np.sum(w * zc**2) / s) if s > 0 else np.nan

    bar = rms(rc < bar_length)
    disk = rms((rc >= 1.5 * bar_length) & (rc < 2.5 * bar_length))
    return bar / disk if (disk and np.isfinite(disk) and disk > 0) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, default=Path("outputs/nbody_shen2010/figures/tng_peanut_census.csv"))
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--manifest", type=Path, default=Path("outputs/tng50_milestone2c_clean3d/manifest.csv"))
    ap.add_argument("--output", type=Path, default=Path("outputs/nbody_shen2010/figures/tng_peanut_crosscheck.png"))
    args = ap.parse_args()

    cen = pd.read_csv(args.census)
    man = pd.read_csv(args.manifest).drop_duplicates("subhalo_id").set_index("subhalo_id")
    t = np.load(args.table)
    gid = np.asarray(t["galaxy_id"])
    re, pe, ze = t["r_edges_kpc"], t["phi_edges_rad"], t["z_edges_kpc"]
    truth = t["truth_density"]

    rows = []
    for _, r in cen.iterrows():
        sid = int(r["subhalo_id"])
        if sid not in man.index:
            continue
        L = float(man.loc[sid, "bar_length"])
        grow = int(np.flatnonzero(gid == sid)[0])
        ratio = bar_thickening_ratio(truth[grow].astype(np.float64), re, pe, ze, L)
        rows.append({
            "strength": float(r["strength_median"]),
            "thickening": ratio,
            "A2_bar": float(man.loc[sid, "A2_bar"]),
            "logM": np.log10(float(man.loc[sid, "stellar_mass_msun"])),
        })
    df = pd.DataFrame(rows).dropna()
    print(f"cross-check on {len(df)} galaxies")

    tests = [
        ("thickening", "bar/disk RMS|z| (B/P thickening)"),
        ("A2_bar", "catalog bar strength A2"),
        ("logM", "log stellar mass (control)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (key, lab) in zip(axes, tests, strict=True):
        x = df[key].to_numpy()
        y = df["strength"].to_numpy()
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        print(f"  strength vs {key:11s}: Pearson r={pr.statistic:+.2f} (p={pr.pvalue:.1e}) | "
              f"Spearman rho={sr.statistic:+.2f} (p={sr.pvalue:.1e})")
        ax.scatter(x, y, s=12, alpha=0.6, color="#4c72b0")
        ax.set_xlabel(lab)
        ax.set_ylabel("bulge boxiness (b4)")
        ax.set_title(f"r={pr.statistic:+.2f}, rho={sr.statistic:+.2f}")
        ax.axhline(0, color="k", lw=0.6)
    fig.suptitle("Peanut metric external cross-check (vs independent B/P expectations)", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
