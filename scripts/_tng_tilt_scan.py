"""How widespread is the x-z tilt (bulge symmetry axis vs grid z) across the 185?

For each galaxy's fine-z truth grid (bar frame), measure the along-bar vertical
tilt = slope of the mass-weighted z-centroid <z>(x) over the bar region, i.e.
slope = sum(m x z) / sum(m x^2) with x = R cos(phi). A clean symmetric peanut
gives ~0; a misaligned (tilted) bulge gives a nonzero tilt. Reports the
distribution -- if many galaxies are tilted, the angular-momentum alignment is
smearing peanuts and inflating any structured-basis rank sample-wide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--bar-radius-kpc", type=float, default=5.0)
    ap.add_argument("--output", type=Path, default=Path("outputs/nbody_shen2010/figures/tng_tilt_scan.png"))
    args = ap.parse_args()

    t = np.load(args.table)
    gid = np.asarray(t["galaxy_id"])
    re, pe, ze = t["r_edges_kpc"], t["phi_edges_rad"], t["z_edges_kpc"]
    rc = 0.5 * (re[:-1] + re[1:])
    pc = 0.5 * (pe[:-1] + pe[1:])
    zc = 0.5 * (ze[:-1] + ze[1:])
    vol = (
        0.5 * (re[1:] ** 2 - re[:-1] ** 2)[:, None, None]
        * np.diff(pe)[None, :, None]
        * np.diff(ze)[None, None, :]
    )
    xg = rc[:, None, None] * np.cos(pc)[None, :, None] * np.ones_like(zc)[None, None, :]
    zg = np.ones((len(rc), len(pc)))[:, :, None] * zc[None, None, :]
    barmask = (rc < args.bar_radius_kpc)[:, None, None]

    truth = t["truth_density"]
    tilts = {}
    for g in np.unique(gid):
        row = int(np.flatnonzero(gid == g)[0])
        mass = truth[row].astype(np.float64) * vol
        w = mass * barmask
        sxx = float(np.sum(w * xg**2))
        sxz = float(np.sum(w * xg * zg))
        slope = sxz / sxx if sxx > 0 else 0.0
        tilts[int(g)] = float(np.degrees(np.arctan(slope)))

    vals = np.array(list(tilts.values()))
    av = np.abs(vals)
    print(f"x-z tilt across {len(vals)} galaxies (deg):")
    print(f"  median |tilt| = {np.median(av):.2f}; p84 = {np.percentile(av, 84):.2f}; "
          f"p95 = {np.percentile(av, 95):.2f}; max = {av.max():.2f}")
    print(f"  fraction |tilt|>3deg = {np.mean(av > 3):.2f}; >5deg = {np.mean(av > 5):.2f}; "
          f">10deg = {np.mean(av > 10):.2f}")
    if 392276 in tilts:
        print(f"  392276 tilt = {tilts[392276]:+.2f} deg")

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True)
    ax.hist(av, bins=np.linspace(0, max(20, av.max()), 30), color="#4c72b0", edgecolor="k", alpha=0.8)
    ax.axvline(np.median(av), color="k", ls="--", label=f"median {np.median(av):.1f} deg")
    if 392276 in tilts:
        ax.axvline(abs(tilts[392276]), color="#b3403c", ls="-", lw=2, label=f"392276 {abs(tilts[392276]):.1f} deg")
    ax.set_xlabel("|x-z tilt| of bar region [deg]  (bulge axis vs grid z)")
    ax.set_ylabel("galaxies")
    ax.set_title(f"TNG50 fine-z truth: along-bar vertical tilt ({len(vals)} galaxies)")
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
