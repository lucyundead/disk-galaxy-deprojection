"""Characterize the Shen2010 N-body barred-galaxy snapshot (units + structure).

Reads the raw ASCII snapshot (x y z vx vy vz), caches positions/velocities as
.npy for fast reuse, and prints the physical scale we need to decide a kpc
scaling: radial/vertical extent, an exponential disk scale length, the m=2 bar
amplitude profile (bar length + strength), and a bar-end vertical profile (the
boxy/peanut/X signature). Read-only characterization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True, help="t800info.dat ASCII path")
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    pos_path = args.cache_dir / "positions_raw.npy"
    vel_path = args.cache_dir / "velocities_raw.npy"
    if pos_path.exists() and vel_path.exists():
        positions = np.load(pos_path)
        velocities = np.load(vel_path)
        print(f"loaded cached arrays: positions {positions.shape}")
    else:
        raw = np.loadtxt(args.raw)
        print(f"raw shape {raw.shape}")
        positions = np.ascontiguousarray(raw[:, 0:3])
        velocities = np.ascontiguousarray(raw[:, 3:6]) if raw.shape[1] >= 6 else None
        np.save(pos_path, positions)
        if velocities is not None:
            np.save(vel_path, velocities)
        print(f"cached to {args.cache_dir}")

    n = positions.shape[0]
    com = positions.mean(axis=0)
    print(f"\nN particles = {n}")
    print(f"COM (raw) = {com}")
    pos = positions - com  # recenter
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    R = np.hypot(x, y)
    absz = np.abs(z)

    def pct(a, qs):
        return {q: float(np.percentile(a, q)) for q in qs}

    qs = [50, 90, 95, 99, 100]
    print("\n-- raw-unit extent --")
    print(f"x range [{x.min():.3f}, {x.max():.3f}]")
    print(f"y range [{y.min():.3f}, {y.max():.3f}]")
    print(f"z range [{z.min():.3f}, {z.max():.3f}]")
    print(f"R percentiles {pct(R, qs)}")
    print(f"|z| percentiles {pct(absz, qs)}")

    if velocities is not None:
        vmag = np.linalg.norm(velocities - velocities.mean(axis=0), axis=1)
        print(f"\nvelocity |v| percentiles {pct(vmag, [50, 90, 99, 100])}")
        # circular speed proxy: median |v_phi| in an annulus
        vphi = (x * velocities[:, 1] - y * velocities[:, 0]) / np.maximum(R, 1e-9)
        for lo, hi in [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0)]:
            m = (R >= lo) & (R < hi)
            if m.sum() > 100:
                print(f"  <|vphi|> R in [{lo},{hi}) = {np.mean(np.abs(vphi[m])):.1f} "
                      f"(n={int(m.sum())})")

    # exponential disk scale length: Sigma(R) ~ exp(-R/Rd)
    r_max_fit = float(np.percentile(R, 98))
    edges = np.linspace(0.0, r_max_fit, 41)
    rc = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(R, bins=edges)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    sigma = counts / area
    good = (sigma > 0) & (rc > 0.2 * r_max_fit) & (rc < 0.8 * r_max_fit)
    coef = np.polyfit(rc[good], np.log(sigma[good]), 1)
    rd = -1.0 / coef[0]
    print("\n-- structure (raw units) --")
    print(f"exp disk scale length Rd ~ {rd:.3f}  (fit annulus {0.2*r_max_fit:.2f}-{0.8*r_max_fit:.2f})")
    print(f"R50={np.percentile(R,50):.3f}  R90={np.percentile(R,90):.3f}")

    # m=2 bar amplitude profile
    phi = np.arctan2(y, x)
    print("\n-- m=2 amplitude A2(R) (bar length & strength) --")
    bar_edges = np.linspace(0.0, np.percentile(R, 95), 31)
    a2_prof = []
    for i in range(len(bar_edges) - 1):
        m = (R >= bar_edges[i]) & (R < bar_edges[i + 1])
        if m.sum() < 50:
            a2_prof.append(0.0)
            continue
        a2 = np.abs(np.mean(np.exp(2j * phi[m])))
        a2_prof.append(a2)
    a2_prof = np.array(a2_prof)
    bc = 0.5 * (bar_edges[:-1] + bar_edges[1:])
    peak_a2 = float(a2_prof.max())
    peak_r = float(bc[np.argmax(a2_prof)])
    # bar length ~ radius where A2 drops to half its peak (outside the peak)
    half = peak_a2 / 2.0
    bar_len = np.nan
    for i in range(int(np.argmax(a2_prof)), len(a2_prof)):
        if a2_prof[i] < half:
            bar_len = float(bc[i])
            break
    print(f"peak A2={peak_a2:.3f} at R={peak_r:.3f}; bar length (A2<half) ~ {bar_len:.3f}")
    for i in range(0, len(bc), 2):
        print(f"  R={bc[i]:5.2f}  A2={a2_prof[i]:.3f}")

    # bar-end vertical profile (boxy/peanut/X): bins along bar, near the bar end
    along = (np.abs(phi) < np.pi / 6) | (np.abs(np.abs(phi) - np.pi) < np.pi / 6)
    bar_end = along & (R > 0.5 * np.nanmax([bar_len, peak_r])) & (R < 1.2 * np.nanmax([bar_len, peak_r]))
    print(f"\n-- bar-end vertical profile (n={int(bar_end.sum())}) --")
    zext = float(np.percentile(np.abs(z[bar_end]), 98))
    zb = np.linspace(-zext, zext, 41)
    h, _ = np.histogram(z[bar_end], bins=zb)
    h = 0.5 * (h + h[::-1]).astype(float)
    zcen = 0.5 * (zb[:-1] + zb[1:])
    hn = h / h.max()
    mid = hn[np.argmin(np.abs(zcen))]
    peakz = abs(zcen[np.argmax(hn)])
    print(f"midplane/peak dip = {mid:.3f}   peak|z| = {peakz:.3f} (raw units)")
    print("  z, normalized count:")
    for i in range(0, len(zcen), 2):
        print(f"  z={zcen[i]:+5.2f}  {hn[i]:.3f}")


if __name__ == "__main__":
    main()
