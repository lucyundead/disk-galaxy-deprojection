"""Rank galaxies by an off-plane-mass proxy to pick X/peanut candidates.

Computed from the production 0.625 kpc grid (all we have for all galaxies). The
proxy is the ratio of off-plane to midplane mass at the bar ends:
m(|z|~0.94 kpc) / m(|z|~0.31 kpc), summed along the bar over R in [0.6, 1.1]L.
The 0.625 grid cannot resolve the X bifurcation, but galaxies with genuine
buckled peanuts carry the most off-plane mass and rank highest, so this is a
sound relative ranking to choose which galaxies to re-bin finely.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_density_residual_pca import _mass_grids  # noqa: E402

TABLE = Path("outputs/tng50_milestone2c_clean3d/density_residual_table.npz")
MANIFEST = Path("outputs/tng50_milestone2c_clean3d/manifest.csv")


def main() -> None:
    t = np.load(TABLE)
    tm, _, _ = _mass_grids(t)
    r_edges = t["r_edges_kpc"]
    phi_edges = t["phi_edges_rad"]
    z_edges = t["z_edges_kpc"]
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    pc = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    along = (np.abs(pc) < np.pi / 6) | (np.abs(np.abs(pc) - np.pi) < np.pi / 6)
    i_mid = np.argmin(np.abs(np.abs(zc) - 0.3125))
    i_off = np.argmin(np.abs(np.abs(zc) - 0.9375))

    info = {}
    for row in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        sid = int(row["subhalo_id"])
        info.setdefault(
            sid,
            (
                float(row["bar_length"]),
                float(row["A2_bar"]),
                float(row["stellar_mass_msun"]),
                row["split"],
            ),
        )

    rows = []
    seen = set()
    for idx in range(tm.shape[0]):
        sid = int(t["galaxy_id"][idx])
        if sid in seen:
            continue
        seen.add(sid)
        length, a2, mstar, split = info[sid]
        r_sel = (rc >= 0.6 * length) & (rc <= 1.1 * length)
        prof = tm[idx][np.ix_(r_sel, along, np.arange(len(zc)))].sum(axis=(0, 1))
        prof = 0.5 * (prof + prof[::-1])
        # off/mid ratio using bins centered ~0.31 and ~0.94 kpc
        ratio = (prof[i_off] + prof[len(zc) - 1 - i_off]) / max(
            prof[i_mid] + prof[len(zc) - 1 - i_mid], 1e-30
        )
        rows.append((ratio, sid, length, a2, mstar, split))

    rows.sort(reverse=True)
    print(f"{'rank':>4} {'subhalo':>8} {'off/mid':>8} {'L_bar':>6} {'A2':>5} {'log Mstar':>9} {'split':>6}")
    for rank, (ratio, sid, length, a2, mstar, split) in enumerate(rows[:18], 1):
        print(f"{rank:>4} {sid:>8} {ratio:>8.3f} {length:>6.1f} {a2:>5.2f} {np.log10(mstar):>9.2f} {split:>6}")
    test_top = [sid for _, sid, *_ in rows if _find_split(rows, sid) == "test"][:8]
    print("\ntop test-split candidates:", test_top)


def _find_split(rows, sid):
    for _, s, _, _, _, split in rows:
        if s == sid:
            return split
    return None


if __name__ == "__main__":
    main()
