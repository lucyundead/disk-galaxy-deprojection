"""Test re-binned candidate grids for a genuine X/peanut signature.

For each fetched 0.3125 kpc truth grid, compute the bar-end vertical mass
profile (along the bar, R in [0.6, 1.1] L_bar) and report:
  - midplane/peak: < 1 means the profile dips at z=0 (off-plane peaks => X);
  - peak |z|: the height of the density maximum.
A real buckled peanut shows midplane/peak well below 1 with the peak off the
midplane.
"""

from __future__ import annotations

import csv
import glob
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_density_residual_pca import cylindrical_bin_volumes_from_edges  # noqa: E402

ZREBIN = "/mnt/e/dgdp-zrebin"
MANIFEST = Path("outputs/tng50_milestone2c_clean3d/manifest.csv")


def bar_end_profile(path, length):
    with h5py.File(path, "r") as f:
        density = f["density_msun_per_kpc3"][:]
        r_edges = f["r_edges_kpc"][:]
        phi_edges = f["phi_edges_rad"][:]
        z_edges = f["z_edges_kpc"][:]
    vol = cylindrical_bin_volumes_from_edges(r_edges, phi_edges, z_edges)
    mass = density * vol
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    pc = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    along = (np.abs(pc) < np.pi / 6) | (np.abs(np.abs(pc) - np.pi) < np.pi / 6)
    r_sel = (rc >= 0.6 * length) & (rc <= 1.1 * length)
    prof = mass[np.ix_(r_sel, along, np.arange(len(zc)))].sum(axis=(0, 1))
    prof = 0.5 * (prof + prof[::-1])
    mid = prof[np.argmin(np.abs(zc))]
    peak = prof.max()
    peak_z = abs(zc[np.argmax(prof)])
    return mid / peak, peak_z


def main() -> None:
    lengths = {}
    for row in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        lengths[int(row["subhalo_id"])] = float(row["bar_length"])
    results = []
    for path in sorted(glob.glob(f"{ZREBIN}/subhalo_*_density_cylindrical.hdf5")):
        m = re.search(r"subhalo_(\d+)", path)
        sid = int(m.group(1))
        if sid not in lengths:
            continue
        dip, peak_z = bar_end_profile(path, lengths[sid])
        results.append((dip, peak_z, sid, lengths[sid]))
    results.sort()
    print(f"{'subhalo':>8} {'L_bar':>6} {'midplane/peak':>14} {'peak|z|':>8}  verdict")
    for dip, peak_z, sid, length in results:
        verdict = "X / PEANUT" if dip < 0.9 and peak_z > 0.3 else ("boxy/thick" if dip < 0.98 else "single-peak")
        print(f"{sid:>8} {length:>6.1f} {dip:>14.3f} {peak_z:>8.2f}  {verdict}")


if __name__ == "__main__":
    main()
