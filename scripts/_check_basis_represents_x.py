"""Can the train-fit PCA basis represent the held-out galaxy's X at all?

Reconstructs the held-out galaxy's truth from its OWN best-fit basis
coefficients (true_coefficients) and measures the bar-end vertical dip. If the
reconstruction keeps the off-plane double-peak, the basis can represent the X
and the MDN simply failed to predict it; if the reconstruction is single-peaked,
the basis itself (fit on an X-free training set) cannot express an X.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from train_density_residual_pca import (
    _mass_grids,
    reconstruct_delta_mass_from_coefficients,
)

M2D = Path("/mnt/e/dgdp-milestone2d")
TABLE = M2D / "density_residual_table_holdout.npz"
PCA = M2D / "density_residual_pca/density_residual_pca.npz"
PRED = M2D / "mdn_z5/density_residual_pca_mdn_predictions.npz"
MANIFEST = Path("outputs/tng50_milestone2c_clean3d/manifest.csv")
SUBHALO = 392276
GRID = (32, 48, 32)


def bar_end_dip(mass, r_edges, phi_edges, z_edges, length):
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    pc = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    along = (np.abs(pc) < np.pi / 6) | (np.abs(np.abs(pc) - np.pi) < np.pi / 6)
    r_sel = (rc >= 0.6 * length) & (rc <= 1.1 * length)
    prof = mass[np.ix_(r_sel, along, np.arange(len(zc)))].sum(axis=(0, 1))
    prof = 0.5 * (prof + prof[::-1])
    prof = prof / prof.max()
    return prof[np.argmin(np.abs(zc))], abs(zc[np.argmax(prof)])


def main() -> None:
    table = np.load(TABLE)
    preds = np.load(PRED)
    pca = np.load(PCA)
    r_edges = table["r_edges_kpc"].astype(np.float64)
    phi_edges = table["phi_edges_rad"].astype(np.float64)
    z_edges = table["z_edges_kpc"].astype(np.float64)

    t_row = int(np.flatnonzero((table["galaxy_id"] == SUBHALO) & (table["projection_id"] == 0))[0])
    p_row = int(np.flatnonzero((preds["galaxy_id"] == SUBHALO) & (preds["projection_id"] == 0))[0])

    truth_mass, baseline_mass, _ = _mass_grids(table)
    scale = table["baseline_grid_mass_msun"][t_row].astype(np.float32)

    length = next(
        float(r["bar_length"]) for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))
        if int(r["subhalo_id"]) == SUBHALO
    )

    def corrected(coeffs):
        delta = reconstruct_delta_mass_from_coefficients(
            coeffs[None, :], components=pca["components"].astype(np.float32),
            mean=pca["mean"].astype(np.float32), scale_msun=np.array([scale], dtype=np.float32),
            grid_shape=GRID,
        )[0]
        return np.clip(baseline_mass[t_row] + delta, 0.0, None)

    raw_dip = bar_end_dip(truth_mass[t_row], r_edges, phi_edges, z_edges, length)
    pca_truth_dip = bar_end_dip(corrected(preds["true_coefficients"][p_row]), r_edges, phi_edges, z_edges, length)
    mdn_dip = bar_end_dip(corrected(preds["posterior_mean_coefficients"][p_row]), r_edges, phi_edges, z_edges, length)
    base_dip = bar_end_dip(baseline_mass[t_row], r_edges, phi_edges, z_edges, length)

    print(f"subhalo {SUBHALO} bar-end vertical (midplane/peak, peak|z|):")
    print(f"  raw truth                       : dip={raw_dip[0]:.3f}  peak|z|={raw_dip[1]:.2f}")
    print(f"  truth via PCA basis (best case) : dip={pca_truth_dip[0]:.3f}  peak|z|={pca_truth_dip[1]:.2f}")
    print(f"  MDN posterior mean              : dip={mdn_dip[0]:.3f}  peak|z|={mdn_dip[1]:.2f}")
    print(f"  geometric baseline              : dip={base_dip[0]:.3f}  peak|z|={base_dip[1]:.2f}")
    verdict = (
        "BASIS CAN represent X -> MDN failed to predict it"
        if pca_truth_dip[0] < 0.9
        else "BASIS CANNOT represent X (fit on X-free train) -> representational limit"
    )
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
