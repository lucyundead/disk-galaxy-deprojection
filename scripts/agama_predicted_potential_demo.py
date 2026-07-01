"""Route-B(2) demo: a predicted 3D density -> native AGAMA potential -> dynamics.

End to end on one held-out galaxy (TNG 554189, test split), with NO retraining:

  observed image  ->  predicted 3D density  ->  agama.Potential  ->  v_c(R) + orbits

The predicted density is the ADOPTED grid PCA-MDN posterior mean from the saved
milestone-2d predictions (``mdn_z5/density_residual_pca_mdn_predictions.npz`` -- a
genuine image->density prediction, loaded, not retrained). It is wrapped two ways
with :mod:`dgdp.agama_density`:

  - directly as an ``agama.DensityAzimuthalHarmonic`` (the raw predicted grid), and
  - as the smooth, potential-ready even-m Fourier x (R,z) form of the same prediction
    (``fit_fourier_rz_from_grid`` -> ``fourier_rz_to_agama_density``),

both preserving the predicted total mass EXACTLY through the wrap (mass conservation
intact). The CylSpline potential then gives the rotation curve and orbits, compared
to the truth-density potential (same grid/frame).

Run:
    PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 \\
        .venv/bin/python scripts/agama_predicted_potential_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

try:
    import agama
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama

from dgdp.agama_density import (
    circular_velocity,
    fourier_rz_to_agama_density,
    to_agama_density,
    to_agama_potential,
)
from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.fourier_rz import fit_fourier_rz_from_grid

agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s


def grid_lookup(density, r_edges, phi_edges, z_edges):
    """Nearest-cell density callable for a precomputed (nR, nphi, nz) grid."""
    def f(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        rr, pp, zz = np.hypot(x[:, 0], x[:, 1]), np.arctan2(x[:, 1], x[:, 0]), x[:, 2]
        ir = np.clip(np.searchsorted(r_edges, rr) - 1, 0, len(r_edges) - 2)
        ip = np.clip(np.searchsorted(phi_edges, pp) - 1, 0, len(phi_edges) - 2)
        iz = np.searchsorted(z_edges, zz) - 1
        valid = (rr < r_edges[-1]) & (iz >= 0) & (iz < len(z_edges) - 1)
        return np.where(valid, density[ir, ip, np.clip(iz, 0, len(z_edges) - 2)], 0.0)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=Path("/mnt/e/dgdp-milestone2d/density_residual_table.npz"))
    ap.add_argument("--predictions", type=Path,
                    default=Path("/mnt/e/dgdp-milestone2d/mdn_z5/density_residual_pca_mdn_predictions.npz"))
    ap.add_argument("--galaxy-id", type=int, default=554189)
    ap.add_argument("--inclination", type=float, default=40.0)
    ap.add_argument("--bar-angle", type=float, default=0.0)
    ap.add_argument("--r-max-rep", type=float, default=15.0, help="Fourier/AZH representation r_max")
    ap.add_argument("--z-max-rep", type=float, default=4.0)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    args = ap.parse_args()

    table = np.load(args.table)
    pred = np.load(args.predictions)
    gid, meta = table["galaxy_id"], table["metadata"]
    r_edges, phi_edges, z_edges = table["r_edges_kpc"], table["phi_edges_rad"], table["z_edges_kpc"]
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    vol = cylindrical_bin_volumes(
        make_cylindrical_grid_spec(z_max_kpc=float(z_edges[-1]), n_z=len(z_edges) - 1)).astype(np.float64)

    sel = np.flatnonzero((gid == args.galaxy_id) & (meta[:, 0] == args.inclination) & (meta[:, 2] == args.bar_angle))
    if sel.size == 0:
        raise SystemExit(f"no row for galaxy {args.galaxy_id} incl {args.inclination} bar {args.bar_angle}")
    row = int(sel[0])
    split = pred["split"].astype(str)[row] if "split" in pred.files else table["split"].astype(str)[row]
    print(f"row {row}: galaxy {args.galaxy_id}  incl {args.inclination}  bar {args.bar_angle}  split={split}")

    pred_mass = pred["posterior_mean_mass"][row].astype(np.float64)
    truth_mass = (table["truth_density"][row].astype(np.float64) * vol)
    pred_density = pred_mass / vol
    truth_density = table["truth_density"][row].astype(np.float64)
    image = table["images"][row].astype(np.float64)
    image = image.sum(axis=0) if image.ndim == 3 else image

    # mass within the smooth representation domain (R<r_max, |z|<z_max)
    in_dom = (r_centers[:, None] < args.r_max_rep) & (np.abs(z_centers)[None, :] < args.z_max_rep)
    m_pred_full = float(pred_mass.sum())
    m_pred_dom = float((pred_mass.sum(axis=1) * in_dom).sum())
    m_truth_full = float(truth_mass.sum())
    print(f"M_pred(full grid) {m_pred_full:.4e}  M_pred(R<{args.r_max_rep:.0f},|z|<{args.z_max_rep:.0f}) "
          f"{m_pred_dom:.4e} ({m_pred_dom / m_pred_full:.1%})  M_truth {m_truth_full:.4e}")

    # ---- wrap the predicted density as native AGAMA densities (mass preserved exactly) ----
    pred_grid_call = grid_lookup(pred_density, r_edges, phi_edges, z_edges)
    azh_pred = to_agama_density(pred_grid_call, total_mass=m_pred_full,
                                r_max=float(r_edges[-1]), z_max=float(z_edges[-1]))
    fmodel = fit_fourier_rz_from_grid(pred_density, r_centers, z_centers, n_r=25, n_z_half=12,
                                      r_max=args.r_max_rep, z_max=args.z_max_rep)
    azh_fourier = fourier_rz_to_agama_density(fmodel, total_mass=m_pred_dom,
                                              r_max=args.r_max_rep, z_max=args.z_max_rep)
    cons = {
        "M_pred_full": m_pred_full, "M_pred_domain": m_pred_dom, "M_truth": m_truth_full,
        "azh_grid_totalMass": float(azh_pred.totalMass()),
        "azh_fourier_totalMass": float(azh_fourier.totalMass()),
        "azh_grid_mass_residual": abs(float(azh_pred.totalMass()) - m_pred_full) / m_pred_full,
        "azh_fourier_mass_residual": abs(float(azh_fourier.totalMass()) - m_pred_dom) / m_pred_dom,
    }
    print(f"AGAMA totalMass: grid-wrap {cons['azh_grid_totalMass']:.4e} (resid {cons['azh_grid_mass_residual']:.1e}), "
          f"fourier-wrap {cons['azh_fourier_totalMass']:.4e} (resid {cons['azh_fourier_mass_residual']:.1e})")

    # ---- potentials (predicted fourier / predicted grid / truth grid), same frame ----
    pot_pred = to_agama_potential(azh_fourier)
    pot_pred_grid = to_agama_potential(azh_pred)
    pot_truth = to_agama_potential(grid_lookup(truth_density, r_edges, phi_edges, z_edges))

    radii = np.linspace(0.4, 12.0, 40)
    vc = {"pred_fourier": circular_velocity(pot_pred, radii),
          "pred_grid": circular_velocity(pot_pred_grid, radii),
          "truth": circular_velocity(pot_truth, radii)}
    inside = radii < args.r_max_rep
    vc_rms = {k: float(np.sqrt(np.mean((vc[k][inside] - vc["truth"][inside]) ** 2)))
              for k in ("pred_fourier", "pred_grid")}
    print(f"v_c(R) rms vs truth (R<{args.r_max_rep:.0f} kpc): "
          f"pred_fourier {vc_rms['pred_fourier']:.1f}  pred_grid {vc_rms['pred_grid']:.1f} km/s")

    # ---- a few orbits in the predicted (fourier) potential ----
    orbits = []
    for r0, vfrac, vz in [(1.5, 0.95, 15.0), (3.0, 0.90, 25.0), (6.0, 0.98, 20.0)]:
        v_c0 = float(circular_velocity(pot_pred, np.array([r0]))[0])
        ic = [r0, 0.0, 0.0, 0.0, vfrac * v_c0, vz]
        _, traj = agama.orbit(potential=pot_pred, ic=ic, time=6.0, trajsize=800)
        orbits.append({"r0": r0, "v_c": v_c0, "traj": np.asarray(traj)})
    print("orbits integrated (6 time units ~ 5.9 Gyr):  "
          + "  ".join(f"R0={o['r0']:.1f} vc={o['v_c']:.0f}" for o in orbits))

    # ---- persist metrics ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "agama_predicted_potential_demo_metrics.json"
    out.write_text(json.dumps({
        "galaxy_id": args.galaxy_id, "row": row, "split": str(split),
        "inclination": args.inclination, "bar_angle": args.bar_angle,
        "mass_conservation": cons, "vc_rms_vs_truth_kms": vc_rms,
        "vc_radii_kpc": radii.tolist(), "vc": {k: v.tolist() for k, v in vc.items()},
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")

    # ---- figure: image | predicted edge-on density | v_c(R) | orbits xy | orbits Rz ----
    fig = plt.figure(figsize=(19, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 5)

    ax0 = fig.add_subplot(gs[0, 0])
    vmax = float(image.max())
    ax0.imshow(image.T, origin="lower", cmap="bone", norm=LogNorm(vmin=vmax * 3e-3, vmax=vmax))
    ax0.set_title(f"observed image\nTNG {args.galaxy_id}  i={args.inclination:.0f} deg")
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-12, 12, 200)
    zs = np.linspace(-4, 4, 100)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    dmap = azh_fourier.density(np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()])).reshape(xx.shape)
    dmax = float(dmap.max())
    ax1.imshow(dmap.T, origin="lower", extent=[-12, 12, -4, 4], cmap="magma", aspect="auto",
               norm=LogNorm(vmin=dmax * 3e-3, vmax=dmax))
    ax1.set_title("predicted 3D density (AGAMA)\nedge-on, mass conserved")
    ax1.set_xlabel("x [kpc]")
    ax1.set_ylabel("z [kpc]")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(radii, vc["truth"], "k-", lw=2, label="truth density")
    ax2.plot(radii, vc["pred_fourier"], color="#c44e52", lw=1.8,
             label=f"pred Fourier (rms {vc_rms['pred_fourier']:.1f})")
    ax2.plot(radii, vc["pred_grid"], color="#4c72b0", lw=1.2, ls="--",
             label=f"pred grid (rms {vc_rms['pred_grid']:.1f})")
    ax2.axvline(args.r_max_rep, color="gray", ls=":", lw=0.8)
    ax2.set_xlabel("R [kpc]")
    ax2.set_ylabel("v_c [km/s]")
    ax2.set_title("rotation curve from\npredicted potential")
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(gs[0, 3])
    cols = ["#1f77b4", "#2ca02c", "#d62728"]
    for o, c in zip(orbits, cols, strict=True):
        ax3.plot(o["traj"][:, 0], o["traj"][:, 1], color=c, lw=0.7, label=f"R0={o['r0']:.1f} kpc")
    ax3.set_aspect("equal")
    ax3.set_xlabel("x [kpc]")
    ax3.set_ylabel("y [kpc]")
    ax3.set_title("orbits in predicted\npotential (face-on)")
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(gs[0, 4])
    for o, c in zip(orbits, cols, strict=True):
        ax4.plot(np.hypot(o["traj"][:, 0], o["traj"][:, 1]), o["traj"][:, 2], color=c, lw=0.7)
    ax4.set_xlabel("R [kpc]")
    ax4.set_ylabel("z [kpc]")
    ax4.set_title("orbits (edge-on)")

    fig.suptitle("Route-B(2): predicted density -> to_agama_density -> agama.Potential -> v_c + orbits "
                 f"(TNG {args.galaxy_id}, held out, no retrain)", fontsize=12)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "figures" / "agama_predicted_potential_demo.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
