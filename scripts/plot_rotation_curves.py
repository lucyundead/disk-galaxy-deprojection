"""Rotation-curve comparison v_c(R) across the 3D density representations.

For each galaxy (Shen2010 + TNG 554189 + 392276) overplots the STELLAR circular
velocity v_c(R) = sqrt(-R <a_R>_phi) in the midplane from:
  - the particles directly (gold reference: azimuthally-averaged direct-sum radial
    force on z=0 rings; v_c in the plane is ~softening-insensitive at R > a few eps),
  - the cylindrical grid, the Fourier x (R,z) full + compressed, the superellipsoid
    (each via the isolated FFT Poisson solver on a common box).

Stars only - this is the stellar contribution to the rotation curve, not the
total (DM+gas) curve.

Run:
    .venv/bin/python scripts/plot_rotation_curves.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgdp.fourier_rz import fit_fourier_rz, reconstruct_fourier_rz
from dgdp.poisson_fft import circular_velocity_profile, isolated_potential_and_forces
from reconstruct_superellipsoid_3d import SPH, get_particles, grid_density_at
from validate_fourier_rz_potential import direct_particle_forces


def direct_rotation_curve(pos, mass, r_mid, *, softening=0.1, n_phi=48):
    """True stellar v_c(R) from azimuthally-averaged direct-sum radial force at z=0."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    v_c = np.zeros(len(r_mid))
    for i, radius in enumerate(r_mid):
        ring = np.column_stack([radius * np.cos(phi), radius * np.sin(phi), np.zeros(n_phi)])
        accel = direct_particle_forces(ring, pos, mass, softening=softening)
        a_radial = (ring[:, 0] * accel[:, 0] + ring[:, 1] * accel[:, 1]) / radius
        v_c[i] = np.sqrt(max(-radius * float(a_radial.mean()), 0.0))
    return v_c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--tng-particle-dir", type=Path, default=Path("/mnt/e/dgdp-fullparticles"))
    ap.add_argument("--tng-ids", type=int, nargs="+", default=[554189, 392276])
    ap.add_argument("--half-xy", type=float, default=10.0)
    ap.add_argument("--half-z", type=float, default=3.5)
    ap.add_argument("--spacing", type=float, default=0.25)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    args = ap.parse_args()

    cases = [("Shen2010", "shen", None)]
    cases += [(f"TNG {i}", "tng", args.tng_particle_dir / f"subhalo_{i}.hdf5") for i in args.tng_ids]
    h = args.spacing
    nxy, nz = int(round(2 * args.half_xy / h)), int(round(2 * args.half_z / h))
    gx = (np.arange(nxy) - nxy / 2 + 0.5) * h
    gz = (np.arange(nz) - nz / 2 + 0.5) * h
    box = np.stack(np.meshgrid(gx, gx, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    shape = (nxy, nxy, nz)
    r_edges = np.linspace(0.4, args.half_xy - 0.5, 26)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

    styles = {
        "particles (truth)": ("k", 2.6, "-"),
        "cylindrical grid": ("#2ca02c", 1.6, ":"),
        "Fourier full (SPH-anchored)": ("#1f77b4", 1.6, "-"),
        "Fourier mass-conserving": ("#dd8452", 2.0, "--"),
    }
    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4.7), constrained_layout=True)
    for col, (name, kind, hdf5) in enumerate(cases):
        pos, mass = get_particles(kind, args.cache_dir, hdf5)
        rho = SPH(pos, mass, k=32)
        rho_box = rho(box).reshape(shape)
        fourier_full = reconstruct_fourier_rz(box, fit_fourier_rz(rho)).reshape(shape)
        # mass-conserving construction: pin the projected surface density (int rho dz) to the
        # truth column-by-column, keeping the Fourier vertical shape -> correct enclosed mass.
        sigma_t = rho_box.sum(axis=2)
        sigma_f = fourier_full.sum(axis=2)
        ratio = np.where(sigma_f > 1e-6 * sigma_f.max(), sigma_t / np.where(sigma_f > 0, sigma_f, 1.0), 0.0)
        densities = {
            "cylindrical grid": grid_density_at(box, pos, mass, z_max=5.0, n_z=32).reshape(shape),
            "Fourier full (SPH-anchored)": fourier_full,
            "Fourier mass-conserving": fourier_full * ratio[:, :, None],
        }
        curves = {"particles (truth)": direct_rotation_curve(pos, mass, r_mid)}
        for key, dens in densities.items():
            _, ax_, ay_, _ = isolated_potential_and_forces(dens, h)
            curves[key] = circular_velocity_profile(ax_, ay_, gx, gx, gz, r_edges, z_slab=2 * h)

        for key, (color, lw, ls) in styles.items():
            axes[col].plot(r_mid, curves[key], color=color, lw=lw, ls=ls, label=key)
        axes[col].set_xlabel("R [kpc]")
        axes[col].set_ylabel("stellar v_c [km/s]")
        axes[col].set_title(f"{name} ({pos.shape[0]:,} star particles)")
        axes[col].set_ylim(bottom=0)
        axes[col].grid(alpha=0.25)
        axes[col].legend(fontsize=8)
        rms = {k: float(np.sqrt(np.nanmean((curves[k] - curves["particles (truth)"]) ** 2)))
               for k in densities}
        print(f"{name}: v_c rms error vs particles [km/s]  "
              + "  ".join(f"{k.split()[0]}={v:.1f}" for k, v in rms.items()))

    fig.suptitle("Stellar rotation curve v_c(R): representations vs the particles (stars only)", fontsize=13)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "figures" / "rotation_curve_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
