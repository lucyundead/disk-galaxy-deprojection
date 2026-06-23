"""Is the potential/force ranking an artifact of the direct-sum softening choice?

Two robustness checks on `validate_fourier_rz_potential.py`:

(a) SOFTENING SWEEP: recompute the gold direct-particle force at a range of Plummer
    softenings eps and re-measure each representation's force error vs eps.

(b) SELF-CONSISTENT PM REFERENCE: deposit the particles on the same Cartesian box
    and run them through the SAME isolated FFT Poisson solver as every
    representation - i.e. everyone solves Poisson identically, no direct sum and no
    softening choice. Measure each representation's force error vs this PM truth.

If the ranking (grid most accurate, smooth Fourier worse) survives both, it is a
property of the density representations, not of the force method or eps.

Run:
    .venv/bin/python scripts/check_potential_softening.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgdp.fourier_rz import fit_fourier_rz, reconstruct_fourier_rz
from dgdp.poisson_fft import GRAV_KPC_KMS2_MSUN, isolated_potential_and_forces
from reconstruct_superellipsoid_3d import SPH, fit_shells, get_particles, grid_density_at, reconstruct
from validate_fourier_rz_potential import load_allocation

REPS = ["sph_truth", "fourier_full", "fourier_compressed", "grid_0.3125", "superellipsoid", "particle_PM"]


def particle_box_density(pos, mass, gx, gz, h):
    """Nearest-grid-point particle density on the Cartesian box (Msun/kpc^3)."""
    ex = np.concatenate([gx - h / 2, [gx[-1] + h / 2]])
    ez = np.concatenate([gz - h / 2, [gz[-1] + h / 2]])
    hist, _ = np.histogramdd(pos, bins=[ex, ex, ez], weights=mass)
    return hist / h**3


def direct_forces_multi_eps(points, pos, mass, eps_list, chunk=16):
    """Softened direct-particle accelerations at ``points`` for several eps (reuses distances)."""
    out = {e: np.zeros((len(points), 3)) for e in eps_list}
    for i in range(0, len(points), chunk):
        diff = points[i : i + chunk, None, :] - pos[None, :, :]
        d2 = diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2
        gm = mass[None, :]
        for e in eps_list:
            inv = (d2 + e * e) ** -1.5
            out[e][i : i + chunk] = -GRAV_KPC_KMS2_MSUN * np.einsum("cp,cpk->ck", inv * gm, diff)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--tng-particle-dir", type=Path, default=Path("/mnt/e/dgdp-fullparticles"))
    ap.add_argument("--tng-ids", type=int, nargs="+", default=[554189, 392276])
    ap.add_argument("--allocation", type=Path, default=Path("outputs/nbody_shen2010/fourier_rz_allocation.json"))
    ap.add_argument("--half-xy", type=float, default=10.0)
    ap.add_argument("--half-z", type=float, default=3.5)
    ap.add_argument("--spacing", type=float, default=0.25)
    ap.add_argument("--n-direct", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    eps_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]

    alloc = load_allocation(args.allocation)
    cases = [("Shen2010", "shen", None)]
    cases += [(f"TNG {i}", "tng", args.tng_particle_dir / f"subhalo_{i}.hdf5") for i in args.tng_ids]
    h = args.spacing
    nxy, nz = int(round(2 * args.half_xy / h)), int(round(2 * args.half_z / h))
    gx = (np.arange(nxy) - nxy / 2 + 0.5) * h
    gz = (np.arange(nz) - nz / 2 + 0.5) * h
    box = np.stack(np.meshgrid(gx, gx, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    shape = (nxy, nxy, nz)

    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4.6), constrained_layout=True)
    for col, (name, kind, hdf5) in enumerate(cases):
        pos, mass = get_particles(kind, args.cache_dir, hdf5)
        rho = SPH(pos, mass, k=32)
        shells = fit_shells(rho)
        densities = {
            "sph_truth": rho(box).reshape(shape),
            "fourier_full": reconstruct_fourier_rz(box, fit_fourier_rz(rho)).reshape(shape),
            "fourier_compressed": reconstruct_fourier_rz(box, fit_fourier_rz(rho, alloc=alloc)).reshape(shape),
            "grid_0.3125": grid_density_at(box, pos, mass, z_max=5.0, n_z=32).reshape(shape),
            "superellipsoid": reconstruct(box, shells).reshape(shape),
            "particle_PM": particle_box_density(pos, mass, gx, gz, h),
        }
        forces = {k: isolated_potential_and_forces(d, h)[1:] for k, d in densities.items()}

        body = densities["sph_truth"] > 1e-2 * densities["sph_truth"].max()
        body_idx = np.argwhere(body)
        sel = body_idx[rng.choice(len(body_idx), min(args.n_direct, len(body_idx)), replace=False)]
        cell_pts = np.stack([gx[sel[:, 0]], gx[sel[:, 1]], gz[sel[:, 2]]], axis=1)
        rep_force_at = {k: np.stack([f[0][tuple(sel.T)], f[1][tuple(sel.T)], f[2][tuple(sel.T)]], axis=1)
                        for k, f in forces.items()}

        direct = direct_forces_multi_eps(cell_pts, pos, mass, eps_list)

        # (a) error vs direct-sum at each eps; (b) error vs the self-consistent PM reference
        def med_err(a_rep, a_ref):
            return float(np.median(np.linalg.norm(a_rep - a_ref, axis=1) / np.maximum(np.linalg.norm(a_ref, axis=1), 1e-30)))

        pm_ref = rep_force_at["particle_PM"]
        print(f"\n=== {name} ({pos.shape[0]} particles) median fractional force error ===")
        print("  rep \\\\ eps[kpc]   " + "  ".join(f"{e:>5.2f}" for e in eps_list) + "   | vs PM-grid")
        sweep = {}
        for rep in ["fourier_full", "fourier_compressed", "grid_0.3125", "superellipsoid", "sph_truth"]:
            row = [med_err(rep_force_at[rep], direct[e]) for e in eps_list]
            sweep[rep] = row
            vs_pm = med_err(rep_force_at[rep], pm_ref)
            print(f"  {rep:18s} " + "  ".join(f"{v:>5.2f}" for v in row) + f"   | {vs_pm:.3f}")
        # how well the PM reference itself matches the direct sum (consistency of the two truths)
        pm_vs_direct = [med_err(pm_ref, direct[e]) for e in eps_list]
        print(f"  {'particle_PM':18s} " + "  ".join(f"{v:>5.2f}" for v in pm_vs_direct) + "   | (reference)")

        styles = {"fourier_full": ("#1f77b4", "-"), "fourier_compressed": ("#c44e52", "--"),
                  "grid_0.3125": ("#2ca02c", ":"), "superellipsoid": ("#9467bd", "-."), "sph_truth": ("k", "-")}
        for rep, (color, ls) in styles.items():
            axes[col].plot(eps_list, sweep[rep], color=color, ls=ls, marker="o", ms=4, label=rep)
        axes[col].plot(eps_list, pm_vs_direct, color="#888888", ls="-", marker="s", ms=3, label="particle_PM (ref)")
        axes[col].axvspan(0.1, 0.3, color="gold", alpha=0.15, label="TNG50/grid softening")
        axes[col].set_xlabel("direct-sum softening eps [kpc]")
        axes[col].set_ylabel("median |da|/|a| vs direct-sum")
        axes[col].set_title(name)
        axes[col].legend(fontsize=7)

    fig.suptitle("Softening sensitivity of the force comparison (vs direct-sum at eps; bands = sim softening)", fontsize=13)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "figures" / "potential_softening_check.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
