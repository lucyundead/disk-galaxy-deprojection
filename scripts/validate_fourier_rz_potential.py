"""Potential/force validation of the 3D density representations.

Does the adopted even-m Fourier x (R,z) representation - and its power-weighted
compression - reproduce the *dynamics* (potential, forces, rotation curve) of the
truth density? Compares, per galaxy (Shen2010 + TNG 554189 + 392276), the SPH-KDE
truth against:
  - Fourier x (R,z) full (2002 coeff),
  - Fourier x (R,z) compressed (the Task-1 power-weighted allocation, ~500 coeff),
  - the cylindrical grid (0.3125 kpc, the eval scaffold),
  - the nested superellipsoid (the compact bulge descriptor).

Each density is sampled on a common fine Cartesian box and run through an isolated
FFT Poisson/force solver (dgdp.poisson_fft; AGAMA's CylSpline needs a compiler the
sandbox lacks, but the maps ARE in CylSpline form - this measures the same
density->force fidelity). Reports the circular-velocity curve v_c(R), the
body-averaged fractional force error, and the potential error.

Run:
    .venv/bin/python scripts/validate_fourier_rz_potential.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgdp.fourier_rz import fit_fourier_rz, reconstruct_fourier_rz
from dgdp.poisson_fft import GRAV_KPC_KMS2_MSUN, circular_velocity_profile, isolated_potential_and_forces
from reconstruct_superellipsoid_3d import SPH, fit_shells, get_particles, grid_density_at, reconstruct


def load_allocation(path: Path) -> dict[int, tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(m): (int(a), int(b)) for m, (a, b) in payload["allocation"].items()}


def direct_particle_forces(points, pos, mass, *, softening, chunk=25):
    """Gold-standard softened acceleration from the particles at ``points`` [(km/s)^2/kpc]."""
    eps2 = softening * softening
    out = np.zeros((len(points), 3))
    for i in range(0, len(points), chunk):
        diff = points[i : i + chunk, None, :] - pos[None, :, :]
        inv = (diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2 + eps2) ** -1.5
        out[i : i + chunk] = -GRAV_KPC_KMS2_MSUN * np.einsum("cp,cpk->ck", inv * mass[None, :], diff)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--tng-particle-dir", type=Path, default=Path("/mnt/e/dgdp-fullparticles"))
    ap.add_argument("--tng-ids", type=int, nargs="+", default=[554189, 392276])
    ap.add_argument("--allocation", type=Path, default=Path("outputs/nbody_shen2010/fourier_rz_allocation.json"))
    # box fully inside the Fourier domain (corner R = half_xy*sqrt(2) < r_max=15,
    # |z| < z_max=4) so no clamping; isolates representation fidelity in the
    # bar + inner-disk region where the bar/peanut dynamics live.
    ap.add_argument("--half-xy", type=float, default=10.0)
    ap.add_argument("--half-z", type=float, default=3.5)
    ap.add_argument("--spacing", type=float, default=0.25)
    ap.add_argument("--n-direct", type=int, default=800, help="body cells for the gold direct-particle force check")
    ap.add_argument("--softening", type=float, default=0.3, help="Plummer softening for the direct sum [kpc]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    alloc = load_allocation(args.allocation)
    cases = [("Shen2010", "shen", None)]
    cases += [(f"TNG {i}", "tng", args.tng_particle_dir / f"subhalo_{i}.hdf5") for i in args.tng_ids]

    # common Cartesian box (cell-centered)
    nxy = int(round(2 * args.half_xy / args.spacing))
    nz = int(round(2 * args.half_z / args.spacing))
    gx = (np.arange(nxy) - nxy / 2 + 0.5) * args.spacing
    gz = (np.arange(nz) - nz / 2 + 0.5) * args.spacing
    box = np.stack(np.meshgrid(gx, gx, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    shape = (nxy, nxy, nz)
    r_edges = np.linspace(0.3, args.half_xy - 1.0, 28)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    print(f"box {shape} at {args.spacing} kpc ({box.shape[0]} cells); v_c slab |z|<{args.spacing * 2:.2f}")

    summary, vc_curves = {}, {}
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 4.3 * len(cases)), constrained_layout=True)
    for row, (name, kind, hdf5) in enumerate(cases):
        pos, mass = get_particles(kind, args.cache_dir, hdf5)
        rho = SPH(pos, mass, k=32)
        shells = fit_shells(rho)
        densities = {
            "truth": rho(box).reshape(shape),
            "fourier_full": reconstruct_fourier_rz(box, fit_fourier_rz(rho)).reshape(shape),
            "fourier_compressed": reconstruct_fourier_rz(box, fit_fourier_rz(rho, alloc=alloc)).reshape(shape),
            "grid_0.3125": grid_density_at(box, pos, mass, z_max=5.0, n_z=32).reshape(shape),
            "superellipsoid": reconstruct(box, shells).reshape(shape),
        }
        fields = {k: isolated_potential_and_forces(d, args.spacing) for k, d in densities.items()}
        body = densities["truth"] > 1e-2 * densities["truth"].max()

        # gold reference: softened direct-particle forces at a sample of body cells
        body_idx = np.argwhere(body)
        sel = body_idx[rng.choice(len(body_idx), min(args.n_direct, len(body_idx)), replace=False)]
        cell_pts = np.stack([gx[sel[:, 0]], gx[sel[:, 1]], gz[sel[:, 2]]], axis=1)
        a_direct = direct_particle_forces(cell_pts, pos, mass, softening=args.softening)
        a_dir_mag = np.linalg.norm(a_direct, axis=1)
        weights = densities["truth"][tuple(sel.T)]

        phi_t = fields["truth"][0]
        phi_rms = float(np.sqrt(np.mean(phi_t[body] ** 2)))
        summary[name] = {"n_particles": int(pos.shape[0]), "n_direct_cells": int(len(sel))}
        vc_curves[name] = {"r_mid": r_mid.tolist()}
        for key, (phi, ax_, ay_, az_) in fields.items():
            v_c = circular_velocity_profile(ax_, ay_, gx, gx, gz, r_edges, z_slab=args.spacing * 2)
            vc_curves[name][key] = v_c.tolist()
            a_rep = np.stack([ax_[tuple(sel.T)], ay_[tuple(sel.T)], az_[tuple(sel.T)]], axis=1)
            frac = np.linalg.norm(a_rep - a_direct, axis=1) / np.maximum(a_dir_mag, 1e-30)
            summary[name][key] = {
                "force_err_median": float(np.median(frac)),
                "force_err_p84": float(np.percentile(frac, 84)),
                "force_err_massweighted": float(np.sum(frac * weights) / np.sum(weights)),
                "pot_err_rms_frac_vs_sphkde": float(np.sqrt(np.mean((phi[body] - phi_t[body]) ** 2)) / phi_rms),
            }
        truth_vc = np.array(vc_curves[name]["truth"])
        for key in fields:
            summary[name][key]["vc_rms_err_kms_vs_truthfield"] = float(
                np.sqrt(np.nanmean((np.array(vc_curves[name][key]) - truth_vc) ** 2)))

        # ---- figure ----
        styles = {"truth": ("k", 2.4, "-"), "fourier_full": ("#1f77b4", 1.4, "-"),
                  "fourier_compressed": ("#c44e52", 1.6, "--"), "grid_0.3125": ("#2ca02c", 1.4, ":"),
                  "superellipsoid": ("#9467bd", 1.4, "-.")}
        for key, (color, lw, ls) in styles.items():
            axes[row, 0].plot(r_mid, vc_curves[name][key], color=color, lw=lw, ls=ls, label=key)
        axes[row, 0].set_xlabel("R [kpc]")
        axes[row, 0].set_ylabel("v_c [km/s]")
        axes[row, 0].set_title(f"{name}: rotation curve")
        axes[row, 0].legend(fontsize=7)

        keys = ["truth", "fourier_full", "fourier_compressed", "grid_0.3125", "superellipsoid"]
        med = [summary[name][k]["force_err_median"] for k in keys]
        p84 = [summary[name][k]["force_err_p84"] for k in keys]
        xpos = np.arange(len(keys))
        axes[row, 1].bar(xpos, med, color=[styles[k][0] for k in keys], alpha=0.85)
        axes[row, 1].errorbar(xpos, med, yerr=[[0] * len(keys), np.array(p84) - np.array(med)],
                              fmt="none", ecolor="k", capsize=3, lw=1)
        axes[row, 1].set_xticks(xpos)
        axes[row, 1].set_xticklabels([k.replace("_", "\n").replace("0.3125", "grid") for k in keys], fontsize=8)
        axes[row, 1].set_ylabel("|da|/|a| vs direct particles")
        axes[row, 1].set_title(f"{name}: force error vs particles (median, p84)")
        print(f"\n{name} ({pos.shape[0]} particles): force err vs direct particles "
              f"(median / mass-wt / p84), pot-err vs SPH, v_c-rms")
        for k in keys:
            s = summary[name][k]
            print(f"  {k:20s}: {s['force_err_median']:.3f} / {s['force_err_massweighted']:.3f} / "
                  f"{s['force_err_p84']:.3f}   pot {s['pot_err_rms_frac_vs_sphkde']:.3f}   "
                  f"vc {s['vc_rms_err_kms_vs_truthfield']:.1f} km/s")

    fig.suptitle("Potential/force validation vs direct-particle forces (isolated FFT Poisson; gold = particles)", fontsize=13)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "figures" / "fourier_rz_potential_validation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    (args.output_dir / "fourier_rz_potential_metrics.json").write_text(
        json.dumps({"box_shape": list(shape), "spacing_kpc": args.spacing,
                    "summary": summary, "vc_curves": vc_curves}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {fig_path}")
    print(f"wrote {args.output_dir / 'fourier_rz_potential_metrics.json'}")


if __name__ == "__main__":
    main()
