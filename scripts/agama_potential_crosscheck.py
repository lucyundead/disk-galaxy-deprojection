"""AGAMA CylSpline cross-check of the potential/force validation.

AGAMA is available (user pre-built; imported from the Agama build dir). Redo the
force comparison with the real CylSpline solver instead of the in-house FFT-Poisson
stand-in. For each galaxy build an AGAMA CylSpline potential from the particles
(gold) and from each representation's density (SPH-KDE truth, Fourier full /
compressed, cylindrical grid, superellipsoid), and compare forces over the galaxy
body. Also verifies AGAMA agrees with dgdp.poisson_fft on the same density (so the
earlier FFT-Poisson conclusions stand).

Run:
    PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 \\
        .venv/bin/python scripts/agama_potential_crosscheck.py
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

try:
    import agama
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama

from dgdp.density3d import build_cylindrical_density_grid, cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.fourier_rz import fit_fourier_rz, reconstruct_fourier_rz
from dgdp.poisson_fft import isolated_potential_and_forces
from dgdp.types import ParticleSet
from reconstruct_superellipsoid_3d import SPH, fit_shells, get_particles, reconstruct
from validate_fourier_rz_potential import load_allocation

agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s
REPS = ["sph_truth", "fourier_full", "fourier_compressed", "grid", "superellipsoid"]


def cylspline(*, density=None, particles=None):
    kwargs = dict(type="CylSpline", symmetry="triaxial", mmax=6,
                  gridSizeR=25, gridSizeZ=25, Rmin=0.1, Rmax=25.0, zmin=0.05, zmax=10.0)
    if density is not None:
        kwargs["density"] = density
    if particles is not None:
        kwargs["particles"] = particles
    return agama.Potential(**kwargs)


def grid_density_callable(pos, mass):
    """Precompute the cylindrical histogram density and return a fast lookup callable."""
    spec = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32)
    grid = build_cylindrical_density_grid(ParticleSet(positions_kpc=pos, masses_msun=mass), spec)
    dens = grid.mass_msun / cylindrical_bin_volumes(spec)
    re, pe, ze = spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc

    def f(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        rr, pp, zz = np.hypot(x[:, 0], x[:, 1]), np.arctan2(x[:, 1], x[:, 0]), x[:, 2]
        ir = np.clip(np.searchsorted(re, rr) - 1, 0, len(re) - 2)
        ip = np.clip(np.searchsorted(pe, pp) - 1, 0, len(pe) - 2)
        iz = np.searchsorted(ze, zz) - 1
        valid = (rr < re[-1]) & (iz >= 0) & (iz < len(ze) - 1)
        return np.where(valid, dens[ir, ip, np.clip(iz, 0, len(ze) - 2)], 0.0)

    return f


def _wrap(fn):
    return lambda x: fn(np.atleast_2d(np.asarray(x, dtype=float)))


def med_force_err(force_a, force_b, ref_mag):
    return float(np.median(np.linalg.norm(force_a - force_b, axis=1) / np.maximum(ref_mag, 1e-30)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--tng-particle-dir", type=Path, default=Path("/mnt/e/dgdp-fullparticles"))
    ap.add_argument("--tng-ids", type=int, nargs="+", default=[554189, 392276])
    ap.add_argument("--allocation", type=Path, default=Path("outputs/nbody_shen2010/fourier_rz_allocation.json"))
    ap.add_argument("--n-body", type=int, default=800)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    args = ap.parse_args()
    alloc = load_allocation(args.allocation)
    rng = np.random.default_rng(0)
    cases = [("Shen2010", "shen", None)]
    cases += [(f"TNG {i}", "tng", args.tng_particle_dir / f"subhalo_{i}.hdf5") for i in args.tng_ids]

    summary = {}
    for name, kind, hdf5 in cases:
        pos, mass = get_particles(kind, args.cache_dir, hdf5)
        rho = SPH(pos, mass, k=32)
        shells = fit_shells(rho)
        fmodel, fmodel_c = fit_fourier_rz(rho), fit_fourier_rz(rho, alloc=alloc)
        callables = {
            "sph_truth": rho,
            "fourier_full": _wrap(lambda x, m=fmodel: reconstruct_fourier_rz(x, m)),
            "fourier_compressed": _wrap(lambda x, m=fmodel_c: reconstruct_fourier_rz(x, m)),
            "grid": grid_density_callable(pos, mass),
            "superellipsoid": _wrap(lambda x, s=shells: reconstruct(x, s)),
        }

        # body sample points = the densest of a random cloud (avoids a full-box SPH eval)
        radius = rng.uniform(0.3, 9.0, 4000)
        phi = rng.uniform(-np.pi, np.pi, 4000)
        cloud = np.column_stack([radius * np.cos(phi), radius * np.sin(phi), rng.uniform(-3.0, 3.0, 4000)])
        pts = cloud[np.argsort(rho(cloud))[-args.n_body:]]

        gold = cylspline(particles=(pos, mass))
        gold_force = gold.force(pts)
        gold_mag = np.linalg.norm(gold_force, axis=1)
        pots = {rep: cylspline(density=callables[rep]) for rep in REPS}

        summary[name] = {"n_particles": int(pos.shape[0])}
        print(f"\n{name} ({pos.shape[0]} particles): AGAMA CylSpline median force error vs particle gold")
        for rep in REPS:
            err = med_force_err(pots[rep].force(pts), gold_force, gold_mag)
            summary[name][rep] = {"agama_force_err": err}
            print(f"  {rep:18s}: {err:.3f}")

        # solver agreement: AGAMA vs in-house FFT-Poisson on the same density (grid + fourier_full)
        h, half_xy, half_z = 0.25, 10.0, 3.5
        nxy, nz = int(2 * half_xy / h), int(2 * half_z / h)
        bx = (np.arange(nxy) - nxy / 2 + 0.5) * h
        bz = (np.arange(nz) - nz / 2 + 0.5) * h
        box = np.stack(np.meshgrid(bx, bx, bz, indexing="ij"), axis=-1).reshape(-1, 3)
        rho_box = rho(box)
        body = rho_box > 1e-2 * rho_box.max()
        body_pts = box[body]
        for rep in ("fourier_full", "grid"):
            dbox = callables[rep](box).reshape(nxy, nxy, nz)
            _, ax_, ay_, az_ = isolated_potential_and_forces(dbox, h)
            fft_force = np.stack([ax_.ravel()[body], ay_.ravel()[body], az_.ravel()[body]], axis=1)
            agama_force = pots[rep].force(body_pts)
            agree = med_force_err(fft_force, agama_force, np.linalg.norm(agama_force, axis=1))
            summary[name][rep]["agama_vs_fftpoisson"] = agree
        print(f"  [solver agreement AGAMA vs FFT-Poisson] fourier_full "
              f"{summary[name]['fourier_full']['agama_vs_fftpoisson']:.3f}  "
              f"grid {summary[name]['grid']['agama_vs_fftpoisson']:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agama_crosscheck_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # ---- figure: AGAMA force error per representation, per galaxy ----
    names = list(summary)
    colors = {"sph_truth": "k", "fourier_full": "#1f77b4", "fourier_compressed": "#c44e52",
              "grid": "#2ca02c", "superellipsoid": "#9467bd"}
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.5), constrained_layout=True)
    for ax, nm in zip(np.atleast_1d(axes), names, strict=True):
        ax.bar(range(len(REPS)), [summary[nm][r]["agama_force_err"] for r in REPS], color=[colors[r] for r in REPS])
        ax.set_xticks(range(len(REPS)))
        ax.set_xticklabels([r.replace("_", "\n") for r in REPS], fontsize=8)
        ax.set_ylabel("median |dF|/|F| vs particle gold")
        ax.set_title(f"{nm}")
    fig.suptitle("AGAMA CylSpline force error vs particle gold (real solver cross-check)", fontsize=13)
    fig_path = args.output_dir / "figures" / "agama_potential_crosscheck.png"
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.output_dir / 'agama_crosscheck_metrics.json'} and {fig_path}")


if __name__ == "__main__":
    main()
