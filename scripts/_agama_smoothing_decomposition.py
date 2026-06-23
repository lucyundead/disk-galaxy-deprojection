"""Decompose the Fourier-vs-CylSpline-from-particles force difference.

The AGAMA cross-check showed ~6-10% force error between the Fourier x (R,z) rep and
the CylSpline-from-particles "gold". Is that the odd-m asymmetry the Fourier rep
drops, or the (R,z)/SPH smoothing? This isolates each contribution by comparing,
all through the SAME AGAMA CylSpline solver, against the full particle potential
(`gold_full`, symmetry='none', mmax=8, which KEEPS the asymmetry):

  - gold_even (triaxial, mmax=6 from particles): drop odd-m asymmetry + m>6;
  - gold_axi  (axisymmetric from particles):     drop all non-axisymmetry (m>=2);
  - fourier_full / compressed densities (triaxial, mmax=6);
  - fourier_full vs gold_even: same even-m<=6 azimuthal content on both, so the
    remainder is purely the (R,z) + SPH-KDE smoothing.

Result (2026-06-23): the odd-m asymmetry costs only ~1-2%; the Fourier error is
almost entirely the (R,z)+SPH smoothing (fourier_full vs gold_even ~= fourier_full
vs gold_full). So the two smoothings differ in the radial/vertical plane, not
azimuthally.

Run:
    .venv/bin/python scripts/_agama_smoothing_decomposition.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import agama
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama

from dgdp.fourier_rz import fit_fourier_rz, reconstruct_fourier_rz
from reconstruct_superellipsoid_3d import SPH, get_particles
from validate_fourier_rz_potential import load_allocation

agama.setUnits(mass=1, length=1, velocity=1)


def cyl(symmetry, mmax, **kwargs):
    return agama.Potential(type="CylSpline", symmetry=symmetry, mmax=mmax, gridSizeR=30, gridSizeZ=30,
                           Rmin=0.1, Rmax=25.0, zmin=0.05, zmax=10.0, **kwargs)


def med(force_a, force_b):
    return float(np.median(np.linalg.norm(force_a - force_b, axis=1)
                           / np.maximum(np.linalg.norm(force_b, axis=1), 1e-30)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--tng-particle-dir", type=Path, default=Path("/mnt/e/dgdp-fullparticles"))
    ap.add_argument("--tng-ids", type=int, nargs="+", default=[554189, 392276])
    ap.add_argument("--allocation", type=Path, default=Path("outputs/nbody_shen2010/fourier_rz_allocation.json"))
    args = ap.parse_args()
    alloc = load_allocation(args.allocation)
    cases = [("Shen2010", "shen", None)]
    cases += [(f"TNG {i}", "tng", args.tng_particle_dir / f"subhalo_{i}.hdf5") for i in args.tng_ids]

    for name, kind, hdf5 in cases:
        pos, mass = get_particles(kind, args.cache_dir, hdf5)
        rho = SPH(pos, mass, k=32)
        fm, fmc = fit_fourier_rz(rho), fit_fourier_rz(rho, alloc=alloc)
        rng = np.random.default_rng(0)
        radius, phi = rng.uniform(0.3, 9.0, 3000), rng.uniform(-np.pi, np.pi, 3000)
        cloud = np.column_stack([radius * np.cos(phi), radius * np.sin(phi), rng.uniform(-3, 3, 3000)])
        pts = cloud[np.argsort(rho(cloud))[-800:]]

        gold_full = cyl("none", 8, particles=(pos, mass))        # keeps odd+even m<=8 (asymmetry in)
        gold_even = cyl("triaxial", 6, particles=(pos, mass))    # even m<=6 only (asymmetry out)
        gold_axi = cyl("axisymmetric", 0, particles=(pos, mass))  # m=0 only
        pot_f = cyl("triaxial", 6, density=lambda x, m=fm: reconstruct_fourier_rz(np.atleast_2d(np.asarray(x, float)), m))
        pot_fc = cyl("triaxial", 6, density=lambda x, m=fmc: reconstruct_fourier_rz(np.atleast_2d(np.asarray(x, float)), m))

        full, even = gold_full.force(pts), gold_even.force(pts)
        print(f"\n{name}: median force err vs gold_full (particles, m<=8 incl odd-m asymmetry)")
        print(f"  drop odd-m asymmetry + m>6 (gold_even):        {med(even, full):.3f}")
        print(f"  drop all non-axisym, m=0 only (gold_axi):      {med(gold_axi.force(pts), full):.3f}")
        print(f"  fourier_full (total):                          {med(pot_f.force(pts), full):.3f}")
        print(f"  fourier_compressed (total):                    {med(pot_fc.force(pts), full):.3f}")
        print(f"  fourier_full vs gold_even ((R,z)+SPH only):    {med(pot_f.force(pts), even):.3f}")


if __name__ == "__main__":
    main()
