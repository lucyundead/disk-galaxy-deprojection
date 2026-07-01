"""Route-B(1) cheap prototype: does AGAMA's native density expansion raise the
deprojection representation ceiling over the even-m Fourier x (R,z) target?

On the three reference galaxies with a smooth SPH-KDE truth (Shen2010 + TNG 554189
/ 392276, full particles), build the density as an AGAMA ``DensityAzimuthalHarmonic``
expansion on a *fixed* (gridsizeR, gridsizez, mmax, symmetry) and verify its
coefficients round-trip exactly through ``export()`` / re-import. Then compare the
*oracle* (true-coefficient) reconstruction fidelity of three smooth representations
against the same SPH-KDE truth, plus their AGAMA-CylSpline force error vs the
particle gold:

  - AGAMA DensityAzimuthalHarmonic (the candidate native expansion),
  - even-m Fourier x (R,z) full + compressed (the current target),
  - the cylindrical grid 0.3125 kpc (the eval scaffold).

The raw (no-PCA) oracle is an *upper bound* on any PCA-32 oracle, so if the AGAMA
expansion is not clearly better here it cannot help the PCA-32 deprojection oracle
either -- the gate for the full milestone-2d MDN/flow retrain. This does NOT
regenerate the frozen ``fourier_rz_allocation.json``.

Run:
    PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 \\
        .venv/bin/python scripts/agama_density_oracle_prototype.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
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

from dgdp.fourier_rz import fit_fourier_rz, n_coefficients, reconstruct_fourier_rz
from agama_potential_crosscheck import cylspline, grid_density_callable, med_force_err
from reconstruct_superellipsoid_3d import SPH, get_particles, grid_density_at, rel_l2
from validate_fourier_rz_potential import load_allocation

agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s


def build_azh(rho, *, gridsizeR: int, gridsizez: int, mmax: int) -> agama.Density:
    """AGAMA DensityAzimuthalHarmonic fit to a density callable (Cartesian N,3 in)."""
    return agama.Density(
        type="DensityAzimuthalHarmonic", density=rho,
        gridsizeR=gridsizeR, gridsizez=gridsizez, mmax=mmax,
        Rmin=0.1, Rmax=15.0, zmin=0.05, zmax=4.0, symmetry="triaxial",
    )


def azh_export_roundtrip(d: agama.Density, pts: np.ndarray) -> tuple[agama.Density, float, dict]:
    """Export coefficients to text, re-import, and report the round-trip density diff.

    Returns ``(reimported_density, median_rel_diff, header_info)`` where header_info
    holds the grid sizes / mmax / coefficient count parsed from the export file.
    """
    tmp = Path(tempfile.mkdtemp()) / "azh.coef"
    d.export(str(tmp))
    txt = tmp.read_text(encoding="utf-8")
    d2 = agama.Density(file=str(tmp))
    v1, v2 = d.density(pts), d2.density(pts)
    rel = float(np.median(np.abs(v1 - v2) / np.maximum(np.abs(v1), 1e-30)))

    def _int(key: str) -> int:
        return int(re.search(rf"{key}=(\d+)", txt).group(1))

    gR, gz, mm = _int("gridSizeR"), _int("gridSizez"), _int("mmax")
    n_harm = len(range(0, mm + 1, 2))  # triaxial keeps even m only
    # data rows per m-block: numeric lines after the "#R(row)\z(col)" header
    blocks = re.split(r"^\d+\t#m", txt, flags=re.M)[1:]
    rows0 = sum(1 for ln in blocks[0].splitlines() if re.match(r"^[-+0-9.]", ln)) if blocks else gR
    z_cols = len(re.search(r"#R\(row\)\\z\(col\)\t(.*)", txt).group(1).split())
    n_coeff = n_harm * rows0 * z_cols
    return d2, rel, {"gridSizeR": gR, "gridSizez": gz, "mmax": mm, "n_coeff": int(n_coeff)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--tng-particle-dir", type=Path, default=Path("/mnt/e/dgdp-fullparticles"))
    ap.add_argument("--tng-ids", type=int, nargs="+", default=[554189, 392276])
    ap.add_argument("--allocation", type=Path, default=Path("outputs/nbody_shen2010/fourier_rz_allocation.json"))
    ap.add_argument("--azh-grid", type=int, nargs=2, default=[25, 25], help="gridsizeR gridsizez")
    ap.add_argument("--azh-mmax", type=int, nargs="+", default=[6, 10])
    ap.add_argument("--n-body", type=int, default=800)
    ap.add_argument("--no-forces", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    args = ap.parse_args()
    alloc = load_allocation(args.allocation)
    gR, gz = args.azh_grid
    rng = np.random.default_rng(0)

    cases = [("Shen2010", "shen", None)]
    cases += [(f"TNG {i}", "tng", args.tng_particle_dir / f"subhalo_{i}.hdf5") for i in args.tng_ids]

    # fine 3D evaluation grid (matches compress_fourier_rz_target.py exactly)
    gx = np.arange(-10, 10.01, 0.2)
    gzv = np.arange(-3.5, 3.51, 0.2)
    grid_x, grid_y, grid_z = np.meshgrid(gx, gx, gzv, indexing="ij")
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    summary: dict = {}
    for name, kind, hdf5 in cases:
        pos, mass = get_particles(kind, args.cache_dir, hdf5)
        rho = SPH(pos, mass, k=32)
        truth = rho(pts)
        mask = truth > 1e-3 * truth.max()

        # ---- Fourier x (R,z): full 25x25 default, full 14x13 study grid, compressed ----
        f_full = fit_fourier_rz(rho)  # 25x25 default
        f_full14 = fit_fourier_rz(rho, n_r=14, n_z_half=6)  # historical study grid
        f_comp = fit_fourier_rz(rho, n_r=14, n_z_half=6, alloc=alloc)  # frozen 496-coeff target
        rec = {
            "fourier_full_25x25": (rel_l2(reconstruct_fourier_rz(pts, f_full), truth, mask),
                                   n_coefficients(f_full)),
            "fourier_full_14x13": (rel_l2(reconstruct_fourier_rz(pts, f_full14), truth, mask),
                                   n_coefficients(f_full14)),
            "fourier_compressed": (rel_l2(reconstruct_fourier_rz(pts, f_comp), truth, mask),
                                   n_coefficients(f_comp)),
            "grid_0.3125": (rel_l2(grid_density_at(pts, pos, mass, z_max=5.0, n_z=32), truth, mask), 49152),
        }

        # ---- AGAMA DensityAzimuthalHarmonic, fixed grid+mmax+sym, export/import round-trip ----
        azh_objs: dict[int, agama.Density] = {}
        roundtrip: dict[int, dict] = {}
        for mm in args.azh_mmax:
            d = build_azh(rho, gridsizeR=gR, gridsizez=gz, mmax=mm)
            d2, rtrip, hdr = azh_export_roundtrip(d, pts)
            azh_objs[mm] = d2
            rec[f"agama_azh_mmax{mm}"] = (rel_l2(d2.density(pts), truth, mask), hdr["n_coeff"])
            roundtrip[mm] = {"reldiff": rtrip, "mmax_used": hdr["mmax"], "n_coeff": hdr["n_coeff"]}

        summary[name] = {"n_particles": int(pos.shape[0]),
                         "rel_l2": {k: float(v[0]) for k, v in rec.items()},
                         "n_coeff": {k: int(v[1]) for k, v in rec.items()},
                         "azh_roundtrip": {str(mm): roundtrip[mm] for mm in roundtrip}}
        print(f"\n{name} ({pos.shape[0]} particles): 3D rel-L2 vs SPH-KDE truth  [n_coeff]")
        for k in ("fourier_full_25x25", "fourier_full_14x13", "fourier_compressed",
                  *[f"agama_azh_mmax{m}" for m in args.azh_mmax], "grid_0.3125"):
            print(f"  {k:22s}: {rec[k][0]:.3f}   [{rec[k][1]}]")
        for mm in args.azh_mmax:
            print(f"  (AZH mmax={mm} -> used mmax={roundtrip[mm]['mmax_used']}, "
                  f"export/import round-trip median rel diff: {roundtrip[mm]['reldiff']:.1e})")

        # ---- AGAMA CylSpline force error vs particle gold ----
        if not args.no_forces:
            radius = rng.uniform(0.3, 9.0, 4000)
            phi = rng.uniform(-np.pi, np.pi, 4000)
            cloud = np.column_stack([radius * np.cos(phi), radius * np.sin(phi),
                                     rng.uniform(-3.0, 3.0, 4000)])
            body = cloud[np.argsort(rho(cloud))[-args.n_body:]]
            gold = cylspline(particles=(pos, mass))
            gforce = gold.force(body)
            gmag = np.linalg.norm(gforce, axis=1)
            force_reps = {
                "agama_azh_mmax%d" % args.azh_mmax[0]: lambda x, d=azh_objs[args.azh_mmax[0]]: d.density(x),
                "fourier_full_25x25": lambda x, m=f_full: reconstruct_fourier_rz(
                    np.atleast_2d(np.asarray(x, dtype=float)), m),
                "grid_0.3125": grid_density_callable(pos, mass),
            }
            ferr = {}
            for rep, dcall in force_reps.items():
                pot = cylspline(density=dcall)
                ferr[rep] = med_force_err(pot.force(body), gforce, gmag)
            summary[name]["force_err"] = {k: float(v) for k, v in ferr.items()}
            print("  AGAMA CylSpline median force err vs particle gold: "
                  + "  ".join(f"{k.split('_')[0]}:{v:.3f}" for k, v in ferr.items()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "agama_density_oracle_metrics.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out}")

    # ---- figure: oracle rel-L2 per representation, per galaxy ----
    names = list(summary)
    reps = ["fourier_full_25x25", "fourier_compressed",
            *[f"agama_azh_mmax{m}" for m in args.azh_mmax], "grid_0.3125"]
    colors = ["#1f77b4", "#c44e52", "#2ca02c", "#8c564b", "#9467bd", "#777777"][: len(reps)]
    fig, axes = plt.subplots(1, len(names), figsize=(5.0 * len(names), 4.6), constrained_layout=True)
    for ax, nm in zip(np.atleast_1d(axes), names, strict=True):
        vals = [summary[nm]["rel_l2"][r] for r in reps]
        ax.bar(range(len(reps)), vals, color=colors)
        ax.set_xticks(range(len(reps)))
        ax.set_xticklabels([r.replace("_", "\n") for r in reps], fontsize=8)
        ax.set_ylabel("3D rel-L2 vs SPH-KDE truth")
        ax.set_title(f"{nm}")
    fig.suptitle("Route-B(1): AGAMA DensityAzimuthalHarmonic vs Fourier x (R,z) oracle "
                 "(smooth-truth representation ceiling)", fontsize=12)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "figures" / "agama_density_oracle_prototype.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
