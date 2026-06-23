"""Power-weighted (R,z) compression of the even-m Fourier x (R,z) target.

A uniform (R,z) knot grid for every harmonic costs ~2000 coefficients. The high
harmonics carry little volume-weighted power, and they need far fewer (R,z) DOF: a
power-weighted reverse water-filling allocates fewest-coefficient per-harmonic
grids that jointly keep >= `capture` of the power across the reference-galaxy
population (Shen2010 + TNG 554189 + 392276, SPH-KDE truth). This sweeps `capture`
to expose the coefficient-count / 3D-rel-L2 knee, then saves one chosen shared
allocation to JSON for reuse as the deprojection target layout (Task 2), with a
figure.

Run:
    .venv/bin/python scripts/compress_fourier_rz_target.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgdp.fourier_rz import (
    EVEN_M,
    derive_power_allocation_multi,
    fit_fourier_rz,
    n_coefficients,
    reconstruct_fourier_rz,
)
from reconstruct_superellipsoid_3d import SPH, get_particles, grid_density_at, rel_l2


def _alloc_n_coeff(alloc: dict[int, tuple[int, int]]) -> int:
    return int(sum((1 if m == 0 else 2) * n_r * (2 * n_zh + 1) for m, (n_r, n_zh) in alloc.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--tng-particle-dir", type=Path, default=Path("/mnt/e/dgdp-fullparticles"))
    ap.add_argument("--tng-ids", type=int, nargs="+", default=[554189, 392276])
    ap.add_argument("--capture", type=float, default=0.95, help="chosen capture saved to JSON/figure")
    ap.add_argument("--capture-sweep", type=float, nargs="+",
                    default=[0.99, 0.98, 0.97, 0.95, 0.93, 0.90, 0.85])
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    args = ap.parse_args()

    cases = [("Shen2010", "shen", None)]
    cases += [(f"TNG {i}", "tng", args.tng_particle_dir / f"subhalo_{i}.hdf5") for i in args.tng_ids]

    # ---- fit the full model per galaxy + cache its eval-grid truth / grid baseline ----
    gx = np.arange(-10, 10.01, 0.2)
    gz = np.arange(-3.5, 3.51, 0.2)
    grid_x, grid_y, grid_z = np.meshgrid(gx, gx, gz, indexing="ij")
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    fitted: dict[str, dict] = {}
    for name, kind, hdf5 in cases:
        pos, mass = get_particles(kind, args.cache_dir, hdf5)
        rho = SPH(pos, mass, k=32)
        model = fit_fourier_rz(rho)
        truth = rho(pts)
        mask = truth > 1e-3 * truth.max()
        fitted[name] = {
            "rho": rho, "model": model, "truth": truth, "mask": mask,
            "rel_full": rel_l2(reconstruct_fourier_rz(pts, model), truth, mask),
            "rel_grid": rel_l2(grid_density_at(pts, pos, mass, z_max=5.0, n_z=32), truth, mask),
            "n_particles": int(pos.shape[0]),
        }
        print(f"{name}: {pos.shape[0]} particles; full rel-L2 {fitted[name]['rel_full']:.3f}; "
              f"grid 0.3125 rel-L2 {fitted[name]['rel_grid']:.3f}")

    models = [fitted[name]["model"] for name in fitted]
    n_full = n_coefficients(models[0])
    _, agg_info = derive_power_allocation_multi(models, capture=0.99)
    print("\naggregate power fraction per m: "
          + "  ".join(f"m{m}={agg_info['power_fraction'][m]:.2e}" for m in EVEN_M))

    # ---- sweep capture: coefficient count vs per-galaxy compressed rel-L2 ----
    print(f"\n{'capture':>8} {'n_coeff':>8} {'%full':>6}  " + "  ".join(f"{n[:9]:>9}" for n in fitted))
    print(f"{'full':>8} {n_full:>8} {'100%':>6}  " + "  ".join(f"{fitted[n]['rel_full']:>9.3f}" for n in fitted))
    sweep_rows = []
    for capture in args.capture_sweep:
        alloc, _ = derive_power_allocation_multi(models, capture=capture)
        n_c = _alloc_n_coeff(alloc)
        rels = {}
        for name in fitted:
            compact = fit_fourier_rz(fitted[name]["rho"], alloc=alloc)
            rels[name] = rel_l2(reconstruct_fourier_rz(pts, compact), fitted[name]["truth"], fitted[name]["mask"])
        sweep_rows.append({"capture": capture, "n_coeff": n_c, "alloc": alloc, "rel": rels})
        print(f"{capture:>8.3f} {n_c:>8} {n_c / n_full:>5.0%}  " + "  ".join(f"{rels[n]:>9.3f}" for n in fitted))
    print(f"{'grid':>8} {49152:>8} {'':>6}  " + "  ".join(f"{fitted[n]['rel_grid']:>9.3f}" for n in fitted))

    # ---- chosen allocation ----
    chosen_alloc, chosen_info = derive_power_allocation_multi(models, capture=args.capture)
    chosen_alloc = dict(sorted(chosen_alloc.items()))
    n_chosen = _alloc_n_coeff(chosen_alloc)
    print(f"\nCHOSEN allocation (capture {args.capture}): {n_chosen} coeff "
          f"({n_chosen / n_full:.0%} of {n_full})")
    results = {}
    for name in fitted:
        compact = fit_fourier_rz(fitted[name]["rho"], alloc=chosen_alloc)
        results[name] = {
            "n_particles": fitted[name]["n_particles"],
            "rel_l2_full": fitted[name]["rel_full"],
            "rel_l2_compressed": rel_l2(reconstruct_fourier_rz(pts, compact), fitted[name]["truth"], fitted[name]["mask"]),
            "rel_l2_grid_0.3125": fitted[name]["rel_grid"],
        }
    for m, (n_r, n_zh) in chosen_alloc.items():
        print(f"  m={m:2d}: n_R={n_r:2d}  n_z={2 * n_zh + 1:2d}  -> {(1 if m == 0 else 2) * n_r * (2 * n_zh + 1)} coeff")

    # ---- persist allocation + metrics ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "capture_target": args.capture,
        "captured_fraction_aggregate": chosen_info["captured_fraction"],
        "n_coeff_uniform": int(n_full),
        "n_coeff_compressed": int(n_chosen),
        "allocation": {str(m): [int(n_r), int(n_zh)] for m, (n_r, n_zh) in chosen_alloc.items()},
        "knot_params": {"n_phi": 64, "n_r_full": 14, "n_z_half_full": 6,
                        "r_max": 15.0, "z_max": 4.0, "r_min": 0.12, "z_min": 0.12},
        "aggregate_power_fraction": {int(m): float(agg_info["power_fraction"][m]) for m in EVEN_M},
        "results": results,
        "sweep": [{"capture": r["capture"], "n_coeff": r["n_coeff"],
                   "rel_l2": {n: r["rel"][n] for n in fitted}} for r in sweep_rows],
    }
    alloc_path = args.output_dir / "fourier_rz_allocation.json"
    alloc_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {alloc_path}")

    # ---- figure: power fraction, capture-sweep knee, chosen rel-L2 ----
    names = list(fitted)
    colors = {names[0]: "#4c72b0", names[1]: "#dd8452", names[2]: "#55a868"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)

    ax[0].bar([f"m={m}" for m in EVEN_M], [agg_info["power_fraction"][m] for m in EVEN_M], color="#777777")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("aggregate volume-weighted power fraction")
    ax[0].set_title("Azimuthal power per harmonic")

    caps = [r["capture"] for r in sweep_rows]
    ncs = [r["n_coeff"] for r in sweep_rows]
    ax[1].plot(ncs, [np.mean([r["rel"][n] for n in names]) for r in sweep_rows], "o-", color="k", label="mean")
    for name in names:
        ax[1].plot(ncs, [r["rel"][name] for r in sweep_rows], "o-", ms=4, color=colors[name], alpha=0.7, label=name)
    ax[1].axhline(np.mean([fitted[n]["rel_grid"] for n in names]), ls="--", color="r", label="grid (mean)")
    ax[1].axvline(n_chosen, ls=":", color="g")
    for nc, cap in zip(ncs, caps, strict=True):
        ax[1].annotate(f"{cap:g}", (nc, ax[1].get_ylim()[0]), fontsize=7, ha="center")
    ax[1].set_xlabel("coefficient count")
    ax[1].set_ylabel("3D rel-L2 vs SPH-KDE truth")
    ax[1].set_title("Capture sweep: coeff count vs fidelity")
    ax[1].legend(fontsize=7)

    width = 0.25
    for i, name in enumerate(names):
        vals = [results[name]["rel_l2_full"], results[name]["rel_l2_compressed"], results[name]["rel_l2_grid_0.3125"]]
        ax[2].bar(np.arange(3) + (i - 1) * width, vals, width, label=name, color=colors[name])
    ax[2].set_xticks(np.arange(3))
    ax[2].set_xticklabels([f"full\n({n_full})", f"compressed\n({n_chosen})", "grid\n(49152)"])
    ax[2].set_ylabel("3D rel-L2 vs SPH-KDE truth")
    ax[2].set_title(f"Chosen (capture {args.capture}) preserves fidelity")
    ax[2].legend(fontsize=8)

    fig.suptitle("Power-weighted (R,z) compression of the even-m Fourier x (R,z) deprojection target", fontsize=13)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "figures" / "fourier_rz_compression.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
