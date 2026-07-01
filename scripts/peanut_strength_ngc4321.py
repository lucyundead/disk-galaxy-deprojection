"""Apply the validated b4/boxiness peanut metric (scripts/peanut_strength.py) to the
NGC 4321 deprojection: the geometric (sech^2) baseline and the TNG-learned density,
measured bar side-on, alongside the ground-truth anchors (Shen2010 strong peanut, thin
exponential disk, tilted ellipsoid) on the same scale.

Convention: strength = median m=4 boxiness in the bulge region, tilt-invariant.
  <0 disky, ~0 ellipse, >0 boxy/peanut.

Run:
    .venv/bin/python scripts/peanut_strength_ngc4321.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from peanut_strength import (
    _grid_from_particles,
    _region_for,
    _shen_aligned_positions,
    peanut_strength,
)


def bar_azimuth(baseline_mass, r_grid):
    """Grid azimuth (rad) of the bar from the m=2 phase of Sigma(R,phi) in the bar region."""
    sig = baseline_mass.sum(axis=2)
    sel = (r_grid > 2.0) & (r_grid < 5.5)
    c2 = np.fft.rfft(sig, axis=1)[:, 2][sel]
    return float(-np.angle(np.sum(c2)) / 2.0)


def side_on_sigma(density, r_e, p_e, z_e, theta_b, *, half_xy=8.0, y_int=4.0, vox=0.1):
    """Bar-side-on Sigma(x,z): integrate the cyl density over |y|<y_int with the bar on x
    (rotate the eval frame by +theta_b into the grid frame). theta_b=0 if already bar-on-x."""
    x = np.arange(-half_xy + 0.5 * vox, half_xy, vox)
    z = np.arange(-z_e[-1] + 0.5 * vox, z_e[-1], vox)
    y = np.arange(-y_int + 0.5 * vox, y_int, vox)
    nr, nphi, nz = len(r_e) - 1, len(p_e) - 1, len(z_e) - 1
    ct, st = np.cos(theta_b), np.sin(theta_b)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    xd, yd = xg * ct - yg * st, xg * st + yg * ct   # bar frame -> grid (disk) frame
    rg, pg = np.hypot(xd, yd), np.arctan2(yd, xd)
    ir = np.clip(np.searchsorted(r_e, rg) - 1, 0, nr - 1)
    ip = np.clip(np.searchsorted(p_e, pg) - 1, 0, nphi - 1)
    valid = rg < r_e[-1]
    sigma = np.zeros((len(x), len(z)))
    for k, zv in enumerate(z):
        iz = int(np.searchsorted(z_e, zv) - 1)
        if 0 <= iz < nz:
            sigma[:, k] = np.where(valid, density[ir, ip, iz], 0.0).sum(axis=1) * vox
    return x, z, sigma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, default=Path("outputs/real_images/ngc4321_arrays.npz"))
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--bar-len-kpc", type=float, default=4.75, help="NGC4321 deprojected bar semi-major")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/real_images"))
    args = ap.parse_args()

    spec = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32)
    r_e, p_e, z_e = spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc
    vol = cylindrical_bin_volumes(spec)
    d = np.load(args.arrays)
    theta_b = bar_azimuth(d["baseline_mass"], d["r_grid"])
    reg_ngc = _region_for(args.bar_len_kpc)

    cases = []  # (label, x, z, sigma, region)
    for lab, mass in (("NGC4321 baseline (sech^2)", d["baseline_mass"]),
                      ("NGC4321 learned (TNG q_m)", d["learned_mass"])):
        x, z, sig = side_on_sigma(mass / vol, r_e, p_e, z_e, theta_b)
        cases.append((lab, x, z, sig, reg_ngc))

    # ground-truth anchors (already bar-on-x after alignment): same grid spec, theta_b=0
    shen_pos = _shen_aligned_positions(args.cache_dir)
    dens_s, rs, ps, zs_e = _grid_from_particles(shen_pos, z_max=5.0, n_z=32)
    cases.append(("Shen2010 (strong peanut)", *side_on_sigma(dens_s, rs, ps, zs_e, 0.0), _region_for(4.4)))
    rng = np.random.default_rng(11)
    rr = -3.0 * np.log(1.0 - rng.random(700000))
    rr = rr[rr < 15.0]
    ph = rng.uniform(0, 2 * np.pi, rr.size)
    zz = 0.3 * np.arctanh(np.clip(2 * rng.random(rr.size) - 1, -0.999, 0.999))
    disk = np.column_stack((rr * np.cos(ph), rr * np.sin(ph), zz))
    dens_d, rd, pd, zd = _grid_from_particles(disk, z_max=5.0, n_z=32)
    cases.append(("thin exp disk", *side_on_sigma(dens_d, rd, pd, zd, 0.0), _region_for(3.0)))
    rng2 = np.random.default_rng(7)
    ell = np.column_stack((rng2.normal(0, 2.5, 600000), rng2.normal(0, 1.0, 600000), rng2.normal(0, 0.5, 600000)))
    dens_e, reo, peo, zeo = _grid_from_particles(ell, z_max=5.0, n_z=32)
    cases.append(("tilted ellipsoid", *side_on_sigma(dens_e, reo, peo, zeo, 0.0), _region_for(6.0)))

    print(f"NGC4321 bar azimuth in grid = {np.degrees(theta_b):.1f} deg (rotated to x for side-on)")
    print("\nb4 peanut strength (>0 boxy/peanut, ~0 ellipse, <0 disky):")
    print(f"  {'case':30s} {'median b4':>10} {'boxiest':>9} {'tilt':>6} {'q':>5}")
    results = {}
    for lab, x, z, sig, reg in cases:
        m = peanut_strength(x, z, sig, **reg)
        results[lab] = m
        print(f"  {lab:30s} {m['strength']:>+10.3f} {m['strength_boxiest']:>+9.3f} "
              f"{m['tilt_deg']:>+6.1f} {m['q']:>5.2f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, len(cases), figsize=(3.5 * len(cases), 4.0), constrained_layout=True)
    for ax, (lab, x, z, sig, _reg) in zip(axes, cases, strict=True):
        vmax = float(sig.max())
        ax.imshow(sig.T, origin="lower", extent=[x[0], x[-1], z[0], z[-1]], cmap="magma",
                  norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax), aspect="auto")
        ax.contour(x, z, sig.T, levels=vmax * np.array([0.04, 0.1, 0.25, 0.5]), colors="cyan", linewidths=0.7)
        ax.set_title(f"{lab}\nb4={results[lab]['strength']:+.3f}", fontsize=9)
        ax.set_xlabel("x [kpc] (bar)")
        ax.set_ylabel("z [kpc]")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-3, 3)
    fig.suptitle("b4/boxiness peanut metric (bar side-on): NGC 4321 baseline & learned vs anchors", fontsize=12)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / "ngc4321_peanut_strength.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
