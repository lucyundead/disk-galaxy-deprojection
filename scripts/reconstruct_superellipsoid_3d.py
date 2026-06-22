"""Reconstruct a smooth 3D density from superellipsoid shells and compare its 3D
fidelity to the cylindrical grid, against an SPH-KDE truth, for Shen2010 and a
strong-bar TNG galaxy (392276).

Pipeline per galaxy (all grid-free except the grid we compare against):
  1. align particles to the disk/bar frame, then to the bulge principal axes
     (absorbs the bar tilt, e.g. 392276's ~12 deg);
  2. adaptive SPH-KDE density (k=32) = the smooth truth, evaluable anywhere;
  3. fit nested superellipsoid shells (semi-axes pinned to iso-surface extents,
     squareness exponents fitted) and reconstruct rho(x) by interpolating the
     shell levels;
  4. evaluate truth / cyl-grid / superellipsoid on a fine 0.2 kpc grid and report
     3D relative-L2, with parameter counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from dgdp.density3d import build_cylindrical_density_grid, cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.orientation import align_particles_to_disk_bar_frame
from dgdp.types import ParticleSet
from peanut_strength import _shen_aligned_positions

EVEN_M = (0, 2, 4, 6, 8, 10)


def cubic_spline_w(q):
    w = np.zeros_like(q)
    a, b = q <= 0.5, (q > 0.5) & (q <= 1.0)
    w[a] = 1 - 6 * q[a] ** 2 + 6 * q[a] ** 3
    w[b] = 2 * (1 - q[b]) ** 3
    return 8.0 / np.pi * w


class SPH:
    def __init__(self, pos, mass, k=32):
        self.tree, self.mass, self.k = cKDTree(pos), mass, k

    def __call__(self, pts):
        d, idx = self.tree.query(pts, k=self.k, workers=-1)
        h = d[:, -1:]
        return (self.mass[idx] * cubic_spline_w(d / h) / h**3).sum(axis=1)


def bulge_align(pos, mass):
    """Rotate into the bulge (r<4 kpc) inertia principal frame; absorbs bar tilt."""
    r = np.linalg.norm(pos, axis=1)
    sel = r < 4.0
    pc, w = pos[sel], mass[sel]
    inertia = (pc * w[:, None]).T @ pc / w.sum()
    evals, evecs = np.linalg.eigh(inertia)
    R = evecs[:, np.argsort(evals)[::-1]]  # axis0 = longest (bar)
    return pos @ R


def get_particles(kind, cache, hdf5):
    if kind == "shen":
        pos = _shen_aligned_positions(cache)  # already disk/bar aligned
        mass = np.full(pos.shape[0], 4.5e10 / pos.shape[0])
    else:
        with h5py.File(hdf5, "r") as f:
            pos = f["PartType4/Coordinates"][:].astype(float)
            mass = f["PartType4/Masses"][:].astype(float)
            vel = f["PartType4/Velocities"][:].astype(float)
        pos = pos - np.median(pos, axis=0)
        keep = np.linalg.norm(pos, axis=1) < 50.0
        pos, mass, vel = pos[keep], mass[keep], vel[keep]
        al = align_particles_to_disk_bar_frame(
            ParticleSet(positions_kpc=pos, masses_msun=mass, velocities_kms=vel),
            normal_radius_kpc=10.0, bar_radius_kpc=5.0,
        )
        pos, mass = al.particles.positions_kpc, al.particles.masses_msun
    return bulge_align(pos, mass), mass


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)])


def fit_shells(rho, n_dir=400, n_shell=24):
    dirs = fib_sphere(n_dir)
    radii = np.geomspace(0.1, 15.0, 90)
    ray = rho((radii[None, :, None] * dirs[:, None, :]).reshape(-1, 3)).reshape(n_dir, len(radii))
    ref = np.median(ray[:, np.argmin(np.abs(radii - 0.5))])
    levels = ref * np.geomspace(3.0, 0.008, n_shell)
    shells = []
    for lev in levels:
        iso = np.full(n_dir, np.nan)
        for j in range(n_dir):
            above = np.where(ray[j] >= lev)[0]
            if above.size == 0 or above[-1] == len(radii) - 1:
                continue
            i1 = above[-1]
            r0, r1, p0, p1 = radii[i1], radii[i1 + 1], ray[j, i1], ray[j, i1 + 1]
            iso[j] = r0 + (lev - p0) * (r1 - r0) / (p1 - p0) if p1 != p0 else r0
        ok = np.isfinite(iso)
        if ok.sum() < 0.7 * n_dir:
            continue
        surf = iso[ok, None] * dirs[ok]
        abc = np.maximum(np.percentile(np.abs(surf), 98, axis=0), 1e-2)
        x, y, z = surf[:, 0], surf[:, 1], surf[:, 2]

        def resid(s, x=x, y=y, z=z, abc=abc):
            return ((x / abc[0]) ** 2) ** s[0] + ((y / abc[1]) ** 2) ** s[1] + ((z / abc[2]) ** 2) ** s[2] - 1.0

        s = least_squares(resid, [1.0, 1.0, 1.0], bounds=([0.2] * 3, [2.0] * 3), max_nfev=2000).x
        shells.append({"abc": abc, "s": s, "level": float(lev)})
    return shells


def reconstruct(pts, shells):
    g = np.empty((pts.shape[0], len(shells)), dtype=np.float64)
    for k, sh in enumerate(shells):
        a, b, c = sh["abc"]
        sa, sb, sc = sh["s"]
        g[:, k] = ((pts[:, 0] / a) ** 2) ** sa + ((pts[:, 1] / b) ** 2) ** sb + ((pts[:, 2] / c) ** 2) ** sc
    gm = np.minimum.accumulate(g, axis=1)               # enforce monotone decrease with shell
    levels = np.array([sh["level"] for sh in shells])
    logL = np.log(levels)
    out = np.zeros(pts.shape[0])
    inside_all = gm[:, 0] < 1.0                          # inside innermost shell
    out[inside_all] = levels[0]
    k1 = np.argmax(gm < 1.0, axis=1)                     # first shell the point is inside
    has = (gm < 1.0).any(axis=1) & ~inside_all
    k1h = k1[has]
    g0, g1 = gm[has, k1h - 1], gm[has, k1h]
    frac = np.clip((g0 - 1.0) / np.maximum(g0 - g1, 1e-12), 0.0, 1.0)
    out[has] = np.exp(logL[k1h - 1] + frac * (logL[k1h] - logL[k1h - 1]))
    return out


def grid_density_at(pts, pos, mass, *, z_max, n_z):
    spec = make_cylindrical_grid_spec(z_max_kpc=z_max, n_z=n_z)
    grid = build_cylindrical_density_grid(ParticleSet(positions_kpc=pos, masses_msun=mass), spec)
    vol = cylindrical_bin_volumes(spec)
    dens = grid.mass_msun / vol
    re, pe, ze = spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc
    rr = np.hypot(pts[:, 0], pts[:, 1])
    pp = np.arctan2(pts[:, 1], pts[:, 0])
    ir = np.clip(np.searchsorted(re, rr) - 1, 0, len(re) - 2)
    ip = np.clip(np.searchsorted(pe, pp) - 1, 0, len(pe) - 2)
    iz = np.searchsorted(ze, pts[:, 2]) - 1
    valid = (rr < re[-1]) & (iz >= 0) & (iz < len(ze) - 1)
    iz = np.clip(iz, 0, len(ze) - 2)
    return np.where(valid, dens[ir, ip, iz], 0.0)


def rel_l2(model, truth, mask):
    return float(np.linalg.norm(model[mask] - truth[mask]) / np.linalg.norm(truth[mask]))


def edgeon(rho_fn, half=6.0, zmax=3.0, vox=0.05):
    xx = np.arange(-half, half + vox, vox)
    zz = np.arange(-zmax, zmax + vox, vox)
    XX, ZZ = np.meshgrid(xx, zz, indexing="ij")
    return xx, zz, rho_fn(np.column_stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()])).reshape(XX.shape)


def fourier_rz_fit(rho, *, n_phi=64, nR=14, nz_half=6, r_max=15.0, z_max=4.0):
    """Grid-free even-m Fourier x smooth-(R,z) model: at control (R,z) knots
    (log R, dense-near-0 z) take the azimuthal Fourier transform of the SPH-KDE
    density and keep even m<=10. NOT a stratified/concentric model and NOT a
    disk/bulge split: a_m(R,z) is a free 2D map per harmonic, so a flat disk and a
    rounder bulge coexist natively. Reconstruction interpolates a_m(R,z) smoothly."""
    R = np.geomspace(0.12, r_max, nR)
    zp = np.geomspace(0.12, z_max, nz_half)
    z = np.concatenate([-zp[::-1], [0.0], zp])  # 2*nz_half+1 knots, dense near plane
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    RR, PP, ZZ = np.meshgrid(R, phi, z, indexing="ij")
    pts = np.column_stack([(RR * np.cos(PP)).ravel(), (RR * np.sin(PP)).ravel(), ZZ.ravel()])
    dens = rho(pts).reshape(len(R), n_phi, len(z))
    coeff = np.fft.rfft(dens, axis=1) / n_phi
    maps = {m: coeff[:, m, :] for m in EVEN_M}
    n_coeff = int(sum((1 if m == 0 else 2) for m in EVEN_M) * len(R) * len(z))
    return {"R": R, "z": z, "maps": maps, "n_coeff": n_coeff}


def fourier_rz_reconstruct(pts, model):
    R = np.hypot(pts[:, 0], pts[:, 1])
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    q = np.column_stack([np.clip(R, model["R"][0], model["R"][-1]),
                         np.clip(pts[:, 2], model["z"][0], model["z"][-1])])
    out = np.zeros(pts.shape[0])
    for m in EVEN_M:
        cm = model["maps"][m]
        re = RegularGridInterpolator((model["R"], model["z"]), cm.real, bounds_error=False, fill_value=0.0)(q)
        if m == 0:
            out += re
        else:
            im = RegularGridInterpolator((model["R"], model["z"]), cm.imag, bounds_error=False, fill_value=0.0)(q)
            out += 2.0 * (re * np.cos(m * phi) - im * np.sin(m * phi))
    return np.clip(out, 0.0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--tng-particle-dir", type=Path, default=Path("outputs/tng50_milestone2c/particles"))
    ap.add_argument("--tng-ids", type=int, nargs="+", default=[554189, 392276])
    ap.add_argument("--output", type=Path, default=Path("outputs/nbody_shen2010/figures/superellipsoid_recon_3d.png"))
    args = ap.parse_args()

    cases = [("Shen2010", "shen", None)]
    cases += [(f"TNG {i}", "tng", args.tng_particle_dir / f"subhalo_{i}.hdf5") for i in args.tng_ids]
    fig, axes = plt.subplots(len(cases), 4, figsize=(20, 4.6 * len(cases)), constrained_layout=True)
    summary = {}
    for row, (name, kind, hdf5) in enumerate(cases):
        pos, mass = get_particles(kind, args.cache_dir, hdf5)
        rho = SPH(pos, mass, k=32)
        shells = fit_shells(rho)
        n_par = len(shells) * 6
        print(f"\n{name}: {pos.shape[0]} particles, {len(shells)} shells ({n_par} params)")

        # fine 3D evaluation grid
        gx = np.arange(-10, 10.01, 0.2)
        gz = np.arange(-3.5, 3.51, 0.2)
        XX, YY, ZZ = np.meshgrid(gx, gx, gz, indexing="ij")
        pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
        truth = rho(pts)
        supe = reconstruct(pts, shells)
        grid625 = grid_density_at(pts, pos, mass, z_max=10.0, n_z=32)   # production 0.625 kpc
        grid312 = grid_density_at(pts, pos, mass, z_max=5.0, n_z=32)    # fine 0.3125 kpc
        fmodel = fourier_rz_fit(rho)
        fourier = fourier_rz_reconstruct(pts, fmodel)
        n_four = fmodel["n_coeff"]
        mask = truth > 1e-3 * truth.max()
        r_supe = rel_l2(supe, truth, mask)
        r_four = rel_l2(fourier, truth, mask)
        r_g625 = rel_l2(grid625, truth, mask)
        r_g312 = rel_l2(grid312, truth, mask)
        bar = mask & (np.hypot(pts[:, 0], pts[:, 1]) < 6.0)
        summary[name] = {
            "n_shell_params": n_par, "n_fourier_coeff": n_four, "n_grid_cells": 32 * 48 * 32,
            "rel_l2_superellipsoid": r_supe, "rel_l2_fourier_rz": r_four,
            "rel_l2_grid_0.625": r_g625, "rel_l2_grid_0.3125": r_g312,
            "rel_l2_superellipsoid_bar": rel_l2(supe, truth, bar),
            "rel_l2_fourier_rz_bar": rel_l2(fourier, truth, bar),
            "rel_l2_grid_0.3125_bar": rel_l2(grid312, truth, bar),
        }
        print("  3D rel-L2 vs SPH-KDE truth (truth>1e-3 max):")
        print(f"    superellipsoid  ({n_par} params): {r_supe:.3f}  [bar {summary[name]['rel_l2_superellipsoid_bar']:.3f}]")
        print(f"    Fourier x (R,z) ({n_four} coeff): {r_four:.3f}  [bar {summary[name]['rel_l2_fourier_rz_bar']:.3f}]")
        print(f"    grid 0.3125 kpc ({32*48*32} cells): {r_g312:.3f}  [bar {summary[name]['rel_l2_grid_0.3125_bar']:.3f}]")
        print(f"    grid 0.625  kpc ({32*48*32} cells): {r_g625:.3f}")

        # edge-on panels
        xx, zz, t_map = edgeon(rho)
        _, _, s_map = edgeon(lambda p, sh=shells: reconstruct(p, sh))
        _, _, g_map = edgeon(lambda p, ps=pos, ms=mass: grid_density_at(p, ps, ms, z_max=5.0, n_z=32))
        _, _, f_map = edgeon(lambda p, fm=fmodel: fourier_rz_reconstruct(p, fm))
        for ax, m, ttl in (
            (axes[row, 0], t_map, f"{name}: SPH-KDE truth (mesh-free)"),
            (axes[row, 1], g_map, f"grid 0.3125 kpc (rel-L2 {r_g312:.2f})"),
            (axes[row, 2], s_map, f"superellipsoid {n_par} par (rel-L2 {r_supe:.2f})"),
            (axes[row, 3], f_map, f"Fourier x (R,z) {n_four} coeff (rel-L2 {r_four:.2f})"),
        ):
            vmax = float(m.max())
            ax.imshow(m.T, origin="lower", extent=[-6, 6, -3, 3], cmap="magma",
                      norm=LogNorm(vmin=vmax * 3e-3, vmax=vmax), aspect="auto")
            ax.contour(xx, zz, m.T, levels=vmax * np.array([0.03, 0.06, 0.12, 0.25, 0.5]),
                       colors="cyan", linewidths=0.7)
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel("x [kpc] (along bar)")
            ax.set_ylabel("z [kpc]")
    fig.suptitle("3D reconstruction: grid vs superellipsoid vs even-m Fourier x (R,z), vs SPH-KDE truth", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
