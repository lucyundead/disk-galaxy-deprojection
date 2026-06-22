"""Foundation for a structured residual basis on the Shen2010 X (no retraining).

Builds a finer-z truth grid for t800, the per-geometry geometric baselines, and
the residual density field delta_rho = rho_true - rho_baseline, then represents
the residual in a STRUCTURED basis:

    delta_rho(R, phi, z) = sum_{m in 0,2,4,6,8,10} [ low-rank a_m(R, z) ] cos/sin(m phi)

i.e. (1) azimuthal Fourier keeping only even, low-order harmonics (drops odd-m
lopsidedness and high-m local noise, enforces bar symmetry), then (2) a per-m
low-rank (R,z) compression (hybrid PCA-per-m: shared modes fit across the 9
projections; per-map SVD reported as the ensemble-free capacity).

It then validates that this basis RECONSTRUCTS THE X as a function of the
coefficient budget, on a grid fine enough (0.125 kpc) to resolve the off-plane
arms. Foundation only: it does not retrain the MDN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from build_tng50_baseline_density_grid import baseline_density_grid_from_image
from dgdp.density3d import (
    build_cylindrical_density_grid,
    cylindrical_bin_volumes,
    make_cylindrical_grid_spec,
)
from dgdp.orientation import align_particles_to_disk_bar_frame
from dgdp.projection import project_to_mock_image
from dgdp.types import Geometry, ParticleSet

EVEN_M = (0, 2, 4, 6, 8, 10)


def edgeon_from_cyl(density, r_edges, phi_edges, z_edges, *, radius=6.0, voxel=0.08, slab=0.6):
    xy = np.arange(-radius + 0.5 * voxel, radius, voxel)
    zz = np.arange(-z_edges[-1] + 0.5 * voxel, z_edges[-1], voxel)
    xg, yg = np.meshgrid(xy, xy, indexing="ij")
    rr = np.hypot(xg, yg)
    pp = np.arctan2(yg, xg)
    ir = np.clip(np.searchsorted(r_edges, rr) - 1, 0, len(r_edges) - 2)
    ip = np.clip(np.searchsorted(phi_edges, pp) - 1, 0, len(phi_edges) - 2)
    valid = rr < r_edges[-1]
    keep = np.abs(xy) <= slab
    panel = np.zeros((len(xy), len(zz)))
    for k, zv in enumerate(zz):
        iz = int(np.searchsorted(z_edges, zv) - 1)
        if 0 <= iz < len(z_edges) - 1:
            cell = np.where(valid, density[ir, ip, iz], 0.0)
            panel[:, k] = cell[:, keep].sum(axis=1)
    return xy, zz, panel


def rms_z_profile(panel, zcen, z_cap_kpc=2.5):
    """Vertical RMS height vs position along the bar, within |z|<z_cap so grids
    of different z-extent compare fairly. Overall thickness, not X-shape."""
    out = np.zeros(panel.shape[0])
    m = np.abs(zcen) <= z_cap_kpc
    zc = zcen[m]
    for xi in range(panel.shape[0]):
        w = np.clip(panel[xi, m], 0.0, None)
        s = float(w.sum())
        if s > 0:
            out[xi] = np.sqrt(np.sum(w * zc**2) / s)
    return 0.5 * (out + out[::-1])


def keep_even_azimuth(coeff, n_phi, ranks=None):
    """Zero odd / high m; optionally low-rank-compress each kept m's (R,z) map."""
    out = np.zeros_like(coeff)
    for m in EVEN_M:
        cm = coeff[:, m, :]
        if ranks is None:
            out[:, m, :] = cm
        else:
            k = ranks[m] if isinstance(ranks, dict) else int(ranks)
            u, s, vh = np.linalg.svd(cm, full_matrices=False)
            out[:, m, :] = (u[:, :k] * s[:k]) @ vh[:k, :]
    return np.fft.irfft(out, n=n_phi, axis=1)


def rz_rank_for_energy(cm, energy=0.99):
    s = np.linalg.svd(cm, compute_uv=False)
    c = np.cumsum(s**2) / max(np.sum(s**2), 1e-30)
    return int(np.searchsorted(c, energy) + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=Path("outputs/nbody_shen2010/cache"))
    ap.add_argument("--prod-grids", type=Path, default=Path("outputs/nbody_shen2010/nbody_shen2010_grids.npz"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/nbody_shen2010"))
    ap.add_argument("--total-mass-msun", type=float, default=4.5e10)
    ap.add_argument("--z-max-kpc", type=float, default=4.0)
    ap.add_argument("--n-z", type=int, default=64)              # 0.125 kpc
    ap.add_argument("--inclinations", type=float, nargs="+", default=[20.0, 40.0, 60.0])
    ap.add_argument("--bar-angles", type=float, nargs="+", default=[20.0, 40.0, 60.0])
    ap.add_argument("--rep-inc", type=float, default=40.0)
    ap.add_argument("--rep-bar", type=float, default=40.0)
    ap.add_argument("--pixel-scale-kpc", type=float, default=0.35)
    ap.add_argument("--image-size", type=int, default=192)
    ap.add_argument("--vertical-scale-height-kpc", type=float, default=0.4)
    args = ap.parse_args()
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- align + fine truth grid ------------------------------------------
    pos = np.load(args.cache_dir / "positions_raw.npy").astype(np.float64)
    vel_path = args.cache_dir / "velocities_raw.npy"
    vel = np.load(vel_path).astype(np.float64) if vel_path.exists() else None
    pos -= pos.mean(axis=0)
    n = pos.shape[0]
    masses = np.full(n, args.total_mass_msun / n)
    aligned = align_particles_to_disk_bar_frame(
        ParticleSet(positions_kpc=pos, masses_msun=masses, velocities_kms=vel),
        normal_radius_kpc=10.0, bar_radius_kpc=5.0,
    )
    spec = make_cylindrical_grid_spec(z_max_kpc=args.z_max_kpc, n_z=args.n_z)
    vol = cylindrical_bin_volumes(spec)
    truth = build_cylindrical_density_grid(aligned.particles, spec)
    truth_dens = truth.mass_msun / vol
    n_phi = len(spec.phi_edges_rad) - 1
    print(f"fine grid: n_R={len(spec.r_edges_kpc)-1} n_phi={n_phi} n_z={args.n_z} "
          f"dz={2*args.z_max_kpc/args.n_z:.3f} kpc")

    # ---- per-geometry residual density fields -----------------------------
    geoms = [(i, b) for i in args.inclinations for b in args.bar_angles]
    residuals = []
    baselines = []
    rep = 0
    for idx, (inc, bar) in enumerate(geoms):
        geom = Geometry(inclination_deg=inc, disk_pa_deg=0.0, bar_angle_deg=bar)
        image = project_to_mock_image(
            aligned.particles, geom, image_size=args.image_size,
            pixel_scale_kpc=args.pixel_scale_kpc, psf_sigma_pixels=0.0,
            noise_sigma_fraction=0.0, seed=0,
        ).image
        base = baseline_density_grid_from_image(
            image.astype(float), geometry=geom, pixel_scale_kpc=args.pixel_scale_kpc,
            spec=spec, vertical_scale_height_kpc=args.vertical_scale_height_kpc,
        )
        base_dens = base.mass_msun / vol
        baselines.append(base_dens)
        residuals.append(truth_dens - base_dens)
        if np.isclose(inc, args.rep_inc) and np.isclose(bar, args.rep_bar):
            rep = idx
    residuals = np.array(residuals)

    # ---- structured basis on the representative residual ------------------
    dres = residuals[rep]
    base_rep = baselines[rep]
    coeff = np.fft.rfft(dres, axis=1)

    # azimuthal power (residual) and the even-m<=10 truncation (full R-z rank)
    w = vol[:, 0, :]
    power = np.array([float(np.sum(np.abs(coeff[:, m, :]) ** 2 * w)) for m in range(coeff.shape[1])])
    power = power / power.max()
    even_full = keep_even_azimuth(coeff, n_phi, ranks=None)

    def l2(a):
        return float(np.sqrt(np.sum((a - dres) ** 2 * vol) / np.sum(dres ** 2 * vol)))

    # per-m (R,z) capacity: per-map SVD rank for 95/99% energy
    ranks_95 = {m: rz_rank_for_energy(coeff[:, m, :], 0.95) for m in EVEN_M}
    ranks_99 = {m: rz_rank_for_energy(coeff[:, m, :], 0.99) for m in EVEN_M}

    # hybrid PCA-per-m: shared (R,z) modes fit across the 9 geometries
    coeffs_all = np.fft.rfft(residuals, axis=2)  # (G, R, K, z)
    pca_var = {}
    for m in EVEN_M:
        maps = coeffs_all[:, :, m, :].reshape(len(geoms), -1)
        feats = np.concatenate([maps.real, maps.imag], axis=1)
        feats = feats - feats.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(feats, compute_uv=False)
        pca_var[m] = (np.cumsum(sv**2) / max(np.sum(sv**2), 1e-30)).tolist()

    # X-reconstruction truncation study: sweep per-map rank k (same k all m)
    rc = 0.5 * (spec.r_edges_kpc[:-1] + spec.r_edges_kpc[1:])
    zc3 = 0.5 * (spec.z_edges_kpc[:-1] + spec.z_edges_kpc[1:])
    bar_w = vol * ((rc[:, None, None] < 6.0) & (np.abs(zc3)[None, None, :] < 2.5))

    def bar_region_relerr(field3d):
        num = float(np.sum((field3d - truth_dens) ** 2 * bar_w))
        den = float(np.sum(truth_dens**2 * bar_w))
        return float(np.sqrt(num / den)) if den > 0 else 0.0

    xy, zz, panel_truth = edgeon_from_cyl(truth_dens, spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc)
    sweep = []
    recon_panels = {}
    for k in (1, 2, 4, 6, 8, 12, 16):
        dres_k = keep_even_azimuth(coeff, n_phi, ranks=k)
        truth_k = np.clip(base_rep + dres_k, 0.0, None)  # positivity, as the pipeline does
        _, _, panel_k = edgeon_from_cyl(truth_k, spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc)
        sweep.append({
            "rz_rank_per_m": k,
            "n_real_coeffs": int(k * len(EVEN_M) * 2),
            "residual_l2_error": l2(dres_k),
            "bar_region_truth_relerr": bar_region_relerr(truth_k),
        })
        recon_panels[k] = panel_k
    even_full_l2 = l2(even_full)
    base_relerr = bar_region_relerr(base_rep)

    # production-grid truth (coarse) for the contour comparison
    g = np.load(args.prod_grids)
    pvol = (
        0.5 * (g["r_edges_kpc"][1:] ** 2 - g["r_edges_kpc"][:-1] ** 2)[:, None, None]
        * np.diff(g["phi_edges_rad"])[None, :, None]
        * np.diff(g["z_edges_kpc"])[None, None, :]
    )
    pdens = g["truth_mass"] / pvol
    px, pz, panel_prod = edgeon_from_cyl(pdens, g["r_edges_kpc"], g["phi_edges_rad"], g["z_edges_kpc"])
    _, _, panel_base = edgeon_from_cyl(base_rep, spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc)
    _, _, panel_even_full = edgeon_from_cyl(
        np.clip(base_rep + even_full, 0.0, None),
        spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc,
    )
    even_full_bar_relerr = bar_region_relerr(np.clip(base_rep + even_full, 0.0, None))

    print(f"residual azimuthal odd-m power fraction: "
          f"{np.sum(power[1:13:2]) / np.sum(power[:13]):.4f}")
    print(f"even-m<=10 (full R-z rank) residual L2 error: {even_full_l2:.4f} "
          f"(captures {100 * (1 - even_full_l2**2):.0f}% of residual variance)")
    print(f"per-m R-z rank for 99% energy: {ranks_99}")
    print(f"bar-region truth recon rel error: baseline={base_relerr:.3f}, "
          f"even-m<=10 full rank={even_full_bar_relerr:.3f} -> structured (by k):")
    for row in sweep:
        print(f"  k={row['rz_rank_per_m']:2d}  coeffs={row['n_real_coeffs']:3d}  "
              f"residual_L2={row['residual_l2_error']:.3f}  "
              f"bar_region_relerr={row['bar_region_truth_relerr']:.3f}")

    # ---- save -------------------------------------------------------------
    np.savez_compressed(
        args.output_dir / "nbody_shen2010_fine_truth_grid.npz",
        truth_mass=truth.mass_msun.astype(np.float32), truth_density=truth_dens.astype(np.float32),
        r_edges_kpc=spec.r_edges_kpc, phi_edges_rad=spec.phi_edges_rad, z_edges_kpc=spec.z_edges_kpc,
    )
    metrics = {
        "fine_grid": {"n_R": len(spec.r_edges_kpc) - 1, "n_phi": n_phi, "n_z": args.n_z,
                      "dz_kpc": 2 * args.z_max_kpc / args.n_z, "z_max_kpc": args.z_max_kpc},
        "representative_geometry": {"inclination_deg": geoms[rep][0], "bar_angle_deg": geoms[rep][1]},
        "residual_odd_m_power_fraction": float(np.sum(power[1:13:2]) / np.sum(power[:13])),
        "even_m_le10_full_rank_residual_l2": even_full_l2,
        "rz_rank_per_m_95pct": ranks_95,
        "rz_rank_per_m_99pct": ranks_99,
        "pca_per_m_cumulative_variance_over_9_geoms": pca_var,
        "baseline_bar_region_relerr": base_relerr,
        "even_m_le10_full_rank_bar_region_relerr": even_full_bar_relerr,
        "truncation_sweep": sweep,
    }
    (args.output_dir / "nbody_structured_basis_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)

    def show(a, panel, xx, zzz, title):
        vm = float(panel.max())
        a.imshow(panel.T, origin="lower", extent=[xx[0], xx[-1], zzz[0], zzz[-1]], cmap="magma",
                 norm=LogNorm(vmin=vm * 5e-3, vmax=vm), aspect="auto")
        a.contour(xx, zzz, gaussian_filter(panel, 1.0).T,
                  levels=vm * np.array([0.04, 0.1, 0.25, 0.5]), colors="cyan", linewidths=0.8)
        a.set_title(title)
        a.set_xlabel("x [kpc] (along bar)")
        a.set_ylabel("z [kpc]")
        a.set_ylim(-2.5, 2.5)

    show(ax[0, 0], panel_truth, xy, zz, "fine truth 0.125 kpc (strong X)")
    show(ax[0, 1], recon_panels[4], xy, zz, "structured: even-m<=10, k=4 R-z modes")
    show(ax[0, 2], recon_panels[12], xy, zz, "structured: even-m<=10, k=12 R-z modes")

    ax[1, 0].semilogy(range(13), power[:13], "o-", color="0.3")
    ax[1, 0].semilogy(range(0, 13, 2), power[0:13:2], "o", color="#d62728", label="even m")
    ax[1, 0].semilogy(range(1, 13, 2), power[1:13:2], "s", color="#1f77b4", label="odd m")
    ax[1, 0].axvline(10.5, color="k", ls=":", lw=1)
    ax[1, 0].set_xlabel("azimuthal order m")
    ax[1, 0].set_ylabel("residual power (norm.)")
    ax[1, 0].set_title("residual azimuthal spectrum")
    ax[1, 0].legend()

    ks = [r["n_real_coeffs"] for r in sweep]
    ax[1, 1].plot(ks, [r["residual_l2_error"] for r in sweep], "o-", color="k", label="residual L2 (whole grid)")
    ax[1, 1].plot(ks, [r["bar_region_truth_relerr"] for r in sweep], "s-", color="#d62728",
                  label="truth recon error (bar region)")
    ax[1, 1].axhline(base_relerr, color="0.5", ls="--", label="baseline (no residual)")
    ax[1, 1].set_xlabel("structured coefficients (per galaxy)")
    ax[1, 1].set_ylabel("relative error")
    ax[1, 1].set_title("dimension vs fidelity")
    ax[1, 1].legend(fontsize=8)

    for panel, zzz, lab, col in (
        (panel_truth, zz, "fine truth", "k"),
        (panel_base, zz, "baseline (sech^2)", "0.5"),
        (recon_panels[4], zz, "structured k=4", "#d62728"),
        (panel_even_full, zz, "structured even-m<=10 (full rank)", "#ff7f0e"),
        (panel_prod, pz, "production 0.625 kpc", "#1f77b4"),
    ):
        xx = xy if zzz is zz else px
        ax[1, 2].plot(xx, rms_z_profile(panel, zzz), "-", color=col, lw=2, label=lab)
    ax[1, 2].set_xlim(-6, 6)
    ax[1, 2].set_xlabel("x [kpc] (along bar)")
    ax[1, 2].set_ylabel("vertical RMS height [kpc] (|z|<2.5, clipped)")
    ax[1, 2].set_title("RMS z(x): overall thickness (not X-shape)")
    ax[1, 2].legend(fontsize=7)

    fig.suptitle("Shen2010 t800: structured residual basis (even-m<=10 x PCA-per-m) preserves the X",
                 fontsize=14)
    fig.savefig(fig_dir / "nbody_structured_basis.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_dir / 'nbody_structured_basis.png'} and metrics/grid to {args.output_dir}")


if __name__ == "__main__":
    main()
