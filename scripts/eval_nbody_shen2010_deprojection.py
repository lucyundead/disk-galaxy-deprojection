"""Out-of-distribution test: deproject the Shen2010 MW N-body bar with the
TNG50-trained MDN.

Takes the Shen et al. 2010 boxy/peanut N-body snapshot (positions already in
kpc), aligns it to the disk/bar frame, builds the true cylindrical density grid,
then for each requested (inclination, bar-angle) projection:

  1. renders the clean mock image (192x192, 0.35 kpc/pix) the model expects;
  2. builds the geometric thin-disk + sech^2 baseline grid from that image;
  3. runs the *adopted* 2c MDN forward pass (PCA-coefficient residual) end to end
     with no retraining;
  4. compares truth / baseline / MDN in 3D: cell-mass MAE, total mass, and the
     physical summaries, plus the bar-end vertical profile (the boxy/peanut/X
     signature this N-body model has and TNG50 lacks).

This reuses the validated forward pass from scan_inclination_sensitivity.py and
the X diagnostic from analyze_mdn_x_recovery.py. The correction heads are
excluded: this measures the core image->3D model on a genuinely out-of-sample,
strongly buckled bar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from analyze_mdn_x_recovery import _bar_end_profile, _edgeon
from build_tng50_baseline_density_grid import baseline_density_grid_from_image
from dgdp.density3d import (
    build_cylindrical_density_grid,
    cylindrical_bin_volumes,
    make_cylindrical_grid_spec,
)
from dgdp.models.mdn import SummaryResidualMDN
from dgdp.orientation import align_particles_to_disk_bar_frame
from dgdp.projection import project_to_mock_image
from dgdp.types import Geometry, ParticleSet
from report_milestone2b_physical_summaries import (
    azimuthal_m2_profiles,
    bar_axis_mass_fraction,
    central_mass_fraction,
    radial_mass_profiles,
    vertical_mass_profiles,
    vertical_rms_height_kpc,
)
from train_density_residual_pca import (
    make_density_residual_features,
    reconstruct_delta_mass_from_coefficients,
)
from train_density_residual_pca_mdn import corrected_mass_from_delta

DEFAULT_PCA = Path(
    "outputs/tng50_milestone2c_clean3d/density_residual_diagnostics/density_residual_pca.npz"
)
DEFAULT_RUN_DIR = Path(
    "outputs/tng50_milestone2c_clean3d/density_residual_pca_mdn_sweep_central/"
    "components_1_seed_20260608"
)
DEFAULT_CACHE = Path("outputs/nbody_shen2010/cache")
DEFAULT_OUTPUT = Path("outputs/nbody_shen2010")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--pca", type=Path, default=DEFAULT_PCA)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--total-mass-msun", type=float, default=4.5e10)
    p.add_argument("--length-scale", type=float, default=1.0, help="kpc per raw unit")
    p.add_argument("--inclinations", type=float, nargs="+", default=[20.0, 40.0, 60.0])
    p.add_argument("--bar-angles", type=float, nargs="+", default=[20.0, 40.0, 60.0])
    p.add_argument("--disk-pa-deg", type=float, default=0.0)
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--pixel-scale-kpc", type=float, default=0.35)
    p.add_argument("--vertical-scale-height-kpc", type=float, default=0.4)
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--sample-batch-size", type=int, default=16)
    p.add_argument("--central-radius-kpc", type=float, default=2.0)
    p.add_argument("--bar-half-angle-deg", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=20260613)
    p.add_argument("--rep-inclination", type=float, default=40.0)
    p.add_argument("--rep-bar-angle", type=float, default=40.0)
    return p.parse_args()


def estimate_bar_length_kpc(positions: np.ndarray, *, r_max_kpc: float = 12.0) -> float:
    """Bar length = radius where the m=2 amplitude drops below half its peak."""
    x, y = positions[:, 0], positions[:, 1]
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    edges = np.linspace(0.0, r_max_kpc, 49)
    rc = 0.5 * (edges[:-1] + edges[1:])
    a2 = np.zeros(len(rc))
    for i in range(len(rc)):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.sum() >= 50:
            a2[i] = np.abs(np.mean(np.exp(2j * phi[m])))
    peak = int(np.argmax(a2))
    half = 0.5 * a2[peak]
    for i in range(peak, len(rc)):
        if a2[i] < half:
            return float(rc[i])
    return float(rc[peak])


def bar_end_stats(mass, r_edges, phi_edges, z_edges, length) -> tuple[float, float, float]:
    """(midplane dip, off-plane peak |z|, RMS |z|) of the bar-end vertical profile."""
    zc, prof = _bar_end_profile(mass, r_edges, phi_edges, z_edges, length)
    total = float(prof.sum())
    if total <= 0.0:
        return 1.0, 0.0, 0.0
    norm = prof / prof.max()
    dip = float(norm[np.argmin(np.abs(zc))])
    peak_z = float(abs(zc[np.argmax(norm)]))
    rms_z = float(np.sqrt(np.sum(prof * zc**2) / total))
    return dip, peak_z, rms_z


def summaries_batch(mass, *, spec, central_radius_kpc, bar_half_angle_deg) -> dict:
    return {
        "radial_mass_profile": radial_mass_profiles(mass),
        "vertical_mass_profile": vertical_mass_profiles(mass),
        "vertical_rms_height": vertical_rms_height_kpc(mass, spec.z_edges_kpc),
        "central_mass_fraction": central_mass_fraction(
            mass, spec.r_edges_kpc, radius_kpc=central_radius_kpc
        ),
        "bar_axis_mass_fraction": bar_axis_mass_fraction(
            mass, spec.phi_edges_rad, half_angle_deg=bar_half_angle_deg
        ),
        "m2_profile": azimuthal_m2_profiles(mass, spec.phi_edges_rad),
    }


def m2_bar_amplitude(mass_row, spec, length) -> float:
    """Mass-weighted mean normalized m=2 amplitude over bar radii (R<=length)."""
    prof = azimuthal_m2_profiles(mass_row[None], spec.phi_edges_rad)[0]
    rc = 0.5 * (spec.r_edges_kpc[:-1] + spec.r_edges_kpc[1:])
    band = rc <= length
    radial = radial_mass_profiles(mass_row[None])[0]
    w = radial * band
    return float(np.sum(prof * w) / max(np.sum(w), 1.0))


def load_model(run_dir: Path):
    ck = torch.load(run_dir / "density_residual_pca_mdn.pt", map_location="cpu", weights_only=False)
    model = SummaryResidualMDN(
        input_dim=int(ck["input_dim"]),
        output_dim=int(ck["output_dim"]),
        hidden_dim=int(ck["hidden_dim"]),
        n_components=int(ck["n_components"]),
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with np.load(run_dir / "density_residual_pca_mdn_normalization.npz") as norm:
        normalization = {k: norm[k].astype(np.float32) for k in norm.files}
    return model, normalization, ck


def main() -> None:
    args = parse_args()
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng_seed = args.seed

    # ---- load + align particles -------------------------------------------
    positions = np.load(args.cache_dir / "positions_raw.npy").astype(np.float64)
    vel_path = args.cache_dir / "velocities_raw.npy"
    velocities = np.load(vel_path).astype(np.float64) if vel_path.exists() else None
    positions = (positions - positions.mean(axis=0)) * args.length_scale
    n_part = positions.shape[0]
    masses = np.full(n_part, args.total_mass_msun / n_part, dtype=np.float64)
    raw_particles = ParticleSet(
        positions_kpc=positions, masses_msun=masses, velocities_kms=velocities
    )
    aligned = align_particles_to_disk_bar_frame(
        raw_particles, normal_radius_kpc=10.0, bar_radius_kpc=5.0
    )
    apos = aligned.particles.positions_kpc
    bar_length = estimate_bar_length_kpc(apos)
    # alignment sanity: bar should now lie along +x (residual angle ~ 0)
    residual_bar_angle = float(
        np.rad2deg(
            0.5
            * np.angle(
                np.sum((apos[:, 0] + 1j * apos[:, 1]) ** 2 * (np.hypot(apos[:, 0], apos[:, 1]) <= 5.0))
            )
        )
    )
    align_info = {
        "disk_normal": [float(v) for v in aligned.disk_normal],
        "orientation_method": aligned.orientation_method,
        "bar_angle_removed_deg": float(aligned.bar_angle_deg),
        "residual_bar_angle_deg": residual_bar_angle,
        "bar_length_kpc": float(bar_length),
        "n_particles": int(n_part),
        "total_mass_msun": float(args.total_mass_msun),
    }
    print(f"alignment: {align_info}")

    # ---- truth grids -------------------------------------------------------
    spec = make_cylindrical_grid_spec()  # production: R 0.05-30, |z|<10, (32,48,32)
    volumes = cylindrical_bin_volumes(spec).astype(np.float32)
    truth_grid = build_cylindrical_density_grid(aligned.particles, spec)
    truth_mass = truth_grid.mass_msun.astype(np.float32)
    # finer-z truth (|z|<5, 0.3125 kpc) to characterise the intrinsic X strength
    spec_fine = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32)
    truth_grid_fine = build_cylindrical_density_grid(aligned.particles, spec_fine)
    truth_dip_prod = bar_end_stats(
        truth_mass, spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc, bar_length
    )
    truth_dip_fine = bar_end_stats(
        truth_grid_fine.mass_msun.astype(np.float32),
        spec_fine.r_edges_kpc, spec_fine.phi_edges_rad, spec_fine.z_edges_kpc, bar_length,
    )
    print(f"truth bar-end dip (production 0.625 kpc): {truth_dip_prod}")
    print(f"truth bar-end dip (fine 0.3125 kpc):      {truth_dip_fine}")
    print(f"truth grid mass fraction in grid: {truth_grid.mass_fraction_in_grid:.4f}")

    # ---- model -------------------------------------------------------------
    model, normalization, ck = load_model(args.run_dir)
    with np.load(args.pca) as pca:
        components = pca["components"].astype(np.float32)
        pca_mean = pca["mean"].astype(np.float32)
    image_feature_size = int(ck["image_feature_size"])
    central_pixel_scale_kpc = (
        float(ck["central_pixel_scale_kpc"]) if ck["central_pixel_scale_kpc"] is not None else None
    )
    grid_shape = tuple(int(s) for s in truth_mass.shape)

    # ---- geometries: images + baselines -----------------------------------
    geoms = [(i, b) for i in args.inclinations for b in args.bar_angles]
    images = np.zeros((len(geoms), args.image_size, args.image_size), dtype=np.float32)
    baseline_mass = np.zeros((len(geoms), *grid_shape), dtype=np.float32)
    metadata = np.zeros((len(geoms), 3), dtype=np.float32)
    for k, (inc, bar) in enumerate(geoms):
        geometry = Geometry(inclination_deg=inc, disk_pa_deg=args.disk_pa_deg, bar_angle_deg=bar)
        images[k] = project_to_mock_image(
            aligned.particles, geometry,
            image_size=args.image_size, pixel_scale_kpc=args.pixel_scale_kpc,
            psf_sigma_pixels=0.0, noise_sigma_fraction=0.0, seed=0,
        ).image.astype(np.float32)
        baseline_mass[k] = baseline_density_grid_from_image(
            images[k].astype(float), geometry=geometry,
            pixel_scale_kpc=args.pixel_scale_kpc, spec=spec,
            vertical_scale_height_kpc=args.vertical_scale_height_kpc,
        ).mass_msun.astype(np.float32)
        metadata[k] = [inc, args.disk_pa_deg, bar]
    baseline_grid_mass = np.sum(baseline_mass, axis=(1, 2, 3)).astype(np.float32)

    # ---- MDN forward pass (all geometries at once) ------------------------
    features = make_density_residual_features(
        images, metadata, baseline_grid_mass_msun=baseline_grid_mass,
        image_feature_size=image_feature_size, central_pixel_scale_kpc=central_pixel_scale_kpc,
    )
    x = (features - normalization["x_mean"][None, :]) / normalization["x_scale"][None, :]
    torch.manual_seed(rng_seed)
    with torch.no_grad():
        sampled_z = model.sample(torch.tensor(x, dtype=torch.float32), n_samples=args.n_samples).numpy()
    sampled_coeff = (
        sampled_z * normalization["y_scale"][None, None, :]
        + normalization["y_mean"][None, None, :]
    ).astype(np.float32)
    mean_coeff = sampled_coeff.mean(axis=1).astype(np.float32)
    delta_mean = reconstruct_delta_mass_from_coefficients(
        mean_coeff, components=components, mean=pca_mean,
        scale_msun=baseline_grid_mass, grid_shape=grid_shape,
    )
    pred_mean_mass = corrected_mass_from_delta(
        baseline_mass, delta_mean, preserve_baseline_total_mass=True
    )

    # truth (broadcast) summaries
    truth_b = np.broadcast_to(truth_mass, (len(geoms), *grid_shape))
    sum_kwargs = dict(
        spec=spec, central_radius_kpc=args.central_radius_kpc,
        bar_half_angle_deg=args.bar_half_angle_deg,
    )
    s_truth = summaries_batch(truth_b, **sum_kwargs)
    s_base = summaries_batch(baseline_mass, **sum_kwargs)
    s_pred = summaries_batch(pred_mean_mass, **sum_kwargs)
    truth_total = float(truth_mass.sum())

    # ---- per-geometry metrics ---------------------------------------------
    rows = []
    for k, (inc, bar) in enumerate(geoms):
        base_mae = float(np.mean(np.abs(baseline_mass[k] - truth_mass)))
        pred_mae = float(np.mean(np.abs(pred_mean_mass[k] - truth_mass)))
        b_dip = bar_end_stats(baseline_mass[k], spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc, bar_length)
        p_dip = bar_end_stats(pred_mean_mass[k], spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc, bar_length)
        # per-sample bar-end dip + RMS z for an honest posterior interval
        dips, rmsz = [], []
        for start in range(0, args.n_samples, args.sample_batch_size):
            stop = min(start + args.sample_batch_size, args.n_samples)
            flat = sampled_coeff[k, start:stop, :]
            d = reconstruct_delta_mass_from_coefficients(
                flat, components=components, mean=pca_mean,
                scale_msun=np.full(stop - start, baseline_grid_mass[k], dtype=np.float32),
                grid_shape=grid_shape,
            )
            m = corrected_mass_from_delta(
                np.broadcast_to(baseline_mass[k], (stop - start, *grid_shape)), d,
                preserve_baseline_total_mass=True,
            )
            for s in range(stop - start):
                dd, _, rr = bar_end_stats(m[s], spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc, bar_length)
                dips.append(dd)
                rmsz.append(rr)
        dips = np.array(dips)
        rmsz = np.array(rmsz)
        rows.append({
            "inclination_deg": inc, "bar_angle_deg": bar,
            "baseline_cell_mass_mae_msun": base_mae,
            "mdn_cell_mass_mae_msun": pred_mae,
            "improvement_pct": 100.0 * (1.0 - pred_mae / base_mae),
            "baseline_total_mass_frac_err": float(abs(baseline_mass[k].sum() - truth_total) / truth_total),
            "mdn_total_mass_frac_err": float(abs(pred_mean_mass[k].sum() - truth_total) / truth_total),
            "central_frac_truth": float(s_truth["central_mass_fraction"][k]),
            "central_frac_baseline": float(s_base["central_mass_fraction"][k]),
            "central_frac_mdn": float(s_pred["central_mass_fraction"][k]),
            "vrms_truth_kpc": float(s_truth["vertical_rms_height"][k]),
            "vrms_baseline_kpc": float(s_base["vertical_rms_height"][k]),
            "vrms_mdn_kpc": float(s_pred["vertical_rms_height"][k]),
            "m2_bar_truth": m2_bar_amplitude(truth_mass, spec, bar_length),
            "m2_bar_baseline": m2_bar_amplitude(baseline_mass[k], spec, bar_length),
            "m2_bar_mdn": m2_bar_amplitude(pred_mean_mass[k], spec, bar_length),
            "barend_dip_truth": truth_dip_prod[0],
            "barend_dip_baseline": b_dip[0],
            "barend_dip_mdn": p_dip[0],
            "barend_dip_mdn_post_mean": float(dips.mean()),
            "barend_dip_mdn_p16": float(np.percentile(dips, 16)),
            "barend_dip_mdn_p84": float(np.percentile(dips, 84)),
            "barend_rmsz_truth_kpc": float(truth_dip_prod[2]),
            "barend_rmsz_baseline_kpc": float(b_dip[2]),
            "barend_rmsz_mdn_kpc": float(p_dip[2]),
            "barend_rmsz_mdn_p16": float(np.percentile(rmsz, 16)),
            "barend_rmsz_mdn_p84": float(np.percentile(rmsz, 84)),
        })
        print(f"i={inc:.0f} bar={bar:.0f}: base MAE={base_mae:.3e} MDN MAE={pred_mae:.3e} "
              f"({rows[-1]['improvement_pct']:.1f}%)  dip t/b/m="
              f"{truth_dip_prod[0]:.2f}/{b_dip[0]:.2f}/{p_dip[0]:.2f}")

    def med(key):
        return float(np.median([r[key] for r in rows]))

    aggregate = {
        "median_baseline_cell_mass_mae_msun": med("baseline_cell_mass_mae_msun"),
        "median_mdn_cell_mass_mae_msun": med("mdn_cell_mass_mae_msun"),
        "median_improvement_pct": med("improvement_pct"),
        "median_mdn_total_mass_frac_err": med("mdn_total_mass_frac_err"),
        "median_vrms_truth_kpc": med("vrms_truth_kpc"),
        "median_vrms_baseline_kpc": med("vrms_baseline_kpc"),
        "median_vrms_mdn_kpc": med("vrms_mdn_kpc"),
        "median_barend_dip_mdn": med("barend_dip_mdn"),
    }

    metrics = {
        "setup": {
            "run_dir": str(args.run_dir),
            "pca": str(args.pca),
            "n_samples": args.n_samples,
            "vertical_scale_height_kpc": args.vertical_scale_height_kpc,
            "correction_heads": "excluded (core MDN only)",
            "truth_grid_mass_fraction": float(truth_grid.mass_fraction_in_grid),
            "truth_barend_dip_production_0p625kpc": truth_dip_prod[0],
            "truth_barend_peakz_production_kpc": truth_dip_prod[1],
            "truth_barend_rmsz_production_kpc": truth_dip_prod[2],
            "truth_barend_dip_fine_0p3125kpc": truth_dip_fine[0],
            "truth_barend_peakz_fine_kpc": truth_dip_fine[1],
            **align_info,
        },
        "per_geometry": rows,
        "aggregate": aggregate,
    }
    (args.output_dir / "nbody_shen2010_deprojection_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "nbody_shen2010_grids.npz",
        truth_mass=truth_mass, baseline_mass=baseline_mass, pred_mean_mass=pred_mean_mass,
        metadata=metadata, r_edges_kpc=spec.r_edges_kpc, phi_edges_rad=spec.phi_edges_rad,
        z_edges_kpc=spec.z_edges_kpc, bar_length_kpc=np.float32(bar_length),
    )

    # ---- figures -----------------------------------------------------------
    _make_figures(
        args, geoms, images, truth_mass, baseline_mass, pred_mean_mass, spec, volumes,
        bar_length, sampled_coeff, components, pca_mean, baseline_grid_mass, grid_shape,
        truth_dip_fine, truth_grid_fine, spec_fine, rows, fig_dir,
    )

    _write_markdown(args.output_dir / "nbody_shen2010_deprojection.md", metrics, rows)
    print(f"\nwrote results to {args.output_dir}")


def _faceon_cartesian(mass, r_edges, phi_edges, *, radius=15.0, voxel=0.2):
    """Gap-free face-on surface-density map by nearest-cell Cartesian lookup."""
    surf = mass.sum(axis=2)
    area = 0.5 * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)[:, None] * np.diff(phi_edges)[None, :]
    sigma = surf / area
    xy = np.arange(-radius + 0.5 * voxel, radius, voxel)
    xg, yg = np.meshgrid(xy, xy, indexing="ij")
    rr = np.hypot(xg, yg)
    pp = np.arctan2(yg, xg)
    ir = np.clip(np.searchsorted(r_edges, rr) - 1, 0, len(r_edges) - 2)
    ip = np.clip(np.searchsorted(phi_edges, pp) - 1, 0, len(phi_edges) - 2)
    return xy, np.where(rr < r_edges[-1], sigma[ir, ip], 0.0)


def _make_figures(args, geoms, images, truth_mass, baseline_mass, pred_mean_mass, spec, volumes,
                  bar_length, sampled_coeff, components, pca_mean, baseline_grid_mass, grid_shape,
                  truth_dip_fine, truth_grid_fine, spec_fine, rows, fig_dir):
    rep = None
    for k, (inc, bar) in enumerate(geoms):
        if np.isclose(inc, args.rep_inclination) and np.isclose(bar, args.rep_bar_angle):
            rep = k
    if rep is None:
        rep = len(geoms) // 2
    inc, bar = geoms[rep]
    re = spec.r_edges_kpc
    pe = spec.phi_edges_rad
    ze = spec.z_edges_kpc

    def edgeon(mass):
        density = mass / volumes
        xy, z, panel = _edgeon(density, re, pe, ze, 8.0, 4.0, 0.1, 1.5)
        return xy, z, panel

    # Figure 1: representative-geometry recovery (face-on + edge-on + bar-end profile)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.log1p(images[rep]), origin="lower", cmap="inferno")
    ax.set_title(f"mock image (i={inc:.0f}, bar={bar:.0f})")
    ax.set_xticks([])
    ax.set_yticks([])

    faceon = {
        "truth": truth_mass, "baseline": baseline_mass[rep], "MDN": pred_mean_mass[rep],
    }
    for col, name in enumerate(("truth", "baseline", "MDN")):
        _, panel = _faceon_cartesian(faceon[name], re, pe, radius=15.0)
        a = fig.add_subplot(gs[0, col + 1])
        vmax = float(panel.max())
        a.imshow(panel.T, origin="lower", extent=[-15, 15, -15, 15], cmap="inferno",
                 norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax))
        a.set_title(f"{name} face-on")
        a.set_xlabel("x [kpc]")
        a.set_ylabel("y [kpc]")

    for col, name in enumerate(("truth", "baseline", "MDN")):
        xy, z, panel = edgeon(faceon[name])
        a = fig.add_subplot(gs[1, col])
        vmax = float(panel.max())
        a.imshow(panel.T, origin="lower", extent=[-8, 8, -4, 4], cmap="magma",
                 norm=LogNorm(vmin=vmax * 5e-3, vmax=vmax), aspect="auto")
        a.contour(xy, z, panel.T, levels=vmax * np.array([0.04, 0.1, 0.25, 0.5]),
                  colors="cyan", linewidths=0.7, alpha=0.7)
        a.set_title(f"{name} edge-on (bar)")
        a.set_xlabel("x [kpc] (along bar)")
        a.set_ylabel("z [kpc]")

    # bar-end vertical profiles
    a = fig.add_subplot(gs[1, 3])
    colors = {"truth": "k", "baseline": "0.6", "MDN": "#b3403c"}
    for name in ("truth", "baseline", "MDN"):
        zc, prof = _bar_end_profile(faceon[name], re, pe, ze, bar_length)
        prof = prof / prof.max()
        dip = prof[np.argmin(np.abs(zc))]
        a.plot(zc, prof, "o-", color=colors[name], lw=2, ms=4, label=f"{name} (dip={dip:.2f})")
    # truth at fine z (what the production grid smears out)
    zc, prof = _bar_end_profile(
        truth_grid_fine.mass_msun.astype(np.float32),
        spec_fine.r_edges_kpc, spec_fine.phi_edges_rad, spec_fine.z_edges_kpc, bar_length,
    )
    prof = prof / prof.max()
    a.plot(zc, prof, ":", color="green", lw=2, label=f"truth fine-z (dip={truth_dip_fine[0]:.2f})")
    a.set_xlim(-3, 3)
    a.set_xlabel("z [kpc]")
    a.set_ylabel("normalized mass/z-bin")
    a.set_title("bar-end vertical profile")
    a.legend(fontsize=8)
    fig.suptitle("Shen2010 N-body bar: TNG50-trained MDN deprojection (out-of-distribution)", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / f"nbody_recovery_i{inc:.0f}_bar{bar:.0f}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: cell-mass MAE + vertical RMS across all geometries
    labels = [f"i{int(r['inclination_deg'])}\nb{int(r['bar_angle_deg'])}" for r in rows]
    xidx = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].bar(xidx - 0.2, [r["baseline_cell_mass_mae_msun"] for r in rows], 0.4, label="baseline", color="0.6")
    axes[0].bar(xidx + 0.2, [r["mdn_cell_mass_mae_msun"] for r in rows], 0.4, label="MDN", color="#b3403c")
    axes[0].set_xticks(xidx)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("cell-mass MAE [Msun]")
    axes[0].set_title("3D cell-mass error")
    axes[0].legend()
    axes[1].axhline(float(np.median([r["vrms_truth_kpc"] for r in rows])), color="k", ls="--", label="truth")
    axes[1].bar(xidx - 0.2, [r["vrms_baseline_kpc"] for r in rows], 0.4, label="baseline", color="0.6")
    axes[1].bar(xidx + 0.2, [r["vrms_mdn_kpc"] for r in rows], 0.4, label="MDN", color="#b3403c")
    axes[1].set_xticks(xidx)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("global vertical RMS height [kpc]")
    axes[1].set_title("vertical thickness")
    axes[1].legend()
    axes[2].axhline([r["central_frac_truth"] for r in rows][0], color="k", ls="--", label="truth")
    axes[2].bar(xidx - 0.2, [r["central_frac_baseline"] for r in rows], 0.4, label="baseline", color="0.6")
    axes[2].bar(xidx + 0.2, [r["central_frac_mdn"] for r in rows], 0.4, label="MDN", color="#b3403c")
    axes[2].set_xticks(xidx)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("central mass fraction (R<2 kpc)")
    axes[2].set_title("central concentration")
    axes[2].legend()
    fig.suptitle("Shen2010 N-body: recovery vs geometry (3x3 inclination x bar angle)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "nbody_recovery_vs_geometry.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(path: Path, metrics: dict, rows: list[dict]) -> None:
    s = metrics["setup"]
    L = [
        "# Shen2010 N-body Bar: Out-of-Distribution MDN Deprojection",
        "",
        "Genuine boxy/peanut N-body bar (Shen et al. 2010 MW model) deprojected with the",
        "**TNG50-trained adopted 2c MDN, no retraining**. Core MDN only (no correction heads).",
        "",
        "## Setup",
        f"- particles: {s['n_particles']}, total mass {s['total_mass_msun']:.3e} Msun (equal-mass)",
        f"- alignment: disk normal {np.round(s['disk_normal'],3).tolist()} via {s['orientation_method']}; "
        f"bar angle removed {s['bar_angle_removed_deg']:.1f} deg (residual {s['residual_bar_angle_deg']:.2f})",
        f"- bar length (m=2 half-max): {s['bar_length_kpc']:.2f} kpc",
        f"- truth mass fraction inside grid: {s['truth_grid_mass_fraction']:.4f}",
        f"- run: `{s['run_dir']}`",
        "",
        "## Intrinsic boxy/peanut strength (truth)",
        f"- bar-end midplane dip at production 0.625 kpc grid: **{s['truth_barend_dip_production_0p625kpc']:.3f}** "
        f"(off-plane peak |z|={s['truth_barend_peakz_production_kpc']:.2f} kpc, RMS z={s['truth_barend_rmsz_production_kpc']:.2f} kpc)",
        f"- bar-end midplane dip at fine 0.3125 kpc grid: **{s['truth_barend_dip_fine_0p3125kpc']:.3f}** "
        f"(off-plane peak |z|={s['truth_barend_peakz_fine_kpc']:.2f} kpc)",
        "  (dip<1 => off-plane X lobes; production grid smears them, per milestone 2d)",
        "",
        "## Aggregate (median over 9 geometries)",
        f"- baseline cell-mass MAE: {metrics['aggregate']['median_baseline_cell_mass_mae_msun']:.3e} Msun",
        f"- MDN cell-mass MAE: {metrics['aggregate']['median_mdn_cell_mass_mae_msun']:.3e} Msun "
        f"(**{metrics['aggregate']['median_improvement_pct']:.1f}%** better)",
        f"- MDN total-mass fractional error: {metrics['aggregate']['median_mdn_total_mass_frac_err']:.4f}",
        f"- vertical RMS height: truth {metrics['aggregate']['median_vrms_truth_kpc']:.2f}, "
        f"baseline {metrics['aggregate']['median_vrms_baseline_kpc']:.2f}, "
        f"MDN {metrics['aggregate']['median_vrms_mdn_kpc']:.2f} kpc",
        f"- MDN bar-end dip (median): {metrics['aggregate']['median_barend_dip_mdn']:.3f} (truth "
        f"{s['truth_barend_dip_production_0p625kpc']:.3f})",
        "",
        "## Per-geometry",
        "",
        "| i | bar | base MAE | MDN MAE | impr% | MDN totM err | cfrac t/b/m | vRMS t/b/m | dip t/b/m |",
        "| -: | -: | -: | -: | -: | -: | :- | :- | :- |",
    ]
    for r in rows:
        L.append(
            f"| {r['inclination_deg']:.0f} | {r['bar_angle_deg']:.0f} | "
            f"{r['baseline_cell_mass_mae_msun']:.2e} | {r['mdn_cell_mass_mae_msun']:.2e} | "
            f"{r['improvement_pct']:.1f} | {r['mdn_total_mass_frac_err']:.3f} | "
            f"{r['central_frac_truth']:.3f}/{r['central_frac_baseline']:.3f}/{r['central_frac_mdn']:.3f} | "
            f"{r['vrms_truth_kpc']:.2f}/{r['vrms_baseline_kpc']:.2f}/{r['vrms_mdn_kpc']:.2f} | "
            f"{r['barend_dip_truth']:.2f}/{r['barend_dip_baseline']:.2f}/{r['barend_dip_mdn']:.2f} |"
        )
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
