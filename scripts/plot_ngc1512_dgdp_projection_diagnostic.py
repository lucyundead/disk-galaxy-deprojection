"""Render an NGC 1512 dgdp deprojection and its sky-plane reprojection check.

The four panels are the dgdp face-on surface density (with isodensity contours),
the S4G input image, the line-of-sight projection of the dgdp density, and the
fractional image residual.  The input is the refilled S4G crop so that the
foreground-star and mosaic holes do not become artificial density deficits.

Run:
    PYTHONPATH=src .venv/bin/python scripts/plot_ngc1512_dgdp_projection_diagnostic.py
    PYTHONPATH=src .venv/bin/python scripts/plot_ngc1512_dgdp_projection_diagnostic.py --dynamical
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.ndimage import gaussian_filter

from dgdp import deproject
from dgdp.density3d import make_cylindrical_grid_spec
from dgdp.image import geometric_baseline, load_image
from dgdp.reproject import project_to_sky

sys.path.insert(0, "scripts")
from plot_ngc4321_figures import make_lookup, project


INPUT_FITS = "NGC1512/NGC1512_refilled.fits"
DISTANCE_MPC = 18.83
INCLINATION_DEG = 42.0
# Geometry from the faint connected outer disk: its 2-sigma isophote has q=0.725
# (thin-disk i_eff=43.5 deg) and PA=241.1 deg.  Retain the adopted i=42 deg.
PA_ONSKY_DEG = 241.1
OUTER_ISOPHOTE_Q = 0.725
STELLAR_MASS_MSUN = 5.25e10
FOV_KPC = 16.0
N_FACE = 320
N_R, N_PHI, N_Z = 256, 192, 128
OUTER_TAPER_INNER_KPC = 6.0
OUTER_TAPER_OUTER_KPC = 10.0
NATIVE_OUT = Path("outputs/real_images/ngc1512_dgdp_projection_diagnostic_pa241p1.png")
DYNAMICAL_OUT = Path("outputs/real_images/ngc1512_dgdp_dynamical_diagnostic_pa241p1_m024_v1.png")
DYNAMICAL_DENSITY_OUT = Path(
    "outputs/real_images/ngc1512_dgdp_dynamical_density_pa241p1_m024_taper6_10_v1.npz"
)


def _bar_azimuth(result) -> float:
    """Measured m=2 bar phase in the dgdp disk-plane convention."""
    sigma = (result.density_3d * result._vol).sum(axis=2)
    radial = (result.grid["r"] > 2.0) & (result.grid["r"] < 5.5)
    return float(-np.angle(np.sum(np.fft.rfft(sigma, axis=1)[:, 2][radial])) / 2.0)


def main(*, dynamical: bool = False) -> None:
    result = deproject(
        INPUT_FITS,
        distance_mpc=DISTANCE_MPC,
        inclination_deg=INCLINATION_DEG,
        pa_onsky_deg=PA_ONSKY_DEG,
        ml=1.0,
        stellar_mass=STELLAR_MASS_MSUN,
    )
    image = load_image(
        INPUT_FITS,
        distance_mpc=DISTANCE_MPC,
        inclination_deg=INCLINATION_DEG,
        pa_onsky_deg=PA_ONSKY_DEG,
    )
    fine = result.regrid(n_r=N_R, n_phi=N_PHI, n_z=N_Z)
    plot_result = (
        fine.bisymmetrize_for_dynamics(
            inner_radius_kpc=OUTER_TAPER_INNER_KPC,
            outer_radius_kpc=OUTER_TAPER_OUTER_KPC,
        )
        if dynamical
        else fine
    )
    native_grid = result._ctx["grid"]
    native_spec = make_cylindrical_grid_spec(
        r_min_kpc=native_grid["r_min"],
        r_max_kpc=native_grid["r_max"],
        n_r=native_grid["n_r"],
        n_phi=native_grid["n_phi"],
        z_max_kpc=native_grid["z_max"],
        n_z=native_grid["n_z"],
    )
    input_sky = geometric_baseline(
        image,
        native_spec,
        scale_height_kpc=0.3,
        stellar_mass=STELLAR_MASS_MSUN,
    )
    fine_spec = make_cylindrical_grid_spec(
        r_min_kpc=native_grid["r_min"],
        r_max_kpc=native_grid["r_max"],
        n_r=N_R,
        n_phi=N_PHI,
        z_max_kpc=native_grid["z_max"],
        n_z=N_Z,
    )

    # Face-on map in the intrinsic bar frame.
    bar_angle = _bar_azimuth(plot_result)
    cos_bar, sin_bar = np.cos(bar_angle), np.sin(bar_angle)
    density_lookup = make_lookup(
        plot_result.density_3d,
        fine_spec.r_edges_kpc,
        fine_spec.phi_edges_rad,
        fine_spec.z_edges_kpc,
    )

    def bar_frame_lookup(x, y, z):
        return density_lookup(x * cos_bar - y * sin_bar, x * sin_bar + y * cos_bar, z)

    face_axis = np.linspace(-FOV_KPC, FOV_KPC, N_FACE)
    face_on = project(bar_frame_lookup, face_axis, face_axis, np.linspace(-5.0, 5.0, 128), along="z")

    # Use the exact grid and sky sampling of dgdp's reprojection loop.  Its input panel is
    # therefore a rotated/resampled S4G image, rather than a different native-pixel sampling.
    sky_edges = input_sky["image_edges_kpc"]
    raw = input_sky["image_tng"]
    projection_result = plot_result if dynamical else result
    model = project_to_sky(
        projection_result.density_3d,
        projection_result.grid["r"],
        projection_result.grid["phi"],
        projection_result.grid["z"],
        INCLINATION_DEG,
        sky_edges,
    )
    projected = model * raw.sum() / max(model.sum(), 1e-30)
    model_threshold = 1e-4 * float(projected.max())
    valid = (raw > 0) & (projected > model_threshold)
    fractional_residual = (raw - projected) / np.maximum(projected, 1e-30)
    input_norm = raw / max(raw.sum(), 1e-30)
    residual_rms = float(np.sqrt(np.sum(input_norm[valid] * fractional_residual[valid] ** 2) /
                                 np.sum(input_norm[valid])))

    extent = [sky_edges[0], sky_edges[-1], sky_edges[0], sky_edges[-1]]
    face_display = gaussian_filter(face_on, sigma=1.5)
    face_vmin = float(np.percentile(face_display[face_display > 0], 10.0))
    face_vmax = float(np.percentile(face_display, 99.8))
    display_values = np.concatenate((raw[raw > 0], projected[projected > 0]))
    image_vmin = float(np.percentile(display_values, 1.0))
    image_vmax = float(np.percentile(display_values, 99.8))

    fig, axes = plt.subplots(1, 4, figsize=(19.2, 5.1), constrained_layout=True)
    im0 = axes[0].imshow(
        face_display.T,
        origin="lower",
        extent=[-FOV_KPC, FOV_KPC, -FOV_KPC, FOV_KPC],
        cmap="magma",
        norm=LogNorm(vmin=face_vmin, vmax=face_vmax),
    )
    contours = np.geomspace(
        np.percentile(face_display[face_display > 0], 55.0), np.percentile(face_display, 99.5), 7
    )
    xx_face, yy_face = np.meshgrid(face_axis, face_axis, indexing="ij")
    axes[0].contour(xx_face, yy_face, face_display, levels=contours, colors="white", linewidths=0.7, alpha=0.78)
    density_title = "dgdp dynamical $\\Sigma_\\star$" if dynamical else "dgdp deprojected $\\Sigma_\\star$"
    axes[0].set_title(f"{density_title}\n(face-on; contours = isodensity)")
    axes[0].set(xlabel="x [kpc] (bar frame)", ylabel="y [kpc]", xlim=(-FOV_KPC, FOV_KPC), ylim=(-FOV_KPC, FOV_KPC))
    axes[0].set_aspect("equal")
    cbar0 = fig.colorbar(im0, ax=axes[0], pad=0.02, fraction=0.046)
    cbar0.set_label(r"$\Sigma_\star$ [$M_\odot$ kpc$^{-2}$]")

    image_norm = LogNorm(vmin=image_vmin, vmax=image_vmax)
    axes[1].set_facecolor("black")
    axes[1].imshow(np.ma.masked_less_equal(raw, 0.0), origin="lower", extent=extent, cmap="gray_r", norm=image_norm)
    axes[1].set_title("S4G 3.6 $\\mu$m input\n(resampled to dgdp sky grid)")
    axes[1].set(xlabel="x [kpc] (major axis)", ylabel="y [kpc] (minor axis)")
    axes[1].set(xlim=(-FOV_KPC, FOV_KPC), ylim=(-FOV_KPC, FOV_KPC))
    axes[1].set_aspect("equal")

    axes[2].set_facecolor("black")
    im2 = axes[2].imshow(np.ma.masked_less_equal(projected, 0.0), origin="lower", extent=extent, cmap="gray_r", norm=image_norm)
    projected_title = "Projected dynamical density" if dynamical else "Projected dgdp density"
    axes[2].set_title(f"{projected_title}\n(same reprojection grid)")
    axes[2].set(xlabel="x [kpc] (major axis)", ylabel="y [kpc] (minor axis)")
    axes[2].set(xlim=(-FOV_KPC, FOV_KPC), ylim=(-FOV_KPC, FOV_KPC))
    axes[2].set_aspect("equal")
    cbar2 = fig.colorbar(im2, ax=[axes[1], axes[2]], pad=0.02, fraction=0.035)
    cbar2.set_label("background-subtracted S4G units (shared scale)")

    im3 = axes[3].imshow(
        np.ma.array(fractional_residual, mask=~valid),
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5),
    )
    residual_name = "dyn" if dynamical else "dgdp"
    axes[3].set_title(
        f"Fractional residual\n$(\\mathrm{{S4G}}-\\mathrm{{{residual_name}}})/"
        f"\\mathrm{{{residual_name}}}$"
    )
    axes[3].set(xlabel="x [kpc] (major axis)", ylabel="y [kpc] (minor axis)")
    axes[3].set(xlim=(-FOV_KPC, FOV_KPC), ylim=(-FOV_KPC, FOV_KPC))
    axes[3].set_aspect("equal")
    cbar3 = fig.colorbar(im3, ax=axes[3], pad=0.02, fraction=0.046, extend="both")
    cbar3.set_label("fractional residual")

    if dynamical:
        title = "NGC 1512: bisymmetric dgdp density for dynamical modelling"
        footer = (
            f"retained modes $m=0,2,4$  ·  outer taper "
            f"{OUTER_TAPER_INNER_KPC:.0f}–{OUTER_TAPER_OUTER_KPC:.0f} kpc  ·  "
            f"dynamical-density projection RMS = {residual_rms:.3f}"
        )
        out = DYNAMICAL_OUT
    else:
        title = f"NGC 1512: native dgdp deprojection with outer-isophote PA = {PA_ONSKY_DEG:.1f}°"
        footer = (
            f"reprojection-loop residual {result.reproj['history'][0]:.3f} → "
            f"{min(result.reproj['history']):.3f}  ·  outer-isophote $q={OUTER_ISOPHOTE_Q:.3f}$  ·  "
            f"flux-weighted fractional RMS = {residual_rms:.3f}"
        )
        out = NATIVE_OUT
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.text(
        0.5,
        -0.025,
        footer,
        ha="center",
        fontsize=9,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    if dynamical:
        np.savez_compressed(
            DYNAMICAL_DENSITY_OUT,
            density_3d=plot_result.density_3d,
            r_kpc=plot_result.grid["r"],
            phi_rad=plot_result.grid["phi"],
            z_kpc=plot_result.grid["z"],
            r_edges_kpc=fine_spec.r_edges_kpc,
            phi_edges_rad=fine_spec.phi_edges_rad,
            z_edges_kpc=fine_spec.z_edges_kpc,
            total_mass_msun=plot_result.total_mass,
            distance_mpc=DISTANCE_MPC,
            inclination_deg=INCLINATION_DEG,
            pa_onsky_deg=PA_ONSKY_DEG,
            bar_angle_rad=bar_angle,
            retained_modes=np.array([0, 2, 4]),
            outer_taper_inner_kpc=OUTER_TAPER_INNER_KPC,
            outer_taper_outer_kpc=OUTER_TAPER_OUTER_KPC,
            source_fits=INPUT_FITS,
            density_unit="Msun/kpc^3",
            product_kind="derived_bisymmetric_dynamical_density",
        )
        print(f"wrote {DYNAMICAL_DENSITY_OUT}")
    print(
        f"bar angle={np.degrees(bar_angle):.1f} deg; reprojection history={result.reproj['history']}; "
        f"flux-weighted fractional RMS={residual_rms:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynamical",
        action="store_true",
        help="write the m=0,2,4 bisymmetric density, its projection diagnostic, and density cube",
    )
    main(dynamical=parser.parse_args().dynamical)
