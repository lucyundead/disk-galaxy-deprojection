"""First-try deprojection of a REAL S4G image (NGC 4321 / M100) with our method.

Uses the parts of the image-anchored, mass-conserving even-m Fourier x (R,z) method
that transfer to observations:

  S4G 3.6um image  ->  background-subtract + measure the apparent ellipse (PA, i)
  ->  deproject the light into the disk-plane surface density Sigma(R,phi)
  ->  baseline sech^2(z/h) vertical profile (mass-conserving)  ->  even-m Fourier
  x (R,z) representation  ->  agama.Density / agama.Potential (dgdp.agama_density)
  ->  rotation curve.

Light is treated as mass up to a constant M/L (a global scaling that does NOT affect
the deprojection geometry); the grid total is normalized to a literature stellar mass,
and v_c scales as sqrt(M*/M*_assumed).

What is INTENTIONALLY not done: the TNG-LEARNED vertical-profile refinement q_m(z;R)
is trained on TNG barred-galaxy mocks at a fixed feature/scale format and is strongly
out-of-distribution for a real S4G image, so this uses the geometric baseline vertical
profile. The deprojection geometry, the smooth Fourier x (R,z) rep, and the AGAMA
potential are the transferable pieces.

Run:
    PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 \\
        .venv/bin/python scripts/deproject_real_image_ngc4321.py
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
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm

try:
    import agama
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama

from astropy import log as astropy_log

from dgdp.agama_density import circular_velocity, fourier_rz_to_agama_density, to_agama_potential
from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.fourier_rz import fit_fourier_rz_from_grid, reconstruct_fourier_rz

agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s
astropy_log.setLevel("ERROR")  # silence the harmless SIP-CTYPE info message
ARCSEC_PER_RAD = 206264.806


def measure_ellipse(light, cx, cy, *, r_in_pix, r_out_pix, thresh):
    """Flux-weighted second-moment ellipse in an annulus (avoids the round bulge/bar).

    Returns ``(pa_rad, axis_ratio b/a, incl_deg)``; the outer annulus traces the disk.
    """
    ny, nx = light.shape
    iy, ix = np.mgrid[0:ny, 0:nx]
    dx, dy = ix - cx, iy - cy
    r = np.hypot(dx, dy)
    sel = (light > thresh) & (r > r_in_pix) & (r < r_out_pix)
    w = light[sel]
    x, y = dx[sel].astype(float), dy[sel].astype(float)
    m = w.sum()
    mxx = (w * x * x).sum() / m
    myy = (w * y * y).sum() / m
    mxy = (w * x * y).sum() / m
    pa = 0.5 * np.arctan2(2.0 * mxy, mxx - myy)
    common = np.sqrt(((mxx - myy) / 2.0) ** 2 + mxy ** 2)
    lam1 = (mxx + myy) / 2.0 + common
    lam2 = (mxx + myy) / 2.0 - common
    ba = float(np.sqrt(max(lam2, 0.0) / lam1))
    return float(pa), ba, float(np.degrees(np.arccos(np.clip(ba, 1e-3, 1.0))))


def onsky_pa(wcs, cx, cy, pa_pix):
    """On-sky position angle (deg E of N) of the pixel-frame major axis at angle pa_pix."""
    c0 = wcs.pixel_to_world(cx, cy)
    c1 = wcs.pixel_to_world(cx + 50.0 * np.cos(pa_pix), cy + 50.0 * np.sin(pa_pix))
    return float(c0.position_angle(c1).deg)


def pixel_pa_from_onsky(wcs, cx, cy, pa_sky_deg):
    """Invert :func:`onsky_pa`: pixel-frame angle whose on-sky PA matches (mod 180 deg)."""
    grid = np.radians(np.linspace(0.0, 180.0, 721, endpoint=False))
    skies = np.array([onsky_pa(wcs, cx, cy, g) % 180.0 for g in grid])
    diff = np.abs((skies - (pa_sky_deg % 180.0) + 90.0) % 180.0 - 90.0)
    return float(grid[int(np.argmin(diff))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=Path, default=Path("NGC4321_m_c_r_f.fits"))
    ap.add_argument("--distance-mpc", type=float, default=15.2)  # Cepheid distance to M100
    ap.add_argument("--inclination-deg", type=float, default=30.0, help="literature M100 disk i")
    ap.add_argument("--pa-onsky-deg", type=float, default=153.0, help="literature on-sky disk PA (E of N)")
    ap.add_argument("--pa-pixel-deg", type=float, default=None, help="override pixel-frame major-axis PA")
    ap.add_argument("--ellipse-annulus-pix", type=float, nargs=2, default=[120.0, 260.0])
    ap.add_argument("--stellar-mass", type=float, default=6.0e10)  # literature M* of M100
    ap.add_argument("--scale-height-kpc", type=float, default=0.3)
    ap.add_argument("--bar-angle-deg", type=float, default=0.0)
    ap.add_argument("--r-max-rep", type=float, default=20.0)
    ap.add_argument("--z-max-rep", type=float, default=3.0)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/real_images"))
    args = ap.parse_args()

    with fits.open(args.fits) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
        wcs = WCS(header)
    cd = np.array([[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]])
    pix_arcsec = float(np.sqrt(np.abs(np.linalg.det(cd))) * 3600.0)
    pix_kpc = pix_arcsec / ARCSEC_PER_RAD * (args.distance_mpc * 1e3)  # kpc/pixel
    print(f"image {data.shape}  pixel {pix_arcsec:.3f} arcsec = {pix_kpc:.4f} kpc  (D={args.distance_mpc} Mpc)")

    # ---- background subtract, find center, get light>=0 ----
    border = np.concatenate([data[0], data[-1], data[:, 0], data[:, -1]])
    bkg = float(np.median(border))
    bkg_std = float(np.std(border))
    light = np.clip(data - bkg, 0.0, None)
    iy0, ix0 = np.unravel_index(np.argmax(light), light.shape)
    ny, nx = light.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    win = (np.hypot(xx - ix0, yy - iy0) < 80) & (light > 5 * bkg_std)
    cx = float((light[win] * xx[win]).sum() / light[win].sum())
    cy = float((light[win] * yy[win]).sum() / light[win].sum())
    print(f"background {bkg:.2f} MJy/sr (std {bkg_std:.2f}); center pixel (x,y)=({cx:.1f},{cy:.1f})")

    # ---- apparent ellipse (outer annulus) for context; default to literature geometry ----
    r_in, r_out = args.ellipse_annulus_pix
    pa_meas, ba, incl_meas = measure_ellipse(light, cx, cy, r_in_pix=r_in, r_out_pix=r_out, thresh=3 * bkg_std)
    incl = args.inclination_deg
    if args.pa_pixel_deg is not None:
        pa_pix = np.radians(args.pa_pixel_deg)
    else:
        pa_pix = pixel_pa_from_onsky(wcs, cx, cy, args.pa_onsky_deg)
    pa_sky = onsky_pa(wcs, cx, cy, pa_pix)
    print(f"annulus ellipse ({r_in:.0f}-{r_out:.0f} px): b/a={ba:.3f} -> i_meas={incl_meas:.1f} deg, "
          f"PA_meas(on-sky)={onsky_pa(wcs, cx, cy, pa_meas):.1f} deg")
    print(f"USING (literature) incl={incl:.1f} deg, PA(on-sky)={args.pa_onsky_deg:.1f} -> "
          f"PA(pixel)={np.degrees(pa_pix):.1f} deg [check on-sky={pa_sky:.1f}], h={args.scale_height_kpc} kpc")

    # ---- deproject light pixels into the face-on disk plane (mass = light, conserved) ----
    sel = light > 2 * bkg_std
    x_sky = (xx[sel] - cx) * pix_kpc
    y_sky = (yy[sel] - cy) * pix_kpc
    mass = light[sel].astype(float)
    cpa, spa = np.cos(pa_pix), np.sin(pa_pix)
    x_major = x_sky * cpa + y_sky * spa
    y_minor = -x_sky * spa + y_sky * cpa
    y_disk = y_minor / max(np.cos(np.radians(incl)), 1e-3)  # de-foreshorten the minor axis
    cba, sba = np.cos(np.radians(args.bar_angle_deg)), np.sin(np.radians(args.bar_angle_deg))
    x_disk = x_major * cba + y_disk * sba  # rotate bar onto x (cosmetic for mass/v_c)
    y_disk = -x_major * sba + y_disk * cba
    radius = np.hypot(x_disk, y_disk)
    phi = np.arctan2(y_disk, x_disk)

    # ---- cylindrical Sigma(R,phi) x normalized sech^2(z/h): mass-conserving 3D grid ----
    spec = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32)
    r_edges, phi_edges, z_edges = spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    sigma_mass, _, _ = np.histogram2d(radius, phi, bins=(r_edges, phi_edges), weights=mass)
    wz = 1.0 / np.cosh(z_centers / args.scale_height_kpc) ** 2
    wz /= wz.sum()
    mass3d = sigma_mass[:, :, None] * wz[None, None, :]
    mass3d *= args.stellar_mass / mass3d.sum()  # normalize grid total to the assumed M*
    density = mass3d / cylindrical_bin_volumes(spec)
    print(f"deprojected light into {mass3d.shape} grid; total M* set to {mass3d.sum():.3e} Msun "
          f"(M/L scaling: v_c ~ sqrt(M*/{args.stellar_mass:.0e}))")

    # ---- even-m Fourier x (R,z) representation + AGAMA potential ----
    model = fit_fourier_rz_from_grid(density, r_centers, z_centers, n_r=25, n_z_half=12,
                                     r_max=args.r_max_rep, z_max=args.z_max_rep)
    azh = fourier_rz_to_agama_density(model, total_mass=float(mass3d.sum()),
                                      r_max=args.r_max_rep, z_max=args.z_max_rep)
    pot = to_agama_potential(azh, r_max=30.0)
    mass_resid = abs(float(azh.totalMass()) - float(mass3d.sum())) / float(mass3d.sum())
    radii = np.linspace(0.3, 18.0, 60)
    vc = circular_velocity(pot, radii)
    i_peak = int(np.argmax(vc))
    print(f"AGAMA density totalMass {azh.totalMass():.3e} (mass residual {mass_resid:.1e}); "
          f"v_c peak {vc[i_peak]:.0f} km/s at R={radii[i_peak]:.1f} kpc, v_c(10 kpc)~{np.interp(10, radii, vc):.0f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ngc4321_deprojection_metrics.json").write_text(json.dumps({
        "galaxy": "NGC4321", "distance_mpc": args.distance_mpc, "pixel_kpc": pix_kpc,
        "background_mjy_sr": bkg, "axis_ratio_ba": ba, "inclination_measured_deg": incl_meas,
        "inclination_used_deg": incl, "pa_pixel_deg": float(np.degrees(pa_pix)), "pa_onsky_deg": pa_sky,
        "scale_height_kpc": args.scale_height_kpc, "stellar_mass_msun": float(mass3d.sum()),
        "mass_residual": mass_resid, "vc_peak_kms": float(vc[i_peak]),
        "vc_peak_radius_kpc": float(radii[i_peak]), "vc_radii_kpc": radii.tolist(), "vc_kms": vc.tolist(),
    }, indent=2, sort_keys=True), encoding="utf-8")

    # ---- figure: observed image | deprojected face-on | edge-on density | rotation curve ----
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    half = 0.5 * min(ny, nx) * pix_kpc
    vmax = float(np.percentile(light[light > 0], 99.9))
    ax[0].imshow(light, origin="lower", cmap="bone", norm=LogNorm(vmin=vmax * 3e-3, vmax=vmax),
                 extent=[(0 - cx) * pix_kpc, (nx - cx) * pix_kpc, (0 - cy) * pix_kpc, (ny - cy) * pix_kpc])
    ax[0].plot([0, 30 * pix_kpc * cpa], [0, 30 * pix_kpc * spa], "c-", lw=1)  # major axis
    ax[0].set_xlim(-half, half)
    ax[0].set_ylim(-half, half)
    ax[0].set_title("S4G 3.6um (bkg-subtracted)\nNGC 4321")
    ax[0].set_xlabel("x [kpc]")
    ax[0].set_ylabel("y [kpc]")

    gx = np.linspace(-args.r_max_rep, args.r_max_rep, 240)
    xx2, yy2 = np.meshgrid(gx, gx, indexing="ij")
    faceon = reconstruct_fourier_rz(np.column_stack([xx2.ravel(), yy2.ravel(), np.zeros(xx2.size)]), model)
    faceon = faceon.reshape(xx2.shape)
    fmax = float(faceon.max())
    ax[1].imshow(faceon.T, origin="lower", extent=[-args.r_max_rep, args.r_max_rep] * 2, cmap="magma",
                 norm=LogNorm(vmin=fmax * 3e-3, vmax=fmax))
    ax[1].set_title("deprojected face-on density\n(Fourier x (R,z), z=0)")
    ax[1].set_xlabel("x [kpc] (bar axis)")
    ax[1].set_ylabel("y [kpc]")

    xs = np.linspace(-args.r_max_rep, args.r_max_rep, 240)
    zs = np.linspace(-2.5, 2.5, 120)
    xe, ze = np.meshgrid(xs, zs, indexing="ij")
    edge = azh.density(np.column_stack([xe.ravel(), np.zeros(xe.size), ze.ravel()])).reshape(xe.shape)
    emax = float(edge.max())
    ax[2].imshow(edge.T, origin="lower", extent=[-args.r_max_rep, args.r_max_rep, -2.5, 2.5], cmap="magma",
                 aspect="auto", norm=LogNorm(vmin=emax * 3e-3, vmax=emax))
    ax[2].set_title(f"edge-on density (AGAMA)\nsech^2 h={args.scale_height_kpc} kpc, mass conserved")
    ax[2].set_xlabel("x [kpc]")
    ax[2].set_ylabel("z [kpc]")

    ax[3].plot(radii, vc, "-", color="#c44e52", lw=2)
    ax[3].set_xlabel("R [kpc]")
    ax[3].set_ylabel("v_c [km/s]")
    ax[3].set_ylim(0, None)
    ax[3].set_title(f"rotation curve from\npredicted potential (M*={mass3d.sum():.1e})")
    ax[3].grid(alpha=0.3)

    fig.suptitle("NGC 4321 (M100): real S4G image deprojected with our image-anchored Fourier x (R,z) "
                 f"method -> AGAMA potential  (i={incl:.0f} deg, baseline vertical)", fontsize=12)
    fig_path = args.output_dir / "ngc4321_deprojection.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output_dir / 'ngc4321_deprojection_metrics.json'} and {fig_path}")


if __name__ == "__main__":
    main()
