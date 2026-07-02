"""Load a galaxy image, resolve its geometry, and build the geometric sech^2 baseline density.

Ported from ``scripts/deproject_real_image_ngc4321_learned.py`` (``deproject_ngc4321`` +
the WCS position-angle helpers), split into I/O+geometry (``load_image``) and deprojection
(``geometric_baseline``). No training, pure numpy + astropy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dgdp.density3d import cylindrical_bin_volumes

ARCSEC_PER_RAD = 206264.806


def _onsky_pa(wcs, cx, cy, pa_pix):
    c0 = wcs.pixel_to_world(cx, cy)
    c1 = wcs.pixel_to_world(cx + 50.0 * np.cos(pa_pix), cy + 50.0 * np.sin(pa_pix))
    return float(c0.position_angle(c1).deg)


def _pixel_pa_from_onsky(wcs, cx, cy, pa_sky_deg):
    grid = np.radians(np.linspace(0.0, 180.0, 721, endpoint=False))
    skies = np.array([_onsky_pa(wcs, cx, cy, g) % 180.0 for g in grid])
    diff = np.abs((skies - (pa_sky_deg % 180.0) + 90.0) % 180.0 - 90.0)
    return float(grid[int(np.argmin(diff))])


@dataclass
class GalaxyImage:
    light: np.ndarray          # background-subtracted, non-negative
    cx: float
    cy: float
    pix_kpc: float
    pa_pix: float              # disk major-axis PA in the pixel frame [rad, math from +x]
    incl_deg: float
    bkg: float
    bkg_std: float


def load_image(source, *, distance_mpc, inclination_deg, pix_arcsec=None, pa_pix_deg=None,
               pa_onsky_deg=None, center=None, mask=None) -> GalaxyImage:
    """Read a FITS path or 2-D array, subtract background, resolve centre/PA/scale.

    Geometry: give ``pix_arcsec`` (+ ``pa_pix_deg``) to bypass WCS (e.g. an S4G cutout); else a
    FITS with a WCS supplies the pixel scale and lets ``pa_onsky_deg`` be converted to the pixel
    frame. Arrays require ``pix_arcsec`` and ``pa_pix_deg`` (no WCS available).
    """
    wcs = None
    if isinstance(source, (str, Path)):
        from astropy.io import fits
        with fits.open(source) as hdul:
            data = np.asarray(hdul[0].data, dtype=float)
            header = hdul[0].header
        if pix_arcsec is None:
            from astropy.wcs import WCS
            wcs = WCS(header)
            cd = np.array([[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]])
            pix_arcsec = float(np.sqrt(np.abs(np.linalg.det(cd))) * 3600.0)
    else:
        data = np.asarray(source, dtype=float)
        if pix_arcsec is None:
            raise ValueError("pix_arcsec is required when passing an array (no WCS)")

    pix_kpc = pix_arcsec / ARCSEC_PER_RAD * (distance_mpc * 1e3)

    if mask is not None:
        from astropy.io import fits
        with fits.open(mask) as mh:
            data = np.where(np.asarray(mh[0].data, dtype=float) > 0, np.nan, data)

    border = np.concatenate([data[0], data[-1], data[:, 0], data[:, -1]])
    bkg, bkg_std = float(np.nanmedian(border)), float(np.nanstd(border))
    light = np.clip(np.nan_to_num(data, nan=bkg) - bkg, 0.0, None)

    ny, nx = light.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    if center is not None:
        cx, cy = float(center[0]), float(center[1])
    else:
        iy0, ix0 = np.unravel_index(np.argmax(light), light.shape)
        win = (np.hypot(xx - ix0, yy - iy0) < 80) & (light > 5 * bkg_std)
        cx = float((light[win] * xx[win]).sum() / light[win].sum())
        cy = float((light[win] * yy[win]).sum() / light[win].sum())

    if pa_pix_deg is not None:
        pa_pix = np.radians(pa_pix_deg)
    elif pa_onsky_deg is not None:
        if wcs is None:
            raise ValueError("pa_onsky_deg needs a FITS WCS; pass pa_pix_deg instead")
        pa_pix = _pixel_pa_from_onsky(wcs, cx, cy, pa_onsky_deg)
    else:
        raise ValueError("provide pa_pix_deg or pa_onsky_deg")

    return GalaxyImage(light, cx, cy, pix_kpc, pa_pix, float(inclination_deg), bkg, bkg_std)


def geometric_baseline(gi: GalaxyImage, spec, *, scale_height_kpc, stellar_mass) -> dict:
    """Geometric deprojection: disk-plane Sigma(R,phi) x sech^2(z/h) -> baseline density grid,
    plus the observed image resampled into the 192x192 TNG mock format (major axis -> axis 1)."""
    ny, nx = gi.light.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    sel = gi.light > 2 * gi.bkg_std
    # 4x4 subpixel deposit: the inner log-R rings are narrower than a pixel, so point deposits
    # there make the anchor harmonics phi-deltas (|2 Sigma_m|/Sigma_0 -> 2) and Sigma(R) a comb
    off = (np.arange(4) + 0.5) / 4.0 - 0.5
    ox, oy = np.meshgrid(off, off)
    x_sky = ((xx[sel][:, None] + ox.ravel()).ravel() - gi.cx) * gi.pix_kpc
    y_sky = ((yy[sel][:, None] + oy.ravel()).ravel() - gi.cy) * gi.pix_kpc
    mass = np.repeat(gi.light[sel].astype(float) / 16.0, 16)
    cpa, spa = np.cos(gi.pa_pix), np.sin(gi.pa_pix)
    x_major = x_sky * cpa + y_sky * spa
    y_minor = -x_sky * spa + y_sky * cpa

    y_disk = y_minor / max(np.cos(np.radians(gi.incl_deg)), 1e-3)
    radius = np.hypot(x_major, y_disk)
    phi = np.arctan2(y_disk, x_major)
    r_edges, phi_edges, z_edges = spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    sigma_mass, _, _ = np.histogram2d(radius, phi, bins=(r_edges, phi_edges), weights=mass)
    wz = 1.0 / np.cosh(z_centers / scale_height_kpc) ** 2
    wz /= wz.sum()
    mass3d = sigma_mass[:, :, None] * wz[None, None, :]
    mass3d *= stellar_mass / mass3d.sum()
    baseline_density = mass3d / cylindrical_bin_volumes(spec)

    fov = 0.5 * 192 * 0.35
    edges = np.linspace(-fov, fov, 193)
    img_tng, _, _ = np.histogram2d(y_minor, x_major, bins=(edges, edges), weights=mass)
    return {"baseline_density": baseline_density, "image_tng": img_tng.astype(np.float32),
            "M_star": float(mass3d.sum()),
            "sigma_mass": sigma_mass * (stellar_mass / sigma_mass.sum()),
            "image_edges_kpc": edges}
