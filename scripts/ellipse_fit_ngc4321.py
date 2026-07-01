"""Isophotal ellipse fitting of the real NGC 4321 S4G image to measure the bar.

Standard method (photutils.isophote.Ellipse, the IRAF `ellipse` algorithm used by the S4G
bar catalogs): fit ellipses to the surface-brightness isophotes -> ellipticity eps(a) and
position-angle PA(a) radial profiles. The bar is the local ellipticity MAXIMUM with a
roughly constant PA, ending where eps drops and PA twists toward the disk. Reports the
apparent (sky-plane) bar parameters and the deprojected (disk-frame) ones, and overlays the
bar isophote on the image.

Run:
    .venv/bin/python scripts/ellipse_fit_ngc4321.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse as EllipsePatch
from photutils.isophote import Ellipse, EllipseGeometry

from astropy import log as astropy_log

astropy_log.setLevel("ERROR")
ARCSEC_PER_RAD = 206264.806


def onsky_pa(wcs, cx, cy, pa_pix):
    c0 = wcs.pixel_to_world(cx, cy)
    c1 = wcs.pixel_to_world(cx + 50.0 * np.cos(pa_pix), cy + 50.0 * np.sin(pa_pix))
    return float(c0.position_angle(c1).deg)


def pixel_pa_from_onsky(wcs, cx, cy, pa_sky_deg):
    grid = np.radians(np.linspace(0.0, 180.0, 721, endpoint=False))
    skies = np.array([onsky_pa(wcs, cx, cy, g) % 180.0 for g in grid])
    diff = np.abs((skies - (pa_sky_deg % 180.0) + 90.0) % 180.0 - 90.0)
    return float(grid[int(np.argmin(diff))])


def deproject_ellipse(a_kpc, b_kpc, pa_bar, disk_pa, incl_deg, rng):
    """Deproject an apparent ellipse to intrinsic (A, B/A, PA-from-disk-major) via point cloud."""
    u, v = rng.uniform(-1, 1, 40000), rng.uniform(-1, 1, 40000)
    keep = u * u + v * v <= 1.0
    ex, ey = a_kpc * u[keep], b_kpc * v[keep]
    xs = ex * np.cos(pa_bar) - ey * np.sin(pa_bar)  # sky-plane
    ys = ex * np.sin(pa_bar) + ey * np.cos(pa_bar)
    x_major = xs * np.cos(disk_pa) + ys * np.sin(disk_pa)
    y_minor = -xs * np.sin(disk_pa) + ys * np.cos(disk_pa)
    x_disk, y_disk = x_major, y_minor / max(np.cos(np.radians(incl_deg)), 1e-3)
    cov = np.cov(np.vstack([x_disk, y_disk]))
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    semi = 2.0 * np.sqrt(np.maximum(evals[order], 0.0))
    vec = evecs[:, order[0]]
    return float(semi[0]), float(semi[1] / semi[0]), float(np.degrees(np.arctan2(vec[1], vec[0])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=Path, default=Path("NGC4321_m_c_r_f.fits"))
    ap.add_argument("--distance-mpc", type=float, default=15.2)
    ap.add_argument("--inclination-deg", type=float, default=30.0)
    ap.add_argument("--pa-onsky-deg", type=float, default=153.0)
    ap.add_argument("--pa0-bar-deg", type=float, default=30.0, help="initial bar PA guess (pixel frame)")
    ap.add_argument("--bar-search-min-kpc", type=float, default=1.0, help="exclude the nuclear core below this")
    ap.add_argument("--bar-search-kpc", type=float, default=8.0, help="max radius to look for the eps peak")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/real_images"))
    args = ap.parse_args()

    with fits.open(args.fits) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header
        wcs = WCS(header)
    cd = np.array([[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]])
    pix_kpc = float(np.sqrt(np.abs(np.linalg.det(cd))) * 3600.0) / ARCSEC_PER_RAD * (args.distance_mpc * 1e3)

    border = np.concatenate([data[0], data[-1], data[:, 0], data[:, -1]])
    bkg, bkg_std = float(np.median(border)), float(np.std(border))
    light = data - bkg
    iy0, ix0 = np.unravel_index(np.argmax(light), light.shape)
    ny, nx = light.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    win = (np.hypot(xx - ix0, yy - iy0) < 80) & (light > 5 * bkg_std)
    cx = float((light[win] * xx[win]).sum() / light[win].sum())
    cy = float((light[win] * yy[win]).sum() / light[win].sum())
    disk_pa = pixel_pa_from_onsky(wcs, cx, cy, args.pa_onsky_deg)
    print(f"center (x,y)=({cx:.1f},{cy:.1f}); pixel {pix_kpc:.4f} kpc; disk PA(pixel)={np.degrees(disk_pa):.1f} deg")

    # ---- isophote ellipse fit (image is already star-subtracted/refilled; sclip rejects residuals) ----
    geom = EllipseGeometry(x0=cx, y0=cy, sma=20.0, eps=0.3, pa=np.radians(args.pa0_bar_deg))
    iso = Ellipse(light, geometry=geom).fit_image(maxsma=320.0, sclip=3.0, nclip=3)
    sma = iso.sma * pix_kpc
    eps, eps_err = np.array(iso.eps), np.array(iso.ellip_err)
    pa = np.degrees(np.array(iso.pa)) % 180.0
    good = np.isfinite(sma) & np.isfinite(eps) & (sma > 0.2)
    sma, eps, eps_err, pa = sma[good], eps[good], eps_err[good], pa[good]
    print(f"fit {good.sum()} isophotes from {sma.min():.2f} to {sma.max():.1f} kpc")

    # ---- identify the bar: eps maximum in [min,max] (excludes the nuclear core), then drop/twist ----
    region = (sma > args.bar_search_min_kpc) & (sma < args.bar_search_kpc)
    i_peak = int(np.flatnonzero(region)[np.argmax(eps[region])])
    eps_bar, a_epsmax, pa_bar = float(eps[i_peak]), float(sma[i_peak]), float(pa[i_peak])
    a_bar_end = a_epsmax
    for j in range(i_peak + 1, len(sma)):
        if eps[j] < eps_bar - 0.1 or abs((pa[j] - pa_bar + 90) % 180 - 90) > 10.0:
            a_bar_end = float(sma[j])
            break
    A, ba, pa_intr = deproject_ellipse(
        a_epsmax, a_epsmax * (1 - eps_bar), np.radians(pa_bar), disk_pa,
        args.inclination_deg, np.random.default_rng(0))
    print("\nBAR (ellipticity-maximum method):")
    inner = sma < args.bar_search_min_kpc
    if inner.any():
        j = int(np.flatnonzero(inner)[np.argmax(eps[inner])])
        print(f"  (nuclear bar/disk: a={sma[j]:.2f} kpc, eps={eps[j]:.2f}, PA={pa[j]:.0f} deg)")
    print(f"  apparent : a(eps-max)={a_epsmax:.2f} kpc, eps={eps_bar:.2f} (b/a={1 - eps_bar:.2f}), "
          f"PA(sky-pixel)={pa_bar:.0f} deg; bar-end (eps drop / PA twist) a={a_bar_end:.2f} kpc")
    print(f"  deprojected: a={A:.2f} kpc, b/a={ba:.2f} (eps={1 - ba:.2f}), "
          f"PA={pa_intr:.0f} deg from disk major axis (disk i={args.inclination_deg:.0f})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ngc4321_ellipse_fit_bar.json").write_text(json.dumps({
        "center_xy": [cx, cy], "pixel_kpc": pix_kpc, "disk_pa_pixel_deg": float(np.degrees(disk_pa)),
        "inclination_deg": args.inclination_deg,
        "bar_apparent": {"a_eps_max_kpc": a_epsmax, "eps": eps_bar, "ba": 1 - eps_bar,
                         "pa_sky_pixel_deg": pa_bar, "a_bar_end_kpc": a_bar_end},
        "bar_deprojected": {"a_kpc": A, "ba": ba, "eps": 1 - ba, "pa_from_disk_major_deg": pa_intr},
        "profile": {"sma_kpc": sma.tolist(), "eps": eps.tolist(), "pa_deg": pa.tolist()},
    }, indent=2), encoding="utf-8")

    # ---- figure: image + ellipses; eps(a); PA(a) ----
    fig = plt.figure(figsize=(16, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_im = fig.add_subplot(gs[:, 0])
    half = 220
    vmax = float(np.percentile(light[light > 0], 99.7))
    ax_im.imshow(light, origin="lower", cmap="bone", norm=LogNorm(vmin=vmax * 3e-3, vmax=vmax))
    for k in range(0, len(iso.sma), max(1, len(iso.sma) // 18)):
        s, e, p = iso.sma[k], iso.eps[k], iso.pa[k]
        if not (np.isfinite(s) and np.isfinite(e)) or s < 1:
            continue
        ax_im.add_patch(EllipsePatch((cx, cy), 2 * s, 2 * s * (1 - e), angle=np.degrees(p),
                                     fill=False, edgecolor="0.6", lw=0.5))
    s = a_epsmax / pix_kpc
    ax_im.add_patch(EllipsePatch((cx, cy), 2 * s, 2 * s * (1 - eps_bar), angle=pa_bar,
                                 fill=False, edgecolor="red", lw=2.2))
    ax_im.set_xlim(cx - half, cx + half)
    ax_im.set_ylim(cy - half, cy + half)
    ax_im.set_title(f"NGC 4321 S4G + fitted isophotes\nbar (red): a={a_epsmax:.1f} kpc, b/a={1 - eps_bar:.2f}")
    ax_im.set_xlabel("x [pixel]")
    ax_im.set_ylabel("y [pixel]")

    ax_e = fig.add_subplot(gs[0, 1:])
    ax_e.errorbar(sma, eps, yerr=eps_err, fmt="o-", ms=3, lw=1, color="#c44e52")
    ax_e.axvline(a_epsmax, color="red", ls="--", lw=1, label=f"eps-max a={a_epsmax:.1f} kpc")
    ax_e.axvline(a_bar_end, color="green", ls=":", lw=1, label=f"bar end a={a_bar_end:.1f} kpc")
    ax_e.set_ylabel("ellipticity  $\\epsilon = 1-b/a$")
    ax_e.set_xlim(0, 16)
    ax_e.set_ylim(0, max(0.7, eps_bar + 0.1))
    ax_e.legend(fontsize=8)
    ax_e.set_title("ellipse-fit profiles (bar = eps maximum, then eps drop + PA twist)")

    ax_p = fig.add_subplot(gs[1, 1:])
    ax_p.plot(sma, pa, "o-", ms=3, lw=1, color="#4c72b0")
    ax_p.axhline(np.degrees(disk_pa) % 180.0, color="k", ls="-", lw=0.8, label="disk PA")
    ax_p.axvline(a_epsmax, color="red", ls="--", lw=1)
    ax_p.axvline(a_bar_end, color="green", ls=":", lw=1)
    ax_p.set_xlabel("semi-major axis a [kpc]")
    ax_p.set_ylabel("PA [deg]")
    ax_p.set_xlim(0, 16)
    ax_p.legend(fontsize=8)

    fig.suptitle("Isophotal ellipse fitting of NGC 4321 (bar from the ellipticity-maximum method)", fontsize=13)
    fig_path = args.output_dir / "ngc4321_ellipse_fit_bar.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {args.output_dir / 'ngc4321_ellipse_fit_bar.json'} and {fig_path}")


if __name__ == "__main__":
    main()
