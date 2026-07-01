"""Figures for the NGC 4321 deprojection (loads arrays from
deproject_real_image_ngc4321_learned.py; no retrain). Geometry from the S4G data release
(i=34.6 deg, disk PA=158.2 deg); the deprojection frame has the BAR along the x axis.

(A) Edge-on thin baseline vs flaring-thick learned disk on a SHARED color scale + RMS|z|(R).
(B) observed S4G | geometric deprojection (face-on) | TNG-learned deprojection (face-on),
    with the bar/oval in red; and BELOW the two deprojections, their edge-on (bar side-on)
    views -- geometric sech^2 vs learned flaring.

Run:
    .venv/bin/python scripts/plot_ngc4321_figures.py
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
from matplotlib.colors import LogNorm

from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec


def make_lookup(density, r_edges, phi_edges, z_edges):
    """Nearest-cell density(X,Y,Z) lookup for a cylindrical grid (Cartesian inputs)."""
    def f(x, y, z):
        radius, phi = np.hypot(x, y), np.arctan2(y, x)
        i_r = np.clip(np.searchsorted(r_edges, radius) - 1, 0, len(r_edges) - 2)
        i_p = np.clip(np.searchsorted(phi_edges, phi) - 1, 0, len(phi_edges) - 2)
        i_z = np.searchsorted(z_edges, z) - 1
        valid = (radius < r_edges[-1]) & (i_z >= 0) & (i_z < len(z_edges) - 1)
        return np.where(valid, density[i_r, i_p, np.clip(i_z, 0, len(z_edges) - 2)], 0.0)
    return f


def project(lookup, axis_a, axis_b, los, *, along):
    """Surface density by integrating the LOS coordinate. along='y' -> edge-on (a=x,b=z);
    along='z' -> face-on (a=x,b=y)."""
    aa, bb = np.meshgrid(axis_a, axis_b, indexing="ij")
    out = np.zeros_like(aa)
    d_los = los[1] - los[0]
    for s in los:
        full = np.full_like(aa, s)
        out += (lookup(aa, full, bb) if along == "y" else lookup(aa, bb, full)) * d_los
    return out


def bar_azimuth(baseline_mass, r_grid):
    """Grid azimuth of the bar (radians) from the m=2 phase of Sigma(R,phi) in the bar region."""
    sig = baseline_mass.sum(axis=2)
    sel = (r_grid > 2.0) & (r_grid < 5.5)
    c2 = np.fft.rfft(sig, axis=1)[:, 2][sel]
    return float(-np.angle(np.sum(c2)) / 2.0)


def deproject_pixels(light, cx, cy, pix_kpc, pa_pix, incl, theta_b, thresh, extent, nbin):
    """Full-resolution geometric deprojection of the observed pixels into the bar frame (bar on x)."""
    ny, nx = light.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    sel = light > thresh
    x_sky, y_sky, w = (xx[sel] - cx) * pix_kpc, (yy[sel] - cy) * pix_kpc, light[sel]
    cpa, spa = np.cos(pa_pix), np.sin(pa_pix)
    x_major = x_sky * cpa + y_sky * spa
    y_minor = -x_sky * spa + y_sky * cpa
    y_disk = y_minor / max(np.cos(np.radians(incl)), 1e-3)
    ctb, stb = np.cos(theta_b), np.sin(theta_b)
    xb = x_major * ctb + y_disk * stb   # rotate disk frame by -theta_b -> bar on x
    yb = -x_major * stb + y_disk * ctb
    edges = np.linspace(-extent, extent, nbin + 1)
    hist, _, _ = np.histogram2d(xb, yb, bins=(edges, edges), weights=w)
    return hist, 0.5 * (edges[:-1] + edges[1:])


def bar_to_sky(bx, by, cx, cy, pix_kpc, pa_pix, incl, theta_b):
    """Project a bar-frame ellipse outline back onto the observed image (pixel coords)."""
    ctb, stb = np.cos(theta_b), np.sin(theta_b)
    x_major = bx * ctb - by * stb          # bar frame -> disk frame (rotate by +theta_b)
    y_disk = bx * stb + by * ctb
    y_minor = y_disk * np.cos(np.radians(incl))
    cpa, spa = np.cos(pa_pix), np.sin(pa_pix)
    x_sky = x_major * cpa - y_minor * spa   # disk -> sky
    y_sky = x_major * spa + y_minor * cpa
    return cx + x_sky / pix_kpc, cy + y_sky / pix_kpc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, default=Path("outputs/real_images/ngc4321_arrays.npz"))
    ap.add_argument("--fits", type=Path, default=Path("NGC4321_m_c_r_f.fits"))
    ap.add_argument("--a-bar-kpc", type=float, default=4.75, help="deprojected bar semi-major")
    ap.add_argument("--ba-bar", type=float, default=0.43, help="deprojected bar b/a")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/real_images"))
    args = ap.parse_args()

    d = np.load(args.arrays)
    spec = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32, n_r=len(d["r_grid"]))
    r_e, p_e, z_e = spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc
    vol = cylindrical_bin_volumes(spec)
    dens_b, dens_l = d["baseline_mass"] / vol, d["learned_mass"] / vol
    r_grid, z_grid = d["r_grid"], d["z_grid"]
    incl, pa_pix = float(d["incl_deg"]), float(d["pa_pix"])
    cx, cy, pix_kpc, bkg = float(d["cx"]), float(d["cy"]), float(d["pix_kpc"]), float(d["bkg"])
    theta_b = bar_azimuth(d["baseline_mass"], r_grid)  # grid azimuth of the bar
    ctb, stb = np.cos(theta_b), np.sin(theta_b)

    def rot_lookup(density):  # display (bar frame) -> grid (disk-major frame): rotate by +theta_b
        base = make_lookup(density, r_e, p_e, z_e)
        return lambda x, y, z: base(x * ctb - y * stb, x * stb + y * ctb, z)

    lk_b, lk_l = rot_lookup(dens_b), rot_lookup(dens_l)
    print(f"geometry: i={incl} deg, disk PA(pixel)={np.degrees(pa_pix):.1f}; measured bar azimuth "
          f"{np.degrees(theta_b):.1f} deg -> rotated onto x; bar a={args.a_bar_kpc} kpc, b/a={args.ba_bar}")

    # bar ellipse outline in the deprojected bar frame (bar along x)
    t = np.linspace(0, 2 * np.pi, 200)
    bx, by = args.a_bar_kpc * np.cos(t), args.a_bar_kpc * args.ba_bar * np.sin(t)

    # ---------- Figure A: edge-on (bar side-on), shared scale ----------
    xs = np.linspace(-18, 18, 240)
    zs = np.linspace(-4, 4, 120)
    los = np.linspace(-18, 18, 160)
    edge_b = project(lk_b, xs, zs, los, along="y")
    edge_l = project(lk_l, xs, zs, los, along="y")
    vmax = float(max(edge_b.max(), edge_l.max()))
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    rms_b = np.sqrt((d["baseline_mass"].sum(1) * z_grid[None] ** 2).sum(1)
                    / np.maximum(d["baseline_mass"].sum((1, 2)), 1e-30))
    rms_l = np.sqrt((d["learned_mass"].sum(1) * z_grid[None] ** 2).sum(1)
                    / np.maximum(d["learned_mass"].sum((1, 2)), 1e-30))
    figa, axa = plt.subplots(1, 3, figsize=(17, 4.3), constrained_layout=True)
    for ax, emap, ttl in ((axa[0], edge_b, "baseline (thin, sech$^2$ h=0.3)"),
                          (axa[1], edge_l, "learned (flaring, TNG q$_m$)")):
        im = ax.imshow(emap.T, origin="lower", extent=[-18, 18, -4, 4], cmap="magma", norm=norm, aspect="auto")
        ax.set_title(f"edge-on, bar side-on\n{ttl}")
        ax.set_xlabel("x [kpc] (bar)")
        ax.set_ylabel("z [kpc]")
    figa.colorbar(im, ax=axa[1], shrink=0.8, label="$\\Sigma$ (shared)")
    axa[2].plot(r_grid, rms_b, color="#4c72b0", lw=2, label="baseline")
    axa[2].plot(r_grid, rms_l, color="#c44e52", lw=2, label="learned")
    axa[2].set_xlim(0, 18)
    axa[2].set_xlabel("R [kpc]")
    axa[2].set_ylabel("RMS |z| [kpc]")
    axa[2].set_title("vertical thickness vs R")
    axa[2].legend(fontsize=8)
    figa.suptitle("NGC 4321 edge-on: thin baseline vs flaring-thick learned disk (shared scale)", fontsize=12)
    figa.savefig(args.output_dir / "ngc4321_edgeon_both_disks.png", dpi=150, bbox_inches="tight")
    plt.close(figa)

    # ---------- Figure B: observed | geo / learned face-on  + edge-on below ----------
    with fits.open(args.fits) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
    bkg_std = float(np.std(np.concatenate([data[0], data[-1], data[:, 0], data[:, -1]])))
    light = np.clip(data - bkg, 0.0, None)
    geo_full, gx = deproject_pixels(light, cx, cy, pix_kpc, pa_pix, incl, theta_b,
                                    2 * bkg_std, 16.0, 160)
    fxs = np.linspace(-16, 16, 240)
    face_l = project(lk_l, fxs, fxs, np.linspace(-5, 5, 64), along="z")
    ex_xs = np.linspace(-16, 16, 240)
    ex_zs = np.linspace(-3, 3, 120)
    ex_los = np.linspace(-16, 16, 140)
    edge_g = project(lk_b, ex_xs, ex_zs, ex_los, along="y")
    edge_l2 = project(lk_l, ex_xs, ex_zs, ex_los, along="y")

    fig = plt.figure(figsize=(16.5, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ny, nx = light.shape

    ax_obs = fig.add_subplot(gs[0, 0])
    ext = [(0 - cx) * pix_kpc, (nx - cx) * pix_kpc, (0 - cy) * pix_kpc, (ny - cy) * pix_kpc]
    ovmax = float(np.percentile(light[light > 0], 99.9))
    ax_obs.imshow(light, origin="lower", extent=[ext[0], ext[1], ext[2], ext[3]], cmap="bone",
                  norm=LogNorm(vmin=ovmax * 3e-3, vmax=ovmax))
    sx, sy = bar_to_sky(bx, by, 0.0, 0.0, 1.0, pa_pix, incl, theta_b)  # in kpc (centered)
    ax_obs.plot(sx, sy, "r-", lw=2)
    ax_obs.set_xlim(-15, 15)
    ax_obs.set_ylim(-15, 15)
    ax_obs.set_aspect("equal")
    ax_obs.set_title("observed S4G 3.6$\\mu$m\n(bar projected onto sky)")
    ax_obs.set_xlabel("x [kpc]")
    ax_obs.set_ylabel("y [kpc]")

    def faceon_panel(ax, fmap, gxv, ttl):
        fmax = float(fmap.max())
        ax.imshow(fmap.T, origin="lower", extent=[gxv.min(), gxv.max(), gxv.min(), gxv.max()],
                  cmap="magma", norm=LogNorm(vmin=fmax * 3e-3, vmax=fmax))
        ax.plot(bx, by, "r-", lw=2)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect("equal")
        ax.set_title(ttl)
        ax.set_xlabel("x [kpc] (bar)")
        ax.set_ylabel("y [kpc]")

    faceon_panel(fig.add_subplot(gs[0, 1]), geo_full, gx, "geometric deprojection (full-res)\nface-on, bar on x")
    faceon_panel(fig.add_subplot(gs[0, 2]), face_l, fxs, "TNG-learned deprojection\nface-on (in-plane = geometric)")

    for col, emap, ttl in ((1, edge_g, "geometric edge-on\n(baseline sech$^2$ h=0.3 kpc)"),
                           (2, edge_l2, "TNG-learned edge-on\n(flaring; RMS|z| 0.5->2 kpc)")):
        ax = fig.add_subplot(gs[1, col])
        emax = float(emap.max())
        ax.imshow(emap.T, origin="lower", extent=[-16, 16, -3, 3], cmap="magma", aspect="auto",
                  norm=LogNorm(vmin=emax * 3e-3, vmax=emax))
        ax.axvline(args.a_bar_kpc, color="cyan", ls=":", lw=0.8)
        ax.axvline(-args.a_bar_kpc, color="cyan", ls=":", lw=0.8)
        ax.set_xlim(-15, 15)
        ax.set_title(ttl)
        ax.set_xlabel("x [kpc] (bar, side-on)")
        ax.set_ylabel("z [kpc]")

    ax_txt = fig.add_subplot(gs[1, 0])
    ax_txt.axis("off")
    ax_txt.text(0.02, 0.5, "S4G geometry:\n  i = 34.6 deg\n  disk PA = 158.2 deg\n\n"
                f"bar (deprojected):\n  a = {args.a_bar_kpc} kpc\n  b/a = {args.ba_bar}\n"
                "  (rotated onto x)\n\nedge-on dashed lines:\n  bar semi-major +-a",
                fontsize=10, va="center", family="monospace")

    fig.suptitle("NGC 4321: observed -> deprojected (bar on x), with edge-on (bar side-on) views below. "
                 "Geometric & learned share the face-on; they differ only in the vertical (edge-on).",
                 fontsize=12)
    fig.savefig(args.output_dir / "ngc4321_deproject_faceon_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure C: rotation curve comparison (finer R, direct-sum, full radial resolution) ----------
    mt = json.loads((args.output_dir / "ngc4321_learned_vs_baseline_metrics.json").read_text())
    radii = np.array(mt["vc_radii_kpc"])
    figc, axc = plt.subplots(1, 1, figsize=(7.0, 5.0), constrained_layout=True)
    axc.plot(radii, mt["vc_baseline_kms"], "-", color="#4c72b0", lw=2.2, label="geometric baseline (sech$^2$ h=0.3)")
    axc.plot(radii, mt["vc_learned_kms"], "--", color="#c44e52", lw=2.2, label="TNG-learned (flaring)")
    axc.plot(radii, mt["vc_uniform_kms"], ":", color="#2ca02c", lw=2.0, label="uniform h=1 kpc (control)")
    axc.axvline(args.a_bar_kpc, color="gray", ls=":", lw=0.8)
    axc.set_xlabel("R [kpc]")
    axc.set_ylabel("v$_c$ [km/s]  (stellar, M$_*$=6e10)")
    axc.set_xlim(0, 18)
    axc.set_ylim(0, None)
    axc.set_title(f"NGC 4321 stellar rotation curve  ({int(mt['n_r_out'])} log R bins, direct-sum)\n"
                  "same image-anchored $\\Sigma$(R); only the vertical structure differs")
    axc.legend(fontsize=9)
    axc.grid(alpha=0.3)
    figc.savefig(args.output_dir / "ngc4321_rotation_curve.png", dpi=150, bbox_inches="tight")
    plt.close(figc)
    print(f"wrote {args.output_dir / 'ngc4321_edgeon_both_disks.png'}, "
          f"{args.output_dir / 'ngc4321_deproject_faceon_bar.png'} and "
          f"{args.output_dir / 'ngc4321_rotation_curve.png'}")


if __name__ == "__main__":
    main()
