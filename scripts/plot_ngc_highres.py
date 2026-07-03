"""High-resolution deprojections via regrid(): NGC 4321 + NGC 4371.

Regrids each deprojection to 256x192x128 (radial bin 0.26 kpc at R=10, dz=0.078 kpc,
azimuthal arc 0.33 kpc at R=10) and renders face-on / edge-on surface density in the bar
frame plus the rotation curve (native grid overlaid -- the two must agree, the fine grid
just resolves more structure).

Run: PYTHONPATH=src .venv/bin/python scripts/plot_ngc_highres.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from dgdp import deproject
from dgdp.density3d import make_cylindrical_grid_spec

sys.path.insert(0, "scripts")
from plot_ngc4321_figures import make_lookup, project

GALAXIES = {
    "NGC4321": dict(image="NGC4321_m_c_r_f.fits", distance_mpc=15.2, inclination_deg=30.0,
                    pa_onsky_deg=153.0, stellar_mass=6.0e10),
    "NGC4371": dict(image="NGC4371/final/NGC4371.fits", distance_mpc=16.194,
                    inclination_deg=58.0, pix_arcsec=0.75, pa_pix_deg=1.8,
                    center=(254.6, 152.8), mask="NGC4371/final/NGC4371_mask.fits",
                    stellar_mass=3.53e10),
}
N_R, N_PHI, N_Z = 256, 192, 128
V_RADII = np.linspace(0.3, 18.0, 80)


def bar_azimuth(res):
    sig = (res.density_3d * res._vol).sum(axis=2)
    sel = (res.grid["r"] > 2.0) & (res.grid["r"] < 5.5)
    return float(-np.angle(np.sum(np.fft.rfft(sig, axis=1)[:, 2][sel])) / 2.0)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.35, 1.0]})
    for row, (name, cfg) in enumerate(GALAXIES.items()):
        img = cfg.pop("image")
        res = deproject(img, ml=1.0, **cfg)
        t0 = time.time()
        fine = res.regrid(n_r=N_R, n_phi=N_PHI, n_z=N_Z)
        g = res._ctx["grid"]
        spec = make_cylindrical_grid_spec(r_min_kpc=g["r_min"], r_max_kpc=g["r_max"],
                                          n_r=N_R, n_phi=N_PHI, z_max_kpc=g["z_max"], n_z=N_Z)
        theta_b = bar_azimuth(fine)
        ctb, stb = np.cos(theta_b), np.sin(theta_b)
        base_lk = make_lookup(fine.density_3d, spec.r_edges_kpc, spec.phi_edges_rad,
                              spec.z_edges_kpc)

        def lk(x, y, z):                       # display (bar) frame -> grid frame
            return base_lk(x * ctb - y * stb, x * stb + y * ctb, z)

        xs = np.linspace(-16, 16, 440)
        face = project(lk, xs, xs, np.linspace(-5, 5, 128), along="z")
        zs = np.linspace(-4, 4, 220)
        edge = project(lk, xs, zs, np.linspace(-16, 16, 300), along="y")
        vc_f = fine.v_circ(V_RADII)
        vc_n = res.v_circ(V_RADII)
        print(f"{name}: regrid+render {time.time()-t0:.0f}s  vc peak fine "
              f"{vc_f.max():.1f} / native {vc_n.max():.1f} km/s")

        # percentile scale: the fine grid resolves the nucleus peak, so .max() would bury
        # the disk at the bottom of the log range
        fmax = float(np.percentile(face[face > 0], 99.7))
        axes[row, 0].imshow(face.T, origin="lower", extent=[-16, 16, -16, 16], cmap="magma",
                            norm=LogNorm(vmin=fmax * 1e-3, vmax=fmax))
        axes[row, 0].set(title=f"{name} face-on (bar on x)", xlabel="x [kpc]", ylabel="y [kpc]")
        axes[row, 0].set_aspect("equal")

        emax = float(np.percentile(edge[edge > 0], 99.7))
        axes[row, 1].imshow(edge.T, origin="lower", extent=[-16, 16, -4, 4], cmap="magma",
                            aspect="auto", norm=LogNorm(vmin=emax * 1e-3, vmax=emax))
        axes[row, 1].set(title=f"{name} edge-on (bar side-on)", xlabel="x [kpc]", ylabel="z [kpc]")

        axes[row, 2].plot(V_RADII, vc_f, color="#c44e52", lw=2,
                          label=f"fine {N_R}x{N_PHI}x{N_Z}")
        axes[row, 2].plot(V_RADII, vc_n, "--", color="#4c72b0", lw=1.5, label="native 64x48x32")
        axes[row, 2].set(xlim=(0, 18), ylim=(0, None), title=f"{name} rotation curve",
                         xlabel="R [kpc]", ylabel="v_c [km/s]")
        axes[row, 2].legend(fontsize=8)

    fig.suptitle(f"High-resolution deprojections via regrid(): {N_R}x{N_PHI}x{N_Z} "
                 "(radial 0.26 kpc @ R=10, dz 0.078 kpc)", fontsize=13)
    out = Path("outputs/real_images/ngc_highres_deprojection.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
