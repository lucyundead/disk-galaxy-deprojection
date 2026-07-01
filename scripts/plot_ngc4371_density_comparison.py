"""Face-on and edge-on density: our TNG-learned NGC 4371 deprojection vs Behzad's MGE.

Same real galaxy, two reconstructions. Surface mass density (M/L=1, Msun/kpc^2):
  face-on  Sigma(x,y) = int rho dz   (line of sight along z)
  edge-on  Sigma(x,z) = int rho dy   (line of sight along y -- a projection, not a slice)
Both are line-of-sight integrals of the 3D density (the learned cylindrical grid is
interpolated, the MGE is analytic), so the two columns are directly comparable. The bar is
put along x in both: the MGE is already in that intrinsic frame; the learned grid is
de-rotated by its measured m=2 bar phase. Shared colour scale per row.

Run: PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313 \\
        .venv/bin/python scripts/plot_ngc4371_density_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, "scripts")
try:
    import agama
except ImportError:
    sys.path.insert(0, "/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313")
    import agama
from ngc4371_mge_benchmark import build_density

_DENS = None


def dens_eval(x, y, zc):
    """MGE density on a (broadcastable) cartesian grid."""
    global _DENS
    if _DENS is None:
        _DENS = build_density(ml=1.0)
    xb, yb, zb = np.broadcast_arrays(x, y, zc)
    return _DENS.density(np.column_stack([xb.ravel(), yb.ravel(), zb.ravel()])).reshape(xb.shape)


def main():
    agama.setUnits(mass=1, length=1, velocity=1)
    d = Path("outputs/real_images")
    a = np.load(d / "ngc4371_arrays.npz")
    lm = a["learned_mass"]                       # (R, phi, z) cell mass
    lm = 0.5 * (lm + lm[:, :, ::-1])             # symmetrise in z (vertical structure is symmetric)
    r, phi, z = a["r_grid"], a["phi_centers"], a["z_grid"]
    nphi, dz = len(phi), float(z[1] - z[0])

    # bar position angle from the m=2 phase over the bar radial range
    sig = lm.sum(axis=2)
    bar = (r > 1) & (r < 6)
    phibar = -0.5 * np.angle((sig[bar] * np.exp(-2j * phi)[None, :]).sum())

    # cell volumes from geometric-midpoint edges (exact for a log-R grid) -> density
    redge = np.concatenate([[r[0] ** 2 / np.sqrt(r[0] * r[1])], np.sqrt(r[:-1] * r[1:]),
                            [r[-1] ** 2 / np.sqrt(r[-2] * r[-1])]])
    ann_over_nphi = np.pi * (redge[1:] ** 2 - redge[:-1] ** 2) / nphi   # cell base area (R,)
    rho = lm / (ann_over_nphi[:, None, None] * dz)
    sigcyl = sig / ann_over_nphi[:, None]
    taper = np.clip((13.0 - r) / 2.0, 0.0, 1.0)   # fade the unreliable large-R reconstruction (R>11)
    rho *= taper[:, None, None]
    sigcyl *= taper[:, None]

    # periodic-phi interpolators (look up learned angle = display angle + phibar)
    phi_ext = np.concatenate([phi, [phi[0] + 2 * np.pi]])
    fo_i = RegularGridInterpolator((r, phi_ext), np.concatenate([sigcyl, sigcyl[:, :1]], axis=1),
                                   bounds_error=False, fill_value=0.0)
    rho_i = RegularGridInterpolator((r, phi_ext, z), np.concatenate([rho, rho[:, :1, :]], axis=1),
                                    bounds_error=False, fill_value=0.0)

    ext = 12.0
    xm = np.linspace(-ext, ext, 120)   # even -> no column exactly at x=0 (avoids the axis seam)
    zm = np.linspace(-4, 4, 60)

    def wrap_phi(ang):  # map into the grid's phi range [phi[0], phi[0]+2pi)
        return phi[0] + np.mod(ang + phibar - phi[0], 2 * np.pi)

    X, Y = np.meshgrid(xm, xm, indexing="ij")
    fo_learn = fo_i((np.hypot(X, Y), wrap_phi(np.arctan2(Y, X))))
    Xe, Ze = np.meshgrid(xm, zm, indexing="ij")
    yint = np.linspace(-ext, ext, 97)
    fo_mge = np.zeros_like(X)
    eo_learn = np.zeros_like(Xe)
    eo_mge = np.zeros_like(Xe)
    for zv in np.linspace(-6, 6, 73):
        fo_mge += dens_eval(X, Y, zv)
    fo_mge *= (12.0 / 72)
    for yv in yint:
        eo_learn += rho_i((np.hypot(Xe, yv), wrap_phi(np.arctan2(yv, Xe)), Ze))
        eo_mge += dens_eval(Xe, yv, Ze)
    eo_learn *= (yint[1] - yint[0])
    eo_mge *= (yint[1] - yint[0])

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    def show(axx, img, extent, vmax, ttl, ylab):
        axx.imshow(img.T, origin="lower", extent=extent, cmap="magma", aspect="auto",
                   norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax))
        axx.set(title=ttl, xlabel="x [kpc]", ylabel=ylab)

    def vmx(*imgs):  # high percentile so the sharp nucleus doesn't saturate the colour scale
        return max(np.percentile(im[im > 0], 99.5) for im in imgs)

    fo_v, eo_v = vmx(fo_learn, fo_mge), vmx(eo_learn, eo_mge)
    show(ax[0, 0], fo_learn, [-ext, ext, -ext, ext], fo_v, "face-on Σ(x,y) — TNG-learned q_m", "y [kpc]")
    show(ax[0, 1], fo_mge, [-ext, ext, -ext, ext], fo_v, "face-on Σ(x,y) — MGE (Behzad)", "y [kpc]")
    show(ax[1, 0], eo_learn, [-ext, ext, -4, 4], eo_v, "edge-on Σ(x,z) — TNG-learned q_m", "z [kpc]")
    show(ax[1, 1], eo_mge, [-ext, ext, -4, 4], eo_v, "edge-on Σ(x,z) — MGE (Behzad)", "z [kpc]")
    fig.suptitle("NGC 4371 deprojected density: TNG-learned q_m vs MGE (bar along x, M/L=1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(d / "ngc4371_density_faceon_edgeon.png", dpi=130)
    print(f"learned bar phase = {np.degrees(phibar):.1f} deg; wrote {d}/ngc4371_density_faceon_edgeon.png")


if __name__ == "__main__":
    main()
