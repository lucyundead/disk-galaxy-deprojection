"""Quick check: swap the TNG-learned vertical profile for a Comeron+2018 thin+thick sech^2
scale-height prior (real S4G edge-on galaxies), and compare the bar-side-on edge-on image
to the TNG-learned one. Reuses the saved deprojection (ngc4321_arrays.npz): SAME in-plane
image anchor Sigma(R,phi); only the vertical profile differs.

Comeron+2018 disks are ~non-flaring, so h_z is constant in R (a function of M* only); the
defaults are representative thin/thick values for M* ~ 6e10 Msun (override via CLI). Note
their h_z is a DISK measurement -- the bulge/bar is a separate, thicker component, so this
prior is a disk-level lower bound in the bar region.

Run:
    .venv/bin/python scripts/ngc4321_comeron_vs_learned.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from plot_ngc4321_figures import bar_azimuth, make_lookup, project


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, default=Path("outputs/real_images/ngc4321_arrays.npz"))
    ap.add_argument("--h-thin-kpc", type=float, default=0.4, help="Comeron+2018 thin-disk h_z (M*~6e10)")
    ap.add_argument("--h-thick-kpc", type=float, default=1.2, help="Comeron+2018 thick-disk h_z")
    ap.add_argument("--thick-frac", type=float, default=0.3, help="thick-disk mass fraction")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/real_images"))
    args = ap.parse_args()

    d = np.load(args.arrays)
    r_grid, z = d["r_grid"], d["z_grid"]
    spec = make_cylindrical_grid_spec(z_max_kpc=5.0, n_z=32, n_r=len(r_grid))
    r_e, p_e, z_e = spec.r_edges_kpc, spec.phi_edges_rad, spec.z_edges_kpc
    vol = cylindrical_bin_volumes(spec)

    # Comeron thin+thick vertical profile (normalized) x the SAME image anchor Sigma(R,phi)
    wt = 1.0 / np.cosh(z / args.h_thin_kpc) ** 2
    wk = 1.0 / np.cosh(z / args.h_thick_kpc) ** 2
    wz = (1.0 - args.thick_frac) * (wt / wt.sum()) + args.thick_frac * (wk / wk.sum())
    sigma = d["baseline_mass"].sum(axis=2)            # image anchor: column mass per (R,phi)
    comeron_mass = sigma[:, :, None] * wz[None, None, :]
    learned_mass = d["learned_mass"]

    tb = bar_azimuth(d["baseline_mass"], r_grid)
    ct, st = np.cos(tb), np.sin(tb)

    def rotlk(density):                               # display (bar on x) -> grid lookup
        lk = make_lookup(density / vol, r_e, p_e, z_e)
        return lambda x, y, zz: lk(x * ct - y * st, x * st + y * ct, zz)

    xs, zs, los = np.linspace(-16, 16, 240), np.linspace(-3, 3, 140), np.linspace(-16, 16, 140)
    edge_c = project(rotlk(comeron_mass), xs, zs, los, along="y")
    edge_l = project(rotlk(learned_mass), xs, zs, los, along="y")

    def rms_eff(mass):
        rms = np.sqrt((mass.sum(1) * z[None] ** 2).sum(1) / np.maximum(mass.sum((1, 2)), 1e-30))
        w = mass.sum((1, 2))[r_grid < 12]
        return rms, float((rms[r_grid < 12] * w).sum() / w.sum())

    rms_c, eff_c = rms_eff(comeron_mass)
    rms_l, eff_l = rms_eff(learned_mass)
    print(f"Comeron thin+thick (h={args.h_thin_kpc}/{args.h_thick_kpc} kpc, f_thick={args.thick_frac}): "
          f"RMS|z|(R<12)={eff_c:.2f} kpc  vs  TNG-learned {eff_l:.2f} kpc  (learned/Comeron = {eff_l / eff_c:.1f}x)")

    vmax = float(max(edge_c.max(), edge_l.max()))
    norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    for a, m, t in ((ax[0], edge_c, f"Comeron+2018 thin+thick prior\nh={args.h_thin_kpc}/{args.h_thick_kpc} kpc, "
                     f"f_thick={args.thick_frac} (RMS|z| {eff_c:.2f})"),
                    (ax[1], edge_l, f"TNG-learned q_m\n(RMS|z| {eff_l:.2f}, flaring)")):
        im = a.imshow(m.T, origin="lower", extent=[-16, 16, -3, 3], cmap="magma", aspect="auto", norm=norm)
        a.set_title(t, fontsize=10)
        a.set_xlabel("x [kpc] (bar)")
        a.set_ylabel("z [kpc]")
        a.set_xlim(-12, 12)
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="$\\Sigma$ (shared)")
    ax[2].plot(r_grid, rms_c, color="#2ca02c", lw=2, label=f"Comeron prior ({eff_c:.2f} kpc)")
    ax[2].plot(r_grid, rms_l, color="#c44e52", lw=2, label=f"TNG-learned ({eff_l:.2f} kpc)")
    ax[2].set_xlim(0, 16)
    ax[2].set_xlabel("R [kpc]")
    ax[2].set_ylabel("RMS |z| [kpc]")
    ax[2].set_title("vertical thickness vs R\n(Comeron ~flat; learned flares)")
    ax[2].legend(fontsize=8)
    fig.suptitle("NGC 4321 edge-on (bar side-on): published thin+thick scale-height prior vs TNG-learned "
                 "(same in-plane $\\Sigma$, shared scale)", fontsize=12)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "ngc4321_comeron_vs_learned.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
