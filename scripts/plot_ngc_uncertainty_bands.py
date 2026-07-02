"""Posterior uncertainty bands for the NGC 4321 / NGC 4371 deprojections.

Draws n samples of the vertical-profile weights from the bundled MDN (needs the mixture
heads, i.e. a bundle trained after the K>1 export), reconstructs each, and plots the
16-84% and 5-95% bands of RMS|z|(R) and v_c(R) around the mixture-mean prediction.
Bands share the loop-corrected Sigma anchors: they quantify the vertical-profile
posterior at fixed in-plane surface density.

Run: PYTHONPATH=src .venv/bin/python scripts/plot_ngc_uncertainty_bands.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgdp import deproject

GALAXIES = {
    "NGC4321": dict(image="NGC4321_m_c_r_f.fits", distance_mpc=15.2, inclination_deg=30.0,
                    pa_onsky_deg=153.0, stellar_mass=6.0e10),
    "NGC4371": dict(image="NGC4371/final/NGC4371.fits", distance_mpc=16.194,
                    inclination_deg=58.0, pix_arcsec=0.75, pa_pix_deg=1.8,
                    center=(254.6, 152.8), mask="NGC4371/final/NGC4371_mask.fits",
                    stellar_mass=3.53e10),
}
N_SAMPLES = 48
R_RMS = np.linspace(0.3, 16.0, 60)
R_VC = np.linspace(0.3, 18.0, 40)


def band(ax, radii, samples, mean, color, ylabel, title):
    lo1, hi1 = np.percentile(samples, [16, 84], axis=0)
    lo2, hi2 = np.percentile(samples, [5, 95], axis=0)
    ax.fill_between(radii, lo2, hi2, color=color, alpha=0.18, lw=0, label="5-95%")
    ax.fill_between(radii, lo1, hi1, color=color, alpha=0.35, lw=0, label="16-84%")
    ax.plot(radii, mean, color=color, lw=2, label="mixture mean")
    ax.set(xlabel="R [kpc]", ylabel=ylabel, title=title, xlim=(0, radii[-1]))
    ax.legend(fontsize=8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, (name, cfg) in enumerate(GALAXIES.items()):
        img = cfg.pop("image")
        res = deproject(img, ml=1.0, n_samples=N_SAMPLES, seed=20260702, **cfg)
        band(axes[row, 0], R_RMS, res.scale_height_samples(R_RMS), res.scale_height(R_RMS),
             "#c44e52", "h_z [kpc]  (ρ ∝ sech²(z/h_z))",
             f"{name}: sech² scale height (i={cfg['inclination_deg']:.0f}°)")
        axes[row, 0].axhline(0.3, color="#4c72b0", lw=1, ls=":", label="_")
        band(axes[row, 1], R_VC, res.v_circ_samples(R_VC), res.v_circ(R_VC), "#55a868",
             "v_c [km/s]", f"{name}: rotation curve")
        axes[row, 1].set_ylim(0, None)
        print(f"{name}: reproj residual {res.reproj['history'][0]:.3f} -> "
              f"{min(res.reproj['history']):.3f}  h_z(2) "
              f"{np.percentile(res.scale_height_samples([2.0]), [16, 84]).round(2)}")
    fig.suptitle(f"Posterior bands ({N_SAMPLES} MDN draws; Sigma anchors fixed to the "
                 "mean prediction's reprojection-corrected values)", fontsize=11)
    out = Path("outputs/real_images/ngc_uncertainty_bands.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
