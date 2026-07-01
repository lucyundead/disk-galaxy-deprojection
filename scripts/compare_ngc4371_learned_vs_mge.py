"""Compare our TNG-learned q_m deprojection of NGC 4371 to Behzad's independent MGE
deprojection of the same galaxy (the benchmark). Tests whether the learned vertical
structure is reasonable, on a real SB0 that is morphologically in the TNG barred class.

Inputs (all already on disk):
  outputs/real_images/ngc4371_arrays.npz                  (our baseline_mass, learned_mass)
  outputs/real_images/ngc4371_learned_vs_baseline_metrics.json   (our v_c by direct sum)
  outputs/real_images/ngc4371_mge_benchmark.npz           (MGE RMS|z|(R), v_c)

Run: .venv/bin/python scripts/compare_ngc4371_learned_vs_mge.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rms_z_profile(mass_grid, z_centers):  # mirrors the pipeline's helper (3 lines)
    m_rz = mass_grid.sum(axis=1)
    tot = np.maximum(m_rz.sum(axis=1), 1e-30)
    return np.sqrt((m_rz * z_centers[None, :] ** 2).sum(axis=1) / tot)


def comeron_rmsz(vc):
    """Comeron+2018 disk-only thin+thick equivalent RMS|z| at circular velocity vc."""
    zt = (26 + 1.23 * vc) / 1000.0
    zT = (-262 + 8.64 * vc) / 1000.0
    f = 0.333
    return 1.8138 * np.sqrt((1 - f) * zt**2 + f * zT**2), zt, zT


def main():
    d = Path("outputs/real_images")
    a = np.load(d / "ngc4371_arrays.npz")
    r = a["r_grid"]
    z = a["z_grid"]
    base_rms = rms_z_profile(a["baseline_mass"], z)
    learn_rms = rms_z_profile(a["learned_mass"], z)

    mge = np.load(d / "ngc4371_mge_benchmark.npz")
    r_mge, rms_mge, vc_mge = mge["r_kpc"], mge["rmsz_kpc"], mge["vc_ml1"]

    m = json.loads((d / "ngc4371_learned_vs_baseline_metrics.json").read_text())
    rv = np.array(m["vc_radii_kpc"])
    vc_base, vc_learn = np.array(m["vc_baseline_kms"]), np.array(m["vc_learned_kms"])

    vc_fid = 180.0  # NGC4371 circular velocity (~MGE stellar peak); for Comeron context
    com_rms, com_zt, com_zT = comeron_rmsz(vc_fid)

    print(f"{'R[kpc]':>7}{'ours base':>10}{'ours learn':>11}{'MGE':>8}{'learn/MGE':>11}")
    for rq in (0.5, 1, 2, 3, 5, 8, 12):
        ob = float(np.interp(rq, r, base_rms))
        ol = float(np.interp(rq, r, learn_rms))
        mg = float(np.interp(rq, r_mge, rms_mge))
        print(f"{rq:7.1f}{ob:10.2f}{ol:11.2f}{mg:8.2f}{ol / mg:11.2f}")
    print(f"\nComeron disk-only (vc={vc_fid:.0f}): zt={com_zt:.2f} zT={com_zT:.2f} "
          f"-> equiv RMS|z|~{com_rms:.2f} kpc")
    print(f"v_c peak: ours baseline {vc_base.max():.0f}, ours learned {vc_learn.max():.0f}, "
          f"MGE {vc_mge.max():.0f} km/s")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(r, base_rms, color="#4c72b0", lw=2, label="ours: baseline sech² h=0.3")
    ax[0].plot(r, learn_rms, color="#c44e52", lw=2, label="ours: TNG-learned q_m")
    ax[0].plot(r_mge, rms_mge, "k--", lw=2, label="MGE benchmark (Behzad)")
    ax[0].axhline(com_rms, color="C2", ls=":", lw=1.5, label=f"Comerón disk eq. ({com_rms:.2f})")
    ax[0].axvspan(14, 20, color="0.9", alpha=0.7)
    ax[0].text(14.2, 0.1, "learned edge\nartifact", fontsize=7, color="0.4")
    ax[0].set(xlabel="R [kpc]", ylabel="RMS|z| [kpc]", xlim=(0, 18), ylim=(0, 2.5),
              title="NGC 4371 vertical structure: ours vs MGE")
    ax[0].legend(fontsize=8)

    ax[1].plot(rv, vc_base, color="#4c72b0", lw=2, label="ours: baseline (thin)")
    ax[1].plot(rv, vc_learn, "--", color="#c44e52", lw=2, label="ours: learned q_m")
    ax[1].plot(r_mge, vc_mge, "k--", lw=2, label="MGE benchmark (M/L=1)")
    ax[1].set(xlabel="R [kpc]", ylabel="v_c [km/s]", xlim=(0, 18), ylim=(0, 200),
              title="stellar v_c (M/L=1)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "ngc4371_learned_vs_mge.png", dpi=130)
    print(f"\nwrote {d}/ngc4371_learned_vs_mge.png")


if __name__ == "__main__":
    main()
