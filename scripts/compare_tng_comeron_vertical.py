"""Compare the measured TNG50 stellar vertical scale (Step 1) to the real S4G
edge-on sample of Comeron, Salo & Knapen 2018 (A&A 610, A5) -> OOD-axis verdict.

REAL VALUES ARE NOT FABRICATED. They are taken verbatim from the paper (the PDF the
user supplied), with line refs:

* Convention (their l.2093 + l.2795-2804, verified against their MW check below):
  the disc luminosity density is rho(z) propto sech^2(z/z0); the REPORTED scale-heights
  zt (thin), zT (thick) are the e-folding lengths of the profile tail = z0/2 = the
  EXPONENTIAL scale height. My TNG fit uses rho propto sech^2(z/2h) so h is also the
  exponential scale height -> zt<->h_thin, zT<->h_thick DIRECTLY, no factor of 2.

* Scale-height vs circular velocity, their Eq. 18 + Table 4 (z in pc, vc in km/s):
      <z_i> = A + B vc
  thin  all (N=124): A=  26, B=1.23, rho=0.55 ; gas-rich (N=83): A= -12, B=1.42, rho=0.77
  thick all (N=124): A=-262, B=8.64, rho=0.56 ; gas-rich (N=83): A=-217, B=7.42, rho=0.71
  MW check (their l.2814-2821): predicts zt~300 pc, zT~1450 pc at vc=218 within 15%
  (gas-rich <5%). Verified: thin 26+1.23*218=294; thick gr -217+7.42*218=1401. OK.

* vc here is the MAXIMUM circular velocity (their l.1923, total-mass proxy); sample
  range vc=50-300, mostly 100-200, MW-sized vc>200 rare (9/141; their l.2045-2048).

* Thick/thin disc MASS ratio (their l.2437-2443): the well-known trend that thick discs
  dominate in low-mass galaxies -- MT/Mt typically 0.2-1 for vc>120 km/s, up to ~3 for
  low-mass -- i.e. the thick mass FRACTION f_thick=MT/(MT+Mt) DECREASES with mass
  (0.17-0.5 for vc>120, up to ~0.75 below).

THE ONE EXTERNAL INPUT: to place TNG (known M*) and Comeron (known vc) on a common axis
we map M*->vc with an MW-anchored stellar Tully-Fisher vc = 218 (M*/M*_MW)^(1/n),
M*_MW=10^10.78 (Licquia & Newman 2015), vc_MW=218 (Bovy 2012 -- the SAME MW point
Comeron validate against), canonical n=4 (M* ~ vc^4); bracketed n in [3.5,4.5]. This
relation does NOT enter any h_z value; it only aligns the two x-axes. Stated per the
user's instruction to be explicit about borrowed numbers.

Run (after measure_tng_vertical_scale.py):
    .venv/bin/python scripts/compare_tng_comeron_vertical.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Comeron+2018 Eq.18 / Table 4 (pc, km/s) -- see module docstring for source.
COM = {"thin_all": (26.0, 1.23), "thin_gas": (-12.0, 1.42),
       "thick_all": (-262.0, 8.64), "thick_gas": (-217.0, 7.42)}
MW_LOGM, MW_VC = 10.78, 218.0  # Licquia&Newman 2015 ; Bovy 2012 (Comeron's MW point)


def comeron_z_kpc(vc, key):
    a, b = COM[key]
    return (a + b * vc) / 1000.0  # pc -> kpc


def vc_from_logM(logM, n=4.0):
    """MW-anchored stellar Tully-Fisher (axis alignment only; see docstring)."""
    return MW_VC * 10.0 ** ((logM - MW_LOGM) / n)


def rms_sech2(zt, zT, f_thick):
    """RMS|z| of a thin+thick sech^2 mix given EXPONENTIAL scale heights (kpc).

    Per component rho propto sech^2(z/2h) -> <z^2> = (pi^2/3) h^2, RMS = 1.814 h.
    Mass-weighted: RMS = 1.814 sqrt(f_thin zt^2 + f_thick zT^2).
    """
    return 1.8138 * np.sqrt((1 - f_thick) * zt ** 2 + f_thick * zT ** 2)


def fthick_band(vc):
    """Comeron real thick mass fraction band from MT/Mt (their l.2441-2443)."""
    r_lo = 0.2
    r_hi = np.where(vc >= 120, 1.0, 1.0 + (3.0 - 1.0) * (120 - np.clip(vc, 70, 120)) / 50)
    return r_lo / (1 + r_lo), r_hi / (1 + r_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="outputs/tng50_vertical_scale")
    ap.add_argument("--tfr-n", type=float, default=4.0)
    args = ap.parse_args()
    d = Path(args.in_dir)

    rows = list(csv.DictReader(open(d / "tng_vertical_scale.csv", newline="")))
    logM = np.array([float(r["logM"]) for r in rows])
    h_thin = np.array([float(r["h_thin"]) for r in rows])
    h_thick = np.array([float(r["h_thick"]) for r in rows])
    f_thick = np.array([float(r["f_thick"]) for r in rows])
    rmsz = np.array([float(r["rmsz_disk"]) for r in rows])
    vc_tng = vc_from_logM(logM, args.tfr_n)

    # ---- per mass-bin comparison table ----
    bins = [9.5, 10.0, 10.5, 11.0, 11.5, 12.5]
    print(f"{'logM*':<11}{'vc_est':>7}{'  | TNG zt/zT/fT/RMS':>26}"
          f"{'  | Comeron(all) zt/zT/RMS':>27}{'  | zT ratio':>11}")
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (logM >= lo) & (logM < hi)
        if not sel.any():
            continue
        vcm = float(np.median(vc_tng[sel]))
        czt, czT = comeron_z_kpc(vcm, "thin_all"), comeron_z_kpc(vcm, "thick_all")
        cf = 0.333  # MT/Mt=0.5 representative
        crms = rms_sech2(czt, czT, cf)
        t = {"bin": f"{lo:.1f}-{hi:.1f}", "n": int(sel.sum()), "vc_est": round(vcm, 0),
             "tng_zt": round(float(np.median(h_thin[sel])), 2),
             "tng_zT": round(float(np.median(h_thick[sel])), 2),
             "tng_fthick": round(float(np.median(f_thick[sel])), 2),
             "tng_rms": round(float(np.median(rmsz[sel])), 2),
             "com_zt": round(czt, 2), "com_zT": round(czT, 2), "com_rms": round(crms, 2),
             "zT_ratio": round(float(np.median(h_thick[sel])) / czT, 2),
             "zt_ratio": round(float(np.median(h_thin[sel])) / czt, 2)}
        table.append(t)
        print(f"{t['bin']:<11}{vcm:>7.0f}   {t['tng_zt']:.2f}/{t['tng_zT']:.2f}/"
              f"{t['tng_fthick']:.2f}/{t['tng_rms']:.2f}      "
              f"{t['com_zt']:.2f}/{t['com_zT']:.2f}/{t['com_rms']:.2f}        "
              f"{t['zT_ratio']:.2f}x")
    json.dump({"tfr_n": args.tfr_n, "mw_anchor": [MW_LOGM, MW_VC], "table": table},
              open(d / "tng_vs_comeron_summary.json", "w"), indent=2)

    # ---- figure ----
    lm = np.linspace(9.5, 12.0, 60)
    vc = vc_from_logM(lm, args.tfr_n)
    vc_lo, vc_hi = vc_from_logM(lm, 4.5), vc_from_logM(lm, 3.5)  # slope bracket
    extrap = lm > MW_LOGM + 4 * np.log10(300 / MW_VC)  # vc>300 beyond Comeron range

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    def overlay(a, key_all, key_gas):
        za, zg = comeron_z_kpc(vc, key_all), comeron_z_kpc(vc, key_gas)
        zb_lo = comeron_z_kpc(vc_lo, key_all)
        zb_hi = comeron_z_kpc(vc_hi, key_all)
        a.fill_between(lm, zb_lo, zb_hi, color="C1", alpha=0.18,
                       label="Comeron all (TFR-slope band)")
        a.plot(lm, za, "C1-", lw=2, label="Comeron all (Eq.18)")
        a.plot(lm, zg, "C1--", lw=1.5, label="Comeron gas-rich")

    # (a) thick scale height
    ax[0, 0].scatter(logM, h_thick, s=12, color="0.6", alpha=0.5)
    overlay(ax[0, 0], "thick_all", "thick_gas")
    binc = [0.5 * (a + b) for a, b in zip(bins[:-1], bins[1:])]
    ax[0, 0].plot(binc[:len(table)], [t["tng_zT"] for t in table], "C3-o", lw=2,
                  label="TNG median")
    ax[0, 0].set(xlabel="log M* [Msun]", ylabel="thick h_z = zT [kpc]",
                 title="(a) THICK disc scale height", ylim=(0, 3))
    ax[0, 0].legend(fontsize=8)

    # (b) thin scale height
    ax[0, 1].scatter(logM, h_thin, s=12, color="0.6", alpha=0.5)
    overlay(ax[0, 1], "thin_all", "thin_gas")
    ax[0, 1].plot(binc[:len(table)], [t["tng_zt"] for t in table], "C3-o", lw=2,
                  label="TNG median")
    ax[0, 1].axhspan(0.0, 0.4, color="0.85", alpha=0.6)
    ax[0, 1].text(9.55, 0.05, "z grid + softening floor (<~0.4 kpc unresolved)",
                  fontsize=7, color="0.4")
    ax[0, 1].set(xlabel="log M* [Msun]", ylabel="thin h_z = zt [kpc]",
                 title="(b) THIN disc scale height", ylim=(0, 1.0))
    ax[0, 1].legend(fontsize=8)

    # (c) thick mass fraction
    f_lo, f_hi = fthick_band(vc)
    ax[1, 0].fill_between(lm, f_lo, f_hi, color="C1", alpha=0.2,
                          label="Comeron real (MT/Mt 0.2-1, ->3 low-M)")
    ax[1, 0].scatter(logM, f_thick, s=12, color="0.6", alpha=0.5)
    ax[1, 0].plot(binc[:len(table)], [t["tng_fthick"] for t in table], "C3-o", lw=2,
                  label="TNG median")
    ax[1, 0].set(xlabel="log M* [Msun]", ylabel="thick mass fraction f_thick",
                 title="(c) thick-disc MASS FRACTION (note opposite trend)", ylim=(0, 1))
    ax[1, 0].legend(fontsize=8)

    # (d) total RMS|z|
    crms_mid = rms_sech2(comeron_z_kpc(vc, "thin_all"), comeron_z_kpc(vc, "thick_all"), 0.333)
    crms_lo = rms_sech2(comeron_z_kpc(vc, "thin_all"), comeron_z_kpc(vc, "thick_all"), f_lo)
    crms_hi = rms_sech2(comeron_z_kpc(vc, "thin_all"), comeron_z_kpc(vc, "thick_all"), f_hi)
    ax[1, 1].fill_between(lm, crms_lo, crms_hi, color="C1", alpha=0.2,
                          label="Comeron (f_thick band)")
    ax[1, 1].plot(lm, crms_mid, "C1-", lw=2, label="Comeron (f_thick=0.33)")
    ax[1, 1].scatter(logM, rmsz, s=12, color="0.6", alpha=0.5)
    ax[1, 1].plot(binc[:len(table)], [t["tng_rms"] for t in table], "C3-o", lw=2,
                  label="TNG median")
    ax[1, 1].set(xlabel="log M* [Msun]", ylabel="RMS|z|_disk [kpc]",
                 title="(d) total disk-region RMS|z|", ylim=(0, 3))
    ax[1, 1].legend(fontsize=8)

    # vc annotation on every panel top axis
    for a in ax.ravel():
        for x in extrap.nonzero()[0][:1]:
            a.axvspan(lm[x], 12.0, color="0.95", alpha=0.5)
            a.text(lm[x], a.get_ylim()[1] * 0.9, " vc>300\n (extrapol.)",
                   fontsize=6, color="0.5")
    fig.suptitle("TNG50 milestone-2d stellar vertical scale vs Comeron+2018 S4G edge-ons "
                 "(x->vc via MW-anchored STFR)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(d / "tng_vs_comeron_vertical.png", dpi=130)
    print(f"\nwrote {d}/tng_vs_comeron_vertical.png + tng_vs_comeron_summary.json")


if __name__ == "__main__":
    main()
