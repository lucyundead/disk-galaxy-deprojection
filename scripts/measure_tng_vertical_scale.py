"""Measure the TNG50 milestone-2d sample's stellar vertical scale (Step 1 of the
OOD-axis decision: is TNG's q_m thickness inflated, or is the OOD morphology?).

For each of the 185 unique galaxies in the fine-z truth grids we measure, in the
DISK region (R beyond the bar, so the bar/bulge is excluded -- the truth density is
total stellar mass, while Comeron+2018 separates components and drops the bulge):

  * RMS|z|(R)                -- model-free vertical second moment vs radius (flaring)
  * RMS|z|_disk              -- single mass-weighted RMS|z| over the disk region
  * h_single                 -- single sech^2(z/2h) fit (h = exp. scale height)
  * h_thin, h_thick, f_thick -- thin+thick sech^2 fit, comparable to Comeron+2018

Convention: rho(z) propto sech^2(z / (2 h)), so at large |z| rho ~ exp(-|z|/h) and h
is the EXPONENTIAL scale height. RMS|z| of one such component is pi/(2 sqrt 3)*2h =
1.814 h. (If Comeron's tables use z0 with sech^2(z/z0), then z0 = 2 h -- state which
when overlaying.)

Resolution caveat: the z grid is 0.3125 kpc and the stellar softening ~0.2-0.3 kpc,
so any h_z <~ 0.4 kpc (i.e. the thin disk, ~1 cell) is only marginally resolved; the
robust discriminators are RMS|z| and h_thick. Reported h_thin is flagged when <0.4.

Comeron comparison is NOT fabricated here -- this script only measures TNG. Overlay
the real edge-on sample in a second pass with the user-provided table.

Run:
    .venv/bin/python scripts/measure_tng_vertical_scale.py
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
from scipy.optimize import least_squares


def rms_z_profile(mass_grid, z_centers):
    """RMS |z| as a function of R (summed over phi).

    Mirrors rms_z_profile in scripts/deproject_real_image_ngc4321_learned.py;
    copied (not imported) because that module pulls in torch/astropy/agama.
    """
    m_rz = mass_grid.sum(axis=1)  # (R, z)
    tot = np.maximum(m_rz.sum(axis=1), 1e-30)
    return np.sqrt((m_rz * z_centers[None, :] ** 2).sum(axis=1) / tot)


def _sech2(x):
    return 1.0 / np.cosh(x) ** 2


def fit_sech2(zabs, prof, two_component):
    """Fit sech^2 vertical profile in log space (weights thin AND thick wings).

    Returns dict with scale height(s); h in the rho ~ exp(-|z|/h) convention.
    """
    good = prof > prof.max() * 1e-3
    z, y = zabs[good], np.log10(prof[good] + 1e-30)
    peak = prof.max()
    if two_component:
        def resid(p):
            a1, h1, a2, h2 = p
            model = a1 * _sech2(z / (2 * h1)) + a2 * _sech2(z / (2 * h2))
            return np.log10(model + 1e-30) - y
        p0 = [0.7 * peak, 0.3, 0.3 * peak, 1.2]
        lo = [0.0, 0.05, 0.0, 0.3]
        hi = [np.inf, 1.5, np.inf, 6.0]
        sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=4000)
        a1, h1, a2, h2 = sol.x
        if h1 > h2:  # order so 1 = thin
            a1, h1, a2, h2 = a2, h2, a1, h1
        # mass propto A*h  (int sech^2(z/2h) dz = 4h)
        f_thick = (a2 * h2) / (a1 * h1 + a2 * h2 + 1e-30)
        return {"h_thin": h1, "h_thick": h2, "f_thick": f_thick,
                "rms_log_resid": float(np.sqrt(np.mean(sol.fun ** 2)))}

    def resid1(p):
        a, h = p
        return np.log10(a * _sech2(z / (2 * h)) + 1e-30) - y
    sol = least_squares(resid1, [peak, 0.5], bounds=([0.0, 0.05], [np.inf, 6.0]),
                        max_nfev=4000)
    return {"h_single": sol.x[1], "rms_log_resid": float(np.sqrt(np.mean(sol.fun ** 2)))}


def load_manifest(path):
    info = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            info[int(row["subhalo_id"])] = {
                "logM": float(np.log10(float(row["stellar_mass_msun"]))),
                "bar_length": float(row["bar_length"]),
                "barred": row["barred_catalog"] == "True",
            }
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="/mnt/e/dgdp-milestone2d/density_residual_table.npz")
    ap.add_argument("--manifest", default="outputs/tng50_milestone2c_clean3d/manifest.csv")
    ap.add_argument("--out-dir", default="outputs/tng50_vertical_scale")
    ap.add_argument("--mass-frac", type=float, default=0.95,
                    help="disk R_out = radius enclosing this fraction of beyond-bar mass")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(args.manifest)

    z = np.load(args.table, allow_pickle=True)
    td = z["truth_density"]                 # (N, 32, 48, 32) = (sample, R, phi, z)
    gid = z["galaxy_id"]
    r_edges = z["r_edges_kpc"].astype(float)
    z_edges = z["z_edges_kpc"].astype(float)
    n_phi = z["phi_edges_rad"].size - 1
    r_cen = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_cen = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = float(np.diff(z_edges).mean())
    # cell volume depends only on R (annulus area / n_phi * dz); uniform over phi,z
    vol_R = (np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2) / n_phi) * dz  # (32,)

    # one truth per galaxy (identical across the 9 projections)
    _, first = np.unique(gid, return_index=True)
    first = np.sort(first)
    gids = gid[first]

    npos = z_cen.size // 2                  # fold +z/-z about midplane
    zabs = z_cen[npos:]                     # (16,) positive |z| centers

    rows = []
    rmsz_R = np.full((len(gids), r_cen.size), np.nan)
    for i, (idx, g) in enumerate(zip(first, gids)):
        if int(g) not in man:
            continue
        dens = td[idx].astype(np.float64)          # (R, phi, z) density
        mass = dens * vol_R[:, None, None]          # (R, phi, z) mass
        rprof = rms_z_profile(mass, z_cen)          # (R,)
        rmsz_R[i] = rprof

        r_in = man[int(g)]["bar_length"]
        rmask = r_cen >= r_in
        if rmask.sum() < 2:
            continue
        # trim sparse outskirts: R_out encloses mass_frac of beyond-bar mass
        m_per_r = mass[rmask].sum(axis=(1, 2))
        csum = np.cumsum(m_per_r) / m_per_r.sum()
        r_region = r_cen[rmask]
        r_out = float(r_region[np.searchsorted(csum, args.mass_frac)])
        rmask = (r_cen >= r_in) & (r_cen <= r_out)
        if rmask.sum() < 2:
            rmask = r_cen >= r_in
            r_out = float(r_cen[rmask][-1])

        m_z = mass[rmask].sum(axis=(0, 1))          # (z,) vertical mass profile
        prof = m_z[npos:] + m_z[npos - 1::-1]       # fold to |z| (16,)
        rms_disk = float(np.sqrt((prof * zabs ** 2).sum() / max(prof.sum(), 1e-30)))

        rin_bin = np.where(rmask)[0][0]
        rout_bin = np.where(rmask)[0][-1]
        flare = float(rprof[rout_bin] / max(rprof[rin_bin], 1e-30))

        try:
            f1 = fit_sech2(zabs, prof, two_component=False)
            f2 = fit_sech2(zabs, prof, two_component=True)
        except Exception as e:  # noqa: BLE001
            print(f"  fit failed gid {g}: {e}")
            f1, f2 = {"h_single": np.nan}, {"h_thin": np.nan, "h_thick": np.nan, "f_thick": np.nan}

        rows.append({
            "galaxy_id": int(g),
            "logM": man[int(g)]["logM"],
            "bar_length": r_in,
            "r_in": r_in, "r_out": r_out, "n_rbins": int(rmask.sum()),
            "rmsz_disk": rms_disk,
            "h_single": f1.get("h_single", np.nan),
            "h_thin": f2.get("h_thin", np.nan),
            "h_thick": f2.get("h_thick", np.nan),
            "f_thick": f2.get("f_thick", np.nan),
            "flare_ratio": flare,
        })

    # ---- save per-galaxy table + profiles ----
    fields = list(rows[0].keys())
    with open(out / "tng_vertical_scale.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    np.savez(out / "tng_vertical_scale_profiles.npz",
             galaxy_id=np.array([r["galaxy_id"] for r in rows]),
             logM=np.array([r["logM"] for r in rows]),
             rmsz_R=rmsz_R, r_cen=r_cen)

    # ---- bin by stellar mass ----
    logM = np.array([r["logM"] for r in rows])
    bins = [9.5, 10.0, 10.5, 11.0, 11.5, 12.5]
    metrics = ["rmsz_disk", "h_single", "h_thin", "h_thick", "f_thick", "flare_ratio"]
    summary = {"convention": "rho ~ sech^2(z/2h); h = exp scale height; RMS|z|=1.814 h per component",
               "region": "R in [bar_length, R_out(95% beyond-bar mass)]; bulge+bar excluded",
               "resolution_kpc": dz, "softening_note": "h_z<~0.4 kpc marginally resolved",
               "n_galaxies": len(rows), "bins": {}}
    print(f"\n{'logM* bin':<14}{'N':>4}{'RMS|z|':>9}{'h_single':>10}{'h_thin':>8}"
          f"{'h_thick':>9}{'f_thick':>9}{'flare':>8}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (logM >= lo) & (logM < hi)
        if sel.sum() == 0:
            continue
        b = {"n": int(sel.sum())}
        line = f"{lo:.1f}-{hi:.1f}      {sel.sum():>4}"
        for mname in metrics:
            vals = np.array([r[mname] for r in rows])[sel]
            vals = vals[np.isfinite(vals)]
            med, p16, p84 = np.percentile(vals, [50, 16, 84])
            b[mname] = {"median": float(med), "p16": float(p16), "p84": float(p84)}
            if mname not in ("f_thick", "flare_ratio"):
                line += f"{med:>8.2f} "
        line += f"{b['f_thick']['median']:>8.2f}{b['flare_ratio']['median']:>8.2f}"
        summary["bins"][f"{lo:.1f}-{hi:.1f}"] = b
        print(line)
    with open(out / "tng_vertical_scale_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure (TNG only; overlay Comeron in pass 2) ----
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # panel 1: RMS|z|(R) median + 16/84 band + sample curves
    med = np.nanmedian(rmsz_R, axis=0)
    p16 = np.nanpercentile(rmsz_R, 16, axis=0)
    p84 = np.nanpercentile(rmsz_R, 84, axis=0)
    ax[0].fill_between(r_cen, p16, p84, alpha=0.25, color="C0", label="16-84%")
    ax[0].plot(r_cen, med, "C0-", lw=2, label="median")
    for k in np.linspace(0, len(rows) - 1, 8).astype(int):
        ax[0].plot(r_cen, rmsz_R[k], color="0.6", lw=0.5, alpha=0.6)
    ax[0].set(xlabel="R [kpc]", ylabel="RMS|z| [kpc]", xlim=(0, 20),
              title="TNG50 RMS|z|(R) (185 gal)")
    ax[0].axvline(np.median([r["bar_length"] for r in rows]), ls=":", color="k",
                  label="median bar length")
    ax[0].legend(fontsize=8)
    # panel 2: h_thin/h_thick vs logM*
    ht = np.array([r["h_thin"] for r in rows])
    hk = np.array([r["h_thick"] for r in rows])
    ax[1].scatter(logM, hk, s=12, color="C3", alpha=0.5, label="h_thick")
    ax[1].scatter(logM, ht, s=12, color="C0", alpha=0.5, label="h_thin")
    ax[1].axhline(0.4, ls="--", color="0.5", lw=1)
    ax[1].text(logM.min(), 0.42, "z resolution limit", fontsize=7, color="0.4")
    ax[1].set(xlabel="log M* [Msun]", ylabel="scale height h [kpc]",
              title="thin+thick sech$^2$ fit")
    ax[1].legend(fontsize=8)
    # panel 3: RMS|z|_disk vs logM*
    rd = np.array([r["rmsz_disk"] for r in rows])
    ax[2].scatter(logM, rd, s=14, color="C2", alpha=0.6)
    cen = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))
    bmed = [summary["bins"].get(f"{lo:.1f}-{hi:.1f}", {}).get("rmsz_disk", {}).get("median", np.nan)
            for lo, hi in zip(bins[:-1], bins[1:])]
    ax[2].plot(cen, bmed, "k-o", lw=2, label="binned median")
    ax[2].set(xlabel="log M* [Msun]", ylabel="RMS|z|$_{disk}$ [kpc]",
              title="disk-region RMS|z| vs mass")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tng_vertical_scale.png", dpi=130)
    print(f"\nwrote {out}/tng_vertical_scale.{{csv,png}} + summary.json + profiles.npz")


if __name__ == "__main__":
    main()
