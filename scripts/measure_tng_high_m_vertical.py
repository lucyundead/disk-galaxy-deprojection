"""Do high-m (arm/odd) azimuthal structures share the m=0 vertical profile in TNG50?

The deprojection assigns the m NOT in {0,2,4} surface-density content the local m=0
vertical profile (dgdp.reproject.high_m_sigma). This measures that assumption on the 185
training galaxies (R=32 truth grids, one projection per galaxy): per (galaxy, R, m), fit
the sech^2 scale height h_z (rho ∝ sech^2(z/h_z)) to the harmonic's vertical amplitude
profile |c_m|(R,z), and report the amplitude-weighted mean ratio h_m / h_0 over the disk
(1 < R < 12 kpc, harmonics with >=5% of the m=0 amplitude). Also aggregates ALL m not in
{0,2,4} ("hi"). Writes outputs/tng50_vertical_scale/high_m_vertical.json.

Run: PYTHONPATH=src .venv/bin/python scripts/measure_tng_high_m_vertical.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dgdp.vertical_mixture import sech2_height_fit

TABLE = Path("outputs/tng50_milestone2c_clean3d/density_residual_table.npz")
OUT = Path("outputs/tng50_vertical_scale/high_m_vertical.json")
M_LIST = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)
R_MIN, R_MAX, AMP_FRAC = 1.0, 12.0, 0.05


def main():
    t = np.load(TABLE)
    gal = t["galaxy_id"]
    _, first = np.unique(gal, return_index=True)
    rho = t["truth_density"][first].astype(np.float64)          # (gals, nR, nphi, nz)
    r = 0.5 * (t["r_edges_kpc"][:-1] + t["r_edges_kpc"][1:])
    z = 0.5 * (t["z_edges_kpc"][:-1] + t["z_edges_kpc"][1:])
    n_g, n_r, n_phi, n_z = rho.shape
    dz = float(z[1] - z[0])

    c = np.fft.rfft(rho, axis=2) / n_phi                        # (gals, nR, m, nz)
    amp = np.abs(c).sum(axis=3) * dz                            # (gals, nR, m) column amplitude
    sel_r = (r > R_MIN) & (r < R_MAX)

    def h_of(prof):                                             # (gals, nR, nz) -> (gals, nR)
        return sech2_height_fit(prof.reshape(-1, n_z), z).reshape(n_g, n_r)

    h0 = h_of(np.abs(c[:, :, 0, :]))
    report = {"n_galaxies": int(n_g), "r_range_kpc": [R_MIN, R_MAX], "amp_frac_cut": AMP_FRAC}
    print(f"{n_g} galaxies | h_z convention rho ∝ sech^2(z/h_z) | weighted by |Sigma_m|(R)")
    num = den = 0.0
    for m in M_LIST:
        hm = h_of(np.abs(c[:, :, m, :]))
        w = amp[:, :, m] * (amp[:, :, m] > AMP_FRAC * amp[:, :, 0]) * sel_r[None, :]
        ratio = float((hm / h0 * w).sum() / max(w.sum(), 1e-30))
        frac_used = float((w > 0).sum() / (n_g * sel_r.sum()))
        report[f"h{m}_over_h0"] = ratio
        print(f"  m={m:2d}: h_m/h_0 = {ratio:5.3f}   ({100 * frac_used:4.1f}% of disk cells pass the amplitude cut)")
        if m not in (0, 2, 4) and m <= 10:                      # the q_hi modes; m>10 = shot noise
            num += float((hm / h0 * w).sum())
            den += float(w.sum())
    report["hi_height_ratio"] = num / max(den, 1e-30)           # amplitude-weighted, m in {1,3,5,6,7,8,10}
    print(f"  amplitude-weighted q_hi squeeze (m not in {{0,2,4}}, m<=10): {report['hi_height_ratio']:.3f}")

    # aggregate of everything NOT in {0,2,4}: vertical profile of the residual field
    keep = np.zeros_like(c)
    keep[:, :, [0, 2, 4], :] = c[:, :, [0, 2, 4], :]
    hi = rho - np.fft.irfft(keep * n_phi, n=n_phi, axis=2)
    prof_hi = np.abs(hi).sum(axis=2)                            # (gals, nR, nz)
    h_hi = h_of(prof_hi)
    w = prof_hi.sum(axis=2) * dz * sel_r[None, :]
    ratio_hi = float((h_hi / h0 * w).sum() / max(w.sum(), 1e-30))
    report["h_hi_over_h0"] = ratio_hi
    print(f"  all m not in {{0,2,4}} combined: h_hi/h_0 = {ratio_hi:5.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
