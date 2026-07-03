"""How much do light-vs-mass systematics move the deprojection? (no SKIRT needed)

S4G 3.6um light is a good but imperfect mass tracer: star-forming arms are over-luminous
(PAH 3.3um + young red supergiants) and M/L drifts slowly with radius. This perturbs the
input image in those two physically motivated ways and measures the response:

  arms+/-15% : unsharp boost/suppress of structure below ~1.5 kpc scale (arm contrast)
  M/L grad   : +/-0.1 dex linear tilt from centre to R=12 kpc (radial M/L gradient)

Reported per variant: h_z at R=2/5/8, v_c at R=2/5/10, and the OOD percentile (does the
perturbation push the galaxy out of the training cloud?). Writes
outputs/real_images/input_sensitivity.json.

Run: PYTHONPATH=src .venv/bin/python scripts/diagnose_input_sensitivity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dgdp import deproject
from dgdp.image import load_image

GALAXIES = {
    "NGC4321": dict(image="NGC4321_m_c_r_f.fits", distance_mpc=15.2, inclination_deg=30.0,
                    pa_onsky_deg=153.0, stellar_mass=6.0e10),
    "NGC4371": dict(image="NGC4371/final/NGC4371.fits", distance_mpc=16.194,
                    inclination_deg=58.0, pix_arcsec=0.75, pa_pix_deg=1.8,
                    center=(254.6, 152.8), mask="NGC4371/final/NGC4371_mask.fits",
                    stellar_mass=3.53e10),
}
R_HZ, R_VC = [2.0, 5.0, 8.0], [2.0, 5.0, 10.0]


def box_blur(img, half):
    """3x iterated box blur ~ Gaussian of that scale (numpy-only)."""
    out = img.astype(float)
    for _ in range(3):
        c = np.cumsum(np.pad(out, ((half + 1, half), (0, 0))), axis=0)
        out = (c[2 * half + 1:] - c[:-2 * half - 1]) / (2 * half + 1)
        c = np.cumsum(np.pad(out, ((0, 0), (half + 1, half))), axis=1)
        out = (c[:, 2 * half + 1:] - c[:, :-2 * half - 1]) / (2 * half + 1)
    return out


def variants(gi):
    """(name, perturbed light) pairs; totals renormalised so only the SHAPE changes."""
    yy, xx = np.mgrid[0:gi.light.shape[0], 0:gi.light.shape[1]]
    r_kpc = np.hypot(xx - gi.cx, yy - gi.cy) * gi.pix_kpc
    smooth = box_blur(gi.light, max(2, int(round(0.75 / gi.pix_kpc))))   # ~1.5 kpc FWHM-ish
    out = {"baseline": gi.light}
    for a in (+0.15, -0.15):
        pert = np.clip(gi.light + a * (gi.light - smooth), 0.0, None)
        out[f"arms{a:+.0%}"] = pert * (gi.light.sum() / pert.sum())
    for g in (+0.1, -0.1):
        pert = gi.light * 10.0 ** (g * np.clip(r_kpc, 0, 12.0) / 12.0)
        out[f"M/L{g:+.1f}dex"] = pert * (gi.light.sum() / pert.sum())
    return out


def main():
    report = {}
    for name, cfg in GALAXIES.items():
        img, mask = cfg.pop("image"), cfg.pop("mask", None)
        gi = load_image(img, distance_mpc=cfg["distance_mpc"],
                        inclination_deg=cfg["inclination_deg"], mask=mask,
                        pix_arcsec=cfg.get("pix_arcsec"), pa_pix_deg=cfg.get("pa_pix_deg"),
                        pa_onsky_deg=cfg.get("pa_onsky_deg"), center=cfg.get("center"))
        pix_arcsec = gi.pix_kpc / (cfg["distance_mpc"] * 1e3) * 206264.806
        rows = {}
        print(f"\n=== {name} ===")
        for tag, light in variants(gi).items():
            res = deproject(light, distance_mpc=cfg["distance_mpc"],
                            inclination_deg=cfg["inclination_deg"], pix_arcsec=pix_arcsec,
                            pa_pix_deg=float(np.degrees(gi.pa_pix)), center=(gi.cx, gi.cy),
                            ml=1.0, stellar_mass=cfg["stellar_mass"])
            rows[tag] = dict(hz=res.scale_height(R_HZ).tolist(),
                             vc=res.v_circ(R_VC).tolist(),
                             ood_percentile=None if res.ood is None else res.ood["percentile"])
            hz, vc = rows[tag]["hz"], rows[tag]["vc"]
            print(f"  {tag:12s} h_z(2/5/8)={hz[0]:.2f}/{hz[1]:.2f}/{hz[2]:.2f} kpc  "
                  f"v_c(2/5/10)={vc[0]:.0f}/{vc[1]:.0f}/{vc[2]:.0f} km/s  "
                  f"OOD pct={rows[tag]['ood_percentile']:.0f}")
        base = rows["baseline"]
        for tag, row in rows.items():
            if tag == "baseline":
                continue
            dh = 100 * np.max(np.abs(np.array(row["hz"]) / base["hz"] - 1))
            dv = 100 * np.max(np.abs(np.array(row["vc"]) / base["vc"] - 1))
            print(f"  {tag:12s} max |dh_z|={dh:.1f}%  max |dv_c|={dv:.1f}%")
        report[name] = rows
    out = Path("outputs/real_images/input_sensitivity.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
