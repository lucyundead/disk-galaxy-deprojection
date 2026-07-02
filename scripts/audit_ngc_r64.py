"""Regenerate the NGC 4321 + NGC 4371 deprojections with the shipped R=64 bundle
(``dgdp.deproject``) and write the arrays/metrics in the format the existing comparison
scripts consume, plus a per-galaxy Part-A summary figure (observed | face-on | edge-on |
RMS|z| | rotation curve).

This replaces the old R=32 ``outputs/real_images/{slug}_arrays.npz`` +
``{slug}_learned_vs_baseline_metrics.json`` in place (outputs/ is gitignored), so
``plot_ngc4321_figures.py`` and the three ``*ngc4371*`` MGE-comparison scripts run unchanged
on the R=64 result.

Run:  PYTHONPATH=src .venv/bin/python scripts/audit_ngc_r64.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from dgdp import rotation
from dgdp import vertical_mixture as vm
from dgdp.deproject import _BUNDLED, deproject
from dgdp.density3d import cylindrical_bin_volumes, make_cylindrical_grid_spec
from dgdp.image import geometric_baseline, load_image
from dgdp.model import DeprojectionModel

# reuse the cartesian renderers / bar-frame helper from the NGC 4321 figure script
from plot_ngc4321_figures import bar_azimuth, make_lookup, project

OUT = Path("outputs/real_images")

# geometry kwargs (shared by load_image + deproject) + mass kwargs (deproject only)
GALAXIES = {
    "NGC4321": {  # WCS path (S4G mosaic), README-verified params
        "fits": "NGC4321_m_c_r_f.fits",
        "geom": dict(distance_mpc=15.2, inclination_deg=30.0, pa_onsky_deg=153.0),
        "stellar_mass": 6.0e10,
    },
    "NGC4371": {  # WCS-less S4G cutout, 2026-06-25 MGE-validation params
        "fits": "NGC4371/final/NGC4371.fits",
        "geom": dict(distance_mpc=16.194, inclination_deg=58.0, pix_arcsec=0.75,
                     pa_pix_deg=1.8, center=(254.6, 152.8), mask="NGC4371/final/NGC4371_mask.fits"),
        "stellar_mass": 3.53e10,
    },
}
SCALE_HEIGHT_KPC = 0.3
V_RADII = np.linspace(0.3, 18.0, 80)


def rms_z_profile(mass_grid, z_centers):
    m_rz = mass_grid.sum(axis=1)
    tot = np.maximum(m_rz.sum(axis=1), 1e-30)
    return np.sqrt((m_rz * z_centers[None, :] ** 2).sum(axis=1) / tot)


def hz_profile(mass_grid, z_centers):
    """Sech^2 scale height h_z(R): rho ∝ sech^2(z/2h) fit per radius (obs-comparable)."""
    return vm.sech2_height_fit(mass_grid.sum(axis=1), z_centers)


def summary_figure(name, light, cx, cy, pix_kpc, baseline_mass, learned_mass, vol,
                   r_grid, z_grid, radii, vc, hz_base, hz_learn, path):
    """observed | face-on (learned) | edge-on (learned) | h_z(R) | rotation curve."""
    spec_e = _SPEC
    r_e, p_e, z_e = spec_e.r_edges_kpc, spec_e.phi_edges_rad, spec_e.z_edges_kpc
    theta_b = bar_azimuth(baseline_mass, r_grid)              # grid azimuth of the bar
    ctb, stb = np.cos(theta_b), np.sin(theta_b)
    base_lk = make_lookup(learned_mass / vol, r_e, p_e, z_e)

    def lk(x, y, z):  # display (bar) frame -> grid (disk-major) frame
        return base_lk(x * ctb - y * stb, x * stb + y * ctb, z)

    xs = np.linspace(-16, 16, 220)
    face = project(lk, xs, xs, np.linspace(-5, 5, 64), along="z")
    zs = np.linspace(-4, 4, 110)
    edge = project(lk, xs, zs, np.linspace(-16, 16, 140), along="y")

    fig, ax = plt.subplots(1, 5, figsize=(23, 4.4), constrained_layout=True)
    ny, nx = light.shape
    ext = [(0 - cx) * pix_kpc, (nx - cx) * pix_kpc, (0 - cy) * pix_kpc, (ny - cy) * pix_kpc]
    ovmax = float(np.percentile(light[light > 0], 99.9))
    ax[0].imshow(light, origin="lower", extent=ext, cmap="bone",
                 norm=LogNorm(vmin=ovmax * 3e-3, vmax=ovmax))
    ax[0].set(xlim=(-16, 16), ylim=(-16, 16), title="observed (sky)", xlabel="x [kpc]", ylabel="y [kpc]")
    ax[0].set_aspect("equal")

    fmax = float(face.max())
    ax[1].imshow(face.T, origin="lower", extent=[-16, 16, -16, 16], cmap="magma",
                 norm=LogNorm(vmin=fmax * 3e-3, vmax=fmax))
    ax[1].set(title="deprojected face-on\n(bar on x)", xlabel="x [kpc]", ylabel="y [kpc]")
    ax[1].set_aspect("equal")

    emax = float(edge.max())
    ax[2].imshow(edge.T, origin="lower", extent=[-16, 16, -4, 4], cmap="magma", aspect="auto",
                 norm=LogNorm(vmin=emax * 3e-3, vmax=emax))
    ax[2].set(title="deprojected edge-on\n(bar side-on)", xlabel="x [kpc]", ylabel="z [kpc]")

    ax[3].plot(r_grid, hz_base, color="#4c72b0", lw=2, label="baseline sech² h=0.3")
    ax[3].plot(r_grid, hz_learn, color="#c44e52", lw=2, label="learned q_m")
    ax[3].set(xlim=(0, 18), title="sech² scale height", xlabel="R [kpc]",
              ylabel="h_z [kpc]  (ρ ∝ sech²(z/h_z))")
    ax[3].legend(fontsize=8)

    ax[4].plot(radii, vc["baseline"], color="#4c72b0", lw=2, label="baseline")
    ax[4].plot(radii, vc["learned"], "--", color="#c44e52", lw=2, label="learned")
    ax[4].plot(radii, vc["uniform"], ":", color="#2ca02c", lw=2, label="uniform h=1")
    ax[4].set(xlim=(0, 18), ylim=(0, None), title="rotation curve", xlabel="R [kpc]", ylabel="v_c [km/s]")
    ax[4].legend(fontsize=8)

    fig.suptitle(f"{name}: R=64 bundle deprojection (bar azimuth {np.degrees(theta_b):.0f}°)", fontsize=13)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    global _SPEC
    OUT.mkdir(parents=True, exist_ok=True)
    g = DeprojectionModel.load(str(_BUNDLED)).grid
    spec = make_cylindrical_grid_spec(r_min_kpc=g["r_min"], r_max_kpc=g["r_max"], n_r=g["n_r"],
                                      n_phi=g["n_phi"], z_max_kpc=g["z_max"], n_z=g["n_z"])
    _SPEC = spec
    r_grid = 0.5 * (spec.r_edges_kpc[:-1] + spec.r_edges_kpc[1:])
    z_grid = 0.5 * (spec.z_edges_kpc[:-1] + spec.z_edges_kpc[1:])
    phi = 0.5 * (spec.phi_edges_rad[:-1] + spec.phi_edges_rad[1:])
    vol = cylindrical_bin_volumes(spec).astype(float)
    wz_unif = 1.0 / np.cosh(z_grid / 1.0) ** 2
    wz_unif /= wz_unif.sum()

    for name, cfg in GALAXIES.items():
        slug = name.lower()
        m_star = cfg["stellar_mass"]
        gi = load_image(cfg["fits"], **cfg["geom"])
        base = geometric_baseline(gi, spec, scale_height_kpc=SCALE_HEIGHT_KPC, stellar_mass=m_star)
        baseline_mass = base["baseline_density"] * vol

        res = deproject(cfg["fits"], **cfg["geom"], ml=1.0, stellar_mass=m_star,
                        scale_height_kpc=SCALE_HEIGHT_KPC)
        learned_mass = res.density_3d * res._vol
        uniform_mass = baseline_mass.sum(axis=2)[:, :, None] * wz_unif[None, None, :]

        vc = {n: rotation.v_circ(mg, r_grid, phi, z_grid, V_RADII)
              for n, mg in (("baseline", baseline_mass), ("learned", learned_mass),
                            ("uniform", uniform_mass))}
        rms_base = rms_z_profile(baseline_mass, z_grid)
        rms_learn = rms_z_profile(learned_mass, z_grid)
        hz_base = hz_profile(baseline_mass, z_grid)
        hz_learn = hz_profile(learned_mass, z_grid)
        inner = r_grid < 12.0
        w_in = baseline_mass.sum(axis=(1, 2))[inner]

        def eff(prof):  # mass-weighted profile mean over the reliable inner disk (R<12 kpc)
            return float((prof[inner] * w_in).sum() / w_in.sum())

        # mass conservation is the core invariant of the pipeline -> assert it
        assert abs(learned_mass.sum() / m_star - 1) < 1e-3, (name, learned_mass.sum(), m_star)
        assert abs(baseline_mass.sum() / m_star - 1) < 1e-3, (name, baseline_mass.sum(), m_star)

        np.savez(OUT / f"{slug}_arrays.npz",
                 baseline_mass=baseline_mass, learned_mass=learned_mass, uniform_mass=uniform_mass,
                 r_grid=r_grid, z_grid=z_grid, phi_centers=phi,
                 incl_deg=float(gi.incl_deg), pa_pix=float(gi.pa_pix),
                 cx=float(gi.cx), cy=float(gi.cy), pix_kpc=float(gi.pix_kpc), bkg=float(gi.bkg),
                 scale_height=SCALE_HEIGHT_KPC, stellar_mass=float(m_star))
        (OUT / f"{slug}_learned_vs_baseline_metrics.json").write_text(json.dumps({
            "galaxy": name, "n_r_out": int(g["n_r"]),
            "hz_baseline_kpc": eff(hz_base), "hz_learned_kpc": eff(hz_learn),
            "hz_ratio": eff(hz_learn) / eff(hz_base),
            "rmsz_baseline_kpc": eff(rms_base), "rmsz_learned_kpc": eff(rms_learn),
            "rmsz_ratio": eff(rms_learn) / eff(rms_base),
            "scale_height_baseline_kpc": SCALE_HEIGHT_KPC,
            "vc_radii_kpc": V_RADII.tolist(),
            "vc_baseline_kms": vc["baseline"].tolist(), "vc_learned_kms": vc["learned"].tolist(),
            "vc_uniform_kms": vc["uniform"].tolist(),
        }, indent=2), encoding="utf-8")

        summary_figure(name, gi.light, gi.cx, gi.cy, gi.pix_kpc, baseline_mass, learned_mass, vol,
                       r_grid, z_grid, V_RADII, vc, hz_base, hz_learn,
                       OUT / f"{slug}_r64_summary.png")

        print(f"{name}: M* {m_star:.2e} Msun  v_c peak base {vc['baseline'].max():.1f} / "
              f"learned {vc['learned'].max():.1f} km/s  h_z(R<12) base {eff(hz_base):.2f} -> "
              f"learned {eff(hz_learn):.2f} kpc ({eff(hz_learn) / eff(hz_base):.1f}x)")
        print("   h_z flare learned: " + "  ".join(
            f"R={rq}:{np.interp(rq, r_grid, hz_learn):.2f}" for rq in (0.5, 1, 2, 5, 8)))
        print(f"   wrote {slug}_arrays.npz, {slug}_learned_vs_baseline_metrics.json, {slug}_r64_summary.png")


if __name__ == "__main__":
    main()
