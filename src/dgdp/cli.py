"""Command-line interface: deproject a galaxy image into a 3D density cube + rotation curve."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dgdp import deproject


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="dgdp-deproject",
        description="Deproject a galaxy image -> 3D stellar-mass density + rotation curve.")
    ap.add_argument("image", help="FITS path (or a .npy 2-D array)")
    ap.add_argument("--distance-mpc", type=float, required=True)
    ap.add_argument("--inclination-deg", type=float, required=True)
    ap.add_argument("--pa-pix-deg", type=float, default=None, help="disk major-axis PA in the pixel frame")
    ap.add_argument("--pa-onsky-deg", type=float, default=None, help="on-sky PA (needs a FITS WCS)")
    ap.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"))
    ap.add_argument("--pix-arcsec", type=float, default=None, help="pixel scale; bypasses WCS")
    ap.add_argument("--mask", default=None, help="mask FITS; pixels>0 blanked")
    ap.add_argument("--ml", type=float, default=1.0, help="mass-to-light ratio")
    ap.add_argument("--stellar-mass", type=float, default=None, help="total M* [Msun] (M/L folded in)")
    ap.add_argument("--luminosity", type=float, default=None, help="total L; M*=ml*L")
    ap.add_argument("--zeropoint", type=float, default=None, help="photometric mag zero-point")
    ap.add_argument("--band-solar-mag", type=float, default=None, help="band solar absolute mag")
    ap.add_argument("--bar-angle-deg", type=float, default=0.0)
    ap.add_argument("--reproject-iters", type=int, default=2,
                    help="reprojection-consistency passes correcting the Sigma anchors (0=off)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--potential",
        action="store_true",
        help="also write mid-plane AGAMA potential samples for every requested output (needs agama)",
    )
    ap.add_argument(
        "--dynamical-output",
        type=float,
        nargs=2,
        default=None,
        metavar=("INNER_KPC", "OUTER_KPC"),
        help=(
            "also write a dynamics-ready m=0,2,4 density in OUTPUT_DIR/dynamical; "
            "non-axisymmetric modes taper to zero between INNER_KPC and OUTER_KPC"
        ),
    )
    ap.add_argument("-o", "--output-dir", default="dgdp_out")
    a = ap.parse_args(argv)

    image = a.image
    if image.endswith(".npy"):
        image = np.load(image)

    result = deproject(
        image, distance_mpc=a.distance_mpc, inclination_deg=a.inclination_deg,
        pa_pix_deg=a.pa_pix_deg, pa_onsky_deg=a.pa_onsky_deg,
        center=tuple(a.center) if a.center else None, pix_arcsec=a.pix_arcsec, mask=a.mask,
        ml=a.ml, stellar_mass=a.stellar_mass, luminosity=a.luminosity, zeropoint=a.zeropoint,
        band_solar_mag=a.band_solar_mag, bar_angle_deg=a.bar_angle_deg,
        reproject_iters=a.reproject_iters)
    result.save(a.output_dir)
    dynamical = None
    dynamical_dir = Path(a.output_dir) / "dynamical"
    if a.dynamical_output is not None:
        inner, outer = a.dynamical_output
        dynamical = result.bisymmetrize_for_dynamics(
            inner_radius_kpc=inner,
            outer_radius_kpc=outer,
        )
        dynamical.save(dynamical_dir)
        np.savez(
            dynamical_dir / "postprocess.npz",
            product_kind="bisymmetric_dynamical_density",
            retained_modes=np.array([0, 2, 4]),
            outer_taper_kpc=np.array([inner, outer]),
            source="native dgdp density.npz",
        )
    if not a.no_figures:
        from dgdp.figures import save_summary_figure
        save_summary_figure(result, a.output_dir)
        if dynamical is not None:
            save_summary_figure(dynamical, str(dynamical_dir))
    if a.potential:
        radii = np.linspace(1.0, 15.0, 40)
        np.savez(f"{a.output_dir}/potential.npz", R_kpc=radii,
                 potential=result.potential(radii, np.zeros_like(radii)))
        if dynamical is not None:
            np.savez(dynamical_dir / "potential.npz", R_kpc=radii,
                     potential=dynamical.potential(radii, np.zeros_like(radii)))
    print(f"wrote outputs to {a.output_dir} (total mass {result.total_mass:.3e} Msun"
          f"{', relative scale' if result.relative else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
