# Disk Galaxy Deprojection (`dgdp`)

Turn a single galaxy image into its deprojected 3-D stellar-mass distribution,
rotation curve, and (optionally) gravitational potential — with calibrated
uncertainties, from a **bundled pretrained model**. No training data or GPU required.

## Motivation

Recovering the 3-D stellar-mass structure of a barred galaxy from one 2-D image is
ambiguous: many 3-D distributions reproject to the same picture. `dgdp` treats the
result as a *calibrated posterior* rather than a single inversion, so every summary
(scale height, rotation curve, …) carries an uncertainty band learned from TNG50
galaxies whose true 3-D structure is known.

## Method

- The in-plane surface density is anchored geometrically from the image.
- A learned fixed-dictionary sech² vertical profile — trained on 185 TNG50 barred
  galaxies × 198 projections — supplies the disk thickness, mass-conserving by
  construction.
- A reprojection loop corrects the surface-density anchors until the thick model
  reproduces the observed image.
- The model is a mixture density network; sampling its head propagates the posterior
  to RMS|z|(R) and v_c(R) bands, calibrated on a held-out TNG50 split.

Full derivation: [`docs/reports/2026-06-30-mixture-qm-fixed-dictionary.md`](docs/reports/2026-06-30-mixture-qm-fixed-dictionary.md).

## Goal

Given an S4G-like image plus geometry (distance, inclination, PA), return a cylindrical
stellar-mass cube ρ\*(R, φ, z), a rotation curve v_c(R) and vertical scale height h_z(R),
posterior draws for uncertainty bands, and optionally the AGAMA CylSpline potential Φ(R, z).

## Install

```bash
pip install git+https://github.com/lucyundead/disk-galaxy-deprojection
```

Core install is light (`numpy`, `astropy`, `matplotlib`) and ships the ~1 MB model.
Retraining needs the `[train]` extra (`pip install '.[train]'`); the optional potential
needs [AGAMA](https://github.com/GalacticDynamics-Oxford/Agama) built from source and on
`PYTHONPATH`.

## Usage

```python
from dgdp import deproject

r = deproject("galaxy.fits", distance_mpc=15.2, inclination_deg=30, pa_onsky_deg=153,
              ml=1.0, stellar_mass=6e10, n_samples=48)

r.density_3d                    # (nR, nphi, nz) cylindrical stellar-mass cube [Msun]
r.v_circ([1, 2, 5, 10])         # rotation curve at those radii [km/s]
r.scale_height([1, 5, 10])      # sech^2 scale height h_z(R) [kpc]
r.scale_height_samples([1, 5])  # posterior draws -> uncertainty bands
r.edge_on, r.face_on            # 2-D renderings
r.save("out/")                  # density.npz + rotation_curve.csv + deprojection.png

# Dynamics-ready derivative: retain m=0,2,4 and taper the bar to an
# axisymmetric outer density between 6 and 10 kpc.
dyn = r.bisymmetrize_for_dynamics(inner_radius_kpc=6, outer_radius_kpc=10)
dyn.save("out/dynamical/")
dyn.potential([1, 2, 5, 10], [0, 0, 0, 0])  # optional AGAMA potential
```

CLI:

```bash
dgdp-deproject galaxy.fits --distance-mpc 15.2 --inclination-deg 30 --pa-onsky-deg 153 \
    --ml 1.0 --stellar-mass 6e10 -o out/

# Also write out/dynamical/{density.npz,rotation_curve.csv,potential.npz,postprocess.npz}.
dgdp-deproject galaxy.fits --distance-mpc 15.2 --inclination-deg 30 --pa-onsky-deg 153 \
    --ml 1.0 --stellar-mass 6e10 --dynamical-output 6 10 --potential -o out/
```

The native density remains the image-reprojection-consistent inference. The optional
``dynamical/`` product is a non-negative, exactly bisymmetric 3-D derivative: it removes
all modes except ``m=0,2,4``, conserves the azimuthally averaged ``(R,z)`` mass, and tapers
``m=2,4`` to zero over the requested radial interval. Its projected image is therefore a
diagnostic rather than an exact reproduction of clumpy input structure.

- **Geometry:** pass `pix_arcsec` + `pa_pix_deg` (+ `center`) to bypass WCS (e.g. an S4G
  cutout), or let a FITS WCS supply the pixel scale for an on-sky `pa_onsky_deg`.
- **Mass scale:** the deprojected *shape* is M/L-independent; M/L only sets the v_c
  amplitude. Fix the absolute scale with `stellar_mass` (total M\*), `luminosity` × `ml`,
  or `zeropoint` + `band_solar_mag` + distance; with none, outputs use a relative scale.

---

Developer docs — project handoff, milestone reports, and the cluster/TNG50 ingestion
workflow — live under [`docs/`](docs/) (full original README: [`docs/README-full.md`](docs/README-full.md)).
