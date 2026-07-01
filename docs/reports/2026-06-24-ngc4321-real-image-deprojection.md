# First end-to-end deprojection of a real S4G galaxy: NGC 4321 (M100)

Date: 2026-06-24

A first test of the deprojection backbone on a real near-infrared image, rather than a
TNG mock or N-body model. The user supplied `NGC4321_m_c_r_f.fits` (S4G 3.6 um, IRAC,
foreground stars subtracted and refilled). This report covers: confirming the image is
NOT pre-deprojected, the adopted geometry, the image-anchored Fourier x (R,z)
deprojection, the bar (NGC 4321 is double-barred) via isophote ellipse fitting, the
rotation curve, the vertical-structure question (geometric sech^2 vs the TNG-learned
q_m), a boxy/peanut (b4) test, and finer-R refinement without retraining.

> **Headline.** The **in-plane** recovery is solid and observationally grounded: the
> deprojected bar (a = 4.75 kpc, b/a = 0.43) matches the user's students' ellipse-fit
> bar, and the stellar rotation curve is sensible (~160 km/s for M* = 6e10, with M100's
> full v_c ~210 km/s requiring DM+gas). The **vertical** structure is the weak,
> prior-dominated part: a face-on image carries no vertical information, so it is set by
> assumption (geometric sech^2) or by the TNG-learned q_m (out-of-distribution). The
> learned profile is much thicker than the (deliberately thin) sech^2 baseline; the
> *direction* is physically right (the old disk / bulge / bar are genuinely thick) but
> the *absolute* thickness is unconstrained here (see the caveat below).

## 1. The image is the original observed mosaic, NOT deprojected

(`scripts/ellipse_fit_ngc4321.py` header check; diagnostics in the session.) Decisive
evidence the FITS is the original sky image (just star-cleaned), not stretched to
face-on:

| test | result | meaning |
| - | - | - |
| WCS pixel scale (SVD of CD) | 0.7500 x 0.7500 arcsec, anisotropy **1.0000** | square original pixels; a deprojection stretch at i=34.6 would force a 1.21 anisotropy |
| SIP distortion (`A_ORDER` ...) | present | native Spitzer/S4G astrometric solution |
| CTYPE / CRVAL | `RA---TAN/DEC--TAN`, ICRS, (185.727, +15.821) | valid sky astrometry at NGC 4321's catalog position |
| outer-disk isophote ellipticity | eps ~ 0.10-0.14 (a > 11 kpc) | an inclined disk (i ~ 25-30 deg); a face-on/deprojected image would be round (eps ~ 0) |

So the deprojection (an anisotropic stretch) is the correct operation to apply, not a
double-deprojection.

## 2. Geometry and units

S4G data release values (confirmed against the image): inclination **i = 34.6 deg**
(matches the outer-disk ELLIP = 0.177), disk **PA = 158.2 +/- 4.5 deg**. Cepheid
distance **D = 15.2 Mpc** -> pixel scale 0.75 arcsec = **0.0553 kpc/pixel**. Light is
treated as mass up to a constant M/L (a global scaling that does NOT affect the
deprojection geometry); the grid total is normalized to a literature stellar mass
**M* = 6e10 Msun**, and v_c scales as sqrt(M*/M*_assumed).

## 3. Deprojection method (the transferable backbone)

`scripts/deproject_real_image_ngc4321.py` (geometric) and
`scripts/deproject_real_image_ngc4321_learned.py` (learned). The image-anchored,
mass-conserving even-m Fourier x (R,z) form: deproject the light into the disk-plane
surface density Sigma_m(R) (measured from the image), apply a vertical profile q_m(z;R),
build the 3D density, and wrap it as an `agama.Potential` (`dgdp.agama_density`). Two
choices for the vertical profile:

- **geometric baseline**: a fixed sech^2(z/h), h = 0.3 kpc (a *chosen* prior, see Section 6);
- **TNG-learned q_m(z;R)**: the conserving method's vertical-profile head (PCA-32 + MDN),
  trained on the TNG milestone-2d table and applied here out-of-distribution.

In both, the in-plane Sigma_m(R) is identical (image-measured); only the vertical
structure differs - so the two are a clean isolation of "what the learned vertical step
changes." Figures (under `outputs/real_images/`): `ngc4321_deprojection.png`,
`ngc4321_deproject_faceon_bar.png` (observed -> face-on, bar on x, with edge-on rows).

## 4. NGC 4321 is double-barred (isophote ellipse fitting)

`scripts/ellipse_fit_ngc4321.py` (photutils `isophote.Ellipse`, the IRAF algorithm the
S4G bar catalogs use). Fit isophotes -> eps(a), PA(a); the bar is the ellipticity
maximum with ~constant PA, ending where eps drops and PA twists. The profile shows the
textbook **double-bar** signature:

| component | a (eps-max) | eps (b/a) | PA |
| - | -: | -: | -: |
| nuclear bar/disk | 0.6 kpc | 0.60 (0.40) | 40 deg |
| **main bar (apparent/sky)** | **4.2 kpc** | **0.55 (0.45)** | **41 deg** |
| main bar (deprojected) | 4.75 kpc | 0.43 | ~57 deg from disk major axis |

with the bar end (eps drop + PA twist 41->65 deg) at a ~ 5.6 kpc. The apparent bar
matches the user's students' ellipse-fit measurement in size, ellipticity, and
orientation. (An earlier flux-weighted second-moment estimate gave b/a ~ 0.95 -
"round" - because azimuthal averaging over the bulge and winding spiral washes the bar
out; the ellipse-fit eps-maximum is the correct method.) Figure
`ngc4321_ellipse_fit_bar.png`.

## 5. Rotation curve (in-plane, well constrained)

`ngc4321_rotation_curve.png` (direct softened summation over the grid cells; full radial
resolution). The stellar rotation curve peaks ~160 km/s and shows resolved inner
structure - a steep rise to ~155 km/s by ~1 kpc, a dip to ~133 at R ~ 3-4 kpc, then a
rise to ~160 at 6-10 kpc and a gentle decline. M100's full v_c (~210 km/s) needs DM+gas,
so ~160 km/s for the stars (M* = 6e10) is consistent.

**Vertical structure barely changes v_c here** (geometric baseline vs learned overlap to
a few percent), for two reasons established with a controlled test
(`scripts/thick_disk_vc_test.py`): (i) both share the same Sigma(R), and (ii) the learned
profile *flares* (thin center, thick outskirts) rather than puffing the center, so the
central v_c - which is set by the central mass - is only mildly affected. The control
confirms the underlying physics the user flagged: an exponential disk with the *same*
Sigma(R) made uniformly 3x thicker (h 0.3 -> 1.0 kpc) drops the central force ~22-27%
(v_c x0.86 at R=1), and 6.7x thicker ~40-48% (v_c x0.72), converging at large R; and the
"uniform h = 1 kpc" control on NGC 4321's own Sigma(R) drops central v_c to x0.81. So
thickness *does* matter for v_c when it is central - the learned model just happens to
put the extra thickness in the outskirts.

## 6. Vertical thickness: thicker than a too-thin baseline (how much is uncertain)

Mass-weighted RMS|z| (R < 12 kpc): geometric baseline **0.27 kpc** (sech^2, h = 0.3) vs
TNG-learned **~1.5 kpc**, with the learned profile flaring from RMS|z| ~ 0.5 kpc at
R = 0.3 to ~2.0 kpc at R = 15. The learned thickness is **not** a mass-extrapolation
artifact: NGC 4321 (M* = 6e10) sits essentially at the TNG training median image mass
(4.75e10), and using NGC 4321's real mass vs the TNG median for the conditioning features
gives the same thickness - so it is driven by morphology + the learned TNG vertical
prior, not by an out-of-range mass feature.

**Interpretation (deliberately cautious).** The ratio "~5.7x thicker" is *relative to the
sech^2 h = 0.3 kpc baseline, which is itself too thin* - 3.6 um traces the old stellar
population, and the bulge/bar region is genuinely vertically extended. So the learned
step's *direction* (much thicker, flaring) is physically reasonable. What is **not**
established is the *absolute* thickness: the learned ~1.5 kpc RMS inherits TNG50's
vertical scale, which may be inflated by the simulation's ~100-300 pc gravitational
softening and numerical disk heating, and a **face-on image provides no vertical
constraint at all**. The true vertical structure of NGC 4321's old disk/bulge/bar lies
somewhere between the too-thin 0.3 kpc baseline and (possibly) the TNG prior, and pinning
it down needs edge-on data or stellar kinematics. We therefore report the learned profile
as "much thicker than the baseline, flaring" and avoid claiming it over- or
under-estimates the truth. (NB: whether the TNG q_m prior is genuinely too thick is itself
**unverified** - the softening-inflation point is a hypothesis, not a measurement. The
check, and the prerequisite for any retrain: measure the TNG milestone sample's stellar
h_z(R) from the truth grids and compare to a real edge-on h_z sample, e.g. Comeron+2018;
that decides whether the OOD is thickness or morphology.)

## 7. Boxy/peanut (b4) test: not a peanut

`scripts/peanut_strength_ngc4321.py` applies the validated tilt-invariant b4 isophote
metric (`scripts/peanut_strength.py`) to the bar-side-on surface density, against the
ground-truth anchors:

| case | median b4 | reading |
| - | -: | - |
| thin exp disk (anchor) | -0.077 | disky |
| NGC 4321 baseline (sech^2) | -0.041 | disky/thin |
| **NGC 4321 learned (TNG q_m)** | **-0.016** | neutral (ellipse-like) |
| tilted ellipsoid (anchor) | -0.005 | neutral |
| Shen2010 (anchor) | +0.042 | strong peanut |

The learned edge-on is **not** a boxy/peanut bulge (median b4 = -0.016, at the neutral
ellipsoid anchor, far below the Shen2010 peanut +0.042). The learned q_m *rounds* the
disk (axis ratio q goes 0.16 -> 0.51) without pinching the isophotes. A "central
double-peak" noticed in an earlier edge-on figure was a **rendering artifact** - that
figure used a thin y=0 *slice* through the central column (plus coherent m=2/4
harmonics); the line-of-sight *projection* (the observable, and the correct view for
peanut-hunting) is single-peaked, consistent with b4 = neutral. Figures
`ngc4321_peanut_strength.png`, `ngc4321_edgeon_slice_vs_proj.png`.

## 8. Finer radial bins without retraining

The conserving design splits a_m(R,z) = Sigma_m(R) q_m(z;R) into an **image-measured**
radial anchor Sigma_m(R) and a **predicted** vertical profile q_m on fixed coarse R-knots.
So the output radial resolution is set by the anchor (the image), not the model: bumping
the NGC 4321 output grid to **128 log R bins** (`--n-r-out 128`) re-measures Sigma_m(R)
finely from the image and interpolates the existing q_m onto it - **no retraining** (the
q_m head stays trained on the 32-R table). This resolves the inner radial structure of
v_c (the R ~ 3-4 kpc dip and the central rise) that a coarse grid smooths. The rotation
curve is computed by direct summation over the 128-R cells (the smooth Fourier/CylSpline
representation otherwise caps the radial resolution at ~25 knots). Retraining would only
be needed to *predict* q_m on more independent R-knots (finer vertical-profile
R-variation), which - given q_m is smooth and the prediction is image->coefficient
limited - would add little.

## 9. Caveats and conclusions

- **In-plane (radial + bar): trustworthy and observationally anchored.** The deprojected
  bar matches an independent ellipse-fit; the rotation curve is sensible and finely
  resolved.
- **Vertical: a prior, not a measurement.** A face-on image cannot constrain the vertical
  structure. The geometric baseline (h = 0.3 kpc) is too thin; the TNG-learned q_m is
  thicker and flaring (direction reasonable) but its absolute scale is OOD and
  unconstrained. Treat it as a prior to be validated against edge-on / kinematic data.
- **No genuine peanut** is recovered for NGC 4321 (b4 neutral); the learned step rounds
  rather than pinches.
- **M/L and distance** only rescale amplitudes (v_c ~ sqrt(M*), lengths ~ D); the
  morphology/geometry results are independent of them.

## Code and artifacts

Scripts (run with `PYTHONPATH=/home/lucyundead/Agama/build/lib.linux-x86_64-cpython-313`
where AGAMA is used):

- `scripts/deproject_real_image_ngc4321.py` - geometric image-anchored Fourier x (R,z)
  deprojection + AGAMA potential + rotation curve.
- `scripts/deproject_real_image_ngc4321_learned.py` - the TNG-learned q_m vertical step
  vs the geometric baseline (`--n-r-out` for the output R bins; saves `ngc4321_arrays.npz`).
- `scripts/ellipse_fit_ngc4321.py` - isophote ellipse fitting -> bar (eps-max method).
- `scripts/peanut_strength_ngc4321.py` - b4/boxiness metric vs ground-truth anchors.
- `scripts/thick_disk_vc_test.py` - controlled thin-vs-thick disk rotation-curve test.
- `scripts/plot_ngc4321_figures.py` - the figures (face-on + edge-on, bar on x; rotation curve).

Figures under `outputs/real_images/`: `ngc4321_deproject_faceon_bar.png`,
`ngc4321_rotation_curve.png`, `ngc4321_ellipse_fit_bar.png`, `ngc4321_peanut_strength.png`,
`ngc4321_learned_vs_baseline.png`, `ngc4321_edgeon_both_disks.png`,
`ngc4321_edgeon_slice_vs_proj.png`, `ngc4321_deprojection.png`, `thick_disk_vc_test.png`.

New dependencies installed into the `.venv` for the real-image work: `astropy`,
`photutils` (binary wheels; FITS/WCS reading and isophote fitting). `ruff check .` clean;
`pytest -q` unaffected (the real-image scripts have no unit tests; the AGAMA-density module
is covered by `tests/test_agama_density.py`).
