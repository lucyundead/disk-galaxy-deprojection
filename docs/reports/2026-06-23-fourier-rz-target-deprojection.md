# Compressing the Fourier x (R,z) target and wiring it into the deprojection

Date: 2026-06-23

This continues the adopted grid-free 3D backbone (SPH-KDE truth + even-m azimuthal
Fourier x smooth (R,z) maps; see
`docs/reports/2026-06-13-nbody-shen2010-deprojection.md`). Two steps:

1. **Compress** the Fourier x (R,z) representation from the uniform ~2002
   coefficients to a few hundred via a power-weighted per-harmonic (R,z)
   allocation.
2. **Wire** the Fourier x (R,z) coefficients in as the deprojection *target* -
   predict them from the image, retrain the MDN and a conditional flow, and
   compare held-out 3D recovery against the current PCA-on-grid pipeline.

New package module `src/dgdp/fourier_rz.py` (the representation, with unit tests
`tests/test_fourier_rz.py`); the previously script-local `fourier_rz_fit` /
`fourier_rz_reconstruct` in `scripts/reconstruct_superellipsoid_3d.py` were moved
here unchanged (the reconstruction numbers reproduce exactly: Shen2010 0.385, TNG
554189 0.307, TNG 392276 0.370). `ruff check .` clean; `pytest -q` 108 passed.

## 1. Power-weighted (R,z) compression

`scripts/compress_fourier_rz_target.py`. The uniform model spends an identical
(14 R x 13 z) knot grid on every even harmonic m<=10 = 2002 real coefficients, but
the harmonics carry very unequal volume-weighted azimuthal power. Measured on the
three full-particle reference galaxies (SPH-KDE truth), the aggregate fractions are

| m | 0 | 2 | 4 | 6 | 8 | 10 |
| - | -: | -: | -: | -: | -: | -: |
| power fraction | 0.854 | 0.100 | 0.026 | 0.010 | 0.006 | 0.005 |

(NOTE: m=6,8,10 carry ~0.5-1% each here, not the ~0.1% recalled from earlier
notes - the high harmonics of an equal-mass N-body / particle SPH-KDE density are
substantially shot-noise.)

**Method (reverse water-filling, `derive_power_allocation` /
`derive_power_allocation_multi`).** For each harmonic and each candidate (R,z) grid
we measure the power discarded by coarsening (resample down then back up). A single
global threshold is raised until the cumulative discarded power reaches a budget
`(1 - capture)`; each harmonic then takes the fewest-coefficient grid within the
threshold, and harmonics whose total power falls below it drop out. A sweep over
`capture` exposes the coefficient-count / fidelity knee.

**Result (capture 0.95 -> 496 coefficients, 25% of 2002).** The allocation keeps
the dynamically important low harmonics - crucially at **full vertical
resolution** (13 z knots), which carries the disk thinness and the boxy/peanut X -
and drops the shot-noise-dominated high harmonics:

| m | 0 | 2 | 4 | 6,8,10 |
| - | -: | -: | -: | -: |
| (n_R, n_z) | (12, 13) | (8, 13) | (6, 11) | dropped |

3D rel-L2 vs the SPH-KDE truth is preserved and stays far below the cylindrical
grid on all three galaxies:

| galaxy | full (2002) | compressed (496) | grid 0.3125 kpc (49152) |
| - | -: | -: | -: |
| Shen2010 | 0.385 | **0.377** | 0.465 |
| TNG 554189 | 0.307 | **0.331** | 0.600 |
| TNG 392276 | 0.370 | **0.380** | 0.444 |

The fidelity is flat down to ~496 coefficients and only degrades below ~350
(`figures/fourier_rz_compression.png`). The chosen allocation is saved to
`outputs/nbody_shen2010/fourier_rz_allocation.json`.

## 2. Fourier x (R,z) as the deprojection target

`scripts/deproject_fourier_rz_compare.py`. A head-to-head where **everything is
held equal except the target representation** the model predicts:

- **PCA-on-grid** (current pipeline): flatten the cylindrical residual grid
  (delta_mass / baseline_total, 49152 cells), cross-galaxy PCA-32;
- **Fourier x (R,z)** (new): azimuthally FFT the residual grid, keep even m<=10 on
  the power-weighted per-harmonic (R,z) allocation derived *on the residual*
  (608 coefficients at capture 0.99: m0 (12,13), m2 (12,13), m4 (10,7), m>=6
  dropped), then cross-galaxy PCA-32 of those coefficients.

Both get the same 586-dim image+geometry features, the same galaxy-level
milestone-2d split (999/333/333 rows), and the same 1-Gaussian MDN; the Fourier
target additionally gets a conditional flow. Held-out recovery reconstructs each
predicted residual back onto the grid (eval scaffold), adds the geometric
baseline, and scores against the TNG truth. Built from the fine-z (0.3125 kpc)
table `/mnt/e/dgdp-milestone2d/density_residual_table.npz` - **no particle
re-extraction needed** (the FFT is taken directly on the stored residual grids).

Note: on the *residual* (not the density), m=0 carries ~95% of the power (the
axisymmetric vertical-thickening correction dominates), so the residual is less
compressible than the density - reaching the even-m representation ceiling needs
~600 coefficients (vs 496 for the density).

### Held-out results (test: 37 galaxies / 333 rows, PCA-32)

| method | n_target | cell-mass MAE [Msun] | density rel-L2 | resid rel-L2 | RMS-z MAE [kpc] | bar m=2 MAE | cov68 |
| - | -: | -: | -: | -: | -: | -: | -: |
| geometric baseline | - | 1.624e6 | 0.495 | - | 1.007 | 0.0409 | - |
| **grid MDN** | 49152 | **5.93e5** | **0.428** | **0.406** | **0.123** | 0.0385 | 0.762 |
| Fourier MDN | 608 | 7.37e5 | 0.787 | 0.462 | 0.192 | 0.0392 | 0.731 |
| Fourier flow | 608 | 7.15e5 | 0.677 | 0.455 | 0.214 | **0.0330** | 0.727 |
| grid oracle (true coeff) | - | 3.70e5 | 0.372 | 0.281 | 0.033 | 0.0088 | - |
| Fourier oracle (true coeff) | - | 5.16e5 | 0.593 | 0.379 | 0.165 | 0.0120 | - |

### Findings

1. **The Fourier x (R,z) target does NOT improve held-out 3D recovery on the noisy
   TNG grid.** Grid PCA is ~20-24% better on cell-mass MAE (5.93 vs 7.37e5), ~11%
   on residual rel-L2, and better on vertical RMS-z and (inner-cell-weighted)
   density rel-L2. The cause is structural: the even-m + (R,z) projection discards
   ~25% of the raw grid residual (odd-m lopsidedness + high-m + fine cell
   structure), and the grid PCA can still *partially predict* some of that, so the
   Fourier representation ceiling (oracle resid rel-L2 0.379) sits below the grid's
   (0.281). This is robust across capture (608-1014 coefficients give the same
   conclusion; the oracle is flat above ~600).

2. **They tie on the smooth bar m=2 amplitude** - the symmetric quantity the
   even-m basis is built for (grid 0.0385, Fourier MDN 0.0392) - and the
   **conditional flow gives the best m=2 of all (0.0330)**, consistent with the
   prior finding that the flow's tighter, mildly non-Gaussian posterior helps
   symmetric structure. Otherwise flow ~ MDN (confirming the slide-15 deferral on
   TNG).

3. **The Fourier target is more predictable relative to its (lower) ceiling**: the
   MDN/oracle cell-mass-MAE gap is 1.43 for Fourier vs 1.60 for the grid (the
   smoother target is easier to learn) - but its ceiling is lower because it
   discards predictable mass structure along with the noise, so the net is a
   tie-to-slight-loss.

4. **Interpretation.** The grid-histogram metric is biased toward the grid
   representation (it rewards matching the grid's own cell noise). The Fourier x
   (R,z) target's established advantage is on a *smooth* SPH-KDE truth (Step 1:
   0.377/0.331/0.380 vs grid 0.465/0.600/0.444) and in being the compact,
   symmetric, potential-ready CylSpline form - **not** improved image->3D
   prediction. This reinforces the standing conclusion that the bottleneck is
   image->coefficient predictability, not the representation or the generator.

## Mass-conserving image-anchored variant (`scripts/deproject_fourier_rz_conserving.py`)

Driven by the deprojection goal (image -> smoothed 3D density with total mass
conserved). Re-parameterize each harmonic as ``a_m(R,z) = Sigma_m(R) * q_m(z;R)``
with ``int q_m dz = 1``, where ``Sigma_m(R)`` (radial azimuthal-Fourier structure)
is *measured* from the deprojected image (the geometric baseline) and ``q_m(z;R)``
(normalized vertical profiles) is *predicted*. Only m=0 carries net mass and
``int q_0 dz = 1``, so the total mass is exactly the image's mass for ANY predicted
profiles - conservation is a hard property, not a fitted correction. The model only
ever predicts the genuinely unobserved vertical structure.

The vertical profiles q_m(z;R) are predicted on the compact, smooth power-allocated
knots (the 496-coeff Task-1 allocation), while Sigma_m(R) is kept on the FINE grid R
(measured from the baseline) so the radial structure - which sets v_c - is not
degraded by coarse knots. A 1-scalar Sigma_0 mass-correction head (predict
log M_truth/M_image) optionally rescales the controlled total to the truth mass.

Held-out (37 gal / 333 rows), MDN. All Fourier variants share the SAME predicted
vertical profiles q_m; only the radial anchor Sigma_m(R) differs - image (measured),
+radial-corr (image + a predicted Sigma_0(R) profile correction = step a), oracle
(true Sigma_m = ceiling). The total mass is a controlled scalar in all (no 3D drift).

| method | n_target | cell-mass MAE | density rel-L2 | mass vs truth |
| - | -: | -: | -: | -: |
| geom baseline | - | 1.624e6 | 0.495 | 0.075 |
| grid MDN (non-conserving, drifts 6.8% off image mass) | 49152 | 5.93e5 | 0.428 | 0.028 |
| Fourier conserving, image anchor | **496** | 6.80e5 | 0.470 | 0.114 |
| + Sigma_0(R) correction (m=0) | **496** | 5.79e5 | 0.378 | 0.061 |
| **+ Sigma_0,2,4(R) correction (all, step a)** | **496** | **5.68e5** | **0.375** | 0.061 |
| Fourier conserving, oracle anchor (ceiling) | **496** | 4.68e5 | 0.337 | 0.032 |

Findings: (1) **exact total-mass control** (output total = a controlled scalar, no
3D-shape drift) with a compact (496-coeff) *smooth* CylSpline-ready field. (2) The
**radial Sigma_m(R) correction heads** (step a; dedicated heads predicting the radial
log-ratio for m=0 and complex ratios c_m(R) for the bar m=2,4) are the key: image->all
improves cell-mass MAE 6.80->5.68e5, rel-L2 0.470->0.375, mass-vs-truth 0.114->0.061.
m=0 does the bulk (->5.79e5); the m=2,4 bar corrections add a further ~2% (->5.68e5).
With them the smooth 496-coeff conserving rep **beats the 49152-cell grid pipeline on
BOTH cell-mass MAE (5.68 vs 5.93e5) and rel-L2 (0.375 vs 0.428)**. (3) **The residual
gap to the oracle anchor (4.68e5, rel-L2 0.337) is predictability-limited**: the oracle
uses the TRUE Sigma_m(R), and the heads cannot fully predict the radial corrections from
the image - the same image->coefficient bottleneck. So the learned anchor is near its
ceiling; closing the rest needs more information (multi-band / kinematics), not a richer
representation or vertical model.

**v_c re-check (`scripts/plot_rotation_curves.py`, mass-conserving construction).**
Pinning the projected surface density to the truth (the mass-conserving construction)
fixes the outer-v_c overshoot where the surface density is accurate: vs the particle
rotation curve, the Fourier rep's v_c rms improves Shen 11.3->5.5 and 392276
10.0->4.3 km/s (both now *better* than the grid's 9.0 / 8.4), confirming the overshoot
was the radial/surface-density error, not the rep form. On the low-N 554189 it does
not help (7.6->11.7) because the anchor surface density is itself noisy there - same
N-dependence as everywhere (smoothing helps at low N; an accurate anchor helps at high
N). Consistent with finding (3): v_c quality is set by the radial anchor.

## Implications and next steps

- For the deprojection goal, prefer the **mass-conserving image-anchored** Fourier x
  (R,z) form above: it conserves mass by construction at ~grid recovery, and focuses
  learning on the unobserved vertical structure.
- Adopt the Fourier x (R,z) representation for what it is good at: a compact
  (~500), smooth, bar-symmetric, **potential-ready** density model that beats the
  grid against a smooth truth. Do **not** expect it to beat PCA-on-grid for raw 3D
  recovery on the noisy TNG grid; keep the grid PCA as the production point
  predictor for now and the grid as the eval scaffold.
- The honest prediction comparison should be on a *smooth* truth, but that needs
  SPH-KDE truth (hence particles) for the held-out galaxies - only 2 are local;
  the rest is the deferred N-body / uncapped-particle work.
- Still open (handoff next-steps): (3) potential/force validation via AGAMA
  CylSpline on Shen2010 (truth vs deprojection) - the Fourier x (R,z) maps are now
  in the CylSpline form, so this is unblocked; (4) a proper B/P census pipeline;
  (5) the N-body library for strong-X training/validation (deferred).

## Code and artifacts

- `src/dgdp/fourier_rz.py` - the representation: `fit_fourier_rz` (from a density
  callable), `fit_fourier_rz_from_grid` (from a cylindrical grid),
  `reconstruct_fourier_rz` (`nonneg=False` for residuals), `coefficient_vector` /
  `model_from_vector`, `harmonic_power`, `derive_power_allocation[_multi]`.
- `tests/test_fourier_rz.py` - round-trip, bar symmetry, power concentration,
  compression, vector round-trip, grid-vs-callable agreement.
- `scripts/compress_fourier_rz_target.py` - Step 1; writes
  `outputs/nbody_shen2010/fourier_rz_allocation.json` and
  `figures/fourier_rz_compression.png`.
- `scripts/deproject_fourier_rz_compare.py` - Step 2; writes
  `outputs/nbody_shen2010/fourier_rz_deprojection_metrics.json` and
  `figures/fourier_rz_deprojection_compare.png`.
- `scripts/reconstruct_superellipsoid_3d.py` - refactored to import the module
  (behaviour unchanged).
