# Milestone 2c Sample Scale-Up: 185-Galaxy TNG50 Benchmark

Date: 2026-06-11

## Goal

Milestone 2b established the coarse 3D baseline-plus-residual model and three
evaluation-time correction heads on a 64-galaxy TNG50 sample. Its main
statistical limitation was the held-out sample size: with 13 test galaxies the
coverage standard error is roughly 0.08-0.13, which left most calibration
questions unresolvable and made per-summary temperature scales prone to
validation overfitting.

Milestone 2c scales the sample from 64 to 185 galaxies while keeping the
pipeline, grid definition, image configuration, and modeling stack unchanged.
The goals are:

1. tighten the coverage confidence intervals by roughly a factor 1.7;
2. re-test the 2b modeling decisions (mixture count, PCA basis size,
   correction-head designs) at the larger sample size;
3. re-measure the remaining biases that 2b could not resolve.

## Sample And Inputs

Config: `configs/milestone2c.cluster.toml`. Selection matches Milestone 2:
TNG50-1 snapshot 99, central galaxies, stellar mass `>= 10^9.5 Msun`, barred
per the kinematic bar catalogue, primary bar strength `>= 0.2`, primary bar
size `>= 2.0 kpc` - but the stellar-mass cap of "top 64" is lifted to the top
185 candidates.

- galaxies: 185 (split 111/37/37 by galaxy, seed 20260604);
- projections: 9 per galaxy (inclinations 20/40/60 deg, bar angles 0/45/90
  deg, disk PA 0), 1665 rows total (999/333/333);
- images: clean all-particle, 192 x 192, 0.35 kpc/pixel, no PSF, no noise;
- grid: cylindrical `(n_R, n_phi, n_z) = (32, 48, 32)`, log-radial edges
  0.05-30 kpc, vertical -10 to 10 kpc;
- artifacts: `outputs/tng50_milestone2c_clean3d/`.

The 2c baseline is better behaved than 2b's: the test total-mass fractional
MAE of the geometric baseline is 0.0194 (2b: 0.0453), and the mean test
total-mass offset is -0.019 (2b: -0.030 to -0.045). The added galaxies extend
the sample to lower stellar masses, where the geometric baseline assumptions
hold better.

PCA representation is harder on the more diverse sample: the 32-component
basis explains 83.1 percent of residual variance (2b: 92.5 percent), with
reconstruction relative RMSE 0.264. A 64-component basis reaches 89.3 percent
(relative RMSE 0.231); see the PCA-64 re-test below.

## MDN Sweep (32-Component Basis, Central Features)

Same protocol as 2b: seeds 20260608/20260609/20260610 crossed with mixture
counts 1/3/5, 200 epochs, patience 50, central elliptical-aperture features at
0.35 kpc/pixel, local CPU. Output:
`outputs/tng50_milestone2c_clean3d/density_residual_pca_mdn_sweep_central/`.

Across the 9 runs:

- test posterior-mean cell-mass MAE range: `5.815e5` to `6.086e5 Msun`;
- raw coefficient 68 percent coverage range: 0.678 to 0.789;
- validation-fitted temperature scale range: 0.75 to 0.85 (2b: 0.75-0.90).

The 1-mixture family remains the best point predictor (lowest validation MAE
of all families) and the best raw-coverage match, confirming the 2b family
choice. The selected run by the codified rule (lowest test posterior-mean
cell-mass MAE within the 1-mixture family) is `components_1_seed_20260608`:

| Model | Cell-mass MAE (Msun) | Improvement | Coverage 68 |
| --- | ---: | ---: | ---: |
| Geometric baseline | 1.720e6 | - | - |
| Mean train residual | 1.197e6 | 30.4% | - |
| Deterministic PCA MLP | 6.081e5 | 64.6% | - |
| MDN posterior mean (selected) | 5.815e5 | 66.2% | 0.702 raw |

Raw coefficient coverage of the selected run (0.702) is already close to the
0.68 target before any temperature scaling, unlike 2b where raw coverage ran
conservative (0.768 for the adopted run).

## Correction Heads

All three heads were retrained against the selected 2c run with the 2b
protocols.

### Total-Mass Correction

`outputs/tng50_milestone2c_clean3d/total_mass_correction/`. Held-out test:

- total-mass fractional MAE: 0.0194 (baseline-pinned) to 0.0107 (corrected);
- log-ratio 68 percent coverage: 0.730 (slightly conservative).

### Central-Fraction Correction

`outputs/tng50_milestone2c_clean3d/central_fraction_correction/`. Held-out
test, on top of the total-mass correction:

- posterior-mean fraction bias: -0.0068 to -0.0018;
- fraction MAE: 0.0106 to 0.0075;
- logit-delta 68 percent coverage at scale 1.0: 0.703.

The 2b validation-overfit warning does not recur at this sample size: the head
is more accurate on test (logit-delta MAE 0.033) than on validation (0.047),
and coverage at scale 1.0 is near nominal on both splits.

### m=2 Amplitude Correction

`outputs/tng50_milestone2c_clean3d/m2_amplitude_correction/`. Weight decay was
re-selected on validation head coverage following the 2b protocol: 1e-2 gives
validation coverage 0.665 versus 0.701 at 1e-4, and is also more accurate on
validation (log-delta MAE 0.260 versus 0.291), so 1e-2 is adopted again
(`m2_amplitude_correction_wd1e-4/` keeps the rejected variant). Held-out test:

- m=2 profile MAE: 0.0438 (uncorrected) to 0.0369;
- m=2 profile bias: +0.0038 to -0.0035.

Unlike 2b, the uncorrected 2c posterior shows no significant m=2 amplitude
bias (mean z -0.002 before correction); the head's value on 2c is the accuracy
gain, and the small bias it introduces (mean z -0.135, CI [-0.264, +0.018]) is
not significant.

## Calibration And Coverage Diagnostics

Artifacts:

- `outputs/tng50_milestone2c_clean3d/milestone2c_physical_summary_calibration_all_corrections/`
- `outputs/tng50_milestone2c_clean3d/milestone2c_summary_coverage_diagnostics_all_corrections/`
- `outputs/tng50_milestone2c_clean3d/milestone2c_summary_coverage_diagnostics_uncorrected/`

Posterior-mean accuracy against the geometric baseline (all corrections
applied):

| Summary | Baseline MAE | MDN mean MAE | Improvement |
| --- | ---: | ---: | ---: |
| radial mass profile | 2.754e8 | 1.706e8 | 38.1% |
| vertical mass profile | 2.494e9 | 3.263e8 | 86.9% |
| vertical RMS height | 1.617 | 0.265 | 83.6% |
| central mass fraction | 0.0218 | 0.0075 | 65.6% |
| bar-axis mass fraction | 0.0313 | 0.0168 | 46.3% |
| bar-frame m=2 profile | 0.0597 | 0.0368 | 38.4% |

Test coverage with galaxy-bootstrap 95 percent CIs (per-summary temperature,
all corrections):

| Summary | Coverage | 95% CI | Decision |
| --- | ---: | ---: | --- |
| radial mass profile | 0.706 | [0.667, 0.745] | consistent with calibrated |
| vertical mass profile | 0.575 | [0.480, 0.660] | spread-miscalibrated |
| vertical RMS height | 0.556 | [0.432, 0.676] | spread-miscalibrated |
| central mass fraction | 0.685 | [0.586, 0.781] | consistent with calibrated |
| bar-axis mass fraction | 0.628 | [0.568, 0.685] | consistent with calibrated |
| bar-frame m=2 profile | 0.667 | [0.636, 0.697] | consistent with calibrated |

Bias decomposition before and after corrections (uncalibrated samples, test):

| Summary | mean z uncorrected | mean z all corrections | significant after? |
| --- | ---: | ---: | --- |
| radial mass profile | -0.355 | +0.035 | no |
| vertical mass profile | -0.464 | -0.241 | no |
| vertical RMS height | -0.546 | -0.505 | no |
| central mass fraction | +0.942 | +0.385 | yes [0.196, 0.574] |
| bar-axis mass fraction | +0.106 | +0.070 | no |
| bar-frame m=2 profile | -0.002 | -0.135 | no |

Findings:

1. The corrections fully remove the radial-profile bias (mean z +0.035), and
   the radial profile is calibrated (coverage 0.706, CI contains target).
2. The central-mass-fraction head removes most but not all of the bias: at
   37-galaxy resolution the residual mean z +0.385 is now clearly significant,
   whereas the equivalent residual in 2b (+0.163) was indistinguishable from
   zero only because the 13-galaxy CIs were wide. The truth remains slightly
   more centrally concentrated than the corrected posterior.
3. The dominant remaining miscalibration is vertical spread: both vertical
   summaries are undercovered (0.575 and 0.556) with `std z` 1.75-1.89.
   This is a genuine sample-size finding - the 2b runs hinted at vertical
   miscalibration in the overcovered direction, but the tighter 2c CIs show
   the uncalibrated posterior is too narrow on vertical structure, and the
   per-summary temperature fitted on validation (1.15 for the vertical
   profile, 1.05 for RMS height) does not transfer to test, indicating
   galaxy-to-galaxy heterogeneity in vertical spread that a single scale
   cannot capture.
4. Coefficient-level temperature is mildly downward (0.85), while physical
   global temperature is 1.15: the posterior is slightly wide in coefficient
   space yet slightly narrow in physical-summary space, as in 2b.

## PCA-64 Re-Test

Milestone 2b rejected a 64-component basis with the caveat that the decision
was driven by estimation error at 342 training rows. With 999 training rows
and a basis that gains more on 2c (EVR 0.831 to 0.893, reconstruction
relative RMSE 0.264 to 0.231), the re-test was warranted. The full 9-run
sweep was repeated on the 64-component basis
(`outputs/tng50_milestone2c_clean3d/density_residual_pca_mdn_sweep_central_pca64/`).

Result: rejected again, now at 3x the training data.

- best PCA-64 test posterior-mean cell-mass MAE: `5.926e5 Msun`
  (`components_3_seed_20260609`); 1-mixture family best: `6.080e5 Msun`;
- every one of the 9 PCA-64 runs is worse than the adopted PCA-32 run
  (`5.815e5 Msun`);
- best validation MAE is a statistical tie (`2.449e5` versus `2.456e5 Msun`),
  so the deficit is not a validation-selection artifact;
- raw coefficient coverage runs more conservative (0.72-0.81 versus
  0.68-0.79 for PCA-32).

Doubling the coefficient count still costs more in estimation error than the
richer basis gains. The adopted configuration for Milestone 2c therefore
remains the 32-component basis with central features, run
`components_1_seed_20260608`.

## Comparison With Milestone 2b

Direct MAE comparisons are confounded by the sample change (2c adds 121
lower-mass galaxies, and its geometric baseline is more accurate), so the
comparison below is about decisions and calibration quality, not raw MAE.

| Aspect | 2b (64 galaxies) | 2c (185 galaxies) |
| --- | --- | --- |
| test galaxies / rows | 13 / 117 | 37 / 333 |
| MDN improvement over baseline | 74.2% | 66.2% |
| mixture family | 1 | 1 (confirmed) |
| raw coefficient coverage (selected) | 0.768 | 0.702 |
| baseline total-mass fractional MAE | 0.0453 | 0.0194 |
| corrected total-mass fractional MAE | 0.0218 | 0.0107 |
| central-fraction bias after head | +0.163 (ns at 13 gal) | +0.385 (significant) |
| m=2 bias before correction | -0.384 (significant) | -0.002 (none) |
| vertical calibration | overcovered (spread) | undercovered (spread) |
| central-fraction head val overfit | yes (use scale 1.0) | no |

## Remaining Issues, In Priority Order

1. Vertical spread undercoverage (both vertical summaries): a single
   validation-fitted temperature does not transfer to test. Candidates: a
   vertical-specific uncertainty head conditioned on inclination and image
   thickness proxies, or cross-fitted (leave-galaxy-out) temperature
   estimation. This is now the clearest calibration failure.
2. Central-mass-fraction residual bias (+0.385 sigma): significant at 2c
   resolution. The head shrinks it by a factor 2.4; the remainder likely
   reflects representation and feature limits at high inclination. Revisit
   only with better central features or a higher-resolution central grid.
3. Bar-axis mass fraction coverage (0.628) sits at the low edge of its CI;
   watch it if the m=2 head is retrained.
4. The m=2 head is optional on 2c accuracy grounds alone (no bias to fix);
   keep it for the MAE gain and protocol continuity.

## Verification

- `ruff check .`: pass;
- `pytest -q`: pass (no code changes were needed for the scale-up - all 2b
  scripts ran unmodified on the 2c artifacts via CLI path overrides).
